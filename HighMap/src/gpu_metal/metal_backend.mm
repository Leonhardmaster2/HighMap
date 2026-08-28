/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU
 * General Public License. The full license is in the file LICENSE, distributed
 * with this software. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include "metal_shader_source.hpp"

#ifndef HIGHMAP_METAL_PRECOMPILED
#define HIGHMAP_METAL_PRECOMPILED 0
#endif

#if HIGHMAP_METAL_PRECOMPILED
#include "metal_library.hpp"
#endif

#include "highmap/gpu/metal.hpp"

#if HIGHMAP_HAS_METAL

namespace hmap::gpu::metal
{

namespace
{

thread_local ExecutionStats current_stats;

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start)
{
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

void begin_operation()
{
  current_stats = {};
}

struct GridParams
{
  int nx;
  int ny;
};

struct BinarySmoothParams
{
  int nx;
  int ny;
  float k;
};

struct NoiseParams
{
  int nx;
  int ny;
  int noise_id;
  float kx;
  float ky;
  std::uint32_t seed;
  int has_noise_x;
  int has_noise_y;
  int period_x;
  int period_y;
  float bbox0;
  float bbox1;
  float bbox2;
  float bbox3;
};

struct AdvectionParams
{
  int nx;
  int ny;
  float advection_length;
  float value_persistence;
};

struct ThermalParams
{
  int nx;
  int ny;
};

struct HydraulicParams
{
  int nx;
  int ny;
  float dt;
  float water_height;
  float k_capacity;
  float k_erode;
  float k_depose;
  float k_discharge_exp;
  float downcutting_max_depth_ratio;
  int flux_diffusion;
  float flux_diffusion_strength;
  float evap_factor;
  float initial_water_volume;
  float rain_volume;
  int maintain_water_volume;
};

struct ReduceParams
{
  int count;
  int threads_per_group;
};

class MetalContext
{
public:
  void ensure_initialized()
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (initialized) return;

    initialized = true;
    @autoreleasepool
    {
      device = MTLCreateSystemDefaultDevice();
      if (!device)
      {
        failure = "MTLCreateSystemDefaultDevice returned no device";
        return;
      }

      name = [[device name] UTF8String];
      queue = [device newCommandQueue];
      if (!queue)
      {
        failure = "Metal command queue creation failed";
        return;
      }

      NSError *error = nil;
#if HIGHMAP_METAL_PRECOMPILED
      dispatch_data_t binary = dispatch_data_create(
          highmap_metal_library,
          highmap_metal_library_len,
          nullptr,
          DISPATCH_DATA_DESTRUCTOR_DEFAULT);
      library = [device newLibraryWithData:binary error:&error];
#else
      NSString *source = [NSString stringWithUTF8String:
                                      HIGHMAP_METAL_SHADER_SOURCE];
      library = [device newLibraryWithSource:source options:nil error:&error];
#endif
      if (!library)
      {
        failure = error ? [[error localizedDescription] UTF8String]
                        : "Metal shader library compilation failed";
        return;
      }
      ready = true;
    }
  }

  bool ready_state() const { return ready; }

  const std::string &device_label() const { return name; }

  id<MTLComputePipelineState> pipeline(const char *function_name)
  {
    ensure_initialized();
    if (!ready) throw std::runtime_error(failure);

    const auto lookup_start = Clock::now();
    std::lock_guard<std::mutex> lock(mutex);
    auto it = pipelines.find(function_name);
    if (it != pipelines.end())
    {
      current_stats.pipeline_lookup_ms += elapsed_ms(lookup_start);
      return it->second;
    }

    @autoreleasepool
    {
      NSError *error = nil;
      NSString *name_string = [NSString stringWithUTF8String:function_name];
      id<MTLFunction> function = [library newFunctionWithName:name_string];
      if (!function)
        throw std::runtime_error(std::string("Metal function not found: ") +
                                 function_name);

      id<MTLComputePipelineState> state =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (!state)
      {
        std::string message = error ? [[error localizedDescription] UTF8String]
                                    : "Metal pipeline creation failed";
        throw std::runtime_error(message);
      }
      pipelines.emplace(function_name, state);
      ++current_stats.pipeline_creations;
      current_stats.pipeline_lookup_ms += elapsed_ms(lookup_start);
      return state;
    }
  }

  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;

private:
  id<MTLLibrary> library = nil;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
  std::mutex mutex;
  bool initialized = false;
  bool ready = false;
  std::string failure;
  std::string name;

  friend MetalContext &context();
  friend bool is_available();
  friend std::string device_name();
};

MetalContext &context()
{
  static MetalContext value;
  return value;
}

void check_shape(const Array &array, glm::ivec2 shape, const char *name)
{
  if (array.shape != shape || array.vector.size() !=
                                  static_cast<size_t>(shape.x * shape.y))
    throw std::invalid_argument(std::string("Metal ") + name +
                                " has an incompatible shape");
}

void check_shape_2d(glm::ivec2 shape)
{
  if (shape.x <= 0 || shape.y <= 0)
    throw std::invalid_argument("Metal kernels require a non-empty grid");
}

id<MTLBuffer> input_buffer(const std::vector<float> &values)
{
  if (values.empty()) throw std::invalid_argument("Metal cannot bind an empty array");
  const auto allocation_start = Clock::now();
  id<MTLBuffer> buffer = [context().device
      newBufferWithLength:values.size() * sizeof(float)
                 options:MTLResourceStorageModeShared];
  if (!buffer) throw std::runtime_error("Metal input buffer allocation failed");
  current_stats.allocation_ms += elapsed_ms(allocation_start);
  const auto upload_start = Clock::now();
  std::memcpy([buffer contents], values.data(), values.size() * sizeof(float));
  current_stats.upload_ms += elapsed_ms(upload_start);
  ++current_stats.buffer_allocations;
  current_stats.upload_bytes += values.size() * sizeof(float);
  return buffer;
}

id<MTLBuffer> zero_buffer(size_t count)
{
  const auto allocation_start = Clock::now();
  id<MTLBuffer> buffer = [context().device
      newBufferWithLength:count * sizeof(float)
                  options:MTLResourceStorageModeShared];
  if (!buffer) throw std::runtime_error("Metal output buffer allocation failed");
  std::memset([buffer contents], 0, count * sizeof(float));
  current_stats.allocation_ms += elapsed_ms(allocation_start);
  ++current_stats.buffer_allocations;
  return buffer;
}

struct DispatchShape
{
  NSUInteger width;
  NSUInteger height;
};

void dispatch(id<MTLComputeCommandEncoder> encoder,
              id<MTLComputePipelineState> pipeline,
              int nx,
              int ny,
              const char *kernel_name)
{
  const NSUInteger max_threads = [pipeline maxTotalThreadsPerThreadgroup];
  const NSUInteger simd_width = [pipeline threadExecutionWidth];

  // A process-level override makes layout experiments reproducible without
  // changing the public API. Examples: HIGHMAP_METAL_THREADGROUP=8x8 or
  // HIGHMAP_METAL_THREADGROUP=16x8. It is read once per process, not per
  // dispatch.
  static const DispatchShape override_shape = [] {
    const char *value = std::getenv("HIGHMAP_METAL_THREADGROUP");
    int width = 0;
    int height = 0;
    if (!value || std::sscanf(value, "%dx%d", &width, &height) != 2 ||
        width <= 0 || height <= 0)
      return DispatchShape{0, 0};
    return DispatchShape{static_cast<NSUInteger>(width),
                         static_cast<NSUInteger>(height)};
  }();

  if (override_shape.width > 0 && override_shape.height > 0 &&
      override_shape.width * override_shape.height <= max_threads)
  {
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(nx),
                                         static_cast<NSUInteger>(ny),
                                         1)
        threadsPerThreadgroup:MTLSizeMake(override_shape.width,
                                          override_shape.height,
                                          1)];
    return;
  }

  // Neighborhood kernels use a 16-wide tile on the measured Apple M3 path;
  // this keeps the tile compact while exposing a full SIMD group. Pointwise
  // kernels have no cross-thread neighborhood and use up to 32 lanes. The
  // limits are clamped by the pipeline's reported SIMD width, and the
  // environment override is available for repeatable tuning sweeps.
  const bool neighborhood = std::strstr(kernel_name, "advection") != nullptr ||
                            std::strstr(kernel_name, "thermal") != nullptr ||
                            std::strstr(kernel_name, "hydraulic") != nullptr;
  NSUInteger width = neighborhood ? std::min<NSUInteger>(16, simd_width)
                                  : std::min<NSUInteger>(32, simd_width);
  width = std::max<NSUInteger>(1, std::min(width, max_threads));
  const NSUInteger height = std::max<NSUInteger>(
      1, std::min<NSUInteger>(8, max_threads / width));
  [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(nx),
                                       static_cast<NSUInteger>(ny),
                                       1)
      threadsPerThreadgroup:MTLSizeMake(width, height, 1)];
}

NSUInteger reduction_threads(id<MTLComputePipelineState> pipeline)
{
  const NSUInteger maximum =
      std::min<NSUInteger>(256, [pipeline maxTotalThreadsPerThreadgroup]);
  NSUInteger threads = 1;
  while (threads * 2 <= maximum) threads *= 2;
  return std::max<NSUInteger>(1, threads);
}

void dispatch_reduction(id<MTLComputeCommandEncoder> encoder,
                        id<MTLComputePipelineState> pipeline,
                        int count,
                        NSUInteger threads)
{
  const NSUInteger groups =
      (static_cast<NSUInteger>(count) + threads - 1) / threads;
  [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

id<MTLCommandBuffer> make_command_buffer()
{
  id<MTLCommandBuffer> value = [context().queue commandBuffer];
  if (!value) throw std::runtime_error("Metal command buffer allocation failed");
  ++current_stats.command_buffers;
  return value;
}

id<MTLComputeCommandEncoder> compute_encoder(id<MTLCommandBuffer> buffer)
{
  id<MTLComputeCommandEncoder> value = [buffer computeCommandEncoder];
  if (!value) throw std::runtime_error("Metal compute encoder creation failed");
  ++current_stats.encoders;
  return value;
}

void record_encoding(Clock::time_point start)
{
  current_stats.encoding_ms += elapsed_ms(start);
}

void wait_for_completion(id<MTLCommandBuffer> command_buffer)
{
  const auto t0 = std::chrono::steady_clock::now();
  ++current_stats.synchronization_count;
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  const auto t1 = std::chrono::steady_clock::now();
  current_stats.wait_ms +=
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  const CFTimeInterval gpu_start = [command_buffer GPUStartTime];
  const CFTimeInterval gpu_end = [command_buffer GPUEndTime];
  if (gpu_end >= gpu_start && gpu_start > 0.0)
    current_stats.gpu_execution_ms += (gpu_end - gpu_start) * 1000.0;
  if ([command_buffer status] == MTLCommandBufferStatusError)
  {
    NSError *error = [command_buffer error];
    throw std::runtime_error(error ? [[error localizedDescription] UTF8String]
                                  : "Metal command buffer failed");
  }
}

void read_buffer(id<MTLBuffer> buffer, std::vector<float> &values)
{
  const auto readback_start = Clock::now();
  std::memcpy(values.data(), [buffer contents], values.size() * sizeof(float));
  current_stats.readback_ms += elapsed_ms(readback_start);
  current_stats.readback_bytes += values.size() * sizeof(float);
}

void set_bytes(id<MTLComputeCommandEncoder> encoder,
               const void *bytes,
               size_t size,
               NSUInteger index)
{
  [encoder setBytes:bytes length:size atIndex:index];
}

void require_ready()
{
  context().ensure_initialized();
  if (!context().ready_state())
    throw std::runtime_error(
        "HighMap Metal backend is unavailable on this host");
}

} // namespace

bool is_available()
{
  context().ensure_initialized();
  return context().ready_state();
}

std::string device_name()
{
  context().ensure_initialized();
  return context().device_label();
}

DeviceCapabilities capabilities()
{
  context().ensure_initialized();
  if (!context().ready_state()) return {};

  const MTLSize max_threads = [context().device maxThreadsPerThreadgroup];
  id<MTLComputePipelineState> gradient_pipeline =
      context().pipeline("gradient_norm");

  DeviceCapabilities result;
  result.device_name = context().device_label();
  result.recommended_max_working_set_size =
      [context().device recommendedMaxWorkingSetSize];
  result.thread_execution_width =
      static_cast<std::uint32_t>([gradient_pipeline threadExecutionWidth]);
  result.max_threads_per_threadgroup =
      static_cast<std::uint32_t>(max_threads.width * max_threads.height *
                                 max_threads.depth);

  // Report the highest generic family advertised by the SDK/runtime pair.
  // This is diagnostic metadata only; dispatch policy remains capability based.
  if (@available(macOS 26.0, *))
  {
    if ([context().device supportsFamily:MTLGPUFamilyMetal4])
      result.gpu_family = static_cast<std::uint32_t>(MTLGPUFamilyMetal4);
  }
  if (result.gpu_family == 0)
  {
    if (@available(macOS 13.0, *))
    {
      if ([context().device supportsFamily:MTLGPUFamilyMetal3])
        result.gpu_family = static_cast<std::uint32_t>(MTLGPUFamilyMetal3);
    }
  }
  if (result.gpu_family == 0 &&
      [context().device supportsFamily:MTLGPUFamilyApple9])
    result.gpu_family = static_cast<std::uint32_t>(MTLGPUFamilyApple9);
  if (result.gpu_family == 0 &&
      [context().device supportsFamily:MTLGPUFamilyApple7])
    result.gpu_family = static_cast<std::uint32_t>(MTLGPUFamilyApple7);
  if (result.gpu_family == 0 &&
      [context().device supportsFamily:MTLGPUFamilyApple1])
    result.gpu_family = static_cast<std::uint32_t>(MTLGPUFamilyApple1);

  return result;
}

ExecutionStats last_execution_stats()
{
  return current_stats;
}

bool supports_noise(NoiseType noise_type)
{
  switch (noise_type)
  {
  case NoiseType::PERLIN:
  case NoiseType::PERLIN_BILLOW:
  case NoiseType::PERLIN_HALF:
  case NoiseType::SIMPLEX2:
  case NoiseType::VALUE:
  case NoiseType::VALUE_LINEAR: return true;
  default: return false;
  }
}

Array gradient_norm(const Array &array)
{
  begin_operation();
  require_ready();
  check_shape_2d(array.shape);

  Array output(array.shape);
  id<MTLBuffer> input = input_buffer(array.vector);
  id<MTLBuffer> result = zero_buffer(array.vector.size());
  GridParams params{array.shape.x, array.shape.y};

  id<MTLComputePipelineState> pipeline = context().pipeline("gradient_norm");
  id<MTLCommandBuffer> command_buffer = make_command_buffer();
  const auto encoding_start = Clock::now();
  id<MTLComputeCommandEncoder> encoder = compute_encoder(command_buffer);
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:input offset:0 atIndex:0];
  [encoder setBuffer:result offset:0 atIndex:1];
  set_bytes(encoder, &params, sizeof(params), 2);
  dispatch(encoder, pipeline, params.nx, params.ny, "gradient_norm");
  [encoder endEncoding];
  record_encoding(encoding_start);
  wait_for_completion(command_buffer);
  read_buffer(result, output.vector);
  return output;
}

Array smooth_extrema(const Array &array1,
                     const Array &array2,
                     float        k,
                     const char  *kernel_name)
{
  begin_operation();
  require_ready();
  check_shape_2d(array1.shape);
  check_shape(array2, array1.shape, "array2");

  Array output(array1.shape);
  id<MTLBuffer> input1 = input_buffer(array1.vector);
  id<MTLBuffer> input2 = input_buffer(array2.vector);
  id<MTLBuffer> result = zero_buffer(output.vector.size());
  BinarySmoothParams params{array1.shape.x, array1.shape.y, k};

  id<MTLComputePipelineState> pipeline = context().pipeline(kernel_name);
  id<MTLCommandBuffer> command_buffer = make_command_buffer();
  const auto encoding_start = Clock::now();
  id<MTLComputeCommandEncoder> encoder = compute_encoder(command_buffer);
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:input1 offset:0 atIndex:0];
  [encoder setBuffer:input2 offset:0 atIndex:1];
  [encoder setBuffer:result offset:0 atIndex:2];
  set_bytes(encoder, &params, sizeof(params), 3);
  dispatch(encoder, pipeline, params.nx, params.ny, kernel_name);
  [encoder endEncoding];
  record_encoding(encoding_start);
  wait_for_completion(command_buffer);
  read_buffer(result, output.vector);
  return output;
}

Array maximum_smooth(const Array &array1, const Array &array2, float k)
{
  return smooth_extrema(array1, array2, k, "maximum_smooth");
}

Array minimum_smooth(const Array &array1, const Array &array2, float k)
{
  return smooth_extrema(array1, array2, k, "minimum_smooth");
}

Array noise(NoiseType     noise_type,
            glm::ivec2    shape,
            glm::vec2     kw,
            std::uint32_t seed,
            const Array  *p_noise_x,
            const Array  *p_noise_y,
            glm::vec4     bbox,
            glm::ivec2    period)
{
  begin_operation();
  require_ready();
  check_shape_2d(shape);
  if (!supports_noise(noise_type))
    throw std::invalid_argument("Metal staged noise kernel does not support this NoiseType");
  if (p_noise_x) check_shape(*p_noise_x, shape, "noise_x");
  if (p_noise_y) check_shape(*p_noise_y, shape, "noise_y");

  Array output(shape);
  std::vector<float> dummy(1, 0.f);
  id<MTLBuffer> result = zero_buffer(output.vector.size());
  id<MTLBuffer> noise_x = input_buffer(p_noise_x ? p_noise_x->vector : dummy);
  id<MTLBuffer> noise_y = input_buffer(p_noise_y ? p_noise_y->vector : dummy);
  NoiseParams params{shape.x,
                     shape.y,
                     static_cast<int>(noise_type),
                     kw.x,
                     kw.y,
                     seed,
                     p_noise_x ? 1 : 0,
                     p_noise_y ? 1 : 0,
                     period.x,
                     period.y,
                     bbox.x,
                     bbox.y,
                     bbox.z,
                     bbox.w};

  id<MTLComputePipelineState> pipeline = context().pipeline("noise");
  id<MTLCommandBuffer> command_buffer = make_command_buffer();
  const auto encoding_start = Clock::now();
  id<MTLComputeCommandEncoder> encoder = compute_encoder(command_buffer);
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:result offset:0 atIndex:0];
  [encoder setBuffer:noise_x offset:0 atIndex:1];
  [encoder setBuffer:noise_y offset:0 atIndex:2];
  set_bytes(encoder, &params, sizeof(params), 3);
  dispatch(encoder, pipeline, params.nx, params.ny, "noise");
  [encoder endEncoding];
  record_encoding(encoding_start);
  wait_for_completion(command_buffer);
  read_buffer(result, output.vector);
  return output;
}

Array advection_warp(const Array &z,
                     const Array &advected_field,
                     const Array &dx,
                     const Array &dy,
                     float        advection_length,
                     float        value_persistence,
                     const Array *p_mask)
{
  begin_operation();
  require_ready();
  const glm::ivec2 shape = z.shape;
  check_shape_2d(shape);
  check_shape(advected_field, shape, "advected_field");
  check_shape(dx, shape, "dx");
  check_shape(dy, shape, "dy");
  if (p_mask) check_shape(*p_mask, shape, "mask");

  Array output(shape);
  Array default_mask(shape, 1.f);
  id<MTLBuffer> z_buffer = input_buffer(z.vector);
  id<MTLBuffer> field_buffer = input_buffer(advected_field.vector);
  id<MTLBuffer> dx_buffer = input_buffer(dx.vector);
  id<MTLBuffer> dy_buffer = input_buffer(dy.vector);
  id<MTLBuffer> mask_buffer = input_buffer(p_mask ? p_mask->vector
                                                  : default_mask.vector);
  id<MTLBuffer> output_buffer = zero_buffer(output.vector.size());
  AdvectionParams params{shape.x, shape.y, advection_length, value_persistence};

  id<MTLComputePipelineState> pipeline = context().pipeline("advection_warp");
  id<MTLCommandBuffer> command_buffer = make_command_buffer();
  const auto encoding_start = Clock::now();
  id<MTLComputeCommandEncoder> encoder = compute_encoder(command_buffer);
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:z_buffer offset:0 atIndex:0];
  [encoder setBuffer:field_buffer offset:0 atIndex:1];
  [encoder setBuffer:dx_buffer offset:0 atIndex:2];
  [encoder setBuffer:dy_buffer offset:0 atIndex:3];
  [encoder setBuffer:mask_buffer offset:0 atIndex:4];
  [encoder setBuffer:output_buffer offset:0 atIndex:5];
  set_bytes(encoder, &params, sizeof(params), 6);
  dispatch(encoder, pipeline, params.nx, params.ny, "advection_warp");
  [encoder endEncoding];
  record_encoding(encoding_start);
  wait_for_completion(command_buffer);
  read_buffer(output_buffer, output.vector);
  return output;
}

void thermal(Array &z, const Array &talus, int iterations)
{
  begin_operation();
  require_ready();
  const glm::ivec2 shape = z.shape;
  check_shape_2d(shape);
  check_shape(talus, shape, "talus");
  if (iterations <= 0) return;

  id<MTLBuffer> input = input_buffer(z.vector);
  id<MTLBuffer> output = zero_buffer(z.vector.size());
  ThermalParams params{shape.x, shape.y};
  id<MTLBuffer> talus_buffer = input_buffer(talus.vector);
  id<MTLComputePipelineState> pipeline = context().pipeline("thermal_pass");
  id<MTLCommandBuffer> command_buffer = make_command_buffer();
  const auto encoding_start = Clock::now();

  for (int it = 0; it < iterations; ++it)
  {
    id<MTLComputeCommandEncoder> encoder = compute_encoder(command_buffer);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:talus_buffer offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    set_bytes(encoder, &params, sizeof(params), 3);
    dispatch(encoder, pipeline, params.nx, params.ny, "thermal_pass");
    [encoder endEncoding];
    std::swap(input, output);
  }

  record_encoding(encoding_start);
  wait_for_completion(command_buffer);
  read_buffer(input, z.vector);
}

void hydraulic_vpipes(Array &z,
                      float  water_height,
                      bool   maintain_water_volume,
                      float  evap_rate,
                      int    iterations,
                      float  dt,
                      float  k_capacity,
                      float  k_erode,
                      float  k_depose,
                      float  k_discharge_exp,
                      float  downcutting_max_depth_ratio,
                      bool   flux_diffusion,
                      float  flux_diffusion_strength,
                      Array *p_rain_map,
                      Array *p_water_depth,
                      Array *p_sediment,
                      Array *p_vel_u,
                      Array *p_vel_v)
{
  begin_operation();
  require_ready();
  const glm::ivec2 shape = z.shape;
  check_shape_2d(shape);
  if (p_rain_map) check_shape(*p_rain_map, shape, "rain_map");

  Array rain(shape, 1.f);
  if (p_rain_map) rain = *p_rain_map;
  const size_t count = z.vector.size();
  const float rain_volume =
      std::accumulate(rain.vector.begin(), rain.vector.end(), 0.f);
  const float initial_water_volume = water_height * rain_volume;

  id<MTLBuffer> z_a = input_buffer(z.vector);
  id<MTLBuffer> z_b = zero_buffer(count);
  id<MTLBuffer> rain_buffer = input_buffer(rain.vector);
  id<MTLBuffer> d_a = zero_buffer(count);
  id<MTLBuffer> d_b = zero_buffer(count);
  id<MTLBuffer> d2 = zero_buffer(count);
  id<MTLBuffer> sediment = zero_buffer(count);
  id<MTLBuffer> sediment_tmp = zero_buffer(count);
  id<MTLBuffer> fl_a = zero_buffer(count);
  id<MTLBuffer> fr_a = zero_buffer(count);
  id<MTLBuffer> ft_a = zero_buffer(count);
  id<MTLBuffer> fb_a = zero_buffer(count);
  id<MTLBuffer> fl_b = zero_buffer(count);
  id<MTLBuffer> fr_b = zero_buffer(count);
  id<MTLBuffer> ft_b = zero_buffer(count);
  id<MTLBuffer> fb_b = zero_buffer(count);
  id<MTLBuffer> velocity_u = zero_buffer(count);
  id<MTLBuffer> velocity_v = zero_buffer(count);

  // d_a starts at water_height * rain_map. This is a one-time initialization
  // in shared unified memory; it is never touched by the host again until the
  // final command buffer has completed.
  id<MTLComputePipelineState> flow_pipeline =
      context().pipeline("hydraulic_flow_pass");
  id<MTLComputePipelineState> water_pipeline =
      context().pipeline("hydraulic_water_pass");
  id<MTLComputePipelineState> erosion_pipeline =
      context().pipeline("hydraulic_erosion_pass");
  id<MTLComputePipelineState> sediment_pipeline =
      context().pipeline("hydraulic_sediment_pass");
  id<MTLComputePipelineState> evaporate_pipeline =
      context().pipeline("hydraulic_evaporate");
  id<MTLComputePipelineState> rescale_pipeline =
      context().pipeline("hydraulic_rescale");
  id<MTLComputePipelineState> sum_pipeline =
      context().pipeline("hydraulic_sum_tiles");
  id<MTLComputePipelineState> reduce_pipeline =
      context().pipeline("hydraulic_sum_reduce");

  const NSUInteger sum_threads = reduction_threads(sum_pipeline);
  const size_t partial_count =
      std::max<size_t>(1, (count + sum_threads - 1) / sum_threads);
  id<MTLBuffer> reduction_a = zero_buffer(partial_count);
  id<MTLBuffer> reduction_b = zero_buffer(partial_count);

  HydraulicParams params{shape.x,
                         shape.y,
                         dt,
                         water_height,
                         k_capacity,
                         k_erode,
                         k_depose,
                         k_discharge_exp,
                         downcutting_max_depth_ratio,
                         flux_diffusion ? 1 : 0,
                         flux_diffusion_strength,
                         1.f - dt * evap_rate,
                         initial_water_volume,
                         rain_volume,
                         maintain_water_volume ? 1 : 0};

  id<MTLCommandBuffer> command_buffer = make_command_buffer();
  const auto encoding_start = Clock::now();
  // Initialize the shared water buffer by a small host-side multiply. This
  // avoids another shader solely for a multiply while preserving GPU residency
  // thereafter.
  float *initial_water = static_cast<float *>([d_a contents]);
  for (size_t i = 0; i < count; ++i) initial_water[i] = water_height * rain.vector[i];

  for (int it = 0; it < std::max(0, iterations); ++it)
  {
    id<MTLComputeCommandEncoder> encoder = compute_encoder(command_buffer);
    [encoder setComputePipelineState:flow_pipeline];
    [encoder setBuffer:z_a offset:0 atIndex:0];
    [encoder setBuffer:fl_a offset:0 atIndex:1];
    [encoder setBuffer:fr_a offset:0 atIndex:2];
    [encoder setBuffer:ft_a offset:0 atIndex:3];
    [encoder setBuffer:fb_a offset:0 atIndex:4];
    [encoder setBuffer:d_a offset:0 atIndex:5];
    [encoder setBuffer:fl_b offset:0 atIndex:6];
    [encoder setBuffer:fr_b offset:0 atIndex:7];
    [encoder setBuffer:ft_b offset:0 atIndex:8];
    [encoder setBuffer:fb_b offset:0 atIndex:9];
    set_bytes(encoder, &params, sizeof(params), 10);
    dispatch(encoder, flow_pipeline, shape.x, shape.y, "hydraulic_flow_pass");
    [encoder endEncoding];

    encoder = compute_encoder(command_buffer);
    [encoder setComputePipelineState:water_pipeline];
    [encoder setBuffer:z_a offset:0 atIndex:0];
    [encoder setBuffer:fl_b offset:0 atIndex:1];
    [encoder setBuffer:fr_b offset:0 atIndex:2];
    [encoder setBuffer:ft_b offset:0 atIndex:3];
    [encoder setBuffer:fb_b offset:0 atIndex:4];
    [encoder setBuffer:d_a offset:0 atIndex:5];
    [encoder setBuffer:d2 offset:0 atIndex:6];
    [encoder setBuffer:velocity_u offset:0 atIndex:7];
    [encoder setBuffer:velocity_v offset:0 atIndex:8];
    set_bytes(encoder, &params, sizeof(params), 9);
    dispatch(encoder, water_pipeline, shape.x, shape.y, "hydraulic_water_pass");
    [encoder endEncoding];

    encoder = compute_encoder(command_buffer);
    [encoder setComputePipelineState:erosion_pipeline];
    [encoder setBuffer:z_a offset:0 atIndex:0];
    [encoder setBuffer:d2 offset:0 atIndex:1];
    [encoder setBuffer:velocity_u offset:0 atIndex:2];
    [encoder setBuffer:velocity_v offset:0 atIndex:3];
    [encoder setBuffer:sediment offset:0 atIndex:4];
    [encoder setBuffer:z_b offset:0 atIndex:5];
    [encoder setBuffer:sediment_tmp offset:0 atIndex:6];
    set_bytes(encoder, &params, sizeof(params), 7);
    dispatch(encoder, erosion_pipeline, shape.x, shape.y, "hydraulic_erosion_pass");
    [encoder endEncoding];

    encoder = compute_encoder(command_buffer);
    [encoder setComputePipelineState:sediment_pipeline];
    [encoder setBuffer:velocity_u offset:0 atIndex:0];
    [encoder setBuffer:velocity_v offset:0 atIndex:1];
    [encoder setBuffer:sediment_tmp offset:0 atIndex:2];
    [encoder setBuffer:sediment offset:0 atIndex:3];
    set_bytes(encoder, &params, sizeof(params), 4);
    dispatch(encoder, sediment_pipeline, shape.x, shape.y, "hydraulic_sediment_pass");
    [encoder endEncoding];

    encoder = compute_encoder(command_buffer);
    [encoder setComputePipelineState:evaporate_pipeline];
    [encoder setBuffer:d2 offset:0 atIndex:0];
    [encoder setBuffer:d_b offset:0 atIndex:1];
    set_bytes(encoder, &params, sizeof(params), 2);
    dispatch(encoder, evaporate_pipeline, shape.x, shape.y, "hydraulic_evaporate");
    [encoder endEncoding];

    if (maintain_water_volume)
    {
      // Sum the new water depth on the GPU. A hierarchical float reduction is
      // used instead of a quantized integer atomic so large maps and high
      // water levels do not overflow and the correction remains numerically
      // faithful. The reduction is only needed for the optional volume
      // correction; the default path avoids these extra command encoders.
      id<MTLBuffer> reduction_input = reduction_a;
      id<MTLBuffer> reduction_output = reduction_b;
      int reduction_count = static_cast<int>(count);
      int next_count = static_cast<int>(partial_count);

      encoder = compute_encoder(command_buffer);
      [encoder setComputePipelineState:sum_pipeline];
      [encoder setBuffer:d_b offset:0 atIndex:0];
      [encoder setBuffer:reduction_input offset:0 atIndex:1];
      ReduceParams reduce_params{reduction_count,
                                 static_cast<int>(sum_threads)};
      set_bytes(encoder, &reduce_params, sizeof(reduce_params), 2);
      dispatch_reduction(encoder, sum_pipeline, reduction_count, sum_threads);
      [encoder endEncoding];
      reduction_count = next_count;

      while (reduction_count > 1)
      {
        next_count = static_cast<int>(
            (static_cast<size_t>(reduction_count) + sum_threads - 1) /
            sum_threads);
        encoder = compute_encoder(command_buffer);
        [encoder setComputePipelineState:reduce_pipeline];
        [encoder setBuffer:reduction_input offset:0 atIndex:0];
        [encoder setBuffer:reduction_output offset:0 atIndex:1];
        reduce_params = {reduction_count, static_cast<int>(sum_threads)};
        set_bytes(encoder, &reduce_params, sizeof(reduce_params), 2);
        dispatch_reduction(encoder,
                           reduce_pipeline,
                           reduction_count,
                           sum_threads);
        [encoder endEncoding];
        reduction_count = next_count;
        std::swap(reduction_input, reduction_output);
      }

      encoder = compute_encoder(command_buffer);
      [encoder setComputePipelineState:rescale_pipeline];
      [encoder setBuffer:d_b offset:0 atIndex:0];
      [encoder setBuffer:rain_buffer offset:0 atIndex:1];
      [encoder setBuffer:reduction_input offset:0 atIndex:2];
      set_bytes(encoder, &params, sizeof(params), 3);
      dispatch(encoder, rescale_pipeline, shape.x, shape.y, "hydraulic_rescale");
      [encoder endEncoding];
    }

    std::swap(z_a, z_b);
    std::swap(d_a, d_b);
    std::swap(fl_a, fl_b);
    std::swap(fr_a, fr_b);
    std::swap(ft_a, ft_b);
    std::swap(fb_a, fb_b);
  }

  record_encoding(encoding_start);
  wait_for_completion(command_buffer);
  read_buffer(z_a, z.vector);

  if (p_water_depth)
  {
    *p_water_depth = Array(shape);
    read_buffer(d_a, p_water_depth->vector);
  }
  if (p_sediment)
  {
    *p_sediment = Array(shape);
    read_buffer(sediment, p_sediment->vector);
  }
  if (p_vel_u)
  {
    *p_vel_u = Array(shape);
    read_buffer(velocity_u, p_vel_u->vector);
  }
  if (p_vel_v)
  {
    *p_vel_v = Array(shape);
    read_buffer(velocity_v, p_vel_v->vector);
  }
}

} // namespace hmap::gpu::metal

#endif
