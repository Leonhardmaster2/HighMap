/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU
 * General Public License. The full license is in the file LICENSE, distributed
 * with this software. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

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

namespace detail
{

struct DeviceSessionState
{
  StorageMode default_storage = StorageMode::shared;
  id<MTLCommandBuffer> command_buffer = nil;
  bool finished = false;
  bool has_work = false;
  std::size_t live_bytes = 0;
  ExecutionStats stats;
  std::vector<id<MTLCommandBuffer>> submitted_command_buffers;
  std::vector<id<MTLBuffer>> upload_staging;
  std::unordered_map<std::size_t, std::vector<id<MTLBuffer>>> shared_pool;
  std::unordered_map<std::size_t, std::vector<id<MTLBuffer>>> private_pool;

  ~DeviceSessionState()
  {
    if (command_buffer && !finished)
    {
      id<MTLCommandBuffer> wait_buffer = nil;
      if (has_work)
      {
        [command_buffer commit];
        wait_buffer = command_buffer;
      }
      else if (!submitted_command_buffers.empty())
        wait_buffer = submitted_command_buffers.back();
      if (wait_buffer) [wait_buffer waitUntilCompleted];
    }
  }
};

struct DeviceArrayState
{
  std::shared_ptr<DeviceSessionState> session;
  id<MTLBuffer> buffer = nil;
  id<MTLBuffer> download_staging = nil;
  glm::ivec2 shape = {0, 0};
  StorageMode storage = StorageMode::shared;
  ResidencyState residency = ResidencyState::device_valid;
  std::string debug_name;
  std::size_t byte_size = 0;
  std::shared_ptr<std::vector<float>> host_shadow;
};

} // namespace detail

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

struct NoiseFbmParams
{
  int nx;
  int ny;
  int noise_id;
  float kx;
  float ky;
  std::uint32_t seed;
  int octaves;
  float weight;
  float persistence;
  float lacunarity;
  int has_ctrl_param;
  int has_noise_x;
  int has_noise_y;
  int period_x;
  int period_y;
  float bbox0;
  float bbox1;
  float bbox2;
  float bbox3;
};

struct SmoothCpulseParams
{
  int nx;
  int ny;
  int ir;
  int pass;
  float weight_sum;
};

struct NormalizeParams
{
  int nx;
  int ny;
  float from_min;
  float from_max;
  float to_min;
  float to_max;
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

struct BorderParams
{
  int nx;
  int ny;
};

struct LinearCombineParams
{
  int nx;
  int ny;
  float weight1;
  float weight2;
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

struct MinMaxParams
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

  id<MTLComputePipelineState> pipeline(const char       *function_name,
                                       ExecutionStats *stats = nullptr)
  {
    ensure_initialized();
    if (!ready) throw std::runtime_error(failure);

    const auto lookup_start = Clock::now();
    std::lock_guard<std::mutex> lock(mutex);
    auto it = pipelines.find(function_name);
    if (it != pipelines.end())
    {
      if (stats)
        stats->pipeline_lookup_ms += elapsed_ms(lookup_start);
      else
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
      if (stats)
      {
        ++stats->pipeline_creations;
        stats->pipeline_lookup_ms += elapsed_ms(lookup_start);
      }
      else
      {
        ++current_stats.pipeline_creations;
        current_stats.pipeline_lookup_ms += elapsed_ms(lookup_start);
      }
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
  current_stats.bytes_allocated += values.size() * sizeof(float);
  current_stats.peak_resident_bytes = std::max(
      current_stats.peak_resident_bytes, current_stats.bytes_allocated);
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
  current_stats.bytes_allocated += count * sizeof(float);
  current_stats.peak_resident_bytes = std::max(
      current_stats.peak_resident_bytes, current_stats.bytes_allocated);
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

id<MTLCommandBuffer> make_command_buffer(ExecutionStats *stats = nullptr)
{
  id<MTLCommandBuffer> value = [context().queue commandBuffer];
  if (!value) throw std::runtime_error("Metal command buffer allocation failed");
  if (stats)
    ++stats->command_buffers;
  else
    ++current_stats.command_buffers;
  return value;
}

id<MTLComputeCommandEncoder> compute_encoder(id<MTLCommandBuffer> buffer,
                                             ExecutionStats    *stats = nullptr)
{
  id<MTLComputeCommandEncoder> value = [buffer computeCommandEncoder];
  if (!value) throw std::runtime_error("Metal compute encoder creation failed");
  if (stats)
    ++stats->encoders;
  else
    ++current_stats.encoders;
  return value;
}

void record_encoding(Clock::time_point start, ExecutionStats *stats = nullptr)
{
  if (stats)
    stats->encoding_ms += elapsed_ms(start);
  else
    current_stats.encoding_ms += elapsed_ms(start);
}

void wait_for_completion(id<MTLCommandBuffer> command_buffer,
                         ExecutionStats    *stats = nullptr,
                         bool               commit = true)
{
  const auto t0 = std::chrono::steady_clock::now();
  if (stats)
    ++stats->synchronization_count;
  else
    ++current_stats.synchronization_count;
  if (commit) [command_buffer commit];
  [command_buffer waitUntilCompleted];
  const auto t1 = std::chrono::steady_clock::now();
  const double wait_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  if (stats)
    stats->wait_ms += wait_ms;
  else
    current_stats.wait_ms += wait_ms;
  const CFTimeInterval gpu_start = [command_buffer GPUStartTime];
  const CFTimeInterval gpu_end = [command_buffer GPUEndTime];
  if (gpu_end >= gpu_start && gpu_start > 0.0)
  {
    const double gpu_ms = (gpu_end - gpu_start) * 1000.0;
    if (stats)
      stats->gpu_execution_ms += gpu_ms;
    else
      current_stats.gpu_execution_ms += gpu_ms;
  }
  if ([command_buffer status] == MTLCommandBufferStatusError)
  {
    NSError *error = [command_buffer error];
    throw std::runtime_error(error ? [[error localizedDescription] UTF8String]
                                  : "Metal command buffer failed");
  }
}

void read_buffer(id<MTLBuffer> buffer,
                 std::vector<float> &values,
                 ExecutionStats    *stats = nullptr)
{
  const auto readback_start = Clock::now();
  std::memcpy(values.data(), [buffer contents], values.size() * sizeof(float));
  if (stats)
  {
    stats->readback_ms += elapsed_ms(readback_start);
    stats->readback_bytes += values.size() * sizeof(float);
  }
  else
  {
    current_stats.readback_ms += elapsed_ms(readback_start);
    current_stats.readback_bytes += values.size() * sizeof(float);
  }
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

void require_session_open(const std::shared_ptr<detail::DeviceSessionState> &session)
{
  if (!session) throw std::runtime_error("Metal DeviceSession is empty");
  if (session->finished)
    throw std::runtime_error(
        "Metal DeviceSession is already finished; create a new session");
  if (!session->command_buffer)
    throw std::runtime_error("Metal DeviceSession has no command buffer");
}

void require_input(const std::shared_ptr<detail::DeviceSessionState> &session,
                  const std::shared_ptr<detail::DeviceArrayState>   &input,
                  const char                                         *name)
{
  if (!input || !input->buffer)
    throw std::invalid_argument(std::string("Metal ") + name +
                                " is an empty DeviceArray");
  if (input->session != session)
    throw std::invalid_argument(std::string("Metal ") + name +
                                " belongs to another DeviceSession");
}

void require_same_shape(const std::shared_ptr<detail::DeviceArrayState> &input,
                        glm::ivec2                                      shape,
                        const char                                      *name)
{
  if (input->shape != shape)
    throw std::invalid_argument(std::string("Metal ") + name +
                                " has an incompatible shape");
}

void record_new_allocation(
    const std::shared_ptr<detail::DeviceSessionState> &session,
    std::size_t                                        bytes)
{
  ++session->stats.buffer_allocations;
  session->stats.bytes_allocated += bytes;
  session->live_bytes += bytes;
  session->stats.resident_bytes = session->live_bytes;
  session->stats.peak_resident_bytes = std::max<std::uint64_t>(
      session->stats.peak_resident_bytes, session->live_bytes);
}

void record_reused_allocation(
    const std::shared_ptr<detail::DeviceSessionState> &session,
    std::size_t                                        bytes)
{
  ++session->stats.buffer_reuses;
  session->stats.bytes_reused += bytes;
  session->live_bytes += bytes;
  session->stats.resident_bytes = session->live_bytes;
  session->stats.peak_resident_bytes = std::max<std::uint64_t>(
      session->stats.peak_resident_bytes, session->live_bytes);
}

id<MTLBuffer> new_session_buffer(
    const std::shared_ptr<detail::DeviceSessionState> &session,
    std::size_t                                        bytes,
    StorageMode                                        mode)
{
  const auto start = Clock::now();
  const MTLResourceOptions options =
      mode == StorageMode::private_storage ? MTLResourceStorageModePrivate
                                           : MTLResourceStorageModeShared;
  id<MTLBuffer> buffer =
      [context().device newBufferWithLength:bytes options:options];
  if (!buffer) throw std::runtime_error("Metal DeviceArray allocation failed");
  session->stats.allocation_ms += elapsed_ms(start);
  record_new_allocation(session, bytes);
  return buffer;
}

std::unordered_map<std::size_t, std::vector<id<MTLBuffer>>> &buffer_pool(
    const std::shared_ptr<detail::DeviceSessionState> &session,
    StorageMode                                        mode)
{
  return mode == StorageMode::private_storage ? session->private_pool
                                               : session->shared_pool;
}

id<MTLBuffer> acquire_session_buffer(
    const std::shared_ptr<detail::DeviceSessionState> &session,
    std::size_t                                        bytes,
    StorageMode                                        mode)
{
  auto &pool = buffer_pool(session, mode)[bytes];
  if (!pool.empty())
  {
    id<MTLBuffer> buffer = pool.back();
    pool.pop_back();
    record_reused_allocation(session, bytes);
    return buffer;
  }
  return new_session_buffer(session, bytes, mode);
}

void release_session_buffer(
    const std::shared_ptr<detail::DeviceSessionState> &session,
    id<MTLBuffer>                                      buffer,
    std::size_t                                        bytes,
    StorageMode                                        mode)
{
  if (!buffer) return;
  buffer_pool(session, mode)[bytes].push_back(buffer);
  if (session->live_bytes >= bytes)
    session->live_bytes -= bytes;
  else
    session->live_bytes = 0;
  session->stats.resident_bytes = session->live_bytes;
}

std::pair<float, float> reduce_range(
    const std::shared_ptr<detail::DeviceSessionState> &session,
    id<MTLBuffer>                                      input,
    std::size_t                                         count)
{
  require_session_open(session);
  if (!input || count == 0)
    throw std::invalid_argument("Metal range reduction requires a non-empty input");

  id<MTLComputePipelineState> tiles_pipeline =
      context().pipeline("minmax_tiles", &session->stats);
  id<MTLComputePipelineState> reduce_pipeline =
      context().pipeline("minmax_reduce", &session->stats);
  const NSUInteger threads = reduction_threads(reduce_pipeline);
  const std::size_t partial_count =
      std::max<std::size_t>(1, (count + threads - 1) / threads);
  const std::size_t partial_bytes = partial_count * 2 * sizeof(float);
  id<MTLBuffer> reduction_a =
      new_session_buffer(session, partial_bytes, StorageMode::shared);
  id<MTLBuffer> reduction_b =
      new_session_buffer(session, partial_bytes, StorageMode::shared);

  const auto encoding_start = Clock::now();
  id<MTLComputeCommandEncoder> encoder =
      compute_encoder(session->command_buffer, &session->stats);
  [encoder setComputePipelineState:tiles_pipeline];
  [encoder setBuffer:input offset:0 atIndex:0];
  [encoder setBuffer:reduction_a offset:0 atIndex:1];
  MinMaxParams params{static_cast<int>(count), static_cast<int>(threads)};
  set_bytes(encoder, &params, sizeof(params), 2);
  dispatch_reduction(encoder, tiles_pipeline, static_cast<int>(count), threads);
  [encoder endEncoding];

  id<MTLBuffer> reduction_input = reduction_a;
  id<MTLBuffer> reduction_output = reduction_b;
  std::size_t reduction_count = partial_count;
  while (reduction_count > 1)
  {
    const std::size_t next_count =
        (reduction_count + threads - 1) / threads;
    encoder = compute_encoder(session->command_buffer, &session->stats);
    [encoder setComputePipelineState:reduce_pipeline];
    [encoder setBuffer:reduction_input offset:0 atIndex:0];
    [encoder setBuffer:reduction_output offset:0 atIndex:1];
    params = {static_cast<int>(reduction_count), static_cast<int>(threads)};
    set_bytes(encoder, &params, sizeof(params), 2);
    dispatch_reduction(encoder,
                       reduce_pipeline,
                       static_cast<int>(reduction_count),
                       threads);
    [encoder endEncoding];
    reduction_count = next_count;
    std::swap(reduction_input, reduction_output);
  }
  session->stats.encoding_ms += elapsed_ms(encoding_start);
  session->has_work = true;

  // Normalization needs two scalar values before its pointwise pass can be
  // encoded. Keep this as a scalar synchronization; the terrain itself never
  // crosses the host boundary.
  wait_for_completion(session->command_buffer, &session->stats);
  session->command_buffer = make_command_buffer(&session->stats);
  session->has_work = false;

  std::vector<float> range(2, 0.f);
  read_buffer(reduction_input, range, &session->stats);
  release_session_buffer(session,
                         reduction_a,
                         partial_bytes,
                         StorageMode::shared);
  release_session_buffer(session,
                         reduction_b,
                         partial_bytes,
                         StorageMode::shared);
  return {range[0], range[1]};
}

id<MTLBlitCommandEncoder> blit_encoder(
    const std::shared_ptr<detail::DeviceSessionState> &session)
{
  id<MTLBlitCommandEncoder> encoder =
      [session->command_buffer blitCommandEncoder];
  if (!encoder) throw std::runtime_error("Metal blit encoder creation failed");
  ++session->stats.blit_encoders;
  return encoder;
}

void encode_copy(const std::shared_ptr<detail::DeviceSessionState> &session,
                 id<MTLBuffer>                                      source,
                 id<MTLBuffer>                                      destination,
                 std::size_t                                        bytes)
{
  const auto start = Clock::now();
  id<MTLBlitCommandEncoder> encoder = blit_encoder(session);
  [encoder copyFromBuffer:source
              sourceOffset:0
                  toBuffer:destination
         destinationOffset:0
                      size:bytes];
  [encoder endEncoding];
  session->stats.encoding_ms += elapsed_ms(start);
  session->has_work = true;
}

id<MTLTexture> new_session_texture(
    const std::shared_ptr<detail::DeviceSessionState> &session,
    glm::ivec2                                             shape,
    StorageMode                                            mode,
    std::size_t                                            bytes)
{
  const auto start = Clock::now();
  MTLTextureDescriptor *descriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                         width:static_cast<NSUInteger>(shape.x)
                                                        height:static_cast<NSUInteger>(shape.y)
                                                     mipmapped:NO];
  descriptor.storageMode = mode == StorageMode::private_storage
                               ? MTLStorageModePrivate
                               : MTLStorageModeShared;
  descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  id<MTLTexture> texture = [context().device newTextureWithDescriptor:descriptor];
  if (!texture) throw std::runtime_error("Metal DeviceArray texture allocation failed");
  session->stats.allocation_ms += elapsed_ms(start);
  ++session->stats.texture_allocations;
  session->stats.bytes_allocated += bytes;
  session->live_bytes += bytes;
  session->stats.resident_bytes = session->live_bytes;
  session->stats.peak_resident_bytes = std::max<std::uint64_t>(
      session->stats.peak_resident_bytes, session->live_bytes);
  return texture;
}

void release_session_texture(
    const std::shared_ptr<detail::DeviceSessionState> &session,
    std::size_t                                            bytes)
{
  if (session->live_bytes >= bytes)
    session->live_bytes -= bytes;
  else
    session->live_bytes = 0;
  session->stats.resident_bytes = session->live_bytes;
}

void encode_buffer_to_texture(
    const std::shared_ptr<detail::DeviceSessionState> &session,
    id<MTLBuffer>                                      source,
    id<MTLTexture>                                     destination,
    glm::ivec2                                         shape,
    std::size_t                                        bytes)
{
  const auto start = Clock::now();
  id<MTLBlitCommandEncoder> encoder = blit_encoder(session);
  [encoder copyFromBuffer:source
             sourceOffset:0
        sourceBytesPerRow:static_cast<NSUInteger>(shape.x * sizeof(float))
      sourceBytesPerImage:static_cast<NSUInteger>(bytes)
               sourceSize:MTLSizeMake(static_cast<NSUInteger>(shape.x),
                                      static_cast<NSUInteger>(shape.y),
                                      1)
                toTexture:destination
         destinationSlice:0
         destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
  [encoder endEncoding];
  session->stats.encoding_ms += elapsed_ms(start);
  session->has_work = true;
}

void encode_texture_to_buffer(
    const std::shared_ptr<detail::DeviceSessionState> &session,
    id<MTLTexture>                                     source,
    id<MTLBuffer>                                      destination,
    glm::ivec2                                         shape,
    std::size_t                                        bytes)
{
  const auto start = Clock::now();
  id<MTLBlitCommandEncoder> encoder = blit_encoder(session);
  [encoder copyFromTexture:source
             sourceSlice:0
             sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(static_cast<NSUInteger>(shape.x),
                                      static_cast<NSUInteger>(shape.y),
                                      1)
                  toBuffer:destination
         destinationOffset:0
    destinationBytesPerRow:static_cast<NSUInteger>(shape.x * sizeof(float))
  destinationBytesPerImage:static_cast<NSUInteger>(bytes)];
  [encoder endEncoding];
  session->stats.encoding_ms += elapsed_ms(start);
  session->has_work = true;
}

void initialize_zero(const std::shared_ptr<detail::DeviceSessionState> &session,
                     id<MTLBuffer>                                      buffer,
                     std::size_t                                        bytes,
                     StorageMode                                        mode)
{
  if (mode == StorageMode::shared)
  {
    std::memset([buffer contents], 0, bytes);
    return;
  }

  const auto start = Clock::now();
  id<MTLBlitCommandEncoder> encoder = blit_encoder(session);
  [encoder fillBuffer:buffer range:NSMakeRange(0, bytes) value:0];
  [encoder endEncoding];
  session->stats.encoding_ms += elapsed_ms(start);
  session->has_work = true;
}

std::shared_ptr<detail::DeviceArrayState> make_array_state(
    const std::shared_ptr<detail::DeviceSessionState> &session,
    glm::ivec2                                        shape,
    id<MTLBuffer>                                      buffer,
    StorageMode                                        mode,
    ResidencyState                                     residency)
{
  auto result = std::make_shared<detail::DeviceArrayState>();
  result->session = session;
  result->buffer = buffer;
  result->shape = shape;
  result->storage = mode;
  result->residency = residency;
  result->byte_size = static_cast<std::size_t>(shape.x) *
                      static_cast<std::size_t>(shape.y) * sizeof(float);
  return result;
}

id<MTLBuffer> upload_session_values(
    const std::shared_ptr<detail::DeviceSessionState> &session,
    const std::vector<float>                           &values,
    StorageMode                                         mode)
{
  const std::size_t bytes = values.size() * sizeof(float);
  if (bytes == 0) throw std::invalid_argument("Metal cannot upload an empty array");

  id<MTLBuffer> destination = new_session_buffer(session, bytes, mode);
  const auto upload_start = Clock::now();
  if (mode == StorageMode::shared)
  {
    std::memcpy([destination contents], values.data(), bytes);
  }
  else
  {
    id<MTLBuffer> staging =
        new_session_buffer(session, bytes, StorageMode::shared);
    std::memcpy([staging contents], values.data(), bytes);
    session->upload_staging.push_back(staging);
    encode_copy(session, staging, destination, bytes);
  }
  session->stats.upload_ms += elapsed_ms(upload_start);
  session->stats.upload_bytes += bytes;
  return destination;
}

void recycle_unique_state(
    const std::shared_ptr<detail::DeviceArrayState> &array)
{
  if (!array || !array->buffer || array.use_count() != 1) return;
  release_session_buffer(array->session,
                         array->buffer,
                         array->byte_size,
                         array->storage);
  array->buffer = nil;
}

void finish_session(const std::shared_ptr<detail::DeviceSessionState> &session)
{
  if (!session || session->finished) return;
  id<MTLCommandBuffer> wait_buffer = nil;
  if (session->has_work)
  {
    wait_buffer = session->command_buffer;
  }
  else if (!session->submitted_command_buffers.empty())
    wait_buffer = session->submitted_command_buffers.back();
  if (wait_buffer)
    wait_for_completion(wait_buffer,
                        &session->stats,
                        session->has_work);
  session->finished = true;
  session->command_buffer = nil;
  session->submitted_command_buffers.clear();
  session->upload_staging.clear();
}

void prepare_download_state(
    const std::shared_ptr<detail::DeviceArrayState> &array)
{
  if (!array || array->storage != StorageMode::private_storage ||
      array->download_staging)
    return;
  require_session_open(array->session);
  array->download_staging = new_session_buffer(
      array->session, array->byte_size, StorageMode::shared);
  encode_copy(array->session,
              array->buffer,
              array->download_staging,
              array->byte_size);
}

Array download_array_state(
    const std::shared_ptr<detail::DeviceArrayState> &array)
{
  if (!array || !array->buffer)
    throw std::invalid_argument("Cannot download an empty Metal DeviceArray");
  auto session = array->session;
  if (!session) throw std::runtime_error("Metal DeviceArray has no session");

  prepare_download_state(array);

  finish_session(session);
  id<MTLBuffer> source = array->storage == StorageMode::private_storage
                             ? array->download_staging
                             : array->buffer;
  Array result(array->shape);
  const auto start = Clock::now();
  std::memcpy(result.vector.data(),
              [source contents],
              array->byte_size);
  session->stats.readback_ms += elapsed_ms(start);
  session->stats.readback_bytes += array->byte_size;
  array->residency = ResidencyState::both_valid;
  return result;
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

bool supports_noise_fbm(NoiseType noise_type)
{
  // The resident FBM kernel shares the staged backend's base-noise coverage.
  // The remaining HighMap FBM families have materially different algorithms
  // and remain on their established CPU/OpenCL paths.
  return supports_noise(noise_type);
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

void thermal_ridge(Array &z, const Array &talus, int iterations)
{
  DeviceSession session;
  auto device_z = session.upload(z);
  auto device_talus = session.upload(talus);
  auto result = session.thermal_ridge(std::move(device_z),
                                      device_talus,
                                      iterations);
  result = session.extrapolate_borders(std::move(result));
  z = session.download(result);
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

DeviceArray DeviceSession::gradient_norm(DeviceArray array)
{
  require_session_open(state_);
  require_input(state_, array.state_, "gradient input");
  const glm::ivec2 shape = array.state_->shape;
  const std::size_t bytes = array.state_->byte_size;
  const StorageMode mode = array.state_->storage;
  id<MTLBuffer> output = acquire_session_buffer(state_, bytes, mode);
  auto result = make_array_state(state_,
                                 shape,
                                 output,
                                 mode,
                                 ResidencyState::device_valid);

  GridParams params{shape.x, shape.y};
  id<MTLComputePipelineState> pipeline =
      context().pipeline("gradient_norm", &state_->stats);
  const auto encoding_start = Clock::now();
  id<MTLComputeCommandEncoder> encoder =
      compute_encoder(state_->command_buffer, &state_->stats);
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:array.state_->buffer offset:0 atIndex:0];
  [encoder setBuffer:output offset:0 atIndex:1];
  set_bytes(encoder, &params, sizeof(params), 2);
  dispatch(encoder, pipeline, shape.x, shape.y, "gradient_norm");
  [encoder endEncoding];
  record_encoding(encoding_start, &state_->stats);
  state_->has_work = true;
  recycle_unique_state(array.state_);
  return DeviceArray(std::move(result));
}

DeviceArray DeviceSession::maximum_smooth(DeviceArray        array1,
                                           const DeviceArray &array2,
                                           float              k)
{
  require_session_open(state_);
  require_input(state_, array1.state_, "smooth input 1");
  require_input(state_, array2.state_, "smooth input 2");
  require_same_shape(array2.state_, array1.state_->shape, "smooth input 2");
  const glm::ivec2 shape = array1.state_->shape;
  const StorageMode mode = array1.state_->storage;
  id<MTLBuffer> output = acquire_session_buffer(state_,
                                                array1.state_->byte_size,
                                                mode);
  auto result = make_array_state(state_,
                                 shape,
                                 output,
                                 mode,
                                 ResidencyState::device_valid);
  BinarySmoothParams params{shape.x, shape.y, k};
  id<MTLComputePipelineState> pipeline =
      context().pipeline("maximum_smooth", &state_->stats);
  const auto encoding_start = Clock::now();
  id<MTLComputeCommandEncoder> encoder =
      compute_encoder(state_->command_buffer, &state_->stats);
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:array1.state_->buffer offset:0 atIndex:0];
  [encoder setBuffer:array2.state_->buffer offset:0 atIndex:1];
  [encoder setBuffer:output offset:0 atIndex:2];
  set_bytes(encoder, &params, sizeof(params), 3);
  dispatch(encoder, pipeline, shape.x, shape.y, "maximum_smooth");
  [encoder endEncoding];
  record_encoding(encoding_start, &state_->stats);
  state_->has_work = true;
  recycle_unique_state(array1.state_);
  return DeviceArray(std::move(result));
}

DeviceArray DeviceSession::minimum_smooth(DeviceArray        array1,
                                           const DeviceArray &array2,
                                           float              k)
{
  require_session_open(state_);
  require_input(state_, array1.state_, "smooth input 1");
  require_input(state_, array2.state_, "smooth input 2");
  require_same_shape(array2.state_, array1.state_->shape, "smooth input 2");
  const glm::ivec2 shape = array1.state_->shape;
  const StorageMode mode = array1.state_->storage;
  id<MTLBuffer> output = acquire_session_buffer(state_,
                                                array1.state_->byte_size,
                                                mode);
  auto result = make_array_state(state_,
                                 shape,
                                 output,
                                 mode,
                                 ResidencyState::device_valid);
  BinarySmoothParams params{shape.x, shape.y, k};
  id<MTLComputePipelineState> pipeline =
      context().pipeline("minimum_smooth", &state_->stats);
  const auto encoding_start = Clock::now();
  id<MTLComputeCommandEncoder> encoder =
      compute_encoder(state_->command_buffer, &state_->stats);
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:array1.state_->buffer offset:0 atIndex:0];
  [encoder setBuffer:array2.state_->buffer offset:0 atIndex:1];
  [encoder setBuffer:output offset:0 atIndex:2];
  set_bytes(encoder, &params, sizeof(params), 3);
  dispatch(encoder, pipeline, shape.x, shape.y, "minimum_smooth");
  [encoder endEncoding];
  record_encoding(encoding_start, &state_->stats);
  state_->has_work = true;
  recycle_unique_state(array1.state_);
  return DeviceArray(std::move(result));
}

DeviceArray DeviceSession::noise(NoiseType     noise_type,
                                 glm::ivec2    shape,
                                 glm::vec2     kw,
                                 std::uint32_t seed,
                                 const DeviceArray *p_noise_x,
                                 const DeviceArray *p_noise_y,
                                 glm::vec4      bbox,
                                 glm::ivec2     period)
{
  require_session_open(state_);
  check_shape_2d(shape);
  if (!supports_noise(noise_type))
    throw std::invalid_argument("Metal resident noise type is unsupported");

  const StorageMode mode = state_->default_storage;
  if (p_noise_x)
  {
    require_input(state_, p_noise_x->state_, "noise_x");
    require_same_shape(p_noise_x->state_, shape, "noise_x");
  }
  if (p_noise_y)
  {
    require_input(state_, p_noise_y->state_, "noise_y");
    require_same_shape(p_noise_y->state_, shape, "noise_y");
  }

  const std::size_t count = static_cast<std::size_t>(shape.x) *
                            static_cast<std::size_t>(shape.y);
  const std::size_t bytes = count * sizeof(float);
  id<MTLBuffer> output = acquire_session_buffer(state_, bytes, mode);
  id<MTLBuffer> noise_x = p_noise_x
                              ? p_noise_x->state_->buffer
                              : new_session_buffer(state_, sizeof(float), mode);
  id<MTLBuffer> noise_y = p_noise_y
                              ? p_noise_y->state_->buffer
                              : new_session_buffer(state_, sizeof(float), mode);
  auto result = make_array_state(state_,
                                 shape,
                                 output,
                                 mode,
                                 ResidencyState::device_valid);

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
  id<MTLComputePipelineState> pipeline =
      context().pipeline("noise", &state_->stats);
  const auto encoding_start = Clock::now();
  id<MTLComputeCommandEncoder> encoder =
      compute_encoder(state_->command_buffer, &state_->stats);
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:output offset:0 atIndex:0];
  [encoder setBuffer:noise_x offset:0 atIndex:1];
  [encoder setBuffer:noise_y offset:0 atIndex:2];
  set_bytes(encoder, &params, sizeof(params), 3);
  dispatch(encoder, pipeline, shape.x, shape.y, "noise");
  [encoder endEncoding];
  record_encoding(encoding_start, &state_->stats);
  state_->has_work = true;
  return DeviceArray(std::move(result));
}

DeviceArray DeviceSession::noise_fbm(NoiseType          noise_type,
                                     glm::ivec2         shape,
                                     glm::vec2          kw,
                                     std::uint32_t      seed,
                                     int                octaves,
                                     float              weight,
                                     float              persistence,
                                     float              lacunarity,
                                     const DeviceArray *p_ctrl_param,
                                     const DeviceArray *p_noise_x,
                                     const DeviceArray *p_noise_y,
                                     glm::vec4          bbox,
                                     glm::ivec2         period)
{
  require_session_open(state_);
  check_shape_2d(shape);
  if (!supports_noise_fbm(noise_type))
    throw std::invalid_argument("Metal resident FBM noise type is unsupported");
  if (octaves < 0)
    throw std::invalid_argument("Metal resident FBM octaves must be non-negative");

  if (p_ctrl_param)
  {
    require_input(state_, p_ctrl_param->state_, "ctrl_param");
    require_same_shape(p_ctrl_param->state_, shape, "ctrl_param");
  }
  if (p_noise_x)
  {
    require_input(state_, p_noise_x->state_, "noise_x");
    require_same_shape(p_noise_x->state_, shape, "noise_x");
  }
  if (p_noise_y)
  {
    require_input(state_, p_noise_y->state_, "noise_y");
    require_same_shape(p_noise_y->state_, shape, "noise_y");
  }

  const StorageMode mode = state_->default_storage;
  const std::size_t count = static_cast<std::size_t>(shape.x) *
                            static_cast<std::size_t>(shape.y);
  const std::size_t bytes = count * sizeof(float);
  id<MTLBuffer> output = acquire_session_buffer(state_, bytes, mode);
  id<MTLBuffer> ctrl = p_ctrl_param
                           ? p_ctrl_param->state_->buffer
                           : new_session_buffer(state_, sizeof(float), mode);
  id<MTLBuffer> noise_x = p_noise_x
                              ? p_noise_x->state_->buffer
                              : new_session_buffer(state_, sizeof(float), mode);
  id<MTLBuffer> noise_y = p_noise_y
                              ? p_noise_y->state_->buffer
                              : new_session_buffer(state_, sizeof(float), mode);
  auto result = make_array_state(state_,
                                 shape,
                                 output,
                                 mode,
                                 ResidencyState::device_valid);

  NoiseFbmParams params{shape.x,
                        shape.y,
                        static_cast<int>(noise_type),
                        kw.x,
                        kw.y,
                        seed,
                        octaves,
                        weight,
                        persistence,
                        lacunarity,
                        p_ctrl_param ? 1 : 0,
                        p_noise_x ? 1 : 0,
                        p_noise_y ? 1 : 0,
                        period.x,
                        period.y,
                        bbox.x,
                        bbox.y,
                        bbox.z,
                        bbox.w};
  id<MTLComputePipelineState> pipeline =
      context().pipeline("noise_fbm", &state_->stats);
  const auto encoding_start = Clock::now();
  id<MTLComputeCommandEncoder> encoder =
      compute_encoder(state_->command_buffer, &state_->stats);
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:output offset:0 atIndex:0];
  [encoder setBuffer:ctrl offset:0 atIndex:1];
  [encoder setBuffer:noise_x offset:0 atIndex:2];
  [encoder setBuffer:noise_y offset:0 atIndex:3];
  set_bytes(encoder, &params, sizeof(params), 4);
  dispatch(encoder, pipeline, shape.x, shape.y, "noise_fbm");
  [encoder endEncoding];
  record_encoding(encoding_start, &state_->stats);
  state_->has_work = true;
  return DeviceArray(std::move(result));
}

DeviceArray DeviceSession::smooth_cpulse(DeviceArray array, int ir)
{
  require_session_open(state_);
  require_input(state_, array.state_, "smooth input");
  if (ir <= 0) return array;

  const glm::ivec2 shape = array.state_->shape;
  const StorageMode mode = array.state_->storage;
  const std::size_t bytes = array.state_->byte_size;
  float weight_sum = 0.f;
  for (int k = -ir; k <= ir; ++k)
  {
    const float d = std::abs(static_cast<float>(k)) / static_cast<float>(ir);
    weight_sum += std::exp(-0.5f * d * d * 9.f);
  }

  id<MTLBuffer> first = acquire_session_buffer(state_, bytes, mode);
  id<MTLBuffer> second = acquire_session_buffer(state_, bytes, mode);
  id<MTLComputePipelineState> pipeline =
      context().pipeline("smooth_cpulse", &state_->stats);
  SmoothCpulseParams params{shape.x, shape.y, ir, 0, weight_sum};
  const auto encoding_start = Clock::now();
  id<MTLComputeCommandEncoder> encoder =
      compute_encoder(state_->command_buffer, &state_->stats);
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:array.state_->buffer offset:0 atIndex:0];
  [encoder setBuffer:first offset:0 atIndex:1];
  set_bytes(encoder, &params, sizeof(params), 2);
  dispatch(encoder, pipeline, shape.x, shape.y, "smooth_cpulse");
  [encoder endEncoding];

  params.pass = 1;
  encoder = compute_encoder(state_->command_buffer, &state_->stats);
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:first offset:0 atIndex:0];
  [encoder setBuffer:second offset:0 atIndex:1];
  set_bytes(encoder, &params, sizeof(params), 2);
  dispatch(encoder, pipeline, shape.x, shape.y, "smooth_cpulse");
  [encoder endEncoding];
  record_encoding(encoding_start, &state_->stats);
  state_->has_work = true;

  release_session_buffer(state_, first, bytes, mode);
  recycle_unique_state(array.state_);
  return DeviceArray(make_array_state(state_,
                                      shape,
                                      second,
                                      mode,
                                      ResidencyState::device_valid));
}

DeviceArray DeviceSession::spectral_equalizer(
    DeviceArray              array,
    const std::vector<float> &weights,
    int                       ir_min,
    int                       ir_max)
{
  require_session_open(state_);
  require_input(state_, array.state_, "spectral input");
  if (weights.empty()) return array;
  if (ir_min <= 0 || ir_max <= 0)
    throw std::invalid_argument("Metal spectral radii must be positive");

  const std::size_t nbands = weights.size();
  std::vector<int> radii;
  radii.reserve(nbands > 0 ? nbands - 1 : 0);
  int previous = 0;
  if (nbands > 1)
  {
    const float log_min = std::log(static_cast<float>(ir_min));
    const float log_max = std::log(static_cast<float>(ir_max));
    for (std::size_t i = 0; i + 1 < nbands; ++i)
    {
      const float t = static_cast<float>(i) /
                      static_cast<float>(nbands - 2);
      int radius = std::max(
          1,
          static_cast<int>(std::exp(log_min + t * (log_max - log_min)) + 0.5f));
      if (radius <= previous) radius = previous + 1;
      previous = radius;
      radii.push_back(radius);
    }
  }

  std::vector<DeviceArray> blurred;
  blurred.reserve(nbands);
  blurred.push_back(std::move(array));
  const DeviceArray source = blurred.front();
  for (const int radius : radii)
    blurred.push_back(this->smooth_cpulse(source, radius));

  DeviceArray output;
  for (std::size_t k = 0; k < nbands; ++k)
  {
    DeviceArray band;
    if (k + 1 < nbands)
      band = this->linear_combine(blurred[k], blurred[k + 1], 1.f, -1.f);
    else
      band = blurred[k];

    const float band_weight = weights[nbands - 1 - k];
    if (output.empty())
    {
      const DeviceArray band_reference = band;
      output = this->linear_combine(std::move(band),
                                    band_reference,
                                    band_weight,
                                    0.f);
    }
    else
      output = this->linear_combine(std::move(output), band, 1.f, band_weight);
  }
  return output;
}

DeviceArray DeviceSession::normalize(DeviceArray array, float vmin, float vmax)
{
  require_session_open(state_);
  require_input(state_, array.state_, "normalize input");
  const glm::ivec2 shape = array.state_->shape;
  const StorageMode mode = array.state_->storage;
  const std::size_t bytes = array.state_->byte_size;
  const auto range = reduce_range(state_, array.state_->buffer, array.size());
  id<MTLBuffer> output = acquire_session_buffer(state_, bytes, mode);
  NormalizeParams params{shape.x, shape.y, range.first, range.second, vmin, vmax};
  id<MTLComputePipelineState> pipeline =
      context().pipeline("normalize", &state_->stats);
  const auto encoding_start = Clock::now();
  id<MTLComputeCommandEncoder> encoder =
      compute_encoder(state_->command_buffer, &state_->stats);
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:array.state_->buffer offset:0 atIndex:0];
  [encoder setBuffer:output offset:0 atIndex:1];
  set_bytes(encoder, &params, sizeof(params), 2);
  dispatch(encoder, pipeline, shape.x, shape.y, "normalize");
  [encoder endEncoding];
  record_encoding(encoding_start, &state_->stats);
  state_->has_work = true;
  recycle_unique_state(array.state_);
  return DeviceArray(make_array_state(state_,
                                      shape,
                                      output,
                                      mode,
                                      ResidencyState::device_valid));
}

DeviceArray DeviceSession::advection_warp(DeviceArray        z,
                                           const DeviceArray &advected_field,
                                           const DeviceArray &dx,
                                           const DeviceArray &dy,
                                           float              advection_length,
                                           float              value_persistence,
                                           const DeviceArray *p_mask)
{
  require_session_open(state_);
  require_input(state_, z.state_, "advection z");
  require_input(state_, advected_field.state_, "advection field");
  require_input(state_, dx.state_, "advection dx");
  require_input(state_, dy.state_, "advection dy");
  const glm::ivec2 shape = z.state_->shape;
  require_same_shape(advected_field.state_, shape, "advection field");
  require_same_shape(dx.state_, shape, "advection dx");
  require_same_shape(dy.state_, shape, "advection dy");
  if (p_mask)
  {
    require_input(state_, p_mask->state_, "advection mask");
    require_same_shape(p_mask->state_, shape, "advection mask");
  }

  const StorageMode mode = z.state_->storage;
  const std::size_t bytes = z.state_->byte_size;
  id<MTLBuffer> output = acquire_session_buffer(state_, bytes, mode);
  id<MTLBuffer> mask = p_mask ? p_mask->state_->buffer : nil;
  if (!mask)
  {
    std::vector<float> ones(z.size(), 1.f);
    mask = upload_session_values(state_, ones, mode);
  }
  auto result = make_array_state(state_,
                                 shape,
                                 output,
                                 mode,
                                 ResidencyState::device_valid);

  AdvectionParams params{shape.x,
                         shape.y,
                         advection_length,
                         value_persistence};
  id<MTLComputePipelineState> pipeline =
      context().pipeline("advection_warp", &state_->stats);
  const auto encoding_start = Clock::now();
  id<MTLComputeCommandEncoder> encoder =
      compute_encoder(state_->command_buffer, &state_->stats);
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:z.state_->buffer offset:0 atIndex:0];
  [encoder setBuffer:advected_field.state_->buffer offset:0 atIndex:1];
  [encoder setBuffer:dx.state_->buffer offset:0 atIndex:2];
  [encoder setBuffer:dy.state_->buffer offset:0 atIndex:3];
  [encoder setBuffer:mask offset:0 atIndex:4];
  [encoder setBuffer:output offset:0 atIndex:5];
  set_bytes(encoder, &params, sizeof(params), 6);
  dispatch(encoder, pipeline, shape.x, shape.y, "advection_warp");
  [encoder endEncoding];
  record_encoding(encoding_start, &state_->stats);
  state_->has_work = true;
  recycle_unique_state(z.state_);
  return DeviceArray(std::move(result));
}

DeviceArray DeviceSession::advection_warp_texture(
    DeviceArray              z,
    const DeviceArray       &advected_field,
    const DeviceArray       &dx,
    const DeviceArray       &dy,
    float                    advection_length,
    float                    value_persistence,
    const DeviceArray       *p_mask)
{
  require_session_open(state_);
  require_input(state_, z.state_, "texture advection z");
  require_input(state_, advected_field.state_, "texture advection field");
  require_input(state_, dx.state_, "texture advection dx");
  require_input(state_, dy.state_, "texture advection dy");
  const glm::ivec2 shape = z.state_->shape;
  require_same_shape(advected_field.state_, shape, "texture advection field");
  require_same_shape(dx.state_, shape, "texture advection dx");
  require_same_shape(dy.state_, shape, "texture advection dy");
  if (p_mask)
  {
    require_input(state_, p_mask->state_, "texture advection mask");
    require_same_shape(p_mask->state_, shape, "texture advection mask");
  }

  const StorageMode mode = z.state_->storage;
  const std::size_t bytes = z.state_->byte_size;
  id<MTLBuffer> output = acquire_session_buffer(state_, bytes, mode);
  id<MTLBuffer> mask = p_mask ? p_mask->state_->buffer : nil;
  if (!mask)
  {
    std::vector<float> ones(z.size(), 1.f);
    mask = upload_session_values(state_, ones, mode);
  }

  id<MTLTexture> z_texture =
      new_session_texture(state_, shape, mode, bytes);
  id<MTLTexture> field_texture =
      new_session_texture(state_, shape, mode, bytes);
  id<MTLTexture> dx_texture =
      new_session_texture(state_, shape, mode, bytes);
  id<MTLTexture> dy_texture =
      new_session_texture(state_, shape, mode, bytes);
  id<MTLTexture> mask_texture =
      new_session_texture(state_, shape, mode, bytes);
  id<MTLTexture> output_texture =
      new_session_texture(state_, shape, mode, bytes);
  encode_buffer_to_texture(state_, z.state_->buffer, z_texture, shape, bytes);
  encode_buffer_to_texture(
      state_, advected_field.state_->buffer, field_texture, shape, bytes);
  encode_buffer_to_texture(state_, dx.state_->buffer, dx_texture, shape, bytes);
  encode_buffer_to_texture(state_, dy.state_->buffer, dy_texture, shape, bytes);
  encode_buffer_to_texture(state_, mask, mask_texture, shape, bytes);

  auto result = make_array_state(state_,
                                 shape,
                                 output,
                                 mode,
                                 ResidencyState::device_valid);
  AdvectionParams params{shape.x,
                         shape.y,
                         advection_length,
                         value_persistence};
  id<MTLComputePipelineState> pipeline =
      context().pipeline("advection_warp_texture", &state_->stats);
  const auto encoding_start = Clock::now();
  id<MTLComputeCommandEncoder> encoder =
      compute_encoder(state_->command_buffer, &state_->stats);
  [encoder setComputePipelineState:pipeline];
  [encoder setTexture:z_texture atIndex:0];
  [encoder setTexture:field_texture atIndex:1];
  [encoder setTexture:dx_texture atIndex:2];
  [encoder setTexture:dy_texture atIndex:3];
  [encoder setTexture:mask_texture atIndex:4];
  [encoder setTexture:output_texture atIndex:5];
  set_bytes(encoder, &params, sizeof(params), 0);
  dispatch(encoder, pipeline, shape.x, shape.y, "advection_texture");
  [encoder endEncoding];
  record_encoding(encoding_start, &state_->stats);
  encode_texture_to_buffer(state_, output_texture, output, shape, bytes);
  state_->has_work = true;
  if (!p_mask) release_session_buffer(state_, mask, bytes, mode);
  release_session_texture(state_, bytes * 6);
  recycle_unique_state(z.state_);
  return DeviceArray(std::move(result));
}

DeviceArray DeviceSession::thermal(DeviceArray        z,
                                   const DeviceArray &talus,
                                   int                iterations)
{
  require_session_open(state_);
  require_input(state_, z.state_, "thermal input");
  require_input(state_, talus.state_, "thermal talus");
  require_same_shape(talus.state_, z.state_->shape, "thermal talus");
  if (iterations <= 0) return z;

  const glm::ivec2 shape = z.state_->shape;
  const StorageMode mode = z.state_->storage;
  const std::size_t bytes = z.state_->byte_size;
  id<MTLBuffer> first = acquire_session_buffer(state_, bytes, mode);
  id<MTLBuffer> second = acquire_session_buffer(state_, bytes, mode);
  id<MTLComputePipelineState> pipeline =
      context().pipeline("thermal_pass", &state_->stats);
  ThermalParams params{shape.x, shape.y};
  id<MTLBuffer> input = z.state_->buffer;
  id<MTLBuffer> output = first;
  const auto encoding_start = Clock::now();
  for (int it = 0; it < iterations; ++it)
  {
    id<MTLComputeCommandEncoder> encoder =
        compute_encoder(state_->command_buffer, &state_->stats);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:talus.state_->buffer offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    set_bytes(encoder, &params, sizeof(params), 3);
    dispatch(encoder, pipeline, shape.x, shape.y, "thermal_pass");
    [encoder endEncoding];
    input = output;
    output = output == first ? second : first;
  }
  record_encoding(encoding_start, &state_->stats);
  state_->has_work = true;
  release_session_buffer(state_, output, bytes, mode);
  recycle_unique_state(z.state_);
  return DeviceArray(make_array_state(state_,
                                      shape,
                                      input,
                                      mode,
                                      ResidencyState::device_valid));
}

DeviceArray DeviceSession::thermal_ridge(DeviceArray        z,
                                         const DeviceArray &talus,
                                         int                iterations)
{
  require_session_open(state_);
  require_input(state_, z.state_, "thermal ridge input");
  require_input(state_, talus.state_, "thermal ridge talus");
  require_same_shape(talus.state_, z.state_->shape, "thermal ridge talus");
  if (iterations <= 0) return z;

  const glm::ivec2 shape = z.state_->shape;
  const StorageMode mode = z.state_->storage;
  const std::size_t bytes = z.state_->byte_size;
  id<MTLBuffer> first = acquire_session_buffer(state_, bytes, mode);
  id<MTLBuffer> second = acquire_session_buffer(state_, bytes, mode);
  id<MTLComputePipelineState> pipeline =
      context().pipeline("thermal_ridge_pass", &state_->stats);
  ThermalParams params{shape.x, shape.y};
  id<MTLBuffer> input = z.state_->buffer;
  id<MTLBuffer> output = first;
  const auto encoding_start = Clock::now();
  for (int it = 0; it < iterations; ++it)
  {
    id<MTLComputeCommandEncoder> encoder =
        compute_encoder(state_->command_buffer, &state_->stats);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:talus.state_->buffer offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    set_bytes(encoder, &params, sizeof(params), 3);
    dispatch(encoder, pipeline, shape.x, shape.y, "thermal_ridge_pass");
    [encoder endEncoding];
    input = output;
    output = output == first ? second : first;
  }
  record_encoding(encoding_start, &state_->stats);
  state_->has_work = true;
  release_session_buffer(state_, output, bytes, mode);
  recycle_unique_state(z.state_);
  return DeviceArray(make_array_state(state_,
                                      shape,
                                      input,
                                      mode,
                                      ResidencyState::device_valid));
}

DeviceArray DeviceSession::extrapolate_borders(DeviceArray z)
{
  require_session_open(state_);
  require_input(state_, z.state_, "border extrapolation input");
  const glm::ivec2 shape = z.state_->shape;
  if (shape.x < 3 || shape.y < 3) return z;

  const StorageMode mode = z.state_->storage;
  const std::size_t bytes = z.state_->byte_size;
  id<MTLBuffer> horizontal = acquire_session_buffer(state_, bytes, mode);
  id<MTLBuffer> vertical = acquire_session_buffer(state_, bytes, mode);
  id<MTLComputePipelineState> horizontal_pipeline =
      context().pipeline("extrapolate_horizontal", &state_->stats);
  id<MTLComputePipelineState> vertical_pipeline =
      context().pipeline("extrapolate_vertical", &state_->stats);
  BorderParams params{shape.x, shape.y};

  const auto encoding_start = Clock::now();
  id<MTLComputeCommandEncoder> encoder =
      compute_encoder(state_->command_buffer, &state_->stats);
  [encoder setComputePipelineState:horizontal_pipeline];
  [encoder setBuffer:z.state_->buffer offset:0 atIndex:0];
  [encoder setBuffer:horizontal offset:0 atIndex:1];
  set_bytes(encoder, &params, sizeof(params), 2);
  dispatch(encoder, horizontal_pipeline, shape.x, shape.y, "extrapolate_horizontal");
  [encoder endEncoding];

  encoder = compute_encoder(state_->command_buffer, &state_->stats);
  [encoder setComputePipelineState:vertical_pipeline];
  [encoder setBuffer:horizontal offset:0 atIndex:0];
  [encoder setBuffer:vertical offset:0 atIndex:1];
  set_bytes(encoder, &params, sizeof(params), 2);
  dispatch(encoder, vertical_pipeline, shape.x, shape.y, "extrapolate_vertical");
  [encoder endEncoding];

  record_encoding(encoding_start, &state_->stats);
  state_->has_work = true;
  release_session_buffer(state_, horizontal, bytes, mode);
  recycle_unique_state(z.state_);
  return DeviceArray(make_array_state(state_,
                                      shape,
                                      vertical,
                                      mode,
                                      ResidencyState::device_valid));
}

DeviceArray DeviceSession::linear_combine(DeviceArray        array1,
                                          const DeviceArray &array2,
                                          float              weight1,
                                          float              weight2)
{
  require_session_open(state_);
  require_input(state_, array1.state_, "linear combine input 1");
  require_input(state_, array2.state_, "linear combine input 2");
  require_same_shape(array2.state_, array1.state_->shape, "linear combine input 2");
  const glm::ivec2 shape = array1.state_->shape;
  const StorageMode mode = array1.state_->storage;
  const std::size_t bytes = array1.state_->byte_size;
  id<MTLBuffer> output = acquire_session_buffer(state_, bytes, mode);
  auto result = make_array_state(state_,
                                 shape,
                                 output,
                                 mode,
                                 ResidencyState::device_valid);
  LinearCombineParams params{shape.x, shape.y, weight1, weight2};
  id<MTLComputePipelineState> pipeline =
      context().pipeline("linear_combine", &state_->stats);
  const auto encoding_start = Clock::now();
  id<MTLComputeCommandEncoder> encoder =
      compute_encoder(state_->command_buffer, &state_->stats);
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:array1.state_->buffer offset:0 atIndex:0];
  [encoder setBuffer:array2.state_->buffer offset:0 atIndex:1];
  [encoder setBuffer:output offset:0 atIndex:2];
  set_bytes(encoder, &params, sizeof(params), 3);
  dispatch(encoder, pipeline, shape.x, shape.y, "linear_combine");
  [encoder endEncoding];
  record_encoding(encoding_start, &state_->stats);
  state_->has_work = true;
  recycle_unique_state(array1.state_);
  return DeviceArray(std::move(result));
}

DeviceArray DeviceSession::hydraulic_vpipes(
    DeviceArray        z,
    float              water_height,
    bool               maintain_water_volume,
    float              evap_rate,
    int                iterations,
    float              dt,
    float              k_capacity,
    float              k_erode,
    float              k_depose,
    float              k_discharge_exp,
    float              downcutting_max_depth_ratio,
    bool               flux_diffusion,
    float              flux_diffusion_strength,
    const DeviceArray *p_rain_map,
    DeviceArray       *p_water_depth,
    DeviceArray       *p_sediment,
    DeviceArray       *p_vel_u,
    DeviceArray       *p_vel_v)
{
  require_session_open(state_);
  require_input(state_, z.state_, "hydraulic terrain");
  const glm::ivec2 shape = z.state_->shape;
  const StorageMode mode = z.state_->storage;
  if (p_rain_map)
  {
    require_input(state_, p_rain_map->state_, "hydraulic rain map");
    require_same_shape(p_rain_map->state_, shape, "hydraulic rain map");
    if (!p_rain_map->state_->host_shadow)
      throw std::invalid_argument(
          "Resident hydraulic rain maps require an explicit host upload");
  }

  const std::size_t count = static_cast<std::size_t>(shape.x) *
                            static_cast<std::size_t>(shape.y);
  const std::size_t bytes = count * sizeof(float);
  std::vector<float> rain_host(
      p_rain_map && p_rain_map->state_->host_shadow
          ? *p_rain_map->state_->host_shadow
          : std::vector<float>(count, 1.f));
  const float rain_volume =
      std::accumulate(rain_host.begin(), rain_host.end(), 0.f);
  const float initial_water_volume = water_height * rain_volume;
  std::vector<float> initial_water(count);
  for (std::size_t i = 0; i < count; ++i)
    initial_water[i] = water_height * rain_host[i];

  // Keep the caller's terrain immutable and make all simulation state owned
  // by this session. The copy is a GPU blit, not a host round trip.
  id<MTLBuffer> z_a = acquire_session_buffer(state_, bytes, mode);
  encode_copy(state_, z.state_->buffer, z_a, bytes);
  id<MTLBuffer> z_b = acquire_session_buffer(state_, bytes, mode);
  id<MTLBuffer> rain_buffer = p_rain_map
                                  ? p_rain_map->state_->buffer
                                  : upload_session_values(state_, rain_host, mode);
  id<MTLBuffer> d_a = upload_session_values(state_, initial_water, mode);
  id<MTLBuffer> d_b = acquire_session_buffer(state_, bytes, mode);
  id<MTLBuffer> d2 = acquire_session_buffer(state_, bytes, mode);
  id<MTLBuffer> sediment = acquire_session_buffer(state_, bytes, mode);
  initialize_zero(state_, sediment, bytes, mode);
  id<MTLBuffer> sediment_tmp = acquire_session_buffer(state_, bytes, mode);
  id<MTLBuffer> fl_a = acquire_session_buffer(state_, bytes, mode);
  id<MTLBuffer> fr_a = acquire_session_buffer(state_, bytes, mode);
  id<MTLBuffer> ft_a = acquire_session_buffer(state_, bytes, mode);
  id<MTLBuffer> fb_a = acquire_session_buffer(state_, bytes, mode);
  id<MTLBuffer> fl_b = acquire_session_buffer(state_, bytes, mode);
  id<MTLBuffer> fr_b = acquire_session_buffer(state_, bytes, mode);
  id<MTLBuffer> ft_b = acquire_session_buffer(state_, bytes, mode);
  id<MTLBuffer> fb_b = acquire_session_buffer(state_, bytes, mode);
  id<MTLBuffer> velocity_u = acquire_session_buffer(state_, bytes, mode);
  id<MTLBuffer> velocity_v = acquire_session_buffer(state_, bytes, mode);

  id<MTLComputePipelineState> flow_pipeline =
      context().pipeline("hydraulic_flow_pass", &state_->stats);
  id<MTLComputePipelineState> water_pipeline =
      context().pipeline("hydraulic_water_pass", &state_->stats);
  id<MTLComputePipelineState> erosion_pipeline =
      context().pipeline("hydraulic_erosion_pass", &state_->stats);
  id<MTLComputePipelineState> sediment_pipeline =
      context().pipeline("hydraulic_sediment_pass", &state_->stats);
  id<MTLComputePipelineState> evaporate_pipeline =
      context().pipeline("hydraulic_evaporate", &state_->stats);
  id<MTLComputePipelineState> rescale_pipeline =
      context().pipeline("hydraulic_rescale", &state_->stats);
  id<MTLComputePipelineState> sum_pipeline =
      context().pipeline("hydraulic_sum_tiles", &state_->stats);
  id<MTLComputePipelineState> reduce_pipeline =
      context().pipeline("hydraulic_sum_reduce", &state_->stats);
  const NSUInteger sum_threads = reduction_threads(sum_pipeline);
  const size_t partial_count =
      std::max<size_t>(1, (count + sum_threads - 1) / sum_threads);
  id<MTLBuffer> reduction_a =
      acquire_session_buffer(state_, partial_count * sizeof(float), mode);
  id<MTLBuffer> reduction_b =
      acquire_session_buffer(state_, partial_count * sizeof(float), mode);

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

  id<MTLBuffer> terrain = z_a;
  id<MTLBuffer> terrain_output = z_b;
  id<MTLBuffer> water = d_a;
  id<MTLBuffer> water_output = d_b;
  const auto encoding_start = Clock::now();
  for (int it = 0; it < std::max(0, iterations); ++it)
  {
    id<MTLComputeCommandEncoder> encoder =
        compute_encoder(state_->command_buffer, &state_->stats);
    [encoder setComputePipelineState:flow_pipeline];
    [encoder setBuffer:terrain offset:0 atIndex:0];
    [encoder setBuffer:fl_a offset:0 atIndex:1];
    [encoder setBuffer:fr_a offset:0 atIndex:2];
    [encoder setBuffer:ft_a offset:0 atIndex:3];
    [encoder setBuffer:fb_a offset:0 atIndex:4];
    [encoder setBuffer:water offset:0 atIndex:5];
    [encoder setBuffer:fl_b offset:0 atIndex:6];
    [encoder setBuffer:fr_b offset:0 atIndex:7];
    [encoder setBuffer:ft_b offset:0 atIndex:8];
    [encoder setBuffer:fb_b offset:0 atIndex:9];
    set_bytes(encoder, &params, sizeof(params), 10);
    dispatch(encoder, flow_pipeline, shape.x, shape.y, "hydraulic_flow_pass");
    [encoder endEncoding];

    encoder = compute_encoder(state_->command_buffer, &state_->stats);
    [encoder setComputePipelineState:water_pipeline];
    [encoder setBuffer:terrain offset:0 atIndex:0];
    [encoder setBuffer:fl_b offset:0 atIndex:1];
    [encoder setBuffer:fr_b offset:0 atIndex:2];
    [encoder setBuffer:ft_b offset:0 atIndex:3];
    [encoder setBuffer:fb_b offset:0 atIndex:4];
    [encoder setBuffer:water offset:0 atIndex:5];
    [encoder setBuffer:d2 offset:0 atIndex:6];
    [encoder setBuffer:velocity_u offset:0 atIndex:7];
    [encoder setBuffer:velocity_v offset:0 atIndex:8];
    set_bytes(encoder, &params, sizeof(params), 9);
    dispatch(encoder, water_pipeline, shape.x, shape.y, "hydraulic_water_pass");
    [encoder endEncoding];

    encoder = compute_encoder(state_->command_buffer, &state_->stats);
    [encoder setComputePipelineState:erosion_pipeline];
    [encoder setBuffer:terrain offset:0 atIndex:0];
    [encoder setBuffer:d2 offset:0 atIndex:1];
    [encoder setBuffer:velocity_u offset:0 atIndex:2];
    [encoder setBuffer:velocity_v offset:0 atIndex:3];
    [encoder setBuffer:sediment offset:0 atIndex:4];
    [encoder setBuffer:terrain_output offset:0 atIndex:5];
    [encoder setBuffer:sediment_tmp offset:0 atIndex:6];
    set_bytes(encoder, &params, sizeof(params), 7);
    dispatch(encoder, erosion_pipeline, shape.x, shape.y, "hydraulic_erosion_pass");
    [encoder endEncoding];

    encoder = compute_encoder(state_->command_buffer, &state_->stats);
    [encoder setComputePipelineState:sediment_pipeline];
    [encoder setBuffer:velocity_u offset:0 atIndex:0];
    [encoder setBuffer:velocity_v offset:0 atIndex:1];
    [encoder setBuffer:sediment_tmp offset:0 atIndex:2];
    [encoder setBuffer:sediment offset:0 atIndex:3];
    set_bytes(encoder, &params, sizeof(params), 4);
    dispatch(encoder,
             sediment_pipeline,
             shape.x,
             shape.y,
             "hydraulic_sediment_pass");
    [encoder endEncoding];

    encoder = compute_encoder(state_->command_buffer, &state_->stats);
    [encoder setComputePipelineState:evaporate_pipeline];
    [encoder setBuffer:d2 offset:0 atIndex:0];
    [encoder setBuffer:water_output offset:0 atIndex:1];
    set_bytes(encoder, &params, sizeof(params), 2);
    dispatch(encoder, evaporate_pipeline, shape.x, shape.y, "hydraulic_evaporate");
    [encoder endEncoding];

    if (maintain_water_volume)
    {
      id<MTLBuffer> reduction_input = reduction_a;
      id<MTLBuffer> reduction_output = reduction_b;
      int reduction_count = static_cast<int>(count);
      int next_count = static_cast<int>(partial_count);

      encoder = compute_encoder(state_->command_buffer, &state_->stats);
      [encoder setComputePipelineState:sum_pipeline];
      [encoder setBuffer:water_output offset:0 atIndex:0];
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
        encoder = compute_encoder(state_->command_buffer, &state_->stats);
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

      encoder = compute_encoder(state_->command_buffer, &state_->stats);
      [encoder setComputePipelineState:rescale_pipeline];
      [encoder setBuffer:water_output offset:0 atIndex:0];
      [encoder setBuffer:rain_buffer offset:0 atIndex:1];
      [encoder setBuffer:reduction_input offset:0 atIndex:2];
      set_bytes(encoder, &params, sizeof(params), 3);
      dispatch(encoder, rescale_pipeline, shape.x, shape.y, "hydraulic_rescale");
      [encoder endEncoding];
    }

    std::swap(terrain, terrain_output);
    std::swap(water, water_output);
    std::swap(fl_a, fl_b);
    std::swap(fr_a, fr_b);
    std::swap(ft_a, ft_b);
    std::swap(fb_a, fb_b);
  }
  record_encoding(encoding_start, &state_->stats);
  state_->has_work = true;

  auto terrain_state = make_array_state(state_,
                                        shape,
                                        terrain,
                                        mode,
                                        ResidencyState::device_valid);
  if (p_water_depth)
    p_water_depth->state_ = make_array_state(state_,
                                             shape,
                                             water,
                                             mode,
                                             ResidencyState::device_valid);
  if (p_sediment)
    p_sediment->state_ = make_array_state(state_,
                                          shape,
                                          sediment,
                                          mode,
                                          ResidencyState::device_valid);
  if (p_vel_u)
    p_vel_u->state_ = make_array_state(state_,
                                       shape,
                                       velocity_u,
                                       mode,
                                       ResidencyState::device_valid);
  if (p_vel_v)
    p_vel_v->state_ = make_array_state(state_,
                                       shape,
                                       velocity_v,
                                       mode,
                                       ResidencyState::device_valid);

  prepare_download_state(terrain_state);
  if (p_water_depth) prepare_download_state(p_water_depth->state_);
  if (p_sediment) prepare_download_state(p_sediment->state_);
  if (p_vel_u) prepare_download_state(p_vel_u->state_);
  if (p_vel_v) prepare_download_state(p_vel_v->state_);

  // Recycle state that is no longer exposed by a live DeviceArray. Reuse is
  // session-local and remains ordered after the encoders above.
  release_session_buffer(state_, terrain_output, bytes, mode);
  release_session_buffer(state_, water_output, bytes, mode);
  release_session_buffer(state_, d2, bytes, mode);
  release_session_buffer(state_, sediment_tmp, bytes, mode);
  release_session_buffer(state_, fl_a, bytes, mode);
  release_session_buffer(state_, fr_a, bytes, mode);
  release_session_buffer(state_, ft_a, bytes, mode);
  release_session_buffer(state_, fb_a, bytes, mode);
  release_session_buffer(state_, fl_b, bytes, mode);
  release_session_buffer(state_, fr_b, bytes, mode);
  release_session_buffer(state_, ft_b, bytes, mode);
  release_session_buffer(state_, fb_b, bytes, mode);
  if (!p_sediment) release_session_buffer(state_, sediment, bytes, mode);
  if (!p_vel_u) release_session_buffer(state_, velocity_u, bytes, mode);
  if (!p_vel_v) release_session_buffer(state_, velocity_v, bytes, mode);
  release_session_buffer(state_,
                         reduction_a,
                         partial_count * sizeof(float),
                         mode);
  release_session_buffer(state_,
                         reduction_b,
                         partial_count * sizeof(float),
                         mode);
  if (!p_rain_map)
    release_session_buffer(state_, rain_buffer, bytes, mode);
  recycle_unique_state(z.state_);
  return DeviceArray(std::move(terrain_state));
}

DeviceArray::DeviceArray(std::shared_ptr<detail::DeviceArrayState> state)
    : state_(std::move(state))
{
}

DeviceArray::~DeviceArray() = default;

bool DeviceArray::empty() const noexcept
{
  return !state_ || !state_->buffer;
}

glm::ivec2 DeviceArray::shape() const
{
  return state_ ? state_->shape : glm::ivec2{0, 0};
}

std::size_t DeviceArray::size() const
{
  return state_ ? static_cast<std::size_t>(state_->shape.x) *
                      static_cast<std::size_t>(state_->shape.y)
                : 0;
}

StorageMode DeviceArray::storage_mode() const
{
  return state_ ? state_->storage : StorageMode::shared;
}

ResidencyState DeviceArray::residency_state() const
{
  return state_ ? state_->residency : ResidencyState::host_valid;
}

std::string DeviceArray::debug_name() const
{
  return state_ ? state_->debug_name : std::string();
}

void DeviceArray::set_debug_name(const std::string &name)
{
  if (!state_ || !state_->buffer)
    throw std::invalid_argument("Cannot name an empty Metal DeviceArray");
  state_->debug_name = name;
  if (!name.empty())
  {
    NSString *label = [NSString stringWithUTF8String:name.c_str()];
    [state_->buffer setLabel:label];
  }
}

Array DeviceArray::to_array() const
{
  return download_array_state(state_);
}

DeviceSession::DeviceSession(StorageMode default_storage)
{
  require_ready();
  state_ = std::make_shared<detail::DeviceSessionState>();
  state_->default_storage = default_storage;
  state_->command_buffer = make_command_buffer(&state_->stats);
}

DeviceSession::DeviceSession(DeviceSession &&other) noexcept = default;

DeviceSession &DeviceSession::operator=(DeviceSession &&other) noexcept =
    default;

DeviceSession::~DeviceSession() noexcept
{
  if (state_)
  {
    try
    {
      finish_session(state_);
    }
    catch (...)
    {
      // Destruction is best effort. Explicit finish/download calls report
      // command-buffer failures to the caller.
    }
  }
}

DeviceArray DeviceSession::upload(const Array &array, StorageMode mode)
{
  require_session_open(state_);
  check_shape_2d(array.shape);
  id<MTLBuffer> buffer =
      upload_session_values(state_, array.vector, mode);
  auto result = make_array_state(state_,
                                 array.shape,
                                 buffer,
                                 mode,
                                 ResidencyState::both_valid);
  result->host_shadow =
      std::make_shared<std::vector<float>>(array.vector);
  return DeviceArray(std::move(result));
}

DeviceArray DeviceSession::allocate(glm::ivec2 shape, StorageMode mode)
{
  require_session_open(state_);
  check_shape_2d(shape);
  const std::size_t bytes = static_cast<std::size_t>(shape.x) *
                            static_cast<std::size_t>(shape.y) * sizeof(float);
  id<MTLBuffer> buffer = acquire_session_buffer(state_, bytes, mode);
  return DeviceArray(make_array_state(state_,
                                      shape,
                                      buffer,
                                      mode,
                                      ResidencyState::device_valid));
}

Array DeviceSession::download(const DeviceArray &array) const
{
  if (!array.state_ || array.state_->session != state_)
    throw std::invalid_argument(
        "Metal DeviceArray belongs to another DeviceSession");
  return download_array_state(array.state_);
}

void DeviceSession::submit()
{
  require_session_open(state_);
  if (!state_->has_work) return;
  [state_->command_buffer commit];
  state_->submitted_command_buffers.push_back(state_->command_buffer);
  state_->command_buffer = make_command_buffer(&state_->stats);
  state_->has_work = false;
}

void DeviceSession::finish() const
{
  finish_session(state_);
}

ExecutionStats DeviceSession::stats() const
{
  return state_ ? state_->stats : ExecutionStats{};
}

} // namespace hmap::gpu::metal

#endif
