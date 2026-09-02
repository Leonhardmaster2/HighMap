/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU
 * General Public License. The full license is in the file LICENSE, distributed
 * with this software. */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "highmap/array.hpp"
#include "highmap/functions.hpp"

#ifndef HIGHMAP_HAS_METAL
#define HIGHMAP_HAS_METAL 0
#endif

namespace hmap::gpu::metal
{

/**
 * @brief Return whether a usable native Metal device/backend is available.
 *
 * The result is capability based and does not inspect the Mac's marketing
 * model name. It is safe to call on non-Apple builds.
 */
bool is_available();

/** @brief Return the selected Metal device name, or an empty string. */
std::string device_name();

/**
 * @brief Capability and dispatch data exposed for diagnostics and tuning.
 *
 * Values are zero on builds without a usable Metal backend. GPU family
 * numbers are reported by Metal and are intentionally not mapped to M-series
 * marketing names.
 */
struct DeviceCapabilities
{
  std::string device_name;
  std::uint64_t recommended_max_working_set_size = 0;
  std::uint32_t thread_execution_width = 0;
  std::uint32_t max_threads_per_threadgroup = 0;
  std::uint32_t gpu_family = 0;
};

DeviceCapabilities capabilities();

/**
 * @brief Counters from the most recent Metal operation on the calling thread.
 *
 * These are diagnostic counters for benchmark and profiling work, not part of
 * the numerical API contract. GPU time is obtained from Metal command-buffer
 * timestamps when the driver reports them.
 */
struct ExecutionStats
{
  std::uint64_t buffer_allocations = 0;
  std::uint64_t pipeline_creations = 0;
  std::uint64_t command_buffers = 0;
  std::uint64_t encoders = 0;
  std::uint64_t synchronization_count = 0;
  std::uint64_t upload_count = 0;
  std::uint64_t upload_bytes = 0;
  std::uint64_t readback_count = 0;
  std::uint64_t readback_bytes = 0;
  std::uint64_t bytes_allocated = 0;
  std::uint64_t bytes_reused = 0;
  std::uint64_t buffer_reuses = 0;
  std::uint64_t texture_allocations = 0;
  std::uint64_t resident_bytes = 0;
  std::uint64_t peak_resident_bytes = 0;
  std::uint64_t blit_encoders = 0;
  double        allocation_ms = 0.0;
  double        upload_ms = 0.0;
  double        pipeline_lookup_ms = 0.0;
  double        encoding_ms = 0.0;
  double        gpu_execution_ms = 0.0;
  double        wait_ms = 0.0;
  double        readback_ms = 0.0;
};

ExecutionStats last_execution_stats();

enum class StorageMode
{
  shared,
  private_storage
};

enum class ResidencyState
{
  device_valid,
  host_valid,
  both_valid
};

namespace detail
{
struct DeviceArrayState;
struct DeviceSessionState;
} // namespace detail

/**
 * @brief Explicit GPU-resident float32 array owned by a Metal session.
 *
 * Copies share the resource. Resident operations are out-of-place unless an
 * operation explicitly documents otherwise; use DeviceSession::download or
 * to_array() for the synchronization boundary.
 */
class DeviceArray
{
public:
  DeviceArray() = default;
  DeviceArray(const DeviceArray &) = default;
  DeviceArray &operator=(const DeviceArray &) = default;
  DeviceArray(DeviceArray &&) noexcept = default;
  DeviceArray &operator=(DeviceArray &&) noexcept = default;
  ~DeviceArray();

  bool empty() const noexcept;
  glm::ivec2 shape() const;
  std::size_t size() const;
  StorageMode storage_mode() const;
  ResidencyState residency_state() const;
  std::string debug_name() const;
  void set_debug_name(const std::string &name);

  /** @brief Download through the owning session and return a host Array. */
  Array to_array() const;

private:
  std::shared_ptr<detail::DeviceArrayState> state_;

  explicit DeviceArray(std::shared_ptr<detail::DeviceArrayState> state);

  friend class DeviceSession;
};

/**
 * @brief One ordered, explicitly synchronized Metal execution session.
 *
 * A session uses one command buffer and one command queue. Resident operations
 * append work without waiting; download() or finish() commits and waits once.
 * A session is movable but not copyable or thread-safe.
 */
class DeviceSession
{
public:
  explicit DeviceSession(StorageMode default_storage = StorageMode::shared);
  DeviceSession(const DeviceSession &) = delete;
  DeviceSession &operator=(const DeviceSession &) = delete;
  DeviceSession(DeviceSession &&) noexcept;
  DeviceSession &operator=(DeviceSession &&) noexcept;
  ~DeviceSession() noexcept;

  DeviceArray upload(const Array &array,
                     StorageMode mode = StorageMode::shared);
  /**
   * @brief Adopt a completed device resource into this session.
   *
   * The source session is finished before the resource is returned. This is
   * intended for bounded, optional caches; callers never receive or store a
   * raw Metal buffer.
   */
  DeviceArray adopt_completed(const DeviceArray &array);
  DeviceArray allocate(glm::ivec2 shape,
                       StorageMode mode = StorageMode::private_storage);

  DeviceArray gradient_norm(DeviceArray array);
  /** @brief Compute a radius-bounded morphological gradient in-resident. */
  DeviceArray morphological_gradient(DeviceArray array, int ir);
  DeviceArray maximum_smooth(DeviceArray        array1,
                             const DeviceArray &array2,
                             float k);
  DeviceArray minimum_smooth(DeviceArray        array1,
                             const DeviceArray &array2,
                             float k);
  DeviceArray noise(NoiseType     noise_type,
                    glm::ivec2    shape,
                    glm::vec2     kw,
                    std::uint32_t seed,
                    const DeviceArray *p_noise_x = nullptr,
                    const DeviceArray *p_noise_y = nullptr,
                    glm::vec4      bbox = {0.f, 1.f, 0.f, 1.f},
                    glm::ivec2     period = {0, 0});
  DeviceArray noise_fbm(NoiseType     noise_type,
                        glm::ivec2    shape,
                        glm::vec2     kw,
                        std::uint32_t seed,
                        int           octaves = 8,
                        float         weight = 0.7f,
                        float         persistence = 0.5f,
                        float         lacunarity = 2.f,
                        const DeviceArray *p_ctrl_param = nullptr,
                        const DeviceArray *p_noise_x = nullptr,
                        const DeviceArray *p_noise_y = nullptr,
                        glm::vec4      bbox = {0.f, 1.f, 0.f, 1.f},
                        glm::ivec2     period = {0, 0});
  /** @brief Evaluate the Gabor-wavelet fBm primitive without host materialization. */
  DeviceArray gabor_wave_fbm(
      glm::ivec2          shape,
      glm::vec2           kw,
      std::uint32_t       seed,
      float               angle_degrees,
      float               angle_spread_ratio,
      int                 octaves,
      float               weight,
      float               persistence,
      float               lacunarity,
      const DeviceArray  *p_ctrl_param = nullptr,
      const DeviceArray  *p_noise_x = nullptr,
      const DeviceArray  *p_noise_y = nullptr,
      const DeviceArray  *p_angle = nullptr,
      glm::vec4           bbox = {0.f, 1.f, 0.f, 1.f});
  /** @brief Gaussian separable blur kept entirely in the session. */
  DeviceArray smooth_cpulse(DeviceArray array, int ir);
  /** @brief Rebuild weighted frequency bands without host materialization. */
  DeviceArray spectral_equalizer(DeviceArray              array,
                                 const std::vector<float> &weights,
                                 int                       ir_min,
                                 int                       ir_max);
  /** @brief Remap using a device reduction for the source range. */
  DeviceArray normalize(DeviceArray array, float vmin = 0.f, float vmax = 1.f);
  DeviceArray advection_warp(DeviceArray z,
                             const DeviceArray &advected_field,
                             const DeviceArray &dx,
                             const DeviceArray &dy,
                             float              advection_length,
                             float              value_persistence,
                             const DeviceArray *p_mask = nullptr);
  /**
   * @brief Texture-backed advection experiment with explicit conversions.
   *
   * Inputs and output remain DeviceArray buffers. Temporary textures are
   * created for this operation and their buffer/texture conversion cost is
   * reported in ExecutionStats.
   */
  DeviceArray advection_warp_texture(
      DeviceArray              z,
      const DeviceArray       &advected_field,
      const DeviceArray       &dx,
      const DeviceArray       &dy,
      float                    advection_length,
      float                    value_persistence,
      const DeviceArray       *p_mask = nullptr);
  DeviceArray thermal(DeviceArray z,
                      const DeviceArray &talus,
                      int                iterations);
  DeviceArray thermal_ridge(DeviceArray z,
                            const DeviceArray &talus,
                            int                iterations);
  /** @brief Apply the legacy one-cell linear border extrapolation in-resident. */
  DeviceArray extrapolate_borders(DeviceArray z);
  DeviceArray linear_combine(DeviceArray        array1,
                             const DeviceArray &array2,
                             float              weight1,
                             float              weight2);
  DeviceArray hydraulic_vpipes(
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
      const DeviceArray *p_rain_map = nullptr,
      DeviceArray       *p_water_depth = nullptr,
      DeviceArray       *p_sediment = nullptr,
      DeviceArray       *p_vel_u = nullptr,
      DeviceArray       *p_vel_v = nullptr);

  Array download(const DeviceArray &array) const;
  /**
   * @brief Commit the current command buffer and continue without waiting.
   *
   * The session remains ordered on the same Metal queue. This is intended for
   * command-buffer strategy experiments; it is not a CPU synchronization
   * boundary.
   */
  void submit();
  void finish() const;
  ExecutionStats stats() const;

private:
  std::shared_ptr<detail::DeviceSessionState> state_;
};

/** @brief Return whether the staged Metal noise kernel supports this type. */
bool supports_noise(NoiseType noise_type);
/** @brief Return whether resident FBM is available for this noise type. */
bool supports_noise_fbm(NoiseType noise_type);

Array gradient_norm(const Array &array);

/** @brief Compute a morphological gradient with the staged Metal backend. */
Array morphological_gradient(const Array &array, int ir);

Array maximum_smooth(const Array &array1, const Array &array2, float k);

Array minimum_smooth(const Array &array1, const Array &array2, float k);

Array noise(NoiseType     noise_type,
            glm::ivec2    shape,
            glm::vec2     kw,
            std::uint32_t seed,
            const Array  *p_noise_x = nullptr,
            const Array  *p_noise_y = nullptr,
            glm::vec4     bbox = {0.f, 1.f, 0.f, 1.f},
            glm::ivec2    period = {0, 0});

Array noise_fbm(NoiseType     noise_type,
                glm::ivec2    shape,
                glm::vec2     kw,
                std::uint32_t seed,
                int           octaves = 8,
                float         weight = 0.7f,
                float         persistence = 0.5f,
                float         lacunarity = 2.f,
                const Array  *p_ctrl_param = nullptr,
                const Array  *p_noise_x = nullptr,
                const Array  *p_noise_y = nullptr,
                glm::vec4     bbox = {0.f, 1.f, 0.f, 1.f},
                glm::ivec2    period = {0, 0});

Array gabor_wave_fbm(
    glm::ivec2        shape,
    glm::vec2         kw,
    std::uint32_t     seed,
    float             angle_degrees,
    float             angle_spread_ratio,
    int               octaves = 8,
    float             weight = 0.7f,
    float             persistence = 0.5f,
    float             lacunarity = 2.f,
    const Array      *p_ctrl_param = nullptr,
    const Array      *p_noise_x = nullptr,
    const Array      *p_noise_y = nullptr,
    const Array      *p_angle = nullptr,
    glm::vec4         bbox = {0.f, 1.f, 0.f, 1.f});

Array smooth_cpulse(const Array &array, int ir);

Array spectral_equalizer(const Array              &array,
                         const std::vector<float> &weights,
                         int                       ir_min,
                         int                       ir_max);

Array advection_warp(const Array &z,
                     const Array &advected_field,
                     const Array &dx,
                     const Array &dy,
                     float        advection_length,
                     float        value_persistence,
                     const Array *p_mask = nullptr);

/**
 * @brief Run the no-optional-output thermal algorithm on Metal.
 *
 * The implementation uses ping-pong buffers for every iteration. This avoids
 * relying on undefined in-dispatch ordering when neighboring cells are read
 * while the terrain is updated.
 */
void thermal(Array &z, const Array &talus, int iterations);

/**
 * @brief Run the thermal ridge algorithm on Metal.
 *
 * The legacy wrapper preserves the same border extrapolation behavior as the
 * OpenCL implementation while using the ordered resident session API.
 */
void thermal_ridge(Array &z, const Array &talus, int iterations);

/**
 * @brief Run hydraulic virtual pipes with all simulation state kept on Metal.
 *
 * The final result is copied back into the supplied arrays only after the
 * complete iteration loop. Optional rain and output maps preserve the existing
 * HighMap API shape.
 */
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
                      Array *p_vel_v);

} // namespace hmap::gpu::metal
