/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU
 * General Public License. The full license is in the file LICENSE, distributed
 * with this software. */
#pragma once

#include <cstdint>
#include <string>

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

/** @brief Return whether the staged Metal noise kernel supports this type. */
bool supports_noise(NoiseType noise_type);

Array gradient_norm(const Array &array);

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
