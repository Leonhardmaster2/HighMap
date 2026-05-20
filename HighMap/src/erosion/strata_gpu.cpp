/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <cstdint> // for uint32_t
#include <vector>  // for allocator, vector

#include "cl_wrapper/run.hpp" // for Run

#include "highmap/array.hpp"                     // for Array, operator*
#include "highmap/erosion.hpp"                   // for strata_cells, strat...
#include "highmap/functions.hpp"                 // for NoiseType
#include "highmap/math/array.hpp"                // for lerp
#include "highmap/opencl/gpu_opencl.hpp"         // for helper_bind_optiona...
#include "highmap/primitives/coherent_noise.hpp" // for noise_fbm

namespace hmap::gpu
{

void strata(Array           &z,
            float            angle,
            float            slope,
            float            gamma,
            std::uint32_t    seed,
            bool             linear_gamma,
            float            kz,
            int              octaves,
            float            lacunarity,
            float            gamma_noise_ratio,
            float            noise_amp,
            const glm::vec2 &noise_kw,
            bool             enable_ridge_noise,
            const glm::vec2 &ridge_noise_kw,
            float            ridge_angle_shift,
            float            ridge_noise_amp,
            float            ridge_clamp_vmin,
            float            ridge_remap_vmin,
            bool             apply_elevation_mask,
            bool             apply_ridge_mask,
            float            mask_gamma,
            const Array     *p_mask,
            const glm::vec4 &bbox)
{
  auto run = clwrapper::Run("strata");

  run.bind_buffer<float>("z", z.vector);

  helper_bind_optional_buffer(run, "mask", p_mask);

  run.bind_arguments(z.shape.x,
                     z.shape.y,
                     angle,
                     slope,
                     gamma,
                     seed,
                     linear_gamma ? 1 : 0,
                     kz,
                     octaves,
                     lacunarity,
                     gamma_noise_ratio,
                     noise_amp,
                     noise_kw,
                     enable_ridge_noise ? 1 : 0,
                     ridge_noise_kw,
                     ridge_angle_shift,
                     ridge_noise_amp,
                     ridge_clamp_vmin,
                     ridge_remap_vmin,
                     apply_elevation_mask ? 1 : 0,
                     apply_ridge_mask ? 1 : 0,
                     mask_gamma,
                     p_mask ? 1 : 0,
                     bbox);

  run.write_buffer("z");
  run.execute({z.shape.x, z.shape.y});
  run.read_buffer("z");
}

void strata_cells(Array        &z,
                  glm::vec2     kw,
                  float         amp,
                  std::uint32_t seed,
                  float         gamma,
                  float         gamma_lateral,
                  float         angle,
                  float         noise_amp,
                  bool          absolute_displacement,
                  float         occurence_probability,
                  const Array  *p_noise_x,
                  const Array  *p_noise_y,
                  glm::vec4     bbox)
{
  const Array *p_mask = nullptr;

  //

  auto run = clwrapper::Run("strata_cells");

  run.bind_buffer<float>("z", z.vector);

  helper_bind_optional_buffer(run, "mask", p_mask);
  helper_bind_optional_buffer(run, "noise_x", p_noise_x);
  helper_bind_optional_buffer(run, "noise_y", p_noise_y);

  run.bind_arguments(z.shape.x,
                     z.shape.y,
                     kw.x,
                     kw.y,
                     seed,
                     gamma,
                     gamma_lateral,
                     angle,
                     amp,
                     noise_amp,
                     absolute_displacement ? 1 : 0,
                     occurence_probability,
                     p_mask ? 1 : 0,
                     p_noise_x ? 1 : 0,
                     p_noise_y ? 1 : 0,
                     bbox);

  run.write_buffer("z");
  run.execute({z.shape.x, z.shape.y});
  run.read_buffer("z");
}

void strata_cells(Array        &z,
                  glm::vec2     kw,
                  float         amp,
                  std::uint32_t seed,
                  const Array  *p_mask,
                  float         gamma,
                  float         gamma_lateral,
                  float         angle,
                  float         noise_amp,
                  bool          absolute_displacement,
                  float         occurence_probability,
                  const Array  *p_noise_x,
                  const Array  *p_noise_y,
                  glm::vec4     bbox)
{
  if (!p_mask)
  {
    strata_cells(z,
                 kw,
                 amp,
                 seed,
                 gamma,
                 gamma_lateral,
                 angle,
                 noise_amp,
                 absolute_displacement,
                 occurence_probability,
                 p_noise_x,
                 p_noise_y,
                 bbox);
  }
  else
  {
    Array z_f = z;
    strata_cells(z_f,
                 kw,
                 amp,
                 seed,
                 gamma,
                 gamma_lateral,
                 angle,
                 noise_amp,
                 absolute_displacement,
                 occurence_probability,
                 p_noise_x,
                 p_noise_y,
                 bbox);
    z = lerp(z, z_f, *(p_mask));
  }
}

void strata_cells_fbm(Array        &z,
                      glm::vec2     kw,
                      float         amp,
                      std::uint32_t seed,
                      float         gamma,
                      float         gamma_lateral,
                      float         angle,
                      bool          enable_default_noise,
                      float         default_noise_amp,
                      bool          absolute_displacement,
                      float         occurence_probability,
                      int           octaves,
                      float         persistence,
                      float         lacunarity,
                      const Array  *p_noise_x,
                      const Array  *p_noise_y,
                      glm::vec4     bbox)
{

  // --- Default noise

  Array dx, dy;

  if (enable_default_noise)
  {
    const glm::vec2 kw_noise = kw;

    dx = default_noise_amp * gpu::noise_fbm(hmap::NoiseType::SIMPLEX2,
                                            z.shape,
                                            kw_noise,
                                            ++seed,
                                            /* octaves */ 8,
                                            /* weight*/ 0.f,
                                            /* persistence */ 0.5f,
                                            /* lacunarity */ 2.f,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            bbox);

    dy = default_noise_amp * gpu::noise_fbm(hmap::NoiseType::SIMPLEX2,
                                            z.shape,
                                            kw_noise,
                                            ++seed,
                                            /* octaves */ 8,
                                            /* weight*/ 0.f,
                                            /* persistence */ 0.5f,
                                            /* lacunarity */ 2.f,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            bbox);

    p_noise_x = &dx;
    p_noise_y = &dy;
  }

  // --- Fbm layers

  float a = 1.f;
  float m = 1.f;

  for (int k = 0; k < octaves; ++k)
  {
    strata_cells(z,
                 m * kw,
                 a * amp,
                 seed,
                 gamma,
                 gamma_lateral,
                 angle,
                 /* noise_amp */ 1.f,
                 absolute_displacement,
                 occurence_probability,
                 p_noise_x,
                 p_noise_y,
                 bbox);

    a *= persistence;
    m *= lacunarity;
  }
}

void strata_cells_fbm(Array        &z,
                      glm::vec2     kw,
                      float         amp,
                      std::uint32_t seed,
                      const Array  *p_mask,
                      float         gamma,
                      float         gamma_lateral,
                      float         angle,
                      bool          enable_default_noise,
                      float         default_noise_amp,
                      bool          absolute_displacement,
                      float         occurence_probability,
                      int           octaves,
                      float         persistence,
                      float         lacunarity,
                      const Array  *p_noise_x,
                      const Array  *p_noise_y,
                      glm::vec4     bbox)
{
  if (!p_mask)
  {
    strata_cells_fbm(z,
                     kw,
                     amp,
                     seed,
                     gamma,
                     gamma_lateral,
                     angle,
                     enable_default_noise,
                     default_noise_amp,
                     absolute_displacement,
                     occurence_probability,
                     octaves,
                     persistence,
                     lacunarity,
                     p_noise_x,
                     p_noise_y,
                     bbox);
  }
  else
  {
    Array z_f = z;
    strata_cells_fbm(z_f,
                     kw,
                     amp,
                     seed,
                     gamma,
                     gamma_lateral,
                     angle,
                     enable_default_noise,
                     default_noise_amp,
                     absolute_displacement,
                     occurence_probability,
                     octaves,
                     persistence,
                     lacunarity,
                     p_noise_x,
                     p_noise_y,
                     bbox);
    z = lerp(z, z_f, *(p_mask));
  }
}

void strata_terrace(Array        &z,
                    float         gamma,
                    std::uint32_t seed,
                    float         kz,
                    bool          linear_gamma,
                    float         gamma_noise_ratio,
                    const Array  *p_noise)
{
  auto run = clwrapper::Run("strata_terrace");

  run.bind_buffer<float>("z", z.vector);
  helper_bind_optional_buffer(run, "noise", p_noise);

  run.bind_arguments(z.shape.x,
                     z.shape.y,
                     gamma,
                     seed,
                     linear_gamma ? 1 : 0,
                     kz,
                     gamma_noise_ratio,
                     p_noise ? 1 : 0);

  run.write_buffer("z");
  run.execute({z.shape.x, z.shape.y});
  run.read_buffer("z");
}

void strata_terrace(Array        &z,
                    float         gamma,
                    std::uint32_t seed,
                    const Array  *p_mask,
                    float         kz,
                    bool          linear_gamma,
                    float         gamma_noise_ratio,
                    const Array  *p_noise)
{
  if (!p_mask)
  {
    strata_terrace(z,
                   gamma,
                   seed,
                   kz,
                   linear_gamma,
                   gamma_noise_ratio,
                   p_noise);
  }
  else
  {
    Array z_f = z;
    strata_terrace(z_f,
                   gamma,
                   seed,
                   kz,
                   linear_gamma,
                   gamma_noise_ratio,
                   p_noise);
    z = lerp(z, z_f, *(p_mask));
  }
}

} // namespace hmap::gpu
