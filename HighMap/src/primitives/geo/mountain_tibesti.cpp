/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <sys/types.h> // for uint

#include <cmath> // for cos, sin, M_PI

#include "highmap/array.hpp"      // for Array, operator*, operator+
#include "highmap/filters.hpp"    // for gamma_correction, smooth_fill
#include "highmap/functions.hpp"  // for NoiseType
#include "highmap/primitives.hpp" // for noise_fbm, gabor_wave_fbm, gaus...
#include "highmap/range.hpp"      // for clamp_min

namespace hmap::gpu
{

Array mountain_tibesti(glm::ivec2   shape,
                       uint         seed,
                       float        scale,
                       int          octaves,
                       float        peak_kw,
                       float        rugosity,
                       float        angle,
                       float        angle_spread_ratio,
                       float        gamma,
                       bool         add_deposition,
                       float        bulk_amp,
                       float        base_noise_amp,
                       glm::vec2    center,
                       const Array *p_noise_x,
                       const Array *p_noise_y,
                       glm::vec4    bbox)
{
  const float     persistence = 0.5f;
  const float     lacunarity = 2.f;
  const float     alpha = angle / 180.f * M_PI;
  const float     half_width = 0.3f;
  const glm::vec2 kw_base = glm::vec2(peak_kw / scale, peak_kw / scale);
  const glm::vec2 kw_noise4 = glm::vec2(4.f / scale, 4.f / scale);
  const glm::vec2 kw_noise2 = glm::vec2(2.f / scale, 2.f / scale);

  // prepare base noise used for displacements
  Array noise4 = gpu::noise_fbm(NoiseType::SIMPLEX2,
                                shape,
                                kw_noise4,
                                seed++,
                                octaves,
                                /* weight */ 0.7f,
                                persistence,
                                lacunarity,
                                /* p_ctrl_param */ nullptr,
                                p_noise_x,
                                p_noise_y,
                                /* p_stretching */ nullptr,
                                bbox);

  noise4 = 0.5f * noise4 + 0.5f;
  clamp_min(noise4, 0.f);
  gamma_correction(noise4, gamma);

  Array noise2 = scale * base_noise_amp *
                 gpu::noise_fbm(NoiseType::SIMPLEX2,
                                shape,
                                kw_noise2,
                                seed++,
                                octaves,
                                rugosity,
                                persistence,
                                lacunarity,
                                /* p_ctrl_param */ nullptr,
                                p_noise_x,
                                p_noise_y,
                                /* p_stretching */ nullptr,
                                bbox);

  // base
  Array dx = noise2 * std::cos(alpha); // perpendicular
  Array dy = noise2 * std::sin(alpha);

  Array gabor = gabor_wave_fbm(shape,
                               kw_base,
                               seed++,
                               angle,
                               angle_spread_ratio,
                               octaves,
                               /* weight */ 0.7f,
                               persistence,
                               lacunarity,
                               /* p_ctrl_param */ nullptr,
                               &dx,
                               &dy,
                               bbox);

  gabor = (0.5f * gabor + 0.5f) * noise4;
  gabor = noise4 * (bulk_amp + gabor) / (bulk_amp + 1.f);

  // enveloppe pulse
  Array pulse = gaussian_pulse(shape,
                               half_width,
                               /* p_ctrl_param */ nullptr,
                               /* p_noise_x */ nullptr,
                               /* p_noise_y */ nullptr,
                               /* p_stretching */ nullptr,
                               center,
                               bbox);

  gabor *= pulse;

  // post
  if (add_deposition)
  {
    int   ir = (int)(0.05f * scale * shape.x);
    float k = 0.05f;
    gpu::smooth_fill(gabor, ir, k);
  }

  return gabor;
}

} // namespace hmap::gpu
