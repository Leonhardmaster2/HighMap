/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <sys/types.h> // for uint

#include <cmath> // for cos, sin, M_PI

#include "highmap/array.hpp"      // for Array, operator*
#include "highmap/filters.hpp"    // for gamma_correction, smooth_fill
#include "highmap/functions.hpp"  // for NoiseType
#include "highmap/primitives.hpp" // for VoronoiReturnType, gaussian_pulse

namespace hmap::gpu
{

Array shattered_peak(glm::ivec2   shape,
                     uint         seed,
                     float        scale,
                     int          octaves,
                     float        peak_kw,
                     float        rugosity,
                     float        angle,
                     float        gamma,
                     bool         add_deposition,
                     float        bulk_amp,
                     float        base_noise_amp,
                     float        k_smoothing,
                     glm::vec2    center,
                     const Array *p_noise_x,
                     const Array *p_noise_y,
                     glm::vec4    bbox)
{
  // apply global scaling to reference values
  const float     half_width = 0.2f * scale;
  const glm::vec2 kw = glm::vec2(peak_kw / scale, peak_kw / scale);

  const float persistence = 0.5f;
  const float lacunarity = 2.f;
  const float alpha = angle / 180.f * M_PI;

  // prepare base noise used for displacements
  Array noise = scale * base_noise_amp *
                gpu::noise_fbm(NoiseType::SIMPLEX2,
                               shape,
                               kw,
                               seed,
                               octaves,
                               rugosity,
                               persistence,
                               lacunarity,
                               /* p_ctrl_param */ nullptr,
                               p_noise_x,
                               p_noise_y,
                               /* p_stretching */ nullptr,
                               bbox);

  Array dx = noise * std::cos(alpha);
  Array dy = noise * std::sin(alpha);

  // enveloppe pulse
  Array pulse = gaussian_pulse(shape,
                               half_width,
                               /* p_ctrl_param */ nullptr,
                               /* p_noise_x */ nullptr,
                               /* p_noise_y */ nullptr,
                               /* p_stretching */ nullptr,
                               center,
                               bbox);

  // base primitives
  glm::vec2         jitter(1.f, 1.f);
  VoronoiReturnType return_type = VoronoiReturnType::EDGE_DISTANCE_SQUARED;

  // roughly in [0, 0.5]
  Array voronoi = voronoi_fbm(shape,
                              kw,
                              seed,
                              jitter,
                              k_smoothing,
                              0.f,
                              return_type,
                              octaves,
                              /* weight */ 0.7f,
                              persistence,
                              lacunarity,
                              /* p_ctrl_param */ nullptr,
                              &dx,
                              &dy,
                              bbox);

  voronoi *= pulse;
  voronoi += bulk_amp * pulse;
  voronoi *= 1.f / (0.5f + bulk_amp); // ~ in [0, 1]

  gamma_correction(voronoi, gamma);

  if (add_deposition)
  {
    int   ir = (int)(0.05f * scale * shape.x);
    float k = 0.05f;
    gpu::smooth_fill(voronoi, ir, k);
  }

  return voronoi;
}

} // namespace hmap::gpu
