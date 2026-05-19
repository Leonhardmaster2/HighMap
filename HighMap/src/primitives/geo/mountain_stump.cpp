/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <sys/types.h> // for uint

#include <cmath> // for cos, sin, M_PI

#include "highmap/array.hpp"      // for Array, operator*
#include "highmap/filters.hpp"    // for gain, gamma_correction, smooth_...
#include "highmap/functions.hpp"  // for NoiseType
#include "highmap/primitives.hpp" // for noise_fbm, VoronoiReturnType
#include "highmap/range.hpp"      // for clamp_min, minimum_smooth

namespace hmap::gpu
{

Array mountain_stump(glm::ivec2   shape,
                     uint         seed,
                     float        scale,
                     int          octaves,
                     float        peak_kw,
                     float        rugosity,
                     float        angle,
                     float        k_smoothing,
                     float        gamma,
                     bool         add_deposition,
                     float        ridge_amp,
                     float        base_noise_amp,
                     glm::vec2    center,
                     const Array *p_noise_x,
                     const Array *p_noise_y,
                     glm::vec4    bbox)
{
  // apply global scaling to reference values
  const float     half_width = 0.1f * scale;
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
  gain(pulse, 2.f);

  // in [0.5, 1]
  Array stump = 0.25f * gpu::noise_fbm(NoiseType::SIMPLEX2,
                                       shape,
                                       kw,
                                       seed,
                                       octaves,
                                       /* rugosity */ 0.7f,
                                       persistence,
                                       lacunarity,
                                       /* p_ctrl_param */ nullptr,
                                       p_noise_x,
                                       p_noise_y,
                                       /* p_stretching */ nullptr,
                                       bbox) +
                0.75f;

  // divide by 0.75f to set amplitude back to [0, 1] (very
  // approximative...)
  const float km = 0.05f;
  stump = hmap::minimum_smooth(stump, pulse, km); // / 0.75f;

  stump.infos();

  // base primitives
  glm::vec2         jitter(1.f, 1.f);
  VoronoiReturnType return_type = VoronoiReturnType::EDGE_DISTANCE_SQUARED;

  // roughly in [0, 1]
  Array voronoi = 2.f * voronoi_fbm(shape,
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
  clamp_min(voronoi, 0.f);
  voronoi *= pulse;
  gamma_correction(voronoi, gamma);

  voronoi = (ridge_amp * voronoi + stump) / (ridge_amp + 1.f);

  if (add_deposition)
  {
    int   ir = (int)(0.05f * scale * shape.x);
    float k = 0.05f;
    gpu::smooth_fill(voronoi, ir, k);
  }

  return voronoi;
}

} // namespace hmap::gpu
