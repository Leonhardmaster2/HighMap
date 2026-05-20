/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <cstdint> // for uint32_t

#include "highmap/array.hpp"                     // for Array, operator*
#include "highmap/filters.hpp"                   // for project_talus_along...
#include "highmap/functions.hpp"                 // for NoiseType
#include "highmap/math/array.hpp"                // for lerp
#include "highmap/primitives/coherent_noise.hpp" // for VoronoiReturnType
#include "highmap/primitives/geo.hpp"            // for plates

namespace hmap::gpu
{

Array plates(glm::ivec2    shape,
             glm::vec2     kw,
             std::uint32_t seed,
             float         talus,
             int           direction,
             float         mix_ratio,
             float         base_noise_amp,
             float         kw_multiplier,
             int           octaves,
             float         rugosity,
             glm::vec4     bbox)
{
  const float persistence = 0.5f;
  const float lacunarity = 2.f;
  const float k_smoothing = 0.f;

  // prepare base noise used for displacements
  Array noise = base_noise_amp * gpu::noise_fbm(NoiseType::SIMPLEX2,
                                                shape,
                                                kw_multiplier * kw,
                                                seed++,
                                                octaves,
                                                rugosity,
                                                persistence,
                                                lacunarity,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                bbox);

  // base primitive
  glm::vec2         jitter(1.f, 1.f);
  VoronoiReturnType return_type = VoronoiReturnType::EDGE_DISTANCE_SQUARED;

  Array voronoi = 2.f * voronoi_fbm(shape,
                                    kw,
                                    seed++,
                                    jitter,
                                    k_smoothing,
                                    0.f,
                                    return_type,
                                    octaves,
                                    /* weight */ 0.7f,
                                    persistence,
                                    lacunarity,
                                    nullptr,
                                    &noise,
                                    nullptr,
                                    bbox);

  Array plates = gpu::project_talus_along_direction(voronoi,
                                                    Array(shape, talus),
                                                    direction);

  return lerp(voronoi, plates, mix_ratio);
}

} // namespace hmap::gpu
