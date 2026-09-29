/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm>
#include <cmath>

#include "highmap/array.hpp"
#include "highmap/gradient.hpp"
#include "highmap/hydrology/hydrology.hpp"
#include "highmap/internal/validation.hpp"
#include "highmap/local_metrics.hpp"

namespace hmap
{

Array topographic_wetness_index(const Array &z,
                                float        talus_ref,
                                float        min_slope)
{
  if (!validate_non_empty(z)) return Array();

  const float tref = talus_ref > 0.f ? talus_ref
                                     : std::max(gradient_talus(z).max(), 1e-6f);

  Array facc = flow_accumulation_dinf(z, tref);
  Array slope = gradient_norm(z);

  Array twi(z.shape);

#pragma omp parallel for schedule(static)
  for (int j = 0; j < z.shape.y; ++j)
    for (int i = 0; i < z.shape.x; ++i)
    {
      const float s = std::max(slope(i, j), min_slope);
      const float a = std::max(facc(i, j), 1e-6f);
      twi(i, j) = std::log(a / s);
    }

  return twi;
}

} // namespace hmap
