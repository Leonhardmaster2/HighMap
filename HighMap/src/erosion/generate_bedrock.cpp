/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/gradient.hpp"
#include "highmap/math/array.hpp"

namespace hmap
{

Array generate_bedrock(const Array &z,
                       float        elevation_strength,
                       float        slope_strength,
                       float        slope_limit,
                       float        zmin,
                       float        zmax)
{
  // resolve elevation range
  if (zmin > zmax)
  {
    zmin = z.min();
    zmax = z.max();
  }

  float zptp = zmax - zmin;
  Array z_bedrock = z;

  // amplitude influence
  if (elevation_strength != 0.f)
    z_bedrock -= elevation_strength * (z - zmin) / zptp;

  // slope influence
  if (slope_strength != 0.f)
  {
    Array slope = gradient_norm(z);
    float slope_width = 0.1f * slope_limit;
    z_bedrock -= slope_strength *
                 (1.f - sigmoid(slope - slope_limit, slope_width));
  }

  return z_bedrock;
}

} // namespace hmap
