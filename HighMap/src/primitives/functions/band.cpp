/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "highmap/array.hpp"
#include "highmap/geometry/grids.hpp"
#include "highmap/math/core.hpp"
#include "highmap/math/profiles.hpp"

namespace hmap
{

Array band(glm::ivec2    shape,
           float         angle,
           float         length,
           float         width,
           RadialProfile profile,
           float         profile_param,
           const Array  *p_noise_r,
           const Array  *p_noise_offset,
           glm::vec2     center,
           glm::vec4     bbox)
{
  const float alpha = angle / 180.f * M_PI;
  const float ca = std::cos(alpha);
  const float sa = std::sin(alpha);
  const float half_length = 0.5f * std::max(0.f, length);
  const float wd = std::max(1e-6f, width);

  Array z = Array(shape);

  auto radial_profile = get_radial_profile_function(profile, profile_param);

  // grid
  bool               endpoint = false;
  std::vector<float> x, y;
  grid_xy_vector(x, y, shape, bbox, endpoint);

  for (int j = 0; j < shape.y; j++)
    for (int i = 0; i < shape.x; i++)
    {
      float dx = x[i] - center.x;
      float dy = y[j] - center.y;

      // band frame: u along the core segment, v transverse
      float u = dx * ca + dy * sa;
      float v = -dx * sa + dy * ca;

      float ds = p_noise_offset ? (*p_noise_offset)(i, j) : 0.f;
      float dr = p_noise_r ? (*p_noise_r)(i, j) : 0.f;

      // capsule distance to the core segment, normalized by the width
      float du = std::max(0.f, std::abs(u) - half_length);
      float d = std::hypot(du / wd, v / wd + ds);
      d = std::max(0.f, d + dr);

      z(i, j) = 1.f - radial_profile(std::min(d, 1.f));
    }

  return z;
}

} // namespace hmap
