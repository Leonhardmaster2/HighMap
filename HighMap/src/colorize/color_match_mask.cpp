/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <glm/glm.hpp>

#include "highmap/array.hpp"

namespace hmap
{

Array color_match_mask(const Array     &r,
                       const Array     &g,
                       const Array     &b,
                       const glm::vec3 &color,
                       float            tolerance)
{
  const glm::ivec2 &shape = r.shape;
  const float       tol2 = tolerance * tolerance;
  Array             mask(shape);

  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      float dr = r(i, j) - color.x;
      float dg = g(i, j) - color.y;
      float db = b(i, j) - color.z;
      float dist2 = dr * dr + dg * dg + db * db;

      mask(i, j) = (dist2 <= tol2) ? 1.f : 0.f;
    }

  return mask;
}

} // namespace hmap
