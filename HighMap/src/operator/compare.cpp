/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/array.hpp"

namespace hmap
{

Array compare(const Array &a,
              const Array &b,
              float        slice_x_pos,
              float        slice_y_pos)
{
  const glm::ivec2 &shape = a.shape;
  Array             out(shape);

  int ic = int(slice_x_pos * (shape.x - 1.f));
  int jc = int(slice_y_pos * (shape.y - 1.f));

  for (int j = 0; j < shape.y; j++)
    for (int i = 0; i < shape.x; i++)
    {
      if (i <= ic && j <= jc)
        out(i, j) = a(i, j);
      else
        out(i, j) = b(i, j);
    }

  return out;
}

} // namespace hmap
