/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/array.hpp"

namespace hmap
{

float variance(const Array &array, float *p_mean)
{
  const glm::ivec2 &shape = array.shape;
  float             mean = p_mean ? *p_mean : array.mean();
  float             var = 0.f;

  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      float d = array(i, j) - mean;
      var += d * d;
    }
  var /= float(shape.x * shape.y);

  return var;
}

} // namespace hmap
