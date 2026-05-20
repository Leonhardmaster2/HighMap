/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <bits/std_abs.h> // for abs
#include <stdlib.h>       // for abs

#include <cmath>      // for floor
#include <functional> // for function

#include "highmap/array.hpp"                // for Array
#include "highmap/operator.hpp"             // for fill_array_using_xy_func...
#include "highmap/primitives/functions.hpp" // for checkerboard

namespace hmap
{

Array checkerboard(glm::ivec2   shape,
                   glm::vec2    kw,
                   const Array *p_noise_x,
                   const Array *p_noise_y,
                   const Array *p_stretching,
                   glm::vec4    bbox)
{
  Array array = Array(shape);

  auto lambda = [&kw](float x, float y, float)
  {
    return std::abs(std::abs((int)std::floor(x) % 2) -
                    std::abs((int)std::floor(y) % 2));
  };

  fill_array_using_xy_function(array,
                               bbox,
                               nullptr,
                               p_noise_x,
                               p_noise_y,
                               p_stretching,
                               lambda);
  return array;
}

} // namespace hmap
