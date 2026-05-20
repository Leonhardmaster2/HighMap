/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

#include <cstdint> // for uint32_t

#include "highmap/array.hpp"                     // for Array
#include "highmap/functions.hpp"                 // for WorleyDoubleFunction
#include "highmap/operator.hpp"                  // for fill_array_using_xy...
#include "highmap/primitives/coherent_noise.hpp" // for worley_double

namespace hmap
{

Array worley_double(glm::ivec2    shape,
                    glm::vec2     kw,
                    std::uint32_t seed,
                    float         ratio,
                    float         k,
                    const Array  *p_ctrl_param,
                    const Array  *p_noise_x,
                    const Array  *p_noise_y,
                    const Array  *p_stretching,
                    glm::vec4     bbox)
{
  hmap::Array                array = hmap::Array(shape);
  hmap::WorleyDoubleFunction f = hmap::WorleyDoubleFunction(kw, seed, ratio, k);

  fill_array_using_xy_function(array,
                               bbox,
                               p_ctrl_param,
                               p_noise_x,
                               p_noise_y,
                               p_stretching,
                               f.get_delegate());
  return array;
}

} // namespace hmap
