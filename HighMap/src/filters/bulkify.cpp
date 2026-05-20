/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/array.hpp"                // for Array, operator*
#include "highmap/primitives/functions.hpp" // for get_primitive_base, Prim...

namespace hmap
{

Array bulkify(const Array         &z,
              const PrimitiveType &primitive_type,
              float                amp,
              const Array         *p_noise_x,
              const Array         *p_noise_y,
              glm::vec2            center,
              glm::vec4            bbox)
{
  return z + amp * get_primitive_base(primitive_type,
                                      z.shape,
                                      p_noise_x,
                                      p_noise_y,
                                      center,
                                      bbox);
}

} // namespace hmap
