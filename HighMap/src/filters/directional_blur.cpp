/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <vector>

#include "cl_wrapper/run.hpp"

#include "highmap/array.hpp"
#include "highmap/math/array.hpp"

namespace hmap::gpu
{

void directional_blur(Array &array, float radius, const Array &angle, int steps)
{
  const glm::ivec2 &shape = array.shape;

  Array out(shape);

  auto run = clwrapper::Run("directional_blur");

  run.bind_imagef("array", array.vector, shape.x, shape.y);
  run.bind_imagef("angle", angle.vector, shape.x, shape.y);
  run.bind_imagef("out", out.vector, shape.x, shape.y, true);

  run.bind_arguments(shape.x, shape.y, radius, steps);

  run.execute({shape.x, shape.y});

  run.read_imagef("out");

  array = out;
}

void directional_blur(Array       &array,
                      float        radius,
                      const Array &angle,
                      const Array *p_mask,
                      int          steps)
{
  if (!p_mask)
  {
    directional_blur(array, radius, angle, steps);
  }
  else
  {
    Array array_f = array;
    directional_blur(array_f, radius, angle, steps);
    array = lerp(array, array_f, *(p_mask));
  }
}

} // namespace hmap::gpu
