/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <vector> // for allocator, vector

#include "cl_wrapper/run.hpp" // for Run

#include "highmap/array.hpp"      // for Array, operator*
#include "highmap/filters.hpp"    // for smooth_cpulse
#include "highmap/gradient.hpp"   // for gradient_x, gradient_y, gradient...
#include "highmap/math/array.hpp" // for atan2, hypot
#include "highmap/range.hpp"      // for maximum

namespace hmap::gpu
{

Array gradient_angle_circular_smoothing(const Array &array,
                                        int          ir,
                                        bool         downward)
{
  // gradients
  Array dx = gradient_x(array);
  Array dy = gradient_y(array);
  Array dn = hypot(dx, dy);
  Array dn_safe = maximum(dn, 1e-9f);

  if (downward)
  {
    dx = -1.f * dx;
    dy = -1.f * dy;
  }

  // angle to unit vector
  Array u = dx / dn_safe;
  Array v = dy / dn_safe;

  // smoothing
  gpu::smooth_cpulse(u, ir);
  gpu::smooth_cpulse(v, ir);
  gpu::smooth_cpulse(dn, ir);

  // renormalize and compute the angle
  dn_safe = maximum(dn, 1e-9f);
  u /= dn_safe;
  v /= dn_safe;

  return atan2(v, u);
}

Array gradient_norm(const Array &array)
{
  Array dm(array.shape);

  auto run = clwrapper::Run("gradient_norm");

  run.bind_buffer<float>("array",
                         const_cast<std::vector<float> &>(array.vector));
  run.bind_buffer<float>("dm", dm.vector);
  run.bind_arguments(array.shape.x, array.shape.y);

  run.write_buffer("array");

  run.execute({array.shape.x, array.shape.y});

  run.read_buffer("dm");

  return dm;
}

Array laplacian_fract(const Array &array, float s, int ir)
{
  Array out(array.shape);

  auto run = clwrapper::Run("laplacian_fract");

  run.bind_imagef("array",
                  const_cast<std::vector<float> &>(array.vector),
                  array.shape.x,
                  array.shape.y);
  run.bind_imagef("out", out.vector, array.shape.x, array.shape.y, true);

  run.bind_arguments(array.shape.x, array.shape.y, ir, s);

  run.execute({array.shape.x, array.shape.y});

  run.read_imagef("out");

  return out;
}

} // namespace hmap::gpu
