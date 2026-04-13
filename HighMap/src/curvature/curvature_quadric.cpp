/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/boundary.hpp"
#include "highmap/curvature.hpp"
#include "highmap/opencl/gpu_opencl.hpp"

namespace hmap::gpu
{

Array curvature_quadric(const Array &z, int ir, CurvatureType curvature_type)
{
  const glm::ivec2 shape = z.shape;
  int              type_id = static_cast<int>(curvature_type);

  Array out(shape);

  auto run = clwrapper::Run("curvature_quadric");

  run.bind_imagef("z",
                  const_cast<std::vector<float> &>(z.vector),
                  shape.x,
                  shape.y);
  run.bind_imagef("out", out.vector, shape.x, shape.y, true);

  run.bind_arguments(shape.x, shape.y, ir, type_id);

  run.execute({shape.x, shape.y});

  run.read_imagef("out");

  if (ir == 0) extrapolate_borders(out);

  return out;
}

} // namespace hmap::gpu
