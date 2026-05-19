/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <vector> // for allocator, vector

#include "cl_wrapper/run.hpp" // for Run

#include "highmap/array.hpp" // for Array

namespace hmap::gpu
{

Array sparse_max_convolution(const Array &array,
                             const Array &kernel,
                             float        k_smoothmax)
{
  const glm::ivec2 &shape = array.shape;

  // no negative values, raises issue with atomic max in OpenCL
  const float offset = array.min();
  Array       out = array + offset;

  auto run = clwrapper::Run("sparse_max_convolution");

  run.bind_buffer<float>("array", out.vector);
  run.bind_imagef("kernel", kernel.vector, kernel.shape.x, kernel.shape.y);
  run.bind_buffer<float>("out", out.vector);

  run.write_buffer("array");
  run.write_buffer("out");

  run.bind_arguments(shape.x,
                     shape.y,
                     kernel.shape.x,
                     kernel.shape.y,
                     k_smoothmax);

  run.execute({shape.x, shape.y});

  run.read_buffer("out");

  return out - offset;
}

} // namespace hmap::gpu
