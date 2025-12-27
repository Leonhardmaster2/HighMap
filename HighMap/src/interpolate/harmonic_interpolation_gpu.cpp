/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/array.hpp"
#include "highmap/opencl/gpu_opencl.hpp"
#include "highmap/operator.hpp"

namespace hmap::gpu
{

Array harmonic_interpolation(const Array &array,
                             const Array &mask_fixed_values,
                             int          iterations_max)
{
  Array array_wrk = array;
  Array out(array.shape);

  auto run = clwrapper::Run("harmonic_interpolation");

  run.bind_buffer<float>("array_wrk", array_wrk.vector);
  run.bind_buffer<float>("out", out.vector);
  run.bind_buffer<float>("mask_fixed_values", mask_fixed_values.vector);
  run.bind_arguments(out.shape.x, out.shape.y);

  run.write_buffer("mask_fixed_values");

  for (int it = 0; it < iterations_max; ++it)
  {
    run.write_buffer("array_wrk");

    run.execute({array_wrk.shape.x, array_wrk.shape.y});

    run.read_buffer("out");
    swap(out, array_wrk);
  }

  return out;
}

} // namespace hmap::gpu
