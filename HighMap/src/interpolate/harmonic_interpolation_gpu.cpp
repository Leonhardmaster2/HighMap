/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <vector>

#include "cl_wrapper/run.hpp"

#include "highmap/array.hpp"
#include "highmap/operator.hpp"

namespace hmap::gpu
{

Array harmonic_interpolation(const Array &array,
                             const Array &mask_fixed_values,
                             int          iterations_max,
                             float        omega)
{
  Array u = array;

  auto run = clwrapper::Run("harmonic_interpolation_red_black");

  run.bind_buffer<float>("u", u.vector);
  run.bind_buffer<float>("mask_fixed_values", mask_fixed_values.vector);
  run.bind_arguments(u.shape.x, u.shape.y, 0, omega);

  run.write_buffer("u");
  run.write_buffer("mask_fixed_values");

  for (int it = 0; it < iterations_max; ++it)
  {
    // Pass 0: Red cells ((i + j) % 2 == 0)
    run.set_argument(4, 0);
    run.execute({u.shape.x, u.shape.y});

    // Pass 1: Black cells ((i + j) % 2 == 1)
    run.set_argument(4, 1);
    run.execute({u.shape.x, u.shape.y});
  }

  run.read_buffer("u");

  return u;
}

Array harmonic_interpolation_legacy_bf(const Array &array,
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
