/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/boundary.hpp"
#include "highmap/filters.hpp"
#include "highmap/local_metrics.hpp"
#include "highmap/math.hpp"
#include "highmap/morphology.hpp"
#include "highmap/opencl/gpu_opencl.hpp"
#include "highmap/range.hpp"

namespace hmap::gpu
{

Array border(const Array &array, int ir)
{
  return array - gpu::erosion(array, ir);
}

Array closing(const Array &array, int ir)
{
  return gpu::erosion(gpu::dilation(array, ir), ir);
}

Array closing_by_reconstruction(const Array &array, int ir, float k_smooth_max)
{
  Array marker = gpu::dilation(array, ir);
  return gpu::reconstruction_by_erosion(marker, array, ir, k_smooth_max);
}

Array dilation(const Array &array, int ir)
{
  return gpu::local_max(array, ir);
}

Array dilation_expand_border_only(const Array &array, int ir)
{
  Array out = gpu::dilation(array, ir);

  // only keep result in the "background" to leave initial vlaues
  // untouched
  for (int j = 0; j < array.shape.y; ++j)
    for (int i = 0; i < array.shape.x; ++i)
    {
      if (array(i, j) != 0.f) out(i, j) = array(i, j);
    }

  return out;
}

Array erosion(const Array &array, int ir)
{
  return gpu::local_min(array, ir);
}

Array morphological_black_hat(const Array &array, int ir)
{
  return gpu::closing(array, ir) - array;
}

Array morphological_gradient(const Array &array, int ir)
{
  float vmin = array.min();
  return gpu::dilation(array - vmin, ir) - gpu::erosion(array - vmin, ir);
}

Array morphological_laplacian(const Array &array, int ir)
{
  float vmin = array.min();
  return gpu::dilation(array - vmin, ir) + gpu::erosion(array - vmin, ir) -
         2.f * array;
}

Array morphological_top_hat(const Array &array, int ir)
{
  return array - gpu::opening(array, ir);
}

Array opening(const Array &array, int ir)
{
  return gpu::dilation(gpu::erosion(array, ir), ir);
}

Array opening_by_reconstruction(const Array &array, int ir, float k_smooth_min)
{
  Array marker = gpu::erosion(array, ir);
  return gpu::reconstruction_by_dilation(marker, array, ir, k_smooth_min);
}

Array reconstruction_by_dilation(const Array &marker,
                                 const Array &mask,
                                 int          ir,
                                 float        k_smooth_min)
{
  constexpr float tol = 1e-6f;

  Array current = marker;
  Array next;

  while (true)
  {
    next = gpu::dilation(current, ir);

    // clamp to mask
    next = hmap::minimum_smooth(next, mask, k_smooth_min);

    float diff = abs(next - current).max();

    if (diff < tol) // convergence
      break;

    current = next;
  }

  return current;
}

Array reconstruction_by_erosion(const Array &marker,
                                const Array &mask,
                                int          ir,
                                float        k_smooth_max)
{
  constexpr float tol = 1e-6f;

  Array current = marker;
  Array next;

  while (true)
  {
    next = gpu::erosion(current, ir);

    // clamp to mask
    next = hmap::maximum_smooth(next, mask, k_smooth_max);

    float diff = abs(next - current).max();

    if (diff < tol) // convergence
      break;

    current = next;
  }

  return current;
}

Array relative_distance_from_skeleton(const Array &array,
                                      int          ir_search,
                                      bool         zero_at_borders,
                                      int          ir_erosion)
{
  Array border = array - gpu::erosion(array, ir_erosion);
  Array sk = gpu::skeleton(array, zero_at_borders);
  Array rdist(array.shape);

  auto run = clwrapper::Run("relative_distance_from_skeleton");

  run.bind_imagef("array",
                  const_cast<std::vector<float> &>(array.vector),
                  array.shape.x,
                  array.shape.y);
  run.bind_imagef("sk", sk.vector, array.shape.x, array.shape.y);
  run.bind_imagef("border", border.vector, array.shape.x, array.shape.y);
  run.bind_imagef("rdist", rdist.vector, array.shape.x, array.shape.y, true);
  run.bind_arguments(array.shape.x, array.shape.y, ir_search);

  run.execute({array.shape.x, array.shape.y});

  run.read_imagef("rdist");

  return rdist;
}

Array skeleton(const Array &array, bool zero_at_borders)
{
  Array sk = array;
  Array prev;
  Array diff;

  auto run = clwrapper::Run("thinning");

  run.bind_imagef("in", sk.vector, array.shape.x, array.shape.y);
  run.bind_imagef("out", sk.vector, array.shape.x, sk.shape.y, true);
  run.bind_arguments(array.shape.x, array.shape.y, 0);

  do
  {
    prev = sk;

    run.set_argument(4, 0); // pass 1
    run.write_imagef("in");
    run.execute({array.shape.x, array.shape.y});
    run.read_imagef("out");

    run.set_argument(4, 1); // pass 2
    run.write_imagef("in");
    run.execute({array.shape.x, array.shape.y});
    run.read_imagef("out");

    diff = sk - prev;

  } while (diff.count_non_zero() > 0);

  // set border to zero
  if (zero_at_borders) zeroed_borders(sk);

  return sk;
}

} // namespace hmap::gpu
