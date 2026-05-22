/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm> // for max, copy
#include <cmath>     // for abs
#include <utility>   // for swap
#include <vector>    // for allocator, vector

#include "cl_wrapper/run.hpp" // for Run

#include "highmap/array.hpp"         // for Array, operator*, count_non_zero
#include "highmap/boundary.hpp"      // for generate_buffered_array, set_bo...
#include "highmap/filters.hpp"       // for smooth_cpulse
#include "highmap/local_metrics.hpp" // for local_max, local_min
#include "highmap/math/array.hpp"    // for is_zero, smoothstep3
#include "highmap/morphology.hpp"    // for distance_transform, relative_di...
#include "highmap/range.hpp"         // for clamp, maximum_smooth, minimum_...

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

Array contour_smoothing(const Array &array, int ir, float transition_ratio)
{
  Array edt = distance_transform(is_zero(array)) - distance_transform(array);
  gpu::smooth_cpulse(edt, 2 * ir);

  float width = transition_ratio * ir;
  edt /= width;
  clamp(edt, -1.f, 1.f);
  edt = smoothstep3(0.5f * edt + 0.5f);

  return edt;
}

Array dilation(const Array &array, int ir)
{
  return gpu::local_max(array, ir);
}

Array dilation_expand_border_only(const Array &array, int ir)
{
  const glm::ivec2 &shape = array.shape;
  Array             out = gpu::dilation(array, ir);

  // only keep result in the "background" to leave initial vlaues
  // untouched
  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
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
  Array           current = marker;
  Array           next;

  while (true)
  {
    next = gpu::dilation(current, ir);
    next = hmap::minimum_smooth(next, mask, k_smooth_min);

    float diff = 0.f;
    for (int j = 0; j < current.shape.y; ++j)
      for (int i = 0; i < current.shape.x; ++i)
        diff = std::max(diff, std::abs(next(i, j) - current(i, j)));

    std::swap(current, next);
    if (diff < tol) break;
  }
  return current;
}

Array reconstruction_by_erosion(const Array &marker,
                                const Array &mask,
                                int          ir,
                                float        k_smooth_max)
{
  constexpr float tol = 1e-6f;
  Array           current = marker;
  Array           next;

  while (true)
  {
    next = gpu::erosion(current, ir);
    next = hmap::maximum_smooth(next, mask, k_smooth_max);

    float diff = 0.f;
    for (int j = 0; j < current.shape.y; ++j)
      for (int i = 0; i < current.shape.x; ++i)
        diff = std::max(diff, std::abs(next(i, j) - current(i, j)));

    std::swap(current, next);
    if (diff < tol) break;
  }
  return current;
}

Array relative_distance_from_skeleton(const Array &array,
                                      const Array &skeleton,
                                      int          ir_search,
                                      int          ir_erosion)
{
  const glm::ivec2 &shape = array.shape;

  Array border = array - gpu::erosion(array, ir_erosion);
  Array rdist(shape);

  auto run = clwrapper::Run("relative_distance_from_skeleton");

  run.bind_imagef("array", array.vector, shape.x, shape.y);
  run.bind_imagef("sk", skeleton.vector, shape.x, shape.y);
  run.bind_imagef("border", border.vector, shape.x, shape.y);
  run.bind_imagef("rdist", rdist.vector, shape.x, shape.y, true);
  run.bind_arguments(shape.x, shape.y, ir_search);

  run.execute({shape.x, shape.y});

  run.read_imagef("rdist");

  return rdist;
}

Array relative_distance_from_skeleton(const Array &array,
                                      int          ir_search,
                                      bool         zero_at_borders,
                                      int          ir_erosion)
{
  Array sk = gpu::skeleton(array, zero_at_borders);
  return gpu::relative_distance_from_skeleton(array, sk, ir_search, ir_erosion);
}

Array skeleton(const Array &array, bool zero_at_borders)
{
  Array sk = generate_buffered_array(array, {1, 1, 1, 1});
  set_borders(sk, 0.f, 1);

  const glm::ivec2 &shape_pad = sk.shape;

  Array prev;
  Array diff;

  auto run = clwrapper::Run("thinning");

  run.bind_imagef("in", sk.vector, shape_pad.x, shape_pad.y);
  run.bind_imagef("out", sk.vector, shape_pad.x, sk.shape.y, true);
  run.bind_arguments(shape_pad.x, shape_pad.y, 0);

  do
  {
    prev = sk;

    run.set_argument(4, 0); // pass 1
    run.write_imagef("in");
    run.execute({shape_pad.x, shape_pad.y});
    run.read_imagef("out");

    run.set_argument(4, 1); // pass 2
    run.write_imagef("in");
    run.execute({shape_pad.x, shape_pad.y});
    run.read_imagef("out");

    diff = sk - prev;

  } while (count_non_zero(diff) > 0);

  // remove padding
  sk = sk.extract_slice({1, sk.shape.x - 1, 1, sk.shape.y - 1});

  // set border to zero
  if (zero_at_borders) zeroed_borders(sk);

  return sk;
}

} // namespace hmap::gpu
