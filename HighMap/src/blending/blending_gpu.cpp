/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <vector>

#include "cl_wrapper/run.hpp"

#include "highmap/array.hpp"
#include "highmap/filters.hpp"
#include "highmap/gradient.hpp"
#include "highmap/internal/validation.hpp"
#include "highmap/math/array.hpp"
#include "highmap/opencl/gpu_opencl.hpp"
#include "highmap/range.hpp"
#include "highmap/texture.hpp"

namespace hmap::gpu
{

Array blend_gradients(const Array &array1, const Array &array2, int ir)
{
  if (!validate_non_empty(array1) || !validate_same_shape(array1, array2))
    return Array();

  Array dn1 = hmap::gradient_norm(array1);
  Array dn2 = hmap::gradient_norm(array2);

  gpu::smooth_cpulse(dn1, ir);
  gpu::smooth_cpulse(dn2, ir);

  Array t = gpu::maximum_smooth(dn1, dn2, 0.1f / (float)array1.shape.x);
  remap(t);

  return lerp(array1, array2, t);
}

Array blend_poisson_bf(const Array &array1,
                       const Array &array2,
                       const int    iterations,
                       const Array *p_mask)
{
  if (!validate_non_empty(array1) || !validate_same_shape(array1, array2))
    return Array();
  if (p_mask && !validate_same_shape(array1, *p_mask)) return Array();

  Array array1_out = array1;

  auto run = clwrapper::Run("blend_poisson_bf");

  run.bind_buffer<float>("array1_out", array1_out.vector);
  run.bind_buffer<float>("array2", array2.vector);
  helper_bind_optional_buffer(run, "mask", p_mask);

  run.bind_arguments(array1.shape.x, array1.shape.y, p_mask ? 1 : 0);

  run.write_buffer("array1_out");
  run.write_buffer("array2");

  for (int it = 0; it < iterations; it++)
    run.execute({array1.shape.x, array1.shape.y});

  run.read_buffer("array1_out");

  return array1_out;
}

Texture blend_poisson_bf(const Texture &texture1,
                         const Texture &texture2,
                         const int      iterations,
                         const Array   *p_mask)
{
  if (!validate_non_empty(texture1) || !validate_same_shape(texture1, texture2))
    return Texture();
  if (!validate_channels(texture2, texture1.num_channels())) return Texture();
  if (p_mask && !validate_same_shape(texture1, *p_mask)) return Texture();

  std::vector<Array> blended_channels;
  blended_channels.reserve(texture1.num_channels());

  for (size_t c = 0; c < texture1.channels.size(); ++c)
  {
    blended_channels.push_back(blend_poisson_bf(texture1.channels[c],
                                                texture2.channels[c],
                                                iterations,
                                                p_mask));
  }

  return Texture(blended_channels);
}

Array transfer(const Array &source,
               const Array &target,
               int          ir,
               float        amplitude,
               bool         target_prefiltering)
{
  if (!validate_non_empty(source) || !validate_same_shape(source, target))
    return Array();

  // high-pass spatial filter
  Array w = -source;
  gpu::smooth_cpulse(w, ir);
  w += source;

  if (target_prefiltering)
  {
    Array target_f = target;
    gpu::smooth_cpulse(target_f, ir);
    w = target_f + amplitude * w;
  }
  else
  {
    w = target + amplitude * w;
  }

  return w;
}

} // namespace hmap::gpu
