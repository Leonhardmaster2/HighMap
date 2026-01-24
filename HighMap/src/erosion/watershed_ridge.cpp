/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "macrologger.h"

#include "highmap/array.hpp"
#include "highmap/erosion.hpp"
#include "highmap/filters.hpp"
#include "highmap/hydrology.hpp"
#include "highmap/internal/vector_utils.hpp"
#include "highmap/math.hpp"
#include "highmap/morphology.hpp"
#include "highmap/range.hpp"
#include "highmap/transform.hpp"

namespace hmap
{

Array watershed_ridge(const Array        &z,
                      float               amplitude,
                      int                 ir,
                      float               edt_exponent,
                      FlowDirectionMethod fd_method)
{
  DrainageBasins   basins;
  bool             remove_lakes = true;
  const glm::ivec2 shape = z.shape;

  // retrieve main channel (valley bottom)
  basins.generate_traversal(z, fd_method, remove_lakes);

  auto mcs = basins.get_main_channels();

  // distance transform to generate valley shape
  Array edt(shape);

  for (const auto &vec : mcs)
    for (const auto &p : vec)
      edt(p) = 1.f;

  edt = distance_transform(edt);
  remap(edt);

  // filter but keep sharp valley bottoms
  Array mask = edt;
  for (auto &v : mask.vector)
    v = 1.f - std::exp(-v * v / 0.01f);
  smooth_cpulse(edt, ir, &mask);

  edt = 1.f - edt;

  // apply
  edt = pow(edt, edt_exponent);

  return z - amplitude * edt;
}

Array watershed_ridge(const Array        &z,
                      Array              *p_mask,
                      float               amplitude,
                      int                 ir,
                      float               edt_exponent,
                      FlowDirectionMethod fd_method)
{
  if (!p_mask)
    return watershed_ridge(z, amplitude, ir, edt_exponent, fd_method);
  else
  {
    Array z_f = watershed_ridge(z, amplitude, ir, edt_exponent, fd_method);
    return lerp(z, z_f, *(p_mask));
  }
}

} // namespace hmap

namespace hmap::gpu
{

Array watershed_ridge(const Array        &z,
                      float               amplitude,
                      int                 ir,
                      float               edt_exponent,
                      FlowDirectionMethod fd_method,
                      const Array        *p_noise_x,
                      const Array        *p_noise_y)
{
  Array        ze;
  const Array *p_z;

  if (p_noise_x || p_noise_y)
  {
    ze = z;
    gpu::warp(ze, p_noise_x, p_noise_y);
    p_z = &ze;
  }
  else
  {
    p_z = &z;
  }

  DrainageBasins   basins;
  bool             remove_lakes = true;
  const glm::ivec2 shape = z.shape;

  // retrieve main channel (valley bottom)
  basins.generate_traversal(*p_z, fd_method, remove_lakes);

  auto mcs = basins.get_main_channels();

  // distance transform to generate valley shape
  Array edt(shape);

  for (const auto &vec : mcs)
    for (const auto &p : vec)
      edt(p) = 1.f;

  edt = distance_transform(edt);
  remap(edt);

  // filter but keep sharp valley bottoms
  Array mask = edt;
  for (auto &v : mask.vector)
    v = 1.f - std::exp(-v * v / 0.01f);
  gpu::smooth_cpulse(edt, ir, &mask);

  edt = 1.f - edt;

  // apply
  edt = pow(edt, edt_exponent);

  return z - amplitude * edt;
}

Array watershed_ridge(const Array        &z,
                      Array              *p_mask,
                      float               amplitude,
                      int                 ir,
                      float               edt_exponent,
                      FlowDirectionMethod fd_method,
                      const Array        *p_noise_x,
                      const Array        *p_noise_y)
{
  if (!p_mask)
    return gpu::watershed_ridge(z,
                                amplitude,
                                ir,
                                edt_exponent,
                                fd_method,
                                p_noise_x,
                                p_noise_y);
  else
  {
    Array z_f = gpu::watershed_ridge(z,
                                     amplitude,
                                     ir,
                                     edt_exponent,
                                     fd_method,
                                     p_noise_x,
                                     p_noise_y);
    return lerp(z, z_f, *(p_mask));
  }
}

} // namespace hmap::gpu
