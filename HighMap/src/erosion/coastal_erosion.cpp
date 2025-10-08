/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/array.hpp"
#include "highmap/filters.hpp"
#include "highmap/hydrology.hpp"

namespace hmap
{

void coastal_erosion_diffusion(Array &z,
                               Array &water_depth,
                               float  additional_depth,
                               int    iterations)
{
  for (int it = 0; it < iterations; ++it)
  {
    const Array z_bckp = z;
    const Array mask = water_mask(water_depth, z, additional_depth);

    // filtering
    hmap::laplace(z, &mask, /* sigma */ 0.125f, 1);

    // adjust water depth so that water height is the same as before
    // filtering
    for (int j = 0; j < z.shape.y; j++)
      for (int i = 0; i < z.shape.x; i++)
      {
        if (water_depth(i, j) > 0.f)
          water_depth(i, j) = z_bckp(i, j) + water_depth(i, j) - z(i, j);
      }
  }
}

} // namespace hmap
