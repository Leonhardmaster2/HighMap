/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/erosion.hpp"
#include "highmap/filters.hpp"
#include "highmap/math.hpp"
#include "highmap/range.hpp"

namespace hmap::gpu
{

void valley_fill(Array       &z,
                 const Array &talus,
                 int          iterations,
                 float        gamma,
                 float        ratio,
                 float        zmin,
                 float        zmax)
{
  if (zmax <= zmin)
  {
    zmin = z.min();
    zmax = z.max();
  }

  // scree deposition
  Array ze = z;
  gpu::thermal_scree(ze, talus, Array(z.shape, zmax), iterations);
  remap(ze, zmin, zmax);

  // mixing mask
  Array t = z;
  remap(t, 0.f, 1.f, zmin, zmax);
  gamma_correction(t, gamma);
  remap(t, 1.f - ratio, 1.f);

  // apply
  z = lerp(ze, z, t);
}

} // namespace hmap::gpu
