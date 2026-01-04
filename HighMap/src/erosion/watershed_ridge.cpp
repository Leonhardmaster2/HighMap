/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "macrologger.h"

#include "highmap/array.hpp"
#include "highmap/erosion.hpp"
#include "highmap/hydrology.hpp"
#include "highmap/internal/vector_utils.hpp"
#include "highmap/math.hpp"
#include "highmap/morphology.hpp"
#include "highmap/range.hpp"

namespace hmap
{

Array watershed_ridge(const Array &z,
                      float        amplitude,
                      bool         smooth_ridge_crest,
                      float        edt_exponent)
{
  Array ze = z;
  Array ids = basin_id(z, hmap::FlowDirectionMethod::FDM_D8);

  ze = border(ids, 1);
  ze = distance_transform(ze);
  remap(ze);

  // apply it...
  if (smooth_ridge_crest) ze = smoothstep3_lower(ze);

  ze = pow(ze, edt_exponent);
  ze = z - amplitude * ze;

  return ze;
}

Array watershed_ridge(const Array &z,
                      Array       *p_mask,
                      float        amplitude,
                      bool         smooth_ridge_crest,
                      float        edt_exponent)
{
  if (!p_mask)
    return watershed_ridge(z, amplitude, smooth_ridge_crest, edt_exponent);
  else
  {
    Array z_f = watershed_ridge(z, amplitude, smooth_ridge_crest, edt_exponent);
    return lerp(z, z_f, *(p_mask));
  }
}

} // namespace hmap
