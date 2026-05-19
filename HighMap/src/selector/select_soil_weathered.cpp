/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/array.hpp"      // for Array, operator*
#include "highmap/curvature.hpp"  // for CurvatureType, curvature_quadric
#include "highmap/filters.hpp"    // for smooth_cpulse
#include "highmap/morphology.hpp" // for morphological_gradient
#include "highmap/range.hpp"      // for ClampMode, clamp
#include "highmap/selector.hpp"   // for select_soil_weathered

namespace hmap::gpu
{

Array select_soil_weathered(const Array &z,
                            int          ir_curvature,
                            int          ir_gradient,
                            ClampMode    curvature_clamp_mode,
                            float        curvature_clamping,
                            float        curvature_weight,
                            float        gradient_weight,
                            float        gradient_scaling_factor)
{
  if (gradient_scaling_factor <= 0.f) gradient_scaling_factor = z.shape.x;

  // gradient (scaling is empirical)
  Array dn = gpu::morphological_gradient(z, ir_gradient) *
             gradient_scaling_factor / 32.f / ir_gradient;

  return select_soil_weathered(z,
                               dn,
                               ir_curvature,
                               curvature_clamp_mode,
                               curvature_clamping,
                               curvature_weight,
                               gradient_weight,
                               gradient_scaling_factor);
}

Array select_soil_weathered(const Array &z,
                            const Array &gradient_norm,
                            int          ir_curvature,
                            ClampMode    curvature_clamp_mode,
                            float        curvature_clamping,
                            float        curvature_weight,
                            float        gradient_weight,
                            float        gradient_scaling_factor)
{
  // curvature
  Array cm = z;

  if (ir_curvature) gpu::smooth_cpulse(cm, ir_curvature);

  cm = -gradient_scaling_factor *
       gpu::curvature_quadric(cm, 0, CurvatureType::CT_MEAN);
  clamp(cm, curvature_clamping, curvature_clamp_mode);

  return curvature_weight * cm + gradient_weight * gradient_norm;
}

} // namespace hmap::gpu
