/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/curvature.hpp"
#include "highmap/filters.hpp"
#include "highmap/morphology.hpp"
#include "highmap/opencl/gpu_opencl.hpp"
#include "highmap/range.hpp"

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

  // curvature
  Array cm = z;

  if (ir_curvature) gpu::smooth_cpulse(cm, ir_curvature);

  cm = gradient_scaling_factor * curvature_mean(cm);
  clamp(cm, curvature_clamping, curvature_clamp_mode);

  // gradient (scaling is empirical)
  Array dn = gpu::morphological_gradient(z, ir_gradient) *
             gradient_scaling_factor / 32.f / ir_gradient;

  return curvature_weight * cm + gradient_weight * dn;
}

} // namespace hmap::gpu
