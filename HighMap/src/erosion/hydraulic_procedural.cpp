/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <functional>
#include <random>

#include "macrologger.h"

#include "highmap/array.hpp"
#include "highmap/blending.hpp"
#include "highmap/erosion.hpp"
#include "highmap/filters.hpp"
#include "highmap/gradient.hpp"
#include "highmap/morphology.hpp"
#include "highmap/opencl/gpu_opencl.hpp"
#include "highmap/operator.hpp"
#include "highmap/primitives.hpp"
#include "highmap/range.hpp"

namespace hmap::gpu
{

void hydraulic_procedural(Array         &z,
                          float          kp_global,
                          float          c_erosion,
                          uint           seed,
                          ErosionProfile erosion_profile,
                          float          erosion_profile_parameter,
                          float          angle_shift,
                          float          phase_smoothing,
                          float          talus_ref,
                          float          gradient_scaling_ratio,
                          float          gradient_power,
                          bool           apply_deposition,
                          float          deposition_strength,
                          bool           enable_default_noise,
                          float          noise_amp,
                          const Array   *p_kp_multiplier,
                          const Array   *p_angle_shift,
                          const Array   *p_noise_x,
                          const Array   *p_noise_y,
                          Array         *p_ridge_mask,
                          glm::vec4      bbox)
{

  // ---Resolve derived parameters

  const glm::ivec2 shape = z.shape;
  const int        kp_ir = int(1.f * shape.x / kp_global);
  const int        angle_filter_ir = kp_ir;
  const bool       rotate90 = false;
  const int        n_kernel_samples = 16;
  const glm::vec2  jitter = {0.5f, 0.5f};
  const int        gradient_prefilter_ir = kp_ir;

  // --- Default noise

  Array dx, dy;

  if (enable_default_noise)
  {
    const glm::vec2 kw_noise = {kp_global, kp_global};

    dx = noise_amp * gpu::noise_fbm(hmap::NoiseType::SIMPLEX2,
                                    z.shape,
                                    kw_noise,
                                    ++seed,
                                    /* octaves */ 8,
                                    /* weight*/ 0.7f,
                                    /* persistence */ 0.5f,
                                    /* lacunarity */ 2.f,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    bbox);

    dy = noise_amp * gpu::noise_fbm(hmap::NoiseType::SIMPLEX2,
                                    z.shape,
                                    kw_noise,
                                    ++seed,
                                    /* octaves */ 8,
                                    /* weight*/ 0.7f,
                                    /* persistence */ 0.5f,
                                    /* lacunarity */ 2.f,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    bbox);

    p_noise_x = &dx;
    p_noise_y = &dy;
  }

  // --- Compute phase field

  Array field_x(shape);
  Array field_y(shape);

  Array phase = gpu::phase_field(z,
                                 seed,
                                 kp_global,
                                 rotate90,
                                 n_kernel_samples,
                                 jitter,
                                 angle_filter_ir,
                                 // nullptr,
                                 p_kp_multiplier,
                                 p_noise_x,
                                 p_noise_y,
                                 &field_x,
                                 &field_y,
                                 bbox);

  // add optional local or global angle shifts
  if (angle_shift != 0.) phase += angle_shift / 180.f * M_PI;

  if (p_angle_shift) phase += *p_angle_shift;

  // --- Ridge profile

  float profile_avg = 0.f;
  auto  pfct = get_erosion_profile_function(erosion_profile,
                                           erosion_profile_parameter,
                                           profile_avg);

  Array ridge(shape);

  for (int j = 0; j < shape.y; j++)
    for (int i = 0; i < shape.x; i++)
    {
      float value = pfct(phase(i, j));

      // modulus-based smoothing to avoid kinky transitions between
      // phases
      float t = 2.f / M_PI *
                std::atan(phase_smoothing *
                          std::hypot(field_x(i, j), field_y(i, j)));

      ridge(i, j) = lerp(profile_avg, value, t);
    }

  // --- Erosion amplitude

  Array amp = flow_accumulation_dinf(z, talus_ref);
  amp = log10(amp);

  // scale erosion with local gradient
  Array gn = hmap::gradient_norm(z);

  {
    gpu::smooth_cpulse(gn, gradient_prefilter_ir);
    remap(gn);
    gn = pow(gn, gradient_power);
    gn = smoothstep5_lower(gn);
    amp *= (1.f - gradient_scaling_ratio) + gradient_scaling_ratio * gn;
  }

  gpu::smooth_cpulse(amp, kp_ir);
  remap(amp);

  ridge *= amp;

  // --- erode

  z -= c_erosion * ridge;

  // --- mimic deposition

  if (apply_deposition)
  {
    int   deposition_ir = kp_ir;
    Array zd = z;
    gpu::smooth_fill_holes(zd, deposition_ir);
    zd = gpu::blend_gradients(zd, z, deposition_ir);
    z = lerp(z, zd, deposition_strength);
  }

  // --- optional output

  if (p_ridge_mask) *p_ridge_mask = cos(phase);
}

void hydraulic_procedural_fbm(Array         &z,
                              float          kp_global,
                              float          c_erosion,
                              uint           seed,
                              ErosionProfile erosion_profile,
                              int            octaves,
                              float          persistence,
                              float          lacunarity,
                              float          erosion_profile_parameter,
                              float          angle_shift,
                              float          phase_smoothing,
                              float          talus_ref,
                              float          gradient_scaling_ratio,
                              float          gradient_power,
                              bool           apply_deposition,
                              float          deposition_strength,
                              bool           enable_default_noise,
                              float          noise_amp,
                              const Array   *p_kp_multiplier,
                              const Array   *p_angle_shift,
                              const Array   *p_noise_x,
                              const Array   *p_noise_y,
                              Array         *p_ridge_mask,
                              glm::vec4      bbox)
{
  float a = 1.f;
  float m = 1.f;

  Array ridge_mask = Array(z.shape);
  float a_sum = 0.f;

  for (int k = 0; k < octaves; ++k)
  {
    Array ridge_mask_current(z.shape);

    hmap::gpu::hydraulic_procedural(z,
                                    m * kp_global,
                                    a * c_erosion,
                                    seed,
                                    erosion_profile,
                                    erosion_profile_parameter,
                                    angle_shift,
                                    phase_smoothing,
                                    talus_ref,
                                    gradient_scaling_ratio,
                                    gradient_power,
                                    apply_deposition,
                                    deposition_strength,
                                    enable_default_noise,
                                    noise_amp,
                                    p_kp_multiplier,
                                    p_angle_shift,
                                    p_noise_x,
                                    p_noise_y,
                                    &ridge_mask_current,
                                    bbox);

    ridge_mask += a * ridge_mask_current;
    a_sum += a;

    a *= persistence;
    m *= lacunarity;
  }

  if (p_ridge_mask) *p_ridge_mask = 0.5f * (ridge_mask / a_sum) + 0.5f;
}

} // namespace hmap::gpu
