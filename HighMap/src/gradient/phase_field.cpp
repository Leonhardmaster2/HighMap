/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <sys/types.h> // for uint

#include <cmath>  // for M_PI, sqrt
#include <vector> // for allocator, vector

#include "cl_wrapper/run.hpp" // for Run

#include "highmap/array.hpp"             // for Array
#include "highmap/gradient.hpp"          // for talus_jump_mask, gradient_x
#include "highmap/math/array.hpp"        // for atan2
#include "highmap/opencl/gpu_opencl.hpp" // for helper_bind_optional_buffer

namespace hmap::gpu
{

void phase_averaging(Array &field_real, Array &field_imag, int ir)
{
  const glm::ivec2 shape = field_real.shape;

  auto run = clwrapper::Run("phase_averaging");

  // inputs
  run.bind_imagef("fr", field_real.vector, shape.x, shape.y);
  run.bind_imagef("fi", field_imag.vector, shape.x, shape.y);

  // outputs
  run.bind_imagef("fr_out", field_real.vector, shape.x, shape.y, true);
  run.bind_imagef("fi_out", field_imag.vector, shape.x, shape.y, true);

  run.bind_arguments(shape.x, shape.y, ir);

  run.execute({shape.x, shape.y});

  // update flux (from GPU to CPU)
  run.read_imagef("fr_out");
  run.read_imagef("fi_out");
}

Array phase_field(const Array     &array,
                  const glm::vec2 &kw,
                  uint             seed,
                  float            kp,
                  bool             rotate90,
                  int              n_kernel_samples,
                  const glm::vec2 &jitter,
                  int              angle_filter_ir,
                  const Array     *p_ctrl_param,
                  const Array     *p_noise_x,
                  const Array     *p_noise_y,
                  Array           *p_modulus,
                  Array           *p_angle_jump_mask,
                  glm::vec4        bbox)
{
  const glm::ivec2 shape = array.shape;

  // --- compute local angle

  float phi = rotate90 ? M_PI : 0.5 * M_PI;
  Array dx = gradient_x(array);
  Array dy = gradient_y(array);
  phase_averaging(dx, dy, angle_filter_ir);
  Array angle = atan2(dy, dx) + phi;

  if (p_angle_jump_mask)
    *p_angle_jump_mask = talus_jump_mask(angle, M_PI, 0.1f * M_PI);

  // --- compute phase

  Array phase(shape);

  auto run = clwrapper::Run("phase_field");

  run.bind_buffer<float>("angle", angle.vector);
  run.bind_buffer<float>("phase", phase.vector);

  helper_bind_optional_buffer(run, "ctrl_param", p_ctrl_param);
  helper_bind_optional_buffer(run, "p_noise_x", p_noise_x);
  helper_bind_optional_buffer(run, "p_noise_y", p_noise_y);
  helper_bind_optional_buffer(run, "p_modulus", p_modulus);

  run.bind_arguments(shape.x,
                     shape.y,
                     kw.x,
                     kw.y,
                     seed,
                     jitter,
                     n_kernel_samples,
                     kp,
                     p_ctrl_param ? 1 : 0,
                     p_noise_x ? 1 : 0,
                     p_noise_y ? 1 : 0,
                     p_modulus ? 1 : 0,
                     bbox);

  run.write_buffer("angle");
  run.write_buffer("phase");

  run.execute({shape.x, shape.y});

  run.read_buffer("phase");
  if (p_modulus) run.read_buffer("p_modulus");

  return phase;
}

Array phase_field(const Array     &array,
                  uint             seed,
                  float            kp_global,
                  bool             rotate90,
                  int              n_kernel_samples,
                  const glm::vec2 &jitter,
                  int              angle_filter_ir,
                  const Array     *p_ctrl_param,
                  const Array     *p_noise_x,
                  const Array     *p_noise_y,
                  Array           *p_modulus,
                  Array           *p_angle_jump_mask,
                  glm::vec4        bbox)
{
  float           kp = std::sqrt(kp_global);
  const glm::vec2 kw = {kp, kp};

  return phase_field(array,
                     kw,
                     seed,
                     kp,
                     rotate90,
                     n_kernel_samples,
                     jitter,
                     angle_filter_ir,
                     p_ctrl_param,
                     p_noise_x,
                     p_noise_y,
                     p_modulus,
                     p_angle_jump_mask,
                     bbox);
}

Array phase_field_angle(const Array     &angle,
                        const glm::vec2 &kw,
                        uint             seed,
                        float            kp,
                        int              n_kernel_samples,
                        const glm::vec2 &jitter,
                        const Array     *p_ctrl_param,
                        const Array     *p_noise_x,
                        const Array     *p_noise_y,
                        Array           *p_modulus,
                        Array           *p_angle_jump_mask,
                        glm::vec4        bbox)
{
  const glm::ivec2 shape = angle.shape;

  if (p_angle_jump_mask)
    *p_angle_jump_mask = talus_jump_mask(angle, M_PI, 0.1f * M_PI);

  // --- compute phase

  Array phase(shape);

  auto run = clwrapper::Run("phase_field");

  run.bind_buffer<float>("angle", angle.vector);
  run.bind_buffer<float>("phase", phase.vector);

  helper_bind_optional_buffer(run, "ctrl_param", p_ctrl_param);
  helper_bind_optional_buffer(run, "p_noise_x", p_noise_x);
  helper_bind_optional_buffer(run, "p_noise_y", p_noise_y);
  helper_bind_optional_buffer(run, "p_modulus", p_modulus);

  run.bind_arguments(shape.x,
                     shape.y,
                     kw.x,
                     kw.y,
                     seed,
                     jitter,
                     n_kernel_samples,
                     kp,
                     p_ctrl_param ? 1 : 0,
                     p_noise_x ? 1 : 0,
                     p_noise_y ? 1 : 0,
                     p_modulus ? 1 : 0,
                     bbox);

  run.write_buffer("angle");
  run.write_buffer("phase");

  run.execute({shape.x, shape.y});

  run.read_buffer("phase");
  if (p_modulus) run.read_buffer("p_modulus");

  return phase;
}

} // namespace hmap::gpu
