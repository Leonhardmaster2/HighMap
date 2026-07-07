/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <cstdint>
#include <vector>

#include "cl_wrapper/run.hpp"

#include "highmap/array.hpp"
#include "highmap/gradient.hpp"
#include "highmap/hydrology/hydrology.hpp"
#include "highmap/opencl/gpu_opencl.hpp"

namespace hmap::gpu
{

Array flow_accumulation_stochastic(const Array  &z,
                                   int           n_samples,
                                   std::uint32_t seed,
                                   const Array  *p_source,
                                   const Array  *p_decay)
{
  Array vx = -gradient_x(z);
  Array vy = -gradient_y(z);

  // hmap::downhill_velocity(z, vx, vy);

  Array flux(z.shape); // zero-initialized

  {
    auto run = clwrapper::Run("flow_accum_stochastic_solve");
    run.bind_buffer<float>("vx", vx.vector);
    run.bind_buffer<float>("vy", vy.vector);
    helper_bind_optional_buffer(run, "source", p_source);
    helper_bind_optional_buffer(run, "decay", p_decay);
    run.bind_buffer<float>("flux", flux.vector);
    run.bind_arguments(p_source ? 1 : 0,
                       p_decay ? 1 : 0,
                       z.shape.x,
                       z.shape.y,
                       n_samples,
                       seed);
    run.write_buffer("vx");
    run.write_buffer("vy");
    run.write_buffer("flux");
    run.execute(n_samples); // 1D — matches the execute(int) convention used
                            // by advection_particle_gpu.cpp
    run.read_buffer("flux");
  }

  {
    // vx/vy are re-bound and re-uploaded here because each clwrapper::Run
    // allocates its own device buffers — there is no buffer sharing across
    // the two Run instances above.
    auto run = clwrapper::Run("flow_accum_stochastic_normalize");
    run.bind_buffer<float>("flux", flux.vector);
    run.bind_buffer<float>("vx", vx.vector);
    run.bind_buffer<float>("vy", vy.vector);
    helper_bind_optional_buffer(run, "source", p_source);
    run.bind_arguments(p_source ? 1 : 0, z.shape.x, z.shape.y, n_samples);
    run.write_buffer("flux");
    run.write_buffer("vx");
    run.write_buffer("vy");
    run.execute({z.shape.x, z.shape.y});
    run.read_buffer("flux");
  }

  return flux;
}

} // namespace hmap::gpu
