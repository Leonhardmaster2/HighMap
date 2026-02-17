R""(
/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

float2 phasor_kernel(const float2 p,
                     const float  kp,
                     const float  angle,
                     const float  phi,
                     const float  radius)
{
  float2 pn = p / radius;
  float  r = length(pn);

  if (r > 1.f) return 0.f;

  float ca = cos(angle);
  float sa = sin(angle);

  float a = 1.f - r * r * (3.f - 2.f * r);
  float rr = pn.x * ca + pn.y * sa;
  float c = cos(3.14159f * kp * rr + phi);
  float s = sin(3.14159f * kp * rr + phi);
  return (float2)(a * c, a * s);
}

float2 base_phasor_cell(const float2 p,
                        const float2 cell_ij,
                        const float2 jitter,
                        const int    n_kernel_samples,
                        const float  kp,
                        const float  angle,
                        const float  fseed)
{
  float2 sum = 0.f;

  for (int n = 0; n < n_kernel_samples; ++n)
  {
    float2 p_relative = p - cell_ij;
    p_relative -= jitter * hash22f(cell_ij + (float2)(n, 0), fseed);

    // random phase
    float phi = 2.f * 3.14159f * hash12f(cell_ij + (float2)(n, 0), fseed);
    float width = 1.f;

    sum += phasor_kernel(p_relative, kp, angle, phi, width);
  }

  return sum;
}

float2 base_phase_field(const float2 p,
                        const float2 jitter,
                        const int    n_kernel_samples,
                        const float  kp,
                        const float  angle,
                        const float  fseed)
{
  float2 cell_ij = floor(p);
  float2 pi;
  float2 sum = 0.f;

  for (int dx = -1; dx <= 1; ++dx)
    for (int dy = -1; dy <= 1; ++dy)
    {
      float2 dr = (float2)(dx, dy);
      sum += base_phasor_cell(p,
                              cell_ij + dr,
                              jitter,
                              n_kernel_samples,
                              kp,
                              angle,
                              fseed);
    }

  return sum;
}

void kernel phase_field(global float *angle,
                        global float *phase,
                        global float *ctrl_param,
                        global float *noise_x,
                        global float *noise_y,
                        global float *modulus,
                        const int     nx,
                        const int     ny,
                        const float   kx,
                        const float   ky,
                        const uint    seed,
                        const float2  jitter,
                        const int     n_kernel_samples,
                        const float   kp,
                        const int     has_ctrl_param,
                        const int     has_noise_x,
                        const int     has_noise_y,
                        const int     has_modulus,
                        const float4  bbox)
{
  int2 g = {get_global_id(0), get_global_id(1)};

  if (g.x >= nx || g.y >= ny) return;

  int idx = linear_index(g.x, g.y, nx);

  uint  rng_state = wang_hash(seed);
  float fseed = rand(&rng_state);

  float ct = has_ctrl_param > 0 ? ctrl_param[idx] : 1.f; // kp muliplier
  float dx = has_noise_x > 0 ? noise_x[idx] : 0.f;
  float dy = has_noise_y > 0 ? noise_y[idx] : 0.f;

  float2 pos = g_to_xy(g, nx, ny, kx, ky, dx, dy, bbox);

  float val = 0.f;

  float2 gabor = base_phase_field(pos,
                                  jitter,
                                  n_kernel_samples,
                                  ct * kp,
                                  angle[idx],
                                  fseed);

  phase[idx] = atan2(gabor.y, gabor.x);

  if (has_modulus) modulus[idx] = hypot(gabor.x, gabor.y);
}
)""
