R""(
/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

float stratify(const float val,
               const float shift,
               const float kz,
               const float gamma)
{
  float out = val + shift;
  out *= kz;

  // apply recruve function using a cell-based (but pointwise)
  // approach
  float i = floor(out);
  float u = out - i;

  // smooth gamma
  {
    float ce = 50.f / gamma;
    u = pow(u, gamma) * (1.f - exp(-ce * u));
  }

  // revert back value to input range
  out = (i + u) / kz - shift;

  return out;
}

void kernel strata(global float *output,
                   const int     nx,
                   const int     ny,
                   const float   angle,
                   const float   slope,
                   const float   gamma,
                   const uint    seed,
                   const float   kz,
                   const int     octaves,
                   const float   lacunarity,
                   const float   gamma_noise_ratio,
                   const float   noise_amp,
                   const float2  noise_kw,
                   const float2  ridge_noise_kw,
                   const float   ridge_angle_shift,
                   const float   ridge_noise_amp,
                   const float   ridge_clamp_vmin,
                   const float   ridge_remap_vmin,
                   const float   mask_gamma,
                   const float4  bbox)
{
  /* float kz = 1.f; */
  /* int   octaves = 4; */
  /* float lacunarity = 2.f; */
  /* float gamma = 0.6f; */
  /* float gamma_noise_ratio = 0.5f; */
  /* // */
  /* float noise_amp = 0.2f; */
  /* float noise_kw = 4.f; */
  /* // */
  /* float2 ridge_noise_kw = (float2)(4.f, 1.2f); */
  /* float  ridge_angle_shift = 45.f; */
  /* float  ridge_noise_amp = 0.5f; */
  /* float  ridge_clamp_vmin = 0.f; */
  /* float  ridge_remap_vmin = 0.f; */
  /* // */
  /* float mask_gamma = 0.4f; */

  // ---

  int2 g = {get_global_id(0), get_global_id(1)};

  if (g.x >= nx || g.y >= ny) return;

  int index = linear_index(g.x, g.y, nx);

  uint  rng_state = wang_hash(seed);
  float fseed = rand(&rng_state);

  float2 pos = g_to_xy(g, nx, ny, 1.f, 1.f, 0.f, 0.f, bbox);

  const float val0 = output[index];
  float       val = val0;
  float       alpha = angle / 180.f * 3.141592f;
  float       alpha_ridge = (angle + ridge_angle_shift) / 180.f * 3.141592f;
  float2      dir = (float2)(cos(alpha), sin(alpha));

  float dr = dot(pos, dir);
  float shift = slope * dr;

  // TODO hardcoded
  float noise = base_perlin_fbm(noise_kw * pos, 6, 1.f, 0.99f, 2.f, fseed);

  // ridges
  float  dr_p = dot(pos, (float2)(cos(alpha_ridge), sin(alpha_ridge)));
  float2 pos_v = (float2)(ridge_noise_kw.x * dr_p + ridge_noise_amp * noise,
                          ridge_noise_kw.y * dr);

  float noise_v = base_voronoi_f1df2(pos_v, (float2)(1.f, 1.f), 0.f, fseed);
  noise_v = clamp(noise_v, ridge_clamp_vmin, 1.f);
  noise_v = remap_from(noise_v,
                       1.f,
                       ridge_remap_vmin,
                       ridge_clamp_vmin,
                       1.f); // reverse

  float gamma_r = gamma * (1.f + gamma_noise_ratio * noise);
  gamma_r = clamp(gamma_r, 0.05f, 10.f);

  // multiscale
  float kz_n = kz;

  for (int k = 0; k < octaves; ++k)
  {
    val = stratify(val, shift + noise_amp * noise / kz_n, kz_n, gamma_r);
    kz_n *= lacunarity;
  }

  // out
  output[index] = lerp(val0, val, pow(val0, mask_gamma) * noise_v);
}
)""
