R""(
/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

kernel void shallow_viscous_flow(read_only image2d_t  z,
                                 read_only image2d_t  h_in,
                                 write_only image2d_t h_out,
                                 int                  nx,
                                 int                  ny,
                                 float                dt,
                                 float                viscosity,
                                 float                power)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  if (i >= nx || j >= ny) return;

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
  const float diag = 0.70710678f;

  float h = TGET(h_in, i, j);
  float zc = TGET(z, i, j);

  // neighbors
  float hxp = TGET(h_in, i + 1, j);
  float hxm = TGET(h_in, i - 1, j);
  float hyp = TGET(h_in, i, j + 1);
  float hym = TGET(h_in, i, j - 1);

  float Hc = h + zc;
  float Hxp = hxp + TGET(z, i + 1, j);
  float Hxm = hxm + TGET(z, i - 1, j);
  float Hyp = hyp + TGET(z, i, j + 1);
  float Hym = hym + TGET(z, i, j - 1);

  // face mobilities
  float Mp = pow(0.5f * (h + hxp), power) / viscosity;
  float Mm = pow(0.5f * (h + hxm), power) / viscosity;
  float Myp = pow(0.5f * (h + hyp), power) / viscosity;
  float Mym = pow(0.5f * (h + hym), power) / viscosity;

  // fluxes through faces
  float qxp = -Mp * (Hxp - Hc);
  float qxm = -Mm * (Hc - Hxm);
  float qyp = -Myp * (Hyp - Hc);
  float qym = -Mym * (Hc - Hym);

  // divergence
  float dhdt = -(qxp - qxm + qyp - qym);

  // explicit update
  float h_new = fmax(0.f, h + dt * dhdt);

  // smoothing
  float h_avg = (TGET(h_in, i - 1, j) + TGET(h_in, i + 1, j) +
                 TGET(h_in, i, j - 1) + TGET(h_in, i, j + 1) +
                 diag * (TGET(h_in, i - 1, j - 1) + TGET(h_in, i + 1, j - 1) +
                         TGET(h_in, i - 1, j + 1) + TGET(h_in, i + 1, j + 1))) /
                (4.f + 4.f * diag);

  float k_visc = 0.001f;
  // h_new = mix(h_new, h_avg, k_visc);

  TSET(h_out, i, j, h_new);
}
)""
