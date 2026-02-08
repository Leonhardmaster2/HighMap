R""(
/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

kernel void snow_simulation(read_only image2d_t  z,
                            read_only image2d_t  s,
                            read_only image2d_t  talus,
                            read_only image2d_t  melting_map,
                            write_only image2d_t s_out,
                            int                  nx,
                            int                  ny,
                            float                dt,
                            float                snow_depth,
                            float                k_snow,
                            float                k_melt,
                            float                k_visc,
                            float                k_depth_ratio,
                            float                k_depth_slope_ratio)
{
  int i = get_global_id(0);
  int j = get_global_id(1);

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

  if (i >= nx || j >= ny) return;

  const float diag = 0.70710678f;
  const float slope_repose = TGET(talus, i, j);
  const float sc = TGET(s, i, j);

  //
  const float sc0 = k_depth_ratio * snow_depth;
  const float depth_factor = 1.f +
                             k_depth_slope_ratio * smoothstep(0.f, sc0, sc);
  const float slope_repose_eff = slope_repose * depth_factor;

  float Hc = TGET(z, i, j) + sc;

  // --- outgoing flux from center

  float Hl = TGET(z, i - 1, j) + TGET(s, i - 1, j);
  float Hr = TGET(z, i + 1, j) + TGET(s, i + 1, j);
  float Hd = TGET(z, i, j - 1) + TGET(s, i, j - 1);
  float Hu = TGET(z, i, j + 1) + TGET(s, i, j + 1);

  float Hdl = TGET(z, i - 1, j - 1) + TGET(s, i - 1, j - 1);
  float Hdr = TGET(z, i + 1, j - 1) + TGET(s, i + 1, j - 1);
  float Hul = TGET(z, i - 1, j + 1) + TGET(s, i - 1, j + 1);
  float Hur = TGET(z, i + 1, j + 1) + TGET(s, i + 1, j + 1);

  float sx = 0.5f * (Hr - Hl);
  float sy = 0.5f * (Hu - Hd);
  float slope = hypot(sx, sy);

  float excess = max(0.f, slope - slope_repose_eff);

  float flux_out = k_snow * excess * dt;
  flux_out = fmin(flux_out, sc);

  // depth at which snow starts to lock
  float mobility = exp(-sc / sc0);
  flux_out *= mobility;

  float wl = max(0.f, Hc - Hl);
  float wr = max(0.f, Hc - Hr);
  float wd = max(0.f, Hc - Hd);
  float wu = max(0.f, Hc - Hu);
  float wdl = max(0.f, Hc - Hdl);
  float wdr = max(0.f, Hc - Hdr);
  float wul = max(0.f, Hc - Hul);
  float wur = max(0.f, Hc - Hur);

  float wsum = wl + wr + wd + wu + diag * (wdl + wdr + wul + wur);

  float out = (wsum > 1e-6f) ? flux_out : 0.f;

  // --- incoming flux from neighbors

  float in = 0.f;

  // LEFT neighbor (i-1,j) sends RIGHT to (i,j)
  {
    float Hn = Hl;
    float wn = max(0.f, Hn - Hc);
    float excess_n = max(0.f, (Hn - Hc) - slope_repose_eff);
    float factor_n = k_snow * excess_n * dt;

    float sum = max(0.f, Hn - Hr) +
                max(0.f, Hn - (TGET(z, i - 1, j - 1) + TGET(s, i - 1, j - 1))) *
                    diag +
                max(0.f, Hn - (TGET(z, i - 1, j + 1) + TGET(s, i - 1, j + 1))) *
                    diag;

    if (sum > 1e-6f) in += factor_n * wn / sum;
  }

  // RIGHT neighbor (i+1,j) sends LEFT to (i,j)
  {
    float Hn = Hr;
    float wn = max(0.f, Hn - Hc);
    float excess_n = max(0.f, (Hn - Hc) - slope_repose_eff);
    float factor_n = k_snow * excess_n * dt;

    float sum = max(0.f, Hn - Hl) +
                max(0.f, Hn - (TGET(z, i + 1, j - 1) + TGET(s, i + 1, j - 1))) *
                    diag +
                max(0.f, Hn - (TGET(z, i + 1, j + 1) + TGET(s, i + 1, j + 1))) *
                    diag;

    if (sum > 1e-6f) in += factor_n * wn / sum;
  }

  // DOWN neighbor (i,j-1) sends UP to (i,j)
  {
    float Hn = Hd;
    float wn = max(0.f, Hn - Hc);
    float excess_n = max(0.f, (Hn - Hc) - slope_repose_eff);
    float factor_n = k_snow * excess_n * dt;

    float sum = max(0.f, Hn - Hu) +
                max(0.f, Hn - (TGET(z, i - 1, j - 1) + TGET(s, i - 1, j - 1))) *
                    diag +
                max(0.f, Hn - (TGET(z, i + 1, j - 1) + TGET(s, i + 1, j - 1))) *
                    diag;

    if (sum > 1e-6f) in += factor_n * wn / sum;
  }

  // UP neighbor (i,j+1) sends DOWN to (i,j)
  {
    float Hn = Hu;
    float wn = max(0.f, Hn - Hc);
    float excess_n = max(0.f, (Hn - Hc) - slope_repose_eff);
    float factor_n = k_snow * excess_n * dt;

    float sum = max(0.f, Hn - Hd) +
                max(0.f, Hn - (TGET(z, i - 1, j + 1) + TGET(s, i - 1, j + 1))) *
                    diag +
                max(0.f, Hn - (TGET(z, i + 1, j + 1) + TGET(s, i + 1, j + 1))) *
                    diag;

    if (sum > 1e-6f) in += factor_n * wn / sum;
  }

  float sc_new = sc - out + in;

  // --- viscosity (numerical damping)

  float s_avg = (TGET(s, i - 1, j) + TGET(s, i + 1, j) + TGET(s, i, j - 1) +
                 TGET(s, i, j + 1) +
                 diag * (TGET(s, i - 1, j - 1) + TGET(s, i + 1, j - 1) +
                         TGET(s, i - 1, j + 1) + TGET(s, i + 1, j + 1))) /
                (4.f + 4.f * diag);

  sc_new = mix(sc_new, s_avg, k_visc);

  // --- melting

  float mt = TGET(melting_map, i, j); // in (0, 1]
  float melt = k_melt * mt * dt;
  melt = fmin(melt, sc_new);
  sc_new -= melt;

  // --- update output

  TSET(s_out, i, j, max(0.f, sc_new));
}
)""
