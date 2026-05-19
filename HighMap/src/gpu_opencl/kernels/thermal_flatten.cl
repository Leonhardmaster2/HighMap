R""(
/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
void kernel thermal_flatten(global float       *z,
                            const global float *talus,
                            const int           nx,
                            const int           ny,
                            const float         sigma_inf,
                            const float         sigma_sup)
{
  int2 g = {get_global_id(0), get_global_id(1)};
  int  index = linear_index(g.x, g.y, nx);

  if (g.x >= nx || g.y >= ny) return;
  if (apply_boundaries_buffer(z, g.x, g.y, nx, ny, 2)) return;

  const int   di[8] = {-1, 0, 0, 1, -1, -1, 1, 1};
  const int   dj[8] = {0, 1, -1, 0, -1, 1, -1, 1};
  const float c[8] = {1.f, 1.f, 1.f, 1.f, 1.414f, 1.414f, 1.414f, 1.414f};

  const int   rotation_offset = (g.x + g.y) % 8;
  const float zc = z[index];

  const float slope_min = 0.f / nx;

  // --- Remove material from current cell

  float delta = 0.f;
  float dmax = 0.f;
  int   ka = -1;

  for (int n = 0; n < 8; ++n)
  {
    const int   k = (n + rotation_offset) & 7;
    const int   ni = g.x + di[k];
    const int   nj = g.y + dj[k];
    const float dz = (zc - z[linear_index(ni, nj, nx)]) / c[k];

    if (dz > dmax)
    {
      dmax = dz;
      ka = k;
    }
  }

  float excess = max(dmax - slope_min, 0.f);

  if (dmax < talus[index]) delta -= sigma_inf * excess;
  if (dmax > talus[index]) delta -= sigma_sup * excess;

  // --- Remove material from neighbors

  for (int n = 0; n < 8; ++n)
  {
    const int   k = (n + rotation_offset) & 7;
    const int   pi = g.x - di[k];
    const int   pj = g.y - dj[k];
    const int   pidx = linear_index(pi, pj, nx);
    const float zp = z[pidx];

    float dmax = 0.f;
    int   ka = -1;

    // find steepest neighbor for source cell
    for (int m = 0; m < 8; ++m)
    {
      const int   kk = (m + rotation_offset) & 7;
      const int   qi = pi + di[kk];
      const int   qj = pj + dj[kk];
      const float dz = (zp - z[linear_index(qi, qj, nx)]) / c[kk];

      if (dz > dmax)
      {
        dmax = dz;
        ka = kk;
      }
    }

    // if current cell is target receiver
    if (ka == k && dmax > 0.0f && dmax < talus[pidx]) delta += 0.5f * dmax;
  }

  z[index] = zc + delta;
}
)""
