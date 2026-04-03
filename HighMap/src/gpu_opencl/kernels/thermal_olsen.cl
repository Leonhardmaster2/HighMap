R""(
/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
void kernel thermal_olsen(global float       *z,
                          const global float *talus,
                          const int           nx,
                          const int           ny)
{
  int2 g = {get_global_id(0), get_global_id(1)};

  if (g.x >= nx || g.y >= ny) return;
  if (apply_boundaries_buffer(z, g.x, g.y, nx, ny, 3)) return;

  // --- thermal erosion

  int idx = linear_index(g.x, g.y, nx);

  const int di[8] = {-1, 0, 0, 1, -1, -1, 1, 1};
  const int dj[8] = {0, 1, -1, 0, -1, 1, -1, 1};

  const float c1 = 1.f;
  const float c2 = 1.414213562f; // TODO add bias parameter
  const float c[8] = {c1, c1, c1, c1, c2, c2, c2, c2};

  float val = z[idx];
  float talus_ref = talus[idx];

  // "reversed" compared to Olsen, loop over neighbors of neighbors to
  // get current cell modification... => each cell is written by only
  // one work-item
  for (int k = 0; k < 8; ++k)
  {
    uint shift = hash21u(g.x, g.y) & 7;
    int  s = (k + shift) & 7;

    int2 gk = (int2)(g.x + di[s], g.y + dj[s]);

    float dsum = 0.f;
    float dmax = 0.f;
    float dz[8];

    for (int p = 0; p < 8; ++p)
    {
      int2 gp = (int2)(gk.x + di[p], gk.y + dj[p]);

      dz[p] = z[linear_index(gk.x, gk.y, nx)] - z[linear_index(gp.x, gp.y, nx)];

      if (dz[p] > talus_ref * c[p])
      {
        dsum += dz[p];
        dmax = max(dmax, dz[p]);
      }
    }

    if (dmax > 0.f)
    {
      for (int p = 0; p < 8; ++p)
      {
        int2 gp = (int2)(gk.x + di[p], gk.y + dj[p]);

        if (gp.x == g.x && gp.y == g.y)
        {
          float amount = 0.25f * (dmax - talus_ref * c[p]) * dz[p] / dsum;
          z[idx] += amount;
          continue;
        }
      }
    }
  }
}
)""
