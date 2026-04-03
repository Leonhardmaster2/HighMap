R""(
/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

void kernel thermal_rib(global float *z,
                        const int     nx,
                        const int     ny,
                        const int     it)
{
  int2 g = {get_global_id(0), get_global_id(1)};
  int  index = linear_index(g.x, g.y, nx);

  if (g.x >= nx || g.y >= ny) return;
  if (apply_boundaries(z, g.x, g.y, nx, ny)) return;

  // --- thermal erosion

  const int   di[8] = {-1, 0, 0, 1, -1, -1, 1, 1};
  const int   dj[8] = {0, 1, -1, 0, -1, 1, -1, 1};
  const float c[8] = {1.f, 1.f, 1.f, 1.f, 1.414f, 1.414f, 1.414f, 1.414f};

  float delta_min = FLT_MAX;
  float val = z[index];

  for (int k = 0; k < 8; k++)
  {
    float dz = fabs(val - z[linear_index(g.x + di[k], g.y + dj[k], nx)]) / c[k];
    delta_min = min(delta_min, dz);
  }

  z[index] -= delta_min;
}
)""
