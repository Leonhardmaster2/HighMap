R""(
/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
void kernel thermal_ridge(global float       *z,
                          const global float *talus,
                          const int           nx,
                          const int           ny)
{
  int2 g = {get_global_id(0), get_global_id(1)};
  int  index = linear_index(g.x, g.y, nx);

  if (g.x >= nx || g.y >= ny) return;
  if (apply_boundaries(z, g.x, g.y, nx, ny)) return;

  // --- thermal erosion

  const int   di[8] = {-1, 0, 0, 1, -1, -1, 1, 1};
  const int   dj[8] = {0, 1, -1, 0, -1, 1, -1, 1};
  const float c[8] = {1.f, 1.f, 1.f, 1.f, 1.414f, 1.414f, 1.414f, 1.414f};

  float val = z[index];
  float sum = 0.f;
  float slope_max = 0.f;

  for (int k = 0; k < 8; k++)
  {
    float dz = (val - z[linear_index(g.x + di[k], g.y + dj[k], nx)]) / c[k];

    if (dz > 0.f) sum += dz;

    slope_max = max(slope_max, fabs(dz));
  }

  float t = talus[index];
  float amp = clamp(1.f - t / slope_max, 0.f, 1.f);
  amp = smoothstep3(amp);

  z[index] += 0.25f * (t - 0.5f * sum) * amp;
}
)""
