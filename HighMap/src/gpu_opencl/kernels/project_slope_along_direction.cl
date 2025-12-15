R""(
/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
void kernel project_talus_along_direction(global float *array,
                                          global float *out,
                                          const int     nx,
                                          const int     ny,
                                          const float   talus,
                                          const int     di,
                                          const int     dj)
{
  int2 g = {get_global_id(0), get_global_id(1)};

  if (g.x >= nx || g.y >= ny) return;

  int   index = linear_index(g.x, g.y, nx);
  float val = array[index];

  // full domain diagonal in worst case scenario
  const int nsteps = (int)(1.414f * max(nx, ny));

  for (int k = 0; k < nsteps; ++k)
  {
    int   i = g.x + (k + 1) * di;
    int   j = g.y + (k + 1) * dj;
    float dr = length((float2)(i - g.x, j - g.y));

    if (i < 0 || i >= nx || j < 0 || j >= ny) break;

    float v = val - dr * talus;

    int   idx = linear_index(i, j, nx);
    float vref = out[idx];

    if (v < vref) break;

    atomic_max(&out[idx], v);
  }
}
)""
