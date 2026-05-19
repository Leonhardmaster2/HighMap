R""(
/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

void kernel sparse_max_convolution(global float       *array,
                                   read_only image2d_t weights,
                                   global float       *out,
                                   const int           nx,
                                   const int           ny,
                                   const int           nkx,
                                   const int           nky,
                                   const float         k_smoothmax)
{
  int2 g = (int2)(get_global_id(0), get_global_id(1));
  if (g.x >= nx || g.y >= ny) return;

  int   index = linear_index(g.x, g.y, nx);
  float val = array[index];

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
  int irx = nkx / 2;
  int iry = nky / 2;

  for (int r = 0; r < nkx; ++r)
    for (int s = 0; s < nky; ++s)
    {
      int i = g.x + (r - irx);
      int j = g.y + (s - iry);
      if (i < 0 || i >= nx || j < 0 || j >= ny) continue;

      // kernel sampled at the center of texel (r, s).
      float2 pos = (float2)((float)r + 0.5f, (float)s + 0.5f);
      float  w = read_imagef(weights, sampler, pos).x * val;

      atomic_smoothmax_float(&out[linear_index(i, j, nx)], w, k_smoothmax);
    }
}
)""
