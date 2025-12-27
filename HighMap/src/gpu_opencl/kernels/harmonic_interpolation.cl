R""(
/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
void kernel harmonic_interpolation(const global float *array,
                                   global float       *out,
                                   const global float *mask_fixed_values,
                                   const int           nx,
                                   const int           ny)
{
  int2 g = {get_global_id(0), get_global_id(1)};
  int  idx = linear_index(g.x, g.y, nx);

  if (g.x >= nx || g.y >= ny) return;

  out[idx] = array[idx];

  bool in_bounds = g.x > 0 && g.x < nx - 1 && g.y > 0 && g.y < ny - 1;
  bool is_fixed = mask_fixed_values[idx] > 0.f;

  if (!in_bounds || is_fixed) return;

  float new_val = 0.25f * (array[linear_index(g.x - 1, g.y, nx)] +
                           array[linear_index(g.x + 1, g.y, nx)] +
                           array[linear_index(g.x, g.y - 1, nx)] +
                           array[linear_index(g.x, g.y + 1, nx)]);
  out[idx] = new_val;
}
)""
