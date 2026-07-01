R""(
/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
void kernel erosion_function_buffer(global float *z,  // read / write
                                    const int     nx, // shape.x
                                    const int     ny, // shape.y
                                    const float   some_parameter,
                                    const int     some_int)
{
  // one worker per cell, so g actually corresponds to one of the
  // array cell, like a (i, j) index
  int2 g = {get_global_id(0), get_global_id(1)};

  // convert (i, j) index to a linear 1D index, use by OpenCL kernel
  // (see _common_index.cl for index helpers)
  int index = linear_index(g.x, g.y, nx);

  // discard out-of-bound indices (can happen because the nb of
  // workers is a 2^n number for performances reason)
  if (g.x >= nx || g.y >= ny) return;

  // --- COMPUTE GOES HERE ---

  // get the value of the current cell
  float val = z[index];

  // get a cell around (always check bounds)
  if (g.x + 1 < nx)
  {
    int   idx = linear_index(g.x + 1, g.y, nx);
    float val_next = z[idx];
  }

  // assign a value (you can assign the value of other cells but race
  // condition between workers will likely create issues)

  z[index] = 0.5f * val; // dummy operation, divides amplitude by 2
}

void kernel erosion_function_image(read_only image2d_t  z,   // in
                                   write_only image2d_t out, // out
                                   const int            nx,
                                   const int            ny,
                                   const float          some_parameter)
{
  // see above for the global indices
  const int2 g = {get_global_id(0), get_global_id(1)};

  if (g.x >= nx || g.y >= ny) return;

  // various sampling possibilities see
  // https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/samplers.html

  // this one corresponds to a sampling based on cell indices '(i, j)'
  // but '(i, j)' can be float, you can sample in the middle of a cell
  // with bilinear interpolation
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

  // get the current value (directly (i, j) positions, it's not a
  // linear index logic). The image buffer has one channel, but the
  // first component still has to be explicitly requested when reading
  // the first (hence the .x at the end)
  float2 pos = (float2)(g.x, g.y);
  float  f0 = read_imagef(z, sampler, pos).x;

  // other position
  float f1 = read_imagef(z, sampler, (float2)(g.x + 0.5f, g.y)).x;

  // write to the output (same operation as above). NB: you can only
  // at integer positionssince its an array...
  write_imagef(out, (int2)(g.x, g.y), f0 * 0.5f);
}
)""
