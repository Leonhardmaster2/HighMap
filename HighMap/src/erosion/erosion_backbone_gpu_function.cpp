/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "cl_wrapper/run.hpp"

#include "highmap/array.hpp"

namespace hmap::gpu
{

void erosion_backbone_gpu_function_buffer(Array &z, float some_parameter)
{
  const glm::ivec2 &shape = z.shape; // (width, height)

  // see in file erosion_backbone_functions.cl, the function name must
  // fit the OpenCL kernel name...
  auto run = clwrapper::Run("erosion_function_buffer");

  run.bind_buffer<float>("z", z.vector); // bind as many buffers as you want

  // template-based w/ ellipsis, bind as many parameters as you want in one call
  int some_int = 0;
  run.bind_arguments(shape.x, shape.y, some_parameter, some_int);

  // copy array content to the GPU
  run.write_buffer("z");

  // compute (one worker per cell)
  run.execute({z.shape.x, z.shape.y});

  // copy GPU content to the array
  run.read_buffer("z");
}

void erosion_backbone_gpu_function_image(Array &z, float some_parameter)
{
  const glm::ivec2 &shape = z.shape; // (width, height)

  // see in file erosion_backbone_functions.cl, the function name must
  // fit the OpenCL kernel name...
  auto run = clwrapper::Run("erosion_function_image");

  // trick here, for portzbility reasons, read/write image buffers are
  // not used. The GPU image buffer can either be read or write but it
  // can be associated to the same CPU data (array 'z' here)
  run.bind_imagef("z", z.vector, shape.x, shape.y);

  // 'true' at the end for an output. Associated array is z, meaning
  // data copy of data from the GPU will be dumped to this array
  run.bind_imagef("out", z.vector, shape.x, shape.y, true);

  run.bind_arguments(shape.x, shape.y, some_parameter);

  // copy array content to the GPU (input only)
  run.write_imagef("z");

  run.execute({shape.x, shape.y});

  // copy GPU content to the array
  run.read_imagef("out");
}

} // namespace hmap::gpu
