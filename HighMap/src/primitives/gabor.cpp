/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <cstdint> // for uint32_t

#include "highmap/array.hpp"      // for Array
#include "highmap/convolve.hpp"   // for convolve2d_svd
#include "highmap/kernels.hpp"    // for gabor
#include "highmap/primitives.hpp" // for white_sparse, gabor_noise

#define SVD_RANK 2

namespace hmap
{

Array gabor_noise(glm::ivec2    shape,
                  float         kw,
                  float         angle,
                  int           width,
                  float         density,
                  std::uint32_t seed)
{
  Array weight = white_sparse(shape, 0.f, 1.f, density, seed);
  Array kernel = gabor({width, width}, kw, angle);

  return convolve2d_svd(weight, kernel, SVD_RANK);
}

} // namespace hmap
