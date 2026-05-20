/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */

/**
 * @file random.hpp
 * @author Otto Link (otto.link.bv@gmail.com)
 * @copyright Copyright (c) 2023
 */
#pragma once

#include "highmap/array.hpp"

namespace hmap
{

/**
 * @brief Return an array filled with white noise.
 *
 * @param  shape Array shape.
 * @param  a     Lower bound of random distribution.
 * @param  b     Upper bound of random distribution.
 * @param  seed  Random number seed.
 * @return       Array White noise.
 *
 * **Example**
 * @include ex_white.cpp
 *
 * **Result**
 * @image html ex_white.png
 *
 * @see          {@link white_sparse}
 */
Array white(glm::ivec2 shape, float a, float b, std::uint32_t seed);

/**
 * @brief Return an array filled `1` with a probability based on a density map.
 *
 * @param  density_map Density map.
 * @param  seed        Random number seed.
 * @return             Array New array.
 *
 * **Example**
 * @include ex_white_density_map.cpp
 *
 * **Result**
 * @image html ex_white_density_map.png
 */
Array white_density_map(const Array &density_map, std::uint32_t seed);

/**
 * @brief Return an array sparsely filled with white noise.
 *
 * @param  shape   Array shape.
 * @param  a       Lower bound of random distribution.
 * @param  b       Upper bound of random distribution.
 * @param  density Array filling density, in [0, 1]. If set to 1, the function
 *                 is equivalent to {@link white}.
 * @param  seed    Random number seed.
 * @return         Array Sparse white noise.
 *
 * **Example**
 * @include ex_white_sparse.cpp
 *
 * **Result**
 * @image html ex_white_sparse.png
 *
 * @see            {@link white}
 */
Array white_sparse(glm::ivec2    shape,
                   float         a,
                   float         b,
                   float         density,
                   std::uint32_t seed);

/**
 * @brief Return an array sparsely filled with random 0 and 1.
 *
 * @param  shape   Array shape.
 * @param  density Array filling density, in [0, 1]. If set to 1, the function
 *                 is equivalent to {@link white}.
 * @param  seed    Random number seed.
 * @return         Array Sparse white noise.
 */
Array white_sparse_binary(glm::ivec2 shape, float density, std::uint32_t seed);

} // namespace hmap