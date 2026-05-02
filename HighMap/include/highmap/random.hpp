/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once
#include <cstddef>
#include <cstdint>

namespace hmap
{

/**
 * @brief Generates a deterministic uniform float in [0,1) using a
 * SplitMix64-style hash.
 *
 * @param  seed Base seed value.
 * @param  k    Sample index.
 * @return      Pseudo-random float in [0,1).
 */
float splitmix64_to_unit_float(unsigned int seed, size_t k);

/**
 * @brief Generates a fast deterministic uniform float in [0,1) using a 32-bit
 * hash.
 *
 * @param  seed Base seed value.
 * @param  k    Sample index.
 * @return      Pseudo-random float in [0,1).
 * @note Faster but lower quality than splitmix64_to_unit_float().
 */
float fast_hash32_to_unit_float(unsigned int seed, size_t k);

} // namespace hmap