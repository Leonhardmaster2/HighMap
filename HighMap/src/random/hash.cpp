/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <cstddef>
#include <cstdint>

namespace hmap
{

float splitmix64_to_unit_float(unsigned int seed, size_t k)
{
  // combine seed + index into 64-bit to avoid truncation
  uint64_t x = static_cast<uint64_t>(seed) ^
               (static_cast<uint64_t>(k) + 0x9e3779b97f4a7c15ull);

  // 64-bit mix (SplitMix64-inspired)
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebull;
  x ^= x >> 31;

  // convert to float in [0,1)
  return static_cast<float>((x >> 40) & 0xFFFFFF) / static_cast<float>(1 << 24);
}

float fast_hash32_to_unit_float(unsigned int seed, size_t k)
{
  uint32_t x = static_cast<uint32_t>(k) ^ seed;
  x ^= x >> 16;
  x *= 0x45d9f3bu;
  x ^= x >> 16;
  return static_cast<float>((x >> 8) * (1.f / float(1 << 24)));
}

} // namespace hmap
