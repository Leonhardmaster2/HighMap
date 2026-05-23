/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <cstddef>
#include <vector>

#include "highmap/array.hpp"

namespace hmap
{

size_t count_non_zero(const Array &array)
{
  size_t count = 0;
  for (const auto &v : array.vector)
    if (v != 0.f) count++;
  return count;
}

size_t count_zero(const Array &array)
{
  size_t count = 0;
  for (const auto &v : array.vector)
    if (v == 0.f) count++;
  return count;
}

} // namespace hmap
