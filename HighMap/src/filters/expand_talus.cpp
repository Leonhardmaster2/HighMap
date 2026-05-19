/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm> // for min
#include <cmath>     // for hypot
#include <cstdint>
#include <queue>  // for make_heap, pop_heap, push_heap
#include <random> // for generate_canonical, mt19937
#include <vector> // for vector

#include "highmap/array.hpp"    // for Array
#include "highmap/boundary.hpp" // for extrapolate_borders
#include "highmap/filters.hpp"  // for expand_talus

namespace hmap
{

void expand_talus(Array        &z,
                  const Array  &mask,
                  float         talus,
                  std::uint32_t seed,
                  int           ir,
                  float         noise_ratio)
{
  // heap queue structure
  struct HeapNode
  {
    float h;
    int   i;
    int   j;

    bool operator<(const HeapNode &other) const
    {
      return h > other.h; // min-heap
    }
  };

  // params
  std::mt19937     gen(seed);
  const glm::ivec2 shape = z.shape;

  // ---  expand the talus upward

  // queue
  Array                 mask_copy = mask;
  std::vector<HeapNode> queue;
  queue.reserve(shape.x * shape.y / 8); // heuristic

  // initialize heap
  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      if (mask_copy(i, j) != 0.f) queue.push_back({z(i, j), i, j});
    }

  std::make_heap(queue.begin(), queue.end());

  // fill
  while (!queue.empty())
  {
    std::pop_heap(queue.begin(), queue.end());
    HeapNode current = queue.back();
    queue.pop_back();

    const int   i0 = current.i;
    const int   j0 = current.j;
    const float z0 = z(i0, j0);

    for (int s = -ir; s <= ir; ++s)
      for (int r = -ir; r <= ir; ++r)
      {
        if (r == 0 && s == 0) continue;

        const int ni = i0 + r;
        const int nj = j0 + s;

        if (ni < 0 || nj < 0 || ni >= shape.x - 1 || nj >= shape.y - 1)
          continue;

        const float dist = std::hypot(r, s);
        const float rd = 2.f * std::generate_canonical<float, 16>(gen) - 1.f;
        const float h = z0 + dist * talus * (1.f + noise_ratio * rd);

        if (mask_copy(ni, nj))
        {
          z(ni, nj) = std::min(z(ni, nj), h);
        }
        else
        {
          z(ni, nj) = h;

          mask_copy(ni, nj) = 1.f;
          queue.push_back({z(ni, nj), ni, nj});
          std::push_heap(queue.begin(), queue.end());
        }
      }
  }

  extrapolate_borders(z, 1);
}

} // namespace hmap
