/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <cmath>
#include <cstdint>
#include <queue>
#include <random>
#include <vector>

#include "highmap/array.hpp"
#include "highmap/filters.hpp"

namespace hmap
{

void fill_talus(Array        &z,
                const Array  &talus,
                std::uint32_t seed,
                int           ir,
                float         noise_ratio,
                const Array  *p_seed_mask)
{
  std::mt19937                          gen(seed);
  std::uniform_real_distribution<float> dis(1.f - noise_ratio,
                                            1.f + noise_ratio);
  // local node type
  struct Node
  {
    float h;
    int   i;
    int   j;

    bool operator<(const Node &other) const
    {
      return h < other.h; // max-heap
    }
  };

  const int nx = z.shape.x;
  const int ny = z.shape.y;

  std::vector<Node> queue;
  queue.reserve(nx * ny);

  // build initial heap
  if (p_seed_mask)
  {
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i)
        if ((*p_seed_mask)(i, j)) queue.push_back({z(i, j), i, j});
  }
  else
  {
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i)
        queue.push_back({z(i, j), i, j});
  }

  std::make_heap(queue.begin(), queue.end());

  // talus propagation
  while (!queue.empty())
  {
    std::pop_heap(queue.begin(), queue.end());
    const Node current = queue.back();
    queue.pop_back();

    const float base = z(current.i, current.j);
    const float talus_ref = talus(current.i, current.j);

    for (int s = -ir; s <= ir; ++s)
      for (int r = -ir; r <= ir; ++r)
      {
        if (r == 0 && s == 0) continue;

        const int p = current.i + r;
        const int q = current.j + s;

        if (p < 0 || p > nx - 1 || q < 0 || q > ny - 1) continue;

        const float dist = std::hypot(r, s);
        const float rd = dis(gen);
        const float h = base - dist * talus_ref * rd;

        if (h > z(p, q))
        {
          z(p, q) = h;
          queue.push_back({h, p, q});
          std::push_heap(queue.begin(), queue.end());
        }
      }
  }
}

void fill_talus(Array        &z,
                float         talus,
                std::uint32_t seed,
                int           ir,
                float         noise_ratio,
                const Array  *p_seed_mask)
{
  fill_talus(z, Array(z.shape, talus), seed, ir, noise_ratio, p_seed_mask);
}

} // namespace hmap
