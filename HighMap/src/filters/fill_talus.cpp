/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <cmath>
#include <random>

#include "macrologger.h"

#include "highmap/boundary.hpp"
#include "highmap/erosion.hpp"
#include "highmap/filters.hpp"
#include "highmap/math.hpp"
#include "highmap/range.hpp"

namespace hmap
{

void fill_talus(Array       &z,
                const Array &talus,
                uint         seed,
                int          ir,
                float        noise_ratio,
                const Array *p_seed_mask)
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

void fill_talus(Array       &z,
                float        talus,
                uint         seed,
                int          ir,
                float        noise_ratio,
                const Array *p_seed_mask)
{
  fill_talus(z, Array(z.shape, talus), seed, ir, noise_ratio, p_seed_mask);
}

void fill_talus_fast(Array    &z,
                     Vec2<int> shape_coarse,
                     float     talus,
                     uint      seed,
                     int       ir,
                     float     noise_ratio)
{
  // apply the algorithm on the coarser mesh (and ajust the talus
  // value)
  int   step = std::max(z.shape.x / shape_coarse.x, z.shape.y / shape_coarse.y);
  float talus_coarse = talus * step;

  // add maximum filter to avoid loosing data (for instance those
  // defined at only one cell)
  Array z_coarse = Array(shape_coarse);
  {
    Array z_filtered = maximum_local(z, (int)std::ceil(0.5f * step));
    z_coarse = z_filtered.resample_to_shape(shape_coarse);
  }

  fill_talus(z_coarse, talus_coarse, seed, ir, noise_ratio);

  // revert back to the original resolution but keep initial
  // smallscale details
  z_coarse = z_coarse.resample_to_shape(z.shape);

  clamp_min(z, z_coarse);
}

} // namespace hmap
