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

void fill_talus(Array &z, float talus, uint seed, float noise_ratio)
{
  std::mt19937                          gen(seed);
  std::uniform_real_distribution<float> dis(1.f - noise_ratio,
                                            1.f + noise_ratio);

  std::vector<int>   di = HMAP_DI;
  std::vector<int>   dj = HMAP_DJ;
  std::vector<float> c = HMAP_CD;
  const uint         nb = di.size();

  // trick to exclude border cells, to avoid checking out of bounds
  // indices
  set_borders(z, 10.f * z.max(), 2);

  // build queue (elevation, index (i, j))
  std::vector<std::pair<float, std::pair<int, int>>> queue;

  for (int i = 2; i < z.shape.x - 2; i++)
    for (int j = 2; j < z.shape.y - 2; j++)
      queue.push_back(std::pair(z(i, j), std::pair(i, j)));

  std::make_heap(queue.begin(), queue.end());

  // fill
  while (queue.size() > 0)
  {
    std::pair<int, std::pair<int, int>> current = queue.back();
    queue.pop_back();

    int i = current.second.first;
    int j = current.second.second;

    for (uint k = 0; k < nb; k++) // loop over neighbors
    {
      int   p = i + di[k];
      int   q = j + dj[k];
      float rd = dis(gen);
      float h = z(i, j) - c[k] * talus * rd;

      if (h > z(p, q))
      {
        z(p, q) = h;
        queue.push_back(std::pair(z(p, q), std::pair(p, q)));
        std::push_heap(queue.begin(), queue.end());
      }
    }
  }

  // clean-up boundaries
  extrapolate_borders(z, 2);
}

void fill_talus_fast(Array    &z,
                     Vec2<int> shape_coarse,
                     float     talus,
                     uint      seed,
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

  fill_talus(z_coarse, talus_coarse, seed, noise_ratio);

  // revert back to the original resolution but keep initial
  // smallscale details
  z_coarse = z_coarse.resample_to_shape(z.shape);

  clamp_min(z, z_coarse);
}

void fold(Array &array, int iterations, float k)
{
  fold(array, array.min(), array.max(), iterations, k);
}

void fold(Array &array, float vmin, float vmax, int iterations, float k)
{
  array -= vmin;
  float vref = (vmax - vmin) / (iterations + 1.f);

  for (int it = 0; it < iterations; it++)
  {
    array -= vref;

    if (k == 0.f)
      array = abs(array);
    else
      array = abs_smooth(array, k);
  }
}

} // namespace hmap
