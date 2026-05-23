/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "highmap/array.hpp"
#include "highmap/boundary.hpp"

namespace hmap
{

glm::ivec2 pick_boundary_cell(const Array   &z,
                              DomainBoundary boundary,
                              std::uint32_t  seed,
                              bool           favor_boundary_center,
                              bool           favor_lower_elevation,
                              bool           favor_sinks)
{
  const int       nx = z.shape.x;
  const int       ny = z.shape.y;
  const glm::vec2 range = z.range();

  // --- candidate cells

  std::vector<glm::ivec2> cells;

  switch (boundary)
  {
  case BOUNDARY_LEFT:
  {
    int i = 0;
    for (int j = 0; j < ny; ++j)
      cells.emplace_back(glm::ivec2(i, j));
  }
  break;

  case BOUNDARY_RIGHT:
  {
    int i = nx - 1;
    for (int j = 0; j < ny; ++j)
      cells.emplace_back(glm::ivec2(i, j));
  }
  break;

  case BOUNDARY_BOTTOM:
  {
    int j = 0;
    for (int i = 0; i < nx; ++i)
      cells.emplace_back(glm::ivec2(i, j));
  }
  break;

  case BOUNDARY_TOP:
  {
    int j = ny - 1;
    for (int i = 0; i < nx; ++i)
      cells.emplace_back(glm::ivec2(i, j));
  }
  break;
  }

  // --- compute weights

  const size_t       ncells = cells.size();
  std::vector<float> weights(ncells, 1.f);

  for (size_t k = 0; k < ncells; ++k)
  {
    if (favor_boundary_center)
    {
      float w = float(k) / float(ncells - 1);
      weights[k] *= 4.f * w * (1.f - w);
    }

    if (favor_lower_elevation)
    {
      float w = (z(cells[k]) - range.x) / (range.y - range.x);
      weights[k] *= (1.f - w) * (1.f - w);
    }

    if (favor_sinks)
    {
      float center = z(cells[k]);
      float min_neigh = center;

      for (int dj = -1; dj <= 1; ++dj)
        for (int di = -1; di <= 1; ++di)
        {
          int ni = cells[k].x + di;
          int nj = cells[k].y + dj;

          if (ni >= 0 && ni < nx && nj >= 0 && nj < ny)
            min_neigh = std::min(min_neigh, z(ni, nj));
        }

      if (center <= min_neigh) weights[k] = std::min(1.f, 4.f * weights[k]);
    }
  }

  // --- pick the cells

  std::mt19937                    gen(seed);
  std::discrete_distribution<int> dist(weights.begin(), weights.end());
  int                             idx = dist(gen);

  return cells[idx];
}

} // namespace hmap
