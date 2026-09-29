/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm>
#include <cstddef>
#include <vector>

#include "highmap/algebra.hpp"
#include "highmap/array.hpp"
#include "highmap/erosion.hpp"
#include "highmap/filters.hpp"
#include "highmap/hydrology/drainage_basin_cell_based.hpp"
#include "highmap/internal/validation.hpp"

namespace hmap
{

void depression_filling_priority_flood(Array &z,
                                       bool   apply_post_filter,
                                       bool   outflow_left,
                                       bool   outflow_right,
                                       bool   outflow_bottom,
                                       bool   outflow_top)
{
  if (!validate_non_empty(z)) return;

  Array z_bckp = apply_post_filter ? z : Array();

  auto db = DrainageBasinCellBased(z);

  std::vector<glm::ivec2> outlets;
  outlets.reserve(2 * (z.shape.x + z.shape.y));

  for (int j = 0; j < z.shape.y; ++j)
    for (int i = 0; i < z.shape.x; ++i)
    {
      if ((outflow_left && i == 0) || (outflow_right && i == z.shape.x - 1) ||
          (outflow_bottom && j == 0) || (outflow_top && j == z.shape.y - 1))
      {
        outlets.push_back({i, j});
      }
    }

  db.set_outlets(outlets);

  db.compute_receivers_priority_flood();
  db.update_traversals();

  auto upstream_traversals = db.compute_upstream_traversals();

  for (const auto &indices : upstream_traversals)
    for (size_t k = 0; k < indices.size(); ++k)
    {
      const auto &i = indices[k];
      const auto &r = db.receivers(i);

      float new_z = std::max(z(i), z(r));
      z(i) = new_z;
    }

  if (apply_post_filter)
  {
    Array deposition = z - z_bckp;
    laplace(deposition);
    z = z_bckp + deposition;
  }
}

} // namespace hmap
