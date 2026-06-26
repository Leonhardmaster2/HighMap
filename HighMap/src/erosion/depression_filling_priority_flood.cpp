/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm>
#include <cstddef>
#include <vector>

#include "highmap/algebra.hpp"
#include "highmap/array.hpp"
#include "highmap/filters.hpp"
#include "highmap/hydrology/drainage_basin_cell_based.hpp"

namespace hmap
{

void depression_filling_priority_flood(Array &z, bool apply_post_filter)
{
  Array z_bckp = apply_post_filter ? z : Array();

  auto db = DrainageBasinCellBased(z);

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
