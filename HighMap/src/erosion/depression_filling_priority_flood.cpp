/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <queue>

#include "macrologger.h"

#include "highmap/array.hpp"
#include "highmap/hydrology/hydrology.hpp"

namespace hmap
{

void depression_filling_priority_flood(Array &z)
{
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
}

} // namespace hmap
