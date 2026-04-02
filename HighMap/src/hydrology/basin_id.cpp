/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <map>
#include <queue>

#include "macrologger.h"

#include "highmap/hydrology/hydrology.hpp"

namespace hmap
{

Array basin_id(const Array &z, FlowDirectionMethod fd_method, bool remove_lakes)
{
  Array ids(z.shape);

  // --- stream structure

  auto db = DrainageBasinCellBased(z);

  if (fd_method == FlowDirectionMethod::FDM_D8)
  {
    db.compute_receivers();
    auto [subroots, has_lake] = db.find_subroots();
    if (has_lake) db.remove_lakes(subroots);
  }
  else
    db.compute_receivers_priority_flood();

  db.update_traversals();

  // --- basin Ids

  auto upstream_traversals = db.compute_upstream_traversals();
  int  id_count = 0;

  for (const auto &indices : upstream_traversals)
  {
    for (size_t k = 0; k < indices.size(); ++k)
    {
      const auto &i = indices[k];
      ids(i) = id_count;
    }
    id_count++;
  }

  return ids;
}

} // namespace hmap
