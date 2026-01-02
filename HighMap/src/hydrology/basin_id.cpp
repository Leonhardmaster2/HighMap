/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <map>
#include <queue>

#include "macrologger.h"

#include "highmap/hydrology.hpp"

namespace hmap
{

Array basin_id_priority_flood(const Array &z)
{
  Array ids(z.shape);

  auto basins = DrainageBasins();
  basins.generate_traversal_priority_flood(z);

  auto lambda = [&ids](int i, int j, int basin_id) { ids(i, j) = basin_id; };

  basins.traverse_upstream(lambda);

  return ids;
}

} // namespace hmap
