/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <map>
#include <queue>

#include "macrologger.h"

#include "highmap/hydrology.hpp"

namespace hmap
{

Array basin_id(const Array &z, FlowDirectionMethod fd_method, bool remove_lakes)
{
  Array ids(z.shape);

  auto basins = DrainageBasins();
  basins.generate_traversal(z, fd_method, remove_lakes);

  auto lambda = [&ids](int i, int j, int basin_id) { ids(i, j) = basin_id; };
  basins.traverse_upstream(lambda);

  return ids;
}

} // namespace hmap
