/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <queue>

#include "macrologger.h"

#include "highmap/array.hpp"
#include "highmap/hydrology.hpp"

namespace hmap
{

void depression_filling_priority_flood(Array &z)
{
  auto basins = DrainageBasins();
  basins.generate_traversal_priority_flood(z);

  auto lambda = [&z](int i, int j, int i_next, int j_next, int /* basin_id */)
  {
    float new_z = std::max(z(i_next, j_next), z(i, j));
    z(i, j) = new_z;
  };

  basins.traverse_upstream(lambda);
}

} // namespace hmap
