/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <queue>

#include "highmap/hydrology.hpp"

namespace hmap
{

struct HelperNode
{
  int   i, j;
  float z;
};

struct HelperNodeCmp
{
  bool operator()(const HelperNode &a, const HelperNode &b) const
  {
    return a.z > b.z; // min-heap
  }
};

Array basin_id_priority_flood(const Array &z)
{
  const Vec2<int> shape = z.shape;
  Array           basin_id(shape, -1);

  Array spill(shape);

  std::priority_queue<HelperNode, std::vector<HelperNode>, HelperNodeCmp> pq;

  int basin_counter = 0;

  // --- initialize boundary

  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      bool boundary = (i == 0 || j == 0 || i == shape.x - 1 ||
                       j == shape.y - 1);

      if (boundary)
      {
        basin_id(i, j) = basin_counter++;
        pq.push({i, j, z(i, j)});
      }
    }

  const int di[8] = {1, 1, 0, -1, -1, -1, 0, 1};
  const int dj[8] = {0, -1, -1, -1, 0, 1, 1, 1};

  // --- flood inward

  while (!pq.empty())
  {
    HelperNode c = pq.top();
    pq.pop();

    int bc = basin_id(c.i, c.j);

    for (int k = 0; k < 8; ++k)
    {
      int i = c.i + di[k];
      int j = c.j + dj[k];

      if (i < 0 || j < 0 || i >= shape.x || j >= shape.y) continue;
      // TODO record border cells

      if (basin_id(i, j) == -1)
      {
        basin_id(i, j) = bc;
        pq.push({i, j, std::max(z(i, j), c.z)});
      }
      else if (basin_id(i, j) != bc)
      {
        float spill = std::max(z(i, j), c.z);
        // TODO record spill between basins

        // TODO record cell topographic order
      }
    }
  }

  return basin_id;
}

} // namespace hmap
