/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <queue>

#include "macrologger.h"

#include "highmap/array.hpp"

namespace hmap
{

void depression_filling_priority_flood(Array &z)
{
  struct Node
  {
    int   i;
    int   j;
    float z;
  };

  struct NodeCmp
  {
    bool operator()(const Node &a, const Node &b) const
    {
      return a.z > b.z; // min-heap
    }
  };

  const Vec2<int> shape = z.shape;
  const int       w = shape.x;
  const int       h = shape.y;

  Array basin_id(shape, -1);

  std::priority_queue<Node, std::vector<Node>, NodeCmp> pq;

  int basin_counter = 0;

  // --- initialize boundary
  for (int j = 0; j < h; ++j)
    for (int i = 0; i < w; ++i)
    {
      if (i == 0 || j == 0 || i == w - 1 || j == h - 1)
      {
        basin_id(i, j) = basin_counter++;
        pq.emplace(i, j, z(i, j));
      }
    }

  // 8-connectivity
  constexpr int di[8] = {1, 1, 0, -1, -1, -1, 0, 1};
  constexpr int dj[8] = {0, -1, -1, -1, 0, 1, 1, 1};

  // --- priority flood
  while (!pq.empty())
  {
    Node c = pq.top();
    pq.pop();

    int bc = basin_id(c.i, c.j);

    for (int k = 0; k < 8; ++k)
    {
      int i = c.i + di[k];
      int j = c.j + dj[k];

      if (i < 0 || j < 0 || i >= w || j >= h) continue;

      if (basin_id(i, j) == -1)
      {
        basin_id(i, j) = bc;

        float new_z = std::max(z(i, j), c.z);
        z(i, j) = new_z;

        pq.emplace(i, j, new_z);
      }
    }
  }
}

} // namespace hmap
