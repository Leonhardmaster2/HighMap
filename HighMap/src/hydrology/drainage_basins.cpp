/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <queue>

#include "macrologger.h"

#include "highmap/hydrology.hpp"

namespace hmap
{

void DrainageBasins::generate_traversal_priority_flood(const Array &z)
{
  this->upstream_traversal.clear();

  // local node type
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

  // algo
  const Vec2<int> shape = z.shape;
  Array           basin_id(shape, -1);

  std::priority_queue<Node, std::vector<Node>, NodeCmp> pq;
  int                                                   basin_counter = 0;

  // --- initialize boundary

  // outlet elevation storage
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

  // --- flood inward

  this->upstream_traversal.resize(2 * shape.x + 2 * (shape.y - 2));

  const int di[8] = {1, 1, 0, -1, -1, -1, 0, 1};
  const int dj[8] = {0, -1, -1, -1, 0, 1, 1, 1};

  Array zm = z;

  while (!pq.empty())
  {
    Node c = pq.top();
    pq.pop();

    int bc = basin_id(c.i, c.j);
    this->upstream_traversal[bc].push_back(Vec2<int>(c.i, c.j));

    for (int k = 0; k < 8; ++k)
    {
      int i = c.i + di[k];
      int j = c.j + dj[k];

      if (i < 0 || j < 0 || i >= shape.x || j >= shape.y) continue;

      if (basin_id(i, j) == -1)
      {
        basin_id(i, j) = bc;
        pq.push({i, j, std::max(z(i, j), c.z)});
      }
    }
  }
}

void DrainageBasins::traverse_upstream(
    std::function<void(int i, int j, int i_next, int j_next, int basin_id)> op)
{
  for (size_t basin_id = 0; basin_id < this->upstream_traversal.size();
       ++basin_id)
  {
    const auto &path = this->upstream_traversal[basin_id];
    if (path.size() < 2) continue;

    for (size_t k = 0; k + 1 < path.size(); ++k)
    {
      auto [i, j] = path[k];
      auto [i_next, j_next] = path[k + 1];
      op(i, j, i_next, j_next, basin_id);
    }
  }
}

void DrainageBasins::traverse_upstream(
    std::function<void(int i, int j, int basin_id)> op)
{
  for (size_t basin_id = 0; basin_id < this->upstream_traversal.size();
       ++basin_id)
  {
    const auto &path = this->upstream_traversal[basin_id];

    for (size_t k = 0; k < path.size(); ++k)
    {
      auto [i, j] = path[k];
      op(i, j, basin_id);
    }
  }
}

void DrainageBasins::traverse_downstream(
    std::function<void(int i, int j, int i_next, int j_next, int basin_id)> op)
{
  for (size_t basin_id = 0; basin_id < this->upstream_traversal.size();
       ++basin_id)
  {
    const auto &path = this->upstream_traversal[basin_id];
    if (path.size() < 2) continue;

    for (size_t k = path.size() - 1; k > 0; --k)
    {
      auto [i, j] = path[k];
      auto [i_next, j_next] = path[k - 1];
      op(i, j, i_next, j_next, basin_id);
    }
  }
}

void DrainageBasins::traverse_downstream(
    std::function<void(int i, int j, int basin_id)> op)
{
  for (size_t basin_id = 0; basin_id < this->upstream_traversal.size();
       ++basin_id)
  {
    const auto &path = this->upstream_traversal[basin_id];

    for (size_t k = path.size(); k-- > 0;)
    {
      auto [i, j] = path[k];
      op(i, j, basin_id);
    }
  }
}

} // namespace hmap
