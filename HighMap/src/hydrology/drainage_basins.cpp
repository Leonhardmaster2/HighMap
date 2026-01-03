/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <queue>

#include "macrologger.h"

#include "highmap/hydrology.hpp"

namespace hmap
{

// TODO define next_i for priority flood

void DrainageBasins::accumulate(const Array &to_accumulate, Array &acc) const
{
  acc = Array(to_accumulate.shape, 0.f);

  for (const auto &basin : this->upstream_traversal)
  {
    // basin order is downstream → upstream
    // so we iterate backwards
    for (int k = int(basin.size()) - 1; k >= 0; --k)
    {
      int i = basin[k].x;
      int j = basin[k].y;

      acc(i, j) += to_accumulate(i, j);

      int ni = this->next_i(i, j);
      int nj = this->next_j(i, j);

      // propagate downstream
      if (ni >= 0) acc(ni, nj) += acc(i, j);
    }
  }
}

Mat<std::vector<Vec2<int>>> DrainageBasins::build_upstream_adjacency()
{
  const Vec2<int>             shape = this->next_i.shape;
  Mat<std::vector<Vec2<int>>> upstream(shape);

  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      int ni = this->next_i(i, j);
      int nj = this->next_j(i, j);

      if (ni >= 0) upstream(ni, nj).push_back(Vec2<int>(i, j));
    }

  return upstream;
}

void DrainageBasins::generate_traversal(const Array        &z,
                                        FlowDirectionMethod fd_method,
                                        bool                remove_lakes)
{
  switch (fd_method)
  {
  case FDM_D8: this->generate_traversal_d8(z, remove_lakes); break;
  case FDM_PRIORITY_FLOOD: this->generate_traversal_priority_flood(z); break;
  }
}

void DrainageBasins::generate_traversal_d8(const Array &z, bool remove_lakes)
{
  this->upstream_traversal.clear();

  const Vec2<int> shape = z.shape;
  this->next_i = Mat<int>(shape, -1);
  this->next_j = Mat<int>(shape, -1);

  const int   di[8] = {1, 1, 0, -1, -1, -1, 0, 1};
  const int   dj[8] = {0, -1, -1, -1, 0, 1, 1, 1};
  const float cd[8] =
      {1.f, M_SQRT1_2, 1.f, M_SQRT1_2, 1.f, M_SQRT1_2, 1.f, M_SQRT1_2};

  // --- compute best downslope neighbor (D8)

  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      float z0 = z(i, j);
      float best_slope = 0.0f;
      int   bi = -1;
      int   bj = -1;

      for (int k = 0; k < 8; ++k)
      {
        int ni = i + di[k];
        int nj = j + dj[k];

        if (ni < 0 || nj < 0 || ni >= shape.x || nj >= shape.y) continue;

        float dz = (z0 - z(ni, nj)) * cd[k];
        if (dz > best_slope)
        {
          best_slope = dz;
          bi = ni;
          bj = nj;
        }
      }

      this->next_i(i, j) = bi;
      this->next_j(i, j) = bj;
    }

  if (remove_lakes) this->remove_lakes_d8(z);
  this->update_traversal();
}

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
  this->next_i = Mat<int>(shape, -1);
  this->next_j = Mat<int>(shape, -1);
  Array basin_id(shape, -1);

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

        this->next_i(i, j) = c.i;
        this->next_j(i, j) = c.j;
      }
    }
  }

  // NB - upstream traversal already updated above and no need to
  // remove lakes since the priority flood already ensures there is no
  // lakes
}

void DrainageBasins::remove_lakes_d8(const Array &z,
                                     float        dz_weight,
                                     float        dz_downstream_cost_ratio)
{
  // scale by array shape to get a cumulative amplitude consistent
  // with distance cost (based on unit cell)
  dz_weight *= z.shape.x;

  struct PQItem
  {
    float cost;
    int   i, j;

    bool operator>(const PQItem &o) const
    {
      return cost > o.cost;
    }
  };

  const Vec2<int> shape = z.shape;

  const int   di[8] = {1, 1, 0, -1, -1, -1, 0, 1};
  const int   dj[8] = {0, -1, -1, -1, 0, 1, 1, 1};
  const float dlen[8] = {1.f,
                         std::sqrt(2.f),
                         1.f,
                         std::sqrt(2.f),
                         1.f,
                         std::sqrt(2.f),
                         1.f,
                         std::sqrt(2.f)};

  // --- find subroots (includes inner domain sinks)

  Vec2<int>      null_cell(-1, -1);
  Mat<Vec2<int>> subroot(shape, null_cell);

  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      if (subroot(i, j) != null_cell) continue;

      std::vector<Vec2<int>> path;
      Vec2<int>              p = {i, j};

      while ((subroot(p) == null_cell) && (this->next_i(p) != -1))
      {
        path.push_back(p);
        p = {this->next_i(p), this->next_j(p)};
      }

      // p is now either:
      // - an outlet
      // - or a lake (self-loop but not outlet)
      // - or already assigned
      Vec2<int> root = p;

      if (subroot(p) != null_cell) root = subroot(p);

      for (const auto &v : path)
        subroot(v) = root;
      subroot(p) = root;
    }

  // --- remove lakes

  std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> pq;
  Mat<int>       visited(shape, 0);
  Mat<Vec2<int>> root(shape, null_cell);

  // initialize heap queue
  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      if (i == 0 || j == 0 || i == shape.x - 1 || j == shape.y - 1)
        if (this->next_i(i, j) == -1)
        {
          root(i, j) = {i, j};
          pq.push({0.f, i, j});
        }
    }

  // helper - lambda
  auto is_inside = [&shape](int i, int j)
  { return i >= 0 && i < shape.x && j >= 0 && j < shape.y; };

  // helper - lambda
  auto reverse_path_to_outlet = [&](Vec2<int> start, Vec2<int> outlet)
  {
    Vec2<int> current = start;
    Vec2<int> prev = outlet;

    while (next_i(current) != -1)
    {
      Vec2<int> next = {next_i(current), next_j(current)};

      next_i(current) = prev.x;
      next_j(current) = prev.y;

      prev = current;
      current = next;
    }

    next_i(current) = prev.x;
    next_j(current) = prev.y;
  };

  // heap queue
  while (!pq.empty())
  {
    auto c = pq.top();
    pq.pop();

    if (visited(c.i, c.j)) continue;

    for (int k = 0; k < 8; ++k)
    {
      int ni = c.i + di[k];
      int nj = c.j + dj[k];

      if (!is_inside(ni, nj) || visited(ni, nj)) continue;

      // connect basin if needed
      auto lake_root = subroot(ni, nj);
      auto outlet_root = subroot(c.i, c.j);

      if (root(lake_root) == null_cell)
      {
        reverse_path_to_outlet({ni, nj}, {c.i, c.j});
        root(lake_root) = root(outlet_root);
      }

      // cumulative distance
      float dz = z(ni, nj) - z(c.i, c.j);
      float dz_cost = dz_weight *
                      (dz < 0.f ? -dz_downstream_cost_ratio * dz : dz);
      float new_cost = c.cost + dlen[k] + dz_cost;
      pq.push({new_cost, ni, nj});
    }

    root(c.i, c.j) = root(subroot(c.i, c.j));
    visited(c.i, c.j) = true;
  }
}

void DrainageBasins::update_traversal()
{
  const Vec2<int> shape = this->next_i.shape;

  Mat<int> basin_id(shape, -1);

  // --- identify outlets (no downslope neighbor)

  int basin_counter = 0;
  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      if (this->next_i(i, j) == -1) basin_id(i, j) = basin_counter++;
    }

  this->upstream_traversal.resize(basin_counter);

  // --- propagate basin ids upstream (iterative relaxation)

  bool updated = true;
  while (updated)
  {
    updated = false;

    for (int j = 0; j < shape.y; ++j)
      for (int i = 0; i < shape.x; ++i)
      {
        if (basin_id(i, j) != -1) continue;

        int ni = this->next_i(i, j);
        int nj = this->next_j(i, j);

        if (ni >= 0 && basin_id(ni, nj) != -1)
        {
          basin_id(i, j) = basin_id(ni, nj);
          updated = true;
        }
      }
  }

  // --- build upstream adjacency

  Mat<std::vector<Vec2<int>>> upstream = this->build_upstream_adjacency();

  // --- build traversal per basin (BFS, receiver-respecting)

  std::queue<Vec2<int>> queue;
  Mat<int>              visited(shape, 0);

  // initialize queue with outlets and sinks
  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
      if (next_i(i, j) == -1)
      {
        queue.push({i, j});
        visited(i, j) = 1;
      }

  while (!queue.empty())
  {
    auto [ci, cj] = queue.front();
    queue.pop();
    int b = basin_id(ci, cj);
    visited(ci, cj) = 1;

    this->upstream_traversal[b].push_back({ci, cj});

    for (const auto &u : upstream(ci, cj))
    {
      if (visited(u.x, u.y)) continue;
      queue.push(u);
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
