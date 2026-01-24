/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <queue>

#include "macrologger.h"

#include "highmap/hydrology.hpp"

namespace hmap
{

void DrainageBasins::accumulate(Array &acc) const
{
  for (const auto &basin : this->upstream_traversal)
  {
    // basin order is downstream → upstream
    // so we iterate backwards
    for (int k = int(basin.size()) - 1; k >= 0; --k)
    {
      glm::ivec2 c = basin[k];
      glm::ivec2 p = this->next(c);

      // propagate downstream
      if (p != this->null_cell) acc(p) += acc(c);
    }
  }
}

void DrainageBasins::generate_traversal(const Array        &z,
                                        FlowDirectionMethod fd_method,
                                        bool                remove_lakes,
                                        const std::vector<glm::ivec2> &outlets)
{
  switch (fd_method)
  {
  case FDM_D8: this->generate_traversal_d8(z, remove_lakes, outlets); break;
  case FDM_PRIORITY_FLOOD:
    this->generate_traversal_priority_flood(z, outlets);
    break;
  }
}

void DrainageBasins::generate_traversal_d8(
    const Array                   &z,
    bool                           remove_lakes,
    const std::vector<glm::ivec2> &outlets)
{
  this->upstream_traversal.clear();

  const glm::ivec2 shape = z.shape;
  this->next = Mat<glm::ivec2>(shape, this->null_cell);

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

      this->next(i, j) = {bi, bj};
    }

  // prescribe some outlets
  if (!outlets.empty())
  {
    for (const auto &p : outlets)
      this->next(p) = this->null_cell;
  }

  if (remove_lakes) this->remove_lakes_d8(z);
  this->update_traversal();
}

void DrainageBasins::generate_traversal_priority_flood(
    const Array                   &z,
    const std::vector<glm::ivec2> &outlets)
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
  const glm::ivec2 shape = z.shape;
  this->next = Mat<glm::ivec2>(shape, this->null_cell);
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
    this->upstream_traversal[bc].push_back(glm::ivec2(c.i, c.j));

    for (int k = 0; k < 8; ++k)
    {
      int i = c.i + di[k];
      int j = c.j + dj[k];

      if (i < 0 || j < 0 || i >= shape.x || j >= shape.y) continue;

      if (basin_id(i, j) == -1)
      {
        basin_id(i, j) = bc;
        pq.push({i, j, std::max(z(i, j), c.z)});

        this->next(i, j) = {c.i, c.j};
      }
    }
  }

  // NB - upstream traversal already updated above and no need to
  // remove lakes since the priority flood already ensures there is no
  // lakes, unless outlets habe changed

  // prescribe some outlets
  if (!outlets.empty())
  {
    for (const auto &p : outlets)
      this->next(p) = this->null_cell;
  }
}

size_t DrainageBasins::get_basins_number() const
{
  return this->upstream_traversal.size();
}

std::vector<std::vector<glm::ivec2>> DrainageBasins::get_main_channels()
{
  const glm::ivec2 shape = this->next.shape;

  // compute flow accumulation
  Array area_acc(shape, 1.f);
  this->accumulate(area_acc);

  const int di[8] = {1, 1, 0, -1, -1, -1, 0, 1};
  const int dj[8] = {0, -1, -1, -1, 0, 1, 1, 1};

  // for each basin, follow downstream the flow acc maximum to get the
  // basin main channel
  size_t                               nbasins = this->get_basins_number();
  std::vector<std::vector<glm::ivec2>> channels(nbasins);

  for (size_t basin_id = 0; basin_id < nbasins; ++basin_id)
  {
    // starting cell (most upstream)
    glm::ivec2 p = this->upstream_traversal[basin_id].back();
    channels[basin_id].push_back(p);

    while (p.x > 0 && p.y > 0 && p.x < shape.x - 1 && p.y < shape.y - 1)
    {
      for (int k = 0; k < 8; ++k)
      {
        int ni = p.x + di[k];
        int nj = p.y + dj[k];

        if (ni < 0 || nj < 0 || ni >= shape.x || nj >= shape.y) continue;

        if (area_acc(ni, nj) > area_acc(p)) p = {ni, nj};
      }

      channels[basin_id].push_back(p);
    }
  }

  return channels;
}

std::vector<glm::ivec2> DrainageBasins::get_outlets() const
{
  const glm::ivec2        shape = this->next.shape;
  std::vector<glm::ivec2> outlets = {};

  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      if (this->next(i, j) == this->null_cell) outlets.push_back({i, j});
    }

  return outlets;
}

std::vector<glm::ivec2> DrainageBasins::get_ridges()
{
  const glm::ivec2        shape = this->next.shape;
  std::vector<glm::ivec2> ridges;

  // --- get basin ids

  Mat<int> ids(shape);
  auto lambda = [&ids](int i, int j, int basin_id) { ids(i, j) = basin_id; };
  this->traverse_upstream(lambda);

  // --- tag ridges

  const int di[8] = {1, 1, 0, -1, -1, -1, 0, 1};
  const int dj[8] = {0, -1, -1, -1, 0, 1, 1, 1};

  auto is_inside = [&shape](int i, int j)
  { return i >= 0 && i < shape.x && j >= 0 && j < shape.y; };

  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      int                     current_id = ids(i, j);
      std::vector<glm::ivec2> ridge_neighbors = {};

      for (int k = 0; k < 8; ++k)
      {
        int ni = i + di[k];
        int nj = j + dj[k];

        if (ids(ni, nj) != current_id && is_inside(ni, nj))
        {
          ridges.push_back({i, j});
          break;
        }
      }
    }

  return ridges;
}

std::vector<std::vector<glm::ivec2>> DrainageBasins::get_ridges_neighbors()
{
  const glm::ivec2                     shape = this->next.shape;
  std::vector<std::vector<glm::ivec2>> ridges;

  // --- get basin ids

  Mat<int> ids(shape);
  auto lambda = [&ids](int i, int j, int basin_id) { ids(i, j) = basin_id; };
  this->traverse_upstream(lambda);

  // --- tag ridges

  const int di[8] = {1, 1, 0, -1, -1, -1, 0, 1};
  const int dj[8] = {0, -1, -1, -1, 0, 1, 1, 1};

  auto is_inside = [&shape](int i, int j)
  { return i >= 0 && i < shape.x && j >= 0 && j < shape.y; };

  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      int                     current_id = ids(i, j);
      std::vector<glm::ivec2> ridge_neighbors = {};

      for (int k = 0; k < 8; ++k)
      {
        int ni = i + di[k];
        int nj = j + dj[k];

        if (ids(ni, nj) != current_id && is_inside(ni, nj))
          ridge_neighbors.push_back({ni, nj});
      }

      if (!ridge_neighbors.empty())
      {
        ridge_neighbors.push_back({i, j}); // ref pt at the end
        ridges.push_back(ridge_neighbors);
      }
    }

  return ridges;
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

  const glm::ivec2 shape = z.shape;

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

  Mat<glm::ivec2> subroot(shape, null_cell);

  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      if (subroot(i, j) != null_cell) continue;

      std::vector<glm::ivec2> path;
      glm::ivec2              p = {i, j};

      while ((subroot(p) == null_cell) && (this->next(p) != this->null_cell))
      {
        path.push_back(p);
        p = this->next(p);
      }

      // p is now either:
      // - an outlet
      // - or a lake (self-loop but not outlet)
      // - or already assigned
      glm::ivec2 root = p;

      if (subroot(p) != null_cell) root = subroot(p);

      for (const auto &v : path)
        subroot(v) = root;
      subroot(p) = root;
    }

  // --- remove lakes

  std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> pq;
  Mat<int>        visited(shape, 0);
  Mat<glm::ivec2> root(shape, null_cell);

  // initialize heap queue
  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      if (i == 0 || j == 0 || i == shape.x - 1 || j == shape.y - 1)
        if (this->next(i, j) == this->null_cell)
        {
          root(i, j) = {i, j};
          pq.push({0.f, i, j});
        }
    }

  // helper - lambda
  auto is_inside = [&shape](int i, int j)
  { return i >= 0 && i < shape.x && j >= 0 && j < shape.y; };

  // helper - lambda
  auto reverse_path_to_outlet = [&](glm::ivec2 start, glm::ivec2 outlet)
  {
    glm::ivec2 current = start;
    glm::ivec2 prev = outlet;

    while (next(current) != this->null_cell)
    {
      glm::ivec2 next = this->next(current);
      this->next(current) = prev;
      prev = current;
      current = next;
    }

    this->next(current) = prev;
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

void DrainageBasins::traverse_upstream(
    std::function<void(int i, int j, int i_next, int j_next, int basin_id)> op)
{
  for (size_t basin_id = 0; basin_id < this->upstream_traversal.size();
       ++basin_id)
  {
    const auto &path = this->upstream_traversal[basin_id];

    for (size_t k = 0; k < path.size(); ++k)
    {
      int i = path[k].x;
      int j = path[k].y;
      int ni = this->next(i, j).x;
      int nj = this->next(i, j).y;

      if (ni >= 0) op(i, j, ni, nj, basin_id);
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
      int i = path[k].x;
      int j = path[k].y;
      op(i, j, int(basin_id));
    }
  }
}

void DrainageBasins::traverse_downstream(
    std::function<void(int i, int j, int i_next, int j_next, int basin_id)> op)
{
  for (size_t basin_id = 0; basin_id < this->upstream_traversal.size();
       ++basin_id)
  {
    const auto &basin = this->upstream_traversal[basin_id];

    // basin order is downstream → upstream
    // so we iterate backwards
    for (int k = int(basin.size()) - 1; k >= 0; --k)
    {
      int i = basin[k].x;
      int j = basin[k].y;
      int ni = this->next(i, j).x;
      int nj = this->next(i, j).y;

      if (ni >= 0) op(i, j, ni, nj, basin_id);
    }
  }
}

void DrainageBasins::traverse_downstream(
    std::function<void(int i, int j, int basin_id)> op)
{
  for (size_t basin_id = 0; basin_id < this->upstream_traversal.size();
       ++basin_id)
  {
    const auto &basin = this->upstream_traversal[basin_id];

    // basin order is downstream → upstream
    // so we iterate backwards
    for (int k = int(basin.size()) - 1; k >= 0; --k)
    {
      int i = basin[k].x;
      int j = basin[k].y;
      op(i, j, basin_id);
    }
  }
}

void DrainageBasins::update_traversal()
{
  const glm::ivec2 shape = this->next.shape;
  Mat<int>         basin_id(shape, -1);

  this->upstream_traversal.clear();

  // --- identify outlets (no downslope neighbor)

  int basin_counter = 0;
  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      if (this->next(i, j) == this->null_cell)
      {
        basin_id(i, j) = basin_counter++;

        if (int(this->upstream_traversal.size()) <= basin_id(i, j))
          this->upstream_traversal.resize(basin_id(i, j) + 1);
        this->upstream_traversal[basin_id(i, j)].push_back({i, j});
      }
    }

  // --- propagate upstream

  const int di[8] = {1, 1, 0, -1, -1, -1, 0, 1};
  const int dj[8] = {0, -1, -1, -1, 0, 1, 1, 1};

  for (size_t bid = 0; bid < this->upstream_traversal.size(); ++bid)
  {
    size_t count = 0;

    while (count < this->upstream_traversal[bid].size())
    {
      glm::ivec2 p = this->upstream_traversal[bid][count];
      for (int k = 0; k < 8; ++k)
      {
        // add upstream nodes whose receiver points to 'p'
        int i = p.x + di[k];
        int j = p.y + dj[k];

        if (i < 0 || j < 0 || i >= shape.x || j >= shape.y) continue;

        if (this->next(i, j) == p)
          this->upstream_traversal[bid].push_back({i, j});
      }
      count++;
    }
  }

  // fail check
  size_t ctot = 0;
  for (auto vec : this->upstream_traversal)
    ctot += vec.size();

  if (int(ctot) != shape.x * shape.y)
    LOG_ERROR("DrainageBasins::update_traversal: wrong count: %ld, %d",
              ctot,
              shape.x * shape.y);
}

} // namespace hmap
