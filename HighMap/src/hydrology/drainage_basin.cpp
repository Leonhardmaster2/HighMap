/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <cstdio>
#include <fstream>
#include <queue>

#include "hmm/src/heightmap.h"
#include "hmm/src/triangulator.h"

#include "delaunator-cpp.hpp"

#include "macrologger.h"

#include "highmap/dbg/timer.hpp"
#include "highmap/hydrology/drainage_basin.hpp"
#include "highmap/math.hpp"

namespace hmap
{

DrainageBasin::DrainageBasin(std::vector<glm::vec3> xyz_) : xyz(xyz_)
{
  // --- generate Delaunay triangulation

  std::vector<float> xy;
  xy.reserve(2 * this->xyz.size());

  for (const auto &p : this->xyz)
  {
    xy.push_back(p.x);
    xy.push_back(p.y);
  }

  delaunator::Delaunator d(xy);

  // --- store triangles and convex hull indices

  // chull
  {
    this->convex_hull.clear();
    size_t start = d.hull_start;
    size_t i = start;

    do
    {
      this->convex_hull.push_back(i);
      i = d.hull_next[i];
    } while (i != start);
  }

  // triangles
  {
    const std::vector<size_t> &tri = d.triangles;
    this->triangles.clear();
    this->triangles.reserve(tri.size() / 3);

    for (std::size_t k = 0; k < tri.size(); k += 3)
      this->triangles.push_back({tri[k], tri[k + 1], tri[k + 2]});
  }

  // --- neighbor list for each vertex and corresponding distances

  this->nbrs_indices.assign(xyz.size(), {});
  this->nbrs_distances.assign(xyz.size(), {});

  auto add_edge = [&](size_t a, size_t b)
  {
    // prevent duplicates
    auto exists = [&](size_t u, size_t v)
    {
      return std::find(nbrs_indices[u].begin(), nbrs_indices[u].end(), v) !=
             nbrs_indices[u].end();
    };

    if (exists(a, b) || exists(b, a)) return;

    float dist = std::hypot(xyz[a].x - xyz[b].x, xyz[a].y - xyz[b].y);

    this->nbrs_indices[a].push_back(b);
    this->nbrs_indices[b].push_back(a);

    this->nbrs_distances[a].push_back(dist);
    this->nbrs_distances[b].push_back(dist);
  };

  for (const auto &t : this->triangles)
  {
    add_edge(t.x, t.y);
    add_edge(t.y, t.z);
    add_edge(t.z, t.x);
  }

  // --- reference length between points for scaling

  this->reference_length = 0.f;

  for (std::size_t k = 0; k < this->size(); ++k)
    for (float d : nbrs_distances[k])
      reference_length = std::max(this->reference_length, d);

  // --- outlets == convex hull by default

  this->outlets_mask.resize(this->xyz.size());
  for (const auto &idx : this->convex_hull)
    this->outlets_mask[idx] = true;
}

void DrainageBasin::accumulate_area_by_outlet(const std::vector<float> &area,
                                              std::vector<float> &acc) const
{
  for (size_t o : this->get_outlets())
  {
    const auto &traversal = this->traversals.at(o);

    for (auto it = traversal.begin(); it != traversal.end(); ++it)
    {
      size_t v = *it;
      acc[v] = area[v];

      for (size_t child : this->children[v])
        acc[v] += acc[child];
    }
  }
}

void DrainageBasin::compute_receivers()
{
  this->receivers.clear();
  this->receivers.resize(this->xyz.size());

  for (size_t k = 0; k < this->xyz.size(); ++k)
  {
    if (this->outlets_mask[k])
    {
      this->receivers[k] = k; // mark as self-receiver
      continue;
    }

    float  best_slope = 0.f;
    size_t best_k = k;

    for (size_t j = 0; j < nbrs_indices[k].size(); ++j)
    {
      size_t n = nbrs_indices[k][j];
      float  dist = nbrs_distances[k][j];

      if (this->xyz[k].z > this->xyz[n].z)
      {
        float slope = (this->xyz[k].z - this->xyz[n].z) / dist;
        if (slope > best_slope)
        {
          best_slope = slope;
          best_k = n;
        }
      }
    }

    this->receivers[k] = best_k;
  }
}

std::vector<size_t> DrainageBasin::compute_strahler_order() const
{
  const size_t n = receivers.size();

  std::vector<size_t> order(n, 1);
  std::vector<size_t> max_child(n, 0);
  std::vector<size_t> count_max(n, 0);

  for (size_t o : this->get_outlets())
  {
    const auto &traversal = this->traversals.at(o);

    for (auto it = traversal.begin(); it != traversal.end(); ++it)
    {
      size_t i = *it;

      size_t child_order_max = 1;
      for (size_t child : this->children[i])
        if (order[child] > child_order_max) child_order_max = order[child];

      order[i] = child_order_max;

      if (this->children[i].size() > 1) order[i]++;
    }
  }

  return order;
}

std::vector<float> DrainageBasin::compute_response_times(
    const std::vector<float> &area_acc,
    const std::vector<float> &erodibility,
    float                     m_exp) const
{
  const size_t       n = this->receivers.size();
  std::vector<float> response_times(n, 0.f);
  float              area_scale = reference_length * reference_length;

  for (const auto &[outlet, traversal] : traversals)
  {
    // downstream => upstream
    for (auto it = traversal.rbegin(); it != traversal.rend(); ++it)
    {
      size_t i = *it;
      size_t j = this->receivers[i];

      if (j != i)
      {
        float distance = glm::length(this->xyz[i] - this->xyz[j]);
        distance /= this->reference_length;

        float celerity = erodibility[i] *
                         std::pow(std::max(area_acc[i] / area_scale, 1e-8f),
                                  m_exp);

        response_times[i] = response_times[j] + distance / celerity;
      }
      else
      {
        response_times[i] = 0.f; // outlet
      }
    }
  }

  return response_times;
}

std::vector<float> DrainageBasin::compute_vertex_areas() const
{
  std::vector<float> areas(this->xyz.size());

  // accumulate 1/3 area of the triangles to each vertex
  for (const auto &idx : this->triangles)
  {
    const auto &p0 = this->xyz[idx.x];
    const auto &p1 = this->xyz[idx.y];
    const auto &p2 = this->xyz[idx.z];

    // edge vectors
    auto e0 = p1 - p0;
    auto e1 = p2 - p0;

    // triangle area
    float triangle_area = 0.5f * glm::length(glm::cross(e0, e1));
    float vertex_contribution = triangle_area / 3.f;

    areas[idx.x] += vertex_contribution;
    areas[idx.y] += vertex_contribution;
    areas[idx.z] += vertex_contribution;
  }

  return areas;
}

std::pair<std::vector<size_t>, bool> DrainageBasin::find_subroots()
{
  const size_t n = receivers.size();

  std::vector<size_t> subroot(n, static_cast<size_t>(-1));
  bool                has_lake = false;

  for (size_t i = 0; i < n; ++i)
  {
    if (subroot[i] != static_cast<size_t>(-1)) continue;

    std::vector<size_t> path;
    size_t              p = i;

    while (subroot[p] == static_cast<size_t>(-1) && receivers[p] != p)
    {
      path.push_back(p);
      p = receivers[p];
    }

    size_t root;

    if (subroot[p] != static_cast<size_t>(-1))
    {
      // already assigned
      root = subroot[p];
    }
    else if (outlets_mask[p])
    {
      // outlet
      root = p;
    }
    else
    {
      // true lake
      has_lake = true;
      root = p;
    }

    for (size_t v : path)
      subroot[v] = root;

    subroot[p] = root;
  }

  return {subroot, has_lake};
}

const std::vector<size_t> &DrainageBasin::for_each_upstream(size_t outlet) const
{
  return traversals.at(outlet);
}

std::vector<size_t> DrainageBasin::get_outlets() const
{
  std::vector<size_t> outlet_indices;

  for (size_t k = 0; k < this->xyz.size(); ++k)
    if (this->outlets_mask[k]) outlet_indices.push_back(k);

  return outlet_indices;
}

const std::vector<size_t> &DrainageBasin::get_receivers() const
{
  return this->receivers;
}

const std::vector<glm::vec3> &DrainageBasin::get_xyz() const
{
  return this->xyz;
}

std::vector<glm::vec3> &DrainageBasin::get_xyz()
{
  return this->xyz;
}

void DrainageBasin::remap(float vmin, float vmax)
{
  if (this->xyz.empty()) return;

  float zmin = this->xyz[0].z;
  float zmax = this->xyz[0].z;

  for (const auto &p : this->xyz)
  {
    zmin = std::min(zmin, p.z);
    zmax = std::max(zmax, p.z);
  }

  float dz = zmax - zmin;

  if (dz == 0.f)
  {
    float mid = 0.5f * (vmin + vmax);
    for (auto &p : this->xyz)
      p.z = mid;
    return;
  }

  float scale = (vmax - vmin) / dz;

  for (auto &p : this->xyz)
    p.z = vmin + (p.z - zmin) * scale;
}

void DrainageBasin::remove_lakes(const std::vector<size_t> &subroot)
{
  const size_t n = receivers.size();

  std::vector<uint8_t> visited(n, 0);
  std::vector<float>   dist(n, std::numeric_limits<float>::max());

  roots.assign(n, invalid_index);

  struct HeapNode
  {
    float  dist;
    size_t node;

    bool operator>(const HeapNode &o) const noexcept
    {
      return dist > o.dist;
    }
  };

  std::vector<HeapNode> heap_storage;
  heap_storage.reserve(n * 2);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"

  std::priority_queue<HeapNode, std::vector<HeapNode>, std::greater<HeapNode>>
      heap(std::greater<HeapNode>(), std::move(heap_storage));

#pragma GCC diagnostic pop

  size_t remaining_lakes = 0;
  for (size_t i = 0; i < n; ++i)
    if (subroot[i] != invalid_index) remaining_lakes++;

  // initialize outlets
  for (size_t k = 0; k < n; ++k)
  {
    if (outlets_mask[k])
    {
      roots[k] = k;
      dist[k] = 0.f;
      heap.push({0.f, k});
    }
  }

  const size_t *subroot_ptr = subroot.data();

  while (!heap.empty())
  {
    HeapNode top = heap.top();
    heap.pop();

    size_t i = top.node;
    if (visited[i]) continue;

    visited[i] = 1;

    auto &nbrs = nbrs_indices[i];
    auto &dists = nbrs_distances[i];

    for (size_t k = 0, sz = nbrs.size(); k < sz; ++k)
    {
      size_t j = nbrs[k];
      if (visited[j]) continue;

      size_t sr = subroot_ptr[j];

      if (sr != invalid_index && roots[sr] == invalid_index)
      {
        size_t k_node = j;
        size_t nk = i;

        while (receivers[k_node] != k_node)
        {
          size_t tmp = receivers[k_node];
          receivers[k_node] = nk;
          nk = k_node;
          k_node = tmp;
        }

        receivers[k_node] = nk;

        roots[subroot[j]] = roots[subroot[i]];

        if (--remaining_lakes == 0) return;
      }

      float new_dist = top.dist + dists[k];

      if (new_dist < dist[j])
      {
        dist[j] = new_dist;
        heap.push({new_dist, j});
      }
    }

    size_t sri = subroot_ptr[i];
    if (sri != invalid_index) roots[i] = roots[sri];
  }
}

void DrainageBasin::set_outlets(const std::vector<size_t> &outlet_indices)
{
  this->outlets_mask = std::vector<bool>(this->size(), false);

  for (const auto &idx : outlet_indices)
    this->outlets_mask[idx] = true;
}

size_t DrainageBasin::size() const
{
  return this->xyz.size();
};

void DrainageBasin::smooth_mesh(float lambda, int iterations)
{
  std::vector<glm::vec3> buffer(this->size());

  for (int it = 0; it < iterations; ++it)
  {
    for (size_t i = 0; i < this->size(); ++i)
    {
      const auto &nbrs = this->nbrs_indices[i];

      if (nbrs.empty())
      {
        buffer[i] = xyz[i];
        continue;
      }

      glm::vec3 sum = glm::vec3(0.f);
      float     sum_w = 0.f;

      for (size_t j : nbrs)
      {
        float dist = glm::length(xyz[i] - xyz[j]);
        if (dist > 1e-30f)
        {
          float w = 1.f / dist;
          sum += w * xyz[j];
          sum_w += w;
        }
      }

      glm::vec3 avg = sum / sum_w;

      buffer[i] = glm::mix(xyz[i], avg, lambda);
    }

    // preserve convex hull
    for (const auto &i : this->convex_hull)
      buffer[i] = xyz[i];

    for (size_t i = 0; i < this->size(); ++i)
      xyz[i] = buffer[i];
  }
}

void DrainageBasin::smooth_mesh_taubin(float lambda, float mu, int iterations)
{
  for (int it = 0; it < iterations; ++it)
  {
    int sub_iterations = 1;
    this->smooth_mesh(lambda, sub_iterations);
    this->smooth_mesh(mu, sub_iterations);
  }
}

void DrainageBasin::to_csv(const std::string &filename) const
{
  std::ofstream f(filename, std::ios::out);
  if (!f.is_open()) return;

  auto order = this->compute_strahler_order();
  auto area = this->compute_vertex_areas();

  std::vector<float> acc(this->size(), 0.f);
  std::vector<float> flow(this->size(), 1.f);
  this->accumulate_area_by_outlet(area, acc);
  this->accumulate_area_by_outlet(area, flow);

  f << "# vertices\n";
  f << "# "
       "vertex_id,x,y,z,is_outlet,receiver,root,order,area,area_acc,flow_"
       "acc\n";

  for (size_t i = 0; i < static_cast<size_t>(xyz.size()); ++i)
  {
    size_t r = (i < receivers.size()) ? receivers[i] : -1;
    size_t rt = (i < roots.size()) ? roots[i] : -1;

    f << i << "," << xyz[i].x << "," << xyz[i].y << "," << xyz[i].z << ","
      << (outlets_mask[i] ? 1 : 0) << "," << r << "," << rt << "," << order[i]
      << "," << area[i] << "," << acc[i] << "," << flow[i] << "\n";
  }

  f.close();
}

float DrainageBasin::update_elevations(const std::vector<float> &response_times,
                                       float                     uplift_rate,
                                       const std::vector<float> &max_slope)
{
  const size_t n = this->receivers.size();
  float        delta_sum = 0.f;

  // precompute outlet_of
  std::vector<size_t> outlet_of(n);

  for (const auto &[outlet, traversal] : traversals)
  {
    for (size_t i : traversal)
      outlet_of[i] = outlet;
  }

  // iterate upstream (outlet first)
  for (const auto &[outlet, traversal] : traversals)
  {
    for (size_t i : traversal)
    {
      size_t j = this->receivers[i];

      if (j == i) continue; // skip outlet

      float new_elevation = this->xyz[outlet].z +
                            uplift_rate * std::max(response_times[i] -
                                                       response_times[outlet],
                                                   0.f);

      // slope limiting
      {
        // xy-plane distance only for splope
        float distance = std::hypot(this->xyz[i].x - this->xyz[j].x,
                                    this->xyz[i].y - this->xyz[j].y);

        float slope = (new_elevation - this->xyz[j].z) / distance;

        if (slope > max_slope[i])
          new_elevation = this->xyz[j].z + max_slope[i] * distance;

        float min_slope = 0.f;
        if (slope < min_slope)
          new_elevation = this->xyz[j].z + min_slope * distance;
      }

      float delta = std::abs(new_elevation - this->xyz[i].z);
      delta_sum += delta;

      this->xyz[i].z = new_elevation;
    }
  }

  delta_sum /= float(this->size());

  return delta_sum;
}

void DrainageBasin::update_stream_tree()
{
  hmap::Timer::Start("update_stream_tree/A");
  this->compute_receivers();
  hmap::Timer::Stop("update_stream_tree/A");

  hmap::Timer::Start("update_stream_tree/B");
  auto [subroots, has_lake] = this->find_subroots();
  hmap::Timer::Stop("update_stream_tree/B");

  hmap::Timer::Start("update_stream_tree/C");
  if (has_lake) this->remove_lakes(subroots);
  hmap::Timer::Stop("update_stream_tree/C");

  hmap::Timer::Start("update_stream_tree/D");
  this->children = invert_receiver_map(this->receivers);
  hmap::Timer::Stop("update_stream_tree/D");

  hmap::Timer::Start("update_stream_tree/E");
  this->update_traversals();
  hmap::Timer::Stop("update_stream_tree/E");
}

void DrainageBasin::update_traversals()
{
  this->traversals.clear();

  for (const auto &o : this->get_outlets())
  {
    std::vector<size_t> traversal;
    traversal.reserve(this->xyz.size());

    traversal.push_back(o);

    size_t i = 0;
    while (i < traversal.size())
    {
      size_t node = traversal[i];

      // add upstream nodes
      for (size_t child : this->children[node])
        traversal.push_back(child);

      ++i;
    }

    std::reverse(traversal.begin(), traversal.end());
    this->traversals[o] = std::move(traversal);
  }
}

// --- functions

std::vector<size_t> find_border_minima(const std::vector<glm::vec3> &xyz,
                                       float                         eps)
{
  const size_t invalid = std::numeric_limits<size_t>::max();

  size_t xmin = invalid;
  size_t xmax = invalid;
  size_t ymin = invalid;
  size_t ymax = invalid;

  float zxmin = std::numeric_limits<float>::max();
  float zxmax = std::numeric_limits<float>::max();
  float zymin = std::numeric_limits<float>::max();
  float zymax = std::numeric_limits<float>::max();

  for (size_t i = 0; i < xyz.size(); ++i)
  {
    const auto &p = xyz[i];

    if (std::abs(p.x) < eps)
    {
      if (p.z < zxmin)
      {
        zxmin = p.z;
        xmin = i;
      }
    }

    if (std::abs(p.x - 1.f) < eps)
    {
      if (p.z < zxmax)
      {
        zxmax = p.z;
        xmax = i;
      }
    }

    if (std::abs(p.y) < eps)
    {
      if (p.z < zymin)
      {
        zymin = p.z;
        ymin = i;
      }
    }

    if (std::abs(p.y - 1.f) < eps)
    {
      if (p.z < zymax)
      {
        zymax = p.z;
        ymax = i;
      }
    }
  }

  return {xmin, xmax, ymin, ymax};
}

std::vector<size_t> sample_border_points(const std::vector<glm::vec3> &xyz,
                                         size_t                        nb)
{
  std::vector<size_t> indices;
  indices.reserve(nb);

  if (xyz.empty() || nb == 0) return indices;

  float perimeter = 4.f;
  float step = perimeter / float(nb);

  for (size_t k = 0; k < nb; ++k)
  {
    float s = k * step;

    float x, y;

    if (s < 1.f) // bottom edge
    {
      x = s;
      y = 0.f;
    }
    else if (s < 2.f) // right edge
    {
      x = 1.f;
      y = s - 1.f;
    }
    else if (s < 3.f) // top edge
    {
      x = 3.f - s;
      y = 1.f;
    }
    else // left edge
    {
      x = 0.f;
      y = 4.f - s;
    }

    float  best_d2 = std::numeric_limits<float>::max();
    size_t best_i = 0;

    for (size_t i = 0; i < xyz.size(); ++i)
    {
      float dx = xyz[i].x - x;
      float dy = xyz[i].y - y;
      float d2 = dx * dx + dy * dy;

      if (d2 < best_d2)
      {
        best_d2 = d2;
        best_i = i;
      }
    }

    indices.push_back(best_i);
  }

  return indices;
}

std::vector<glm::vec3> heightmap_retopology(const Array &z,
                                            float        max_error,
                                            int          max_triangles,
                                            int          max_points)
{
  const glm::ivec2 &shape = z.shape;

  const auto   p_hmap = std::make_shared<Heightmap>(shape.y,
                                                  shape.x,
                                                  z.get_vector());
  Triangulator tri(p_hmap);
  tri.Run(max_error, max_triangles, max_points);

  const auto &points = tri.Points(1.f);

  // x, y normalization coefficients
  const float ax = 1.f / float(shape.y - 1);
  const float ay = 1.f / float(shape.x - 1);

  std::vector<glm::vec3> xyz;
  xyz.reserve(points.size());

  for (const auto &p : points)
    xyz.push_back({ay * p.y, ax * p.x, p.z});

  return xyz;
}

std::vector<std::vector<size_t>> invert_receiver_map(
    const std::vector<size_t> &receivers)
{
  const size_t n = receivers.size();

  std::vector<std::vector<size_t>> children(n);

  for (size_t v = 0; v < n; ++v)
  {
    size_t r = receivers[v];
    if (r != v) children[r].push_back(v);
  }

  return children;
}

} // namespace hmap
