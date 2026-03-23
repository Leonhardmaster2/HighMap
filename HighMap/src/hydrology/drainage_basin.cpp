/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <cstdio>
#include <fstream>
#include <limits>
#include <queue>

#include "hmm/src/heightmap.h"
#include "hmm/src/triangulator.h"

#include "delaunator-cpp.hpp"

#include "macrologger.h"

#include "highmap/hydrology/drainage_basin.hpp"
#include "highmap/math.hpp"
#include "highmap/random.hpp"

namespace hmap
{

DrainageBasin::DrainageBasin(std::vector<glm::vec3> xyz)
{
  this->mesh = TerrainTriMesh(xyz);

  // outlets == convex hull by default
  this->outlets_mask.resize(this->mesh.size());

  for (const auto &idx : this->mesh.get_convex_hull())
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
  this->receivers.resize(this->mesh.size());

  const TerrainTriMesh::NeighborData &nbrs_data = this->mesh.get_neighbors();

  for (size_t k = 0; k < this->mesh.size(); ++k)
  {
    if (this->outlets_mask[k])
    {
      this->receivers[k] = k; // mark as self-receiver
      continue;
    }

    float  best_slope = 0.f;
    size_t best_k = k;

    for (const auto &nb : nbrs_data.adjacency[k])
    {
      size_t n = nb.index;
      float  dist = nb.distance2d;
      float  dz = this->mesh.get_points()[k].z - this->mesh.get_points()[n].z;

      if (dz > 0.f)
      {
        float slope = dz / dist;
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

void DrainageBasin::compute_receivers(unsigned int seed, float noise_strength)
{
  this->receivers.clear();
  this->receivers.resize(this->mesh.size());

  const TerrainTriMesh::NeighborData &nbrs_data = this->mesh.get_neighbors();
  const auto                         &points = this->mesh.get_points();

  for (size_t k = 0; k < this->mesh.size(); ++k)
  {
    if (this->outlets_mask[k])
    {
      this->receivers[k] = k; // mark as self-receiver
      continue;
    }

    float  best_score = -std::numeric_limits<float>::infinity();
    size_t best_k = k;

    for (const auto &nb : nbrs_data.adjacency[k])
    {
      float dz = points[k].z - points[nb.index].z;

      if (dz > 0.f)
      {
        float slope = dz / nb.distance2d;

        // use deterministic noise in [-1, 1) to bias slope with noise
        // perturbation
        float noise = 2.f * hash_to_unit_float(seed, k ^ nb.index) - 1.f;
        float score = slope * (1.f + noise_strength * noise);

        if (score > best_score)
        {
          best_score = score;
          best_k = nb.index;
        }
      }
    }

    // float  best_slope = 0.f;
    // size_t best_k = k;

    // for (const auto &nb : nbrs_data.adjacency[k])
    // {
    //   size_t n = nb.index;
    //   float  dist = nb.distance2d;
    //   float  dz = this->mesh.get_points()[k].z -
    //   this->mesh.get_points()[n].z;

    //   if (dz > 0.f)
    //   {
    //     float slope = dz / dist;
    //     if (slope > best_slope)
    //     {
    //       best_slope = slope;
    //       best_k = n;
    //     }
    //   }

    this->receivers[k] = best_k;
  }

  // for (size_t k = 0; k < this->mesh.size(); ++k)
  // {
  //   if (this->outlets_mask[k])
  //   {
  //     this->receivers[k] = k;
  //     continue;
  //   }

  //   float total_weight = 0.f;

  //   // first pass - compute total weight
  //   for (const auto &nb : nbrs_data.adjacency[k])
  //   {
  //     float dz = points[k].z - points[nb.index].z;
  //     if (dz > 0.f) total_weight += dz / nb.distance2d;
  //   }

  //   if (total_weight <= 0.f)
  //   {
  //     this->receivers[k] = k;
  //     continue;
  //   }

  //   // second pass - pick the best (with a random bias)

  //   // deterministic random in [0, total_weight]
  //   float  r = hash_to_unit_float(seed, k);
  //   r = 1.f * total_weight;

  //   float  cumulative = 0.f;
  //   size_t chosen = k;

  //   for (const auto &nb : nbrs_data.adjacency[k])
  //   {
  //     float dz = points[k].z - points[nb.index].z;

  //     if (dz > 0.f)
  //     {
  //       float weight = dz / nb.distance2d;
  //       cumulative += weight;

  //       if (r >= cumulative)
  //       {
  //         chosen = nb.index;
  //         break;
  //       }
  //     }
  //   }

  //   this->receivers[k] = chosen;
  // }
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

  glm::vec3 dref = this->mesh.get_reference_lengths();

  for (const auto &[outlet, traversal] : traversals)
  {
    // downstream => upstream
    for (auto it = traversal.rbegin(); it != traversal.rend(); ++it)
    {
      size_t i = *it;
      size_t j = this->receivers[i];

      if (j != i)
      {
        const auto &pi = this->mesh.get_points()[i];
        const auto &pj = this->mesh.get_points()[j];
        float       vx = (pi.x - pj.x) / dref.x;
        float       vy = (pi.y - pj.y) / dref.y;
        float       distance = glm::length(glm::vec2(vx, vy));

        float celerity = erodibility[i] *
                         std::pow(std::max(area_acc[i], 1e-8f), m_exp);

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
  std::vector<float> areas(this->mesh.size());

  // accumulate 1/3 area of the triangles to each vertex
  for (const auto &idx : this->mesh.get_triangles())
  {
    const auto &p0 = this->mesh.get_points()[idx.a];
    const auto &p1 = this->mesh.get_points()[idx.b];
    const auto &p2 = this->mesh.get_points()[idx.c];

    // edge vectors
    auto e0 = p1 - p0;
    auto e1 = p2 - p0;

    // triangle area
    float triangle_area = 0.5f * glm::length(glm::cross(e0, e1));
    float vertex_contribution = triangle_area / 3.f;

    areas[idx.a] += vertex_contribution;
    areas[idx.b] += vertex_contribution;
    areas[idx.c] += vertex_contribution;
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

const TerrainTriMesh &DrainageBasin::get_mesh() const
{
  return this->mesh;
}

TerrainTriMesh &DrainageBasin::get_mesh()
{
  return this->mesh;
}

std::vector<size_t> DrainageBasin::get_outlets() const
{
  std::vector<size_t> outlet_indices;

  for (size_t k = 0; k < this->mesh.size(); ++k)
    if (this->outlets_mask[k]) outlet_indices.push_back(k);

  return outlet_indices;
}

const std::vector<size_t> &DrainageBasin::get_receivers() const
{
  return this->receivers;
}

const std::vector<glm::vec3> &DrainageBasin::get_xyz() const
{
  return this->mesh.get_points();
}

void DrainageBasin::remap(float vmin, float vmax)
{
  if (this->mesh.get_points().empty()) return;

  float zmin = this->mesh.get_points()[0].z;
  float zmax = this->mesh.get_points()[0].z;

  for (const auto &p : this->mesh.get_points())
  {
    zmin = std::min(zmin, p.z);
    zmax = std::max(zmax, p.z);
  }

  float dz = zmax - zmin;

  if (dz == 0.f)
  {
    float mid = 0.5f * (vmin + vmax);
    for (auto &p : this->mesh.get_points())
      p.z = mid;
    return;
  }

  float scale = (vmax - vmin) / dz;

  for (auto &p : this->mesh.get_points())
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

  const size_t                *subroot_ptr = subroot.data();
  TerrainTriMesh::NeighborData nbrs_data = this->mesh.get_neighbors();

  while (!heap.empty())
  {
    HeapNode top = heap.top();
    heap.pop();

    size_t i = top.node;
    if (visited[i]) continue;

    visited[i] = 1;

    auto &nbrs = nbrs_data.adjacency[i];

    for (const auto &nb : nbrs)
    {
      size_t j = nb.index;
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

      float distance = glm::length(mesh.get_points()[i] - mesh.get_points()[j]);
      float new_dist = top.dist + distance;

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
  return this->mesh.get_points().size();
};

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

  for (size_t i = 0; i < static_cast<size_t>(mesh.get_points().size()); ++i)
  {
    size_t r = (i < receivers.size()) ? receivers[i] : -1;
    size_t rt = (i < roots.size()) ? roots[i] : -1;

    f << i << "," << mesh.get_points()[i].x << "," << mesh.get_points()[i].y
      << "," << mesh.get_points()[i].z << "," << (outlets_mask[i] ? 1 : 0)
      << "," << r << "," << rt << "," << order[i] << "," << area[i] << ","
      << acc[i] << "," << flow[i] << "\n";
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

  // get z range to scale slope limiters
  glm::vec2 zr = this->mesh.get_range_z();
  float     zptp = zr.y - zr.x;

  // iterate upstream (outlet first)
  std::vector<glm::vec3> &points = this->mesh.get_points();

  for (const auto &[outlet, traversal] : traversals)
  {
    for (size_t i : traversal)
    {
      size_t j = this->receivers[i];

      if (j == i) continue; // skip outlet

      float dt = std::max(response_times[i] - response_times[outlet], 0.f);
      float new_elevation = points[outlet].z + uplift_rate * dt;

      // slope limiting
      {
        // xy-plane distance only for splope
        float distance = std::hypot(points[i].x - points[j].x,
                                    points[i].y - points[j].y);

        float slope = (new_elevation - points[j].z) / distance;
        float max_slope_n = max_slope[i] * zptp;

        // thermal erosion
        if (slope > max_slope_n)
          new_elevation = points[j].z + max_slope_n * distance;
      }

      float delta = std::abs(new_elevation - points[i].z);
      delta_sum += delta;

      points[i].z = new_elevation;
    }
  }

  delta_sum /= float(this->size());

  return delta_sum;
}

void DrainageBasin::update_stream_tree(unsigned int seed, float noise_strength)
{
  this->compute_receivers(seed, noise_strength);

  auto [subroots, has_lake] = this->find_subroots();
  if (has_lake) this->remove_lakes(subroots);

  this->children = invert_receiver_map(this->receivers);
  this->update_traversals();
}

void DrainageBasin::update_stream_tree()
{
  // deactivate noise for receivers
  float noise_strength = 0.f;
  uint  seed = 0; // dummy value

  this->update_stream_tree(seed, noise_strength);
}

void DrainageBasin::update_traversals()
{
  this->traversals.clear();

  for (const auto &o : this->get_outlets())
  {
    std::vector<size_t> traversal;
    traversal.reserve(this->mesh.get_points().size());

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

std::vector<size_t> find_border_sinks(TerrainTriMesh &mesh, float eps)
{
  const auto &pts = mesh.get_points();
  const auto &nbrs_data = mesh.get_neighbors();
  const auto &bbox = mesh.get_bbox();

  std::vector<size_t> sinks;

  for (size_t i = 0; i < pts.size(); ++i)
  {
    const auto &p = pts[i];

    // border test
    bool is_border = (std::abs(p.x - bbox.min.x) < eps) ||
                     (std::abs(p.x - bbox.max.x) < eps) ||
                     (std::abs(p.y - bbox.min.y) < eps) ||
                     (std::abs(p.y - bbox.max.y) < eps);

    if (!is_border) continue;

    const auto &nbrs = nbrs_data.adjacency[i];

    if (nbrs.empty()) continue;

    bool is_sink = true;

    for (const auto &nb : nbrs)
    {
      size_t j = nb.index;

      if (j >= pts.size()) continue;

      // --- strictly lower neighbor → NOT a sink ---
      if (pts[j].z < p.z - eps)
      {
        is_sink = false;
        break;
      }
    }

    if (is_sink) sinks.push_back(i);
  }

  return sinks;
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
