/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <glm/geometric.hpp>

#include "hmm/src/heightmap.h"
#include "hmm/src/triangulator.h"

#include "delaunator-cpp.hpp"

#include "macrologger.h"

#include "highmap/interpolate2d.hpp"
#include "highmap/terrain_tri_mesh.hpp"

namespace hmap
{

// ------------------------------------------------------------
// BoundingBox
// ------------------------------------------------------------

bool TerrainTriMesh::BoundingBox::contains(const glm::vec2 &p) const
{
  return (p.x >= this->min.x && p.x <= this->max.x && p.y >= this->min.y &&
          p.y <= this->max.y);
}

glm::vec2 TerrainTriMesh::BoundingBox::clamp(const glm::vec2 &p) const
{
  return {std::clamp(p.x, this->min.x, this->max.x),
          std::clamp(p.y, this->min.y, this->max.y)};
}

// ------------------------------------------------------------
// TerrainTriMesh
// ------------------------------------------------------------

TerrainTriMesh::TerrainTriMesh(const std::vector<glm::vec3> &ref_points)
    : points(ref_points)
{
  this->triangulate_delaunay();
  this->compute_neighbors();
}

void TerrainTriMesh::compute_neighbors()
{
  this->neighbors.clear();
  this->neighbors.resize(this->points.size());

  const auto &pts = this->points;

  auto add_edge = [&](size_t a, size_t b)
  {
    // check if already exists (avoid duplicates)
    for (const auto &nb : this->neighbors.adjacency[a])
    {
      if (nb.index == b) return;
    }

    float vx = pts[a].x - pts[b].x;
    float vy = pts[a].y - pts[b].y;
    float dist = glm::length(glm::vec2(vx, vy));

    this->neighbors.adjacency[a].push_back({b, dist});
    this->neighbors.adjacency[b].push_back({a, dist});
  };

  for (const auto &tri : this->triangles)
  {
    add_edge(tri.a, tri.b);
    add_edge(tri.b, tri.c);
    add_edge(tri.c, tri.a);
  }
}

bool TerrainTriMesh::export_obj(const std::string &filepath) const
{
  std::ofstream file(filepath);
  if (!file.is_open()) return false;

  // vertices
  for (const auto &p : this->points)
  {
    file << "v " << p.x << " " << p.y << " " << p.z << "\n";
  }

  // faces (OBJ is 1-based indexing)
  for (const auto &tri : this->triangles)
  {
    file << "f " << (tri.a + 1) << " " << (tri.b + 1) << " " << (tri.c + 1)
         << "\n";
  }

  return true;
}

TerrainTriMesh::BoundingBox TerrainTriMesh::get_bbox() const
{
  glm::vec3 pmin(std::numeric_limits<float>::max());
  glm::vec3 pmax(std::numeric_limits<float>::lowest());

  for (const auto &p : this->points)
  {
    pmin = glm::min(pmin, p);
    pmax = glm::max(pmax, p);
  }

  TerrainTriMesh::BoundingBox bbox;
  bbox.min = pmin;
  bbox.max = pmax;

  return bbox;
}

const std::vector<size_t> &TerrainTriMesh::get_convex_hull() const
{
  return this->convex_hull;
}

const TerrainTriMesh::NeighborData &TerrainTriMesh::get_neighbors() const
{
  if (this->neighbors.adjacency.empty())
    throw std::runtime_error(
        "TerrainTriMesh::NeighborData: neighbors data has not been "
        "initialized, make sure TerrainTriMesh::compute_neighbors has been "
        "called before-hand.");

  return this->neighbors;
}

const std::vector<glm::vec3> &TerrainTriMesh::get_points() const
{
  return this->points;
}

std::vector<glm::vec3> &TerrainTriMesh::get_points()
{
  return this->points;
}

glm::vec2 TerrainTriMesh::get_range_z() const
{
  if (this->points.empty()) return glm::vec2(0.f);

  float zmin = std::numeric_limits<float>::max();
  float zmax = std::numeric_limits<float>::lowest();

  for (const auto &p : this->points)
  {
    if (p.z < zmin) zmin = p.z;
    if (p.z > zmax) zmax = p.z;
  }

  return glm::vec2(zmin, zmax);
}

float TerrainTriMesh::get_reference_area() const
{
  float total = 0.0f;

  for (const auto &tri : this->triangles)
  {
    const glm::vec3 &p0 = this->points[tri.a];
    const glm::vec3 &p1 = this->points[tri.b];
    const glm::vec3 &p2 = this->points[tri.c];

    glm::vec3 e0 = p1 - p0;
    glm::vec3 e1 = p2 - p0;

    float area = 0.5f * glm::length(glm::cross(e0, e1));

    total += area;
  }

  return total;
}

float TerrainTriMesh::get_reference_area_xy() const
{
  float total = 0.0f;

  for (const auto &tri : this->triangles)
  {
    const glm::vec2 p0 = this->to_xy(this->points[tri.a]);
    const glm::vec2 p1 = this->to_xy(this->points[tri.b]);
    const glm::vec2 p2 = this->to_xy(this->points[tri.c]);

    float area = 0.5f * std::abs((p1.x - p0.x) * (p2.y - p0.y) -
                                 (p2.x - p0.x) * (p1.y - p0.y));

    total += area;
  }

  return total;
}

glm::vec3 TerrainTriMesh::get_reference_lengths() const
{
  glm::vec3 total(0.0f);
  int       count = 0;

  for (const auto &tri : this->triangles)
  {
    auto accumulate = [&](const glm::vec3 &a, const glm::vec3 &b)
    {
      glm::vec3 d = glm::abs(b - a);
      total += d;
      count++;
    };

    accumulate(this->points[tri.a], this->points[tri.b]);
    accumulate(this->points[tri.b], this->points[tri.c]);
    accumulate(this->points[tri.c], this->points[tri.a]);
  }

  if (count > 0) total /= static_cast<float>(count);

  return total;
}

const std::vector<TerrainTriMesh::Triangle> &TerrainTriMesh::get_triangles()
    const
{
  return this->triangles;
}

std::vector<float> TerrainTriMesh::get_vertex_areas(bool normalized) const
{
  std::vector<float> areas(this->points.size(), 0.0f);

  glm::vec3 ref(1.f);

  if (normalized) ref = this->get_reference_lengths();

  for (const auto &tri : this->triangles)
  {
    const auto &p0 = this->points[tri.a];
    const auto &p1 = this->points[tri.b];
    const auto &p2 = this->points[tri.c];

    glm::vec3 e0 = p1 - p0;
    glm::vec3 e1 = p2 - p0;

    if (normalized)
    {
      // anisotropic normalization
      e0 /= ref;
      e1 /= ref;
    }

    float triangle_area = 0.5f * glm::length(glm::cross(e0, e1));
    float contribution = triangle_area / 3.f;

    areas[tri.a] += contribution;
    areas[tri.b] += contribution;
    areas[tri.c] += contribution;
  }

  return areas;
}

std::string TerrainTriMesh::info_string() const
{
  std::ostringstream oss;

  oss << "---- TerrainTriMesh Info ----\n";

  // --- counts ---
  const size_t nb_vertices = this->points.size();
  const size_t nb_triangles = this->triangles.size();

  oss << "vertices   : " << nb_vertices << "\n";
  oss << "triangles  : " << nb_triangles << "\n";

  if (this->points.empty())
  {
    oss << "empty mesh\n";
    return oss.str();
  }

  // --- bounding box ---
  glm::vec3 pmin(std::numeric_limits<float>::max());
  glm::vec3 pmax(std::numeric_limits<float>::lowest());

  for (const auto &p : this->points)
  {
    pmin = glm::min(pmin, p);
    pmax = glm::max(pmax, p);
  }

  glm::vec3 extent = pmax - pmin;

  oss << "min        : (" << pmin.x << ", " << pmin.y << ", " << pmin.z
      << ")\n";
  oss << "max        : (" << pmax.x << ", " << pmax.y << ", " << pmax.z
      << ")\n";
  oss << "extent     : (" << extent.x << ", " << extent.y << ", " << extent.z
      << ")\n";

  // --- derived metrics ---
  if (!this->triangles.empty())
  {
    float     area_xy = this->get_reference_area_xy();
    float     area_3d = this->get_reference_area();
    glm::vec3 ref_len = this->get_reference_lengths();

    oss << "area_xy    : " << area_xy << "\n";
    oss << "area_3d    : " << area_3d << "\n";
    oss << "roughness  : " << (area_xy > 0.f ? area_3d / area_xy : 0.f) << "\n";

    oss << "ref_len    : (" << ref_len.x << ", " << ref_len.y << ", "
        << ref_len.z << ")\n";
  }

  oss << "-----------------------------\n";

  return oss.str();
}

void TerrainTriMesh::print_info() const
{
  std::cout << this->info_string();
}

void TerrainTriMesh::remap_z(float vmin, float vmax)
{
  if (this->points.empty()) return;

  // find current z range
  float zmin = std::numeric_limits<float>::max();
  float zmax = std::numeric_limits<float>::lowest();

  for (const auto &p : this->points)
  {
    zmin = std::min(zmin, p.z);
    zmax = std::max(zmax, p.z);
  }

  float dz = zmax - zmin;

  // avoid division by zero (flat terrain)
  if (dz < 1e-30f)
  {
    float zmid = 0.5f * (vmin + vmax);
    for (auto &p : this->points)
      p.z = zmid;
    return;
  }

  float scale = (vmax - vmin) / dz;

  // remap
  for (auto &p : this->points)
    p.z = vmin + (p.z - zmin) * scale;
}

void TerrainTriMesh::relax_xy(float lambda, int iterations, bool preserve_chull)
{
  if (this->neighbors.adjacency.empty()) this->compute_neighbors();

  for (int it = 0; it < iterations; ++it)
  {
    std::vector<glm::vec3> new_points = this->points;

    for (size_t i = 0; i < this->points.size(); ++i)
    {
      const auto &nbrs = this->neighbors.adjacency[i];
      if (nbrs.empty()) continue;

      glm::vec2 avg(0.f);

      for (const auto &nb : nbrs)
        avg += this->to_xy(this->points[nb.index]);

      avg /= static_cast<float>(nbrs.size());

      glm::vec2 current = this->to_xy(this->points[i]);
      glm::vec2 relaxed = glm::mix(current, avg, lambda);

      new_points[i].x = relaxed.x;
      new_points[i].y = relaxed.y;
    }

    // preserve convex hull
    if (preserve_chull)
    {
      for (const auto &i : this->convex_hull)
      {
        new_points[i].x = points[i].x;
        new_points[i].y = points[i].y;
      }
    }

    this->points = std::move(new_points);
  }
}

void TerrainTriMesh::relax_xyz(float lambda,
                               int   iterations,
                               bool  preserve_chull)
{
  if (this->neighbors.adjacency.empty()) this->compute_neighbors();

  for (int it = 0; it < iterations; ++it)
  {
    std::vector<glm::vec3> buffer(this->points.size());

    for (size_t i = 0; i < this->points.size(); ++i)
    {
      const auto &nbrs = this->neighbors.adjacency[i];

      if (nbrs.empty())
      {
        buffer[i] = this->points[i];
        continue;
      }

      glm::vec3 sum(0.f);
      float     sum_w = 0.f;

      const glm::vec3 &pi = this->points[i];

      for (const auto &nb : nbrs)
      {
        size_t           j = nb.index;
        const glm::vec3 &pj = this->points[j];

        float dist = glm::length(pi - pj);

        if (dist > 1e-30f)
        {
          float w = 1.f / dist;
          sum += w * pj;
          sum_w += w;
        }
      }

      if (sum_w > 0.f)
      {
        glm::vec3 avg = sum / sum_w;
        buffer[i] = glm::mix(pi, avg, lambda);
      }
      else
      {
        buffer[i] = pi;
      }
    }

    // preserve convex hull
    if (preserve_chull)
    {
      for (size_t i : this->get_convex_hull())
        buffer[i] = this->points[i];
    }

    this->points = std::move(buffer);
  }
}

void TerrainTriMesh::relax_xyz_taubin(float lambda,
                                      float mu,
                                      int   iterations,
                                      bool  preserve_chull)
{
  for (int it = 0; it < iterations; ++it)
  {
    int sub_iterations = 1;
    this->relax_xyz(lambda, sub_iterations, preserve_chull);
    this->relax_xyz(mu, sub_iterations, preserve_chull);
  }
}

size_t TerrainTriMesh::size() const
{
  return this->points.size();
}

void TerrainTriMesh::slope_limiter(float max_slope, int iterations, float sigma)
{
  if (this->neighbors.adjacency.empty()) this->compute_neighbors();

  for (int it = 0; it < iterations; ++it)
  {
    std::vector<glm::vec3> buffer(this->points.size());

    for (size_t i = 0; i < this->points.size(); ++i)
    {
      const auto      &nbrs = this->neighbors.adjacency[i];
      const glm::vec3 &pi = this->points[i];

      buffer[i] = pi;

      if (nbrs.empty()) continue;

      for (const auto &nb : nbrs)
      {
        size_t           j = nb.index;
        const glm::vec3 &pj = this->points[j];

        float dist = glm::length(glm::vec2(pi.x - pj.x, pi.y - pj.y));
        float dz = pi.z - pj.z;
        float slope = std::abs(dz) / dist;

        if (slope > max_slope) buffer[i].z -= sigma * dz;
      }
    }

    this->points = std::move(buffer);
  }
}

void TerrainTriMesh::subdivise()
{
  std::unordered_map<Edge, size_t, EdgeHash> midpoint_map;
  std::vector<Triangle>                      new_triangles;

  for (const auto &tri : triangles)
  {
    size_t a = tri.a;
    size_t b = tri.b;
    size_t c = tri.c;

    auto get_midpoint = [&](size_t i0, size_t i1) -> size_t
    {
      Edge e(i0, i1);
      auto it = midpoint_map.find(e);
      if (it != midpoint_map.end()) return it->second;

      glm::vec3 mid = 0.5f * (points[i0] + points[i1]);
      size_t    idx = points.size();
      points.push_back(mid);
      midpoint_map[e] = idx;
      return idx;
    };

    size_t ab = get_midpoint(a, b);
    size_t bc = get_midpoint(b, c);
    size_t ca = get_midpoint(c, a);

    // Create 4 new triangles
    new_triangles.push_back({a, ab, ca});
    new_triangles.push_back({b, bc, ab});
    new_triangles.push_back({c, ca, bc});
    new_triangles.push_back({ab, bc, ca});
  }

  triangles = std::move(new_triangles);
}

Array TerrainTriMesh::to_array(const glm::ivec2         &shape,
                               const std::vector<float> &values,
                               const glm::vec4          &bbox) const
{
  // prepare data
  std::vector<float> xc, yc, zc;
  const size_t       np = this->points.size();

  if (values.size() != np)
  {
    xc.reserve(np);
    yc.reserve(np);
    zc.reserve(np);

    for (const auto &p : this->points)
    {
      xc.push_back(p.x);
      yc.push_back(p.y);
      zc.push_back(p.z);
    }
  }
  else
  {
    xc.resize(np);
    yc.resize(np);
    zc.resize(np);

    for (size_t k = 0; k < np; ++k)
    {
      xc[k] = this->points[k].x;
      yc[k] = this->points[k].y;
      zc[k] = values[k];
    }
  }

  // interpolate
  Array out = hmap::interpolate2d(shape,
                                  xc,
                                  yc,
                                  zc,
                                  hmap::InterpolationMethod2D::ITP2D_DELAUNAY,
                                  nullptr,
                                  nullptr,
                                  bbox);
  return out;
}

glm::vec2 TerrainTriMesh::to_xy(const glm::vec3 &p) const
{
  return {p.x, p.y};
}

void TerrainTriMesh::triangulate_delaunay()
{
  // generate Delaunay triangulation
  std::vector<float> xy;
  xy.reserve(2 * this->points.size());

  for (const auto &p : this->points)
  {
    xy.push_back(p.x);
    xy.push_back(p.y);
  }
  delaunator::Delaunator d(xy);

  // triangles
  {
    const std::vector<size_t> &tri = d.triangles;
    this->triangles.clear();
    this->triangles.reserve(tri.size() / 3);

    for (std::size_t k = 0; k < tri.size(); k += 3)
      this->triangles.push_back({tri[k], tri[k + 1], tri[k + 2]});
  }

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
}

// ------------------------------------------------------------
// FUNCTIONS
// ------------------------------------------------------------

std::vector<float> cubic_pulse(const TerrainTriMesh &mesh)
{
  const TerrainTriMesh::BoundingBox &bbox = mesh.get_bbox();

  // center and radius
  float xc = 0.5f * (bbox.max.x + bbox.min.x);
  float yc = 0.5f * (bbox.max.y + bbox.min.y);

  float lx = bbox.max.x - bbox.min.x;
  float ly = bbox.max.y - bbox.min.y;
  float rc = std::min(lx, ly);

  std::vector<float> values;
  values.reserve(mesh.size());

  for (auto &p : mesh.get_points())
  {
    float r = std::hypot(p.x - xc, p.y - yc) / rc;

    if (r < 1.f)
      r = 1.f - r * r * (3.f - 2.f * r);
    else
      r = 0.f;

    values.push_back(r);
  }

  return values;
}

TerrainTriMesh generate_terrain_tri_mesh_from_heightmap(const Array &z,
                                                        float        max_error,
                                                        int max_triangles,
                                                        int max_points)
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

  return TerrainTriMesh(xyz);
}

} // namespace hmap
