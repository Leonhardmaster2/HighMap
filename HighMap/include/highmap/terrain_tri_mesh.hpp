/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once
#include <optional>
#include <unordered_map>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "highmap/array.hpp"

namespace hmap
{

class TerrainTriMesh
{
public:
  struct BoundingBox
  {
    glm::vec2 min;
    glm::vec2 max;

    bool      contains(const glm::vec2 &p) const;
    glm::vec2 clamp(const glm::vec2 &p) const;
  };

  struct Edge
  {
    size_t v0, v1;
    Edge(size_t a, size_t b) : v0(std::min(a, b)), v1(std::max(a, b))
    {
    }
    bool operator==(const Edge &other) const
    {
      return v0 == other.v0 && v1 == other.v1;
    }
  };

  struct EdgeHash
  {
    std::size_t operator()(const Edge &e) const
    {
      return std::hash<size_t>()(e.v0) ^ (std::hash<size_t>()(e.v1) << 1);
    }
  };

  struct Triangle
  {
    size_t a, b, c;
  };

  struct Neighbor
  {
    size_t index;
    float  distance2d;
  };

  struct NeighborData
  {
    std::vector<std::vector<Neighbor>> adjacency;

    void clear()
    {
      adjacency.clear();
    }

    void resize(size_t n)
    {
      adjacency.resize(n);
    }
  };

public:
  TerrainTriMesh() = default;
  TerrainTriMesh(const std::vector<glm::vec3> &ref_points);

  // --- Core processing ---

  void triangulate_delaunay();
  void compute_neighbors();

  // --- Geometry ops ---

  void relax_xy(float lambda = 0.5f,
                int   iterations = 1,
                bool  preserve_chull = true);

  void relax_xyz(float lambda = 0.5f,
                 int   iterations = 1,
                 bool  preserve_chull = true);

  void relax_xyz_taubin(float lambda = 0.5f,
                        float mu = -0.55f,
                        int   iterations = 1,
                        bool  preserve_chull = true);

  void remap_z(float vmin = 0.f, float vmax = 1.f);

  void subdivise();

  // --- Metrics ---

  TerrainTriMesh::BoundingBox get_bbox() const;
  glm::vec2                   get_range_z() const;
  glm::vec3                   get_reference_lengths() const;
  std::vector<float>          get_vertex_areas(bool normalized) const;

  float get_reference_area_xy() const;
  float get_reference_area() const;

  size_t size() const;

  // --- Accessors ---

  const std::vector<glm::vec3> &get_points() const;
  std::vector<glm::vec3>       &get_points();
  const std::vector<Triangle>  &get_triangles() const;
  const std::vector<size_t>    &get_convex_hull() const;
  const NeighborData           &get_neighbors() const;

  // --- IO ---

  bool        export_obj(const std::string &filepath) const;
  std::string info_string() const;
  void        print_info() const;
  Array       to_array(const glm::ivec2         &shape,
                       const std::vector<float> &values = {},
                       const glm::vec4          &bbox = {0.f, 1.f, 0.f, 1.f}) const;

private:
  std::vector<glm::vec3> points;
  std::vector<Triangle>  triangles;
  std::vector<size_t>    convex_hull;
  NeighborData           neighbors;

private:
  glm::vec2 to_xy(const glm::vec3 &p) const;
};

// --- FUNCTIONS

std::vector<float> cubic_pulse(const TerrainTriMesh &mesh);

TerrainTriMesh generate_terrain_tri_mesh_from_heightmap(const Array &z,
                                                        float        max_error,
                                                        int max_triangles = 0,
                                                        int max_points = 0);

} // namespace hmap