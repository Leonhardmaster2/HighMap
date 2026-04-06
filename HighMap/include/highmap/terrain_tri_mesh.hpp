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
  TerrainTriMesh(const std::vector<float> &x,
                 const std::vector<float> &y,
                 const std::vector<float> &z);

  // --- Core processing ---

  // in this order...
  void triangulate_delaunay();
  void compute_neighbors();
  void compute_gradients();

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

  void slope_limiter(float max_slope, int iterations = 10, float sigma = 0.1f);
  void slope_limiter(const std::vector<float> &max_slope,
                     int                       iterations = 10,
                     float                     sigma = 0.1f);

  void subdivise();

  // --- Triangle walk and interp ---

  bool barycentric(const glm::vec2 &p,
                   size_t           i0,
                   size_t           i1,
                   size_t           i2,
                   float           &w0,
                   float           &w1,
                   float           &w2) const;

  int find_triangle(const glm::vec2 &p,
                    int              start_tri = 0,
                    bool             linear_search = false) const;
  int neighbor_triangle(int tri_index, int edge_index) const;

  float interpolate_z_linear(const glm::vec2 &p,
                             int             &last_tri,
                             float            fill_value = 0.f) const;
  float interpolate_z_linear_gradient(const glm::vec2 &p,
                                      int             &last_tri,
                                      float            fill_value = 0.f,
                                      float gradient_scaling = 1.f) const;
  float interpolate_z_nearest(const glm::vec2 &p) const;
  float interpolate_z_nearest_approx(const glm::vec2 &p,
                                     int             &last_tri,
                                     float            fill_value = 0.f) const;

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
  void        to_csv(const std::string &fname) const;

private:
  std::vector<glm::vec3> points;
  std::vector<Triangle>  triangles;
  std::vector<size_t>    halfedges;
  std::vector<size_t>    convex_hull;
  NeighborData           neighbors;
  std::vector<glm::vec2> gradients;

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