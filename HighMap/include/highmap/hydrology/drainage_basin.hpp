/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once
#include <memory>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

#include "highmap/array.hpp"

namespace hmap
{

class DrainageBasin
{
public:
  // --- Construction

  DrainageBasin(std::vector<glm::vec3> xyz_);

  const std::vector<glm::vec3> &get_xyz() const;
  std::vector<glm::vec3>       &get_xyz();
  size_t                        size() const;
  void                          to_csv(const std::string &filename) const;

  // --- Geometry / Mesh

  std::vector<float> compute_vertex_areas() const;
  void               remap(float zmin = 0.f, float zmax = 1.f);
  void               smooth_mesh(float lambda, int iterations = 1);
  void smooth_mesh_taubin(float lambda, float mu, int iterations = 1);

  // --- Flow graph construction

  void compute_receivers();
  void update_stream_tree();
  void update_traversals();

  std::vector<size_t> get_outlets() const;
  void                set_outlets(const std::vector<size_t> &outlet_indices);
  const std::vector<size_t> &get_receivers() const;

  // --- Basin topology utilities

  std::vector<size_t>                  compute_strahler_order() const;
  std::pair<std::vector<size_t>, bool> find_subroots();
  void remove_lakes(const std::vector<size_t> &subroot);

  // --- Hydrology computations

  std::vector<float> compute_response_times(
      const std::vector<float> &area_acc,
      const std::vector<float> &erodibility,
      float                     m_exp) const;

  float update_elevations(const std::vector<float> &response_times,
                          float                     uplift_rate,
                          const std::vector<float> &max_slope);

  void accumulate_area_by_outlet(const std::vector<float> &area,
                                 std::vector<float>       &acc) const;

  // --- Traversal helpers

  const std::vector<size_t> &for_each_upstream(size_t outlet) const;

  auto for_each_downstream(size_t outlet) const
  {
    const auto &t = traversals.at(outlet);
    return std::make_pair(t.rbegin(), t.rend());
  }

private:
  // --- Geometry

  std::vector<glm::vec3>           xyz;
  std::vector<size_t>              convex_hull;
  std::vector<glm::ivec3>          triangles;
  std::vector<std::vector<size_t>> nbrs_indices;
  std::vector<std::vector<float>>  nbrs_distances;
  float                            reference_length;

  // --- Flow graph

  std::vector<size_t>              receivers;
  std::vector<size_t>              roots;
  std::vector<std::vector<size_t>> children;
  std::vector<bool>                outlets_mask;

  // --- Traversal cache

  std::unordered_map<size_t, std::vector<size_t>> traversals;

  // --- Constants

  const size_t invalid_index = size_t(-1);
};

// --- FUNCTIONS

std::vector<size_t> find_border_minima(const std::vector<glm::vec3> &xyz,
                                       float eps = 1e-6f);

std::vector<glm::vec3> heightmap_retopology(const Array &z,
                                            float        max_error,
                                            int          max_triangles = 0,
                                            int          max_points = 0);

std::vector<std::vector<size_t>> invert_receiver_map(
    const std::vector<size_t> &receivers);

std::vector<size_t> sample_border_points(const std::vector<glm::vec3> &xyz,
                                         size_t                        nb);

} // namespace hmap