/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once
#include <memory>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

#include "highmap/array.hpp"
#include "highmap/terrain_tri_mesh.hpp"

namespace hmap
{

class DrainageBasin
{
public:
  DrainageBasin(std::vector<glm::vec3> xyz_);

  const std::vector<glm::vec3> &get_xyz() const;
  size_t                        size() const;
  void                          to_csv(const std::string &filename) const;

  // --- Geometry / Mesh ---

  const TerrainTriMesh &get_mesh() const;
  TerrainTriMesh       &get_mesh();

  std::vector<float> compute_vertex_areas() const;
  void               remap(float zmin = 0.f, float zmax = 1.f);

  // --- Flow graph construction ---

  void compute_receivers();
  void compute_receivers(unsigned int seed, float noise_strength = 0.25f);
  void update_stream_tree(unsigned int seed, float noise_strength);
  void update_stream_tree();
  void update_traversals();

  std::vector<size_t> get_outlets() const;
  void                set_outlets(const std::vector<size_t> &outlet_indices);
  const std::vector<size_t> &get_receivers() const;

  // --- Basin topology utilities ---

  std::vector<bool>                    compute_is_ridge_node() const;
  std::vector<size_t>                  compute_strahler_order() const;
  std::pair<std::vector<size_t>, bool> find_subroots();
  std::vector<std::vector<size_t>>     get_main_channels() const;
  void remove_lakes(const std::vector<size_t> &subroot);

  // --- Hydrology computations ---

  std::vector<float> compute_response_times(
      const std::vector<float> &area_acc,
      const std::vector<float> &erodibility,
      float                     m_exp) const;

  float update_elevations(const std::vector<float> &response_times,
                          float                     uplift_rate,
                          const std::vector<float> &max_slope);

  void accumulate_area_by_outlet(const std::vector<float> &area,
                                 std::vector<float>       &acc) const;

  // --- Traversal helpers ---

  const std::vector<size_t> &for_each_upstream(size_t outlet) const;

  auto for_each_downstream(size_t outlet) const
  {
    const auto &t = traversals.at(outlet);
    return std::make_pair(t.rbegin(), t.rend());
  }

private:
  // --- Geometry ---

  TerrainTriMesh mesh;

  // --- Flow graph ---

  std::vector<size_t>              receivers;
  std::vector<size_t>              roots; // basin ID
  std::vector<std::vector<size_t>> children;
  std::vector<bool>                outlets_mask;

  // --- Traversal cache ---

  std::unordered_map<size_t, std::vector<size_t>> traversals;

  // --- Constants ---

  const size_t invalid_index = size_t(-1);
};

// --- FUNCTIONS

std::vector<size_t> find_border_minima(const std::vector<glm::vec3> &xyz,
                                       float eps = 1e-6f);

std::vector<size_t> find_border_sinks(TerrainTriMesh &mesh, float eps = 1e-6f);

std::vector<glm::vec3> heightmap_retopology(const Array &z,
                                            float        max_error,
                                            int          max_triangles = 0,
                                            int          max_points = 0);

std::vector<std::vector<size_t>> invert_receiver_map(
    const std::vector<size_t> &receivers);

std::vector<size_t> sample_border_points(const std::vector<glm::vec3> &xyz,
                                         size_t                        nb);

} // namespace hmap
