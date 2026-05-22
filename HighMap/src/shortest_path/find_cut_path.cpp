/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <cstddef> // for size_t
#include <cstdint> // for uint32_t
#include <vector>  // for vector

#include "highmap/array.hpp"         // for Array
#include "highmap/boundary.hpp"      // for pick_boundary_cell, DomainBoundary
#include "highmap/geometry/path.hpp" // for Path
#include "highmap/shortest_path.hpp" // for find_path_dijkstra, find_path_m...

namespace hmap
{

Path find_cut_path_dijkstra(const Array   &z,
                            DomainBoundary start,
                            DomainBoundary end,
                            float          dijk_elevation_ratio,
                            float          dijk_distance_exponent,
                            float          dijk_upward_penalization,
                            std::uint32_t  seed,
                            bool           favor_boundary_center,
                            bool           favor_lower_elevation,
                            bool           favor_sinks)
{
  const int nx = z.shape.x;
  const int ny = z.shape.y;

  // --- pick start and end cells

  glm::ivec2 start_pt = pick_boundary_cell(z,
                                           start,
                                           seed,
                                           favor_boundary_center,
                                           favor_lower_elevation,
                                           favor_sinks);
  glm::ivec2 end_pt = pick_boundary_cell(z,
                                         end,
                                         seed,
                                         favor_boundary_center,
                                         favor_lower_elevation,
                                         favor_sinks);

  // --- find cut path

  std::vector<int> i_path, j_path;

  find_path_dijkstra(z,
                     glm::ivec2(start_pt.x, start_pt.y),
                     glm::ivec2(end_pt.x, end_pt.y),
                     i_path,
                     j_path,
                     dijk_elevation_ratio,
                     dijk_distance_exponent,
                     dijk_upward_penalization);

  // --- build the output path

  std::vector<float> x, y, v;
  x.reserve(i_path.size());
  y.reserve(i_path.size());
  v.reserve(i_path.size());

  for (size_t k = 0; k < i_path.size(); ++k)
  {
    x.push_back(float(i_path[k]) / float(nx - 1));
    y.push_back(float(j_path[k]) / float(ny - 1));
    v.push_back(z(i_path[k], j_path[k]));
  }

  return Path(x, y, v);
}

Path find_cut_path_midpoint(const Array   &z,
                            DomainBoundary start,
                            DomainBoundary end,
                            std::uint32_t  seed,
                            float          offset_ratio,
                            int            steps,
                            bool           favor_boundary_center,
                            bool           favor_lower_elevation,
                            bool           favor_sinks)
{
  const int nx = z.shape.x;
  const int ny = z.shape.y;

  // --- pick start and end cells

  glm::ivec2 start_pt = pick_boundary_cell(z,
                                           start,
                                           seed,
                                           favor_boundary_center,
                                           favor_lower_elevation,
                                           favor_sinks);
  glm::ivec2 end_pt = pick_boundary_cell(z,
                                         end,
                                         seed,
                                         favor_boundary_center,
                                         favor_lower_elevation,
                                         favor_sinks);

  // --- find cut path

  int max_it = 0; // => autoset by algo

  std::vector<glm::ivec2> indices = find_path_midpoint(z,
                                                       start_pt,
                                                       end_pt,
                                                       offset_ratio,
                                                       max_it,
                                                       steps);

  const size_t       npts = indices.size();
  std::vector<float> x, y, v;

  x.reserve(npts);
  y.reserve(npts);
  v.reserve(npts);

  for (const auto &p : indices)
  {
    x.push_back(float(p.x) / float(nx - 1));
    y.push_back(float(p.y) / float(ny - 1));
    v.push_back(z(p));
  }

  auto path = Path(x, y, v);
  return path;
}

} // namespace hmap
