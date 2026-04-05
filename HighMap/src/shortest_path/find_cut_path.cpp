/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/geometry/path.hpp"
#include "highmap/primitives.hpp"
#include "highmap/shortest_path.hpp"

namespace hmap
{

Path find_cut_path_dijkstra(const Array   &z,
                            DomainBoundary start,
                            DomainBoundary end,
                            float          dijk_elevation_ratio,
                            float          dijk_distance_exponent,
                            float          dijk_upward_penalization,
                            uint           seed,
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
                            uint           seed,
                            int            midp_iterations,
                            float          midp_sigma,
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
  i_path = {start_pt.x, int(0.5f * (start_pt.x + end_pt.x)), end_pt.x};
  j_path = {start_pt.y, int(0.5f * (start_pt.y + end_pt.y)), end_pt.y};

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

  auto path = Path(x, y, v);

  path.fractalize(midp_iterations, seed, midp_sigma);
  path.bspline();

  return path;
}

} // namespace hmap
