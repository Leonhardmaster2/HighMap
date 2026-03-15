/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "macrologger.h"

#include "highmap/array.hpp"
#include "highmap/erosion.hpp"
#include "highmap/filters.hpp"
#include "highmap/hydrology/hydrology.hpp"
#include "highmap/range.hpp"

namespace hmap
{

void hydraulic_spl(Array &z)
{
  int                 iterations = 50;
  FlowDirectionMethod fd_method = FlowDirectionMethod::FDM_D8;
  // fd_method = FlowDirectionMethod::FDM_PRIORITY_FLOOD;
  bool  remove_lakes = true;
  float m_exp = 0.3f;
  float uplift_rate = 2.f / z.shape.x;
  bool  preserve_elevation_range = true;
  Array talus = Array(z.shape, 4.f / z.shape.x);
  Array erodability(z.shape, 1.f);

  //
  erodability = 1.f - z; // maximum(1.f - z, 0.01f);
  // erodability = 1.f - relative_elevation(z, 16);
  erodability.to_png("ero.png", Cmap::MAGMA);

  // talus = z;
  // remap(talus, 0.1f / z.shape.x, 2.f / z.shape.x);

  // ---------------------------------------------------------------------------

  // --- initialization

  const glm::ivec2        shape = z.shape;
  auto                    basins = DrainageBasinCellBased();
  const float             zmin = z.min();
  const float             zmax = z.max();
  std::vector<glm::ivec2> outlets;

  // --- SPL loop

  for (int it = 0; it < iterations; ++it)
  {
    LOG_DEBUG("%d", it);

    // --- flow routing / basin traversal

    basins.generate_traversal(z, fd_method, remove_lakes, outlets);

    // backup the initial outlets to reinforce them, in order to keep
    // the overall input structure
    if (it == 1) outlets = basins.get_outlets();

    // --- surface accumulation (downstream accumulation)

    Array area_acc(shape, 1.f);
    // basins.accumulate(area_acc);
    area_acc = flow_accumulation_dinf(z, 1.f);

    // --- response time (upstream accumulation)

    Array response_time(shape);

    auto accumulate_response_time =
        [&response_time,
         &area_acc,
         &erodability,
         m_exp](int i, int j, int i_next, int j_next, int /* basin_id */)
    {
      if (erodability(i, j) > 0.f)
      {
        float celerity = erodability(i, j) * std::pow(area_acc(i, j), m_exp);
        float dist = std::hypot(i_next - i, j_next - j);
        response_time(i, j) += response_time(i_next, j_next) + dist / celerity;
      }
      else
      {
        response_time(i, j) += response_time(i_next, j_next);
      }
    };

    basins.traverse_upstream(accumulate_response_time);

    // --- perform erosion

    auto erode_cell =
        [&response_time, &z, &basins, &talus, uplift_rate](int i,
                                                           int j,
                                                           int i_next,
                                                           int j_next,
                                                           int basin_id)
    {
      const auto &traversal = basins.upstream_traversal[basin_id];
      if (traversal.empty()) return;

      const glm::ivec2 outlet = traversal.front();

      // uplift
      float z_out = z(outlet.x, outlet.y);
      float dt = response_time(i, j) - response_time(outlet);
      float new_z = z_out + uplift_rate * dt;

      // talus limiter
      float dist = std::hypot(i_next - i, j_next - j);
      float dz = new_z - z(i_next, j_next);
      if (dz / dist > talus(i, j))
        new_z = z(i_next, j_next) + talus(i, j) * dist;

      // update
      z(i, j) = new_z;
    };

    basins.traverse_upstream(erode_cell);
  }

  // --- post-process

  z.infos();

  extrapolate_borders(z); // remove outlets with no donor
  laplace(z, 0.1f);
  if (preserve_elevation_range) remap(z, zmin, zmax);
}

} // namespace hmap
