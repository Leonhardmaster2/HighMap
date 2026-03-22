/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "macrologger.h"

#include "highmap/geometry/cloud.hpp"
#include "highmap/hydrology/drainage_basin.hpp"
#include "highmap/math.hpp"
#include "highmap/primitives.hpp"
#include "highmap/range.hpp"

namespace hmap
{

void hydraulic_saleve(TerrainTriMesh           &mesh,
                      const std::vector<float> &erodibility,
                      const std::vector<float> &max_slope,
                      float                     m_exp,
                      float                     uplift_rate,
                      float                     tolerance,
                      int                       max_iterations)
{
  auto db = DrainageBasin(mesh.get_points());
  db.set_outlets(find_border_sinks(db.get_mesh()));

  for (int it = 0; it < max_iterations; ++it)
  {
    std::vector acc(db.size(), 0.f);

    db.update_stream_tree();

    auto area = db.get_mesh().get_vertex_areas(true);
    db.accumulate_area_by_outlet(area, acc);

    auto  response_times = db.compute_response_times(acc, erodibility, m_exp);
    float diff = db.update_elevations(response_times, uplift_rate, max_slope);

    if (diff < tolerance) break;
  }

  // override input with eroded field
  mesh = TerrainTriMesh(db.get_mesh());
}

Array hydraulic_saleve(const Array &z,
                       uint         seed,
                       size_t       control_points_count,
                       float        m_exp,
                       float        uplift_rate,
                       float        tolerance,
                       int          max_iterations,
                       float        smin,
                       float        smax,
                       float        strength,
                       bool         scale_erodibility_with_z,
                       float        erodibility_distrib_exp,
                       const Array *p_noise_x,
                       const Array *p_noise_y)
{
  const glm::ivec2 shape = z.shape;
  const glm::vec4  bbox = {0.f, 1.f, 0.f, 1.f};
  const float      zmin = z.min();
  const float      zmax = z.max();

  // safeguard
  smin = std::min(smin, smax);

  // --- generate triangle mesh

  Cloud cloud = random_cloud_jittered(control_points_count,
                                      {0.5f, 0.5f},
                                      {0.f, 0.f},
                                      seed,
                                      bbox);
  cloud.snap_points_to_bounding_box(bbox);
  cloud.set_values_from_array(z, bbox);
  auto mesh = TerrainTriMesh(cloud.to_vec3());

  // --- spatial parameters

  // erodibility align with elevation
  std::vector<float> erodibility;

  if (scale_erodibility_with_z && (zmin != zmax))
  {
    erodibility = cloud.get_values();

    for (auto &v : erodibility)
    {
      v = (v - zmin) / (zmax - zmin);
      v = std::pow(1.f - v, erodibility_distrib_exp);
    }
  }
  else
  {
    erodibility = std::vector<float>(control_points_count, 1.f);
  }

  // slope varying with distance to the boundary
  std::vector<float> max_slope = cubic_pulse(mesh);

  for (auto &v : max_slope)
    v = (smax - smin) * v + smin;

  // --- erode the triangle mesh

  hydraulic_saleve(mesh,
                   erodibility,
                   max_slope,
                   m_exp,
                   uplift_rate,
                   tolerance,
                   max_iterations);

  // --- interpolate back to an heightmap

  // make sure the input noise displacement do not modifiy the convex
  // hull limits to avoid issues with the nautral neighbor
  // interpolation
  Array dx;
  Array dy;

  if (p_noise_x)
  {
    dx = (*p_noise_x) * biquad_pulse_x(shape);
    p_noise_x = &dx;
  }

  if (p_noise_y)
  {
    dy = (*p_noise_y) * biquad_pulse_y(shape);
    p_noise_y = &dy;
  }

  // interpolate
  std::vector<float> xc, yc, zc;
  for (const auto &p : mesh.get_points())
  {
    xc.push_back(p.x);
    yc.push_back(p.y);
    zc.push_back(p.z);
  }

  Array ze = interpolate2d(shape,
                           xc,
                           yc,
                           zc,
                           InterpolationMethod2D::ITP2D_NNI,
                           p_noise_x,
                           p_noise_y);
  remap(ze, zmin, zmax);

  return lerp(z, ze, strength);
}

} // namespace hmap
