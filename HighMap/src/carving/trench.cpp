/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "macrologger.h"

#include "highmap/array.hpp"
#include "highmap/carving.hpp"
#include "highmap/filters.hpp"
#include "highmap/geometry/grids.hpp"
#include "highmap/geometry/kd_tree.hpp"
#include "highmap/geometry/path.hpp"
#include "highmap/morphology.hpp"
#include "highmap/range.hpp"

namespace hmap
{

void trench(Array                       &z,
            const Path                  &path,
            float                        width,
            bool                         enable_width_depth_scaling,
            RadialProfile                radial_profile,
            float                        radial_profile_parameter,
            ElevationLongitudinalProfile longitudinal_profile,
            float                        elevation_shift,
            float                        shift_ramp_start_ratio,
            float                        shift_ramp_end_ratio,
            float                        min_slope,
            size_t                       k_neighbors,
            const Array                 *p_noise_r,
            Array                       *p_bending_mask,
            glm::vec4                    bbox)
{
  if (path.points.empty()) return;

  const glm::ivec2 &shape = z.shape;

  // path working copy
  Path   path_copy = path;
  auto  &points = path_copy.points;
  size_t npts = points.size();

  // --- Force path resolution

  {
    float lx = bbox.y - bbox.x;
    float ly = bbox.w - bbox.z;
    float dmin = std::min(lx / shape.x, ly / shape.y);
    path_copy.resample(dmin);
  }

  // --- Adjust longitudinal elevation

  // overall shift
  for (size_t k = 0; k < npts; ++k)
  {
    // shape factor
    float t = float(k) / float(npts - 1); // in [0, 1]

    // start and end ramps
    if (t < shift_ramp_start_ratio)
      t /= shift_ramp_start_ratio;
    else if (t > 1.f - shift_ramp_end_ratio)
      t = (1.f - t) / shift_ramp_end_ratio;
    else
      t = 1.f;

    t = smoothstep3(t);

    // apply
    points[k].v += t * elevation_shift;
  }

  // monotonicity
  for (size_t k = 1; k < npts; ++k)
  {
    // minimum elevation delta (use when monotonicity is enforced)
    float dz_min = min_slope *
                   glm::length(glm::vec2(points[k].x - points[k - 1].x,
                                         points[k].y - points[k - 1].y));

    switch (longitudinal_profile)
    {
    case ElevationLongitudinalProfile::ELP_FLAT:
      points[k].v = points[0].v;
      break;

    case ElevationLongitudinalProfile::ELP_DECREASING:
    {
      points[k].v = std::min(points[k].v, points[k - 1].v - dz_min);
    }
    break;

    case ElevationLongitudinalProfile::ELP_INCREASING:
      points[k].v = std::max(points[k].v, points[k - 1].v + dz_min);
      break;

    case ElevationLongitudinalProfile::ELP_UNCHANGED:
      // nothing here
      break;
    }
  }

  // --- SDF-based transform

  Array zp(shape);
  Array blending_mask(shape);

  std::vector<float> xp = path_copy.get_x();
  std::vector<float> yp = path_copy.get_y();

  KDTreeContext tree(xp, yp);

  // interpolation base grid
  std::vector<float> xg, yg;
  grid_xy_vector(xg, yg, shape, bbox, /* endpoint */ false);

  // radial profile
  auto profile_fct = get_radial_profile_function(radial_profile,
                                                 radial_profile_parameter);

  std::vector<size_t> indices;
  std::vector<float>  distances;

  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      float xi = xg[i];
      float yi = yg[j];

      tree.neighbor_search(xi, yi, k_neighbors, indices, distances);

      float value = 0.f;
      float value_mask = 0.f;
      float dr = p_noise_r ? (*p_noise_r)(i, j) : 0.f;
      float effective_width = std::max(0.f, width * (1.f + dr));

      if (enable_width_depth_scaling)
      {
        // simplify... use only 1st neighbor for various width scalings
        size_t k0 = indices[0];
        float  dz = std::abs((z(i, j) - points[k0].v) / elevation_shift);
        effective_width *= std::clamp(dz, 0.f, 1.f);
      }

      for (size_t k = 0; k < k_neighbors; ++k)
      {
        float zref = points[indices[k]].v;
        float r = std::sqrt(distances[k]) / effective_width;

        if (r >= 0.f && r <= 1.f)
        {
          float t = profile_fct(r);
          value += lerp(zref, z(i, j), t);
          value_mask += 1.f - t;
        }
        else
        {
          value += z(i, j);
        }
      }

      zp(i, j) = value / float(k_neighbors);
      blending_mask(i, j) = value_mask / float(k_neighbors);
    }

  // --- outputs

  if (p_bending_mask) *p_bending_mask = std::move(blending_mask);

  z = std::move(zp);
}

} // namespace hmap
