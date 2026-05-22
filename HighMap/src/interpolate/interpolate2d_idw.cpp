/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <cmath>   // for pow
#include <cstddef> // for size_t
#include <vector>  // for vector

#include "highmap/array.hpp"            // for Array
#include "highmap/geometry/grids.hpp"   // for grid_xy_vector
#include "highmap/geometry/kd_tree.hpp" // for KDTreeContext
#include "highmap/math/core.hpp"        // for smoothstep3

#include "nanoflann.hpp" // for ResultItem

namespace hmap
{

Array interpolate2d_idw(glm::ivec2                shape,
                        const std::vector<float> &x,
                        const std::vector<float> &y,
                        const std::vector<float> &values,
                        const Array              *p_noise_x,
                        const Array              *p_noise_y,
                        glm::vec4                 bbox,
                        float                     distance_exp,
                        float                     radius)
{
  // failsafe
  if (x.empty() || x.size() != y.size() || values.size() != x.size() ||
      x.size() < 2)
    return Array(shape);

  // KD-tree
  KDTreeContext tree(x, y);

  // automatically set the search radius
  if (radius == 0.f)
  {
    size_t    k_nbrs = 32;
    glm::vec2 drange = tree.compte_neighbor_distance_range(k_nbrs);
    radius = 2.5f * drange.y;
  }

  const float radius2 = radius * radius;

  // interpolation base grid
  std::vector<float> xg, yg;
  grid_xy_vector(xg, yg, shape, bbox, /* endpoint */ false);

  // interpolate
  Array out(shape);

  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      float dx = p_noise_x ? (*p_noise_x)(i, j) : 0.f;
      float dy = p_noise_y ? (*p_noise_y)(i, j) : 0.f;
      float xi = xg[i] + dx;
      float yi = yg[j] + dy;

      // use a radius search to avoid hard neighbor switching
      auto matches = tree.radius_search(xi, yi, radius);

      if (matches.empty())
      {
        out(i, j) = 0.f;
        continue;
      }

      float v = 0.f;
      float sum = 0.f;

      for (const auto &res : matches)
      {
        size_t nk = res.first;
        float  dist = res.second;

        if (dist < 1e-10f)
        {
          v = values[nk];
          break;
        }

        float d = dist / radius2;

        float w = 1.f - d;
        w = smoothstep3(w);
        w /= std::pow(d, 0.5f * distance_exp);

        v += values[nk] * w;
        sum += w;
      }

      if (sum > 0.f) out(i, j) = v / sum;
    }

  return out;
}

} // namespace hmap
