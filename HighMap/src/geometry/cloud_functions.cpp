/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <stddef.h> // for size_t

#include <algorithm>  // for remove_if
#include <cmath>      // for sqrt
#include <functional> // for function
#include <random>     // for uniform_real_distribution
#include <vector>     // for vector

#include "highmap/array.hpp"            // for Array, uint
#include "highmap/functions.hpp"        // for make_xy_function_from_array
#include "highmap/geometry/cloud.hpp"   // for Cloud, cloud_sdf_to_array
#include "highmap/geometry/grids.hpp"   // for grid_xy_vector
#include "highmap/geometry/kd_tree.hpp" // for KDTreeContext
#include "highmap/geometry/point.hpp"   // for Point

namespace hmap
{

Array cloud_sdf_to_array(const Cloud &cloud,
                         glm::ivec2   shape,
                         glm::vec4    bbox_array,
                         const Array *p_noise_x,
                         const Array *p_noise_y)
{
  Array array(shape);

  if (cloud.size() < 1) return array;

  // --- KD-tree

  std::vector<float> x = cloud.get_x();
  std::vector<float> y = cloud.get_y();
  KDTreeContext      tree(x, y);

  // --- SDF

  // array base grid
  std::vector<float> xg, yg;
  grid_xy_vector(xg, yg, shape, bbox_array, /* endpoint */ false);

  std::vector<size_t> indices;
  std::vector<float>  distances;

  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      float dx = p_noise_x ? (*p_noise_x)(i, j) : 0.f;
      float dy = p_noise_y ? (*p_noise_y)(i, j) : 0.f;
      float xi = xg[i] + dx;
      float yi = yg[j] + dy;

      tree.neighbor_search(xi,
                           yi,
                           /* k_neighbors */ 1,
                           indices,
                           distances);

      array(i, j) = std::sqrt(distances[0]);
    }

  return array;
}

std::vector<float> interpolate_values_from_array(const Cloud &cloud,
                                                 const Array &array,
                                                 glm::vec4    bbox)
{
  const float      inv_width = 1.0f / (bbox.y - bbox.x);
  const float      inv_height = 1.0f / (bbox.w - bbox.z);
  const glm::ivec2 shape = {array.shape.x - 1, array.shape.y - 1};

  std::vector<float> values;
  values.reserve(cloud.points.size());

  for (const auto &p : cloud.points)
  {
    const float xn = (p.x - bbox.x) * inv_width;
    const float yn = (p.y - bbox.z) * inv_height;

    if (xn < 0.0f || xn > 1.0f || yn < 0.0f || yn > 1.0f)
    {
      values.push_back(0.0f);
      continue;
    }

    const float x_scaled = xn * shape.x;
    const float y_scaled = yn * shape.y;
    const int   i = static_cast<int>(x_scaled);
    const int   j = static_cast<int>(y_scaled);

    if (i >= 0 && i < array.shape.x && j >= 0 && j < array.shape.y)
    {
      const float uu = x_scaled - i;
      const float vv = y_scaled - j;
      values.push_back(array.get_value_bilinear_at(i, j, uu, vv));
    }
    else
    {
      values.push_back(0.0f);
    }
  }
  return values;
}

void rejection_filter_density(Cloud           &cloud,
                              const Array     &density_mask,
                              uint             seed,
                              const glm::vec4 &bbox)
{
  std::mt19937                          gen(seed);
  std::uniform_real_distribution<float> dis(0.f, 1.f);

  auto density_fct = make_xy_function_from_array(density_mask, bbox);

  std::remove_if(cloud.points.begin(),
                 cloud.points.end(),
                 [&](Point p)
                 {
                   float rnd = dis(gen);
                   return (rnd > density_fct(p.x, p.y));
                 });
}

} // namespace hmap
