/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/geometry/grids.hpp"
#include "highmap/geometry/kd_tree.hpp"
#include "highmap/operator.hpp"

namespace hmap
{

Array interpolate2d_nearest(glm::ivec2                shape,
                            const std::vector<float> &x,
                            const std::vector<float> &y,
                            const std::vector<float> &values,
                            const Array              *p_noise_x,
                            const Array              *p_noise_y,
                            glm::vec4                 bbox)
{
  // failsafe
  if (x.size() < 2 || x.size() != y.size() || x.size() != values.size())
    return Array(shape);

  // KD-tree
  KDTreeContext tree(x, y);
  // interpolation base grid
  std::vector<float> xg, yg;
  grid_xy_vector(xg, yg, shape, bbox, /* endpoint */ false);

  // interpolate
  Array out(shape);

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

      out(i, j) = values[indices[0]];
    }

  return out;
}

} // namespace hmap
