/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/interpolate2d.hpp"
#include "highmap/array.hpp"
#include "highmap/boundary.hpp"
#include "highmap/functions.hpp"
#include "highmap/geometry/grids.hpp"
#include "highmap/operator.hpp"
#include "highmap/primitives.hpp"

namespace hmap
{

Array interpolate2d(glm::ivec2                shape,
                    const std::vector<float> &x,
                    const std::vector<float> &y,
                    const std::vector<float> &values,
                    InterpolationMethod2D     interpolation_method,
                    const Array              *p_noise_x,
                    const Array              *p_noise_y,
                    glm::vec4                 bbox)
{
  switch (interpolation_method)
  {
  case InterpolationMethod2D::ITP2D_DELAUNAY:
  {
    return interpolate2d_delaunay(shape,
                                  x,
                                  y,
                                  values,
                                  p_noise_x,
                                  p_noise_y,
                                  bbox);
  }

  case InterpolationMethod2D::ITP2D_NEAREST:
  {
    return interpolate2d_nearest(shape,
                                 x,
                                 y,
                                 values,
                                 p_noise_x,
                                 p_noise_y,
                                 bbox);
  }

  case InterpolationMethod2D::ITP2D_IDW:
  {
    return interpolate2d_idw(shape, x, y, values, p_noise_x, p_noise_y, bbox);
  }

  case InterpolationMethod2D::ITP2D_GAUSSIAN:
  {
    return interpolate2d_gaussian(shape,
                                  x,
                                  y,
                                  values,
                                  p_noise_x,
                                  p_noise_y,
                                  bbox);
  }

  case InterpolationMethod2D::ITP2D_NNI:
  {
    return interpolate2d_nni(shape, x, y, values, p_noise_x, p_noise_y, bbox);
  }

  default: throw std::runtime_error("unknown 2D interpolation method");
  }
}

} // namespace hmap
