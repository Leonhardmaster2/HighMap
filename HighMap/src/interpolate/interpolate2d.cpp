/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "delaunator-cpp.hpp"
#include "macrologger.h"

#include "highmap/array.hpp"
#include "highmap/boundary.hpp"
#include "highmap/functions.hpp"
#include "highmap/geometry/grids.hpp"
#include "highmap/interpolate2d.hpp"
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

Array interpolate2d_delaunay(glm::ivec2                shape,
                             const std::vector<float> &x,
                             const std::vector<float> &y,
                             const std::vector<float> &values,
                             const Array              *p_noise_x,
                             const Array              *p_noise_y,
                             glm::vec4                 bbox)
{
  if (p_noise_x || p_noise_y) // slow
  {
    // triangulate
    std::vector<float> coords(2 * x.size());

    for (size_t k = 0; k < x.size(); k++)
    {
      coords[2 * k] = x[k];
      coords[2 * k + 1] = y[k];
    }

    delaunator::Delaunator<float> d(coords);

    // compute and store triangles area
    std::vector<float> area(d.triangles.size());

    for (size_t k = 0; k < d.triangles.size(); k += 3)
    {
      int p0 = d.triangles[k];
      int p1 = d.triangles[k + 1];
      int p2 = d.triangles[k + 2];

      // true area
      area[k] = 0.5f * (-y[p1] * x[p2] + y[p0] * (-x[p1] + x[p2]) +
                        x[p0] * (y[p1] - y[p2]) + x[p1] * y[p2]);

      // but stored like this to avoid doing it at each evaluation while
      // interpolating
      area[k] = 1.f / (2.f * area[k]);
    }

    auto itp_fct = [&x, &y, &values, &d, &area](float x_, float y_, float)
    {
      // https://stackoverflow.com/questions/2049582

      // compute barycentric coordinates to find in which triangle the
      // point (x_, y_) is inside
      for (size_t k = 0; k < d.triangles.size(); k += 3)
      {
        int p0 = d.triangles[k];
        int p1 = d.triangles[k + 1];
        int p2 = d.triangles[k + 2];

        float s = area[k] * (y[p0] * x[p2] - x[p0] * y[p2] +
                             (y[p2] - y[p0]) * x_ + (x[p0] - x[p2]) * y_);
        float t = area[k] * (x[p0] * y[p1] - y[p0] * x[p1] +
                             (y[p0] - y[p1]) * x_ + (x[p1] - x[p0]) * y_);

        if (s >= 0.f && t >= 0.f && s + t <= 1.f)
        {
          s = smoothstep5(s);
          t = smoothstep5(t);

          return values[p0] + s * (values[p1] - values[p0]) +
                 t * (values[p2] - values[p0]);
        }
      }

      return 0.f;
    };

    Array array_out = Array(shape);

    fill_array_using_xy_function(array_out,
                                 bbox,
                                 nullptr,
                                 p_noise_x,
                                 p_noise_y,
                                 nullptr,
                                 itp_fct);

    return array_out;
  }
  else
  {
    std::vector<float> coords(2 * x.size());

    for (size_t i = 0; i < x.size(); ++i)
    {
      coords[2 * i] = x[i];
      coords[2 * i + 1] = y[i];
    }

    delaunator::Delaunator<float> d(coords);

    Array out(shape);

    float dx = (bbox.y - bbox.x) / shape.x;
    float dy = (bbox.w - bbox.z) / shape.y;

    for (size_t k = 0; k < d.triangles.size(); k += 3)
    {
      int p0 = d.triangles[k];
      int p1 = d.triangles[k + 1];
      int p2 = d.triangles[k + 2];

      float x0 = x[p0];
      float y0 = y[p0];
      float x1 = x[p1];
      float y1 = y[p1];
      float x2 = x[p2];
      float y2 = y[p2];

      float v0 = values[p0];
      float v1 = values[p1];
      float v2 = values[p2];

      float area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
      if (std::abs(area) < 1e-12f) continue;

      float inv_area = 1.f / area;

      float minx = std::min({x0, x1, x2});
      float maxx = std::max({x0, x1, x2});
      float miny = std::min({y0, y1, y2});
      float maxy = std::max({y0, y1, y2});

      int ix0 = std::max(0, int((minx - bbox.x) / dx));
      int ix1 = std::min(shape.x - 1, int((maxx - bbox.x) / dx));

      int iy0 = std::max(0, int((miny - bbox.z) / dy));
      int iy1 = std::min(shape.y - 1, int((maxy - bbox.z) / dy));

      for (int j = iy0; j <= iy1; ++j)
      {
        float py = bbox.z + (j + 0.5f) * dy;

        for (int i = ix0; i <= ix1; ++i)
        {
          float px = bbox.x + (i + 0.5f) * dx;

          float w0 = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1);
          float w1 = (px - x2) * (y0 - y2) - (py - y2) * (x0 - x2);
          float w2 = (px - x0) * (y1 - y0) - (py - y0) * (x1 - x0);

          if (w0 >= 0 && w1 >= 0 && w2 >= 0)
          {
            w0 *= inv_area;
            w1 *= inv_area;
            w2 *= inv_area;

            out(i, j) = -(w0 * v0 + w1 * v1 + w2 * v2);
          }
        }
      }
    }

    out.dump();

    return out;
  }
}

Array interpolate2d_gaussian(glm::ivec2                shape,
                             const std::vector<float> &x,
                             const std::vector<float> &y,
                             const std::vector<float> &values,
                             const Array              *p_noise_x,
                             const Array              *p_noise_y,
                             glm::vec4                 bbox,
                             float                     sigma)
{
  constexpr float epsilon = 1e-10f;
  float           invs2 = 0.5f / (sigma * sigma);

  auto itp_fct = [&x, &y, &values, invs2](float x_, float y_, float)
  {
    float weighted_sum = 0.f;
    float weight_sum = 0.f;

    for (size_t k = 0; k < x.size(); ++k)
    {
      float dx = x_ - x[k];
      float dy = y_ - y[k];
      float d2 = dx * dx + dy * dy;

      // exact sample match
      if (d2 < epsilon) return values[k];

      float w = std::exp(-d2 * invs2);

      weighted_sum += w * values[k];
      weight_sum += w;
    }

    return (weight_sum > 0.f) ? (weighted_sum / weight_sum) : 0.f;
  };

  Array array_out = Array(shape);

  fill_array_using_xy_function(array_out,
                               bbox,
                               nullptr,
                               p_noise_x,
                               p_noise_y,
                               nullptr,
                               itp_fct);

  return array_out;
}

Array interpolate2d_idw(glm::ivec2                shape,
                        const std::vector<float> &x,
                        const std::vector<float> &y,
                        const std::vector<float> &values,
                        const Array              *p_noise_x,
                        const Array              *p_noise_y,
                        glm::vec4                 bbox,
                        float                     distance_exp)
{
  constexpr float epsilon = 1e-10f;

  auto itp_fct = [&x, &y, &values, distance_exp](float x_, float y_, float)
  {
    float weighted_sum = 0.f;
    float weight_sum = 0.f;

    for (size_t k = 0; k < x.size(); ++k)
    {
      float dx = x_ - x[k];
      float dy = y_ - y[k];
      float d2 = dx * dx + dy * dy;

      // exact sample match
      if (d2 < epsilon) return values[k];

      float w = 1.f / std::pow(d2, 0.5f * distance_exp);

      weighted_sum += w * values[k];
      weight_sum += w;
    }

    return (weight_sum > 0.f) ? (weighted_sum / weight_sum) : 0.f;
  };

  Array array_out = Array(shape);

  fill_array_using_xy_function(array_out,
                               bbox,
                               nullptr,
                               p_noise_x,
                               p_noise_y,
                               nullptr,
                               itp_fct);

  return array_out;
}

Array interpolate2d_nearest(glm::ivec2                shape,
                            const std::vector<float> &x,
                            const std::vector<float> &y,
                            const std::vector<float> &values,
                            const Array              *p_noise_x,
                            const Array              *p_noise_y,
                            glm::vec4                 bbox)
{
  auto itp_fct = [&x, &y, &values](float x_, float y_, float)
  {
    float dmax = std::numeric_limits<float>::max();
    float value_;
    for (size_t k = 0; k < x.size(); k++)
    {
      float d = std::hypot(x_ - x[k], y_ - y[k]);
      if (d < dmax)
      {
        dmax = d;
        value_ = values[k];
      }
    }

    return value_;
  };

  Array array_out = Array(shape);

  fill_array_using_xy_function(array_out,
                               bbox,
                               nullptr,
                               p_noise_x,
                               p_noise_y,
                               nullptr,
                               itp_fct);

  return array_out;
}

Array interpolate2d_nni(glm::ivec2                shape,
                        const std::vector<float> &x,
                        const std::vector<float> &y,
                        const std::vector<float> &values,
                        const Array              *p_noise_x,
                        const Array              *p_noise_y,
                        glm::vec4                 bbox)
{
  std::vector<float> xg, yg;
  grid_xy_vector(xg, yg, shape, bbox, false);

  std::vector<float> xout, yout;
  xout.reserve(shape.x * shape.y);
  yout.reserve(shape.x * shape.y);

  if (p_noise_x || p_noise_y)
  {
    for (int j = 0; j < shape.y; j++)
      for (int i = 0; i < shape.x; i++)
      {
        float dx = p_noise_x ? (*p_noise_x)(i, j) : 0.f;
        float dy = p_noise_y ? (*p_noise_y)(i, j) : 0.f;

        xout.push_back(xg[i] + dx);
        yout.push_back(yg[j] + dy);
      }
  }
  else
  {
    for (int j = 0; j < shape.y; j++)
      for (int i = 0; i < shape.x; i++)
      {
        xout.push_back(xg[i]);
        yout.push_back(yg[j]);
      }
  }

  NaturalNeighborInterpolator nn;
  nn.setup_output_points(xout, yout);
  nn.build(x, y);

  std::vector<float> result;
  nn.interpolate(values, result);

  Array array_out = Array(shape);

  size_t k = 0;
  for (int j = 0; j < shape.y; j++)
    for (int i = 0; i < shape.x; i++)
    {
      array_out(i, j) = result[k];
      k++;
    }

  extrapolate_borders(array_out);

  return array_out;
}

} // namespace hmap
