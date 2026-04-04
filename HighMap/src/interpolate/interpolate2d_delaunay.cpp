/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <limits>

#include "delaunator-cpp.hpp"

#include "highmap/geometry/grids.hpp"
#include "highmap/geometry/kd_tree.hpp"
#include "highmap/operator.hpp"

namespace hmap
{

Array interpolate2d_delaunay(glm::ivec2                shape,
                             const std::vector<float> &x,
                             const std::vector<float> &y,
                             const std::vector<float> &values,
                             const Array              *p_noise_x,
                             const Array              *p_noise_y,
                             glm::vec4                 bbox)
{
  // failsafe
  if (x.size() < 3 || x.size() != y.size() || x.size() != values.size())
    return Array(shape);

  // --- build delaunay input

  std::vector<double> coords;
  coords.reserve(2 * x.size());
  for (size_t i = 0; i < x.size(); ++i)
  {
    coords.push_back(x[i]);
    coords.push_back(y[i]);
  }

  delaunator::Delaunator d(coords);

  // --- helper: neighbor triangle

  auto neighbor_triangle = [&](int halfedge) -> int
  {
    int opp = d.halfedges[halfedge];
    return (opp < 0) ? -1 : (opp / 3);
  };

  // --- helper: barycentric

  auto barycentric = [&](float  px,
                         float  py,
                         size_t p0,
                         size_t p1,
                         size_t p2,
                         float &w0,
                         float &w1,
                         float &w2) -> bool
  {
    float x0 = x[p0], y0 = y[p0];
    float x1 = x[p1], y1 = y[p1];
    float x2 = x[p2], y2 = y[p2];

    float det = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
    if (std::abs(det) < 1e-8f) return false;

    w0 = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1);
    w1 = (px - x2) * (y0 - y2) - (py - y2) * (x0 - x2);
    w2 = (px - x0) * (y1 - y0) - (py - y0) * (x1 - x0);

    const float eps = -1e-6f;
    if (w0 < eps || w1 < eps || w2 < eps) return false;

    float inv = 1.f / det;
    w0 *= inv;
    w1 *= inv;
    w2 *= inv;

    return true;
  };

  // --- helper: triangle walk

  const int max_iter = int(x.size());

  auto find_triangle = [&](float px, float py, int start_tri) -> int
  {
    int tri = start_tri;

    for (int iter = 0; iter < max_iter; ++iter)
    {
      int t = tri * 3;

      size_t p0 = d.triangles[t + 0];
      size_t p1 = d.triangles[t + 1];
      size_t p2 = d.triangles[t + 2];

      float w0, w1, w2;

      // compute signed barycentric (no rejection yet)
      float x0 = x[p0], y0 = y[p0];
      float x1 = x[p1], y1 = y[p1];
      float x2 = x[p2], y2 = y[p2];

      float det = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
      if (std::abs(det) < 1e-8f) return -1;

      w0 = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1);
      w1 = (px - x2) * (y0 - y2) - (py - y2) * (x0 - x2);
      w2 = (px - x0) * (y1 - y0) - (py - y0) * (x1 - x0);

      if (w0 >= 0 && w1 >= 0 && w2 >= 0) return tri;

      // walk across edge
      if (w0 < 0)
      {
        int n = neighbor_triangle(t + 1);
        if (n < 0) return -1;
        tri = n;
      }
      else if (w1 < 0)
      {
        int n = neighbor_triangle(t + 2);
        if (n < 0) return -1;
        tri = n;
      }
      else
      {
        int n = neighbor_triangle(t + 0);
        if (n < 0) return -1;
        tri = n;
      }
    }

    return -1;
  };

  // --- grid

  std::vector<float> xg, yg;
  grid_xy_vector(xg, yg, shape, bbox, false);

  Array out(shape);

  int last_tri = 0;

  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      float dx = p_noise_x ? (*p_noise_x)(i, j) : 0.f;
      float dy = p_noise_y ? (*p_noise_y)(i, j) : 0.f;

      float px = xg[i] + dx;
      float py = yg[j] + dy;

      int tri = find_triangle(px, py, last_tri);

      if (tri >= 0)
      {
        last_tri = tri;

        int t = tri * 3;

        size_t p0 = d.triangles[t + 0];
        size_t p1 = d.triangles[t + 1];
        size_t p2 = d.triangles[t + 2];

        float w0, w1, w2;

        if (barycentric(px, py, p0, p1, p2, w0, w1, w2))
        {
          out(i, j) = -(w0 * values[p0] + w1 * values[p1] + w2 * values[p2]);
          continue;
        }
      }

      // nearest neighbor for fallback (outside hull or failure)
      float best_d2 = std::numeric_limits<float>::max();
      float best_v = 0.f;

      for (size_t k = 0; k < x.size(); ++k)
      {
        float dx = px - x[k];
        float dy = py - y[k];
        float d2 = dx * dx + dy * dy;

        if (d2 < best_d2)
        {
          best_d2 = d2;
          best_v = values[k];
        }
      }

      out(i, j) = best_v;
    }

  return out;
}

} // namespace hmap
