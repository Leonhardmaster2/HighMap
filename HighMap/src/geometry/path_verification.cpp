/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <stddef.h> // for size_t

#include <algorithm> // for min
#include <vector>    // for vector

#include "macrologger.h" // for LOG_ERROR, LOG_INFO

#include "highmap/geometry/path.hpp"  // for Path, assert_start_end_points
#include "highmap/geometry/point.hpp" // for Point, distance

#include <float.h> // for FLT_MAX

namespace hmap
{

bool assert_start_end_points(const Path &path1,
                             const Path &path2,
                             float       tol,
                             bool        verbose)
{
  if (path1.size() < 1 || path2.size() < 1)
  {
    LOG_ERROR("not enough Path points");
    return false;
  }

  const Point &p1s = path1.points.front();
  const Point &p1e = path1.points.back();
  const Point &p2s = path2.points.front();
  const Point &p2e = path2.points.back();

  float ds = distance(p1s, p2s);
  float de = distance(p1e, p2e);
  bool  assert = ds < tol && de < tol;

  if (verbose)
    LOG_INFO("ds: %f, de: %f, assert: %s", ds, de, assert ? "T" : "F");

  return assert;
}

float chamfer_distance(const Path &a, const Path &b)
{
  auto avg = [](const Path &p, const Path &q)
  {
    float sum = 0.f;
    for (auto &pa : p.points)
    {
      float min_d = FLT_MAX;
      for (auto &pb : q.points)
        min_d = std::min(min_d, distance(pa, pb));
      sum += min_d;
    }
    return sum / float(p.size());
  };

  return avg(a, b) + avg(b, a);
}

bool has_duplicates(const Path &path, float tol)
{
  const auto &pts = path.points;

  for (size_t i = 0; i < pts.size(); ++i)
    for (size_t j = i + 1; j < pts.size(); ++j)
      if (distance(pts[i], pts[j]) < tol) return true;

  return false;
}

} // namespace hmap
