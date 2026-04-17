/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "macrologger.h"

#include "highmap/geometry/path.hpp"

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

} // namespace hmap
