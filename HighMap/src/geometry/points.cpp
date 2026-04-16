/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <cmath>
#include <iostream>

#include "highmap/geometry/point.hpp"
#include "highmap/math.hpp"

namespace hmap
{

void Point::set_value_from_array(const Array &array, glm::vec4 bbox)
{
  // scale to unit interval
  float xn = (this->x - bbox.x) / (bbox.y - bbox.x);
  float yn = (this->y - bbox.z) / (bbox.w - bbox.z);

  // scale to array shape
  xn *= (float)(array.shape.x - 1);
  yn *= (float)(array.shape.y - 1);

  int i = (int)xn;
  int j = (int)yn;

  if (i >= 0 && i < array.shape.x && j >= 0 && j < array.shape.y)
  {
    float uu = xn - (float)i;
    float vv = yn - (float)j;
    this->v = array.get_value_bilinear_at(i, j, uu, vv);
  }
  else
    this->v = 0.f; // if outside array bounding box
}

void Point::print()
{
  std::cout << "(" << this->x << ", " << this->y << ", " << this->v << ")"
            << std::endl;
}

float angle(const Point &p1, const Point &p2)
{
  return std::atan2(p2.y - p1.y, p2.x - p1.x);
}

float angle(const Point &p0, const Point &p1, const Point &p2)
{
  // Calculate vectors
  float dx1 = p1.x - p0.x;
  float dy1 = p1.y - p0.y;
  float dx2 = p2.x - p0.x;
  float dy2 = p2.y - p0.y;

  // Compute the angle using the atan2 function to get the signed angle
  float angle1 = std::atan2(dy1, dx1);
  float angle2 = std::atan2(dy2, dx2);

  // Calculate the angle difference
  float angle = angle2 - angle1;

  // Normalize the angle to be in the range [-π, π]
  if (angle > M_PI)
    angle -= 2 * M_PI;
  else if (angle < -M_PI)
    angle += 2 * M_PI;

  return angle;
}

float classify_point(const Point &p_prev,
                     const Point &p,
                     const Point &p_next,
                     const Point &pq)
{
  float c = curvature_signed(p_prev, p, p_next);
  float s = side(p_prev, p, p_next, pq);

  float v = c * s;

  if (v > 0.f) return 1.f;  // convex
  if (v < 0.f) return -1.f; // concave
  return 0.f;
}

float cross_product(const Point &p0, const Point &p1, const Point &p2)
{
  // calculate vectors v1 = p1 - p0 and v2 = p2 - p0
  float v1x = p1.x - p0.x;
  float v1y = p1.y - p0.y;
  float v2x = p2.x - p0.x;
  float v2y = p2.y - p0.y;

  // calculate the 2D cross product v1 x v2
  return v1x * v2y - v1y * v2x;
}

float cross_product(const Point &p1, const Point &p2)
{
  return p1.x * p2.y - p1.y * p2.x;
}

float curvature(const Point &p1, const Point &p2, const Point &p3)
{
  float d = distance(p1, p2) * distance(p2, p3) * distance(p1, p3);

  if (d > 0.f)
    return 4.f * triangle_area(p1, p2, p3) / d;
  else
    return 0.f;
}

float curvature_signed(const Point &p1, const Point &p2, const Point &p3)
{
  float d = distance(p1, p2) * distance(p2, p3) * distance(p1, p3);

  if (d > 0.f)
    return 4.f * triangle_area_signed(p1, p2, p3) / d;
  else
    return 0.f;
}

float distance(const Point &p1, const Point &p2)
{
  float dx = p1.x - p2.x;
  float dy = p1.y - p2.y;
  return std::hypot(dx, dy);
}

Point interp_bezier(const Point &p_start,
                    const Point &p_ctrl_start,
                    const Point &p_ctrl_end,
                    const Point &p_end,
                    float        t)
{
  // https://github.com/chen0040/cpp-spline
  Point pi = p_start;
  pi = pi +
       t * t * t *
           ((-1.f) * p_start + 3.f * p_ctrl_start - 3.f * p_ctrl_end + p_end);
  pi = pi + t * t * (3.f * p_start - 6.f * p_ctrl_start + 3.f * p_ctrl_end);
  pi = pi + t * ((-3.f) * p_start + 3.f * p_ctrl_start);
  return pi;
}

Point interp_bspline(const Point &p0,
                     const Point &p1,
                     const Point &p2,
                     const Point &p3,
                     float        t)
{
  // https://github.com/chen0040/cpp-spline
  Point pi;
  pi = t * t * t * (-1.f * p0 + 3.f * p1 - 3.f * p2 + p3) / 6.f;
  pi = pi + t * t * (3.f * p0 - 6.f * p1 + 3.f * p2) / 6.f;
  pi = pi + t * (-3.f * p0 + 3.f * p2) / 6.f;
  pi = pi + (p0 + 4.f * p1 + p2) / 6.f;
  return pi;
}

Point interp_catmullrom(const Point &p0,
                        const Point &p1,
                        const Point &p2,
                        const Point &p3,
                        float        t)
{
  // https://github.com/chen0040/cpp-spline
  Point pi = p1;
  pi = pi + t * t * t * ((-1) * p0 + 3 * p1 - 3 * p2 + p3) / 2;
  pi = pi + t * t * (2 * p0 - 5 * p1 + 4 * p2 - p3) / 2;
  pi = pi + t * ((-1) * p0 + p2) / 2;
  return pi;
}

Point interp_decasteljau(const std::vector<Point> &points, float t)
{
  if (points.size() == 1) return points[0];

  std::vector<Point> new_points;
  for (size_t i = 0; i < points.size() - 1; ++i)
    new_points.push_back(points[i] * (1 - t) + points[i + 1] * t);

  return interp_decasteljau(new_points, t);
}

glm::vec4 intersect_bounding_boxes(const glm::vec4 &bbox1,
                                   const glm::vec4 &bbox2)
{
  // Calculate the boundaries of the intersection
  float min_x = std::max(bbox1.x, bbox2.x);
  float max_x = std::min(bbox1.y, bbox2.y);
  float min_y = std::max(bbox1.z, bbox2.z);
  float max_y = std::min(bbox1.w, bbox2.w);

  // Check if there is an overlap
  if (min_x <= max_x && min_y <= max_y)
  {
    return glm::vec4{min_x, max_x, min_y, max_y};
  }

  // else return an "impossible" bounding box with xmin > xmax and ymin > ymax
  return glm::vec4(1.f, -1.f, 1.f, -1.f);
}

bool is_point_within_bounding_box(Point p, glm::vec4 bbox)
{
  return p.x >= bbox.x && p.x <= bbox.y && p.y >= bbox.z && p.y <= bbox.w;
}

bool is_point_within_bounding_box(float x, float y, glm::vec4 bbox)
{
  return x >= bbox.x && x <= bbox.y && y >= bbox.z && y <= bbox.w;
}

Point lerp(const Point &p1, const Point &p2, float t)
{
  return p1 + t * (p2 - p1);
}

Point midpoint(const Point &p1,
               const Point &p2,
               int          orientation,
               float        distance_ratio,
               float        t)
{
  // interpolated midpoint based on t
  Point pmid = lerp(p1, p2, t);

  // apply orientation (0 for random direction, 1 for inflation, -1
  // for deflation)
  distance_ratio = orientation == 0 ? distance_ratio
                                    : std::abs(distance_ratio) *
                                          std::copysign(1.0f, orientation);

  // normalize the perpendicular vector and apply the orientation and
  // distance ratio (NB - point distance is embedded in dx and dy)
  float dx = p2.x - p1.x;
  float dy = p2.y - p1.y;

  float perp_x = -dy * distance_ratio;
  float perp_y = dx * distance_ratio;

  // apply the perpendicular displacement to the midpoint
  Point displaced_midpoint(pmid.x + perp_x, pmid.y + perp_y, pmid.v);

  return displaced_midpoint;
}

std::optional<Point> segment_intersection(const Point &p1,
                                          const Point &p2,
                                          const Point &q1,
                                          const Point &q2)
{
  Point r{p2.x - p1.x, p2.y - p1.y};
  Point s{q2.x - q1.x, q2.y - q1.y};
  float rxs = cross_product(r, s);
  float qpxr = cross_product({q1.x - p1.x, q1.y - p1.y}, r);

  if (std::abs(rxs) < 1e-10) return std::nullopt; // parallel or collinear

  float t = cross_product({q1.x - p1.x, q1.y - p1.y}, s) / rxs;
  float u = qpxr / rxs;

  if (t >= 0 && t <= 1 && u >= 0 && u <= 1)
    return Point{p1.x + t * r.x, p1.y + t * r.y, p1.v + t * r.v};

  return std::nullopt; // no intersection
}

// HELPER
bool cmp_inf(Point &a, Point &b)
{
  return (a.x < b.x) | (a.x == b.x && a.y < b.y) |
         (a.x == b.x && a.y == b.y && a.v < b.v);
}

float side(const Point &p1, const Point &p2, const Point &p3, const Point &pq)
{
  // tangent direction at p2 (no need to normalize — only sign matters)
  float tx = p3.x - p1.x;
  float ty = p3.y - p1.y;

  // vector from p2 to query point
  float qx = pq.x - p2.x;
  float qy = pq.y - p2.y;

  // z-component of cross product T × Q
  float cross = tx * qy - ty * qx;

  if (cross > 0.f)
    return 1.f; // left of curve  (CCW side)
  else if (cross < 0.f)
    return -1.f; // right of curve (CW side)
  else
    return 0.f; // on the tangent line
}

void sort_points(std::vector<Point> &points)
{
  std::sort(points.begin(), points.end(), cmp_inf);
}

float triangle_area(const Point &p1, const Point &p2, const Point &p3)
{
  return std::abs(triangle_area_signed(p1, p2, p3));
}

float triangle_area_signed(const Point &p1, const Point &p2, const Point &p3)
{
  // positive = CCW, negative = CW
  return 0.5f *
         (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y));
}

glm::vec4 unit_square_bbox()
{
  return glm::vec4(0.f, 1.f, 0.f, 1.f);
}

} // namespace hmap
