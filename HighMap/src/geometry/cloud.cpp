/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "delaunator-cpp.hpp"
#include "macrologger.h"
#include "point_sampler/metrics.hpp"
#include "point_sampler/point.hpp"
#include "point_sampler/utils.hpp"

#include "highmap/array.hpp"
#include "highmap/geometry/cloud.hpp"
#include "highmap/geometry/graph.hpp"
#include "highmap/geometry/point.hpp"
#include "highmap/geometry/point_sampling.hpp"
#include "highmap/interpolate2d.hpp"
#include "highmap/math/core.hpp"
#include "highmap/operator.hpp"

namespace hmap
{

Cloud::Cloud(int npoints, std::uint32_t seed, glm::vec4 bbox)
{
  this->points.resize(npoints);
  this->randomize(seed, bbox);
};

Cloud::Cloud(const std::vector<float> &x,
             const std::vector<float> &y,
             float                     default_value)
{
  if (x.size() != y.size())
    throw std::invalid_argument("x and y vectors must be of equal size");

  this->points.reserve(x.size());

  for (size_t k = 0; k < x.size(); ++k)
    this->points.emplace_back(x[k], y[k], default_value);
}

Cloud::Cloud(const std::vector<float> &x,
             const std::vector<float> &y,
             const std::vector<float> &v)
{
  if (x.size() != y.size() || x.size() != v.size())
    throw std::invalid_argument("x, y, and v vectors must be of equal size");

  this->points.reserve(x.size());

  for (size_t k = 0; k < x.size(); ++k)
    this->points.emplace_back(x[k], y[k], v[k]);
}

Cloud::Cloud(const std::vector<glm::ivec2> &indices,
             const glm::ivec2              &shape,
             const glm::vec4               &bbox)
{
  this->points.reserve(indices.size());

  for (const auto &ij : indices)
  {
    float x = float(ij.x) / float(shape.x - 1);
    float y = float(ij.y) / float(shape.y - 1);

    x = lerp(bbox.x, bbox.y, x);
    y = lerp(bbox.z, bbox.w, y);

    this->points.emplace_back(x, y, 1.f);
  }
}

Cloud::Cloud(const std::vector<glm::vec3> &xyv)
{
  this->points.reserve(xyv.size());

  for (const auto &p : xyv)
    this->points.emplace_back(p.x, p.y, p.z);
}

void Cloud::add_point(const Point &p)
{
  this->points.push_back(p);
}

void Cloud::clear()
{
  this->points.clear();
}

bool Cloud::from_csv(const std::string &fname)
{
  std::ifstream file(fname);
  if (!file.is_open())
  {
    LOG_ERROR("Could not open file: %s", fname.c_str());
    return false;
  }

  std::vector<Point> new_points;
  std::string        line;
  size_t             line_num = 0;
  const auto         old_locale = std::locale::global(std::locale("C"));

  while (std::getline(file, line))
  {
    ++line_num;
    if (line.empty()) continue;

    std::istringstream ss(line);
    std::vector<float> values;
    std::string        token;

    while (std::getline(ss, token, ','))
    {
      try
      {
        values.push_back(std::stof(token));
      }
      catch (const std::invalid_argument &)
      {
        LOG_ERROR("Invalid number format in CSV line %zu: '%s'",
                  line_num,
                  token.c_str());
        std::locale::global(old_locale);
        return false;
      }
    }

    if (values.size() == 2)
    {
      new_points.emplace_back(values[0], values[1], 0.0f);
    }
    else if (values.size() == 3)
    {
      new_points.emplace_back(values[0], values[1], values[2]);
    }
    else
    {
      LOG_ERROR("Invalid number of values (%zu) in CSV line %zu",
                values.size(),
                line_num);
      std::locale::global(old_locale);
      return false;
    }
  }

  std::locale::global(old_locale);
  this->points = std::move(new_points);
  return true;
}

glm::vec4 Cloud::get_bbox() const
{
  if (this->size() == 0) return glm::vec4();

  std::vector<float> x = this->get_x();
  std::vector<float> y = this->get_y();
  glm::vec4          bbox;
  {
    float xmin = *std::min_element(x.begin(), x.end());
    float xmax = *std::max_element(x.begin(), x.end());
    float ymin = *std::min_element(y.begin(), y.end());
    float ymax = *std::max_element(y.begin(), y.end());
    bbox = {xmin, xmax, ymin, ymax};
  }
  return bbox;
}

Point Cloud::get_center() const
{
  if (this->size() == 0) return Point(0.f, 0.f);

  Point center = Point();
  for (auto &p : this->points)
    center = center + p;
  return center / (float)this->points.size();
}

std::vector<int> Cloud::get_convex_hull() const
{
  delaunator::Delaunator d(this->get_xy());

  std::vector<int> chull = {(int)d.hull_start};

  int inext = (int)d.hull_next[chull.back()];
  while (inext != chull[0])
  {
    chull.push_back(inext);
    inext = d.hull_next[chull.back()];
  }

  return chull;
}

std::vector<float> Cloud::get_values() const
{
  std::vector<float> values;
  values.reserve(this->size());
  for (auto &p : this->points)
    values.push_back(p.v);
  return values;
}

float Cloud::get_values_max() const
{
  if (this->size() == 0) return 0.f;

  std::vector<float> values = this->get_values();
  return *std::max_element(values.begin(), values.end());
}

float Cloud::get_values_min() const
{
  if (this->size() == 0) return 0.f;

  std::vector<float> values = this->get_values();
  return *std::min_element(values.begin(), values.end());
}

std::vector<float> Cloud::get_x() const
{
  std::vector<float> x;
  x.reserve(this->size());
  for (auto &p : this->points)
    x.push_back(p.x);
  return x;
}

std::vector<float> Cloud::get_xy() const
{
  std::vector<float> xy;
  xy.reserve(2 * this->size());
  for (auto &p : this->points)
  {
    xy.push_back(p.x);
    xy.push_back(p.y);
  }
  return xy;
}

std::vector<float> Cloud::get_y() const
{
  std::vector<float> y;
  y.reserve(this->size());
  for (auto &p : this->points)
    y.push_back(p.y);
  return y;
}

size_t Cloud::nearest_point(const glm::vec2 &xy) const
{
  size_t kn = 0;
  float  d2max = std::numeric_limits<float>::max();

  for (size_t k = 0; k < this->size(); k++)
  {
    glm::vec2 diff = xy - glm::vec2(this->points[k].x, this->points[k].y);
    float     d2 = glm::dot(diff, diff);

    if (d2 < d2max)
    {
      d2max = d2;
      kn = k;
    }
  }

  return kn;
}

void Cloud::print()
{
  std::cout << "Cloud" << std::endl;

  glm::vec4 bbox = this->get_bbox();
  Point     center = this->get_center();

  std::cout << "  - bounding box: {" << bbox.x << ", " << bbox.y << ", "
            << bbox.z << ", " << bbox.w << "}" << std::endl;

  std::cout << "  - center: {" << center.x << ", " << center.y << "}"
            << std::endl;

  std::cout << "  - points:" << std::endl;
  for (size_t k = 0; k < this->size(); k++)
  {
    std::cout << std::setw(6) << k;
    std::cout << std::setw(12) << this->points[k].x;
    std::cout << std::setw(12) << this->points[k].y;
    std::cout << std::setw(12) << this->points[k].v;
    std::cout << std::endl;
  }
}

void Cloud::randomize(std::uint32_t seed, glm::vec4 bbox)
{
  Cloud cloud_rnd = random_cloud(this->size(),
                                 seed,
                                 PointSamplingMethod::RND_LHS,
                                 bbox);

  for (size_t k = 0; k < this->size(); ++k)
    this->points[k] = cloud_rnd.points[k];
}

void Cloud::remap_values(float vmin, float vmax)
{
  const auto [current_min, current_max] = std::minmax_element(
      this->points.begin(),
      this->points.end(),
      [](const Point &a, const Point &b) { return a.v < b.v; });

  if (current_min->v == current_max->v) return;

  const float scale = (vmax - vmin) / (current_max->v - current_min->v);
  for (auto &p : this->points)
    p.v = (p.v - current_min->v) * scale + vmin;
}

void Cloud::remove_point(int point_idx)
{
  this->points.erase(this->points.begin() + point_idx);
}

void Cloud::set_points(const std::vector<float> &x, const std::vector<float> &y)
{
  if (x.size() != y.size() || x.size() != this->points.size())
    throw std::invalid_argument("New values size must match number of points");

  for (size_t k = 0; k < this->points.size(); ++k)
  {
    this->points[k].x = x[k];
    this->points[k].y = y[k];
  }
}

void Cloud::set_values(const std::vector<float> &new_values)
{
  if (new_values.size() != this->points.size())
    throw std::invalid_argument("New values size must match number of points");

  for (size_t k = 0; k < this->points.size(); ++k)
    this->points[k].v = new_values[k];
}

void Cloud::set_values(float new_value)
{
  for (size_t k = 0; k < this->points.size(); ++k)
    this->points[k].v = new_value;
}

void Cloud::set_values_from_array(const Array &array, const glm::vec4 &bbox)
{
  for (auto &p : this->points)
    p.set_value_from_array(array, bbox);
}

void Cloud::set_values_from_border_distance(const glm::vec4 &bbox)
{
  std::array<std::vector<float>, 2> xy = {this->get_x(), this->get_y()};
  std::vector<ps::Point<float, 2>>  points = ps::merge_by_dimension(xy);

  std::vector<float> dist = ps::distance_to_boundary(points,
                                                     bbox_to_ranges2d(bbox));

  this->set_values(dist);
}

void Cloud::set_values_from_chull_distance()
{
  std::vector<int> chull = this->get_convex_hull();

  for (size_t i = 0; i < this->size(); i++)
  {
    float dmax = std::numeric_limits<float>::max();
    for (size_t k = 0; k < chull.size(); k++)
    {
      float dist = distance(this->points[i], this->points[chull[k]]);
      if (dist < dmax)
      {
        dmax = dist;
        this->points[i].v = dist;
      }
    }
  }
}

void Cloud::set_values_from_min_distance()
{
  std::array<std::vector<float>, 2> xy = {this->get_x(), this->get_y()};
  std::vector<ps::Point<float, 2>>  points = ps::merge_by_dimension(xy);
  std::vector<float> dist = ps::first_neighbor_distance_squared(points);

  for (auto &v : dist)
    v = std::sqrt(v);

  this->set_values(dist);
}

size_t Cloud::size() const
{
  return this->points.size();
}

void Cloud::snap_points_to_bounding_box(const glm::vec4 &bbox,
                                        float            tolerance_ratio)
{
  if (!this->size()) return;

  // reference distance based on point density
  float lx = bbox.y - bbox.x;
  float ly = bbox.w - bbox.z;
  float dref = tolerance_ratio * std::sqrt(lx * ly / float(this->size()));

  // snap points to border if close enough
  for (auto &p : this->points)
  {
    float dl = std::abs(p.x - bbox.x);
    float dr = std::abs(p.x - bbox.y);
    float db = std::abs(p.y - bbox.z);
    float dt = std::abs(p.y - bbox.w);

    float dmin = std::min(std::min(dl, dr), std::min(db, dt));

    if (dmin > dref) continue;

    if (dmin == dl)
      p.x = bbox.x;
    else if (dmin == dr)
      p.x = bbox.y;
    else if (dmin == db)
      p.y = bbox.z;
    else
      p.y = bbox.w;
  }

  // corners
  glm::vec2 corners[4] = {{bbox.x, bbox.z},
                          {bbox.y, bbox.z},
                          {bbox.y, bbox.w},
                          {bbox.x, bbox.w}};

  // snap closest point to each corner
  for (int c = 0; c < 4; ++c)
  {
    float best_d2 = std::numeric_limits<float>::max();
    int   best_i = -1;

    for (size_t i = 0; i < this->points.size(); ++i)
    {
      glm::vec2 d = glm::vec2(this->points[i].x, this->points[i].y) -
                    corners[c];
      float d2 = glm::dot(d, d);

      if (d2 < best_d2)
      {
        best_d2 = d2;
        best_i = int(i);
      }
    }

    if (best_i >= 0)
    {
      this->points[best_i].x = corners[c].x;
      this->points[best_i].y = corners[c].y;
    }
  }
}

void Cloud::shuffle(float dx, float dy, std::uint32_t seed, float dv)
{
  std::mt19937                          gen(seed);
  std::uniform_real_distribution<float> dis(-1.f, 1.f);

  for (auto &p : this->points)
  {
    p.x += dx * dis(gen);
    p.y += dy * dis(gen);
    p.v += dv * dis(gen);
  }
}

void Cloud::to_array(Array &array, glm::vec4 bbox) const
{
  int   ni = array.shape.x;
  int   nj = array.shape.y;
  float ai = (ni - 1) / (bbox.y - bbox.x);
  float bi = -bbox.x * (ni - 1) / (bbox.y - bbox.x);
  float aj = (nj - 1) / (bbox.w - bbox.z);
  float bj = -bbox.z * (nj - 1) / (bbox.w - bbox.z);

  for (auto &p : this->points)
  {
    const int i = static_cast<int>(std::round(ai * p.x + bi));
    const int j = static_cast<int>(std::round(aj * p.y + bj));

    if ((i > -1) and (i < ni) and (j > -1) and (j < nj)) array(i, j) = p.v;
  }
}

Array Cloud::to_array(glm::ivec2 shape, glm::vec4 bbox) const
{
  Array array(shape);
  this->to_array(array, bbox);
  return array;
}

void Cloud::to_array_interp(Array                &array,
                            glm::vec4             bbox,
                            InterpolationMethod2D interpolation_method,
                            Array                *p_noise_x,
                            Array                *p_noise_y,
                            glm::vec4             bbox_array) const
{
  std::vector<float> x = this->get_x();
  std::vector<float> y = this->get_y();
  std::vector<float> v = this->get_values();

  const float     lx = bbox.y - bbox.x;
  const float     ly = bbox.w - bbox.z;
  const glm::vec4 bbox_expanded = {bbox.x - lx,
                                   bbox.y + lx,
                                   bbox.z - ly,
                                   bbox.w + ly};
  expand_points_domain_corners(x, y, v, bbox_expanded, 0.f);

  array = interpolate2d(array.shape,
                        x,
                        y,
                        v,
                        interpolation_method,
                        p_noise_x,
                        p_noise_y,
                        bbox_array);
}

void Cloud::to_csv(const std::string &fname) const
{
  std::ofstream f(fname, std::ios::out);

  if (!f.is_open()) throw std::runtime_error("Failed to open file: " + fname);

  // Use C locale for consistent number formatting
  f.imbue(std::locale("C"));
  f << std::fixed << std::setprecision(9);

  for (const auto &p : this->points)
    f << p.x << ',' << p.y << ',' << p.v << '\n';
}

Graph Cloud::to_graph_delaunay()
{
  delaunator::Delaunator d(this->get_xy());
  Graph                  graph = Graph(*this);

  for (std::size_t e = 0; e < d.triangles.size(); e++)
  {
    int i = (int)d.halfedges[e];
    if (((int)e > i) or (i == -1))
    {
      int next_he = (e % 3 == 2) ? e - 2 : e + 1;
      graph.add_edge({(int)d.triangles[e], (int)d.triangles[next_he]});
    }
  }

  return graph;
}

void Cloud::to_png(const std::string &fname,
                   int                cmap,
                   glm::vec4          bbox,
                   int                depth,
                   glm::ivec2         shape)
{
  Array array = Array(shape);
  this->to_array(array, bbox);
  array.to_png(fname, cmap, depth);
}

std::vector<glm::vec3> Cloud::to_vec3() const
{
  std::vector<glm::vec3> vec;
  vec.reserve(this->size());

  for (const auto &p : this->points)
    vec.push_back({p.x, p.y, p.v});

  return vec;
}

Cloud merge_cloud(const Cloud &cloud1, const Cloud &cloud2)
{
  std::vector<float> x1 = cloud1.get_x();
  std::vector<float> y1 = cloud1.get_y();
  std::vector<float> v1 = cloud1.get_values();

  std::vector<float> x2 = cloud2.get_x();
  std::vector<float> y2 = cloud2.get_y();
  std::vector<float> v2 = cloud2.get_values();

  x1.insert(x1.end(), x2.begin(), x2.end());
  y1.insert(y1.end(), y2.begin(), y2.end());
  v1.insert(v1.end(), v2.begin(), v2.end());

  return Cloud(x1, y1, v1);
}

Cloud merge_clouds(const std::vector<Cloud> &clouds)
{
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> v;

  // reserve total size to avoid reallocations
  std::size_t total_size = 0;
  for (const auto &cloud : clouds)
    total_size += cloud.size();

  x.reserve(total_size);
  y.reserve(total_size);
  v.reserve(total_size);

  for (const auto &cloud : clouds)
  {
    const auto &cx = cloud.get_x();
    const auto &cy = cloud.get_y();
    const auto &cv = cloud.get_values();

    x.insert(x.end(), cx.begin(), cx.end());
    y.insert(y.end(), cy.begin(), cy.end());
    v.insert(v.end(), cv.begin(), cv.end());
  }

  return Cloud(x, y, v);
}

Cloud random_cloud(size_t                     count,
                   std::uint32_t              seed,
                   const PointSamplingMethod &method,
                   const glm::vec4           &bbox)
{
  auto xy = random_points(count, seed, method, bbox);
  auto v = random_vector(0.f, 1.f, xy[0].size(), ++seed);
  return Cloud(xy[0], xy[1], v);
}

Cloud random_cloud_density(size_t           count,
                           const Array     &density,
                           std::uint32_t    seed,
                           const glm::vec4 &bbox)
{
  auto xy = random_points_density(count, density, seed, bbox);
  auto v = random_vector(0.f, 1.f, xy[0].size(), ++seed);
  return Cloud(xy[0], xy[1], v);
}

Cloud random_cloud_distance(float            min_dist,
                            std::uint32_t    seed,
                            const glm::vec4 &bbox)
{
  auto xy = random_points_distance(min_dist, seed, bbox);
  auto v = random_vector(0.f, 1.f, xy[0].size(), ++seed);
  return Cloud(xy[0], xy[1], v);
}

Cloud random_cloud_distance(float            min_dist,
                            float            max_dist,
                            const Array     &density,
                            std::uint32_t    seed,
                            const glm::vec4 &bbox)
{
  auto xy = random_points_distance(min_dist, max_dist, density, seed, bbox);
  auto v = random_vector(0.f, 1.f, xy[0].size(), ++seed);
  return Cloud(xy[0], xy[1], v);
}

Cloud random_cloud_distance_power_law(float            dist_min,
                                      float            dist_max,
                                      float            alpha,
                                      std::uint32_t    seed,
                                      const glm::vec4 &bbox)
{
  auto xy = random_points_distance_power_law(dist_min,
                                             dist_max,
                                             alpha,
                                             seed,
                                             bbox);
  auto v = random_vector(0.f, 1.f, xy[0].size(), ++seed);
  return Cloud(xy[0], xy[1], v);
}

Cloud random_cloud_distance_weibull(float            dist_min,
                                    float            lambda,
                                    float            k,
                                    std::uint32_t    seed,
                                    const glm::vec4 &bbox)
{
  auto xy = random_points_distance_weibull(dist_min, lambda, k, seed, bbox);
  auto v = random_vector(0.f, 1.f, xy[0].size(), ++seed);
  return Cloud(xy[0], xy[1], v);
}

Cloud random_cloud_jittered(size_t           count,
                            const glm::vec2 &jitter_amount,
                            const glm::vec2 &stagger_ratio,
                            std::uint32_t    seed,
                            const glm::vec4 &bbox)
{
  auto xy = random_points_jittered(count,
                                   jitter_amount,
                                   stagger_ratio,
                                   seed,
                                   bbox);
  auto v = random_vector(0.f, 1.f, xy[0].size(), ++seed);
  return Cloud(xy[0], xy[1], v);
}

} // namespace hmap
