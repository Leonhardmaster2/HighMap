/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/geometry/point_sampling.hpp"
#include "highmap/functions.hpp"

namespace hmap
{

size_t helper_estimate_count(const glm::vec4 &bbox, float distance)
{
  size_t count = static_cast<size_t>(2.f * (bbox.y - bbox.x) / distance *
                                     (bbox.w - bbox.z) / distance);
  return count;
}

std::array<std::pair<float, float>, 2> bbox_to_ranges2d(const glm::vec4 &bbox)
{
  std::array<std::pair<float, float>, 2> ranges = {
      std::make_pair(bbox.x, bbox.y),
      std::make_pair(bbox.z, bbox.w)};
  return ranges;
}

std::function<float(const ps::Point<float, 2> &)>
make_pointwise_function_from_array(const Array &array, const glm::vec4 &bbox)
{
  return [&array, &bbox](const ps::Point<float, 2> &p) -> float
  {
    float x = (p[0] - bbox.x) / (bbox.y - bbox.x);
    float y = (p[1] - bbox.z) / (bbox.w - bbox.z);

    x = std::clamp(x, 0.f, 1.f);
    y = std::clamp(y, 0.f, 1.f);

    float xn = x * (array.shape.x - 1);
    float yn = y * (array.shape.y - 1);

    int   i = static_cast<int>(xn);
    int   j = static_cast<int>(yn);
    float u = xn - i;
    float v = yn - j;

    return array.get_value_bilinear_at(i, j, u, v);
  };
}

std::array<std::vector<float>, 2> random_points(
    size_t                     count,
    uint                       seed,
    const PointSamplingMethod &method,
    const glm::vec4           &bbox)
{
  std::vector<ps::Point<float, 2>> points;
  auto                             ranges = bbox_to_ranges2d(bbox);

  switch (method)
  {
  case PointSamplingMethod::RND_RANDOM:
  {
    points = ps::random<float, 2>(count, ranges, seed);
  }
  break;
  //
  case PointSamplingMethod::RND_HALTON:
  {
    points = ps::halton<float, 2>(count, ranges, seed);
  }
  break;
  //
  case PointSamplingMethod::RND_HAMMERSLEY:
  {
    points = ps::hammersley<float, 2>(count, ranges, seed);
  }
  break;
  //
  case PointSamplingMethod::RND_LHS:
  {
    points = ps::latin_hypercube_sampling<float, 2>(count, ranges, seed);
  }
  break;
    //
  }

  return ps::split_by_dimension(points);
}

std::array<std::vector<float>, 2> random_points_density(size_t       count,
                                                        const Array &density,
                                                        uint         seed,
                                                        const glm::vec4 &bbox)
{
  auto ranges = bbox_to_ranges2d(bbox);
  auto density_fct = make_pointwise_function_from_array(density, bbox);

  auto points = ps::rejection_sampling<float, 2>(count,
                                                 ranges,
                                                 density_fct,
                                                 seed);
  return ps::split_by_dimension(points);
}

std::array<std::vector<float>, 2> random_points_distance(float min_dist,
                                                         uint  seed,
                                                         const glm::vec4 &bbox)
{
  auto   ranges = bbox_to_ranges2d(bbox);
  size_t count = helper_estimate_count(bbox, min_dist);

  auto points = ps::poisson_disk_sampling_uniform(count,
                                                  ranges,
                                                  min_dist,
                                                  seed);
  return ps::split_by_dimension(points);
}

std::array<std::vector<float>, 2> random_points_distance(float        min_dist,
                                                         float        max_dist,
                                                         const Array &density,
                                                         uint         seed,
                                                         const glm::vec4 &bbox)
{
  auto ranges = bbox_to_ranges2d(bbox);

  // convert density [0, 1] to scale (when density = 0, enforce
  // max_dist, when density = 1, enforce min_dist)
  Array scale = (1.f - density) * (max_dist / min_dist - 1.f) + 1.f;
  auto  scale_fct = make_pointwise_function_from_array(scale, bbox);

  // estimate a maximum count using the minimum distance
  size_t count = helper_estimate_count(bbox, min_dist);

  auto points = ps::poisson_disk_sampling<float, 2>(count,
                                                    ranges,
                                                    min_dist,
                                                    scale_fct,
                                                    seed);
  return ps::split_by_dimension(points);
}

std::array<std::vector<float>, 2> random_points_distance_power_law(
    float            dist_min,
    float            dist_max,
    float            alpha,
    uint             seed,
    const glm::vec4 &bbox)
{
  auto   ranges = bbox_to_ranges2d(bbox);
  size_t count = helper_estimate_count(bbox, dist_min);

  auto points = ps::poisson_disk_sampling_power_law<float, 2>(count,
                                                              dist_min,
                                                              dist_max,
                                                              alpha,
                                                              ranges,
                                                              seed);
  return ps::split_by_dimension(points);
}

std::array<std::vector<float>, 2> random_points_distance_weibull(
    float            dist_min,
    float            lambda,
    float            k,
    uint             seed,
    const glm::vec4 &bbox)
{
  auto   ranges = bbox_to_ranges2d(bbox);
  size_t count = helper_estimate_count(bbox, dist_min);

  auto points = ps::poisson_disk_sampling_weibull<float, 2>(count,
                                                            lambda,
                                                            k,
                                                            dist_min,
                                                            ranges,
                                                            seed);
  return ps::split_by_dimension(points);
}

std::array<std::vector<float>, 2> random_points_jittered(
    size_t           count,
    const glm::vec2 &jitter_amount,
    const glm::vec2 &stagger_ratio,
    uint             seed,
    const glm::vec4 &bbox)
{
  auto                 ranges = bbox_to_ranges2d(bbox);
  std::array<float, 2> jt = {jitter_amount.x, jitter_amount.y};
  std::array<float, 2> sr = {stagger_ratio.x, stagger_ratio.y};

  auto points = ps::jittered_grid(count, ranges, jt, sr, seed);
  return ps::split_by_dimension(points);
}

} // namespace hmap
