/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/geometry/kd_tree.hpp"

namespace hmap
{

// --- NanoflannPointCloudAdaptor ---

NanoflannPointCloudAdaptor::NanoflannPointCloudAdaptor(
    const std::vector<float> &x_,
    const std::vector<float> &y_)
    : x(x_), y(y_)
{
}

size_t NanoflannPointCloudAdaptor::kdtree_get_point_count() const
{
  return x.size();
}

float NanoflannPointCloudAdaptor::kdtree_get_pt(const size_t idx, int dim) const
{
  if (dim == 0)
    return x[idx];
  else
    return y[idx];
}

KDTreeContext::KDTreeContext(const std::vector<float> &x_,
                             const std::vector<float> &y_)
    : x(x_),
      y(y_),
      adaptor(x, y),
      index(2, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10))
{
  index.buildIndex();
}

// --- KDTreeContext ---

glm::vec2 KDTreeContext::compte_neighbor_distance_range(
    size_t k_neighbors) const
{
  float dmin = std::numeric_limits<float>::max();
  float dmax = 0.f;

  std::vector<size_t> indices;
  std::vector<float>  distances;

  for (size_t k = 0; k < this->x.size(); ++k)
  {
    this->neighbor_search(this->x[k],
                          this->y[k],
                          k_neighbors,
                          indices,
                          distances);

    for (const auto &d : distances)
    {
      dmax = std::max(dmax, d);
      if (d > 0.f) dmin = std::min(dmin, d);
    }
  }

  return {dmin, dmax};
}

void KDTreeContext::neighbor_search(float                x_query,
                                    float                y_query,
                                    size_t               k_neighbors,
                                    std::vector<size_t> &indices,
                                    std::vector<float>  &distances) const
{
  indices.resize(k_neighbors);
  distances.resize(k_neighbors);

  nanoflann::KNNResultSet<float> result_set(k_neighbors);
  result_set.init(indices.data(), distances.data());

  float query_pt[2] = {x_query, y_query};
  this->index.findNeighbors(result_set,
                            query_pt,
                            nanoflann::SearchParameters());
}

std::vector<nanoflann::ResultItem<unsigned int, float>> KDTreeContext::
    radius_search(float x_query, float y_query, float radius) const
{
  std::vector<nanoflann::ResultItem<unsigned int, float>> matches;
  nanoflann::SearchParameters                             params;

  float query_pt[2] = {x_query, y_query};
  this->index.radiusSearch(query_pt, radius * radius, matches, params);

  return matches;
}

} // namespace hmap
