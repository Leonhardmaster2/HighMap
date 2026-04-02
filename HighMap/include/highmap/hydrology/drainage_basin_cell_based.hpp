/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once
#include <functional>
#include <limits>

#include "highmap/array.hpp"

namespace hmap
{

enum FlowDirectionMethod : int
{
  FDM_D8,
  FDM_PRIORITY_FLOOD
};

class DrainageBasinCellBased
{
public:
  // --- Construction ---

  DrainageBasinCellBased() = default;
  DrainageBasinCellBased(const Array &z_);

  // --- Geometry / Mesh ---

  const Array &get_z() const;

  // --- Flow graph construction ---

  void compute_receivers(unsigned int seed = 0, float noise_strength = 0.f);
  void compute_receivers_priority_flood();
  void update_stream_tree(unsigned int seed, float noise_strength);
  void update_traversals();

  std::vector<glm::ivec2> get_outlets() const;
  void set_outlets(const std::vector<glm::ivec2> &outlet_indices);

  std::vector<std::vector<glm::ivec2>> compute_upstream_traversals();

  // --- Basin topology utilities ---

  std::pair<Mat<glm::ivec2>, bool> find_subroots();
  void                             remove_lakes(const Mat<glm::ivec2> &subroot);

  std::vector<std::vector<glm::ivec2>> get_main_channels() const;

  // --- Hydrology computations ---

  Array compute_response_times(const Array &area_acc,
                               const Array &erodibility,
                               float        m_exp) const;

  float update_elevations(const Array &response_times,
                          float        uplift_rate,
                          const Array &max_slope);

  void accumulate_area_by_outlet(Array &acc) const;

  // --- Members ---

  Array z;

  Mat<int>                     outlets_mask;
  Mat<glm::ivec2>              receivers;
  Mat<std::vector<glm::ivec2>> children;
  Mat<glm::ivec2>              roots;

  std::unordered_map<glm::ivec2, std::vector<glm::ivec2>, IVec2Hash> traversals;

  const glm::ivec2 null_cell = glm::ivec2(-1, -1);

private:
  // constants
  const int   di[8] = {1, 1, 0, -1, -1, -1, 0, 1};
  const int   dj[8] = {0, -1, -1, -1, 0, 1, 1, 1};
  const float cd[8] =
      {1.f, M_SQRT1_2, 1.f, M_SQRT1_2, 1.f, M_SQRT1_2, 1.f, M_SQRT1_2};
};

Mat<std::vector<glm::ivec2>> invert_receiver_map(
    const Mat<glm::ivec2> &receivers);

} // namespace hmap