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
  std::vector<std::vector<glm::ivec2>> upstream_traversal;
  Mat<glm::ivec2>                      next = Mat<glm::ivec2>({0, 0});
  glm::ivec2                           null_cell = glm::ivec2(-1, -1);

  size_t                               get_basins_number() const;
  std::vector<std::vector<glm::ivec2>> get_main_channels();
  std::vector<glm::ivec2>              get_outlets() const;
  std::vector<glm::ivec2>              get_ridges();
  std::vector<std::vector<glm::ivec2>> get_ridges_neighbors();

  void generate_traversal(
      const Array                   &z,
      FlowDirectionMethod            fd_method = FlowDirectionMethod::FDM_D8,
      bool                           remove_lakes = true,
      const std::vector<glm::ivec2> &outlets = {});

  void generate_traversal_d8(const Array                   &z,
                             bool                           remove_lakes = true,
                             const std::vector<glm::ivec2> &outlets = {});
  void generate_traversal_priority_flood(
      const Array                   &z,
      const std::vector<glm::ivec2> &outlets = {});

  void accumulate(Array &acc) const;
  void traverse_downstream(std::function<void(int, int, int, int, int)> op);
  void traverse_downstream(std::function<void(int, int, int)> op);
  void traverse_upstream(std::function<void(int, int, int, int, int)> op);
  void traverse_upstream(std::function<void(int, int, int)> op);

private:
  void remove_lakes_d8(const Array &z,
                       float        dz_weight = 1.f,
                       float        dz_downstream_cost_ratio = 0.f);
  void update_traversal();
};

} // namespace hmap