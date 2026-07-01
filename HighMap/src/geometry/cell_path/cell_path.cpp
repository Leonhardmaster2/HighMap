/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <utility>
#include <vector>

#include "highmap/geometry/cell_path.hpp"

namespace hmap
{

CellPath::CellPath(const std::vector<glm::ivec2> &indices) : indices(indices)
{
}

CellPath::CellPath(std::vector<glm::ivec2> &&indices)
    : indices(std::move(indices))
{
}

std::vector<glm::ivec2> &CellPath::get_indices()
{
  return this->indices;
}

const std::vector<glm::ivec2> &CellPath::get_indices() const
{
  return this->indices;
}

} // namespace hmap
