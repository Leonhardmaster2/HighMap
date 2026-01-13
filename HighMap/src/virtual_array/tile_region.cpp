/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <glm/glm.hpp>

#include "macrologger.h"

#include "highmap/virtual_array/tile_region.hpp"

namespace hmap
{

TileRegion::TileRegion(const TileKey    &key,
                       const glm::vec4  &bbox,
                       const glm::ivec2 &shape,
                       const glm::vec4  &halo)
    : key(key), bbox(bbox), shape(shape), halo(halo)
{
}

glm::vec2 TileRegion::cell_center(int i, int j) const
{
  const float x_min = bbox.x;
  const float x_max = bbox.y;
  const float y_min = bbox.z;
  const float y_max = bbox.w;

  const float dx = (x_max - x_min) / float(shape.x);
  const float dy = (y_max - y_min) / float(shape.y);

  float x = x_min + (i + 0.5f) * dx;
  float y = y_min + (j + 0.5f) * dy;

  return {x, y};
}

glm::vec2 TileRegion::cell_corner(int i, int j) const
{
  float x = glm::mix(bbox.x, bbox.y, float(i) / shape.x);
  float y = glm::mix(bbox.z, bbox.w, float(j) / shape.y);
  return {x, y};
}

} // namespace hmap
