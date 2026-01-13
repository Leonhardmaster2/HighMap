/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
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

} // namespace hmap
