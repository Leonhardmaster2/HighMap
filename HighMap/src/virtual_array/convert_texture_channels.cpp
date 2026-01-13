/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "macrologger.h"

#include "highmap/virtual_array/virtual_texture.hpp"

namespace hmap
{

VirtualTexture convert_texture_channels(const VirtualTexture &src,
                                        int                   dst_channels,
                                        float                 fill_value,
                                        const ComputeMode    &cm)
{
  if (dst_channels <= 0)
    throw std::runtime_error("convert_texture_channels: invalid channel count");

  const int src_channels = src.channels();
  const int copy_channels = std::min(src_channels, dst_channels);

  // Create destination texture
  VirtualTexture dst(src.shape,
                     src.bbox,
                     src.tile_shape,
                     src.halo,
                     dst_channels,
                     src.channel(0).storage->clone());

  // Build VirtualArray pointer lists
  std::vector<VirtualArray *> src_vas;
  std::vector<VirtualArray *> dst_vas;

  for (int c = 0; c < copy_channels; ++c)
  {
    src_vas.push_back(const_cast<VirtualArray *>(&src.channel(c)));
    dst_vas.push_back(&dst.channel(c));
  }

  // Tile-wise copy
  for_each_tile(
      src_vas,
      [&](std::vector<Array *> &src_tiles, const TileRegion &region)
      {
        std::vector<Array *> dst_tiles;
        dst_tiles.reserve(copy_channels);

        for (int c = 0; c < copy_channels; ++c)
          dst_tiles.push_back(&dst_vas[c]->storage->get_tile(region));

        for (int j = 0; j < region.shape.y; ++j)
          for (int i = 0; i < region.shape.x; ++i)
            for (int c = 0; c < copy_channels; ++c)
              (*dst_tiles[c])(i, j) = (*src_tiles[c])(i, j);

        for (int c = 0; c < copy_channels; ++c)
          dst_vas[c]->storage->release_tile(region);
      },
      cm);

  // Fill extra channels
  for (int c = copy_channels; c < dst_channels; ++c)
  {
    for_each_tile(
        dst.channel(c),
        [&](Array &tile, const TileRegion &region)
        {
          for (int j = 0; j < region.shape.y; ++j)
            for (int i = 0; i < region.shape.x; ++i)
              tile(i, j) = fill_value;
        },
        cm);
  }

  return dst;
}

} // namespace hmap
