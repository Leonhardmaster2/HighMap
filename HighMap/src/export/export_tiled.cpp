/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "macrologger.h"

#include "highmap/array.hpp"
#include "highmap/internal/string_utils.hpp"
#include "highmap/math.hpp"

namespace hmap
{

glm::ivec4 helper_compute_tile_bounds(int               tile_x,
                                      int               tile_y,
                                      const glm::ivec2 &tile_count,
                                      const glm::ivec2 &array_shape,
                                      bool              overlapping_edges,
                                      bool              reverse_tile_y_indexing)
{
  const int tile_width = ceil_div(array_shape.x, tile_count.x);
  const int tile_height = ceil_div(array_shape.y, tile_count.y);

  const int x1 = tile_x * tile_width;
  int       x2 = (tile_x + 1) * tile_width;

  const int ty = reverse_tile_y_indexing ? tile_count.y - 1 - tile_y : tile_y;

  const int y1 = ty * tile_height;
  int       y2 = (ty + 1) * tile_height;

  if (overlapping_edges)
  {
    ++x2;
    ++y2;
  }

  // Clamp to exclusive bounds
  x2 = std::min(x2, array_shape.x);
  y2 = std::min(y2, array_shape.y);

  return {x1, x2, y1, y2};
}

void export_tiled(const std::string &fname_radical,
                  const std::string &fname_extension,
                  const Array       &array,
                  const glm::ivec2  &tiling,
                  int                leading_zeros,
                  int                depth,
                  bool               overlapping_edges,
                  bool               reverse_tile_y_indexing)
{
  if (tiling.x <= 0 || tiling.y <= 0) return;

  for (int tx = 0; tx < tiling.x; ++tx)
    for (int ty = 0; ty < tiling.y; ++ty)
    {
      const glm::ivec4 b = helper_compute_tile_bounds(tx,
                                                      ty,
                                                      tiling,
                                                      array.shape,
                                                      overlapping_edges,
                                                      reverse_tile_y_indexing);

      const int x1 = b.x;
      const int x2 = b.y;
      const int y1 = b.z;
      const int y2 = b.w;

      if (x1 >= x2 || y1 >= y2) continue;

      Array tile = array.extract_slice(x1, x2, y1, y2);

      const std::string str_tx = zfill(std::to_string(tx), leading_zeros);
      const std::string str_ty = zfill(std::to_string(ty), leading_zeros);

      const std::string fname = fname_radical + "_" + str_tx + "_" + str_ty +
                                "." + fname_extension;

      tile.to_png_grayscale(fname, depth);
    }
}

} // namespace hmap
