/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <string> // for string

#include "highmap/array.hpp"    // for Array
#include "highmap/export.hpp"   // for export_normal_map_png
#include "highmap/gradient.hpp" // for normal_map
#include "highmap/tensor.hpp"   // for Tensor

namespace hmap
{

void export_normal_map_png(const std::string &fname,
                           const Array       &array,
                           int                depth)
{
  Tensor nmap = normal_map(array);
  nmap.to_png(fname, depth);
}

} // namespace hmap
