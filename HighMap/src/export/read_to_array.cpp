/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <string> // for string

#include <opencv2/core/mat.hpp>  // for Mat
#include <opencv2/imgcodecs.hpp> // for ImreadModes, imread

#include "macrologger.h" // for LOG_ERROR

#include "highmap/array.hpp" // for Array, cv_mat_to_array

namespace hmap
{

Array read_to_array(const std::string &fname, bool flip_j, bool remap)
{
  cv::Mat mat = cv::imread(fname, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);

  if (mat.data == nullptr)
  {
    LOG_ERROR("error while reading the image file: %s", fname.c_str());
    return Array();
  }
  else
  {
    return cv_mat_to_array(mat, remap, flip_j);
  }
}

} // namespace hmap
