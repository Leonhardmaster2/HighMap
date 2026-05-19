/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm> // for min, max
#include <cmath>     // for cos, sin, M_PI
#include <limits>    // for numeric_limits
#include <utility>   // for pair

#include "highmap/coord_frame.hpp"                 // for CoordFrame
#include "highmap/virtual_array/virtual_array.hpp" // for VirtualArray

namespace hmap
{

CoordFrame::CoordFrame()
    : origin(glm::vec2(0.f, 0.f)), size(glm::vec2(1.f, 1.f))
{
  this->set_rotation_angle(0.f);
}

CoordFrame::CoordFrame(glm::vec2 origin, glm::vec2 size, float rotation_angle)
    : origin(origin), size(size)
{
  this->set_rotation_angle(rotation_angle);
}

glm::vec4 CoordFrame::compute_bounding_box() const
{
  // Define the four corners relative to the origin
  std::pair<float, float> corners[4] = {{0.f, 0.f},
                                        {1.f, 0.f},
                                        {1.f, 1.f},
                                        {0.f, 1.f}};

  // Rotate each corner and find min/max x and y
  float min_x = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float min_y = std::numeric_limits<float>::max();
  float max_y = std::numeric_limits<float>::lowest();

  for (const auto &corner : corners)
  {
    glm::vec2 g = this->map_to_global_coords(corner.first, corner.second);

    // Update bounding box extents
    min_x = std::min(min_x, g.x);
    max_x = std::max(max_x, g.x);
    min_y = std::min(min_y, g.y);
    max_y = std::max(max_y, g.y);
  }

  return glm::vec4(min_x, max_x, min_y, max_y);
}

float CoordFrame::get_heightmap_value_bilinear(const VirtualArray &h,
                                               float               gx,
                                               float               gy,
                                               float fill_value) const
{
  glm::vec2 rel = this->map_to_relative_coords(gx, gy);

  if (rel.x >= 0.f && rel.x <= 1.f && rel.y >= 0.f && rel.y <= 1.f)
  {
    return h.get_bilinear(rel.x, rel.y);
  }
  else
    return fill_value;
}

float CoordFrame::get_rotation_angle() const
{
  return this->rotation_angle;
}

// Method to determine if a point (x, y) is within the rotated terrain
bool CoordFrame::is_point_within(float gx, float gy) const
{
  glm::vec2 rel = this->map_to_relative_coords(gx, gy);

  // Check if the unrotated point is within the unrotated bounds
  return (rel.x >= 0.f && rel.x <= 1.f && rel.y >= 0.f && rel.y <= 1.f);
}

// Method to set the rotation angle and update cos_angle and sin_angle
void CoordFrame::set_rotation_angle(float new_angle)
{
  this->rotation_angle = new_angle;

  float angle_rad = this->rotation_angle * M_PI / 180.0f;

  this->cos_angle = std::cos(angle_rad);
  this->sin_angle = std::sin(angle_rad);
}

glm::vec2 CoordFrame::map_to_global_coords(float rx, float ry) const
{
  rx *= this->size.x;
  ry *= this->size.y;

  float gx = this->origin.x + rx * this->cos_angle - ry * this->sin_angle;
  float gy = this->origin.y + rx * this->sin_angle + ry * this->cos_angle;

  return glm::vec2(gx, gy);
}

// Method to apply inverse rotation to a point relative to the terrain origin
glm::vec2 CoordFrame::map_to_relative_coords(float gx, float gy) const
{
  // Translate the point to be relative to the terrain's origin
  glm::vec2 translated = glm::vec2(gx, gy) - this->origin;

  // Unrotate the point using precomputed cos_angle and sin_angle
  float unrotated_x = translated.x * this->cos_angle +
                      translated.y * this->sin_angle;
  float unrotated_y = -translated.x * this->sin_angle +
                      translated.y * this->cos_angle;

  unrotated_x /= this->size.x;
  unrotated_y /= this->size.y;

  return glm::vec2(unrotated_x, unrotated_y);
}

float CoordFrame::normalized_distance_to_edges(float gx, float gy) const
{
  // switch to unit-square coordinates
  glm::vec2 rel = this->map_to_relative_coords(gx, gy);

  // distances to the 4 edges, times 2 to get something in [0, 1]
  float dmin = 2.f * std::min(1.f - rel.y,
                              std::min(rel.y, std::min(rel.x, 1.f - rel.x)));

  return dmin;
}

float CoordFrame::normalized_shape_factor(float gx, float gy) const
{
  // switch to unit-square coordinates
  glm::vec2 rel = this->map_to_relative_coords(gx, gy);

  return 256.f * rel.x * rel.x * (1.f - rel.x) * (1.f - rel.x) * rel.y * rel.y *
         (1.f - rel.y) * (1.f - rel.y);
}

} // namespace hmap
