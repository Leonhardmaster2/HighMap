/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */

/**
 * @file particles.hpp
 * @author Otto Link (otto.link.bv@gmail.com)
 * @copyright Copyright (c) 2023
 */
#pragma once
#include <random>

#include <glm/glm.hpp>

#include "highmap/array.hpp"

namespace hmap
{

/**
 * @brief Spawn a point uniformly along the border of a 2D grid.
 *
 * The border is defined as a strip of thickness `nbuffer` around the grid
 * edges. Sampling is uniform over the entire border area (top, bottom, left,
 * right).
 *
 * @param  rng     Random number generator.
 * @param  nbuffer Border thickness in cells (clamped to at least 1).
 * @param  shape   Grid dimensions (width, height).
 * @return         Random integer coordinate on the border.
 */
glm::ivec2 spawn_borders(std::mt19937     &rng,
                         int               nbuffer,
                         const glm::ivec2 &shape);

/**
 * @brief Spawn a point uniformly inside a disk.
 *
 * Uses polar coordinates with sqrt-radius correction to ensure uniform area
 * density.
 *
 * @param  rng    Random number generator.
 * @param  center Disk center in grid coordinates.
 * @param  radius Disk radius.
 * @param  shape  Grid bounds used for clamping.
 * @return        Random point inside the disk.
 */
glm::ivec2 spawn_disc(std::mt19937     &rng,
                      const glm::ivec2 &center,
                      float             radius,
                      const glm::ivec2 &shape);

/**
 * @brief Spawn a point using a 2D Gaussian distribution.
 *
 * The distribution is centered on `center` with standard deviation `sigma`.
 *
 * @param  rng    Random number generator.
 * @param  center Mean position of the distribution.
 * @param  sigma  Standard deviation (spread).
 * @param  shape  Grid bounds used for clamping.
 * @return        Random Gaussian-distributed point.
 */
glm::ivec2 spawn_gaussian(std::mt19937     &rng,
                          const glm::vec2  &center,
                          float             sigma,
                          const glm::ivec2 &shape);

/**
 * @brief Spawn a point uniformly within an annulus (ring).
 *
 * The distribution is uniform over area between `radius_min` and `radius_max`.
 *
 * @param  rng        Random number generator.
 * @param  center     Center of the ring.
 * @param  radius_min Inner radius.
 * @param  radius_max Outer radius.
 * @param  shape      Grid bounds used for clamping.
 * @return            Random point inside the annulus.
 */
glm::ivec2 spawn_ring(std::mt19937     &rng,
                      const glm::ivec2 &center,
                      float             radius_min,
                      float             radius_max,
                      const glm::ivec2 &shape);

/**
 * @brief Spawn a point uniformly over the full grid.
 *
 * Each cell has equal probability of being selected.
 *
 * @param  rng   Random number generator.
 * @param  shape Grid dimensions (width, height).
 * @return       Random grid coordinate.
 */
glm::ivec2 spawn_uniform(std::mt19937 &rng, const glm::ivec2 &shape);

} // namespace hmap