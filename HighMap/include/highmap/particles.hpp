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
#include "highmap/functions.hpp"

namespace hmap
{

/**
 * @brief Rasterizes a segment between two integer points using Bresenham's
 * algorithm.
 *
 * Ensures full grid connectivity by filling all intermediate integer cells
 * between points a and b.
 *
 * @param out Output container receiving the rasterized points.
 * @param a   Start point.
 * @param b   End point.
 */
void add_line_bresenham(std::vector<glm::ivec2> &out,
                        glm::ivec2               a,
                        glm::ivec2               b);

/**
 * @brief Applies transverse noise displacement along an integer path.
 *
 * The function computes a global direction from first to last point, derives a
 * perpendicular axis, and displaces each point using procedural noise.
 *
 * @param ipath       Input/output integer path to deform.
 * @param seed        Random seed for noise generation.
 * @param kw          Noise spatial frequency (scale).
 * @param amp         Maximum displacement amplitude.
 * @param shape       Optional domain scaling / bounds for noise sampling.
 * @param noise_type  Type of procedural noise to use.
 * @param octaves     Number of noise octaves for fractal noise.
 * @param weight      Blend factor between octaves.
 * @param persistence Amplitude decay per octave.
 * @param lacunarity  Frequency increase per octave.
 *
 * **Example**
 * @include ex_add_noise_ipath.cpp
 *
 * **Result**
 * @image html ex_add_noise_ipath.png
 */
void add_noise(std::vector<glm::ivec2> &ipath,
               std::uint32_t            seed,
               float                    kw,
               float                    amp,
               const glm::ivec2        &shape,
               NoiseType                noise_type = NoiseType::PERLIN,
               int                      octaves = 8,
               float                    weight = 0.7f,
               float                    persistence = 0.5f,
               float                    lacunarity = 2.f);

/**
 * @brief Ensures that an integer path is grid-adjacent and gap-free.
 *
 * Reconstructs the path so that every consecutive step moves to a neighboring
 * grid cell (8-connected), filling missing cells if needed.
 *
 * @param ipath Input/output path to repair and densify.
 */
void enforce_path_adjacency(std::vector<glm::ivec2> &ipath);

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