/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */

#pragma once
#include "highmap/boundary.hpp"
#include "highmap/geometry/path.hpp"

namespace hmap
{

/**
 * @brief Dig a path on a heightmap.
 *
 * This function modifies a heightmap array by "digging" a path into it based on
 * the provided path. The path is represented by a `Path` object, and the
 * function adjusts the height values in the `z` array to create the appearance
 * of a dug path. The width, border decay, and flattening radius parameters
 * control the characteristics of the dig. Optionally, the path can be forced to
 * have a monotonically decreasing elevation.
 *
 * **Example**
 * @include ex_dig_path.cpp
 *
 * **Result**
 * @image html ex_dig_path.png
 *
 * @param z                 Input array representing the heightmap to be
 *                          modified.
 * @param path              The path to be dug into the heightmap, with
 *                          coordinates with respect to a unit-square. The path
 *                          will be processed to create the dig effect.
 * @param width             Radius of the path width in pixels. This determines
 *                          how wide the dug path will be.
 * @param decay             Radius of the path border decay in pixels. This
 *                          controls how quickly the effect of the path fades
 *                          out towards the edges.
 * @param flattening_radius Radius used to flatten the elevation of the path,
 *                          creating a smooth transition. This is measured in
 *                          pixels.
 * @param force_downhill    If `true`, the path's elevation will be forced to
 *                          decrease monotonically, creating a downhill effect.
 * @param bbox              Bounding box specifying the region of the heightmap
 *                          to consider for the digging operation. It defines
 *                          the area where the path is applied.
 * @param depth             Optional depth parameter to specify the maximum
 *                          depth of the dig. If not provided, the default depth
 *                          of 0.f is used.
 */
void dig_path(Array    &z,
              Path     &path,
              int       width = 1,
              int       decay = 2,
              int       flattening_radius = 16,
              bool      force_downhill = false,
              glm::vec4 bbox = {0.f, 1.f, 0.f, 1.f},
              float     depth = 0.f);

/**
 * @brief Modifies the elevation array to carve a river along a specified path.
 *
 * This function adjusts the elevation values in the input array `z` to simulate
 * a river along the provided `path`. It incorporates parameters for riverbed
 * and riverbank slopes, noise effects, and merging behavior to create a
 * realistic river profile.
 *
 * @param z               The input 2D array representing the elevation map.
 *                        This array will be modified in place.
 * @param path            The path along which the river is to be carved,
 *                        represented as a sequence of points, with coordinates
 *                        with respect to a unit-square.
 * @param riverbank_talus The slope of the riverbank, controlling how steep the
 *                        river's edges are.
 * @param merging_ir      The merging radius, specifying how far the effects of
 *                        multiple rivers combine.
 * @param riverbed_talus  The slope of the riverbed, controlling how steep the
 *                        riverbed is (default: 0.0).
 * @param noise_ratio     The proportion of random noise applied to the river's
 *                        shape for realism (default: 0.9).
 * @param seed            The seed for the random noise generator, ensuring
 *                        reproducibility (default: 0).
 *
 * **Example**
 * @include ex_flow_stream.cpp
 *
 * **Result**
 * @image html ex_flow_stream.png
 */
void dig_river(Array                   &z,
               const std::vector<Path> &path_list,
               float                    riverbank_talus,
               int                      river_width = 0,
               int                      merging_width = 0,
               float                    depth = 0.f,
               float                    riverbed_talus = 0.f,
               float                    noise_ratio = 0.9f,
               uint                     seed = 0,
               Array                   *p_mask = nullptr);

void dig_river(Array      &z,
               const Path &path,
               float       riverbank_talus,
               int         river_width = 0,
               int         merging_width = 0,
               float       depth = 0.f,
               float       riverbed_talus = 0.f,
               float       noise_ratio = 0.9f,
               uint        seed = 0,
               Array      *p_mask = nullptr);

/**
 * @brief Blends a flatbed carve into an existing heightmap.
 *
 * Carves and blends a flatbed shape along a path into the provided array, using
 * a smooth falloff mask.
 *
 * @param z                        Heightmap to modify in place.
 * @param path                     Path defining the bed centerline.
 * @param bottom_extent            Radius of the flatbed region.
 * @param vmin                     Base height value.
 * @param depth                    Bed depth.
 * @param falloff_distance         Falloff width outside the bed.
 * @param outer_slope              Linear slope beyond the bed.
 * @param preserve_bedshape        Preserve relative bed shape under radial
 *                                 noise.
 * @param radial_profile           Radial profile type.
 * @param radial_profile_parameter Shape parameter for the profile.
 * @param p_falloff_mask           Optional output falloff mask.
 * @param p_noise_r                Optional radial noise.
 * @param bbox                     World-space bounding box.
 *
 * **Example**
 * @include ex_flatbed_carve.cpp
 *
 * **Result**
 * @image html ex_flatbed_carve.png
 */
void flatbed_carve(Array        &z,
                   const Path   &path,
                   float         bottom_extent = 32.f, // pixels
                   float         vmin = 0.1f,
                   float         depth = 0.05f,
                   float         falloff_distance = 128.f,
                   float         outer_slope = 0.1f,
                   bool          preserve_bedshape = true,
                   RadialProfile radial_profile = RadialProfile::RP_GAIN,
                   float         radial_profile_parameter = 2.f,
                   Array        *p_falloff_mask = nullptr,
                   const Array  *p_noise_r = nullptr,
                   glm::vec4     bbox = {0.f, 1.f, 0.f, 1.f});

} // namespace hmap