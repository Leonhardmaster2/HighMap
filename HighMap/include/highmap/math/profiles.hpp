/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */

/**
 * @file distance_functions.hpp
 * @author Otto Link (otto.link.bv@gmail.com)
 * @copyright Copyright (c) 2023
 */
#pragma once
#include <functional>

namespace hmap
{

/**
 * @brief Phasor angular profile type.
 */
enum PhasorProfile : int
{
  PP_COSINE_BULKY,
  PP_COSINE_PEAKY,
  PP_COSINE_SQUARE,
  PP_COSINE_STD,
  PP_TRIANGLE,
};

/**
 * @brief Radial profile type.
 */
enum RadialProfile : int
{
  RP_GAIN,
  RP_LINEAR,
  RP_POW,
  RP_SMOOTHSTEP,
  RP_SMOOTHSTEP_UPPER,
  RP_FLAT_BOTTOM,
  RP_SQRT
};

/**
 * @brief Generates a function representing a phasor profile based on the
 * specified type and parameters.
 *
 * This function returns a callable object (`std::function`) that computes the
 * value of the specified phasor profile for a given phase angle (phi).
 * Optionally, it can compute the average value of the profile over the range
 * [-π, π].
 *
 * @param  phasor_profile The type of phasor profile to generate.
 * @param  delta          A parameter that can influence the profile (depending
 *                        on the profile choice).
 * @param  p_profile_avg  Optional pointer to a float. If not `nullptr`, it will
 *                        store the average value of the profile over the range
 *                        [-π, π].
 * @return                A `std::function<float(float)>` that computes the
 *                        phasor profile for a given phase angle.
 *
 * @throws std::invalid_argumentIftheprovided`phasor_profile`isinvalid.
 *
 * @note The average value is computed using numerical integration over 50
 * sample points within [-π, π].
 */
std::function<float(float)> get_phasor_profile_function(
    PhasorProfile phasor_profile,
    float         delta,
    float        *p_profile_avg = nullptr);

/**
 * @brief Returns a normalized radial profile function.
 *
 * The returned function maps x ∈ [0,1] to a value in [0,1], with f(0) = 0 and
 * f(1) = 1. It is typically used for radial or distance-based falloff in
 * procedural heightmaps.
 *
 * @param  radial_profile Radial profile type.
 * @param  delta          Shape parameter used by some profiles.
 *
 * @return                Radial profile evaluation function.
 *
 * @throws std::invalid_argumentIftheprofileisinvalid.
 */
std::function<float(float)> get_radial_profile_function(
    RadialProfile radial_profile,
    float         delta);

} // namespace hmap