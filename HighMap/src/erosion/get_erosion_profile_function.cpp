/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <fstream>
#include <functional>
#include <iomanip>

#include "macrologger.h"

#include "highmap/array.hpp"
#include "highmap/erosion.hpp"
#include "highmap/operator.hpp"

namespace hmap
{

std::vector<ErosionProfile> check_erosion_profile_function(float delta)
{
  std::vector<ErosionProfile> profiles = {
      ErosionProfile::EP_COSINE,
      ErosionProfile::EP_COSINE_BULK,
      ErosionProfile::EP_COSINE_PEAK,
      ErosionProfile::EP_PARABOL,
      ErosionProfile::EP_SAW_SHARP,
      ErosionProfile::EP_SAW_SMOOTH,
      ErosionProfile::EP_SHARP_VALLEYS,
      ErosionProfile::EP_SQRT,
      ErosionProfile::EP_TRIANGLE_GRENIER,
      ErosionProfile::EP_TRIANGLE_SHARP,
      ErosionProfile::EP_TRIANGLE_SMOOTH,
  };

  // test grid
  int                nd = 200;
  std::vector<float> phi = linspace(-M_PI, M_PI, nd);

  // --- CSV file

  // open output file
  std::ofstream file("erosion_profiles.csv");
  file << std::setprecision(8);

  // header
  file << "phi";
  for (auto ep : profiles)
    file << "," << std::to_string(ep);
  file << "\n";

  // --- Precompute functions

  struct ProfileData
  {
    ErosionProfile              profile;
    std::function<float(float)> func;
  };

  std::vector<ProfileData> profile_data;

  for (auto ep : profiles)
  {
    float profile_avg = 0.f;
    auto  fct = get_erosion_profile_function(ep, delta, profile_avg);
    profile_data.push_back({ep, fct});
  }

  // --- Write rows

  for (auto v : phi)
  {
    file << v;

    for (auto &p : profile_data)
      file << "," << p.func(v);

    file << "\n";
  }

  file.close();

  // return profile list for conveniency
  return profiles;
}

std::function<float(float)> get_erosion_profile_function(
    const ErosionProfile &erosion_profile,
    float                 delta,
    float                &profile_avg)
{
  std::function<float(float)> lambda_p;

  switch (erosion_profile)
  {
  case ErosionProfile::EP_COSINE:
    lambda_p = [](float phi) { return 0.5f - 0.5f * std::cos(phi); };
    break;
  //
  case ErosionProfile::EP_COSINE_BULK:
    lambda_p = [](float phi)
    { return 0.5f - 0.5f * std::cos(M_PI * std::pow(phi / M_PI, 2)); };
    break;
  //
  case ErosionProfile::EP_COSINE_PEAK:
    lambda_p = [](float phi)
    {
      float t = phi / M_PI;
      t = std::fmod(t + 2.f, 2.f) - 1.f;
      return 0.5f * std::cos(M_PI * t * t) + 0.5f;
    };
    break;
  //
  case ErosionProfile::EP_PARABOL:
    lambda_p = [](float phi) { return std::pow(phi / M_PI, 2); };
    break;
  //
  case ErosionProfile::EP_SAW_SHARP:
  {
    lambda_p = [](float phi)
    {
      float t = phi / M_PI + 1.f;
      t = std::fmod(t + 2.f, 2.f) - 1.f;
      return 0.5f * (t - int(t)) + 0.5f;
    };
  }
  break;
  //
  case ErosionProfile::EP_SAW_SMOOTH:
  {
    float n = 1.f + 0.02f / delta;
    float dn = 2.f * n;
    float coeff = std::pow(1.f / dn, 1.f / 2.f / n) * 2.f * n / dn;
    coeff = 1.f / coeff;

    lambda_p = [n, coeff](float phi)
    {
      float t = phi / M_PI + 1.f;
      t = std::fmod(t + 2.f, 2.f) - 1.f;
      t = coeff * t * (1.f - std::pow(t, 2.f * n));
      t = 0.5f * (1.f + t);
      return t;
    };
  }
  break;
  //
  case ErosionProfile::EP_SHARP_VALLEYS:
  {
    lambda_p = [delta](float phi)
    {
      float t = phi / M_PI;
      t = std::fmod(t + 2.f, 2.f) - 1.f;
      float v = (1.f - t * t) / (1.f + t * t / delta);
      return v;
    };
  }
  break;
    //
  case ErosionProfile::EP_SQRT:
    lambda_p = [](float phi) { return std::pow(std::abs(phi) / M_PI, 0.5f); };
    break;
  //
  case ErosionProfile::EP_TRIANGLE_GRENIER:
  {
    // https://onlinelibrary.wiley.com/doi/epdf/10.1111/cgf.14992
    lambda_p = [delta](float phi)
    {
      float t = phi / M_PI + 1.f;
      t = std::fmod(t + 2.f, 2.f) - 1.f;

      float value = std::sqrt((1.f + 2.f * std::sqrt(delta)) * t * t + delta) -
                    std::sqrt(delta);
      return value;
    };
  }
  break;
  //
  case ErosionProfile::EP_TRIANGLE_SHARP:
  {
    lambda_p = [](float phi)
    {
      float t = phi / M_PI;
      t = std::fmod(t + 2.f, 2.f) - 1.f;
      return 1.f - std::abs(t);
    };
  }
  break;
  //
  case ErosionProfile::EP_TRIANGLE_SMOOTH:
  {
    // https://mathematica.stackexchange.com/questions/38293
    float coeff = 0.5f / (std::acos(delta - 1.f) / M_PI - 0.5f);

    lambda_p = [delta, coeff](float phi)
    {
      float v = 0.5f + coeff * (std::acos((1.f - delta) *
                                          std::sin(phi + 0.5f * M_PI)) /
                                    M_PI -
                                0.5f);
      return v;
    };
  }
  break;
  }

  // average value
  profile_avg = 0.f;
  {
    int                nd = 50;
    std::vector<float> phi = linspace(-M_PI, M_PI, nd);
    for (auto &v : phi)
      profile_avg += lambda_p(v);
    profile_avg /= (float)nd;
  }

  return lambda_p;
}

} // namespace hmap
