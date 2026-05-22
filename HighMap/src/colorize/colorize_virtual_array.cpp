/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm>        // for reverse, clamp
#include <cstddef>          // for size_t
#include <functional>       // for function
#include <initializer_list> // for initializer_list
#include <vector>           // for vector

#include "macrologger.h" // for LOG_ERROR

#include "highmap/array.hpp"                         // for Array, operator*
#include "highmap/colorize.hpp"                      // for NormalMapBlendi...
#include "highmap/colormaps.hpp"                     // for get_colormap_data
#include "highmap/interpolate1d.hpp"                 // for Interpolator1D
#include "highmap/math/array.hpp"                    // for lerp, pow
#include "highmap/operator.hpp"                      // for linspace
#include "highmap/virtual_array/tile_region.hpp"     // for TileRegion
#include "highmap/virtual_array/virtual_array.hpp"   // for VirtualArray
#include "highmap/virtual_array/virtual_texture.hpp" // for VirtualTexture

namespace hmap
{

void colorize(VirtualTexture    &out,
              VirtualArray      &level,
              const ComputeMode &cm,
              float              vmin,
              float              vmax,
              int                cmap,
              VirtualArray      *p_alpha,
              bool               reverse,
              VirtualArray      *p_noise)
{
  std::vector<glm::vec3> colors = get_colormap_data(cmap);

  std::vector<float> positions = linspace(0.f, 1.f, colors.size(), true);

  colorize(out,
           level,
           cm,
           vmin,
           vmax,
           positions,
           colors,
           p_alpha,
           reverse,
           p_noise);
}

void colorize(VirtualTexture               &out,
              VirtualArray                 &level,
              const ComputeMode            &cm,
              float                         vmin,
              float                         vmax,
              const std::vector<float>     &positions,
              const std::vector<glm::vec3> &colormap_colors,
              VirtualArray                 *p_alpha,
              bool                          reverse,
              VirtualArray                 *p_noise)
{
  if (out.channels() < 3)
  {
    LOG_ERROR("VirtualTexture must have at least 3 channels to be colorized");
    return;
  }

  // copy to reverse them if requested
  auto cpos = positions;
  auto colors = colormap_colors;

  if (reverse)
  {
    std::reverse(colors.begin(), colors.end());
    std::reverse(cpos.begin(), cpos.end());

    for (auto &p : cpos)
      p = 1.f - p;
  }

  // colorize fct
  auto lambda = [&](std::vector<Array *> &p_arrays, const TileRegion &region)
  {
    Array &z = *p_arrays[0];
    Array *pa_noise = p_arrays[1];
    Array *pa_alpha = p_arrays[2];
    Array &r = *p_arrays[3];
    Array &g = *p_arrays[4];
    Array &b = *p_arrays[5];

    // color interpolators
    std::vector<float> cc_r, cc_g, cc_b;
    for (const auto &col : colors)
    {
      cc_r.push_back(col[0]);
      cc_g.push_back(col[1]);
      cc_b.push_back(col[2]);
    }

    Interpolator1D citp_r = hmap::Interpolator1D(
        cpos,
        cc_r,
        hmap::InterpolationMethod1D::LINEAR);

    Interpolator1D citp_g = hmap::Interpolator1D(
        cpos,
        cc_g,
        hmap::InterpolationMethod1D::LINEAR);

    Interpolator1D citp_b = hmap::Interpolator1D(
        cpos,
        cc_b,
        hmap::InterpolationMethod1D::LINEAR);

    // colorize
    for (int j = 0; j < region.shape.y; ++j)
      for (int i = 0; i < region.shape.x; ++i)
      {
        float v = z(i, j) + (pa_noise ? (*pa_noise)(i, j) : 0.f);
        v = (v - vmin) / (vmax - vmin);
        v = std::clamp(v, 0.f, 1.f);

        r(i, j) = citp_r(v);
        g(i, j) = citp_g(v);
        b(i, j) = citp_b(v);
      }

    // alpha channel
    if (pa_alpha && p_arrays.size() == 7)
    {
      *p_arrays[6] = *pa_alpha;
    }
    else
    {
      *p_arrays[6] = 1.f;
    }
  };

  // apply
  std::vector<VirtualArray *> ptrs = {&level, p_noise, p_alpha};
  for (auto &ptr : out.channels_ptr())
    ptrs.push_back(ptr);

  for_each_tile(ptrs, lambda, cm);
}

void luminance(VirtualArray &out, VirtualTexture &tex, const ComputeMode &cm)
{
  if (tex.channels() < 3)
  {
    LOG_ERROR("VirtualTexture: inputs mismatch, virtual textures must have 3 "
              "channels for luminance.");
    return;
  }

  auto lambda = [](std::vector<Array *> &p_arrays, const TileRegion &)
  {
    hmap::Array &lum = *p_arrays[0];
    hmap::Array &r = *p_arrays[1];
    hmap::Array &g = *p_arrays[2];
    hmap::Array &b = *p_arrays[3];

    lum = 0.299f * r + 0.587f * g + 0.114f * b;
  };

  std::vector<VirtualArray *> ptrs = {&out};
  for (auto ptr : tex.channels_ptr())
    ptrs.push_back(ptr);

  for_each_tile(ptrs, lambda, cm);
}

void mix(VirtualTexture    &out,
         VirtualTexture    &tex1,
         VirtualTexture    &tex2,
         const ComputeMode &cm,
         bool               use_sqrt_avg)
{
  // --- failsafe

  if (out.channels() != 4 || tex1.channels() != 4 || tex2.channels() != 4 ||
      out.shape != tex1.shape || out.shape != tex2.shape)
  {
    LOG_ERROR("VirtualTexture: inputs mismatch, virtual textures must have 4 "
              "channels and same shape.");
    return;
  }

  // --- mixing function

  std::function<void(Array &, Array &, Array &, Array &)> lambda_blend;

  if (use_sqrt_avg)
  {
    lambda_blend = [](Array &out, Array &in1, Array &in2, Array &t)
    { out = pow((1.f - t) * in1 * in1 + t * in2 * in2, 0.5f); };
  }
  else
  {
    lambda_blend = [](Array &out, Array &in1, Array &in2, Array &t)
    { out = lerp(in1, in2, t); };
  }

  // --- colorize fct

  auto lambda =
      [&lambda_blend](std::vector<Array *> &p_arrays, const TileRegion &)
  {
    //      R  G  B  A
    // out  0  1  2  3
    // tex1 4  5  6  7
    // tex2 8  9  10 11

    Array &a1 = *p_arrays[7];
    Array &a2 = *p_arrays[11];

    Array t = a2 / (a2 + a1 * (1.f - a2)); // mixing factor

    for (int nch = 0; nch < 3; ++nch)
      lambda_blend(*p_arrays[0 + nch],
                   *p_arrays[4 + nch],
                   *p_arrays[8 + nch],
                   t);

    // alpha channel
    *p_arrays[3] = a1 + a2 * (1.f - a1);
  };

  // apply
  std::vector<VirtualArray *> ptrs = {};
  for (auto &plist :
       {out.channels_ptr(), tex1.channels_ptr(), tex2.channels_ptr()})
  {
    for (auto &ptr : plist)
      ptrs.push_back(ptr);
  }

  for_each_tile(ptrs, lambda, cm);
}

void mix(VirtualTexture                &out,
         std::vector<VirtualTexture *> &texs,
         const ComputeMode             &cm,
         bool                           use_sqrt_avg)
{
  if (texs.size() == 0) return;

  out.copy_from(*texs.front(), cm);

  for (size_t k = 1; k < texs.size(); k++)
    mix(out, out, *(texs[k]), cm, use_sqrt_avg);
}

void mix_normal_map(VirtualTexture         &out,
                    VirtualTexture         &nmap_base,
                    VirtualTexture         &nmap_detail,
                    const ComputeMode      &cm,
                    float                   detail_scaling,
                    NormalMapBlendingMethod blending_method)
{
  // output, also used to store first normal map
  out.copy_from(nmap_base, cm);

  // mix and then re-normalize values assuming a RGB channels
  // represent a normal vector
  auto lambda = [detail_scaling, blending_method](std::vector<Array *> p_arrays,
                                                  const TileRegion    &region)
  {
    Array *pa_r1 = p_arrays[0];
    Array *pa_g1 = p_arrays[1];
    Array *pa_b1 = p_arrays[2];
    // skip alpha channel
    Array *pa_r2 = p_arrays[4];
    Array *pa_g2 = p_arrays[5];
    Array *pa_b2 = p_arrays[6];

    std::function<glm::vec3(glm::vec3 &, glm::vec3 &)> blending_fct;

    switch (blending_method)
    {
    case NormalMapBlendingMethod::NMAP_LINEAR:
    {
      blending_fct = [](glm::vec3 &n1, glm::vec3 &n2) { return n1 + n2; };
    }
    break;
    //
    case NormalMapBlendingMethod::NMAP_DERIVATIVE:
    {
      blending_fct = [](glm::vec3 &n1, glm::vec3 &n2)
      {
        glm::vec3 vn = glm::vec3(n1.x * n2.z + n2.x * n1.z,
                                 n1.y * n2.z + n2.y * n1.z,
                                 n1.z * n2.z);
        return vn;
      };
    }
    break;
    //
    case NormalMapBlendingMethod::NMAP_UDN:
    {
      blending_fct = [](glm::vec3 &n1, glm::vec3 &n2)
      {
        glm::vec3 vn = glm::vec3(n1.x + n2.x, n1.y + n2.y, n1.z);
        return vn;
      };
    }
    break;
      //
    case NormalMapBlendingMethod::NMAP_UNITY:
    {
      blending_fct = [](glm::vec3 &n1, glm::vec3 &n2)
      {
        glm::vec3 m0 = glm::vec3(n1.z, n1.x, -n1.x);
        glm::vec3 m1 = glm::vec3(n1.x, n1.z, -n1.y);
        glm::vec3 m2 = glm::vec3(n1.x, n1.y, n1.z);

        glm::vec3 vn = glm::vec3(n2.x * m0.x + n2.y * m1.x + n2.z * m2.x,
                                 n2.x * m0.y + n2.y * m1.y + n2.z * m2.y,
                                 n2.x * m0.z + n2.y * m1.z + n2.z * m2.z);
        return vn;
      };
    }
    break;
    //
    case NormalMapBlendingMethod::NMAP_WHITEOUT:
    default:
    {
      blending_fct = [](glm::vec3 &n1, glm::vec3 &n2)
      {
        glm::vec3 vn = glm::vec3(n1.x + n2.x, n1.y + n2.y, n1.z * n2.z);
        return vn;
      };
    }
    }

    for (int j = 0; j < region.shape.y; j++)
      for (int i = 0; i < region.shape.x; i++)
      {
        // do some rescaling because RGBA texture expected in [0, 1]
        // but normal vector expected in [-1, 1]

        glm::vec3 v111 = glm::vec3(1.f, 1.f, 1.f);
        glm::vec3 n1 = 2.f * glm::vec3((*pa_r1)(i, j),
                                       (*pa_g1)(i, j),
                                       (*pa_b1)(i, j)) -
                       v111;
        glm::vec3 n2 = 2.f * glm::vec3((*pa_r2)(i, j),
                                       (*pa_g2)(i, j),
                                       (*pa_b2)(i, j)) -
                       v111;

        n2.x *= detail_scaling;
        n2.y *= detail_scaling;
        n2.z *= detail_scaling;

        glm::vec3 vn = blending_fct(n1, n2);
        vn = glm::normalize(vn);

        (*pa_r1)(i, j) = 0.5f * vn.x + 0.5f;
        (*pa_g1)(i, j) = 0.5f * vn.y + 0.5f;
        (*pa_b1)(i, j) = 0.5f * vn.z + 0.5f;
      }
  };

  // apply
  std::vector<VirtualArray *> ptrs = {};
  for (auto &plist : {out.channels_ptr(), nmap_detail.channels_ptr()})
    for (auto &ptr : plist)
      ptrs.push_back(ptr);

  for_each_tile(ptrs, lambda, cm);
}

} // namespace hmap
