/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */

/**
 * @file colorize.hpp
 * @author Otto Link (otto.link.bv@gmail.com)
 * @brief Header file for functions related to applying colorization and
 * hillshading.
 *
 * This file contains the declarations of functions that are used to apply
 * colorization and hillshading effects to images and arrays. The functions
 * allow for creating grayscale, histogram-based, and colored images from
 * elevation data and other input arrays.
 *
 * @copyright Copyright (c) 2023 Otto Link
 */
#pragma once
#include "highmap/array.hpp"
#include "highmap/tensor.hpp"
#include "highmap/virtual_array/virtual_texture.hpp"

namespace hmap
{

enum NormalMapBlendingMethod : int
{
  NMAP_LINEAR,
  NMAP_DERIVATIVE,
  NMAP_UDN,
  NMAP_UNITY,
  NMAP_WHITEOUT
};

static std::map<std::string, int> normal_map_blending_method_as_string = {
    {"Linear", NMAP_LINEAR},
    {"Partial derivative", NMAP_DERIVATIVE},
    {"Unreal Developer Network", NMAP_UDN},
    {"Unity", NMAP_UNITY},
    {"Whiteout", NMAP_WHITEOUT},
};

struct ColorAdjust
{
  float in_min = 0.0f;
  float in_max = 1.0f;
  float exposure = 0.0f;
  float contrast = 1.0f;
  float saturation = 1.0f;
  float temperature = 0.0f;
  float gamma = 1.f;
  float dither_amp = 0.f;
  bool  filmic_tonemap = false;
  bool  aces_tonemap = false;
  bool  agx_tonemap = false;
};

/**
 * @enum Cmap
 * @brief Enumeration for different colormap types.
 *
 * This enumeration is defined in the highmap/colormap.hpp file and is used for
 * selecting the colormap to be applied during colorization.
 */
enum Cmap : int; // highmap/colormap.hpp

/**
 * @brief Apply hillshading to a Tensor image.
 *
 * This function applies a hillshading effect to the provided tensor image using
 * the elevation data in the input array. The effect is controlled by the power
 * exponent and can be scaled by specifying the minimum and maximum values.
 *
 * @param img      Input image represented as a Tensor.
 * @param array    Elevation data used to generate the hillshading effect.
 * @param vmin     Minimum value for scaling the hillshading effect (default is
 *                 0.f).
 * @param vmax     Maximum value for scaling the hillshading effect (default is
 *                 1.f).
 * @param exponent Power exponent applied to the hillshade values (default is
 *                 1.f).
 */
void apply_hillshade(Tensor      &img,
                     const Array &array,
                     float        vmin = 0.f,
                     float        vmax = 1.f,
                     float        exponent = 1.f);

/**
 * @brief Apply hillshading to an 8-bit image.
 *
 * This function applies a hillshading effect to a vector of 8-bit image data.
 * The effect is computed based on the elevation data in the input array. The
 * image can be either RGB or RGBA format depending on the is_img_rgba flag.
 *
 * @param img         Input image represented as a vector of 8-bit data.
 * @param array       Elevation data used to generate the hillshading effect.
 * @param vmin        Minimum value for scaling the hillshading effect (default
 *                    is 0.f).
 * @param vmax        Maximum value for scaling the hillshading effect (default
 *                    is 1.f).
 * @param exponent    Power exponent applied to the hillshade values (default is
 *                    1.f).
 * @param is_img_rgba Boolean flag indicating if the input image has an alpha
 *                    channel (default is false).
 */
void apply_hillshade(std::vector<uint8_t> &img,
                     const Array          &array,
                     float                 vmin = 0.f,
                     float                 vmax = 1.f,
                     float                 exponent = 1.f,
                     bool                  is_img_rgba = false);

void color_adjust(VirtualTexture    &tex,
                  ColorAdjust        param,
                  const ComputeMode &cm);

/**
 * @brief Apply colorization to an array.
 *
 * This function applies a colormap to the input array and optionally applies
 * hillshading and noise. The colorization can be reversed, and the function
 * returns a Tensor representing the colorized image.
 *
 * @param  array       Input array to be colorized.
 * @param  vmin        Minimum value for scaling the colormap.
 * @param  vmax        Maximum value for scaling the colormap.
 * @param  cmap        Colormap to be applied (using the Cmap enum).
 * @param  hillshading Boolean flag to apply hillshading.
 * @param  reverse     Boolean flag to reverse the colormap (default is false).
 * @param  p_noise     Optional pointer to a noise array (default is nullptr).
 * @return             Tensor Colorized Tensor image.
 */
Tensor colorize(const Array &array,
                float        vmin,
                float        vmax,
                int          cmap,
                bool         hillshading,
                bool         reverse = false,
                const Array *p_noise = nullptr);

/**
 * @brief Colorize a scalar field into a texture using a predefined colormap.
 * @param out     Output texture.
 * @param level   Input scalar values.
 * @param cm      Compute mode (CPU/GPU).
 * @param vmin    Lower bound for normalization.
 * @param vmax    Upper bound for normalization.
 * @param cmap    Colormap identifier.
 * @param p_alpha Optional alpha channel.
 * @param reverse Reverse colormap mapping.
 * @param p_noise Optional noise for dithering.
 *
 * **Example**
 * @include ex_virtual_texture.cpp
 *
 * **Result**
 * @image html ex_virtual_texture.png
 */
void colorize(VirtualTexture    &out,
              VirtualArray      &level,
              const ComputeMode &cm,
              float              vmin,
              float              vmax,
              int                cmap,
              VirtualArray      *p_alpha = nullptr,
              bool               reverse = false,
              VirtualArray      *p_noise = nullptr);

/**
 * @brief Colorize a scalar field into a texture using a custom colormap.
 * @param out             Output texture.
 * @param level           Input scalar values.
 * @param cm              Compute mode (CPU/GPU).
 * @param vmin            Lower bound for normalization.
 * @param vmax            Upper bound for normalization.
 * @param positions       Normalized color positions.
 * @param colormap_colors RGB(A) colors per position.
 * @param p_alpha         Optional alpha channel.
 * @param reverse         Reverse colormap mapping.
 * @param p_noise         Optional noise for dithering.
 *
 * **Example**
 * @include ex_virtual_texture.cpp
 *
 * **Result**
 * @image html ex_virtual_texture.png
 */
void colorize(VirtualTexture               &out,
              VirtualArray                 &level,
              const ComputeMode            &cm,
              float                         vmin,
              float                         vmax,
              const std::vector<float>     &positions,
              const std::vector<glm::vec3> &colormap_colors,
              VirtualArray                 *p_alpha = nullptr,
              bool                          reverse = false,
              VirtualArray                 *p_noise = nullptr);

/**
 * @brief Convert an array to a grayscale image.
 *
 * This function converts the input array to an 8-bit grayscale Tensor image.
 *
 * @param  array Input array.
 * @return       Tensor Grayscale Tensor image.
 */
Tensor colorize_grayscale(const Array &array);

/**
 * @brief Convert an array to a histogram-based grayscale image.
 *
 * This function converts the input array to an 8-bit grayscale Tensor image
 * using a histogram-based method for enhanced contrast.
 *
 * @param  array Input array.
 * @return       Tensor Grayscale Tensor image with histogram-based contrast.
 */
Tensor colorize_histogram(const Array &array);

/**
 * @brief Colorizes a slope height heatmap based on the gradient norms of a
 * given array.
 *
 * This function computes a colorized heatmap using a two-dimensional histogram
 * that considers the gradient norm of the input array. It normalizes both the
 * input array values and their corresponding gradient norms, calculates a 2D
 * histogram, and then applies a colormap to visualize the heatmap.
 *
 * @param  array The input array for which the slope height heatmap is computed.
 *               This should be a 2D array representing height values.
 * @param  cmap  An integer representing the colormap to be used for
 *               colorization. Colormap options depend on the colorization
 *               function used internally.
 *
 * @return       Tensor representing the colorized heatmap, which visualizes the
 *               distribution of height values and their corresponding gradient
 *               norms.
 *
 * @details
 * - The function normalizes the height values and gradient norms independently
 * using the minimum and maximum values.
 * - A 2D histogram is constructed, where each bin corresponds to a pair of
 * normalized height and gradient values.
 * - The colormap is then applied to this histogram to produce a colorized
 * output.
 *
 * @note If the input array has a constant value (i.e., min == max), no
 * normalization is applied, and the function may not produce meaningful
 * results.
 *
 * @warning Ensure that the input array is non-empty and has valid dimensions.
 *
 * **Example**
 * @include ex_colorize_slope_height_heatmap.cpp
 *
 * **Result**
 * @image html ex_colorize_slope_height_heatmap.png
 */
Tensor colorize_slope_height_heatmap(const Array &array, int cmap);

/**
 * @brief Combine two arrays into a colored image.
 *
 * This function takes two input arrays and combines them into a single 8-bit
 * colored Tensor image. The resulting image uses the data from both arrays to
 * create a composite color representation.
 *
 * @param  array1 First input array.
 * @param  array2 Second input array.
 * @return        Tensor Colorized Tensor image.
 *
 * **Example**
 * @include ex_colorize_vec2.cpp
 *
 * **Result**
 * @image html ex_colorize_vec2.png
 */
Tensor colorize_vec2(const Array &array1, const Array &array2);

/**
 * @brief Compute luminance from a texture.
 *
 * Computes a grayscale luminance array from the RGB channels of a virtual
 * texture and stores the result in @p out.
 *
 * @param out Output luminance array.
 * @param tex Input virtual texture (expects at least 3 channels).
 * @param cm  Compute mode (execution and storage behavior).
 */
void luminance(VirtualArray &out, VirtualTexture &tex, const ComputeMode &cm);

/**
 * @brief Mix two textures into an output texture.
 * @param out          Output texture.
 * @param tex1         First input texture.
 * @param tex2         Second input texture.
 * @param cm           Compute mode (CPU/GPU).
 * @param use_sqrt_avg Use square-root averaging.
 *
 * **Example**
 * @include ex_virtual_texture.cpp
 *
 * **Result**
 * @image html ex_virtual_texture.png
 */
void mix(VirtualTexture    &out,
         VirtualTexture    &tex1,
         VirtualTexture    &tex2,
         const ComputeMode &cm,
         bool               use_sqrt_avg = true);

void mix(VirtualTexture                &out,
         std::vector<VirtualTexture *> &texs,
         const ComputeMode             &cm,
         bool                           use_sqrt_avg = true);

/**
 * @brief Blend two normal maps into a single output normal map.
 *
 * Combines a base normal map with a detail normal map using the specified
 * blending method and scaling factor.
 *
 * @param out             Output normal map texture.
 * @param nmap_base       Base normal map texture.
 * @param nmap_detail     Detail normal map texture.
 * @param cm              Compute mode (execution and storage behavior).
 * @param detail_scaling  Strength of the detail normal map.
 * @param blending_method Normal map blending method.
 */
void mix_normal_map(VirtualTexture         &out,
                    VirtualTexture         &nmap_base,
                    VirtualTexture         &nmap_detail,
                    const ComputeMode      &cm,
                    float                   detail_scaling,
                    NormalMapBlendingMethod blending_method);

} // namespace hmap
