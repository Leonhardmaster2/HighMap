/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "macrologger.h"

#include "highmap/virtual_array/virtual_texture.hpp"

#include "highmap/tensor.hpp"

namespace hmap
{

VirtualTexture::VirtualTexture(glm::ivec2                   shape,
                               glm::vec4                    bbox,
                               glm::ivec2                   tile_shape,
                               int                          halo,
                               int                          channels,
                               std::unique_ptr<TileStorage> storage_proto)
    : shape(shape), bbox(bbox), tile_shape(tile_shape), halo(halo)
{
  arrays.reserve(channels);

  for (int c = 0; c < channels; ++c)
  {
    arrays.emplace_back(shape, bbox, tile_shape, halo, storage_proto->clone());
  }
}

VirtualTexture::VirtualTexture(glm::ivec2  shape,
                               glm::vec4   bbox,
                               glm::ivec2  tile_shape,
                               int         halo,
                               int         channels,
                               StorageMode storage_mode)
    : shape(shape), bbox(bbox), tile_shape(tile_shape), halo(halo)
{
  arrays.reserve(channels);

  for (int c = 0; c < channels; ++c)
  {
    arrays.emplace_back(shape, bbox, tile_shape, halo, storage_mode);
  }
}

VirtualTexture::VirtualTexture(glm::ivec2  shape,
                               glm::ivec2  tile_shape,
                               int         halo,
                               int         channels,
                               StorageMode storage_mode)
    : shape(shape),
      bbox({0.f, 1.f, 0.f, 1.f}),
      tile_shape(tile_shape),
      halo(halo)
{
  arrays.reserve(channels);

  for (int c = 0; c < channels; ++c)
  {
    arrays.emplace_back(shape, bbox, tile_shape, halo, storage_mode);
  }
}

int VirtualTexture::channels() const
{
  return int(arrays.size());
}

VirtualArray &VirtualTexture::channel(int c)
{
  return arrays[c];
}
const VirtualArray &VirtualTexture::channel(int c) const
{
  return arrays[c];
}

std::vector<VirtualArray *> VirtualTexture::channels_ptr()
{
  std::vector<VirtualArray *> out;
  out.reserve(arrays.size());

  for (auto &a : arrays)
    out.push_back(&a);

  return out;
}

std::vector<const VirtualArray *> VirtualTexture::channels_ptr() const
{
  std::vector<const VirtualArray *> out;
  out.reserve(arrays.size());

  for (const auto &a : arrays)
    out.push_back(&a);

  return out;
}

void VirtualTexture::fill(float value, const ComputeMode &cm)
{
  for (auto &va : this->get_arrays())
    va.fill(value, cm);
}

void VirtualTexture::fill(int c, float value, const ComputeMode &cm)
{
  if (c > this->channels() - 1) return;

  this->channel(c).fill(value, cm);
}

std::vector<VirtualArray> &VirtualTexture::get_arrays()
{
  return this->arrays;
}

std::vector<uint8_t> VirtualTexture::to_img_8bit(const glm::ivec2  &img_shape,
                                                 const ComputeMode &cm)
{
  if (this->channels() < 3)
  {
    LOG_ERROR("VirtualTexture::to_img_8bit: not enough channels");
    return {0};
  }

  Array r = this->channel(0).to_array(img_shape, cm);
  Array g = this->channel(1).to_array(img_shape, cm);
  Array b = this->channel(2).to_array(img_shape, cm);

  if (this->channels() == 3)
  {
    Tensor t(img_shape, 3);
    t.set_slice(0, r);
    t.set_slice(1, g);
    t.set_slice(2, b);
    return t.to_img_8bit();
  }
  else
  {
    Array  a = this->channel(3).to_array(img_shape, cm);
    Tensor t(img_shape, 4);
    t.set_slice(0, r);
    t.set_slice(1, g);
    t.set_slice(2, b);
    t.set_slice(3, a);
    return t.to_img_8bit();
  }
}

void VirtualTexture::to_png(const glm::ivec2  &img_shape,
                            const std::string &fname,
                            const ComputeMode &cm,
                            int                depth) const
{
  if (this->channels() < 3)
  {
    LOG_ERROR("VirtualTexture::to_img_8bit: not enough channels");
    return;
  }

  Array r = this->channel(0).to_array(img_shape, cm);
  Array g = this->channel(1).to_array(img_shape, cm);
  Array b = this->channel(2).to_array(img_shape, cm);

  if (this->channels() == 3)
  {
    Tensor t(img_shape, 3);
    t.set_slice(0, r);
    t.set_slice(1, g);
    t.set_slice(2, b);
    t.to_png(fname, depth);
  }
  else
  {
    Array  a = this->channel(3).to_array(img_shape, cm);
    Tensor t(img_shape, 4);
    t.set_slice(0, r);
    t.set_slice(1, g);
    t.set_slice(2, b);
    t.set_slice(3, a);
    t.to_png(fname, depth);
  }
}

void VirtualTexture::to_png(const std::string &fname,
                            const ComputeMode &cm,
                            int                depth) const
{
  this->to_png(this->shape, fname, cm, depth);
}

} // namespace hmap
