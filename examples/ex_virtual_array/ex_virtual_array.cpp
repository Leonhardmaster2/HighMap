#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "highmap.hpp"

#define EXPECT(cond, msg)                                                      \
  do                                                                           \
  {                                                                            \
    if (!(cond))                                                               \
      std::cerr << "[FAIL] " << msg << "\n";                                   \
    else                                                                       \
      std::cout << "[ OK ] " << msg << "\n";                                   \
  } while (0)

int main()
{
  hmap::gpu::init_opencl();

  // ===========================================================================
  // Configuration
  // ===========================================================================

  const glm::vec4 bbox{0.f, 1.f, 0.f, 1.f};

  const glm::ivec2 shape{1024, 1024};
  const glm::ivec2 tile_shape{220, 312};
  const size_t     halo = 32;

  // const auto storage_mode = hmap::StorageMode::VA_RAM;
  const auto storage_mode = hmap::StorageMode::VA_DISK_LRU;
  // const auto storage_mode = hmap::StorageMode::VA_DISK_LRU_MIN;
  // const auto storage_mode = hmap::StorageMode::VA_DISK_SEQUENTIAL;

  // const auto foreach_mode = hmap::ForEachMode::VA_DISTRIBUTED;
  // const auto foreach_mode = hmap::ForEachMode::VA_SEQUENTIAL;
  const auto foreach_mode = hmap::ForEachMode::VA_SINGLE_ARRAY;
  // const auto foreach_mode = hmap::ForEachMode::VA_SINGLE_ARRAY_STRIDED;
  // const auto foreach_mode = hmap::ForEachMode::VA_SINGLE_ARRAY_DOWNSCALED;

  const hmap::ComputeMode compute_mode{.mode = foreach_mode,
                                       .trim_storage = true,
                                       .stride = 8};

  // ===========================================================================
  // Virtual arrays
  // ===========================================================================

  hmap::VirtualArray varray(shape, bbox, tile_shape, halo, storage_mode);
  hmap::VirtualArray varray_binary(shape, bbox, tile_shape, halo, storage_mode);

  // ===========================================================================
  // Tile operations
  // ===========================================================================

  const auto generate_noise =
      [](hmap::Array &tile, const hmap::TileRegion &region)
  {
    tile = hmap::noise_fbm(hmap::NoiseType::PERLIN,
                           region.shape,
                           {2.f, 2.f},
                           0,
                           8,
                           0.7f,
                           0.5f,
                           2.f,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           region.bbox);
  };

  const auto apply_gamma = [](hmap::Array &tile, const hmap::TileRegion &)
  { hmap::gamma_correction(tile, 2.f); };

  const auto smooth = [](hmap::Array &tile, const hmap::TileRegion &)
  { hmap::smooth_cpulse(tile, 64); };

  hmap::for_each_tile(varray, generate_noise, compute_mode);
  hmap::for_each_tile(varray, apply_gamma, compute_mode);
  // hmap::for_each_tile(varray, smooth, compute_mode);

  // specific
  {
    const auto erosion = [](hmap::Array &tile, const hmap::TileRegion &)
    { hmap::gpu::hydraulic_stream_log(tile, 0.2f, 1e0f); };

    hmap::ComputeMode cm_dwn = {
        .mode = hmap::ForEachMode::VA_SINGLE_ARRAY_DOWNSCALED,
        .trim_storage = false,
        .k_cutoff = 0.3f};

    hmap::for_each_tile(varray, erosion, cm_dwn);
  }

  const hmap::ComputeMode cm_local = {.mode = hmap::ForEachMode::VA_SEQUENTIAL,
                                      .trim_storage = false};

  varray.to_array(cm_local).to_png("out_noise.png", hmap::Cmap::JET);

  // ===========================================================================
  // Multi-array operation
  // ===========================================================================

  const auto binarize_copy =
      [](std::vector<hmap::Array *> arrays, const hmap::TileRegion &)
  {
    auto &src = *arrays[0];
    auto &dst = *arrays[1];

    dst = src;
    hmap::make_binary(dst, 0.5f);
  };

  varray.trim_storage();

  hmap::for_each_tile({&varray, &varray_binary}, binarize_copy, compute_mode);

  varray_binary.to_array(cm_local).to_png("out_binary.png", hmap::Cmap::JET);

  // ===========================================================================
  // Global operations
  // ===========================================================================

  varray.smooth_overlap_buffers();
  varray.remap(0.f, 1.f, compute_mode);

  // ===========================================================================
  // Tests
  // ===========================================================================

  // --- Determinism

  {
    auto a0 = varray.to_array(compute_mode);
    auto a1 = varray.to_array(compute_mode);

    float max_diff = 0.f;
    for (int j = 0; j < a0.shape.y; ++j)
      for (int i = 0; i < a0.shape.x; ++i)
        max_diff = std::max(max_diff, std::abs(a0(i, j) - a1(i, j)));

    EXPECT(max_diff == 0.f, "deterministic to_array()");
  }

  // --- Range invariant

  {
    float min_v = varray.min(compute_mode);
    float max_v = varray.max(compute_mode);

    EXPECT(min_v >= 0.f, "min >= 0 after remap");
    EXPECT(max_v <= 1.f, "max <= 1 after remap");
  }

  // --- Binary validity

  {
    auto values = varray_binary.unique_values(compute_mode);

    bool ok = true;
    for (float v : values)
      ok &= (v == 0.f || v == 1.f);

    EXPECT(ok, "binary array contains only {0,1}");
  }

  // --- Reduction consistency

  {
    auto arr = varray.to_array(compute_mode);

    double sum = 0.0;
    for (int j = 0; j < arr.shape.y; ++j)
      for (int i = 0; i < arr.shape.x; ++i)
        sum += arr(i, j);

    double mean_ref = sum / (arr.shape.x * arr.shape.y);
    double mean_va = varray.mean(compute_mode);

    EXPECT(std::abs(mean_ref - mean_va) < 1e-6f, "mean == sum / N");
  }

  // --- from_array round-trip

  {
    hmap::Array src = hmap::noise_fbm(hmap::NoiseType::PERLIN,
                                      shape,
                                      {2.f, 2.f},
                                      0);

    hmap::VirtualArray v(shape, bbox, tile_shape, halo, storage_mode);
    v.from_array(src, compute_mode);

    auto dst = v.to_array(compute_mode);

    float max_diff = 0.f;
    for (int j = 0; j < shape.y; ++j)
      for (int i = 0; i < shape.x; ++i)
        max_diff = std::max(max_diff, std::abs(src(i, j) - dst(i, j)));

    EXPECT(max_diff < 1e-6f, "from_array() round-trip");
  }

  // --- Interpolation sanity

  {
    float v0 = varray.get_nearest(bbox.x, bbox.z);
    float v1 = varray.get_bilinear(bbox.y, bbox.w);

    EXPECT(std::isfinite(v0), "nearest interpolation finite");
    EXPECT(std::isfinite(v1), "bilinear interpolation finite");
  }

  // ===========================================================================
  // Small tile stress test (edge case)
  // ===========================================================================

  {
    const glm::ivec2 small_tile{8, 8};
    const size_t     big_halo = 7;

    hmap::VirtualArray v_small(shape, bbox, small_tile, big_halo, storage_mode);

    hmap::for_each_tile(v_small, generate_noise, compute_mode);
    v_small.smooth_overlap_buffers();
    v_small.remap(0.f, 1.f, compute_mode);

    float min_v = v_small.min(compute_mode);
    float max_v = v_small.max(compute_mode);

    EXPECT(min_v >= 0.f, "small tile min >= 0");
    EXPECT(max_v <= 1.f, "small tile max <= 1");

    auto arr = v_small.to_array(compute_mode);
    EXPECT(arr.shape == shape, "small tile shape preserved");
  }

  return 0;
}
