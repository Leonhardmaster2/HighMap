/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <string>
#include <vector>

#include "cl_wrapper/device_manager.hpp"
#include "cl_wrapper/kernel_manager.hpp"
#include "cl_wrapper/run.hpp"

#include "highmap/array.hpp"
#include "highmap/opencl/gpu_opencl.hpp"

namespace hmap::gpu
{

void helper_bind_optional_buffer(clwrapper::Run    &run,
                                 const std::string &id,
                                 const Array       *p_array)
{
  std::vector<float> dummy_vector(1);

  if (p_array)
  {
    run.bind_buffer<float>(id, p_array->vector);
    run.write_buffer(id);
  }
  else
    run.bind_buffer<float>(id, dummy_vector);
}

bool init_opencl()
{
  if (!clwrapper::DeviceManager::get_instance().is_ready()) return false;

  // load and build kernels
  std::string code;
  code.reserve(270000);

  code +=
#include "kernels/_common_index.cl"
      ;
  code +=
#include "kernels/_common_math.cl"
      ;
  code +=
#include "kernels/_common_rand.cl"
      ;
  code +=
#include "kernels/_common_sort.cl"
      ;
  //
  code +=
#include "kernels/advection_particle.cl"
      ;
  code +=
#include "kernels/advection_warp.cl"
      ;
  code +=
#include "kernels/bilateral_filter.cl"
      ;
  code +=
#include "kernels/blend_poisson_bf.cl"
      ;
  code +=
#include "kernels/coastal_fetch.cl"
      ;
  code +=
#include "kernels/curvature_quadric.cl"
      ;
  code +=
#include "kernels/directional_blur.cl"
      ;
  code +=
#include "kernels/eulerian_transport.cl"
      ;
  code +=
#include "kernels/expand.cl"
      ;
  code +=
#include "kernels/flow_direction_d8.cl"
      ;
  code +=
#include "kernels/gabor_wave.cl"
      ;
  code +=
#include "kernels/gavoronoise.cl"
      ;
  code +=
#include "kernels/generate_riverbed.cl"
      ;
  code +=
#include "kernels/gradient_norm.cl"
      ;
  code +=
#include "kernels/harmonic_interpolation.cl"
      ;
  code +=
#include "kernels/hemisphere_field.cl"
      ;
  code +=
#include "kernels/hydraulic_particle.cl"
      ;
  code +=
#include "kernels/hydraulic_schott.cl"
      ;
  code +=
#include "kernels/hydraulic_vpipes.cl"
      ;
  code +=
#include "kernels/interpolate_array.cl"
      ;
  code +=
#include "kernels/jump_flooding.cl"
      ;
  code +=
#include "kernels/laplace.cl"
      ;
  code +=
#include "kernels/laplacian_fract.cl"
      ;
  code +=
#include "kernels/local_max.cl"
      ;
  code +=
#include "kernels/local_mean.cl"
      ;
  code +=
#include "kernels/local_min.cl"
      ;
  code +=
#include "kernels/local_relief.cl"
      ;
  code +=
#include "kernels/local_skewness.cl"
      ;
  code +=
#include "kernels/local_variance.cl"
      ;
  code +=
#include "kernels/local_z_score.cl"
      ;
  code +=
#include "kernels/maximum_smooth.cl"
      ;
  code +=
#include "kernels/mean_shift.cl"
      ;
  code +=
#include "kernels/median_3x3.cl"
      ;
  code +=
#include "kernels/minimum_smooth.cl"
      ;
  code +=
#include "kernels/mountain_range_radial.cl"
      ;
  code +=
#include "kernels/noise_a.cl"
      ;
  code +=
#include "kernels/noise_b.cl"
      ;
  code +=
#include "kernels/normal_displacement.cl"
      ;
  code +=
#include "kernels/phase_averaging.cl"
      ;
  code +=
#include "kernels/phase_field.cl"
      ;
  code +=
#include "kernels/plateau.cl"
      ;
  code +=
#include "kernels/polygon_field.cl"
      ;
  code +=
#include "kernels/project_slope_along_direction.cl"
      ;
  code +=
#include "kernels/rotate.cl"
      ;
  code +=
#include "kernels/ruggedness.cl"
      ;
  code +=
#include "kernels/rugosity.cl"
      ;
  code +=
#include "kernels/sdf_2d_polyline.cl"
      ;
  code +=
#include "kernels/shallow_viscous_flow.cl"
      ;
  code +=
#include "kernels/skeleton.cl"
      ;
  code +=
#include "kernels/smooth_cpulse.cl"
      ;
  code +=
#include "kernels/snow_simulation.cl"
      ;
  code +=
#include "kernels/sparse_max_convolution.cl"
      ;
  code +=
#include "kernels/thermal.cl"
      ;
  code +=
#include "kernels/thermal_flatten.cl"
      ;
  code +=
#include "kernels/thermal_inflate.cl"
      ;
  code +=
#include "kernels/thermal_olsen.cl"
      ;
  code +=
#include "kernels/thermal_rib.cl"
      ;
  code +=
#include "kernels/thermal_ridge.cl"
      ;
  code +=
#include "kernels/thermal_schott.cl"
      ;
  code +=
#include "kernels/thermal_scree.cl"
      ;
  code +=
#include "kernels/topographic_position_index.cl"
      ;
  code +=
#include "kernels/vorolines.cl"
      ;
  code +=
#include "kernels/voronoi_base.cl"
      ;
  code +=
#include "kernels/voronoi_edge_distance.cl"
      ;
  code +=
#include "kernels/voronoi_fbm.cl"
      ;
  code +=
#include "kernels/voronoi_main.cl"
      ;
  code +=
#include "kernels/voronoise.cl"
      ;
  code +=
#include "kernels/vororand_main.cl"
      ;
  code +=
#include "kernels/warp.cl"
      ;
  code +=
#include "kernels/water_depth_filter.cl"
      ;
  code +=
#include "kernels/wavelet_noise.cl"
      ;
  //
  code +=
#include "kernels/rifts.cl"
      ;
  code +=
#include "kernels/strata.cl"
      ;
  code +=
#include "kernels/strata_cells.cl"
      ;
  code +=
#include "kernels/strata_terrace.cl"
      ;

  std::string opencl_build_options = "-cl-fast-relaxed-math "
                                     "-cl-mad-enable "
                                     "-cl-no-signed-zeros "
                                     "-cl-denorms-are-zero "
                                     "-cl-finite-math-only ";

  clwrapper::KernelManager::get_instance().set_build_options(
      opencl_build_options);

  constexpr bool clear_sources = true;
  clwrapper::KernelManager::get_instance().add_kernel(code, clear_sources);

  return true;
}

} // namespace hmap::gpu
