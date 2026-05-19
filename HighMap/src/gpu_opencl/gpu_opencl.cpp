/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <vector> // for vector

#include "cl_wrapper/device_manager.hpp" // for DeviceManager
#include "cl_wrapper/kernel_manager.hpp" // for KernelManager

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
  const std::string code =
#include "kernels/_common_index.cl"
#include "kernels/_common_math.cl"
#include "kernels/_common_rand.cl"
#include "kernels/_common_sort.cl"
  //
#include "kernels/advection_particle.cl"
#include "kernels/advection_warp.cl"
#include "kernels/bilateral_filter.cl"
#include "kernels/blend_poisson_bf.cl"
#include "kernels/coastal_fetch.cl"
#include "kernels/curvature_quadric.cl"
#include "kernels/directional_blur.cl"
#include "kernels/eulerian_transport.cl"
#include "kernels/expand.cl"
#include "kernels/flow_direction_d8.cl"
#include "kernels/gabor_wave.cl"
#include "kernels/gavoronoise.cl"
#include "kernels/generate_riverbed.cl"
#include "kernels/gradient_norm.cl"
#include "kernels/harmonic_interpolation.cl"
#include "kernels/hemisphere_field.cl"
#include "kernels/hydraulic_particle.cl"
#include "kernels/hydraulic_schott.cl"
#include "kernels/hydraulic_vpipes.cl"
#include "kernels/interpolate_array.cl"
#include "kernels/jump_flooding.cl"
#include "kernels/laplace.cl"
#include "kernels/laplacian_fract.cl"
#include "kernels/local_max.cl"
#include "kernels/local_mean.cl"
#include "kernels/local_min.cl"
#include "kernels/local_relief.cl"
#include "kernels/local_skewness.cl"
#include "kernels/local_variance.cl"
#include "kernels/local_z_score.cl"
#include "kernels/maximum_smooth.cl"
#include "kernels/mean_shift.cl"
#include "kernels/median_3x3.cl"
#include "kernels/minimum_smooth.cl"
#include "kernels/mountain_range_radial.cl"
#include "kernels/noise.cl"
#include "kernels/normal_displacement.cl"
#include "kernels/phase_averaging.cl"
#include "kernels/phase_field.cl"
#include "kernels/plateau.cl"
#include "kernels/polygon_field.cl"
#include "kernels/project_slope_along_direction.cl"
#include "kernels/rotate.cl"
#include "kernels/ruggedness.cl"
#include "kernels/rugosity.cl"
#include "kernels/sdf_2d_polyline.cl"
#include "kernels/shallow_viscous_flow.cl"
#include "kernels/skeleton.cl"
#include "kernels/smooth_cpulse.cl"
#include "kernels/snow_simulation.cl"
#include "kernels/sparse_max_convolution.cl"
#include "kernels/thermal.cl"
#include "kernels/thermal_flatten.cl"
#include "kernels/thermal_inflate.cl"
#include "kernels/thermal_olsen.cl"
#include "kernels/thermal_rib.cl"
#include "kernels/thermal_ridge.cl"
#include "kernels/thermal_schott.cl"
#include "kernels/thermal_scree.cl"
#include "kernels/topographic_position_index.cl"
#include "kernels/vorolines.cl"
#include "kernels/voronoi_base.cl"
#include "kernels/voronoi_edge_distance.cl"
#include "kernels/voronoi_fbm.cl"
#include "kernels/voronoi_main.cl"
#include "kernels/voronoise.cl"
#include "kernels/vororand_main.cl"
#include "kernels/warp.cl"
#include "kernels/water_depth_filter.cl"
#include "kernels/wavelet_noise.cl"
//
#include "kernels/rifts.cl"
#include "kernels/strata.cl"
#include "kernels/strata_cells.cl"
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
