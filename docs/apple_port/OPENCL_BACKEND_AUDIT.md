# OpenCL backend audit

## Runtime architecture

HighMap's only current GPU abstraction is the `clwrapper` submodule.
`hmap::gpu::init_opencl()` in `HighMap/src/gpu_opencl/gpu_opencl.cpp` first
checks `DeviceManager::is_ready()`, clears the singleton `KernelManager`, sets
fast-math build options, appends 81 kernel source files (including four common
headers), and builds one OpenCL program. The kernel sources are included as C++
raw strings at compile time.

`DeviceManager` enumerates platforms/devices, scores GPUs ahead of accelerators
and CPUs, then selects the highest score. `KernelManager` creates an OpenCL
context for the selected device and compiles the concatenated source. Both are
singletons protected by shared mutexes, but a `Run` owns its own command queue.

`clwrapper::Run` does the following:

1. Constructor creates a command queue and looks up one kernel by name.
2. `bind_arguments` assigns sequential scalar kernel arguments.
3. `bind_buffer` allocates a fresh `cl::Buffer` and stores a raw host vector
   pointer; `write_buffer` is a blocking `enqueueWriteBuffer`.
4. `bind_imagef` allocates a fresh `cl::Image2D`; input images use
   `CL_MEM_COPY_HOST_PTR`, output images are write-only; `write_imagef` and
   `read_imagef` are blocking operations.
5. `execute` rounds a 1D/2D global range to a multiple of eight, enqueues the
   kernel, calls `finish`, and optionally reports host wall time around the
   finish.
6. Destructor calls `queue.finish()` again.

There is no queue/context/resource pool, no event graph, no pipeline cache at
the HighMap layer, no asynchronous readback, and no representation of device
resident HighMap arrays.

## Kernel inventory and C++ call sites

The following groups cover every OpenCL source file and entry point in this
snapshot. The C++ implementation column is the source file that constructs
the `Run`; repeated uses of a kernel are grouped.

| Kernel source / entry points | C++ caller(s) | Access pattern and migration class |
|---|---|---|
| `_common_index.cl`, `_common_math.cl`, `_common_rand.cl`, `_common_sort.cl` | Included by `gpu_opencl.cpp` | Shared indexing, math, hashing, atomic helpers and sorting; translate selectively into MSL inline functions |
| `gradient_norm.cl: gradient_norm` | `gradient/gradient_gpu.cpp` | Buffer, 4-neighbor read, pointwise output; simple |
| `advection_warp.cl: advection_warp` | `transform/advection_warp_gpu.cpp` | Six images, linear sampling, bounded path array and reverse walk; intermediate |
| `advection_particle.cl: advection_particle` | `transform/advection_particle_gpu.cpp` | Particle state and irregular path behavior; complex |
| `warp.cl: warp_x`, `warp_y`, `warp_xy` | `transform/warp_gpu.cpp` | Image linear sampling; simple/intermediate |
| `rotate.cl: rotate_hmap` | `transform/transform_gpu.cpp` | Image transform; simple |
| `noise_a.cl`, `noise_b.cl: noise`, `noise_fbm` | `primitives/coherent_noise/primitives_gpu.cpp` | Hash/noise functions and per-pixel octave loops; simple/intermediate after parity definition |
| `gabor_wave.cl`, `gavoronoise.cl`, `vorolines.cl`, `voronoi_*`, `voronoise.cl`, `vororand_main.cl`, `wavelet_noise.cl` | `primitives/coherent_noise/primitives_gpu.cpp` | Procedural per-pixel loops, optional maps, substantial branching; intermediate/complex |
| `hemisphere_field.cl`, `mountain_range_radial.cl`, `polygon_field.cl` | `primitives/coherent_noise/primitives_gpu.cpp` | Procedural fields and optional inputs; intermediate |
| `local_max.cl`, `local_min.cl`, `local_mean.cl`, `local_relief.cl`, `local_skewness.cl`, `local_variance.cl`, `local_z_score.cl`, `ruggedness.cl`, `rugosity.cl`, `topographic_position_index.cl` | `local_metrics/local_metrics_gpu.cpp` | Window neighborhood/reductions; a reusable Metal window layer is possible |
| `curvature_quadric.cl` | `curvature/curvature_quadric.cpp` | Neighborhood fit and optional smoothing; intermediate |
| `smooth_cpulse.cl`, `maximum_smooth.cl`, `minimum_smooth.cl`, `expand.cl`, `laplace.cl`, `mean_shift.cl`, `median_3x3.cl`, `bilateral_filter.cl`, `directional_blur.cl`, `normal_displacement.cl`, `plateau.cl`, `project_slope_along_direction.cl` | `filters/filters_gpu.cpp`, `range/range_gpu.cpp`, `filters/bilateral_filter.cpp`, `filters/directional_blur.cpp` | Window/multi-pass filters; simple to complex depending on window and iterations |
| `sparse_max_convolution.cl` | `convolve/sparse_max_convolution.cpp` | Atomic max/reduction over sparse kernel; complex, negative values are already called out as an OpenCL issue |
| `blend_poisson_bf.cl` | `blending/blending_gpu.cpp` | Iterative Poisson-like/blend pass, composed with filters; GPU-residency candidate |
| `interpolate_array.cl` | `interpolate/interpolate_array_gpu.cpp` | Nearest/bilinear/bicubic/Lagrange image sampling; intermediate |
| `harmonic_interpolation.cl` | `interpolate/harmonic_interpolation_gpu.cpp` | Red-black iterative solver, two kernels; synchronization between colors/iterations is required |
| `phase_field.cl`, `phase_averaging.cl`, `laplacian_fract.cl` | `gradient/phase_field.cpp`, `gradient/gradient_gpu.cpp` | Iterative phase/neighbor passes; intermediate/complex |
| `hydraulic_vpipes.cl: hydraulic_vpipes_flow_pass`, `...water_pass`, `...erosion_pass`, `...sediment_transport_pass` | `erosion/hydraulic_vpipes_gpu.cpp`, `hydrology/flow_simulation.cpp` | Four dependent image passes per iteration; current implementation is CPU-synchronized after each pass |
| `thermal.cl: thermal`, `thermal_with_bedrock`, `thermal_auto_bedrock` | `erosion/thermal_gpu.cpp` | In-place buffer neighbor reads; host loops over iterations, each dispatch blocks |
| `thermal_flatten.cl`, `thermal_inflate.cl`, `thermal_olsen.cl`, `thermal_rib.cl`, `thermal_ridge.cl`, `thermal_schott.cl`, `thermal_scree.cl` | `erosion/thermal_gpu.cpp` | Iterative neighborhood updates and optional deposition maps; host loops block per iteration |
| `hydraulic_schott.cl` | `erosion/hydraulic_schott_gpu.cpp` | Iterative hydraulic and erosion stages; multiple arrays and readbacks |
| `hydraulic_mcdonald.cl` | `erosion/hydraulic_mcdonald_gpu.cpp` | Solver/filter/debris stages; iterative and stateful |
| `hydraulic_particle.cl` | `erosion/hydraulic_particle.cpp` | Particle-like irregular flow/erosion; branch-heavy |
| `flow_direction_d8.cl` | `hydrology/flow_accumulation_d8_gpu.cpp` | Direction selection from neighbors; graph dependency |
| `flow_accum_stochastic.cl` | `hydrology/flow_accumulation_stochastic_gpu.cpp` | Stochastic solve plus normalization; reductions and graph convergence |
| `eulerian_transport.cl`, `shallow_viscous_flow.cl` | `hydrology/flow_accumulation_from_velocity_field.cpp`, `hydrology/flow_simulation.cpp` | Transport/flow neighborhood passes; iterative |
| `water_depth_filter.cl`, `snow_simulation.cl`, `coastal_fetch.cl`, `generate_riverbed.cl` | Hydrology source modules | Pointwise/neighborhood hydrology; mixed |
| `jump_flooding.cl`, `skeleton.cl` | `morphology/distance_transform_jfa.cpp`, `morphology/morphology_gpu.cpp` | Multi-step jump flooding/thinning; synchronization and branch-heavy |
| `sdf_2d_polyline.cl` | `sdf/sdf_2d_polyline_gpu.cpp` | Per-pixel segment distance; parallel but loop/branch heavy |
| `rifts.cl`, `strata.cl`, `strata_cells.cl`, `strata_terrace.cl` | `erosion/rifts_gpu.cpp`, `erosion/strata.cpp` | Procedural/neighbor terrain transforms; intermediate |

The source inventory above includes all functional `.cl` files; the four
underscore-prefixed files are common source fragments rather than kernels.

## Exact synchronization and transfer observations

The following are direct consequences of reading all current GPU callers and
the `Run` implementation:

| Pattern | Evidence | Cost/impact |
|---|---|---|
| Fresh allocation per invocation | Every `Run::bind_buffer`/`bind_imagef` creates a new `cl::Buffer`/`cl::Image2D` | Repeated allocation and image setup; no resource reuse |
| Blocking input upload | `bind_imagef(IN)` copies at construction; callers also call `write_buffer` after binding buffers | Host-to-device transfer is part of each operation |
| Blocking output readback | Callers invoke `read_buffer`/`read_imagef` before returning or before the next stage | Every high-level operation returns host data; composed GPU functions repeatedly cross the boundary |
| Dispatch wait | Both `execute` overloads call `queue.finish()` | Kernel timing includes a mandatory host/device synchronization |
| Destructor wait | `Run::~Run()` calls `queue.finish()` | Additional defensive wait and queue lifetime per pass |
| Thermal iteration loop | `thermal_gpu.cpp` sets the iteration argument then calls `execute` once per iteration | Host submits and waits for every iteration; state remains in an OpenCL allocation but not under an explicit multi-pass graph |
| Hydraulic virtual-pipe loop | `hydraulic_vpipes_gpu.cpp` launches flow, reads four outputs; launches water, reads three; launches erosion, reads two; launches sediment, reads one; computes water volume with CPU `sum()` | At least ten blocking image transfers/synchronization boundaries per iteration, plus four `Run` objects and allocations |
| Hydraulic water conservation | `d = d2 * ...`, then `d.sum()` and host-side redistribution when `maintain_water_volume` is true | A reduction and state update are necessarily host-visible in the current code; Metal can move this to a reduction/normalization pass |
| Composed morphology/filter functions | `morphology_gpu.cpp`, `filters_gpu.cpp`, `local_metrics_gpu.cpp`, and blending code call other `gpu::` functions that each read back | High-level GPU compositions do not retain intermediate state on device |
| Harmonic red/black solver | GPU dispatches alternate color passes in a host loop | Host controls convergence/iteration and synchronizes each pass |
| Advection | One dispatch/readback, but each thread walks a path in private local storage up to `MAX_STEPS=1024` | No transfer loop, but register/local memory pressure and divergence are likely bottlenecks |

## OpenCL correctness risks to preserve/document

* Many kernels use image address modes (`CLAMP_TO_EDGE`, mirrored repeat) while
  other kernels use raw buffer indices. Metal ports must reproduce the exact
  boundary behavior, not merely the arithmetic.
* `thermal.cl` writes `z` while neighboring work-items read `z` in the same
  dispatch. The host iteration boundary is synchronized, but there is no
  per-dispatch ordering guarantee. A Metal implementation should use ping-pong
  buffers for deterministic behavior and compare aggregate/perceptual results
  against the existing reference.
* OpenCL is built with `-cl-fast-relaxed-math`, `-cl-mad-enable`, no signed
  zeros, denormals-as-zero and finite-math-only. Metal parity tests therefore
  need tolerances and error statistics rather than bit identity.
* The wrapper rounds global sizes to multiples of eight and kernels guard
  against out-of-range work-items. Metal should retain the guards while using a
  capability-appropriate threadgroup size.

## Highest-value restructuring targets

1. Hydraulic virtual pipes: one Metal context, reusable buffers, four encoded
   passes per iteration, one optional GPU reduction/normalization, final
   readback only.
2. Thermal erosion: ping-pong state buffers and a command-buffer loop without
   host wait per iteration.
3. Filter/morphology chains: a `DeviceArray` or explicit Metal buffer handle
   so composed operations do not return through `Array` after every pass.
4. Harmonic interpolation: red/black passes encoded with explicit barriers or
   separate command encoders and a device-side convergence strategy.

The first implementation deliberately starts with independent operations and
then uses hydraulic/thermal to validate the multi-pass design.
