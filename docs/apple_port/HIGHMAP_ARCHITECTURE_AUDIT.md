# HighMap architecture audit for Apple Silicon

## Scope and source snapshot

This audit was performed against the `dev` branch at commit
`2cc066e1002f598e10bb4f493ac2a811e8e09300` (`fix(gpu): correct sampling offset
and regularization in hydraulic_vpipes kernel`). The working branch is
`feature/apple-metal-backend`. The repository was cloned with all submodules;
the OpenCL wrapper at this snapshot is `7be8d7448f9c0f108896765ba94ad9ec3099061f`.

The host used for the initial audit is arm64 macOS 27.0, Apple Clang
21.0.0.21000327, CMake 4.1.2. No source or public API changes are assumed by
this document unless explicitly called out in the implementation documents.

## Architectural summary

HighMap is a C++20 static library whose primary data object is `hmap::Array`.
An array owns a column-major-in-concept / x-fastest flat `std::vector<float>`:
`vector[j * shape.x + i]`. Most CPU algorithms operate directly on this
storage. The public surface is organized by domain headers and is re-exported
from `highmap.hpp`.

The current GPU path is not an abstraction layer. Feature-specific `*_gpu.cpp`
files construct `clwrapper::Run` objects directly, bind host vectors to new
OpenCL buffers or images, execute synchronously, and read results back into
`Array`. This makes the existing path easy to call but prevents GPU resource
reuse and makes multi-pass composition expensive.

## Public data and utility layers

| Area | Public API/header | Implementation and behavior | Data/algorithm notes | Apple suitability |
|---|---|---|---|---|
| Array/data model | `Array`, `array.hpp` | `array/array.cpp`, `methods.cpp`, `array_functions.cpp`, `io.cpp`, `opencv_wrapper.cpp` | 2D `float`, flat contiguous storage, explicit shape, arithmetic operators, reductions, slices, bilinear/cubic sampling, normalization, OpenCV conversion and file I/O | Excellent CPU cache locality; a future GPU handle must preserve simple host semantics |
| Algebra/math | `algebra.hpp`, `math.hpp`, `math/*` | `algebra.cpp`, `math.cpp`, `core.cpp`, profile/distance helpers | Pointwise arithmetic, scalar/vector helpers, profiles, distance functions, Chebyshev and reductions | Auto-vectorizable where loops are contiguous; avoid Apple-only replacements without measurements |
| Operators | `operator.hpp` | `operator/operator.cpp`, `compare.cpp`, `vector.cpp`, `fill_array.cpp`, `inpainting_gaussian.cpp`, stitching helpers | Pointwise comparisons, masks, fill, vector fields, Gaussian inpainting | Mostly CPU-friendly; selected pointwise operations can share a GPU buffer abstraction |
| Boundaries | `boundary.hpp` | `boundary/*` | Border extrapolation, zeroed edges, boundary picking | Small/branch-heavy; CPU usually wins at small sizes |
| Random/statistics | `random.hpp`, `statistics.hpp` | `random/*`, `statistics/*` | PDF sampling, normalization, histograms, detrending, autocorrelation | Reductions and random state need dedicated GPU strategies; not first Metal targets |
| OpenMP | `openmp.hpp` | `openmp/openmp.cpp` | Runtime diagnostics and thread-count initialization | Useful for CPU algorithms; Apple Clang needs external `libomp` |
| Texture/color | `texture.hpp`, `colorize.hpp`, `colormaps.hpp` | `tensor/texture.cpp`, `colorize/*`, colormap data | CPU texture/color operations, normal maps and image-like results | CPU/OpenCV path; Metal textures are a later interoperability option, not an Array replacement |
| Virtual arrays | `virtual_array.hpp` and subheaders | `virtual_array/*` | Tiled/disk-backed arrays, LRU/ram storage, virtual textures and tile regions | GPU residency must be tile-aware; do not force whole-array uploads |

## Procedural generation and primitives

| Feature | Public API | CPU/GPU implementation | Data and GPU characteristics | Migration priority |
|---|---|---|---|---|
| Coherent noise | `primitives/coherent_noise.hpp` | CPU `primitives/coherent_noise/noise.cpp`, `fbm_functions.cpp` and helpers; OpenCL `noise_a.cl`, `noise_b.cl`, Voronoi/Gabor/Worley families through `primitives_gpu.cpp` | One output float per pixel; deterministic hash/noise; optional displacement/control arrays; FBM is per-pixel iterative | High: embarrassingly parallel, good Metal candidate after a simple kernel |
| Procedural scalar functions | `functions.hpp` | `functions/*`, field/fbm/Parberry implementations | Pointwise/field evaluation with optional coordinate transforms | High for pointwise kernels, but API dispatch should remain CPU-safe |
| Primitive functions | `primitives/functions.hpp` | band, cone, checkerboard, multisteps, polar, swirl, wave | Pointwise analytic fields | High, low migration complexity |
| Geo primitives | `primitives/geo.hpp` | mountain, island, crater, rift, caldera, badlands, plates, etc. | Compositions of noise, masks, filters and sometimes GPU noise; branch and multi-pass mix | Medium; port constituents first |
| Random primitives | `primitives/random.hpp` | white-noise CPU implementation | Pointwise random map | High, but seed parity must be specified |
| Geometry/points/clouds | `geometry.hpp` and subheaders | `geometry/*`, kd-tree/nanoflann/delaunay dependencies | Paths, grids, graphs, clouds, point sampling and spatial queries | Mostly CPU/branch-heavy; GPU only for carefully isolated maps |
| Roads/authoring | `roads.hpp`, `authoring.hpp` | `roads/*`, `authoring/*` | Path/elevation constraints, stamps, ridgelines, sparse constraints | CPU orchestration with GPU primitives as substeps |

## Transform, blending, filters, and local analysis

| Feature | Public API | CPU implementation | Existing GPU implementation | GPU shape |
|---|---|---|---|---|
| Warp/transform | `transform.hpp` | `transform/*`, texture transforms | `transform_gpu.cpp` (`rotate_hmap`), `warp_gpu.cpp` (`warp_x/y/xy`), `advection_warp_gpu.cpp`, `advection_particle_gpu.cpp` | Neighborhood/texture sampling; advection has a bounded per-thread path and significant branching |
| Blending | `blending.hpp` | `blending/blending.cpp` | `blending_gpu.cpp` plus `blend_poisson_bf.cl` and composed smoothing | Multi-pass, some reductions/iterative Poisson behavior; GPU benefit depends on residency |
| Range/pointwise filters | `range.hpp`, `filters.hpp` | `range/*`, `filters/*` | `range_gpu.cpp`, `filters_gpu.cpp` | Local windows, separable/multi-pass smoothing, masks; `maximum_smooth`/`minimum_smooth` are pointwise and are the first Metal filter slice |
| Convolution | `convolve.hpp` | `convolve/*` | `sparse_max_convolution.cl` invoked by CPU implementation | Neighborhood/reduction and atomic max for sparse kernels; Metal needs a distinct reduction/atomic design |
| Bilateral/directional filters | `filters.hpp` | CPU helpers | `bilateral_filter.cl`, `directional_blur.cl` | Neighborhood sampling, branch-heavy; texture/buffer choice must be profiled |
| Morphology | `morphology.hpp` | `morphology/*` | `morphology_gpu.cpp`: local extrema, reconstruction, skeleton/relative distance and composed filters | Window operations plus iterative reconstruction; GPU resident chains are valuable |
| Gradients/normals | `gradient.hpp` | `gradient/gradient.cpp`, normal/phase/Poisson solvers | `gradient_gpu.cpp` (`gradient_norm`, `laplacian_fract`) | Local 4/8-neighbor reads, regular dispatch, excellent first kernel |
| Curvature | `curvature.hpp` | quadric and level-set curvature CPU code | `curvature_gpu.cpp`, `curvature_quadric.cl` | Neighborhood fit/branching; medium complexity |
| Local metrics | `local_metrics.hpp` | `local_metrics/*` | `local_metrics_gpu.cpp`: max/min/mean/relief/skewness/variance/z-score/ruggedness/rugosity/TP index | Window reads; some metrics are reductions; good family for shared Metal helpers |
| SDF | `sdf.hpp` | polyline CPU routines | `sdf_2d_polyline_gpu.cpp`, SDF kernels | Per-pixel segment loops; potentially GPU-friendly but branch-heavy |

## Erosion, hydraulic simulation, and hydrology

| Feature | Public API | CPU implementation | GPU implementation/kernels | Iteration/reduction/sampling assessment |
|---|---|---|---|---|
| Thermal erosion | `erosion.hpp` | `thermal_schott.cpp` and other CPU variants | `thermal_gpu.cpp`; `thermal.cl`, `thermal_flatten.cl`, `thermal_inflate.cl`, `thermal_olsen.cl`, `thermal_rib.cl`, `thermal_ridge.cl`, `thermal_schott.cl`, `thermal_scree.cl` | Iterative; 4/8-neighbor reads; current in-place updates are synchronized only between dispatches and have intra-pass ordering sensitivity |
| Hydraulic virtual pipes | `erosion.hpp` | CPU simulation in `hydraulic_vpipes` support paths | `hydraulic_vpipes_gpu.cpp`; four passes in `hydraulic_vpipes.cl` | Iterative, four dependent passes, neighborhood reads, optional global `sum()` on CPU for water-volume conservation; highest value GPU-residency target |
| Hydraulic Schott | `erosion.hpp` | CPU and `hydraulic_schott.cpp` | `hydraulic_schott_gpu.cpp`, `hydraulic_schott.cl` | Iterative multi-stage and sediment/flow state; current GPU path has repeated command work |
| Hydraulic McDonald | `erosion.hpp` | CPU orchestration | `hydraulic_mcdonald_gpu.cpp`, `hydraulic_mcdonald.cl` | Iterative solver/filter/debris stages; reductions and multiple intermediate maps |
| Hydraulic particle | `erosion.hpp` | particle path in `hydraulic_particle.cpp` | `hydraulic_particle.cl` | Many independent particles, random/branch-heavy, irregular writes; lower initial suitability |
| Hydraulic stream/procedural | `erosion.hpp` | `hydraulic_stream.cpp`, `hydraulic_procedural.cpp`, `deposition.cpp` | composed GPU maps and stream log helpers | Multi-pass orchestration; benefit primarily from chaining residency |
| Other erosion | `erosion.hpp` | coastal, mudslide, depression fill, bedrock, rifts, strata, valley fill, convective erosion | Some use GPU rifts/strata/thermal/filter primitives | Mixed; priority-flood and graph-like portions remain CPU candidates |
| Flow direction/accumulation | `hydrology.hpp` | `hydrology/*` | `flow_direction_d8.cl`, `flow_accum_stochastic.cl` and GPU wrappers | Graph dependency, atomics/reductions, irregular convergence; complex Metal migration |
| Flow simulation | `hydrology.hpp` | flow simulation CPU code | virtual-pipe and shallow-viscous kernels | Iterative neighborhood updates; GPU resident state is promising |
| Flooding/lakes/basins | `hydrology.hpp` | boundary flood, depression filling, watershed code | Mostly CPU | Priority queues/connected components; CPU likely competitive unless redesigned |
| Snow/coastal | `hydrology.hpp`, `erosion.hpp` | CPU implementations | `snow_simulation.cl`, coastal kernels | Multi-pass and masks; later targets |

## Interpolation, mesh, import/export, and dependencies

| Area | Implementation | Notes |
|---|---|---|
| Interpolation | `interpolate/*`, `interpolate_array_gpu.cpp`, `harmonic_interpolation_gpu.cpp` | 1D/2D methods include nearest, bilinear, bicubic, Lagrange, IDW, Gaussian, Delaunay, natural-neighbor and harmonic red/black iterations. GPU interpolation has neighborhood sampling; harmonic interpolation is iterative and GPU-friendly only if all iterations stay device-side. |
| Mesh/terrain analysis | `terrain_tri_mesh/*`, `features/*`, `shortest_path/*`, `geometry/*` | Triangulation, paths, connected components, geomorphons and clustering are irregular/branch-heavy and should remain CPU-first. |
| Import/export | `export/*`, `array/io.cpp`, OpenCV wrapper | PNG/EXR/TIFF/RAW/ASCII/PLY/asset/splatmap/cubemap and OpenCV conversions. These are host-facing and should force explicit readback at the API boundary. |
| OpenCV | Root `find_package(OpenCV)`, `Array` wrapper | Image I/O/conversion, image processing support. Avoid making Metal resources the default OpenCV interchange type. |
| GSL | Root `find_package(GSL)` | Numerical support used by library algorithms; host-side. |
| Assimp | Root `find_package(assimp)` | Asset/mesh import/export; host-side. |
| GLM | Root and public headers | Vector/math types; host-side API ABI. |
| FastNoiseLite/Noise | Submodules | Noise implementations and NoiseLib; NoiseLib currently requires OpenMP. |
| Delaunator/dkm/hmm/libnpy/nanoflann/nn-c/PointSampler/terrain-descriptors/mixbox | `external/` and CMake | Geometry, clustering, array interchange, sampling and color helpers; mostly CPU. |
| CLWrapper/CLErrorLookup | `external/CLWrapper` | Existing OpenCL device/program/run layer; must remain available when OpenCL is enabled. |

## Main architectural conclusions

1. `Array` is intentionally a simple host value type. A separate GPU handle or
   execution context is needed to get residency without making every existing
   API asynchronous.
2. The first Metal slice should use flat `device float` buffers. It matches the
   existing `Array` layout, avoids premature texture conversion, and is enough
   for gradient, noise, advection and erosion kernels. Texture-backed resources
   can be added where sampling hardware measurably helps.
3. The high-value difference is command/resource lifetime: current OpenCL
   `Run` allocates and synchronously copies for each invocation. A Metal context
   should cache pipeline states and offer explicit multi-pass command encoding.
4. CPU, OpenCL and Metal implementations must remain separate at the lowest
   level, with a small selection layer in high-level GPU entry points. Existing
   public signatures can remain unchanged.
5. Iterative thermal/hydraulic algorithms need explicit ping-pong resources in
   Metal. In-place neighborhood writes should not be assumed deterministic.
6. Reduction-heavy and graph-like algorithms should not be ported merely for
   coverage. Benchmarks and parity tests determine whether a Metal path is
   justified.
