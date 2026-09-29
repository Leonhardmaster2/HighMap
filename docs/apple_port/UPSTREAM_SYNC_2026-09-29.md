# Upstream sync and macOS re-optimisation — 2026-09-29

Branch `feature/apple-metal-backend` was 103 commits behind `upstream/dev`.
It now merges `upstream/dev` at `7f439892f` (TWI, 2026-09-29), which also
contains `89f329403`, the HighMap revision pinned by upstream Hesiod `dev`.
The pre-sync tip is kept as `backup/apple-metal-pre-sync-2026-09-29`.

## Merge

| Area | Resolution |
|---|---|
| CMake | Kept optional OpenCL / Metal / OpenMP. Adopted upstream's FetchContent dependencies and the new OpenEXR, TIFF, nlohmann_json, zstd and LightUSD requirements. CLWrapper stays behind `HIGHMAP_ENABLE_OPENCL`. |
| `gpu::gradient_norm`, `gpu::maximum_smooth`, `gpu::minimum_smooth` | Removed upstream (pointwise work moved back to the CPU). The public hooks are gone; `metal::` and `DeviceSession` versions remain. Tests and benchmarks that used the deleted OpenCL kernels now compare against the CPU. |
| VirtualArray | Took upstream's reorganised sources; re-applied the `halo == 1` divide-by-zero guard in `sync_overlap_buffers`. |
| New upstream GPU sources | `jagged`, `recast_cliff` and `hydraulic_musgrave_gpu` include `internal/opencl_run.hpp`, so OpenCL-disabled builds still compile. |
| Tests | Upstream's test used `va_copy` as a variable name, which is a `<stdarg.h>` macro on Apple Clang; renamed. |
| Local fixes | The uncommitted local CPU/VirtualArray fixes (convolve cache order, `smooth_flat` normalisation, sequential-storage clone ownership) were committed before the merge. |

## Semantic drift fixed

Upstream's thermal kernels changed how they handle borders: each pass now copies
the adjacent interior cell into the border (`apply_boundaries_io`), and
diagonals use `1.41421356`. The Metal `thermal_pass` and `thermal_ridge_pass`
still used the old rule (border unchanged, `1.414`), so macOS results drifted
from upstream near the edges. Both kernels now follow the new rule. A direct
Metal-vs-OpenCL test (`MetalBackend.ThermalMatchesOpenCLKernel`) covers 1, 2
and 9 iterations.

## New Metal routes

| Public wrapper | Notes |
|---|---|
| `gpu::smooth_cpulse` | 46 internal call sites. The kernel reads a host-computed weight table; before, it evaluated `exp()` for every tap of every pixel. |
| `gpu::local_max` / `gpu::local_min` (DISK, SQUARE, OCTAGON) | Brings dilation, erosion, opening, closing, top-hat, black-hat, `morphological_operators` and `local_relief` onto Metal. All separable passes run in one command buffer (the OpenCL octagon path round-trips through the host 4×). Matches the brute-force reference and the OpenCL disk kernel exactly. |
| `gpu::morphological_gradient` (DISK) | Fused max − min kernel. |
| `gpu::noise_fbm` | PERLIN, PERLIN_BILLOW, PERLIN_HALF, SIMPLEX2, VALUE, VALUE_LINEAR. Raw output is within 3.2e-6 of OpenCL for all control-map, warp, bbox and period variants. |

## Synchronous buffer reuse

Every synchronous Metal call used to allocate fresh shared buffers and pay
first-touch page faults on the upload memcpy. These calls now draw from a
small, mutex-protected cache, bounded to min(256 MB, working set / 8) and 24
entries. An RAII `OperationScope` returns the buffers once the operation has
waited for its command buffer. Sync `smooth_cpulse` and `noise_fbm` now
encode on those buffers directly instead of going through a `DeviceSession`.
An 8-thread stress test (`MetalRoutes.ConcurrentSyncCallsShareBufferCacheSafely`)
passes 30/30 repeated runs.

`highmap_benchmarks --benchmark_filter=BM_Route_ --benchmark_repetitions=5`,
median wall time, radius 16, Apple M3 / 8 GB, Release. Each call includes
upload and readback. The OpenCL side runs the upstream host code (including
its host round trips); the Metal side calls the public `hmap::gpu` wrapper.

| Operation | Size | OpenCL | Metal | Speed-up |
|---|---|---:|---:|---:|
| `smooth_cpulse` | 1024² | 9.08 ms | 3.18 ms | 2.9× |
| | 2048² | 22.59 ms | 5.36 ms | 4.2× |
| | 4096² | 78.86 ms | 19.93 ms | 4.0× |
| `local_max` DISK | 1024² | 20.61 ms | 9.06 ms | 2.3× |
| | 2048² | 83.24 ms | 33.48 ms | 2.5× |
| | 4096² | 331.06 ms | 132.62 ms | 2.5× |
| `local_max` OCTAGON | 1024² | 8.17 ms | 1.53 ms | 5.3× |
| | 2048² | 24.06 ms | 5.48 ms | 4.4× |
| | 4096² | 92.79 ms | 20.43 ms | 4.5× |

Before buffer reuse, a direct probe measured sync Metal `smooth_cpulse` at
~12 ms (2048²) and ~45 ms (4096²).

The benchmarks live in `benchmarks/src/bm_apple_routes.cpp`.

## Test matrix (Release, Apple M3)

| Configuration | Passed | Skipped | Failed |
|---|---:|---:|---:|
| Metal ON / OpenCL ON | 669 | 2 | 3 |
| Metal OFF / OpenCL ON | 613 | 58 | 3 |
| Metal ON / OpenCL OFF | 581 | 92 | 1 (`PathSplines`) |

These numbers were recorded right after the merge. After the stability fixes
below, Metal ON / OpenCL ON is 673 passed, 2 skipped, 2 failed (`PathSplines`
and the upstream roughness CPU/GPU mismatch; `ConvErosion.BasicExecution` now
passes). `HIGHMAP_DISABLE_METAL=1` gives 617 passed, 58 skipped, the same 2
failures.

For the Metal-only configuration, 28 new upstream tests (jagged, recast
cliff, Musgrave GPU, conv-erosion, GPU roughness, convolution scaling) call
OpenCL-only kernels. They now start with `HMAP_SKIP_IF_NO_OPENCL()`, like the
fork's earlier tests, instead of failing on the intended "OpenCL disabled"
exception.

The same three tests fail in both OpenCL configurations, so none of them
comes from the Metal backend:

* `PathSplines.PreservePathShape`: pre-existing and documented earlier.
* `ConvErosion.BasicExecution`: upstream's conv-erosion is marked WIP
  (`61a502cea`), and the output is unchanged by the call.
* `LocalMetrics.Roughness_CpuGpuEquivalenceAndWrapper`: upstream's GPU
  roughness now adds `local_mean` + smoothing + border extrapolation, while
  the CPU version does not, so the CPU/GPU equality in the test cannot hold.

## Upstream issue noticed

Upstream `gpu::hydraulic_vpipes` (OpenCL) still calls the kernels
`hydraulic_vpipes_flow_simulation`, `_velocity`, `_erosion_deposition` and
`_advection`, none of which exist in the upstream kernel set any more. On
macOS the Metal route bypasses that code, so the Metal implementation is the
only working GPU path for this function.

## Stability fixes found through Hesiod (second pass)

Hesiod was run headless over all 258 example graphs. Each graph was also
evaluated with Metal on and with `HIGHMAP_DISABLE_METAL=1`, and every node
output was dumped and diffed. That turned up these HighMap issues, all present
upstream as well:

| Issue | Symptom | Fix |
|---|---|---|
| `RamTileStorage::get_tile` inserted into an unguarded `unordered_map` while `VA_DISTRIBUTED` workers created tiles lazily | SIGSEGV in `VirtualArray::to_array`, heap corruption, or a hung graph update in 9 of 258 examples | Mutex, same as the LRU storages; clone copies under the lock |
| nn-c / `triangle.c` keep global state | Natural-neighbour interpolation is unsafe from concurrent tile workers | One process-wide mutex around all nn-c calls |
| `remap` computed `x * scale + (vmin - min * scale)` | Cancellation for data far from zero; with FMA (Apple Silicon) the minimum became slightly negative, and `pow()` made it NaN. `hydraulic_stream_log` then spread one NaN pixel into a ~43k-pixel NaN region (HydraulicSaleve), and many graphs had NaN outputs | `(x - min) * scale + vmin`, clamped to `[vmin, vmax]` when remapping onto the array's own range |

`HIGHMAP_DISABLE_METAL=1` is a new runtime switch that sends every wrapper back
to the CPU/OpenCL path, for this kind of A/B check.

Result of the final A/B (512², 2×2 tiles), 1106 node outputs:

* The Metal and OpenCL host paths agree to within 1e-4 relative, except for
  `HydraulicParticle`. That upstream OpenCL kernel is non-deterministic: two
  OpenCL runs differ by up to 59%.
* NaN outputs dropped from 59 to 2 (`PolarShape`, NaN on both backends).
* Untiled, resident source nodes match the host path to about 1e-7. Downstream
  erosion nodes amplify that into at most a few percent at isolated pixels;
  with 2×2 tiling, the untiled resident evaluation differs from the tiled host
  evaluation near tile seams.
