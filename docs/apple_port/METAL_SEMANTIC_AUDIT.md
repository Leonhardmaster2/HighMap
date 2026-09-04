# HighMap Metal semantic audit

Date: 2026-09-02  
Reference policy: current upstream CPU/OpenCL behavior remains authoritative.

## Operation matrix

`Sync Metal` means the public synchronous Metal wrapper. `Resident Metal` means the `DeviceSession`/`DeviceArray` path. A dash means that the operation is not currently implemented by the resident API, not that a CPU/OpenCL feature has been removed.

| Operation | CPU reference | OpenCL reference | Sync Metal | Resident Metal | Parameters/fallback | Current parity evidence |
|---|---|---|---|---|---|---|
| `noise` | `hmap::noise` | `hmap::gpu::noise` | yes | yes | Supported noise types are the Metal shader set; unsupported resident types throw. Public GPU routing retains OpenCL fallback when Metal is unavailable. | Phase 8 and hardening resident/sync tests |
| `noise_fbm` | `hmap::noise_fbm` | `hmap::gpu::noise_fbm` | yes | yes | Same seed, scale, octave, and persistence semantics as the existing Metal wrapper. | Existing Phase 8 parity tests |
| `gabor_wave_fbm` | CPU Gabor-wave fBm | OpenCL Gabor-wave fBm | yes | yes | Current scale/seed/octave parameters are forwarded; no upstream parameter is silently dropped. | Existing Phase 8 parity tests |
| `gradient_norm` | `hmap::gradient_norm` | `hmap::gpu::gradient_norm` | yes | yes | Shape-preserving finite-difference gradient. | Small-resolution and non-square sweeps |
| `maximum_smooth`, `minimum_smooth` | CPU smooth extrema | OpenCL smooth extrema | yes | yes | Shape match and smoothing parameter are validated. | Phase 8 and hardening tests |
| `morphological_gradient` | CPU morphology has square-window semantics | GPU/OpenCL morphology uses the disk-radius GPU contract | yes | yes | Radius is forwarded. Comparisons use `hmap::gpu::morphological_gradient` for the GPU semantic contract rather than incorrectly comparing disk morphology to the CPU square-window implementation. | Six-resolution sweep |
| `smooth_cpulse` | CPU smooth pulse | OpenCL smooth pulse | yes | yes | Existing pulse parameters are forwarded. | Existing Metal backend tests |
| `spectral_equalizer` | CPU spectral equalizer | OpenCL spectral equalizer | yes | yes | Frequency-band parameters and iteration count are forwarded. | Hybrid boundary and existing Phase 8 tests |
| `normalize` | `hmap::remap` range contract | OpenCL reduction/remap | yes | yes | Two scalar range values are read internally; output remains resident until an explicit download. Constant arrays map to the requested lower bound without NaNs. | Constant, odd-size, and transfer-stat tests |
| `advection_warp` | CPU advection warp | OpenCL advection warp | yes | yes | Buffer path preserves current interpolation/boundary parameters. | Existing buffer-chain parity tests |
| `advection_warp_texture` | CPU advection equivalent | OpenCL texture path | yes | yes | Texture path is explicit; no default switch was made from benchmark noise alone. | Existing texture/buffer benchmark and parity coverage |
| `thermal` | CPU thermal erosion | OpenCL thermal | yes | yes | Iteration count and talus input are forwarded. | Six-resolution and resident tests |
| `thermal_ridge` | CPU thermal-ridge behavior | OpenCL thermal-ridge kernel | yes | yes | Real resident `thermal_ridge` execution is now tested for odd/non-square shapes and 1/3/7 iterations. | Dedicated synchronous-vs-resident parity test and sweep |
| `extrapolate_borders` | CPU border extrapolation | OpenCL border helper | yes | yes | Applied explicitly where the current resident thermal-ridge contract requires boundary completion. | Thermal-ridge parity tests |
| `linear_combine` | CPU linear combination | OpenCL linear combination | yes | yes | Coefficients and shape checks are preserved. | Existing Phase 8 tests |
| `hydraulic_vpipes` | CPU virtual-pipes reference | OpenCL virtual-pipes passes | sync wrapper and resident chain | yes | Resident path preserves the multi-pass flow/water/erosion/sediment sequence and avoids per-iteration host transfers. | Existing hydraulic tests and Chain C benchmark |

## Transfer and synchronization semantics

`upload_count`/`upload_bytes` count host-to-Metal value copies performed by the HighMap API. `readback_count`/`readback_bytes` count Metal-to-host value copies. Metal-internal resource copies, scratch-buffer reuse, and command-buffer boundaries are not misreported as host transfers. Normalize's scalar range read is intentionally visible as a readback; a resident result still does not incur a full terrain readback until `download`.

## Fallback and availability

The public `hmap::gpu` wrappers retain the established CPU/OpenCL fallback routing when native Metal is unavailable. The explicit `hmap::gpu::metal` API reports an unavailable backend through empty capabilities/device name and deterministic exceptions from session/array operations. Hardware-dependent Metal tests skip in a Metal-disabled build; the two portability tests execute there and validate the fallback and stub contract.

## Upstream changes not silently ported

The new upstream hydraulic-particle/multiscale semantics, including the detail-preserving `mix` parameter, are not currently Metal-backed. They remain available through the current CPU/OpenCL implementation. Because no existing Metal wrapper claims to implement that algorithm, this is an explicit coverage boundary rather than a stale-parameter bug.

## Phase 4 build and backend boundary audit

Date: 2026-09-04

Phase 4 keeps the semantic rule above intact while making the legacy OpenCL
backend optional. `HIGHMAP_ENABLE_OPENCL=OFF` removes OpenCL discovery,
CLWrapper compilation, OpenCL include paths, and OpenCL framework linkage. The
GPU wrapper translation units still compile against the internal
`opencl_run.hpp` compatibility boundary, whose disabled `clwrapper::Run`
stand-in throws a deterministic `std::runtime_error` if an explicitly
OpenCL-only operation is requested. This preserves a clear failure instead of
silently producing an incomplete result.

The public Metal path is unaffected by this boundary: native Metal wrappers
and resident `DeviceSession` operations continue to use the same MSL source,
buffer ownership, transfer counters, and synchronization rules. The optional
backend contract is covered by `OpenCLBackend.BuildTimeContract` in all four
matrix configurations. OpenCL-requiring tests now skip when the optional
backend is disabled or has no runtime device; Metal and CPU coverage remains
active.

The only known source-level semantic failure in the complete matrix is the
pre-existing `PathSplines.PreservePathShape` tolerance failure (chamfer
distance `0.15608564019203186` versus tolerance `0.15000000596046448`). It is
also present on an unmodified upstream baseline and is not attributable to
the Phase 4 backend boundary.
