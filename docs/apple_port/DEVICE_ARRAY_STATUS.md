# DeviceArray status — Phase 3

Date: 2026-08-28  
Branch: `feature/apple-metal-backend`  
Host used for Metal evidence: Apple M3, macOS 27.0, arm64

## Delivered

| Area | Status | Evidence / boundary |
|---|---|---|
| Explicit resident model | Complete | `metal::DeviceSession` + `metal::DeviceArray`; no hidden state in `Array` |
| Ownership and lifetime | Complete | Shared opaque state, shallow copies, moves, session lifetime retained by arrays |
| Upload/download | Complete | Shared direct memcpy; Private staging blits; explicit `download()` |
| Residency semantics | Complete | `device_valid`, `host_valid`, `both_valid`; metadata getters do not synchronize |
| Gradient and smooth extrema | Complete | Resident Metal paths and synchronous parity tests |
| Noise | Complete | Supported Phase 2 noise types, resident output, optional resident fields |
| Advection | Complete | Buffer-backed resident path; optional resident mask |
| Thermal | Complete | One session command buffer, operation-local ping-pong, no per-iteration allocation |
| Hydraulic virtual pipes | Complete | Resident state, optional resident outputs, GPU hierarchical volume reduction |
| Reuse | Complete | Session-local size/storage keyed pool; move-consuming inputs; allocation/reuse counters |
| Composition | Complete | Chains A/B/C in the benchmark target with shared/private resident variants |
| Command-buffer strategy | Complete | One-buffer default plus `submit()` split-without-wait experiment |
| Shared vs Private | Complete | End-to-end chain measurements and explicit blit counters |
| Buffer vs Texture | Complete as experiment | `advection_warp_texture()` uses temporary `R32Float` textures and explicit conversions |
| Error handling | Complete | Backend, session, shape, ownership, unsupported-operation, finish and download checks |
| No-Metal build | Complete | Stub API compiles; all 25 Metal tests skip cleanly |

## Deliberate limits

* The public resident API is Metal-specific and currently supports 2D float32
  arrays. The state layout is backend-neutral in concept, but no OpenCL
  `DeviceArray` implementation is claimed in Phase 3.
* A session owns one queue and one current command buffer and is not
  thread-safe. `submit()` can split buffers without a CPU wait, but the default
  chain remains one buffer.
* Resident operations are out-of-place. Move-consuming inputs permit safe
  buffer reuse; no speculative clamp/scale in-place kernel was added.
* The texture path is intentionally an experiment: each advection call creates
  temporary textures and pays five buffer-to-texture plus one texture-to-buffer
  blit. Buffers remain the default representation.
* Generic public sum/min/max/mean reductions were not added. Hydraulic reuses
  its existing hierarchical sum/reduction kernels internally, which is the
  only reduction needed by the Phase 3 chain.
* Private results should be downloaded through the owning session before the
  session is explicitly finished unless the operation pre-prepared staging
  for its exposed outputs. This keeps staging explicit instead of allocating
  hidden host mirrors for every private value.
* The normal composed benchmark range is 256²–4096². An 8192² resident Chain A
  case is available only with `HIGHMAP_PHASE3_EXTENDED=1` because of its
  approximately 805 MB peak resident allocation.

## Verification snapshot

* Metal-focused suite: **25/25 passed**.
* Full Metal-enabled suite after the final source/test rebuild: **346 passed,
  1 known upstream `PathSplines.PreservePathShape` failure**.
* Metal-disabled focused suite: **25 skipped cleanly**.
* The historical Phase 2 documents remain unchanged; the Phase 2 result was
  335/336 before the additional Phase 3 tests.

