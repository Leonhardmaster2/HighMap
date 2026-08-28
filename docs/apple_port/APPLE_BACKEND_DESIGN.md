# Apple Metal backend design — Phase 2

## Goals and current boundary

The Metal backend is an optional native compute backend for Apple platforms.
It coexists with CPU and OpenCL, preserves the existing public HighMap
signatures, and keeps Metal objects out of ordinary `Array` callers.

The Phase 2 boundary is deliberately measurable: gradient, smooth extrema,
noise, advection, thermal and hydraulic virtual pipes have native Metal
implementations, instrumented timing counters, expanded parity tests and a
common benchmark matrix. The public API is still synchronous and returns
host-resident `Array` values. A GPU-resident composition API is the next
architecture milestone, not an implicit property of these wrappers.

## Layering

```text
HighMap public API (Array and existing gpu::* functions)
             |
             v
  small backend selection / Metal adapter
       |                         |
       v                         v
  existing OpenCL path       hmap::gpu::metal
  (CLWrapper)                MetalContext + MSL pipelines
             |                         |
             +------------+------------+
                          v
                       Apple GPU
```

The `hmap::gpu::metal` namespace is a low-level optional capability surface.
The high-level staged functions select Metal when it is available and retain
the existing OpenCL fallback otherwise. No OpenCL source or API is removed.

## Resource model and measured choices

`MetalContext` owns the default `MTLDevice`, one command queue, one shader
library, a function-name pipeline cache and capability metadata. Initialization
is lazy and thread-safe. Pipeline lookup is cached; the instrumented path
reports cache misses separately from encoding and GPU time.

Current staged calls use `MTLBuffer` objects containing the exact flat
`Array::vector` layout (`j * nx + i`). This is the right boundary for the
existing synchronous API because callers expect a returned or mutated CPU
`Array` immediately. All current buffers use unified-memory `Shared` storage;
that makes upload and final readback explicit and reliable on Apple Silicon.

The Phase 2 benchmark does not justify switching these isolated calls to
`Private` storage: each call still has to upload inputs and read back its
result, so a private resource would add a staging/blit path without removing
the public boundary. A private-storage experiment should wait for a chain that
keeps multiple results on the device.

Buffers are used for every staged operation. Pointwise and reduction kernels
need linear indexing, and advection's custom mirrored-boundary sampler fits
the buffer representation. The OpenCL reference uses images for advection,
but the Metal measurements did not establish a texture win after accounting
for allocation and readback; texture-backed sampling remains a targeted future
experiment rather than a speculative rewrite.

## Device capabilities and dispatch

The backend reports device name, recommended working-set size, thread
execution width, maximum threads per threadgroup and a generic Metal family
number. It does not infer an M-series model from marketing names.

The measured default dispatch policy uses up to 32 lanes for pointwise kernels
and a compact 16-wide tile for neighborhood kernels, clamped by the pipeline's
reported SIMD width and maximum threadgroup size. `HIGHMAP_METAL_THREADGROUP`
provides a reproducible override for future sweeps. The measured sweeps showed
32-wide pointwise dispatch improving gradient and 16-wide neighborhood
dispatch improving advection relative to the initial 8-wide baseline.

## Command buffers, synchronization and residency

The synchronous adapter encodes one or more compute passes, commits a command
buffer, waits at the public result boundary and then copies results to the
caller. Simple operations use one encoder and one wait. Thermal uses one
command buffer with one encoder per ping-pong iteration and one final wait.

Hydraulic is the main residency target. It allocates the full ping-pong state
once, encodes flow, water, erosion, sediment and evaporation passes for every
iteration, keeps the state on the GPU between passes, and commits exactly one
command buffer. When volume maintenance is requested, the water sum and
rescaling are also encoded in the same command buffer using a hierarchical
float reduction. Only the final requested arrays are read back. The correctness
suite asserts one command buffer and one synchronization for this path.

This is a meaningful architecture difference from the benchmark's OpenCL
hydraulic copy, which reads outputs after each dependent pass and performs
host-side volume/state work between passes. The hydraulic speedup is therefore
not attributed to Metal hardware alone.

## Reductions and numerical behavior

The hydraulic volume correction uses a two-stage GPU reduction: a tile sum
followed by recursively reduced partial buffers, then a GPU rescale pass. The
intermediate scalar remains device-resident; only final public arrays are
downloaded. The reduction uses float accumulation rather than a quantized
integer atomic so larger maps and water levels do not silently overflow.

Metal uses float32 and preserves the existing boundary and sampling rules.
Stress tests cover flat and non-square gradients, seeded noise with optional
maps/bounds/periods, masked advection, zero/one/many thermal iterations and
non-square hydraulic output/invariant cases.

## Fusion and a future `DeviceArray`

The current wrappers do not fuse public operations. Fusing two calls behind
the existing `Array` API would either change when the caller can observe a
result or silently retain hidden device state, so it would be an unsafe
optimization at this boundary. The next design is an internal or public
GPU-resident `DeviceArray` with:

* shape, element format and backend ownership;
* an `MTLBuffer` or `MTLTexture` handle and storage mode;
* host/device dirty state with explicit upload/download operations;
* lifetime tied to the Metal device/queue and command-buffer dependencies;
* thread-safe ownership rules, or an explicit single-queue contract;
* conversion to `Array` only at a requested synchronization boundary.

With that type, a caller could compose gradient → smooth → noise or a
multi-pass terrain chain without allocating, uploading and downloading every
stage. The benchmark evidence supports doing this for hydraulic and other
multi-pass workloads first. It does not yet support claiming a generic fusion
engine or migrating the remaining OpenCL catalog.

## CPU crossover and selection policy

The measured results show no universal GPU threshold. CPU wins small gradient
and remains competitive through 4096²; Metal is clearly ahead of CPU for noise
from 512² in the measured samples; advection benefits at 1024² and 2048² but
loses its advantage at 4096² when transfer cost dominates; hydraulic benefits
strongly from the single-command-buffer design. The high-level scheduler
should therefore eventually consider operation, dimensions, iteration count,
current residency and requested result residency. Until residency exists, the
selection policy remains operation-specific and benchmark-backed rather than a
single pixel-count cutoff.

## Build and portability

`HIGHMAP_ENABLE_METAL` is optional. Objective-C++ and Metal framework linkage
are enabled only when the Apple SDK is usable. Release builds use a build-time
`metallib` when standalone `xcrun metal` and `metallib` tools are available;
otherwise the embedded MSL source is compiled by `newLibraryWithSource` at
runtime. Non-Apple builds use the stub API and compile without Objective-C++.

The current host lacks the standalone Metal compiler, so the runtime source
path is the active build path. The build-time path is implemented and will be
selected automatically on a tool-complete Apple SDK.

## Next milestone

Implement and benchmark the GPU-resident `DeviceArray` boundary, then measure
buffer versus texture and shared versus private storage for composed chains.
Prioritize reductions and linear pointwise stages that can reuse resident
state. Expand the kernel catalog only after those measurements establish that
the residency boundary, rather than another isolated wrapper, is the limiting
factor.
