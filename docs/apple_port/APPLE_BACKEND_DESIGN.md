# Apple Metal backend design

## Goals

The Metal backend is an optional native compute backend for Apple platforms.
It must coexist with the existing CPU and OpenCL implementations, preserve
existing public HighMap signatures, and avoid making callers manage Metal
objects for ordinary `hmap::gpu::*` calls.

The first slice is intentionally small: a capability-detected Metal device,
cached compute pipelines, flat float buffers, explicit command submission, and
representative gradient/filter/noise/advection/thermal/hydraulic entry points. The
canonical shader source is kept in `HighMap/src/gpu_metal/highmap.metal` and is
embedded into the library at configure time so a runtime source checkout is
not required.

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
             |
             v
          Apple GPU
```

The `hmap::gpu::metal` namespace is a low-level, optional capability surface.
The existing `hmap::gpu::*` functions use it opportunistically for the staged
representative operations and fall back to their current OpenCL behavior when
Metal is unavailable. No OpenCL source or API is removed.

## Resource model

### Device and context

`MetalContext` owns:

* the default `MTLDevice`, selected through `MTLCreateSystemDefaultDevice()`;
* one command queue for the current synchronous API;
* one shader library compiled once from the embedded MSL source;
* a map of compute pipeline states keyed by function name;
* capability metadata obtained from Metal objects, never from M-series names.

Initialization is lazy and thread-safe. Pipeline lookup is cached. Failure to
find Metal or compile the shader library leaves `is_available()` false and is
reported through a descriptive diagnostic; the CPU/OpenCL paths remain usable.

### Data representation

The first backend uses `MTLBuffer` objects containing the exact flat
`Array::vector` layout (`j * nx + i`). This keeps copies predictable and lets
the same kernels support both pointwise and neighborhood operations. The
abstraction does not expose a graphics texture or a texture sampler to public
HighMap APIs yet.

The intended next step is a private `DeviceArray` with:

* shape and element format;
* `MTLBuffer` or `MTLTexture` storage;
* storage mode (`Shared` for upload/readback-friendly data, `Private` for long
  GPU chains);
* host/device dirty state;
* explicit `upload`, `download`, and `synchronize` operations.

That type should be introduced only after measurements show a composed pipeline
benefits from residency; `Array` remains a simple host value type.

### Command submission

The synchronous adapter encodes one or more compute passes, commits a command
buffer, and waits only at the public result boundary. The hydraulic path uses
two state buffers and encodes all dependent passes for an iteration before the
command buffer is committed. It therefore avoids the current OpenCL
flow-readback/water-readback/erosion-readback/sediment-readback sequence.

The multi-pass design uses separate buffers for a pass's input and output. The
Metal implementation must not rely on in-place neighbor writes. A future
asynchronous API can return a command token or expose a `DeviceArray`, but that
is intentionally outside this milestone.

## Public and internal contracts

The low-level header provides:

* `is_available()` and `device_name()` for diagnostics;
* `gradient_norm`, `noise`, and `advection_warp` returning `Array` values;
* `thermal` and `hydraulic_vpipes` mutating/outputting arrays with the existing
  signatures.

These functions throw only for an explicitly requested Metal operation after
Metal was reported available but a runtime operation failed. High-level
selection checks availability before calling them and otherwise preserves the
existing OpenCL behavior.

## Backend selection policy for this milestone

There is no speculative dynamic scheduler. For the five staged operations:

* if Metal is available on macOS, the selected high-level GPU function uses
  Metal;
* if Metal is unavailable, it uses the existing OpenCL implementation;
* non-Apple builds compile the stub and use OpenCL;
* future selection can incorporate operation, dimensions, iteration count,
  current residency and requested result residency after benchmark evidence.

The benchmark harness records CPU/OpenCL/Metal separately using end-to-end
host API wall time, including the current upload, dispatch, synchronization and
readback costs. Separate upload/dispatch/readback columns remain a follow-up
once the backend exposes nonblocking timing hooks. On the current host, the
OpenCL and Metal cases are explicitly unavailable at runtime, so no backend
speed claim is made yet.

## CMake and portability

`HIGHMAP_ENABLE_METAL` defaults on only as a request; actual availability is
detected from Metal framework headers/library. Objective-C++ is enabled only on
that successful detection. The Objective-C++ backend source is excluded from non-Apple and
SDK-incomplete builds. Foundation/Metal frameworks are linked privately to the
HighMap library.

OpenCL remains the existing required compatibility backend in this milestone;
its source and submodule are untouched. The existing OpenMP option is made
robust for Apple Clang + Homebrew `libomp`, while builds without a usable
OpenMP runtime continue with the library's existing serial behavior.

## Numerical policy

Metal uses float32 and follows the algorithm's existing boundary and sampling
rules. Parity tests report maximum absolute error, mean absolute error and
RMSE. Iterative erosion additionally checks invariants and aggregate terrain
statistics. Any intentional difference caused by ping-pong ordering or Metal
math lowering is recorded in `METAL_PORT_STATUS.md`.

## Deferred work

* texture-backed sampling after profiling buffer-vs-texture behavior;
* device-side reductions for all hydrology algorithms;
* GPU-resident public `DeviceArray` composition;
* full 81-file OpenCL kernel migration;
* asynchronous public API and event ownership;
* per-GPU-family paths. Capability checks, not M-series marketing names, are
  the extension point.
