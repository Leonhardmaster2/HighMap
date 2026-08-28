# GPU-resident `DeviceArray` design — Phase 3

## Decision

Phase 3 adds an explicit Metal execution model without adding hidden state to
`hmap::Array`:

```text
Array --upload--> DeviceArray --resident operations--> DeviceArray
                                      |
                                      +-- download --> Array
```

The public boundary is a small `DeviceSession` that owns an ordered Metal
command buffer, and a `DeviceArray` that owns one resource in that session.
The existing synchronous `Array` APIs remain unchanged.

Example:

```cpp
hmap::gpu::metal::DeviceSession session;
auto noise = session.noise(hmap::NoiseType::PERLIN,
                           {1024, 1024},
                           {4.f, 4.f},
                           42u);
auto gradient = session.gradient_norm(std::move(noise));
auto result = session.download(gradient);
```

`download()` is the explicit synchronization boundary. Shape queries, storage
queries, moves and destruction do not synchronize. Resident operation methods
append encoders to the session command buffer and return device-backed values;
they do not copy results to CPU.

## Ownership and lifetime

`DeviceArray` uses shared ownership of an opaque implementation object. A copy
is shallow: it refers to the same Metal resource and session state. Resident
operations are out-of-place unless explicitly documented otherwise, so copied
inputs are safe to use concurrently as read-only values. Move construction and
move assignment transfer the shared handle without allocating or copying the
GPU data.

The implementation is Objective-C++ only. Ordinary C++ headers expose no
`id<MTLBuffer>` and no Foundation type; the resource is hidden behind an
opaque `detail` state. A `DeviceArray` keeps its session state alive, so it may
outlive the `DeviceSession` C++ object. A session destructor finishes an open
command buffer before releasing it, preventing a resource from being destroyed
while GPU work is outstanding. Device resources cannot outlive the underlying
Metal device; if initialization fails, construction/upload throws a normal
`std::runtime_error` and never returns an invalid array.

The initial implementation uses one lazily initialized Metal context and one
command queue. A session owns one command buffer on that queue, created during
session construction after Metal capability validation. Multiple
sessions are independent and may be created by different threads, but a
single session is not thread-safe; callers must serialize access to it. The
global pipeline cache remains mutex-protected, and diagnostic statistics are
returned per session as well as through the existing thread-local synchronous
counter.

## Representation and validity

Each value stores:

* a two-dimensional shape and float32 element format;
* backend ownership (`Metal` in this phase);
* an opaque `MTLBuffer` resource;
* `Shared` or `Private` storage mode;
* an explicit validity state: host-valid, device-valid, or both-valid;
* a debug name and byte count;
* a shared reference to the session/lifetime state.

The initial resident implementation uses buffers rather than textures because
the staged kernels already use flat indexing and the chain benchmark is
intended to isolate residency first. Texture-backed advection is a separate
experiment exposed as `advection_warp_texture()`: it converts the resident
buffers to temporary `R32Float` textures and converts the result back, so the
same mirrored boundary behaviour can be tested without changing DeviceArray's
resource model. The measured path is retained as an experiment, not selected
as the default.

`Shared` is the default for direct host upload because it permits a single
explicit memcpy and final readback. `Private` is available for resident values;
upload uses a temporary shared staging buffer and a blit encoded before the
first compute pass, and download uses one final shared staging buffer. No
implicit blit is inserted by `shape()` or resource inspection.

## Command-buffer dependencies

Resident operations append to the session's command buffer in call order.
Metal's ordered encoders provide the dependency chain for a single session;
there is no CPU wait between stages. `download()` closes the chain by:

1. encoding final private-to-shared staging copies when needed;
2. committing the command buffer;
3. waiting once for completion;
4. copying shared contents into a new `Array`.

Calling `download()` twice is supported only as a read of the completed final
resource and performs no second command-buffer commit. Appending another
operation after a session has been finished is an error, because reopening a
command buffer would make dependency ownership ambiguous. A future asynchronous
API can expose a completion token, but Phase 3 keeps the ownership rule small.

For command-buffer experiments, `DeviceSession::submit()` commits the current
buffer and opens the next buffer on the same queue without waiting. Queue
ordering preserves dependencies, and the session still reports one
synchronization when the final result is downloaded. The default composed path
uses one buffer; the split path measures the driver/encoding tradeoff.

## Shape, backend and error rules

All resident operation inputs must belong to the same session, have matching
shapes where the operation requires it, and use float32 buffers. Mismatches
throw `std::invalid_argument` before an encoder is submitted. DeviceArray is
initially Metal-specific at the API level, but the concepts are backend-neutral:
shape, format, storage, ownership, validity and explicit transfer. A future
OpenCL implementation can add another backend state without changing `Array`
or pretending that an OpenCL resource is a Metal object.

Metal unavailable, unsupported noise types, allocation failures, pipeline
failures, command-buffer errors and invalid downloads all throw descriptive
HighMap errors. There is no fallback from a resident Metal operation to OpenCL:
that would silently break residency and change the execution model. Existing
host-resident APIs retain their existing fallback behaviour.

## Temporary and reuse policy

Resident output values own their resources. Thermal and hydraulic use
operation-local ping-pong and reduction buffers, allocated once per operation,
and do not allocate inside their iteration loops. The session also tracks a
small size-and-storage keyed scratch pool for explicitly consumed temporary
values. Reuse is allowed only after the previous encoder has been appended and
the resource is no longer exposed by a live `DeviceArray`; the command buffer's
ordering keeps reuse after the previous read.

The pool records allocations, reused buffers, allocated bytes, reused bytes and
peak resident bytes in `ExecutionStats`. It is session-local rather than a
global heap, so lifetime, device ownership and memory pressure are observable.
`MTLHeap` is not used.

The public Phase 3 operations remain out-of-place. Move-consuming inputs let a
later operation recycle an input buffer when no shallow copies remain; thermal
and hydraulic retain ping-pong state where in-place updates would change the
algorithm. No speculative clamp/scale kernel was added just to expose an
in-place API.

## Operation surface

The first resident surface mirrors the validated Phase 2 families:

* `gradient_norm(DeviceArray)`;
* smooth maximum/minimum on resident arrays;
* seeded noise created in a session;
* masked or unmasked `advection_warp` on resident arrays;
* thermal ping-pong on resident state;
* hydraulic virtual pipes with optional resident output arrays and GPU
  hierarchical volume reduction.

The only new supporting kernel is `advection_warp_texture`, added for the
buffer-versus-texture experiment. Existing gradient, smoothing, noise,
advection, thermal, hydraulic and reduction kernels are reused.

The synchronous `Array` wrappers remain the compatibility layer. They may use
the same internal encoders, but their timing and host-visible behaviour do not
change. No broad OpenCL catalog migration is part of Phase 3.

## CPU and OpenCL coexistence

`Array` remains the natural CPU/OpenCL interchange type. Passing an `Array` to
an existing CPU or OpenCL function never discovers or synchronizes a hidden
Metal resource. A resident caller explicitly chooses `DeviceSession`; if a
CPU-only branch is needed, `download()` is the visible boundary. A future
backend-neutral session can add OpenCL/Vulkan/CUDA resource ownership, but
cross-backend transfers must remain explicit rather than pretending resources
are interchangeable.

## API synchronization contract

| API | Synchronizes? | Notes |
|---|---:|---|
| `DeviceSession()` | no, unless initialization is required | Creates and owns one session command buffer |
| `upload(Array)` | no for shared; encodes a blit for private | Does not wait |
| resident operation | no | Appends ordered compute encoders |
| `DeviceArray::shape()` | no | Metadata only |
| `DeviceArray::storage_mode()` | no | Metadata only |
| `DeviceArray::set_debug_name()` | no | Metadata/resource label only |
| `DeviceSession::submit()` | no | Commits the current buffer and opens the next one |
| `download(DeviceArray)` | yes, once | Final synchronization and host copy |
| `DeviceSession::finish()` | yes, once | Explicit completion without download |
| session destruction with open work | yes | Safety completion rule |

This contract is intentionally explicit and testable. The Phase 3 benchmark
asserts one upload, one final download, no intermediate readbacks, one command
buffer and one synchronization for resident chains.
