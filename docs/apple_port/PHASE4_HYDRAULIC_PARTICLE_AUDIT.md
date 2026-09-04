# HighMap Phase 4 — hydraulic-particle audit

Date: 2026-09-04

## Implementation location and backend

The public particle API is declared in
`HighMap/include/highmap/erosion.hpp` and implemented in
`HighMap/src/erosion/hydraulic_particle.cpp`. Its GPU namespace implementation
constructs `clwrapper::Run("hydraulic_particle")` and binds the OpenCL kernel
from `HighMap/src/gpu_opencl/kernels/hydraulic_particle.cl`.

There is no corresponding hydraulic-particle implementation in the native
Metal source or resident `DeviceSession` API. The current Metal hydraulic
coverage is virtual-pipes (`hydraulic_vpipes`), which is a different algorithm
and is not a substitute for particle erosion.

## Semantics

The implementation supports:

* particle count and deterministic seed;
* optional bedrock, moisture, and elevation-shift maps;
* optional erosion and deposition outputs;
* capacity, erosion, deposition, inertia, gravity, drag, evaporation, talus,
  and collapse parameters;
* a mask overload; and
* a multiscale wrapper that resamples a coarse-to-fine ladder and blends the
  original input at each level.

The `iterations` parameter splits particles across passes. Each pass uploads
and executes the OpenCL particle kernel against the terrain modified by the
previous pass. This deliberate serialization lets later particles follow
channels carved by earlier particles, but it is not a device-resident Metal
session and is not safe to relabel as a generic GPU implementation.

## Phase 4 result

The algorithm remains CPU/OpenCL-facing and is explicitly outside the Metal
coverage claim. In an OpenCL-disabled build, an explicit call reaches the
disabled compatibility boundary and throws the documented deterministic error.
The build itself remains valid because OpenCL headers, CLWrapper, and the
OpenCL framework are not required. No particle kernel or numerical behavior was
changed in Phase 4.

Porting this algorithm is a separate project: it would require defining the
particle-state ownership model, preserving optional-map and output semantics,
choosing an atomic/deposition strategy, and adding cross-backend correctness
tests before performance work.
