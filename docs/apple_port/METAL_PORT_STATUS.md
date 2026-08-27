# Metal port status

Status values: `planned`, `implemented`, `compiled`, `tested`, or `blocked`.
“Implemented” means source exists; it does not imply runtime parity.

| Function | CPU | OpenCL | Metal | Tested | Benchmark | Notes |
|---|---|---|---|---|---|---|
| `gradient_norm` | yes | yes | implemented / compiled | runtime-blocked | CPU only | Flat-buffer local-neighbor kernel; CPU/OpenCL parity tests are present, but no Metal device is exposed here |
| `maximum_smooth` / `minimum_smooth` | yes | yes | implemented / compiled | runtime-blocked | CPU only | Pointwise smooth-extrema filter; included as the first simple filter stage |
| `noise` (Perlin/simplex/value subset) | yes | yes | implemented / compiled | runtime-blocked | CPU only | Perlin, Perlin billow/half, simplex2, value and value-linear paths; deterministic and OpenCL parity tests are present |
| `advection_warp` | yes | yes | implemented / compiled | runtime-blocked | CPU only | Bounded path walk; OpenCL parity test uses a 2e-3 mirrored-boundary tolerance |
| `thermal` | yes | yes | implemented / compiled | runtime-blocked | CPU only | Metal uses ping-pong state; high-level wrapper preserves border extrapolation |
| `hydraulic_vpipes` | yes | yes | implemented / compiled | runtime-blocked | CPU only | Four dependent passes stay resident in one Metal command buffer; optional water conservation uses a hierarchical float reduction |
| Other 76+ OpenCL entry points | yes/mixed | yes | planned | pending | pending | Port only where measurements justify it |

The current host has Metal headers/frameworks, and the Objective-C++ backend
compiles. Runtime MSL compilation and kernel execution could not be exercised:
`MTLCreateSystemDefaultDevice()` returned no usable device. The standalone
`xcrun metal` compiler is also absent, so when a device is available shader
compilation is intentionally performed through
`MTLDevice::newLibraryWithSource`.
