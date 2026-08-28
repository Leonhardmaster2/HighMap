# Metal port status

Status values: `planned`, `implemented`, `compiled`, `tested`, or `blocked`.
“Implemented” means source exists; it does not imply numerical parity by
itself.

| Function | CPU | OpenCL | Metal | Tested | Benchmark | Notes |
|---|---|---|---|---|---|---|
| `gradient_norm` | yes | yes | implemented / compiled | 14-test suite | measured | Flat-buffer local-neighbor kernel; CPU parity and boundary stress coverage |
| `maximum_smooth` / `minimum_smooth` | yes | yes | implemented / compiled | compiled | measured harness | Pointwise smooth-extrema filters |
| `noise` | yes | yes | implemented / compiled | 14-test suite | measured | Seed, optional-map, bbox and period parity coverage |
| `advection_warp` | yes in library, no benchmark reference | yes | implemented / compiled | 14-test suite | measured | Buffer sampler with mask and non-square stress coverage |
| `thermal` | yes in library, no benchmark reference | yes | implemented / compiled | 14-test suite | measured | Ping-pong state; one command buffer for all iterations |
| `hydraulic_vpipes` | yes in library, no benchmark reference | yes | implemented / compiled | 14-test suite | measured | GPU-resident state, one command buffer, GPU hierarchical volume reduction |
| Other OpenCL entry points | yes/mixed | yes | planned | pending | pending | Port only when residency-aware measurements justify it |

The current host exposes both an Apple M3 OpenCL device and a usable native
Metal device. Metal-focused tests pass 14/14, and the same tests skip cleanly
when configured with Metal disabled. The standalone `xcrun metal` compiler is
not installed on this host, so the tested Release build uses embedded MSL
runtime compilation; the build-time `metallib` path is enabled automatically
when the SDK tools are available.

Detailed timings and the Phase 2 decision are recorded in
`BENCHMARK_RESULTS.md` and `PHASE2_PERFORMANCE_REVIEW.md`.
