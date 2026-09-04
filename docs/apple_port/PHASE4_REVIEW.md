# HighMap Phase 4 — code and quality review

Date: 2026-09-04
Review scope: optional OpenCL boundary, configuration matrix, Metal runtime
qualification, resident workload evidence, and documentation

## Findings

### Optionality is implemented at the correct layers

The switch is not a runtime label only. CMake conditionally discovers headers,
adds CLWrapper, and links OpenCL; the source include boundary prevents
OpenCL headers from entering OFF builds; the target exports the compile-time
contract to consumers; and the final binaries were inspected independently.
This is the key quality requirement for a real Metal-only build.

### The disabled behavior is explicit

The compatibility `Run` stand-in compiles all existing OpenCL wrapper
translation units but throws when explicitly used. This keeps the build graph
stable and avoids silent data corruption. Tests that are intentionally
OpenCL-dependent skip based on the same availability contract, while the new
contract test verifies both the readiness probe and a public OpenCL operation.

### The Metal implementation remains isolated

No Metal shader, `DeviceArray` ownership rule, transfer counter, command-buffer
policy, or public fallback routing was changed for optionality. The A and B
configurations therefore exercise the same native Metal path, with B proving
that it does not carry an OpenCL link dependency.

### The matrix is meaningful

All four combinations configure and build. The expected backend-specific skips
are separated from failures, and the one remaining failure is reproduced from
upstream. The large-workload benchmarks use explicit single-iteration control
after the harness's unitless-time over-repetition behavior was observed.

## Remaining risks

* Full Xcode Metal compiler and Instruments tooling are unavailable on this
  host, so precompiled library and GPU-counter claims remain unverified.
* Resident Chain C reaches 1.611 GB peak allocation at 4096² on an 8 GB Mac;
  product callers need workload limits or memory-aware scheduling.
* Hydraulic-particle/multiscale remains OpenCL-only and is documented rather
  than incorrectly presented as Metal coverage.
* The project retains noisy third-party header warnings and the known upstream
  spline tolerance failure; neither was broadened by this phase.

## Verdict

Accept the implementation for Phase 4 publication on
`Leonhardmaster2/HighMap`. The code-level quality is good for the stated scope:
the dependency boundary is complete, behavior is explicit, the matrix is
covered, and no speculative optimization was retained. Do not begin Phase 5
from this review; the next work should be a separately authorized milestone
after the published branch has been tested by its intended caller.
