# HighMap Phase 4 — backend configuration matrix

Date: 2026-09-04
Test binary: `highmap_tests`, 436 registered tests per configuration

## Matrix result

| Config | Metal | OpenCL | Build | Passed | Skipped | Known failure | Result |
|---|---:|---:|---|---:|---:|---|---|
| A | ON | ON | Release tests + benchmarks | 433 | 2 | `PathSplines.PreservePathShape` | expected baseline-only failure |
| B | ON | OFF | Release tests + benchmarks | 387 | 42 | `PathSplines.PreservePathShape` | Metal-only gate passed |
| C | OFF | ON | Release tests | 385 | 50 | `PathSplines.PreservePathShape` | OpenCL-only gate passed |
| D | OFF | OFF | Release tests | 348 | 81 | `PathSplines.PreservePathShape` | CPU-only gate passed |

The skipped tests are backend-specific: B skips OpenCL-dependent coverage, C
skips native Metal coverage, and D skips both. No new test failure appeared in
any configuration. The known spline failure has the same exact chamfer value
and tolerance as the unmodified upstream baseline.

## Commands

The primary builds were configured with these switches:

```bash
# A
cmake -S . -B build-phase4-metal -DCMAKE_BUILD_TYPE=Release \
  -DHIGHMAP_ENABLE_METAL=ON -DHIGHMAP_ENABLE_OPENCL=ON \
  -DHIGHMAP_ENABLE_TESTS=ON -DHIGHMAP_ENABLE_BENCHMARKS=ON \
  -DHIGHMAP_ENABLE_EXAMPLES=OFF

# B
cmake -S . -B build-phase4-metal-only -DCMAKE_BUILD_TYPE=Release \
  -DHIGHMAP_ENABLE_METAL=ON -DHIGHMAP_ENABLE_OPENCL=OFF \
  -DHIGHMAP_ENABLE_TESTS=ON -DHIGHMAP_ENABLE_BENCHMARKS=ON \
  -DHIGHMAP_ENABLE_EXAMPLES=OFF

# C
cmake -S . -B build-no-metal -DCMAKE_BUILD_TYPE=Release \
  -DHIGHMAP_ENABLE_METAL=OFF -DHIGHMAP_ENABLE_OPENCL=ON \
  -DHIGHMAP_ENABLE_TESTS=ON -DHIGHMAP_ENABLE_BENCHMARKS=OFF \
  -DHIGHMAP_ENABLE_EXAMPLES=OFF

# D
cmake -S . -B build-phase4-cpu-only -DCMAKE_BUILD_TYPE=Release \
  -DHIGHMAP_ENABLE_METAL=OFF -DHIGHMAP_ENABLE_OPENCL=OFF \
  -DHIGHMAP_ENABLE_TESTS=ON -DHIGHMAP_ENABLE_BENCHMARKS=OFF \
  -DHIGHMAP_ENABLE_EXAMPLES=OFF
```

Each build completed. `OpenCLBackend.BuildTimeContract` passed in all four
configurations. The focused Metal/portability/hardening gate ran 45 tests in
A: 43 passed and 2 expected portability skips. In B, it ran 45 tests: 33
passed and 12 expected parity/portability skips. The native Metal runtime
tests themselves executed successfully; parity tests that require a second
OpenCL implementation skipped because B deliberately disables it.

## Interpretation

Configuration B is the release-critical gate: native Metal works without an
OpenCL SDK, OpenCL headers, CLWrapper target, or OpenCL framework dependency.
Configurations A, C, and D preserve the compatibility and CPU fallback paths.
