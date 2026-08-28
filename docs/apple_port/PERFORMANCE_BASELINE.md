# Performance baseline — Phase 1 historical record

This document records the pre-Phase-2 environment and unavailable-device
baseline. Current Apple M3 OpenCL/Metal measurements are in
`BENCHMARK_RESULTS.md`; the Phase 2 interpretation is in
`PHASE2_PERFORMANCE_REVIEW.md`.

## Environment

| Field | Value |
|---|---|
| Source | `dev` at `e0d1279fced75cf7556de233128abfb5650e25b5` |
| Host | arm64 MacBook Air host (Darwin reports `RELEASE_ARM64_T8122`; exact M-series model was not exposed by the available system queries) |
| OS | macOS 27.0, build `26A5378j` |
| Compiler | Apple Clang 21.0.0.21000327, target `arm64-apple-darwin27.0.0` |
| Build system | CMake 4.1.2; C++20; Release requested |
| OpenMP | Apple Clang does not auto-detect libomp; Homebrew `/opt/homebrew/opt/libomp` is installed |
| OpenCL | Command Line Tools OpenCL framework is discoverable, but runtime device enumeration finds no matching device |
| Metal | `Metal.h` and `Metal.framework` are discoverable in the installed Command Line Tools SDK; `xcrun --find metal` has no standalone compiler, so this port uses runtime `newLibraryWithSource` compilation |

## Unmodified baseline result

The first configure used the repository's original CMake files and
`cmake -S . -B build-baseline -DCMAKE_BUILD_TYPE=Release
-DHIGHMAP_ENABLE_EXAMPLES=OFF`. It failed before generation because the
project calls `find_package(OpenMP REQUIRED)` and Apple Clang did not expose a
discoverable OpenMP runtime.

A second configure with manually supplied Apple `libomp` values generated, but
the project then passed the space-separated value
`-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include` as one compiler
argument. The build failed in external targets with `unknown argument`. This
is the motivation for the CMake fix in this branch. The corrected branch now
configures and builds successfully with Apple Clang, Homebrew `libomp`, and
Metal enabled.

## Measurement protocol

The benchmark plan covers `128²`, `256²`, `512²`, `1024²`, `2048²`, `4096²`
and `8192²` where memory/time permit. Each case must:

1. construct inputs outside the timed region;
2. warm up CPU and GPU paths;
3. run multiple repetitions and report median plus sample count;
4. separate upload, kernel/dispatch, synchronization and readback when the
   backend exposes those boundaries;
5. report effective pixels/s (or samples/s) and iteration rate;
6. record whether the result is host-resident or device-resident.

The benchmark tree now includes an Apple-backend harness alongside the
existing `bm_*` cases. New results belong in `BENCHMARK_RESULTS.md`; this file
records the environment and protocol rather than inventing measurements.

## Baseline status

| Backend | 128²–8192² | Status |
|---|---:|---|
| CPU | built and measured | A filtered CPU-oriented run completed 297 tests: 288 passed and 9 Metal tests skipped. The excluded legacy GPU suites require an unavailable OpenCL device; `PathSplines.PreservePathShape` remains a separate pre-existing failure in the unfiltered run. |
| OpenCL | compiled, unavailable at runtime | `DeviceManager::is_ready()` reports no OpenCL device on the host. OpenCL-dependent tests/benchmarks skip instead of aborting. |
| Metal | compiled, unavailable at runtime | Metal headers/framework are found and Objective-C++ backend code builds, but `MTLCreateSystemDefaultDevice()` returns no usable device in this execution environment. Metal tests/benchmarks skip. |

The measured CPU samples and the unavailable-backend evidence are recorded in
`BENCHMARK_RESULTS.md`. No CPU-vs-OpenCL-vs-Metal decision gate can be closed
until the same binaries run on a host exposing both compute devices.

The unfiltered executable was also run: 331 tests produced 292 passes, 9
Metal skips, and 30 failures. Twenty-nine failures are legacy OpenCL-dependent
tests that throw the documented no-device error because those older tests do
not yet self-skip; the remaining `PathSplines.PreservePathShape` failure is
separate. They are not treated as CPU or Metal regressions. The new Metal tests
and benchmark cases self-skip cleanly when Metal is unavailable.
