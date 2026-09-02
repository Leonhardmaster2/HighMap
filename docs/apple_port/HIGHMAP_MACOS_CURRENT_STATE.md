# HighMap macOS / Apple Silicon current state

Date: 2026-09-02

## Repository state

```text
Repository: Leonhardmaster2/HighMap
Branch: feature/apple-metal-backend
Upstream base: upstream/dev at 269e9b6b77fa0926916d97c18656d8344800c9da
Published implementation commit validated from a clean clone: d93bdfa8a
```

The Phase 1–8 Metal implementation was rebased onto current upstream `dev` without a feature-range merge commit. HighMap is standalone; no Hesiod changes were made.

## Backend surface

The explicit Metal API provides capability queries, synchronous wrappers, and the advanced resident `DeviceSession`/`DeviceArray` API. Current resident coverage includes generated noise, gradient, smooth extrema, morphology, normalization, spectral equalization, advection, thermal/thermal-ridge, border extrapolation, linear combination, Gabor-wave fBm, and virtual-pipes chains. Public `hmap::gpu` wrappers continue to prefer the established GPU route and fall back to OpenCL when native Metal is unavailable.

Resident sessions keep intermediate arrays on the device and expose upload/readback, finish, submit, completed-resource adoption, storage modes, and execution statistics. Transfer counters now report actual host copies rather than Metal-internal blits.

## Build requirements and tested configurations

The tested commands are:

```bash
cmake -S . -B build-phase4-metal -DCMAKE_BUILD_TYPE=Release \
  -DHIGHMAP_ENABLE_METAL=ON -DHIGHMAP_METAL_RUNTIME_COMPILE=ON \
  -DHIGHMAP_ENABLE_TESTS=ON -DHIGHMAP_ENABLE_BENCHMARKS=ON \
  -DHIGHMAP_ENABLE_EXAMPLES=OFF
cmake --build build-phase4-metal -j2

cmake -S . -B build-phase4-metal-debug -DCMAKE_BUILD_TYPE=Debug \
  -DHIGHMAP_ENABLE_METAL=ON -DHIGHMAP_METAL_RUNTIME_COMPILE=ON \
  -DHIGHMAP_ENABLE_TESTS=ON -DHIGHMAP_ENABLE_BENCHMARKS=OFF \
  -DHIGHMAP_ENABLE_EXAMPLES=OFF
cmake --build build-phase4-metal-debug -j2

cmake -S . -B build-no-metal -DCMAKE_BUILD_TYPE=Release \
  -DHIGHMAP_ENABLE_METAL=OFF -DHIGHMAP_ENABLE_TESTS=ON \
  -DHIGHMAP_ENABLE_BENCHMARKS=OFF -DHIGHMAP_ENABLE_EXAMPLES=OFF
cmake --build build-no-metal -j2
```

Apple Clang Release, Apple Clang Debug, Metal-enabled runtime MSL, and Metal-disabled builds all configure/build successfully on the tested host. OpenMP is discovered through CMake's `find_package(OpenMP)` and linked through the imported target; the host uses Homebrew `libomp`, without a hard-coded user path in the project. The current repository still requires OpenCL because existing GPU wrappers and CLWrapper are unconditional; `HIGHMAP_ENABLE_OPENCL` is not yet a complete optional dependency switch.

## Runtime shader path

Runtime MSL compilation is validated on the M3 host. Precompiled metallib was not validated because `xcrun --find metal` and `xcrun --find metallib` are unavailable in the current Command Line Tools installation. The CMake status explicitly reports that precompiled mode is off and runtime mode is on.

## Current test status

The current Metal-enabled Release suite has 398 tests: 395 pass, 2 skip (the no-Metal portability cases), and 1 known pre-existing `PathSplines.PreservePathShape` failure. The no-Metal Release suite has 398 tests: 347 pass, 50 skip, and the same one pre-existing failure. The focused Metal/Phase 8/hardening gate is 50/50 in Release and Debug. The no-Metal portability/hardening gate is 13 tests: 2 pass and 11 correctly skip native-Metal-only cases.

The sanitizer configuration (`build-asan-no-metal`) built successfully. Its practical no-Metal portability/validation subset passed under AddressSanitizer with no memory findings. A full UBSan run stops on a signed integer overflow in upstream `external/FastNoiseLite/Cpp/FastNoiseLite.h`; this is a third-party diagnostic, not a Metal-path finding. Existing compile warnings are dominated by upstream third-party headers (`nn-c`, legacy C prototypes, and external deprecations), not new Metal diagnostics.

A fresh clone of the published feature branch at `d93bdfa8a`, with all submodules initialized, configured and built successfully in a clean Metal-disabled build directory. Its full test run reproduced the expected 347 pass / 50 skip / 1 known PathSplines failure result.

## Performance characteristics and limitations

Resident execution is strongest when a graph contains multiple GPU operations and only one terminal download. At 1024², Chain A is about 1.5 ms with one final readback; the combined thermal/hydraulic Chain C is about 28.7 ms for 10 iterations with one command buffer but high variance. Synchronous hydraulic execution scales to about 191 ms at 100 iterations and is dominated by GPU synchronization/encoding.

Known limitations are the pre-existing PathSplines baseline failure, no precompiled shader-tool validation on this host, no 4096²/8192² measurement in this pass, OpenCL remaining a required build dependency, and the fact that upstream hydraulic-particle/multiscale additions are not Metal-backed. No Qt, Hesiod, GNode, or QTerrainRenderer dependency is present in the Metal backend, and Objective-C++ types remain isolated in implementation files.
