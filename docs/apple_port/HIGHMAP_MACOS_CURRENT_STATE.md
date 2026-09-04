# HighMap macOS / Apple Silicon current state

Date: 2026-09-04

## Repository state

```text
Repository: Leonhardmaster2/HighMap
Branch: feature/apple-metal-backend
Upstream base: upstream/dev at c63c44b16e4ffa0e73f035999009d42f83f8a6dd
Published implementation baseline validated from a clean clone: 4350fbb819c5f9106de76a538e372945a1cbe749
Phase 4 source/docs are recorded on top of that historical baseline; the final tip is recorded in the release audit and push report.
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

Apple Clang Release, Apple Clang Debug, Metal-enabled runtime MSL, and Metal-disabled builds all configure/build successfully on the tested host. OpenMP is discovered through CMake's `find_package(OpenMP)` and linked through the imported target; the host uses Homebrew `libomp`, without a hard-coded user path in the project. Phase 4 makes OpenCL optional at configure, compile, link, and runtime-contract levels. With `HIGHMAP_ENABLE_OPENCL=OFF`, CLWrapper is not added, OpenCL headers are not searched, the OpenCL framework is not linked, and explicit OpenCL-only calls fail deterministically while CPU/Metal paths remain usable.

## Runtime shader path

Runtime MSL compilation is validated on the M3 host. Precompiled metallib was not validated because `xcrun --find metal` and `xcrun --find metallib` are unavailable in the current Command Line Tools installation. The CMake status explicitly reports that precompiled mode is off and runtime mode is on.

## Current test status

The Phase 4 matrix has 436 tests in each configuration. Metal+OpenCL ON: 433 pass, 2 skip, and 1 known pre-existing `PathSplines.PreservePathShape` failure. Metal ON/OpenCL OFF: 387 pass, 42 skip, and the same failure. Metal OFF/OpenCL ON: 385 pass, 50 skip, and the same failure. Metal OFF/OpenCL OFF: 348 pass, 81 skip, and the same failure. The focused Metal gate is 43 pass / 2 expected portability skips in A, and 33 pass / 12 expected parity or portability skips in B; the optional-backend contract test passes in all four configurations.

The sanitizer configuration (`build-asan-no-metal`) built successfully. Its practical no-Metal portability/validation subset passed under AddressSanitizer with no memory findings. A full UBSan run stops on a signed integer overflow in upstream `external/FastNoiseLite/Cpp/FastNoiseLite.h`; this is a third-party diagnostic, not a Metal-path finding. Existing compile warnings are dominated by upstream third-party headers (`nn-c`, legacy C prototypes, and external deprecations), not new Metal diagnostics.

A fresh clone gate for the post-Phase 4 branch was completed at `/tmp/highmap-phase4-clean.9FqKh9`. The earlier clean clone at `d93bdfa8a` remains historical evidence; the new clean-clone result is recorded in `PHASE4_RELEASE_READINESS.md`.

## Performance characteristics and limitations

Resident execution is strongest when a graph contains multiple GPU operations and only one terminal download. The post-rebase single-sample sweep measured shared-storage Chain A at 6.790 ms / 8.971 ms / 27.481 ms for 1024² / 2048² / 4096², and resident Chain C (advection + thermal + hydraulic virtual pipes, 10 iterations) at 35.747 ms / 136.185 ms / 828.614 ms. These are workload evidence, not release thresholds: the host was under substantial system load and the benchmark reports one measured iteration to avoid over-repeating high-memory cases. Peak resident allocation reached 201.327 MB for Chain A 4096² and 1.611 GB for Chain C 4096².

Known limitations are the pre-existing PathSplines baseline failure, no precompiled shader-tool validation on this host because only Command Line Tools are installed, the benchmark harness's tendency to over-repeat expensive cases when a time suffix is omitted, and the fact that upstream hydraulic-particle/multiscale additions are not Metal-backed. OpenCL is now optional and remains a first-class ON configuration. No Qt, Hesiod, GNode, or QTerrainRenderer dependency is present in the Metal backend, and Objective-C++ types remain isolated in implementation files.
