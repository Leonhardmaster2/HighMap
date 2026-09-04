# HighMap Phase 4 — OpenCL dependency audit

Date: 2026-09-04
Branch: `feature/apple-metal-backend`

## Configure-time boundary

`HIGHMAP_ENABLE_OPENCL` defaults to `ON` for compatibility. When it is `OFF`:

| Layer | OpenCL OFF behavior |
|---|---|
| Root CMake | Does not call `find_package(OpenCL)` or search for `CL/opencl.hpp` / `CL/cl.h` |
| External CMake | Does not add `external/CLWrapper` |
| HighMap target | Does not link `clwrapper` or `${OpenCL_LIBRARIES}` |
| Compile contract | Publishes `HIGHMAP_HAS_OPENCL=0` |
| Source wrappers | Include `highmap/internal/opencl_run.hpp`, not CLWrapper directly |
| Runtime | `init_opencl()` returns `false`; explicit OpenCL-only `Run` construction throws |

When it is `ON`, the existing CLWrapper path, OpenCL header checks, kernel
source registration, and OpenCL framework link remain active. The CMake target
publishes `HIGHMAP_HAS_OPENCL=1` and the existing OpenCL C++ wrapper version
macros.

## Source boundary

The internal header is the only HighMap source boundary for `clwrapper::Run`.
The real CLWrapper header is included only under `#if HIGHMAP_HAS_OPENCL`.
The disabled stand-in intentionally does not emulate computation: it throws
with:

```text
HighMap OpenCL backend is disabled at build time (HIGHMAP_ENABLE_OPENCL=OFF)
```

This is important for correctness. A caller that explicitly selects an
OpenCL-only function receives a visible failure; it does not get an empty or
partially initialized result. CPU and native Metal paths do not construct the
stand-in.

## Binary evidence

Configuration B (`HIGHMAP_ENABLE_METAL=ON`, `HIGHMAP_ENABLE_OPENCL=OFF`) was
built as Release. The generated flags contain:

```text
-DHIGHMAP_HAS_OPENCL=0 -DHIGHMAP_HAS_METAL=1
```

`otool -L build-phase4-metal-only/bin/highmap_tests` and the corresponding
`highmap_benchmarks` executable contain the normal project dependencies,
Metal/Foundation, and Homebrew `libomp`, but no `OpenCL.framework`. The same
inspection on configuration D (`HIGHMAP_ENABLE_METAL=OFF`, OpenCL OFF) contains
neither `OpenCL.framework` nor the Metal framework. `nm -u` on both B
executables produces no symbol matching `opencl`, `clwrapper`, or `metal`.

The disabled build also omits the CLWrapper target from the generated target
graph. This proves the OFF path at discovery, compilation, target linkage, and
binary dependency levels rather than only at runtime.

## Test coverage

`OpenCLBackend.BuildTimeContract` passes in A, B, C, and D. Existing tests that
intentionally require OpenCL now use a shared availability guard and skip when
the optional backend is disabled or when the host has no OpenCL device. This
keeps a missing optional device distinct from a test failure.

One ancillary build issue found during the audit was that tests and benchmarks
used `spdlog` directly while receiving it transitively from CLWrapper. They now
declare and link `spdlog::spdlog` explicitly, so the OFF build does not depend
on an unrelated transitive link edge.
