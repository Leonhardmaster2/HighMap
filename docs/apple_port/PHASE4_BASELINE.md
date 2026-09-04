# HighMap Apple Silicon — Phase 4 baseline

Date: 2026-09-04
Repository: `Leonhardmaster2/HighMap`
Branch: `feature/apple-metal-backend`

## Scope

Phase 4 qualifies the HighMap branch for a true Metal-only build while keeping
OpenCL available as an optional, first-class configuration. The work is scoped
to HighMap. No Hesiod, Qt, GNode, QTerrainRenderer, or application-level files
are part of this change.

The feature branch was already based directly on current upstream `dev` before
this phase:

```text
upstream/dev: 269e9b6b77fa0926916d97c18656d8344800c9da
feature baseline: 4350fbb819c5f9106de76a538e372945a1cbe749
```

`git fetch upstream --prune` was run at the start of the phase and the fetched
`upstream/dev` resolved to the SHA above. No upstream commit authorship was
changed.

## Host and toolchain

| Field | Value |
|---|---|
| Hardware | Apple M3 MacBook Air, 8 GB |
| OS | macOS 27.0 |
| Architecture | arm64 |
| Compiler | AppleClang 21.0.0.21000327 |
| Build system | CMake 4.1.2, C++20 |
| OpenMP | Homebrew `libomp`, discovered through CMake |
| Metal SDK | Headers/framework discoverable |
| Standalone Metal compiler | unavailable in installed Command Line Tools |
| Runtime shader mode | MSL source compiled by `newLibraryWithSource` |

The host exposes both an Apple M3 Metal device and an Apple M3 OpenCL 1.2
device at runtime. `xcrun --find metal`, `xcrun --find metallib`, and the
usable `xctrace` developer tool are unavailable, so precompiled shader and
Instruments captures are recorded as environment limitations rather than
silently inferred.

## Baseline behavior

Before Phase 4, the root project unconditionally discovered OpenCL, built
CLWrapper, and linked the OpenCL framework even when the caller requested a
Metal-only build. The Metal implementation and resident `DeviceSession` work
were otherwise functional and already validated by the Phase 1–8 test and
benchmark history.

The known baseline test failure is:

```text
PathSplines.PreservePathShape
chamfer distance = 0.15608564019203186
tolerance       = 0.15000000596046448
```

The same failure is present on an unmodified upstream baseline and is not used
as evidence against the Phase 4 backend changes.

## Phase 4 changes

* `HIGHMAP_ENABLE_OPENCL` now gates OpenCL discovery, headers, CLWrapper, and
  link dependencies.
* `opencl_run.hpp` provides a compile-only disabled boundary whose explicit
  OpenCL-only operations throw a deterministic runtime error.
* OpenCL-dependent tests skip cleanly when the backend is disabled or has no
  device; the four build configurations have an explicit contract test.
* Tests and benchmarks link `spdlog` directly instead of relying on the
  previously transitive CLWrapper dependency.
* No Metal kernel or resident ownership/synchronization semantic was changed.

The exact dependency and matrix evidence is in the companion Phase 4 audit
documents.
