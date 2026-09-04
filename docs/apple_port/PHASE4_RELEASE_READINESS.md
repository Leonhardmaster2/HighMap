# HighMap Phase 4 — release readiness

Date: 2026-09-04
Repository: `Leonhardmaster2/HighMap`
Branch: `feature/apple-metal-backend`

## Gate status

| Gate | Status | Evidence |
|---|---|---|
| Latest upstream base revalidated | PASS | `upstream/dev = 269e9b6b77fa0926916d97c18656d8344800c9da` |
| Metal ON / OpenCL ON | PASS with known baseline failure | 396 pass, 2 skip, 1 `PathSplines` failure |
| Metal ON / OpenCL OFF | PASS | 357 pass, 41 skip, 1 same baseline failure |
| Metal OFF / OpenCL ON | PASS | 348 pass, 50 skip, 1 same baseline failure |
| Metal OFF / OpenCL OFF | PASS | 318 pass, 80 skip, 1 same baseline failure |
| Metal-only dependency proof | PASS | no OpenCL/CLWrapper symbols or framework in `otool`/`nm` audit |
| Runtime MSL path | PASS | focused Metal tests execute on Apple M3 |
| Precompiled metallib path | NOT AVAILABLE | standalone `metal`/`metallib` tools absent |
| 1024² resident Chain A | PASS | completed with one final sync/readback |
| Hydraulic resident workload | PASS | Chain C completed at 1024²/2048²/4096², 10 iterations |
| Hydraulic-particle Metal coverage | DOCUMENTED LIMIT | remains OpenCL-only |
| Additional numerical optimization | EXPLICITLY DEFERRED | no stable low-risk candidate justified |

## Known issue

Every complete test matrix run reports the pre-existing
`PathSplines.PreservePathShape` failure. The exact value is
`0.15608564019203186` versus tolerance `0.15000000596046448`; an unmodified
upstream build reports the same failure. It is not a Phase 4 regression.

## Clean-clone and publication record

The final publication step must clone the feature branch from
`https://github.com/Leonhardmaster2/HighMap`, configure at least the Metal-only
and CPU-only builds, and rebuild the relevant test target. The exact published
tip, feature commit count, merge count, author audit, and push result are
recorded in the final task report after that gate. No Hesiod repository or
remote is in scope.

## Readiness decision

Phase 4 is ready to publish as a HighMap feature branch with the documented
baseline failure and toolchain limitations. It is not a claim that every
OpenCL-only HighMap algorithm has a Metal implementation, nor a claim that a
precompiled metallib was validated on this host.
