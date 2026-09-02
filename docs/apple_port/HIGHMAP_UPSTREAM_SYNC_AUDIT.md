# HighMap upstream synchronization audit

Date: 2026-09-02  
Repository: `Leonhardmaster2/HighMap`  
Branch: `feature/apple-metal-backend`  
Upstream: `ottolink-dev/HighMap`, branch `dev`

## Result

The feature branch was rebased onto the fetched `upstream/dev` at:

```text
269e9b6b77fa0926916d97c18656d8344800c9da
```

The previous Metal base was:

```text
e0d1279fced75cf7556de233128abfb5650e25b5
```

The rebase preserved the Phase 1–8 Metal implementation and produced a linear feature range. At the time of this audit:

```text
git merge-base HEAD upstream/dev = 269e9b6b77fa0926916d97c18656d8344800c9da
git rev-list --count --merges upstream/dev..HEAD = 0
```

No Hesiod checkout, submodule, source file, commit, or remote was modified.

## Relevant upstream changes

The complete range `e0d1279fced75cf7556de233128abfb5650e25b5..upstream/dev` was inspected with both `git log` and `git diff`. The relevant changes are classified below.

| Upstream area | Representative commits | Classification | Metal interaction and resolution |
|---|---|---|---|
| Custom logger and removal of `macro-logger` | `1750d8ece`, `bdb4d7c86`, `75a06517f`, `8c0ad0f47` | `BUILD_CHANGE`, `POTENTIAL_CONFLICT`, `API_CHANGE` | The rebase retained upstream `hmap::log` and its source-location API. Metal sources compile against the current logger; the old tracked dependency removal is preserved. The untracked local `external/macro-logger/` artifact was not touched. |
| Array, authoring, blending, boundary, and general input validation | `c17fead91`, `36946f0ed`, `3cbca7f0d`, `416e73324`, `fc98d140c` | `API_CHANGE`, `TEST_CHANGE` | Upstream validation remains authoritative. Metal code was adapted to current public signatures and the hardening tests exercise invalid shapes, moved arrays, closed sessions, and shape mismatches without changing upstream validation policy. |
| Hydrology flow fixing, MST, riverbed carving, and related refactors | `58a8d0800`, `169645d87`, `28b2e2edd`, `92826e600` | `ALGORITHM_CHANGE`, `NO_INTERACTION`, `NEW_METAL_OPPORTUNITY` | These are CPU/OpenCL hydrology features not currently routed through the Metal backend. No speculative new kernel port was introduced. |
| New hydrology/erosion parameters and validation | `667f74758`, `ea665deb4`, `6e6aaffc6`, `1d2e81868`, `05b9d6254`, `4a063924b`, `0dbf8d07f`, `64fe708ff`, `269e9b6b7` | `ALGORITHM_CHANGE`, `API_CHANGE`, `NEW_METAL_OPPORTUNITY` | The hydraulic-particle/multiscale work is a separate upstream algorithm. Existing Metal coverage is for virtual pipes and its signatures remain current. The new particle semantics are documented as unported rather than silently ignored by a Metal wrapper. |
| Path squiggle, path utilities, profiles, geometry, and terrain additions | `b4df5d859`, `146856146`, `3293e79df`, `9a54c1c3b`, `1267feed2` | `API_CHANGE`, `ALGORITHM_CHANGE`, `NO_INTERACTION` | No existing Metal-backed operation changed semantics. The recurring spline test was separately investigated against clean upstream. |
| Convolution, curvature, filters, primitives, shadows, statistics, synthesis, and validation expansion | `5e16f8583`, `14d91444a`, `755a1fdbf`, `96b949b60`, `315c146ea`, `36a2aca74` | `API_CHANGE`, `TEST_CHANGE`, `NEW_METAL_OPPORTUNITY` | These additions remain first-class CPU/OpenCL functionality. No stale Metal wrapper signature was found for an operation already implemented by Metal. |
| CMake, formatting, and platform/build cleanup | `e9269d402`, `49dcc7447`, `9fdaf3953` and related changes | `BUILD_CHANGE`, `TEST_CHANGE` | The feature build was regenerated with Apple Clang in Metal-enabled and Metal-disabled modes. Current Metal tool detection and portable OpenMP discovery remain in the rebased tree. |

The remaining upstream commits in the range are formatting, merge-history, or unrelated implementation changes and were classified `NO_INTERACTION` after source inspection. Upstream merge commits were not copied into the feature-only range.

## Wrapper/signature audit

The public headers, OpenCL wrappers, Metal headers, Metal implementation, tests, and benchmarks were searched after the rebase. The upstream `hydraulic_particle` and `hydraulic_particle_multiscale` changes do not have a Metal counterpart, so there is no Metal wrapper that silently drops the new `mix` or particle parameters. The existing Metal-backed operations compile and are covered by the semantic audit in `METAL_SEMANTIC_AUDIT.md`.

## Source equivalence check

The source diff from the rebased feature branch retains the Phase 1–8 Metal paths. The only new implementation change in this hardening pass is authoritative host-transfer counters (`upload_count` and `readback_count`); the existing Metal kernels and resident execution behavior remain intact. Focused Metal parity tests, full suites, and the no-Metal build were run after the rebase.

