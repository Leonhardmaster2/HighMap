# Regression baseline

## Starting point

Phase 2 starts from the committed first Metal milestone:

| Field | Value |
|---|---|
| Branch | `feature/apple-metal-backend` |
| Upstream source | `dev` at `2cc066e1002f598e10bb4f493ac2a811e8e09300` |
| Phase 1 checkpoint | `0663afa1f` (`feat(metal): add first Apple Silicon backend milestone`) |
| Host | arm64 Apple Silicon MacBook Air (`RELEASE_ARM64_T8122`) |
| OS | macOS 27.0, build `26A5378j` |
| Compiler | Apple Clang 21.0.0.21000327 |
| Build | Release, C++20, arm64, examples disabled, tests enabled |

The upstream source was checked out in a detached worktree with all recorded
submodules initialized. Its original CMake configuration does not build on
Apple Clang because it applies `-Xpreprocessor -fopenmp` globally; the flag
then captures CMake's `-MD` dependency option. A temporary compiler launcher
removed only that malformed forwarding for the comparison build, and supplied
the same Homebrew OpenMP/OpenCL include locations. No upstream source was
modified.

## Exact comparison

| Test | Untouched `dev` | Metal branch | Same failure? |
|---|---|---|---|
| `Blending.BlendExclusion` | fail | fail | yes |
| `Blending.BlendNegate` | fail | fail | yes |
| `Blending.BlendOverlay` | fail | fail | yes |
| `Blending.BlendSoft` | fail | fail | yes |
| `Gradient.Laplacian_ConstantField` | fail | pass | no; branch fixes no new failure |
| `PathSplines.PreservePathShape` | fail | fail | yes |

The untouched upstream run produced **316/322 passing tests** with six
failures. The Metal branch produced **323/328 passing tests** with five
failures; the six additional tests are the Metal-focused suite and all pass.
The branch introduces no test failure that is absent from untouched `dev`.
The five remaining failures are therefore existing repository failures, not
Metal regressions.

The upstream build workaround is recorded because an unmodified build attempt
fails before tests with the original OpenMP flag propagation. The branch fixes
that issue through target-scoped OpenMP linkage and remains buildable without
Metal as well.

## Phase 2 verification run

After the Phase 2 changes, the current branch test executable ran 336 tests:
335 passed and the same pre-existing `PathSplines.PreservePathShape` test
failed with a chamfer-distance error of 0.1561 against a 0.1500 tolerance.
The focused Metal suite ran 14/14, including the hydraulic one-command-buffer
assertion. This later count reflects the current checkout's full test
registration; the exact 322-versus-328 comparison above remains the preserved
Phase 2 starting-point comparison against untouched `dev`.
