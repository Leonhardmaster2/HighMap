# PathSplines baseline investigation

Date: 2026-09-02

## Finding

`PathSplines.PreservePathShape` still fails with the same deterministic result on clean upstream and on the rebased feature branch:

```text
chamfer distance: 0.15608564019203186
test tolerance:   0.15000000596046448
```

Observed configurations:

| Configuration | Result |
|---|---|
| Clean upstream `dev` at `269e9b6b77fa0926916d97c18656d8344800c9da` | Fails with the value above |
| Rebasing feature source, Metal disabled | Fails with the value above |
| Rebasing feature source, Metal enabled/runtime MSL | Fails with the value above |

The feature failure is not in a Metal test and does not depend on the Metal backend.

## Method

A temporary worktree was checked out at the exact upstream SHA and initialized with its submodules. The upstream default macOS configuration exposed two independent host-build issues: its OpenMP flag injection applies `-Xpreprocessor` to dependency compilation flags, and the default probe did not provide the Homebrew OpenCL C++ header include paths. For this source-level baseline test only, the temporary worktree's generated CMake input was adjusted to remove the problematic global OpenMP compile-option injection and supplied the host's OpenMP/OpenCL include and library locations. No such temporary edit was made in the HighMap repository.

The same single-test command was then run against the clean upstream build and both current feature builds. All three produced the same failure value.

## Classification

This is a pre-existing upstream/toolchain-sensitive path-spline baseline failure, not a Metal regression. The algorithm and tolerance were deliberately left unchanged. The full feature and no-Metal suites therefore report one known upstream failure while all newly added Metal and portability gates pass.

