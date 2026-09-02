# HighMap macOS hardening review

Date: 2026-09-02  
Scope: HighMap only; no Hesiod changes.

## Decision

The current HighMap Metal branch is rebased onto current upstream, builds in the tested Apple Clang configurations, passes all new Metal correctness gates, has authoritative host-transfer counters, and has a measured profile. The only full-suite failure is reproduced on clean upstream and is documented as pre-existing. No unstable dispatch experiment was promoted to a default optimization.

## Required answers

1. **Current `upstream/dev` SHA:** `269e9b6b77fa0926916d97c18656d8344800c9da`.
2. **Published implementation HEAD validated from a clean clone:** `d93bdfa8a`; the final branch tip including this documentation update is recorded in the final report/branch audit.
3. **Old merge base:** `e0d1279fced75cf7556de233128abfb5650e25b5`.
4. **Upstream commits since that base:** logger migration, validation expansion, path/geometry additions, hydrology flow/MST changes, erosion hydraulic-particle/multiscale changes, CMake/OpenMP cleanup, and the associated formatting/tests; exact representative hashes and classifications are in `HIGHMAP_UPSTREAM_SYNC_AUDIT.md`.
5. **Upstream changes interacting with Metal:** build/logger/API changes and current validation; no existing Metal kernel had a conflicting semantic change. Hydraulic-particle changes are a documented new Metal opportunity, not an existing wrapper conflict.
6. **How incorporated:** clean rebase onto fetched `upstream/dev`; upstream headers, logger, validation, and build behavior were retained while Metal sources were compiled/adapted against them.
7. **Stale Metal semantics found:** no stale existing Metal semantics were found. The morphology comparison was corrected to use the GPU/OpenCL disk-radius contract rather than the CPU square-window helper.
8. **Wrappers missing new parameters:** none for operations already claimed by Metal. The unported hydraulic-particle/multiscale API is explicitly not routed through Metal.
9. **Final Metal test count:** 50 focused Metal/Phase 8/hardening tests; 50 passed in Release and Debug.
10. **Final full-suite result:** 398 total, 395 passed, 2 skipped, 1 known pre-existing PathSplines failure in Metal-enabled Release.
11. **Final no-Metal result:** 398 total, 347 passed, 50 skipped, and the same known pre-existing PathSplines failure.
12. **No-Metal tests that execute:** two portability tests validate deterministic unavailable-backend behavior and public GPU OpenCL fallback; they pass. Native Metal cases skip with an explicit reason.
13. **PathSplines on clean upstream:** yes, it reproduces exactly.
14. **Root classification:** pre-existing upstream/toolchain-sensitive baseline failure, unrelated to Metal.
15. **Misleading/brittle tests corrected:** the unsupported noise test was renamed from a false fallback claim to `UnsupportedNoiseTypeIsRejected`; a thermal-plus-normalize test was renamed to match its actual operations; exact command-buffer assertions were replaced with transfer/submission/residency invariants.
16. **`thermal_ridge` parity:** yes; resident execution is compared with the synchronous Metal reference over 29×13, 63×37, and 127×65 shapes and 1/3/7 iterations, plus the six-resolution sweep.
17. **Exact command-buffer tests:** no feature hardening test freezes the old exact count. Statistics remain observable and lower-bound invariants verify work submission.
18. **Authoritative transfer statistics:** yes, at the HighMap `ExecutionStats`/`DeviceSession` level.
19. **Upload/readback counts and bytes:** yes; counters increment only at host copies, including explicit upload/download and normalize's scalar range read. Internal Metal blits are excluded.
20. **Completed-resource adoption stress:** yes; 64 deterministic producer/consumer iterations pass.
21. **Use-after-free:** none observed in the tested lifetime/adoption paths.
22. **Retained-resource leak:** no; resident bytes return to zero after finish/download in stress tests. A full allocator-level leak tool was not available.
23. **Small-resolution parity problem:** none reproduced in HighMap Metal at 32² through 1024².
24. **Worst small-resolution errors:** normalization max absolute error `1.78814e-7`; morphology max `1.19209e-7`; gradient max `1.86265e-9`; smooth max `5.96046e-8`; thermal and thermal-ridge were zero against their synchronous Metal references.
25. **Non-square shapes:** yes; odd/non-square reduction shapes 127×63 and 255×257 pass, as do thermal-ridge representative non-square shapes.
26. **Odd reductions:** yes; 127×63 and 255×257 normalize reductions cover all elements and match the CPU range contract.
27. **Normalization edges:** constant negative, zero, and positive arrays across 1×257, 37×19, and 255×257 remain finite and map to the requested lower bound.
28. **All current Metal wrappers compared:** the operation matrix in `METAL_SEMANTIC_AUDIT.md` records CPU/OpenCL references, resident/sync status, parameters, fallback, and evidence for each current wrapper.
29. **macOS Debug build:** passes with Apple Clang and runtime MSL enabled.
30. **macOS Release build:** passes for Metal-enabled and no-Metal configurations; test executables run as reported above.
31. **Metal-disabled build:** passes and its fallback/portability gate runs.
32. **OpenMP detection:** portable CMake discovery/imported target; no project hard-coded user path.
33. **Metal tool detection:** clear; framework availability and build-time compiler/metallib availability are reported separately.
34. **Runtime MSL:** validated on the M3 host, including source load, pipeline compilation, dispatch, and parity tests.
35. **Precompiled metallib:** not validated; the host lacks `metal` and `metallib` in the installed Command Line Tools.
36. **Missing tool exactness:** `xcrun --find metal` and `xcrun --find metallib` do not resolve on this host.
37. **Runtime/metallib parity:** not measurable on this host because precompiled mode is unavailable; runtime source is the only exercised shader path.
38. **Pipeline caching:** working; warm benchmark samples report zero pipeline creations and repeated-operation tests pass.
39. **Independent-session threading:** yes; two independent sessions run concurrently and match synchronous Metal references.
40. **Scratch/resource pooling bounded:** tested repeated same-shape reuse and post-finish zero residency; no evidence of unbounded growth in this pass.
41. **Peak memory at 4096²:** not run in this pass.
42. **Peak memory at 8192²:** not run in this pass; the benchmark guards it behind an explicit extended-workload opt-in.
43. **Largest current bottlenecks:** multi-pass hydraulic synchronization/encoding, synchronous transfer boundaries, and combined resident Chain C memory/dispatch volume.
44. **Optimizations attempted:** transfer accounting, residency/lifetime/test hardening, and repeatable 8×8/16×8/32×4 threadgroup experiments.
45. **Retained:** authoritative upload/readback counters and the expanded correctness/portability instrumentation; existing evidence-backed resident dispatch policy remains.
46. **Reverted/not promoted:** no speculative threadgroup default, kernel fusion, MTLHeap, universal scheduler, or broad new kernel port was retained.
47. **Material speedups:** no new default kernel speedup is claimed; the main measurable quality improvement is eliminating ambiguity in transfer accounting. Resident Chain A remains approximately 1.5 ms at 1024².
48. **Important-size regressions:** none in the focused parity sweep or full source tests beyond the same upstream PathSplines baseline failure.
49. **Standalone/no Qt/Hesiod:** yes; no Qt/Hesiod/GNode/QTerrainRenderer dependency appears in the Metal backend.
50. **Objective-C isolation:** yes; public GPU headers remain standard C++ and Objective-C++ syntax is confined to `.mm` implementation code.
51. **Clean clone build:** yes; `/tmp/highmap-clean-jkiXmj` cloned the published branch at `d93bdfa8a`, initialized every submodule, configured a clean Metal-disabled build, built `highmap_tests`, and reproduced 347 passed / 50 skipped / 1 known PathSplines failure.
52. **Local paths absent:** yes in tracked HighMap source/build configuration; the host-specific Homebrew paths were only supplied to the temporary upstream baseline and are not project settings.
53. **Linear history:** yes.
54. **Merge commits in feature range:** 0.
55. **AI author/co-author counts:** 0/0; full commit bodies were audited case-insensitively.
56. **Branch publication:** yes for the implementation commit via `git push --force-with-lease origin feature/apple-metal-backend`; this final documentation update is pushed with the same safe mechanism.
57. **Remaining macOS HighMap issues:** precompiled metallib cannot be tested on this CLT-only host; OpenCL is still effectively required; upstream hydraulic-particle/multiscale has no Metal path; 4096²/8192² memory evidence is still outstanding; PathSplines remains an upstream baseline failure.
58. **Next HighMap-only focus:** install/use a full Metal shader toolchain for metallib parity, then profile 2048²–4096² resident hydraulic and advection workloads with Instruments/Metal validation before considering another narrowly measured optimization.

## HESIOD_FOLLOWUP

None. This pass made no Hesiod changes and identified no required Hesiod-side implementation.
