# HighMap Apple Silicon optimization — Phase 3 review

Date: 2026-08-28  
Branch: `feature/apple-metal-backend`

## Decision

Phase 3 is **accepted as an implementation checkpoint**. The project now has
an explicit Metal-resident execution model and measured composed-chain wins,
without changing `Array` value semantics or removing CPU/OpenCL paths. The
resident API is ready for carefully selected Metal callers; a universal graph
planner or automatic backend scheduler is intentionally deferred.

## Review questions

1. **Is the ownership model safe?** Yes. `DeviceArray` holds shared ownership
   of opaque Metal state; copies are shallow, moves transfer the handle, and the
   session state stays alive until arrays release it. Session destruction waits
   for open work.

2. **Are synchronization boundaries explicit?** Yes. Resident operations and
   metadata getters do not wait. `download()`, `finish()` and destruction with
   open work are the synchronization boundaries. Private staging is encoded at
   the boundary or pre-prepared for hydraulic outputs.

3. **Did Phase 3 avoid hidden GPU state in `Array`?** Yes. No `Array` layout or
   copy semantics were changed. A caller opts into `DeviceSession` explicitly.

4. **Does the model preserve backend separation?** Yes. The public resident
   type is Metal-specific for this phase, while the state concepts are
   backend-neutral. OpenCL remains available through existing `Array` APIs.

5. **Are the required resident operations present?** Yes: gradient, smooth
   extrema, supported noise, advection, thermal and hydraulic virtual pipes.
   Hydraulic optional resident outputs and GPU hierarchical volume reduction
   are included.

6. **Does composition actually avoid intermediate transfers?** Yes. The
   resident tests and counters show one command buffer and one final sync for
   the default chains. Chain A reads back 4.2 MB at 1024² instead of 16.8 MB
   across four synchronous public calls.

7. **Is there a real allocation/reuse strategy?** Yes. The session has a
   size-and-storage keyed scratch pool, explicit move-consuming inputs and
   allocation/reuse/peak counters. Chain A reports buffer reuse; thermal and
   hydraulic allocate ping-pong state once outside their iteration loops.

8. **Is one command buffer always assumed optimal?** No. The default uses one
   buffer, and `submit()` provides ordered no-wait splitting. The 1024² split
   experiment was within scheduling noise, so the simpler one-buffer policy is
   retained.

9. **What did Shared versus Private show?** Shared was the better default for
   these end-to-end chains. Private is correct and useful for long-lived device
   state, but its explicit staging and blits made it slower in the measured
   multi-input B/C samples. It remains available rather than being forced.

10. **What did Buffer versus Texture show?** Mixed results at 1024², 2048² and
    4096². Temporary texture conversion is measurable, so textures were not
    promoted to the default representation.

11. **Does the model work at large sizes?** Resident Chain A was measured
    through 4096² and an extended 8192² Shared case completed at about 805 MB
    peak resident allocation. B/C are registered through 4096² but only the
    256²–1024² samples were included in this review table.

12. **Are correctness tolerances preserved?** Yes. Metal-focused parity and
    stress tests pass without loosening Phase 2 tolerances. The texture path
    matches the buffer advection path within the established 2e-3 advection
    cross-runtime tolerance.

13. **Are failures explicit?** Yes. Empty arrays, invalid shapes, mixed
    sessions, unsupported noise, unavailable Metal, finished sessions and
    invalid downloads throw descriptive standard exceptions. The no-Metal stub
    fails explicitly while existing benchmark/test skip behavior remains clean.

14. **Was scope controlled?** Yes. Only one supporting kernel,
    `advection_warp_texture`, was added for a directly requested experiment.
    No broad OpenCL catalog port or hidden scheduler was introduced. Generic
    public reductions and speculative in-place kernels remain future work.

15. **What should Phase 4 do?** First integrate resident primitives into one
    real HighMap/Hesiod workload and choose its explicit CPU/device boundary.
    Then profile private memory pressure, consider reusable textures only if
    the workload justifies them, and decide whether a backend-neutral session
    is warranted by an actual OpenCL/Vulkan use case.

## Verification

* Metal-focused tests: **25/25 passed**.
* Full Metal-enabled suite: **346 passed, 1 known upstream failure** in
  `PathSplines.PreservePathShape`; no new regression was observed.
* Metal-disabled focused suite: **25 skipped cleanly**.
* Historical Phase 2 benchmark and review documents were not rewritten.

