# HighMap Phase 4 — optimization candidates and decision

Date: 2026-09-04

## Ranked candidates

| Rank | Candidate | Evidence | Decision |
|---:|---|---|---|
| 1 | Keep multi-operation graphs resident through one terminal download | Chain A uses one command buffer/final sync; Chain C avoids intermediate host readbacks | Retained from Phase 3; highest measured value |
| 2 | Reduce hydraulic multi-pass synchronization/encoding | Chain C reaches 101 encoders at 10 iterations and dominates 4096² wall time | Already resident; further fusion would alter numerical/pass ordering and needs Instruments |
| 3 | Persistent/private resource pooling at 4096² | Peak Chain C reaches 1.611 GB; shared mode reports reuse but pressure is real | Defer; needs full memory trace and product workload limits |
| 4 | Change default threadgroup shape | Earlier 32×4 experiments sometimes helped Chain A but were noisy and did not generalize to Chain C | Reject speculative default change |
| 5 | MTLHeap or broad kernel fusion | Could reduce allocation overhead but affects ownership, lifetime, and error paths | Do not introduce without full Xcode profiling and a concrete graph |
| 6 | Port hydraulic-particle to Metal/DeviceSession | The algorithm is OpenCL-only, branch-heavy, and has multi-pass particle semantics | Out of scope for this phase; document as a coverage boundary |

## Optimization decision

No additional numerical kernel optimization is justified by the available
evidence. The low-risk, high-value work in this phase is the genuine optional
OpenCL boundary plus test coverage and dependency proof. The resident API is
already the effective optimization for composed Metal graphs. Changing a
shader, storage default, or pass schedule without Instruments and without a
stable product graph would trade correctness confidence for an unproven gain.

This is an explicit “none worthwhile” conclusion for the additional
optimization slot, not an unmeasured assumption. The next useful measurement
requires full Xcode Metal tools and a representative HighMap caller.
