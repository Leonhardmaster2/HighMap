# Current HighMap Metal profile

Date: 2026-09-04
Host: Apple M3 MacBook Air, 8 GB  
Compiler: AppleClang 21.0.0.21000327  
Build: Release, `HIGHMAP_ENABLE_METAL=ON`, runtime MSL enabled, tests and benchmarks enabled

## Runtime/toolchain path

The Metal framework is detected and the runtime MSL path compiles and executes successfully. Build-time `metal`/`metallib` tools were not available through the installed Command Line Tools, so this host reports:

```text
HIGHMAP_METAL_PRECOMPILED=OFF
HIGHMAP_METAL_RUNTIME_COMPILE=ON
```

The Metal tests exercise source loading, runtime shader compilation, pipeline creation, dispatch, and result download. A precompiled metallib parity run could not be performed because the `metal` and `metallib` executables are missing; there is no claim that a precompiled artifact was tested.

The representative table below is the earlier Phase 1–8 profile retained for
historical comparison. The authoritative post-rebase Phase 4 sweep is recorded
in the final section and in `PHASE4_LARGE_WORKLOADS.md`.

## Representative measured results

Google Benchmark was run warm, with five measured repetitions and aggregate statistics. Times below are real-time medians; the workload includes the transfer boundary stated in each row.

| Workload | Median | Encoders / command buffers | Host transfers | Peak allocated |
|---|---:|---:|---|---:|
| Resident Chain A, 1024² | 1.466 ms in the selected five-repeat run; a separate ten-repeat run measured 1.518 ms | 4 / 1 | 0 uploads, 1 final readback (4.194 MB) | 12.583 MB |
| Resident Chain C, 1024², 10 iterations | 28.667 ms | 101 / 1 | 8 uploads (33.554 MB), 1 final readback (4.194 MB) | 100.696 MB |
| Synchronous hydraulic virtual pipes, 1024², 10 iterations | 27.232 ms | 90 / 1 | 2 uploads (8.389 MB), 1 readback (4.194 MB) | 75.530 MB |
| Synchronous hydraulic virtual pipes, 1024², 100 iterations | 190.986 ms | 900 / 1 | 2 uploads, 1 readback | 75.530 MB |

The Chain C and hydraulic samples are sensitive to system load; the selected five-repeat Chain C run had high variance. The counters show the important boundary: the resident chain does not read back intermediate terrain, while synchronous wrappers include host transfers.

## Bottleneck ranking

1. Multi-pass hydraulic execution is dominated by command completion/synchronization and repeated encoders, especially at 100 iterations.
2. The synchronous advection path is transfer- and sampling-sensitive; its historical 1024² runs are hundreds of milliseconds with large input upload volume, while the resident path removes intermediate host boundaries.
3. Resident Chain C is substantially more expensive than Chain A because it combines advection, thermal, and 10 hydraulic iterations and retains roughly 100.7 MB during the chain.
4. Chain A is already low-latency; its one final readback is a visible fraction of total time.
5. Warm pipeline creation is not a steady-state bottleneck: measured samples report zero new pipeline creations after the warm-up.

## Dispatch experiment

The existing capability-based default uses a compact 16-wide neighborhood layout and up to 32 lanes for pointwise kernels, with an environment override for repeatable experiments. At 1024², 8×8, 16×8, and 32×4 overrides were measured for Chain A/C. 32×4 sometimes measured faster for Chain A (1.414 ms median in the ten-repeat run versus 1.518 ms default), but the result carried about 25% coefficient of variation and did not improve all graphs consistently; Chain C varied independently. No speculative default change was retained.

## Memory and pooling

Repeated resident operations reuse scratch buffers and report peak allocation through `peak_resident_bytes`; the completed sample returns `resident_bytes` to zero after download/finish. Phase 4 single-iteration samples measured Chain A peak allocation of 12.583 MB / 50.332 MB / 201.327 MB at 1024² / 2048² / 4096², and Chain C peak allocation of 100.696 MB / 402.784 MB / 1.611 GB at the same sizes. Chain A reports one reuse; Chain C reports three reuses. Existing benchmarks guard 8192² behind `HIGHMAP_PHASE3_EXTENDED=1` to avoid accidental pathological runs.

## Optimization decision

The retained improvements in this pass are authoritative transfer accounting and regression coverage around residency, lifetime, reductions, non-square shapes, non-finite values, and no-Metal behavior. The dispatch alternatives were measured and rejected as insufficiently stable to justify changing the generic default. No kernel fusion, MTLHeap, scheduler, or broad new kernel port was introduced. Phase 4 profiling did not justify an additional numerical kernel optimization: the dominant risks are large-workload memory pressure, synchronization/encoding in hydraulic chains, and the absence of full Xcode Instruments/Metal compiler tooling on this host.

## Phase 4 workload evidence

The following targeted runs used the Release Metal-enabled benchmark binary,
one warm-up, `--benchmark_min_time=1x`, one measured iteration, and the shared
resident storage path. The machine was under variable system load, so these
measurements are comparative evidence rather than hard performance budgets.

| Workload | 1024² | 2048² | 4096² |
|---|---:|---:|---:|
| Resident Chain A, real time | 6.790 ms | 8.971 ms | 27.481 ms |
| Resident Chain A, peak resident | 12.583 MB | 50.332 MB | 201.327 MB |
| Resident Chain C, 10 iterations, real time | 35.747 ms | 136.185 ms | 828.614 ms |
| Resident Chain C, peak resident | 100.696 MB | 402.784 MB | 1.611 GB |

Both chains reported one command buffer and one final synchronization. Chain A
performed no input uploads and one final readback; Chain C performed eight
input uploads and one final readback. Chain C includes advection, thermal, and
hydraulic virtual pipes, so its synchronization and memory growth are expected
to dominate at 4096². The first attempt with an omitted benchmark time suffix
was killed after over-repetition; it is excluded from the measurements above.
