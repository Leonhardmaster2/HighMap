# HighMap Phase 4 — large resident workloads

Date: 2026-09-04
Host: Apple M3 MacBook Air, 8 GB
Build: Release, Metal ON, OpenCL OFF, runtime MSL

## Protocol

The targeted runs used the resident shared-storage benchmark cases with one
untimed warm-up and one measured iteration:

```bash
./build-phase4-metal-only/bin/highmap_benchmarks \
  --benchmark_filter='BM_Phase3_DeviceArrayShared_ChainA/(1024|2048|4096)/real_time' \
  --benchmark_min_time=1x --benchmark_repetitions=1

./build-phase4-metal-only/bin/highmap_benchmarks \
  --benchmark_filter='BM_Phase3_DeviceArrayShared_ChainC/<size>/10/real_time' \
  --benchmark_min_time=1x --benchmark_repetitions=1
```

`Chain A` is two noise fields, resident gradient normalization, and resident
smooth maximum. `Chain C` is resident advection, thermal, and hydraulic
virtual pipes for 10 iterations. Both cases perform one terminal download.

## Measurements

| Workload | 1024² | 2048² | 4096² |
|---|---:|---:|---:|
| Chain A wall time | 6.790 ms | 8.971 ms | 27.481 ms |
| Chain A peak resident | 12.583 MB | 50.332 MB | 201.327 MB |
| Chain C wall time, 10 iterations | 35.747 ms | 136.185 ms | 828.614 ms |
| Chain C peak resident | 100.696 MB | 402.784 MB | 1.611 GB |

Chain A reported 0 input uploads, 1 final readback, 1 command buffer, and 1
final synchronization. Chain C reported 8 input uploads, 1 final readback, 1
command buffer, and 1 final synchronization. The 4096² Chain C sample
completed successfully; its peak allocation is the relevant memory-pressure
signal for future graph planning. These are the post-rebase validation samples
from `upstream/dev` at `c63c44b16e4ffa0e73f035999009d42f83f8a6dd`; the earlier
Phase 4 exploratory samples remain in the profile history for comparison.

## Caveats

The host load average was high and the benchmark executable reports an
unavailable CPU frequency, so these values are evidence for scale and transfer
behavior, not performance budgets. An earlier exploratory command omitted the
time suffix; this version of Google Benchmark over-repeated the expensive
hydraulic case and the process was killed. That sample is discarded. All
reported values use the explicit `1x` single-iteration setting.
