# Benchmark results and decision gate

The corrected Apple-Silicon build has a common CPU/OpenCL/Metal harness. On the
current host, only the CPU backend can execute: the OpenCL framework exposes no
usable device and Metal reports no usable default device. Consequently, this
file records CPU measurements and the explicit runtime skips; it does not
invent GPU timings.

| Operation | Backend | Shape | Repetitions | Warmup | Wall median (ms) | Pixels/s (wall-derived) |
|---|---|---:|---:|---:|---:|---:|
| gradient_norm | CPU | 128² | 3 | 1 | 0.021535 | 760.8M |
| gradient_norm | CPU | 256² | 3 | 1 | 0.077229 | 848.6M |
| gradient_norm | CPU | 512² | 3 | 1 | 0.302889 | 865.5M |
| gradient_norm | CPU | 1024² | 3 | 1 | 1.158667 | 905.0M |
| maximum_smooth | CPU | 128² | 3 | 1 | 0.004223 | 3.88G |
| maximum_smooth | CPU | 256² | 3 | 1 | 0.016502 | 3.97G |
| maximum_smooth | CPU | 512² | 3 | 1 | 0.066775 | 3.93G |
| maximum_smooth | CPU | 1024² | 3 | 1 | 0.266925 | 3.93G |
| noise | CPU | 128² | 3 | 1 | 0.113074 | 144.9M |
| noise | CPU | 256² | 3 | 1 | 0.457129 | 143.4M |
| noise | CPU | 512² | 3 | 1 | 1.898000 | 138.1M |
| noise | CPU | 1024² | 3 | 1 | 7.380500 | 142.1M |
| gradient_norm | OpenCL | 128²–8192² | — | — | unavailable | no OpenCL device |
| noise | OpenCL | 128²–8192² | — | — | unavailable | no OpenCL device |
| gradient_norm | Metal | 128²–8192² | — | — | unavailable | no usable Metal device |
| noise | Metal | 128²–8192² | — | — | unavailable | no usable Metal device |

The CPU run used `UseRealTime()`, `--benchmark_min_time=0.01s`, three
repetitions and aggregate medians. Inputs were constructed outside the timed
region. Google Benchmark reports an invalid CPU frequency/affinity warning on
this host; that affects metadata only, not the measured durations. This is a
partial size sweep, not a backend-selection policy.

The Metal and OpenCL benchmark cases are still present and explicitly report a
skip/error when their runtime device is unavailable. The current harness does
not yet expose separate upload, dispatch, synchronization and readback timers;
those require a nonblocking/resource-owning backend API and are part of the
next benchmark stage.

For subsequent runs, record one row per operation, size and backend with:

```text
operation, backend, shape, repetitions, warmup,
wall_ms_median, dispatch_ms_median, upload_ms_median,
readback_ms_median, sync_ms_median, pixels_per_second,
iterations_per_second, max_abs_error, mean_abs_error, rmse
```

The decision gate remains open until measurements answer:

* Metal versus OpenCL on this Mac;
* the CPU/GPU crossover size for each representative operation;
* synchronization/copy cost in current and GPU-resident hydraulic paths;
* whether remaining kernels warrant migration;
* which operations should remain CPU-side;
* whether `DeviceArray` composition is justified.
