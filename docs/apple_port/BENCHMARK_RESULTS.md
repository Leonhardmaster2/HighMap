# Apple backend benchmark results — Phase 2

## Environment and protocol

Measurements below were collected on the arm64 Apple Silicon host used for
this port: macOS 27.0, Apple Clang 21, Release C++20, with both the Apple M3
OpenCL device and the native Metal device available. The benchmark executable
was built from `feature/apple-metal-backend` after the Phase 1 checkpoint.

Inputs are constructed outside the timed region. Each case performs one
untimed warm-up, then three or five measured repetitions unless noted. The
reported `real_time` is the end-to-end host-visible operation, including
allocation, upload, command submission, synchronization and result readback.
The harness also records backend timing breakdowns; Metal GPU time comes from
command-buffer timestamps when available. OpenCL's `device_or_finish_ms` is the
blocking `Run::execute` duration and is a finish/synchronization proxy rather
than a pure GPU timestamp.

The registered matrix is:

* pointwise gradient, smooth extrema and noise: 128², 256², 512², 1024²,
  2048², 4096² and 8192²;
* advection, thermal and hydraulic: 128², 256², 512², 1024², 2048² and
  4096²;
* thermal and hydraulic iteration counts: 10, 50, 100 and 250.

High-cost hydraulic cases at 2048² and above were sampled at lower iteration
counts where a complete 250-iteration run was not practical. The executable
still exposes every requested size/iteration combination for targeted runs.

## End-to-end matrix samples

### Pointwise operations

| Operation | Size | CPU | OpenCL | Metal |
|---|---:|---:|---:|---:|
| gradient_norm | 128² | 0.022 | 3.11 | 0.305 |
| gradient_norm | 512² | 0.305 | 2.20 | 0.456 |
| gradient_norm | 1024² | 1.21 | 3.06 | 1.50 |
| gradient_norm | 2048² | 5.41 | 6.15 | 5.35 |
| gradient_norm | 4096² | 24.1 | 14.3 | 26.2 |
| gradient_norm | 8192² | — | 48.9 | 59.4 |
| noise | 128² | 0.124 | 1.41 | 0.248 |
| noise | 512² | 1.96 | 1.08 | 0.515 |
| noise | 1024² | 7.46 | 1.93 | 1.27 |
| noise | 2048² | 28.5 | 5.02 | 3.81 |
| noise | 4096² | 116 | 14.8 | 14.9 |
| noise | 8192² | — | 57.1 | 178 |

CPU gradient remains faster through this measured range; Metal's best
gradient result is approximately tied with CPU at 2048². Noise is the
stronger Metal candidate, with a clear advantage over CPU from 512² onward
and a 1.3–5.7× advantage over OpenCL in the 128²–2048² samples. The 8192²
noise result is transfer-bound and was highly variable, so it is not used as a
selection threshold.

### Neighborhood and iterative operations

| Operation | Size / iterations | CPU | OpenCL | Metal |
|---|---:|---:|---:|---:|
| advection_warp | 1024² | — | 15.0 | 5.57 |
| advection_warp | 2048² | — | 40.4 | 21.1 |
| advection_warp | 4096² | — | 139 | 229 |
| thermal | 1024² / 10 | — | 9.16 | 2.79 |
| thermal | 1024² / 50 | — | 39.0 | 8.09 |
| thermal | 1024² / 100 | — | 35.5 | 13.8 |
| thermal | 1024² / 250 | — | 86.8 | 32.4 |
| thermal | 2048² / 10 | — | 24.5 | 12.9 |
| thermal | 2048² / 50 | — | 48.9 | 37.4 |
| thermal | 2048² / 100 | — | 92.7 | 64.5 |
| thermal | 2048² / 250 | — | 222 | 151 |
| thermal | 4096² / 10 | — | 40.7 | 46.1 |
| hydraulic_vpipes | 128² / 10 | — | 46.1 | 0.652 |
| hydraulic_vpipes | 128² / 100 | — | 456 | 3.37 |
| hydraulic_vpipes | 1024² / 10 | — | 475 | 23.2 |
| hydraulic_vpipes | 1024² / 50 | — | 2392 | 95.3 |
| hydraulic_vpipes | 2048² / 10 | — | 1583 | 93.2 |
| hydraulic_vpipes | 4096² / 10 | — | 6533 | 971 |

HighMap does not currently expose CPU reference implementations for
`advection_warp`, `thermal` or `hydraulic_vpipes` with the benchmark API, so
those CPU cells are intentionally marked unavailable rather than filled with
an invented reference implementation. At 4096², advection and hydraulic show
that allocation and readback can erase the benefit of the kernel itself.

## Timing breakdowns

The following are representative warm medians from instrumented runs. Times
are milliseconds per public operation; bytes are decimalized for readability.

| Operation / backend | Allocation | Upload | Pipeline lookup | Encoding | GPU or finish proxy | Wait / sync | Readback | Host between passes | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gradient 1024² / OpenCL | 0.003 | 1.414 | 0.087 | 0 | 0.701 | 0.701 | 0.430 | 0 | 2.73 |
| gradient 1024² / Metal | 0.257 | 0.301 | 0.000 | 0.003 | 0.114 | 0.635 | 0.163 | 0 | 1.43 |
| thermal 1024² / 100 / OpenCL | 2.750 | 1.177 | 0.092 | 0 | 33.516 | 33.516 | 0.331 | 0 | 35.47 |
| thermal 1024² / 100 / Metal | 0.302 | 0.636 | 0.125 | 0.035 | 11.867 | 12.436 | 0.294 | 0 | 13.81 |
| hydraulic 1024² / 10 / OpenCL | 58.840 | 0 | 6.590 | 0 | 331.706 | 331.706 | 47.842 | 13.225 | 474.96 |
| hydraulic 1024² / 10 / Metal | 3.593 | 0.546 | 0.460 | 0.221 | 16.747 | 17.780 | 0.142 | 0 | 23.18 |

For the OpenCL hydraulic copy, image binding and host-visible allocation are
reported in the allocation column, so upload is intentionally zero. Every
flow, water, erosion and sediment pass is followed by a blocking output read
in that benchmark copy; `host_between_pass_ms` includes the CPU volume and
state work. This is why the hydraulic comparison is primarily an execution
architecture comparison, not a pure Metal-versus-OpenCL shader comparison.

An independent 15-sample run at 1024² for `gradient_norm` produced this wall
time distribution:

| Backend | P10 | Median | P90 | Population stddev |
|---|---:|---:|---:|---:|
| CPU | 1.160 | 1.160 | 1.186 | 0.047 |
| OpenCL | 2.060 | 2.590 | 3.514 | 0.821 |
| Metal | 1.536 | 1.590 | 2.808 | 0.686 |

The Metal and OpenCL tails include occasional driver/queue scheduling spikes;
medians are therefore used for the matrix and crossover discussion.

## Residency and command submission evidence

| Operation | Metal command buffers | Metal synchronizations | Encoders | Readback behavior |
|---|---:|---:|---:|---|
| gradient 1024² | 1 | 1 | 1 | final result only |
| thermal 1024² / 100 | 1 | 1 | 100 | final result only |
| hydraulic 1024² / 10 | 1 | 1 | 90 | final terrain only in benchmark |

Hydraulic uses 5 simulation encoders per iteration plus one GPU reduction and
rescale sequence when volume maintenance is enabled. It allocates 20 buffers
for the complete state and commits once after all iterations. No CPU readback
or queue wait occurs between hydraulic passes or iterations. The correctness
suite asserts the one-command-buffer/one-synchronization contract directly.

## Current decision

Metal is the preferred path for noise, advection at moderate sizes, thermal,
and hydraulic when a single operation owns the full state. CPU remains the
preferred path for small pointwise work and for gradient across the measured
range. Large 4096² neighborhood calls need a residency-aware API before a
strong selection rule can be justified; the next milestone is a GPU-resident
`DeviceArray`, not a broad second wave of isolated kernels.
