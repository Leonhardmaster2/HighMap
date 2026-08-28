# HighMap Apple Silicon optimization — Phase 2 review

## Scope and evidence

This review uses the Release benchmark harness on the arm64 Apple Silicon
host, with both the Apple M3 OpenCL device and native Metal device available.
The matrix covers pointwise sizes from 128² through 8192² where practical,
neighborhood sizes through 4096², and thermal/hydraulic iteration counts of
10, 50, 100 and 250. Results and timing semantics are in
`BENCHMARK_RESULTS.md`.

The most important qualification is that the public HighMap API is still
synchronous and returns CPU `Array` values. Therefore the isolated benchmark
numbers include allocation, upload and final readback. Hydraulic is also
measured against an OpenCL benchmark copy that performs the existing
per-dependent-pass readbacks, while Metal keeps state resident.

## Review questions

### 1. Where is Metal faster than OpenCL?

Metal is faster for:

* gradient at 128²–2048², by about 10.2×, 4.8×, 2.0× and 1.15×;
* noise at 128²–2048², by about 5.7×, 2.1×, 1.5× and 1.3×;
* advection at 1024² and 2048², by about 2.7× and 1.9×;
* thermal at the measured 1024² and 2048² iteration counts, generally
  1.3–4.8×;
* hydraulic at 1024² / 10 and / 50, by about 20.5× and 25.1×, and at
  2048² / 10 by about 17×.

At 4096², Metal loses for gradient, noise is effectively tied, advection is
slower, thermal / 10 is nearly tied, and hydraulic remains faster but with a
smaller approximately 6.7× margin. These large-map results are transfer- and
queue-variability-sensitive.

### 2. Where is Metal faster than CPU?

Noise is the clearest CPU comparison: Metal is about 3.8×, 5.9×, 7.5× and
7.8× faster at 512², 1024², 2048² and 4096². It is about 2× faster even at
128² in the measured run.

Metal is not faster than CPU for gradient in the current isolated API. CPU
wins at 128², 512² and 1024², is approximately tied at 2048², and is slightly
faster at 4096². There is no CPU implementation exposed through this harness
for advection, thermal or hydraulic, so no CPU claim is made for those paths.

### 3. Which operations benefit most?

Hydraulic benefits most from execution architecture: one Metal command buffer,
one wait, GPU-resident ping-pong state and GPU volume reduction replace the
OpenCL benchmark's repeated pass readbacks and CPU work. Noise benefits most
among isolated pointwise kernels. Thermal benefits consistently because many
iterations are encoded before the single wait. Advection benefits at moderate
sizes but is sensitive to the large input/output transfer volume.

### 4. Which operations benefit least?

Gradient benefits least relative to CPU and loses to Metal at 4096². Advection
also loses its Metal advantage at 4096². The common cause is not necessarily
the kernel: the synchronous API allocates and transfers every input and reads
back every result for each public call.

### 5. Where does CPU remain preferable?

CPU is preferable for small pointwise work, especially gradient and 128²
noise. CPU gradient remains preferable throughout the measured size range.
For a caller that already owns host-resident data and needs only one simple
operation, Metal's device setup and transfer boundary can outweigh its GPU
throughput.

### 6. What is the crossover size?

For noise, the measured CPU/Metal crossover is between 128² and 512². For
gradient, no Metal-over-CPU crossover is established through 4096². Crossover
points for advection, thermal and hydraulic are not available because the
benchmark harness has no CPU reference API for those operations. A future
scheduler must also include iteration count and current data residency; a
single pixel-count threshold would be misleading.

### 7. What synchronization costs dominate?

Metal performs one public-boundary wait per operation. Examples include one
wait for thermal / 100 and one wait for hydraulic / 10, regardless of the
number of encoded passes. The Metal hydraulic / 1024² / 10 sample records 90
encoders but one command buffer and one synchronization.

The OpenCL hydraulic comparison performs blocking reads after each dependent
pass. At 1024² / 10, its finish/synchronization proxy is 331.7 ms and its
readback is 47.8 ms, versus 16.7 ms GPU timestamp and 0.14 ms final readback
for Metal. OpenCL's timing is explicitly a blocking-finish proxy, not a pure
GPU timestamp.

### 8. What allocation and copy costs dominate?

For gradient / 1024², the representative Metal breakdown is approximately
0.257 ms allocation, 0.301 ms upload and 0.163 ms readback; the OpenCL
breakdown is approximately 0.003 ms allocation, 1.414 ms upload and 0.430 ms
readback. Metal's lower transfer overhead is offset in small cases by the
synchronous command wait and public result boundary.

Hydraulic / 1024² / 10 allocates 20 Metal buffers once, uploads the initial
terrain/rain state, and reads back only the final requested result. The OpenCL
benchmark repeatedly creates/binds image state and reads intermediate maps;
its measured allocation and readback costs are 58.8 ms and 47.8 ms before
counting 331.7 ms of finish/synchronization proxy time.

### 9. Is hydraulic actually GPU-resident?

Yes for the Metal implementation. The loop allocates the state buffers once,
encodes all dependent flow, water, erosion, sediment and evaporation passes,
keeps the buffers on the command buffer between passes, and commits once after
all iterations. Volume maintenance uses the same command buffer for the
hierarchical reduction and rescale passes. The test suite asserts exactly one
command buffer and one synchronization for the stress case, and benchmark
counters report one command buffer for all sampled hydraulic cases.

### 10. Is the speedup architecture or hardware?

Both contribute, but hydraulic is predominantly architecture. The comparison
changes the synchronization and residency model as well as the shader
language, so its 20–25× improvement cannot be attributed to M3 Metal hardware
alone. Noise and advection are more useful hardware/kernel comparisons, but
their end-to-end results still include public API transfers. The review treats
the measurements as backend-path evidence, not a controlled silicon-only
microbenchmark.

### 11. Which optimizations paid off?

The changes with direct evidence are:

* cached Metal pipeline states, eliminating steady-state pipeline creation;
* operation-specific 32-wide pointwise and 16-wide neighborhood dispatch;
* one command buffer and one wait for iterative Metal operations;
* persistent ping-pong buffers for thermal and hydraulic;
* GPU hierarchical reduction and rescaling for hydraulic volume maintenance;
* explicit allocation/upload/readback counters so future decisions can target
  the actual bottleneck.

The build also has a Release-time `metallib` path when standalone Metal tools
are present; this host lacks `xcrun metal`, so the tested build uses runtime
embedded-source compilation.

### 12. Which experiments were not retained?

No private-storage, texture-backed, `MTLHeap` or generic kernel-fusion path is
claimed as a measured win. The current public boundary would still require
staging and readback, so retaining those changes without a resident chain
would add complexity without proving value. Those experiments belong after a
resident data type exists.

The Release `metallib` path is implemented, but this host does not provide the
standalone `xcrun metal`/`metallib` tools. Consequently a before/after startup
measurement of runtime MSL compilation versus a precompiled library is not
available yet; the tested build uses runtime compilation and reports pipeline
creation separately after warm-up. That comparison should be the first
measurement once a tool-complete Apple SDK is available.

### 13. Should more kernels be ported now?

Not broadly. The five staged operation families are enough to expose the
current limiting factor: isolated calls repeatedly cross the host/device
boundary. Porting many more isolated OpenCL entry points would multiply
maintenance while preserving that cost. A small number of reduction and
linear pointwise kernels should be added only to support a resident chain.

### 14. Which next kernels are most promising?

Prioritize reduction primitives and simple pointwise transforms that can reuse
resident buffers, followed by operations that naturally form a terrain chain.
Hydrology reductions are already the proof point. Texture sampling and more
neighborhood kernels should wait for a buffer-versus-texture experiment on a
resident workload.

### 15. Is a `DeviceArray` worth implementing?

Yes. It is the highest-value next milestone. A `DeviceArray` should own shape,
format, Metal resource, storage mode, dirty state, queue/lifetime dependency
and explicit conversion to/from `Array`. It would allow composed operations to
avoid per-stage allocation, upload and readback, make private storage useful,
and give the scheduler the residency information needed for a real crossover
policy.

## Decision and next step

Phase 2 validates the Metal execution model and identifies a concrete
architecture win, but it does not justify a universal GPU threshold or a
second wave of isolated ports. Proceed with the `DeviceArray` boundary and
measure composed chains, shared versus private storage, and buffer versus
texture sampling there. Keep CPU gradient and small pointwise calls available
as first-class choices.
