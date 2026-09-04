# HighMap Phase 4 — Metal library validation

Date: 2026-09-04

## Available path

The tested Release Metal build reports:

```text
HIGHMAP_METAL_AVAILABLE=ON
HIGHMAP_METAL_PRECOMPILED=OFF
HIGHMAP_METAL_RUNTIME_COMPILE=ON
```

Metal source is embedded and compiled at runtime through the existing
Objective-C++ implementation. The Metal focused suite on the Metal-only
configuration ran 45 tests, with 33 passing and 12 expected skips. The passing
tests cover backend availability, runtime pipeline creation, resident session
ownership, transfer statistics, reductions, non-square shapes, non-finite
values, thermal/hydraulic behavior, and repeated session chaining.

## Toolchain limitation

The host uses the installed Command Line Tools rather than a full Xcode
developer directory. These probes fail because the standalone tools are not
installed:

```text
xcrun --find metal    -> unable to find utility "metal"
xcrun --find metallib -> unable to find utility "metallib"
xcrun xctrace list templates -> unable to find utility "xctrace"
```

Therefore this phase does not claim precompiled `.metallib` validation or an
Instruments/Metal System Trace capture. Runtime source compilation is the
authoritative shader path on this host and is covered by the executable tests.

## Release implication

The source path is qualified for this Mac's runtime Metal device. A packaged
precompiled library remains a separate packaging check that requires full
Xcode or an equivalent Metal toolchain; enabling it was not faked by checking
for headers alone.
