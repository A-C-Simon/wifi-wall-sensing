# Code audit

The audit covered acquisition, parsing, calibration, motion scoring, map
matching, validation protocols, live displays and diagnostic probes. Fixes
were committed and pushed separately.

## Main corrections

- Removed future scan evidence from earlier motion scores and made fine
  windows causal. Calibration and measurement use the same saved parameters.
- Prevented unsupported localization and degenerate map validation. Matching
  uses measured shared anchors, and changed maps require a new verdict.
- Aligned validation phases with capture timestamps. Occupancy uncertainty
  uses independent A/B pairs rather than correlated fine samples.
- Fixed native termination, stale live decisions, partial record reads,
  malformed inputs, fractional durations and coordinate collisions.
- Fixed probe compilation, latency units, ping invocation and input errors.
  Invalid RSSI readings now discard the partial acquisition bucket.
- Reduced fine feature work with prefix sums, retained live filter state,
  prepared map matching once and bounded pairwise separation memory.

## Verification

- All 57 Python regression tests pass with the Agg plotting backend.
- The native capture program and all five retained Swift probes compile.
- Probe input checks reject invalid durations and missing interfaces.
- All 21 existing recordings parse, and all three existing radio maps load.
- Integration checks exercised baseline analysis, repeatability, map
  validation, localization, heatmaps, coarsening and distance calibration.
- Short native captures completed under a time limit and SIGTERM. Headless
  dashboard rendering checked missing-data labels.

Local benchmarks measured about 117 times less time for fine feature
extraction on 40,000 buckets. Cached live scoring took about 0.9 ms versus
61 ms for replaying that history. A 500-cell, 28-anchor separation check used
about 0.11 MiB of temporary Python-tracked memory versus 106.81 MiB, while
preserving the result. These timings depend on the machine and workload.

No new operator-led occupancy or localization experiment was performed.
Synthetic tests, archived recordings and short capture checks establish
software behavior, not improved physical detection accuracy. The traffic
probe was compiled but its full traffic experiment was not rerun. Capture
gap handling was compiled but not tested by disconnecting the interface.
The hardware and scan cadence limits described in the README still apply.
