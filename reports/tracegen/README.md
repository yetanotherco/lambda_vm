# Trace-generation baseline — ethrex 1-tx block

Phase-0 measurement for the "make trace generation faster" effort. Adds per-stage +
per-table timing to `trace_build` and captures a baseline on the real workload.

## Setup
- Workload: `ethrex.elf` + `executor/tests/ethrex_simple_tx.bin` (1 tx, **1,792,841 cycles**).
- Machine: macOS, 14 cores, 36 GB — **CPU-only (no CUDA)**. So the *proving* phase here
  is CPU LDE/Merkle (slow); on the GPU box proving collapses and trace-build — which is
  CPU either way — becomes a much larger share. The trace-build numbers below are
  GPU-independent.
- Build: `cargo build --release -p cli --features instruments`
- Run: `LAMBDA_VM_TIMELINE_JSON=... cli prove ethrex.elf --private-input <bin> --time`
- 3 runs; trace-build was stable to ~2%.

## Result — trace_build ≈ 1.46 s (median), broken down

Wall-clock span tree (serial critical path; the `fill:*` lines are per-table CPU-time
contributions that run in parallel *under* `p5_generate_tables`):

```
trace_build                     1.46 s   (100% of trace build)
  p0_decode                    ~31 ms      2%
  p1_cpu_ops                   ~36 ms      2%
  p2a_collect_cpu (mem walk)  ~184 ms     13%   sequential by nature
  p2b_collect_all             ~222 ms     15%   sequential (routing + register ops)
  p3to5_build_traces          ~836 ms     57%
    p4_bitwise_collect        ~399 ms     27%   <-- fan-out collection
    p5_generate_tables        ~410 ms     28%   critical path ≈ PAGE fill
      fill:gen_pages          ~375 ms            <-- the p5 tall pole
      fill:gen_bitwise        ~150 ms            (overlaps under pages)
      fill:gen_lts             ~89 ms            (overlaps)
      fill:gen_cpus            ~65 ms            (overlaps)
      … all other fills < 50 ms, overlap under pages
```

## Real op volume (the fan-out multiplier)
```
cpu                     1,792,841     (= cycles; one CPU row per instruction)
memw_register           3,884,623     <-- largest real table by rows
memw_aligned              482,594
load                      266,826
store                     223,754
lt   (post-fanout)        708,337
bitwise (post-fanout)  39,991,182     <-- ~22× the instruction count
```

## What the data says (levers, ranked)

1. **BITWISE is the single biggest theme — ~550 ms (~38% of trace build).**
   `p4_bitwise_collect` (~399 ms) + `fill:gen_bitwise` (~150 ms). Driven by **~40 M**
   bitwise lookup ops fanned out from 1.8 M instructions. The collect is a chain of
   sequential `.extend()` calls, each `collect_bitwise_from_X` independent and
   order-independent (multiplicities are summed) → **parallelize the collect** (per-source
   vecs in parallel, then merge). Likely a large, low-risk win.

2. **PAGE fill is the parallel-fill critical path — ~375 ms.** `fill:gen_pages` ≈ the
   whole `p5` wall because every other fill overlaps beneath it. It's one closure building
   ~5 pages of 262 K rows each. **Spawn pages as independent rayon tasks** (and/or tighten
   the per-page fill) to drop the p5 tall pole. (PAGE is also ~5×0.3 s downstream in
   rounds 2–4 — worth a hard look.)

3. **Sequential front-end — ~406 ms** (`p2a` mem walk ~184 ms + `p2b` collect_all
   ~222 ms). Inherently sequential (state threading); lower ceiling, harder. Prior probe
   confirmed the memory walk alone is not the lever. Revisit only after 1 & 2.

Note: the heavy *individual* table fills (CPU 65 ms, LT 89 ms, MEMW_R 46 ms) are **not**
on the critical path today — they already overlap under PAGE. Optimizing them in isolation
won't move trace-build wall until PAGE and the bitwise collect shrink.

## Reproduce
```
make executor/program_artifacts/rust/ethrex.elf
cargo build --release -p cli --features instruments
LAMBDA_VM_TIMELINE_JSON=reports/tracegen/timeline_run1.json \
  target/release/cli prove executor/program_artifacts/rust/ethrex.elf \
  --private-input executor/tests/ethrex_simple_tx.bin --time --cycles \
  -o /tmp/x.proof > reports/tracegen/prove_run1.stdout 2> reports/tracegen/prove_run1.stderr
```
Raw logs: `prove_run{1,2,3}.{stdout,stderr}`, `timeline_run{1,2,3}.json`.
