# Trace generation — analysis & redesign brief

Consolidated understanding of the `trace_build` phase, the measured cost breakdown and
how it scales, and the framing for a ground-up redesign. Baseline run details in
[`README.md`](README.md); this doc is the shared brief for the redesign effort.

## The contract (fixed — a redesign may change everything *between* these)

**Inputs**
- **Execution log** — `Vec<Log>`, one record per executed instruction:
  `{current_pc, next_pc, src1_val, src2_val, dst_val}` (40 B, POD, fixed stride). Timestamps
  are *positional* (`ts = 4·i + 4`), assigned in trace-gen, not by the executor. ECALLs
  repurpose the value fields (syscall no. + params).
- **ELF** — instruction stream (decoded once), initial memory image, entry point.
- **Private input** — raw bytes (seeds memory; marks private pages).

**Outputs**
- ~27 **trace tables** (main columns = the witness), each padded to a power-of-two height
  and split into chunks. Column layouts are fixed by the AIR. These feed LDE + Merkle commit.
  Many tables ride a **permutation-invariant LogUp bus** → row *order within a table is free*
  (only the multiset matters); a few (CPU, memory, keccak) are position-sensitive.

Field: Goldilocks, **direct canonical u64** (no Montgomery). Tables are effectively
row-major `Vec<u64>`.

## The pipeline today (instrument span names)
- **p0_decode** — ELF → per-instruction fields (once; program-sized). Cheap (~31 ms, flat).
- **p1_cpu_ops** — log → CPU operations: assign ts, resolve register reads, compute ALU /
  branch results. Sequential, cheap.
- **p2a_collect_cpu** — **the memory-model walk**: replay every memory & register access in
  time order, threading `MemoryState` (byte→(val,ts)) + `RegisterState` (32-entry array), so
  each access records its *predecessor's* (value, timestamp). **Sequential** (last-writer
  dependency).
- **p2b_collect_all** — partition MEMW into register / aligned / general, gather register
  accesses (M1/M3/M5 + PC), collect BRANCH/MUL/DVRM/EQ/… from CPU fields. Sequential.
- **p3/p4 fan-out** — MEMW→LT (timestamp-ordering checks), then **all→BITWISE**: every table
  emits its range-check / bit-decomposition lookups into one op stream. Generates **tens to
  hundreds of millions** of bitwise ops. Order-independent `.extend()` chains.
- **p5_generate_tables** — fill every column of every row for all ~27 tables, pad to 2ⁿ,
  chunk. **Parallel** (`rayon::scope` per table + `into_par_iter` per chunk).

## Measured cost & scaling (macOS 14-core, CPU-only, `cli count-elements`)

Stage wall time (ms) — **1-tx / 5-tx / 10-tx** (cycles 1.79 M / 4.04 M / 6.81 M):

| stage | 1-tx | 5-tx | 10-tx |
|---|---|---|---|
| p0 decode | 31 | 31 | 31 |
| p1 log→cpu ops | 36 | 83 | 129 |
| **p2a memory walk** | 184 | 518 | **898** |
| **p2b route/collect** | 222 | 428 | **680** |
| p4 bitwise collect | 399 | 510 | 602 |
| p5 fills (parallel) | 410 | 501 | 549 |
| other/overhead | 178 | 329 | 441 |
| **TOTAL trace_build** | **1.46 s** | **2.40 s** | **3.33 s** |
| bitwise ops (post-fanout) | 40 M | 86 M | 141 M |

**Scaling facts**
- Total is *sub-linear* per cycle (0.82 → 0.59 → 0.49 µs/cyc) — fixed costs amortize.
- **Sequential front-end (walk + route) share grows: 28% → 39% → 47%.** The memory walk is
  *super-linear* per cycle (103 → 128 → 132 ns/cyc). **Correction (confirmed by investigation):**
  the cause is NOT a HashMap — `MemoryState` is `PagedMem` (dense page arrays + a `binary_search`
  over a growing `Vec` of touched pages, plus O(pages) `Vec::insert` on first touch). Sort+scan
  or an O(1) page directory eliminates it.
- **Parallel fills shrink in share: 28% → 21% → 17%** — they already use every core.
- Per-table fills have ~10–20% run-to-run variance (parallel scheduling); at 5–10 tx **CPU
  and PAGE** trade the top spot.

## Lever ranking (corrected for scale)
1. **Sequential front-end (memory walk + op routing)** — dominant and growing (~47% at 10 tx),
   but inherently serial (last-writer dependency) → the hard, high-value target.
2. **Bitwise fan-out collect** — 18–27%, order-independent → the tractable parallelization win.
3. **Fills** — already parallel, shrinking ceiling → low priority.

## The crux question that gates the whole solution space
The walk is a "previous-occurrence-per-key" computation — classically parallelizable via
**sort-by-(addr,ts) + segmented scan**. The open question: **is sequential replay actually
necessary, or are all values already in the executor log** (so the walk reduces to pure
metadata bookkeeping)? Prior probes validated a parallel sort+scan for byte-memory accesses
byte-for-byte, but flagged a **precompile blocker** (keccak/ecsm/commit "read live memory
mid-walk"). Whether that blocker is real *given the executor already recorded precompile I/O*
is the first thing to settle — it decides whether the #1 lever can be parallelized on CPU or GPU.

## North star (user)
Eventually the **whole prover on GPU** — "witness in, proof out, nothing in between." So a
redesign that lands the witness *device-resident* (feeding LDE with no host↔device copy) is
strategically preferred, even if intermediate transfers look bad in isolation.
