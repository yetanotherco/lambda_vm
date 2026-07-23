# Trace-gen optimization experiments (log)

Experimentation on `main` (uncommitted). Workload: ethrex 10-transfers (6.81 M cycles).
Measured with `cli count-elements --features instruments` (trace-gen only, no proving).
Baseline `trace_build` = 3.33 s (p2a 916 / p2b 680 / p4 602 / p5 549 + overhead).

| # | Change | Result | Verdict |
|---|---|---|---|
| E1 | **Parallelize p4 bitwise collect** (run the ~15 independent, order-independent `collect_bitwise_from_*` collectors via `par_iter`, concat) | p4 **602 → ~410 ms**; trace_build **3.33 → 3.00 s** | ✅ **KEPT** |
| E2 | **Parallelize p2b routing** (rayon `partition` + `par_iter` on the filter/map passes) | p2b **680 → ~766 ms** (regressed) | ❌ **REVERTED** |
| E3 | **Single-pass 3-way partition** in p2b (one classification pass instead of two `partition` calls that move aligned+general ops twice) | p2b **680 → ~585 ms**; trace_build → **~2.79 s** | ✅ **KEPT** |
| E4 | **Walk-fusion**: route each `MemwOperation` into its bucket (register/aligned/general) *during the walk* via a `MemwBuckets` router, so p2b needs no partition at all. Register collector made generic over a `MemwSink` trait so the scratch/test caller still uses a plain `Vec`. | p2b **585 → ~62 ms** (p2a +~75 ms absorbs the routing); trace_build → **~2.43 s** | ✅ **KEPT** |
| E5 | **Move CPU bitwise fan-out (`op.collect_bitwise_ops()`) out of the serial walk into a parallel p4 collector** (rayon `flat_map`-collect over cpu_ops) | p2a −110 ms but **p4 420 → ~1.7 s** (rayon flat_map-collect over 6.8 M per-op Vecs = huge alloc+merge overhead); trace_build → 3.5 s | ❌ **REVERTED** |
| E6 | **Shrink `MemwOperation`** `value`/`old` from `[u64;8]` to `[u32;8]` (elements are bytes or 32-bit register halves; `new`/`with_old` still take `[u64;8]` and convert, so all ~20 construct sites are unchanged — only the persisted struct halves those fields, 216→~152 B) | p2a ~1.02 → ~0.96 s; trace_build → **~2.36 s** (~3%). prove+verify green (no truncation). | ✅ **KEPT** |

| E7 | **Direct-to-column fill for MEMW_R** — register accesses fill the memw_register columns from a compact 48 B `RegRow` instead of materializing a ~152 B `MemwOperation` each (skips the largest table's op-Vec). Prover-only, no executor change. General/aligned fallback still builds `MemwOperation`. | p2a −12% (~110 ms), trace_build −~6.5%. Byte-identical memw_register table (all chunks incl padding) + prove+verify green. | ✅ **KEPT** |

## Final cumulative (trace_build, count-elements, clean runs)
| block | baseline | optimized | Δ |
|---|---|---|---|
| 1-tx | 1.46 s | **1.11 s** | −24% |
| 5-tx | 2.40 s | **1.55 s** | −35% |
| 10-tx | 3.33 s | **2.14 s** | −36% |

Kept: E1 (parallel bitwise collect) + E3 (single-pass partition) + E4 (walk-fusion / MemwBuckets)
+ E6 (MemwOperation shrink) + E7 (direct-to-column MEMW_R). All prove+verify green, integrated on `main` (uncommitted).

| E8 | **Direct-to-column for MEMORY tables** (MEMW_A/LOAD/STORE) — same technique as E7 | Byte-identical + prove+verify green, but **NO speed win** (aligned bucket ~1.4M ops vs register ~16M; below noise). NOT integrated (correct-but-not-worth-it; in worktree). | ➖ shelved |
| E9 | **Bitwise histogram-on-the-fly** — accumulate multiplicities into a dense counter array instead of materializing the ~140M-op / ~560MB `Vec<BitwiseOperation>` | BITWISE table byte-identical + prove+verify green. **Peak RSS −2.27 GB (−22%)**; gen_bitwise fill 4.5× (~354→79ms); trace_build ~flat (p4 histogram alloc offsets it). INTEGRATED. | ✅ **KEPT (memory win)** |

| E10 | **Move in-walk CPU bitwise (~49M ops, ~145ms in p2a) to a parallel p4 histogram fold** (`cpu_ops.par_chunks().fold()` into per-thread histograms) | p2a −120ms but **p4 +650ms** (500→1150ms). The 80MB histogram is too big to allocate/merge many times; the CPU bitwise is cheapest as a sequential append. | ❌ **REVERTED** |

## Integrated final (on main, uncommitted, prove+verify green): trace_build −21%/−32%/−34% (1/5/10-tx) + peak RSS −22%
Kept: E1, E3, E4, E6, E7, E9. Shelved: E2, E5, E8, E10 (E2/E5/E10 regressed; E8 correct/no-win).

## CPU CEILING REACHED
After E7 (register materialization eliminated) + E9 (bitwise Vec eliminated, −22% RSS), the
remaining p2a (~800ms/10-tx) is the memory-state threading + register threading + the
sequential in-walk bitwise append + precompiles — all either cheap-but-necessary or
bandwidth-bound. Every further CPU attempt (E2/E5/E8/E10) either regressed or was below noise.
The histogram approach wins on MEMORY (E9) but its 80MB alloc/merge fixed cost loses on
wall-time for cheap-to-append sources (E10). No reachable CPU walk lever remains without
executor fusion (Tier 2, regressed on registers) or the GPU direction.

## The winning lever for the walk: eliminate materialization (not parallelize/relocate)
- Route B (parallelize the walk): REGRESSED — bandwidth-bound.
- Route A (relocate register predecessor to executor): REGRESSED — just moved cost; registers' predecessor recovery is cheap.
- **E7 direct-to-column (eliminate the register op-Vec materialization): WON (~6.5%).** This is the confirmed technique. Extending it to the MEMORY tables (load/store/memw/memw_aligned — the rest of the walk's materialization) is the next step; the memory walk is the largest remaining p2a cost.

## Learnings (what parallelism/structure can and can't do here)
- **Compute-bound + order-independent → parallelize** (bitwise collect: win).
- **Bandwidth-bound → don't parallelize, restructure** (p2b partition: parallel regressed;
  eliminating the pass via walk-fusion won big).
- **Fine-grained rayon `flat_map`-collect loses to serial `.extend`** (E5).
- **Struct size matters but sub-linearly** — the walk also pays PagedMem lookups + fill compute,
  so a 30% smaller struct bought ~3%, not 30%.
- The remaining walk (`p2a`, deeply interleaved register+memory+precompile in one ts-ordered
  pass) is the last big lever; decomposing it (prover-side parallel predecessor scan for the
  register/memory slices; executor log-enrichment for precompile I/O) is the documented next
  phase (`EXECUTOR-FUSION-PLAN.md`), a soundness-critical multi-crate refactor.

## Correctness
prove+verify on ethrex 1-tx with the optimized code: **Verification succeeded** (1.27 s).
The LogUp bus balances, so no op was dropped/duplicated/misrouted. E1 is byte-identical
(order-independent histogram); E3/E4 preserve push order within each bucket → byte-identical
to the old stable partition.

## Cumulative result (trace_build, count-elements, clean runs)
| block | baseline | optimized | Δ |
|---|---|---|---|
| 1-tx (1.79 M cyc) | 1.46 s | **1.14 s** | −22% |
| 5-tx (4.04 M cyc) | 2.40 s | **1.73 s** | −28% |
| 10-tx (6.81 M cyc) | 3.33 s | **2.49 s** | −25% |

All CPU-only, uncommitted on `main`. Wins: parallel bitwise collect (compute-bound,
order-independent) + walk-fusion (eliminates the bandwidth-bound partition pass). The
front-end walk (`p2a`, ~1.03 s at 10-tx) is now the dominant remaining stage — next levers:
histogram-on-the-fly for the bitwise fill, shrinking the `MemwOperation` struct (SoA), or
executor fusion to delete the re-walk.

## Why E2 regressed — the key finding
`p2b`'s cost is the two `partition`s that move **millions of `MemwOperation` structs**
(register M1/M3/M5 ≈ 3 accesses/instruction). That is **memory-bandwidth-bound**, not
compute-bound — rayon just added coordination overhead without buying anything, because the
bottleneck is DRAM bandwidth, not cores.

The same is true of `p2a` (the walk): `PagedMem` lookups are ~constant (only a handful of
pages — the super-linear hypothesis was a red herring); the real cost is **materializing the
millions of op-structs**. So the whole sequential front-end (`p2a`+`p2b` ≈ 55%) is
bandwidth-bound.

**Implication:** throwing cores at the front-end doesn't work. It needs a *structural* change
that materializes/moves **less**:
- **Fuse the partition into the walk** — have `collect_ops_from_cpu` push directly into the
  register / aligned / general buckets instead of building one `memw_ops` vec and partitioning
  it in `p2b`. Eliminates the two bandwidth-bound partition passes. Byte-parity safe (same ops,
  order within each bucket preserved). *Next CPU experiment.*
- **Shrink the op structs (SoA)** — cut bytes moved per access.
- **Executor fusion** (Tier 2) — emit access records once during execution (executor already
  threads the state), deleting the re-walk entirely. The deepest fix; execution is 40–50×
  cheaper than trace_build so there's headroom.

Where parallelism *does* help: work that is compute-bound and order-independent — e.g. the
bitwise collect (E1). The remaining bitwise headroom is the histogram-on-the-fly (don't
materialize the ~140 M-record Vec at all; accumulate multiplicities directly).

## Current state (uncommitted, 10-transfers)
`trace_build` ≈ **3.00 s** (was 3.33 s). Stages: p2a ~955 / p2b ~700 / p4 ~410 / p5 ~520.
