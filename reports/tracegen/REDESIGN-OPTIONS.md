# Trace generation — redesign options & test plan (CPU)

Synthesis of four parallel investigations (crux / executor-fusion / work-reduction /
CPU-parallelize). Scope: **CPU only** (GPU deferred). Context in [`ANALYSIS.md`](ANALYSIS.md).
Baseline: `trace_build` 1.46 / 2.40 / 3.33 s for 1 / 5 / 10-tx; the sequential front-end
(memory walk + routing) grows to ~47% at 10-tx, the parallel fills shrink to ~17%.

## The convergent picture

All four tracks agree on the shape of the problem and the fix:

- **The sequential memory-model walk (`p2a`) is the #1 lever and it is NOT fundamentally
  sequential.** It only computes each access's predecessor `(old_value, old_ts)` metadata plus
  re-reads precompile operands. That is a "previous-occurrence-per-key" problem →
  **sort-by-(addr,ts) + segmented scan**, or emitted directly by the executor (which already
  threads the state). The *only* true serial residue is precompiles (keccak/ecsm/commit)
  reading live memory mid-walk — and that dissolves once the executor records the I/O it
  already computes (today it discards it and trace-gen redundantly re-runs `keccak_f1600` /
  `compute_witness`).
- **The bitwise fan-out (`p4`) is wasted materialization.** 40–141 M `BitwiseOperation`
  records (~564 MB) exist *only* to be counted into a histogram, and that histogram
  (`gen_bitwise`) runs as a *single serial rayon task*. Accumulate multiplicities directly →
  eliminates the Vec and parallelizes the count. Zero soundness change.
- **The super-linear walk cost is `PagedMem`** (binary-search + insert over a growing page
  Vec), not a HashMap. Fixable with an O(1) page directory even without restructuring.
- **The parallel fills (`p5`) are already near-optimal** — shrinking share, low priority.
- **Execution is 40–50× cheaper than trace_build** (30–80 ms vs 1.46–3.33 s) → huge headroom
  to move work into the executor.

Two tiers of change: **Tier 1** = safe, high-value, ship-independently CPU wins (no
architecture change). **Tier 2** = eliminate the sequential walk (the big structural lever).

---

## Tier 1 — safe parallelization / reduction wins (ship first, independently)

### T1.1 — Bitwise: histogram-on-the-fly (biggest safe win)
**Mechanism.** Replace `collect_bitwise_from_*` → `Vec<BitwiseOperation>` → serial
`update_multiplicities` with **direct per-thread counter arrays** (`[u64; 2^20 × n_types]`),
incremented in place, then tree-reduced (the histogram is a commutative monoid). Fuses `p4`
collect and the serial `gen_bitwise` fill into one parallel pass; never materializes the
~564 MB op stream.
**Target / ceiling.** `p4` (18–27%) **+** the currently-serial bitwise fill inside `p5`.
Removes the largest transient allocation; parallelizes a serial stage.
**Risk.** None to soundness — multiplicity values are bit-identical (addition is
associative/commutative; already order-independent). Pure generation-method change.
**Memory.** ~84 MB per counter array; per-thread copies ~1.2 GB on 14 cores (vs the 564 MB
Vec) — shard by type or use `AtomicU64` if tight.
**Test.** **Byte-parity** of BITWISE multiplicity columns, new vs old, on 1/5/10-tx.

### T1.2 — Parallelize `p2b` routing / op collection
**Mechanism.** `collect_all_ops` is ~10 independent order-preserving `filter/map/collect`
passes + two `partition`s on `memw_ops`. Convert to `rayon::join` / ordered `par_iter`; make
the 3-way MEMW partition a parallel bucketing that preserves order.
**Target / ceiling.** `p2b` (15% → growing); ~3–5× on 14 cores → ~150–200 ms at 10-tx.
**Risk.** Low — order preserved; consumers are order-free except CPU/MEMW which stays ordered.
**Test.** Byte-parity of `CollectedOps` vectors vs sequential + prove+verify.

### T1.3 — Fix `PagedMem` super-linearity
**Mechanism.** Replace the sorted-`Vec` page directory (binary_search + O(pages) insert) with
an O(1) lookup: `HashMap<page_base, Box<Page>>` or a direct-mapped last-page cache + fallback
(footprint is ~98% two contiguous blocks). Drop-in behind the existing `get/set/iter` API.
**Target / ceiling.** Shaves the super-linear tail of `p2a` (~15–25% of `p2a` at 10-tx, more
at higher tx). Helps even after T2 if any sequential residue remains.
**Risk.** Very low. **Test.** Byte-parity + existing page/continuation tests.

### T1.4 — Cache preprocessed columns
**Mechanism.** BITWISE (11 cols), KECCAK_RC, DECODE preprocessed columns are
program-independent (or fixed-per-run) yet regenerated as field elements every proof/epoch.
Memoize into `OnceLock<Arc<[FE]>>`; keep immutable preprocessed buffers separate from
per-proof multiplicity buffers (composes with T1.1).
**Target / ceiling.** Modest for a single proof; compounds linearly across continuation
epochs. Structural enabler for "compute once."
**Risk.** None (values identical). **Test.** Byte-parity; existing static-commitment drift tests.

### T1.5 — Lift `ecsm::compute_witness` off the serial walk
**Mechanism.** The secp256k1 witness (heavy BigInt limb work, ~512 steps/op) currently runs
*inside* the sequential walk. Move it to a post-walk `par_iter` over the ecsm ops.
**Target / ceiling.** Pure win on ecrecover-heavy blocks (ethrex with real signatures);
removes a serial hitter. **Risk.** Low. **Test.** Byte-parity of ECSM/ECDAS tables + prove+verify.

---

## Tier 2 — eliminate the sequential walk (the structural lever, ~47% and growing)

Prerequisite (**the enabler**): **executor records precompile I/O** (keccak in/out lanes,
ecsm operands+result, commit bytes) instead of discarding it, and stops trace-gen's redundant
recompute. Small executor change; unblocks everything below.

### T2.A — Prover-side parallel walk (sort + segmented scan)
**Mechanism.** In one parallel pass emit raw accesses `(addr, ts, new_value, width, kind)`
(all log-derived after the enabler) → **parallel sort by (addr, ts)** → **segmented scan**
fills `old_value`/`old_ts` per access and yields per-address **final (value, ts)** for
PAGE/L2G/register final-state for free. Registers: same on a 34-key space (counting sort or
trivial per-key scan).
**Target / ceiling.** Eliminates the sequential dependency of `p2a` entirely; cost becomes a
parallel sort (O(N log N)) + linear scan. Keeps the executor thin; portable to GPU later.
**Risk.** Medium — must reproduce predecessor semantics exactly; precompile-touching accesses
handled via the recorded I/O (enabler) or a sequential fallback.
**Test.** Byte-parity of produced op vectors vs current `p2a` on the non-precompile subset;
full vectors after the enabler; prove+verify.

### T2.B — Executor-side emission (fusion)
**Mechanism.** Executor tracks `last_ts` per cell (it already has `old_value` = the overwritten
cell) and emits per-access records with `(old_value, old_ts)` attached + precompile I/O. `p2a`
is **deleted**, not parallelized. Evolvable toward the executor writing trace rows directly
(the eventual north star).
**Target / ceiling.** Deletes `p2a` at ~zero executor cost (40–50× headroom). Highest ceiling;
simplest algorithm (no sort). **Risk.** Medium — couples executor↔prover; larger record volume
(stream per epoch, mechanism exists); executor memory grows (but nets out — moved from trace-gen).
**Test.** Byte-parity of `MemwOperation`/`Load`/`Keccak`/`Ecsm` vectors vs current p2a; prove+verify.

### T2.C — CPU-only partial: parallel register walk, no executor change
**Mechanism.** Registers are a 32-key space and the walk records only `old_ts`. Parallelize as
per-register segmented scans (the bulk — memw_register is the largest table by rows), leaving
byte-memory + precompiles on a sequential fallback. No executor change required.
**Target / ceiling.** Attacks the bulk of `p2a` without the enabler; a lower-risk first cut at
the walk. **Risk.** Medium (predecessor semantics; precompile residue stays serial).
**Test.** Byte-parity of memw_register ops + prove+verify.

**Route recommendation.** T2.C first (no executor dep, de-risks the scan), then the enabler +
T2.B (fusion — simplest, biggest ceiling) as the destination. T2.A is the alternative if we
want to keep the executor thin for a future GPU port.

---

## Test plan

**Principles (from the investigations):**
- **Deterministic generation-method changes** (T1.1, T1.2, T1.3, T1.4, T1.5, walk changes) →
  **byte-parity**: assert new output columns/op-vectors equal old, cell-for-cell.
- **Order-independent tables** (anything on the LogUp bus — bitwise, ALU) → **multiset
  equality** (sort/histogram compare), not byte-parity.
- **Anything that could alter the witness or bus** → **prove+verify roundtrip** (mandatory
  end-to-end gate). None of the Tier-1/Tier-2 options should change the constraint system; if
  one ever does, it needs the cross-version verification harness.
- **Keep the old path behind a switch** for A/B — add a `LAMBDA_VM_LEGACY_TRACEGEN` env
  kill-switch (none exists today) so each change is one flag apart at runtime.

**Step 0 — regression harness (do once, first).**
- Add a byte-parity self-check mode: build traces both ways (old vs new) and assert equal
  op-vectors / columns; wire it as an `#[ignore]` test over the 1/5/10-tx fixtures.
- Lock the A/B timing: `cli count-elements --features instruments` (already gives per-stage +
  per-table spans), 3-run median, on the many-core box. This is the perf gate.

**Cheap spikes to validate each ceiling BEFORE full build:**
- **Spike A (T1.1):** prototype the parallel counter-array histogram for the two biggest
  collectors (memw_register, page); measure `p4` + `gen_bitwise` drop. Expect the largest win.
- **Spike B (T1.3):** micro-benchmark `PagedMem` with an O(1) directory replaying the 10-tx
  access trace; confirm the super-linear tail flattens.
- **Spike C (T2.A/C):** emit the access list, run a parallel sort+scan, byte-parity the
  `old_ts`/`old_value` against the sequential walk on the non-precompile subset; measure vs
  `p2a`. (A prior `collect_memory_ops_parallel` probe exists — reuse it.)
- **Spike D (T2 enabler):** measure executor overhead of tracking `last_ts` + emitting records
  — expected to be noise given the 40–50× headroom.

**Correctness gate per change:** byte-parity (or multiset) self-check green on 1/5/10-tx +
prove+verify roundtrip green. **Perf gate:** target stage improves and total `trace_build`
does not regress (3-run median).

**Recommended sequence (dependency- and risk-ordered):**
1. Step 0 harness + kill-switch.
2. **T1.1 histogram-on-the-fly** (biggest safe win) → measure.
3. **T1.2 parallel p2b** + **T1.3 PagedMem O(1)** + **T1.4 preprocessed cache** + **T1.5 ecsm
   lift** — independent, byte-parity-gated, land in parallel → re-measure.
4. **T2.C parallel register walk** (no executor dep) → measure the walk drop.
5. **T2 enabler** (executor precompile I/O) → then **T2.B fusion** (delete p2a) → final measure.

**Expected impact (rough, Amdahl-aware):** Tier 1 attacks `p4` (27%) + `p2b` (15%) + the
`p2a` super-linear tail — most is parallelizable, so on a many-core box these should cut a
large fraction of their stages. Tier 2 removes `p2a` outright (the growing ~47% at scale) —
the decisive structural win. Combined, trace_build should drop well below half at 10-tx, with
the ceiling then set by the (now-parallel) fan-out and fills.
