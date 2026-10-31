# Stage A — CENSUS FIT MAP

**Result: Gate A FAILS at every point.** No `(epoch size, preset)` in the sweep fits
93 GiB (box) or 110 GiB (rigs) under keccak-inner. The cheapest real-block point
overshoots by 13×; the worst by 78×.

Measured 2026-08-12 on the rented GPU box (79.161.122.162, 32 cores / 93 GiB / RTX
5090), repo `/root/lambda_vm` @ `blake3-real-hash`.

**★ All artifacts are checkpointed OFF the rented box** (it will not outlive the
campaign) at `~/workspace/lambda_vm_bench_cache/lfm_census_2026-08-12/`:

| file | what |
|---|---|
| `census_harness.diff` | the 374-line harness — **re-derive the reviewed commit from this** |
| `census_results.md` | all points + verdict tables as produced |
| `census_logs/` | 28 raw per-point logs, incl. full leg-shape dumps |
| `project.py`, `fitmap.py`, `final.py` | analytic chip model + Gate-A projections |
| `tower.py` | the Gate-D1 tower model |
| `census_run.sh`, `sweep.sh` | runners — `census_run.sh <tag> <preset> <epoch_log2> <elf> <input> <timeout> <mode> <queries>` reproduces any point |

On the box itself (while it lives): the same files under `/root/`.

**Inner workload:** `ethrex.elf` (sha `133816f0`) + real mainnet block **25368371**
(`ethrex_mainnet_25368371.bin`, 1,110,156 B, sha `61eba49b`). Every ethrex point
censuses **epoch 0** of that block. Per-epoch profiles vary across a block, so a
later epoch would move the trace-length profile somewhat — but not the verdict,
which fails by more than an order of magnitude.

---

## 1. The fit map

Projected peak = census cells × 33.7 bytes/cell (`MEASURED_BYTES_PER_CELL`, the
slice-0 anchor: 481,327,124 cells → 15.1 GiB RSS).

| epoch | sub-proofs | preset | total keccak perms | KECCAK_RND chunks | cells | projected peak | fits 93 GiB | fits 110 GiB |
|---|---|---|---|---|---|---|---|---|
| 2^4 fixture | 25 | blowup2/219q | 311,214 | 15 | 22.3B+ | 701 GiB | NO (7.5×) | NO |
| 2^4 fixture | 25 | blowup4/110q | 164,120+ | 8 | 12.7B+ | 399 GiB | NO (4.3×) | NO |
| **2^20** | 28 | blowup2/219q | 920,273 | 43 | 72.0B | 2,261 GiB | NO (24×) | NO |
| **2^20** | 28 | blowup4/110q | 490,998 | 23 | 38.2B | **1,199 GiB** | **NO (13×)** | NO |
| **2^21** | 32 | blowup2/219q | 1,194,441 | 55 | 93.3B | 2,929 GiB | NO (31×) | NO |
| **2^21** | 32 | blowup4/110q | 637,057 | 30 | 49.5B | 1,554 GiB | NO (17×) | NO |
| **2^22** | 43 | blowup2/219q | 1,797,233 | 83 | 140.8B | 4,421 GiB | NO (48×) | NO |
| **2^22** | 43 | blowup4/110q | 960,063 | 44 | 74.8B | 2,348 GiB | NO (25×) | NO |
| **2^23** | 64 | blowup2/219q | 2,895,737 | 133 | 230.6B | 7,242 GiB | NO (78×) | NO |
| **2^23** | 64 | blowup4/110q | 1,546,363 | 71 | 122.5B | 3,848 GiB | NO (41×) | NO |

✓ MEASURED per point: sub-proof count, per-chip log-heights (full leg-shape dumps in
the logs), closed-form leg permutations, spine permutations and spine instruction mix.
? PROJECTED: cells at the real query counts (analytic chip model, §3) and peak RSS
(the 33.7 B/cell coefficient, §5).

### Epoch trace-length profiles (log2), measured

| epoch | profile |
|---|---|
| 2^4 fixture | `[2 ×15, 3, 4 ×4, 5 ×3, 7, 20]` |
| 2^20 | `[2 ×10, 4, 5, 7, 17 ×4, 18 ×2, 19 ×4, 20 ×4, 21]` |
| 2^21 | `[2 ×8, 5, 7, 10 ×2, 12, 17, 18 ×2, 19 ×8, 20 ×7, 22]` |
| 2^22 | `[2 ×7, 5, 7, 11, 13 ×2, 14, 17, 18 ×2, 19 ×12, 20 ×14, 22]` |
| 2^23 | `[2 ×7, 5, 7, 11, 14, 15, 16, 19 ×25, 20 ×25, 22]` |

---

## 2. Two results that need no extrapolation

### 2a. The 219-query program cannot even be BUILT

The assembled verifier at blowup2/219q on the **16-cycle fibonacci fixture** — the
smallest epoch that exists, 25 sub-proofs — was **OOM-killed (signal 9) at 89.1 GiB
during EMISSION**, after 3 m 56 s, before any proving began.

Emission is strictly cheaper than proving, so this alone settles the gate: the
smallest possible epoch at the cheapest secure preset does not fit on the box even
as a program in memory.

### 2b. Eight of the required 219 queries already exceed the box

Fully emitted censuses of a **real 2^21 block epoch at blowup2**, query count reduced
via `LFM_CENSUS_QUERIES` (never a security claim — the count travels with every number):

| queries | cells | chunks | projected peak | fits 93 GiB |
|---|---|---|---|---|
| 4 | 1,965,702,420 | 2 | 61.7 GiB | YES |
| 8 | 3,706,469,652 | 3 | 116.4 GiB | **NO** |
| 16 | — | — | 225.7 GiB | **NO** |

Spine/leg split at the q=4 point: spine 1,725,376 instr / 2,667 perms / 9,010 words;
legs 4,948,202 instr / 21,736 perms / 75,804 words — i.e. **5,434 leg permutations per
query**, closed-form CHECKED against `epoch_verify::query_permutations`.

---

## 3. The scaling law

**Linear in query count.** Measured ×1.89 then ×1.94 across a 4× span (61.7 → 116.4 →
225.7 GiB). The shortfall below an exact ×2 is fixed-height chips (BITWISE at 2^20,
KECCAK_RC, LFM_RANGE) plus power-of-two row padding, both of which dilute as the point
grows — so linearity is the right extrapolation and it is mildly conservative.

**Roughly linear in sub-proof count**, which grows 28 → 32 → 43 → 64 across 2^20…2^23
as VM tables split at `max_rows`.

**Logarithmic in epoch size per table** (Merkle depth and committed FRI layers).

Net: the epoch-size lever is **weak** — 2^23 → 2^20 buys only 3.2× — and it multiplies
the number of wraps the tower must then aggregate, so it partly pays itself back.

**One chip is the bill.** `KECCAK_RND` is **92.5% of cells** at the real 2^21 point
(87.6% at the fixture; the share rises with size). One full chunk is 2^19 rows ×
(1480 main + 3×516 aux) = 3,028 base-field-equivalent cells/row = 1.588B cells =
12.7 GB raw trace = **49.8 GiB projected**. Therefore:

> The 93 GiB budget holds ~**40.8k permutations** — 1.9 chunks — in total.
> The sweep needs 491k…2.90M.

Equivalently, per point, the query count that *would* fit 93 GiB: 2^20 → 9.7 queries;
2^21 → 7.5; 2^22 → 5.0; 2^23 → 3.1. Against a required 219.

### Where the permutations go (2^21/blowup2 leg dump)

Two distinct cost centres, which matter because they respond to different levers:

- **Wide tables → leaf absorption.** Leg idx 4 is the inner proof's own `KECCAK_RND`
  sub-proof: only 2^2 rows, but 1480 main + 516 aux columns, so one query's leaf costs
  ~359 permutations. 79,935 perms at 219 queries from a 4-row table. Independent of
  epoch size.
- **Deep tables → Merkle + FRI paths.** Leg idx 31 at 2^22 rows / depth 22 / 14 FRI
  layers costs 288 perms per query almost entirely in path steps. Grows with epoch size
  and with table count.

---

## 4. Levers, and the headline conclusion

From the cheapest real point, **2^20 / blowup4 / 110q = 1,199 GiB (13× over)**:

| lever | result | still over 93 GiB |
|---|---|---|
| baseline | 1,199 GiB | 13× |
| + inner hash blake3-6r (4.06×, the plan's own hash matrix 11.17B → 2.75B) | **295 GiB** | **3.2×** |
| + a further 2× from anywhere | 148 GiB | 1.6× |
| + a further 4× from anywhere | 74 GiB | fits |

> **The inner-hash switch is NECESSARY BUT NOT SUFFICIENT.**

The plan anticipated that a keccak-inner failure would promote the inner-hash switch
"from optimization to prerequisite". The measurement says something stronger: after the
switch, the best point in the whole sweep is still **3.2× over** the box.

### Coefficient-free floor — this is not an artifact of the 33.7 B/cell anchor

Counting only the raw committed trace at 8 bytes per felt — zero LDE, zero Merkle trees,
zero quotient, zero allocator overhead:

| epoch | preset | raw trace | vs 93 GiB | after blake3-6r |
|---|---|---|---|---|
| 2^20 | blowup4/110q | 285 GiB | 3.1× | 70 GiB (fits) |
| 2^21 | blowup2/219q | 695 GiB | 7.5× | 171 GiB (1.8×) |
| 2^23 | blowup2/219q | 1,718 GiB | 18.5× | 423 GiB (4.6×) |

No prover, however efficient, holds the 2^20/blowup4 trace in 93 GiB today. The
blake3 switch brings the *floor* under the budget at exactly one point — which is
what makes the residual ~3.2× a question about prover residency rather than about
the census.

### The remaining ~3.2× — candidate

**Bounded-residency proving.** Peak is currently the *sum* over 23–133 `KECCAK_RND`
chunks because `airs.air_trace_pairs(&mut traces)` hands every trace to a single
`multi_prove` call. One resident chunk at a time would be ~50 GiB regardless of chunk
count. There is a `disk-spill` feature and a `StorageMode` enum already in the tree
(`StorageMode::Ram` is passed explicitly at the epoch-prove sites), which may already
be most of this.

> **★ ANSWERED — this paragraph was written before the read; see
> `residency-seam-audit.md` and Part 2 §1.** Two things above are now known to be
> wrong: `disk-spill`/`StorageMode` do **not** already provide most of this (the
> LDE is never spilled and the path is unreachable from LFM), and "~50 GiB
> regardless of chunk count" needs the TRACE streamed as well as the LDE dropped —
> LDE-only bounding lands at 309-819 GiB. Bounded residency is a real refactor with
> named seams (S1-S7), not a flag. The `airs.air_trace_pairs` observation stands.

✗ **UNCERTAIN at the time of writing — I had not read `multi_prove`'s residency
behaviour.** That read was the cheapest next step and is now done; it determined
that the hash switch alone does **not** close the gate at 2^20/blowup4.

Adjacent parked item: `max_rows` / `KECCAK_RND_MAX_CHUNK_ROWS` tuning changes the chunk
*count* but not total rows, so it does **not** move peak unless residency is bounded.

---

## 5. Method, and what the numbers cannot see

**The analytic chip model** (`project.py`) reproduces `lfm_chip_census` from shapes
alone: per-chip padded rows = `next_pow2(instruction count)`, `KECCAK_RND` rows from the
chunk policy (21,845 perms per 2^19-row chunk), fixed heights for `BITWISE` (2^20),
`KECCAK_RC` (32), `LFM_RANGE` (2^16). It was validated against **three independent
measured censuses** — the fixture at 1 query, and the real 2^21 epoch at 4 and 8
queries — reproducing `main`, `aux`, total cells and chunk count **exactly** in every
case. Leg instruction cost per query came from differencing the emitted q=8 and q=4
censuses (the spine cancels exactly); spine permutations and mixes are measured
directly at the real query counts via `LFM_CENSUS_SPINE_ONLY`.

**The 33.7 B/cell coefficient is the one soft link.** It rests on a single anchor
(slice 0, one `KECCAK_RND` chunk). Extrapolating it assumes peak RSS stays roughly
linear in total cells across 23–133 chunks, which holds only while all traces are
simultaneously resident — the same assumption the bounded-residency lever above would
break. This is why §4's coefficient-free floor is stated: the verdict does not depend
on the coefficient.

**Not covered by either side of the cells count:** preprocessed columns, the composition
polynomial's own commitment, LDEs and Merkle trees. The recursion ratio quoted by the
harness (4.0× its own trace cells at the real 2^21/q=4 point) is trace-to-trace only.

---

## 6. Traps discovered

1. **2^23/blowup4 inner prove dies on a 32 GiB 5090.** `CUDA_ERROR_OUT_OF_MEMORY`
   (`[gpu] resident aux LDE failed (rows=524288 cols=10 blowup=4)`), then it **panics
   instead of falling back**: `crypto/stark/src/prover.rs:1637` — *"R2 composition fell
   back to the host evaluator, but the trace is device-only (empty)"* — and
   `prover.rs:2321` for R4 DEEP, surfacing as `prover.rs:701` "a scoped thread panicked".
   Same class as issue **#927** (uncovered cliff asserts). Reproduced twice.
   `LAMBDA_VM_GPU_LDE_THRESHOLD=999999999` does **NOT** avoid it. The point only
   completed with `LAMBDA_VM_DISABLE_DEVICE_ONLY=1 LAMBDA_VM_DISABLE_GPU_COMPOSITION=1
   LAMBDA_VM_NO_GPU_LOGUP=1` (57.5 s on CPU, 45 GiB RSS). **This will bite Stage B/C**,
   which must prove real 2^23 epochs.
2. **Emission is the first wall, not proving** — 89 GiB merely to *build* the 219q
   program. Any future census must budget for the emitter, and `LFM_CENSUS_SKIP_EMIT` /
   `LFM_CENSUS_SPINE_ONLY` exist for exactly this.
3. **The July-built `ethrex.elf` (sha `133816f0`) executes fine** against the box's
   August `blake3-real-hash` tree with the real block input — no guest rebuild needed,
   and the ELF-drift risk flagged in the brief did not materialise.
4. `pgrep -f` / `pkill -f` on this box match the invoking ssh command's own argv — I
   self-killed a running point that way. Use a bracketed pattern (`lambda_vm_prov[e]r`).
5. The fibonacci fixture ELF was copied aside to `/root/fibonacci.elf.SAFE` before any
   work; no make target that would overwrite it was run.

---

## 7. Harness change (for the reviewed commit later)

Two files, +260 lines, all test-only.

**`prover/src/lfm/epoch_tests.rs`** — `real_epoch_with` now reads three environment
overrides, defaulting **byte-for-byte to the existing fibonacci fixture path** so no
existing test moves:

- `LFM_CENSUS_ELF` — inner guest ELF path (default: `proof_fixture::read_inner_elf()`)
- `LFM_CENSUS_INPUT` — private input path (default: empty)
- `LFM_CENSUS_EPOCH_LOG2` — epoch size (default: `FIXTURE_EPOCH_LOG2` = 4)

The private input is threaded into `Executor::new`, `build_initial_image_paged` and
`Traces::from_image_and_logs` (position 6 of that call was `&[]` and *is* the private
input — easy to miss). Everything else already mirrors `continuation::prove_epoch`
faithfully (no PAGE configs, L2G bookend, REGISTER preprocessed with FINI), so the
generalisation is genuinely minimal. Plus one `eprintln!` reporting inner prove time
and sub-proof count.

**`prover/src/lfm/wrap_tests.rs`** — one new `#[ignore]`d test,
`the_census_fit_map_point`, driven by:

- `LFM_CENSUS_PRESET` — `min|blowup2|blowup4|blowup8`
- `LFM_CENSUS_QUERIES` — query-count override for the linearity calibration
- `LFM_CENSUS_SKIP_EMIT` — stop after the leg-shape dump (what made the big points measurable)
- `LFM_CENSUS_SPINE_ONLY` — emit the spine alone

It prints the per-leg shape table (`log2_trace_length`, LDE, main/aux width, Merkle
depth, groups, committed FRI layers, `query_permutations`), then — when emitting — the
full chip census, the spine/leg split with the closed form **asserted** equal to the
emitted leg permutations, the recursion ratio, and the fit verdict against 93/110 GiB.

The runner (`/root/census_run.sh`) wraps every point in `timeout` + `/usr/bin/time -v`
and appends headlines to `/root/census_results.md` as they land.

---

# Part 2 — Residency, the emission wall, and the tower node

Follow-up to Part 1's Gate-A failure, answering the three questions its verdict
raised. Read-only analysis in `/Users/maurofab/workspace/lambda_vm-blake3-impl`
(branch `blake3-real-hash`); no builds.

> ### Companion documents — read these alongside
>
> Residency and emission were each analysed **twice, independently**. The
> standalone audits are the primary records; §1 and §2 below are the second read,
> and where the two differed the audits won (corrections are marked in place):
>
> | topic | primary record | this document |
> |---|---|---|
> | bounded-residency proving (P-b) | **`residency-seam-audit.md`** — S1-S7 seams, the `17.37·N + 30.2·k GiB` peak model, the 309-819 / 48-56 GiB ladder, coefficient correction, confidence ledger | §1 below (second read) |
> | emission memory (P-c) | **`emitter-memory-audit.md`** — the row-intermediate + scope-held `read_counts` mechanism, the four wins, streaming seams, "emission is not the last wall" | §2 below (second read — **its mechanism was wrong**, see the correction box) |
> | tower node / Gate D1 | §3 below (only record) | — |
>
> Both audits agree with §1-§2 on **every verdict**; the differences are in
> accounting and in which allocation dominates. `PLAN.md` §A cites all three.

## 1. RESIDENCY — verdict **(b) moderate refactor, one named seam**

### The code answers it directly

`crypto/stark/src/prover.rs:265-274`, the `Lde` struct's own doc comment:

> *"Memory trade-off, asymmetric since the per-table scheduler fused aux build,
> aux commit and rounds 2-4 into one task:*
> - *main: produced by the Round 1 main commit, **which is a phase-wide barrier,
>   so all N tables' main LDEs are live at once** (O(N x main_cols x lde_size)).*
> - *aux: produced and consumed inside the same fused task, so **at most
>   `table_parallelism()` of them coexist** (O(k x aux_cols x lde_size))."*

So: **all N main traces AND main LDEs are simultaneously resident; aux is already
bounded to k.** ✓ VERIFIED. Note the `debug-checks` caveat in the same comment —
there the fused task is split and aux becomes all-N-live too.

`air_trace_pairs` hands `multi_prove` a vector of `(air, &mut trace, &publics)`,
so every trace must be materialized *before* the call: there is no per-table
streaming at the entry point either.

### This makes Part 1's projection ~2x CONSERVATIVE

Part 1 charged all N tables for aux cells as well as main. Correcting for the
bounded aux (k = `table_parallelism()` = cores x 2/3 = **21** on the 32-core box,
`prover.rs:588-605`, overridable by the `TABLE_PARALLELISM` env var at `:591`):

| point | N | main-side (xN) | aux-side (xk=21) | corrected | Part 1 said |
|---|---|---|---|---|---|
| 2^20/blowup4 | 23 | 666 GiB | 635 GiB | **1,300 GiB** | 1,199 GiB |
| 2^21/blowup2 | 55 | 956 GiB | 381 GiB | **1,337 GiB** | 2,929 GiB |
| 2^22/blowup2 | 83 | 1,442 GiB | 381 GiB | **1,823 GiB** | 4,421 GiB |
| 2^23/blowup2 | 133 | 2,311 GiB | 381 GiB | **2,692 GiB** | 7,242 GiB |

**Gate A's verdict is unchanged** — every point still fails by 14x-29x — but the
margin at the large points is roughly half what Part 1 reported. (At 2^20 the two
agree closely because N < k there, so nothing was over-charged.)

### Where peak accretes

| buffer | scope | one 2^19-row KECCAK_RND chunk, blowup 2 |
|---|---|---|
| main trace | **all N** (input to `multi_prove`) | 5.8 GiB |
| main LDE | **all N** (Round-1 phase-wide barrier) | 11.6 GiB |
| main tree | all N (small) | 0.03 GiB |
| aux trace | k concurrent | 6.0 GiB |
| aux LDE | k concurrent | 12.1 GiB |
| composition / DEEP / FRI | inside the fused per-table task, k concurrent | — |

**The binding constraint is the main LDEs**: 532 GiB (2^20/blowup4) to 1,538 GiB
(2^23/blowup2) on their own.

### What `disk-spill` / `StorageMode` bound today

`StorageMode` is `{Ram (default), Disk}` (`crypto/stark/src/storage_mode.rs:4-8`).
The feature is opt-in — `prover/Cargo.toml:8` has `default = ["parallel"]`, spill
at `:17` — and **our Stage-A builds used `--features cuda` only, so spill was
compiled out and every Part-1 measurement ran in `Ram`.**

What it actually spills: `prover.rs:3113-3118`, *"Spill main traces to mmap before
Round 1 LDE"* — the main **traces**, via `spill_to_disk()`, and only under
`StorageMode::Disk`. It does **not** spill the main **LDEs**, which are the
binding buffer. So spilling bounds the input side, not the peak.

### The seam, and the achievable floor

Cumulative, at 2^21/blowup2 (N=55):

| change | peak | reachable today? |
|---|---|---|
| today (N main traces + LDEs, k=21 aux) | 1,335 GiB | — |
| `TABLE_PARALLELISM=1` | 972 GiB | flag exists (`prover.rs:591`) — but see correction |
| + `disk-spill` on main traces | 654 GiB | **NO — see correction** |
| + **main-LDE re-derivation at query time** | **35 GiB — FITS** | does not exist |

> ### ⚠ CORRECTION — superseded by `residency-seam-audit.md` (2026-08-12)
>
> Full detail, including the S1-S7 seam list and the confidence ledger, is in
> **`residency-seam-audit.md`** (§4 "Verdict" and §4 "The seams" / "The floor").
> The "exists" column above was **too generous**, and the ladder should not be
> quoted as a set of free levers:
>
> - **`disk-spill` is UNREACHABLE from the LFM path.** The feature is off in our
>   builds *and* the LFM prove call site pins RAM — `lfm/proof.rs:140` calls
>   `Prover::multi_prove` directly, and the test path is literally
>   `test_utils::multi_prove_ram` (`test_utils.rs:134-142`). ✓ VERIFIED. Wiring
>   spill through to LFM is therefore **part of P-b**, not a precondition of it.
> - **`TABLE_PARALLELISM` bounds only the aux / rounds-2-4 transients**, which is
>   the k-term — it does nothing to the O(N) main-LDE term that binds.
> - **The 33.7 B/cell coefficient is ~2.1x high for the KECCAK_RND shape.** The
>   audit's direct peak model for that family is **17.37·N + 30.2·k GiB**, which
>   reads the Gate-A band as ~560-3,200 GiB rather than the figures in Part 1.
>   **No verdict moves** — every point still fails by a wide margin — but the
>   Part-1 absolute numbers are upper bounds, not estimates.
> - Bounding only the LDE lands at **309-819 GiB**; the flat floor additionally
>   needs the TRACE streamed. That is available because chunks are pure functions
>   of their `round_ops` slice (no cross-chunk logic), giving **~48-56 GiB flat
>   regardless of N, at ? +40-60% wall time**. Seams S1-S7 are named in
>   `residency-seam-audit.md` §4.
>
> What survives from my reading: the `Lde` doc comment (`prover.rs:265-274`)
> establishing that **all N main LDEs are live while aux is k-bounded**, and the
> soundness argument below.

**Verdict (b).** No flag bounds residency today. The change that turns O(N) into
O(1) is dropping each table's main LDE after its root is committed and re-deriving
it when Round 4 needs openings — plus streaming the trace for the flat floor.

**Why this is a refactor and not a protocol change:** the Round-1 barrier itself is
required by soundness — every main root must be in the transcript before the shared
LogUp challenges are sampled (`prover.rs:3216`). But only the **roots** are needed
for that; retaining the **LDEs** is a performance choice. The seam is the `Lde`
struct (`prover.rs:275-287`) and the `main_ldes: Vec<(Vec<FieldElement<Field>>,
usize)>` accumulator at `prover.rs:3145`, which is what makes retention O(N).

**Cost:** one extra LDE + tree pass per table, i.e. roughly 2x prover time on the
hash chips, traded for O(1) memory.

⚠ **This corrects my earlier "one chunk resident at a time ≈ 50 GiB flat" claim**
as recorded in PLAN.md §A and the campaign memory. The ~35-50 GiB figure is a
*target reachable only with the re-derivation change*; it is not what flags buy
today.

---

## 2. EMISSION WALL — verdict **(b), with two nearly-free wins first**

The 219q emission was OOM-killed at 89.1 GiB. No architectural change is implicated
— but the dominant allocations are **not** the ones I first named, so read the
correction box before the arithmetic.

`Addr` is `pub struct Addr(pub u64)` — 8 bytes (`instr.rs:20`). The largest `Instr`
variant is `Unpack { input: Addr, outs: [Addr;4], mults: [u64;4] }` = 72 bytes of
payload, so `size_of::<Instr>()` is **80 bytes** with tag and alignment
(? INFERRED — reasoned from the field types; not measured, since builds were out
of scope). `KeccakF(Box<KeccakOperands>)` is boxed, which is what keeps the enum
this small.

At 219 queries the program is ~272M instructions (measured leg slope 1,237,050
instr/query + measured spine):

> ### ⚠ CORRECTION — my attribution below was WRONG; see `emitter-memory-audit.md`
>
> The primary record for this section is **`emitter-memory-audit.md`** (§0 measured
> type sizes, §1 "What dominates", §4 the verdict and the four wins). Read it
> instead of the arithmetic below.
>
> I attributed the 89 GiB peak to a **`Vec<Instr>` doubling spike**. That was an
> inference from the enum size, never a reading of the emitter's allocation path,
> and `emitter-memory-audit.md` §1, which did read it, found otherwise:
>
> - **The instruction stream is only ~24% of the peak** (271M x 80 B = 21.7 GB).
> - The dominant term is the **per-instruction `Vec<Vec<FE>>` row intermediate**
>   (~47 GB, with ~80% capacity waste — 10-wide rows landing at capacity 18).
>   ✓ VERIFIED it exists: `compiler.rs:38`, `fn from_rows(width: usize, rows: Vec<Vec<FE>>)`.
> - Plus a **drained-but-unshrunk `read_counts` HashMap** (~18.3 GB) held by scope
>   through the peak. ✓ VERIFIED: `compiler.rs:140` binds it `mut`, `take(...)`
>   drains it entry by entry (`:164-201`) and `:208` asserts it empty — but a
>   `HashMap`'s allocation does not shrink on removal, so it sits at full capacity
>   across the subsequent column-group emission.
>
> **The correct cheap wins are therefore not mine but these:**
> 1. `drop(read_counts)` before `emit_column_groups` — **-18.8 GB, one line**.
> 2. A flat-append `ColumnGroupBuilder` replacing the row-of-`Vec`s — **-27 GB,
>    ~50 lines**, zero semantic change (`program_id` commits over matrices, so the
>    result is bit-identical).
>
> Together: peak **~99-102 GB -> ~53-56 GB**.
>
> My two suggestions (`Vec::with_capacity`, dense `read_counts`) are not wrong as
> micro-optimisations, but they target the ~24% term and would not have moved the
> wall. **Task #29 tracks `emitter-memory-audit.md` §4's list, not mine.**
>
> ⚠ And emission is **not** the last wall even once streamed: `execute` still wants
> ~21 GB of memory plus ~10 GB of records, and `LFM_BALU` pads to 2^28 rows at
> 219q. The P-b prover streaming stays load-bearing.

The superseded arithmetic is left below for the record.

| term | size |
|---|---|
| final `Vec<Instr>` (272M x 80 B) | 20.3 GiB |
| ~~`Vec` doubling spike~~ (SUPERSEDED — not the mechanism) | ~~60.0 GiB~~ |
| `read_counts: HashMap<Addr, u64>` (`builder.rs:101`), ~1 entry per instruction | 8.5 GiB |

`Addr` is `pub struct Addr(pub u64)` — 8 bytes (`instr.rs:20`); the largest `Instr`
variant is `Unpack` at 72 bytes of payload, so `size_of::<Instr>()` is ~80 bytes
(? INFERRED, never measured — `emitter-memory-audit.md` §0 measured the type
sizes directly and its 21.7 GB for this term agrees).

Whether full per-leg streaming is possible hinges on the machine being
straight-line; `emitter-memory-audit.md` §3 names the seams (builder `instrs` field; `compile` merging
into the builder; the executor needing a 10-way merge by destination — the one new
algorithm). I did not verify the straight-line property end to end. ✗ UNCERTAIN.

---

## 3. TOWER NODE PROJECTION — Gate D1: **FAILS** (1.3x-4.9x over)

The plan expects "census says the 1-proof verifier fits comfortably (expected:
yes - 14 tables vs ~25-31, blake3 legs vs keccak)". **Falsified**, though by far
less than Gate A.

Node's own options blowup 2; legs recompute BLAKE3 per `COMMIT.md` §1.4; 6-round
chip; projected peak = cells x 33.7 B/cell.

| node | inner LFM proof | 110q | 219q |
|---|---|---|---|
| D1 (verify 1 proof) | fixture wrap (exists today) | **124 GiB** (1.3x) | 247 GiB (2.7x) |
| D1 | real 2^21 wrap, D0-consistent | 227 GiB (2.4x) | 452 GiB (4.9x) |
| D2 (aggregate 2) | fixture wrap | 248 GiB (2.7x) | 493 GiB (5.3x) |
| D2 | real 2^21 wrap | 454 GiB (4.9x) | 904 GiB (9.7x) |

keccak control (D1 / fixture / 219q): 1,274 GiB. So blake3 buys **5.2x** here -
better than the plan's 4.06x aggregate.

### Model validation — exact on four measured legs

The cost model reproduces MEASURED per-query permutation counts **exactly** for
four structurally different real sub-proofs:

| leg | shape | model | measured |
|---|---|---|---|
| leg 4 | 2^2 rows, 1480+516 cols (wide+shallow) | 365 | 365 |
| leg 31 | 2^22 rows, 9+3 cols, 14 FRI layers (narrow+deep) | 288 | 288 |
| leg 3 | 2^2 rows, 511+67 cols | 92 | 92 |
| leg 22 | 2^20 rows, 10+4 cols, 12 FRI layers | 239 | 239 |

This confirms the leaf/Merkle/FRI decomposition, the group counts, the FRI layer
counts, and `num_parts = 2`.

### The D1 lever is the LEAF RATE, not the hash

blake3's advantage is **highly non-uniform**:

- **Merkle parents: 14.7x cheaper.** One invocation either way - keccak
  `KECCAK_RND` costs 24 rows x (1480 main + 3x516 aux) = 72,672 cells per
  permutation; the blake3 chip is 1 row x (3056 + 3x630) = 4,946 cells per
  compression (`blake3_chip.rs:162,224`, `airs.rs:246` for the
  `interactions.div_ceil(2)` aux rule).
- **Leaf absorption: only 1.73x cheaper.** keccak absorbs 17 felts per
  permutation; `COMMIT.md` §1.4 and `LEAF.md` §1.4 give blake3 **2 felts per
  compression** (a 4-felt `LFML` row plus one `LFMC` fold). blake3 needs 8.5x
  more invocations, nearly cancelling its per-invocation edge.

And leaf absorption is **69.8%** of the D1 node's per-query bill (Merkle 10.7%,
FRI 19.5%), concentrated in the two wide chips: `KECCAK_RND` 3,229
compressions/query + `LFM_KECCAK` 1,164 = 65% of everything.

Raising the `LFML` rate is therefore the dominant D1 lever - and it is still an
**open spec decision** (`COMMIT.md` is DRAFT, S1 is the gating item), so it is
cheap now and expensive later:

| node | inner | today | rate x2 | rate x4 |
|---|---|---|---|---|
| D1 | fixture, 110q | 124 | **81 FITS** | **59 FITS** |
| D1 | real, 110q | 227 | 148 | 108 |
| D2 | fixture, 110q | 248 | 161 | 118 |

? INFERRED headroom: blake3's compression block is 64 B; a 4-felt `LFML` row uses
36 B of it (`LEAF.md` §1.3: 8 lanes x 4 B + a 4-byte tag), so ~7 felts fit one
block. **I have not checked what the chip's constraint layout can support** -
this is a question for the spec owner, not a claim that it is free.

### Why the plan's expectation was directionally right but short

| what is verified | tables | perms/query (keccak) |
|---|---|---|
| RV64 epoch 2^21 (Gate A inner) | 32 | 5,434 (MEASURED) |
| LFM fixture wrap (D1 inner) | 14 | 2,384 |
| LFM real-2^21 wrap (D1 inner) | 15 | 3,306 |

D1's inner IS 2.3x cheaper per query than Gate A's, because the LFM proof has far
fewer DEEP tables (where Merkle depth and FRI dominate). But fewer tables does not
mean cheap: the 14 LFM chips are far WIDER (`KECCAK_RND` 1480+516) than RV64
tables (mostly <50 columns), and **leaf cost is set by width, not height**.

### The tower does not get cheaper as it climbs

An upper-layer node whose inner has every chip at its 4-row floor still costs
**97 GiB at 110q**, against 124 GiB for the base layer - only 22% less. The 14
chip WIDTHS are fixed by the machine, so only the Merkle/FRI 30% shrinks with
height. The plan's "per-layer cost is the D1 census number x 2" is therefore
sound. For N=36 base wraps: 6 layers, 38 aggregation nodes; peak is PER NODE
(sequential), so the binding constraint is the largest single node.

### Two build-config traps

1. **`blake3-6round` is OFF by default.** `prover/Cargo.toml:22` declares it;
   `blake3.rs:82-85` cfg-gates `BLAKE3_ROUNDS` to `BLAKE3_STANDARD_ROUNDS` (7)
   unless the feature is on. The chip is `8 x rounds` G-blocks wide, so the
   default 7-round chip is 3,552 columns against 3,072 - **+16% on every tower
   number** (D1/fixture/110q: 124 -> 144 GiB). The plan's hash matrix says
   "blake3-6r", so the campaign intends the feature ON; it must be named
   explicitly in the build.
2. **Second-order feedback:** under D0 the machine's own hash chip becomes BLAKE3
   at 3,056 main columns, which is **wider than `KECCAK_RND`'s 1,480**. The LFM
   proof's widest table therefore gets wider and each tower layer pays more to
   re-hash its leaf: D1/real/219q moves 381 -> 452 GiB (+19%). The hash switch is
   still right, but it is not monotonically cheaper at every layer.

### Sensitivity — the FAIL verdict is robust

| variation | D1 fixture 110q | vs base |
|---|---|---|
| baseline (aux 630, num_parts 2, non-hash 6.5%) | 124 GiB | -- |
| blake3 aux 500 | 114 | 0.92x |
| blake3 aux 750 | 133 | 1.07x |
| num_parts 1 / 4 | 123 / 125 | 1.00x / 1.01x |
| non-hash chips 3% / 15% | 119 / 136 | 0.96x / 1.10x |

Every variation stays above 93 GiB. The one soft input is the blake3 aux width:
`bus_interactions()` is built with `Vec::with_capacity(1_259)`
(`blake3_chip.rs:913`) and **no test asserts the final count**, so 630 is
? INFERRED; +-20% moves the result only +-8%.

---

# Part 3 — spill ladder, measured

Measured 2026-08-13 on the rented 5090 box (32 cores, 60.45 GiB RAM, 64 GB
overlay disk, 71 GiB of swap present), branch `blake3-real-hash`, wiring commit
`c5ffadf3`. Every number below is ✓ MEASURED unless marked otherwise; raw
`.meta`/`.samples`/`.timev`/`.stdout` per rung are in
`~/workspace/lambda_vm_bench_cache/lfm_spill_2026-08-13/`.

This part answers the question PLAN.md's P-b fallback line poses — *"existing
flags + spill wiring first; streaming only if the numbers still demand it"* —
by wiring the flags and running them.

## 0. What had to be built before anything could be measured

`disk-spill` was unreachable from the wrap on two counts, and one of them was
not in the seam audit's list:

1. `lfm/proof.rs:140-145` passed `Default::default()` = `Ram`. Now it calls
   `auto_storage::decide_lfm()`, which honours `FORCE_DISK_SPILL`. There is
   deliberately no estimate: `decide` keys off the RV64 executor's
   `TableLengths`, and the wrap has no analogue — its table set is program
   shape, and `KECCAK_RND`'s column profile was never calibrated into that
   model.
2. **The fixture path does not run.** `real_epoch_with` passes an empty private
   input to a fibonacci guest that reads its iteration count *from* private
   input, so the guest halts inside the first epoch and every test asserting an
   INTERMEDIATE epoch fails. This is the known "19 failing `lfm::` tests" drift,
   and it is a fixture bug, not a prover bug. `LFM_CENSUS_INPUT` (a file holding
   the input; unset = today's behaviour exactly) is what makes the wrap
   harness runnable; the ladder ran with an 8-byte `n = 1000`.

`LFM_WRAP_QUERIES` raises the blowup-8 wrap's inner query count above 1.
`make lint` already covers `lambda-vm-prover/disk-spill` (Makefile:660), so no
lint-matrix change was needed, and all four of its clippy passes are clean.
✓ VERIFIED

**Default behaviour is unchanged**, checked against the oracle rather than
argued: with every knob unset the `lfm::` suite is **307 passed / 19 failed** —
the known baseline exactly. (All 19 are the fixture-input failures described
above; `machine_tests::continuation_fixture_generates_two_epochs` is the one
that names the cause outright.) ✓ MEASURED

## 1. What the ladder lever actually moves

The wrap spends **2,228 permutations of spine + ~1,565 per inner query**, and a
`KECCAK_RND` chunk holds 21,845 (`chunking.rs:40`, 2^19 rows). So:

- q = 1 … 12 → **one** chunk, growing in height to its 2^19 cap;
- q = 13 → the **second** chunk appears (the first point where peak is a sum
  over chunks at all);
- q = 20 would be the first point with two *full* 2^19 chunks.

That is the whole reachable range: the box refuses the proof long before the
third chunk. The real wrap needs **N = 23 … 133**.

## 2. The ladder

`rung` = storage mode + `TABLE_PARALLELISM`. "anon" is `RssAnon`, "file" is
`RssFile` (both sampled at 2 Hz from `/proc/<pid>/status`); "peak RSS" is
`/usr/bin/time -v`. "spill vol" is the filesystem high-water mark — spill files
are `tempfile()`s, unlinked at creation, so they are invisible to `du` and only
show up as filesystem usage.

| q | perms | chunks | rung | peak RSS | peak anon | peak file | spill vol | wall | prove | result |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3793 | 1 | ram_def | 16.64 | — | — | 0.00 | 24s | 14.4s | ok |
| 1 | 3793 | 1 | ram_tp1 | 14.63 | — | — | 0.00 | 27s | 17.6s | ok |
| 1 | 3793 | 1 | spill_tp1 | 14.23 | — | — | 4.82 | 29s | 18.6s | ok |
| 2 | 5358 | 1 | ram_def | 17.14 | 16.69 | 0.01 | 0.00 | 25s | 15.0s | ok |
| 2 | 5358 | 1 | ram_tp1 | 15.35 | 15.35 | 0.01 | 0.00 | 28s | 18.3s | ok |
| 2 | 5358 | 1 | spill_tp1 | 14.52 | 12.39 | 3.65 | 4.86 | 29s | 19.1s | ok |
| 3 | 6923 | 1 | ram_def | 22.50 | 22.50 | 0.01 | 0.00 | 40s | 28.9s | ok |
| 3 | 6923 | 1 | ram_tp1 | 23.50 | 23.49 | 0.01 | 0.00 | 43s | 32.2s | ok |
| 3 | 6923 | 1 | spill_tp1 | 23.58 | 19.99 | 6.62 | 7.82 | 45s | 34.3s | ok |
| 4 | 8488 | 1 | ram_def | 31.33 | 30.42 | 0.01 | 0.00 | 47s | 32.1s | ok |
| 4 | 8488 | 1 | ram_tp1 | 27.28 | 27.27 | 0.01 | 0.00 | 53s | 37.6s | ok |
| 4 | 8488 | 1 | spill_tp1 | 26.37 | 22.27 | 7.13 | 9.35 | 56s | 40.1s | ok |
| 6 | 11643 | 1 | ram_def | 41.53 | 41.52 | 0.01 | 0.00 | 76s | 59.8s | ok |
| 6 | 11643 | 1 | ram_tp1 | 43.67 | 43.67 | 0.01 | 0.00 | 86s | 69.7s | ok |
| 6 | 11643 | 1 | spill_tp1 | 44.54 | 37.51 | 13.08 | 15.31 | 89s | 72.8s | ok |
| 8 | 14773 | 1 | ram_tp1 | 51.04 | 51.04 | 0.01 | 0.00 | 104s | 77.5s | ok |
| 8 | 14773 | 1 | spill_tp1 | 49.97 | 41.91 | 14.11 | 18.30 | 111s | 84.5s | ok |
| 12 | 21058 | 1 | spill_tp1 | 51.02 | 42.94 | 14.13 | 18.43 | 112s | 83.3s | ok |
| 13 | 22648 | **2** | ram_tp1 | 55.35 | 55.35 | 0.01 | 0.00 | 114s | 84.2s | ok |
| 13 | 22648 | **2** | spill_tp1 | 52.05 | 43.94 | 14.17 | 18.86 | 117s | 87.8s | ok |
| 14 | 24213 | 2 | ram_tp1 | 56.62 | 56.62 | 0.01 | 0.00 | 119s | 88.1s | ok, **swapped 71 MB** |
| 14 | 24213 | 2 | spill_tp1 | 53.00 | 44.88 | 14.17 | 19.22 | 121s | 91.1s | ok, no swap |
| 16 | 27343 | 2 | ram_tp1 | 57.41 | 56.71 | 0.01 | 0.00 | 74s | — | **OOM-KILLED (SIGKILL)** |
| 16 | 27343 | 2 | spill_tp1 | 57.38 | 49.63 | 13.57 | 24.41 | 144s | 111.9s | **ok** |

All GiB. Every `ok` row proved *and* verified *and* passed the three
falsifications (the harness asserts all of them; a rung that only proved would
have failed the test).

Sampler note: the q=1 anon/file cells are blank because the 2 Hz sampler was
still latching onto the wrong pid on those three runs; their `time -v` peak RSS
is unaffected. Fixed from q=2 onward — where the sampler and `time -v` agree to
within the 0.5 s sampling gap.

## 3. Paired, at equal q, both at `TABLE_PARALLELISM=1`

| q | anon Ram → spill | Δ anon | RSS Ram → spill | Δ RSS | wall Ram → spill | Δ wall |
|---|---|---|---|---|---|---|
| 2 | 15.35 → 12.39 | **−19.3%** | 15.35 → 14.52 | −5.4% | 28 → 29s | +3.3% |
| 3 | 23.49 → 19.99 | −14.9% | 23.50 → 23.58 | +0.3% | 43 → 45s | +4.5% |
| 4 | 27.27 → 22.27 | −18.3% | 27.28 → 26.37 | −3.3% | 53 → 56s | +4.3% |
| 6 | 43.67 → 37.51 | −14.1% | 43.67 → 44.54 | +2.0% | 86 → 89s | +3.7% |
| 8 | 51.04 → 41.91 | −17.9% | 51.04 → 49.97 | −2.1% | 104 → 111s | +6.7% |
| 13 | 55.35 → 43.94 | −20.6% | 55.35 → 52.05 | −6.0% | 114 → 117s | +3.0% |
| 14 | 56.62 → 44.88 | **−20.7%** | 56.62 → 53.00 | −6.4% | 119 → 121s | +2.3% |

**Spill takes 14–21% off the anonymous working set for 2–7% of wall time.**
Mauro's recollection that spill "used to be quite efficient" is confirmed on the
time axis — this is a cheap mechanism, and it is now reachable.

**But peak RSS barely moves (0 to −6%), and that gap is the whole story.** What
spill does is *convert* anonymous pages into file-backed ones: at q=8 it wrote
18.30 GiB to disk, dropped anon by 9.13 GiB, and grew `RssFile` from 0.01 to
14.11 GiB. The bytes are still resident — they are just **evictable** now.

That is exactly why the ceiling moves and the peak does not. At q=16 the two
rungs peak at the *same* RSS (57.41 vs 57.38 GiB) and one dies:

- Ram: 56.71 GiB of it is anonymous → nothing to reclaim → SIGKILL.
- spill: only 49.63 GiB is anonymous, 13.57 GiB is reclaimable page cache →
  the kernel reclaims and the proof finishes.

**Peak RSS is the wrong metric for this question. Peak anon is the right one.**

## 4. Largest point each rung completes, on a 60.45 GiB box

| rung | largest q that fits | chunks | first failure |
|---|---|---|---|
| Ram, `TABLE_PARALLELISM` default (21) | **q = 6** (41.53 GiB) | 1 | q=8 **OOM-killed** at 57.27 GiB |
| Ram, `TABLE_PARALLELISM=1` | **q = 14** (56.62 GiB, and it already had to swap 71 MB) | 2 | q=16 **OOM-killed** at 57.41 GiB |
| spill, `TABLE_PARALLELISM=1` | **q = 16** (57.38 GiB, no swap) | 2 | q=20 does not fit |

q=20 (the first point with two *full* 2^19 chunks) reached **52.34 GiB
anonymous with 0.99 GiB of swap in use** and its anon still climbing, having
made ~7 minutes' progress against the 144 s that q=16 took. It was terminated by
the operator rather than burning the 50-minute timeout, so it is recorded as
`rc=143` (SIGTERM), **not** as an OOM kill and not as a completed run. Read it
as "did not fit": it was already paging on a box where every rung that fit used
no swap at all, and the two rungs that were killed outright had reached the same
place. ✗ NOT PROVEN that it would have failed — it was not run to conclusion.

Two levers, and they are **not** the same size:

- **`TABLE_PARALLELISM=1` buys q = 6 → 14.** This is the big one, and it is
  invisible in the peak-RSS column at small q — there it looks like noise (−12%
  at q=1/2/4, but *+*0.3–5% at q=3/6) for ~13% wall. At the ceiling it is
  decisive: at q=8 the default k=21 was OOM-killed at 57.27 GiB while k=1
  finished the same point at 51.04 GiB. The k-term the `Lde` doc bounds is small
  until it isn't.
- **Spill buys q = 14 → 16 on top of that.** In the units that matter that is
  +2 inner queries, +3,130 permutations, and **zero additional chunks** — both
  ceilings sit at N = 2.

✓ MEASURED. The lesson for anyone quoting the paired table in §3: judge these
levers by where the rung breaks, not by the peak-RSS delta at a comfortable
point. The two disagree in both directions.

## 5. Build-side spill for `LfmTraces.keccak_rnd` — measured NOT needed

The brief asked whether the eager chunk-vector build (`trace.rs:162-167`, all
chunks materialised before `multi_prove`) needs its own spill-at-build. It does
not, and the RSS timeline says so directly rather than by argument:

| run | span | peak occurs at | peak |
|---|---|---|---|
| q8 ram_tp1 | 104s | **t+87s (84% in)** | 51.0 GiB |
| q8 spill_tp1 | 111s | t+63s (57% in) | 50.0 GiB RSS / 41.9 anon |

The inner epoch builds in 4.9s and `lfm_prove` runs 77.5s; the trace build is
the small early plateau (~8–11 GiB in the sampled timeline), and the peak is
4–5× higher and lands deep inside `multi_prove` — *after* the existing pre-R1
spill point at `prover.rs:3113-3122`. Arithmetically it could not be otherwise:
a main LDE is `blowup` × its trace, so Σ traces is at most half of Σ main LDEs
at blowup 2. Spilling at build time would move a number that is not the peak.

✓ MEASURED. The seam audit's S6 (lazy per-index trace generation) is still
worth what it claims — but as part of streaming, not as a spill target.

> ### ⚠ CORRECTION — true at N = 2, and it does NOT extrapolate (2026-08-13)
>
> This section's verdict is sound for the range it was measured in and wrong as
> a general statement. The whole ladder above ran at **N = 1 or 2** chunks,
> where the trace build really is a small early plateau. At **N = 15** the trace
> build IS the peak and `multi_prove` is never reached at all.
>
> ✓ MEASURED on a 60 GiB / 32-core box: the **real-block** wrap (block
> 25368371, epoch 0 at 2^16, inner blowup4 / **110 queries** — the secure
> preset) is **OOM-killed at 56.91 GiB anon after 2m30s, BEFORE proving
> starts**. Spill volume 0.00 GiB, `RssFile` peak 0.01 GiB, disk untouched
> (60.11 GiB still free at the minimum). Emission SUCCEEDS first and prints its
> full census (26,197,950,740 base-field-equivalent cells), so the emitter is
> not the wall either. 15 chunks × 5.78 GiB of main trace = **87 GiB** in
> `build_traces_with_hasher` (`prover/src/lfm/trace.rs:162-167`) before
> `multi_prove` is called.
>
> The arithmetic argument above — "a main LDE is `blowup` × its trace, so
> Σ traces is at most half of Σ main LDEs" — is correct and beside the point:
> it compares two things that are only both alive if the prove is reached. Σ
> traces is what has to be resident *to call* `multi_prove`, so at large N it
> binds first no matter what the LDE side costs.
>
> Consequence: **S3 Phase A+B bound residency INSIDE `multi_prove` and are
> never reached at production query counts on a <128 GiB box.** S6 (lazy
> per-index chunk traces) is the enabler for the 64-128 GiB class, not an
> optimisation. On a 258 GiB box the eager build fits and S6 is not required.
>
> Build-side spill does not rescue the 110q rung either: 87 GiB of trace
> against 61 GiB of disk.

## 5a. The two knobs — epoch size is WEAK, query count is STRONG

> ### ⚠ CORRECTION to the campaign's climb strategy (2026-08-13)
>
> Part 1 §3 measured "logarithmic in epoch size per table" and concluded the
> epoch-size lever is **weak** (2^23 → 2^20 buys only 3.2×). That is right, and
> the operational consequence was never drawn: **epoch size is not the knob that
> decides whether a wrap fits.** The chunk count is, and it is
>
> ```
> chunks = (spine_perms + per_query_perms × queries) / 21,845
> ```
>
> (`chunking.rs:40`). Per-query cost is dominated by **leaf absorption, which is
> set by table WIDTH** — Part 1 §3 says so itself ("Independent of epoch size").
> ✓ MEASURED: **2,946.0 perms/query at 2^16** against the census's **5,434 at
> 2^21/blowup2** — a 32× change in epoch size moves per-query cost by 1.8×,
> while the query count moves the chunk count **linearly**.
>
> So a climb that walks epoch size looking for a fitting point at 110 queries
> finds nothing at any size, and a climb that walks the query count finds the
> boundary immediately. Measured boundary at 2^16 on a 36 GiB laptop under
> `RecomputeLde` + `TABLE_PARALLELISM=1`: q=4 → 1 chunk (19.76 GiB), q=8 → 2
> (21.43), q=12 → 2 (22.87, completes), q=16 → 3 (killed).

## 6. Spill volume against the disk

| q | spill volume | of 61 GiB free |
|---|---|---|
| 1 | 4.82 GiB | 8% |
| 8 | 18.30 GiB | 30% |
| 13 | 18.86 GiB | 31% |
| 16 | 24.41 GiB | 40% |

Volume tracks the trace+tree bytes, not the LDE, exactly as §3 of the seam audit
predicts. No run came close to the 62 GiB disk, and `posix_fallocate`
(`mmap_util.rs:47-70`) reserves blocks up front, so a full disk would surface as
a `ProvingError::DiskSpill` rather than a mid-write SIGBUS. **The disk is not
the binding constraint at any point this box can prove** — but note the ratio:
at q=16 the spill volume is 40% of the disk while buying 14% of RAM. Scaled to
the N=23 wrap the volume, not the disk headroom, is what would run out first.

One trap worth recording: `/tmp` is on the overlay filesystem on this box, so
spill files really do land on disk. On a systemd-default distro `/tmp` is tmpfs
and **spill would be a no-op** — anonymous pages moved to RAM-backed files.
`mmap_util.rs:53-55` says so in its own comment; set `TMPDIR` to a disk-backed
path before trusting any spill measurement.

## 7. GPU interaction — documented, one build, three runs

Built `--features cuda,disk-spill` on the same box (RTX 5090, 32,607 MiB VRAM).
Mauro's caveat that spill is "not compatible with GPU" is **half right, and the
other half matters**:

| run | result |
|---|---|
| q=1, spill + `TP=1`, cuda | **ok.** 16.10 GiB RSS / 11.59 anon / 4.00 file, **4.26 GiB spilled**, 24.0s |
| q=13, spill + `TP=1`, cuda | **panic** — `prover.rs:1657` |
| q=13, **Ram** + `TP=1`, cuda (control) | **panic** — `prover.rs:1657`, *identical* |

The panic is
`"R2 composition fell back to the host evaluator, but the trace is device-only
(empty)"` — the #927-class uncovered cliff assert, fired with VRAM at
**32,086 of 32,607 MiB**. **The control settles the attribution: it is a
pre-existing VRAM-pressure failure, not a spill bug.** ✓ MEASURED. The same
q=13 point proves fine on the non-cuda build in every rung of §2.

What is genuinely GPU-specific about spill:

- **Aux Merkle trees are not spilled when a GPU aux commit succeeds.** Both cuda
  aux arms `return Ok(...)` at `prover.rs:3442` and `:3480`, before the
  `spill_tree(&mut tree, storage_mode, "aux Merkle tree")` at `:3524` that the
  CPU fallback reaches. ✓ VERIFIED by reading. It shows up in the volume: at
  q=1 the cuda build spilled **4.26 GiB against the CPU build's 4.82 GiB**, on
  a byte-identical proof.
- **`StorageMode::Disk` disables the precomputed-tree cache** —
  `prover.rs:1151-1157` sets `cache_ok = storage_mode != StorageMode::Disk`, so
  spilled runs lose cross-prove reuse of preprocessed trees. The wrap has 11
  chips carrying preprocessed instruction column groups, so this is a real
  (unmeasured, ? INFERRED) wall-time cost on repeated proves.
- Host RSS is *higher* under cuda at the same point (16.10 vs 14.23 GiB at q=1),
  and wall is lower (24.0 vs 28.6s).

**Net:** spill and cuda compose without corrupting anything — the q=1 cuda spill
run proved and verified — but on this GPU the wrap hits the VRAM cliff at
q=13 regardless of storage mode, so the CPU build remains the honest instrument
for residency work, exactly as briefed.

## 8. Verdict — does spill suffice?

**No. Keep it, but it is not the fix.**

What was bought, measured end to end: **q = 6 → 14 from `TABLE_PARALLELISM=1`,
then 14 → 16 from spill.** Two levers, both now reachable from the wrap, both
cheap (spill costs 2–7% wall). What is needed: the real wrap has **N = 23 to
133** `KECCAK_RND` chunks. This ladder died at **N = 2**.

### Calibrating the seam audit against measurement

The marginal cost of a chunk falls straight out of the spill rung: q=12 (one
full 2^19 chunk) peaks at 42.94 GiB anon, q=16 (that chunk plus a 2^18 one) at
49.63 — **6.69 GiB for half a chunk, so ≈ 13.4 GiB per full 2^19 chunk with
spill on.** The seam audit's model says **17.37 GiB/chunk** persistent without
spill. The two agree to within 23%, and 23% is precisely the anon reduction §3
measures. **The audit's coefficient survives contact with a real wrap.** That is
the most reusable thing in Part 3.

Extrapolating on the measured marginal (? INFERRED — the non-chunk base also
grows with query count, so treat these as a floor):

| point | N | spill + `TP=1` | vs a 124 GiB rig |
|---|---|---|---|
| 2^20 / blowup4 | 23 | ~337 GiB | 2.7× over |
| 2^23 / blowup2 | 133 | ~1,810 GiB | 15× over |

### The specific question: post-blake3 (÷~4), on 124 GiB rigs?

**Not a yes.** ÷4 on the keccak family takes the cheapest point to N ≈ 6, i.e.
~110 GiB by the marginal above — which lands *on* the 124 GiB line, not safely
under it, and that estimate ignores Part 2 §3's own trap: under BLAKE3 the
machine's `LFM_HASH` chip becomes **3,056 columns, wider than `KECCAK_RND`'s
1,480**, so the non-chunk base grows at the same time the chunk count shrinks.
A single geometry choice decides it either way. ? INFERRED.

And the flags are now **spent**: both are on in that estimate. There is no third
flag.

### So the recommendation stands, with one change

**Main-LDE re-derivation (seam S3) remains required**, and the measurement
sharpens why: spill removes the *trace*, and the trace is the cheap half. At
q=16 spill wrote 24.41 GiB to disk to take 7.08 GiB off anon, because a main LDE
is `blowup` × its trace and **never** spills (`LDETraceTable` has no mmap field,
`trace.rs:316-343`). The buffer that binds is the one no flag touches.

The change from PLAN.md's framing: spill is no longer "unreachable, therefore
unknown". It is reachable, it is cheap, it is worth keeping wired — it buys a
rung for free and it will multiply whatever structural fix lands. It simply
cannot be the structural fix, and now that is measured rather than modelled.

### One more thing the ladder found

`real_epoch_with` cannot run the fixture at all without `LFM_CENSUS_INPUT` (§0).
The 19 failing `lfm::` tests are **a fixture-input bug, not prover drift** —
the fibonacci guest reads its iteration count from a private input the fixture
never supplies. Worth fixing properly at the fixture rather than carrying as
known-red.
