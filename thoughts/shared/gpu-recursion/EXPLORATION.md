# GPU for the LFM recursion machine — post-merge map, remaining levers, box plan

**Date:** 2026-08-12
**Worktree read:** `/Users/maurofab/workspace/lambda_vm-blake3-merge`, branch
`blake3-real-hash-mainmerge`, **mid-merge** (`MERGE_HEAD` = `58160b6f` = `origin/main`
tip "Feat/hint ecall (#876)"). The merged content is staged but not committed, so
every line number below is against the *working tree* of that worktree. ✓ VERIFIED
by `git rev-parse MERGE_HEAD` + `git status`.
**Method:** read-only. No cargo, no edits. Every claim is marked ✓ VERIFIED (read the
code) or ? INFERRED (derived, not executed).

---

## 0. Headline

The premise of the earlier finding — *"the LFM machine gets ~zero GPU because the
GPU main-LDE path excludes preprocessed tables"* — **is now largely obsolete.**
Main's #863 (`d83b4d9e`, "halve GPU continuation proving time") added a dedicated
**GPU split-tree path for preprocessed tables**, and #875 (`5749a956`, "device-resident
rounds 2-4") made R2 fully device-resident. Neither is preprocessed-gated.

Post-merge, for a preprocessed table that clears the 2^19 LDE threshold — which for
LFM means **BITWISE, the table that is 97–99% of all committed cells** — the LDE, both
Merkle trees, the LogUp aux build, the composition polynomial, the OOD barycentric,
DEEP and FRI **all run on device**.

What is still excluded is narrower and precisely locatable: **the host D2H is never
skipped for a preprocessed table**, because `device_only_gate` still carries
`&& !is_preprocessed`.

**But that is not the biggest lever for recursion.** The measured census of the real
epoch-verifier wrap (§3.1) shows the LFM machine's shape is **short and very wide** —
`KECCAK_RND` is 1,480 columns and carries **88.1% of all main cells** — whereas the
GPU LDE gate is a **row-count** threshold that is completely blind to width
(`lde_size = n × blowup`, no column term, `gpu_lde.rs:698-700, 799-801`). In the only
LFM wrap that has actually been proved end to end, that threshold drops the
88%-of-cells chip to the CPU while admitting BITWISE, which is 4.8%. **Fixing the
threshold to be cell-aware is the top lever, it needs no new kernels, and the
experiment costs one env var** (§4).

**Three claims in the earlier LFM-GPU note should be retired:** (a) "no chip reaches
the GPU main-LDE/composition path" — false post-merge; (b) "R1 GPU main-LDE is inside
`if precomputed.is_none()`" — that `if` is now a two-way route, not an exclusion;
(c) "value scratch is 1.5 MiB per IR node → KECCAK_RND wants 23.9 GiB" — superseded by
liveness slot reuse (§3). The note's *suggested fix* ("lift `!is_preprocessed`")
survives, but it now buys a D2H skip rather than the whole GPU pipeline.

---

## 1. Current map: what runs on GPU for an LFM per-table proof

The LFM machine proves through the *same* `Prover::multi_prove` as the RV64 VM
(`prover/src/lfm/proof.rs:140`), so this is the generic prover path, read for the
preprocessed case.

| Stage | Preprocessed table today | Gate (file:line) | Status |
|---|---|---|---|
| R1 main LDE (row-major NTT) | **GPU** | `prover.rs:1127-1157` → `gpu_lde.rs:779-858` → `math-cuda/src/lde.rs:718-830` | ✓ VERIFIED |
| R1 precomputed-columns Merkle tree | **GPU** (built on device, nodes D2H'd once, then process-cached) | `math-cuda/src/lde.rs:782-789`; cache `prover.rs:175-214` | ✓ VERIFIED |
| R1 multiplicity Merkle tree | **GPU, stays resident** (host tree is root-only) | `math-cuda/src/lde.rs:793-802`; `gpu_lde.rs:839-845` | ✓ VERIFIED |
| R1 main-LDE host copy (D2H) | **CPU / always paid** | `math-cuda/src/lde.rs:806-807` (unconditional), comment at `:804-805` | ✓ VERIFIED — **the remaining exclusion** |
| R1 LogUp aux build | **GPU**, reading the resident trace | `logup_gpu.rs:418-453`, threshold `1<<10` at `logup_gpu.rs:26`; preprocessed handles threaded at `prover.rs:3253-3260` | ✓ VERIFIED |
| R1 aux LDE + aux Merkle | **GPU** | `prover.rs:3394-3424` (resident) / `:3428-3463` (fused) | ✓ VERIFIED |
| R1 aux host copy (D2H) | **CPU / always paid** (same `device_only` flag) | `prover.rs:3407`, `:3449` — `!device_only` | ✓ VERIFIED |
| R2 composition `H(row)` | **GPU**, device-resident (`GpuCompH`) | `evaluator.rs:388-421` → `:302-332`; **no `is_preprocessed` check** | ✓ VERIFIED |
| R2 decompose + half-extend (d=2) | **GPU** | `prover.rs:1589-1621`, `gpu_lde.rs:580` | ✓ VERIFIED |
| R2 composition-parts host copy | **CPU / always paid** | `prover.rs:1606` — retain = `!host_trace_empty()` | ✓ VERIFIED |
| R2 composition Merkle tree | **GPU from the device parts handle** | `prover.rs:1737-1756` | ✓ VERIFIED |
| R3 OOD barycentric (main) | **GPU** off the handle | `trace.rs:766-776`; gate `gpu_lde.rs:1198-1233` (no prep check), threshold `1<<14` at `gpu_lde.rs:1020` | ✓ VERIFIED |
| R3 OOD barycentric (aux) | **GPU** | `trace.rs:824-834` | ✓ VERIFIED |
| R4 DEEP composition | **GPU** (device inv-denoms + device parts) | `prover.rs:2250-2278`, `:2190` | ✓ VERIFIED |
| R4 Merkle authentication paths | **GPU** — including the preprocessed table's multiplicity tree | `prover.rs:2705-2731` (no prep check) | ✓ VERIFIED |
| R4 opening **values** | **CPU host gather** for preprocessed | `prover.rs:2758` `(!is_preprocessed)`; consumer `prover.rs:2836-2876` `gather_main_row_range` | ✓ VERIFIED — **second exclusion** |
| FRI commit + query phase | **GPU** | `fri/mod.rs:62`, `:164`; `prover.rs:2008` | ✓ VERIFIED, not prep-gated |

### The two `if precomputed.is_none()` / `is_preprocessed` sites, precisely

1. **`prover.rs:1075` — `if precomputed.is_none()`.** This is *no longer an
   exclusion*. It is a two-way route: `:1075` takes the plain fused path for normal
   tables, `:1128` takes the **split-tree** path for preprocessed ones. Both end in
   `return Ok((commit, main_data, Some(handle)))` — the preprocessed table gets a
   device handle, which is what unlocks everything downstream. ✓ VERIFIED
2. **`gpu_lde.rs:210` — `&& !is_preprocessed` inside `device_only_gate`.** This one
   *is* still a real exclusion, and it is the only place the preprocessed property
   changes GPU behaviour in R1. Its consequence is a *host copy*, not a CPU
   computation. ✓ VERIFIED
3. **`prover.rs:2758` — `(!is_preprocessed)` on `main_dev_values`.** Downstream of
   (2): preprocessed R4 openings read values from the host LDE. ✓ VERIFIED

`crypto/stark/src/verifier.rs:225` and `:1274` also mention `is_preprocessed` — those
are verify-side and irrelevant here. ✓ VERIFIED

---

## 2. What main's split-tree path actually does on device

`math_cuda::lde::coset_lde_row_major_split_trees` (`crypto/math-cuda/src/lde.rs:718`),
read line by line: ✓ VERIFIED

- `:747` one H2D of the row-major trace → `expand_row_major_on_stream` (fused
  row-major NTT), retaining the trace-domain column-major snapshot.
- `:759-778` a closure that builds **one subset Merkle tree per column range** by
  launching `keccak_base_row_major_row_pair_range(col_start, col_end)` over the
  resident LDE, then `build_inner_tree_levels`. Leaves are bit-identical to the CPU
  `commit_rows_bit_reversed_subset` pair (asserted by the wrapper's doc,
  `gpu_lde.rs:766-777`).
- `:782-789` precomputed tree: built on device; nodes D2H'd **only when
  `build_precomputed` is true**, which is `cached_pre.is_none()` at `prover.rs:1156`
  — i.e. only on a process-cache miss.
- `:793-802` multiplicity tree: built on device and **kept resident**; only the
  32-byte root comes back.
- `:806-807` **the row-major LDE is D2H'd unconditionally**, with the comment
  `"preprocessed tables always keep the host copy — they are excluded from the
  device-only gate"`. This is the single line the whole remaining lever hangs on.
- `:810` row→column-major transpose on device, producing the `GpuLdeBase.buf` every
  downstream round reads.

The precomputed-tree cache (`prover.rs:175-214`) is process-wide, type-erased and
keyed by the commitment root, so a long-lived prover process builds BITWISE's
precomputed tree **once**, not once per proof. ✓ VERIFIED. (A fresh process per proof
— e.g. a CLI invocation — rebuilds it, but on GPU.)

---

## 3. Does the LFM machine actually reach these paths?

Working through the gates against the LFM chip set:

- **All 14 chips are the same generic AIR type** (`AirWithBuses`,
  `prover/src/lfm/airs.rs:31`), proved by one `multi_prove` call
  (`prover/src/lfm/proof.rs:140-146`). ✓ VERIFIED
- **13/14 preprocessed** — `build_air` chains `.with_preprocessed`
  (`airs.rs:331-349`); only `KECCAK_RND` uses `build_air_no_prep` (`airs.rs:313`,
  slot constant at `:74`). ✓ VERIFIED (unchanged from the earlier finding)
- **BITWISE: 2^20 rows, 21 columns, 11 preprocessed** (`prover/src/tables/bitwise.rs:98,
  94, 101`) — fixed in *every* LFM proof by the fixed-machine principle
  (`airs.rs:34-49`). ✓ VERIFIED
- **KECCAK_RC: 32 rows, 10 columns, 9 preprocessed** (`prover/src/tables/keccak_rc.rs:46,
  36, 40`). ✓ VERIFIED
- **LFM_RANGE: 2^16 rows**, fixed and program-independent
  (`prover/src/lfm/layout.rs:261-266`) — the second-largest always-present table.
  At blowup 2 its LDE is 2^17, **4× below the 2^19 threshold**, so it is entirely
  CPU today. ✓ VERIFIED
- All other chips are `padded_rows(real_rows) = real_rows.next_power_of_two().max(4)`
  (`prover/src/lfm/layout.rs:271-276`) — program-dependent. ✓ VERIFIED
- **Blowup = 2** in every registry entry (`prover/src/lfm/registry.rs:196, 280, 364,
  448, 532, 616`). ✓ VERIFIED. So **LDE size = 2 × trace rows**, and a chip needs
  **≥ 2^18 trace rows** to clear the 2^19 LDE threshold (`gpu_lde.rs:45`).
- **`split_col` precondition** `0 < split_col < m`: BITWISE gives 11 < 21 ✓;
  KECCAK_RC gives 9 < 10 ✓. ✓ VERIFIED
- **Composition parts = `max_degree − 1`** (`lookup.rs:1110-1126`). LogUp batched
  terms are degree 3, so every LFM chip is ≥ degree 3 ⇒ **2 parts**, which is exactly
  the `number_of_parts == 2` condition the device-resident R2 path requires
  (`prover.rs:1592`). ? INFERRED (degree-3 LogUp is stated at `lookup.rs:1113`; I did
  not enumerate each chip's `max_degree()`).
- **`end_exemptions == 0` everywhere** — stated and measured across all tables at
  `prover/src/lfm/constraints.rs:783-785`, and `RowDomain::ALL` is the only row
  domain used in `prover/src/lfm/`. So `zerofier_uniform` holds. ✓ VERIFIED
- **`transition_offsets = [0, 1]`** for every `AirWithBuses` (`lookup.rs:959`) ⇒
  `offsets_are_contiguous` ✓. ✓ VERIFIED
- **`has_aux_trace` + non-empty `constraints_meta`**: even chips built on
  `EmptyConstraints` (BITWISE, KECCAK_RC, LFM_CONST, LFM_LANES, LFM_HINT, LFM_PUBLIC,
  LFM_RANGE) get LogUp metas appended by the framework (`lookup.rs:931-937`), and
  BITWISE has 10 bus interactions ⇒ 6 aux columns. Both preconditions at
  `prover.rs:1031` hold. ✓ VERIFIED

## 3.1 What the *recursion* workload actually looks like (measured, in-repo)

This is the part that changes the conclusion, and it is measured rather than derived.
**Do not reason about LFM-on-GPU from the registered programs** — they are toy-sized.
The recursion target is the assembled **epoch-verifier wrap**.

**The wrap that has actually been proved** — inner epoch min preset, wrap options
blowup 2 / 219 queries / grinding 20, 14 LFM sub-proofs, prove 19.5 s, verify 0.09 s,
peak RSS 15.1 GiB on an 11-core laptop
(`others/lfm-agent-status.log:183, 186`) ✓ VERIFIED (checked-in measurement):

| chip | rows | cols | main cells | share | LDE @ blowup 2 | ≥ 2^19? |
|---|---|---|---|---|---|---|
| **KECCAK_RND** | 131,072 = 2^17 | 1,480 | 193,986,560 | **88.1%** | 2^18 | **✗ misses by 2×** |
| LFM_BALU | 2^21 | 4 (+10 prep) | 8,388,608 | 3.8% | 2^22 | ✓ |
| BITWISE | 2^20 | 10 (+11 prep) | 10,485,760 | 4.8% | 2^21 | ✓ |
| LFM_XALU | 2^17 | — | — | — | 2^18 | ✗ |
| LFM_LANES, LFM_RANGE | 2^16 | — | — | — | 2^17 | ✗ |

Totals for that wrap: 220,107,920 main + 87,073,068 aux ext = 481,327,124 base-field
equivalents. The "fixed-machine floor" (empty program) is 10,560,752 main = **4.8%**.

**Two consequences, and they point in opposite directions from the old note:**

1. **BITWISE is ~5% of the recursion workload, not 97–99%.** That 97–99% figure came
   from the *registered toy programs* (trivial / keccak_sponge / statement_replay),
   where the fixed machine is the whole proof. Carrying it into a recursion argument
   is a category error. ✓ VERIFIED by the census above.
2. **The dominant chip is `KECCAK_RND` — the one non-preprocessed chip** — and it is
   short and enormously wide (2^17 × 1,480). The row-based threshold sees only 2^17
   rows and refuses it, even though it is ~9× BITWISE's cell count.

**Chunking is what decides whether this bites.** `KECCAK_RND_MAX_CHUNK_ROWS = 1 << 19`
(`prover/src/lfm/chunking.rs:41`, 24 rows/permutation at `:34`) ✓ VERIFIED. So a
**full** chunk is 2^19 rows → LDE 2^20 at blowup 2 → **clears the threshold**. It is
**partial / small-epoch chunks that fall off the GPU**, which is exactly the case in
the only wrap proved end to end. A production-sized epoch verify (the blowup-8 census
at `others/lfm-hash-matrix-scope.md:1146-1151` — 6 chunks, 2,883,584 rows, plus
LFM_BALU 2^27, BITDEC 2^21, LANES 2^21) would have its chunks at the cap and would
clear — but that configuration **has never been proved** (350.6 GiB projected peak;
`wrap_tests.rs` `the_wrap_census_at_blowup_8` is `#[ignore]`d). ✓ VERIFIED

**Net:** at blowup 2, `LFM_BALU` and `BITWISE` clear every gate for the GPU split path
today and (being preprocessed) fail only `device_only_gate`. `KECCAK_RND` fails the
size gate at small chunk heights and clears it at full ones — and because it is *not*
preprocessed, once it clears it also qualifies for full device-only residency with
**no code change at all**.

### A correction to the previous finding's cost model

The earlier note recorded *"value scratch is 1.5 MiB per IR node → `KECCAK_RND` at
full chunk height wants 23.9 GiB"*. **That is stale.** Main's device lowering now does
**liveness slot reuse**: slots are freed at an operand's last use and the kernel
allocates `num_base_slots`/`num_ext_slots` per thread, not one slot per node
(`crypto/stark/src/constraint_ir/device.rs:24-30`, `:168-171`;
`crypto/math-cuda/src/constraint_interp.rs:99-101`). The working set is now the
*live* set, not the node count. ✓ VERIFIED. Whether that brings `KECCAK_RND` under
VRAM is a box measurement, not a code read — but the 23.9 GiB figure should not be
carried forward.

---

## 4. The remaining levers, in priority order

### Lever 1 (top) — make the GPU LDE gate cell-aware, not row-only

The threshold is `lde_size < gpu_lde_threshold()` where `lde_size = n × blowup`
(`gpu_lde.rs:698-700`, `:799-801`, `:881-883`) — **there is no column term anywhere in
it**. Its own doc justifies this as "the check is on lde size, not trace length,
because that's what determines the FFT workload" (`gpu_lde.rs:36-44`) — true for the
RV64 VM's tall-and-narrow tables, **false for the LFM machine**, whose chips are short
and extremely wide. The work is `num_cols` FFTs of length `lde_size`, so the honest
cost proxy is `lde_size × num_cols`.

What it costs today, on the wrap that has actually been proved:

| chip | LDE × cols | bytes | on GPU? |
|---|---|---|---|
| **KECCAK_RND** | 2^18 × 1,480 | **~3.0 GiB** | **✗ CPU — 88.1% of main cells** |
| LFM_BALU | 2^22 × 14 | ~469 MiB | ✓ |
| BITWISE | 2^21 × 21 | ~352 MiB | ✓ |

? INFERRED arithmetic over ✓ VERIFIED row/column counts.

**Why this lever is the best one:** `KECCAK_RND` is the **one non-preprocessed chip**
(`airs.rs:313`, `:74`). Once it clears the size gate it satisfies `device_only_gate`
outright — no `!is_preprocessed` problem, no new kernel, **no code change at all**.
Everything from §1 (device LDE, device Merkle, GPU composition, R3, R4, FRI) plus full
device-only residency applies to it immediately.

**Test it with one env var** — `LAMBDA_VM_GPU_LDE_THRESHOLD=262144` (2^18) admits the
2^17-row chunk at blowup 2. The permanent fix is a cell-aware gate; the env var proves
the thesis first.

**Caveat, stated honestly:** full `KECCAK_RND` chunks are capped at 2^19 rows
(`chunking.rs:41`), so at blowup 2 a *full* chunk already clears at 2^20. This lever
therefore matters most for **partial chunks and small epochs** — which is precisely
the only configuration anyone has proved end to end. On a production-sized epoch the
chunks sit at the cap and this lever shrinks; lever 2 grows correspondingly.

### Lever 2 — lift `!is_preprocessed` from `device_only_gate` (`gpu_lde.rs:210`)

Per proof of the actually-proved wrap, at blowup 2, from the two preprocessed chips
that are already on GPU:

| Buffer | Size | Gate that retains it |
|---|---|---|
| BITWISE main LDE host copy | 2^21 × 21 × 8 B = **352 MiB** | `math-cuda/src/lde.rs:806` |
| BITWISE aux LDE host copy (6 ext3 cols) | 2^21 × 6 × 24 B = **288 MiB** | `prover.rs:3407` / `:3449` |
| BITWISE composition parts (2 ext3 parts) | 2^21 × 2 × 24 B = **96 MiB** | `prover.rs:1606` |
| LFM_BALU main LDE host copy | 2^22 × 14 × 8 B = **469 MiB** | same |
| **total (main+aux+parts, both chips)** | **≳ 1.2 GiB** | |

? INFERRED arithmetic. For calibration: the same D2H skip on the RV64 VM measured
**−12.5%** prove time (memory `gpu-constraint-eval`, 5 ABBA pairs, RTX 5090). This
lever **grows** on production-sized epochs, where more and larger preprocessed chips
clear the size gate.

**Cost:** three changes, all small, all in already-guarded code.

1. `gpu_lde.rs:210` — drop `&& !is_preprocessed`.
2. Thread a `retain_host_lde` flag into `try_expand_split_trees_row_major_keep`
   (`gpu_lde.rs:779`) and `coset_lde_row_major_split_trees`
   (`math-cuda/src/lde.rs:718`), so the D2H at `lde.rs:806` becomes conditional —
   exactly mirroring what `coset_lde_row_major_with_merkle_tree_keep` already does
   for the plain path.
3. `prover.rs:2758` — let preprocessed tables use `main_dev_values`. The device
   gather returns **full** rows (`math-cuda/src/barycentric.rs:357-390`, no column
   range), and preprocessed openings need only columns
   `[num_precomputed_cols, total_cols)`, so this needs a range-slicing variant of
   `device_row_pair` (`prover.rs:2633`) — the values are already in hand, it is a
   slice, not a new kernel.

The safety net already exists: every host-read fallback carries a
`host_trace_empty()` hard-abort (`evaluator.rs:240`, `:1637`, `trace.rs:786`, `:844`,
`prover.rs:2320`, `:2617`, `:2627`), so a missed precondition aborts loudly instead of
producing a wrong proof. ✓ VERIFIED

### Lever 3 — the hash, which is a machine-design question, not a GPU one

**84.0% of the production-shaped epoch verify's cells are the hash**
(`LFM_KECCAK + KECCAK_RND`), at 36,256 main + 13,912 aux cells per permutation
(`others/lfm-agent-status.log:196`) ✓ VERIFIED. No amount of GPU work changes that
ratio — it is what the BLAKE3 column exists to attack. Note also that the
production-shaped wrap is **not provable at 124 GiB** (350.6 GiB projected,
`lfm-agent-status.log:198`), so "make recursion fit" may outrank "make recursion fast".

**Note on BITWISE:** the earlier framing — "BITWISE is 97–99% of cells, shrinking it
dominates everything" — holds only for the *registered toy programs*. In the recursion
wrap BITWISE is 4.8%. Do not spend design effort there on recursion's account.

---

## 5. Staged plan for the GPU box

### Stage 0 — establish the baseline and falsify the map (do this first)

The single most valuable measurement, and it is a *falsification test of §1*:

**Measure the WRAP, not the registered programs.** The registered programs are
toy-sized (§3.1) and will tell you almost nothing. The target is the assembled
epoch-verifier wrap in `prover/src/lfm/wrap_tests.rs` — the harnesses there are
`#[ignore]`d, so they need `--ignored`. Use the **min inner preset** (the one that has
actually been proved: 19.5 s, 15.1 GiB peak on an 11-core laptop); the blowup-8
production shape is not provable at 124 GiB.

**Falsifiable predictions.** On an LFM *wrap* prove under `--features cuda`, post-merge:
- `gpu_composition_calls()` **≥ 2** (BITWISE and LFM_BALU both clear the size gate).
  The earlier finding predicted **0** for everything; if it is still 0, §1 is wrong.
- `gpu_merkle_tree_calls()` **≥ 4** cold (two subset trees each for BITWISE and
  LFM_BALU), fewer warm as the precomputed-tree cache fills.
- `gpu_device_only_calls()` **exactly 0** — BITWISE and LFM_BALU are preprocessed, and
  `KECCAK_RND` (the only non-preprocessed chip) is at 2^17 rows, under the gate. This
  is the sharpest prediction in the document: **it is 0 only because of a threshold,
  and one env var should flip it to ≥ 1.**
- `gpu_lde_calls()` should **exclude** `KECCAK_RND`'s 1,480 columns — i.e. the counter
  should be far below the machine's total column count. That is lever 1's evidence.

Commands (do **not** run locally — these are for the box):

```bash
# 0. Confirm the box's own GPU stack is healthy BEFORE touching LFM.
#    Both targets are ✓ VERIFIED in the Makefile (:565, :572).
make test-math-cuda
make test-cuda-integration

# 1. Build. Do not add RUSTFLAGS; sccache cache keys must stay stable.
cargo build --release -p lambda-vm-prover --features cuda

# 2. THE measurement: the assembled epoch-verifier wrap, min inner preset.
#    Counter-gated work shares global atomics, so keep --test-threads=1 throughout.
cargo test --release -p lambda-vm-prover --features cuda \
  lfm::wrap_tests -- --ignored --nocapture --test-threads=1

# 3. A cheap smoke check first, if the wrap is slow to iterate on.
cargo test --release -p lambda-vm-prover --features cuda \
  lfm::machine_tests::machine_proves_the_sample_replay -- --nocapture --test-threads=1
```

There is **no counter-printing harness for LFM in-tree** — the PR #915 artifacts
(`prover/src/lfm/device_parity_tests.rs`, `prover/tests/gpu_lfm_constraint_interp.rs`,
`thoughts/shared/lfm-gpu/`) live on branch `lfm-gpu-experiments` /
`/Users/maurofab/workspace/lambda_vm-lfm-gpu` and are **not present in this worktree**
(✓ VERIFIED by `ls`). **Stage 0's real deliverable is a ~30-line test** that copies the
shape of `prover/tests/cuda_path_integration.rs:22-36` — `reset_all_gpu_call_counters()`
(`gpu_lde.rs:70`), prove one LFM program, print/assert every counter, verify. Note
those tests are `#[ignore]`d and need `--ignored`; follow the same convention so
no-GPU CI keeps skipping.

Timing baseline, for the A/B in Stage 1:

```bash
# Per-phase timings (feature `instruments`) — shows where LFM prove time actually goes.
cargo test --release -p lambda-vm-prover --features cuda,instruments \
  lfm::machine_tests::machine_proves_the_sample_replay -- --nocapture --test-threads=1

# Per-chip geometry, so §6 open question 1 stops being open. `lfm_chip_census`
# (airs.rs:138) already computes rows/main_cols/aux_cols per chip.
cargo test --release -p lambda-vm-prover lfm:: -- --nocapture --test-threads=1 2>&1 | grep -i census
```

### Stage 1 — A/B the levers that need no code change

```bash
# (a) ★ LEVER 1, the headline experiment. 2^18 admits the 2^17-row KECCAK_RND chunk
#     (88.1% of main cells) at blowup 2. Expect gpu_device_only_calls() to go 0 -> >=1
#     with ZERO code changed, because KECCAK_RND is the non-preprocessed chip.
LAMBDA_VM_GPU_LDE_THRESHOLD=262144 cargo test --release -p lambda-vm-prover \
  --features cuda lfm::wrap_tests -- --ignored --nocapture --test-threads=1
# then 2^17 (adds XALU/LANES/RANGE) and 2^12 (everything) to find the real knee:
LAMBDA_VM_GPU_LDE_THRESHOLD=131072 ...
LAMBDA_VM_GPU_LDE_THRESHOLD=4096   ...

# (b) Is the GPU composition path worth anything on LFM? In-binary A/B.
LAMBDA_VM_DISABLE_GPU_COMPOSITION=1 cargo test --release -p lambda-vm-prover \
  --features cuda lfm::wrap_tests -- --ignored --test-threads=1 --nocapture

# (c) Table concurrency: 14 AIRs of wildly unequal size. Default is cores*2/3 under
#     cuda (prover.rs:588-616). Sweep it.
TABLE_PARALLELISM=1  ... ; TABLE_PARALLELISM=4 ... ; TABLE_PARALLELISM=14 ...

# (d) Other knobs available (grep-verified, gpu_lde.rs / logup_gpu.rs / prover.rs):
#     LAMBDA_VM_DISABLE_DEVICE_ONLY, LAMBDA_VM_GPU_BARY_THRESHOLD,
#     LAMBDA_VM_NO_GPU_LOGUP, LAMBDA_VM_VRAM_BUDGET_MB, LAMBDA_VM_LOGUP_TIMING
```

**Watch VRAM on (a).** `KECCAK_RND` at 2^18 LDE × 1,480 columns is ~3.0 GiB of LDE
before scratch; `estimate_table_vram_bytes` (`prover.rs:622`) will price it around
6–7 GiB. That fits a 5090, but it is the first LFM table big enough to make the
admission gate matter — if it thrashes, `LAMBDA_VM_VRAM_BUDGET_MB` is the knob.

**Prediction for (a):** **large** — this is the one to bet on. It moves 88.1% of the
machine's main cells from CPU to a fully device-resident path in one env var. If it
does *nothing*, the most likely explanations are (i) `KECCAK_RND`'s constraint program
does not fit the interpreter's per-thread slot scratch and silently falls back to CPU
(a fallback, not an error — check `gpu_composition_calls()` did not rise), or (ii) the
1,480-column shape breaks a kernel launch assumption. Both are worth knowing.

**Prediction for (b):** small, per the VM's −2.7%-vs-−12.5% split.

### Stage 2 — build the lever (the three-part change in §4, lever 2)

Order matters, because each step is independently verifiable:

1. Make the split-path D2H conditional (`lde.rs:806`, `gpu_lde.rs:779`) but keep
   `device_only_gate` unchanged. **Behaviour-neutral** — nothing sets the flag yet.
   Gate: existing `cuda_path_integration` suite still green.
2. Add the range-slicing device opening for preprocessed tables (`prover.rs:2758`,
   `:2633`) with the release cross-check at `prover.rs:2640` left on. **Still
   behaviour-neutral** for correctness; the cross-check is the oracle.
3. Flip `gpu_lde.rs:210`. Now `gpu_device_only_calls()` should go from 0 to ≥1 on an
   LFM prove, and no `host_trace_empty` assert may fire.

**Falsifiable prediction for Stage 2:** after step 3, an LFM prove reports
`gpu_device_only_calls() ≥ 1`, zero guard panics, verify green, and prove time drops.
If a `host_trace_empty` assert fires, a precondition in `device_only_gate` is not
implied by some dispatch — that is the documented LOCKSTEP hazard at
`gpu_lde.rs:185-188`, and the message names the round.

**Gate before believing any speedup:** proofs are non-deterministic, so never diff
bytes. Use prove→verify plus cross-version verify, per the house rule.

### Stage 3 — make the threshold permanently cell-aware

If Stage 1(a) confirms lever 1, replace the row-only check with a cell-aware one
(`gpu_lde.rs:698-700`, `:799-801`, `:881-883`, and the `device_only_gate` mirror at
`:207-209`). The proxy is `lde_size × num_cols` — every call site already has
`num_cols` in scope. Recalibrate the constant against the box rather than reusing
2^19, whose doc says it was calibrated on a 46-core machine for the VM's shape
(`gpu_lde.rs:36-44`). Keep the row check too if a minimum FFT length matters for
launch efficiency; the point is that width must enter the decision.

---

## 6. Open questions I could not settle from code alone

1. ~~**Heights of the program-dependent chips.**~~ **SETTLED** by the checked-in
   census (§3.1). Fixed tables: BITWISE 2^20, LFM_RANGE 2^16, KECCAK_RC 32. Proved
   wrap: KECCAK_RND 2^17×1,480, LFM_BALU 2^21, LFM_XALU 2^17. Blowup-8 production
   shape (never proved): BALU 2^27, BITDEC 2^21, LANES 2^21, KECCAK_RND 6 chunks.
   All ✓ VERIFIED.
2. **Does `KECCAK_RND`'s constraint program actually run on the GPU interpreter?**
   This is now the load-bearing unknown, because lever 1 depends on it. It is the
   biggest program in the machine (16,317 IR nodes on the old count) and the kernel
   allocates per-thread slot scratch. Liveness reuse (§3.1 correction) should make it
   fit, but "should" is a code read. **A silent OOM here is a CPU fallback, not an
   error** — so the counter must be checked, not just the wall clock.
3. **Whether the R2 device path fires for each chip.** The gate is
   `number_of_parts == 2` and `max_degree` is per-chip; I derived degree-3 from LogUp
   but did not enumerate each chip. Stage 0's counter test settles it.
4. ~~**VRAM headroom.**~~ Settled for the current shape by reading
   `estimate_table_vram_bytes` (`prover.rs:622-633`): BITWISE ≈ 1.19 GiB, and
   `VramGate` (`prover.rs:635-647`) admits an oversized table alone so nothing
   deadlocks. ✓ VERIFIED by reading. **Reopens under lever 1**: `KECCAK_RND` at
   2^18 × 1,480 prices at ~6–7 GiB, the first LFM table where admission matters.
5. **How many wrap proves a real recursion campaign needs**, and whether the
   production-shaped epoch (350.6 GiB projected, `lfm-agent-status.log:198`) is ever
   provable — GPU speed is moot if the shape does not fit at all.
6. **Whether the precomputed-tree cache survives the LFM proving schedule.** It is
   process-wide and root-keyed (✓ VERIFIED), so it should — but if LFM proving runs
   one process per proof, the precomputed trees are rebuilt every time (on GPU, but
   still a full leaf hash plus a node D2H at `lde.rs:785`).

---

## 7. Provenance

- Split-tree preprocessed GPU path introduced by **`d83b4d9e` — "perf(prover): halve
  GPU continuation proving time (#863)"** (found via `git log -S
  try_expand_split_trees_row_major_keep origin/main`). Its own message says:
  *"Precomputed-column Merkle trees are cached process-wide keyed by their commitment
  root, so preprocessed tables (DECODE/BITWISE/range) stop rebuilding identical trees
  on every prove; only the multiplicity columns are recommitted."* ✓ VERIFIED
- Device-resident rounds 2-4 introduced by **`5749a956` — "perf(prover):
  device-resident rounds 2-4 and fused NTT for GPU continuations (#875)"**. ✓ VERIFIED
- Both predate this worktree's merge and postdate the earlier LFM-GPU finding, which
  is why that finding's map no longer holds.
