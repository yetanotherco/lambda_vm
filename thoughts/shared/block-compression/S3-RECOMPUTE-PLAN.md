# S3 — Recompute-instead-of-retain: bounded-memory proving for the wrap

**Status: DRAFT — awaiting Mauro's read. The hunted design is FOUND and this plan is its
revival; provenance below governs naming and salvage.**

## 0-pre. Provenance — this design already existed, was built, and was benched

The alternative continuation design = **"Approach 1: Prove-and-retire"** from the
streaming spec (spec PR #642, branch `spec/streaming`; today a 4-line footnote in
`streaming.typ` — "additional engineering complexity and re-executions"). The "initial
discussion" that chose Approach 2 (today's continuations, PR #685) was never written
down on GitHub. But the design has three recorded homes:

1. **PR #647 "Feat/streaming prover" (diegokingston, Jun 2026, closed unmerged; branch
   `origin/feat/streaming-prover` still live, 10 commits):** a COMPLETE implementation —
   `LAMBDA_STREAM_LDE=1` retire-LDE recomputing on demand via `reconstruct_round1` (M1),
   leaf-drop Merkle trees keeping internal nodes (T1 — the same keep-the-tree choice §2
   makes), `Executor::snapshot/from_snapshot` VM checkpoints (B), deterministic trace
   builds via sorted dedup (C.2a — a HARD prerequisite: HashMap-order row
   nondeterminism breaks commit-vs-rebuild root equality), on-demand per-table trace
   rebuild (C.2b ≈ Phase C here), batched per-lde_size FRI. **Byte-identical proofs
   flag-on vs flag-off, verified** (grinding disabled).
2. **The kill bench, and why it does NOT carry to the wrap:** #647 benched on
   monolithic `fib_iterative_8M`: **−0.9% peak heap / +14.1% prove time** → closed with
   "We are now using #685". On that workload the retired LDE cache was ~1% of peak
   (peak lived in the trace builder/executor). On the WRAP, the retained LDEs are the
   MEASURED dominant term (11.56 of the 13.4 GiB/chunk marginal; 532–1,538 GiB summed
   over chunks). Same trade, opposite workload shape: the +14% time now buys the entire
   fit. The historical verdict was correct FOR ITS WORKLOAD and is not a verdict on this one.
3. **`memory/streaming-proving-vs-zisk.md`** (2026-06-03): the axis analysis. Approach
   1's 2× re-execution was forced by a single global proof's FS ordering. NOTE: S3 does
   not inherit that — the wrap's tables already have per-table transcript forks; S3's
   recompute sits entirely below the transcript.

**Salvage map (verified against today's branch):** `reconstruct_round1` SURVIVES in-tree
(`prover.rs:1358`, debug-checks path) — the recompute engine exists and Phase A largely
promotes it out of cfg(debug-checks) under the new mode. From `feat/streaming-prover`
(June-era, big drift vs main — cherry-pick ideas/tests, not rebase): the byte-identical
oracle test, the two-pass round split, T1's tree handling, and for Phase C the
`VmSnapshot` machinery (never landed on main) + the C.2a determinism fix — **check
whether main's LT/MUL/DVRM/BRANCH builders still have HashMap-order row nondeterminism;
if yes it is a live Phase-C precondition** (it changes proof output = a re-bless).
Naming follows #647's (M1/T1/C.x) where it overlaps.

Grounding: `residency-seam-audit.md` (the verified retention map; every file:line there),
`CENSUS.md` Part 2 §1 (independent second read) and Part 3 (the measured spill ladder that
this plan's marginal predictions extend). Measured anchors: **13.4 GiB marginal per full
KECCAK_RND chunk with spill on** (q=12→16 differencing); the audit's model 17.37·N + 30.2·k
GiB validated within the expected anon delta.

---

## 0. The one-sentence design

Commit each table's Round-1 root exactly as today, then **drop the main LDE** (keep the
32 B/row Merkle tree and the trace); when that table's fused task (aux → R2 → R3 → R4)
runs after the shared challenge, **recompute the LDE from the trace** into a task-local
buffer that dies with the task — turning the N-way LDE retention into a k-way transient,
with zero change to roots, transcript order, or proof bytes.

## 1. Why this is sound (the protocol argument, verified in code)

- Fiat–Shamir requires all main roots to be absorbed before the shared LogUp challenge
  (`prover.rs:3196-3225`; verifier mirror `verifier.rs:1295-1317`). It requires the
  ROOTS — nothing about the LDE buffers. Retention is a performance choice, stated as
  such by the `Lde` struct's own doc (`prover.rs:263-274`).
- After the per-table transcript fork (`prover.rs:3263-3271`) tables are independent;
  each `StarkProof` is self-contained (`prover.rs:3856-3887`).
- Recomputation is deterministic: same trace + same twiddles (process-cached,
  `prover.rs:517-574`) → bit-identical LDE → identical opening values against the KEPT
  tree. The tree is never recomputed, so there is no "recomputed root must match" hazard
  at all — the root that entered the transcript is the root openings are checked against.

## 2. What is dropped, kept, recomputed — and why the tree is KEPT

| buffer (per KECCAK_RND chunk, blowup 2) | size | S3 decision | rationale |
|---|---|---|---|
| main LDE | 11.56 GiB | **DROP after R1 commit; RECOMPUTE in fused task** | the binding buffer; one extra NTT to recompute |
| main Merkle tree | 0.03 GiB | **KEEP** | keeping it makes recompute = one NTT, NOT NTT + full leaf re-hash; R4 auth paths read the tree, only opening VALUES read the LDE |
| main trace | 5.78 GiB | keep (Phase A); lazy-regenerate (Phase C) | it is the recompute input |
| aux trace | 6.05 GiB | keep (Phase A); **free at fused-task end (Phase B)** | written into the caller's TraceTable (`lookup.rs:1209-1211`) and today never freed; nothing reads it after the table's proof is done |
| aux LDE, composition, DEEP, FRI | ~12.4 GiB | unchanged | already k-bounded inside the fused task |

**Marginal-per-chunk prediction (falsifiable on the box):** today with spill ≈ **13.4**
(measured). Phase A → **≈ 11.9** (trace 5.78 + aux trace 6.05 + tree 0.03). Phase A+B →
**≈ 5.8**. Phase A+B+spill (traces to mmap) → **≈ 0.03 resident** — the flat floor.
If the measured Phase-A marginal is not ≈ LDE-sized lower than 13.4, the implementation
missed a retention point; that is the acceptance test, not wall-clock.

## 3. The phases

### Phase A — core S3 (effort M; crypto/stark only, no public-signature changes)

1. `ResidencyMode { Retain, RecomputeLde }` — a NEW enum next to `StorageMode`, no cargo
   feature (pure code path, no disk dependency), default `Retain` so every existing
   caller is byte-identical. Threaded like storage_mode into `multi_prove`; LFM call
   site opts in via env (`LAMBDA_VM_RESIDENCY=recompute`) through `auto_storage::decide_lfm`
   (the c5ffadf3 seam).
2. R1: after the commit produces `(root, tree, lde)`, under `RecomputeLde` push root+tree
   as today but drop the LDE instead of accumulating it into `main_ldes`
   (`prover.rs:3144-3145, :3201`). The `main_lde_cells` accounting (`:3306-3316`)
   follows the mode.
3. Fused task entry: under `RecomputeLde`, recompute the table's main LDE from its trace
   (same `coset_lde_full_expand_row_major` the commit used, minus tree building) into a
   task-local; every downstream consumer inside the task (aux build's `columns_main`,
   R2 evaluator, R3 barycentric, R4 DEEP + opening values `gather_main_row_range`)
   reads it exactly as it reads the retained buffer today — same type, different lifetime.
4. Preprocessed tables: the precomputed-columns tree stays process-cached (untouched);
   the multiplicity LDE gets the same drop/recompute treatment. Verify the cache path
   (`prover.rs:1151-1159`) is mode-independent.
5. cfg surfaces that ASSUME retention get gated: `debug-checks` reconstruction
   (`prover.rs:1338+`) forces `Retain` (mirroring how `device_only_gate` already returns
   false under debug-checks); the cuda `device_only`/handle paths are DISJOINT from this
   mode in Phase A — `RecomputeLde` is documented CPU-prove-oriented, and under cuda it
   forces the host path per-table (same posture as spill; the fit story is CPU proving,
   per Mauro's own framing).

**Oracles A:** full suites at exact baselines (prover 859/34, stark 241/0, lfm 307/19);
a NEW roots-equality test — same trace, same statement: `Retain` and `RecomputeLde`
produce IDENTICAL commitment roots (roots are diffable even though whole proofs are not,
per the house never-diff-proof-bytes rule); cross-mode verify (proof made under
`RecomputeLde` verifies with the standard verifier — same bytes format, this is nearly
tautological and that is the point); `make lint`/`fmt`; then the BOX LADDER re-run at
q=12/16/20 — q=16 must complete with marginal ≈ 11.9 GiB/chunk, q=20 (which paged out
at 52 GiB anon) should now complete.

### Phase B — aux-trace release (effort S-M)

Free each table's aux columns from the caller-owned `TraceTable` when its fused task
completes (they are dead weight after the table's proof exists). This mutates
caller-visible state, so it is part of the documented `RecomputeLde` contract, not a
silent change to `Retain`. Oracle: suites + marginal drops to ≈ 5.8 GiB/chunk on the box.

### Phase C — lazy chunk traces (effort L; ONLY if post-P-a numbers demand it)

The audit's S1/S2/S6: `multi_prove` takes a per-index trace producer; `LfmTraces` stops
materializing all KECCAK_RND chunks (each is a pure function of its `round_ops` slice,
`chunking.rs:12-21` — regeneration trivially available); each chunk's trace is generated
twice (R1 commit, fused task) and never coexists with its siblings. Floor → tree-roots
only, ~0.03 GiB/chunk marginal + one working set. This changes the `AirTracePair`
signature — a real API refactor, separately reviewed, and the point where the hunted
alternative continuation design (if found) must be reconciled first.

**Decision gate for C:** ~~after P-a lands, re-census. Model says Phase A+B post-blake3 at
the cheapest geometry ≈ fits the 124 GiB rigs with margin (traces are the only O(N) term
left and they divide by the hash shrink too); if the re-census disagrees, C proceeds.~~

> ### ★ GATE C RESOLVED BY MEASUREMENT (2026-08-13) — S6 is box-class-dependent
>
> The gate no longer waits on a re-census. ✓ MEASURED on a 60 GiB / 32-core box:
> the real-block wrap (block 25368371, epoch 0 at 2^16, inner blowup4 / **110
> queries**) is **OOM-killed at 56.91 GiB anon, BEFORE proving starts** — spill
> volume 0.00 GiB, disk untouched, `RssFile` peak 0.01 GiB. Emission succeeds
> and prints its full census first, so neither the emitter nor the prover is the
> wall. The wall is `build_traces_with_hasher`
> (`prover/src/lfm/trace.rs:162-167` = **S6**), which materialises all 15
> `KECCAK_RND` chunk traces into one `Vec` — 87 GiB — before `multi_prove` is
> called. **Phases A and B bound residency inside `multi_prove` and are never
> reached.**
>
> So the gate splits by box class rather than by census:
>
> | box RAM | verdict on S6 at 110q |
> |---|---|
> | 64-128 GiB | **REQUIRED.** 87 GiB of eager trace alone; build-side spill does not rescue it either (87 GiB against a 61 GiB disk). |
> | ~258 GiB | **NOT required.** The eager build fits; A+B then bound the prove. |
>
> S6 is therefore the enabler for the 64-128 GiB class, not an optimisation, and
> the campaign can reach the secure inner preset today by using a big-memory box
> instead of building it. Phase C's cost/benefit is now a hardware-procurement
> question rather than a proving-architecture one.
>
> Corollary worth carrying: the arithmetic that made build-side spill look
> pointless — "a main LDE is `blowup` × its trace, so Σ traces is at most half of
> Σ main LDEs" (CENSUS Part 3 §5) — compares two quantities that are only both
> alive if the prove is reached. Σ traces is what must be resident *to call*
> `multi_prove`, so at large N it binds first regardless of the LDE side.

## 4. Cost model (stated honestly)

Recompute cost = ONE extra forward NTT per table per prove (the tree is kept, so no
re-hashing — this roughly halves the audit's +40-60% wall estimate, which priced
LDE+tree recompute; ? MODELED, the box ladder measures it). k concurrent recomputed
LDEs bounded by TABLE_PARALLELISM exactly as today's transients are.

## 5. Composition with everything else in flight

- **Spill (c5ffadf3):** composes — spill moves the traces Phase A retains onto mmap;
  spill+A+B is the best CPU configuration short of Phase C.
- **P-a:** orthogonal (hash choice never appears in this plan); the ÷4 multiplies.
- **GPU:** untouched in Phase A (mode forces host path per-table under cuda).
  **★ PHASE A2 — DEVICE-RECOMPUTE (promoted from deferred to the designated follow-up,
  Mauro 08-13):** the same seam, re-expanding into VRAM instead of host RAM — drop the
  device LDE handle after the root is absorbed, re-expand on device at fused-task entry
  (one NTT on the card), composed with the existing VRAM admission gate scheduling
  tables through the 32 GiB budget. This is the GPU-native bounded-memory prover and
  the endgame configuration under the 64-GiB-preferred production budget: host holds
  traces (lazy via S6 or spilled), VRAM holds one table's working set. The +7.8%
  CPU-side recompute cost shrinks toward noise on device. Sequenced after P-a's GPU
  stages (needs the blake3 kernels for blake3-committed tables; works under keccak
  immediately).
- **D0/tower:** unaffected; tower nodes already fit without S3.

## 6. Risks

1. A downstream consumer reading the LDE OUTSIDE the fused task that the audit missed —
   the loud guard: under `RecomputeLde`, poison the dropped buffer path (the existing
   `host_trace_empty`-style assert pattern) so a missed consumer aborts instead of
   silently reading empty data.
2. debug-checks / test-utils paths that reconstruct or cross-check from retained LDEs —
   gated to `Retain` (step A5); the suites run both modes to keep coverage honest.
3. The disk-spill + recompute interaction on the SAME table (spilled trace → recompute
   reads through mmap = page-cache pressure instead of anon): measured on the box, not
   assumed.
4. `StorageMode::Disk` disabling the precomputed-tree cache (spill ladder finding)
   compounds if both modes are on — measure the preprocessed-heavy fixture point.

## 7. What this is NOT

Not a protocol change, not a proof-format change, not the full streaming redesign, not
epoch-level checkpoint/re-execution (that is the hunted alternative design's territory —
if it surfaces, it likely replaces Phase C, not Phases A/B, since A/B live entirely
below the epoch abstraction).
