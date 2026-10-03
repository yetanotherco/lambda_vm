# PLAN — compress the Ethereum bench block (25368371) with the LFM machine

**Objective:** one proof attesting block 25368371 (74.8M cycles). A blowup-2/219q LFM
STARK as the single output is the campaign target; a further "small final proof" layer is
Stage E, explicitly optional and decided later.

**Grounding (all measured unless marked projected):**
- Block proves as N continuation epochs: 9 × 2^23 / 13-18 × 2^22 / ~36 × 2^21. Epoch size is
  a free knob of `prove_continuation`.
- The LFM wrap proves + verifies ONE epoch-verify today — but only for the 16-cycle fixture
  at the 1-query diagnostics preset (7.2 s on a 5090 with `LAMBDA_VM_GPU_LDE_THRESHOLD=262144`,
  BOX-RESULTS.md). Secure inner presets are blowup2/219q and blowup4/110q.
- Per-epoch verify cost has a floor independent of epoch size (queries × ~25-31 table proofs ×
  Merkle depth). Smaller epochs shrink each wrap but grow the total.
- The wrap's legs recompute the INNER prover's commitment hash. RV64 epoch proofs commit with
  keccak → base-layer wraps pay the hosted keccak family (84% of cells at production shape).
  Hash matrix (measured): epoch-verify 11.17B cells under keccak vs 2.75B under blake3-6r.
- The LFM proof's OWN commitments/transcript are `DefaultTranscript` (keccak) today; the
  machine's native real-blake3 domains (LFMC/LFML/LFMT) are what its programs compute, and
  FriToyV0 already proves+verifies blake3-shaped proofs. ✓ VERIFIED in `lfm/proof.rs`.
- Only production-shape census on record: blowup-8/73q single-epoch wrap → 350.6 GiB projected
  peak (unprovable). blowup2/219q and blowup4/110q have NEVER been censused.
- Census is free: `query_permutations` closed form + `projected_peak_bytes` — no proving needed.

---

## Shape: two tracks that meet

```
TRACK 1 (BASE, real epochs in)          TRACK 2 (TOWER, N→1)
A  census fit map                       D0 LFM-proof hash decision
B  one real ethrex epoch wrapped        D1 LFM-proof-verifier emitter
C  all N epochs wrapped + chaining      D2 aggregate 2→1 (on fixture wraps!)
        \                               D3 binary tree
         \                             /
          block → N base wraps → tower → ONE proof        [E: small final proof]
```

Track 2 starts immediately in parallel: D1/D2 prototype against TODAY'S fixture wrap —
they never wait on real epochs.

---

## Track 1 — real epochs into the wrap

### A. Census fit map (effort S, ~1-2 days, zero proving)
Sweep epoch_log2 ∈ {20, 21, 22, 23} × {blowup2/219, blowup4/110} × hash {keccak,
blake3-6r-modelled}. Per point: emitted-program cells, KECCAK_RND chunk count, projected
peak RSS. Trace-length profiles per epoch size come from EXECUTING the block (cheap),
not proving it. Harness: `real_epoch_with` + `report_census` generalized over the profile.
- **Gate A:** some (epoch size, preset) fits ~90-110 GiB (the box / rigs). If keccak-inner
  fits nowhere → the inner-hash switch (RV64 commits blake3-6r) is promoted from
  optimization to prerequisite and goes to Mauro as a decision.
- **★ GATE A VERDICT (2026-08-12, measured — CENSUS.md): FAIL AT EVERY POINT.** Cheapest
  real point (2^20/blowup4) projects 1,199 GiB — 13× over budget; the 219q program cannot
  even be EMITTED (OOM at 89 GiB during emission on the 16-cycle fixture). Scaling: linear
  in queries and sub-proof count, only logarithmic relief from epoch size (2^23→2^20 buys
  3.2×). KECCAK_RND is 92.5% of cells. **The inner-hash switch is NECESSARY BUT NOT
  SUFFICIENT** — blake3-6r's 4.06× leaves 295 GiB at the cheapest point (3.2× over). The
  coefficient-free floor after blake3 fits at exactly one point (2^20/blowup4 → 70 GiB), so
  the residual is PROVER RESIDENCY: peak is the SUM over 23-133 chunks in one multi_prove.
  Track 1 therefore adds two structural prerequisites: **(P-a) inner RV64 → blake3-6r**,
  **(P-b) bounded-residency proving**, and likely **(P-c) streamed emission** (the emitter
  itself OOMs first). ⚠ P-b CORRECTED: "one chunk ≈ 50 GiB flat" holds ONLY if nothing but
  the root survives per chunk; one full chunk's working set is 35.5 GiB, and if each
  chunk's main LDE + tree must survive Fiat-Shamir to answer openings, the floor is
  retained×N (267 GiB at N=23 … 1,542 GiB at N=133). Real P-b is likely RE-DERIVATION
  (retain roots, recompute chunk LDE+tree at query time, ~2× prover hash time for O(1)
  memory). Note Gate-A ran with `disk-spill` compiled OUT (not a default feature).
- **★ P-b RESOLVED BY AUDIT (residency-seam-audit.md; CENSUS.md Part 2 §1 is the
  independent second read): a real refactor with named seams, NOT a
  flag.** Nothing bounds residency today (`TABLE_PARALLELISM` bounds only aux/R2-4
  transients; disk-spill never touches the LDE and is unreachable from the LFM path).
  Fiat-Shamir forces only the ROOTS before the shared LogUp challenge — LDE retention is
  a perf choice, so the refactor is protocol- and wire-compatible. Peak model, KECCAK_RND
  family: **17.37·N + 30.2·k GiB** (N=23 → ~430 GiB today). Bounding only the LDE lands
  at 309-819 GiB; the flat floor needs the TRACE streamed too — chunks are pure functions
  of their `round_ops` slice (zero cross-chunk logic), so regeneration is trivially
  available → **~48-56 GiB flat regardless of N, at ? +40-60% wall time**. Seams S1-S7
  named in residency-seam-audit.md (multi_prove takes a per-index producer; LfmTraces
  goes lazy; drop-and-recompute LDE/trees). Also corrects Gate A's coefficient: 33.7 B/cell is ~2.1×
  high for the KECCAK_RND shape — the Gate-A band reads ~560-3,200 GiB; verdicts unchanged.
- **P-b LADDER (census Part 2, reconciled with the seam audit):** existing levers reach
  ~654 GiB at 2^21/blowup2 (`TABLE_PARALLELISM=1` → 972; + disk-spill on traces → 654 —
  note disk-spill is currently UNREACHABLE from the LFM path: feature off + `lfm/proof.rs`
  hardcodes Ram, so wiring is part of P-b). The missing piece either way is **main-LDE
  re-derivation at query time** (drop each LDE once its root is absorbed; Round-1 barrier
  requires only the ROOTS by soundness) → ~35 GiB with spill, ~48-56 GiB with the pure
  regeneration variant. Aux side is k-bounded per the `Lde` doc (`prover.rs:265-274`),
  which halves the big-epoch Gate-A figures: corrected band **1,300-2,692 GiB** — still
  14-29× over, verdicts unchanged. ~~**P-b is the highest-value item in the campaign.**~~
  **⛔ REORDERED BY MAURO (2026-08-12, verbatim: "Instead of doing weird streaming stuff,
  change the hash of the prover to blake3 first"): P-a GOES FIRST.** P-b demoted to
  fallback — after P-a lands, re-census and take the cheapest sufficient memory measure
  (existing flags + spill wiring first; streaming only if the numbers still demand it).
  Rationale that holds: P-a is needed at every layer forever, shrinks the workload at the
  source, and step 2's StarkHash parameterization makes it a second config instance
  rather than surgery. Kept visible: ÷4.06 alone projects ~320 GiB at the cheapest point
  (still over 93-124 GiB boxes), so SOME memory measure likely remains; the P-b seam
  analysis stays valid for that day. P-a staged plan: PA-PLAN.md (scoping in flight).
  Flip-time decisions RESOLVED by Mauro (2026-08-12): **6-round blake3** ("I'd prefer the
  6 round to see if this works") — 6r is the target, 7r stays buildable via the existing
  feature structure; and **blake3 CUDA kernels are pre-authorized as an agent dispatch
  whenever needed** ("send an agent to do the blake3 cuda kernels whenever it's needed")
  — closes the GPU regression window; kernel list from the GPU audit (row-pair leaves,
  column-range leaves, ext3 comp-poly leaves, FRI leaves, level/tail compressors), parity
  oracle = the in-repo host 6r implementation.
- **★ P-c RESOLVED BY AUDIT: the 89 GiB emission OOM is NOT the instruction stream**
  (271M × 80 B = 21.7 GB, ~24%). Dominant: the per-instruction `Vec<Vec<FE>>` row
  intermediate (~47 GB, with 80% capacity waste on 10-wide rows landing at cap 18) plus a
  drained-but-unshrunk `read_counts` HashMap (~18.3 GB) held by scope through the peak.
  **Two nearly-free wins: `drop(read_counts)` before `emit_column_groups` (−18.8 GB, one
  line) and a flat-append `ColumnGroupBuilder` (−27 GB, ~50 lines) → peak ~99-102 GB →
  ~53-56 GB, zero semantic change** (program_id commits over matrices, bit-identical).
  Full per-leg streaming: seams named (builder `instrs` field; compile merges into the
  builder; executor needs a 10-way merge by destination — the one new algorithm). ⚠ But
  emission is not the last wall: even streamed, execute wants ~21 GB memory + ~10 GB
  records, and LFM_BALU pads to 2^28 rows at 219q — the P-b prover streaming remains
  load-bearing. Trap for Stage B/C: real 2^23
  inner proves die on the 5090 via #927-class cliff panics; workaround = disable device
  paths (57.5 s CPU).

### B. First real rung (effort M, ~2-4 days)
1. Generalize the `RealEpoch` builder: parameterize ELF + private input + epoch_log2 +
   options (today it hardcodes the fibonacci fixture + empty input).
2. Prove ONE real-block epoch at the Gate-A geometry (RV64 continuation prove, GPU box).
3. Census the real epoch-verify program; falsify A's projection against it.
4. **Gate B: wrap it — prove + verify on the box.** First real compression artifact:
   one real-block epoch proof → one LFM proof. Everything downstream is scale-out.

### C. The whole block as N wraps (effort S code / compute-bound)
- Wrap all N epochs sequentially on GPU.
- Chaining: verify the emitter publishes the epoch boundary state (the spine already binds
  continuation roots per #844-adjacent design — VERIFY, don't assume). If the publics need
  additions, that is emitter/soundness surface → adversarial-debate review before merge
  (house rule from the merge-fix lesson).
- **Gate C:** block attested by N LFM proofs + host adjacency check, all verifying.

## Track 2 — the tower (N→1)

### D0. LFM-proof hash decision — **DECIDED 2026-08-12: Blake3 (Mauro: "Switch the blake3, yes")**
Tower legs recompute the LFM proof's OWN trees. Today that's keccak (`DefaultTranscript`)
→ the tower would pay the expensive chips forever. Switching the LFM proof's
commitments/transcript/FRI to the machine's native blake3 domains makes every tower layer
~4× cheaper in cells — and the chips already exist, z3-gated, proven in FriToyV0.
Proof-breaking for LFM proofs only (no RV64 impact). **Recommend: switch before D1 so the
emitter targets one format.**

**★ GATE D1 VERDICT (2026-08-12, projected on a 4-leg-validated model): FAILS as spec'd,
FIXABLE in the spec.** D1 node (verify one fixture wrap, blake3 legs, 110q) = 124 GiB
(1.3× over); real-wrap inner 227 GiB; D2 2-proof node 248-454 GiB. blake3 buys 5.2× vs
keccak here — but non-uniformly: Merkle parents 14.7×, **leaf absorption only 1.73×**
(LFML takes 2 felts/compression vs keccak's 17/permutation), and leaf absorption is 69.8%
of the node bill. **The dominant lever is the LFML leaf RATE — ×2 makes the D1 fixture
node FIT (81 GiB), ×4 → 59 GiB.** Spec census (COMMIT.md §1.5) refines it two ways:
(1) **LFM_HASH itself dominates the tower's leaf bill at 57%** (3,457 cols under Blake3-7r,
2.3× KECCAK_RND) — the tower spends most of its budget re-absorbing the hash chip's own
trace; (2) the missing ×2 is located precisely: the LFMC fold costs one compression per
4 felts because the socket pins the chaining value to IV. **D7 SUPERSEDED → RATE=4 ADOPTED (COMMIT.md board 85/85; the intermediate RATE=5 draft
was found UNBUILDABLE by the chip read — hash rows read whole 4-felt CELLS
(`instr.rs:99-110` num_input_cells gates the LFM_HASH bus; `word.rs:15`), so felts/row
must be a multiple of 4).** RATE=4 = accumulator cell + one felt cell in ONE compression
(13 of 16 message words), landing on the EXISTING 2-cells-in/1-out bus arity — the frozen
bus shape does not move. ⚠ REFUTE PASS RESULTS (2 refuted, 1 refuted-as-stated, 3 confirmed): the "multiple of 4"
argument is a NON-SEQUITUR (cell receives bind all four felts to memory; unused felts are
sound) — the real constraint is the compile-time lane map, and re-packing via
Pack/Unpack makes **RATE=5 buildable after all (~19% cheaper on an UNPRICED sketch)**;
"the frozen bus shape doesn't move" is also wrong — arity stays but the receive
MULTIPLICITY moves (Sum3 → 4-way selector) and `num_input_cells(Leaf)=2` panics
`emit_unread_input_pins` as written. CONFIRMED: the +16-col arithmetic (exact), the
per-lane-range gate hazard at `blake3_socket.rs:1304` (acc lanes constrained, felt-half
lanes not), and the two-chips width reconciliation (socket arm 2,964 main @6r is what
the tower pays). **★ Ship-breaking hazard found: a SILENT release-mode constraint-index
collision** (lane identities 6..17 overlap unused-output pins at 14+; `EmitTracker`
asserts only under debug_assertions; constraint COUNT unchanged so every count-based
guard is blind — lanes 8-11 would lose their identities). **Gate D1 ≈81 GiB / ~13%
margin STANDS at RATE=4** (working default). **NEW DECISION D9 (Mauro): RATE=4
(fully priced) vs RATE=5 (block ceiling, ~19% sketch, unpriced LfmMem/padding/re-pack
costs) — an optimization decision, not a fit decision.** D8 (sequencing,
Mauro): fold the RATE=4 re-bless INTO D0's re-bless pass (zero marginal cost) vs a
second re-bless later. Spec REQUIRES blake3-6round ON for tower builds. Build traps: `blake3-6round` OFF by default (+16% if forgotten); the BLAKE3
chip is the machine's widest table under D0.

### D1. LFM-proof-verifier emitter (effort L — the campaign's center of mass)
Same emitter machinery as the epoch verifier, pointed at an LFM proof: 14 fixed tables,
known log-heights, wrap options fixed → the program shape is static per (K, options).
Census first (closed form), then emit, then prove. Prototype input: the FIXTURE wrap's
proof — exists today, no Track-1 dependency.
- **Gate D1:** census says the 1-proof verifier fits comfortably (expected: yes — 14 tables
  vs ~25-31, blake3 legs vs keccak).

### D2. Aggregate 2→1 (effort M)
One program verifying TWO LFM proofs + consistency of their published words.
- **Gate D2:** wrap-of-two-fixture-wraps proves + verifies, tamper controls reject
  (both falsification directions, honest-path control per house rule).

### D3. Binary tree (effort S code / compute)
N base wraps → ⌈log2 N⌉ layers → one proof. With N ≤ 36 that is ≤ 6 layers; per-layer
cost is the D1 census number × 2.
- **Gate D3 = THE OBJECTIVE:** one LFM proof attesting block 25368371, with the boundary
  publics chaining genesis→final state.

## E. Small final proof (deferred, decide after D3)
A high-blowup/low-query wrap of the last aggregate (or an outer SNARK later). The wrap's
own options at blowup 8 multiply ITS trace memory ×4 — needs its own census. Not on the
critical path: D3's single STARK already IS "the block, compressed".

---

## Cross-cutting

- **GPU:** the −57% threshold env var is operational on every wrap; the permanent
  admission-token gate (BOX-RESULTS.md Stage-3 design) lands as its own reviewed PR.
- **Fit levers if a census gate fails:** smaller epochs (Track 1 only), inner-hash switch
  (4× on base-layer cells), `max_rows`/chunk-cap tuning (parked memory: max-rows-should-be-
  tunable), lever-2 D2H skip (host-RAM relief). Escalate hash decisions to Mauro; they gate
  batching/design choices per the July campaign.
- **House rules in force:** census before prove; ABBA for any perf claim; adversarial
  debate on emitter/soundness diffs; honest-path controls beside every falsification;
  checkpoint measurements off rented boxes as produced; no artifacts in the PR diff.
- **Known trap:** the fibonacci fixture ELF drift (BOX-RESULTS.md) — pin or fix before it
  bites another box; Track-1 work stops depending on the fixture at Gate B anyway.

## Order of operations (first two weeks)

1. A census sweep (box is warm now) — days 1-2.
2. D0 decision + D1 census — days 1-3, parallel.
3. B real-epoch feeding + Gate B first real wrap — days 3-7.
4. D1 emitter on fixture wraps — week 2+.
5. C scale-out whenever B lands; D2/D3 when D1 lands.

Single biggest unknown: Gate A / Gate D1 census numbers. Both are free to compute and
both are scheduled first — the plan self-corrects on real numbers before any large build.
