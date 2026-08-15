# MMCS-PLAN — batched commitments for the Lambda VM recursion campaign

**Status: SCOPING, implementation-ready. Read-only analysis; no code touched.**

Mauro green-lit batched-MMCS work ("we were delaying it until we had evidence we
needed it but we were expecting to do batched mmcs"). This document supplies the
numbers, the rebase-vs-reimplement verdict, the design, and the sequencing
against S3 and P-a.

---

## 0. Verdict — read this before scheduling anything

**1. Batching projects as the largest single lever in the campaign, larger than
the inner-hash switch.** At the real 2^21/blowup2 point it takes the wrap's leg
cost from 5,434 to 1,264 hash invocations per query — **4.30×**, against P-a's
measured 4.06×. At 2^23 it is **9.06×**. It is the only lever whose value
*grows* with epoch size, and it composes with P-a (combined 11.1× at 2^21,
20.8× at 2^23). DERIVED from the calibrated model; the model's validation is
§1.0 and its reconciliation against the one relevant measurement is §1.3.

**2. The bill is not where the brief assumed, and that changes the design.**
Today's wrap spends **54–66% of its leg permutations on FRI** and 24–25% on
Merkle walks; **leaf absorption is only 9–22%**. Batching the *trees* alone
(MMCS-only) buys 1.28–1.33×. Batching the *FRI* alone buys 1.96–2.80×. The
MMCS is the smaller half of the win at the inner layer — do both, and if
anything sequence FRI first.

**3. After batching, epoch size stops mattering for the wrap's memory.** All
three geometries land within 121–135 GiB (Part-2-corrected baselines, P-a
composed). The per-table O(N_tables) terms are what made 2^23 six times more
expensive than 2^20 in Gate A's sweep; remove them and the residual is leaf
absorption, which is set by table *widths* and barely moves. **Strategic
consequence: the campaign can use LARGE
epochs — fewer wraps for the tower to aggregate — at no memory cost.** That
reverses PLAN.md's "smaller epochs shrink each wrap but grow the total" trade.

**4. In the TOWER the answer is the opposite and much weaker: batching buys
1.31×** (D1 fixture node 122 → 93 GiB at the D0 blake3-socket width). Leaf
absorption is 72% of a tower node and batching does not touch payload. The
tower's lever remains the LFML **RATE** (D9), and batching *magnifies* D9's
value rather than substituting for it.

**5. #768 is bigger than the campaign's notes said, and both facts matter.** It
is not "batched FRI with a digests-only MMCS" — it is a complete wired
implementation: `fri/mmcs.rs` (+1,015, 7 tests), `fri/batched.rs` (+499),
prover integration (+2,053), soundness tests (+237). **Its MMCS layout is
line-for-line the construction this model prices**, arrived at independently —
the strongest structural validation available (§1.3). But it is 25 commits
behind, `CONFLICTING`, and predates StarkHash (879bdc0f), #877's rewrite of the
very R1 loop it changes, #863/#875/#914 and #909. **Verdict: port
`fri/mmcs.rs` + `fri/batched.rs` + the soundness tests; reimplement the
integration on StarkHash.** §2.

**5b. ⚠ ONE NUMBER DOES NOT RECONCILE AND IT GATES QUOTING §1.** The −57% keccak
figure predicts −76.7% under this model with #768's MMCS wired. And that figure
is **not** the PR's CI number — ✓ VERIFIED the only CI-posted result is
**+3.61% cycles / −573 keccak at ONE query** at an intermediate commit; −57%
comes from Mauro's own sims in `pr768-batched-fri-state.md`. So step one is
pinning which measurement is being compared, not re-measuring. **Until then,
treat §1.1's ratios as upper bounds on end-to-end reduction.** The leg
arithmetic is exact and validated four ways, and for the *wrap* the legs are a
MEASURED 99.6% of permutations, so the wrap-side numbers stand. Item M-11, S.

**5c. Two of my own claims were falsified by reading the branch, and both are
recorded in place rather than quietly fixed** (§2.1 correction box, §2.4): I had
`StarkHash` on main when it lives only on `blake3-real-hash`, and I called
#768's terminal-poly gap "not reproduced" after grepping the non-batched
verifier instead of `batched.rs`, where it plainly is. The second one has a
design consequence worth keeping: **#768's width binding is implemented and
tested; the residual gap is that the transcript absorbs heights but not widths**
(`batched.rs:196`), which §3.4's addendum now folds in as requirement M3.

**6. The July caveat "hash choice gates the batching decision" is retired.**
Batching wins by 3.6–9.1× under keccak and 2.4–5.1× under blake3. The hash
changes the *size* of the win, never its sign, and never the design. §4.4.

---

## 1. PROJECTION

### 1.0 Method, and why the numbers are trustworthy

The per-query cost function is the campaign's own closed form, re-derived from
source and then validated against measurement.

✓ VERIFIED the closed form in the code — `epoch_verify.rs:552-559`:

```
per_query = leaf_permutations(shape)                    // Σ over groups
          + groups * shape.sub.merkle_depth             // one parent per level per group
          + shape.fri.permutations_per_query()
```

with `leaf_permutations` = `Σ_groups num_blocks(leaf_bytes)` (`:413-419`),
`leaf_bytes = ROWS_PER_LEAF · num_columns · (24 if ext else 8)`
(`sub_proof.rs:88-90`), `merkle_depth = log2_lde − 1` (`sub_proof.rs:160-166`),
and `permutations_per_query = num_committed + path_steps_per_query` with
`layer_path_len(i) = n − i − 2` (`fri.rs:133-144, :146-155`).

✓ MEASURED validation — the model reproduces the census harness's own
`query_permutations` **exactly, to the unit**, on all four real-block points:

| point | sub-proofs | model | measured |
|---|---|---|---|
| 2^20/blowup2/219q | 28 | 4,185 /query | 4,185 |
| 2^20/blowup4/110q | 28 | 4,434 /query | 4,434 |
| 2^21/blowup2/219q | 32 | 5,434 /query | 5,434 |
| 2^23/blowup2/219q | 64 | 13,196 /query | 13,196 |

Leg shapes are the MEASURED dumps in
`~/workspace/lambda_vm_bench_cache/lfm_census_2026-08-12/census_logs/ethrex_e2*_skip.log`.
Tooling: `mmcs_project.py` / `mmcs2.py` / `mmcs3.py`, parked beside `project.py`
and `tower.py` in the same directory. Run `python3 mmcs2.py` to reproduce every
number in this section.

**The batched cost function**, same primitives, Plonky3 MMCS semantics
(mixed-height, tallest matrix sets the depth, shorter matrices injected at the
level whose subtree height matches):

```
digest = H(rows of the tallest matrices)              # level 0
for l in 1..=D:
    digest = compress(digest, sibling)
    if matrices inject at level l:
        digest = compress(digest, H(their rows))
```

The total absorbed payload over the path equals `Σ_matrices row_pair_bytes` —
**batching does not reduce the payload, only the framing and the walk.** That is
the brief's premise and the model honours it: at 2^21 the leaf term moves 967 →
932 (−3.6%), which is only the saved per-group padding block.

**One MMCS per commitment ROUND, not one overall.** Fiat–Shamir requires every
main root in the transcript before the shared LogUp challenge — ✓ VERIFIED
`prover.rs:3213-3238` ("All main trace commitments must be in the transcript
before sampling LogUp challenges … the one ordering Fiat-Shamir requires"), with
the verifier mirror absorbing the same roots at `verifier.rs:1288-1316`
immediately before "Round 1, Phase B: Sample shared LogUp challenges"
(`:1319-1322`). So main / aux / composition-parts cannot share a tree.
Preprocessed is committed at setup. Four trees, not one. That is the "~1-3 trees"
of the brief, made precise.

All projections below are **DERIVED-from-calibrated-model** unless a cell is
marked MEASURED.

### 1.1 (a) BATCHED INNER — what the WRAP pays

Per-query hash invocations in the wrap's legs, and the four corners of the
design space:

**keccak inner (today's RV64 commitment hash)**

| point | trees | today | FRI-only | MMCS-only | BOTH | BOTH ratio |
|---|---|---|---|---|---|---|
| 2^20/blowup4 | 88 → 4 | 4,434 | 2,236 | 3,426 | **1,228** | **3.61×** |
| 2^20/blowup2 | 88 → 4 | 4,185 | 2,135 | 3,261 | **1,211** | **3.46×** |
| 2^21/blowup2 | 100 → 4 | 5,434 | 2,517 | 4,181 | **1,264** | **4.30×** |
| 2^23/blowup2 | 196 → 4 | 13,196 | 4,713 | 9,940 | **1,457** | **9.06×** |

**blake3 inner (after P-a)**

| point | today | FRI-only | MMCS-only | BOTH | BOTH ratio |
|---|---|---|---|---|---|
| 2^20/blowup4 | 5,530 | 3,332 | 4,534 | **2,336** | **2.37×** |
| 2^21/blowup2 | 6,556 | 3,639 | 5,322 | **2,405** | **2.73×** |
| 2^23/blowup2 | 14,529 | 6,046 | 11,321 | **2,838** | **5.12×** |

**The split, which is the finding.** Where the per-query bill goes, today vs
batched (keccak inner):

| point | today leaf / merkle / FRI | batched leaf / merkle / FRI |
|---|---|---|
| 2^20/blowup4 | 21.1% / 24.6% / **54.3%** | 73.9% / 9.2% / 16.9% |
| 2^21/blowup2 | 17.8% / 24.5% / **57.7%** | 73.7% / 9.1% / 17.2% |
| 2^23/blowup2 | 9.1% / 24.9% / **65.9%** | 77.2% / 7.9% / 14.9% |

Per-table FRI dominates because every one of 28–64 sub-proofs runs its own FRI
instance down to `fri_final_poly_log_degree = 7`, and a deep table's FRI is
expensive: leg 31 of the 2^21 epoch (2^22 rows, 14 layers) costs 288
permutations per query of which **217 are FRI** — the census called it "almost
entirely in path steps" (CENSUS Part 1 §3) and this decomposes that remark.
Batched FRI replaces `Σ_t fri_t` with one instance over the largest domain:
3,134 → 217 per query at 2^21.

The walks collapse as promised (1,333 → 115 at 2^21, a 4-tree walk of depth 22
plus injection compressions). The leaf does not (967 → 932). **After batching,
leaf absorption becomes 74–77% of the wrap's bill** — i.e. batching moves the
inner layer into the same regime the tower is already in, where the only
remaining lever is the leaf rate.

**Peak projection, full census** (leg permutations + MEASURED spine → KECCAK_RND
chunks → cells → 33.7 B/cell). Non-`KECCAK_RND` chips are held constant, which
is conservative (their instruction counts shrink too):

| point | variant | leg perms | chunks N | cells | projected peak |
|---|---|---|---|---|---|
| 2^20/blowup4/110q | today | 487,740 | 23 | 35.7B | 1,122 GiB |
| | #768 FRI-only | 245,960 | 12 | 18.3B | 573 GiB |
| | **BOTH** | **135,080** | **7** | **10.3B** | **324 GiB** |
| 2^21/blowup2/219q | today | 1,190,046 | 55 | 87.3B | 2,742 GiB |
| | #768 FRI-only | 551,223 | 26 | 40.5B | 1,271 GiB |
| | **BOTH** | **276,816** | **13** | **20.6B** | **648 GiB** |
| 2^23/blowup2/219q | today | 2,889,924 | 133 | 211.1B | 6,630 GiB |
| | #768 FRI-only | 1,032,147 | 48 | 76.2B | 2,393 GiB |
| | **BOTH** | **319,083** | **15** | **23.8B** | **748 GiB** |

Spine permutations are MEASURED (`*_spine.log`): 3,258 / 4,395 / 8,417. They are
0.3–0.7% of the total and are held constant; batching in fact shrinks them
slightly (4 roots to absorb instead of 100).

**Chunk count is the number that matters for S3.** N = 55 → 13 at 2^21, N = 133
→ 15 at 2^23. The S3 residency model is `17.37·N + 30.2·k GiB` (CENSUS Part 2
§1, calibrated to a measured 13.4 GiB/chunk marginal, Part 3 §8) — batching
attacks the N term directly, which is exactly the term S3's Phase C exists to
flatten. §4.2 works the interaction.

### 1.2 Composition with P-a, and the fit

Batching factor is DERIVED here; the hash factor is the campaign's **MEASURED**
hash matrix (epoch-verify 11.17B cells keccak → 2.75B blake3-6r = 4.06×). They
are applied to both published baselines — Part 1's `33.7 B/cell` projection and
Part 2 §1's aux-corrected numbers, which the census itself calls upper bounds:

| point | baseline (P1 / P2-corrected) | + batching | + P-a only | **+ BOTH** |
|---|---|---|---|---|
| 2^20/blowup4/110q | 1,199 / 1,300 | 332 / 360 | 295 / 320 | **125 / 135** |
| 2^21/blowup2/219q | 2,929 / 1,337 | 681 / 311 | 721 / 329 | **265 / 121** |
| 2^23/blowup2/219q | 7,242 / 2,692 | 800 / 297 | 1,784 / 663 | **348 / 130** |

Two things to read off this table:

- **The combined lever is 9.6× (2^20) to 20.8× (2^23)**, against Gate A's
  required 13–29×. It does not close the gate on its own at every point, but it
  is the first lever that gets within a factor of ~1.3 of the 93 GiB box and
  *inside* the 124 GiB rigs at two of three points, before S3 contributes
  anything.
- **Epoch size becomes nearly free.** 121 / 130 / 135 GiB across 2^20 → 2^23.
  PLAN.md's framing — "the epoch-size lever is weak, 2^23 → 2^20 buys only 3.2×,
  and it multiplies the number of wraps the tower must aggregate" — is
  *reversed* by batching: the lever's remaining value is ~1.1×, so the campaign
  should take the LARGEST epoch that proves, minimising N wraps and therefore
  tower layers. That is a scheduling decision worth surfacing to Mauro
  independently of when MMCS lands.

⚠ These are wrap **work** numbers. Whether the work fits in RAM is the S3
residency question, which is orthogonal and multiplies (§4.2).

### 1.3 #768 — what it validates, and one number that does NOT reconcile

**★ The construction is independently confirmed.** #768's `fri/mmcs.rs` is a
mixed-height row-pair MMCS, and its documented layout is the *same* construction
this model prices, arrived at independently. ✓ VERIFIED, quoting its module doc
(`crypto/stark/src/fri/mmcs.rs:1-56` on `origin/feat/batched-fri-per-epoch`):

> *"A matrix of `log_height h` is injected at layer index `i = h_max - h` …
> Base layer node `k`: `layer0[k] = H( CONCAT_{m : h_m == h_max} (row_m(2k) ||
> row_m(2k+1)) )` … Climb: `parent = C(layer_i[2j], layer_i[2j+1])`. If any
> matrix has `h_m == inject_h`, then `layer_{i+1}[j] = C( parent, H( CONCAT_{m :
> h_m == inject_h} (row_m(2j) || row_m(2j+1)) ) )` … For query `iota`, matrix
> `m` is opened at leaf `k_m = iota >> (h_max - h_m)`."*

That is `batched_tree_cost` line for line — same injection level, same
concatenation, same `C(parent, H(injected))` two-compression step, same
index truncation. **The model's semantics are not a guess.** It also
retroactively justifies §3.4's recommendation to express injection as one extra
compression rather than a new step type: #768 reached the same shape.

**⚠ But the magnitude does NOT reconcile, and I am recording that as open rather
than explaining it away.** Memory `recursive-verifier-batched-fri` records
**−36.9% cycles / −57% keccak** at real query counts.

⚠ **First, that number's provenance, which is weaker than it looked.** ✓ VERIFIED
the only CI-posted figure on the PR is **+3.61% cycles / −573 keccak calls at ONE
query**, dated 2026-07-17 at intermediate commit `0880cff6` — not at the head,
and not at real query counts. The −57% / −36.9% figures come from **Mauro's own
sims recorded in `pr768-batched-fri-state.md`**, not from the PR. So the target
this model is being reconciled against is itself a simulation whose geometry and
denominator are not stated in the PR. **Pinning which number is being compared
is the first half of M-11**, and it may dissolve the discrepancy without any
re-measurement.

I initially inferred from `pr768-memfix-mmcs-digest-only`
("digests-only MMCS by design") that #768 batched FRI only, which would put the
measurement at the FRI-only corner — where the model predicts −53.7% at 2^21,
within 3.3 points. **That inference is FALSIFIED by the branch.** ✓ VERIFIED all
three round-MMCS instances are built, absorbed and opened:
`prover.rs:612-614` (`main_mmcs` / `aux_mmcs` / `comp_mmcs`), `:2666-2667`
(built, root appended to transcript), `:4413-4415` (`open_batch` per query).
The "not yet wired into the prover/verifier" note at `mmcs.rs:10-12` is **stale**
— it was written at Task 1 and the branch went on to wire it.

So the measurement should sit at the BOTH corner, where the model predicts
**−76.7%** at 2^21, not −57%.

| point | model FRI-only | model BOTH | measured (#768) |
|---|---|---|---|
| 2^20/blowup4/110q | −49.6% | −72.3% | |
| 2^21/blowup2/219q | −53.7% | **−76.7%** | **−57%** |
| 2^23/blowup2/219q | −64.3% | −89.0% | |

**Three candidate explanations, none yet checked:**

1. **Different denominator.** The #768 number is the *whole guest verifier's*
   keccak count; this model prices the *legs* only. The guest also absorbs the
   transcript, samples challenges, and checks grinding, and none of that
   shrinks. For BOTH to read −57% the fixed remainder would have to be ~26% of
   the guest's keccak. In the wrap the spine is 0.4% — but the RV64 guest is a
   different program, so this is plausible and is the explanation I would bet on.
2. **Different geometry.** The ratio is strongly geometry-dependent (−71% to
   −89% across the sweep). A measurement on a small or toy epoch would land
   lower.
3. **Measured mid-branch**, after the FRI work and before the MMCS wiring —
   45 commits, and the stale module note shows the branch was built in tasks.

**This must be resolved before any projection in §1.1–1.2 is quoted as a
schedule input.** It is cheap to resolve: re-read the #768 bench record for its
denominator and geometry, or re-run it. Until then, treat §1.1's ratios as
**upper bounds on the achievable end-to-end reduction** — the *leg* arithmetic
is exact and validated four ways, but the fraction of the wrap that legs
constitute is measured (99.6%) only for the wrap, not for the guest verifier
#768 measured.

**What survives regardless:** the leg model reproduces the measured census
exactly at four points (§1.0), the construction matches a real implementation
(above), and the *wrap's* leg share is measured at 99.6%, so for **the wrap** —
which is what this campaign's memory problem is about — the ratios stand.

### 1.4 (b) BATCHED LFM MACHINE — the TOWER node

Same model, `native = true`: the LFM proof's own commitments under the
COMMIT.md §1.2 LFML/LFMC construction at the adopted **RATE = 4**
(`compressions = ceil(2·num_cols·kind / 4)`, §1.4), parents 1 compression.

**Baseline reconciliation first.** `tower.py` as published prices the OLD rate
(`2·ceil(felts/4)`) and gives D1/fixture/110q = 124 GiB; PLAN.md's RATE=4
headline of ≈81 GiB is that number with the leaf term halved. This model at
RATE=4 with the same `LFM_HASH=(28,3)` width gives **78 GiB** — 4% from the
published 81, and the per-query split (leaf 71.7% / merkle 10.9% / FRI 17.4%)
matches CENSUS Part 2 §3's independently derived 69.8% / 10.7% / 19.5%. ✓ The
tower model is anchored — and note this is a *second* model agreeing with the
first, not a measurement: the tower has never been censused on hardware, unlike
the inner layer where §1.0's four points are real.

| inner | LFM_HASH width | q | today | FRI-only | MMCS-only | **BOTH** | ratio |
|---|---|---|---|---|---|---|---|
| FIXTURE wrap | (28,3) | 110 | 78 GiB | 60 | 66 | **49 GiB** | 1.58× |
| FIXTURE wrap | (2964,630) D0 | 110 | 122 GiB | 104 | 110 | **93 GiB** | 1.31× |
| REAL 2^21 wrap | (28,3) | 110 | 116 GiB | 91 | 103 | **78 GiB** | 1.49× |
| REAL 2^21 wrap | (2964,630) D0 | 110 | 160 GiB | 135 | 147 | **122 GiB** | 1.31× |
| FIXTURE wrap | (2964,630) D0 | 219 | 242 GiB | 208 | 220 | **185 GiB** | 1.31× |

Per-query split, D1 fixture at the D0 width: today leaf 4,773 (72%) / merkle 727
(11%) / FRI 1,156 (17%) → batched leaf 4,769 (94%) / merkle 120 / FRI 208.

**Why the tower gains so much less than the inner layer.** A tower node verifies
14–15 sub-proofs, not 28–64, and those chips are *wide* (`LFM_HASH` 2,964 main
at 6r, `KECCAK_RND` 1,480, `LFM_KECCAK` 736) rather than deep. Leaf absorption
is already 72% of the bill and batching leaves it untouched. **MMCS does not
rescue the tower; the RATE does.**

**Batching magnifies D9 rather than substituting for it** (§ mmcs3.py G):

| | per-table trees | batched MMCS | batching buys |
|---|---|---|---|
| RATE = 4 (adopted) | 122 GiB | 93 GiB | 1.31× |
| RATE = 5 (D9 open) | 105 GiB | 76 GiB | 1.38× |
| RATE = 8 (hypothetical) | 77 GiB | 50 GiB | 1.55× |

After batching, leaf is 94% of the node, so the rate scales 94% of the cost
linearly instead of 72%. **If D9 goes to Mauro, this table belongs in the
question**: RATE=5 is worth 17 GiB unbatched and 17 GiB batched, but it is the
*only* remaining lever once MMCS lands.

**★ A discrepancy found in CENSUS Part 2 §3, flagged for its owner.** That
section states the D0 second-order feedback (the machine's own hash chip
becoming BLAKE3 and therefore the widest table) is **+19%**, "D1/real/219q moves
381 → 452 GiB". Re-running the published `tower.py` with
`WIDTH['LFM_HASH'] = (3056, 630)` gives **559 GiB, i.e. +47%** — and this
model's independent D0-width variant shows +38% to +56% depending on the point.
? UNRESOLVED which input produced 452. It does not change any verdict here (the
+19% figure is the optimistic one, so the tower baseline is *worse* than
published and the case for every tower lever is stronger), but the tower numbers
circulating in the campaign should be re-derived before they gate a decision.

---

## 2. REBASE vs REIMPLEMENT

**Verdict: SALVAGE THE PRIMITIVES, REIMPLEMENT THE INTEGRATION.** Do not rebase
the branch. But `fri/mmcs.rs` is far more reusable than the campaign's notes
suggested and should be the starting point, not a reference.

### 2.0 What is actually on the branch — ✓ VERIFIED

`gh pr view 768`: head `feat/batched-fri-per-epoch`, base `main`, **state OPEN,
mergeable CONFLICTING**, opened 2026-07-02, **+5,097 / −1,099 across 25 files**.

Merge base `3ea4f916` (2026-07-17, "verify continuation proofs in place via
rkyv (#845)"). **25 commits behind main; 45 commits on the branch.**

`git diff --stat 3ea4f916..origin/feat/batched-fri-per-epoch`, the load-bearing
rows:

| file | Δ | what |
|---|---|---|
| `crypto/stark/src/fri/mmcs.rs` | **+1,015 NEW** | `MixedMmcs`, `BorrowedMatrix`, `MixedOpening` — mixed-height row-pair MMCS, 7 tests, layout documented as "the single source of truth" |
| `crypto/stark/src/fri/batched.rs` | **+499 NEW** | `combine_by_height` — mixes DEEP codewords by FRI height with `alpha^i`, the batched-FRI core |
| `crypto/stark/src/prover.rs` | **+2,053** | the integration: `main_mmcs`/`aux_mmcs`/`comp_mmcs` (`:612-614`), built + absorbed (`:2666-2667`), `open_batch` per query (`:4413-4415`) |
| `crypto/stark/src/verifier.rs` | +764 | batched verify, `fri::terminal::FriFoldLayout`, terminal-codeword reconstruction (`:386-400, :436-474`) |
| `crypto/stark/src/proof/stark.rs` | +96 | **wire format changes** — `MixedOpening` enters the proof type |
| `prover/src/continuation.rs` | −810 net | rewritten |
| `crypto/stark/src/tests/bus_tests/batched_soundness_tests.rs` | **+237 NEW** | soundness oracles for the batched path |

So the campaign's shorthand was wrong twice: it is **not** "batched FRI with a
digests-only MMCS", it is a **complete, wired, tested batched-commitment
implementation** — three round-MMCS instances plus batched FRI, exactly the
design this document scopes.

### 2.1 Why it still cannot be rebased

The merge base is 2026-07-17 and **every structural change this design depends
on landed after it** — ✓ VERIFIED by `git log 3ea4f916..origin/main`:

| main-side commit | why it collides |
|---|---|
| `7644043b` **#877 per-table scheduler with VRAM admission for `multi_prove`** | rewrote the R1 commit loop into `run_admitted` — the exact region the branch changes most. It also **deleted `plan_table_chunks`, which #768 still calls** (`prover.rs:4683` on the branch) |
| `5749a956` #875 device-resident rounds 2-4 + fused NTT | rewrote the fused per-table task |
| `d83b4d9e` #863 halve GPU continuation proving | commitment production on device |
| `6949ceb9` **#909 pin each trace-opening column width to the AIR, not just their sum** | the opening-width soundness fix — lands directly on the batched leaf's binding surface |
| `d898a423` #914 VRAM pressure / R2 corruption race | same prover regions |

Plus S3's `ResidencyMode` rewrite of the same R1 loop, **in flight this week**.

> ### ⚠ CORRECTION — `StarkHash` is NOT on main
>
> An earlier revision of this section listed `879bdc0f` (StarkHash, D0 step 2)
> as a main-side commit. **It is not.** ✓ VERIFIED three ways:
> `git grep -c StarkHash origin/main` → zero matches;
> `git branch -a --contains 879bdc0f` → **`origin/blake3-real-hash` only**; and
> it does not appear in the 25 commits of `git log 3ea4f916..origin/main`. The
> `config.rs:55-192` I read is the **campaign branch's** file, not main's.
>
> **Consequence, and it is a scheduling fact not a nitpick:** `StarkHash` is a
> D0 artifact living on `blake3-real-hash` alongside P-a and S3. So "reimplement
> on `StarkHash`" means **building on the campaign branch**, and the eventual
> main merge is a separate, later problem that this plan does not cost. It also
> means M-1 is not independent of D0 — it inherits D0's merge risk. Anyone
> scheduling M-1 against main will not find the trait.
>
> The rebase argument is unaffected: the branch predates `StarkHash` either way,
> and the four genuinely-on-main commits above are sufficient on their own.

⚠ Correction to the brief's list: **#823 (`a8648320`) and #826 (`18f3b8f2`)
predate the merge base** and are already in the branch's history; they are not
sources of conflict, and any note saying #768 "must rebase over #826+#823" is
stale. #863, #875, #877, #909 and #914 are the real collisions.

`gh` reports the PR as **draft, CONFLICTING, zero reviews, empty body**, against
a 25-commit gap that includes two rewrites of the file the branch changes most.
The branch was never rebased — 13 of its 45 commits are `Merge branch 'main'`.

**Two further reasons a merge is the wrong instrument**, both ✓ VERIFIED:

- **The batched lane has ZERO CUDA.** `feature = "cuda"` appears 0 times in
  `mmcs.rs`, 0 in `batched.rs`, and 0 across the branch's
  `batched_table_deep_codeword..batched_round_4` region. Meanwhile 4 of the 5
  main-side `prover.rs` commits since the merge base are GPU work
  (#863/#875/#877/#914, +1,913 lines in that one file). A merged batched path
  would be CPU-only on a prover whose recent history is entirely GPU.
- **A merge would silently delete main's #845 zero-copy verify layer.**
  `git merge-tree` conflicts in only two files, but in the resulting tree
  `EpochProofView`, `ContinuationProofView`, `verify_continuation_view`,
  `access_recursion_archive` and `verify_l2g_commitment_binding_view` all have
  count **zero** — dropped during an on-branch conflict resolution and never
  re-touched. The branch documents the debt itself
  (`prover/src/lib.rs`: *"TODO(batched-fri): port the view machinery to
  `BatchedMultiProof` to restore fully in-place verification"*), priced in the
  campaign notes at **≈ +136M guest cycles**. This is the most dangerous
  property of a rebase: it is a silent deletion that no conflict marker shows.

### 2.2 What to salvage, and it is a lot

The branch's value is concentrated in the two files that do **not** depend on
`multi_prove`'s structure:

- **`fri/mmcs.rs` (1,015 lines, 7 tests).** A standalone primitive taking
  matrices as `(row_major_bit_reversed_lde, log_height, width)`. Its only
  coupling to main's churn is the concrete keccak backend, which is precisely
  what `StarkHash::Mmcs<F>` (§2.3) parameterizes. **Port this file, do not
  rewrite it** — and keep its module doc, which is the clearest statement of the
  layout contract anywhere in the campaign and should become the normative
  reference §3.4's spec delta cites.
- **`fri/batched.rs`'s `combine_by_height` (499 lines).** Mixes codewords by
  height with `alpha^i` and returns one combined codeword per height. That is
  M-3's core and it is written.
- **`batched_soundness_tests.rs` (237 lines).** Oracles for a path that does not
  exist on main yet — the most expensive thing to write from scratch and the
  cheapest to port.

⚠ **One salvage caveat, and it is the streaming requirement from §3.3.**
`combine_by_height` takes `inputs: &[(Vec<FieldElement<E>>, usize)]` — *all*
codewords materialised at once. That is the `O(N)` shape §3.3 warns against. The
same question must be asked of `MixedMmcs::commit`'s matrix input. **Check both
before porting**; if they materialise, the port is the right moment to make them
streaming, since nothing downstream on main depends on the eager signature yet.

### 2.3 `StarkHash` as the home — with one caveat

`StarkHash` is the right home in shape: it is already the place where "how a
leaf becomes a digest" is named, and it already documents the
Batched/Pair split as "separate families because they hash different leaf
shapes, not because they are different hashes" (`config.rs:78-82`). An MMCS is a
third leaf shape.

⚠ **But it does not fit `Batched<F>` as declared, and the reason is worth
stating before someone tries.** `Batched<F>`'s `Data = Vec<FieldElement<F>>` is
a single-field, single-matrix leaf. A mixed-height MMCS leaf is not a `Vec` at
all — matrices are injected at *interior* levels, so the tree builder must know
the injection schedule; `IsStreamingLeafBackend` has no vocabulary for it.

Two saving graces, both ✓ VERIFIED from the round structure:

- **Per-round batching is field-homogeneous.** Main is all base, aux all ext,
  parts all ext. So the *field* generic is not the problem — a `Mmcs<F>` member
  works. Had the design batched across rounds it would have needed a
  heterogeneous leaf, and it cannot batch across rounds anyway (Fiat-Shamir,
  §1.0).
- The digest type is already fixed: `Node` is "deliberately **not** an
  associated type: it is `Commitment` for every implementation"
  (`config.rs:88-91`), which is what keeps `StarkProof`'s rkyv derives
  byte-identical. An MMCS root is one `Commitment`, so the wire format is
  undisturbed by the *root*; what changes is that there is one root per round
  instead of one per table (§3.2).

**Recommended shape:** add a third member `type Mmcs<F>` to `StarkHash` whose
implementation is built on the *same* `hash_bytes` as `Batched<F>` — the same
discipline PA-PLAN §1.4 prescribes for the Pair/Batched invariant ("do not prove
that two independently-written encodings coincide; make them one function"), and
extend the existing invariant test (`tests/commitment_tests.rs:110-121`) with an
MMCS arm asserting that a single-matrix MMCS equals `Batched` on the same data.
That arm is the whole backward-compatibility argument in one test.

### 2.4 The two recorded gaps — status after reading the branch

- **The terminal-polynomial early-stop gap — ✓ CONFIRMED PRESENT.** (My own
  first pass called this "not reproduced"; that was wrong, and the mistake is
  instructive: I grepped `verifier.rs`, which handles the **non-batched** path
  correctly, and never opened `batched.rs`.) The gap is in
  `fri/batched.rs::batched_commit_phase` (`:90`): `num_committed_layers =
  h_max.saturating_sub(1)` (`:129`), folding all the way to a **scalar** with
  `transcript.append_field_element(&last_value)` (`:179`). The function takes no
  `final_poly_log_degree` argument, and `terminal|FriFoldLayout|final_poly`
  matches **one line** in the whole file. The non-batched
  `commit_phase_from_evaluations` on the *same branch* does it right, returning
  `final_poly_coeffs` from `terminal::coeffs_from_terminal_codeword`
  (`fri/mod.rs:138-147`). Two stale doc comments date the lane: `batched.rs:87-89`
  claims termination "mirrors `commit_phase_from_evaluations`" (it does not), and
  `:186-188` says "#729 is not on this branch" — but #729 (`b3f85b79`) **is** on
  it, merged via `1f3c0cec`, which is why `fri/terminal.rs` exists there at all.

  **Priced against this model: +2.5% to +3.7%** on the batched per-query cost
  (2^21: 1,264 → 1,300, ratio 4.30× → 4.18×; 2^23: 1,457 → 1,493, 9.06× →
  8.84×). ~9 extra committed layers per proof, matching the campaign note's own
  estimate of `k + blowup_log`. **So it is a real defect but a small one, and it
  is a bug not to inherit rather than a tax on the design** — every projection in
  §1 prices the *correct* construction with the early stop. The free acceptance
  check: assert `num_committed(batched) == num_committed(largest leg)` using
  `FriShape::effective_k = terminal_log − blowup_log` and `num_committed =
  total_folds − 1` (`fri.rs:97-116`), plus the closed-form equality
  `wrap_tests::the_census_fit_map_point` already asserts.
- **"Leaf-binding" — two different things were being conflated, and the
  distinction changes the design.** `git grep -i 'leaf.?bind'` over `origin/main`
  → zero hits; the campaign note's "leaf-binding fix required pre-production"
  attaches to **PR #857 (LogUp-GKR port)**, not #768.

  On #768 the width/leaf-boundary binding is **implemented and tested**:
  `verify_batch(root, iota, opening, heights, widths)` (`mmcs.rs:423-428`), with
  widths derived AIR-side and never from the proof (`verifier.rs:2099-2128`), and
  a negative test `batched_rejects_main_opening_width_mismatch`
  (`batched_soundness_tests.rs:122`). That is #909's invariant, already honoured.

  **★ The residual gap is the TRANSCRIPT half, and it is a genuine design
  requirement this plan must absorb** — flagged in-module at `mmcs.rs:78-81`
  verbatim: *"`widths` and `heights` should ALSO be bound into the Fiat-Shamir
  transcript by the consumer. Scope A's `absorb_height_histogram` currently binds
  heights only; extending it to `(height, width)` pairs is a Task 4 / verifier
  concern."* ✓ VERIFIED against `batched.rs:196`: the signature is
  `absorb_height_histogram<E,T>(transcript, heights: &[usize])` — heights only.
  See §3.4's addendum, which folds this into the spec delta.

**Also harvest:** `batched_soundness_tests.rs` (+237), and note PR #846 has since
landed "measure the verifier at real query counts, over real blocks" — that is
the instrument to settle §1.3's open denominator question with.

---

## 3. DESIGN — MMCS in the LFM machine

Scope note: §3.1–3.3 are the *machine's own* proof (application (b), the tower).
§3.5 is the emitter change, which is what application (a) needs and is the
larger of the two. They share the leaf construction in §3.4.

### 3.1 Registry — `roots: [Commitment; 14]` → one root plus shape

✓ VERIFIED today (`registry.rs:53-79`):

```rust
pub struct LfmRegistryEntry {
    kind, blowup_factor,
    roots: [Commitment; NUM_LFM_CHIPS],      // 14
    log_heights: [u8; NUM_LFM_CHIPS],
    keccak_rnd_chunks: usize,
    hasher: HasherKind,
    program_id: Commitment,
}
```

and `build_artifacts_with_hasher` (`:151-200`) commits eleven instruction column
groups with `commit_group` plus `keccak_rc::preprocessed_commitment` and
`bitwise::preprocessed_commitment` into slots 12/13, leaving slot 11
(`KECCAK_RND`, no preprocessed columns) as an all-zero sentinel, then derives
`program_id = lfm_program_id(&roots, &log_heights, keccak_rnd_chunks, hasher)`
(`:192`).

**The change.** The 14 preprocessed roots become ONE batched preprocessed root
over the same 13 non-sentinel matrices (mixed heights: `log_heights` in the
registry runs 2…20). Keep `log_heights` — it is no longer only a height record,
it is the **injection schedule**, and the verifier needs it to build the walk.

```rust
pub struct LfmRegistryEntry {
    kind, blowup_factor,
    prep_root: Commitment,                   // was [Commitment; 14]
    log_heights: [u8; NUM_LFM_CHIPS],        // now load-bearing: injection levels
    prep_widths: [u16; NUM_LFM_CHIPS],       // NEW: leaf shape per matrix (see §3.4)
    keccak_rnd_chunks, hasher, program_id,
}
```

Three consequences, each of which has to be written down rather than inherited:

1. **`program_id`'s preimage changes → re-bless.** `lfm_program_id` is fed the
   root array today. Feeding it `prep_root` + `log_heights` + `prep_widths`
   moves every digest. The regeneration path exists and is governed
   (`cargo run --bin compute_lfm_registry --release`, drift tests on every PR,
   "a drift failure is investigated, never re-blessed to silence the test",
   `registry.rs:1-13`). **Sequence this re-bless INTO D0's** — the same argument
   D8 makes for folding the RATE=4 re-bless in (PLAN.md §D1). Three separate
   re-blesses of the same digest in one campaign is three chances to bless a
   drift.
2. **The all-zero sentinel for `KECCAK_RND` stops being expressible** as a root
   and becomes an *absence* in the injection schedule. The soundness argument at
   `registry.rs:98-110` — the chip is program-independent in both directions, so
   binding nothing is sound, and what the entry pins is `keccak_rnd_chunks` — is
   unchanged in substance, but the mechanism moves from "a zero root" to "a
   matrix not in the batched tree". Write it that way; a reader who greps for
   the sentinel must land on the new statement.
3. **`prep_widths` is new registry data and is soundness-bearing.** Under the
   per-table scheme, a group's width is implied by its own root plus the AIR.
   Under an MMCS the widths determine how the leaf is parsed, so they must be
   program shape, pinned, and folded into `program_id` — never read off the
   proof. This is `verifier.rs:639`'s instruction ("do not re-derive it from the
   proof") applied one level up, and it is the same rule COMMIT.md §1.3 states
   for the header.

Effort: **M**.

### 3.2 `LfmArtifacts` / `verify_against` / `lfm_verify`

✓ VERIFIED `verify_against` takes `roots: &[Commitment; NUM_LFM_CHIPS]` and
hands it to `LfmAirs::new_with_hasher(roots, options, keccak_rnd_chunks, hasher)`
(`proof.rs:251-291`); `lfm_prove` does the same via
`prove_traces_with_hasher` (`:170-175`). Both signatures change to
`(prep_root, log_heights, prep_widths)`.

Two guards to preserve verbatim, because they are the shape of the soundness
argument and an MMCS refactor is exactly the kind of change that erodes them:

- The `keccak_rnd_chunks == 0` rejection and the
  `view.len() != num_lfm_airs(keccak_rnd_chunks)` length check
  (`proof.rs:262-268`) — under batching the *AIR set* is still per-chip, so both
  survive unchanged. Keep them; the batched root binds the preprocessed
  matrices, not the chip count.
- The exhaustive `const _: () = match stark::config::COMMITMENT_HASH { ... }`
  tripwire at `registry.rs:158-160`. PA-PLAN §4.2 already flags that this guard
  becomes a half-truth once a second `StarkHash` exists. **Adding an `Mmcs`
  member is a second reason to revisit it in the same pass** — the guard's job
  is "the list of places that have to be revisited before this crate can commit
  under two hashes" (`config.rs:55-62`), and "two leaf shapes" belongs on that
  list too.

Effort: **S** once §3.1 lands (mechanical signature threading).

### 3.3 The prover's Round-1 commit

✓ VERIFIED the seam. `multi_prove` commits each table independently under
`run_admitted` and then absorbs roots sequentially in index order —
`prover.rs:3240-3295`, with the comment stating the requirement exactly: "the
transcript only needs the roots absorbed in index order, done sequentially below
once every commit completed — the one ordering Fiat-Shamir requires before
sampling the shared challenges."

Under batching that loop produces **one** tree and absorbs **one** root.

**★★ The requirement that makes or breaks this: the batched tree build must be
STREAMING PER MATRIX.** This is the single most important implementation
constraint in the document, and getting it wrong silently undoes a large part of
the win.

✓ VERIFIED the property at risk — the `Lde` struct's own doc
(`prover.rs:265-274`): main LDEs are all-N-live because Round 1 is a phase-wide
barrier, but **aux is produced and consumed inside the same fused task, so at
most `table_parallelism()` of them coexist**. Batching the aux round introduces
a *new* phase barrier at aux-commit. A naive implementation — materialise every
table's aux LDE, then build one tree over all of them — converts the aux term
from `O(k)` to `O(N)`, which per CENSUS Part 2 §1's table is 18.15 GiB per
`KECCAK_RND` chunk moving from ×21 to ×N. **That would give back a large
fraction of what batching buys, in the same commit.**

The fix is available and is the same insight S3 rests on: retain the digest,
drop the buffer. Because COMMIT.md §1.2's absorption is a sequential **chain**,
a leaf can be accumulated matrix by matrix:

```
acc[leaf] = H0                                   // per-leaf accumulator, 16 B
for each matrix m, in commitment order:
    compute m's LDE  (one at a time, k-bounded exactly as today)
    for each leaf: acc[leaf] = absorb(acc[leaf], m's header ‖ m's rows)
    drop m's LDE
build the tree from acc[]                         // + injection levels
```

Retained state is `O(num_leaves × 16 B)` instead of `O(N × aux_cols × lde_size)`
— for a 2^19-row chunk that is ~4 MiB against 12.1 GiB. This is what "digests-only
MMCS by design" means in memory `pr768-memfix-mmcs-digest-only`, and it is why
that design note matters more than the code it describes.

**Write it into the acceptance test, not the prose.** The falsifiable check is
the same shape as S3's: prove the same statement with per-table and batched
commitments and assert the batched run's peak anon is not higher. If the
streaming build was missed, that test fails loudly instead of the campaign
discovering it in a census three weeks later.

The same treatment applies to the composition-parts round, and batched FRI needs
the analogous care: the combined codeword is a linear combination, so accumulate
`combined += α^i · quotient_i` one table at a time rather than materialising all
quotients. Both are `O(1)` in N if written that way and `O(N)` if not.

**What does move: the per-table transcript fork.** The fork
(`prover.rs:3361-3370`, `t.append_bytes(&(idx as u64).to_le_bytes())`) exists so
that "aux build, aux commit and rounds 2-4 run FUSED per table … tables never
wait on a phase barrier" (`:3315-3326`). Batching reimposes barriers at
aux-commit, parts-commit and FRI, because a batched root cannot be absorbed
until every contributing matrix exists. So the fork survives only for the
per-table constraint/OOD work; the commitment and FRI stages rejoin the shared
transcript. **This is a restructure of `multi_prove`'s phase architecture, not
only of its R1 loop** — it is why M-4 is L and not M, and it is the reason the
sequencing in §4.3 puts it after S3 Phase A+B rather than beside them.

**★ One verifier check that a batched preprocessed tree changes shape.**
✓ VERIFIED `verifier.rs:1288-1312`: for a preprocessed table the verifier
compares the proof's precomputed root against `air.precomputed_commitment()` —
"the critical soundness check - ensures prover used correct precomputed values" —
and absorbs BOTH the precomputed and the main root. Under a batched preprocessed
MMCS this becomes one comparison against the registry's `prep_root` instead of
one per table, and the per-table `precomputed_commitment()` accessor stops being
the thing that is checked. **That is a consolidation of a soundness check, which
is exactly the kind of change that quietly loses coverage**: the batched
comparison must still fail if *any* single table's preprocessed matrix is wrong,
which it does only if `prep_widths` and `log_heights` pin the parse (§3.1
item 3). Put a tamper control on it per matrix, not just on the tree.

⚠ **The protocol change this forces, stated plainly for the soundness review:**
today each sub-proof samples its own query indices from its own domain
(219 independent queries per table). Batched MMCS + batched FRI means **one
index per query, shared across all tables**, with shorter matrices opened at
`index >> (D − depth_t)`. That is the standard batched-FRI construction and it
is where the security parameters must be re-derived — not assumed to carry over.
It is the one part of this plan that is a protocol change rather than a
refactor, and per the house rule it goes to adversarial-debate review with
tamper controls in both directions plus an honest-path control.

Effort: **L** (this is the center of mass on the prover side).

### 3.4 ★ SPEC DELTA — the batched leaf, for Mauro's ratification

This section is written to be lifted into `commit-spec/COMMIT.md` as a new
subsection under §1. It states what changes in the RATE=4 construction and what
does not.

**What does not change.** The `LFML_row` function, the RATE, the tag, the
accumulator-in-the-message design, the chain-not-tree fold, `ROWS_PER_LEAF = 2`,
the node codec (§3.1), the strict decode (§3.2), the power-of-two leaf-count
assertion (§3.3). ✓ All of COMMIT.md §1.2's primitives are reused unchanged.

**What changes: a leaf covers many matrices, so one header no longer describes
it.** COMMIT.md §1.2 sets `H = [LEAF_MARK, num_cols, kind, ROWS_PER_LEAF]` —
one header, one `num_cols`, one `kind`. An MMCS leaf interleaves matrices of
different widths. A single header binding a single `num_cols` would bind
*nothing* about how the felt stream splits between matrices, which reopens
§1.1's hazard in a new dress: not "moving columns between trees absorbed at
different times" but **moving columns between matrices inside one absorption**.

**★ RECOMMENDED (option b): one header cell per MATRIX, at its injection
level.** The leaf at level ℓ absorbs, in matrix order:

```
for each matrix m injected at level l, in registry index order:
    acc = LFML_row(acc, [LEAF_MARK, m.num_cols, m.kind, m.matrix_index])
    for each chunk c of RATE felts of serialize(m.rows):
        acc = LFML_row(acc, c)
```

Why this one:

- **COMMIT.md §1.3's argument survives verbatim, per matrix.** "The header binds
  `num_cols` AND `kind`" and "the verifier must build the header from the AIR,
  never from the opening" are unchanged statements; they now hold once per
  matrix instead of once per leaf. The C2/C3/C4 executed collisions carry over
  as-is, and the `m=1`-padded / `m=2`-unpadded collision (C4) is if anything more
  necessary here, since adjacent matrices' padding meets inside one stream.
- **Matrix order is bound for free**, by the same property §1.3 already relies
  on: "the chain binds chunk order for free."
- `ROWS_PER_LEAF` leaves the header (it is a global constant, already bound once
  per proof) and `matrix_index` takes its slot — which is what closes the
  reordering question the multi-matrix leaf introduces. Field count is
  unchanged, so the header is still exactly one 4-felt cell and one compression.
- **It is cheap, and the cost is now measured rather than assumed:**

| context | batched cost/query | + headers | overhead |
|---|---|---|---|
| tower, D1 fixture | 5,097 | +55 | **+1.1%** |
| tower, D1 real-2^21 | 6,670 | +58 | **+0.9%** |
| inner 2^21, blake3 | 2,405 | +100 | +4.2% |
| inner 2^23, blake3 | 2,838 | +196 | +6.9% |

  In the tower — where this spec applies — it is a rounding error. At the inner
  layer (where the hash is byte-oriented, not LFML) the framing overhead is a
  keccak/blake3 padding block per matrix and the same 4–14% band applies; still
  far inside the 2.4–9.1× the batching buys.

**Option (a), considered and not recommended:** one header per leaf binding a
*shape digest* over the ordered `(matrix_index, num_cols, kind)` vector. Cheaper
(the digest is program shape, computed once, ~0 per query), but it introduces a
second commitment object whose preimage rules need their own C-tests, and it
makes the leaf's binding indirect at exactly the point §1.3 argues it must be
direct. Take (a) only if the inner layer's 4–14% turns out to bind, and then
only there.

**Injection structure and step typing.** The walk uses only the two existing
primitives:

```
digest = absorb(matrices at level 0)
for l in 1..=D:
    digest = LFMC_compress(digest, sibling)
    if any matrix injects at l:
        digest = LFMC_compress(digest, absorb(matrices at l))
```

No third step type and therefore **no new domain tag** — which matters, because
COMMIT.md §4.1.3 flags tag changes as a read-before-touching surface and the
crate anchor (C9) depends on the message staying a plain byte string. Injection
is expressed as one extra `LFMC_compress`, not as a compress-with-payload.

**What the tree-shape rules become.** §3.3's "assert the leaf count is a power of
two — do not pad" still holds and gets *stronger*: the batched tree's leaf count
is the tallest matrix's `lde_size / 2`, and every injected matrix's own leaf
count must divide it exactly, which is automatic since all are powers of two.
Assert `D − depth_m == log2(leaves_D / leaves_m)` per matrix; it is free on the
honest path and it is what stops a matrix being walked in at the wrong level.

**★ ADDENDUM — the transcript must bind the SHAPE, not only the leaf.** This is
the residual half of #768's leaf-binding item (§2.4) and it is the one place
where reading that branch changed this design rather than confirming it.

The per-matrix header above binds `(num_cols, kind, matrix_index)` **inside the
leaf preimage**, so a mis-parsed opening fails authentication. That is
necessary and not sufficient: the injection schedule itself — *which* matrices
exist, at *which* heights, with *which* widths — is what the verifier builds its
walk from, and it must be pinned before any challenge that depends on it.
#768 pins heights and not widths: ✓ VERIFIED `absorb_height_histogram(transcript,
heights: &[usize])` (`batched.rs:196`), with its own module flagging the omission
(`mmcs.rs:78-81`).

**Requirement: absorb `(height, width)` pairs, in commitment order, before the
batched root.** In the LFM machine this is nearly free because the shape is
already registry data — §3.1's `log_heights` and the new `prep_widths` are
exactly the two vectors, and they are already folded into `program_id`. So for
the tower the binding is *doubly* covered (program_id and transcript) and costs
one absorption per proof. For application (a) — the RV64 epoch proof, whose
table set is not a registry constant — it is the load-bearing one.

Why it cannot be skipped on the grounds that "the verifier builds the walk from
the AIR anyway": that argument is exactly the one `verifier.rs:633-639` records
as having been a **live break** for aux opening widths — the check existed
upstream, but the root was absorbed after the challenge, so the prover got to
choose. An unbound shape vector reopens the same door one level up. Bind it, and
bind it before the root.

**Open for ratification (the three decisions this section needs):**
- **M1.** Per-matrix headers (b) vs shape digest (a). Recommendation: (b).
- **M3.** Confirm `(height, width)` histogram absorption before the batched root
  is the right placement, and that ordering it in commitment order (rather than
  sorted) is what the walk reconstruction needs.
- **M2.** Does `matrix_index` replacing `ROWS_PER_LEAF` in the header satisfy
  the reviewer that the multi-matrix reordering surface is closed, or does the
  header need both (5 fields = 2 cells = 2× the header cost, still ~2% in the
  tower)? This is the one place a second opinion is cheap now and expensive
  later, exactly as §1.4.1 was.

### 3.5 The EMITTER — `sub_proof.rs`'s leg structure

This is where application (a)'s 4.3× materialises, and it is the largest single
code item in this plan.

✓ VERIFIED today's structure. `SubProofShape::groups()` returns
`trace_groups ‖ parts_group` (`sub_proof.rs:128-132`); `emit_query_from_bits`
loops `for (commitment, opening) in commitments.iter().zip(openings)` calling
`emit_group_authentication` (`:446-448`), which does
`emit_leaf_hash` → `keccak_merkle_walk(leaf, bits, siblings)` →
`assert_word_eq_lanes` against the root (`:277-292`). The arena stride is
`values + 2·merkle_depth·groups` per query (`:146-150`), and
`emit_sub_proof_with_bits` declares `roots` as `2 · groups.len()` words
(`:548`).

**The shape change.** `SubProofShape` describes ONE sub-proof; the batched
verifier's unit is a ROUND across sub-proofs. Introduce:

```rust
pub struct MmcsRoundShape {
    /// Matrices in commitment order, with their injection levels.
    pub matrices: Vec<(GroupShape, /*depth*/ usize, /*matrix_index*/ u32)>,
    pub depth: usize,                 // = max over matrices
    pub log2_lde_length: u32,
    pub coset_offset: FE,
}
```

and one emitter `emit_mmcs_query(b, round, root_lanes, openings, bits)` that
walks once, absorbing injected rows at their levels. Per query the wrap then
emits **4 walks instead of 100** (2^21) or **4 instead of 196** (2^23).

Four properties of today's emitter that the change must preserve — all four are
load-bearing and all four are documented in the module header as things built by
construction rather than by convention:

1. **The join.** `emit_group_authentication` "takes cells and cannot hint, so
   the only values it can authenticate are the caller's, and `emit_query` hands
   those same cells to the DEEP fold" (`sub_proof.rs:6-12`). The batched emitter
   must keep the same discipline: the injected rows it absorbs are the same
   cells DEEP folds. This gets *easier* under batching, not harder, because the
   crossing described at `:14-36` ("the authentication groups by matrix and the
   fold groups by point") is now one absorption in matrix order feeding one DEEP
   fold in point order — the same two orders, one fewer tree.
2. **One index, shared.** `bits` are decomposed once and drive the walk *and*
   the point derivation (`:38-52`, `:409`). Under batching this becomes
   structurally true across *tables* as well, which removes a whole class of
   hazard: there is no longer a per-table index that could disagree.
   `QueryOutput::bits`/`point` (`:357-389`) keep serving the FRI join, and the
   FRI join gets simpler for the same reason — one FRI, one bit vector.
3. **The two-consumer root hazard.** `GroupCommitment::from_lanes` exists so a
   root reaches the leg "as the SAME cells the transcript absorbed rather than
   as a second hint" (`:201-214`). With 4 roots instead of 100 this is 25× less
   surface, but the constructor discipline must not be relaxed while the count
   shrinks.
4. **The arena stride assertion** (`:621-625`, cursor must equal
   `num_queries * query_words()`). Recompute `query_words` for the batched
   layout — `values + 2·D` per round rather than `values + 2·depth·groups` — and
   keep the assertion. It is the cheapest guard in the file.

**Interplay with COMMIT.md §1.2/S1, stated for the record.** The emitter's
`emit_leaf_hash` (`:245-269`) currently renders base groups through
`edsl::keccak_leaf_hash` and ext groups by unpacking lanes 0..3 and byteswapping
— with an explicit note that lane 3 is not hashed and why that is sound
(`:236-244`). Under an MMCS leaf **that argument must be re-made per matrix**,
because it rests on "every extension value a query opens is also consumed as an
ext operand by the DEEP fold", which remains true but is now asserted across a
concatenated stream. Keep the note, scope it per matrix, and keep the
`debug_assert_eq!(len_bytes, shape.leaf_bytes())` (`:267`) as a per-matrix
assertion — under batching it becomes the thing that catches a mis-parsed
boundary between two matrices' felts.

Effort: **L**.

### 3.6 Item summary

| # | item | effort |
|---|---|---|
| M-1 | `StarkHash::Mmcs<F>` member + keccak instance + single-matrix-equals-`Batched` invariant test | **M** |
| M-2 | **PORT `fri/mmcs.rs` from #768** (1,015 lines + 7 tests), re-parameterize over `StarkHash::Mmcs`, **make the build streaming-per-matrix (§3.3)** with the peak-anon acceptance test | **M** (was L before the branch was read) |
| M-3 | Batched FRI — **port `fri/batched.rs`'s `combine_by_height`**; one instance over the largest domain, smaller matrices folded in at the matching layer; make the accumulation streaming | **M** (was L) |
| M-4 | `multi_prove` R1/aux/parts commit → one tree per round; one root absorbed | **L** |
| M-5 | Verifier mirror + shared query-index derivation | **M** |
| M-6 | Registry: `prep_root` + `prep_widths`, `program_id` re-bless (fold into D0's) | **M** |
| M-7 | `LfmArtifacts` / `verify_against` / `lfm_verify` signature threading | **S** |
| M-8 | Emitter: `MmcsRoundShape` + `emit_mmcs_query`, arena stride, FRI join | **L** |
| M-9 | COMMIT.md spec delta (§3.4) + C-tests for the per-matrix header | **M** |
| M-10 | Security-parameter re-derivation for shared query indices + adversarial review | **M** |

| M-11 | **Settle §1.3's −57% denominator** before §1.1 is quoted as a schedule input. Step 1 is free: pin which measurement it is (`pr768-batched-fri-state.md`'s sim, not the PR's CI +3.61%-at-1-query). Only if that fails, re-measure with PR #846's harness | **S** |
| M-12 | Do not inherit #768's terminal-poly gap: batched FRI must stop at `fri_final_poly_log_degree`, with the free `num_committed` equality assertion (§2.4). Worth +2.5–3.7% | **S** |
| M-13 | Extend `absorb_height_histogram` to `(height, width)` pairs, absorbed before the batched root (§3.4 addendum / M3) | **S** |

Whole item: **L**, comparable to P-a — but two of the three critical-path items
(M-2, M-3) drop from L to M once #768's primitives are ported rather than
rewritten, which is the main practical consequence of reading the branch.
**M-11 is S and should be done first**: it is the only item that can change the
size of the prize. M-12 and M-13 are both S and both come straight from #768's
defects — cheap to carry, expensive to rediscover.

⚠ **One dependency that is easy to miss:** M-1 targets `StarkHash`, which lives
on `blake3-real-hash` and **not on main** (§2.1). Costing this item as
independent of D0 would be wrong.

---

## 4. COLLISION MAP AND SEQUENCING

### 4.1 Shared files

| file | MMCS needs | S3 (in flight NOW) | P-a stages |
|---|---|---|---|
| `crypto/stark/src/prover.rs` R1 loop (`:3240-3295`) | **rewrite** — per-table commit → one tree | **rewrite** — `MainLdeSlot::{Retained,Dropped}`, Phase A | Stage 2 threads `H` (light here) |
| `crypto/stark/src/prover.rs` fused task | reads the batched opening | **rewrite** — recompute LDE, Phase B aux release | — |
| `crypto/stark/src/config.rs` | **new `Mmcs<F>` member** | — | **new `Blake3StarkHash` instance** (Stage 1) |
| `crypto/stark/src/commitment.rs` | new mixed-height builder beside `commit_bit_reversed` | — | leaf backend swap (Stage 1) |
| `crypto/stark/src/fri/**` | **rewrite** — one batched instance | — | **thread `H`, ~13 sites** (Stage 2, §4.1) |
| `crypto/stark/src/verifier.rs` | batched path auth + shared index | — | `H::Batched` already threaded |
| `prover/src/lfm/registry.rs` | `prep_root`, `prep_widths`, re-bless | — | `COMMITMENT_HASH` tripwire (Stage 6) |
| `prover/src/lfm/proof.rs` | signature threading | **`ResidencyMode` threading (landed, `:103-212`)** | — |
| `prover/src/lfm/sub_proof.rs` + `epoch_verify.rs` + `fri.rs` | **rewrite** — batched leg | — | Stage 5 emitter switch (§4.6) |
| `crypto/math-cuda/**` | batched tree kernels | — | **nine blake3 kernels** (track G) |

**Two hard collisions and one soft one:**

- **`fri/**` is contested by MMCS (M-3, rewrite) and P-a Stage 2 (thread `H`,
  ~13 sites, PA-PLAN §4.1).** These must not run concurrently. P-a Stage 2 is
  the smaller and is already scheduled; MMCS's FRI rewrite should land *after*
  it and inherit the threading.
- **`prover.rs`'s R1 loop is contested by MMCS (M-4) and S3 Phase A.** S3 is in
  flight this week. MMCS must not touch that loop until Phase A lands.
- Soft: `config.rs` gets a new member from each of MMCS (`Mmcs<F>`) and P-a
  (`Blake3StarkHash`). Different axes of the same trait, mergeable, but they
  should not be written in the same week by different agents — the
  `const _: fn()` tie-in block (`config.rs:170-192`) is a magnet for conflicts.

### 4.2 How MMCS and S3 interact — they are complements, not substitutes

This is the most important scheduling fact in this document and it is easy to
get backwards.

- **S3 attacks residency:** peak = `17.37·N + 30.2·k` → flat in N. It does not
  reduce the work; it stops the work from being simultaneously resident. Cost:
  one extra forward NTT per table (S3-RECOMPUTE-PLAN §4).
- **MMCS attacks the work:** N itself, 55 → 13 at 2^21, 133 → 15 at 2^23.

Multiply them and the fit closes with margin from either side; take only one and
it is tight. Take only S3 and the flat floor is ~48–56 GiB *plus* whatever the
non-chunk base has grown to. Take only MMCS and N=13 at the measured
13.4 GiB/chunk marginal is ~205 GiB — better than 654, still over.

**⚠ But there is one way they fight, and it is the aux term.** S3's Phase B
("free each table's aux columns when its fused task completes") is written
against the fused per-table task. Batching reimposes a barrier at aux-commit
(§3.3), so Phase B's "end of the fused task" moves and, if the batched builder
is not streaming, the aux LDEs become all-N-live — the exact property S3 is
trying to fix on the main side. **Whoever writes M-4 must read S3 Phase B
first**, and the streaming builder requirement in §3.3 is what keeps the two
compatible. If M-4 lands before S3 Phase B, Phase B's design has to be rewritten
against the new phase structure; if after, it is a small adaptation. That is a
second, independent reason for the ordering in §4.3.

**MMCS also makes S3 Phase C less likely to be needed.** S3's own decision gate
says Phase C proceeds only "if the re-census disagrees" after P-a
(S3-RECOMPUTE-PLAN §3). Batching cuts N by a further 4–9×, which is a second
reason for that gate to come back negative. **Recommendation: re-run the S3
Phase-C decision gate after MMCS, not only after P-a.**

### 4.3 Proposed order

```
NOW ───────────────────────────────────────────────────────────────────────
  S3 Phase A  (in flight)          P-a Stages 1-3        [D0 blake3 switch]
      │                                  │
      ├── S3 Phase B                     ├── P-a Stage 2 (fri/ threading)
      │                                  │        │
      ▼                                  ▼        ▼
  ══ MMCS may start here ═══════════════════════════════════════════════
  M-11 settle the -57% denominator   ← unblocked NOW, S, sizes the prize
  M-10 security-parameter derivation ← unblocked NOW, could invalidate the plan
  M-9  COMMIT.md spec delta+M1/M2/M3 ← unblocked NOW (spec work, no code)
  M-1  StarkHash::Mmcs member        ← needs S3 Phase A *and* D0's StarkHash
                                        (which is on blake3-real-hash, NOT main)
      │
      ▼
  M-2  PORT #768's fri/mmcs.rs       ← needs M-1
  M-6  registry + re-bless           ← FOLD INTO D0's re-bless pass
      │
      ▼
  M-3  batched FRI                   ← WAITS for P-a Stage 2 (fri/ threading)
  M-4  multi_prove batched rounds    ← WAITS for S3 Phase A+B (prover.rs R1)
      │
      ▼
  M-5  verifier mirror
  M-8  emitter                       ← WAITS for P-a Stage 5 (same four sites)
  M-7  signature threading
```

**What can begin the moment S3 Phase A lands:** M-1 (the `StarkHash` member) and
M-2 (porting the tree builder) — both are additive in `crypto/stark`, behind a
configuration, with keccak per-table remaining the default. Neither touches the
R1 loop. ⚠ **But both must be cut from `blake3-real-hash`, not main**, because
that is where `StarkHash` lives (§2.1 correction box) — so M-1 inherits D0's
merge risk, and the MMCS lane becomes a third passenger on the campaign branch
alongside P-a and S3. If D0 is expected to take a long time to reach main, the
alternative is to write M-2's port against the concrete backend first (as #768
did) and parameterize it when D0 lands; that trades one refactor for
independence.

**What can begin RIGHT NOW, before anything:** **M-11** (settle the −57%
denominator — S, and it sizes the prize), **M-10** (re-deriving the security
parameters for shared query indices — pure analysis, and the one item that could
invalidate the whole plan), and **M-9** (the COMMIT.md spec delta, §3.4 — spec
text and C-tests, and it should go to Mauro's ratification in the *same* pass as
D9/RATE=5 since batching changes D9's arithmetic, §1.4). All three are
analysis/spec work with no code dependency and no collision with S3 or P-a.

**What waits for P-a:** M-3 (fri/) waits for Stage 2's threading; M-8 (emitter)
waits for Stage 5, which switches the same four emitter sites (PA-PLAN §4.6).
Doing them in the other order means writing the batched emitter twice.

### 4.4 The July caveat, re-examined

The recorded caveat is *"hash choice gates the batching decision"*, from the
July campaign. **It does not survive, and here is precisely why.**

That claim was about RV64-**guest** economics, where the verifier's bill is
*cycles* and keccak's rate-17 sponge makes leaf absorption cheap relative to
tree walks — so which hash you pick changes which term dominates and therefore
whether batching is worth its complexity.

For the LFM wrap the currency is **permutations/compressions**, i.e. chip cells,
i.e. memory. In that currency:

| | batching buys | leaf share after batching |
|---|---|---|
| keccak inner | 3.46× – 9.06× | 74–77% |
| blake3 inner | 2.28× – 5.12× | 86–88% |

Batching wins decisively under both hashes; the hash changes the multiplier by
about 1.6× and changes nothing structural. **What does survive of the caveat,
restated correctly:** the hash choice governs *what is left to optimise after*
batching. Under either hash the post-batching bill is 74–88% leaf absorption, so
after MMCS lands the only remaining levers anywhere in the stack are the leaf
RATE (tower) and the leaf payload itself (inner). That is a genuinely useful
reframing of the caveat — and it is an argument for doing MMCS *before* spending
more effort on Merkle/FRI micro-optimisation anywhere.

---

## 5. Confidence ledger

| claim | mark |
|---|---|
| The closed form `leaf + groups·depth + fri` and every constant in it | ✓ VERIFIED, `epoch_verify.rs:552-559`, `sub_proof.rs:88-90,:160-166`, `fri.rs:97-116,:133-144` |
| Model reproduces measured `query_permutations` exactly at 4 real points | ✓ MEASURED, census_logs |
| Leg shapes (widths, depths, FRI layers, sub-proof counts) | ✓ MEASURED, `ethrex_e2*_skip.log` |
| Spine permutations at real query counts | ✓ MEASURED, `ethrex_e2*_spine.log` |
| Per-round batching is forced (Fiat-Shamir), not chosen | ✓ VERIFIED, `prover.rs:3216`, `verifier.rs:1295-1317` |
| `StarkHash` shape, `Batched`/`Pair` members, `Node = Commitment` | ✓ VERIFIED, `config.rs:55-192` |
| Registry root array, `program_id` preimage, sentinel slot 11 | ✓ VERIFIED, `registry.rs:53-200` |
| R1 commit loop, root absorption order, per-table transcript fork | ✓ VERIFIED, `prover.rs:3240-3295, :3361-3370` |
| Emitter per-group walk, arena stride, join discipline | ✓ VERIFIED, `sub_proof.rs:128-150, :277-292, :424-492, :534-628` |
| All batched projections (§1.1–1.4 tables) | **DERIVED from the calibrated model** |
| 4.06× hash factor | ✓ MEASURED (campaign hash matrix), composed multiplicatively — ⚠ the fixed non-hash floor does not shrink, so the product is mildly optimistic |
| `KECCAK_RND` cell cost 72,672 / blake3 4,946 | MEASURED / ? INFERRED on the 630 aux width (`tower.py`'s own caveat) |
| #768 branch exists, OPEN, CONFLICTING, +5,097/−1,099 / 25 files, merge base `3ea4f916` (2026-07-17), 25 behind / 45 ahead | ✓ VERIFIED, `gh pr view 768` + `git merge-base` |
| #768's MMCS is wired (3 round instances, built, absorbed, opened) | ✓ VERIFIED `prover.rs:612-614, :2666-2667, :4413-4415` on the branch |
| #768's MMCS layout == this model's `batched_tree_cost` semantics | ✓ VERIFIED, `fri/mmcs.rs:1-56` module doc quoted in §1.3 |
| ~~`879bdc0f` (StarkHash) is a main-side commit~~ | **✗ FALSIFIED** — StarkHash is **not on main at all**; `git grep -c StarkHash origin/main` → 0, `git branch --contains 879bdc0f` → `origin/blake3-real-hash` only. It is a D0 campaign-branch artifact, so M-1 inherits D0's merge risk (§2.1 correction box) |
| ~~#768 = FRI-only, digests-only MMCS~~ | **✗ FALSIFIED** — my inference from memory notes, refuted by the branch's three wired round-MMCS instances. §1.3 records what it cost |
| ~~terminal-poly early-stop gap not reproduced~~ | **✗ FALSIFIED** — the gap is real, in `batched.rs:129,:179`; I had grepped only the non-batched `verifier.rs`. Priced at +2.5–3.7% (§2.4) |
| Width/leaf-boundary binding on #768 | ✓ VERIFIED **implemented and tested** (`mmcs.rs:423-428`, `verifier.rs:2099-2128`, negative test `batched_soundness_tests.rs:122`) — the campaign's "leaf-binding fix" note attaches to #857, not #768 |
| Transcript binds heights but NOT widths | ✓ VERIFIED `batched.rs:196` + the module's own flag `mmcs.rs:78-81` — folded into §3.4 as a design requirement (M3) |
| #768's batched lane contains no CUDA; #877 deleted `plan_table_chunks` which it calls | ✓ VERIFIED (§2.1) |
| A merge silently deletes #845's view machinery (≈ +136M guest cycles) | ✓ VERIFIED via `git merge-tree`, and documented on-branch as a TODO (§2.1) |
| The −57% ↔ model −76.7% discrepancy | **✗ OPEN** — and the target is itself a sim (`pr768-batched-fri-state.md`), not the PR's CI number (+3.61% cycles at 1 query). Gates quoting §1.1 end-to-end |
| CENSUS Part 2 §3's "+19%" D0 feedback | ✗ DOES NOT REPRODUCE — re-running `tower.py` gives +47%; flagged, not resolved |
| Batching factors composed with P-a | DERIVED × MEASURED |

## 6. Reproduction

Projections (the tooling sits beside `project.py` / `tower.py`, the calibrated
chip model it imports):

```
cd ~/workspace/lambda_vm_bench_cache/lfm_census_2026-08-12
python3 mmcs_project.py    # model validation + (a) and (b) headline
python3 mmcs2.py           # four-corner decomposition, #768 comparison, composites
python3 mmcs3.py           # header cost, prep sensitivity, RATE sweep
```

§2's branch facts:

```
gh pr view 768 --json headRefName,state,mergeable,additions,deletions,changedFiles
git fetch origin 'refs/heads/feat/batched-fri-per-epoch:refs/remotes/origin/feat/batched-fri-per-epoch'
git merge-base origin/main origin/feat/batched-fri-per-epoch          # -> 3ea4f916
git merge-base --is-ancestor 879bdc0f 3ea4f916 && echo pre || echo post   # -> post
git diff --stat 3ea4f916..origin/feat/batched-fri-per-epoch
git show origin/feat/batched-fri-per-epoch:crypto/stark/src/fri/mmcs.rs | head -60
git grep -n mmcs origin/feat/batched-fri-per-epoch -- crypto/stark/src prover/src
```

The tower discrepancy in §1.4:

```
python3 -c "
import sys; sys.path.insert(0,'.')
import tower; from tower import *
from project import BYTES_PER_CELL, GIB
tower.WIDTH['LFM_HASH']=(3056,630)
inv,_ = node_cost(REAL21_WRAP, REAL21_RND, 219, 'blake3')
print(inv*BLAKE3_CELLS_PER_COMPRESSION/0.935*BYTES_PER_CELL/GIB)   # 559, not 452
"
```

---

## ADDENDUM A — M-13b: are any rounds-1-3 challenges shape-exploitable?

Written on `mmcs-integration` alongside M-3 / M-4. §M-10.4 split M-13 into (a)
add widths to the round-4 histogram (done on `mmcs-primitives`) and (b) this
question, which gates §3.4's addendum ratification.

### A.0 Answer

**No round-1-3 challenge is shape-exploitable — but the argument that makes it
so is NOT a transcript-ordering argument, and that is worth knowing before
ratifying §3.4.** Every round-1-3 challenge is drawn strictly before the shape
histogram is absorbed, so the only thing standing between a prover and a
post-hoc shape choice is that each round's *root* implicitly binds its own
shape through the tree structure. That holds, and the primitive's tests pin the
two ways it could fail. It is a collision-resistance argument, one link longer
than the ordering argument §3.4 uses for round 4.

**Recommendation (S, and it removes the extra link): absorb the shape histogram
ONCE more, at the very start of round 1, before the first batched root.** Cost
is two field-sized absorptions per table per proof — the same encoding
`absorb_shape_histogram` already defines, and for the LFM machine the vectors
are registry constants that are already in `program_id`. With it, every
challenge in every round is drawn after the shape is explicitly bound, and the
answer above becomes true for the same reason round 4's is.

### A.1 What "the shape" is, and which parts the prover picks

| component | who supplies it | ✓/? |
|---|---|---|
| per-table `width` | the AIR set, never the proof | ✓ VERIFIED — `verifier.rs:639` states the rule for aux widths and `trace_opening_widths_well_formed` enforces it; `MixedMmcs::verify_batch` takes `widths` from the caller |
| table count and order | the AIR set (`airs` argument) | ✓ VERIFIED `verifier.rs:1232-1252` — `multi_verify_views` rejects `airs.len() != proofs.view_len()` |
| per-table `log_height` | **the PROOF** (`trace_length`) | ✓ VERIFIED `verifier.rs:1269`, `:1484` — read from the proof and used to build the domain |

✓ VERIFIED **`trace_length` is never absorbed into the transcript.** Grepping
`verifier.rs` for `trace_length` returns only domain construction and the
part-count check at `:1270-1273`. So heights are the prover-chosen half of the
shape, today as well as under batching.

### A.2 The challenge order, verified

| round | challenge | drawn after | site |
|---|---|---|---|
| 1 | LogUp challenges | every main root | ✓ `prover.rs:3271-3314`, `verifier.rs:1263-1330` |
| 2 | `beta` (constraint coefficients) | the aux root | ✓ `prover.rs:3617` then `:3869` |
| 2 | — | composition-parts root absorbed | ✓ `prover.rs:3903` |
| 3 | `z` (OOD) | the parts root | ✓ `prover.rs:3909-3913` |
| 4 | `gamma` (DEEP) | the OOD evaluations | ✓ `prover.rs:2032` |
| 4 | shape histogram → `alpha` | — | the batched path, `batched/round4.rs` |

Under batching the histogram lands in round 4, so **(1), (2) and (3) are all
drawn before the shape is explicitly bound.** That is the whole question.

### A.3 The four exploits considered, and why each fails

**E1 — move columns between matrices inside one round (the aux-width break,
one level up).** `verifier.rs:633-649` records the live break: the aux root was
absorbed after the shared challenges, so a prover that moved main columns into
the aux tree chose them after seeing `z`/`alpha`. The batched analogue is
shifting a boundary inside a height group's concatenated leaf — lengthen one
matrix's `evaluations` by one, shorten its `evaluations_sym` by one, and the
flat bytes are unchanged. ✗ **Blocked, and by the same remedy: the width, not
the ordering.** `MixedMmcs::verify_batch` length-checks every matrix's opening
against verifier-derived widths. ✓ VERIFIED, `boundary_shift_forgery_rejected`
asserts the flat concatenation is byte-identical first, so the rejection
provably comes from the width binding and not from a differing hash.

**E2 — relabel a matrix's height after seeing a challenge.** The injection
level is `h_max - h`, so a relabelled height changes where the matrix enters the
climb. ✗ **Blocked by the committed tree**: the root was absorbed before the
challenge, and a relabelled schedule no longer reproduces it. ✓ VERIFIED,
`rejects_a_relabelled_injection_height`.

**E3 — claim a different `h_max`.** ✗ Blocked: `verify_batch` requires
`merkle_path.len() == h_max - 1` and rejects `iota >= 2^(h_max-1)`, and the M-14
guard makes an index from a taller domain a rejection rather than a silent
mis-binding. ✓ VERIFIED, `verify_batch_rejects_malformed_shapes_without_panicking`
and `short_round_low_bit_convention_is_exercised`.

**E4 — commit ONE root that opens validly under TWO shapes, then pick the shape
after seeing the round-1-3 challenges.** This is the residual, and it is the
one that is not closed by a check. ✗ Blocked only because producing such a root
is a collision on the leaf/parent hash. **? INFERRED** — no reduction is written
down, and none is attempted here; it is the standard Merkle binding assumption
the rest of the system already rests on. What is new under batching is that
MORE of the epoch's structure (the injection schedule, the group boundaries)
now hangs off that same assumption, where per-table commitments carried one
root per table and bound the count structurally.

E4 is exactly what the A.0 recommendation removes: absorb the histogram before
the first root and the shape is pinned by the transcript, so no root has to be
binding for the shape to be.

### A.4 Consequence for §3.4's ratification

§3.4's addendum ("the transcript must bind the SHAPE, not only the leaf") is
**confirmed and should be ratified**, with one amendment:

> Absorb the `(height, width)` histogram before the FIRST batched root of the
> proof, not only before the batched root of round 4.

Two smaller notes for the same pass:

1. **§3.4's recommended option (b) — one header cell per matrix, at its
   injection level — is NOT what the primitive implements.** `hash_group_leaf`
   concatenates rows with no per-matrix header; the boundary is pinned by the
   verifier-supplied `widths` length check instead (E1 above). That is sound for
   the boundary question, and it is free where option (b) costs 0.9-6.9% per
   query (§3.4's table). What it does NOT bind is `kind`: two matrices of equal
   width at the same height are distinguished only by their position in the
   input order. Since the verifier fixes that order from the AIR set, position
   is as good as a label — but the argument is now "the order is fixed
   externally" rather than "the leaf says which matrix this is", and M1 should be
   ratified in those terms or the header added.
2. **M2 is moot as posed.** It asks whether `matrix_index` replacing
   `ROWS_PER_LEAF` in the header closes the reordering surface. With no header
   at all, reordering is closed by `rejects_swapped_openings_within_a_height_group`
   — the concatenation binds input order, so two same-shape matrices' openings
   swapped is a different leaf. ✓ VERIFIED.

### A.5 What this does NOT answer

- Whether the *AIR set itself* is program shape in the RV64 epoch proof, as it
  is in the LFM machine (where `num_lfm_airs(keccak_rnd_chunks)` comes from the
  registry). A.1's table assumes it is, on the strength of `multi_verify_views`
  taking `airs` from the caller. If an epoch's table set is ever derived from
  the proof, the whole of A.3 has to be revisited — every "the AIR set supplies
  it" row becomes a prover choice.
- The eps_C consequence of batching, which is a separate question with its own
  addendum below.

---

## ADDENDUM B — the `eps_C` delta from batching

Required before the batched path may ship: the 2026-08-15 security audit
(SECURITY-LEVELS §1.3) found the proximity-gaps batching term to be the system's
soundness FLOOR, and batching moves the two inputs it depends on. Grounded in
the REAL census — `bench_cache/lfm_census_2026-08-12/census_logs/ethrex_e20_blowup2_skip.log`,
block 25368371, epoch 2^20, 28 sub-proofs with measured per-leg
`log2_trace_length`, LDE, main and aux widths.

### B.0 Answer

**Batching costs 6.87 bits (union framing) / 8.43 bits (RBR-max framing) at epoch
2^20, at BOTH blowups.** Robust to ±0.2 bits across epochs 2^20/2^21/2^22, both
blowups, a keccak-heavy synthetic profile, and all six `L`-model variants.

| | blowup 2 | blowup 4 |
|---|---|---|
| today, worst single instance | 97.08 | 97.06 |
| today, union over 28 instances | 95.52 | 95.50 |
| **batched, conservative `(L − 1/2)`** | **88.65** | **88.62** |
| delta (union framing) | **−6.87** | **−6.88** |

With the two design adjustments in B.5 the batched floor is **94.19** — a
conservative cost of ~1.3 bits against today's union — and after the `eta`/`m`
retune (Mauro-gated, SECURITY-LEVELS §2.3) the residual is **−1.93 bits at
worst**. That package is the ruling; §B.5 is what the integration implements.

### B.1 The cause, and the framing that gets it backwards

**100% of the penalty is the `|D0|²` lift of short tables. 0% is the batch
size.** ✓ VERIFIED by construction: 28 separate instances all at 2^22 give
88.653 bits; one batched instance carrying `Sum L_t` at 2^22 gives 88.649. For an
epoch whose tables are all the same height, batching is soundness-neutral to
within 0.004 bits.

The reason is worth stating because the natural framing has it exactly wrong.
"Thirty instances collapse to one, so the union bound over thirty goes away" is
FALSE — **the union bound over per-table instances already sums the `L_t`**, so
the batch-size factor is identical on both sides:

```
eps_today = (C/|F|) · Sum_t (L_t − 1/2) · |D0_t|²
eps_batch = (C/|F|) · (Sum_t L_t − 1/2) · |D0_max|²
```

`C` depends only on `m` and `rho` — per-proof, not per-table — so it cancels and
no part of Haböck Thm 2 has to be re-derived to compare them. Dropping the `1/2`
(every `L_t` is in the hundreds), with `w_t = L_t / Sum L`:

```
R = eps_batch / eps_today = 1 / Sum_t ( w_t · 4^-(h_max − h_t) )
bits lost = log2(R) >= 0,  equality iff every table sits at h_max
```

`R >= 1` always: **batching is never a soundness improvement.**

### B.2 Why the real epoch is the bad case

The loss is governed by how much of the batch's WIDTH sits below the tallest
table, and it is insensitive to how many short tables there are. The measured
profile is close to the worst arrangement of that quantity:

- `Sum L_t = 5018` over 28 legs; `h_max` = 2^22 (blowup 2) / 2^23 (blowup 4).
- **The 13 legs at `log2_trace_length <= 7` carry `Sum L = 4601` — 92% of the
  batch — and each is lifted 28-38 bits.**
- `KECCAK_RND` alone is `L = 1999` at 2^3 rows: 127.98 bits on its own domain,
  **89.98 lifted to 2^22**.
- The table that sets `h_max` is `LOCAL_TO_GLOBAL`, which is 9 main + 3 aux
  columns — `L ~ 15`, about 0.3% of the batch weight.

So nearly all the width sits at the bottom and nearly all the height at the top.
`L` and `h` are close to anti-correlated across this table set, which is the
configuration `R` punishes hardest.

### B.3 Corrections to §1.3's inputs, found while grounding this

Four, all ✓ VERIFIED against the census and the code, and all of which make
TODAY's floor better than §1.3 reports rather than worse:

1. **§1.3's 92.0 is a single worst-instance figure (its own §5 item 2 says so)
   and the instance is HYPOTHETICAL.** It pairs the widest table's `L`
   (1480 + 516) with the deepest table's `|D0|` (2^21). No real table has both:
   the 1996-column table is `KECCAK_RND` at 4 ROWS; the 2^21-row table has 12
   columns. Today's real floor is **95.52** (union) / 97.08 (worst instance).
2. **"trace <= 2^20" is false.** Measured epochs contain 2^21 and 2^22-row
   tables.
3. **`LOCAL_TO_GLOBAL` has NO `max_rows` entry** — ✓ VERIFIED,
   `prover/src/tables/mod.rs:83-99` lists 14 tables and it is not among them. It
   is therefore the table that sets `h_max` at every epoch size.
4. **The union costs only 1.0-1.6 bits, not `log2(28) = 4.8`**, because one tall
   table dominates the sum.
5. `L_t = 2 (parts) + (mainW + auxW) + 1 (next row)`. ✓ VERIFIED `step_size = 1`
   and `transition_offsets = [0,1]` for every production AIR, and
   `trace_ood_next_row_columns()` returns exactly ONE column (the LogUp
   accumulator) — not the conservative full-width default.

### B.4 The affine `3/2` is NOT available — do not ship claiming it

✓ VERIFIED `HeightCombiner::absorb` scales by `next_power` and then does
`next_power *= alpha` (`fri/batched.rs:74-85`): powers of ONE challenge, so the
outer level is a degree-`(T-1)` curve carrying `(T - 1/2)`, not `3/2`.

Two further reasons it does not become available cheaply:

- Even if the outer level WERE affine, the composite coefficient is
  `c_t · gamma_t^i` — the inner per-table `gamma` ladders survive, so
  `Sum_t (L_t − 1/2)` stays. Outer-affine buys `log2(27.5/1.5) = 4.2` bits on a
  term already 8 bits below the dominant one: **net ~0.00**.
- ✓ VERIFIED `inject_bucket` adds `beta² · bucket` using the SAME `beta` as that
  layer's fold (`fri/batched.rs:308-326`), so each fold-and-inject step is a
  degree-2 curve in `beta`. The affine reading is not available even for the fold
  steps.

The `3/2` is only reachable by making the INNER per-table DEEP batching affine —
SECURITY-LEVELS R2, roughly 5000 extra transcript squeezes in the recursion
guest. That is a prover change with a real cycle cost, not a parameter change.

### B.5 The remedies, and the ruling

Epoch 2^20, blowup 2, union framing. Full batch = 88.65.

| remedy | floor | recovered |
|---|---|---|
| exclude the `friL == 0` tables (13 legs, `log2tr <= 7`) | 92.24 | **+3.59** |
| cap `LOCAL_TO_GLOBAL` at `max_rows` 2^20 | 90.64 | **+2.00** |
| **both** | **94.19** | **+5.54** (81% of the loss) |
| two batched instances split at 2^19-2^20 | ~92.6 | +4.0 |
| affine INNER batching (R2) | ~100 | +11, real guest cycles |
| `eta`/`m` retune (R1) | see below | the answer |

**Excluding the `friL == 0` tables is not a compromise, it is a correction.**
Those 13 legs have ZERO committed FRI layers per the census, so batching them
buys no FRI-layer saving whatsoever while paying the full `|D0|²` lift — pure
loss for zero gain. They keep their own trivial per-table instances. This also
REDUCES integration work.

**R1 is the answer to the residual.** Batched + R1 = 112.42 bits (`m = 9`);
today + R1 = 114.35 (`m = 16`). So after the retune the batching penalty is only
**−1.93 bits** (blowup 4: 116.43 → 114.87, −1.56). Zero queries, zero prover
cost, one expression in `with_params`. ⚠ The constant is **Mauro's ratification
item** (SECURITY-LEVELS §2.3) and is not implemented here.

**Queries cannot buy any of this back.** ✓ VERIFIED: at `m = 106`, blowup 2,
`s = 219` gives floor 88.65; `s = 10,000` gives floor 88.65 (the query term
reaches 4952 bits and the floor does not move). This reproduces §1.3's
falsification pass 3. The query term actually IMPROVES under batching
(123.21 union → 128.01 single instance) but sits 39 bits above the floor and is
inert. More blowup does not help either: `eps_C` is blowup-independent —
`m^7·rho^-1.5·|D0|²` with `m ~ sqrt(rho)` and `|D0| ~ 1/rho` gives `rho^0`.

### B.6 ✗ UNCERTAIN — the gate item, carried verbatim

**No theorem in the campaign's cited literature (BCIKS20, Haböck 2022/1216,
Block et al. 2024/1161) covers STAGED, HEIGHT-INJECTED, MIXED-DOMAIN batched
FRI.** All three treat `L` codewords on a COMMON `D0`. The conservative
`L = Sum L_t` figure above is defensible as a two-level hierarchical union with
BOTH levels instantiated at `|D_max|` — **that derivation is ours, not a
citation.**

A third reading that is plausibly physically right — "staged", where each table's
`(L_t − 1/2)` attaches to its own injection-layer domain and only the ~22
fold/inject steps and ~13 bucket-`alpha` curves are paid at the taller domains —
gives **94.57 bits, delta only −0.95**. If a citable analysis for
mixed-degree/staged FRI turns up (Plonky3-style, or STIR/WHIR degree
correction), the penalty likely collapses from −6.9 to −1.0.

**Until then: the claim that ships is the conservative two-level hierarchical
union at `|D_max|`, NOT the affine `3/2`.**

### B.7 One soundness positive, verified in passing

`absorb_shape_histogram` binds heights and widths BEFORE `alpha` is sampled
(`fri/batched.rs:445-447`), so an adversary cannot choose the height profile
after seeing `alpha`. That binding is load-bearing for everything above — the
whole analysis assumes the height profile is fixed. Keep it. (Addendum A's
recommendation to absorb it once more before the FIRST batched root strengthens
the same property for rounds 1-3.)
