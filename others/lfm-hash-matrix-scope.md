# The hash matrix — slice 0 scoping report

Wave 8 (`[hash-w8]`), 2026-08-04, branch `feat/lfm-hash-matrix` off `feat/lfm-assembly`
@ 891f534f. Written BEFORE any hash was built, which is the point: it exists to be
falsified by wave 9's measurement rather than to stand in for it.

Provenance is marked throughout (standing-decisions method rule 6). **VERIFIED**
means I read the code and cite it. **DERIVED** means arithmetic over verified
numbers, shown so it can be rechecked. **INHERITED** means it comes from another
agent's report or the team lead's notes and I did not re-establish it.

---

## 0. Headline

**The machine already has a hash swap surface, and it is NOT the socket keccak is
plugged into.** `prover/src/lfm/hash.rs` is titled "The LFM hash interface — the
machine's swap surface" and freezes a contract — the `LfmHasher` trait, a 12-felt
sponge state, a 4-felt digest, and the `LFM_HASH` bus tuples and opcode numbers —
behind which sits `TestPermutation`, explicitly labelled **NOT cryptographic**
(`hash.rs:1-12`, `hash.rs:46-53`). Its own doc comment names the candidate set:
"Poseidon2 is broken; candidates are Poseidon-original, RPO/XHash, Monolith and
reduced-round Blake2s" (`hash.rs:3-5`). VERIFIED.

That reframes the brief's phrase "behind the same socket". There are **two**
sockets, and they are documented as not interchangeable:

| | socket 1 — keccak hosting | socket 2 — the `LFM_HASH` chiplet |
|---|---|---|
| what fills it | production `KECCAK_RND`/`KECCAK_RC`/`BITWISE` AIRs, unchanged, via `keccak_adapter` | one degree-3 chip, `chips::hash`, 28 value columns, 1 row per permutation |
| gadget | `edsl::keccak_merkle_walk`, `keccak256` | `edsl::merkle_walk`, `edsl::SpongeVar` |
| digest | 2 machine cells (8 felts, 32 bytes) | 1 machine cell (4 felts) |
| status | measured: 11.17 B cells / epoch verify | placeholder permutation only |

`edsl.rs:137-143` states it outright: "the two are not interchangeable:
`merkle_walk` compresses with `LFM_HASH`/`TestPermutation`, the deliberately
non-cryptographic Milestone-C placeholder, so it can only ever authenticate the
Milestone-C fixture tree. Production trees are keccak throughout." VERIFIED.

So a candidate column is **not** a variation on the keccak column. It is socket 2
carrying the epoch verifier's real workload for the first time, which is why the
matrix's other columns "ARE the number" rather than a refinement of it.

**★ THE RESULT THAT MATTERS MOST, and it is not the one I expected: every candidate
fits the 124 GiB box.** Keccak's wrap needs 290–350 GiB; the predicted candidates
span ~48 GiB (RPO) to ~79 GiB (Blake, the only one with a real in-AIR measurement).
**The hash decision is therefore not cost-gated** — it is very nearly a purely
cryptographic choice, which is a far better position than my first pass described.
Matrix in §2.3, my reversal on blake in §2.5.

**Recommendation, in one line: build Poseidon-original first — not because it is
cheapest, but because it has a direct `poseidon1-air` donor in the vendored Plonky3
tree, an in-tree HADES skeleton, and zero AIR data anywhere in the corpus, so its
column is the one that adds information rather than confirming it.** Reasons in §4.

---

## 1. The cost model, validated against the measured keccak column

The census formula, read from source (VERIFIED, `airs.rs:122-128`, `airs.rs:246-249`,
`airs.rs:234-236`):

```
main_cells = padded_rows × (NUM_COLUMNS − PREP_WIDTH)
aux_cells  = padded_rows × ceil(bus_interactions / 2)
base-field-equivalent = main_cells + 3 × aux_cells        (aux are cubic-extension elements)
```

`airs.rs:270-278` confirms `LfmChipCells.main_cols` is "the AIR's width less its
preprocessed prefix" and `aux_cols` is "one per pair of bus interactions".

Writing the hash bill as `P × (m + 3a) × padding`, where `P` = permutations per
verify, `m` = main cells per permutation, `a` = aux cells per permutation:

| quantity | keccak, production shape | source |
|---|---|---|
| `P` | 118,080 | INHERITED (ledger entry 10) |
| `m` | 36,256 = 736 (`LFM_KECCAK`, 1 row) + 24 × 1,480 (`KECCAK_RND`) | INHERITED, widths pinned by a test (status log 2026-07-30T15:40Z) |
| `a` | 13,912 | INHERITED (ledger entry 10) |
| `m + 3a` | **77,992** base-field-equivalent cells per permutation | DERIVED |
| padding | 1.01871 | DERIVED (see below) |

DERIVED check — the model reproduces the ledger's measured numbers exactly:

```
total  base-equiv = 5,077,422,224 + 3 × 2,029,461,548 = 11,165,806,868   ✓ ledger
hash   base-equiv = 4,364,173,312 + 3 × 1,672,478,720 =  9,381,609,472   ✓ 84.02 % of total
non-hash residue  = 11,165,806,868 − 9,381,609,472    =  1,784,197,396   (15.98 %)
P × (m + 3a)      = 118,080 × 77,992                  =  9,209,295,360
implied padding   = 9,381,609,472 / 9,209,295,360     =  1.01871
```

The 1.87 % padding agrees with the ledger's independent statement that default
chunking wastes 1.7 % of round rows at the production shape — close but not
identical, the residue presumably being `LFM_KECCAK`'s own power-of-two padding. I
did not resolve that split and it does not move any conclusion.

**What this model cannot see** (rule 6): it prices only the hash chips. It assumes
the non-hash residue of 1,784,197,396 cells is candidate-independent, which is
FALSE in a direction that favours every candidate — part of that residue is
byte-serialization work (`felt_be_halves`, `Unpack`s, `LFM_BITDEC`/`LFM_BALU`) that
a field-native hash deletes outright. Treating the residue as fixed therefore makes
my candidate predictions **conservative** (too big), not optimistic. It also cannot
see prover wall-time or RSS, only cells.

### 1.1 `LFM_HASH` as it stands today

VERIFIED from `layout.rs` (`pub mod hash`) and `chips.rs:479-523`:

- `PREP_WIDTH = 11` — `IN_ADDR0..2`, `OUT_ADDR0..2`, `MODE_C`, `MODE_P`, `MULT0..2`.
- `NUM_COLUMNS = PREP_WIDTH + 28`; the 28 value columns are `IN0..IN11` (12),
  `S8..S11` (4 materialized capacity columns), `OUT0..OUT11` (12).
- 6 bus interactions, all `BusId::LfmMem` (3 receivers, 3 senders) → `aux_cols = 3`.
- `max_degree() = 3` (`chips.rs:532-534`); one row per permutation.

So `TestPermutation` costs `m = 28`, `a = 3`, i.e. **37 base-field-equivalent cells
per permutation** — 2,108× cheaper than keccak's 77,992. That number is a floor
with no cryptographic content whatsoever: it is ONE degree-3 round. Any real
candidate multiplies it by its round count and S-box overhead. Quoting 37 as a
candidate cost would be the single worst error available in this leg.

### 1.2 The rate penalty — the axis that moves the WRONG way

VERIFIED: keccak's rate is 136 bytes (`layout.rs:116`), and the machine serializes
each felt as 8 bytes (two 4-byte halves, `keccak_host.rs:15`,
`edsl.rs:128-134`). So keccak absorbs **17 felts per permutation**.

VERIFIED: the `LFM_HASH` sponge is "state = 3 cells (rate 2, capacity 1)"
(`edsl.rs:16-17`), i.e. **8 felts per permutation**.

DERIVED: on absorption-bound work a candidate behind socket 2 pays **2.125×** as
many permutations as keccak. Not a rounding detail — it directly offsets the
cells-per-permutation win, and it is a consequence of the FROZEN
`HASH_STATE_FELTS = 12`.

Where it bites, and where it does not (VERIFIED against `fri.rs:139-144` and
`edsl.rs:149-155`):

- **Merkle parent step** — keccak hashes 64 bytes = 8 felts, "64 bytes sits inside
  one 136-byte rate block, so a level is exactly ONE permutation". A candidate
  absorbs 8 felts into a rate of 8 → also exactly one permutation. **1:1, no penalty.**
- **FRI layer leaf** — a 48-byte pair = 6 felts, one rate block either way. **1:1.**
- **Trace leaf hash** — a row PAIR column-major over `c` columns is `2c` felts.
  keccak: `ceil(16c / 136)` permutations. Candidate: `ceil(2c / 8)`. At `c = 10`
  that is 2 vs 3; at `c = 1480`, 175 vs 370. **Here the candidate is up to 2.125×
  worse.**
- **Transcript/spine** — absorption-bound, so ~2.125×.

This is why `P` must be measured rather than assumed, and why I recommend pinning it
first (§4).

---

## 2. Predictions

These are the falsifiable content of this report. Every candidate row is an
ESTIMATE; the derivations are given so wave 9 can kill them with a measurement.

### 2.1 Where each candidate's `P` comes from — and why it is measurable now

`epoch_verify::query_permutations` is already a **closed form over shapes**
(VERIFIED, `epoch_verify.rs:414-434`):

```
per_query = leaf_permutations(sub) + groups × sub.merkle_depth + fri.permutations_per_query()
leaf_permutations = Σ_groups num_blocks(g.leaf_bytes())      num_blocks(n) = n/136 + 1
```

Its doc comment carries the warning that matters here: "A leaf is NOT one
permutation. It covers `ROWS_PER_LEAF · num_columns` elements at 8 or 24 bytes
each … so the epoch's widest table (2,056 OOD columns) has a leaf worth hundreds of
permutations while a FRI layer's one-column leaf is worth one. Predicting the leg's
bill as 'one leaf plus one per level' undercounts it by the whole width of the
trace" (`epoch_verify.rs:404-412`). VERIFIED.

**Consequence, and the single most useful thing in this report:** a candidate's `P`
is a function of the SHAPES only — not of any permutation's internals. Swapping
`num_blocks(bytes) = bytes/136 + 1` for `ceil(felts/8)` yields the candidate's `P`
**with no hash implemented at all**. That is a pure-arithmetic, additive,
differentially-testable slice, and it should come first (§4).

Only the leaf and spine terms move; parent steps and FRI leaves are 1:1 (§1.2). So

```
P_candidate = 2.125 × (absorption-bound part of P) + 1.0 × (path-bound part of P)
```

and the split is exactly what the closed form computes.

**UPDATE — slice A ran, so this axis is now MEASURED, not bounded.** I built the
rate-parameterised closed form and the numbers below come out of the suite
(`epoch_verify_tests::the_assembled_epoch_verifier_runs`, blowup 8 / 73 queries,
real trace lengths):

```
keccak   rate 17 felts/perm:  115,413 permutations  (67,671 leaves + 47,742 paths/FRI)
LFM_HASH rate  8 felts/perm:  187,902 permutations  (140,160 leaves + 47,742 paths/FRI)
candidate/keccak = 1.6281x    (leaf term alone 2.0712x)
absorption-bound share of the keccak bill: 58.6%     widest leaf: 3,456 felts
```

**The keccak side reproduces the ledger exactly: 115,413 is entry 10's own legs
figure** (118,080 = 2,667 spine + 115,413 legs). That is the corroboration that
makes the candidate side trustworthy — the same function, at a different rate.

So the interval collapses to a point: the candidate pays **1.63×** the permutations,
not the 2.125× ceiling, because 41.4 % of the keccak bill is path/FRI work that is
1:1 at any rate. Adding the spine (2,667, absorption-bound, so bounded between 1.0×
and 2.125×) gives

```
P_candidate ∈ [190,569 , 193,569]  =  1.614x – 1.639x keccak's 118,080
```

— a ±0.8 % spread, so the spine's uncertainty is immaterial and I use ~192,000.

### 2.2 Cells per permutation

**REVISED after the corpus extraction landed.** My first pass estimated round counts
from my own domain knowledge; the corpus supplies a MEASURED anchor that is better
than my estimate and, critically, one that normalizes onto our socket exactly.

**The normalization is free, and this is the single luckiest fact in the leg.** The
corpus's primary hash artifact (Miden's BlakeG addendum) states: *"BlakeG keeps
Poseidon2's exact sponge geometry (state 12, rate 8, digest 4), so invocation counts
are hash-invariant and the whole price is per-invocation trace cost."* **State 12,
rate 8, digest 4 is exactly our frozen `LFM_HASH` contract**
(`HASH_STATE_FELTS = 12`, rate 2 cells = 8 felts, `HASH_DIGEST_FELTS = 4`). So the
corpus's per-2-to-1 cell figures transfer to our socket directly, and every
field-native candidate shares ONE permutation count — the `P ≈ 192,000` measured in
§2.1.

Measured anchors from the corpus, per 2-to-1 compression (Miden, **Goldilocks — our
field**):

| | main cells | aux (EF) | base-equiv `m+3a` | provenance |
|---|---|---|---|---|
| Poseidon2 (the dead baseline) | 256 | 16 | 304 | MEASURED, 16-col AIR × 16 rows/perm |
| BlakeG 32-row | 4,096 | 768 | **6,400** | MEASURED — 13.9× main, 48× aux |
| BlakeG 64-row (1st gen) | 5,120 | 768 | 7,424 | MEASURED |

Plus, for any Blake-class candidate, an `And8Lookup` table AIR at a **fixed 2¹⁶ × 10
= 655,360 cells in every proof regardless of workload** (the corpus notes our `RANGE`
chip can absorb that role, so it need not add a chip).

⚠ **My own estimate was 2× CONSERVATIVE, and I am keeping it as the pessimistic
bound rather than discarding it.** I derived ~608 main cells for Poseidon-original
in a one-row layout; Miden's measured Poseidon2 at t=12 over Goldilocks is 256 main.
Their 16 columns × 16 rows beats my unrolled 608 because a row-per-round layout
reuses the state columns instead of allocating fresh ones per round. Since the
corpus rates Poseidon-original at **≈1× Poseidon2 in-AIR** ("AIR cost is
S-box-dominated; round counts match"), 304 base-equivalent is the central case and
617 is my conservative bound. Both are in the matrix.

⚠ **PROVENANCE of each candidate's figure, per the corpus's own marking** — this is
the part that decides how much weight each column carries:

- **Blake: the ONLY candidate with a real in-AIR measurement.** 13.9× main / 48× aux
  vs Poseidon2, read off Miden's BlakeG branches. Also the only one with a shipped
  production existence proof (Airbender runs blake2s-7 as its only hash), **caveat:
  at an 80-bit target; ours is 100/128-bit**, which raises query counts and the bill
  proportionally.
- **RPO: a ROW-COUNT ratio, not a benchmark.** 0.5× is `HASH_CYCLE_LEN` 8 vs 16 read
  off Miden source at an *assumed-equal column count*. The corpus does not check
  whether an RPO AIR needs the same 16 columns, and RPO's inverse S-box typically
  needs its own witness per lane. **Read 0.5× as rows, with columns unverified.**
- **Poseidon-original: reasoned, with ZERO AIR data anywhere in the corpus.** The
  ≈1× is inferred from S-box dominance and matching round counts. What IS measured
  is its *migration* bill: ZisK's upstream PR = 181 files, +49,324/−13,097.
- **Monolith: the weakest-evidenced row** — "few×", priced by analogy to Miden's
  `And8Lookup`, with the native claim coming from the designers' own design goal.
  The corpus's own verdict: "Watch, don't bet the protocol yet."

The layout constraint remains real and VERIFIED: `max_degree() = 3` for the
`LFM_HASH` chip (`chips.rs:532-534`), and over-declaration is safe while
under-declaration is not
(`prover/src/tests/constraint_set_tests_a.rs:66-74`). An `x⁷` S-box at degree 3
needs two intermediate columns (`x²`, `x³`, then `x⁷ = (x³)²·x`).

The layout constraint is real and VERIFIED: `max_degree() = 3` for the `LFM_HASH`
chip (`chips.rs:532-534`), and `max_degree` "is what the engine uses as the
composition-poly degree bound … over-declaration is safe, under-declaration is not"
(`prover/src/tests/constraint_set_tests_a.rs:66-74`). Raising it is possible but
the wrap runs at blowup 2, so a higher-degree composition polynomial costs LDE
cells — self-defeating for a memory play. **So every candidate must express its
S-box in degree ≤ 3, which for `x^7` means two intermediate columns per S-box**
(`x²`, `x³`, then `x⁷ = (x³)² · x`, degree 3 over columns).

⚠ **Correction to my own first pass, kept because it decides layout.** I initially
carried `a = 3` into every layout. Wrong: `aux_cells = rows × ceil(interactions/2)`
scales with ROWS, so a 30-rows-per-permutation layout pays 90 aux cells (270
base-equivalent), not 3. This is also why Miden's 16-row Poseidon2 shows 16 aux and
not 1. Aux count triple, so row count is not free even for a lookup-free hash.

For a purely algebraic candidate the aux bill is otherwise just the chip's existing
6 `LfmMem` interactions (`aux_cols = 3` per row); row-to-row state wiring is
transition constraints, not buses, so it adds none. **The collapse from keccak's
13,912 aux per permutation is structural** — that number is `KECCAK_RND`'s BITWISE
lookups, which ARE bus interactions, and an algebraic hash has none.

**A Blake-class candidate does NOT get that collapse**, and this is where I have to
correct myself hardest (see §2.5): its 768 aux per compression is 48× Poseidon2's,
for exactly the reason keccak's is large. What I got wrong was the conclusion I drew
from it.

### 2.3 The predicted matrix, assembled

One `P` for every field-native candidate (they share the socket's rate-8 sponge, and
the corpus confirms invocation counts are hash-invariant at this geometry), so the
matrix is a single multiplication per row. Memory uses the **two-term model** wave 7
established after falsifying the one-parameter 33.7 B/cell figure: **≈27 B/cell plus
≈190 MB per sub-proof** (peak RSS carries a per-sub-proof term).

| candidate | base-equiv per perm | `P` | hash cells | **total cells** | vs keccak | projected RSS |
|---|---|---|---|---|---|---|
| **keccak — MEASURED, ours** | 77,992 | 118,080 | 9,381.6 M | **11.166 B** | 1.00× | 284 GiB (band 290–350) |
| RPO (0.5× P2 rows, cols unverified) | 152 | 192,000 | 29.7 M | **1.814 B** | **6.16×** | **48 GiB** |
| Poseidon-original (corpus ≈1× P2) | 304 | 192,000 | 59.5 M | **1.844 B** | **6.06×** | **49 GiB** |
| Poseidon-original (MY conservative est.) | 617 | 192,000 | 120.7 M | 1.905 B | 5.86× | 50 GiB |
| Monolith (few×, band — weakest evidence) | ~850 | 192,000 | 166.9 M | ~1.951 B | ~5.7× | ~52 GiB |
| **BlakeG 32-row — MEASURED (Miden)** | 6,400 | 192,000 | 1,252.4 M | **3.037 B** | **3.68×** | **79 GiB** |

Blake's row includes the fixed 655,360-cell `And8Lookup` table (negligible at this
scale, and our `RANGE` chip can absorb the role rather than adding a chip).

**★ THE HEADLINE, AND IT REVERSES WHAT I TOLD YOU FIRST: every candidate fits the
124 GiB box, blake included.** The keccak wrap needs 290–350 GiB; the cheapest
candidate needs ~48 GiB and the most expensive ~79 GiB. The hash choice is therefore
**not** gated on cost — all four make the production wrap provable on hardware we
have. That reframes the decision as almost purely cryptographic, which is a much
better position than the one my first pass described.

Two robustness notes:
- **The residue dominates every candidate row.** Once the hash is cheap, 1.784 B of
  a ~1.85 B total is the already-measured non-hash verifier. So the algebraic rows
  are insensitive to their (weakly-evidenced) cell estimates: RPO vs Poseidon vs
  Monolith differ by 7 % in total cells while their per-permutation estimates differ
  by 5.6×. **Choosing among the algebraic candidates on predicted wrap size is
  choosing on noise.**
- Blake is the one row where the hash still matters — 1.25 B of its 3.04 B — so it
  is also the only row whose estimate is worth refining, and it is the row that is
  already measured.

### 2.4 The 2-to-1 normalization, stated rather than assumed

The corpus's cross-system figures are per **2-to-1 compression**; ours are keccak
**permutations** at a 136-byte rate. Comparing them directly would be wrong, so the
split at the production shape (DERIVED from §2.1's measured decomposition):

| | permutations | is it a 2-to-1 compression? |
|---|---|---|
| Merkle parents + FRI layer steps | 47,742 | **yes** — 64 bytes / 8 felts, one block either way |
| wide trace-leaf absorbs | 67,671 | **no** — up to 3,456 felts, a sponge run |
| spine (transcript) | 2,667 | no — absorption |

**So our apples-to-apples compression count is ≈47,742, i.e. 6.2× Airbender's 7,685
— not the 15.4× a naive 118,080/7,685 gives, and nowhere near the 117× the corpus
flagged for the old guest verifier at ~900,000.** The LFM has already retired most
of the corpus's headline anomaly; that is worth recording, because §I.7's
"recursion diverges at a Blake-class hash" conclusion was reasoned at ~900,000
compressions and does not transfer to this machine at 47,742. It is the main reason
blake lands at 3.68× rather than the corpus's ~220 %.

⚠ Our leaf term is 59 % of the bill and has **no analogue** in the per-2-to-1
figures. It is also the term the frozen rate-8 state penalises (§1.2). Any
cross-system comparison that omits it understates us by 1.6×.

All numbers name their epoch shape: fixture epoch, profile
`[2 ×14, 3, 4 ×4, 5 ×3, 7, 20]`, 24 sub-proofs, fibonacci guest, 16-cycle
INTERMEDIATE epoch, inner blowup 8 / 73 queries; wrap options blowup 2 / 219
queries / grinding 20 (entry 10's rule).

---

### 2.5 Where I was wrong about blake, and why

My first pass said: *"Blake2s is ARX on 32-bit words, so it needs the same
bit-decomposition mechanism that makes keccak's aux 53.5 % of its cost. So blake's
in-AIR character is keccak-like, not Poseidon-like … the hash decision may not buy
the 2.8× memory relief the wrap needs at all."*

**The premise was right and the conclusion was wrong.** Blake IS bit-oriented, and it
does pay a 48× aux penalty against Poseidon2 — that part survives contact with the
corpus's measurement. What I inferred from it does not, for a reason I had no excuse
to miss: **keccak-like in mechanism is not keccak-like in magnitude.** Our keccak
costs 77,992 base-equivalent cells per permutation because `KECCAK_RND` is 1,480
columns over 24 rows; BlakeG is 128 columns over 32 rows. Same mechanism, **12×
apart**. Blake lands at 3.68× better than keccak and ~79 GiB — comfortably inside the
box, not outside it.

Two lessons I would keep:
1. I reasoned from a *mechanism* to a *cost ratio* without ever multiplying the
   widths, which the census formula in §1 was sitting right there to do. A ratio
   claim needs the arithmetic even when the qualitative story is correct.
2. The brief told me blake was the probable ship choice, and I built a narrative
   ("the decision-critical column") that made my analysis load-bearing for it. The
   corpus **renders no pick at all**. Being handed a leading hypothesis is a reason
   for more falsification, not less.

Also corrected: the brief's framing that blake is "the most probable final ship
choice" is not what the review says. Ranked by *evidence strength* rather than
preference: **Blake** (only real in-AIR measurement, plus a shipped 80-bit production
system) > **RPO** (a rows-only ratio, columns unverified) > **Poseidon-original**
(reasoned ≈1×, zero AIR data, but a measured 181-file migration bill) > **Monolith**
("few×", priced by analogy). All four columns are scoped here; none is privileged.

---

## 2.6 What already exists in-tree — searched structurally, and it changes §4's risks

I ran this myself after the dispatched inventory leg failed to report. Method note
worth recording because it nearly cost me a false claim: my first pass used
`grep -r --include=*.rs` **unquoted**, which the shell tried to glob and failed with
"no matches found" — indistinguishable from grep finding nothing. Two of my
"nothing exists" readings were shell errors, not evidence. Re-run quoted.

Searched: `find` over every `.rs`/`.toml` in the repo for
`blake|poseidon|rescue|rpo|monolith|griffin|sha2|sha256|anemoi|reinforced`; then
`grep -rn --include='*.rs'` (quoted) for the same terms plus `hades_permutation`,
`PermutationParameters`; then read the files found.

**FOUND — a Poseidon-original skeleton (VERIFIED):** `crypto/crypto/src/hash/poseidon/`
(96 + 45 lines). A `Poseidon` trait over `PermutationParameters` whose
`hades_permutation` is `N_FULL_ROUNDS/2` full rounds → `N_PARTIAL_ROUNDS` partial →
`N_FULL_ROUNDS/2` full (`mod.rs:28-41`). **That is Poseidon-original's HADES
structure, and it independently confirms the round SHAPE my §2.2 estimate assumed.**
The trait carries `RATE`, `CAPACITY`, `ALPHA`, `N_FULL_ROUNDS`, `N_PARTIAL_ROUNDS`,
`MDS_MATRIX`, `ROUND_CONSTANTS` (`parameters.rs:11-27`), with a default `mix`.

**But it has NO concrete instance.** `grep` for `PermutationParameters for` /
`impl PermutationParameters` across every `.rs` in the repo returns nothing, so
there is no parameter set, no round constants, no MDS matrix and no field binding
anywhere in-tree. The permutation is a generic skeleton, not a usable hash.

**FOUND — Poseidon Merkle backends (VERIFIED):** `TreePoseidon<P: Poseidon>`
(`crypto/crypto/src/merkle_tree/backends/field_element.rs:50-71`) and
`BatchPoseidonTree<P>` (`field_element_vector.rs:206`) both implement
`IsMerkleTreeBackend` with `Node = Data = FieldElement<P::F>` — i.e. a
**field-element** tree, next to the byte-oriented `Digest`-generic backend in the
same file. So the commitment layer is already a trait with a field-native Poseidon
implementation behind it. UNVERIFIED, and important: whether the *prover* is generic
over that trait or pins a concrete backend. I did not establish it.

**FOUND — sha256 AIR SPECS (VERIFIED as files, not as an AIR):** `spec/src/sha256.toml`,
`sha256round.toml`, `sha256msgsched.toml`, `sha256consts.toml` — 749 lines. I found
no generated Rust AIR for them in `prover/src` or `crypto`. Relevance: sha256 is
bit-oriented like blake, so this is the closest in-tree precedent for what a blake
AIR's shape and effort look like — worth reading before costing blake.

**ABSENT — blake, Rescue/RPO, Monolith, Griffin, Anemoi: no AIR and no software
implementation, in any spelling.** One trap resolved: `monolith` matches 26 times
across `prover/src` (`statement.rs`, `paged_mem.rs`, `page.rs`, `lib.rs`,
`recursion.rs`), and every occurrence is the *monolithic proof* concept, nothing to
do with the Monolith hash. A term-only search would have reported a Monolith
implementation that does not exist.

### 2.6.1 Donor AIRs — the vendored Plonky3 tree, which the corpus never analyzed

VERIFIED by listing the tree myself: the main checkout's `others/Plonky3` (@ 4aed8fe4)
carries **`poseidon1-air/`, `poseidon2-air/`, `blake3-air/`, `monolith-air/`,
`keccak-air/`** as crates, alongside bare permutations in `poseidon1/`, `monolith/`,
`rescue/`. The corpus explicitly lists Plonky3 as unscoped
(`recursion_architectures.md:772`: "Candidates still unscoped: `others/leanVM-b`,
`others/Plonky3`"), so none of this is in the review.

This **materially changes the per-candidate build estimates**, and it reorders them:

| candidate | donor AIR | build risk |
|---|---|---|
| **Poseidon-original** | **`poseidon1-air`** — a direct donor for exactly this hash | **lowest** |
| Monolith | `monolith-air` (+ `monolith/` = Monolith-64 Goldilocks, width 8/16) | low, and un-analyzed by the corpus |
| Blake | `blake3-air`, plus Miden's BlakeG branches (unmerged, 13→21 files / 3.8→5.1 k lines) | high — bit-oriented, needs the lookup table |
| **RPO** | **NONE.** Only the permutation (`rescue/src/rpo/goldilocks.rs`, 394 lines); **no `rescue-air` crate exists**, and the corpus's only AIR pointer is Miden git history | **highest** |

⚠ **That inverts the naive ranking.** RPO is the cheapest predicted column (~48 GiB,
6.16×) and has the *worst* donor situation — its 0.5× is a rows-only ratio with
unverified columns AND there is no AIR to copy. Poseidon-original is within 7 % of
RPO on predicted total cells (§2.3's residue argument) and has a direct donor. **So
the cheapest-looking column is the expensive one to build, and the difference it
would buy is inside the noise.**

⚠ **A donor warning that transfers, VERIFIED in the corpus**
(`openvm-port-study-brief.md:214-221`): *"The hash is the wall, and it is worse than
'swap constants.' `Poseidon2SubAir` is a single-variant enum locked to
`BabyBearPoseidon2LinearLayers` … Round constants convert to any `F` by type but
produce numerically meaningless values … A Goldilocks Poseidon2/RPO chip is a **new
chip of the same shape**, not a parameter change."* Expect a donor to supply
structure, not code.

**Consequence for §4's worst risk — it shrinks but does not vanish.** The oracle
problem is no longer "write a Poseidon from nothing and check it against itself".
The HADES structure is in-tree and reviewed, and the remaining input is a
**parameter set** (α, round counts, MDS, round constants for the chosen field and
`t = 12`), which must come from a published reviewed source. Route that keeps it
additive: implement `PermutationParameters` for a LOCAL type inside
`prover/src/lfm/` — a foreign trait on a local type needs no `crypto/**` edit,
where adding a parameter set WOULD be an always-stop change.

---

## 3. Build inventory

Split by the standing-decisions boundary. **Nothing on the critical path for a
CELLS measurement touches `crypto/**` or `prover/src/tables/**`.**

### 3.1 Additive LFM work (pre-authorized)

| piece | what | oracle for differential testing |
|---|---|---|
| ~~A. candidate-`P` closed form~~ **DONE** | `blocks_at_rate`/`leaf_permutations_at_rate`/`query_permutations_at_rate` in `epoch_verify.rs`, rate as a parameter | **the rate-17 case reproduces `query_permutations` exactly** — a real differential, because the new function is written through FELTS and the old through BYTES and `keccak_host::num_blocks`, and NEITHER delegates to the other (rule 7's trap avoided deliberately; making one delegate would have made the test vacuous). The existing assert ties `query_permutations` to the EMITTED count, so the chain reaches the emitter |
| B. `LfmHasher` impl for the candidate | `permute([FE;12]) -> [FE;12]` + `compress_iv` | a reference Poseidon implementation over Goldilocks with the same round constants / MDS; test vectors. **This is the piece with a real oracle problem — see §4 risks** |
| C. the chip's constraint block | replace `chips::hash::HashConstraints`' `t_i` block with the candidate round function at degree ≤ 3 | `constraint_set_tests_a`-style degree check (`measured <= max_degree`), plus prove+verify: rule 2 says execute-only tests prove nothing about chips |
| D. gadgets on socket 2 | a candidate `merkle_walk` / sponge already exist (`edsl::merkle_walk`, `SpongeVar`) and are hash-agnostic by construction | they are already exercised against `fixture::HostSponge`, which mirrors the trait — so B's correctness carries them |
| E. re-emit + census | emit the epoch verifier with socket-2 gadgets, run `report_census` | `the_census_agrees_with_the_traces_the_prover_builds` (exists, green) |

### 3.2 Always-stop / out of scope for a cells measurement

- **An inner prover under the candidate hash — ALWAYS-STOP, and it needs the USER,
  not the team lead.** The inner-prover trace (INHERITED from the dispatched leg via
  the team lead; I did not verify it myself) is: the transcript **hardcodes**
  `PlatformKeccak256`; `config.rs` **pins three Merkle aliases**; `ProofOptions` has
  **no hash field at all**; grinding is hardcoded. The cheapest seam ("Case A") is
  about **4 files on the CPU path but is non-additive inside `crypto/**`**; the
  general version ("Case B") additionally breaks a trait and the proof format.

  **Scope it as a proposal, do not build it.** The proposal: a `config.rs`
  feature-flag seam, a defaulted `D` type parameter on `DefaultTranscript`, and
  `D`-generic grinding — with the CUDA path and the pinned static commitments
  costed, since both are affected. That is a `crypto/**` decision and therefore the
  user's call.

  Note §2.6 found `TreePoseidon`/`BatchPoseidonTree` already implementing
  `IsMerkleTreeBackend` over field elements, so the Merkle half has an
  implementation waiting. **Do not read "additive" into that** — whether the prover
  is generic over the trait or pins a concrete backend is exactly what `config.rs`
  pinning three aliases suggests it is not.
- **Widening `HASH_STATE_FELTS`** past 12 to cut the 2.125× rate penalty (§1.2).
  The contract is frozen and the bus tuples/opcodes are pinned; this is a team-lead
  decision, and it is the single cleanest lever on the candidate's `P`.
- **Raising `max_degree` above 3** to shorten the S-box. Interacts with the wrap's
  blowup 2; almost certainly a net loss, but it is a framework-ceiling question and
  rule "report a ceiling rather than working around it" applies.

### 3.3 TWO STAGES — and stage 1 must not block on stage 2's authorization

This is the governance shape the matrix should be built in, so that no column waits
on a `crypto/**` decision:

**Stage 1 — UNGATED, entirely inside `prover/src/lfm/**` (mine to build).**
A candidate's column factorises into two independently obtainable numbers:

```
column = permutations-per-verify  ×  cells-per-permutation
         └─ GEOMETRY: derived from the wave-7 census once rate/digest is
            normalized (§2.1 DONE, §2.4 normalization stated)
         └─ MEASURABLE: host a candidate AIR behind the socket and
            differential it against a reference implementation
```

Both halves are additive LFM work. **That yields measured-not-projected columns for
the whole matrix without touching `crypto/**` at all** — which is the point, because
it means the hash decision gets real numbers before anyone has to authorize anything.

**Stage 2 — GATED on the user's `crypto/**` call.** A genuinely candidate-hashed
inner proof, verified end to end. This validates stage 1's columns against reality
and is the only thing that makes the column a cryptographic claim rather than a
geometric one. It needs §3.2's proposal authorized first.

### 3.4 The measurement this buys, and what it does NOT buy

Slices A–E produce a **geometry** measurement: the true cell cost of an epoch
verifier that hashes with the candidate, at the real production shape. It is the
matrix column the phase asked for.

It does **not** verify a real candidate-hashed proof, because no such proof can be
produced without §3.2's inner-prover work. Stating that limit precisely is a rule-6
obligation: the column is *"cells to verify an epoch of this shape, hashing with
H"*, and its cryptographic content is the same as the placeholder's until an inner
proof under `H` exists. That is a fair trade for the decision the matrix feeds
(size), and a bad trade for any soundness claim.

---

## 4. Order of work, and the risks

**Recommended order:**

1. ~~**Slice A first — the `P` predictor.**~~ **DONE, in this session** (§2.1).
   Measured 1.63×, and it answered the question it was built to answer: the epoch
   is 58.6 % absorption-bound, so the frozen 12-felt state IS costing real
   permutations — but 1.63×, not the 2.125× ceiling. Widening the state is worth
   raising (§3.2) and is NOT urgent: it would recover at most 1.63 → 1.0, i.e.
   ~0.07 B cells of a ~1.9 B total (4 %), because the residue dominates once the
   hash is cheap. **That is a decision this measurement retires** rather than
   escalates.
2. **Then Poseidon-original** (slices B, C, E) — **but for a different reason than
   my first pass gave.** Not "cheapest column": §2.3 shows the algebraic candidates
   are within 7 % of each other on total cells, so cheapness is not a
   discriminator. The reasons that survive are: a **direct `poseidon1-air` donor**
   (§2.6.1), an in-tree HADES skeleton whose round structure is already confirmed,
   the **lowest build risk of any candidate**, and — decisively — the corpus has
   **zero AIR data** for it, so measuring it *adds* information instead of
   re-confirming a number Miden already published. It also validates socket 2 under
   real load for the first time.
3. **Then blake.** Its column is already measured externally (13.9×/48×), so
   building it mainly **calibrates our cost model against an independent
   measurement** — worth real money for trusting every other column. Budget it as
   the expensive build: bit-oriented, needs the lookup table (our `RANGE` can absorb
   the `And8Lookup` role), and Miden's own effort was 13→21 files / 3.8→5.1 k lines.
4. **Then Monolith** — `monolith-air` is a Goldilocks donor and the corpus never
   analyzed it, so this is the second-highest information-per-effort column.
5. **RPO last, despite being the cheapest predicted column.** It has no `rescue-air`
   donor anywhere (only Miden git history), its 0.5× is rows-only with unverified
   columns, and what it would buy over Poseidon-original is ~1 GiB of a ~49 GiB wrap.
   **Highest build risk for the smallest real difference.**

**A cheap cross-check available before any of this:** blake's measured 13.9×/48×
against Poseidon2 can be run through §1's census formula *today*, at our `P`. I did
exactly that in §2.3 and it is what produced the reversal in §2.5. Any candidate the
corpus has numbers for should get this treatment before it gets a build.

**Risks, worst first:**

- **The oracle problem for slice B — DOWNGRADED by §2.4, not eliminated.** A HADES
  permutation with Poseidon-original's exact round structure is already in-tree
  (`crypto/crypto/src/hash/poseidon/`), so the chip can be differentialled against
  a reviewed software reference rather than against itself. What is missing is a
  concrete `PermutationParameters` — and that is a *cryptographic* input, not an
  engineering one (next risk). Rule 7 still applies at the end: once the chip and
  the reference share a code path, the differential dies and must be replaced by an
  absolute property of the output.
- **Parameter selection is the real remaining risk, and it is not mine to make.**
  α, round counts, the MDS matrix and the round constants for the chosen field at
  `t = 12` must come from a published, reviewed source. Picking them ad hoc yields
  a measured column for a hash nobody would ship — decision-irrelevant, exactly the
  failure mode the Poseidon2 ban exists to avoid. Note the in-tree skeleton is
  field-generic (`type F: IsPrimeField`), so **which field the candidate is over is
  itself an open input** I did not resolve.
- **My round-count estimates are partly corroborated, not verified** (§2.2, §2.4).
  The HADES *structure* (R_F/2 · R_P · R_F/2) is confirmed from in-tree source; the
  specific 8-full/22-partial counts are still my own domain knowledge and set `m`.
  Cheap to fix: read the corpus, which I could not (below).
- **Four research legs never reported.** I dispatched agents for the inner-prover
  hash blast radius, the in-tree AIR inventory, the corpus's Part I.7 candidate
  data, and a full socket spec; none had returned when I closed. I covered the
  inventory myself (§2.4) and the socket myself (§0, §1.1) — the **corpus data and
  the inner-prover blast radius are the two genuine gaps in this report**, and both
  are cheap for wave 9 to close.
- **Parameter selection is a cryptographic act.** Round counts, MDS matrix and round
  constants for Poseidon-original over Goldilocks at t=12 must come from a
  published, reviewed source, not from me. Picking them ad hoc would produce a
  measured column for a hash nobody would ship — decision-irrelevant, exactly the
  failure mode the Poseidon2 ban exists to avoid.
- **The residue is not actually candidate-independent** (§1). It makes my numbers
  conservative, so it is a soundness-of-argument risk rather than a wrong-direction
  one, but a candidate column that quietly keeps keccak's byte-serialization
  gadgets in the residue would understate the win.
- **`TestPermutation`'s 37 cells/permutation is a trap.** It is one non-cryptographic
  degree-3 round. Anyone reading the census after slice A/E without reading §1.1
  could report a 2,108× win. Guard: the report and any test that prints it should
  carry the "NOT cryptographic" label the source does.


---

## 5. Corrections ledger — claims elsewhere that this document supersedes

Recorded so they stop propagating.

1. **The brief's "blake is the most probable final ship choice."** The review renders
   NO pick (§2.5). Ranked by evidence strength: Blake > RPO > Poseidon-original >
   Monolith. Scope all four; privilege none.

2. **My own "the hash decision may not buy the 2.8× the wrap needs."** Wrong — every
   candidate buys it, blake included (§2.3, §2.5). Right premise, unmultiplied
   arithmetic.

3. **The one-parameter 33.7 B/cell memory model — FALSIFIED** by wave 7's follow-up.
   Use the two-term fit: **≈27 B/cell + ≈190 MB/sub-proof**, and the keccak wrap
   ceiling is a **band, 290–350 GiB (2.3–2.8× the box)**, not a point. Every RSS
   figure in this document uses the two-term model. My earlier 59.8–67.8 GiB
   Poseidon figures were computed on the falsified coefficient and are superseded by
   §2.3's ~49 GiB.

4. **The RESUME's wave-7 line "one options change plus the hash swap."** Wrong on the
   options half: `ProofOptions` has **no hash field at all** (§3.2), so there is no
   options change to make — the swap is a `crypto/**` seam, always-stop, user's call.

5. **My own status-log implication that `chunking.rs`'s commit 6dbc5795 was the live
   agent's new work.** It is dated 2026-07-29 — the ORIGINAL chunking leg. At the
   time I looked there were **zero commits past 891f534f** on `feat/lfm-assembly`;
   the only new material was the uncommitted edit. The collision was real, my
   inference about which artifacts evidenced it was not.

6. **The KECCAK_RND chunk knob is TWO-SIDED and cannot buy the memory** (wave-7
   follow-up, measured by proving): retuning cut min-preset RSS 15.1 → 10.1 GiB but
   grew the proof +78 % (30.7 → 54.6 MB) and verify 2.4×; at the production shape
   padding waste is already 1.7 %. **A cheaper hash is the only large memory lever**,
   which is this document's motivation.

7. **§I.7's "hash choice decides batching" is CONTESTED in-corpus** — the guest-model
   version is falsified (the crossover constant is off ~5,500×, so unbatched wins on
   both axes at any hash price) and the native version is unmeasured. Do not import
   it. Relatedly, §I.7's "recursion diverges at a Blake-class hash" was reasoned at
   ~900,000 compressions; **this machine does ≈47,742** (§2.4), so it does not
   transfer.

8. **Arity-4 Merkle and layer-0 4-fold FRI are measured dead ends** on our economics
   (+7M net, and "DEEP doubling loses everywhere"). Arity trades permutations for
   bytes and so wins only when permutations are expensive — which the candidates make
   *less* true, not more.

9. **§2.3's whole candidate table — SUPERSEDED by §8.6** (`[hash-w10]`, MEASURED).
   Every candidate row held the 1,784,197,396 residue fixed, and that residue is
   overwhelmingly the `felt_be_halves` byteswap gadget, which the field-native
   candidates delete. ⚠ Base discipline (team-lead, post-eval catch; corrected by
   `[hash-w10]` before it could propagate wrong): the measured **95.84 %** is
   against the CHIP-HEIGHT CENSUS base of 1,757,982,868 (keccak permutation chips
   AND their lookup tables excluded; byteswap + R_native sum to it exactly);
   against the LEDGER base of 1,784,197,396 it is **94.44 %**. The 26,214,528
   between the bases is NOT an open reconciliation gap — it is exactly
   `BITWISE` (1,048,576 × (10 + 3×5) = 26,214,400) + `KECCAK_RC` (32 × 4 = 128),
   keccak's own lookup tables, which the ledger base includes. This CLOSES the
   gap `fma-vm-analysis.md:191` flagged. **R_native stays 73,072,788**: a
   field-native machine has no chip sending to any BITWISE-served bus
   (structurally checked — one hit in chips.rs, keccak's absorb XOR, LfmMem as
   positive control), so folding keccak's tables into its residue is a category
   error. Caveat: that costing assumes a CHIP-SET change; under the frozen-14-chip
   principle a field-native hash in today's set still carries BITWISE as a dead
   26.2 M table — a design decision, not a measurement. Do not quote
   "95.8 % of 1.784 B"; pick a base and name it. §2.3's "48–79 GiB, so the choice is not cost-gated" splits into ~71 GiB
   (blake, which keeps the gadget AND `BITWISE`) and ~4–8 GiB (algebraic). The
   "every candidate fits the 124 GiB box" half survives; the "choosing on predicted
   wrap size is choosing on noise" half does not, across families.

10. **The two-term RSS model's sub-proof count is 19 at this shape, not 24** —
    and it is candidate-dependent (§8.6). 24 is the INNER epoch's leg count; the
    per-sub-proof term is about the WRAP being produced, which carries 13 chip
    classes plus 6 `KECCAK_RND` chunks. Immaterial for keccak, but it is 27–41 % of
    the projection for the field-native rows, where it becomes the dominant term.

11. **"Delegation" is not an available lever for this machine** (§8.7, priced).
    A separate blake circuit plus verifying its proof costs +66 % over hosting the
    chip in the epoch verifier's own multi-proof. Airbender's delegation circuit
    exists to move hash work out of a FIXED-SIZE main circuit; the LFM has none,
    so its multi-AIR proof already is that pattern.

---

## 6. Slice 1a DONE — the permutation, and the pinned prediction for the chip

**Landed:** `prover/src/lfm/poseidon.rs` — `PoseidonGoldilocks` implementing
`LfmHasher`, with parameters and an external oracle.

### 6.1 Parameter provenance (condition (b), discharged)

From the vendored `others/Plonky3/goldilocks/src/poseidon1.rs` @ 4aed8fe4, which
documents them as Grain-LFSR generated per the Poseidon paper Appendix E with
`field_type=1, alpha=7 (exp_flag=0), n=64, t=12, R_F=8, R_P=22`, via
`poseidon/generate_constants.py --field goldilocks --width 12`. MDS is CIRCULANT
with first row `[1,1,2,1,8,9,10,7,5,9,4,10]` (`goldilocks/src/mds.rs:92`).

**This independently confirms my slice-0 estimate of 8 full + 22 partial**, which
was my own domain knowledge and is now cited. The corpus corroborates from a second
direction: ZisK's shipped PLONKish Poseidon is width-16, 8 full + 22 partial.

⚠ **Ship-grade parameter selection remains a separate cryptographic decision for the
ecosystem, NOT settled by this measurement.** Cells depend on round counts and S-box
degree, not on the constants' numeric values, so the measurement is valid; what to
ship is not ours. Likewise `compress_iv` is ZERO capacity — plain sponge
compression — and domain separation is deliberately not invented here.

### 6.2 The oracle (condition (d), discharged — but NOT as specified)

⚠ **The brief's first oracle is unusable and this is a real finding.** Condition (d)
asked for "the in-tree HADES skeleton instantiated with the same parameters". That
skeleton (`crypto/crypto/src/hash/poseidon/mod.rs`) hardcodes an `x^3` S-box, and
**`x^3` is not a permutation over Goldilocks**: `p - 1 = 2^32 · 3 · 5 · 17 · 257 ·
65537`, so 3 is not coprime to the group order. Differentialling against it would
have validated my implementation against a non-permutation.

Used instead: **Plonky3's own known-answer vector** for width 12 (input `0..11`),
which nothing in this repository produced. `the_permutation_matches_the_plonky3_
known_answer_vector` matched **on the first run**, with a Python cross-check of the
same convention beforehand.

**Falsified three ways (rule 1), each restored** — the KAT pins every convention it
needs to:
| mutation | result |
|---|---|
| `x^7 → x^6` (wrong exponent) | FAILED, correctly |
| circulant MDS transposed (`(j−i)` → `(i−j)`) | FAILED, correctly |
| partial-round S-box lane 0 → lane 11 | FAILED, correctly |

A second test asserts `gcd(α, p−1) = 1` and that 3 and 5 fail it — the skeleton's
bug, encoded as a guard.

### 6.3 PINNED PREDICTION for the chip — falsify this next

My degree-3 layout, one row per permutation (`x⁷ = (x³)²·x` needs `x²`,`x³` as
columns; the MDS is linear so it costs no columns):

```
IN0..11 + S8..11                      =  16
8 full rounds  × (12·x² + 12·x³ + 12 out) = 288
22 partial     × (x² + x³ + 12 out)       = 308
                                  m = 612 value columns, 1 row
                                  a = 3   (the chip's 6 LfmMem interactions)
base-equiv per permutation = 612 + 3·3    = 621
```

At the measured `P ≈ 192,000`: hash cells **121.5 M** with a chunking sibling
(1.019 padding) or **162.8 M** unchunked (pads to 2¹⁸, 1.365) — so the epoch verify
totals **1.906–1.947 B cells = 5.73–5.86× smaller than keccak**, RSS **≈50–51 GiB**.

⚠ **612 is an UPPER BOUND, and knowingly 2× off a known-achievable layout.** Miden's
measured Poseidon2 at the same width is 256 main + 16 aux = 304 base-equivalent, via
16 columns × 16 rows. A smarter layout could roughly halve my hash term — which
moves the TOTAL by ~3 %, because the residue dominates (§2.3). So the column is
worth measuring at 612 and not worth optimising.

### 6.4 Slice 1b — the chip, specified to be executable (NOT built)

**Why this is a spec and not code:** my context ran thin, and
`lfm-standing-decisions.md`'s coordination rule is explicit — "checkpoint and write
a handoff file rather than delivering a half-built slice. Quality over completion."
A 612-column constraint set that compiles but is unfalsified would be worse than
this document. Everything below is derived, not guessed; the arithmetic is checked
against §6.3.

#### Column layout (value section, after `PREP_WIDTH = 11`)

| block | columns | offset |
|---|---|---|
| `IN0..IN11` | 12 | 0 |
| `S8..S11` (capacity materialization) | 4 | 12 |
| per FULL round (×8): `x2[0..12]`, `x3[0..12]`, `out[0..12]` | 36 each | 16 + … |
| per PARTIAL round (×22): `x2`, `x3` (lane 0 only), `out[0..12]` | 14 each | … |
| **total value columns** | **612** | = 16 + 8·36 + 22·14 |

#### Constraints (601 total: 4 + 1 + 8·36 + 22·14)

Let `m = MODE_C + MODE_P` (the existing mode-sum column pair), and per round `r`
let `a_i = state_i + rc[r][i] · m` — an EXPRESSION, degree 1, where `state` is
`IN`/`S` on round 0 and the previous round's `out` afterwards.

1. **Capacity copy** (4): `S_i − MODE_P · IN_{8+i} = 0`. Degree 2. Note Poseidon's
   `compress_iv` is ZERO, so the `MODE_C · IV_i` term of the `TestPermutation`
   version vanishes — do not carry it over.
2. **Mode boolean** (1): `m · (1 − m) = 0`, unchanged from today.
3. **Per active lane**: `x2_i − a_i · a_i = 0` (degree 2) and
   `x3_i − x2_i · a_i = 0` (degree 2, since `x2_i` is a column).
4. **Per round output** (12 each): `out_j − Σ_i M[j][i] · f_i = 0` where
   `f_i = (x3_i)² · a_i` for S-boxed lanes (degree 3) and `f_i = a_i` otherwise
   (degree 1). `M[j][i] = MDS_CIRC_ROW[(i − j) mod 12]`, matching
   `poseidon::PoseidonGoldilocks::mds`.

**Degree is exactly 3**, so `max_degree()` stays 3 and the wrap's blowup 2 is
unaffected — the whole reason the S-box is decomposed rather than written `a^7`.

**Padding obligation, and it is already solved by the existing trick:** scaling the
round constant by `m` (as `chips.rs:548-553` does today) makes an all-zero row
satisfy everything — `m = 0 ⇒ a = 0 ⇒ x2 = x3 = out = 0` — WITHOUT a degree-4 gate.
Keep it; it is load-bearing, not decoration.

#### Trace generator contract

Mirror `poseidon::PoseidonGoldilocks::permute` but RECORD `x2`, `x3` and the
post-MDS state per round. It must use the **same association** —
`x2 = a·a`, `x3 = x2·a`, `x⁷ = (x3)²·a` — which is why `poseidon.rs::sbox` is
already written that way. Any other association gives the same field element and a
different trace, and the constraints would reject it.

#### Test plan (all five needed before the number is real)

1. `max_degree` measured ≤ declared, via the `CaptureBuilder` route
   `prover/src/tests/constraint_set_tests_a.rs:75-94` uses.
2. **Satisfaction**: a real Poseidon row (from the generator) makes every one of
   the 601 constraints evaluate to zero.
3. **Rejection**: perturb ONE column — one `x2`, one `x3`, one `out`, and one
   capacity cell, separately — and assert a constraint fires each time. Rule 1.
4. **Padding**: an all-zero row satisfies everything.
5. **Prove+verify** — rule 2: execute-only tests prove nothing about a chip, so the
   column is not MEASURED until the production prover runs this AIR. This is the
   step that makes §6.3's prediction a measurement.

#### Registration — much smaller than a new chip

`LFM_HASH` is **already** slot-registered (`airs.rs` `LFM_CHIP_NAMES`), so the
8-site checklist for ADDING a chip does not apply. What changes: `cols::NUM_COLUMNS`
(28 → 612 value columns), the constraint set body, the trace filler, and the census
picks the new width up automatically because it reads `hash::cols::NUM_COLUMNS`.
`PREP_WIDTH` stays 11 and the preprocessed group is untouched, so **the registry
root for this chip should NOT move** — verify that rather than assume it, and
regenerate `LFM_REGISTRY` if any digest shifts (pre-authorized).

⚠ **The one genuine hazard, and it is why this is not a small change:** the chips
bake the hasher's constants into their constraints, so `proof.rs:52-54` requires
execution to use the SAME hasher. Swapping `HashConstraints` to Poseidon therefore
breaks every existing call site that executes with `TestPermutation` (~30 across
`epoch_tests`, `constraint_tests`, `epoch_verify_tests`, `machine_tests`, and
`fixture::HostSponge`). **Do not do that swap to get a cells number.** The cells
number needs only the AIR's declared width plus tests 1-5 above; the global hasher
swap is a separate, larger decision about what the machine's default hash IS, and it
should be taken deliberately rather than as a side effect of a measurement.

### 6.5 What slice 1 still owes (superseded by 6.4 — kept for the index)

Not built: the chip's constraint block (replace `chips::hash::HashConstraints`'
`TestPermutation` round with the 30-round chain), `cols::NUM_COLUMNS` 28 → 612, the
census array, `LFM_REGISTRY` regeneration if any digest moves, and the prove+verify
measurement (rule 2: execute-only tests prove nothing about a chip). The padding
trap (condition (c)) must be handled as a chunking sibling OR an explicit
padding-corrected line beside the raw one — both numbers are pinned above so the
first measurement cannot silently read 36 % high.

---

## 7. Slice 1b DONE — the chip is BUILT, PROVED, and the prediction CONFIRMED

**Landed** (`[hash-w9]`): the Poseidon-original `LFM_HASH` chip behind a
construction-time hasher choice, plus 15 tests. `lfm` suite **230 passed / 0
failed / 5 ignored**, `make lint` exit 0 (make's own status).

### 7.1 The measurement, number by number against §6.3

Every figure below is measured through the SAME census instrument that produced
entry 10's keccak column (`main + 3·aux`), so the two columns of the matrix are
comparable by construction rather than by argument.

| §6.3 pinned | measured | verdict |
|---|---|---|
| 612 value columns | **612** | CONFIRMED |
| 601 constraints | **601** | CONFIRMED |
| `max_degree` 3 | **3** declared, **3** measured max | CONFIRMED |
| 621 base-equiv cells/permutation | **621** (612 + 3·3) | CONFIRMED |
| hash cells 121.5 M chunked | **121,497,408** | CONFIRMED |
| hash cells 162.8 M unchunked | **162,791,424** (2^18 pad = 1.365×) | CONFIRMED |
| epoch total 1.906 B chunked | **1,905,694,804** = 5.86× under keccak | CONFIRMED |
| epoch total 1.947 B unchunked | **1,946,988,820** = 5.74× under keccak | CONFIRMED |

⚠ **Provenance, because only one of these inputs is new.** Wave 9 measured
exactly one number: **621 cells per permutation**, off an AIR the production
prover built and the production verifier accepted. `P` (wave 8's closed form)
and the 1,784,197,396 residue (entry 10) are inherited measurements; the epoch
lines are arithmetic over all three. Across wave 8's whole `P` interval
[190,569, 193,569] the chunked total moves only 1,904.8 M → 1,906.7 M
(5.856–5.862×), so the conclusion does not depend on `P` being exactly 192,000.

### 7.2 ⚠ CORRECTION — the RSS figure, and it is a units error, not a cells error

§6.3's "RSS ≈50–51 GiB" does **not** reproduce from the stated two-term model
(27 B/cell + 190 MB/sub-proof) over this epoch's 24 sub-proofs:

- cell term alone: 47.92 GiB chunked / 48.96 GiB unchunked — **but 51.5 / 52.6
  GB**, which is almost certainly where "50–51" came from: the cell term
  computed in GB and labelled GiB, with the sub-proof term dropped.
- both terms, in GiB: **52.2 GiB chunked / 53.2 GiB unchunked**.

Use **≈52–53 GiB**. Nothing downstream changes — every figure in the band is
far inside the 124 GiB box, which is the only claim the number carries — but
the ~4% understatement is recorded so the matrix's other rows (which were
computed the same way in §2.3) get re-derived before anyone compares them at
that precision.

### 7.3 The prediction that was NOT confirmed, and it was never a cells claim

§6.3 called 612 "an UPPER BOUND, knowingly 2× off Miden's measured 304". That
is untouched by this measurement: 612 is what MY layout costs, and a row-per-
round layout reusing state columns would still be roughly half. The instruction
not to optimise it stands for the reason given — halving the hash term moves the
epoch total from 1.906 B to 1.846 B, i.e. **3.2%**, because the residue
dominates at 93.6% of the chunked total. The hash term is no longer the thing
worth engineering.

### 7.4 What the chip actually is

- **Layout.** The frozen `IN`/`S`/`OUT` prefix keeps its offsets and the final
  round's post-MDS output IS `OUT`, so `bus_interactions()` is
  hasher-independent and the `LFM_HASH` tuple contract stays literally frozen:
  `28 + 7·36 + 24 + 22·14 = 612`, the same 612 as §6.4's `16 + 8·36 + 22·14`
  arranged differently. Both totals are asserted, against each other and against
  the built width.
- **Degree exactly 3**, via `x⁷ = (x³)²·x` over witnessed `x²`/`x³`. Asserted
  both ways: nothing exceeds 3, and something reaches it — a decomposition that
  quietly went quadratic would mean the S-box had stopped being computed.
- **The padding trick is load-bearing, now demonstrated rather than asserted.**
  See F4 below: removing the round constant's mode-sum scaling breaks the
  padding row and NOTHING ELSE, because on a real row `m = 1` and the
  permutation is unchanged. That is the cleanest possible evidence for §6.4's
  "keep it; it is load-bearing, not decoration".

### 7.5 Falsification (rule 1) — four mutations, each of the CHIP only

Mutating chip *and* executor together proves nothing: they would move as one.
Each mutation below changes only the constraint body, so the chip stops agreeing
with the permutation the external Plonky3 KAT pins. Instrument checked against a
known-green control first, and failures read from the trailing summary block
(per rule 7's corollary, per-test lines do not name failures).

| mutation | result |
|---|---|
| F1 `x⁷ → x⁵` in the chip | 5 failed, incl. prove+verify |
| F2 circulant MDS transposed `(i−o) → (o−i)` | 5 failed, incl. prove+verify |
| F3 partial-round S-box binds lane 1, not lane 0 | 5 failed, incl. prove+verify |
| F4 round constant no longer scaled by the mode sum | **exactly 2 failed**: the padding row and prove+verify |
| CONTROL (unmutated) | 21 passed, 0 failed |

F4's *discrimination* is the interesting one — satisfaction, the KAT-output
check and every rejection test stay green, so the padding trick is isolated to
padding exactly as §6.4 claimed.

### 7.6 The seam, and the two things asserted rather than assumed

Per the team lead's ruling the hasher is a **construction-time** choice
(`HasherKind`) threaded to the constraint body, the width, the trace filler and
the executor. `Test` remains the default; every pre-existing call site keeps its
signature and its behaviour. **Nothing was flipped** — the machine's real hash is
the ecosystem decision this measurement feeds.

1. **No program digest moves with the hasher.** `PREP_WIDTH` is 11 in both
   layouts and the preprocessed group is untouched, so every root and every
   program id is bit-identical, and the census's row counts and aux widths are
   too — only `LFM_HASH`'s value width moves. `LFM_REGISTRY` did not need
   regenerating. Asserted, because a hash experiment silently reassigning
   program identity is exactly the failure that must not pass quietly.
2. **A proof does not verify under the other hasher**, in both directions.

### 7.7 What this leg does NOT settle

It does not choose a hash, and it is not evidence that Poseidon-original should
be the machine's default. Parameters are published ones adequate to measure an
AIR's *shape*; `compress_iv` is zero because domain separation is a
cryptographic decision deliberately not invented here. Per §0c the decision is
not cost-gated anyway — every candidate fits the box — so this column adds
information (the corpus had zero Poseidon AIR data) without rendering a pick.

---

## 8. Slice 2 DONE — the blake column MEASURED, and the residue turns out to BE the byteswap

Wave 10 (`[hash-w10]`), 2026-08-06. Donor: PR #903 `feat(prover,executor): BLAKE3
6-round compression accelerator`, head **`89aeeb8c2b0389e9d21a861c9e3a10a7b1b5704e`**.

**Landed:** `prover/src/lfm/blake3.rs` (the primitive + the 10 canonical 6-round
vectors + negative controls), `prover/src/lfm/blake3_chip.rs` (the chip, hosted
on `LfmMem`), `prover/src/lfm/blake3_probe.rs` (prove+verify, falsification, and
two `#[ignore]`d measurement instruments). `lfm` suite **244 passed / 0 failed /
5 ignored** (was 230/0/5), `make lint` exit 0 (make's own status).

### 8.1 The headline, and it is not the column

The blake column came out where §2.3 predicted (3.7–4.1× under keccak against a
predicted 3.68×). **The finding that matters is the one the column was measured
against: the 1,784,197,396-cell "non-hash residue" that every candidate row in
§2.3 sits on is 95.8 % a single gadget** — `felt_be_halves`, the felt →
big-endian-u32-halves serializer, at 1,684,910,080 cells.

§1's warning was right and an order of magnitude too quiet. It said "part of that
residue is byte-serialization work … a field-native hash deletes outright" and
concluded the candidate predictions were therefore conservative. They were
conservative by **~10×**, not by a few percent, and §2.3's robustness note —
"choosing among the algebraic candidates on predicted wrap size is choosing on
noise" — is now **false in the one comparison the decision turns on**: blake and
the field-native candidates are ~11× apart, not 1.6×.

### 8.2 The measurement: cells per compression, on our stack

MEASURED, prove+verify (rule 2), `blake3_probe::the_hosted_chip_proves_and_verifies`:
the chip is built with its real 1,259 interactions and its real 769 constraints,
its preprocessed prefix is committed for real via `commit_columns`, and both its
buses are closed — `ByteAlu`/`AreBytes` against the UNCHANGED production
`BITWISE` table, `LfmMem` against a mirror AIR.

| | #903, syscall variant | hosted here | basis |
|---|---|---|---|
| value columns | 3,219 | **3,056** | MEASURED |
| bus interactions | 1,397 | **1,259** | MEASURED |
| aux columns (`⌈i/2⌉`) | 699 | **630** | DERIVED |
| **base-equiv `m + 3a`** | **5,316** | **4,946** | **MEASURED** |
| constraints | 814 | **769** | MEASURED |
| max degree | 3 | **3** declared, **3** measured | MEASURED |

5,316 reproduces #903's own stated figure exactly, which is the corroboration
that the two are being counted the same way.

**The 370-cell (7.0 %) saving is entirely I/O.** Dropped: the `Ecall` receiver,
the x10 register read, 22 `Memw` dword ops, the 32 `OLD_OUT` `AreBytes`, 4 addr
`AreBytes` + the alignment `AND`, and 88 pointer `IsHalfword`s (149 interactions);
and the `TIMESTAMP`/`ADDR`/`PTR`/`OLD_OUT` columns (162), plus `MU` moving into
the preprocessed prefix. Added: 11 `LfmMem` word tokens (7 reads of the 28 input
`u32`s, 4 writes of the 16 output `u32`s), the machine word being four `u32`
lanes exactly as `LFM_KECCAK` defines it.

⚠ **Dropping those range checks is sound, not merely cheaper, and the argument is
worth keeping.** Each guarded something that no longer exists: `OLD_OUT` is the
previous memory content in a `Memw` write's `old` field and an `LfmMem` write has
none; the address checks guard a prover-witnessed pointer read out of x10, where
here every address is a preprocessed column the admission validator vouches for.
The byte-range coverage of the DATA columns is untouched — `m`'s 64 bytes keep
their 32 explicit `AreBytes` (they are never XOR-consumed), `h` is an operand of
the feed-forward XOR, `t_lo`/`t_hi`/`block_len`/`flags` are `v[12..16]` and hence
`vd` operands of round-0 `G`s, and all 64 `OUT` bytes are XOR *results*. So every
byte a token recomposes is range-checked before the recomposition, which is the
same transitive argument `chips::keccak` records for its 400 state bytes.

**Basis label: hosted-measured, not registered.** `LFM_BLAKE3` is not in the
fixed AIR set — `airs.rs` still names 14 chips — so what is proved is the chip
under our AIR framework with its buses closed by a synthetic memory, not an
epoch verifier that hashes with blake. Registration would move every program
digest and is a separate decision. What the probe therefore *cannot* see is
listed in its module doc: whether an LFM program can drive the chip, whether the
validator accepts the address assignment, and anything cryptographic about A6R.

### 8.3 The geometry: blake's `P` is the rate-8 count, and no extra compressions

VERIFIED against #903's ABI. A 2-to-1 Merkle compression is `compress(h = IV,
m = left‖right)` — 64 bytes of message, ONE compression, 1:1 with keccak's
64-bytes-inside-a-136-byte-rate parent. An absorb of `N` felts is `⌈N/8⌉`
compressions at 8 bytes per felt. The counter `t`, `block_len` and `flags` live
in `v[12..16]`, i.e. in the *state*, not in message space, so **the message-mode
framing forces no extra compressions**. Blake therefore shares the field-native
candidates' `P`, re-derived on this run rather than quoted:

```
legs @ rate 17 (keccak)          115,413   = the EMITTED count, exactly (assert)
legs @ rate  8 (blake and field-native) 187,902
spine                              2,667   absorption-bound, so 1.0x–2.125x
P at rate 8 in [190,569 , 193,570]         — §2.1's interval, reproduced
```

⚠ One conservatism carried deliberately: `blocks_at_rate` uses keccak's
`⌊n/rate⌋ + 1` padding convention, which always spends a trailing block. BLAKE3
signals length in `block_len` and needs none, so blake's true count is between
`⌈N/8⌉` and this. Using the same convention on both sides is what makes the
rate-17 case reproduce the emitted count exactly, so it is kept and the
direction recorded: **blake's `P` here is an upper bound.**

### 8.4 The residue, split — MEASURED

Instrument: `blake3_probe::the_blake_column_and_the_residue_split` (`#[ignore]`d;
proves a real inner epoch at blowup 8 and emits ~2.25M instructions). Epoch
`[2 ×14, 3, 4 ×4, 5 ×3, 7, 20]`, inner blowup 8 / 73 queries, **19 sub-proofs**.

```
LFM_BALU     134,217,728 rows ×  4 main /  2 aux   1,342,177,280   12.02%
LFM_BITDEC     2,097,152 rows × 66 main / 33 aux     346,030,080    3.10%
LFM_LANES      2,097,152 rows ×  4 main /  5 aux      39,845,888    0.36%
KECCAK_RND     2,883,584 rows ×1480 main /576 aux   9,250,537,472   82.85%
LFM_KECCAK       131,072 rows ×736 main / 88 aux     131,072,000    1.17%
BITWISE        1,048,576 rows × 10 main /  5 aux      26,214,400    0.23%
(+ 8 more chips, 0.63% between them)                ------------
TOTAL                                              11,165,806,868
```

The keccak permutation chips come to **9,381,609,600**, which reconciles §1's
9,381,609,472 to within `KECCAK_RC`'s 128 cells — so §1's residue of
1,784,197,396 was `total − LFM_KECCAK − KECCAK_RND` and **included the `BITWISE`
table**. Stated cleanly:

| | cells | basis |
|---|---|---|
| keccak permutation chips (`LFM_KECCAK`+`KECCAK_RND`+`KECCAK_RC`) | 9,381,609,600 | MEASURED |
| `BITWISE`, fixed 2²⁰ (blake keeps it, field-native deletes it) | 26,214,400 | MEASURED |
| residue | 1,757,982,868 | MEASURED |
| — of which the byteswap gadget | **1,684,910,080 (95.84 %)** | MEASURED |
| **residue, byte-oriented** (blake: gadget + `BITWISE` kept) | **1,757,982,868** | MEASURED |
| **residue, field-native** (both deleted) | **73,072,788** | DERIVED |

**How the byteswap share is counted, and why it is exact rather than attributed.**
`felt_be_halves` is one `BitDec(64)` plus 64 `BALU` rows per felt
(`machine_tests::felt_be_halves_cost` pins that). Every other production
`bit_dec` site passes 32 bits or a Merkle depth — `sample_u64_pow2` *asserts*
`nbits ≤ 32` — so a 64-bit decomposition in this program IS the gadget. The
instrument prints the whole width histogram so a future 64-bit caller shows up
instead of being silently folded in:

```
BitDec widths: {4: 1022, 5: 73, 6: 292, 7: 219, 9: 73, 22: 73, 32: 398, 64: 1,122,145}
```

1,122,145 gadget calls ⇒ 71,817,280 `BALU` rows, **99.78 % of all `LFM_BALU`
rows**. Padding-aware: the two chips cost 1,688,207,360 with the gadget and
3,297,280 without, hence the 1,684,910,080.

⚠ **The field-native line is DERIVED by subtraction from a keccak-shaped
emission**, not measured on a re-emitted field-native verifier, and it is an
**upper** bound: a field-native absorb also deletes Pack/Unpack traffic around
the gadget, and `LFM_LANES` still costs 39,845,888 here (55 % of the whole
field-native residue).

⚠ **The gadget's cost is structural given the current ISA, not an endianness
accident.** Big-endian order is what forces the 32-term weighted recombination,
but *any* felt → two-`u32`-halves split needs a range-checked decomposition, and
the LFM has no 32-bit range-check instruction (`LFM_RANGE` is a 2¹⁶ table that
`chips::range`'s own comment calls "idle in v0"). Wiring one would be the lever;
it is unbuilt and unmeasured and is NOT assumed anywhere above.

### 8.5 ★ A cheap lever nobody has pulled: chunk `LFM_BALU`

`LFM_BALU` has 71,974,504 real rows and pads to 2²⁷ = 134,217,728 — an **86 %
overshoot**, 622 M cells of pure padding. `LFM_BITDEC` pads 1,124,295 → 2,097,152
for another 161 M. Together **≈783 M cells, 7.0 % of the keccak total and 28 % of
blake's**, recoverable by the chunking policy `KECCAK_RND` already has
(`airs.rs`'s chunk machinery is generic; nothing about it is keccak-specific).
DERIVED from this run's census; not attempted, and it does not move the
field-native rows, whose `BALU` is tiny.

### 8.6 The matrix, RE-DERIVED — and §2.3's rows are superseded

`P = 192,000`; hash cells are `rows × cells-per-permutation` with rows either
chunked (≈1.9 % waste, the `KECCAK_RND` policy) or padded to the next power of
two (36 % waste); RSS is the two-term model with the **candidate's own sub-proof
count** (see the correction below).

| candidate | basis of cells/perm | hash cells | **total cells** | vs keccak | subs | **RSS GiB** |
|---|---|---|---|---|---|---|
| **keccak — MEASURED, ours** | 77,992 | 9,407,824,000 | **11,165,806,868** | 1.00× | 19 | **284.1** |
| **BLAKE3-6r, chunked** | **4,946 MEASURED** | 967,402,978 | **2,751,600,246** | **4.06×** | 12 | **71.3** |
| BLAKE3-6r, padded | 4,946 MEASURED | 1,296,564,224 | 3,080,761,492 | 3.62× | 12 | 79.6 |
| **Poseidon-original, chunked** | **621 MEASURED (w9)** | 121,463,253 | **194,536,041** | **57.4×** | 10 | **6.7** |
| Poseidon-original, padded | 621 MEASURED (w9) | 162,791,424 | 235,864,212 | 47.3× | 10 | 7.7 |
| RPO, chunked | 152 INHERITED est. | 29,730,136 | 102,802,924 | 108.6× | 10 | 4.4 |
| Monolith, chunked | ~850 INHERITED est. | 166,254,050 | 239,326,838 | 46.7× | 10 | 7.8 |

**What changed and why.** §2.3 put every candidate in a 48–79 GiB band and
concluded the choice was not cost-gated. The first half survives — **every
candidate still fits the 124 GiB box** — but the band was an artefact of holding
the byteswap gadget fixed across rows that delete it. The real spread is
**4.4 GiB to 71 GiB, and blake is ~11× the field-native candidates**, so wrap
size *is* a discriminator between the byte-oriented and the algebraic families
(though still not among the algebraic ones, where §2.3's noise argument holds:
RPO, Poseidon and Monolith differ by 2.4× on cells-per-permutation and land
within 1.8× on total).

⚠ **Correction to the sub-proof count, which wave 9 and §2.3 both got wrong.**
The two-term model's per-sub-proof term is about the proof being *produced* — the
wrap. At this shape the wrap has **19** sub-proofs (13 chip classes + 6
`KECCAK_RND` chunks), not 24; 24 is the INNER epoch's leg count. It is also
candidate-dependent: blake drops three keccak-family chips and adds one (≈12),
field-native drops four including `BITWISE` (≈10). At keccak's scale this moves
nothing, but for the field-native rows **the sub-proof term is 27–41 % of the
projection** — it is the dominant term there, which makes those the weakest RSS
numbers in the table.

⚠ **Both RSS coefficients were calibrated on keccak-shaped runs** — a machine
whose largest tables are a 1,480-column round chip and a 2²⁰-row lookup table.
Nothing has checked that 27 B/cell survives a machine whose widest table is a
3,056-column single-row chip, still less one with no lookup table at all. Every
GiB figure above is a projection carrying that caveat.

For the record, §2.3's own rows recomputed with BOTH terms in GiB at its stated
24 sub-proofs (the wave-9 erratum discharged — the gap is the dropped sub-proof
term against a GB-labelled-GiB cell term, which happened to cancel to ~2 %):

| §2.3 row | its cells | cell term GiB | + sub-proof term | §2.3 printed |
|---|---|---|---|---|
| RPO | 1.814 B | 45.6 | **49.9** | 48 |
| Poseidon-original | 1.844 B | 46.4 | **50.6** | 49 |
| Poseidon (conservative) | 1.905 B | 47.9 | **52.1** | 50 |
| Monolith | 1.951 B | 49.1 | **53.3** | ~52 |
| BlakeG 32-row | 3.037 B | 76.4 | **80.6** | 79 |
| keccak | 11.166 B | 280.8 | **285.0** | 284 |

These are corrected in place but **superseded** by the table above: their cell
totals all carry the byteswap gadget.

### 8.7 The delegation topology — priced, and it is a net LOSS here

User request: price blake3 in a SEPARATE specialized circuit (Airbender's
pattern — their blake2s delegation circuit does ~19 proofs' Merkle work in one
2²⁰ instance) against in-trace hosting. Instrument:
`blake3_probe::the_delegation_topology_priced_against_in_machine_hosting`,
arithmetic over the same closed form the epoch's own permutation count comes
from. Inputs INHERITED from the epoch's 2²⁰ leg: 2 composition parts, 73 queries,
198 FRI compressions per query, blowup 8.

```
IN-MACHINE   LFM_BLAKE3 as one more AIR of the epoch verifier's multi-proof:
             192,000 rows x 4,946 = 967,402,978 cells. Nothing else changes.

DELEGATED    (a) the delegation proof's own trace (LFM_BLAKE3 + its BITWISE)
                                                        993,617,378 cells
             (b) verifying that proof inside the epoch verifier:
                 LFM_BLAKE3 AIR (2^18 rows, 3,056 main + 630 aux)
                                        1,523/query x 73 = 111,179 compressions
                 BITWISE AIR    (2^20 rows, 10 main + 5 aux)
                                          298/query x 73 =  21,754 compressions
                 = 132,933 compressions = 657,486,618 extra cells, ON TOP of (a)
```

**Verdict: delegation costs (a) + (b) where in-machine costs (a) alone — a net
loss of 657 M cells, +66 %.** The reason is structural rather than a tuning
accident. What Airbender's delegation circuit buys *them* is moving hash work out
of a **fixed-size** main circuit (a 2²⁰-cycle RISC-V trace) whose cycles the
hashing would otherwise consume. **The LFM has no fixed-size box**: every chip's
height is program shape, and the proof is already a multi-AIR proof over
independently-sized tables connected by a bus. Our architecture *is* the
delegation pattern; a second proof only adds a verification.

The term that makes (b) expensive is the leaf term, the same one §2.4 flagged as
having no analogue in cross-system 2-to-1 figures: a 3,056-column AIR has a
6,112-felt main leaf, which is 765 compressions to absorb, 73 times per query.
**A delegation circuit is wide by construction, and wide traces have expensive
leaves** — so the wider and more efficient you make the delegated chip, the worse
its proof is to verify.

Two variants considered and priced the same way. *Batching K epochs' compressions
into one instance* (the literal Airbender shape) saves the fixed `BITWISE` table
K−1 times — 26.2 M cells each, 468 M at K = 19 — but still pays (b) once, so it
is a loss until K ≳ 25 and it gives up one-proof-per-epoch. *Padding
amortisation* buys nothing: 192,000 pads to 2¹⁸ and 384,000 to 2¹⁹, the same
36 % either way, and chunking already fixes it (§8.5).

⚠ This prices CELLS only. It cannot see prover wall time, proof size on the
wire, or the engineering cost of a second circuit and its glue — and those are
where a delegation argument would have to be made if anyone wants to remake it.

### 8.8 Falsification (rule 1) — chip-only mutations, control-validated

Mutating the chip *and* the primitive together would prove nothing, so each
mutation below changes only `blake3_chip.rs`, leaving `blake3.rs` — pinned by the
canonical vectors — intact. That works because the probe's mirror AIR computes
its `LfmMem` words from `blake3::blake3_compress_6round`, which is an
INDEPENDENT implementation of the compression: the bus is a genuine differential
between the primitive and the chip's own dataflow, and neither delegates to the
other (rule 7's trap avoided deliberately).

Failures read from the trailing summary block, and a green control was run first
and again after restoring (rule 7's corollary).

| mutation | result |
|---|---|
| CONTROL (unmutated) | 14 passed, 0 failed |
| F1 `rotr8`'s free byte relabel transposed in the WIRE interpretation only | **exactly 2 failed**, both prove+verify |
| F2 message schedule transposed in the chip's `run_flow` (`sched[p] = prev[i]`) | **exactly 2 failed**, both prove+verify |
| F3 the `LfmMem` read multiplicity ungated (`Column(MU)` → `One`), so padding rows read | **exactly 2 failed**, both prove+verify |
| CONTROL again (restored) | 14 passed, 0 failed |

The *discrimination* is the useful part: in all three the only casualties are
`the_hosted_chip_proves_and_verifies` and the control, while the six trace-tamper
tests and the layout/degree tests stay green — so the mutations are isolated to
what the proof sees, which is what a chip-only falsification is supposed to show.

⚠ **A fourth mutation is recorded because it FAILED to be a falsification.**
Changing `ROT_SHIFT_R` from `[4, 9]` to `[4, 10]` made 8 tests fail in 0.01 s —
`ValueFlow::rot_shift`'s `debug_assert_eq!` panics before any proof is built. The
mutation *is* caught, but by an assert, not by prove+verify, so it is evidence
about the debug assert and not about the constraint set. Reported rather than
quietly replaced: a falsification harness that counts a panic as a constraint
rejection would be exactly the "my mutation changed nothing" instrument bug rule
7's corollary warns about, inverted.

Six trace-tamper tests back the chip-only set: a flipped OUT byte, a flipped
message byte (the one that would go green if the 32 message `AreBytes` were ever
dropped as redundant), a flipped add3 carry bit, a padding row turned real, a
bumped read multiplicity, and the all-zero-padding assertion.

### 8.9 Provenance of the primitive, and why rule 9 is discharged differently

Rule 9 wants an EXTERNAL known-answer vector that nothing in this repository
produced. **That is impossible in the usual form here**: the 6-round variant is
not standard BLAKE3, so no published vector and no crate exposes it. The chain
#903 supplies, recorded rather than waved at:

1. a z3-proved model of the compression dataflow (`z3_blake_verify.py`);
2. a Python oracle (`blake3_ref.py`) whose **7-round** instantiation is pinned
   against the official `blake3` crate's published vectors — so the G-function,
   message schedule, counter split and feed-forward are externally validated and
   only the round count varies;
3. that oracle at `rounds = 6` emitting the 10 canonical vectors this port is
   pinned against.

The external anchor is therefore one step removed. To check that the vectors
nevertheless *discriminate* rather than merely being reachable,
`breaking_one_convention_at_a_time_breaks_the_vectors` runs a parameterised
control at four broken conventions — rotr12→rotr13, rotr16↔rotr8 (the cheapest
possible error, since both are free byte relabels in the chip), the message
schedule transposed, and 7 rounds instead of 6 — and each stops reproducing
vector 0. A fifth test shows the counter's two halves are not interchangeable.

⚠ Security assumption **A6R** (6-round collision resistance) is named and
unratified. Nothing here ratifies it; this leg prices the AIR.

### 8.10 What this leg does NOT settle

- It does not register `LFM_BLAKE3`, so no epoch verifier has ever hashed with
  blake. The column is "cells to verify an epoch of this shape if it hashed with
  H", the same limit §3.4 states for every column in this matrix.
- The field-native residue is a subtraction, not a re-emission (§8.4).
- It does not choose a hash. It does sharpen the choice: the decision is no
  longer between candidates that are all within 1.6× on size, but between a
  byte-oriented family at ~71 GiB and an algebraic family at ~4–8 GiB — with the
  byte-oriented family holding the only real in-AIR measurements and a shipped
  production existence proof, and the algebraic family holding the size.

### 8.11 ★ Reconciliation against `hash-delegation-eval.md` §3.1 / §4.1

Requested by the team lead, who put the byteswap's CELL share near ~50% against
the eval's instruction-derived 88–93%. **Both figures are right, about different
quantities, and the whole gap is a padding term neither included.**

```
byteswap share of the residue, UNPADDED closed form   903,326,725 = 51.38%   <- the ~50% estimate
byteswap share of the residue, PADDING-AWARE        1,684,910,080 = 95.84%   <- what the machine pays
padding multiplier on the gadget                                    1.865x
```

The gadget is what DRIVES `LFM_BALU` to 2²⁷ (71,974,504 real rows) and
`LFM_BITDEC` to 2²¹ (1,124,295 real rows), so removing it does not remove
`rows × width` — it removes two padded power-of-two tables. The team lead's
"`BALU` rows are cheap at 10 base-equiv" is correct and is exactly why the
unpadded number is ~50 %; what it misses is (a) the gadget's other half,
`LFM_BITDEC`, at **165** base-equiv per row — 20 % of the unpadded cost from 1.6 %
of the rows — and (b) the padding. **The measurement is the padding-aware one: a
machine that does not byteswap does not build those tables at all.**

Item by item against the eval's own numbers:

| eval claim | MEASURED | verdict |
|---|---|---|
| §3.1 byteswap = 88–93 % of residue | **95.84 %** (cells) | eval LOW by 3–8 pts, right conclusion |
| §3.1 `R_native` = 0.024–0.20 B, central 0.059 B | **0.073 B** (73,072,788) | **inside the band**, 24 % above central |
| §3.1 "residue collapses ~10–70×, central ~30×" | **24.1×** | inside the band |
| §4.1 Poseidon in-trace + `R_native` = 0.145–0.32 B, c. **0.18 B** | **0.195 B** | **✅ SURVIVES**, 8 % above central |
| §4.1 BLAKE3-6 in-trace + `R_native` = 1.06–1.24 B, c. **1.10 B** | **2.752 B** | **❌ DOES NOT SURVIVE as built — 2.5× out** |
| §4.1 hosted chip "should land near ~5,150 (≈3 % lower)" | **4,946 (7.0 % lower)** | direction right, magnitude 2.3× understated |
| §4.1 BLAKE3 hash cells 1,040,064,768 | **967,402,978** | −7.0 %, follows from the line above |
| §3.1 open: "does an LFM-hosted BLAKE3 chip shed the byteswap? It should" | **NO** | ❌ **the unfavourable answer** |

**Why blake does not shed it, and it is not an implementation choice I made.**
The chip consumes machine words of four `u32` lanes — the `LFM_KECCAK` convention
— and `u32` halves are precisely what `felt_be_halves` produces. The gadget is
UPSTREAM of the chip's input format, so hosting the chip cannot delete it. Blake
therefore pays the byte-oriented residue AND keeps `BITWISE`.

**But the eval's 1.10 B is recoverable, and the convergence is exact.** A variant
that receives full 64-bit felts and decomposes them to bytes inside its own
constraints does shed the gadget. It needs one thing the eval's sketch omits: a
**canonicity gate per absorbed felt**. `Σ byteₖ·256ᵏ = v` over the field does NOT
pin the byte string — `v` and `v + p` both satisfy it — so without a `< p`
argument the prover chooses what gets absorbed and Fiat–Shamir breaks. (That is
why `felt_be_halves` routes through `bit_dec`, whose doc says outright: "`bit_dec`
also enforces canonicity (`< p`) … production renders `canonical_u64()`" —
VERIFIED, `transcript_replay.rs:735-736`.) A borrow-chain `< p` gate at degree 3
is small against a 3,056-column chip; at my ESTIMATE of ~156 extra base-equiv per
compression (8 absorbed felts × ~20):

```
felt-absorbing BLAKE3 (ESTIMATE, UNBUILT): 5,102/compression + R_native + BITWISE
                                         = 1,097,202,674 = 1.097 B, 10.2x, ~30 GiB
eval §4.1 central                        = 1.10 B
```

**So the 2.5× discrepancy is not an arithmetic disagreement — it is precisely the
value of the unbuilt felt-absorbing variant, ≈1.65 B.** The eval priced a design;
I measured the one that exists. Both numbers should be carried, labelled.

⚠ **I did not build it, and the reason is a cryptographic decision, not effort.**
The input side is engineering; the OUTPUT side is not. A blake output word is 32
bits, so a felt built from 8 output bytes is a 64-bit value reduced mod `p` and
the map is not injective. How a blake digest becomes felts — truncate to four
`u32`s, reduce, domain-separate — changes the security argument, the digest
width, and the token count. That is the ecosystem's call, the same boundary
§6.1/§7.7 draw around Poseidon's parameters and `compress_iv`.

**Net effect on the eval's verdict.** Its §4.1 conclusion strengthens rather than
weakens: the field VM's fully-delegated floor is `R_native` + the stub, measured
at **0.073 B + ~7.7 M ≈ 0.081 B** against its predicted 0.067 B central — and its
"delegation's field-side win over a field-native algebraic hash is 0.113 B / 63 %"
becomes **0.195 − 0.081 = 0.114 B / 58 %**, i.e. essentially unchanged. What
changes is the in-trace blake row it is competing against, and only for the
variant nobody has built.

⚠ And one finding of mine that bears directly on the eval's scheme (§8.7):
**delegation as a SEPARATE PROOF is a net loss of +66 % on this machine.** The LFM
has no fixed-size main circuit to escape — every chip's height is program shape —
so its multi-AIR proof already IS the delegation pattern, and a second proof only
adds a verification whose leaf term is large precisely because a delegated hash
chip is wide.
