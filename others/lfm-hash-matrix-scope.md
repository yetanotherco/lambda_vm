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

### 6.4 What slice 1 still owes (handoff)

Not built: the chip's constraint block (replace `chips::hash::HashConstraints`'
`TestPermutation` round with the 30-round chain), `cols::NUM_COLUMNS` 28 → 612, the
census array, `LFM_REGISTRY` regeneration if any digest moves, and the prove+verify
measurement (rule 2: execute-only tests prove nothing about a chip). The padding
trap (condition (c)) must be handled as a chunking sibling OR an explicit
padding-corrected line beside the raw one — both numbers are pinned above so the
first measurement cannot silently read 36 % high.
