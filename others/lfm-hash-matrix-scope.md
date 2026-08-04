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

**Recommendation, in one line: build Poseidon-original first, and pin the
permutation-count axis before building any permutation at all.** Reasons in §4.

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

⚠ **PROVENANCE, stated plainly: the round counts and S-box degrees below are my
own domain knowledge, not read out of this repo and not (yet) confirmed against
the corpus.** They are the weakest link in this report and wave 9 should check them
against `recursion_architectures.md` before building. Everything about how a round
count becomes a cell count IS verified (§1, §1.1).

The layout constraint is real and VERIFIED: `max_degree() = 3` for the `LFM_HASH`
chip (`chips.rs:532-534`), and `max_degree` "is what the engine uses as the
composition-poly degree bound … over-declaration is safe, under-declaration is not"
(`prover/src/tests/constraint_set_tests_a.rs:66-74`). Raising it is possible but
the wrap runs at blowup 2, so a higher-degree composition polynomial costs LDE
cells — self-defeating for a memory play. **So every candidate must express its
S-box in degree ≤ 3, which for `x^7` means two intermediate columns per S-box**
(`x²`, `x³`, then `x⁷ = (x³)² · x`, degree 3 over columns).

| candidate | S-box | rounds (est.) | S-boxes | `m` est. | `a` est. | base-equiv `m+3a` | vs keccak 77,992 |
|---|---|---|---|---|---|---|---|
| keccak (MEASURED) | — | 24 | — | 36,256 | 13,912 | **77,992** | 1.0× |
| Poseidon-original t=12, **Layout B** | x⁷ | 8 full + 22 partial | 118 | ~608 | 3 | **~617** | **126× cheaper** |
| Poseidon-original t=12, Layout A | x⁷ | 8 full + 22 partial | 118 | ~1,080 | 90 | ~1,350 | 58× cheaper |
| RPO t=12 (Layout B) | x⁷ and x^(1/7) | 7 (both layers) | 168 | ~850–1,500 | 3 | ~860–1,510 | 52–91× cheaper |
| Monolith t=12 | Bars (lookup) | 6 | — | low rows, + a lookup AIR | **>3** | not estimated | needs a lookup table |
| Blake2s (reduced) | ARX on 32-bit words | 10 (or fewer) | — | **bit-oriented — see below** | **≫3** | not estimated | **NOT expected to be orders cheaper** |

Derivation for Poseidon-original, the one I recommend building (DERIVED from the
estimated parameters above plus the VERIFIED census formula):

- 8 full rounds × 12 S-boxes + 22 partial rounds × 1 S-box = 118 S-boxes.
- **Layout B, fully unrolled, one row per permutation:** `m ≈ 12 + 8 × (12 + 24) +
  22 × (12 + 2) = 608` value columns, 1 row. `a = 3`, so `m + 3a ≈ 617`.
- **Layout A, one row per round:** width ≈ 12 state + 12 × 2 intermediates = 36
  value columns × 30 rows → `m ≈ 1,080`.

⚠ **Correction to my own first pass, worth stating because it flips the layout
choice from "either" to "clearly B".** I initially carried `a = 3` into both
layouts. That is wrong: the census formula is `aux_cells = rows × ceil(interactions/2)`,
so aux scales with ROWS, and Layout A pays `30 × 3 = 90` aux cells per permutation,
not 3. Since aux count triple, Layout A's aux term alone is 270 base-equivalent
cells — it more than doubles Layout A's disadvantage. **Layout B wins on both cell
count (617 vs 1,350) and on padding** (§2.3's second caveat).

`a` is 3 in Layout B because Poseidon is purely algebraic: **no lookups, so no new
bus interactions**, and the chip's existing 6 `LfmMem` interactions are the whole
aux bill. Row-to-row state wiring, if any, is transition constraints and not buses,
so it adds no aux.

**The aux collapse is the biggest single effect and it is structural, not an
estimate.** Keccak's `a = 13,912` per permutation exists because `KECCAK_RND`
lookups into `BITWISE` are bus interactions and `aux_cols = ceil(interactions/2)`.
An algebraic hash has none. Aux cells are cubic-extension elements and so count
**triple** in the base-field-equivalent metric — 3 × 13,912 = 41,736 of keccak's
77,992 per-permutation cells, i.e. **53.5 % of the hash bill is aux alone**, and
essentially all of it is the bitwise lookups.

**Why blake is not in the cheap column.** Blake2s is ARX over 32-bit words: XOR and
32-bit rotation. In a Goldilocks prime field those are not field operations — they
need bit decomposition or a lookup table, i.e. the same mechanism that makes
keccak's aux bill 53.5 % of its cost. So blake's in-AIR character is keccak-like,
not Poseidon-like. This matters because blake is (INHERITED, team lead) the most
probable ship choice on cryptographic-trust grounds. **If that is right, the hash
decision may not buy the 2.8× memory relief the wrap needs at all** — which is
exactly the sort of finding the matrix exists to surface, and it is the reason
blake's column is decision-critical even though Poseidon's is cheaper to build.

### 2.3 The predicted matrix column, assembled

DERIVED, using conservative assumptions (§1's residue held fixed, which favours
the candidate; `P` as an interval):

```
candidate total = P_cand × (m + 3a)_cand × 1.019   +   1,784,197,396
```

For Poseidon-original, with `P ≈ 192,000` MEASURED (§2.1) and the two free design
choices — layout and whether the candidate's AIR gets a chunking sibling — taken at
BOTH extremes, so the answer is a box rather than a point:

| layout | padding | hash cells | total | vs 11.17 B | projected RSS |
|---|---|---|---|---|---|
| **B** (1 row/perm, 617) | chunked, 1.9 % | 0.121 B | **1.905 B** | **5.86× smaller** | **59.8 GiB** |
| **B** | unchunked, pads to 2¹⁸ (+36.5 %) | 0.162 B | 1.946 B | 5.74× smaller | 61.1 GiB |
| A (30 rows/perm, 1,350) | chunked | 0.264 B | 2.048 B | 5.45× smaller | 64.3 GiB |
| A | unchunked, pads to 2²³ | 0.378 B | 2.162 B | 5.17× smaller | 67.8 GiB |

**Every cell of that table is inside the 124 GiB box, and the spread across it is
1.13× while the win is 5.2–5.9×.** That is the robustness claim, and it is what
makes this prediction worth acting on despite resting on an estimated round count:
the conclusion survives being wrong about layout, wrong about padding, and wrong
about `m` by a factor of two.

**The headline that falls out: Poseidon-original plausibly brings the production
wrap from 350.6 GiB to roughly 60–63 GiB, i.e. inside the 124 GiB box** — and with
the `P` axis measured, the only estimated input left is the round count. Note what
does the work — once the hash is cheap, the *residue* dominates (1.78 B of ~1.9 B),
so the prediction is insensitive to the hash estimate and mostly sensitive to a
number that is already measured. That is a robustness argument, and it also means
further hash optimisation past Poseidon buys almost nothing at this shape.

⚠ **Two caveats on the 1.019 padding factor, which I carried over from keccak and
which does NOT transfer cleanly.** It is `KECCAK_RND`'s chunk padding at the
production shape (ledger entry 10). A candidate on socket 2 has ONE row per
permutation, so its trace height is `P` itself and its padding is however far
`P ≈ 192,000` sits below a power of two — `2^18 = 262,144`, i.e. **a 36 % waste, not
1.9 %**, unless the candidate's AIR is chunked the way `KECCAK_RND` is. That
pushes the low end from 0.118 B to ~0.155 B of hash cells and the total from 1.90 B
to ~1.94 B — still ~5.8× and still inside the box, so it changes no conclusion, but
it means **the chunking work `chunking.rs` did for keccak will need a sibling for
the candidate**, and a naive first measurement will read ~36 % high on the hash
term. Flagging it because it is exactly the kind of thing that gets discovered after
someone reports a number.

Second: a multi-row layout (Layout A, 30 rows/permutation) makes the trace 30× taller
and the padding question correspondingly different. `m` is roughly layout-invariant
but PADDING is not, which is an argument for Layout B (unrolled, one row per
permutation) beyond its lower cell count.

All numbers name their epoch shape: fixture epoch, profile
`[2 ×14, 3, 4 ×4, 5 ×3, 7, 20]`, 24 sub-proofs, fibonacci guest, 16-cycle
INTERMEDIATE epoch, inner blowup 8 / 73 queries; wrap options blowup 2 / 219
queries / grinding 20 (entry 10's rule).

---

## 2.4 What already exists in-tree — searched structurally, and it changes §4's risks

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

- **An inner prover under the candidate hash.** To VERIFY a real proof committed
  under candidate `H`, the inner prover must commit under `H` — transcript, Merkle
  backend, grinding. UNRESOLVED, and §2.4 partly overturns my first guess: the
  commitment layer is a TRAIT (`IsMerkleTreeBackend`) that already has a
  field-element Poseidon implementation, so the Merkle half may be additive rather
  than invasive. What I did NOT establish is whether the prover is generic over that
  trait or pins a concrete backend, nor anything about the transcript or grinding.
  **Do not read "additive" into this — read "cheaper to find out than I assumed".**
  §3.3's limit stands either way: the cells measurement does not need it.
- **Widening `HASH_STATE_FELTS`** past 12 to cut the 2.125× rate penalty (§1.2).
  The contract is frozen and the bus tuples/opcodes are pinned; this is a team-lead
  decision, and it is the single cleanest lever on the candidate's `P`.
- **Raising `max_degree` above 3** to shorten the S-box. Interacts with the wrap's
  blowup 2; almost certainly a net loss, but it is a framework-ceiling question and
  rule "report a ceiling rather than working around it" applies.

### 3.3 The measurement this buys, and what it does NOT buy

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
2. **Then Poseidon-original** (slices B, C, E) — algebraic, so `a` stays at 3 and
   the aux collapse (53.5 % of the hash bill) is banked; fits degree 3 with two
   intermediates per S-box; and it is where the ecosystem is going now that
   Poseidon2 is broken. This validates socket 2 under real load for the first time.
3. **Then blake**, because its column is the one most likely to CHANGE the
   decision (§2.2). Expect it to need a lookup/bit-decomposition mechanism, so
   budget it as a keccak-class build, not a Poseidon-class one.
4. Monolith and RPO only if 2 and 3 leave the decision open.

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

