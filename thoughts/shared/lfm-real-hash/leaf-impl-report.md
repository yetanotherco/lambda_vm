# The `LFML` leaf mode (option C) — implementation report

**Status:** GREEN **after the post-review fixes of §10**. **★ `FriToyV0` proves
and verifies under BLAKE3 — F3.4 is retired.** One material deviation from the
spec's pricing, named in §6.

⚠ **The first submission of this work carried a HIGH soundness defect (D1) that
this report did not disclose** — `MODE_L`'s unread input cells were pinned on the
BLAKE3 arm and on nothing else, which was a Fiat–Shamir break under `Test` and
`Poseidon`. It is fixed, single-sourced and regression-tested; §10 has the
record, and §7's board below is the post-fix state. **Date:** 2026-08-11.

**Ground:** worktree `lambda_vm-blake3-impl`, branch `blake3-real-hash`, on
committed B1 (`9bcc9ee2`), uncommitted. **Spec:** `leaf-spec/LEAF.md`, binding.

Claims are ✓ EXECUTED / ✓ VERIFIED / ✗ OPEN.

---

## 0. Board

| item | result |
|---|---|
| leaf KATs L1/L5/L6 — per-row vectors, 6 and 7 rounds | ✓ EXECUTED, PASS |
| L2/L3 — boundary felts and non-canonical rejects | ✓ EXECUTED, PASS |
| L4 — the predicate IS `v < p` | ✓ EXECUTED, PASS (boundaries + 24k sweep) |
| crate anchor: leaf == `blake3::hash(LE32(lanes)‖"LFML")[..16]` @7r | ✓ EXECUTED, PASS |
| **`FriToyV0` proves + verifies under BLAKE3** | ✓ **EXECUTED, PASS — the milestone** |
| `FriToyV0` + `TrivialV0` under Test / Poseidon / BLAKE3 | ✓ EXECUTED, 3/3 each |
| M9 — mode confusion, six ordered pairs | ✓ EXECUTED, all fire |
| M10 — a `MODE_L` row cannot skip canonicity | ✓ EXECUTED, fires, incl. the alias |
| negative canonicity leg in the assembled proof | ✓ EXECUTED, §5 |
| sweep for other `LFMC` leaf-hashing | ✓ EXECUTED, §4 — none found |
| D6 — hasher-parameterised fixture | ✓ done |
| full `lfm::` suite | **306 pass / 19 fail** — failure set `diff`-identical to the B1 baseline |
| `make fmt` + `make lint` (4 combos) + `blake3-6round` clippy | ✓ clean |
| **cost: `TrivialV0` 16,551 @7r** | ✓ EXECUTED, matches the spec exactly |
| **cost: `FriToyV0`** | ⚠ **513,081, not the spec's 502,047** — §6 |

---

## 1. What was built

A fourth preprocessed selector, `MODE_L`, carrying the `"LFML"` tag and
**felt-input semantics**. A leaf row reads ONE cell as four arbitrary Goldilocks
elements, splits each into a checked `lo`/`hi` `u32` pair, and hashes the eight
halves through the same socket every other mode uses.

```
v = lo + 2^32·hi ,  lo, hi < 2^32
v < p  ⟺  NOT( hi = 2^32−1  AND  lo ≥ 1 )      (p − 1 = 0xFFFFFFFF_00000000)
```

Per felt: the halves binding plus three canonicity constraints in `LFM_BITDEC`'s
own `Z`/`GINV` idiom — **2 witness columns and 4 constraints per felt, zero new
sends, max degree still 3.**

| | before | after |
|---|---|---|
| selectors | `MODE_C`, `MODE_P`, `MODE_T` | + **`MODE_L` at index 9** |
| `NUM_SELECTORS` / `PREP_WIDTH` | 3 / 12 | **4 / 13** (`MULT0..2` → 10..12) |
| socket value columns @7r | 3,436 | **3,444** (+8 canonicity witnesses) |
| framing constraints | 26 | **46** |
| live domains | `LFMC`, `LFMT` | + **`LFML`** |

### File:line map

| what | where |
|---|---|
| `MODE_L`, `NUM_SELECTORS` 4, `PREP_WIDTH` 13 | `prover/src/lfm/layout.rs:80-133` |
| `HashMode::Leaf` + `num_input_cells`/`num_output_cells` | `prover/src/lfm/instr.rs:47-115` |
| `LfmBuilder::leaf` | `prover/src/lfm/builder.rs:285-305` |
| `LfmHasher::leaf` / `leaf_out` (+ `HasherKind` dispatch) | `prover/src/lfm/hash.rs:92-118`, `:290-308` |
| `TAG_LFML`, `is_canonical`, `felt_halves`, `leaf_lanes`, `leaf_digest*` | `prover/src/lfm/blake3_socket.rs:195-380` |
| BLAKE3 `leaf`/`leaf_out` + the `admits` leaf arm | `prover/src/lfm/blake3_socket.rs:470-520` |
| `MU_COLUMNS` (3), `DIGEST_MODE_COLUMNS`, `CANON` block, `canon_z`/`canon_ginv` | `prover/src/lfm/blake3_socket.rs:455-560` |
| the leaf constraints (idx 26–45) | `prover/src/lfm/blake3_socket.rs:1130-1190` |
| `lanes_from_row` + `fill_canonicity_witness` | `prover/src/lfm/blake3_socket.rs:1045-1105` |
| the §2.2 warning, verbatim in substance | `prover/src/lfm/blake3_socket.rs:39-75` (module docs) |
| O5 retirement, rewritten | `prover/src/lfm/blake3_socket.rs:120-145` (module docs) |
| `edsl::leaf_hash_pair`, `SpongeVar::absorb_felts` | `prover/src/lfm/edsl.rs:98-175` |
| `fixture::host_leaf_hash_pair`, hasher-parameterised tree/prover | `prover/src/lfm/fixture.rs:136-300` |
| `FriToyV0`'s three leaf sites + the two data absorbs | `prover/src/lfm/programs.rs:629,637,676`, `:600-607` |
| leaf KAT vectors (generated from the spec JSON) | `prover/src/lfm/leaf_kats.rs` |
| leaf tests (14) | `prover/src/lfm/leaf_tests.rs` |

---

## 2. KAT results — ✓ EXECUTED

The Rust table is **rendered from `leaf_kats.json`**, not hand-copied.

| | check | evidence |
|---|---|---|
| L1 | 5 leaf rows, lanes and digest, at 6 **and** 7 rounds | `every_leaf_vector_reproduces_at_both_round_counts` |
| L1′ | **the crate anchor** — `blake3::hash(LE32(lanes)‖"LFML")[..16]` @7r, message rebuilt byte-level | `seven_rounds_is_blake3_of_the_leaf_message` |
| L2 | six boundary felts round-trip, `p − 1` included (the tight case) | `every_boundary_felt_round_trips_through_its_halves` |
| L3 | three non-canonical values rejected, **not reduced** — and the test derives the alias each one collides with | `non_canonical_values_are_rejected_not_reduced` |
| L4 | predicate == `v < p` over every boundary, ±2,000 around the wrap, and a 20k stride | `the_canonicity_predicate_is_exactly_less_than_p` |
| L5 | `LFMC`/`LFMT`/`LFML` give three different digests from the SAME eight lanes, both round counts | `the_three_domains_differ_on_the_same_lanes` |
| L6 | an 8-felt leaf is 2 `LFML` + 1 `LFMC`, and the HOST path agrees | `an_eight_felt_leaf_is_two_leaf_rows_and_one_parent` |

`leaf_tests`: **14 passed** at 7 rounds; **14 passed** under
`--features blake3-6round`.

---

## 3. M9 / M10 — ✓ EXECUTED, both fire

| | statement | result |
|---|---|---|
| **M9** | a row in domain X whose witness computes domain Y's digest | **all six ordered confusions rejected**; the three same-domain cases accepted (the honest control, in the same loop) |
| **M10** | a `MODE_L` row that skips canonicity | **rejected**, two ways |

M10's second way is the one worth reading. It installs **the alias**: felt `0`
re-encoded as `(lo = 1, hi = 2^32−1)`. That is the *same field element* — the
binding constraint `v = lo + 2^32·hi` is satisfied — so nothing except canonicity
can catch it, and the test asserts the violated index is **`canon-c` for felt 0
(idx 33)** specifically rather than "something fired". Without the block, one
felt would have two leaf digests, which is a collision in the felt→digest map.

**What this makes checkable:** "`MODE_L` implies felt-input semantics" is now a
constraint, not a convention.

---

## 4. The `LFMC` leaf-hashing sweep — ✓ EXECUTED, nothing else found

Every `LFM_HASH` call site in non-test code, classified:

| site | kind | verdict |
|---|---|---|
| `edsl::merkle_walk:187` | parent over two digests | correct as `LFMC` |
| `edsl::leaf_hash_pair:170` | parent over two leaf digests | correct as `LFMC` |
| `FriToyV0` ×3 (`programs.rs:629,637,676`) | **leaf over trace rows / folded ext values** | **moved to `MODE_L`** |
| `TrivialV0` ×3 (`programs.rs:60-62`) | raw arena data | **left as `LFMC`, deliberately** |

`TrivialV0`'s three are the only remaining place raw arena data enters a
compress, and they form a **chain, not a tree** — there is no leaf and no parent,
so there is no confusion for `MODE_L` to separate. The consequence, recorded at
the call site: that program's arena words must be `u32`-laned under BLAKE3, which
its tests supply. Data that cannot be is exactly what `leaf` exists for.

**O5 is retired and now enforced by the tag.** A leaf digest is
`BLAKE3(…‖"LFML")` and a parent is `BLAKE3(…‖"LFMC")`, so an internal node cannot
be replayed as a leaf whatever the tree's shape. Fixed depth remains true of
every current program but **is no longer load-bearing**. The module docs were
rewritten accordingly — the previous text said "nothing implements `LFML` yet"
and rested the argument on fixed depth.

---

## 5. ★ The milestone, and its negative leg

`leaf_tests::fri_toy_proves_and_verifies_under_blake3` — ✓ EXECUTED. A real
verification program, over real FRI data (LDE evaluations and folded extension
elements, **124 of the fixture's 128 committed values are ≥ 2^32**), proved under
the machine's real hash and accepted by the production verifier. The attested
public output is checked against the inner proof's own roots.

The four replacement criteria the old tripwire's doc set:

1. it is deleted only now that `FriToyV0` proves and verifies under BLAKE3 ✓
2. the replacement is a **prove+verify**, not an execute ✓
3. honest control: the same program proves under `Test` (and Poseidon) ✓ —
   `fri_toy_proves_and_verifies_under_every_hasher`, 3/3
4. ⚠ **RETIRED AS UNSATISFIABLE — not met, and it cannot be.** The criterion
   asked for "a non-canonical arena value must make the proof fail, and fail FOR
   canonicity". No such arena value exists: an arena word is `[FE; 4]` and every
   `FE` is canonical by construction, so the input the criterion describes is
   unconstructible. (By the same argument the `admits` leaf arm's canonicity
   check is dead code on today's call paths — it is kept as the boundary's
   statement, not as a reachable rejection.)

   **What I shipped in its place is NOT that test.**
   `fri_toy_rejects_a_fixture_built_under_another_hasher` is a hasher-mismatch
   test: it shows a fixture whose leaves were hashed differently fails to
   authenticate. Useful, and it does exercise the leaf digests end-to-end — but
   it is a root-mismatch rejection, not a canonicity one, and the first version
   of this report presented it as satisfying criterion 4. It does not.

   **Canonicity's necessity is shown elsewhere, and adequately:** by M10's alias
   leg (a non-canonical half-pair for a felt that the binding constraint accepts
   and only `canon-c` rejects, `leaf_tests.rs`) and, in z3, by the oracle's WA8
   dropped-leg — canonicity removed ⇒ the same felt becomes provable. The
   assembled-proof evidence criterion 4 wanted would need a trace tamper rather
   than an arena value.

Plus `the_fixture_data_is_still_not_u32_and_that_is_the_point`, which keeps the
premise visible: proving over `u32`-shaped data would have proved nothing about
the leaf mode.

### D6 — the fixture, closed

`HostTree::build` and `fixture_prove_columns` now take a `HasherKind`, and
`host_leaf_hash_pair` mirrors the machine's leaf exactly (2 `LFML` + 1 `LFMC`).
The review called this a completeness trap at exactly this milestone and it was:
with `TestPermutation` hard-coded, every BLAKE3 run would have failed inside a
query walk with an authentication error rather than at the mismatch.

---

## 6. ⚠ DEVIATION — `FriToyV0` costs 513,081, not the spec's 502,047

**This is a spec premise that does not hold, not an implementation choice.**

`LEAF.md` §5 says *"`FriToyV0` compresses 67 → **91** … transcript unchanged at
11"*. The transcript's step count IS unchanged at 11 — that half is right. But
two of the four cells `FriToyV0` absorbs are **`t0` and `t1`, the terminal
polynomial's coefficients** — arbitrary field elements, not digests. Absorbing
them raw hands the socket lanes that are not `u32`, so the row is unprovable
under BLAKE3. ✓ EXECUTED: the fixture panicked in
`Blake3Permutation::step` until this was fixed; the probe showed
`commitments[2]` and `[3]` each carry 3 non-`u32` lanes.

**The fix, and why this shape:** data enters the transcript the same way it
enters a tree — through the leaf encoding. `SpongeVar::absorb_felts(c)` hashes
the cell to a digest under `"LFML"` and absorbs *that*, binding the data up to
the leaf hash's collision resistance. One uniform rule ("`absorb` for digests,
`absorb_felts` for data"), stated at both the machine and host sponge.

**Cost:** two extra `LFML` rows.

| | spec | built | note |
|---|---:|---:|---|
| `LFMC` rows | 56 | 56 | 4 queries × (3 leaf parents + 11 walk steps) |
| `LFMT` rows | 11 | 11 | the transcript, unchanged as §5 says |
| `LFML` rows | 24 | **26** | 4 × 3 data leaves × 2, **+2 terminal coefficients** |
| total rows | 91 | **93** | |
| cell-equiv @7r | 502,047 | **513,081** | +2.2% |

`TrivialV0` is unaffected and reproduces the spec's **16,551** exactly (3 rows ×
5,517), including §5's own correction that it is *not* cost-unchanged — the
canonicity witnesses exist on every row.

**For the decision record:** the per-row price (5,517 @7r, 4,749 @6r) and
`TrivialV0` are exactly as ratified; only `FriToyV0`'s row count moved, and it
moved because a program that absorbs field data needs the leaf encoding there
too. If the leaf spec's 91 is quoted anywhere downstream it should be corrected
to 93.

---

## 7. Verification

| gate | result |
|---|---|
| full `lfm::` suite | **306 passed / 19 failed** — `diff`-identical failure set to the B1 baseline (the `fibonacci.elf` 19). From B1's 290: **+16 passes**, being 16 new `leaf_tests` less the deleted O1 tripwire, plus the transcript preamble's split assertions |
| `lfm::leaf_tests` | **16** pass @7r, 16 pass @6r |
| `lfm::blake3_socket_tests` | **34** pass @7r, 34 @6r — the same 34; the file has no `cfg`-gated test, and the earlier "35/34, one 7r-only" in this report was simply wrong |
| `lfm::transcript_tests` | 17 pass @7r, 17 @6r |
| `make fmt` + `make lint` (4 feature combos) | clean, exit 0 |
| `clippy --features blake3-6round` | clean, exit 0 |

### Registry re-bless — once, riding `PREP_WIDTH` 12 → 13

All six `program_id`s moved:

| entry | new (first 8 bytes) |
|---|---|
| `TrivialV0` | `7087e2838dae1171` |
| `FriToyV0` | `82b53911e83ceb53` |
| `KeccakChainV0` | `e830e1f5f9f1ebaf` |
| `KeccakSpongeV0` | `d4f94944580b18eb` |
| `TranscriptReplayV0` | `998273f096ab6b57` |
| `StatementReplayV0` | `788129775db248d2` |

Diff scope: 19 modified files under `prover/src/lfm/` plus 2 new
(`leaf_kats.rs`, `leaf_tests.rs`). Nothing outside the LFM surface; the keccak
wrap path is untouched.

---

## 8. What the oracle's re-gate needs — exposed, mirroring the `MODE_T` pass

| the gate needs | where it is |
|---|---|
| the leaf selector | `cols::MODE_L` |
| the four-way mu | `cols::MU_COLUMNS` (3 entries: C, T, L) and `cols::DIGEST_MODE_COLUMNS` (C, T — the lane identity's gate) |
| the tag map, verbatim | `TAG_SELECTOR` — `(column, tag)` pairs, now three |
| `TAG_LFML` for the pin's checked set | `blake3_socket::TAG_LFML = 0x4C4D464C` |
| the canonicity witness columns | `cols::canon_z(i)` / `cols::canon_ginv(i)`, `cols::CANON` |
| the half lanes | `cols::leaf_lo_lane(i)` / `cols::leaf_hi_lane(i)` |
| constraint indices | 0–3 capacity, 4 mode-sum, 5 `MODE_P` pin, 6–13 lane identity (**digest modes only**), 14–21 unused `OUT`, 22–25 digest recomposition, **26–33 unread-`IN` pins** (`chips::hash::emit_unread_input_pins`, shared by all three arms), **34–49 leaf** (per felt: binding, canon-a, canon-b, canon-c — located by `blake3_socket::LEAF_IDX`), 50+ core |
| the predicate | `blake3_socket::is_canonical` |

**WA8's honest leg is already covered on the Rust side** by
`m10_a_leaf_row_cannot_skip_canonicity`'s control and by
`fri_toy_proves_and_verifies_under_blake3`; the "canonicity dropped ⇒ SAT" leg
needs the gate, since it requires editing the constraint set.

## 9. Open

| item | status |
|---|---|
| WA8 / M8-four-way / M9 / M10 in z3, and the pin gaining `TAG_LFML` + pairwise-distinct | ✗ OPEN — the oracle's, `gate-oracle/` untouched by this build |
| tag tables marking `"LFML"` **live** rather than reserved | ✗ OPEN — flagged, not edited (the B1 pass established that these are the oracle's) |
| `LEAF.md` §5's 91 / 502,047 | ✗ OPEN — §6; needs correcting to 93 / 513,081 |
| single-domain hashers do not separate leaf from parent | ✗ OPEN by design — recorded at `LfmHasher::leaf_out`; under `Test`/`Poseidon` O5 still rests on fixed depth, as it did before. Neither is a production hash |

---

## 10. Post-review fixes (leaf-verify.md D1–D6)

The adversarial review confirmed the leaf mode's own machinery sound and found
one **HIGH soundness defect** in the other two arms, plus five claim/doc defects.
All are fixed.

### ★ D1 — HIGH, soundness. `MODE_L`'s unread input cells were free under `Test` and `Poseidon`.

**The defect, and it was mine.** `MODE_L` reads ONE cell. Three places were
taught that — the `LfmMem` receive, the validator's address-slot check, and the
BLAKE3 AIR's value-column pin — and two were not: `eval_test` and `eval_poseidon`
read `A_i = IN_i` for every `i < 8` in round 0. So on a leaf row under those
hashers, `IN4..8` received nothing from the bus, were pinned by nothing, and were
**read by the permutation the AIR proves** — four free Goldilocks felts, and
`leaf(c)` stopped being a function of `c`. The reviewer executed it end to end:
Poseidon proved AND verified with attacker junk in those columns. For any program
that absorbs data through `absorb_felts` — `FriToyV0` does — that is a complete
Fiat–Shamir break, since the prover re-randomises the junk and chooses `alpha`,
`zeta0`, `zeta1` and every query index with the public statement unchanged.

**⚠ This report talked someone out of the fix before it was needed.** §1's earlier
text called the pin "hygiene rather than soundness". That was true of the BLAKE3
arm in isolation and false as a general statement, and it is exactly the sentence
a reader would have cited to skip the other two arms. The comment is reworded at
the source and the claim is retracted here.

**The fix** — `chips::hash::emit_unread_input_pins`, ONE derivation from
`HashMode::num_input_cells()`, called by all three arms:

- for each input cell some mode does not read, `Σ(selectors of modes that do not
  read it) · IN_col = 0`, four constraints per cell, degree 2;
- **both** unread cells are pinned, not only the one that broke: cell 2 was
  previously safe because nothing read it, which is precisely the reasoning that
  failed for cell 1. `MODE_SELECTORS` is the single mode↔column table it reads.
- constraint counts: `Test` 17 → 25, Poseidon 601 → 609, BLAKE3 framing 46 → 50
  (`NUM_CONSTRAINTS` 942 → 946 @7r). **No cell counts move** — constraints do not
  enter the census — so §6's pricing is unchanged and ✓ EXECUTED: all six
  `program_id`s are byte-identical to the pre-fix re-bless.

**Regression tests**, `leaf_tests`:

- `d1_the_unread_input_pins_are_load_bearing_under_every_hasher` — **shaped like
  WA9**, on the oracle's suggestion, because "the junk row is rejected" would
  pass for a set that rejected it incidentally and would say nothing about
  whether the new constraints are needed. For each of the three arms: the honest
  leaf row still satisfies every constraint (the mandatory honest control), and a
  **consistent** forgery — junk in the unread cell with the whole rest of the row
  rebuilt from it, which is what a prover controlling the trace would actually
  submit — has a violated set that is **exactly** the four pins for that cell,
  read from `chips::hash::unread_input_pin_base` rather than a literal.

  That equality carries both legs at once: WITH the pins the row is rejected, and
  WITHOUT them — delete those four and every remaining constraint still evaluates
  to zero on this row — it is **ACCEPTED**. That is the dropped-leg, and it is
  what makes the pins necessary rather than merely present. On `Test` and
  `Poseidon` that acceptance was the shipped behaviour and an executed
  Fiat–Shamir break; on BLAKE3 the row is inert either way, which is precisely
  the WA9 mirror-image the oracle named — the same constraint is hygiene on one
  arm and soundness on the two whose round 0 reads `IN4..8`.
- `d1_the_pins_come_from_one_derivation` — the selector table is the layout's
  one-hot span, and the pin count follows from `num_input_cells()`, so a mode
  added later cannot acquire free columns by an arm forgetting a line.

### D2 — MEDIUM. The transcript end-to-end vector modelled a transcript `FriToyV0` no longer runs. FIXED — **consumed from the oracle, not invented here.**

`transcript_kats.rs` carried a `✓ VERIFIED against fri_toy_program_source` marker
on a vector built from `absorb2(t0w, t1w)`, but the program now does
`absorb_felts(t0w); absorb_felts(t1w)`. The transcript's own **step count is
unchanged at 11**, which is why nothing went red — and why the stale marker
mattered: the state vector is the one anchor from an independent reference.

**The oracle found the identical staleness in its own `transcript_kats.json` and
regenerated it at both round counts.** `transcript_kats.rs` is re-rendered from
that file; the vector is theirs.

**✓ EXECUTED cross-check, worth recording.** Before their regeneration was
visible I had composed a replacement myself from the same two oracle references
(`transcript_ref` chained, `leaf_ref` for the leaf step). The two agree
**bit-for-bit** across all 11 states at 7 rounds — two independent compositions
landing on the same vector. Mine is discarded; theirs is what ships.

**Their convention, adopted:** `FRI_TOY_COMPRESSIONS` counts the preamble's total
socket cost — **13**, being 11 transcript steps plus the 2 `LFML` rows
`absorb_felts` adds — rather than transcript steps alone. That is the
decomposition that closes the total in the vectors instead of only in prose:
**4 queries × 20 + 13 = 93**, which is the number §6 measured. The tests assert
both halves and their sum, so neither can drift alone.

The four replay sites now model the real preamble: the reference replay, the
`HostSponge` replay, the emitted preamble program, and the machine-vs-host
agreement test. The preamble's arena carries the terminal coefficients as FELTS,
which is what the leaf encoding consumes.

### D3 — criterion 4 retired as unsatisfiable. §5, rewritten above.

### D4 — the report's own arithmetic. Fixed in §0 and §7.

`blake3_socket_tests` is **34 at both round counts** with no `cfg`-gated test —
the earlier "35 @7r / 34 @6r, one 7r-only" was wrong (the difference was the
tripwire I had already deleted). The suite total is now 306, and the delta from
B1's 290 is **+16 passes**, stated as such rather than as "+13".

### D5 — the leaf/parent separation caveat was in the wrong place. FIXED.

`instr.rs` and `layout.rs` stated "O5 retired by the tag" **hasher-independently**
in the ISA docs, while the correctly qualified version sat in `hash.rs` — exactly
backwards, since a single-domain hasher does not separate the domains at all.
Both ISA sites now say the mode is a machine-level SHAPE and that whether a leaf
and a parent are different functions is the hasher's business, pointing at
`LfmHasher::leaf_out`.

### D6 — "hygiene rather than soundness". FIXED.

The comment on the BLAKE3 pin now says the pin is **load-bearing on any arm whose
constraints read `IN`**, names the two that do, and records that it shipped
missing there — so the next reader is warned rather than reassured.

---
