# The O1 leaf convention — options note

> # ✅ DECIDED — OPTION C WITH `LFML`, ratified by the user 2026-08-11
>
> On this paper's presentation, with §6's unsettled items disclosed. The
> recommendation in §5 carried.
>
> **§6's open point is resolved BY DECISION, not by assumption:** `MODE_L`
> **implies felt-input semantics** — the cheaper form, which keeps the one-hot
> selector span contiguous.
>
> **Sequencing:** the `MODE_L` change builds on **committed** B1, not into B1's
> current uncommitted diff. Spec first: `leaf-spec/LEAF.md`.
>
> **Pricing correction (2026-08-11, post-build):** `FriToyV0` measures **93
> compresses / 513,081 @7r**, not the 91 / 502,047 quoted in §4–§5. The `t0`/`t1`
> transcript absorbs are arbitrary field elements and need the felt path — two
> more `LFML` rows. §6 had flagged those absorbs as ✗ OPEN and the number was
> written as if they were settled. **The decision is unaffected:** option C is
> still ~12% under option A, and the ranking never depended on those two rows.
>
> This paper is now the record of *why*; `leaf-spec/` is the record of *what*.

**Decision paper — the recommendation was adopted. NOT an implementation.**
§5 carries the recommendation; §4 carries the finding it rests on.

**Date:** 2026-08-11. **Question:** how do arbitrary Goldilocks field elements
reach the BLAKE3 socket, given obligation O1 requires `u32` lanes? This is what
blocks `FriToyV0` and with it the second half of F3.4.

Claims are ✓ VERIFIED (read the code, cited), ✓ EXECUTED (ran it), ? INFERRED.

---

## 1. The blocker, and its exact shape

✓ VERIFIED (`transcript-impl-report.md` §7.1, reproducing the builder's measured
run): `execute(fri_toy_program(), …, Blake3)` returns
`HasherRejected("BLAKE3 compress input lane is not a u32")`, with **124 of the
fixture's 128 committed column values ≥ 2^32**.

The cause is structural, not a fixture accident: `FriToyV0` hashes **FRI data** —
Merkle leaves over LDE evaluations and folded ext values — and the evaluations of
a low-degree polynomial over a coset are arbitrary elements mod `p`. No choice of
polynomial changes that.

**What is NOT blocked** — worth stating, because it bounds the problem:

- **Digests are fine.** Obligation O2: the socket's output is four `u32`s by
  construction, so every internal Merkle node and every sibling already
  satisfies O1. Only *leaf data* is arbitrary.
- **The transcript is fine** except where it absorbs raw felts (`t0w`/`t1w`, the
  terminal-polynomial coefficients).
- **`TrivialV0` is already retired** — F3.4 is closed for that entry.

So the question is narrow: **a leaf-data encoding**, ~104 field elements per
`FriToyV0` proof.

### 1.1 Why "just reduce mod 2^32" is not an option

Stated because it is the tempting shortcut and it is the same bug O1 itself
names. If a felt `v` reached the hash by reduction, then `v` and `v + 2^32`
(where both are < p) would hash alike — the prover picks which. The encoding
must be a **checked decomposition, rejecting out-of-range inputs, not reducing
them**: the same reject-don't-reduce shape as O1.

Concretely, any option must supply two things, and **neither alone suffices**:

1. **Binding** — the halves are *the* halves of the committed felt:
   `v = lo + 2^32·hi`, as a constraint, not a convention.
2. **Range + canonicity** — `lo, hi < 2^32` **and** `v < p`. Range alone is not
   enough: `lo + 2^32·hi` ranges over `[0, 2^64)` while the field has `p ≈ 2^64 −
   2^32` elements, so without canonicity two distinct half-pairs collide onto one
   felt and a prover opens one leaf two ways.

---

## 2. Option A — the `felt_be_halves` precedent (bit-decompose per felt)

The keccak path's existing shape. ✓ VERIFIED `transcript_replay.rs:743-761`:
`felt_be_halves` calls `bit_dec(v, 64)` and recomposes two 32-bit halves from the
bits with `mul`/`mul_add`.

**Soundness.** Strong, and already reviewed: `LFM_BITDEC` supplies both
obligations. ✓ VERIFIED `chips.rs`: 64 booleanity constraints, plus the
canonicity witness pair — `G = (2^32−1) − top32`, `Z·G = 0` (so `G ≠ 0 ⇒ Z = 0`)
and `IS_REAL·(1 − Z − G·GINV) = 0` (so `G = 0 ⇒ Z = 1`). Binding comes from the
bus receiver, which reads the value as the **linear recomposition** `Σ 2^i·B_i`
rather than as a separate column — so there is no "is this the same value?" gap
at all. This is the machine's established canonicity idiom.

**KAT-ability.** Unchanged: the socket still hashes `u32` lanes, so every
compress remains `blake3::hash(a‖b‖tag)[..16]`.

**Cost — the problem.** ✓ EXECUTED (`leaf-convention-cost.py`, whose compress
formula reproduces the gated census first):

| | per felt |
|---|---:|
| `LFM_BITDEC` row (66 main + 65 sends) | **165** |
| 64 × `LFM_BALU` recomposition ops | **640** |
| **total per felt** | **805** |

104 felts ⇒ **+83,720 cell-equiv**, on top of the leaf restructuring every option
pays. `FriToyV0` end-to-end: **585,039 vs 369,103 = +58.5%**.

**Blast radius.** Moderate: `FriToyV0`'s arena layout and program identity move;
no chip changes. **Gate impact: none** — the socket is untouched, so the pinned
board still describes it.

---

## 3. Option B — keep `FriToyV0` off BLAKE3 (the honest do-nothing baseline)

The registry entry stays on `Test`/`Poseidon`; nothing is built.

**Cost:** zero. **Soundness:** nothing new to argue. **What it costs instead is
the claim.** The disclosure would have to read, permanently and precisely:

> `TrivialV0` proves under BLAKE3 and its hashing is cryptographically
> meaningful. **`FriToyV0` does not.** Its Merkle authentication and its
> Fiat–Shamir transcript run under `TestPermutation`, which is not a hash;
> collisions are trivially constructible. The FRI verification that entry
> performs is cryptographically vacuous.

That is the original F3.4 disclosure, surviving for the entry it was mostly
about. **The machine would ship a registry in which its only non-trivial program
cannot use its real hash** — and the reason would be an encoding gap, not a
cryptographic one, which is an uncomfortable thing to have to explain.

Worth saying plainly: B is not absurd. The wrap is hash-neutral, so nothing in
production depends on this. But it leaves the interesting entry permanently on a
placeholder.

---

## 4. ★ Option C — a felt-input mode in the socket (my finding; cheapest and simplest)

The observation the other two miss: **the socket's existing O1 machinery already
does most of a half-decomposition.** Every input lane is byte-decomposed and
`AreBytes`-checked, and the lane identity pins `IN_lane = Σ bytes·2^{8k} < 2^32`.
So `lo` and `hi` are *already* forced to be `u32`s. The only missing piece is
**canonicity**.

And canonicity over two halves is far cheaper than over 64 bits. ✓ EXECUTED
(200,007 cases including every boundary):

```
with lo, hi < 2^32 and v = lo + 2^32·hi:
    v < p   ⟺   NOT( hi = 2^32−1  AND  lo ≥ 1 )
```

because `p − 1 = 0xFFFFFFFF_00000000` — `hi = 2^32−1`, `lo = 0`. So the whole
canonicity check is *"if `hi` is maximal then `lo` is zero"*, which is the
**same `Z`/`GINV` trick `LFM_BITDEC` already uses**, applied to two halves
instead of 64 bits:

```
G      = (2^32 − 1) − hi
idx a  MU · Z · G          = 0        (G ≠ 0 ⇒ Z = 0)                deg 3
idx b  MU · (1 − Z − G·GINV) = 0      (G = 0 ⇒ Z = 1)                deg 3
idx c  MU · Z · lo         = 0        (hi maximal ⇒ lo = 0)          deg 3
       MU · (v − lo − 2^32·hi) = 0    (binding)                      deg 2
```

**Per felt: 2 witness columns (`Z`, `GINV`) and 4 constraints. Per row (4 felts):
8 columns, 16 constraints, ZERO extra sends.** Max degree stays 3.

**Soundness.** Both obligations are met and neither is inherited on faith:
binding is the explicit identity; range comes from the *existing* lane machinery
(the same `AreBytes` sends WA1/WA2 already gate); canonicity is the `Z`/`GINV`
pair above. It is reject-don't-reduce: a non-canonical input has no satisfying
witness, so the row is unprovable rather than silently reduced.

**KAT-ability.** Fully preserved. The message layout does not change — 4 felts
occupy the same 8 lanes 8 `u32`s did — so a felt-mode compress is still exactly
`blake3::hash(lo₀‖hi₀‖…‖tag)[..16]`.

**Cost.** ✓ EXECUTED: **502,047 vs 369,103 = +36.0%**, and **14.2% cheaper than
option A**. The per-row price moves only `5,509 → 5,517`; the increase is almost
entirely the leaf restructuring that *every* option pays (a leaf covers 8 field
elements = 2 felt-mode compresses + 1 combine, so leaves go 1 → 3 compresses and
`FriToyV0` goes 67 → 91 compresses).

**Blast radius.** Larger than A on the chip and smaller everywhere else: a new
preprocessed mode column, the canonicity block, an executor/trace arm — but no
`bit_dec` traffic, no 64-op recomposition per felt, and no memory round-trip.
`FriToyV0`'s program identity moves either way.

**Gate impact.** Real but well-understood, and it is my work: the pinned board
must be re-transcribed (a new mode, 8 new columns, 16 new constraints), plus a
new width-audit pair in the field domain — *canonicity present → a
non-canonical felt is unprovable (UNSAT); canonicity dropped → the same felt
becomes provable (SAT)* — with the honest leg that canonical felts still prove.
That pair is the direct analogue of WA1/WA2 and is the thing that would make the
argument checked rather than asserted.

---

## 4.1 The `LFML` question — live now, and my answer is yes

The lead is right that this is the moment. ✓ VERIFIED and worth stating clearly:
**`FriToyV0` already performs leaf hashing today**, compressing raw trace rows
into leaves under the **`LFMC`** tag (`programs.rs`, the `leaf_a`/`leaf_b`/
`l1_leaf` compresses). The claim "no leaf-hashing path exists" is false; O5's
safety today rests on **fixed depth alone** — every eDSL circuit is fixed-shape
at build time, so no variable-depth second-preimage confusion is reachable.

That is a real argument but a fragile one: it is a property of every *current*
program, not of the construction, and nothing enforces it.

**Recommendation: leaf compresses should use `LFML`.** O5's ratified wording
already requires it ("any future leaf-hashing path MUST use the reserved `LFML`
tag"), the leaf convention is being redesigned anyway, and program identity moves
regardless — so the cost of adopting it now is zero and the cost of adopting it
later is another re-bless. Doing so **retires the fixed-depth crutch**: with
leaves and parents in separate domains the confusion is closed structurally, and
a future variable-depth tree stops being a latent hazard.

Mechanically this fits option C neatly: make the leaf mode `MODE_L`, carrying
both the `LFML` tag *and* felt-input semantics (leaves take felts; parents and
transcript steps take digests). One more preprocessed selector, one more
`TAG_SELECTOR` entry, and **M8's one-hot control extends to cover it unchanged**.

---

## 5. Comparison and recommendation

| | **A — `felt_be_halves`** | **B — stay off BLAKE3** | **★ C — in-socket felt mode** |
|---|---|---|---|
| binding | bus reads the recomposition | — | explicit identity |
| range | `LFM_BITDEC` booleanity | — | **existing** lane `AreBytes` |
| canonicity | 64-bit `Z`/`GINV` | — | 2-half `Z`/`GINV` (✓ EXECUTED) |
| per-felt tax | **805 cell-equiv** | 0 | **2 columns** |
| `FriToyV0` total | 585,039 (**+58.5%**) | n/a | **502,047 (+36.0%)** |
| KAT-able | yes | n/a | yes |
| chip change | none | none | new mode + canonicity block |
| gate impact | **none** | none | re-transcribe + 1 new audit pair |
| retires F3.4 for `FriToyV0` | yes | **no** | yes |

**My recommendation: option C, with leaves under `LFML`.**

1. **It is the cheapest and by a real margin** — 36% over the counterfactual
   against A's 58.5%, because it adds 8 cells per row instead of 805 per felt.
2. **It reuses the machine's own canonicity idiom** rather than inventing one:
   the `Z`/`GINV` pair is `LFM_BITDEC`'s, and the two-half criterion is executed,
   not argued.
3. **It needs no new range machinery at all.** O1's existing lane check already
   forces `u32` halves; only canonicity was missing. That is the whole finding.
4. **It closes O5's fixed-depth dependence** as a side effect, at zero marginal
   cost, because program identity moves anyway.

The honest case against C: it is the only option that touches the **chip**, so it
is the only one that invalidates the current pin and needs a re-gate. I am the
one who pays that, and it is a day's work of the kind just done twice — I do not
think it should drive the decision, but it should be visible.

If cost is not a concern and minimising chip churn is, **A is a perfectly
defensible choice** — it is the reviewed, precedented path and costs the gate
nothing. **B should be chosen only deliberately**, with §3's disclosure written
down, not drifted into.

---

## 6. What I could not settle

- ? INFERRED: option A's 805/felt assumes `felt_be_halves`' 64 `mul`/`mul_add`
  ops each cost one `LFM_BALU` row (4 main + ~4 interactions). The chip widths
  are ✓ VERIFIED; the op count is read off the source loop; but no profile was
  run and memory traffic for the 64 bit-senders is not priced. Treat A's figure
  as a floor.
- ? INFERRED: the leaf restructuring (1 → 3 compresses) assumes a leaf keeps
  covering 2 trace rows. A different leaf arity changes every option equally.
- ✗ OPEN: whether `t0w`/`t1w`'s absorbs want felt mode or a separate convention —
  they are 8 of the 104 felts and do not change the ranking.
- ✗ OPEN: option C's exact constraint count depends on whether `MODE_L` implies
  felt-input or the two are separate selectors. I assumed implied, which is
  cheaper and keeps the one-hot span contiguous.
