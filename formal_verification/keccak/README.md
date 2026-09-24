# Formal verification of chip round-wiring — z3/QF-BV baseline

This directory is the **canonical, reusable template** for machine-checking that a
bit/byte hash chip's per-round transition wiring computes the function it claims,
*given* the contracts of the helper chips it calls. The worked instance here is
`prover/src/tables/keccak_rnd.rs` (one Keccak-f[1600] round); the method is written
to be copied for the next chip (SHA-2/3 variants, BLAKE3, …).

If you are verifying a new chip, read the **Method** and **Mandatory discipline**
sections, then clone this file layout and swap in your chip's reference + contracts.

---

## What is proven, in one line

For **every** constraint-satisfying assignment of the chip's trace columns, the
chip's declared output equals an **independent** reference implementation of the
round — *assuming* each helper-chip lookup obeys its typed contract. Formally:
assert `chip_output ≠ reference(input)` and ask z3 for a counterexample.

- **UNSAT** ⇒ no such assignment exists ⇒ the wiring is correct (given the contracts).
- **SAT**   ⇒ the constraints permit a wrong output ⇒ under-constrained / mis-wired,
  and the model hands you the forging assignment.

## Method: oracle + checker

The verification is an **assume-guarantee** argument split into two halves with a
deliberate trust boundary:

**Oracle (human-owned, per-chip).** Three artifacts a person writes and reviews:
1. **Reference `f`** — the round recomputed straight from the spec (FIPS-202 here),
   in a representation *structurally independent* of the circuit's wiring. Here it is
   64-bit-lane bitvector ops (`RotateLeft`/`xor`/`and`/`not`) written from the
   standard, in `keccak_ref.py` (`zref_round` in `z3_verify.py`), anchored against
   Python's `hashlib` SHA3 and the repo constant tables (`test_ref.py`).
2. **Column-role map** — which trace column plays which algebraic role in the round
   (which byte of which lane, which carry, which RC). This is the transcription of
   the chip's `bus_interactions` / constraint set into equations.
3. **Chip-contract library** — the typed guarantee each helper lookup provides
   (below). These are *assumed*; each is itself a separately-verified chip.

**Checker (generic, reusable).** z3 in the quantifier-free bitvector theory
(**QF-BV**). Every trace column becomes a free bitvector; every bus interaction and
eval constraint becomes an equation over those frees under the referenced contract;
the output columns are whatever the constraints force. The checker is chip-agnostic
— only the oracle changes between chips.

The trust boundary is the point: a mis-transcription in the oracle is caught by the
**concrete mirror** (`model_dataflow.py`, validated forward against the reference in
`test_dataflow.py` over random and structured inputs) and by the **negative
controls** (below). z3 never sees the Rust; faithfulness of the model to the Rust is
a human obligation, and the long-term fix is to *generate* the model from the
constraint IR instead of hand-transcribing it.

## Typed contract library (assume-guarantee)

Each helper lookup is modeled by its contract, not its implementation:

| Contract | Guarantee modeled |
|---|---|
| `ByteAlu(op, a, b, c)` | `a,b,c` are bytes and `c = a op b`, `op ∈ {XOR, AND, OR}` (the three `BitwiseOperationType::ByteAlu*` rows; keccak_rnd uses only XOR and AND). Operands passed as linear combinations must themselves be bytes — the lookup table only has byte rows — modeled as `ULE(Σ, 255)` on the field value with the low 8 bits used. |
| `AreBytes(a, b)` | both `a` and `b` lie in `0..256` (a range check). In QF-BV this is supplied structurally by declaring the column an 8-bit vector and packing pairs into 16 bits. |
| `Hwsl(in16, s, left16, right16)` (halfword shift) | `left16 = (in16 << s) mod 2¹⁶`, `right16 = in16 >> (16 − s)` (with `right16 = 0` at `s = 0`). Keccak's θ/ρ shifts no longer *call* this lookup — they are inline linear identities (see "Which round variant") — but the QF-BV encoding of the decomposition is identical, so the same contract row models both. |
| 32-bit / word recomposition lookups | a wide value equals the range-checked recomposition of its limbs (`word = Σ limb_i · 2^{8i}`), each limb a byte. Not exercised by keccak_rnd; listed because the template's next targets (e.g. 32-bit-lane hashes) need it. |
| `KeccakRc(round, rc[8])` | `rc` = little-endian bytes of `KECCAK_RC[round]`, for `round ∈ [0, 24)`. The committed table also carries padding rows `24..31` with `rc = 0`, which this contract does **not** cover: the gate assumes they are unreachable, which holds by the bus topology (`keccak.rs` supplies the round endpoints as constants and chains `ROUND+1`) but is a cross-row property outside what QF-BV checks here. Pinning those rows to row 0 instead of zero would remove the assumption. |

## Mandatory discipline (do not skip any of these)

1. **Negative controls are not optional.** "UNSAT = verified" is meaningless unless
   you have shown the encoding is *falsifiable*: inject a bug into the model and
   confirm it flips to **SAT**. If a bug does not flip the result, the encoding is
   vacuous and every UNSAT it ever produced is worthless. This directory ships
   controls of two kinds — **changed** constraints and **removed** constraints —
   because an over-constrained model can hide a missing constraint. See
   `tamper_test.py` and the `bug=` cases in `z3_verify.py`.

   **A constraint encoded as a variable's *sort* is outside the falsifiable set.**
   Check for this explicitly — it is the one gap a full board of green controls
   cannot reveal. Here, `AreBytes` on `Cxz_left`/`rot_left`/`rot_right` is carried by
   declaring those variables 8-bit bitvectors, and the θ carry `Cxz_right` is pinned
   directly to `ZeroExt(7, Extract(15, 15, in))`, with the separate `IS_BIT` disjunct
   redundant *in the model*. Both are `load-bearing` in the circuit
   (`keccak_rnd.rs:7-8`, `:840-842`: the 20 μ-gated `IS_BIT`s make the θ shift
   decomposition unique). Because neither can be removed from the model's constraint
   list, **deleting them from the Rust leaves this gate printing `VERIFIED`** while
   the θ `left` halfwords go free — `2¹⁶` is invertible mod `p`, so the forged
   assignment exists. The model's pin is a sound *consequence* of the shipped
   constraints today, which is why the current board is meaningful; what it is not is
   a test that those constraints are still there.

2. **Positive control (non-vacuity).** Pin the input to a concrete value, drop the
   diff assertion, and confirm the constraint system is **SAT** *and* uniquely pins
   the output to the reference. This proves the UNSATs are "no counterexample",
   not "no models at all". See `positive_control` in `z3_verify.py`.

3. **Width audit — the field-lift trap.** The circuit lives over a large prime field;
   the model lives in fixed-width bitvectors. That is only faithful if **every**
   byte/word width in the model is backed, in the circuit, by (a) a real range-check
   contract *and* (b) a non-overflow side condition guaranteeing the field arithmetic
   never wraps the modulus. A field-level attacker who can make an "8-bit" value hold
   `> 255`, or make a sum overflow `p`, escapes a bitvector model that silently
   assumed the bound. For each lifted width, cite the exact contract that pins it
   (here: `AreBytes` on `Cxz_left`/`rot_left`/`rot_right`, `IS_BIT` on the θ carry
   `Cxz_right`) and confirm the operands cannot overflow. Where the circuit replaced
   a lookup with a **linear identity** (keccak's inlined θ/ρ shifts), the bound is
   *load-bearing at the field level* in a way QF-BV cannot see: `2¹⁶` is invertible
   mod the Goldilocks prime, so without the range bound the `(left, right)`
   decomposition is ambiguous. QF-BV proves the wiring given the bound; proving the
   bound *suffices* mod `p` needs an integer/field model (see Scope + follow-ups).

4. **Independent reference.** The reference must be derived from the spec, not from
   the circuit or the repo's constant tables, then anchored to an outside
   implementation (`hashlib`) and cross-checked against the repo constants. A
   reference that copies the circuit proves only that the circuit equals itself.

5. **Fail-open is the only dangerous failure mode.** A gate that wrongly rejects an
   honest chip is a nuisance you will notice immediately. A gate that is **green for
   the wrong reason** — vacuous encoding, a dropped constraint the model never had,
   a width the model assumed but the circuit never checks — silently blesses an
   unsound chip. Every item above exists to close a fail-open hole. When in doubt,
   assume the gate is lying and add a control that would catch it.

## Which round variant this instance verifies

The model verifies the **shipped** Keccak round on `main` as of #889
(`perf(keccak): inline θ/ρ halfword shifts as μ-gated identities`), commit
`6a280121`. On main the θ rotate-by-1 and the ρ per-lane shifts are enforced by
inline μ-gated linear identities in `KeccakRndConstraints`:

```
θ:  μ · (in·2      − right·2¹⁶ − left) = 0     (right = 1-bit carry, IS_BIT-pinned)
ρ:  μ · (in·2^rnc  − right·2¹⁶ − left) = 0     (rnc = KECCAK_RHO[x][y] % 16)
```

with `left`/`right` the range-checked (`AreBytes`) byte-pair halves. The QF-BV model
encodes each shift as the *unique* decomposition those identities force —
`left = (in << rnc) mod 2¹⁶`, `right = in >> (16 − rnc)`, the byte-pair widths
supplying the `[0, 2¹⁶)` bound — so the gate is faithful to the inlined round even
though the module comments describe it as the pre-inline "HWSL" circuit; the two are
constraint-identical in QF-BV. Verified: `keccak_rnd.rs` is byte-identical across
`main`, this branch, and `6a280121` (same git blob `51b7759f`), so the wiring the
model transcribes is the shipped wiring.

The `rs:NNN` line citations in the code comments are nevertheless **stale**. They were
written against the pre-#889 revision (`d83b4d9e`, blob `1b121a8b`, 926 lines), where
each lands exactly on the construct it names; **#889 — the change that inlined the
HWSL shifts — invalidated them all.** Most now point at a neighbouring construct:
`rs:539-588` ("theta: Cxz XOR chain") is the KeccakRc sender, the chain being at
`546-597`; `rs:796-870` ("chi: AND then XOR") is Iota, Chi's AND/XOR being at
`716-759` and `761-794`.

Two cannot be repointed at all: `rs:593-631` (θ HWSL) and `rs:723-766` (ρ HWSL) cite
`BusInteraction::sender(BusId::Hwsl, …)` blocks that #889 **deleted outright** — the
file now contains zero `BusId::Hwsl` sends, and that content lives in the inline
identities at `:882-894` and `:896-920`. So it is not the case that every referenced
construct still exists.

Locate a construct by its `// --- Step: … ---` banner rather than by these line
numbers.

**Known scope gap carried as the first follow-up:** QF-BV cannot test that the
`AreBytes`/`IS_BIT` bounds are *sufficient* mod `p` for the inline identities (bit
vectors make `2¹⁶` a zero divisor, not the invertible element it is mod the
Goldilocks prime). That companion proof — an integer-mod-`p` model showing that
dropping a range bound makes the decomposition ambiguous (SAT) — was written for the
optimization PR that introduced the identities and is *not* included in this
baseline. Porting it here (or moving to a solver with native field support) is the
first extension of this template.

## Scope

- **In scope: bit/byte-oriented hashes** — Keccak/SHA-3, SHA-2, BLAKE3. Their round
  functions are boolean/byte algebra, which QF-BV models exactly and z3 decides
  efficiently.
- **Out of scope: native-field chips** — e.g. Poseidon/Poseidon2, whose round is
  arithmetic in the STARK field. Bitvectors are the wrong theory; use a finite-field
  solver (`cvc5` with the `FF` theory) or a proof assistant (Lean). This baseline
  deliberately does not attempt them.
- The check is **one round's transition given the helper-chip contracts**. It does
  not re-verify the helper chips (BITWISE is a fully enumerated `2²⁰`-row
  preprocessed table; the range chips are separate), nor cross-row/multiplicity
  gating beyond what the μ column expresses.

## Sibling verifications following this method

- The **BLAKE3 chip gate** (`thoughts/blake3/` on `feat/blake3-accelerator`, PR #903)
  uses the same oracle+QF-BV+controls structure, under an older layout. It is not
  merged, so it is cited by branch rather than by path.
- This directory is the canonical write-up; new chip gates should mirror its file
  layout and its Mandatory-discipline checklist.

## Files

- `z3_verify.py` — the gate: free-var QF-BV model of the round, the typed contracts,
  the 24-round UNSAT check, the positive control, and the `bug=` negative controls.
- `z3_parallel.py` — parallel driver for the gate (all-UNSAT ×24 + controls).
- `tamper_test.py` — changed-constraint **and** removed-constraint controls, with the
  forged witnesses exhibited for the removed-constraint cases.
- `keccak_ref.py`, `test_ref.py` — independent FIPS-202 reference (RC/RHO generated
  from the spec recurrences) + external anchoring against `hashlib` and the repo
  constants.
- `model_dataflow.py`, `test_dataflow.py` — concrete byte-level forward mirror of the
  modeled equations, validated against the reference over random/structured inputs
  and confirmed to move under each injected bug.

## Running the gate

z3's Python bindings are the only dependency (no cargo, no repo build):

```
pip install z3-solver            # if not already importable
cd formal_verification/keccak
python3 test_ref.py              # reference constants + SHA3 vs hashlib
python3 test_dataflow.py         # concrete mirror vs reference (+ bug sanity)
python3 z3_parallel.py           # the gate: 24 rounds + changed-constraint controls
# or, single-process with inline printout:
python3 z3_verify.py
python3 tamper_test.py           # the removed-constraint controls (see discipline 1)
```

`tamper_test.py` is not optional: `z3_parallel.py` runs only the *changed*-constraint
controls, so a run that skips it exercises no **removed**-constraint control at all.

Expected board: positive control PASS, all negative controls **SAT** (caught), all
24 rounds **UNSAT**. Anything else is a real signal — investigate before trusting.
