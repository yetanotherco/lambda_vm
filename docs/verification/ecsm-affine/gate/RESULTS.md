# ECSM affine-selector z3 gate — lemma board & soundness theorem

Reverse-order verification of the surface **PR #879** adds to `prover/src/tables/ecsm.rs`
and `executor/src/vm/instruction/execution.rs`, following the playbook of the two earlier
campaigns in this tree:

- `thoughts/ec-recover-opt/gate/RESULTS.md` on branch `feat/ec-lincomb2`, commit `1d2b4dd7`
  — the ORIGINAL ECSM/ECDAS board. Its lemmas L1–L7 and contracts C1–C7 are **imported as
  hypotheses** here; this board does not re-derive them.
- `thoughts/blake3/` on branch `feat/blake3-accelerator` (PR #903) — the gate-then-implement
  pattern, and its postmortem on which pass is worth doing (the transcription audit).

Both paths are under the gitignored `thoughts/` and exist only on those unmerged branches, not
on `main`; this campaign lives in `docs/verification/` instead (rationale in `../README.md`).

Model transcribed from the Rust with `file:line` citations inline in the scripts; independent
reference = `../oracle/ecsm_affine_ref.py`, a from-scratch secp256k1 implementation (no
`k256`, no `num_bigint`, no repo code).

**Faithfulness anchors, both green, both run before any UNSAT was read:**

1. **Function** — `../oracle/test_oracle.py`: 9/9 anchors, `ORACLE STATUS: VALIDATED`.
   Repo constants parsed and matched, 5 published multiples of `G`, 216 x-only-agreement
   pairs, 200 root-dependence instances, the executor's 8 rejections, the ABI predicates over
   193 offsets, 60 ecrecover-equivalence instances, 25 vectors against the PyPI `ecdsa`
   package.
2. **Columns** — `a6_real_witness.py` over 32 witnesses emitted by the repo's own
   `ecsm::compute_witness{,_with_y}` (`../harness/`): every field the model reads is
   re-derived, including all four overflow chains, both convolution relations and their carry
   windows.

---

## Board

| Lemma | Statement | Verdict | Notes |
|---|---|---|---|
| **A1-PRIME** | `p_g = 2^64 − 2^32 + 1` is prime | CERTIFIED (sympy) | the one assumed algebraic fact; the field property every root argument uses |
| **A1a** | `IS_BIT(IS_AFFINE)` (idx 421) ⇒ `IS_AFFINE ∈ {0,1}` | PROVED (complete split over GF(p_g)) | root set `{0,1}`, factorisation complete ⇒ exhaustive |
| **A1b** | `AffineZeroOnPadding` (idx 422) ⇒ `µ=0` forces `IS_AFFINE=0` | PROVED (z3 UNSAT) | what lets the `yG`/`yR` buses use `Multiplicity::Column(IS_AFFINE)` with no separate padding argument |
| **A1c** | the `Ecall` tuple is **injective** in `IS_AFFINE` | PROVED | the pinning: a row flipping the selector no longer matches the CPU's real `a7` |
| **A1c-ctl** | the degenerate syscall pair un-pins the selector; the repo pair does not | PROVED + 1 SAT | identical numbers ⇒ constant tuple ⇒ `IS_AFFINE` free |
| **A1c-assert** | today's pinning rests **entirely** on the LOW word | PROVED | high word separates nothing (shared `0xFFFF_FFFF`) ⇒ `execution.rs`'s `const _:` assert is load-bearing, not decorative |
| **A1d** | both new constraints are degree 2; `max_degree() == 3` still holds | PROVED | `YrLtP` reuses the existing degree-3 shape |
| **A1f** | **drop idx 421** | **SAT — FORGES** | the row's `Ecall` tuple can be made **another accelerator's**: `IS_AFFINE = 20` → HINT, `IS_AFFINE = p_g − 9` → KECCAK. `IS_BIT` is the only thing confining it to `{0,1}`. **LOAD-BEARING — and not for the reason its comment gives** |
| **A1e** | **drop idx 422** | **SAT — FORGES** | all 423 constraints walked: idx 422 is the ONLY one violated when kept, all 422 remaining are satisfied when dropped, honest padding satisfies all 423 — and the dropped row then fires 8 `IS_AFFINE`-gated MEMW ops. **LOAD-BEARING** |
| **A2a** | `YrLtP` word-carry lift: field recurrence ⇒ integer equation | PROVED (z3 UNSAT) | `\|A_i\| < 2^33 ≪ p_g`, so no `p_g` wrap |
| **A2b** | `OverflowRequired(YrLtP)` ⇒ `yR < p` | PROVED (z3 UNSAT) | `p` pinned as a numeral, not a free constant — so the conclusion is `yR < p`, not `yR < const` |
| **A2b-nv** | the same system without the denial is SAT | SAT (expected) | non-vacuity: `yR = p−1` is reachable |
| **A2c** | every `YrLtP` LHS integer value ≪ p_g under the contracts | PROVED (exact corners) | `max \|A_i\| = 2^33 − 1 = 4.7·10⁻¹⁰·p_g` |
| **A2c-ctl** | wrong constant `p → p+2` / `p → N` | **SAT — CATCHES** | with the witness held FIXED, the honest columns stop satisfying the chain ⇒ it binds `p` itself (keccak wrong-RC analogue) |
| **A2d** | contract **C4-YR**: where `YR`'s byte bound comes from | CONTRACT | `ecsm.rs` byte-checks `{X2, Q0, YG, Q1}` — **not** `YR`. Inherited via two exhaustive cases on `len_k`; bus-level, so outside this gate (C5 + imported L6) |
| **A2d-obs** | `YrLtP` is **µ**-gated, not `IS_AFFINE`-gated | NOTED | binds x-only rows too: strictly stronger, and completeness holds because `witness.rs` fills `y_r_sub_p` on both paths |
| **A2e** | honest-witness anchor for the chain | PROVED | 14 witnesses (4 x-only, 10 affine): every `c_i ∈ {0,1}`, `c_7 = 1`, halfwords in `[0,2^16)` |
| **A2f** | **the forgery, fully instantiated** | **SAT — FORGES** | see below |
| **A2g** | **drop `YrLtP`** | **SAT — FORGES** | the A2f witness is accepted; the guest receives `yR + p`. **LOAD-BEARING** (the `yR`-side analogue of the earlier board's N6 / `XR_SUB_P`) |
| **A2h** | the excluded band is populated by real curve points | PROVED | `2^256 − p = 2^32 + 977`, and a real secp256k1 point has **`y = 1`** |
| **A3a** | `yG`'s parity is arithmetically FREE | PROVED (complete split over GF(p)) | `Y² − (x³+b) = (Y−y)(Y+y)`; both roots distinct at all 6 tested x (no `y=0` point) |
| **A3b** | **the parity forgery, fully instantiated** | **SAT — FORGES** | two complete witnesses over the same `(xG, k)`, both satisfying all 423 in-table constraints, same `xR`, different `yR`. Swept over 12 random `(P, k)` |
| **A3c** | the `yG` read pins `YG` bit-for-bit | PROVED | 4 dwords × 8 bytes ↔ `addr_xG + [32, 64)`, order-preserving, each column covered exactly once |
| **A3d** | **drop the `yG` read** | **SAT — FORGES** | the two A3b witnesses become indistinguishable. The read is the **only** thing pinning input parity. **LOAD-BEARING** |
| **A3e** | the imported **L7** conclusion survives verbatim | PROVED | `x(k·P) = x(k·(−P))` over 20 instances ⇒ x-only rows may still leave parity free |
| **A3f** | `YrLtP` is **not** a parity defence | PROVED | both `±yR` are canonical ⇒ two orthogonal gaps, two fixes; do not conflate them |
| **A4a** | the `Alu` LT bound **==** the executor's `addr_limb_ok`, both modes | PROVED (z3 UNSAT ×2) | same accept set ⇒ no provable-but-halting execution, no legal execution made unprovable |
| **A4b** | the affine `+32 + 8i` span cannot cross `2^32` | PROVED (z3 UNSAT) | 128 touched byte offsets, max `+63`, all `< 2^32` under the bound ⇒ reusing the high limb is safe |
| **A4c** | the **seven-value band** the LT senders close | PROVED | exactly 7 per mode: `[2^32−31, 2^32−24)` and `[2^32−63, 2^32−56)`. The PR comment's number, measured |
| **A4d** | `k`'s bound is flat in both modes, correctly | PROVED (z3 UNSAT) | `k` is 32 B on both arms |
| **A4e** | the overlap guard **==** exact interval disjointness | PROVED (z3 UNSAT) | not a distance bound: `addr_k + 32 == addr_xg` is legal and must stay legal |
| **A4e-ctl** | **the `u64` wrap** | **SAT — FORGES** | `addr_xg = 2^64 − 64` passes `addr_limb_ok(·, 63)` and wraps the pre-fix `+64`, slipping a **total** operand overlap past the guard. The `u128` widening is **LOAD-BEARING** |
| **A4f** | timestamp layout is collision-free | PROVED | `{xG, yG}@ts`, `k@ts+1`, `xR@ts+2`, `yR@ts+3`, stride 4 (parsed from the builder). `xG`/`yG` share `ts` but are address-disjoint |
| **A4g** | the mode-dependent bound is necessary in **both** directions | PROVED | a flat 64-byte bound rejects 32 legal x-only addresses (completeness); a flat 32-byte bound admits 32 illegal affine ones (soundness) |
| **A5** | transcription audit | 19/19 premises READ, 19/19 mutations bite | `TRANSCRIPTION-AUDIT.md` |
| **A6** | real-witness anchor | PROVED (+1 forgery exhibit) | 32 witnesses from `crypto/ecsm` itself; 9 ±yG pairs reproduce A3b outside the model |

Non-vacuity: **seven distinct attacks** — A1c-ctl, A1e, A1f, A2c-ctl, A2f/A2g, A3b/A3d,
A4e-ctl. They surface as **11 `SAT` results** across the lemma files, because several are
exhibited from more than one angle (the parity attack appears as A3b's instance, A3b's
12-instance sweep, A3d's counterfactual, and again as A6c straight out of `crypto/ecsm`). Every new check in the PR now has a control showing it is load-bearing;
none is dead weight. A1f was **missing from the first version of this board** — idx 421 was
proved *correct* (A1a) but never shown *necessary*, and "every new check is load-bearing" was
asserted on five of the six. See Finding 7.

---

## A2f — the `yR + p` forgery, carried all the way through

The PR's soundness section claims the excluded band is populated: "such points are
constructible: `3 | p−1` makes cubing 3-to-1, so a small target `y` has a cube-root preimage
about a third of the time". `../oracle/small_y_point.py` builds the point, and the claim is
not just true but extreme — **the first candidate, `y = 1`, works**:

```
y  = 1
x  = 0x1fe1e5ef3fceb5c135ab7741333ce5a6e80d68167653f6b2b24bcbcfaaaff507   (on curve)
y + p = 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc30 < 2^256
```

Reached through the chip's generic path as `2·(2^{−1}·Q)` (`k = 2`, one ECDAS doubling row —
not the `k = 1` echo). The forged witness publishes `yR' = yR + p` and compensates with
`q2' = q2 − 1`; the relation's residual is `−p·1 + p·1 = 0`. A2f checks **12** facts about it,
including the two that a value-level argument would skip:

- the forged ECDAS `Yr` relation holds **exactly** — all 64 limb divisions are exact and the
  chain closes at `c_63 = 0`;
- the forged **carries stay inside** the `IsHalfword` window `[−16320, 49216)`.

A forgery its carry windows reject is not a forgery, which is why those two are in the list.
`crypto/ecsm` itself returns `y_r = 1` for this instance (A6e), so nothing here depends on the
model being right about the arithmetic.

---

## A3 — the parity gap, and why it is new

The chip constrains `yG` only through `yG² ≡ xG³ + b (mod p)`. Both roots satisfy it, and the
earlier board says so in as many words: L7 concludes `xR = x(k·P)` **"for both yG sign
classes"**, *because* `x(k·P) = x(k·(−P))`. That was a fine conclusion for an x-only ABI.

Publishing `yR` retires the premise, not the lemma. A witness may take `−yG`, compute a
perfectly correct multiple of a **different** point, and hand the guest its `y`:

- the AIR cannot tell — an on-curve check passes for either root;
- the guest cannot tell — knowing the parity of `k·P` is precisely the work it delegated.

A3b builds both witnesses out of real values and evaluates the whole in-table constraint set
on each. Note which columns move: `x2` and `q0` depend on `xG` alone and are **identical**;
`q1` differs, because the `Yg` relation's numerator uses the *integer* `yG²` and
`(p−y)² ≠ y²` over ℤ. So the forged witness needs its own quotient — and it fits its 33-byte
contract, and the `c1` carries still fit the `+16319` window. A6c reproduces the same pair
straight out of `ecsm::compute_witness_with_y`, 9 times.

The fix is the `IS_AFFINE`-gated MEMW read, and A3c checks the thing that would quietly break
it: **coverage**. The forgery needs only one byte of freedom, so a read covering 31 of the 32
bytes would close nothing. It covers all 32, exactly once, order-preserving.

**A3f is a trap worth stating explicitly:** `YrLtP` is *not* a second line of defence here.
Both `±yR` are canonical field elements, so the range check admits either. Two orthogonal
gaps — input parity (A3) and output representation (A2) — with two independent fixes.

---

## Soundness theorem (what this board adds)

Under contracts C1–C7 (imported) + C4-YR + A1-PRIME, and given the earlier board's L1–L7, any
accepted trace satisfies, for every ECSM row with `µ = 1` and ecall timestamp `ts`:

> `IS_AFFINE` equals the mode of the ecall the CPU actually executed. When it is 0, the row's
> behaviour is **bit-identical** to the pre-PR chip and the imported conclusion stands
> unchanged: `xR = x(k·P)` for either lift of `xG`. When it is 1, the witnessed `yG` is the
> 32 bytes the caller placed at `addr_xG + 32`, so `(xG, yG)` is the caller's own point; the
> published `(xR, yR)` are the canonical affine coordinates of `k·(xG, yG)`, both `< p`; and
> the operand addresses are exactly those the executor accepts.

Chain of proof: `IS_AFFINE` is a bit (A1a), dead on padding (A1b) and pinned to the executed
ecall (A1c) → the affine buses fire exactly on affine rows → the `yG` read pins the input
point (A3c), closing the parity freedom A3a/A3b exhibits → the imported L1–L7 give
`(xR, yR) ≡ k·(xG, yG) (mod p)` → `XR_SUB_P` and the new `YrLtP` (A2a/A2b) make both
coordinates canonical → the LT senders align the AIR's address set with the VM's (A4a–A4c) and
the `+32 … +63` span cannot leave its limb (A4b).

**Completeness** (no honest rejection): A2e and A6a — real witnesses from `crypto/ecsm`
satisfy every new constraint, including on the x-only path where `YrLtP` also binds (A2d-obs);
A4g shows the mode-dependent bound is what keeps legal x-only addresses provable.

---

## Contracts (assume-guarantee boundary)

Imported verbatim from the earlier board: **C1** AreBytes, **C2** IsHalfword, **C3**
IS_BIT/booleans, **C4** MEMW byte authority, **C5** LogUp multiset soundness, **C6** Ecall
binding, **C7** timestamp uniqueness, **A-PRIME** (`p`, `N` prime).

New, and the one this board had to look up rather than assume:

- **C4-YR** — `YR`'s bytes are in `[0, 256)`. `ecsm.rs` does **not** emit this: its `is_byte`
  list is `{X2, Q0, YG, Q1}`. The bound is inherited by two exhaustive cases on `len_k`:
  `len_k ≥ 1` ⇒ the `Ecdas` drain tuple equals an ECDAS sender's byte-checked `yR`;
  `len_k = 0` (`k = 1`) ⇒ no ECDAS row can receive round `−1`, so balance forces
  drain = seed, i.e. `YR = YG`, which *is* byte-checked here. Both cases are bus-level (C5 +
  imported L6) and therefore **invisible to this gate** — which is why C4-YR is written down
  instead of being quietly used.
- **A1-PRIME** — `p_g` prime (sympy-certified).

---

## Findings

1. **[confirmed claim] The `y = 1` point.** The PR's constructibility argument for the
   `YrLtP` band is correct, and stronger than stated: the smallest possible `y` is attained,
   so the attack instance sits at the very bottom of a `2^32 + 977`-wide band. No action —
   recorded because "astronomically rare" was the phrasing used for the `XR_SUB_P` analogue
   on the earlier board (Finding 5), and it is not rare here in any useful sense.
2. **[confirmed claim] The seven-value address band.** `ecsm.rs`'s comment claims the LT
   senders close "a seven-value band per operand". Measured: exactly 7, in both modes
   (A4c). No action.
3. **[observation, no action] `YrLtP` is µ-gated, not `IS_AFFINE`-gated.** So it binds on
   x-only rows, where nothing observes `yR`. Strictly stronger ⇒ sound, and completeness
   holds because `compute_witness_inner` fills `y_r_sub_p` on both paths. Gating it on
   `IS_AFFINE` would save 16 halfword sends + 8 constraints on x-only rows; not worth the
   asymmetry with the other three chains. Recorded so it reads as a choice rather than an
   oversight.
4. **[the pinning is narrower than it looks] `IS_AFFINE` is separated by ONE 32-bit word.**
   The two syscall numbers share their high word, so that word's `IS_AFFINE` coefficient is
   zero and carries no mode information (A1c-assert). The entire pinning is the low word, and
   `execution.rs:53-58`'s `const _: () = assert!` is what keeps it that way. The assert is
   *conservative* — it also rejects a pair differing only in the high word, which would in
   fact still be injective — and that is the right side to err on. No action; the audit's P5
   fails if the assert is removed.
5. **[audit method] A blind check is worse than a missing one.** P18 (timestamp stride)
   originally matched `ecsm.rs`'s *comment* about the stride and compared it against a
   hard-coded 4 in the model. It passed. The mutation control then showed it survived
   reducing the real stride to 3 — i.e. it was checking documentation. Now it parses the
   stride out of `trace_builder.rs` and compares the parsed value. **Every premise here is
   mutation-tested for exactly this reason.**
6. **[control hygiene] The wrong-constant control was initially vacuous.** A2c's first form
   recomputed the witness addend for the perturbed constant, so *any* constant passed. Fixed
   by holding the witness fixed — the prover commits its columns against the real `p`, and
   that is what the control must model. Both this and Finding 5 are the same lesson: a green
   control is worth nothing until you have seen it go red.

## The rule those two findings generalise to

> **Every negative control must be PAIRED with the specific check that the dropped premise is
> load-bearing for — and the pairing must be evaluated in both directions.**

A control that drops a premise and then re-runs a check whose reference never mentioned that
premise *cannot fail*, and reads green. Findings 5 and 6 are both instances; a parallel
campaign on the DMA memcpy PR (#874) hit the identical shape three times
(`drop_tail_lane_zero`, `drop_lt_bound`, `drop_reg32`), which is enough independent recurrence
to write the rule down rather than the anecdotes.

Concretely, each control here states both halves:

| Control | half 1: premise KEPT must block | half 2: premise DROPPED must admit |
|---|---|---|
| A1e | idx 422 is the only violated constraint of 423 | all 422 remaining satisfied, 8 MEMW ops fire |
| A2c-ctl | honest witness (addend fixed) valid under `p` | rejected under `p+2` and under `N` |
| A2g | A2f established every *other* constraint holds | the witness is accepted, guest gets `yR + p` |
| A3d | A3c: the read's tuples differ between the two | `check_all_constraints` clean on both |
| A4e-ctl | the `u128` form rejects the overlap | `addr_limb_ok` passes *and* the `u64` form accepts |
| A1c-ctl | the repo pair is injective | the degenerate pair collides |
| A1f | idx 421 kept ⇒ only ECSM/ECSM_AFFINE reachable | dropped ⇒ HINT and KECCAK reachable |

**The rule paid for itself twice.** A1f's first implementation had an inverted sign in its
modular solve, so it found no foreign syscalls. Half 2 alone would have reported
"idx 421 REDUNDANT" — a confident, wrong conclusion. Half 1 (`idx 421 kept ⇒ exactly
`{ECSM, ECSM_AFFINE}` reachable`) failed instead, which is what surfaced the bug.

A1e originally listed "the constraint families I believe remain" rather than walking the index
map. That is the mirror-image hazard — an over-permissive control reports a false FORGES by
forgetting a constraint that would have blocked the state — so it now enumerates all 423 and
asserts the count.

---

7. **[the board's own gap, now closed] `IS_BIT(IS_AFFINE)` is load-bearing for a reason
   nobody had written down.** The first version of this board proved idx 421 *correct* (A1a)
   and never asked whether it was *necessary*. It is, and the argument is not local to the
   ECSM pair at all.

   The receiver's syscall words are `xonly + a·(affine − xonly)` per 32-bit word, and for the
   repo's numbers the coefficients are `−1` (low) and **`0`** (high, since both share
   `0xFFFF_FFFF`). So as `a` sweeps the field the high word is CONSTANT and the low word
   sweeps everything — and every syscall is `u64::MAX − k`, so they all share that high word.
   Each is therefore reachable at `a = (target_lo − xonly_lo)/(−1)`:

   | reached tuple | `IS_AFFINE` |
   |---|---|
   | `ECSM` | 0 (honest) |
   | `ECSM_AFFINE` | 1 (honest) |
   | **`HINT`** | **20** |
   | **`KECCAK`** | **p_g − 9** |

   Drop idx 421 and `a` is free on a `µ=1` row (idx 422 only binds at `µ=0`), so an ECSM row
   can consume the `Ecall` send of a *different* accelerator: the guest's HINT call gets
   proven as a scalar multiplication that writes 32 or 64 bytes wherever the ECSM row's
   register columns point.

   **What does not close this:** `execution.rs`'s `const _: () = assert!` compares only the
   two ECSM numbers' low words. It cannot see a *third* syscall sitting an integer offset
   away — which is the reachable case. That is a documentation gap in #879, not a bug: idx 421
   is present and does the job. Recorded as a review note below.

## Method notes

- **`x·(1−x) ≡ 0 (mod p_g)` is not a z3 query.** Handed to z3 in lifted integer form
  (`x(1−x) = m·p_g`, `m` free) it does not terminate at this modulus — nonlinear integer
  arithmetic with an unbounded quotient. Bit-blasting it (160-bit `URem`) does not terminate
  either. These are **root-of-a-polynomial-over-a-field** statements, and they are discharged
  as such: sympy factors over `GF(q)` and the split is checked **complete** (linear factors
  account for the full degree), so the returned root set is provably exhaustive. z3 is kept
  for what it is good at here — the bounded/linear lifts (A2a, A2b), the predicate-equivalence
  sweeps (A4a, A4d) and the quantified interval statement (A4e).
- **Two different fields.** The AIR is enforced over `GF(p_g)`; the curve lives over `GF(p)`.
  `field_roots` takes the modulus explicitly because an early version of A3a factored the
  curve polynomial over Goldilocks and correctly reported FAIL.
- **Forgeries are constructive.** Where a control is a genuine forgery it is exhibited as
  fixed numerals evaluated against the transcribed constraints — the solver is never asked
  whether an attack *might* exist.
- **Never negate an equation that carries a witness quotient.** The lifts here put the free
  quotient on the *hypothesis* side and deny a quotient-free conclusion: A2a asserts
  `A − 2^32·c == m·PG` and denies `A != 2^32·c`; A1b asserts `a − m·PG == 0` and denies
  `a != 0`. That direction is sound, because extra freedom in `m` only strengthens the
  adversary. The inverse is a trap that produces a **bogus SAT**: `Not(a − b == k·m)` is
  satisfiable by picking `k != 0`, so the solver "refutes" a true statement. Under negation the
  claim has to be spelled out witness-free — an explicit residue variable bounded to `[0, p)`,
  or a two-way disjunction. Audited: every z3 query on this board asserts its quotient
  equations positively, and no denial contains a quotient variable (A2b, A4a, A4d and A4e carry
  no quotients at all).
- **The tractable rewrite, where it applies.** Keeping `Int` (not `BitVec`) and rewriting
  `a ≡ b (mod m)` as `a − b == k·m` with a fresh free `k` is linear whenever `m` is a constant,
  which is why A2a/A2b are milliseconds. Rewriting `x·(1−x) == 0` as `Or(x == 0, x == 1)` is
  the same trick one level up and is used throughout the downstream lemmas (A2b's carry bits,
  A1e's padding row). It is **not** available for A1a itself — there the disjunction *is* the
  conclusion, so assuming it would be circular, and the field-factorisation argument is what
  earns it.

---

## Reproduction

From `docs/verification/ecsm-affine/`:

```bash
python3 -m venv .venv && ./.venv/bin/pip install z3-solver sympy ecdsa
./run_gate.sh                 # everything, logs to gate/logs/
./run_gate.sh --quick         # reuse the existing witness dump (skip the cargo build)
```

`run_gate.sh` cds to its own directory, so it also works from the repo root as
`docs/verification/ecsm-affine/run_gate.sh`. The two scripts that read repo source locate the
root by marker (a workspace `Cargo.toml` next to `prover/`), not by a hard-coded `parents[N]`
— see the note under "Where to send the next reviewer".

Individual stages, in the order the board depends on them:

```bash
cd gate   && python audit_transcription.py   # A5 — premises still true of the code?
cd oracle && python test_oracle.py           # the function anchor
cd oracle && python small_y_point.py         # builds the y = 1 attack instance
cd harness && cargo run --release -- > ../gate/logs/real_witnesses.jsonl
cd gate   && python a6_real_witness.py       # the column anchor
cd gate   && python a1_selector.py           # A1
cd gate   && python a2_yr_lt_p.py            # A2
cd gate   && python a3_parity_binding.py     # A3
cd gate   && python a4_addressing.py         # A4
```

Total runtime is a few seconds plus the `cargo build` (the harness depends only on
`crypto/ecsm`, deliberately — a harness that needs the prover does not get run).

---

## Where to send the next reviewer

- **Bus-level reasoning is outside this gate**, as it was outside both earlier ones. C4-YR,
  the `Ecall` pinning's LogUp step (A1c reduces to it), and A3c's "at most one witness matches
  the caller's buffer" all bottom out in C5 + the imported L6. A gate that models the
  arithmetic cannot see a mis-wired bus; the e2e `prove + verify` tests in
  `prover/src/tests/ecsm_tests.rs` and `prove_elfs_tests.rs` are what cover that, and they are
  green on this branch.
- **The `q1` growth on the `−yG` branch** is checked to fit 33 bytes at the instances tested,
  not proved to fit for all inputs. It is a completeness question about a witness nobody
  should be building, so it is deliberately left as an observation — but if the affine path
  ever *needs* both roots, it becomes a real bound to establish.
- **Nothing here re-proves the double-and-add chain.** If PR #879's rebase ever touches the
  ECDAS chain or the `Ecdas`/`Bit` buses, the imported board is the one to re-run, not this
  one.

## Spec gap — a review note on #879, deliberately NOT fixed here

`spec/` carries one `.typ` chapter per table (34 of them: add, bitwise, commit, keccak, lt,
memw, mul, page, sha256, shift, store, …). **There is no ECSM chapter, under any name.** No
`ecsm`/`ecdas` file, no `spec/src/ecsm.toml`, and
`grep -rlin "ecsm|scalar mul|secp256" spec/*.typ spec/src/*.toml` returns nothing.
(`spec/signatures.typ` is a meta-chapter that renders bus/template signatures — unrelated to
ECDSA.)

Worse, `prover/src/tables/ecsm.rs:19` says **"See `spec/src/ecsm.toml`"** — a file that does
not exist and never has. That is a dangling reference in the very file #879 modifies.

Both are **pre-existing** and not #879's to create: PR #903 added `spec/blake3.typ` because it
introduced a *new* table, whereas #879 extends an existing one whose chapter was never written.
Writing the missing chapter inside a verification PR would also mean a `spec/book.typ` edit —
a shared merge-conflict surface — for a gap that predates the branch. Raise it as a separate
review note on #879 and a follow-up issue instead. (The parallel DMA campaign found the same
class of gap on #874, which *does* add a new fixed table — `FIXED_TABLE_COUNT` 10 → 11 — with
no `spec/dma.typ`; that one is arguably blocking, this one is not.)

For the same reason nothing is added to `docs/SUMMARY.md`: there is no `book.toml` anywhere in
the repo, no workflow or Makefile target references mdbook, and `SUMMARY.md` already omits 7
existing `docs/` files (`ai-review.md` plus 6 under `cryptography/`). So an unindexed file
breaks nothing, while a new top-level heading would put an internal verification campaign into
a user-facing TOC and widen the diff into a shared conflict surface.
