# ECSM affine-selector z3 gate — lemma board & soundness theorem

Reverse-order verification of the surface **PR #879** adds to `prover/src/tables/ecsm.rs` and
`executor/src/vm/instruction/execution.rs`. Model transcribed from the Rust with `file:line`
citations inline in the scripts; independent reference = `ecsm_affine_ref.py`, a from-scratch
secp256k1 implementation (no `k256`, no `num_bigint`, no repo code).

Two earlier campaigns supply the playbook, both on unmerged branches under the gitignored
`thoughts/` (this campaign lives in `formal_verification/` instead, per PR #923 — see
`README.md`):

- `thoughts/ec-recover-opt/gate/RESULTS.md` (`feat/ec-lincomb2` @ `1d2b4dd7`) — the original
  ECSM/ECDAS board. Its **L1–L7 and C1–C7 are imported as hypotheses**; not re-derived here.
- `thoughts/blake3/` (`feat/blake3-accelerator`, PR #903) — the gate-then-implement pattern and
  its postmortem naming the transcription audit as the highest-value pass.

**Faithfulness anchors, both run before any UNSAT was read:**

1. **Function** — `test_oracle.py`, 9/9 (`VALIDATED`): repo constants parsed and matched, 5
   published multiples of `G`, 216 x-only-agreement pairs, 200 root-dependence instances, the
   executor's 8 rejections, ABI predicates over 193 offsets, 60 ecrecover-equivalence
   instances, 25 vectors against the PyPI `ecdsa` package.
2. **Columns** — `a6_real_witness.py` over 32 witnesses from the repo's own
   `ecsm::compute_witness{,_with_y}` (`harness_dump.rs`): every field the model reads is
   re-derived, including all four overflow chains and both convolution relations.

---

## Board

| Lemma | Statement | Verdict | Notes |
|---|---|---|---|
| **A1-PRIME** | `p_g = 2^64 − 2^32 + 1` is prime | CERTIFIED (sympy) | the one assumed algebraic fact |
| **A1a** | `IS_BIT(IS_AFFINE)` (idx 421) ⇒ `IS_AFFINE ∈ {0,1}` | PROVED (complete split over GF(p_g)) | root set `{0,1}`, split complete ⇒ exhaustive |
| **A1b** | `AffineZeroOnPadding` (idx 422) ⇒ `µ=0` forces `IS_AFFINE=0` | PROVED (z3 UNSAT) | lets the `yG`/`yR` buses use `Multiplicity::Column(IS_AFFINE)` with no separate padding argument |
| **A1c** | the `Ecall` tuple is **injective** in `IS_AFFINE` | PROVED | the pinning: a row flipping the selector no longer matches the CPU's real `a7` |
| **A1c-ctl** | the degenerate syscall pair un-pins the selector; the repo pair does not | PROVED + 1 SAT | identical numbers ⇒ constant tuple ⇒ `IS_AFFINE` free |
| **A1c-assert** | today's pinning rests **entirely** on the LOW word | PROVED | high word separates nothing (shared `0xFFFF_FFFF`) ⇒ the `const _:` assert is load-bearing |
| **A1d** | both new constraints are degree 2; `max_degree() == 3` holds | PROVED | `YrLtP` reuses the existing degree-3 shape |
| **A1e** | **drop idx 422** | **SAT — FORGES** | all 423 walked: idx 422 is the only one violated when kept, all 422 satisfied when dropped, honest padding satisfies all 423 — the dropped row fires 8 gated MEMW ops. **LOAD-BEARING** |
| **A1f** | **drop idx 421** | **SAT — FORGES** | the row's `Ecall` tuple becomes **another accelerator's**: `IS_AFFINE = 20` → HINT, `p_g − 9` → KECCAK. **LOAD-BEARING**, and not for the reason its comment gives (Finding 7) |
| **A2a** | `YrLtP` word-carry lift: field recurrence ⇒ integer equation | PROVED (z3 UNSAT) | `\|A_i\| < 2^33 ≪ p_g`, so no `p_g` wrap |
| **A2b** | `OverflowRequired(YrLtP)` ⇒ `yR < p` | PROVED (z3 UNSAT) | `p` pinned as a numeral ⇒ the conclusion is `yR < p`, not `yR < const` |
| **A2b-nv** | the same system without the denial is SAT | SAT (expected) | non-vacuity: `yR = p−1` reachable |
| **A2c** | every `YrLtP` LHS integer value ≪ p_g under the contracts | PROVED (exact corners) | `max \|A_i\| = 2^33 − 1 = 4.7·10⁻¹⁰·p_g` |
| **A2c-ctl** | wrong constant `p → p+2` / `p → N` | **SAT — CATCHES** | witness held FIXED ⇒ honest columns stop satisfying the chain (keccak wrong-RC analogue) |
| **A2d** | contract **C4-YR**: where `YR`'s byte bound comes from | CONTRACT | `ecsm.rs` byte-checks `{X2, Q0, YG, Q1}` — **not** `YR`; inherited, bus-level, outside this gate |
| **A2d-obs** | `YrLtP` is **µ**-gated, not `IS_AFFINE`-gated | NOTED | binds x-only rows too: strictly stronger, completeness holds (Finding 3) |
| **A2e** | honest-witness anchor for the chain | PROVED | 14 witnesses (4 x-only, 10 affine): `c_i ∈ {0,1}`, `c_7 = 1`, halfwords in `[0,2^16)` |
| **A2f** | **the `yR + p` forgery, fully instantiated** | **SAT — FORGES** | 12 facts, incl. the ECDAS `Yr` relation holding **exactly** and the forged carries staying inside `[−16320, 49216)` — a forgery its windows reject is not one |
| **A2g** | **drop `YrLtP`** | **SAT — FORGES** | the A2f witness is accepted; the guest receives `yR + p`. **LOAD-BEARING** (the `yR`-side analogue of the earlier board's N6 / `XR_SUB_P`) |
| **A2h** | the excluded band is populated by real curve points | PROVED | `2^256 − p = 2^32 + 977`, and a real secp256k1 point has **`y = 1`** (Finding 1) |
| **A3a** | `yG`'s parity is arithmetically FREE | PROVED (complete split over GF(p)) | `Y² − (x³+b) = (Y−y)(Y+y)`; roots distinct at all 6 tested x (no `y=0` point) |
| **A3b** | **the parity forgery, fully instantiated** | **SAT — FORGES** | two complete witnesses over the same `(xG, k)`, both satisfying all 423 in-table constraints, same `xR`, different `yR`; swept over 12 random `(P, k)` |
| **A3c** | the `yG` read pins `YG` bit-for-bit | PROVED | 4 dwords × 8 bytes ↔ `addr_xG + [32, 64)`, order-preserving, each column exactly once — one free byte would suffice for A3b |
| **A3d** | **drop the `yG` read** | **SAT — FORGES** | the two A3b witnesses become indistinguishable. The read is the **only** thing pinning input parity. **LOAD-BEARING** |
| **A3e** | the imported **L7** conclusion survives verbatim | PROVED | `x(k·P) = x(k·(−P))` over 20 instances ⇒ x-only rows may still leave parity free |
| **A3f** | `YrLtP` is **not** a parity defence | PROVED | both `±yR` are canonical ⇒ two orthogonal gaps (input parity A3, output representation A2), two fixes |
| **A3g** | `yG` canonicality is **UNCHECKED** | **SAT — FORGES** | no `YgLtP` in `OverflowKind`; `yG = p + 1` is a byte-representable non-canonical encoding of a real point, accepted by the AIR, rejected by the executor. Consequence BENIGN ⇒ VM-parity gap. **Medium**, Finding 8 |
| **A4a** | the `Alu` LT bound **==** the executor's `addr_limb_ok`, both modes | PROVED (z3 UNSAT ×2) | same accept set ⇒ no provable-but-halting execution, no legal execution made unprovable |
| **A4b** | the affine `+32 + 8i` span cannot cross `2^32` | PROVED (z3 UNSAT) | 128 touched offsets, max `+63`, all `< 2^32` ⇒ reusing the high limb is safe |
| **A4c** | the **seven-value band** the LT senders close | PROVED | exactly 7 per mode: `[2^32−31, 2^32−24)` and `[2^32−63, 2^32−56)` (Finding 2) |
| **A4d** | `k`'s bound is flat in both modes, correctly | PROVED (z3 UNSAT) | `k` is 32 B on both arms |
| **A4e** | the overlap guard **==** exact interval disjointness | PROVED (z3 UNSAT) | not a distance bound: `addr_k + 32 == addr_xg` is legal and must stay legal |
| **A4e-ctl** | **the `u64` wrap** | **SAT — FORGES** | `addr_xg = 2^64 − 64` passes `addr_limb_ok(·, 63)` and wraps the pre-fix `+64`, slipping a **total** overlap past the guard. The `u128` widening is **LOAD-BEARING** |
| **A4f** | timestamp layout is collision-free | PROVED | `{xG, yG}@ts`, `k@ts+1`, `xR@ts+2`, `yR@ts+3`, stride 4 parsed from the builder; `xG`/`yG` share `ts` but are address-disjoint |
| **A4g** | the mode-dependent bound is necessary in **both** directions | PROVED | a flat 64-byte bound rejects 32 legal x-only addresses; a flat 32-byte one admits 32 illegal affine ones |
| **A5** | transcription audit | 20/20 premises READ, 20/20 mutations bite | `TRANSCRIPTION-AUDIT.md` |
| **A6** | real-witness anchor | PROVED (+1 forgery exhibit) | 32 witnesses from `crypto/ecsm` itself; 9 ±yG pairs reproduce A3b outside the model |

Non-vacuity: **seven distinct attacks** (A1c-ctl, A1e, A1f, A2c-ctl, A2f/A2g, A3b/A3d,
A4e-ctl), surfacing as **11 `SAT` results** because several are exhibited from more than one
angle. Every new check in the PR has a control showing it is load-bearing; none is dead weight.

**The two headline exhibits.** The PR argues the `YrLtP` band is populated because `3 | p−1`
makes cubing 3-to-1; `small_y_point.py` builds the point and the *first* candidate works —
`y = 1`, at `x = 0x1fe1e5ef…aaaff507`, reached through the chip's generic path as `2·(2^{−1}·Q)`.
And the parity gap is built, not argued: A3b's two witnesses differ in `yG` and `q1` (the `Yg`
numerator uses the *integer* `yG²`, and `(p−y)² ≠ y²` over ℤ) while agreeing on `x2`, `q0` and
`xR` — and `crypto/ecsm` reproduces both, 9 times.

---

## Soundness theorem (what this board adds)

Under contracts C1–C7 (imported) + C4-YR + A1-PRIME, and given the earlier board's L1–L7, any
accepted trace satisfies, for every ECSM row with `µ = 1` and ecall timestamp `ts`:

> `IS_AFFINE` equals the mode of the ecall the CPU actually executed. When it is 0, the row's
> behaviour is **bit-identical** to the pre-PR chip and the imported conclusion stands
> unchanged: `xR = x(k·P)` for either lift of `xG`. When it is 1, the witnessed `yG` is the
> 32 bytes the caller placed at `addr_xG + 32`, so `(xG, yG) mod p` is the caller's own point;
> the published `(xR, yR)` are the canonical affine coordinates of `k·(xG, yG)`, both `< p`;
> and the operand addresses are exactly those the executor accepts.

**The `mod p` is load-bearing and was missing from the first version of this theorem** —
nothing constrains `yG < p` (A3g, Finding 8). The reduction is harmless, but the theorem cannot
claim the witnessed bytes *are* canonical. `xG` carries no such caveat: `XgLtP` pins it.

Chain of proof: `IS_AFFINE` is a bit (A1a), dead on padding (A1b), pinned to the executed ecall
(A1c) → the affine buses fire exactly on affine rows → the `yG` read pins the input point
(A3c), closing the parity freedom A3a/A3b exhibits → imported L1–L7 give
`(xR, yR) ≡ k·(xG, yG) (mod p)` → `XR_SUB_P` and the new `YrLtP` (A2a/A2b) make both
coordinates canonical → the LT senders align the AIR's address set with the VM's (A4a–A4c) and
the `+32 … +63` span cannot leave its limb (A4b).

**Completeness:** A2e and A6a — real witnesses satisfy every new constraint, including on the
x-only path where `YrLtP` also binds; A4g shows the mode-dependent bound is what keeps legal
x-only addresses provable.

---

## Contracts (assume-guarantee boundary)

Imported verbatim: **C1** AreBytes, **C2** IsHalfword, **C3** IS_BIT/booleans, **C4** MEMW byte
authority, **C5** LogUp multiset soundness, **C6** Ecall binding, **C7** timestamp uniqueness,
**A-PRIME** (`p`, `N` prime). New: **A1-PRIME** (`p_g` prime, sympy-certified) and:

- **C4-YR** — `YR`'s bytes are in `[0, 256)`. `ecsm.rs` does **not** emit this. The bound is
  inherited by two exhaustive cases on `len_k`: `len_k ≥ 1` ⇒ the `Ecdas` drain tuple equals an
  ECDAS sender's byte-checked `yR`; `len_k = 0` (`k = 1`) ⇒ balance forces drain = seed, i.e.
  `YR = YG`, which *is* byte-checked here. Both are bus-level (C5 + imported L6) and therefore
  invisible to this gate — which is why C4-YR is written down instead of quietly used.

---

## Findings

1. **[confirmed] The `y = 1` point.** The PR's constructibility claim holds and is stronger than
   stated — the smallest possible `y` is attained. Recorded because "astronomically rare" was
   the phrasing used for the `XR_SUB_P` analogue on the earlier board, and it is not rare here.
2. **[confirmed] The seven-value address band.** `ecsm.rs`'s comment claims the LT senders close
   "a seven-value band per operand". Measured: exactly 7, both modes (A4c). No action.
3. **[observation] `YrLtP` is µ-gated, not `IS_AFFINE`-gated**, so it binds x-only rows where
   nothing observes `yR`. Strictly stronger ⇒ sound; completeness holds because
   `compute_witness_inner` fills `y_r_sub_p` on both paths. Gating on `IS_AFFINE` would save 16
   halfword sends + 8 constraints; not worth the asymmetry with the other three chains.
4. **[narrower than it looks] `IS_AFFINE` is separated by ONE 32-bit word.** The two syscall
   numbers share their high word, so its `IS_AFFINE` coefficient is zero and carries no mode
   information (A1c-assert). The whole pinning is the low word, kept that way by
   `execution.rs`'s `const _: () = assert!`. Audit premise P5 fails if the assert is removed.
5. **[audit method] A blind check is worse than a missing one.** P18 matched `ecsm.rs`'s
   *comment* about the timestamp stride against a hard-coded 4 and passed — the mutation control
   showed it survived reducing the real stride to 3, i.e. it was checking documentation. It now
   parses the stride from `trace_builder.rs`. Every premise is mutation-tested for this reason.
6. **[control hygiene] The wrong-constant control was initially vacuous.** A2c's first form
   recomputed the witness addend for the perturbed constant, so *any* constant passed. Fixed by
   holding the witness fixed. With Finding 5: a green control is worth nothing until you have
   seen it go red.
7. **[gap, now closed] `IS_BIT(IS_AFFINE)` is load-bearing for an unrecorded reason.** The first
   version of this board proved idx 421 *correct* (A1a) and never asked whether it was
   *necessary*. It is, and the argument is not local to the ECSM pair. The receiver's syscall
   words are `xonly + a·(affine − xonly)` per 32-bit word, with coefficients `−1` (low) and
   **`0`** (high, both sharing `0xFFFF_FFFF`) — so as `a` sweeps the field the high word is
   constant and the low word sweeps everything. Every syscall is `u64::MAX − k` and shares that
   high word, so each is reachable at `a = (target_lo − xonly_lo)/(−1)`: `ECSM` at 0,
   `ECSM_AFFINE` at 1, **`HINT` at 20**, **`KECCAK` at `p_g − 9`**. Drop idx 421 and `a` is free
   on a `µ=1` row (idx 422 binds only at `µ=0`), so an ECSM row can consume a *different*
   accelerator's `Ecall` send — a HINT call proven as a scalar multiplication writing 32 or 64
   bytes wherever the row's register columns point. `execution.rs`'s assert cannot see this: it
   compares only the two ECSM low words, not a third syscall an integer offset away. A
   documentation gap in #879, not a bug — idx 421 is present and does the job.
8. **[gap, now closed] `yG` canonicality was never checked.** A3c proved the read *pins* `yG`
   and stopped there; nobody asked whether the bytes are canonical. They need not be —
   `OverflowKind` is `{XgLtP, KLtN, XrLtP, YrLtP}`, with no `YgLtP`, while the executor rejects
   `yG ≥ p`. #879's treatment is asymmetric: it closed the address band (`1a994313`) and left
   the `yG` band open, though `xG`'s sibling check has existed all along. Reachable with this
   board's own artifact (`yG = p + 1`); consequence benign, since every relation is a congruence
   mod `p`, so the computed point and output are unchanged. What is lost is VM-parity — a proof
   can attest to an ecall the executor would have halted on, exactly the class A4c measures.
   **Medium**; the soundness theorem is corrected accordingly, and audit premise **P20** parses
   `OverflowKind` so this cannot go stale silently.

### The rule findings 5–8 generalise to

> **Every negative control must be PAIRED with the specific check the dropped premise is
> load-bearing for — and the pairing must be evaluated in both directions.**

A control that drops a premise and re-runs a check whose reference never mentioned that premise
*cannot fail*, and reads green. A parallel campaign on the DMA memcpy PR (#874) hit the identical
shape three times, which is enough recurrence to write down the rule rather than the anecdotes.

| Control | half 1: premise KEPT must block | half 2: premise DROPPED must admit |
|---|---|---|
| A1c-ctl | the repo pair is injective | the degenerate pair collides |
| A1e | idx 422 is the only violated constraint of 423 | all 422 remaining satisfied, 8 MEMW ops fire |
| A1f | only ECSM/ECSM_AFFINE reachable | HINT and KECCAK reachable |
| A2c-ctl | honest witness (addend fixed) valid under `p` | rejected under `p+2` and under `N` |
| A2g | A2f established every *other* constraint holds | the witness is accepted, guest gets `yR + p` |
| A3d | A3c: the read's tuples differ between the two | `check_all_constraints` clean on both |
| A3g | no constraint bounds `yG < p` | the consequence is benign (same reduced point) |
| A4e-ctl | the `u128` form rejects the overlap | `addr_limb_ok` passes *and* the `u64` form accepts |

**It paid for itself twice.** A1f's first implementation had an inverted sign in its modular
solve and found no foreign syscalls; half 2 alone would have reported "idx 421 REDUNDANT" — a
confident, wrong conclusion. Half 1 failed instead and surfaced the bug. A1e originally listed
"the constraint families I believe remain" rather than walking the index map — the mirror-image
hazard, where an over-permissive control reports a false FORGES by forgetting a constraint that
would have blocked the state; it now enumerates all 423 and asserts the count.

---

## Method notes

- **`x·(1−x) ≡ 0 (mod p_g)` is not a z3 query.** In lifted integer form (`x(1−x) = m·p_g`, `m`
  free) it does not terminate at this modulus, and 160-bit `URem` bit-blasting does not either.
  These are root-of-a-polynomial-over-a-field statements: sympy factors over `GF(q)` and the
  split is checked **complete**, so the root set is provably exhaustive. z3 keeps the
  bounded/linear work — the carry lifts (A2a, A2b), the predicate sweeps (A4a, A4d), the
  quantified interval statement (A4e).
- **Two different fields.** The AIR is over `GF(p_g)`; the curve over `GF(p)`. `field_roots`
  takes the modulus explicitly because an early A3a factored the curve polynomial over
  Goldilocks and correctly reported FAIL.
- **Never negate an equation carrying a witness quotient.** The lifts put the free quotient on
  the *hypothesis* side and deny a quotient-free conclusion (A2a asserts `A − 2^32·c == m·PG`,
  denies `A != 2^32·c`). That direction is sound — extra freedom in `m` only strengthens the
  adversary. The inverse yields a **bogus SAT**: `Not(a − b == k·m)` is satisfiable by picking
  `k != 0`. Audited: every query here asserts its quotient equations positively, and no denial
  contains a quotient variable.
- **The tractable rewrite.** Keeping `Int` and rewriting `a ≡ b (mod m)` as `a − b == k·m` is
  linear when `m` is constant, which is why A2a/A2b are milliseconds. Rewriting `x·(1−x) == 0`
  as `Or(x == 0, x == 1)` is the same trick one level up, used throughout the downstream lemmas
  — but **not** available for A1a, where that disjunction *is* the conclusion.
- **Forgeries are constructive** — fixed numerals evaluated against the transcribed
  constraints; the solver is never asked whether an attack *might* exist.

---

## Reproduction

```bash
python3 -m venv .venv && ./.venv/bin/pip install z3-solver sympy ecdsa
./run_gate.sh            # everything, transcript to gate.log
./run_gate.sh --quick    # reuse the witness dump, skip the cargo build
```

`run_gate.sh` cds to its own directory, so it also works from the repo root. Individual stages
run in dependency order: `audit_transcription.py` → `test_oracle.py` → `small_y_point.py` →
`cargo run --release -- > real_witnesses.jsonl` → `a6_real_witness.py` → `a1`–`a4`. A few
seconds plus one `cargo build`; the harness depends only on `crypto/ecsm`, deliberately. The two
scripts that read repo source locate the root by marker (a workspace `Cargo.toml` next to
`prover/`), not a hard-coded `parents[N]`.

---

## Where to send the next reviewer

- **Bus-level reasoning is outside this gate**, as it was outside both earlier ones. C4-YR, the
  `Ecall` pinning's LogUp step (A1c reduces to it), and A3c's "at most one witness matches the
  caller's buffer" all bottom out in C5 + imported L6. The e2e `prove + verify` tests in
  `prover/src/tests/ecsm_tests.rs` and `prove_elfs_tests.rs` cover that; green on this branch.
- **The `q1` growth on the `−yG` branch** is checked to fit 33 bytes at the instances tested,
  not proved for all inputs. A completeness question about a witness nobody should build — but
  if the affine path ever *needs* both roots, it becomes a real bound to establish.
- **Nothing here re-proves the double-and-add chain.** If #879's rebase touches the ECDAS chain
  or the `Ecdas`/`Bit` buses, the imported board is the one to re-run, not this one.

## Where the spec lives — and a correction

The ECSM spec **exists**: `spec/ecsm.typ`, `spec/src/ecsm.toml`, `spec/src/ecdas.toml`, on the
long-lived **`spec/main`** branch. `prover/src/tables/ecsm.rs:19`'s "See `spec/src/ecsm.toml`"
is a correct reference. PR **#932** specs the affine variant reviewed here, and its constraints
correspond to this board's subjects one-for-one; two divergences are deliberate — the spec
derives `addr_yG`/`addr_yR` with a full 64-bit `ADD` into dedicated columns (so it needs no
limb bound, where the implementation uses the `Alu` LT senders A4 verifies), and it spends 32
columns the implementation never materialises.

> **Corrected 2026-08-12.** An earlier version of this section claimed there was no ECSM chapter
> "under any name" and that `spec/src/ecsm.toml` "does not exist and never has". Both were
> wrong — the mistake was checking `spec/` on `main`, a stale snapshot; the spec is maintained
> on `spec/main`, which is *not* an ancestor of `main` (102 commits behind) and reaches it only
> via batch sync PRs. Recorded rather than quietly deleted, because the same error had already
> been made and corrected once before on this codebase: **any claim about the spec must be
> checked against the newest `spec/*` branch, never `main`.**
