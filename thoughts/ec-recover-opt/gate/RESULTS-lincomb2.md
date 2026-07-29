# lincomb2 z3 gate (phase E) — lemma board & soundness theorem

Verification of the JOINT-CHAIN chips `prover/src/tables/ecsm2.rs` (1,155 cols)
and `ecdas2.rs` (658 cols, 288 constraints). Companion board to
[`RESULTS.md`](RESULTS.md), which covers the original single-scalar chips.

Model transcribed by READING the chips; independent reference =
`../oracle/lincomb2_ref.py` + `jacobian_ref.py`, themselves anchored on four
lineages (see `../oracle/README.md`). The transcription itself was then audited
in the model ⊆ chip direction — the one the positive anchor structurally cannot
see — in [`TRANSCRIPTION-AUDIT.md`](TRANSCRIPTION-AUDIT.md).

**Faithfulness anchor**: before any UNSAT was trusted, the transcribed model was
evaluated on REAL prover witnesses (`ecsm::lincomb2_witness` + `dinv_witness`
via the oracle harness): 265 cases → **5,960 ECDAS2 rows, 3,317,087 checks**,
every one of the 288 constraint values ≡ 0 mod p_g, every carry inside its
IsHalfword window, every quotient inside 33 bytes, every schedule relation
satisfied, and the prover's own `D_INV`/`Q3`/`C3` columns equal to an
independent group-law derivation. (`positive_real_witness2.py`.)

**Layout pins** (settled, verified in the tree): `cols::D_INV = 529`,
`NUM_COLUMNS = 658`, `debug_assert_eq!(idx, 288)` = 28 + 4×65, four-entry
relation loop ending `(Relation::Dinv, cols::C3)`, and the `NB..MU = 519..528`
block preserved by appending rather than inserting the new columns.

> **This gate found two real forgeries.** Both were open in the chips when phase
> E began and are now closed. The gate is written so that it keeps telling the
> truth as the chips change: `gate2_common.chip_state()` **parses** the chip
> source at run time — the emitted constraint expressions and the bus
> multiplicity expressions, never a comment — and a control whose defence is
> missing is reported as **LIVE HOLE** rather than scored as a passing ablation.
> Every parser fails closed: an unrecognised shape reports the defence ABSENT.
>
> That sentence was **false when first written** (`chip_state` matched comments
> and token presence — `TRANSCRIPTION-AUDIT.md` F2), which is why
> `audit_transcription.py` now tampers each defence and requires the detector to
> fire. A detector nobody has watched fail is not a detector.

## Board

| Lemma | Statement | Verdict | Notes |
|---|---|---|---|
| PORT | ECDAS2's `Lambda`/`Xr`/`Yr` relation arms, the `s_i` **operand bindings**, `conv_carry` and the shared helpers are ECDAS's, modulo the documented `XG/YG → XB/YB` rename | PROVED (mechanical text comparison) | so every lemma quantified over those three relations transfers verbatim. The prologue and `conv_carry` were added after `TRANSCRIPTION-AUDIT.md` F3: without them a chip whose relations read the **wrong columns** compared identical |
| PORT-M | ECSM2's `X2`/`Yg` membership relations are ECSM's, modulo the column rename and the `µ → OK` gate swap | PROVED (mechanical) | new. The soundness theorem's "`P2` is on the curve" clause rests on this port and nothing checked it. `OK` is IS_BIT with `OK·(1−MU) = 0`, so `OK = 1` rows are a subset of `MU = 1` rows and every ECSM lemma applies verbatim |
| L1 | carry recurrence + `c₆₃ = 0` ⇒ `Σ256^i·S_i = 0` over ℤ | PORTED | unchanged relations ⇒ unchanged proof |
| L2a | per-limb `\|256c − c⁻ − S\|` < p_g with a **varying, non-canonical** addend | PROVED (exact intervals + z3 ×72 + 36k corners) | worst case 2^24.4 vs 2^64 — 2^39 headroom. `../lincomb2/WIDTH-AUDIT.md` |
| L2b | honest carries fit the unchanged `CARRY_OFFSET_*` windows | PROVED (6,346 real rows, differentially checked vs the prover) | λ `[-4303, 6728]`, xR `[-112, 8308]`, yR `[-465, 5914]`, D_INV `[-581, 6041]` |
| L2c | word-carry lift + strict-inequality chains | PORTED | same overflow-witness machinery |
| L3a/L3b | convolutions capture the intended polynomial; chord/tangent stays on curve | PORTED | |
| L4a/b/c | λ, xR, yR pinned mod p given the side condition | PORTED | the side condition is now discharged by L5b′ |
| L5a | no `y ≡ 0` point on secp256k1 | PORTED | discharges the doubling side condition |
| **L5b′** | **REPLACES L5b.** `D_INV·(xB − xA) ≡ 1 (mod p)` is imposed on exactly the addend-consuming rows and is unsatisfiable exactly when `xB ≡ xA (mod p)` | PROVED (z3, 5 sub-lemmas) | **unconditional** — no dlog assumption. §2 below |
| **L6** | joint-schedule counting: the chain is exactly the honest schedule | PROVED **given the idx 22..=27 gate**, which this gate's break forced | §3 below; `../lincomb2/L6-COUNTING.md` |
| L6.5 | exact-MSB pinning | **DROPPED** | blinding makes any `len ∈ [max_msb+1, 256]` yield the same Q; `len ≤ 256` is structural via the EC_T0 table |
| L7 | drained Q equals fully-enumerated ground truth | PROVED (265 comparisons, 0 mismatches) | 256 small-joint-scalar cases with ground truth by repeated group addition |
| L8 | negative / sensitivity controls | PASSED (**7 forgeries**, 2 redundancies, 0 live holes) | §4. One control (N4b) was mis-aimed and has been re-pointed — read §4 before quoting the count |
| AUDIT | the model asserts nothing the chips do not enforce | PROVED for the schedule block (2^11 brute force vs an independent second transcription, symmetric difference empty **both ways**); 4 unchecked premises found and closed | `TRANSCRIPTION-AUDIT.md`, regression suite `audit_transcription.py` (22/22) |

## 1. What replaced the NUMS assumption

DESIGN §4 proposed closing the incomplete-addition edge with NUMS accumulator
blinding, converting one lemma from unconditional to a named dlog assumption.
**That reduction does not hold** — the prover chooses `P2`, so setting
`P2 = μ·T₀` cancels `T₀`'s coefficient out of the collision equation and no
discrete log is needed (`../lincomb2/FINDING-nums-blinding.log`, 5/5
constructions, corroborated by the Python reference, the Rust witness and an
independent Jacobian path).

The chip instead carries `D_INV·(xB − xA) ≡ 1 (mod p)` as a fourth convolution
relation. **The gate is therefore unconditional again: lincomb2 rests on no
cryptographic assumption that the original chips did not.** The blind is kept
only for the `len` simplification (L6.5), which is a convenience, not a soundness
property.

## 2. L5b′ — the non-degeneracy relation

`l1_l5_port2.py` §3, all PROVED:

| | statement | verdict |
|---|---|---|
| a1 | `OP = ΣS` on every live row, so the chip's `ΣS` gate coincides with `OP` | z3 UNSAT |
| a2 | all five addend-consuming row types (AddP1/P2/P12, Precompute, **Correction**) always consume an addend; doublings never do | z3 UNSAT ×5 + SAT |
| b1 | unsatisfiable whenever `xB ≡ xA (mod p)` — including byte-different encodings that agree mod p | z3 UNSAT (`p·(d·k − m) = 1`) |
| b2 | satisfiable whenever `xB ≢ xA (mod p)` — costs no completeness | 204 residues incl. 1, 2, p−1, p−2 |
| c | the **gated-off branch pins `q3 = µ·3p`** — a doubling is not a hole | z3 UNSAT |
| d | discharges L4a's side condition unconditionally | — |

Sub-lemma (c) is worth its own line because "the check is gated off here" is
exactly the shape a real hole takes. With `g = 0` only `rq` survives, and L1's
telescoping turns that into `p·(µ·R − q3) = 0` — so `q3` is **pinned** to `3p` on
a live doubling and `0` on a padding row, not left free. Confirmed on real
witnesses as well: every doubling row's `Q3` column is exactly `3p` with zero
carries.

The chip gates by `ΣS` rather than `OP`. That is strictly better than the `OP`
gate the audit proposed: it ties the check to the very expression that counts the
Addend receive, so it cannot drift away from the rows that consume one. a1 shows
the two coincide.

**"Cannot drift" is now enforced, not asserted.** `TRANSCRIPTION-AUDIT.md` F1
found that (a1)/(a2) quantify over a *Python* expression and nothing read the
chip's actual gate: deleting the `S_CORR` term left the correction row with no
non-degeneracy check — a working forgery, reachable by one point subtraction
(`u1 = u2 = 1`, `P2 = −2^(len+1)·T₀ − G` makes `xA = xB` **and** `yA = yB` there,
so λ is free) — while every verdict on this board stayed green.
`gate2_common.dinv_gate_state()` now parses the `Relation::Dinv` arm, requires
the gate to be literally a sum of plain columns, and requires that column set to
**equal** the `Addend` receive's `Multiplicity::Linear` terms (each with
coefficient 1). Tamper and untampered results in `audit_transcription.py` §E.

## 3. L6 — the joint-schedule counting argument

Full derivation in `../lincomb2/L6-COUNTING.md`. Established, in order:

1. **Phase separation** — `PHASE = PH1 + 2·PH2` with `PH1·PH2 = 0`; carried
   unchanged along every row, so it is a path invariant pinned by the seeds.
2. **One segment per phase** — all six ECSM2 chain interactions carry
   multiplicity `OK`, which is `IS_BIT`.
3. **No cycles; paths end only at drains** — `round` never increases, a `Δ = 0`
   step forces an add next which decrements, and the `−1` drain round is not a
   byte so no row can receive it.
4. **Phases 0 and 2 are exactly one row each**, with their selectors pinned by
   idx 20 / idx 21.
5. **The main chain is one doubling per round plus an add iff a digit is set**
   (idx 13).
6. **Digit consumption is exact** — balance gives
   `#{rows at round i with D = 1} = 2·u_bit(i)`, each row contributes 0 or 1, so
   both the doubling and its add must exist and agree.
7. **The addend matches the digits** — idx 14/17/18/19 force
   `(1,0) ⇒ S1`, `(0,1) ⇒ S2`, `(1,1) ⇒ S3`, and make `(0,0)` with `OP = 1`
   unsatisfiable.

`len ≥ max_msb + 1` follows from balance (a set bit above `LEN_M1` has a 2×
receive with no possible sender); `len ≤ 256` is structural.

**The 2× JointBit multiplicity is load-bearing**, with a concrete 1×
counterexample: at a round where both digits are set, a 1× prover splits them
across the two rows and the add selects **P2 where the schedule calls for P12**.
z3 SAT at 1×, UNSAT at 2×.

**Bus-28 separator — holds, but not for the reason stated in the source.**
`alpha_offset` advances by `num_bus_elements()` unconditionally
(`lookup.rs:1651-1663`), so zeros never shift positions and there is no
re-alignment risk; the `if result != zero` skip at `:679` is a multiply-avoidance
that still consumes its α slot. What separates the two chains is that tuple
position 1 is constant `0` (old) vs `1` (new), so the difference of the two
fingerprint polynomials has a non-zero α¹ coefficient. Trailing-zero aliasing of
a shorter tuple is real in general — that is what the `sel ≠ 0` note on the
Addend bus guards — but is not the mechanism here. No need for a fresh bus id;
the comment is worth rewording.

## 4. L8 battery

`l8_negative2.py`. Every control is first shown BLOCKED on the untampered chip,
then re-run ablated. Forgeries are constructive (fixed numerals; z3 only checks).

| Control | Verdict | Meaning |
|---|---|---|
| **N1** drop `(1−MU)·D1/D2` | **SAT — FORGES** | padding rows send live JointBit digits ⇒ a set scalar bit is consumed with **no add on the chain**. **Was a live hole**; closed by idx 22..=27 |
| **N2** drop `D_INV` | **SAT — FORGES** | degenerate add ⇒ λ unconstrained. **Was a live hole**; closed by the fourth relation |
| **N3** drop `xQ`/`yQ < p` | SAT — FORGES | a `+p`-shifted output coordinate is the same field element but a different byte string ⇒ the guest keccaks a different address |
| **N4** drop idx 18/19 | SAT — FORGES | an add consumes an addend its digits do not select (P2 with digits `(1,0)`) |
| **N4b** drop `OP = ΣS`, **live** `PH1 = 0` rows (precompute, correction) | SAT — FORGES | an `OP = 0` doubling that still consumes its addend, where idx 17/18/19 are vacuous. **Re-pointed** — see below |
| **N4c** drop `OP = ΣS`, live main-chain doubling | UNSAT — REDUNDANT | idx 17/18/19 already force every selector to 0 when `PH1 = 1, OP = 0`. Keep idx 14 — N4b is what it covers |
| **N5** drop `MU·(STATUS·S_INV − (1−OK))` | SAT — FORGES | `status = 0` with `OK = 0` ⇒ the guest consumes an unproven result |
| **N6** drop the `EcT0` lookup | SAT — FORGES | the correction addend becomes free ⇒ the chain lands on any chosen Q (`W = target − acc`) |
| **N7** drop `yP2 < p` | **UNSAT — REDUNDANT (expected)** | see below |

**N7 is the expected result, not a gate bug** (IMPL-PLAN §11 risk 7). Three
independent reasons, all recorded rather than papered over:

1. the bytes are MEMW-bound to what the guest wrote (contract C4), and the guest
   derives `y` by field arithmetic, so `y < p` already holds;
2. the only other 32-byte encoding congruent mod p is `y + p`, which needs
   `y < 2^256 − p = 2^32 + 977` — and denotes the **same point**;
3. a `< p` test cannot separate a point from its negation anyway: both `y` and
   `p − y` are below `p`. Parity is the guest's authority, backed by MEMW.

Keep the column as defence in depth — it lets the chip's argument stand on its
own constraints rather than on guest-code correctness — but it closes no forgery,
and DESIGN §3's parity-flip justification for it is wrong.

### N4b was mis-aimed, and the count moved twice

`TRANSCRIPTION-AUDIT.md` F4. N4b used to aim at a `MU = 0` padding row **and
build it with `padding_gate=False`**, i.e. against a chip without idx 22..=27.
Its `SAT — FORGES` therefore came from the model, not the chip: on the real
constraint set idx 24..=27 zero every selector on a padding row regardless of
idx 14, and the query is UNSAT. So the board's non-vacuity was really **6**
genuine forgeries plus one artefact, not 7.

Re-pointed at the two **live** `PH1 = 0` phases — where idx 17/18/19 are vacuous
and idx 20/21 still force `S2`/`S_CORR = 1` — it is a real forgery again, and
exhibits its witness rows:

```
precompute (MU=1, PH1=PH2=0) untampered unsat  ablated sat
   forged row: MU=1 OP=0 NB=0 D1=0 D2=0 S1=0 S2=1 S3=0 SC=0 PH1=0 PH2=0
correction  (MU=1, PH2=1)    untampered unsat  ablated sat
   forged row: MU=1 OP=0 NB=0 D1=0 D2=0 S1=0 S2=0 S3=0 SC=1 PH1=0 PH2=1
```

**So the count is back to 7 — for a different reason than this board previously
claimed.** *Keep idx 14* survives unchanged; only its justification was wrong.
The root cause was `Ecdas2Row(padding_gate=…)` defaulting to `False`, a model
strictly weaker than the chip. It now defaults to `True` and ablations opt out
explicitly.

**Non-vacuity: 7 genuine forgeries.** Two of them (N1, N2) were live holes in the
chip, which is the strongest possible evidence that this gate can see real bugs.

## 5. Soundness theorem

Under contracts C1–C7 + A-PRIME (unchanged from `RESULTS.md`), any accepted
trace satisfies: for every ECSM2 row with `OK = 1` at ecall timestamp `ts`,
inputs `P1 = G` (constant), `P2`, `u1`, `u2` bound by the MEMW reads, and output
`xQ‖yQ` bound by the MEMW writes:

> `P2` is on the curve; `0 < u1, u2 < N`; and `(xQ, yQ)` is the canonical affine
> representation of `Q = u1·G + u2·P2`, or the row reports `status ≠ 0` and
> proves nothing (whereupon the guest's software fallback runs, which is sound
> because guest code is proven execution).

Chain of proof: field constraints →(L2a/L2c) integer identities →(L1, L3a) value
relations →(L4, sides by L5a/**L5b′**) per-step group law mod p, on-curve
preserved (L3b) →(L6) the whole chain is the honest joint schedule with each
digit consumed exactly once and the selected addend →(L7) the drain equals
fully-enumerated ground truth on concrete instances.

**Status: green, and unconditional.** No lemma is OPEN. The board rests on the
same eight contracts as the original chips plus nothing else — in particular the
NUMS dlog assumption proposed by DESIGN §4 is **not** required and is **not**
sufficient; it should not be signed off.

## 6. Reproduction

```sh
cd thoughts/ec-recover-opt/gate
<venv>/bin/python positive_real_witness2.py   # anchor — run FIRST (3.3M checks)
<venv>/bin/python l1_l5_port2.py              # port arguments, L2b, L5b′
<venv>/bin/python l6_joint_counting.py        # L6 + the padding-row break
<venv>/bin/python l8_negative2.py             # the control battery
<venv>/bin/python audit_transcription.py      # model ⊆ chip + detector regressions
```

Current verdicts: `l1_l5_port2` 10/10 PROVED, `l6_joint_counting` all PASS /
CONFIRMED / DEMONSTRATED, `l8_negative2` 7 forgeries + 2 redundancies + **0 live
holes** + 0 broken controls, `audit_transcription` 22/22.

venv: `python3 -m venv venv && ./venv/bin/pip install ecdsa z3-solver sympy`.
The harness must be built: `(cd ../oracle/repo-harness && cargo build --release)`.
Logs in `logs/`.

## 7. Caveats, honestly

- **Column indices are settled** (pins listed at the top). They moved once —
  `D_INV` landed *while this gate was being written*, taking ECDAS2 from 529 →
  658 columns and 217 → 288 constraints mid-run — which `chip_state()` caught.
  The gate is written against constraint identities and bus multiplicities and
  re-reads the source at run time, so it survives further movement; indices
  appear only in prose.
- **The positive anchor evaluates CONSTRAINT VALUES only — never a multiplicity
  and never a gating expression.** Every honest row has `MU = 1`, so idx 22..=27
  and idx 20's `MU` factor are vacuous in all 3.3M checks *by construction*. The
  volume is real, but it is volume in one dimension: the ECDAS2 JointBit bug (an
  ungated `Multiplicity::Column(D1)`) would have passed the anchor unchanged, and
  so would a narrowed `D_INV` gate. **No amount of honest-witness evaluation can
  catch a gating error, and none of the 3.3M checks is evidence about one.**
  Those two classes are covered by `gate2_common.chip_state()`'s parsed
  invariants and by the L8 controls instead — see `TRANSCRIPTION-AUDIT.md`
  F1/F2, and `audit_transcription.py` §D/§E, which tamper each defence and
  require the detector to fire.

- ~~**`PH*`/`S*` are a modelled step**~~ — **closed.** They are still not fields
  of `Lincomb2Witness`, and `positive_real_witness2.py` still reproduces the
  chip's `JointSel → (PH1, PH2)` / `(S1, S2, S3, S_CORR)` derivation by hand —
  but `check_sel_maps()` now compares that hand copy against
  `Ecdas2Operation::phase_bits` / `selector_bits` parsed from the chip, arm for
  arm, plus the `JointSel` variant list from `crypto/ecsm/src/witness.rs`. A
  changed *or added* arm fails the anchor instead of silently re-deriving a
  different chip (`audit_transcription.py` §F).
- **L2b is a measurement plus a structural invariance argument**, not a
  closed-form worst-case bound. Completeness-only: a miss costs an unprovable
  honest witness, never a forgery. See `WIDTH-AUDIT.md` §4.
- **The standing residual risk is hand transcription**, mitigated (not
  eliminated) by the anchor *and*, since `TRANSCRIPTION-AUDIT.md`, by an explicit
  model ⊆ chip audit: the schedule block was brute-forced over all 2^11 boolean
  assignments against an independently written second transcription, with the
  symmetric difference empty in **both** directions. That covers idx 0..=27. It
  does **not** cover the convolution blocks, which rest on the (now broader)
  PORT comparison plus the anchor. The durable fix is still the one the keccak
  note calls for: generate the SMT from the constraint IR
  (`air.constraint_program()`).

- **The gate's own detectors are now regression-tested.** `chip_state()` used to
  match comments and token presence; deleting a defence while keeping its
  comment reported the defence present. Everything is parsed from the emitted
  expression, fails closed on an unrecognised shape, and
  `audit_transcription.py` re-runs each tamper and requires detection (22/22).
  Two structural invariants are enforced in both directions: gated columns ==
  raw bus multiplicities, and the `D_INV` gate == the `Addend` receive terms.

- **Still unverified from inside this gate**, unchanged: contracts C1–C7 and
  A-PRIME (only the *sending* side of each bus is read here — not `bitwise.rs`'s
  receivers, MEMW's byte authority, the CPU↔Ecall binding, timestamp uniqueness,
  or the generic LogUp balance argument); and the completeness-side mirroring of
  `collect_bitwise_from_ecdas2` / `collect_bitwise_from_ecsm2`, whose failure
  mode is an unprovable honest witness rather than a forgery. Note also that
  C4 as written in `RESULTS.md` names `xG`/`k`/`xR`; its ECSM2 instance is
  `X_P2`/`Y_P2` byte-ness via the eight MEMW dword reads at `ecsm2.rs:631-649`,
  which is a different (and unstated) instance of the same contract.
