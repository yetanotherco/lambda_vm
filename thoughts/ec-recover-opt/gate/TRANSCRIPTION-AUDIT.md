# Transcription audit — does the lincomb2 gate assert more than the chips enforce?

Adversarial, one-directional audit of `thoughts/ec-recover-opt/gate/` against
`prover/src/tables/ecdas2.rs` (658 cols, 288 constraints) and `ecsm2.rs`
(1,155 cols, 693 constraints), branch `feat/ec-lincomb2`, PR #871.

Only one direction is dangerous. A model **weaker** than the chip yields a
spurious SAT — a false alarm. A model **stronger** than the chip yields UNSAT on
a chip that is genuinely forgeable — false assurance. The 3.3M-check positive
anchor cannot see the second kind: an honest witness satisfies a correct model
and an over-strong model equally well. That blind spot is what this audit
covers.

Reproduce: `<venv>/bin/python audit_transcription.py` (z3 only; §E also needs
`../oracle`). No `.rs` file was modified — the tampers in §C/§D/§E are done on
in-memory copies.

---

> ## STATUS: all five findings FIXED and regression-tested.
>
> `audit_transcription.py` is now the regression suite for the fixes: it re-runs
> every tamper below and requires the detector to fire — **22/22 pass**. The
> chips were not touched (they were correct; the gate was the problem).
>
> | # | fix | where |
> |---|---|---|
> | F1 | `dinv_gate_state()` parses the `Relation::Dinv` arm, requires the gate to be a plain sum of columns, and requires that set to equal the `Addend` receive's unit-coefficient `Multiplicity::Linear` terms | `gate2_common.py` |
> | F2 | `padding_gate_state()` parses the emitted `(1 − MU)·X` expression and enforces **gated columns == raw bus multiplicities**, in both directions. Every parser fails closed | `gate2_common.py` |
> | F3 | `relation_bodies_identical()` extended to the `s_i` prologue and `conv_carry`; new `membership_bodies_identical()` for the ECSM/ECSM2 pair | `gate2_common.py`, reported by `l1_l5_port2.py` |
> | F4 | `Ecdas2Row.padding_gate` defaults to `True`; N4b re-pointed at live `PH1 = 0` rows and now prints its witness rows | `gate2_common.py`, `l8_negative2.py` |
> | — | `JointSel → PH*/S*` compared arm for arm against the chip (§7's last modelled step) | `joint_sel_maps()`, `check_sel_maps()` |
> | — | doc hygiene: `L6-COUNTING.md` status header, stale cites, position-0/1 wording | `L6-COUNTING.md`, `RESULTS-lincomb2.md`, the scripts' docstrings |
>
> **One fix failed its own regression test on the first attempt**, and is
> recorded rather than quietly patched: §G's tamper
> `let g = gate_expr(b) * b.main(0, cols::S1) + …` left the `cols::` **set**
> unchanged, so the first version of the F1 check accepted it while an opaque
> factor could gate the relation off at will. `_is_plain_column_sum()` closed it
> — set equality was not enough; the expression must *be* the sum.
>
> Board impact: **the L8 non-vacuity count was 6, not 7** (F4). After
> re-pointing N4b it is 7 again, for a different and now-correct reason. No
> lemma changed verdict. `l1_l5_port2` 10/10, `l6_joint_counting` all pass,
> `l8_negative2` 7 forgeries / 2 redundancies / **0 live holes**.

## Verdict

**No live over-strong assertion was found.** The boolean core of the model is
*exactly* equivalent to the chip: brute-forcing all 2^11 assignments of
`{MU, OP, NB, D1, D2, S1, S2, S3, S_CORR, PH1, PH2}` against an independent
second transcription of `Ecdas2Constraints::eval` idx 0..=27 gives **11 admitted
assignments on each side, symmetric difference empty in both directions**
(`audit_transcription.py` §A). Every multiplicity, gating factor, bus tuple and
enum mapping listed in the table below matches the source.

But three of the gate's assertions rest on premises the gate **does not check
and would not notice becoming false**, and one negative control reports a
forgery that the chip actually blocks:

| # | severity | what |
|---|---|---|
| **F1** | **high** | The D_INV gate `g = S1+S2+S3+S_CORR` is asserted by every L5b′ lemma and read by nothing. A one-term deletion breaks it, produces a working forgery, and leaves every gate verdict green. |
| **F2** | **high** | `chip_state()` — the mechanism the board credits with "keeping the gate honest as the chips change" — detects `idx 22..=27` from a **comment**, and `D_INV` from token presence. Delete the defences, keep the comments: still reported present. |
| **F3** | medium | The PORT lemma's mechanical check compares only the `match` arms of `s_i`. Rebinding an operand column in the prologue (e.g. `xa → cols::XR`) falsifies L3a/L4a/b/c and is still reported "identical". |
| **F4** | low (wrong direction) | L8 control **N4b** claims `SAT — FORGES`; on the chip's real constraint set the same ablation is **UNSAT**. The SAT comes from a model that omits idx 22..=27. Non-vacuity is 6 genuine forgeries, not 7. |
| F5 | cosmetic | One literal over-strong assertion — `ROUND ∈ [0,255]` on `MU = 0` rows — verified benign. |

None of F1–F3 is a chip bug. All three are **unverified premises presented as
verified results**, which is the failure mode the ECDAS2 JointBit incident
already demonstrated once.

---

## F1 — the D_INV gating expression is asserted everywhere and checked nowhere

`RESULTS-lincomb2.md` §2 states the chip "gates by `ΣS` rather than `OP`… it ties
the check to the very expression that counts the Addend receive, so it cannot
drift away from the rows that consume one", and L5b′(a1)/(a2) are the lemmas
that discharge L4a's side condition — the whole unconditionality claim.

The chip's gate is at **`ecdas2.rs:810-813`**:

```rust
let g = b.main(0, cols::S1)
    + b.main(0, cols::S2)
    + b.main(0, cols::S3)
    + b.main(0, cols::S_CORR);
```

`l1_l5_port2.py` §3 (a1) and (a2) quantify over `Ecdas2Row.addend_receive()` —
`self.s1 + self.s2 + self.s3 + self.sc`, a Python expression
(`gate2_common.py:229-231`). **Nothing connects the two.** A grep over
`gate/*.py` for any reference to `Relation::Dinv`, `cols::S_CORR` or the `let g`
expression returns nothing; `relation_bodies_identical()` deliberately excludes
the Dinv arm; `chip_state()["dinv_relation"]` is `"D_INV" in src and "Dinv" in
src`.

### The forgery a narrowed gate would hide

Drop one term — `g = S1 + S2 + S3` — and the **correction row** (which carries
`S_CORR = 1` and every other selector 0, by idx 21 + idx 14) has `g = 0`: no
non-degeneracy check at all. Its λ relation then degenerates whenever the
accumulator entering it equals its addend `W = −2^len·T₀`.

That is cheap to arrange and needs **no discrete log**:

```
acc = 2^len·T₀ + u1·G + u2·P2        (the chain's accumulator at the correction row)
acc = W = −2^len·T₀   ⟺   u1·G + u2·P2 = −2^(len+1)·T₀

take u1 = u2 = 1  (len = 1)   ⇒   P2 = −2^(len+1)·T₀ − G      — one point subtraction
```

`audit_transcription.py` §E runs it:

```
P2 = (0x6d2601d3ba4652d07eec0066ed211b5cdbdf3c1cf8d0cbe07877b0871e15eb14,
      0x15d65f47ef5cb786daaf7273f57fd2d9d80c4d9a8ac63e0bd716c5bad616431c)
correction row: xA == xB True, yA == yB True
lambda relation identically 0 (free lambda): True
honest Q  = (0x3f0bcb55c7dbd41b34aaa2e82d5503a722cbc385fd8aaf74a6505d1ae51e77b, ...)
forged Q' family, 5 distinct values, none equal honest: True
```

With `xA = xB` *and* `yA = yB`, `op·(Σλ_j(xB−xA)_{i−j} + yA_i − yB_i)` is
identically zero for **every** λ, so `xR = λ² − 2xA`, `yR = λ(xA − xR) − yA` is a
free one-parameter family of outputs — the same shape as the NUMS finding this
relation was added to close, relocated to the correction row. `P2` is on the
curve and MEMW-bound, both scalars are in `[1, N)`, the schedule is the honest
one, the Addend and JointBit balances hold, and `status = 0`.

And with the tampered source in place:

```
chip_state dinv_relation  True     ecdas2_constraints 288     PORT all-identical True
l1_l5_port2 §3 (a1) unsat   (a2) Correction consumes an addend unsat
```

Every verdict on the board is unchanged. **The gate cannot distinguish the chip
that blocks this from the chip that does not.**

To be explicit: **the chip as committed has the `S_CORR` term** (verified by
reading `ecdas2.rs:810-813`), so this forgery is blocked today —
`d·(xB − xA) = 1` with `xB = xA` is unsatisfiable, which §E also confirms. F1 is
a hole in the gate's coverage, not in the chip.

**Fix:** have `chip_state()` extract the `Relation::Dinv` arm and assert the `g`
expression is literally the four selector columns, in the same style
`relation_bodies_identical()` uses for the other three arms — and assert it
matches the Addend receive's `Multiplicity::Linear` term list, which is the
property the prose actually claims.

---

## F2 — `chip_state()` detects the comment, not the defence

`gate2_common.py:9-21` is explicit about why this function exists: two soundness
fixes were outstanding, and "the gate must say so rather than quietly reporting
the expected SAT… so this file keeps telling the truth as `ecdas2.rs` changes
underneath it." `RESULTS-lincomb2.md` repeats the claim in a block quote.

`padding_digit_gate` (`gate2_common.py:122-126`) tries three regexes. The chip
emits idx 22..=27 as a loop over a column list (`ecdas2.rs:988-1003`):

```rust
for (i, col) in [cols::D1, cols::D2, cols::S1, cols::S2, cols::S3, cols::S_CORR] … {
    b.emit_base(22 + i, (one - mu) * x);
}
```

The multiplicand is the loop variable `x`, so the two **code** regexes
(`(one|1)\s*-\s*mu…\*\s*d1` and `d1\s*\*\s*\(\s*(one|1)\s*-\s*mu`) never match.
Only the **comment** regex fires — on the module-header table (`ecdas2.rs:116`)
and on the constraint map (`ecdas2.rs:669`). §D deletes the emitting loop while
leaving every comment intact:

```
emitting loop present : False
padding_digit_gate    : True     <- should be False
```

A chip in that state is the exact L6-E arbitrary-chosen-sender forgery, and
`l8_negative2.py` would score N1 as `SAT — FORGES` (a passing ablation) instead
of `LIVE HOLE`.

`dinv_relation` is `"D_INV" in src and "Dinv" in src`. §D removes
`(Relation::Dinv, cols::C3)` from the emit loop — the relation is then never
emitted, the block shrinks from 288 to 223 constraints — and the detector still
reports `True`, because the columns, the enum variant and the comments survive.

The other four `chip_state()` fields (`ecdas2_columns`, `ecdas2_constraints`,
`ecsm2_columns`, `jointbit_multiplicity`) are **display-only** — grepped, no
lemma or verdict reads them. `jointbit_multiplicity` in particular correctly
reports `Multiplicity::Column`, but if the chip changed it, nothing would react.

**Fix:** detect from the emitted expression, or better, assert on the *shape*:
parse the `for … b.emit_base(22 + i, (one - mu) * x)` loop and its column list,
and cross-check the list against the set of columns that appear in any
`Multiplicity::` of `bus_interactions()`. That last check is the invariant the
header actually claims ("every column that is a bus *multiplicity*") and it
would have caught the original JointBit bug from the other side.

---

## F3 — the PORT lemma compares the arms, not the operands

`relation_bodies_identical()` (`gate2_common.py:90-107`) extracts the
`Relation::{Lambda,Xr,Yr} => { … }` arms of `s_i` from both chips, normalises out
comments/whitespace and the `XB/YB → XG/YG` rename, and compares. Every lemma
marked PORTED rests on the result.

The operand bindings are **not in the arms**. They are in the `s_i` prologue,
before the `match` (`ecdas2.rs:747-755`):

```rust
let lam = |j| Self::byte_at(b, cols::LAMBDA, 32, j);
let xg  = |j| Self::byte_at(b, cols::XB,     32, j);
let xa  = |j| Self::byte_at(b, cols::XA,     32, j);
…
```

§C rebinds `xa` to `cols::XR` in the ECDAS2 prologue — so every occurrence of
`xA` in the λ, xR and yR relations reads `xR` instead, falsifying L3a's
composition identity and L4a/b/c's pinning outright — and re-runs the gate's own
check:

```
s_i::Lambda : identical      s_i::Xr : identical      s_i::Yr : identical
-> port argument still reports all-identical: True
```

`conv_carry` and the emit loop are not compared either (§C prints
`conv_carry bodies identical: False` — expected, ECDAS2 gained a `Dinv` arm —
but the function is never called by the port check, so nothing depends on it).

I verified by reading that the prologues **are** identical today, modulo exactly
the documented `XG→XB` / `YG→YB` rename, and the audit script confirms it under
the same normalisation. So the PORT premise holds in fact; it is just not
established by the check that claims to establish it.

**Fix:** compare `_norm(prologue(ECDAS1)) == _norm(prologue(ECDAS2))` and
`_fn_body(…, "conv_carry")` with the `Dinv` arm excised, alongside the existing
per-arm comparison. Three lines.

---

## F4 — N4b's forgery does not exist on the chip (model *weaker*, not stronger)

`l8_negative2.control_4b` builds its row with `Ecdas2Row(s, "pad", ablate=…)` —
`padding_gate` defaults to `False` (`gate2_common.py:194`), i.e. a chip **without
idx 22..=27**. It then sets `MU = 0, PH1 = PH2 = 0, OP = 0`, asks for a live
Addend receive, gets `unsat / sat`, and reports `SAT — FORGES`.

On the chip's real constraint set the same ablation is **UNSAT** (§B): idx
24..=27 zero every selector on a `MU = 0` row regardless of idx 14. Ablating idx
14 alone cannot mint the spurious receive.

This is the *safe* direction — a spurious SAT, a false alarm — but it means:

- the board's headline "**7 genuine forgeries**" non-vacuity claim is **6**;
- `RESULTS-lincomb2.md` §4's reading of N4b ("Load-bearing exactly where PH1 = 0:
  padding rows and the two off-chain phases") is half wrong: the padding-row half
  is covered by idx 22..=27, not by idx 14;
- N4c's note "Keep idx 14 — N4b is what it covers" cites the wrong evidence.

The conclusion *keep idx 14* survives — §B runs the probe N4b should have run,
on **live** `PH1 = 0` rows, where idx 22..=27 is vacuous:

```
precompute (MU=1, PH1=PH2=0)   untampered unsat   ablate-14 sat
correction (MU=1, PH2=1)       untampered unsat   ablate-14 sat
```

Without idx 14 the precompute or correction row can be an `OP = 0` doubling that
still consumes its addend — a genuine forgery. Only the justification needs
rewriting.

Same root cause, worth checking before the next run: `l1_l5_port2.py` §3(a1)/(a2)
and `l8` controls 4 and 4c also use `padding_gate=False`. Those are all `MU = 1`
queries, where idx 22..=27 is vacuous, so their verdicts are unaffected — but the
default is a trap. Make `padding_gate` default to `True` and have the ablations
pass `False` explicitly.

---

## F5 — the one literal over-strong assertion (benign)

`Ecdas2Row.__init__` (`gate2_common.py:200-201`) and
`l6_joint_counting.Row.__init__` (`:52-53`) assert `0 ≤ ROUND ≤ 255` on **every**
row. The chip's only `ROUND` range check is the `AreBytes` pair
`(ROUND, Q0+32)` at multiplicity `MU` (`ecdas2.rs:566`), which is **dead on
padding rows** — the model's own comment says so ("AreBytes(ROUND), MU-gated"),
and L6-COUNTING §2.2 notes it explicitly.

Verified benign. On a `MU = 0` row, idx 22..=27 force `D1 = D2 = S* = 0`, idx 14
then forces `OP = 0` and idx 13 forces `NB = 0`, so `ROUND` feeds only:
the JointBit send (multiplicity `D1`/`D2` = 0), the Ecdas receive/send
(multiplicity `MU` = 0) and the AreBytes pair (multiplicity `MU` = 0). It is
free and unread. Nothing is hidden.

Two smaller instances of the same shape, both benign for the same reason:
`l1_l5_port2` §3(c) bounds `q3 < 2^264` and §3(b1)/`l8` control 2 bound
`0 ≤ d < P`, where the chip gives `q3 < 2^264` and `d < 2^256` **only at
`MU = 1`**; both arguments (`p·(µR − q3) = 0`, `p | 1`) are independent of the
bound, so the conclusions stand.

---

## The assertion table

Chip references are `file:line` on `feat/ec-lincomb2` @ `bc62f00e`. Verdicts:
**match** = model = chip; **stronger** = model asserts more than the chip;
**weaker** = model omits a chip constraint; **unchecked** = the gate asserts it
but never reads the source.

### ECDAS2 constraints — `Ecdas2Row` / `check_row` / `l6_joint_counting.Row`

| model assertion | chip | verdict |
|---|---|---|
| `bit_var` on 11 columns MU, OP, NB, D1, D2, S1, S2, S3, S_CORR, PH1, PH2 | `ecdas2.rs:851-870` idx 0..=10, same 11 columns, `x·(1−x)`, ungated | match |
| idx 11 `PH1·PH2` | `:873-875` | match |
| idx 12 `OP·NB` | `:879-881` | match |
| idx 13 `(1−OP)(NB−D1−D2+D1·D2)` | `:886-891` | match |
| idx 14 `OP−S1−S2−S3−S_CORR` (ungated) | `:897-902`, ungated | match |
| idx 15/16 `(1−PH1)·D1`, `(1−PH1)·D2` | `:909-916` | match |
| idx 17 `PH1·S_CORR` | `:920-922` | match |
| idx 18 `PH1·(S1+S3−OP·D1)` | `:933-938` | match |
| idx 19 `PH1·(S2+S3−OP·D2)` | `:939-944` | match |
| idx 20 `MU·(1−PH1−PH2)·(S2−1)` — MU-gated | `:950-956`, MU-gated | match |
| idx 21 `PH2·(S_CORR−1)` — **not** MU-gated | `:960-963`, not MU-gated | match |
| idx 22..=27 `(1−MU)·{D1,D2,S1,S2,S3,S_CORR}` (opt-in) | `:988-1003`, exactly those 6 columns | match — but see F2 for how it is *detected*, F4 for the default |
| all constraints hold on every row | `emit_base` → `RowDomain::ALL` (`crypto/stark/src/constraints/builder.rs:133-135`); no `emit_base_rows`, no `except_last` in either chip | match |
| whole boolean block, 2^11 brute force | independent second transcription | **match, both directions, 11 = 11** |
| `ROUND ∈ [0,255]` on every row | `:566` AreBytes pair at multiplicity `MU` — dead at `MU = 0` | **stronger** (F5, benign) |
| relation `Dinv` gated by `ΣS`; `g == op` on live rows | `:810-813` `g = S1+S2+S3+S_CORR` | match — **unchecked** (F1) |
| `Dinv` `S_i = g·(Σd_j(xB−xA)_{i−j} − [i=0]) + rq(Q3)` | `:802-814` | match (read; `positive_real_witness2.py:176-183`) |
| 4 relation blocks × (64 ConvCarry + ColIsZero(c_63)) = idx 28..=287 | `:1006-1021`, `debug_assert_eq!(idx, 288)` | match |
| `IsHalfword` on `c_0..c_62` only (63 per relation), `c_63` pinned by ColIsZero | `:590` `for i in 0..63`; `:1018-1019` | match |
| λ/xR/yR arms identical to `ecdas.rs` modulo `XG/YG → XB/YB` | `ecdas2.rs:757-789` vs `ecdas.rs:363-395`; prologues `:747-755` vs `:353-361` | match (read) — **check does not cover the prologue** (F3) |
| `rq`, `p_byte_expr`, `r_byte_expr`, `byte_at` identical | `:685-738` vs `ecdas.rs:290-345` | match (mechanically compared) |
| `s_ecdas_lambda/xr/yr` (`gate_common.py:154-187`) | `ecdas.rs:363-395` | match (read, term by term) |
| `CARRY_OFFSET_DINV = CARRY_OFFSET_XR = 8161` | `:139`, and `OFF["ecdas_xr"]` in `positive_real_witness2.py:186` | match |

### ECDAS2 bus interactions — `digit_send` / `addend_receive` / the L6 prose

| model assertion | chip | verdict |
|---|---|---|
| Ecdas receive multiplicity `MU` | `:466-478` `mu()` | match |
| Ecdas send multiplicity `MU` | `:616-638` `mu()` | match |
| `digit_send` = raw `D1`/`D2`, **ungated** | `:601-612` `Multiplicity::Column(col)` | match |
| `addend_receive` = `S1+S2+S3+S_CORR`, **ungated** | `:487-504` `Multiplicity::Linear`, coefficient 1 each | match |
| 131 AreBytes at `MU` | `:545-572` — 8 bases × 16 + `(ROUND,Q0[32])` + `(Q1[32],Q2[32])` + `(Q3[32],0)` = 131, all `Column(cols::MU)` | match |
| 252 IsHalfword at `MU` | `:584-597` — 4 × 63, `mu()` | match |
| no other column supplies a multiplicity | grep of `Multiplicity::` in the file: `MU`, `D1`, `D2`, `{S1,S2,S3,S_CORR}` only | match |
| joint tuple `[1, ts_lo, ts_hi, phase, accX(32), accY(32), round, op]`, arity 70 | `:399-420`, `JOINT_CHAIN_ID = 1` at `:172` | match |
| `phase` = `PH1 + 2·PH2`, same expression on receive and send | `:446-457`, used at `:471` and `:621` | match |
| send round = `ROUND + NB − 1`, send op = `NB` | `:623-634` | match |
| addend tuple `[ts_lo, ts_hi, sel, x(32), y(32)]`, `sel = S1+2S2+3S3+4S_CORR` | `:423-439`, `:508-525` | match |
| JointBit send tuple `[ts_lo, ts_hi, ROUND, stream]`, stream ∈ {1,2} | `:601-611` | match |
| bus-28 separator: old chain pins tuple position 0 to `0`, joint chain to `1` ⇒ different α¹ coefficient | `ecsm.rs:616` `constant(0)`; `ecdas2.rs:411` `constant(1)`; `lookup.rs:1653` `alpha_offset = 1` advanced unconditionally, `:679` skips only the multiply | match — the board's correction of the source comment is right |

### ECSM2 — the L6/L8 prose and `l8.control_5`

| model assertion | chip | verdict |
|---|---|---|
| idx 2 `OK·(1−MU)`, idx 3 `OK·STATUS`, idx 4 `MU·(STATUS·S_INV−(1−OK))` | `ecsm2.rs:1156-1177` | match |
| all six chain seeds/drains at multiplicity `OK`; `OK` is IS_BIT | `:836-920` all `ok()`; `:1149-1153` idx 1 | match |
| seeds: phase 0 `(G, 0, round 0, op 1)`, phase 1 `(T₀, 1, LEN_M1, op 0)`, phase 2 `(ACC, 2, round 0, op 1)`; all drains `round = −1, op = 0` | `:836-920` | match |
| phase-1 drain and phase-2 seed use **the same columns** (a literal relay) | `:881` and `:899` both `coord(cols::ACC_X)/coord(cols::ACC_Y)` | match |
| JointBit receive multiplicity `2·u_bit`, raw | `:771-787` `Multiplicity::Linear([{coefficient: 2, column: base+i}])` | match |
| …made inert by idx 517/518 `(Σ u_bit)·(1−OK)` | `:1194-1203` | match |
| Addend publishes `N1`/`N2`/`N3` raw, correction at `OK` | `:795-817` | match |
| …made inert by idx 519..=521 `N·(1−OK)` | `:1207-1213` | match |
| everything else `OK`-gated; Ecall + x10 MEMW `MU`-gated | `:553-582` `mu()`, `:587-826` `ok()` | match |
| `len ≤ 256` structural via EC_T0 | `ec_t0.rs:116` `NUM_ROWS = 256`, no padding; `:353-369` receive key `LEN_M1 + 1` | match |
| EcT0 tuple `[len, x(32), y(32)]`, arity 65, both sides | `ecsm2.rs:821-826` / `ec_t0.rs:353-368` | match |
| `u1, u2 ≠ 0` via the Zero bus | `:749-765` — sums the 32 **bytes** (coefficient `1 << (b % 8)`), which is zero iff every byte is zero | match (the byte-sum form is correct, and is the only form expressible in `i64` coefficients) |
| membership relations X2/Yg are ECSM's with `µ → OK` | `ecsm2.rs:1048-1082` vs `ecsm.rs:744-782` | match (read) — **no lemma or mechanical check in the lincomb2 board covers this port** |

### Derived / modelled values

| model assertion | chip | verdict |
|---|---|---|
| `PHASE = {Precompute:(0,0), Correction:(0,1), Double/AddP1/AddP2/AddP12:(1,0)}` (`positive_real_witness2.py:72-73`) | `ecdas2.rs:258-264` `phase_bits`, all 6 `JointSel` arms | **match, arm for arm** |
| `SELECT = {Double:(0,0,0,0), AddP1:(1,0,0,0), AddP2:(0,1,0,0), Precompute:(0,1,0,0), AddP12:(0,0,1,0), Correction:(0,0,0,1)}` (`:74-76`) | `ecdas2.rs:268-276` `selector_bits`, all 6 arms | **match, arm for arm** |
| dict keys are the harness's `sel` strings | `oracle/repo-harness/src/main.rs:59-68` `sel_name`, all 6 | match |
| `JointSel` has exactly these 6 variants | `crypto/ecsm/src/witness.rs:551-565` | match |

This is `RESULTS-lincomb2.md` §7's flagged "one remaining modelled gap". It is
**exact today**, verified arm for arm against an exhaustive `match` — but it is a
hand copy with no mechanical link, so it belongs in the same class as F1/F2/F3.
`JointSel` is not `#[non_exhaustive]`, so a new variant forces a compile error in
`phase_bits`/`selector_bits`, but silently gets a `KeyError` (a crash, not a
false pass) on the Python side — acceptable.

---

## Cross-checks that came back clean

- **Row domains.** Neither chip calls `emit_base_rows`, `RowDomain::except_last`
  or any end-exemption; `emit_base` is `RowDomain::ALL`
  (`crypto/stark/src/constraints/builder.rs:133-135`). The model's "holds on
  every row" is exact. No padding-row exemption exists to be over-modelled.
- **Integer vs field modelling.** Every schedule constraint the model evaluates
  over ℤ has range within `[−4, 1]` on boolean inputs, so `≡ 0 mod p_g` and
  `= 0` over ℤ coincide. No wraparound is available to a prover.
- **Multiplicity semantics.** `Multiplicity::Column` is the raw column value and
  `Linear` the linear combination (`crypto/stark/src/lookup.rs:1328-1363`); there
  is no product form, so L6-COUNTING §4's "make the multiplicities `MU·D1`"
  suggestion is not currently expressible. The counting argument's "each row
  contributes 0 or 1" holds because `D1`/`D2` are IS_BIT.
- **`N1/N2/N3` need no range check.** The matching side is a sum of at most
  `2^k` bits with `k ≪ 64`, so its integer value is far below `p_g`; a "negative"
  or inflated count cannot balance. The chip comment is right.
- **Padding-row inertness, ECDAS2.** With idx 22..=27: `S* = 0` ⇒ (idx 14)
  `OP = 0` ⇒ (idx 13) `NB = 0`; `D1 = D2 = 0`. Every remaining interaction is
  `MU`-gated. Nothing a padding row can carry reaches a bus.
- **Padding/error-row inertness, ECSM2.** `MU = 0` ⇒ (idx 2) `OK = 0`; idx
  517-521 kill the four raw-multiplicity interactions; the rest are `OK`- or
  `MU`-gated. An error row (`MU = 1, OK = 0`) fires only the Ecall receive and
  the x10 access, and idx 4 forces `STATUS ≠ 0`.
- **L6-COUNTING §3(f)'s "at most two live rows share a round."** Re-derived:
  `OP·NB = 0` gives add ⇒ `Δround = −1`; a double with `NB = 1` forces the
  successor to be an add at the same round, which then decrements; so no cycle
  and no third row at a round. Balance decomposes each phase into one seed→drain
  path (cycles excluded by monotonicity), and the `round = −1` drain is
  unreceivable because live `ROUND` is byte-checked. The counting step is sound.
- **Byte-ness inheritance for `XB`/`YB`.** `point_coord_busvalues`
  (`ecsm.rs:284-286`) is `(0..32).map(packed)` — one element per byte — so tuple
  equality is per-limb and byte-ness transfers. `WIDTH-AUDIT.md` §3.1 already
  flags this as load-bearing. Roots: `G` constant, `P2` via MEMW, `P12` = the
  phase-0 drain of an AreBytes-checked `XR`/`YR`, `T₀N` from the preprocessed
  table. No cycle.
- **`gate_common.s_ecdas_*` vs `ecdas.rs`.** Compared term by term. Identical,
  including the `byte_at` zero-padding and the 33-limb quotients.
- **L3a/L4a/L4b/L4c models vs the composed relations.** `l3_l4_value.py:69-92`
  and `:148-231` state exactly the composed forms of `ecdas.rs`'s three
  relations with `µ = 1`; the operand bounds used (`lam < 2^256`, `q < 2^264`)
  are what the chip's AreBytes checks give at `MU = 1`.

---

## Could not verify

Listed so the boundary is explicit rather than implied.

1. **Contracts C1–C7 and A-PRIME.** I read the *sending* side in both chips, not
   the receiving side: `bitwise.rs`'s AreBytes/IsHalfword tables, MEMW's byte
   authority for `X_P2`/`Y_P2`, the CPU↔Ecall binding, timestamp uniqueness, and
   the generic LogUp balance argument. `RESULTS-lincomb2.md` says the contracts
   are "unchanged from RESULTS.md", but C4 as written there names `xG`/`k`/`xR`;
   its ECSM2 instance (`X_P2`, `Y_P2` bytes from the 8 MEMW dword reads at
   `ecsm2.rs:631-649`) is a different, unstated instance. Worth restating C4 for
   the joint chip.
2. **The ECSM2 membership port.** `Relation::X2`/`Yg` (`ecsm2.rs:1048-1082`) are
   ECSM's bodies with the column rename and `µ → OK`; I verified that by reading,
   but no lemma on the lincomb2 board and no mechanical check covers it, even
   though the soundness theorem's "`P2` is on the curve" clause depends on it.
   Same fix as F3: extend `relation_bodies_identical()` to the ECSM/ECSM2 pair.
3. **`positive_real_witness2.py` was not executed** — it needs the built Rust
   harness (`../oracle/repo-harness/target/release/ecsm-oracle-harness`). I read
   its model and verified its `PHASE`/`SELECT` maps and its idx 11..=21 / Dinv
   transcription against the source; I did not reproduce the 3.3M checks.
   Note that the anchor checks **constraint values only** — it never evaluates a
   multiplicity or a gating expression, so it could not have caught the original
   JointBit bug and cannot catch F1/F2 either.
4. **`l1_l2_lift.py`, `l3_l4_value.py`, `l5_sides.py`, `l8_negative.py`
   (the old-gate lemmas) were not re-run**, nor were `width_audit.py` /
   `width_audit_z3.py` (~25 min). I audited the *premise* they are ported on
   (F3) and read their models against `ecdas.rs`; I did not re-derive their
   proofs.
5. **Completeness-side mirroring.** `collect_bitwise_from_ecdas2` /
   `collect_bitwise_from_ecsm2` must mirror the AreBytes send layouts. Not
   checked — a mismatch is an unprovable honest witness, not a forgery.
6. **Whether the constructions in F1 can be packaged as well-formed `(z, v, r, s)`
   ecrecover inputs.** The chip-level forgery needs only an on-curve `P2` in
   memory, which the construction gives. Packaging additionally needs
   `x(P2) < N`, which is overwhelmingly likely but was not checked for the
   specific `P2` printed above.

---

## Documentation defects found along the way

Not soundness, but a reader following the citations is misled:

- **`L6-COUNTING.md`'s headline verdict is stale.** It still opens with "**L6
  does NOT hold for `ecdas2.rs` as written. There is a constructive break**",
  and §2 presents the padding-row forgery as live. `RESULTS-lincomb2.md` §3
  links it as "Full derivation in `../lincomb2/L6-COUNTING.md`". The fix landed
  as idx 22..=27; the doc needs a status header.
- **Stale line citations.** `L6-COUNTING.md` §2.1 and
  `l6_joint_counting.py:76,229` cite the JointBit send as `ecdas2.rs:459-470`;
  it is now `:601-612`. `l6_joint_counting.py:4` and
  `positive_real_witness2.py:15-18,80` still say ECDAS2 has 217 constraints
  / `idx 0..=216` / `22..=216`; it has 288 / `0..=287` / `28..=287` (the code
  itself does check all four relation blocks — only the prose is stale).
- **Tuple position off by one between docs.** `RESULTS-lincomb2.md` §3 says the
  separator is "tuple position 1"; `L6-COUNTING.md` §5.2 says "position 0". It
  is `values[0]`, which lands at α¹ because `alpha_offset` starts at 1
  (`lookup.rs:1653`). Both are defensible readings of "position"; pick one.
- **`gate2_common.py`'s module docstring** still describes both fixes as
  outstanding ("Two soundness fixes were outstanding when this gate was
  written"). True as history, confusing as a header.

---

## Recommended actions, in priority order — ALL DONE

1. **F1 — DONE.** `dinv_gate_state()` extracts the `Relation::Dinv` arm, checks
   `g` is applied as `g * s`, checks it is a plain sum of columns, and requires
   that set to equal the `Addend` receive's `Multiplicity::Linear` terms (each
   with coefficient 1, since a `coefficient: 2` would change the multiplicity
   without changing the set).
2. **F2 — DONE.** `padding_gated_columns()` parses the emitted expression in
   both the loop and unrolled forms; `padding_gate_state()` requires the gated
   set to **equal** `multiplicity_columns() − {MU}`, which resolves loop-bound
   `Multiplicity::Column(col)` through its `for` header. `Dinv` presence is read
   from the emit loop. Unparsed multiplicities are surfaced, never ignored.
3. **F3 — DONE.** `relation_bodies_identical()` now covers `s_i::prologue` and
   `conv_carry` (Dinv dispatch excised); `membership_bodies_identical()` covers
   the ECSM/ECSM2 pair, closing gap 2. `carry_chain` is excluded and the reason
   is stated in the docstring.
4. **F4 — DONE.** `padding_gate` defaults to `True`; N4b re-points at the live
   `PH1 = 0` phases and prints its witness rows; the count is corrected in
   `RESULTS-lincomb2.md` §4 with the history recorded.
5. **Doc hygiene — DONE.** `L6-COUNTING.md` has a status header and its stale
   cites/counts refreshed; `gate2_common.py`, `l6_joint_counting.py` and
   `positive_real_witness2.py` docstrings corrected; the position-0 / position-1
   wording reconciled (element 0, exponent α¹).

Beyond the list: the `JointSel → PH*/S*` mapping is compared arm for arm against
the chip, which closes `RESULTS-lincomb2.md` §7's last "modelled step", and §7
now records the anchor's structural limitation (constraint values only — never a
multiplicity, never a gating expression).

None of this changed a chip. The chips, as far as this audit can determine,
enforce everything the gate says they do — and the gate now checks that claim
instead of asserting it.

## Regression suite

`audit_transcription.py`, 22/22:

```
A  model ⊆ chip and chip ⊆ model on idx 0..=27 (2^11 cases, empty both ways)
   padding_gate defaults to the chip
B  the OLD N4b target is NOT a forgery; the NEW one IS
C  untampered ECDAS/ECDAS2 and ECSM/ECSM2 ports clean;
   rebound operand column DETECTED; broken carry recurrence DETECTED;
   tampered ECSM2 membership relation DETECTED
D  gated set == raw multiplicity set; defence-deleted-comment-kept DETECTED;
   digit send escaping the gate DETECTED; a NEW ungated multiplicity DETECTED;
   Dinv dropped from the emit loop DETECTED
E  gate == Addend receive; S_CORR dropped DETECTED; the forgery it hides,
   constructed, and blocked by the real chip
F  JointSel arms match; a changed arm DETECTED
G  unparsed gate shape reports ABSENT; opaque gate helper rejected
```
