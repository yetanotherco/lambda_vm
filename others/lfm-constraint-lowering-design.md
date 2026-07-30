# Design: lowering a `ConstraintArtifact` to LFM instructions

Design (α) from `lfm-design.md` §3 — the constraint-evaluation leg of the epoch
verifier. Written by the phase0 agent 2026-07-30 against
`feat/phase0-constraint-ir`. **Design only; no semantics touched.**

Every number below is measured by `constraint_op_census` in
`prover/src/tests/constraint_artifact_tests.rs`, which is a standing instrument —
run it, do not trust this file's copy of the numbers after the constraints change.

Cost facts about the machine (fusion parity, `MulBase` parity, free base→ext,
program-wide constant interning, one-instruction-one-row) come from the ISA
inventory of `prover/src/lfm/`, relayed by the team lead; §2.3 marks which of
them I confirmed against the IR myself and which I took on report.

---

## 0. Headline

**The leg comes in materially UNDER budget.** `lfm-design.md` §5.2 claimed ≈69K
instructions at 25 AIRs, implicitly assuming roughly one instruction per IR node.
Measured at **28** AIRs, with the machine's actual cost model applied:

```
upper bound (one instruction per arithmetic node)   66,652
− MulAdd fusion (9,069 pairs)                       −9,069
= ESTIMATE                                          57,583      ~16.5% under the ≈69K claim
```

Fusion is not an optimization here. `MulAdd` costs the same single row as `Mul`,
so emitting `Mul` then `Add` where one instruction would do is pure waste — the
node count is an **upper bound**, not an estimate, until it is applied.

Three corrections to how the number should be read:

1. **The IR's `dim` tags are the wrong split to budget against** — they describe
   the prover, and the machine runs the verifier (§3). Budgeting from declared
   dims understates extension traffic by 14×.
2. **Constants are interned program-wide**, so the 655 per-AIR pooled constants
   are **315** actual `Const` rows (§4.2).
3. **57,583 is per distinct AIR.** The per-EPOCH figure is now measured (§8.2):
   **≈65K at 1–2M cycles, ≈96K at 20M**, a 1.01–1.49× multiplier. Small, because
   chunking multiplies the cheap AIRs while the expensive ones are never chunked.
   §8.2.1 corrects an earlier claim of mine that the leg is workload-shaped — it
   is not, and the architecture says so.

**Nothing in the IR is structurally inexpressible on a straight-line machine.**
The IR is already in precisely the form the machine's soundness argument demands
— §9, the most reassuring section here.

---

## 1. What the pass consumes and produces

Input: a `ConstraintArtifact` (the flat POD program, per-constraint metadata, AIR
shape, composition degree multiplier). Output: a straight-line `Vec<Instr<F>>`
fragment plus the addresses of the per-AIR quotient contributions.

The pass runs **at registry-build time on the host**, so it may do arbitrary
host-side work — constant folding, peephole fusion, fanout analysis — none of
which costs machine instructions. What it emits is fixed program text whose
digest the registry pins. `Instr::Const` values live in the `LFM_CONST`
preprocessed columns, so they are program data covered by that same digest: this
is what lets a constraint artifact be embedded without a separate commitment
scheme, and it is the concrete reason design (β) was not needed.

---

## 2. Node → instruction mapping, and it is total

Eleven IR ops. Six are leaves that resolve to an address and emit nothing; five
are arithmetic. One instruction is exactly one row on exactly one chip.

| IR op | verify-time value | machine lowering | rows |
|---|---|---|---|
| `Var{main,offset,row,col}` | ext | address in the OOD frame region | 0 |
| `RapChallenge{idx}` | ext | address in the challenge region | 0 |
| `AlphaPow{idx}` | ext | address in the alpha-power region | 0 |
| `TableOffset` | ext | address of the per-proof `L/N` | 0 |
| `ConstBase(idx)` | **base** | `Const{(c,0,0,0)}`, interned program-wide | 1 per distinct word |
| `ConstExt(idx)` | ext | `Const{(c0,c1,c2,0)}`, interned program-wide | 1 per distinct word |
| `Add(a,b)` | ext | `ExtAlu{Add}`, or folded into `MulAdd` (§5) | 1 or 0 |
| `Sub(a,b)` | ext | `ExtAlu{Sub}` | 1 |
| `Mul(a,b)` | ext | `ExtAlu{Mul}`, `MulBase` if one operand is base, `MulAdd` if fused | 1 |
| `Neg(a)` | ext | **`ExtAlu{Sub, a: ZERO, b: a}`** — §2.1 | 1 |
| `Embed(a)` | ext | **nothing** — §2.2 | 0 |

### 2.1 `Op::Neg` has no instruction

`ExtOp` is `Add | Sub | Mul | Div | MulAdd | MulBase`. There is no unary negate.
`Neg(a)` lowers to `Sub` from a pooled zero, which every program already has
(`IrBuilder` reserves node id 0 as the base-field zero, and zero is interned once
program-wide anyway). The mapping is total, but only via that identity — worth
writing down rather than rediscovering.

### 2.2 `Op::Embed` is free, and this is a payoff of the word model

Base→ext conversion costs **no instruction at all**: a base word IS a valid
extension word, the distinction being only which lanes are zero, and those zero
lanes are pinned by constant expressions in the bus tuple rather than by columns
(`SOUNDNESS.md` §4). So `Embed` is a pure address alias — the emitter records
that node `i` refers to node `a`'s address and emits nothing.

The converse is not free: ext→base costs 1 `LANES` row (`Unpack`). **The
constraint leg never needs it.** Nothing in the IR narrows an extension value to
a base one — `Dim` only ever widens through `binop`'s join. That asymmetry is
what makes the all-extension verifier evaluation (§3) affordable despite carrying
20× more extension traffic than the IR's tags suggest.

**Measured: 0 `Embed` nodes across all 28 production AIRs**, and 0 `ConstExt`.
Both arms are correctness-only today. Keep them: a missing arm is a panic in
`ConstraintArtifact::program` or, worse, a silent wrong answer in the CUDA kernel.

### 2.3 What I verified vs what I took on report

Confirmed by me against the IR and the artifact: the eleven-op inventory,
`Op::Neg` having no ISA counterpart, `Embed`/`ConstExt` being unused in
production, the absence of any ext→base narrowing, and every count in §8.

Taken from the ISA inventory without independent verification: one instruction =
one row; `MulAdd` and `MulBase` costing the same row as `Mul`; base→ext being
free; program-wide constant interning; group heights padding to
`next_power_of_two().max(4)`. If any of those is wrong the instruction counts in
§8 still stand — they are counts of instructions — but the row/cell conclusions
drawn from them do not.

---

## 3. The split that matters: prover dims are not machine dims

This is the correction I most want on the record.

`Dim` records what the **prover** computes. Its frame is base-field, so a
trace-only subexpression stays base, and the IR tags 42,137 of 67,103 arithmetic
nodes `Dim::Base`.

The machine runs the **verifier's** evaluation at the OOD point, where the frame
holds only extension elements — `eval_program_verifier` resolves every `Var` to
`Value::Ext` regardless of `main`, because the verifier has openings, not trace
cells. Propagating that through `interp::binop`'s rule (base only when both
operands are base values *and* the declared dim is base), a node is base at
verify time **only if its entire subtree is constants**.

```
arithmetic nodes                67,103
  base by the IR's own dim      42,137   <- prover-side. NOT the machine's split.
  base at verify time            2,916   <- constant-only subtrees
  extension                     59,146   <- 94% of the arithmetic
```

**A 14× discrepancy.** Anyone sizing this leg from the IR's `dim` column would
conclude most of the work is cheap base arithmetic. It is not.

Two consequences.

**The 2,916 base nodes cost nothing at all.** A constant-only subtree is a
compile-time constant: the emitter folds it during the host-side pass and interns
the result. Zero rows, which is why they are excluded from §0 rather than charged
as `BaseAlu`.

**5,041 multiplies must be routed through `MulBase`.** An ext×base multiply is
1 `XALU` row through `MulBase`, versus 4+ if lowered by hand as three base
multiplies plus a repack. So this is not a *reduction* against `Mul` — both are
one row — it is a **routing obligation**: the emitter must recognise the case, or
it pays 4× for it. The eligible operand must be a genuine base cell, because
`LFM_XALU` constrains its shared B-columns to zero on `MulBase` rows so the
received token matches a base writer's (`SOUNDNESS.md` §4); at verify time that
means a folded constant, which is exactly the 5,041 the census counts.

Note the count would be 9,413 if one used the prover dims — nearly double, and
wrong.

---

## 4. Where operands come from

### 4.1 Four regions, and the distinction is a soundness boundary

| region | source | authentication |
|---|---|---|
| OOD frame values (`Var`) | arena, hint-fed | the DEEP/opening leg — the machine hashes them into the openings it checks |
| challenges (`RapChallenge`) | transcript replay | computed in-machine by `LFM_HASH` rows; **never** hinted |
| alpha powers (`AlphaPow`) | derived from α | computed in-machine, once per proof |
| table offset (`TableOffset`) | derived `L/N` | computed in-machine, once per proof |
| constants | the program's own pool | program text; registry-digest-pinned |

The arena rule (`SOUNDNESS.md` §5) says an arena value is unconstrained by the
reading chip and must be transitively authenticated by a hash the machine
performs. OOD frame values satisfy it because the DEEP leg absorbs them;
challenges must never come from an arena and do not.

**The constraint leg pays nothing marginal for any of this.** Measured **5,964
leaf nodes**, all addresses of values other legs already materialized. That is
the single biggest reason the leg is ~1% of the program despite 73,722 nodes.

### 4.2 Constants are interned program-wide

Each distinct 4-lane word is one `Const` row regardless of how many nodes, or how
many AIRs, reference it. Summing per-AIR pools overcounts badly, because small
structural constants (0, 1, 2, `2^8`, `2^16`, `2^24`) recur in every table.

```
per-AIR pools, summed     655
interned program-wide     315      <- the actual Const row count
```

More than half the apparent constant cost is duplication across tables.

### 4.3 Address assignment and `mult`

Addresses are dense and compiler-assigned in emission order: the emitter walks
the node list in index order, assigning address = base + i and skipping folded,
aliased and fused nodes.

`mult(a)` — the statically known read count every write carries — is the node's
fanout in the IR DAG, plus one if the node is a constraint root (the quotient
recombination reads it). **Measured max fanout 1,632**, so the multiplicity
column holds values into the low thousands; it is a field element so this is
comfortable, but it is not the "small" value one might assume when sizing a range
check on it.

**Measured: 3 dead nodes (fanout 0) across all 28 AIRs.** Tiny, but the emitter
must DCE them rather than emit zero-multiplicity writes — the registrar's (M)
check is mult-equality.

---

## 5. `MulAdd` fusion is mandatory, not an optimization

`ExtAlu` carries `MulAdd` as a first-class op **at the same one-row cost as
`Mul`**. The IR has no `MulAdd` node — `CaptureBuilder` emits `Mul` then `Add` —
so an unfused emitter pays two rows where one would do, every time.

**The fusion is valid only when the `Mul` has exactly one consumer.** Hash-consing
means a shared `Mul` feeds several `Add`s; fusing it into each would recompute it
per consumer. (Fusing into just one of several consumers is cost-neutral, not a
saving: the `Mul` row still has to exist for the others.) A node that is a
constraint root also counts as a consumer — fusing it away would delete the value
§6 needs.

**Measured: 9,069 fusable `Add` nodes**, 13.6% of the leg. This is the difference
between the upper bound and the estimate in §0.

Worth naming as a near-miss: the hash-consing that makes the IR compact is the
same property that makes naive fusion unsound. Shared subexpressions are an asset
for program length and a hazard for peepholes — any future fusion needs the same
single-consumer guard.

---

## 6. Zerofier and quotient recombination

The composition quotient is `H = Σ_c β^c · C_c / Z_c`.

### 6.1 Uniform zerofiers make this cheap, and the saving is large

**All 28 production AIRs emit through `RowDomain::ALL`** — measured; nothing under
`prover/src` calls `RowDomain::except_last`. So `end_exemptions = 0` everywhere,
and every constraint of an AIR shares one zerofier `Z = ζ^N − 1`, depending only
on the sub-proof's trace length.

Per sub-proof:

```
ζ^N        repeated squaring    log2(N) × ExtAlu{Mul}   ≈ 20–24 rows
ζ^N − 1    one Sub against the interned 1              1 row
1/Z        one ExtAlu{Div}                             1 row
                                                 ────────────
                                                 ≈ 22–26 rows
```

Because `Z` is shared, the division factors out of the sum:
`H_air = (Σ_c β^c · C_c) / Z` — **one division per AIR, not per constraint**. The
sum is a Horner fold, one `ExtAlu{MulAdd}` per constraint, 2,150 total.

The saving is the entire value of the uniform-zerofier finding, so it is worth a
number. The naive shape — what `main` does today, recomputing `ζ^N` and a full
extension inversion once per constraint (`lfm-design.md` §5.2 hygiene item 1) —
costs `2,150 × ~24 ≈ 51,600` rows. Once per AIR costs `28 × ~24 ≈ 672`.
**≈50,900 rows saved, comparable to the entire rest of the leg.** The GPU path's
uniform-zerofier precondition holding in fact rather than by luck is the same
fact, cashed differently.

Total recombination ≈ 2,150 `MulAdd` + 28 `Div` + ~672 zerofier ≈ **2,850 rows**,
of which §0 counts the 2,150 β-folds and folds the rest into per-sub-proof
overhead.

### 6.2 The final comparison

Comparing `H` against the claimed composition parts is `assert_eq`, which is not
an instruction: it is 2 ALU rows plus an interned constant, via the
division-by-zero mechanism (`div` is constrained `B·OUT = A`, so `B = 0` forces
`A = 0`). A handful of rows per sub-proof; negligible against the above.

### 6.3 If a constraint ever grows an exemption

The zerofier gains factors and the emitter must evaluate one **per distinct
`end_exemptions` value per AIR**, not per constraint. The artifact's
`ConstraintMeta` carries exactly what is needed to group them, and
`transition_zerofier_evaluations_grouped` already keys its dedup on that field
host-side. Cost scales with the number of distinct values, currently one.

---

## 7. Boundary constraint and the next-row read

**Boundary.** Every VM AIR uses `NullBoundaryConstraintBuilder`, so the only
boundary constraint is the framework's `acc[0] = 0` per chip. At ζ that is
`(P(ζ) − 0)/(ζ − 1)`: one Sub for the denominator, one Div, numerator is the
opened value itself. **≈3 rows per sub-proof.** The accumulator's circularity
needs no boundary constraint of its own — it rides the plain `ζ^N − 1` zerofier.

**The next-row read.** The machine has no rows, so "next row" is not a concept it
needs: `Op::Var{offset: 1, col}` is simply a different address, and the DEEP leg
supplies the `g·ζ` opening alongside the `ζ` ones. Zero extra rows.

What makes this cheap is a shape fact worth re-verifying rather than assuming:
**every AIR declares exactly one next-row column** (the LogUp accumulator), or
none. That is not folklore — `ood_window_ir_tests` derives the true next-row read
set from the captured IR and asserts equality with the declaration for all 28
AIRs, and that check is what stands between a correct verifier and one that
silently reconstructs an omitted `g·ζ` column as ZERO. It now covers the three
continuation AIRs, which it did not before this phase.

---

## 8. Measured counts

### 8.1 Per AIR (28 tables; the artifact is blowup- and trace-length-invariant)

`instr` = extension ALU + MulBase, before fusion and before program-wide constant
interning (both of which are global, so they cannot be attributed per row).
Leaves are free; constant-only subtrees fold at build time.

| table | nodes | leaves | fold | ext | mulbase | **instr** |
|---|---:|---:|---:|---:|---:|---:|
| CPU | 600 | 75 | 4 | 417 | 72 | **489** |
| BITWISE | 158 | 33 | 3 | 106 | 6 | **112** |
| LT | 160 | 32 | 2 | 106 | 10 | **116** |
| SHIFT | 393 | 48 | 5 | 299 | 22 | **321** |
| EQ | 124 | 25 | 2 | 77 | 11 | **88** |
| BYTEWISE | 185 | 41 | 0 | 120 | 18 | **138** |
| STORE | 201 | 40 | 2 | 136 | 13 | **149** |
| CPU32 | 516 | 77 | 3 | 356 | 58 | **414** |
| MEMW | 552 | 89 | 3 | 429 | 19 | **448** |
| MEMW_A | 392 | 66 | 3 | 303 | 8 | **311** |
| MEMW_R | 202 | 41 | 2 | 129 | 24 | **153** |
| LOAD | 225 | 48 | 1 | 144 | 18 | **162** |
| DECODE | 35 | 15 | 0 | 18 | 0 | **18** |
| MUL | 388 | 48 | 2 | 276 | 44 | **320** |
| DVRM | 511 | 61 | 7 | 362 | 61 | **423** |
| BRANCH | 147 | 29 | 2 | 96 | 12 | **108** |
| HALT | 825 | 49 | 37 | 600 | 101 | **701** |
| COMMIT | 438 | 55 | 8 | 313 | 46 | **359** |
| PAGE | 63 | 16 | 2 | 34 | 7 | **41** |
| REGISTER | 49 | 15 | 2 | 23 | 6 | **29** |
| KECCAK | 3,997 | 784 | 30 | 2,960 | 186 | **3,146** |
| KECCAK_RND | 16,317 | 2,262 | 17 | 12,677 | 1,339 | **14,016** |
| KECCAK_RC | 51 | 23 | 0 | 26 | 0 | **26** |
| ECSM | 22,162 | 1,093 | 1,513 | 17,611 | 1,653 | **19,264** |
| ECDAS | 24,848 | 851 | 1,262 | 21,424 | 1,294 | **22,718** |
| L2G_GLOBAL | 47 | 15 | 1 | 24 | 3 | **27** |
| L2G_MEMORY | 93 | 21 | 1 | 60 | 5 | **65** |
| GLOBAL_MEMORY | 43 | 12 | 2 | 20 | 5 | **25** |
| **TOTAL** | **73,722** | **5,964** | **2,916** | **59,146** | **5,041** | **64,187** |

Program-wide: + 315 interned `Const` rows + 2,150 β-folds = **66,652 upper
bound**; − 9,069 fused = **57,583 estimate**.

### 8.2 The per-epoch multiplier, MEASURED

The §8.1 total is per distinct AIR. An epoch evaluates the leg once per
SUB-PROOF, and the 14 split-table families are chunked —
`chunks = ceil(rows / max_rows[table])`, with `max_rows` sized per table so each
chunk costs about the same memory (`tables/mod.rs::max_rows`). So

```
constraint rows per epoch = Σ over sub-proofs  instr(that sub-proof's AIR)
```

Measured by `epoch_chunk_multiplier`, which builds real traces so the chunk
counts are the prover's own splitting rather than a reconstruction of it:

| fixture | cycles | chunked sub-proofs | chunked | fixed | pages | **epoch total** | **multiplier** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fib_iterative_1M` | 1.0M | 16 | 4,282 | 60,389 | 41 | **64,712** | **1.01×** |
| `fib_iterative_2M` | 2.0M | 20 | 5,566 | 60,389 | 41 | **65,996** | **1.03×** |
| `array_multipass_20M` | 20.4M | 123 | 34,938 | 60,389 | 205 | **95,532** | **1.49×** |

**The multiplier is small, and the reason is structural**: chunking multiplies
the CHEAP AIRs. CPU is 489 instructions, MEMW_R 153; even 40 CPU chunks at 20M
cycles adds only 19,560. The expensive AIRs — ECSM 19,264, ECDAS 22,718,
KECCAK_RND 14,016 — are never chunked, contributing exactly one sub-proof each.

So the leg runs **≈65K per epoch at 1–2M cycles, ≈96K at 20M**. `lfm-design.md`
§5.2's ≈69K was closer to right than my earlier warning implied; the correction
is a modest growth term in epoch size, not a multiplier on the whole figure.

### 8.2.1 CORRECTION — my "workload-shaped" claim was wrong

An earlier version of this document said the leg was workload-shaped: that
ECDAS + ECSM + KECCAK_RND are 87% of the per-AIR total, so an epoch doing no
elliptic-curve work would drop 65%. **That is false, and the architecture says
so plainly.**

`FIXED_TABLE_COUNT = 10` is documented as "tables that always contribute exactly
one sub-proof, **regardless of `TableCounts`**: bitwise, decode, halt, commit,
keccak, keccak_rnd, keccak_rc, register, ecsm, ecdas" (`prover/src/lib.rs`).
ECSM, ECDAS and the keccak tables are present in **every** epoch whether the
workload touches them or not — a zero-row table still needs its sub-proof, since
dropping it would remove its constraints from verification.

The measurement above confirms it: the `fib_iterative` fixtures use no
elliptic-curve and no keccak work, and still carry the full 60,389-instruction
fixed block.

The correct statement is the opposite of what I wrote: **the constraint leg is
essentially workload-INDEPENDENT.** ~94% of it is the always-present fixed block;
what varies is the chunked remainder, which tracks epoch size rather than
instruction mix. That is a better property to have — the leg is predictable — but
I asserted the reverse from the census alone without checking how sub-proofs are
actually assembled, and the census cannot see that.

### 8.2.2 Continuation epochs specifically

The table above is the monolithic shape. A continuation epoch differs in three
small ways, none of which changes the magnitude:

- **PAGE does not appear.** Continuation epochs pass `page_configs = &[]`
  (`continuation.rs:693`, `:797`, enforced prover-side at `:677-681`), so the
  41-instruction-per-page term vanishes from the epoch proof.
- **One L2G_MEMORY sub-proof** per epoch: +65 instructions.
- **Intermediate epochs drop HALT** (9 fixed tables, not 10): −701 instructions.

The global proof carries one L2G_GLOBAL per epoch (27 each) plus one
GLOBAL_MEMORY per touched page (25 each) — negligible at any plausible page
count.

### 8.3 Against the design doc's claim

| | design doc (25 AIRs) | measured (28 AIRs) |
|---|---:|---:|
| IR nodes | 73,539 | 73,722 |
| arithmetic ops | 66,982 | 67,103 |
| constraint-leg instr | ≈69K | 66,652 upper bound |
| with mandatory fusion | — | **57,583** |

The ≈69K claim assumed roughly 1:1 with nodes. Fusion is common — 9,069 pairs —
so the per-distinct-AIR figure lands **16.5% under**. Applying §8.2's measured
per-epoch multiplier (1.01–1.49×) puts a real epoch at **≈58K–86K instructions**,
which brackets the ≈69K claim rather than contradicting it.

### 8.4 On converting rows to cells

One instruction is one row on one chip, but group heights pad to
`next_power_of_two().max(4)`, so marginal row cost is zero until a boundary is
crossed and the meaningful metric is per-chip padded height × value width.
`airs::lfm_cell_counts` is the instrument for that. Everything above is in
INSTRUCTIONS/rows; the padded-cell figure needs the per-chip distribution, which
depends on how this leg's rows interleave with the rest of the program's — not
something the leg can be costed for in isolation.

---

## 9. Structural expressibility: nothing blocks

There is nothing the machine cannot express, and the reason is stronger than "it
happens to work".

- **The IR is a pure DAG with no control flow.** No branches, no loops, no
  data-dependent addressing. `ConstraintProgram` is a topologically ordered node
  list, which is what a straight-line program *is*.
- **`nodes[i]` references only `< i`.** The IR's own documented invariant,
  enforced by `ConstraintArtifact::validate_self`. It is *identical* to the
  machine's acyclicity premise (A) — "operand address < destination address"
  (`SOUNDNESS.md` §2). Dense address assignment in node order satisfies (A) **by
  construction**, with no reordering pass and no verification burden beyond the
  check the artifact already runs.
- **Fanout is statically known**, so `mult` comes straight off the DAG. The
  write-once model needs exactly this and the IR already has it.
- **No division in the constraint algebra.** `Op` has no `Div`. Division enters
  only at the zerofier/quotient step (§6), a handful of rows per sub-proof.
- **No ext→base narrowing anywhere**, so the one conversion that costs a row
  (`Unpack`) is never needed by this leg.

The only genuine mismatch is trivial: `Op::Neg` has no instruction and lowers to
a subtract from zero (§2.1). A lowering detail, not a structural obstacle.

---

## 10. What I did not verify

- **A continuation-epoch fixture.** §8.2's multiplier is measured on MONOLITHIC
  runs; §8.2.2 derives the continuation differences by reading the code rather
  than by running one. The differences are small and structural, but they are
  inferred.
- **The machine-side cost facts listed in §2.3**, taken from the ISA inventory
  rather than read by me. The instruction counts survive if any is wrong; the row
  and cell conclusions do not.
- **Padded-cell cost**, which needs the whole program's per-chip distribution
  (§8.4), not this leg alone.
