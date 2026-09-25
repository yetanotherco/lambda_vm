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
3. **57,583 is per distinct AIR.** For the shape we actually recurse — a
   CONTINUATION EPOCH — the leg is **63,393 instructions over 24 sub-proofs** at
   the minimum epoch and **64,035 over 26 at 2^20 cycles**, both measured
   (§8.2.2). Doubling the epoch past CPU's chunk bound costs 642 instructions, so
   the leg is ≈63–65K across any plausible epoch size. §8.2.1 corrects an earlier
   claim of mine that the leg is workload-shaped — it is not, the architecture
   says so, and that is what collapses the registry ladder to one dimension.

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
instruction mix.

#### Why this matters more than an erratum: it collapses the registry ladder

This lands directly on the open profile-ladder question — *how many distinct
programs must the registry carry?*

A constraint leg that were workload-shaped would make the emitted program vary
with workload class, and the registry would have to carry a **cross-product**:
workload classes × epoch shapes. That is the feared outcome, and it is the shape
that makes registry entries hard to enumerate.

Because the leg is ~94% fixed, the emitted program barely varies with what the
workload computes. What remains is the chunked term, which tracks **epoch SIZE**
— the 1.01× → 1.49× growth measured in §8.2. So the ladder is
**one-dimensional**: a short list of epoch shapes, not a cross-product. Each rung
is an epoch size, and every workload of that size shares a program.

That is the most consequential consequence of the measurement, and it is the
opposite of what my erratum-version claimed. It also composes with the
`page_base` uniform promotion, which removes the *other* source of
workload-dependence (§0.1 of the uniform proposal): with both, the emitted
program's identity depends on epoch shape alone.

#### The generalizable lesson

The node census cannot see how sub-proofs are **assembled**. It reads captured
IR, one AIR at a time; nothing in it knows that `FIXED_TABLE_COUNT` forces a
sub-proof for a zero-row table. Any inference about workload sensitivity, epoch
composition, or sub-proof count is therefore outside what that instrument can
support, however tempting the per-AIR table makes it. I asserted one anyway.

### 8.2.2 The continuation epoch — the shape we actually recurse

The table above is the monolithic shape, which is the wrong one for the target.
A continuation epoch differs in three ways:

- **PAGE does not appear.** Epochs pass `page_configs = &[]`
  (`continuation.rs:693`, `:797`, enforced prover-side at `:677-681`), so
  `create_page_air` is never called and the per-page term vanishes.
- **One L2G_MEMORY sub-proof** per epoch: +65 instructions.
- **Intermediate epochs drop HALT** (9 fixed tables, not 10): −701.

Composition and totals, computed by `continuation_epoch_constraint_leg`:

```
14 split families (>= 1 chunk each)      3,640
 9 fixed, no HALT                       59,688
 1 L2G_MEMORY                               65
INTERMEDIATE epoch                      63,393 instr over 24 sub-proofs
FINAL epoch (+HALT)                     64,094 instr over 25 sub-proofs
```

**The 24/25 sub-proof count is independently measured** on the LFM fibonacci
epoch fixture, and the test asserts that this composition reproduces it — so the
shape is pinned rather than inferred. If the epoch shape changes, the arithmetic
stops matching and the test fails.

Those 24/25 are the **minimum**: one chunk per family, i.e. an epoch of ≤2^19
cycles. `continuation_epoch_chunk_counts_measured` drives the real continuation
path — `Executor::resume_with_limit` for one epoch's cycles, then
`Traces::from_image_and_logs` — to measure a larger epoch first-hand:

| epoch | cycles | chunked sub-proofs | **total sub-proofs** | **instr** |
|---|---:|---:|---:|---:|
| minimum | ≤2^19 | 14 | **24** | **63,393** |
| measured | 2^20 | 16 (CPU ×2, MEMW_R ×2) | **26** | **64,035** |

Doubling the epoch past CPU's 2^19 chunk bound costs **642 instructions** — one
extra CPU chunk (489) and one extra MEMW_R (153). That is the whole growth term,
and it is why §8.2's monolithic 1.49× at 20M cycles is an over-estimate for an
epoch: an epoch never gets that large, because it is capped at `epoch_size`.

Two things fell out of running it that are worth more than the numbers:

- **`fib_iterative_2M` and `array_multipass_20M` produce IDENTICAL chunk counts**
  for their first 2^20 cycles — two quite different workloads, same 16 sub-proofs
  and same 4,282 instructions. Workload-independence, visible directly rather
  than argued from `FIXED_TABLE_COUNT`.
- **The test asserts `traces.page_configs.is_empty()`**, so "a continuation epoch
  never builds PAGE" is now pinned by a run rather than read off a comment.

**94% of it is the fixed block**, which is the sharpest statement of §8.2.1: the
constraint leg for a continuation epoch is ≈63K instructions essentially
regardless of what the workload does, growing only with epoch size as cheap AIRs
chunk.

The global proof carries one L2G_GLOBAL per epoch (27 each) plus one
GLOBAL_MEMORY per touched page (25 each) — negligible at any plausible page
count, which is what settles the page-base question as an identity problem rather
than a size one.

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

- Nothing remaining on the epoch numbers. §8.2.2 is now first-hand for the
  continuation path at both the minimum epoch and 2^20 cycles; §8.2's monolithic
  table is retained only as the whole-execution comparison, and is explicitly
  the wrong shape for the target.
- **The machine-side cost facts listed in §2.3**, taken from the ISA inventory
  rather than read by me. The instruction counts survive if any is wrong; the row
  and cell conclusions do not.
- **Padded-cell cost**, which needs the whole program's per-chip distribution
  (§8.4), not this leg alone.

---

# Corrections from building it — the emitter agent, 2026-07-30

Added by the agent that implemented this design as `prover/src/lfm/constraints.rs`
(branch `feat/lfm-constraint-emitter`). The original text above is left as its
author wrote it; everything below is measured by
`lfm::constraint_tests::constraint_leg_instruction_census` and
`..::continuation_epoch_constraint_leg_cost`, both of which fail if the numbers
move. Where a correction is a judgement rather than a measurement, it says so.

## What reproduced exactly

- **§8.1's whole per-AIR `instr` column**, all 28 tables, total **64,187**. The
  census test asserts it table by table and fails loudly if any entry drifts.
- **§8.2.2's `63,393` intermediate-epoch budget**, rebuilt from those counts by
  the same 14-families / 9-fixed / 1-L2G_MEMORY composition.
- **§4.3's "3 dead nodes (fanout 0)"** — but see below for what they are.
- **§2.2's claim that production has no `Embed` and no `ConstExt`**, and §9's
  claim that dense address assignment in node order satisfies acyclicity by
  construction. The emitter needed no reordering pass.

## What the emitter actually costs

```
per distinct AIR (28)      64,187 unfused  →  55,147 emitted
per INTERMEDIATE epoch     54,358 leg + 2,894 recombination = 57,252 over 24 sub-proofs
per FINAL epoch            55,058 leg + 2,944 recombination = 58,002 over 25 sub-proofs
```

**9.7% under the 63,393 budget**, with the recombination included — which §8.2.2
does not count.

## Three corrections

### 1. `MulBase` is cost-neutral, not a 4× routing obligation (§3)

§3 says an emitter that fails to detect the ext×base case "pays 4× for it",
comparing against a hand-lowering as three base multiplies plus a repack.
**That comparison has no basis.** Read against `chips::xalu`: `Mul` and
`MulBase` are selectors on the SAME chip at the same width, so both are one row;
and the `B` operand is received through `ext_token` on every selector, so a
base-valued word `(c, 0, 0, 0)` is already a legal `Mul` operand yielding the
same product. Nobody would lower an ext×base multiply by hand when `ExtOp::Mul`
exists, so the 4× alternative is not a lowering anyone would reach for.

Detection is therefore **optional, not obligatory, and worth zero rows**. The
emitter does it anyway — it states the intent, and the chip's constraints 18–19
pin the operand's high lanes to zero on those rows — but a reader sizing this leg
should not expect a saving, and 5,041 is a count of `MulBase` rows rather than of
rows avoided.

### 2. Fusion saves 9,040, not 9,069 (§5)

Measured by construction: the emitter writes exactly 9,040 `MulAdd` rows.

**9,113** `(Add, Mul)` operand pairs individually satisfy the single-consumer
guard, but an `Add` carries ONE multiply, so a sum whose two operands are both
single-consumer products can absorb only one of them. There are 73 such sums.
The achievable saving is bounded by the pairs, not equal to them — 9,069 sits
between the two counts and I could not reproduce it under either rule.

The consequence for §0's arithmetic is small (29 rows on 57,583) but the shape of
the claim matters: a candidate count is an upper bound on a fusion saving, never
the saving itself.

### 3. The "3 dead nodes" cost no rows, and a reachability count is not comparable

§4.3's three fanout-0 nodes reproduce exactly under its own local measure — but
**none of them is arithmetic**. Across all 28 AIRs there are **zero** arithmetic
nodes with local fanout 0, so dead-code elimination saves **zero rows** on any
production artifact. The emitter does DCE anyway, and its test has to INJECT an
unreachable node to exercise the path, because the capture front-end does not
produce one.

Separately, **2,376 nodes are unreachable from any root** once one notices that a
folded constant does not keep its operands alive. Every one of them is itself a
constant, and the census already counts them under `fold` — ECSM 1,186 of its
1,513, ECDAS 1,190 of its 1,262. **Anyone adding §8.1's `fold` column to a
reachability-based dead count will double-count exactly those 2,376.** The
emitter's report keeps them in a separate `unreached_const` field for this
reason; it was the one place the implementation and the design first disagreed,
and the disagreement was in the bookkeeping, not in the program.

## Two deliberate departures from the spec

- **Two extra rows per sub-proof for reciprocal guards.** §6 divides the β-fold
  by `Z` and §7 divides the boundary numerator by `ζ − p`. Under the machine's
  `0/0 = 1` convention a direct divide silently returns 1 when the denominator
  AND numerator vanish, so a `ζ` on the trace domain would be accepted. The
  emitter inverts against the interned one instead (`1/0` has no satisfying
  assignment) and multiplies. §6.1's ≈22–26 row zerofier block measures at ≈24–28.
  The out-of-domain sampler already excludes such a `ζ`, but the sampler is not
  in this leg, and per method rule 5 a deferral's safety argument is itself a
  claim — the guard costs two rows and removes the need for one.
- **Boundary constraints are an explicit shape parameter**, not read from the
  artifact, because `AIR::boundary_constraints` is a function of the public
  inputs and the artifact deliberately excludes it. §7 is right that production
  has exactly one per interacting AIR (`acc[0] = 0` on the last aux column, at
  `g^0`), and the emitter carries the general `{col, point, value}` form anyway.

## What the implementation cannot tell you

The differential runs every one of the 28 AIRs against `eval_program_verifier`,
and the composition check runs against a real proof of L2G_MEMORY — but only
L2G_MEMORY is checked against a real proof, and only its trace length, part count
and single boundary constraint are exercised end to end. Nothing here says a
27-AIR epoch assembles correctly; that is the assembly leg's question, and the
per-epoch figures above are compositions of per-AIR measurements, not a run.
