# Design: lowering a `ConstraintArtifact` to LFM instructions

Design (α) from `lfm-design.md` §3 — the constraint-evaluation leg of the epoch
verifier. Written by the phase0 agent 2026-07-30 against
`feat/phase0-constraint-ir`. **Design only; no semantics touched.**

Every number below is measured by `constraint_op_census` in
`prover/src/tests/constraint_artifact_tests.rs`, which is a standing instrument —
run it, do not trust this file's copy of the numbers after the constraints change.

---

## 0. Headline

**The budget holds.** The design doc claimed ≈69K instructions for the constraint
leg at 25 AIRs. Measured at **28** AIRs (the three continuation tables included):

```
constraint-leg instructions   64,842
+ quotient recombination       2,150
= total                       66,992      (≈58K after MulAdd fusion)
```

Two corrections to how that number should be read, both material:

1. **The IR's `dim` tags are the wrong split to budget against** — they describe
   the prover, and the machine runs the verifier. See §3. Budgeting from the
   declared dims would understate extension traffic by 14×.
2. **66,992 is per distinct AIR, not per epoch.** An epoch evaluates the
   constraint leg once per SUB-PROOF, and chunking gives a table family several.
   See §8.2. This is the one place the design doc's figure is optimistic.

**Nothing in the IR is structurally inexpressible on a straight-line machine.**
In fact the IR is already in precisely the form the machine's soundness argument
demands — see §9, which is the most reassuring section here.

---

## 1. What the pass consumes and produces

Input: a `ConstraintArtifact` (the flat POD program, the per-constraint metadata,
the AIR shape, the composition degree multiplier). Output: a straight-line
`Vec<Instr<F>>` fragment plus the addresses of the per-AIR quotient contributions.

The pass runs **at registry-build time on the host**, so it may do arbitrary
host-side work — constant folding, peephole fusion, fanout analysis. None of it
costs machine instructions. What it emits is fixed program text whose digest the
registry pins.

---

## 2. Node → instruction mapping, and it is total

Eleven IR ops. Six are leaves that resolve to an address and emit nothing; five
are arithmetic.

| IR op | verify-time value | machine lowering | instrs |
|---|---|---|---|
| `Var{main,offset,row,col}` | ext | address in the OOD frame region | 0 |
| `RapChallenge{idx}` | ext | address in the challenge region | 0 |
| `AlphaPow{idx}` | ext | address in the alpha-power region | 0 |
| `TableOffset` | ext | address of the per-proof `L/N` | 0 |
| `ConstBase(idx)` | **base** | `Const{value:(c,0,0,0)}`, pooled | 1 (pooled) |
| `ConstExt(idx)` | ext | `Const{value:(c0,c1,c2,0)}`, pooled | 1 (pooled) |
| `Add(a,b)` | ext | `ExtAlu{Add}` | 1 |
| `Sub(a,b)` | ext | `ExtAlu{Sub}` | 1 |
| `Mul(a,b)` | ext | `ExtAlu{Mul}`, or `ExtAlu{MulBase}` when one operand is base | 1 |
| `Neg(a)` | ext | **`ExtAlu{Sub, a: ZERO, b: a}`** — see below | 1 |
| `Embed(a)` | ext | **nothing** — see below | 0 |

Two entries are not the obvious ones, and both are worth stating explicitly
because a reader would otherwise assume a 1:1 correspondence that does not exist.

### 2.1 `Op::Neg` has no instruction

`ExtOp` is `Add | Sub | Mul | Div | MulAdd | MulBase`. There is no unary negate.
`Neg(a)` lowers to `Sub` from a pooled zero, which every program already has
(`IrBuilder` reserves node id 0 as the base-field zero). So the mapping is total
— but only via that identity, and it is worth writing down rather than
rediscovering. Cost is unchanged at one instruction.

### 2.2 `Op::Embed` is free, and this is a payoff of the `[F;4]` word model

A base value is stored as `(v,0,0,0)`. Its extension embedding is `(v,0,0)` in
lanes 0–2 with lane 3 zero — **the same word**. So `Embed` is a pure address
alias: the emitter records that node `i` refers to node `a`'s address and emits
nothing.

Had the word been `[F;3]`, this would still hold. Had base and extension used
distinct representations, `Embed` would cost a real instruction on every
base→ext boundary — and §3 shows those boundaries are where nearly all the
traffic is.

**Measured: 0 `Embed` nodes across all 28 production AIRs.** The builder
documents the op as unreachable from the single-body capture path and the census
confirms it. So this arm is correctness-only today. Keep it: the arm is three
lines, and the day a constraint body calls `embed()` explicitly, a missing arm
is a panic in `ConstraintArtifact::program` or — worse — a silent wrong answer in
the CUDA kernel.

Likewise **0 `ConstExt` nodes**: no production constraint uses an extension
literal. The pooled-`Const` arm must still handle it.

---

## 3. The split that matters: prover dims are not machine dims

This is the correction I most want on the record.

`Dim` records what the **prover** computes. Its frame is base-field, so a
trace-only subexpression stays in the base field, and the IR duly tags 42,137 of
67,103 arithmetic nodes as `Dim::Base`.

The machine runs the **verifier's** evaluation at the OOD point. There the frame
holds only extension elements — `eval_program_verifier` resolves every `Var` to
`Value::Ext` regardless of `main`, because the verifier has openings, not trace
cells. Propagating that through `interp::binop`'s rule (base only when both
operands are base values *and* the declared dim is base), a node is base at
verify time **only if its entire subtree is constants**.

```
arithmetic nodes                67,103
  base by the IR's own dim      42,137   <- prover-side. NOT the machine's split.
  base at verify time            2,916   <- constant-only subtrees
  extension                     59,146
```

**A 14× discrepancy.** Anyone sizing the constraint leg from the IR's `dim`
column would conclude that most of the work is cheap base arithmetic. It is not:
94% of it is extension arithmetic.

Two consequences follow, and they pull in opposite directions.

**Bad news — `MulBase` applies less often than the IR suggests.** `MulBase` needs
a genuine base operand, and at verify time that means a folded constant. Measured
**5,041 MulBase-eligible multiplies**, against 9,413 if one (wrongly) counted
using prover dims. The `LFM_XALU` chip must still constrain its shared B-columns
to zero on `MulBase` rows so the received token matches a base writer's
(`SOUNDNESS.md` §4) — which is exactly why the operand has to be a real base
cell and cannot be a zero-high-lane extension value that merely looks like one.

**Good news — the 2,916 base nodes cost nothing at all.** A constant-only subtree
is a compile-time constant. The emitter folds it during the host-side pass and
interns the result in the pool. Those nodes emit zero instructions, which is why
they are excluded from the count in §0 rather than charged as `BaseAlu`.

---

## 4. Where operands come from

Four regions, and the distinction is a soundness boundary, not bookkeeping.

| region | source | authentication |
|---|---|---|
| OOD frame values (`Var`) | arena, hint-fed | the DEEP/opening leg — the machine hashes them into the openings it checks |
| challenges (`RapChallenge`) | transcript replay | computed in-machine by `LFM_HASH` rows; **never** hinted |
| alpha powers (`AlphaPow`) | derived from α | computed in-machine, once per proof |
| table offset (`TableOffset`) | derived `L/N` | computed in-machine, once per proof |
| constants | the program's own pool | program text; digest-pinned |

The arena rule (`lfm-design.md` §4, `SOUNDNESS.md` §5) says an arena value is
unconstrained by the reading chip and must be transitively authenticated by a
hash the machine performs. OOD frame values satisfy it because the DEEP leg
absorbs them; challenges must never come from an arena and do not.

**The constraint leg pays nothing marginal for any of this.** Measured **5,964
leaf nodes** across 28 AIRs, all of which are addresses of values other legs have
already materialized. That is the single biggest reason the leg is ~1% of the
program despite being 73,722 nodes.

### 4.1 Address assignment and `mult`

Addresses are dense and compiler-assigned in emission order. The emitter walks
the node list in index order and assigns address = base + i, skipping folded and
aliased nodes.

`mult(a)` — the statically known read count every write carries — is the node's
fanout in the IR DAG, plus one if the node is a constraint root (the quotient
recombination reads it). Measured **max fanout 1,632**, so the multiplicity
column must hold values into the low thousands; it is a field element, so this is
comfortable, but it is not the "small" value one might assume when sizing a
range check on it.

**Measured: 3 dead nodes (fanout 0) across all 28 AIRs.** Tiny, but the emitter
must DCE them rather than emit zero-multiplicity writes — the registrar's (M)
check is mult-equality, and a write nobody reads is at best noise in the digest.

---

## 5. Peephole: `MulAdd` fusion

`ExtAlu` carries `MulAdd` as a first-class op (Horner is the dominant pattern
elsewhere in the verifier). The IR has no `MulAdd` node — `CaptureBuilder` emits
`Mul` then `Add` — so the emitter can fuse `Add(Mul(a,b), c)` into one
instruction.

**The fusion is only valid when the `Mul` has exactly one consumer.** Hash-consing
means a shared `Mul` feeds several `Add`s; fusing it into each would recompute it
per consumer, turning a saving into a loss. A node that is a constraint root also
counts as a consumer — fusing it away would delete the value §6 needs.

**Measured: 9,069 fusable `Add` nodes**, taking the leg from 66,992 to **57,923**
— a 13.5% reduction for a host-side peephole with no chip work. Worth doing in
v0; it is a pass over a DAG the emitter already walks.

---

## 6. Zerofier and quotient recombination

The composition quotient is `H = Σ_c β^c · C_c / Z_c`.

### 6.1 Uniform zerofiers make this cheap, and the saving is large

**All 28 production AIRs emit through `RowDomain::ALL`** — measured; nothing
under `prover/src` calls `RowDomain::except_last`. So `end_exemptions = 0`
everywhere and every constraint of an AIR shares one zerofier, `Z = ζ^N − 1`,
depending only on the sub-proof's trace length.

Per sub-proof:

```
ζ^N          repeated squaring         log2(N) ExtAlu{Mul}   ≈ 20–24
ζ^N − 1      one Sub against pooled 1               1
1/Z          one ExtAlu{Div}                        1
                                        ────────────────────
                                        ≈ 22–26 instructions
```

And because `Z` is shared, the division factors out of the sum:
`H_air = (Σ_c β^c · C_c) / Z` — **one division per AIR, not per constraint**. The
sum is a Horner fold: one `ExtAlu{MulAdd}` per constraint, 2,150 total.

The saving is worth stating as a number, because it is the entire value of the
uniform-zerofier finding. The naive shape — what `main` does today, recomputing
`ζ^N` and a full extension inversion once per constraint (`lfm-design.md` §5.2
hygiene item 1) — costs `2,150 × ~24 ≈ 51,600` instructions. Doing it once per
AIR costs `28 × ~24 ≈ 672`. **≈50,900 instructions saved, ~44% of the unfused
leg.** The GPU path's uniform-zerofier precondition holding in fact rather than
by luck is the same fact, cashed differently.

Total recombination: 2,150 MulAdd + 28 Div + ~672 zerofier ≈ **2,850
instructions**, of which the §0 figure counts the 2,150 β-folds and folds the
rest into the per-sub-proof overhead.

### 6.2 If a constraint ever grows an exemption

The zerofier becomes `(ζ^N − 1) / (ζ − g^{N-e})·…` and the emitter must evaluate
one zerofier **per distinct `end_exemptions` value per AIR**, not per constraint.
The `ConstraintMeta` in the artifact carries exactly what is needed to group
them, and `transition_zerofier_evaluations_grouped` already keys its dedup on
that field host-side. Cost scales with the number of distinct values, which is
currently one.

---

## 7. Boundary constraint and the next-row read

**Boundary.** Every VM AIR uses `NullBoundaryConstraintBuilder`, so the only
boundary constraint is the framework's `acc[0] = 0` per chip. At ζ that is
`(P(ζ) − 0) / (ζ − 1)`: one Sub for the denominator, one Div, and the numerator
is the opened value itself. **≈3 instructions per sub-proof.** The accumulator's
circularity needs no boundary constraint of its own — it rides the plain
`ζ^N − 1` zerofier.

**The next-row read.** The machine has no rows, so "next row" is not a concept it
needs: `Op::Var{offset: 1, col}` is simply a different address, and the DEEP leg
supplies the `g·ζ` opening alongside the `ζ` ones. Zero extra instructions.

What makes this cheap is a shape fact worth re-verifying rather than assuming:
**every AIR declares exactly one next-row column** (the LogUp accumulator), or
none. That is not folklore — `ood_window_ir_tests` derives the true next-row read
set from the captured IR and asserts equality with the declaration, for all 28
AIRs, and that check is what stands between a correct verifier and one that
silently reconstructs an omitted `g·ζ` column as ZERO. It now covers the three
continuation AIRs, which it did not before this phase.

---

## 8. Measured instruction counts

### 8.1 Per AIR (28 tables, blowup 2 — the artifact is blowup-invariant)

`instr` = extension ALU + MulBase + pooled constants. Leaves are free;
constant-only subtrees fold at build time.

| table | nodes | leaves | const | fold | ext | mulbase | **instr** |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU | 600 | 75 | 32 | 4 | 417 | 72 | **521** |
| BITWISE | 158 | 33 | 10 | 3 | 106 | 6 | **122** |
| LT | 160 | 32 | 10 | 2 | 106 | 10 | **126** |
| SHIFT | 393 | 48 | 19 | 5 | 299 | 22 | **340** |
| EQ | 124 | 25 | 9 | 2 | 77 | 11 | **97** |
| BYTEWISE | 185 | 41 | 6 | 0 | 120 | 18 | **144** |
| STORE | 201 | 40 | 10 | 2 | 136 | 13 | **159** |
| CPU32 | 516 | 77 | 22 | 3 | 356 | 58 | **436** |
| MEMW | 552 | 89 | 12 | 3 | 429 | 19 | **460** |
| MEMW_A | 392 | 66 | 12 | 3 | 303 | 8 | **323** |
| MEMW_R | 202 | 41 | 6 | 2 | 129 | 24 | **159** |
| LOAD | 225 | 48 | 14 | 1 | 144 | 18 | **176** |
| DECODE | 35 | 15 | 2 | 0 | 18 | 0 | **20** |
| MUL | 388 | 48 | 18 | 2 | 276 | 44 | **338** |
| DVRM | 511 | 61 | 20 | 7 | 362 | 61 | **443** |
| BRANCH | 147 | 29 | 8 | 2 | 96 | 12 | **116** |
| HALT | 825 | 49 | 38 | 37 | 600 | 101 | **739** |
| COMMIT | 438 | 55 | 16 | 8 | 313 | 46 | **375** |
| PAGE | 63 | 16 | 4 | 2 | 34 | 7 | **45** |
| REGISTER | 49 | 15 | 3 | 2 | 23 | 6 | **32** |
| KECCAK | 3,997 | 784 | 37 | 30 | 2,960 | 186 | **3,183** |
| KECCAK_RND | 16,317 | 2,262 | 22 | 17 | 12,677 | 1,339 | **14,038** |
| KECCAK_RC | 51 | 23 | 2 | 0 | 26 | 0 | **28** |
| ECSM | 22,162 | 1,093 | 292 | 1,513 | 17,611 | 1,653 | **19,556** |
| ECDAS | 24,848 | 851 | 17 | 1,262 | 21,424 | 1,294 | **22,735** |
| L2G_GLOBAL | 47 | 15 | 4 | 1 | 24 | 3 | **31** |
| L2G_MEMORY | 93 | 21 | 6 | 1 | 60 | 5 | **71** |
| GLOBAL_MEMORY | 43 | 12 | 4 | 2 | 20 | 5 | **29** |
| **TOTAL** | **73,722** | **5,964** | **655** | **2,916** | **59,146** | **5,041** | **64,842** |

### 8.2 Two things that scale it — read this before using the total

**The leg is workload-shaped.** ECDAS + ECSM + KECCAK_RND = 56,329 instructions,
**86.9% of the total**. An epoch that does no elliptic-curve work has no
ECSM/ECDAS sub-proofs at all and drops 42,291 instructions (65%). Quoting a
single number for "the constraint leg" is therefore misleading in both
directions; it should be quoted per workload class.

**The total is per distinct AIR, not per epoch.** Each SUB-PROOF carries its own
trace and needs its own constraint evaluation, and an epoch has more sub-proofs
than 28: `T_epoch = table_counts.total()` (14 split-table families, **chunked**)
`+ 9 or 10 fixed + page_configs.len() + 1` (`lfm-target-shape.md`). So

```
constraint-leg instructions per epoch = Σ over sub-proofs  instr(that sub-proof's AIR)
```

A chunked family contributes its AIR's count once per chunk, and each touched
page contributes PAGE's 45. **This is the one place `lfm-design.md` §5.2 is
optimistic** — its 69K line reads as a per-epoch figure but is a per-distinct-AIR
figure. The correction is bounded and cheap for the small AIRs (PAGE at 45
instructions per page is nothing), but a family chunked k ways multiplies a
four-figure count by k. I do not have the chunk counts for a realistic workload,
so I am not giving a multiplier — flagging the formula instead.

### 8.3 Against the design doc's claim

| | design doc (25 AIRs) | measured (28 AIRs) |
|---|---:|---:|
| IR nodes | 73,539 | 73,722 |
| arithmetic ops | 66,982 | 67,103 |
| constraint-leg instr | ≈69K | 66,992 |
| after MulAdd fusion | — | 57,923 |

**The budget holds**, with three more AIRs, and the fusion peephole gives ~13.5%
headroom on top. The design doc's ≈1%-of-program framing survives — subject to
§8.2's per-sub-proof multiplier, which is the number that actually needs pinning
next.

---

## 9. Structural expressibility: nothing blocks

The lead asked me to flag anything a straight-line machine cannot express. There
is nothing — and the reason is stronger than "it happens to work".

- **The IR is a pure DAG with no control flow.** No branches, no loops, no
  data-dependent addressing. `ConstraintProgram` is a topologically ordered node
  list, which is what a straight-line program *is*.
- **`nodes[i]` references only `< i`.** This is the IR's own documented
  invariant, and `ConstraintArtifact::validate_self` enforces it. It is
  *identical* to the machine's acyclicity premise (A) — "operand address <
  destination address" (`SOUNDNESS.md` §2). A dense address assignment in node
  order satisfies (A) **by construction**, with no reordering pass and no
  verification burden beyond the check the artifact already runs.
- **Fanout is statically known**, so `mult` comes straight off the DAG. The
  machine's write-once model needs exactly this and the IR already has it.
- **No division in the constraint algebra.** `Op` has no `Div`. Division enters
  only at the zerofier and quotient step (§6), where it is a handful of
  instructions per sub-proof rather than per node.

The one genuine mismatch is the trivial one: `Op::Neg` has no instruction and
lowers to a subtract from zero (§2.1). That is a lowering detail, not a
structural obstacle.

Worth naming as a near-miss: the hash-consing that makes the IR compact is the
same property that makes naive `MulAdd` fusion unsound (§5). Shared
subexpressions are an asset for program length and a hazard for peepholes. Any
future fusion — not just `MulAdd` — needs the same single-consumer guard.

---

## 10. What I did not verify

- **Chunk counts per table family for a realistic workload**, hence no per-epoch
  multiplier in §8.2. This is now the most valuable missing number for the
  budget.
- **The `[F;4]` lane semantics for `MulBase`'s base operand** I took from
  `SOUNDNESS.md` §4 rather than from the chip's constraints. If `LFM_XALU`'s
  actual `MulBase` row shape differs, §3's MulBase count is still right but its
  cost claim (3 base multiplies, not 9) should be re-derived.
- **Instruction → trace-row cost.** `lfm-design.md` §1.3 gives `LFM_XALU` ~10
  main columns per op, but I have not confirmed one instruction is one row. Every
  count here is in INSTRUCTIONS; converting to rows or cells needs that factor.
- **Whether the emitter should share the constant pool across AIRs.** 655 pooled
  constants over 28 tables, and small integers (0, 1, 2, 2^8, 2^16, 2^24) surely
  recur; a program-wide pool would shrink it. Not measured, likely minor.
