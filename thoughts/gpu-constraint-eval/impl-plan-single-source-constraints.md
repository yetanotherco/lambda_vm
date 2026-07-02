# Implementation plan: single-source constraints (Phase 2.5, PRs A + B)

**Audience: the implementing agent.** Self-contained: read this + the referenced code
and you can build it without the design discussion that produced it. All file:line
refs verified on branch `spike/constraint-ir-builder-part2` (PR #757, head of the
constraint-IR stack: #739 = Part 1, #757 = Part 2).

**Companion docs** (context, not required reading to implement):
- `survey-constraint-frontends.md` — how Plonky3/OpenVM/SP1/risc0/zisk/airbender do this.
- `roadmap.md` — the overall GPU-constraint-eval program (this plan = its Phase 2.5).
- `plan-generic-ir-fable.md` — superseded by this file.

---

## 1. Goal and non-goals

**Goal.** Every transition constraint is defined exactly **once**, and from that
single definition we derive: (a) the compiled CPU prover evaluation, (b) the
verifier evaluation at the OOD point (identical code path in the recursion guest),
(c) the flat IR (`ConstraintProgram`) that the CPU interpreter and the future GPU
kernel consume. Today every constraint is written **twice** (`evaluate` +
`capture`); that duplication is the thing being deleted.

**Non-goals / hard constraints:**
- The stark engine **stays generic** over `<F: IsSubFieldOf<E>, E>`. Do not
  concretize the prover/verifier to Goldilocks.
- **Do not** make the interpreter the CPU proving path. This was measured
  (2026-07-01, ABBA on the bench server, ethrex 20-transfer fixture,
  `spike/constraint-ir-default-on` vs `spike/constraint-ir-builder-part2`):
  interpreted constraint eval costs **~9% total prove time** (pairs: −8.54%,
  −9.36%). The compiled folder path is mandatory.
- No DSL, no codegen, no checked-in generated files.
- Protocol semantics are untouchable: same constraints, same zerofier structure
  (per-constraint period/offset/exemptions + grouped evaluation), same transcript.
  Proofs must be **bit-for-bit identical** before/after (golden-proof gate below).
- The recursion guest (verifier compiled to RISC-V) must never hash and never
  interpret: its constraint evaluation is the compiled folder. Capture (which
  hash-conses) must not run on the guest path — see §4.6.

## 2. Settled decisions (do not relitigate)

| Decision | Choice | Why (evidence) |
|---|---|---|
| Single-source mechanism | One generic body per **table**, `fn eval<B: ConstraintBuilder>` | The Plonky3/SP1/OpenVM `Air<AB>` pattern (survey §1-3); object-safety handled by monomorphizing inside concrete impls, so `&dyn AIR` keeps working |
| CPU prover path | Compiled `EvalFolder` (re-run body per row) | Bench: interpreter = −9% prove time; p3+SP1 do the same |
| GPU path | Capture → flat `ConstraintProgram<F,E>` → device interpreter (roadmap Phase 4) | The whole point of the program; OpenVM/zisk-validated |
| Per-constraint objects | **Deleted.** Constraints are expressions emitted by the table body; metadata is plain data (`Vec<ConstraintMeta>`) | Simplest model; removes `Vec<Box<dyn TransitionConstraintEvaluator>>`, the adapter, `boxed()`, per-constraint structs |
| Constants in the IR | Side tables (`Op::ConstBase(u32)` → `base_consts: Vec<FieldElement<F>>`) | Keeps `Op` POD `Copy+Eq+Hash` with zero bounds on F; `IsField::BaseType` has no Eq/Hash (`crypto/math/src/field/traits.rs:101`), and `FieldElement`'s derived-Hash/manual-Eq disagree on non-canonical reps (`element.rs:47`, `goldilocks.rs:411`) — inline constants would poison the CSE map's key type |
| "FieldConsts" associated consts (roadmap §2.5 step 1) | **Not needed** | Every residue-using constraint is concretely `<GoldilocksField, GoldilocksExtension>` (`prover/src/constraints/templates.rs:81,543`, `cpu.rs:112-749`); field-generic code (lookup.rs) uses only structural u64/i64 constants that `FieldElement::<F>::from` handles for any field |
| `degree()` | Stays **declared**, in `ConstraintMeta`; host-side test asserts declared == measured-from-IR | Measuring requires capture; capture must not run in the guest (verifier needs `composition_poly_degree_bound`, `lookup.rs:1006-1020`) |
| CSE / hashing | Only in the flatten step (existing `IrBuilder` hash-consing), host-side, once per AIR, lazily | p3 doesn't CSE at all; OpenVM only Arc-identity. Guest never flattens |
| Emission order | Explicit `constraint_idx` everywhere (`emit_*` takes idx; meta is idx-ordered) | Order/index alignment is load-bearing in every surveyed system; we keep it explicit + debug-assert completeness |

## 3. Architecture (end state)

```
per table (e.g. eq.rs):
  EqConstraints (small struct: nothing or col config)
    ├─ fn meta(&self)  -> Vec<ConstraintMeta>          // idx-ordered metadata, plain data
    └─ fn eval<B: ConstraintBuilder<F,E>>(&self, b)    // THE single body: emits every constraint

framework (lookup.rs): LogUp constraints emitted the same way, generated from the
  interaction config (single definitions = today's capture helpers, generalized)

three interpretations of the same body:
  ProverEvalFolder   Expr = FieldElement<F>  → per LDE row, compiled   (CPU prover hot path)
  VerifierEvalFolder Expr = FieldElement<E>  → once at OOD point       (verifier + recursion guest)
  CaptureBuilder     Expr = owned expr tree  → once at setup, host     (flatten → ConstraintProgram<F,E>)
                                                                        ├─ CPU interpreter (tests / GPU parity)
                                                                        └─ GPU lowering (Phase 4)

engine (unchanged shape): &dyn AIR; AirWithBuses stores the table's ConstraintSet +
  Vec<ConstraintMeta>; zerofier machinery reads meta; one virtual call per row per table.
```

## 4. PR A — generic IR

Self-contained first PR. Makes `constraint_ir` generic so `CaptureBuilder` can
target it for any field and the `unsafe` bridge dies. **Behavior identical** —
gates are bit-for-bit.

### 4A.1 `crypto/stark/src/constraint_ir/ir.rs`
- Rename `Dim::{D1, D3}` → `Dim::{Base, Ext}`.
- Replace `Op::Const1(u64)` / `Op::Const3([u64;3])` with `Op::ConstBase(u32)` /
  `Op::ConstExt(u32)` — indices into new fields on the program:
  ```rust
  pub struct ConstraintProgram<F: IsField = GoldilocksField,
                               E: IsField = Degree3GoldilocksExtensionField> {
      pub nodes: Vec<Op>,          // Op stays Copy+Eq+Hash (u32 payloads only)
      pub dims: Vec<Dim>,
      pub base_consts: Vec<FieldElement<F>>,
      pub ext_consts: Vec<FieldElement<E>>,
      pub roots: Vec<u32>,
      pub num_base: usize,
      pub complete: bool,
  }
  ```
- Bounds: `F: IsField, E: IsField` only. Default type params = Goldilocks tower so
  existing concrete code compiles unchanged during migration.
- Verified: `Const1`/`Const3`/`const_ext` have **zero** users outside
  `constraint_ir/` (including tests), so the const redesign is module-contained.

### 4A.2 `crypto/stark/src/constraint_ir/builder.rs`
- `IrBuilder<F: IsField = Goldilocks…, E: IsField = …3>`; fields gain
  `base_consts`/`ext_consts`; delete `const_cache: HashMap<u64,u32>`.
- `const_base(v: u64)` / `const_signed(v: i64)`: `FieldElement::<F>::from(v)`
  (generic `From<i64>` exists, `element.rs:149`), then intern.
- `intern_base(fe)`: linear scan `base_consts.iter().position(|c| c == &fe)`
  (PartialEq → `F::eq`, canonicalizing — exact dedup, no Hash needed; tables are
  tiny and this runs once at setup), push if absent, then
  `push(Op::ConstBase(idx), Dim::Base)` (the `(Op, Dim)` cse map is unchanged —
  `Op` is still POD). Same for `intern_ext`.
- `const_ext` signature becomes `const_ext(v: FieldElement<E>)`.
- Keep id-0 convention: `new()` interns zero first → node 0 = `ConstBase(0)`,
  `base_consts[0] = 0`. Node ids are assigned in first-use order exactly as
  today ⇒ **node counts in `prover/src/tests/constraint_ir_tests.rs` must not
  change** (product_zero 4, is_bit_uncond 5, is_bit_cond 7, add_carry_0 14,
  add_carry_1 21; full-table: CPU 616 nodes / EQ 142).

### 4A.3 trait plumbing (temporary — PR B replaces it)
- `Capture<F: IsField = …, E: IsField = …>` in `constraint_ir/mod.rs:43`;
  `TransitionConstraintEvaluator::capture(&self, b: &mut IrBuilder<F, E>)`
  (`constraints/transition.rs:40`) — object-safe (F,E are trait params; precedent:
  `evaluate_verifier` already takes `&TransitionEvaluationContext<F, E>`).
  With the default type params, the ~35 concrete `impl Capture for …` in the
  prover crate compile **unchanged**. `AIR::constraint_program()`
  (`traits.rs:330`) returns `ConstraintProgram<Self::Field, Self::FieldExtension>`.
- lookup.rs capture helpers (`capture_multiplicity` etc., `lookup.rs:1733-1997`)
  and the two LogUp `capture` overrides (`lookup.rs:2130,2336`) gain `<F, E>`.

### 4A.4 `crypto/stark/src/constraint_ir/interp.rs` + delete the bridge
- `Value<F, E> { Base(FieldElement<F>), Ext(FieldElement<E>) }` — `Clone`, not
  `Copy` (not provable for generic F); use `.clone()`; for Goldilocks these
  compile to register copies.
- `eval_program` / `eval_program_verifier` / `eval_program_base` become generic
  `<F: IsSubFieldOf<E>, E: IsField>` (add `IsFFTField`/`'static`/`Send+Sync` only
  if call sites force it). Const resolution reads the side tables (no more
  per-row `Fp::from` re-reduction).
- **Delete `constraint_ir/bridge.rs`** (99 lines, all the module's `unsafe`).
- `constraints/evaluator.rs`: field becomes
  `Option<ConstraintProgram<Field, FieldExtension>>` (line 30); the hook at
  lines 110-125 loses the `ran` fallback boolean — call `eval_program` directly
  when the program is `Some`. The `complete:false → None` guard at :239-243 stays.
- `verifier.rs:254-274`: call `eval_program_verifier` directly; keep the
  `prog.complete` boxed fallback.

### 4A.5 PR A gates
- `cargo test -p lambda-vm-prover constraint_ir_tests -- --nocapture` — node
  counts + full-table prover/verifier diff gates bit-identical.
- `cargo test --release -p lambda-vm-prover --features stark/constraint-ir`
  (incl. `test_prove_elfs_*`) and the default suite; `cargo test -p stark`.
- New test: capture+interpret over a non-Goldilocks tower (Stark252, `E = F`;
  reflexive `IsSubFieldOf` impl at `traits.rs:28`) — proves the genericity.
- `grep -rn unsafe crypto/stark/src/constraint_ir/` → empty.
- `cargo fmt` + `cargo clippy` clean (required before every push).

## 5. PR B — single-source constraints

### 5.1 Step 0 — readability spike  ✅ DONE (2026-07-01)

**Outcome: operator style wins; the trait surface in §5.2 is PINNED.** Reference
implementation (concrete Goldilocks): branch `spike/constraint-builder-step0`,
commit `57ee832e` — `crypto/stark/src/constraints/builder.rs` +
`prover/src/tests/constraint_builder_spike.rs`. All differential gates passed
first try (EqXor / IsBit / Add-pair: ProverEvalFolder == old `evaluate::<Gl,Gl3>`,
VerifierEvalFolder == old `evaluate::<Gl3,Gl3>`, capture→flatten→interpret == old
evaluate, 1000 rows each; tree-measured degree == declared). Ergonomics: clone
noise 2/1/2 per body (only for genuine reuse; Rc clone = pointer bump), **zero
`.into()`** (leaves return `Expr` directly — no `Var`/`Expr` split, dodging SP1's
noise), zero borrow-checker fights. Converted EqXor body, verbatim:

```rust
let res = b.main(0, eq_cols::RES);
let eq = b.main(0, eq_cols::EQ);
let invert = b.main(0, eq_cols::INVERT);
let two = b.const_base(2);
b.emit_base(idx, res - (eq.clone() + invert.clone() - two * eq * invert));
```

Bonus finding: emitting the Add lo/hi pair from ONE template function lets the
IrBuilder hash-consing share the whole `carry_0` subtree across the pair — 24
nodes vs 14+21 for the old separate per-constraint captures. Per-table programs
will therefore be smaller than the sum of the Phase-0 per-constraint node counts;
**PR 2 gates compare folder-vs-interpreter values, never node counts.**

### 5.2 Trait surface — PINNED by the spike (`crypto/stark/src/constraints/builder.rs`)

Generic lift of the spike's concrete form (add `<F: IsField, E: IsField>`,
`FieldElement<F>`/`<E>` for `Fp`/`Fp3`; the verifier folder's const embed needs
`F: IsSubFieldOf<E>`):

```rust
pub trait ExprOps<Ext>: Sized + Clone
    + Add<Self, Output = Self> + Sub<Self, Output = Self>
    + Mul<Self, Output = Self> + Neg<Output = Self>
    + Add<Ext, Output = Ext> + Sub<Ext, Output = Ext> + Mul<Ext, Output = Ext> {}
// + blanket impl for any type meeting the bounds

pub trait ExtExprOps: Sized + Clone
    + Add<Self, Output = Self> + Sub<Self, Output = Self>
    + Mul<Self, Output = Self> + Neg<Output = Self> {}

pub trait ConstraintBuilder<F: IsField, E: IsField> {
    type Expr: ExprOps<Self::ExprE>;
    type ExprE: ExtExprOps;
    fn main(&self, offset: usize, col: usize) -> Self::Expr;
    fn aux(&self, offset: usize, col: usize) -> Self::ExprE;
    fn periodic(&self, idx: usize) -> Self::Expr;
    fn challenge(&self, idx: usize) -> Self::ExprE;          // rap_challenges[idx]
    fn alpha_pow(&self, idx: usize) -> Self::ExprE;          // logup_alpha_powers[idx]
    fn table_offset(&self) -> Self::ExprE;                   // logup L/N
    fn const_base(&self, v: u64) -> Self::Expr;              // ONLY constant path
    fn const_signed(&self, v: i64) -> Self::Expr;
    fn one(&self) -> Self::Expr { self.const_base(1) }       // keep these defaults
    fn zero(&self) -> Self::Expr { self.const_base(0) }
    fn emit_base(&mut self, constraint_idx: usize, e: Self::Expr);
    fn emit_ext(&mut self, constraint_idx: usize, e: Self::ExprE);
}
```

Spike corrections to the original sketch (binding for PR 1b — do not deviate):
- The alias shape is `Expr: ExprOps<Self::ExprE>` — cross-field ops live on the
  **base** side with base always the LEFT operand (the field tower only
  implements subfield∘superfield); `ExtExprOps` takes **no** type params.
- **No `From<FieldElement<F>>` bound on `Expr`** — wrong for `VerifierEvalFolder`
  where `Expr = FieldElement<E>`; `const_base`/`const_signed` are the only
  constant path (verifier folder: `FieldElement::<F>::from(v).to_extension()`).
- `CaptureBuilder::finish(num_base)` returns
  `(ConstraintProgram<F, E>, Vec<(usize, usize)>)` — per-root (idx, measured
  degree); this IS the degree-measurement API for gate §5.9.2.
- Concrete folder leaves use `*` derefs (clippy `clone_on_copy`); generic folders
  use `.clone()` (no lint fires on generics). The capture tree's `Mul` needs
  `#[allow(clippy::suspicious_arithmetic_impl)]` (degree = sum of operands).

```rust
/// One table's constraints: metadata + THE single body.
pub trait ConstraintSet<F: IsField, E: IsField>: Send + Sync {
    fn meta(&self) -> Vec<ConstraintMeta>;                   // idx-ordered
    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B);   // emits every constraint
}

pub struct ConstraintMeta {
    pub constraint_idx: usize,
    pub kind: RootKind,               // Base | Ext; Base entries MUST be a prefix
    pub degree: usize,                // declared; asserted == measured (test, §5.8)
    pub period: usize,                // default 1
    pub offset: usize,                // default 0
    pub exemptions_period: Option<usize>,
    pub periodic_exemptions_offset: Option<usize>,
    pub end_exemptions: usize,        // default 0
}
```
Invariants (debug-assert in the folders and at AIR construction): meta is dense
and idx-ordered; `kind == Base` entries form a prefix (this IS `num_base`, matching
the existing convention, `traits.rs:239-243`); `eval` emits **every** idx exactly
once (folders track a seen-bitset in debug builds).

### 5.3 The three builder implementations (framework, written once)

1. **`ProverEvalFolder<'a, F, E>`** — `Expr = FieldElement<F>`,
   `ExprE = FieldElement<E>`. Constructed per row from the Prover
   `TransitionEvaluationContext` (`traits.rs:73-95`) + output slices; leaves read
   the frame exactly as `interp.rs:176-192` does today
   (`frame.get_evaluation_step(offset).get_main_evaluation_element(0, col)`);
   `emit_base` writes `base_evals[idx]`, `emit_ext` writes `ext_evals[idx]`.
   All ops are plain `FieldElement` arithmetic — after inlining this is the same
   machine code as today's `evaluate` bodies. **This is the CPU hot path.**
2. **`VerifierEvalFolder<'a, F, E>`** — `Expr = FieldElement<E>` (the OOD frame is
   `Frame<E, E>`), `ExprE = FieldElement<E>`; `const_base` embeds via
   `FieldElement::<F>::from(v).to_extension::<E>()`; `emit_base` promotes and
   writes `ext_evals[idx]` (mirrors `TransitionConstraintAdapter`,
   `constraints/transition.rs:459`). Runs once at the OOD point. **This exact
   monomorphization, compiled into the guest binary, is the recursion-guest
   path — no capture, no hashing, no interpretation in-circuit.**
3. **`CaptureBuilder<F, E>`** — `Expr`/`ExprE` = small owned tree
   (`enum IrExpr { Leaf(...), Add(Rc<IrExpr>, Rc<IrExpr>), … }`, each node also
   storing an eagerly-computed `degree` — leaf var 1, const 0, mul sums, add/sub
   max, p3's `degree_multiple`). Operators allocate nodes — **no arena, no
   RefCell, no thread-local, no hashing during capture**. `emit_*` flattens the
   finished tree into the PR A `IrBuilder<F, E>` (recursive walk; hash-consing
   there = structural CSE, host-side) and records the root + measured degree.
   Produces `ConstraintProgram<F, E>` + measured degrees.

### 5.4 Table conversion (the bulk — mechanical)

Per table (17 production tables in `prover/src/tables/*.rs`): replace the
`*_constraints(idx_start) -> (Vec<Box<dyn …>>, usize)` function with a
`XxxConstraints` struct implementing `ConstraintSet`. Recipe, using EQ
(`prover/src/tables/eq.rs:253-345`) as the model:

```rust
pub struct EqConstraints;   // holds col config only if the table needs it

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for EqConstraints {
    fn meta(&self) -> Vec<ConstraintMeta> {
        let mut m = templates::add_pair_meta(0);            // idx 0,1: b + diff = a
        m.extend(templates::is_bit_meta(2, 1));             // idx 2: IS_BIT(invert)
        m.push(ConstraintMeta::base(3, /*degree*/ 2));      // idx 3: res = eq XOR invert
        m
    }
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        templates::emit_add_pair(b, 0, vec![], AddOperand::dword(cols::B_0),
            AddOperand::from_dword_hl(cols::DIFF_0), AddOperand::dword(cols::A_0));
        templates::emit_is_bit(b, 2, cols::INVERT, None);
        let (res, eq, invert) = (b.main(0, cols::RES), b.main(0, cols::EQ), b.main(0, cols::INVERT));
        let two = b.const_base(2);
        b.emit_base(3, res - (eq + invert - two * eq * invert));
    }
}
```

- **Templates become functions**: `AddConstraint`/`IsBitConstraint`/
  `ProductZeroConstraint`/the cpu.rs constraint structs
  (`prover/src/constraints/{templates,cpu}.rs`) turn into `emit_*` +
  `*_meta` function pairs in the same files. Their existing `capture` bodies are
  the starting point for `emit_*` (they're already builder-call style); their
  `evaluate` operator text is the readability reference. Delete both old bodies
  and the structs' trait impls when each table converts.
- The multi-kind mega-constraints (Dvrm 11 kinds / Cpu32 8 / Shift 7 / Lt·Load·Mul 6)
  convert the same way — their `compute()` loops are statically bounded and
  already unrolled in the existing `capture` impls.
- Index bookkeeping: the old `idx_start` threading disappears; each table's meta
  is self-contained 0..n. (LogUp indices are appended by the framework — §5.5.)

### 5.5 LogUp (framework side, `crypto/stark/src/lookup.rs`)

- Reduce `LookupBatchedTermConstraint` / `LookupAccumulatedConstraint` to plain
  config data (a `LogUpLayout`: committed pairs, absorbed interactions,
  `term_column_idx`s, `acc_column_idx`, `num_term_columns`) — this is exactly
  what `AirWithBuses::new` already computes at `lookup.rs:858-880`
  (`split_interactions`, absorbed slice).
- The single definitions are the **existing capture helpers**
  (`capture_multiplicity`, `capture_linear_terms`, `capture_packing_fingerprint`,
  `capture_fingerprint`, `lookup.rs:1733-1997`) generalized over
  `B: ConstraintBuilder<F, E>`, plus two `emit_logup_batched_term` /
  `emit_logup_accumulated` functions transcribed from the current `capture`
  overrides (`lookup.rs:2130`, `:2336` — including the 1-absorbed vs 2-absorbed
  branches and the `aux(1, col)` next-row reads).
- **Delete** the `evaluate_*` twins (`evaluate_batched_term_constraint`,
  `evaluate_accumulated_constraint`) and the two structs' boxed-trait impls
  (`lookup.rs:2039-2196`, `:2197+`).
- Framework meta: reproduce the current structs' `period/offset/end_exemptions`
  answers exactly (read them off the current impls before deleting).
- The runtime `BusValue::Linear` zero-skip optimization is already intentionally
  not reproduced in capture (value-preserving; see the honesty note at
  `lookup.rs:1725-1729`) — with one body this asymmetry disappears entirely;
  verify the golden-proof gate still passes (it must: the skip is value-neutral).

### 5.6 Engine rewiring

- **`AirWithBuses`** (`lookup.rs:805-830`): gains a type param
  `CS: ConstraintSet<F, E>`; field `transition_constraints: Vec<Box<dyn …>>` is
  replaced by `constraint_set: CS`, `logup: LogUpLayout`, and
  `meta: Vec<ConstraintMeta>` (= `cs.meta()` + framework-appended LogUp meta;
  compute `num_base` from the Base-prefix). `new` (`lookup.rs:849`) takes the
  `CS` value instead of the boxed vec; everything else it computes stays.
- **`AIR` trait** (`crypto/stark/src/traits.rs`):
  - `transition_constraints()` (`:315-317`) — **deleted**.
  - New: `fn constraints_meta(&self) -> &[ConstraintMeta]`.
  - `compute_transition_prover` (`:255`) / `compute_transition` (`:224`) lose
    their boxed-loop defaults and become required methods. `AirWithBuses`
    implements them as one-liners into free generic helpers:
    `run_transition_prover(&self.constraint_set, &self.logup, ctx, base, ext)`
    (constructs `ProverEvalFolder`, runs `cs.eval` + `emit_logup_*`); same for
    the verifier folder and for `constraint_program()` (capture + flatten,
    **lazily, cached in a `OnceLock` — the guest never calls it**, see §5.7).
  - `composition_poly_degree_bound` (`lookup.rs:1006-1020`): max over
    `meta.degree` instead of `c.degree()`.
- **Zerofier machinery**: `transition_zerofier_evaluations_grouped`
  (`traits.rs:343-370`) reads `ZerofierGroupKey` fields from
  `constraints_meta()`; the big default methods
  `zerofier_evaluations_on_extended_domain` / `evaluate_zerofier` /
  `end_exemptions_*` (`constraints/transition.rs:127-337`) become free functions
  of `(&ConstraintMeta, &Domain | z)` in a new `constraints/zerofier.rs` — they
  only ever consumed the metadata getters (verified). Bodies move verbatim.
- **`ConstraintEvaluator`** (`constraints/evaluator.rs`): unchanged flow; the
  `eval_row` hook calls `air.compute_transition_prover(&ctx, base_buf, transition_buf)`
  as today (now one virtual call into the monomorphized folder run instead of 33).
  The `constraint-ir` feature hook from PR A stays as the interpreter reference
  path for tests/GPU parity — **off by default** (bench: −9%).
- **Delete**: `TransitionConstraintEvaluator`, `TransitionConstraintAdapter`,
  `TransitionConstraint` (old signature), `Capture`, `boxed()` —
  all of `constraints/transition.rs` except what moves to `zerofier.rs`.

### 5.7 Guest-safety rule (recursion)

The verifier path must run: AIR construction (no capture — `constraint_program`
is lazy and only the prover/GPU/tests force it) → `VerifierEvalFolder` at the OOD
point. Add a test or debug assertion that the verify path never constructs an
`IrBuilder` (e.g. feature-gate a counter, or simply grep-audit + document).
Degree is read from declared meta, so `composition_poly_degree_bound` needs no
capture. This preserves the no-HashMap-in-guest rule with zero special-casing.

### 5.8 Examples + tests migration

- The 13 example AIRs (`crypto/stark/src/examples/*.rs`) and
  `tests/transition_tests.rs` implement `TransitionConstraintEvaluator` directly
  today; each becomes a `ConstraintSet` impl (bodies are 1-3 trivial constraints)
  + the three forwarding one-liners on their `AIR` impls. The
  `complete: false` fallback machinery (`ConstraintProgram::complete`,
  `IrBuilder::mark_unsupported`) can then be **retired** — every AIR captures.
- `prover/src/tests/constraint_ir_tests.rs`: the per-constraint Phase-0 diff
  tests convert to compare `ProverEvalFolder` output vs interpreted program on
  random rows (same assertion, derived from one body now). Full-table gates
  unchanged in spirit: folder vs interpreter vs (during migration only) the old
  boxed path.

### 5.9 PR B gates — all must pass

0. **Pre-flight, from the PR 1 fresh-eyes review (do these FIRST, before any
   conversion):**
   - `num_base` has two independent sources of truth — the interpreter routes by
     `c < prog.num_base` (panics via `.as_base()` on mismatch) while the folders
     route by which `emit_*` the body calls, and `CaptureBuilder::finish(num_base)`
     takes it as a bare argument. Everywhere PR 2 wires these, the value MUST be
     `num_base_from_meta(&meta)`, and add a test asserting it equals the captured
     base-emit count (release-checked, not debug-only).
   - Extend the folder↔capture differential test to cover an `aux(1, col)`
     next-row read and a second alpha index — the real 1-/2-absorbed LogUp bodies
     use both and the PR 1 sample body covers neither.
1. **Golden proofs** (the transcription safety net — this replaces the oracle
   role the duplication accidentally provided): proofs are deterministic given
   trace+params. Before starting conversion, record proof-bytes hashes for a
   fixed ELF set (e.g. the `test_prove_elfs_*` inputs) on the pre-PR-B commit;
   assert identical hashes after. Any slip in any constraint body changes the
   composition polynomial and flips the hash.
2. **Degree assert**: for every table, measured degree (CaptureBuilder trees) ==
   declared `meta.degree`.
3. **Backend consistency**: folder vs interpreted `ConstraintProgram` on 1000
   random rows, every production table (extends the existing gate pattern).
4. Full suite: `cargo test --release -p lambda-vm-prover` (default) and with
   `--features stark/constraint-ir`; `cargo test -p stark`.
5. **ABBA sanity** on the bench server (expect ≈ 0, possibly small win from
   removing 33 virtual calls/row):
   `scripts/bench_abba.sh origin/<pr-b-branch> origin/spike/constraint-ir-builder-part2 20`.
6. `cargo fmt` + `cargo clippy` before every push. No AI attribution anywhere
   (commits, PR bodies) — repo rule.

## 6. Sequencing, branch mechanics & PR packaging

**The spike PRs (#737, #739, #757) are NOT merged and get closed** once the new
branches exist — the user wants human reviewers to see only the real design,
never the transitional scaffolding (bridge `unsafe`, `Capture`-alongside-
`evaluate` duplication, boxed adapter). Their branches stay in the remote as
provenance; close each with "superseded by the single-source constraints PRs;
code absorbed". **Work from their code, not their PRs**: develop on a branch cut
from `spike/constraint-ir-builder-part2` (it has the IR/interpreter/capture
bodies to absorb), but the PRs opened against `main` present the end state
fresh.

Ship as **two PRs against main**, both containing only end-state code:

- **PR 1 — framework** (≈ §4 + §5.2-5.3): `constraint_ir` module arriving
  *already generic* (main never sees a concrete-Goldilocks IR or a bridge) +
  `ConstraintBuilder` + `ProverEvalFolder`/`VerifierEvalFolder` +
  `CaptureBuilder` + `ConstraintMeta` + zerofier free functions. Not wired into
  production paths; fully exercised by its own tests (§4A.5 gates reshaped as
  folder-vs-interpreter, the non-Goldilocks-tower test, spike-derived node-count
  tests). Zero behavior change.
- **PR 2 — the switch** (≈ §5.4-5.9): all tables + LogUp converted,
  `AirWithBuses`/`AIR`/zerofier rewired, old trait machinery deleted, golden
  proofs byte-identical. Structure the commits per table group so the diff reads
  as old-`evaluate`-deleted next to new-body-added in each file.

Notes:
- The internal build order within the work branch can still follow §4 then §5
  (the PR A/PR B labels elsewhere in this doc = the work phases; PR 1/PR 2 = the
  review packaging). The golden-proof baseline hashes are taken on `main`
  immediately before PR 2's conversion starts.
- GPU work (roadmap Phase 4) consumes `ConstraintProgram<F,E>` — stable after
  PR 1 merges; it can proceed in parallel with PR 2.
- Housekeeping: close #737 now; close #739/#757 when PR 1 opens; delete the
  bench branch `spike/constraint-ir-default-on` after PR 2 lands (keep it until
  then for ABBA re-runs).

## 7. What NOT to do (guardrails)

- Do not interpret constraints in the CPU prover default path (−9%, measured).
- Do not put `FieldElement` values inside `Op` (breaks POD/CSE; §2 table).
- Do not introduce hashing, capture, or interpretation into the verifier path
  (recursion guest). `VerifierEvalFolder` only.
- Do not change constraint semantics, indexing, zerofier structure, `num_base`
  ordering, or anything transcript-visible — golden proofs must hold.
- Do not add a `degree`-measuring pass to the verifier; declared meta + host test.
- Do not build packed/SIMD folders, register allocation, or codec work now —
  that's roadmap Phase 6, gated on GPU profiles.

## 8. Current-code map (for orientation)

| What | Where (verified) |
|---|---|
| IR + interpreter + builder + bridge | `crypto/stark/src/constraint_ir/{ir,interp,builder,bridge,mod}.rs` (758 lines total) |
| Boxed constraint trait + adapter + zerofier defaults | `crypto/stark/src/constraints/transition.rs` |
| Prover eval loop + IR hook | `crypto/stark/src/constraints/evaluator.rs:89-160` |
| Verifier OOD eval + IR hook | `crypto/stark/src/verifier.rs:241-274` |
| AIR trait (compute_transition*, zerofier grouping, constraint_program) | `crypto/stark/src/traits.rs:224-336,343-370` |
| AirWithBuses (the one production AIR) + LogUp constraints + capture helpers | `crypto/stark/src/lookup.rs:805+,965+,1733-2400` |
| Constraint templates + CPU constraints (evaluate/capture pairs) | `prover/src/constraints/{templates,cpu}.rs` |
| Table constraint builders (eq, lt, mul, dvrm, shift, …) | `prover/src/tables/*.rs` (e.g. `eq.rs:253-345`) |
| Existing diff-test gates | `prover/src/tests/constraint_ir_tests.rs` |
| Goldilocks residue constants | `prover/src/tables/types.rs:387-423` |
| Example AIRs (to migrate) | `crypto/stark/src/examples/*.rs` (13 files) |
| Bench harness | `scripts/bench_abba.sh` (runs on the bench server only) |
