# Plan: GPU-ready constraint evaluation via a "Symbolic field" capture

> **Status:** the CPU spike from this plan is **implemented** (PR #737, branch
> `spike/constraint-ir-symfield`). For the as-built state and the detailed,
> checkbox continuation plan, see **[`roadmap.md`](./roadmap.md)** — that is the
> execution / handoff doc. This file remains the full design rationale.

**Approach:** keep the ~29 constraint bodies UNCHANGED; introduce a recording
field type `SymField`/`SymExt` whose field operations build an expression graph
instead of computing. Run each constraint's existing generic
`evaluate::<SymField, SymExt>(...)` (and the LogUp helpers) ONCE at setup to
capture a flat single-field Goldilocks IR, then INTERPRET that IR on CPU (prover
over the LDE coset + verifier at the OOD point) and on GPU (one universal
Goldilocks interpreter kernel).

All file/line references below were read directly from the current tree.

---

## 1. Overview & end-state

After this change, each `AIR` (per table) owns, in addition to its existing
`Vec<Box<dyn TransitionConstraintEvaluator>>`, a captured **constraint program**:
a flat list of typed Goldilocks IR ops plus a per-constraint root id. The program
is built once, at AIR construction, by running every constraint through a
recording field (`SymField`/`SymExt`) and recording the LogUp framework
constraints (`LookupBatchedTermConstraint`, `LookupAccumulatedConstraint`) via
the same recording field. At evaluation time, an **interpreter** walks the IR:
on CPU it replaces the per-row `air.compute_transition_prover(...)` call inside
`ConstraintEvaluator::evaluate_transitions` (crypto/stark/src/constraints/evaluator.rs:100)
and the verifier's `air.compute_transition(...)` call
(crypto/stark/src/verifier.rs:209); on GPU it is one Goldilocks kernel that
reads the serialized IR plus the device-resident LDE columns and produces the
per-constraint `Cᵢ` values. The accumulation `Σ βᵢ·Cᵢ·Zᵢ⁻¹ + boundary` and all
zerofier/coefficient machinery stay exactly where they are in
`evaluate_transitions` — the IR only replaces the step that produces each
constraint's scalar `Cᵢ`.

```
                         ┌─ capture (ONCE, at AIR::new, concrete types known) ─┐
constraint structs ──►  run evaluate::<SymField,SymExt>(sym_step)               │
LogUp framework    ──►  run evaluate_batched/accumulated::<SymField,SymExt>(...) │
                         records into thread-local arena ──► ConstraintProgram   │
                         └────────────────────────────────────────────────────┘
                                          │  (serialize)
        ┌─────────────────────────────────┼───────────────────────────────┐
   CPU prover (per LDE row)         CPU verifier (1 OOD point)         GPU kernel
   interp(program, frame) ─► Cᵢ     interp(program, ood_frame) ─► Cᵢ    interp over device cols
        │                                │                               │
        └─► Σ βᵢ·Cᵢ·Zᵢ⁻¹ (unchanged accumulation in evaluate_transitions / verifier)
```

The boxed `dyn TransitionConstraintEvaluator` path is retained verbatim as a
fallback and as the differential-test oracle (Section 9, 12).

---

## 2. The IR (concrete Rust data structures)

The IR is **single-field over Goldilocks**, with a dimension tag distinguishing
base (`dim1`, one u64) from extension (`dim3`, three u64). New crate module:
`crypto/stark/src/symbolic/ir.rs`.

```rust
/// Field-arithmetic dimension of a node's value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dim { D1, D3 }   // base Goldilocks, or its degree-3 extension

/// A leaf input slot, resolved by the interpreter against the current frame
/// and the per-proof uniform inputs.
#[derive(Clone, Copy, Debug)]
pub enum Leaf {
    /// Main trace column read: step.data[row][col], offset selects frame step.
    Main  { step: u8, row: u8, col: u16 },   // dim1 (base) for prover, dim3 for verifier
    /// Aux trace column read: step.aux_data[row][col].
    Aux   { step: u8, row: u8, col: u16 },    // always dim3
    /// Periodic column value at this row.
    Periodic { idx: u16 },                    // dim1
    /// rap_challenges[idx]  (z, alpha, ...)
    Rap   { idx: u16 },                       // dim3
    /// logup_alpha_powers[idx]
    AlphaPow { idx: u16 },                    // dim3
    /// logup_table_offset
    TableOffset,                              // dim3
    /// One of the three precomputed packing shift constants (2^8, 2^16, 2^24)
    Shift { which: u8 },                      // dim1 (prover) / dim3 (verifier)
}

/// One IR instruction. Indices are u32 ids into the program's `nodes` arena.
#[derive(Clone, Copy, Debug)]
pub enum Op {
    Const1(u64),          // dim1 literal (from FieldElement::from(u64/i64), one(), zero())
    Const3([u64; 3]),     // dim3 literal (rare: produced by to_extension / from(u64) in E)
    Leaf(Leaf),
    Add(u32, u32),
    Sub(u32, u32),
    Mul(u32, u32),
    Neg(u32),
    // Embed a dim1 value into dim3 (the to_extension() / IsSubFieldOf::embed step,
    // and the implicit base→ext promotion that F×E ops perform).
    Embed(u32),
}

/// A captured program for one table's transition constraints.
pub struct ConstraintProgram {
    pub nodes: Vec<Op>,            // topologically ordered (id i only references < i)
    pub dims:  Vec<Dim>,           // dims[i] = result dimension of nodes[i]
    pub roots: Vec<u32>,           // roots[c] = node id of constraint c's value Cᵢ
    pub num_base: usize,           // first num_base roots are dim1 (base-field) constraints
    // metadata needed to size interpreter input arrays:
    pub max_step: u8, pub max_main_col: u16, pub max_aux_col: u16,
}
```

**Typing rule.** Every node carries a `Dim`. `Add/Sub/Mul` of (D1,D1)→D1;
any operand D3 ⇒ result D3 (the interpreter auto-`Embed`s the D1 operand,
matching the `F: IsSubFieldOf<E>` mixed-arithmetic the field tower performs at
crypto/math/src/field/element.rs:344). `Embed(D1)→D3`. This mirrors the real
arithmetic exactly: a base×ext multiply is 3 Goldilocks muls (the
`IsSubFieldOf::mul` at crypto/math/src/field/extensions_goldilocks.rs:413), an
ext×ext multiply is one `dot_product_3` schoolbook (extensions_goldilocks.rs:297).

**Serialization for GPU.** `nodes` is encoded as a packed `Vec<u32>` opcode
stream: `[opcode_tag, operand_a, operand_b]` (3×u32 per node; `Const1`/`Const3`
store their literal in a side `Vec<u64>` indexed by operand_a). `dims` is a
`Vec<u8>`. `roots` is a `Vec<u32>`. This is a flat POD layout that copies to the
device as three buffers (`ops: &[u32]`, `consts: &[u64]`, `roots: &[u32]`),
following the same "reinterpret as `&[u64]`/`&[u32]`, transmute-free POD"
discipline used by the GPU LDE bridge in crypto/stark/src/gpu_lde.rs.

---

## 3. Capture front-end — `SymField` design (the distinguishing section)

`SymField` is a **marker type** that implements `IsField`, exactly like
`GoldilocksField` is a zero-sized marker whose `BaseType = u64`
(crypto/math/src/field/goldilocks.rs:70-73). The constraint bodies are generic
over the *field marker* `F` and operate on `FieldElement<F>`, whose data is
`F::BaseType` (crypto/math/src/field/element.rs:50-52). So we choose:

```rust
pub struct SymField;        // base-field recorder (dim1)
pub struct SymExt;          // extension recorder (dim3)
impl IsField for SymField { type BaseType = SymId; ... }
impl IsField for SymExt    { type BaseType = SymId; ... }
```

where `SymId` wraps a `u32` node id (plus the `Dim` it denotes, see arena
decision). Because `BaseType` is just an id, every `IsField::add/mul/...` call
*records* a node into a thread-local arena and returns a fresh id.

### Q1 — ARENA PROBLEM: thread-local arena returning u32 ids (chosen)

`IsField` ops are static, contextless `fn mul(a: &BaseType, b: &BaseType) -> BaseType`
(crypto/math/src/field/traits.rs:104-112). There is no `&self`/arena parameter
to thread. Two options:

* **`BaseType = Arc<Expr>` (tree, hash-consed).** Each op allocates an `Arc`
  node holding its children `Arc`s. Dedup requires hash-consing through a
  thread-local `HashMap<ExprKey, Arc<Expr>>`. *Downsides:* Arc clone/drop traffic
  during capture, recursion in `Drop` for deep trees, and we *still* need a
  thread-local for the hash-cons table — so it buys nothing over ids while
  costing pointer-chasing and an `Arc` per node. Rejected.

* **Thread-local arena returning `u32` ids (CHOSEN).** A `thread_local!` arena:

  ```rust
  thread_local! {
      static ARENA: RefCell<Option<Arena>> = const { RefCell::new(None) };
  }
  struct Arena {
      nodes: Vec<Op>,
      dims:  Vec<Dim>,
      cse:   HashMap<Op, u32>,   // hash-consing: (opcode + operand ids) → id
  }
  ```

  `BaseType` is a small `Copy` struct:

  ```rust
  #[derive(Clone, Copy, Debug, Default)]
  pub struct SymId { id: u32, dim: Dim }   // Default = id 0 ... see Q2 Default note
  ```

  Each op does `ARENA.with(|a| { let a = a.borrow_mut().as_mut().unwrap();
  a.push(Op::Mul(x.id, y.id)) })` where `push` consults `cse` for dedup
  (hash-consing gives a DAG, not a tree, for free). Capture is wrapped in
  `with_arena(|| { ... run constraints ...; arena.take() })` which installs a
  fresh `Arena`, runs the closure, and extracts `(nodes, dims, roots)`.

  This avoids `Arc` entirely, gives DAG dedup via the `cse` map, is `Copy`
  (so `.clone()` in constraint bodies — used heavily, e.g. templates.rs:97,
  cpu.rs:147 — is free and correct), and the only state lives in one
  `thread_local`. Capture runs single-threaded per program (it's a setup-time
  one-shot per table), so the thread-local is uncontended. **This is the
  pick.**

  Hash-consing is mandatory, not optional: without it the ADD-carry templates
  (templates.rs:414-440, `compute_carry_1` recomputes `compute_carry_0`) and
  the LogUp fingerprints (each `compute_fingerprint_from_step` re-reads the same
  columns) would blow up the node count. With `cse`, `compute_carry_0`'s subtree
  is shared.

### Q2 — TRAIT-METHOD COVERAGE (exhaustive)

`SymField` must satisfy `IsField` and the `BaseType` bounds. `SymExt` must
satisfy `IsField`. The `IsSubFieldOf<SymExt> for SymField` impl is also needed
because constraint bodies are bounded `F: IsSubFieldOf<E>` and `evaluate`
returns `FieldElement<F>` (transition.rs:352-355). Below, every required method
with its symbolic implementation or a flag.

**`IsField for SymField` (BaseType = SymId, dim D1):**

| Method | Symbolic impl |
|---|---|
| `type BaseType = SymId` | id+dim, Copy |
| `add(a,b)` | record `Add(a,b)` → D1 |
| `sub(a,b)` | record `Sub(a,b)` → D1 |
| `mul(a,b)` | record `Mul(a,b)` → D1 |
| `neg(a)` | record `Neg(a)` → D1 |
| `double(a)` | default `add(a,a)` works; or record `Add(a,a)` |
| `square(a)` | default `mul(a,a)` works |
| `zero()` | record/return `Const1(0)` id (default `BaseType::default()`; see note) |
| `one()` | record/return `Const1(1)` id |
| `from_u64(x)` | record `Const1(GoldilocksField::from_u64(x))` id |
| `from_base_type(x)` | identity (return x) |
| `inv(a)` | **PROBLEM if ever called** — emit `Op::Inv` only if needed; NOT used by any algebraic constraint nor by the LogUp framework constraints (verified: no `.inv()`/`.div()`/`.pow()` in prover/src/constraints/; LogUp clears denominators so the constraint bodies never invert — fingerprints are *subtracted/multiplied*, not divided, in `evaluate_batched_term_constraint` lookup.rs:1759 and `evaluate_accumulated_constraint` lookup.rs:1887). **Make `inv` `unimplemented!("symbolic inv")`** — if capture ever hits it we want a loud failure, not silent wrong IR. |
| `div(a,b)` | same: `unimplemented!()` (not reached) |
| `eq(a,b)` | **SUBTLE** — returns `bool`, can't be symbolic. Used by `result != FieldElement::zero()` short-circuits in lookup.rs:675, lookup.rs:790. Must return a **conservative `false`** so the "skip zero term" optimization is *not* taken during capture (we always record the multiply). See Q5; this is correct because the skip is a runtime data optimization, and the captured IR must be data-independent. |
| `pow<T>`, `sqrt`, `legendre_symbol` | not reached; default impls call `mul`/`square` and would work but should never run. |

**`BaseType: Clone + Debug + ByteConversion + Default + Send + Sync`
(traits.rs:101):**

| Bound | For `SymId` |
|---|---|
| `Clone + Copy` | derive (it's `{u32, Dim}`) |
| `Debug` | derive |
| `Default` | derive — **but** `Default` is used by `FieldElement::default()` → `value: F::zero()` (element.rs:488) and by `Frame::preallocate` (frame.rs:90-95). During capture we *don't* call `preallocate`; we build a symbolic frame by hand (Q4). A derived `SymId::default()` = `{id:0,dim:D1}` is fine as long as id 0 is a valid node — we reserve node id 0 = `Const1(0)` so a stray default is the zero element. **Resolved, no problem.** |
| `Send + Sync` | `SymId` is `Copy` POD ⇒ auto. The thread-local arena is not part of `SymId`, so no `Send` issue. |
| `ByteConversion` | **FLAG — must implement but never call.** `write_bytes_be/to_bytes_be/from_bytes_be/from_bytes_le` ⇒ `unimplemented!()`. ByteConversion is only exercised by transcript/serialization paths (goldilocks.rs:436), which capture never touches. Acceptable: it's a trait-bound satisfier, not a live method. |

**`IsField for SymExt` (BaseType = SymId, dim D3):** identical table, but every
recorded node is tagged D3, and `from_u64(x)` records `Const3([from_u64(x),0,0])`
(matching `Degree3...::from_u64` extensions_goldilocks.rs:399). `one()`→
`Const3([1,0,0])`, `zero()`→`Const3([0,0,0])`. `inv`/`div` `unimplemented!()`.

**`IsSubFieldOf<SymExt> for SymField` (traits.rs:17-25):** this is the mixed
base×ext arithmetic surface the field-element operators dispatch through
(element.rs:223,295,346). Each must record the correct mixed node:

| Method | Symbolic impl |
|---|---|
| `mul(a: &SymId/*D1*/, b: &SymId/*D3*/) -> SymId/*D3*/` | record `Mul(a,b)` tagged D3 (the interpreter sees a D1×D3 mul and does the 3-mul base×ext path) |
| `add(a,b) -> D3` | record `Add(a,b)` D3 |
| `sub(a,b) -> D3` | record `Sub(a,b)` D3 |
| `div(a,b)` | `unimplemented!()` (not reached) |
| `embed(a: SymId/*D1*/) -> SymId/*D3*/` | record `Embed(a)` → D3 |
| `to_subfield_vec(b)` | `unimplemented!()` (not reached; only serialization uses it) |

Note the blanket `impl IsSubFieldOf<F> for F` (traits.rs:27-60) automatically
gives us `IsSubFieldOf<SymField> for SymField` and `IsSubFieldOf<SymExt> for
SymExt` (the prover-frame `evaluate` with FF=F and the verifier-frame with
FF=E both rely on these reflexive impls). Those route to `SymField::mul` etc.,
so no extra code.

**`IsFFTField for SymField`?** The `AIR` trait bounds `Field: IsFFTField`
(traits.rs:139) and `AirWithBuses` further bounds `Field: IsPrimeField`
(lookup.rs:805). **But capture does NOT instantiate any `AIR<Field=SymField>`.**
Capture only calls the *constraint object's* generic `evaluate::<SymField,
SymExt>(step)` and the LogUp helper fns `::<SymField, SymExt>` — those are
bounded only `FF: IsSubFieldOf<EE>, EE: IsField` (transition.rs:352-355,
lookup.rs:1759, lookup.rs:1887). So `SymField` needs **only** `IsField +
IsSubFieldOf<SymExt>`, NOT `IsFFTField`/`IsPrimeField`. This is the single most
important feasibility fact: it sidesteps `IsFFTField::{TWO_ADICITY, root,
field_name}` and `IsPrimeField::{canonical, from_hex, field_bit_size}` entirely
(none are reachable from `evaluate`). *Verified:* `evaluate`'s only bound is
`FF: IsSubFieldOf<EE>` (transition.rs:354), the LogUp inner fns'
only bound is `A: IsSubFieldOf<B>, B: IsField` (lookup.rs:1759, lookup.rs:1887,
lookup.rs:1679, lookup.rs:1689). Capture never builds the AIR with sym types.

### Q3 — Constants & `to_extension` / `one()` / `zero()`

* `FieldElement::<F>::from(i64/u64)` → `From<u64>`/`From<i64>` (element.rs:136,149)
  → `F::from_u64(value)`. For `F = SymField` this records `Const1(c)` with
  `c = GoldilocksField::from_u64(value)` (we *fold the real Goldilocks reduction*
  at capture time so the literal stored is canonical). `i64` negatives go through
  `-Self::from(abs)` (element.rs:157) → records `Neg(Const1(abs))`; or we can
  constant-fold to `Const1(p - abs)`. Either is correct; constant-folding negatives
  keeps the IR smaller. Examples captured this way: `inv_2_32` (templates.rs:30-36,
  a `from(INV_SHIFT_32)`), `SHIFT_16` (cpu.rs:69), `AddLinearTerm` coefficients
  `1<<16`, `1<<8`, `1<<24` (templates.rs:266-326), bus `LinearTerm` coefficients
  (lookup.rs:656,772).
* `FieldElement::one()`/`zero()` (element.rs:550,556) → `F::one()`/`F::zero()`.
  For `SymField` → `Const1(1)`/`Const1(0)`; for `SymExt` → `Const3([1,0,0])`/
  `Const3([0,0,0])`. The literal `FieldElement::<F>::one()` appears all over
  (templates.rs:98, cpu.rs:146).
* `.to_extension::<L>()` (element.rs:566) → `<F as IsSubFieldOf<L>>::embed(value)`.
  Used by the adapter's verifier path `...evaluate(...).to_extension()`
  (transition.rs:431). For `F=SymField, L=SymExt` this records `Embed(child)`.
  **However** — see Section 4: in the *prover* capture we run the adapter with
  FF=F (base), and in the *verifier* capture we run FF=E (already D3); we will
  capture the constraint's **base value** (the `evaluate` result, dim D1) and let
  the interpreter/accumulator handle the embed, mirroring how the real prover
  keeps base constraints in `base_evals: &mut [FieldElement<F>]`
  (evaluator.rs:106-110). So `to_extension` is mostly *not* in the captured graph
  for base constraints; it only appears if a constraint body itself calls
  `to_extension`, which none of the algebraic ones do (they return D1).

### Q4 — SYMBOLIC FRAME

Capture needs a `TableView<SymField, SymExt>` (and `Frame<SymField, SymExt>`)
whose column reads return `Leaf` nodes. `TableView` is
`{ data: Vec<Vec<FieldElement<F>>>, aux_data: Vec<Vec<FieldElement<E>>> }`
(table.rs:397-399) and reads go through `get_main_evaluation_element(row, col)`
(table.rs:410) / `get_aux_evaluation_element` (table.rs:414). So we build a
symbolic frame by filling each cell with a `FieldElement::from_raw(SymId)` whose
id is a recorded `Leaf::Main { step, row, col }` / `Leaf::Aux { ... }`:

```rust
fn symbolic_frame(num_steps, rows_per_step, num_main, num_aux) -> Frame<SymField, SymExt> {
    let steps = (0..num_steps).map(|step| {
        let data = (0..rows_per_step).map(|r|
            (0..num_main).map(|c|
                FieldElement::<SymField>::from_raw(record_leaf(Leaf::Main{step,row:r,col:c}))
            ).collect()).collect();
        let aux_data = (0..rows_per_step).map(|r|
            (0..num_aux).map(|c|
                FieldElement::<SymExt>::from_raw(record_leaf(Leaf::Aux{step,row:r,col:c}))
            ).collect()).collect();
        TableView::new(data, aux_data)
    }).collect();
    Frame::new(steps)
}
```

`num_steps` = `offsets.len()` (= 2 for LogUp tables, `transition_offsets:
vec![0,1]` lookup.rs:909). `rows_per_step` = step_size/blowup (1 for these
tables). The two `TransitionEvaluationContext` variants needed for capture
(Q5/Q6): a `Prover { frame: &Frame<SymField,SymExt>, periodic_values:
&[FieldElement<SymField>], rap_challenges: &[FieldElement<SymExt>],
logup_alpha_powers, logup_table_offset, packing_shifts: &PackingShifts<SymField> }`.
Each uniform input (periodic, rap, alpha pow, table offset, shifts) is also a
recorded `Leaf` (`Periodic`, `Rap`, `AlphaPow`, `TableOffset`, `Shift`). The
shift constants `PackingShifts::<SymField>::new()` (lookup.rs:54) call
`FieldElement::<SymField>::from(SHIFT_8/16)` and `&shift_8 * &shift_16` — those
record `Const1` + `Mul` automatically; but to keep the IR clean we instead
construct `PackingShifts` with `Leaf::Shift{0/1/2}` ids so the interpreter
injects the real precomputed constants at eval time (they're loop-invariant and
the existing code precomputes them once, lookup.rs:64). Both are correct; the
`Leaf::Shift` version matches the existing precompute and keeps shifts uniform.

---

## 4. Capturing the algebraic constraints (the ~29 structs, via the adapter)

The ~29 algebraic constraints implement the user-facing `TransitionConstraint`
trait and are wrapped by `TransitionConstraintAdapter` (transition.rs:393).
Their bodies are generic `evaluate<FF,EE>(&self, step: &TableView<FF,EE>) ->
FieldElement<FF>` (transition.rs:352). **We do NOT touch any body.** Capture
calls each constraint's `evaluate::<SymField, SymExt>(sym_step)` directly and
reads the returned `SymId` (the root for that constraint).

**Count, verified by grep: there are 33 (not ~29) algebraic
`impl TransitionConstraint<GoldilocksField, GoldilocksExtension>` structs**, not
just the CPU ones the team-lead memo listed. Beyond templates.rs/cpu.rs they
span prover/src/tables/: branch.rs:519, commit.rs:837, cpu32.rs:645,
dvrm.rs:1219, ec_scalar.rs:291, ecdas.rs:{363,402,426}, ecsm.rs:{663,698,791,816},
eq.rs:262, keccak.rs:503, load.rs:572, lt.rs:536, memw_aligned.rs:708,
memw_register.rs:388, memw.rs:921, mul.rs:847, shift.rs:914, store.rs:282
(plus the 11 in templates.rs/cpu.rs). The "zero body edits" win therefore
applies to **all 33**, including the large ones (keccak, ecsm, dvrm, mul) — a
bigger payoff than the memo implied, but those large bodies also drive risk 5/6
(node count / GPU scratch).

These constraints all return a base-field (`FF=F`) value, so we capture them as
**dim-D1 roots** placed in `roots[0..num_base]`, matching the prover's base
split (evaluator.rs:50, evaluator.rs:106). **Safe-op audit (first-hand + grep,
load-bearing for feasibility):** every body uses only `clone`, `+`, `-`, `*`,
`neg` (via `-x`), `FieldElement::from`, `one()`, `get_main_evaluation_element` —
e.g. `IsBitConstraint` (templates.rs:92-107: `&cond * &x * (one - x)`),
`AddConstraint` (templates.rs:442-467 + the carry helpers, which multiply by the
constant `inv_2_32`), `ProductZeroConstraint` (cpu.rs:105-112), `Arg2Constraint`
(cpu.rs:277-303), `BranchRvdConstraint`/`NextPcAddConstraint`
(cpu.rs:394-446, cpu.rs:518-571). Crucially, a grep over the **entire**
`prover/src` (non-test) finds **zero** `.inv()`/`.pow()`/`.div()`/`.sqrt()`/
`.legendre_symbol()` calls and **zero** field-value conditionals (`== FieldElement`,
`.is_zero()`, `if …value()…`) across all 17 table files — so no body, and no
helper any body transitively calls, performs division/inversion/exponentiation
or branches on a field value. (The per-struct degree + body summary from the
enumeration sub-agent is appended at the end.)

**Framework glue changes** (minimal, additive):

1. New trait method on `TransitionConstraint` with a default that **panics**, and
   override it for the adapter is *not* the route — instead add a free function
   `capture_user_constraint<T: TransitionConstraint<GoldilocksField,
   GoldilocksExtension>>(c: &T, step: &TableView<SymField,SymExt>) -> SymId`
   that just calls `c.evaluate::<SymField, SymExt>(step)`. Because the adapter
   stores `T` (transition.rs:393, `TransitionConstraintAdapter<T>(pub T)`), but
   the AIR only keeps `Box<dyn TransitionConstraintEvaluator>` (lookup.rs:813),
   we cannot recover `T` from the boxed object. **Therefore capture must run at
   the point where concrete constraint types still exist — i.e. inside each
   table's constraint-list builder** (e.g. `create_all_cpu_constraints`
   cpu.rs:619), *before* `.boxed()`. See Section 9.

2. Add a capture entry point to the `TransitionConstraintEvaluator` trait:
   `fn capture(&self, ctx: &SymCaptureCtx) -> SymId;` with a default that calls
   `evaluate_verifier` against a symbolic context... **but** `evaluate_verifier`
   needs `&mut [FieldElement<E>]` slots, and for the adapter it calls
   `self.0.evaluate(...).to_extension()` (transition.rs:431). Running *that*
   under sym types records the constraint plus a trailing `Embed`, giving a D3
   root. That is acceptable for capture purposes (the embed is a no-op cost on
   D1→D3 and the accumulator can treat the root as D3). **This is the cleaner,
   object-safe route:** add `fn capture(&self, ctx, &mut [SymId])` to
   `TransitionConstraintEvaluator`, default-implemented by calling a sym version
   of `evaluate_verifier`. The adapter's `capture` runs
   `self.0.evaluate::<SymField,SymExt>(frame.step).to_extension()` → D3 root, OR
   (better, to keep base/ext split) runs `evaluate` and stores the D1 root for
   `idx < num_base`. We implement the latter: `capture` mirrors `evaluate_prover`
   (transition.rs:439) — D1 root into base slot for base constraints, D3 for the
   LogUp ones. This keeps the captured program's `num_base` aligned with
   `air.num_base_transition_constraints()` (lookup.rs:1025).

The recommended concrete design: add **one** method
`fn capture(&self, ctx: &SymCaptureContext, base_roots: &mut Vec<SymId>,
ext_roots: &mut Vec<SymId>)` to `TransitionConstraintEvaluator`
(crypto/stark/src/constraints/transition.rs). Default impl: run the verifier-
style body symbolically and push a D3 root. Adapter override
(transition.rs:395): run `self.0.evaluate::<SymField,SymExt>` and push a D1 root
when `idx < base_roots.capacity-marker`, else D3. The two LogUp framework
structs override `capture` to run their `evaluate_*_constraint` inner fns under
sym types (Section 5).

---

## 5. Capturing the LogUp / extension framework constraints (Q5 — the crux)

The two LogUp constraints do **not** go through the adapter; they
`impl TransitionConstraintEvaluator` directly and branch on the
`TransitionEvaluationContext` enum (lookup.rs:1741, lookup.rs:1868). The decisive
question: **are their helpers generic enough to run under SymField/SymExt?**

**Verdict: YES — they are fully capturable, no hand-emit needed.** Evidence:

* `compute_multiplicity_from_step<A: IsSubFieldOf<B>, B: IsField>` (lookup.rs:1679)
  — generic; body is `multiplicity.evaluate_with(|col| step.get_main_evaluation_element(0,col).clone())`
  → `Multiplicity::evaluate_with<F,G>` (lookup.rs:1252) uses only `one()`, `+`, `-`,
  `*`, `FieldElement::from(coeff)`. All recordable. ✓
* `compute_fingerprint_from_step<A: IsSubFieldOf<B>, B: IsField>` (lookup.rs:1689)
  — generic; body builds `FieldElement::<B>::from(bus_id)` then loops
  `bv.accumulate_fingerprint_from_step(...)` (lookup.rs:738) which uses
  `get_main_evaluation_element`, `Packing::accumulate_fingerprint_with`
  (lookup.rs:272: only `+`,`*`, shift consts), and `z - &linear_combination`.
  All recordable. ✓
* `evaluate_batched_term_constraint<A: IsSubFieldOf<B>, B: IsField>`
  (lookup.rs:1759) — generic inner fn; computes `c * fp_a * fp_b - term_a -
  term_b`. ✓
* `evaluate_accumulated_constraint<A: IsSubFieldOf<B>, B: IsField>`
  (lookup.rs:1887) — generic; `delta * f - m*sign` etc. ✓

**The two sign/branch subtleties, and why they're still capturable:**

1. **`is_sender` sign logic** (lookup.rs:1780-1790, lookup.rs:1927-1932,
   lookup.rs:1954-1956): these are `if interaction.is_sender { term } else {
   -term }` — branching on a **compile-time-known `bool` field of the
   interaction struct**, NOT on a field *value*. During capture `is_sender` is a
   concrete `bool`, so the branch is resolved at capture time and we record
   either `term` or `Neg(term)`. ✓ No data dependence.

2. **`result != FieldElement::<A>::zero()` short-circuit** in
   `accumulate_fingerprint_from_step` (lookup.rs:790) and the column-major
   variant (lookup.rs:675): this branches on a *field value* via `PartialEq` →
   `F::eq`. For `SymField` we make `eq` return **`false` always** (Q2), so the
   capture path *always records the multiply* (`*acc += result * alpha_powers[..]`).
   This is **correct and conservative**: the skip is a runtime optimization for
   rows where the value happens to be zero; the IR must be valid for *all* rows,
   so it must include the multiply unconditionally. The interpreter then always
   does the multiply — slightly more work than the optimized CPU path on
   all-zero rows, but bit-identical results. ✓ (If we wanted to preserve the
   optimization we could detect "operand is a `Const1(0)` node" at capture time
   and constant-fold, recovering the bus-id-padding skip statically. Recommended
   as a cheap IR peephole.)

**Building the capture context.** We construct
`TransitionEvaluationContext::Prover { frame: &Frame<SymField,SymExt>,
rap_challenges: &[FieldElement<SymExt>], logup_alpha_powers:
&[FieldElement<SymExt>], logup_table_offset: &FieldElement<SymExt>,
packing_shifts: &PackingShifts<SymField>, periodic_values:
&[FieldElement<SymField>] }` (the enum at traits.rs:77-84). Every slice element
is a `Leaf` node (`Rap{idx}`, `AlphaPow{idx}`, `TableOffset`, `Shift{}`). The
frame has 2 steps (acc uses `frame.get_evaluation_step(0)` and `(1)`,
lookup.rs:1972-1973). We call the constraint's `evaluate_verifier` (or the new
`capture`) with this Prover context; the matching `match` arm
(lookup.rs:1794, lookup.rs:1963) fires the generic inner fn under sym types and
returns a D3 root. **No fallback hand-emit is required** — this is the key win
over a hand-written LogUp IR.

One caveat to call out: `evaluate_verifier` writes into `transition_evaluations:
&mut [FieldElement<E>]` (lookup.rs:1827). Under sym types `E=SymExt`, so the
result is a `FieldElement<SymExt>` whose value is the root `SymId` — we read it
back from the slot. The slice must be pre-filled with a sentinel; we size it to
`num_transition_constraints` and read `slot[constraint_idx]` after the call. ✓

---

## 6. CPU interpreter

New module `crypto/stark/src/symbolic/interp.rs`. Two entry points, one shared
core.

**Core:** `fn eval_program(prog: &ConstraintProgram, inputs: &Inputs, out: &mut Outputs)`
walks `prog.nodes` in id order, computing each node into a value array. Because
nodes are topologically ordered (id i references < i) a single forward pass with
a `Vec<Value>` (len = nodes.len()) suffices; `Value` is an enum
`{ D1(FieldElement<F>), D3(FieldElement<E>) }` with auto-embed on mixed ops.
`inputs` resolves `Leaf`s: `Main`/`Aux` from the current frame step/row/col,
`Periodic/Rap/AlphaPow/TableOffset/Shift` from the per-proof uniform arrays
(Section 8). Final: `out.base[c] = values[roots[c]]` for `c<num_base` (as D1),
`out.ext[c] = values[roots[c]]` (as D3) otherwise.

**Prover slot** — replaces the per-row body of `eval_row` in
`ConstraintEvaluator::evaluate_transitions` (evaluator.rs:79-135). Today that
closure fills `frame`, fills `periodic_buf`, builds the `ctx`, and calls
`air.compute_transition_prover(&ctx, base_buf, transition_buf)`
(evaluator.rs:100). We keep the frame fill (frame.fill_from_lde,
evaluator.rs:86) and periodic fill, then call
`eval_program(prog, &Inputs{ frame, periodic_buf, rap_challenges,
logup_alpha_powers, logup_table_offset, packing_shifts }, &mut Outputs{ base_buf,
transition_buf })` **instead of** `compute_transition_prover`. The boundary —
the accumulation `acc + zerofier·eval·beta` (evaluator.rs:102-132) — is
untouched. Base/ext split is preserved because the program's `roots[0..num_base]`
are D1 and the rest D3.

**Verifier slot** — replaces `air.compute_transition(&ctx)` at verifier.rs:209.
The verifier frame is `TableView<E,E>` (verifier.rs:198, `into_frame`) so *all*
reads are D3; we run `eval_program` with an `Inputs` whose `Main` leaves resolve
to the OOD frame's D3 cells (interpreter reads them as D3 directly — the program
is the same, only the leaf-resolution dimension differs). Output is the
`Vec<FieldElement<E>>` consumed by the zerofier fold (verifier.rs:218-225),
untouched.

**Base/ext handling.** The interpreter must do D1×D3 the cheap way (3 muls,
matching `IsSubFieldOf::mul` extensions_goldilocks.rs:413) and D3×D3 via
`dot_product_3` (one `Degree3...::mul` extensions_goldilocks.rs:297). We reuse
the real `FieldElement<GoldilocksField>` / `FieldElement<Degree3...>` arithmetic
inside `Value`, so the interpreter's per-node cost equals the boxed path's — the
IR overhead is just the opcode dispatch (a `match` per node), which is cheap
relative to a Goldilocks mul. For the prover the program is run with
`F=GoldilocksField, E=Degree3...`; for the verifier with `F=E=Degree3...`.

---

## 7. GPU interpreter sketch

One universal Goldilocks kernel, modeled on the gpu_lde TypeId+transmute seam.

**Host seam** (`crypto/stark/src/symbolic/gpu_interp.rs`): a
`try_eval_program_gpu<F,E>(prog, lde_trace, uniforms, out) -> Option<()>` that,
exactly like check_base_layout (gpu_lde.rs:106) / the barycentric dispatchers
(gpu_lde.rs:811), gates on `TypeId::of::<F>() == GoldilocksField` and
`TypeId::of::<E>() == Degree3...` (gpu_lde.rs:826-831), a size threshold, and a
device-resident main/aux LDE handle (`lde_trace.gpu_main()`/`gpu_aux()`,
gpu_lde.rs:832,915). On mismatch → `None` → CPU interpreter fallback. The
program's three POD buffers (`ops: &[u32]`, `consts: &[u64]`, `roots: &[u32]`,
Section 2) plus the uniform arrays (rap challenges, alpha powers, table offset,
periodic columns, shift consts — all reinterpreted to `&[u64]` via the same
`#[repr(transparent)]` pattern as weights_to_u64 gpu_lde.rs:196) are H2D-copied
**once** (they're constant across all LDE rows). The columns are already on
device from the R1 LDE keep-handles (gpu_lde.rs:459, `GpuLdeBase`/`GpuLdeExt3`).

**Device kernel** (new file under crypto/math-cuda/src/, e.g.
`symbolic_interp.cu` + a `math_cuda::symbolic` Rust wrapper): one thread per LDE
row. Each thread allocates a small per-node scratch in registers/shared/local
memory (`nodes.len()` Goldilocks values — programs are small, ~hundreds of nodes
per table) and runs the same forward pass as the CPU core, using the existing
math-cuda Goldilocks device primitives: base mul/add/sub (the same reduce128
identities as goldilocks.rs:197), and ext3 mul as device `dot_product_3`
(mirroring goldilocks.rs:290). The kernel writes `out[row*num_constraints + c]`
for each root. **What crosses the host/device boundary:** program buffers + uniforms
(small, once); columns (already resident); output = `num_constraints × lde_size`
ext3 values (or, with the base/ext split, `num_base × lde_size` base + the rest
ext3) — D2H once. The accumulation `Σ βᵢ·Cᵢ·Zᵢ⁻¹` can stay on host (cheap) or be
fused into the kernel later; for v1 keep it on host to minimize surface, matching
how `apply_ext3_scalar` post-processes on host (gpu_lde.rs:694).

The single-field design means **one kernel** handles every table — the per-table
difference is entirely in the data buffers (`ops/consts/roots`), so there is no
per-table CUDA codegen. This is the whole point of the interpreter approach.

---

## 8. Inputs plumbing (Q6)

Periodic values, rap_challenges, logup_alpha_powers, logup_table_offset, and
packing_shifts vary **per proof** but are **constant across all rows** of one
table's evaluation. They are already computed once per `evaluate_transitions`
call: `logup_alpha_powers` (evaluator.rs:53), `packing_shifts` (evaluator.rs:64),
`rap_challenges` (passed in), `logup_table_offset` (evaluator.rs:47),
`lde_periodic_columns` (evaluator.rs:251 — note periodic is **per-row**, indexed
by `col[i]`, so it is a row-varying leaf resolved like a column). They become IR
**leaf inputs** with these resolutions in the interpreter's `Inputs`:

| Leaf | CPU resolution | GPU resolution |
|---|---|---|
| `Main{step,row,col}` | `frame.get_evaluation_step(step).get_main_evaluation_element(row,col)` | device LDE main column, strided by step·lde_step_size (frame.fill_from_lde logic, frame.rs:117) |
| `Aux{...}` | `...get_aux_evaluation_element` | device LDE aux column |
| `Periodic{idx}` | `periodic_buf[idx]` (= `lde_periodic_columns[idx][i]`) | device periodic column |
| `Rap{idx}` | `rap_challenges[idx]` | uniform buffer slot |
| `AlphaPow{idx}` | `logup_alpha_powers[idx]` | uniform buffer slot |
| `TableOffset` | `logup_table_offset` | uniform buffer slot |
| `Shift{which}` | `packing_shifts.{shift_8,16,24}` | uniform buffer slot |

At capture time, the leaf *indices* (which rap challenge, which alpha power) are
fixed by how the constraint reads them (`rap_challenges[0]` = z, lookup.rs:1769;
`alpha_powers[alpha_offset]` walked in packing, lookup.rs:294). So the program
encodes the exact indices; the interpreter just gathers from the per-proof arrays.
The arrays' lengths are known at eval time (`max_bus_elements` →
`compute_alpha_powers` count, evaluator.rs:55). No re-capture per proof.

---

## 9. Coexistence & object-safety

* **Where capture runs.** Because the AIR only stores `Box<dyn
  TransitionConstraintEvaluator>` (lookup.rs:813) and the adapter erases the
  concrete `T` (transition.rs:393), the cleanest object-safe route is to add a
  `capture` method to the **`TransitionConstraintEvaluator` trait** (which the
  boxed objects *do* expose). The adapter's `capture` (transition.rs:395) calls
  `self.0.evaluate::<SymField,SymExt>` — concrete `T` is in scope there. The two
  LogUp structs override `capture` to run their generic inner fns. Then a single
  pass over `air.transition_constraints()` (the existing `Vec<Box<dyn ...>>`,
  traits.rs:314) captures the whole program. This means **the AIR builds its
  `ConstraintProgram` once in a new default method**
  `AIR::constraint_program(&self) -> ConstraintProgram` that iterates the boxed
  constraints and calls `capture` on each within a `with_arena` scope. No table
  builder needs editing.

* **`capture` and object safety.** Adding `fn capture(&self, ctx:
  &SymCaptureContext, base: &mut Vec<SymId>, ext: &mut Vec<SymId>)` to the
  trait keeps it object-safe (no generics in the method signature; `SymField`/
  `SymExt` are concrete). The default impl runs the verifier-shaped body
  symbolically. ✓

* **Generic boxed path retained as fallback.** `compute_transition_prover`
  (traits.rs:254) and `compute_transition` (traits.rs:223) stay. A feature flag
  `symbolic-interp` (or a runtime toggle) selects, inside
  `evaluate_transitions` (evaluator.rs:100) and the verifier (verifier.rs:209),
  whether to call the IR interpreter or the boxed path. Default off until the
  differential test (Section 12) is green; then default on.

* **TypeId gating for GPU.** The GPU interpreter only engages for the real
  `GoldilocksField`/`Degree3...` instantiation (Section 7), identical to
  gpu_lde.rs:119-152. For any other field the host code transparently uses the
  CPU interpreter or the boxed path.

* **Cache the program.** `ConstraintProgram` is built once per AIR and stored in
  the AIR (or in `ConstraintEvaluator::new`, evaluator.rs:188, alongside
  `boundary_constraints`). It is immutable and `Send + Sync` (POD), so it's
  shared across all Rayon workers and reused across proofs of the same table
  shape.

---

## 10. Exhaustive file-by-file change list

**New files:**

* `crypto/stark/src/symbolic/mod.rs` — module root, re-exports.
* `crypto/stark/src/symbolic/sym_field.rs` —
  `pub struct SymField; pub struct SymExt; #[derive(Clone,Copy,Default,Debug)]
  pub struct SymId{id:u32,dim:Dim}`; `impl IsField for SymField/SymExt`;
  `impl IsSubFieldOf<SymExt> for SymField`; `impl ByteConversion for SymId`
  (unimplemented stubs); the `thread_local! ARENA` + `with_arena` +
  `record(Op)->SymId` (hash-consing) + `record_leaf(Leaf)->SymId`.
* `crypto/stark/src/symbolic/ir.rs` — `Dim`, `Leaf`, `Op`, `ConstraintProgram`,
  serialization (`to_pod()` → `(Vec<u32>, Vec<u64>, Vec<u32>)`).
* `crypto/stark/src/symbolic/capture.rs` — `SymCaptureContext`
  (builds the symbolic `Frame`/`TableView`/uniform leaves, Q4),
  `fn capture_program(constraints: &[Box<dyn TransitionConstraintEvaluator<...>>],
  layout, num_base, ...) -> ConstraintProgram`.
* `crypto/stark/src/symbolic/interp.rs` — `Value`, `Inputs`, `Outputs`,
  `fn eval_program(prog,&Inputs,&mut Outputs)` (CPU core + prover & verifier
  adapters).
* `crypto/stark/src/symbolic/gpu_interp.rs` — `try_eval_program_gpu<F,E>(...)
  -> Option<()>` (TypeId gate + H2D uniforms + kernel launch + D2H), guarded by
  the cuda feature.
* `crypto/math-cuda/src/symbolic_interp.rs` (+ `.cu`) — `math_cuda::symbolic::
  eval_program_*` device wrapper and the one universal Goldilocks/ext3 kernel.

**Modified files:**

* `crypto/stark/src/constraints/transition.rs` — add
  `fn capture(&self, ctx: &SymCaptureContext, base: &mut Vec<SymId>, ext: &mut
  Vec<SymId>)` to `TransitionConstraintEvaluator` (default impl runs verifier-
  shaped body symbolically); override in `TransitionConstraintAdapter`
  (transition.rs:395) to run `self.0.evaluate::<SymField,SymExt>`.
* `crypto/stark/src/lookup.rs` — override `capture` for
  `LookupBatchedTermConstraint` (lookup.rs:1741) and
  `LookupAccumulatedConstraint` (lookup.rs:1868) to run their generic inner fns
  under sym types. The inner fns are **unchanged** (already generic).
* `crypto/stark/src/traits.rs` — add default method
  `fn constraint_program(&self) -> ConstraintProgram` (iterates
  `self.transition_constraints()` + `with_arena`).
* `crypto/stark/src/constraints/evaluator.rs` — in `evaluate` (evaluator.rs:216)
  build/fetch the cached `ConstraintProgram`; in `evaluate_transitions`
  (evaluator.rs:100) replace `air.compute_transition_prover(&ctx, base_buf,
  transition_buf)` with `eval_program(...)` (behind the feature/toggle), with the
  GPU dispatch tried first (gpu_interp `try_eval_program_gpu`, else CPU).
* `crypto/stark/src/verifier.rs` — at verifier.rs:209 replace
  `air.compute_transition(&ctx)` with the verifier interpreter (same toggle).
* `crypto/stark/src/lib.rs` (or `mod.rs`) — `pub mod symbolic;`.
* `crypto/math/src/field/...` — **no change** (SymField lives in the stark
  crate; it only needs the public `IsField`/`IsSubFieldOf` traits, which are
  already public). If `ByteConversion` for `SymId` must be impl'd where the
  trait is defined due to orphan rules, add a thin impl in math; otherwise keep
  in stark (SymId is a stark type, ByteConversion is a math trait — the impl is
  allowed in stark since SymId is local). ✓ orphan-rule-safe.

**Key new signatures:**
```rust
impl IsField for SymField { type BaseType = SymId; fn mul(a:&SymId,b:&SymId)->SymId {record(Op::Mul(a.id,b.id),Dim::D1)} ... }
impl IsSubFieldOf<SymExt> for SymField { fn mul(a:&SymId,b:&SymId)->SymId {record(Op::Mul(a.id,b.id),Dim::D3)} fn embed(a:SymId)->SymId{record(Op::Embed(a.id),Dim::D3)} ... }
pub fn capture_program(cs: &[Box<dyn TransitionConstraintEvaluator<GoldilocksField,GoldilocksExtension>>], layout:(usize,usize), num_base:usize, offsets:&[usize], step_size:usize) -> ConstraintProgram;
pub fn eval_program(prog:&ConstraintProgram, inp:&Inputs<'_>, out:&mut Outputs<'_>);
pub(crate) fn try_eval_program_gpu<F:'static,E:'static>(prog:&ConstraintProgram, lde:&LDETraceTable<F,E>, uni:&Uniforms<E>, out:&mut [FieldElement<E>]) -> Option<()>;
```

---

## 11. Risks & unknowns, ranked

1. **IsField-contract friction is LOW — feasibility CONFIRMED.** The decisive
   finding: capture never instantiates `AIR<Field=SymField>`, only calls
   `evaluate::<SymField,SymExt>` and the LogUp inner fns, whose bounds are only
   `IsSubFieldOf + IsField` (transition.rs:354, lookup.rs:1759/1887/1679/1689).
   So `SymField` needs **no** `IsFFTField`/`IsPrimeField` — the
   `TWO_ADICITY`/root/canonical/from_hex methods are unreachable. The remaining
   `IsField` methods that can't be symbolic (`inv`, `div`, `eq`) are either never
   reached (`inv`/`div`: no division in any constraint body, verified by grep +
   reading templates.rs/cpu.rs/lookup.rs) or handled by a conservative `eq=false`
   (the only `eq` use is a runtime zero-skip optimization, lookup.rs:675/790,
   which capture must *not* take). `ByteConversion`/`to_subfield_vec` are
   bound-satisfier stubs that never run. **Residual risk:** a future constraint
   body that calls `.inv()`/`.pow()`/branches on a field value would panic at
   capture; mitigate with the loud `unimplemented!()` + a CI check.

2. **LogUp capturability is HIGH-confidence YES.** The helpers are already
   generic over `A: IsSubFieldOf<B>, B` (lookup.rs:1679/1689) and the constraint
   inner fns too (lookup.rs:1759/1887); `is_sender` is a compile-time bool, not a
   field value (lookup.rs:1780); the only field-value branch is the zero-skip,
   handled by `eq=false`. So **no hand-emit of LogUp IR is needed** — this is the
   approach's biggest advantage over hand-writing. **Residual risk:** the `eq`
   short-circuit means the captured IR always multiplies even by `Const1(0)`
   bus-padding; mitigate with a constant-fold peephole (detect `Mul(_,Const(0))`/
   `Add(x,Const(0))` at capture) so the IR matches the optimized path's node
   count and the GPU kernel doesn't waste lanes. Low effort, high value.

3. **Bit-for-bit equivalence of the interpreter vs the boxed path.** The
   interpreter reuses the real `FieldElement` arithmetic, so per-op results are
   identical; the risk is in *order of operations* (field add/mul are
   associative/commutative in value but the existing code's specific fold order
   is what the OOD/LDE evaluations must match for the proof to verify). Since we
   capture the *exact* call sequence the body executes (recording in evaluation
   order), the IR's forward-pass order equals the body's order. **Residual
   risk:** the zero-skip fold (lookup.rs:672) changes the *additive grouping* on
   zero rows; with `eq=false` we always add, which is value-identical (adding 0).
   So equivalence holds. Validate empirically (Section 12).

4. **Capture-time arena correctness with hash-consing.** A wrong `cse` key
   (e.g. not distinguishing D1 vs D3 nodes with the same operands) would alias
   nodes of different dimension. Mitigate: include `Dim` in the `cse` key, or
   never CSE across dims. Low risk, but must be tested.

5. **GPU per-thread scratch pressure.** Each thread needs `nodes.len()`
   Goldilocks values live. If a table's program is large (hundreds of nodes ×
   ext3 = hundreds × 24 bytes), register/shared pressure could limit occupancy.
   Mitigate: liveness analysis to reuse scratch slots (a node's value is dead
   after its last use), or spill to local memory. This is a perf risk, not a
   correctness risk, and v1 can keep the accumulation on host. Medium.

6. **Unknown: exact node counts per table.** Not yet measured, and there are
   **33** algebraic constraints across many tables — the largest bodies
   (keccak.rs, ecsm.rs, dvrm.rs, mul.rs, commit.rs) are big polynomial circuits
   and will dominate node count. ADD/LogUp with hash-consing should be small (low
   hundreds), but keccak/ecsm could be thousands of nodes, directly amplifying
   risk 5 (GPU per-thread scratch). Resolve by instrumenting `capture_program`
   to print `nodes.len()` per table during the differential test, and prioritize
   liveness-based scratch reuse for the large tables.

**Overall feasibility verdict: HIGH.** The SymField approach is sound; the
IsField-contract friction is manageable (the unreachable-method insight is the
crux) and LogUp captures cleanly with zero hand-emit.

---

## 12. Effort estimate & validation strategy

**Effort (engineer-days, by workstream):**

* W1 — `SymField`/`SymExt`/`SymId` + arena + hash-consing + IsField/IsSubFieldOf
  impls + stubs: **2–3 d**. (Mechanical once the unreachable-method set is fixed.)
* W2 — IR types + serialization + capture context (symbolic frame/uniforms) +
  `capture` trait method + adapter/LogUp overrides + `AIR::constraint_program`:
  **3–4 d**. (LogUp override is the fiddly part but the inner fns are unchanged.)
* W3 — CPU interpreter (core + prover slot in evaluator.rs + verifier slot) +
  feature toggle: **3–4 d**.
* W4 — Differential test harness + peephole constant-fold + fix discrepancies:
  **2–3 d**.
* W5 — GPU host seam + universal kernel + math-cuda wrapper + D2H wiring:
  **6–10 d** (the largest and riskiest; v1 keeps accumulation on host).

**Total: ~16–24 engineer-days**, with W1–W4 (~10–14 d) delivering a working,
validated CPU IR interpreter and W5 the GPU kernel.

**Validation strategy (bit-for-bit, on a real table):**

1. **Per-row prover diff.** In `evaluate_transitions` (evaluator.rs:79), for each
   LDE row run BOTH `air.compute_transition_prover(&ctx, base_a, ext_a)` and
   `eval_program(prog, ..., base_b, ext_b)`, and `assert_eq!` the base and ext
   buffers element-by-element. Gate behind a `debug-checks`-style feature so it's
   on in tests, off in production. Run against the existing test
   `cargo test --release -p lambda-vm-prover test_prove_elfs_test_sb_sh_8`
   (from project memory) for the CPU table (which exercises ADD/IS_BIT/LogUp).
2. **Per-constraint verifier diff.** At verifier.rs:209 compare
   `air.compute_transition(&ctx)` vs the verifier interpreter at the single OOD
   point; `assert_eq!` the full `Vec<FieldElement<E>>`. Cheapest oracle (one
   point).
3. **End-to-end.** With the interpreter as the live path, run the full
   prove→verify test suite; a passing verify is the strongest equivalence check
   (the composition poly and FRI depend on every `Cᵢ`). Run across all tables
   (CPU, MEMW, LOAD, DECODE, MUL, BRANCH, REGISTER, PAGE, BITWISE, LT, HALT) so
   every constraint shape and every `Packing`/`Multiplicity` variant is covered.
4. **GPU diff.** Compare `try_eval_program_gpu` output against the CPU
   interpreter output (not the boxed path) element-wise, reusing the
   `test-cuda-faults` style harness (gpu_lde.rs:1001) to also exercise the
   CPU-fallback path.
5. **Node-count instrumentation.** Log `prog.nodes.len()` per table to size GPU
   scratch and confirm hash-consing is effective (risk 4/5/6).

---

## Appendix — full constraint enumeration (verified by reading every body)

**33 algebraic `impl TransitionConstraint<GoldilocksField, GoldilocksExtension>`
structs + 2 framework LogUp `TransitionConstraintEvaluator` structs.** Every one
uses ONLY capturable ops: field `+ - *` and negation, `FieldElement::from(u64/
i64)`, `one()`/`zero()`, `.clone()`, `get_main_evaluation_element`/
`get_aux_evaluation_element`, and `to_extension()`. **Zero** uses of `.inv()`,
`.pow()`, `/`, `.sqrt()`, field-value conditionals, or data-dependent loops.
Every conditional branches on **metadata** (`carry_idx`, `is_sender`, kind
enums), never on a field value. Helper fns (`carry_chain`,
`compute_multiplicity_from_step`, `compute_fingerprint_from_step`) contain only
statically-bounded loops. → For SymField this confirms the `IsField` impl needs
only add/sub/mul/neg/from(u64)/one/zero (+ `to_extension`/`embed`) to be
functional; `inv`/`pow`/`div`/real-`ByteConversion` can be `unimplemented!()`
stubs because no body invokes them.

Algebraic structs (file:line):
- prover/src/constraints/cpu.rs: ProductZeroConstraint:96, Arg2ExclusiveConstraint:132,
  MemFlagsBitConstraint:168, RegNotReadIsZeroConstraint:211, Arg2Constraint:266,
  RvdEqResConstraint:331, BranchRvdConstraint:422, BranchCondConstraint:464,
  NextPcAddConstraint:546
- prover/src/constraints/templates.rs: IsBitConstraint:80, AddConstraint:470
  (AddOperand/AddLinearTerm with i64 coeffs → `from(i64)` Const nodes)
- prover/src/tables/: BranchConstraint(branch.rs:519), CommitConstraint(commit.rs:837),
  Cpu32Constraint(cpu32.rs:645), DvrmConstraint(dvrm.rs:1219), EqXorConstraint(eq.rs:262),
  MulZeroConstraint(ec_scalar.rs:291), ConvCarry(ecdas.rs:363), ColIsZero(ecdas.rs:402),
  MulZero(ecdas.rs:426), ConvCarry(ecsm.rs:663), ColIsZero(ecsm.rs:698),
  CarryBit(ecsm.rs:791), OverflowRequired(ecsm.rs:816),
  KeccakAddressNoOverflowConstraint(keccak.rs:503), LoadConstraint(load.rs:572),
  LtConstraint(lt.rs:536), MemwConstraint(memw.rs:921), MemwAlignedConstraint(memw_aligned.rs:708),
  MemwRegisterMuSumIsBit(memw_register.rs:388), MulConstraint(mul.rs:847),
  ShiftConstraint(shift.rs:914), StoreConstraint(store.rs:282)

LogUp framework (lookup.rs): LookupBatchedTermConstraint:1741
(`c·fp_a·fp_b − sign_a·m_a·fp_b − sign_b·m_b·fp_a`, degree 3),
LookupAccumulatedConstraint:1868 (running sum over acc col at row 0 AND row 1,
1–2 absorbed interactions, degree 2–3).

**Multi-kind dispatch structs — IMPORTANT for the IR.** Several "structs" are
really one type that, via a kind-enum matched on **metadata at capture time**,
implements many distinct constraint kinds; each kind must capture to **its own
IR root** (the capture pass iterates the boxed objects, and each boxed object's
`constraint_idx()` already gives it a distinct root slot, so this falls out
naturally — but the plan's `roots` count is driven by `num_transition_constraints`,
not by the 33 struct count): ShiftConstraint(7 kinds), Cpu32Constraint(8),
LtConstraint(6), LoadConstraint(6), MulConstraint(6), DvrmConstraint(11),
BranchConstraint(5), MemwConstraint(3), MemwAlignedConstraint(3),
StoreConstraint(2). The total transition-constraint *count* (and thus root
count) is therefore well above 33; the IR's `roots` vector is sized by
`air.num_transition_constraints()` (traits.rs:286), which the capture pass
already respects by writing `roots[constraint_idx()]`.
