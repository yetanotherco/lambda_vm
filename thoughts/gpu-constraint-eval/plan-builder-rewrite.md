# Plan: Capture-Method Rewrite ("Change the constraints") for GPU-ready STARK constraint evaluation

> Approach: rewrite each transition constraint so it *emits* its polynomial into a
> builder/capture abstraction once at setup, producing a flat single-field
> Goldilocks IR, then interpret that IR on CPU (prover over the LDE coset; verifier
> at the OOD point) and later on GPU. This is the head-to-head sibling of the
> "wrap the field type / shadow `IsField`" approach.

All file/line references below were read and verified against the working tree
(branch `main`) unless explicitly marked `? INFERRED` or `✗ UNCERTAIN`.

---

## 1. Overview & end-state

After this change, every table's transition constraints are *captured once* into a
flat per-table IR program (`TableProgram`) at AIR-construction time. The
per-row/per-OOD hot path no longer dispatches through
`Vec<Box<dyn TransitionConstraintEvaluator>>` calling a generic `evaluate<FF,EE>`;
instead `air.compute_transition_prover` (prover) and `air.compute_transition`
(verifier) call a single **interpreter** that walks the IR against the current
`Frame`/`TableView`, writing each constraint's scalar `Cᵢ` into the existing
`base_evals`/`ext_evals` buffers. The accumulation `Σ βᵢ·Cᵢ·Zᵢ⁻¹ + boundary`,
zerofiers, and `ZerofierEvaluations` machinery in
`ConstraintEvaluator::evaluate_transitions` are **untouched** — the IR/interpreter
only replaces the step that produces each `Cᵢ`. Because the IR is a flat array of
Goldilocks-typed ops, the same bytes feed a single universal Goldilocks
interpreter CUDA kernel, dispatched through the existing TypeId+transmute seam used
by `gpu_lde.rs`.

```
                          ┌─────────────── setup (once per AIR) ──────────────┐
 constraint structs  ──►  capture(&mut IrBuilder)  ──►  TableProgram (flat IR) │
 (IsBit, Add, Mul…)       (column reads/+/-/* → IR nodes)   { ops, consts,     │
                                                              emits, n_dim1 }   │
                          └───────────────────────────────────────────────────┘
                                            │ stored in the AIR
                ┌───────────────────────────┼───────────────────────────────┐
                ▼ prover, per LDE row        ▼ verifier, at OOD z             ▼ GPU
   interpret(program, frame_prover)  interpret(program, frame_verifier)  cuda kernel(program, lde)
        → base_evals[], ext_evals[]       → ext_evals[]                  → Cᵢ per row, per table
                │                                  │                            │
                └────── feeds unchanged ───────────┴── ConstraintEvaluator ─────┘
                        Σ βᵢ·Cᵢ·Zᵢ⁻¹ + boundary  (evaluator.rs:102-134, untouched)
```

---

## 2. The IR (concrete Rust data structures)

The IR is **single-field over Goldilocks** with explicit base (`dim1`) vs cubic-ext
(`dim3`) typing on each node, so the interpreter knows the storage width and which
arithmetic routine to use. It lives in a new module `crypto/stark/src/ir.rs`.

### 2.1 Node typing

`Dim` records the field width of a value. Goldilocks base = `Dim1` (`[u64;1]`),
`Degree3GoldilocksExtensionField` = `Dim3` (`[u64;3]`). (Verified: `IsField for
Degree3GoldilocksExtensionField { type BaseType = [FpE;3] }`,
`extensions_goldilocks.rs:277`; base = `repr(transparent)` `u64`.)

```rust
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dim { D1, D3 }

/// Index into the program's node arena. Nodes are in topological (emission) order,
/// so an interpreter can evaluate left-to-right into a value stack/arena.
pub type NodeId = u32;
```

### 2.2 Op / node enum

```rust
#[derive(Clone, Copy, Debug)]
pub enum Op {
    // ---- leaves (inputs) ----
    /// main trace column read: step offset (0 = this row, 1 = next row), column idx.
    Main   { offset: u8, col: u16 },          // always Dim1 on prover; Dim3 on verifier (see §6)
    /// aux trace column read.
    Aux    { offset: u8, col: u16 },           // Dim3 (aux is extension-valued)
    /// constant base-field element, index into `consts_d1`.
    ConstD1 { k: u32 },                         // Dim1
    /// constant ext element, index into `consts_d3`.
    ConstD3 { k: u32 },                         // Dim3  (rarely needed for algebraic; see §5)
    /// periodic column j at this row (uniform per-row input).
    Periodic { j: u16 },                        // Dim1
    /// LogUp challenge i (z=0, alpha=1 by convention), uniform per-proof.
    Challenge { i: u16 },                       // Dim3
    /// alpha power k (precomputed Σ over the proof), uniform per-proof.
    AlphaPow { k: u16 },                        // Dim3
    /// logup_table_offset uniform per-proof.
    TableOffset,                                // Dim3
    /// packing shift constant (8,16,24) — small base consts, can also be ConstD1.
    // (shifts are just ConstD1 entries; no dedicated op needed.)

    // ---- arithmetic (operands are NodeIds already emitted) ----
    Add { a: NodeId, b: NodeId },
    Sub { a: NodeId, b: NodeId },
    Mul { a: NodeId, b: NodeId },
    Neg { a: NodeId },
}

#[derive(Clone, Copy, Debug)]
pub struct Node { pub op: Op, pub dim: Dim }
```

The interpreter's typing rule for `Add/Sub/Mul`: `dim = max(dim(a), dim(b))`
(D3 > D1). A `Mul` of `D1×D3` is the cheap subfield mul (componentwise scalar,
`GoldilocksField: IsSubFieldOf<Ext3>::mul` = `[a*b0, a*b1, a*b2]`, verified
`extensions_goldilocks.rs:413`); `D3×D3` is the full cubic mul
`c0=a0b0+2(a1b2+a2b1)`, … (verified `extensions_goldilocks.rs:298-306`); `D1×D1`
is a plain `GoldilocksField::mul`. This single rule subsumes every "mixed base×ext"
case the current code handles via `IsSubFieldOf`.

### 2.3 Per-table program

```rust
pub struct TableProgram {
    pub nodes: Vec<Node>,            // topological arena
    pub consts_d1: Vec<u64>,         // deduplicated base constants
    pub consts_d3: Vec<[u64; 3]>,    // deduplicated ext constants (usually empty)
    /// emit[c] = NodeId of constraint c's root value. Length = num_transition_constraints.
    pub emits: Vec<NodeId>,
    /// First `num_base` constraints are D1 (base-field), matching
    /// `num_base_transition_constraints()`. Used to split base/ext eval buffers.
    pub num_base: usize,
    /// metadata for input plumbing / GPU upload
    pub num_main_cols: u16,
    pub num_aux_cols: u16,
    pub max_offset: u8,              // 1 (next-row) for LogUp accumulated; else 0
}
```

### 2.4 Serialization for GPU

`nodes` is `Vec<Node>`; `Node`/`Op` are `Copy` plain-old-data. For GPU we lower
`Op` to a fixed-width tagged record `struct GpuOp { tag: u32, dim: u32, a: u32, b: u32 }`
(16 bytes) — leaves pack their immediates into `a`/`b` (e.g. `Main`: a=offset,
b=col). `consts_d1`/`consts_d3` upload as `&[u64]`. This is a flat `Vec<u32>`/`Vec<u64>`
blob: H2D once per table at setup, reused for every LDE row (the kernel runs the
program per row). No per-row host work crosses the boundary — only the device-
resident LDE columns (already kept on device by R1, see `gpu_lde.rs` `gpu_main()`/
`gpu_aux()` handles) plus the uniforms (challenges, alpha powers, periodic, table
offset). `Op`'s representation is internal; we do **not** need `serde` on it unless
we want to cache programs to disk (out of scope).

---

## 3. Capture front-end — builder/capture API & object-safety (distinguishing section)

### 3.1 Object-safety decision (Question 1) — **RECOMMENDATION: non-generic `capture(&self, &mut IrBuilder)`**

The constraints are stored heterogeneously as
`Vec<Box<dyn TransitionConstraintEvaluator<F,E>>>`
(verified `traits.rs:316`, `lookup.rs:813`). A method generic over a builder type
`fn capture<AB: AirBuilder>(&self, &mut AB)` is **NOT object-safe** (generic methods
can't go through a vtable), so it could not be called on `Box<dyn …>`. Two ways out:

- **(a) non-generic `capture(&self, builder: &mut IrBuilder)` with a CONCRETE
  builder.** Object-safe. Runs **once at setup** (not in the hot loop), so the
  concrete builder costs nothing at steady state. The builder is a struct, not a
  trait. This is the minimal, lowest-risk change: the existing
  `TransitionConstraint` trait gains one object-safe method.
- (b) builder-generic `eval<AB>` (Plonky3/SP1 `AirBuilder` style). To call it
  through a boxed trait object you must either (i) monomorphize per concrete AIR
  (de-box: store constraints as a concrete `enum`/typed vec per table, a much
  bigger refactor touching every table's assembly fn and `AirWithBuses`), or
  (ii) add a non-generic shim per constraint anyway (which is just (a) again).

**Recommendation: (a).** Reasoning:
1. It is object-safe, so it drops straight into the existing
   `Box<dyn …>` storage with zero changes to how tables assemble constraints
   (`create_all_cpu_constraints`, `mul_constraints`, `dvrm_constraints`, …).
2. Capture is a one-time setup cost; there is no monomorphization win to be had at
   runtime because the runtime work is the interpreter, not the constraint body.
3. The interpreter is the single hot path; we want exactly one concrete builder so
   the IR is canonical and identical for CPU and GPU. A generic builder would let
   callers instantiate it with an "eval-directly" builder, re-introducing the very
   `IsField` trait-tower fight this approach exists to avoid.

The one real cost of (a): the builder is monomorphic on Goldilocks, so a
constraint can't be captured for a non-Goldilocks field. That is exactly the
project's constraint (base = Goldilocks, ext = degree-3 Goldilocks), so it's a
non-issue here. The generic `evaluate<FF,EE>` is retained transitionally for
migration/validation (see §6, §9) and deleted at the end.

### 3.2 The `IrBuilder` surface (Question 2)

```rust
pub struct IrBuilder {
    nodes: Vec<Node>,
    consts_d1: Vec<u64>,
    consts_d3: Vec<[u64; 3]>,
    emits: Vec<NodeId>,            // indexed by constraint_idx
    const_d1_cache: HashMap<u64, u32>,        // dedupe constants
    const_d3_cache: HashMap<[u64;3], u32>,
    num_main_cols: u16,
    num_aux_cols: u16,
    max_offset: u8,
    // CSE cache: (Op canonicalized) -> NodeId, to coalesce repeated subexpressions
    // (e.g. `one`, `1 - x`, shift consts). Optional but cheap and shrinks the IR a lot.
    cse: HashMap<NodeKey, NodeId>,
}

/// Typed handle so `+ - *` compose with compile-time dim tracking and a tiny op set.
#[derive(Clone, Copy)]
pub struct Expr { id: NodeId, dim: Dim }

impl IrBuilder {
    // ---- leaves ----
    pub fn main(&mut self, offset: u8, col: usize) -> Expr;        // Dim1
    pub fn aux(&mut self, offset: u8, col: usize) -> Expr;         // Dim3
    pub fn const_base(&mut self, v: u64) -> Expr;                  // Dim1 (dedup)
    pub fn const_signed(&mut self, v: i64) -> Expr;                // Dim1, maps i64→field
    pub fn const_ext(&mut self, v: [u64;3]) -> Expr;               // Dim3 (dedup)
    pub fn one(&mut self) -> Expr;                                 // = const_base(1)
    pub fn periodic(&mut self, j: usize) -> Expr;                  // Dim1
    pub fn challenge(&mut self, i: usize) -> Expr;                 // Dim3
    pub fn alpha_power(&mut self, k: usize) -> Expr;               // Dim3
    pub fn table_offset(&mut self) -> Expr;                        // Dim3
    pub fn bus_id(&mut self, id: u64) -> Expr;                     // = const_base(id) (α⁰ term)

    // ---- arithmetic (auto dim = max) ----
    pub fn add(&mut self, a: Expr, b: Expr) -> Expr;
    pub fn sub(&mut self, a: Expr, b: Expr) -> Expr;
    pub fn mul(&mut self, a: Expr, b: Expr) -> Expr;
    pub fn neg(&mut self, a: Expr) -> Expr;

    // ---- output ----
    /// Record that constraint `constraint_idx` evaluates to `e`.
    pub fn emit(&mut self, constraint_idx: usize, e: Expr);

    pub fn finish(self) -> TableProgram;
}
```

Notes on the surface vs the prompt's sketch:
- No `table_offset()` for periodic *exemption offsets* — those stay in the
  zerofier machinery (`transition.rs:160`), which is outside the boundary.
- `Expr` carries `dim`, so `mul(d1, d3)` is legal and lowers to the cheap subfield
  mul; `Expr` makes the constraint bodies read almost identically to today.
- CSE + constant dedup are pure size optimizations; correctness doesn't depend on
  them. (`one`, `shift_16`, `INV_SHIFT_32` recur across most bodies.)

### 3.3 Trait change

`TransitionConstraint` (`transition.rs:332`) gains:

```rust
/// Emit this constraint's polynomial into the builder. Called once at setup.
/// `builder.emit(self.constraint_idx(), root)` records the result.
fn capture(&self, builder: &mut IrBuilder);
```

`TransitionConstraintEvaluator` (`transition.rs:10`, object-safe) gains a forwarding
non-generic method:

```rust
fn capture(&self, builder: &mut IrBuilder);
```

The adapter `TransitionConstraintAdapter` (`transition.rs:395`) forwards
`capture` to `self.0.capture(builder)`. During migration the adapter keeps its
`evaluate_verifier`/`evaluate_prover` too (used by the parallel old path for
bit-for-bit validation, §12).

---

## 4. Rewriting the algebraic constraints (Question 3 + full scope)

### 4.1 Before/after: `IsBitConstraint` (`templates.rs:80-108`)

**Before** (`evaluate<F,E>`):
```rust
let x = step.get_main_evaluation_element(0, self.value_col).clone();
let one = FieldElement::<F>::one();
match self.cond_col {
    Some(cond_col) => { let cond = step.get_main_evaluation_element(0, cond_col).clone();
                        &cond * &x * (one - x) }
    None           => &x * (one - &x),
}
```
**After** (`capture`):
```rust
fn capture(&self, b: &mut IrBuilder) {
    let x   = b.main(0, self.value_col);
    let one = b.one();
    let omx = b.sub(one, x);
    let root = match self.cond_col {
        Some(c) => { let cond = b.main(0, c); let xm = b.mul(x, omx); b.mul(cond, xm) }
        None    => b.mul(x, omx),
    };
    b.emit(self.constraint_idx, root);
}
```

### 4.2 Before/after: `AddConstraint` — `AddOperand`/`AddLinearTerm` mapping (`templates.rs:359-486`)

The lo/hi-limb abstraction with i64 coefficients maps cleanly. `AddLinearTerm::eval`
(`templates.rs:164`) becomes `capture`:
```rust
impl AddLinearTerm {
    fn capture(&self, b: &mut IrBuilder) -> Expr {
        match self {
            AddLinearTerm::Column { coefficient, column } => {
                let col = b.main(0, *column);
                let k   = b.const_signed(*coefficient);   // i64 → field, was FieldElement::from(*coefficient)
                b.mul(col, k)
            }
            AddLinearTerm::Constant(v) => b.const_signed(*v),
        }
    }
}
fn eval_terms_capture(terms: &[AddLinearTerm], b: &mut IrBuilder) -> Expr {
    // empty → zero
    let mut acc = b.const_base(0);
    for t in terms { let e = t.capture(b); acc = b.add(acc, e); }
    acc
}
```
`AddOperand::eval_lo/eval_hi` → `capture_lo/capture_hi` (DWordWL reads
`main(0,start)` / `main(0,start+1)`; Linear → `eval_terms_capture`). Then
`compute_carry_0` (`templates.rs:414`):
```rust
// carry_0 = (lhs_lo + rhs_lo - sum_lo) * 2^(-32)
let inv = b.const_base(INV_SHIFT_32);                 // templates.rs:30, precomputed 2^-32
let s   = b.sub(b.add(lhs_lo, rhs_lo), sum_lo);
let c0  = b.mul(s, inv);
```
`compute_carry_1` adds `carry_0` then the same `*inv`. `compute` then folds the cond
columns (`fold(zero, +)` → chain of `add`) and emits
`cond * carry * (one - carry)` (or unconditional). **The i64 coefficients are the
only subtlety** and they vanish because `const_signed(i64)` reproduces
`FieldElement::<F>::from(i64)` exactly (the field's `From<i64>` already canonicalizes
negatives mod p). The lo/hi limb logic is pure compile-time structure; the captured
IR is a flat add/mul chain identical in value to the current `evaluate`.

### 4.3 Before/after: `ProductZeroConstraint` (`cpu.rs:96-113`)

**Before:** `step.get_main(0,col_a) * step.get_main(0,col_b)`.
**After:**
```rust
fn capture(&self, b: &mut IrBuilder) {
    let a = b.main(0, self.col_a); let c = b.main(0, self.col_b);
    let r = b.mul(a, c); b.emit(self.constraint_idx, r);
}
```

### 4.4 Before/after: a more complex algebraic constraint — `MulConstraint::RawProduct` (`mul.rs:766-844`)

This is the representative "mega-constraint": a `kind` enum dispatched in
`compute()` (`mul.rs:721`), with a convolution body whose `for k` / `for j` loops
are bounded by compile-time `i` (not data). Capturing it **runs those loops once**,
unrolling them into a flat IR chain:

```rust
// raw_product[i] - Σ_k 2^(16k) Σ_j lhs_ext[j]·rhs_ext[idx-j]
fn capture_raw_product(&self, i: usize, b: &mut IrBuilder) -> Expr {
    let lhs = [cols::LHS_0, cols::LHS_1, cols::LHS_2, cols::LHS_3].map(|c| b.main(0, c));
    let rhs = [cols::RHS_0, cols::RHS_1, cols::RHS_2, cols::RHS_3].map(|c| b.main(0, c));
    let ln  = b.main(0, cols::LHS_IS_NEGATIVE);
    let rn  = b.main(0, cols::RHS_IS_NEGATIVE);
    let sf  = b.const_base(SIGN_FILL);
    let mut lhs_ext = [b.const_base(0); 8];
    let mut rhs_ext = [b.const_base(0); 8];
    lhs_ext[..4].copy_from_slice(&lhs);  rhs_ext[..4].copy_from_slice(&rhs);
    for j in 4..8 { lhs_ext[j] = b.mul(sf, ln); rhs_ext[j] = b.mul(sf, rn); }
    let shift_16 = b.const_base(SHIFT_16);
    let mut sum = b.const_base(0);
    for k in 0..=1usize {
        let idx = 2*i + k;
        if idx < 8 {
            let mut inner = b.const_base(0);
            for j in 0..=idx { if j < 8 && idx-j < 8 {
                inner = b.add(inner, b.mul(lhs_ext[j], rhs_ext[idx-j])); } }
            sum = if k==0 { b.add(sum, inner) } else { b.add(sum, b.mul(inner, shift_16)) };
        }
    }
    let raw = b.main(0, raw_col_for(i));
    b.sub(raw, sum)
}
```
**This is the central churn-reducing insight: no algebraic body has data-dependent
control flow.** Every loop bound, conditional, and column index is a function of
`self` only. So `capture` is a *mechanical mirror* of the existing body: swap
`FieldElement` constructors for builder leaves and `+ - *` for `b.add/sub/mul`. The
`kind`-enum dispatch in `compute` becomes a `kind`-enum dispatch in `capture`.

### 4.5 Full rewrite scope (counts verified by grep + reads)

`grep -rn "impl TransitionConstraint"` across `prover/src/` yields the following
**distinct constraint structs** (each implements the user trait once; structs with a
`kind` enum produce many constraint *instances* but are ONE body to rewrite):

**`prover/src/constraints/` (11 structs):**
- `templates.rs`: `IsBitConstraint`, `AddConstraint` (+ `AddOperand`/`AddLinearTerm`
  helper enums — these get `capture` helpers, not trait impls).
- `cpu.rs`: `ProductZeroConstraint`, `Arg2ExclusiveConstraint`, `MemFlagsBitConstraint`,
  `RegNotReadIsZeroConstraint`, `Arg2Constraint`, `RvdEqResConstraint`,
  `BranchRvdConstraint`, `BranchCondConstraint`, `NextPcAddConstraint` (+ helper
  `res_word`). All small (≤ ~30-line bodies).

**`prover/src/tables/` (per verified grep, 21 impl sites across 17 files; some files
hold several structs):**
- `mul.rs` `MulConstraint` (kind enum, ~250 lines incl. helpers; convolution).
- `dvrm.rs` `DvrmConstraint` (kind enum, 11 variants; the biggest, ~1300-line file —
  body+helpers the largest single rewrite).
- `shift.rs` `ShiftConstraint` (kind enum; ~1100-line file).
- `cpu32.rs` `Cpu32Constraint` (kind enum; ~845-line file).
- `memw.rs` `MemwConstraint`; `memw_aligned.rs` `MemwAlignedConstraint`;
  `memw_register.rs` `MemwRegisterMuSumIsBit`.
- `load.rs` `LoadConstraint`; `store.rs` `StoreConstraint`.
- `lt.rs` `LtConstraint`; `eq.rs` `EqXorConstraint`.
- `branch.rs` `BranchConstraint`; `commit.rs` `CommitConstraint`.
- `keccak.rs` `KeccakAddressNoOverflowConstraint` (one small struct). NOTE: keccak's
  51 constraints are mostly **reused `AddConstraint` instances** (`from_dword_bl` +
  `constant` + `from_dword_hl`, verified `keccak.rs:545-557`) — so keccak adds almost
  no rewrite cost once `AddConstraint::capture` exists, and its program is small (no
  GPU register-pressure risk).
- `ec_scalar.rs` `MulZeroConstraint`.
- `ecsm.rs`: `ConvCarry`, `ColIsZero`, `CarryBit`, `OverflowRequired` (4 structs).
- `ecdas.rs`: `ConvCarry`, `ColIsZero`, `MulZero` (3 structs).

**Authoritative count (enumeration-verified): 33 algebraic
`impl TransitionConstraint` structs across 19 files + 2 framework LogUp
`TransitionConstraintEvaluator` structs (§5).** Breakdown:
- `prover/src/constraints/cpu.rs` (9): ProductZero, Arg2Exclusive, MemFlagsBit,
  RegNotReadIsZero, Arg2, RvdEqRes, BranchRvd, BranchCond, NextPcAdd.
- `prover/src/constraints/templates.rs` (2): IsBit, Add (Add carries the
  AddOperand/AddLinearTerm combinators — the trickiest single rewrite, §4.2).
- `prover/src/tables/` (22): Branch, Commit, Cpu32, Dvrm, EqXor, MulZero(ec_scalar),
  ConvCarry+ColIsZero+MulZero(ecdas, 3), ConvCarry+ColIsZero+CarryBit+OverflowRequired
  (ecsm, 4), KeccakAddressNoOverflow, Load, Lt, Memw, MemwAligned, MemwRegisterMuSumIsBit,
  Mul, Shift, Store.

**Scope driver — multi-kind dispatch structs** (one struct, a `kind` enum + a
`compute()` helper; each kind is a separate constraint *instance* needing its own
capture path). Verified kind counts: Dvrm(11), Cpu32(8), Shift(7), Lt(6), Load(6),
Mul(6), Branch(5), Memw(3), MemwAligned(3), Store(2). Dvrm/Cpu32/Shift dominate.
**Rough total evaluate/compute body LOC ≈ 600-800 across the 19 files** — far less
than the raw file sizes suggest, because the kind-enum bodies are short matches that
delegate to `compute()`, and the heavy loops (carry chains, raw-product convolution,
shift formulas) are **statically bounded / metadata-driven**, so they unroll into
builder calls at capture time without per-kind hand-coding of each iteration.
(I read `mul.rs`/`dvrm.rs` bodies in full; the rest share the single-`evaluate<F,E>`
→`compute()` pattern, kind counts enumeration-verified.)

---

## 5. Rewriting the LogUp / extension framework constraints (Question 4 — the crux)

These two live in `crypto/stark/src/lookup.rs` and are the only constraints that use
extension arithmetic, challenges, alpha powers, and (for the accumulated one)
next-row reads. They are **not** `TransitionConstraint` impls — they directly
implement the object-safe `TransitionConstraintEvaluator` (`lookup.rs:1741`,
`lookup.rs:1868`). So for these we write `capture` directly on the evaluator impl.
I read both bodies in full; here is how each maps.

### 5.1 Fingerprint, multiplicity, sign — the shared pieces

`compute_fingerprint_from_step` (`lookup.rs:1689-1709`) builds
`z − (bus_id + Σ α^k · vₖ)` where `vₖ` are the packed bus elements. In IR:

```rust
// fingerprint(interaction) -> Expr (Dim3, because z and alpha powers are Dim3)
fn capture_fingerprint(b: &mut IrBuilder, bi: &BusInteraction) -> Expr {
    let z = b.challenge(0);                              // rap_challenges[0]
    // α⁰ term: bus_id is a base const, added directly (matches lookup.rs:1697)
    let mut lc = b.bus_id(bi.bus_id);                    // Dim1 const, promoted on first add to Dim3
    let mut alpha_idx = 1usize;                          // α⁰ handled, start at α¹ (lookup.rs:1698)
    for bv in &bi.values {
        alpha_idx += capture_busvalue_fingerprint(b, bv, alpha_idx, &mut lc);
    }
    b.sub(z, lc)                                         // z - lc  (Dim3)
}
```

`BusValue::accumulate_fingerprint_from_step` (`lookup.rs:738-796`) and
`Packing::accumulate_fingerprint_with` (`lookup.rs:272-369`) are the packing
formulas. They are pure compile-time structure (the `match self { Packing::Word2L =>
h0 + 2^16·h1, … }`), so capturing them unrolls the same way as §4.4:

```rust
fn capture_busvalue_fingerprint(b: &mut IrBuilder, bv: &BusValue,
                                alpha_off: usize, lc: &mut Expr) -> usize {
    match bv {
        BusValue::Packed { start_column, packing } => {
            // mirror accumulate_fingerprint_with: e.g. Word2L:
            //   combined = col[start] + col[start+1]·shift_16   (Dim1)
            //   *lc += combined · alpha_powers[alpha_off]       (Dim1 · Dim3 -> Dim3)
            let elems = capture_packing(b, *packing, *start_column); // Vec<Expr> (Dim1)
            for (i, e) in elems.iter().enumerate() {
                let ap  = b.alpha_power(alpha_off + i);            // Dim3
                let t   = b.mul(*e, ap);                           // Dim1·Dim3 -> Dim3
                *lc = b.add(*lc, t);
            }
            packing.num_bus_elements()
        }
        BusValue::Linear(terms) => {
            // result = Σ coeff·col + const   (Dim1), then  *lc += result·α^alpha_off
            let mut r = b.const_base(0);
            for t in terms { match t {
                LinearTerm::Column{coefficient, column}
                | LinearTerm::ColumnUnsigned{coefficient, column}=> {
                    let col=b.main(0,*column); let k=b.const_signed(*coefficient as i64);
                    r=b.add(r, b.mul(col,k)); }
                LinearTerm::Constant(v)=> r=b.add(r, b.const_signed(*v)),
            }}
            let ap=b.alpha_power(alpha_off); *lc=b.add(*lc, b.mul(r, ap));
            1
        }
    }
}
```
> **Honesty note on the runtime zero-skip:** the current code skips the
> `result · α` multiply when `result == 0` *on that row* (`lookup.rs:675-677`,
> `790-792`). That is a *data-dependent* optimization the IR **cannot** reproduce —
> the IR is row-agnostic. The IR always emits the multiply. This is the one place
> the capture approach is strictly less optimal than the current per-row code: a
> few extra D1×D3 muls per row for bus elements that happen to be zero. It does
> **not** change the result (adding `0·α` is a no-op), only cost. Quantify in
> validation; likely negligible vs. the dispatch savings. (✓ VERIFIED the skip
> exists and is value-preserving.)

`compute_multiplicity_from_step` (`lookup.rs:1679-1684`) = `Multiplicity::evaluate_with`
(`lookup.rs:1252-1282`): `One→1`, `Column→col`, `Sum→a+b`, `Negated→1-col`,
`Diff→a-b`, `Sum3→a+b+c`, `Linear→Σ`. All Dim1, captured as add/sub/mul chains.

**The sign** (`is_sender`) is a **compile-time bool on the interaction**, so it is
resolved during capture by choosing `add` vs `sub` (or wrapping in `neg`) — never an
IR value. This matches the current "conditional negation instead of E×E sign
multiplication" (`lookup.rs:1779-1790`).

### 5.2 `LookupBatchedTermConstraint::capture` (was `lookup.rs:1754-1831`)

Formula (verified `lookup.rs:1791`):
`c·fp_a·fp_b − sign_a·m_a·fp_b − sign_b·m_b·fp_a`.

```rust
fn capture(&self, b: &mut IrBuilder) {
    let c    = b.aux(0, self.term_column_idx);          // Dim3
    let fp_a = capture_fingerprint(b, &self.interaction_a);
    let fp_b = capture_fingerprint(b, &self.interaction_b);
    let m_a  = capture_multiplicity(b, &self.interaction_a.multiplicity);  // Dim1
    let m_b  = capture_multiplicity(b, &self.interaction_b.multiplicity);
    let term_a = b.mul(m_a, fp_b);                       // Dim1·Dim3 -> Dim3
    let term_a = if self.interaction_a.is_sender { term_a } else { b.neg(term_a) };
    let term_b = b.mul(m_b, fp_a);
    let term_b = if self.interaction_b.is_sender { term_b } else { b.neg(term_b) };
    let main = b.mul(b.mul(c, fp_a), fp_b);
    let root = b.sub(b.sub(main, term_a), term_b);
    b.emit(self.constraint_idx, root);
}
```
Clean. Degree 3, all Dim3 at the top, exactly mirrors the read body.

### 5.3 `LookupAccumulatedConstraint::capture` (was `lookup.rs:1881-2005`) — the messy one

This is the only constraint that reads **two row offsets** (`acc_curr` at offset 0,
`acc_next` and the term columns at offset 1) — verified
`first_step.get_aux(0, acc)` / `second_step.get_aux(0, …)` where `first_step =
frame.get_evaluation_step(0)` and `second_step = frame.get_evaluation_step(1)`
(`lookup.rs:1971-1972`, `1899-1905`). The IR addresses next-row values with
`b.aux(1, col)` — this is exactly why `Op::Main/Aux` carry an `offset: u8` and why
the program records `max_offset` (the interpreter must fill a 2-step frame for these
tables; the prover already builds frames with `offsets = [0,1]`, see
`AirWithBuses` context `transition_offsets: vec![0,1]`, `lookup.rs:909`).

```rust
fn capture(&self, b: &mut IrBuilder) {
    let acc_curr = b.aux(0, self.acc_column_idx);                 // offset 0
    let acc_next = b.aux(1, self.acc_column_idx);                 // offset 1  <-- next row
    // terms_sum over committed term columns at offset 1 (lookup.rs:1903)
    let mut terms = b.const_base(0);
    for i in 0..self.num_term_columns { terms = b.add(terms, b.aux(1, i)); }
    // delta = acc_next - acc_curr - terms_sum + L/N
    let off = b.table_offset();                                   // logup_table_offset (Dim3)
    let delta = b.add(b.sub(b.sub(acc_next, acc_curr), terms), off);
    match self.absorbed.len() {
        1 => {  // delta·f - sign·m            (lookup.rs:1932)
            let f = capture_fingerprint_at(b, &self.absorbed[0], /*offset*/1);
            let m = capture_multiplicity_at(b, &self.absorbed[0].multiplicity, 1);
            let mt = if self.absorbed[0].is_sender { m } else { b.neg(m) };
            let root = b.sub(b.mul(delta, f), mt);
            b.emit(self.constraint_idx, root);
        }
        2 => {  // delta·f1·f2 - sign1·m1·f2 - sign2·m2·f1   (lookup.rs:1957)
            let f1=capture_fingerprint_at(b,&self.absorbed[0],1);
            let f2=capture_fingerprint_at(b,&self.absorbed[1],1);
            let m1=capture_multiplicity_at(b,&self.absorbed[0].multiplicity,1);
            let m2=capture_multiplicity_at(b,&self.absorbed[1].multiplicity,1);
            let t1=b.mul(m1,f2); let t1=if self.absorbed[0].is_sender{t1}else{b.neg(t1)};
            let t2=b.mul(m2,f1); let t2=if self.absorbed[1].is_sender{t2}else{b.neg(t2)};
            let root=b.sub(b.sub(b.mul(b.mul(delta,f1),f2),t1),t2);
            b.emit(self.constraint_idx, root);
        }
        _ => unreachable!(),
    }
}
```
> **The messiness, stated honestly:**
> 1. `capture_fingerprint`/`capture_multiplicity` need an **offset parameter** because
>    the absorbed interactions read columns at the *next* row (`second_step`,
>    `lookup.rs:1919-1946`), whereas the batched-term constraint reads the *current*
>    row. The fingerprint/packing capture helpers (§5.1) must thread `offset: u8`
>    through to every `b.main(offset, …)`/`b.aux(offset, …)`. This is a real but
>    mechanical generalization (one extra arg).
> 2. The `1` vs `2` absorbed cases have different degree (2 vs 3) and different
>    formulas; both must be captured (matches the existing `match absorbed.len()`).
> 3. `logup_table_offset` becomes the `TableOffset` uniform leaf (§8). It is `L/N`,
>    a single Dim3 value computed in `ConstraintEvaluator::new` (`evaluator.rs:199`)
>    and passed via the context — already a per-proof uniform.
>
> **Verdict:** LogUp maps to the builder *cleanly but with one wart* — the per-row
> zero-skip (§5.1) is lost, and the fingerprint helpers must be offset-parameterized.
> Neither blocks the approach; both are mechanical. This is materially **less messy**
> than fighting `IsField` to make a shadow-field type carry the same z/α/alpha-power
> uniforms through `compute_fingerprint_from_step`'s generic `<A: IsSubFieldOf<B>>`
> signature (the sibling approach's burden). The deciding factor leans toward this
> approach because the capture is a near-verbatim transcription of the existing,
> already-factored helpers.

---

## 6. CPU interpreter & the boundary (Question 5)

### 6.1 Where it slots in

The boundary is exactly `air.compute_transition_prover` (prover, `traits.rs:254`)
and `air.compute_transition` (verifier, `traits.rs:223`). Today both loop over
`transition_constraints()` calling `evaluate_prover`/`evaluate_verifier`. After the
rewrite, `AirWithBuses` (the only production AIR, `lookup.rs:964`) overrides both to
call the interpreter against its stored `TableProgram`:

```rust
fn compute_transition_prover(&self, ctx, base_evals, ext_evals) {
    interpret_prover(&self.program, ctx, base_evals, ext_evals);
}
fn compute_transition(&self, ctx) -> Vec<FieldElement<E>> {
    let mut ext = vec![FieldElement::zero(); self.num_transition_constraints()];
    interpret_verifier(&self.program, ctx, &mut ext);
    ext
}
```

`ConstraintEvaluator::evaluate_transitions` (`evaluator.rs:79-135`) is **unchanged**:
it still calls `air.compute_transition_prover(&ctx, base_buf, transition_buf)`
(`evaluator.rs:100`) and accumulates with zerofiers (`evaluator.rs:102-134`). The IR
sits entirely inside the AIR's override.

### 6.2 Base vs ext handling — two interpreters, shared walk

- **Prover** frame is `Frame<F=Goldilocks, E=Ext3>`: `main` reads are **Dim1**
  (base), `aux` reads are **Dim3**. So `interpret_prover` evaluates each node into
  either a `u64` (D1) or `[u64;3]` (D3) slot. The first `num_base` constraints are
  D1-rooted and written into `base_evals: &mut [FieldElement<F>]`; the rest are
  D3-rooted into `ext_evals[num_base..]`. This reproduces the existing F×E split
  (`evaluator.rs:104-114`, `transition.rs:439-458`). Verified: base constraints
  must be the first `num_base_transition_constraints()` and the LogUp constraints
  are appended last (`lookup.rs:857`, `traits.rs:244`).
- **Verifier** frame is `Frame<E=Ext3, E=Ext3>`: there is no base field; every value
  is Dim3 (the verifier "works with a frame that contains only elements from the
  extension", `traits.rs:69-71`). So `interpret_verifier` runs the *same* node walk
  but treats `Main` reads as Dim3 (the column value is already an ext element) and
  every op as Dim3. The IR's per-node `dim` is the prover's typing; the verifier
  simply promotes D1 leaves to D3. One IR, two interpreters differing only in leaf
  loading and whether D1 storage is used.

Implementation: a value arena `Vec<Val>` where `enum Val { D1(u64), D3([u64;3]) }`,
or two parallel arenas (`Vec<u64>` for D1 ids, `Vec<[u64;3]>` for D3 ids) keyed by
node dim. Arithmetic dispatches on `(dim(a),dim(b))` using the raw Goldilocks ops
(`GoldilocksField::add/mul`, the cubic-ext formulas). Reuse the per-thread buffer
pattern already in `evaluate_transitions` (`map_init`, `evaluator.rs:142`): the value
arena is a per-thread scratch `Vec` sized to `program.nodes.len()`.

### 6.3 Fate of `TransitionConstraintAdapter` (Question 5)

**End state:** `TransitionConstraint::evaluate<FF,EE>` and the adapter's
`evaluate_prover`/`evaluate_verifier` are **deleted**. The user trait keeps
`degree/constraint_idx/period/offset/exemptions/end_exemptions` + the new `capture`.
`TransitionConstraintEvaluator` keeps the zerofier/degree/index methods + `capture`,
and **drops** `evaluate_prover`/`evaluate_verifier` (the per-row eval path no longer
goes through the trait object — it goes through the interpreter). The adapter shrinks
to a forwarder for `capture` and the metadata methods.

**Transitional:** during migration we keep both `evaluate*` and `capture` so the old
per-row path and the new interpreter can run in parallel and be diff'd
(§9, §12). Only after every table validates bit-for-bit do we delete the old methods.

---

## 7. GPU interpreter sketch (Question 7)

Model on the `gpu_lde.rs` seam: TypeId checks gate entry, `repr(transparent)`/`[u64;3]`
layout lets us reinterpret `FieldElement` slices as raw `u64`, and a `_keep` device
handle holds the LDE columns resident from R1.

- **Entry/dispatch.** A new `try_compute_transition_gpu<F,E>(program, lde_trace, uniforms)`
  guarded by `TypeId::of::<F>()==Goldilocks && TypeId::of::<E>()==Ext3` and an
  lde-size threshold (mirror `check_base_layout`, `gpu_lde.rs:106`). Returns
  `Option<Vec<FieldElement<E>>>` of length `num_transition · lde_size` (the per-row
  `Cᵢ` values), or `None` to fall back to the CPU interpreter. It is called from the
  AIR's `compute_transition_prover` analog — but note the current
  `evaluate_transitions` calls `compute_transition_prover` *per row*; for GPU we add a
  batched override that produces all rows at once and feeds the accumulation loop
  (this is a small refactor of `evaluate_transitions` to optionally accept a
  precomputed `Cᵢ` matrix; the accumulation stays on whichever side is cheaper).
  `✗ UNCERTAIN`: exact placement of the batched call (per-row vs whole-table) needs a
  design pass — the cleanest is a new `air.compute_transitions_batched(lde) ->
  Option<Cmatrix>` that `evaluate_transitions` tries before the per-row loop.
- **What crosses the boundary (once per table).** The program blob (`GpuOp[]` +
  `consts_d1` + `consts_d3`), the uniforms (challenges, alpha_powers, periodic
  columns, table_offset, packing shifts-as-consts). The LDE main/aux columns are
  already on device (`lde_trace.gpu_main()`/`gpu_aux()`, `gpu_lde.rs:832,915`). No
  per-row H2D.
- **Kernel.** One `interpret_transition_ext3` kernel, one thread per LDE row
  (strided like `barycentric_*_strided`). Each thread walks `nodes` left-to-right
  into a small per-thread register/local array indexed by NodeId (program is tiny —
  hundreds of nodes — fits in local/shared memory), loading `Main/Aux` from the
  resident LDE at `(row + offset·stride)`, doing D1/D3 ops with the existing device
  primitives (`gl_add/gl_mul/gl_sub` and `ext3_add/ext3_mul/ext3_sub`, verified
  present `device.rs:124-131`). Writes `Cᵢ` for each emit. Because the program is
  uniform across rows, this is an embarrassingly parallel single-field kernel — the
  whole point of the IR. New `.cu` file `transition_interp.cu` + `Backend` field +
  `load_function` (mirror `device.rs:227-229`).
- **Fallback.** Any unsupported op/dim, sub-threshold size, or non-Goldilocks → CPU
  interpreter (identical IR, identical result). Same `Option`-returning contract as
  every `try_*` in `gpu_lde.rs`.

---

## 8. Inputs plumbing (Question 6)

The interpreter needs the per-proof/per-row uniforms that today live in
`TransitionEvaluationContext` (`traits.rs:72-93`). They become **leaf opcodes** read
from a uniform table the interpreter is handed alongside the program:

| Current source (verified) | IR leaf | Const-vs-varies |
|---|---|---|
| `periodic_values[j]` (`evaluator.rs:88-90`, filled per row) | `Op::Periodic{j}` | varies per row (Dim1) |
| `rap_challenges[i]` (`ctx`, `traits.rs:80`) | `Op::Challenge{i}` | per proof (Dim3) |
| `logup_alpha_powers[k]` (precomputed `evaluator.rs:53`) | `Op::AlphaPow{k}` | per proof (Dim3) |
| `logup_table_offset` (`evaluator.rs:199`, `traits.rs:82`) | `Op::TableOffset` | per proof (Dim3) |
| `packing_shifts` (8/16/24, `lookup.rs:53`) | `Op::ConstD1` | program constant |

The interpreter signature:
```rust
fn interpret_prover(prog: &TableProgram, ctx: &TransitionEvaluationContext<F,E>,
                    base: &mut [FieldElement<F>], ext: &mut [FieldElement<E>]);
```
pulls `frame`, `periodic_values`, `rap_challenges`, `logup_alpha_powers`,
`logup_table_offset` straight out of `ctx` (already plumbed through
`evaluate_transitions`, `evaluator.rs:92-99`). **No new plumbing into the
evaluator** — the context already carries everything; we only change what *consumes*
it. For GPU, these uniforms upload once per table (challenges/alpha/offset are
per-proof; periodic is `num_periodic · lde_size` Dim1, uploaded once).

---

## 9. Coexistence & migration (Question 9)

- **Table-by-table migration is fully supported.** The interpreter dispatch is on the
  AIR. We add `capture` to all constraints up front (it can default to a `todo!()`
  or, better, a generic auto-capture, see below), but flip an AIR to *use* the
  interpreter independently. Concretely, `AirWithBuses` gets an `Option<TableProgram>`:
  when `Some`, `compute_transition_prover` interprets; when `None`, it falls back to
  the existing `transition_constraints().iter()…evaluate_prover` loop (the current
  `traits.rs:267-269` default). So a table is "migrated" by building its program in
  `AirWithBuses::new`; unmigrated tables keep the old path verbatim.
- **Auto-capture bridge (optional but valuable):** because every algebraic body is
  data-independent, we *could* provide a blanket `capture` that runs the existing
  generic `evaluate` against a recording `TableView` whose elements are IR nodes —
  i.e. a `TableView<IrField, IrField>` where `IrField` is a field-like type whose
  `add/mul` push IR nodes. **However** that is precisely the "shadow IsField" trick
  the sibling approach owns, and making `IrField: IsField` is the trait-tower fight
  we're avoiding. So for *this* approach we hand-write `capture` per struct and do
  **not** rely on an auto-bridge. (Mentioned for completeness; explicitly rejected
  here to keep the approaches distinct.)
- **Feature/TypeId gating:** GPU path behind the existing `cuda` feature + TypeId
  guard (no new feature). CPU interpreter is unconditional. A `LAMBDA_VM_USE_IR`
  env/feature can force the old path for A/B benchmarking during migration.

---

## 10. Exhaustive file-by-file change list

**New files:**
- `crypto/stark/src/ir.rs` — `Dim`, `NodeId`, `Op`, `Node`, `Expr`, `IrBuilder`
  (full API §3.2), `TableProgram`, const/CSE dedup. `~400 LOC`.
- `crypto/stark/src/interpreter.rs` — `interpret_prover`, `interpret_verifier`,
  `Val` arena, op dispatch, D1/D3 raw arithmetic helpers. `~300 LOC`.
- `crypto/math-cuda/src/transition_interp.rs` + `cuda/transition_interp.cu` — GPU
  kernel + host wrapper `compute_transition_ext3`. `~400 LOC + kernel`.
- `crypto/stark/src/gpu_transition.rs` — `try_compute_transition_gpu<F,E>` dispatch
  (TypeId guard, blob upload, fallback). `~250 LOC`. (Or fold into `gpu_lde.rs`.)

**Modified — framework:**
- `crypto/stark/src/constraints/transition.rs`:
  - `TransitionConstraint`: add `fn capture(&self, &mut IrBuilder)`; delete
    `evaluate<FF,EE>` (end state).
  - `TransitionConstraintEvaluator`: add object-safe `fn capture(&self, &mut
    IrBuilder)`; delete `evaluate_prover`/`evaluate_verifier` (end state).
  - `TransitionConstraintAdapter`: forward `capture`; drop `evaluate_*`.
- `crypto/stark/src/lookup.rs`:
  - `LookupBatchedTermConstraint`: replace `evaluate_verifier` body with `capture`
    (§5.2). `LookupAccumulatedConstraint`: replace with `capture` (§5.3).
  - Add offset-parameterized capture helpers mirroring
    `compute_fingerprint_from_step` (1689), `compute_multiplicity_from_step` (1679),
    `BusValue::accumulate_fingerprint_from_step` (738), `Packing::accumulate_*` (272),
    `Multiplicity::evaluate_with` (1252).
  - `AirWithBuses`: add `program: Option<TableProgram>`; build it in `new`
    (`lookup.rs:848`) by `capture`-ing every constraint after assembly; override
    `compute_transition_prover`/`compute_transition` to interpret.
- `crypto/stark/src/traits.rs`: optionally add
  `fn compute_transitions_batched(&self, lde) -> Option<Vec<…>>` default `None`
  (GPU batched hook for `evaluate_transitions`).
- `crypto/stark/src/constraints/evaluator.rs`: (optional) try the batched GPU hook
  before the per-row loop; otherwise **unchanged**.
- `crypto/stark/src/lib.rs` / `crypto/math-cuda/src/lib.rs`: module decls.

**Modified — every constraint struct (`capture` body, delete `evaluate`):**
- `prover/src/constraints/templates.rs`: `IsBitConstraint`, `AddConstraint`,
  `AddOperand::capture_lo/hi`, `AddLinearTerm::capture`, `eval_terms`→`capture`.
- `prover/src/constraints/cpu.rs`: `ProductZeroConstraint`, `Arg2ExclusiveConstraint`,
  `MemFlagsBitConstraint`, `RegNotReadIsZeroConstraint`, `Arg2Constraint`,
  `RvdEqResConstraint`, `BranchRvdConstraint`, `BranchCondConstraint`,
  `NextPcAddConstraint`, `res_word`→capture helper.
- `prover/src/tables/`: `mul.rs (MulConstraint+compute helpers)`, `dvrm.rs
  (DvrmConstraint)`, `shift.rs (ShiftConstraint)`, `cpu32.rs (Cpu32Constraint)`,
  `memw.rs (MemwConstraint)`, `memw_aligned.rs (MemwAlignedConstraint)`,
  `memw_register.rs (MemwRegisterMuSumIsBit)`, `load.rs (LoadConstraint)`,
  `store.rs (StoreConstraint)`, `lt.rs (LtConstraint)`, `eq.rs (EqXorConstraint)`,
  `branch.rs (BranchConstraint)`, `commit.rs (CommitConstraint)`,
  `keccak.rs (one struct)`, `ec_scalar.rs (MulZeroConstraint)`,
  `ecsm.rs (ConvCarry, ColIsZero, CarryBit, OverflowRequired)`,
  `ecdas.rs (ConvCarry, ColIsZero, MulZero)`.

**Key new type/function signatures (summary):**
```rust
pub struct TableProgram { nodes, consts_d1, consts_d3, emits, num_base, … }
pub struct IrBuilder { … }   impl IrBuilder { main/aux/const_*/periodic/challenge/alpha_power/table_offset/add/sub/mul/neg/emit/finish }
pub fn interpret_prover(&TableProgram, &TransitionEvaluationContext, &mut[FE<F>], &mut[FE<E>]);
pub fn interpret_verifier(&TableProgram, &TransitionEvaluationContext, &mut[FE<E>]);
trait TransitionConstraint { fn capture(&self, &mut IrBuilder); … }            // generic evaluate removed
trait TransitionConstraintEvaluator { fn capture(&self, &mut IrBuilder); … }   // evaluate_* removed
pub(crate) fn try_compute_transition_gpu<F,E>(&TableProgram, &LDETraceTable<F,E>, …) -> Option<Vec<FieldElement<E>>>;
```

---

## 11. Risks & unknowns, ranked (brutally honest)

1. **Breadth of the manual rewrite (33 structs / 19 files, ~600-800 LOC of bodies).**
   This is the dominant cost and risk. Every body is mechanical but the multi-kind
   mega-constraints (`dvrm` 11 kinds, `cpu32` 8, `shift` 7) have many capture paths
   that are easy to transcribe subtly wrong. *Mitigation:* the bit-for-bit
   parallel-path validation (§12) catches any divergence immediately; migrate one
   table at a time behind the `Option<TableProgram>` flag.
2. **LogUp `LookupAccumulatedConstraint` offset handling + lost per-row zero-skip.**
   The fingerprint helpers must thread `offset` (next-row reads, §5.3) and the IR
   cannot do the data-dependent `result==0` multiply-skip (§5.1). Correctness is
   safe (value-preserving); the cost is a few extra D1×D3 muls/row. *Risk:* the
   skip might matter more than expected on wide-bus tables; measure before deleting
   the old path. `? INFERRED` it's negligible vs. dispatch savings — not yet
   measured.
3. **Verifier-side typing (`Main` reads are Dim3 in the verifier).** The IR's
   per-node `dim` is the prover's; the verifier interpreter must promote D1 leaves
   to D3 and run everything as D3. If any constraint body relied on F-specific
   behavior (e.g. `inv()` in base field) this would break — but I verified the
   algebraic bodies only use `+ - * ` and `const` (the only "division" is
   multiply-by-precomputed-`INV_SHIFT_32` const, `templates.rs:30`, which is just a
   `Mul` by a constant — safe in any field). ✓ VERIFIED no body calls `inv()` at
   eval time.
4. **GPU kernel program-size / divergence.** Programs are small (hundreds of nodes)
   and uniform across rows (no divergence), but the per-thread value arena must fit
   in registers/local mem; a large mega-constraint program (`dvrm`/`shift` are the
   biggest) could spill. Keccak is NOT a concern (mostly reused `AddConstraint`s,
   verified). *Mitigation:* per-thread arena lives in shared/local mem indexed by
   NodeId; CPU fallback always available; GPU is opt-in per table above a threshold.
5. **Refactor of `evaluate_transitions` for the batched GPU hook.** The current loop
   is per-row (`evaluator.rs:79`); a whole-table GPU call needs either a batched
   path or accepting a precomputed `Cᵢ` matrix. `✗ UNCERTAIN` on the cleanest seam;
   CPU interpreter needs none of this (it slots into the existing per-row call).
6. **CSE/const-dedup correctness.** Optional, but if the CSE key mis-merges two ops
   with the same shape but different dim, results corrupt. *Mitigation:* key on
   `(Op, Dim)`; or ship without CSE first (correctness independent of it).

---

## 12. Effort estimate & validation strategy

### Effort (by workstream)
- **IR + builder + CPU interpreter (framework):** `ir.rs` + `interpreter.rs` +
  trait changes + `AirWithBuses` wiring. **~4-5 days.** Highest design value;
  unblocks everything.
- **Rewrite algebraic constraints (33 structs, ~600-800 LOC):** 11 small structs in
  `constraints/` (~1 day) + 22 in `tables/`, of which ~10 are multi-kind dispatch
  structs. Budget the small ones at ~10-15/day; the multi-kind ones at ~0.5-1.5 each
  (dvrm/cpu32/shift the costliest): **~4-6 days** total.
- **LogUp framework (2 constraints + offset-parameterized helpers):** **~2-3 days**
  — small count but the highest per-line care (the crux).
- **Validation harness (parallel old/new diff):** **~1-2 days.**
- **GPU interpreter (kernel + dispatch + batched hook):** **~5-7 days** incl. the
  `evaluate_transitions` batched seam and parity tests. Can land *after* the CPU
  path is fully migrated and validated.
- **Total: ~2.5-3.5 weeks** for CPU-complete + validated; +1-1.5 weeks for GPU.

### Validation (bit-for-bit, real tables, parallel paths)
1. **Keep the old generic `evaluate*` alongside `capture` during migration.** In a
   `#[cfg(test)]` / debug harness, for each table and each LDE row, run BOTH:
   the old `compute_transition_prover` (current trait-object loop) and
   `interpret_prover(program, …)`, then `assert_eq!` the full `base_evals` and
   `ext_evals` arrays. This is exactly the existing `validate_trace`-style
   debug-assert pattern referenced in project memory; here it asserts
   *evaluator equality* not trace validity.
2. **Drive it with the existing prove test** (`cargo test --release -p
   lambda-vm-prover test_prove_elfs_test_sb_sh_8`) and the per-table bus tests
   (`prover/src/tests/*_bus_tests.rs`, `*_tests.rs`) — these already exercise every
   table's full constraint set on real traces. A mismatch pinpoints the exact
   constraint_idx and row.
3. **Verifier parity:** at the OOD point, diff `air.compute_transition` (old) vs
   `interpret_verifier` for the same frame — small (one frame), cheap, catches the
   D1→D3 promotion bugs (Risk 3).
4. **GPU parity:** standard `gpu_lde.rs` pattern — compute on GPU and on CPU
   interpreter, assert equal (the math-cuda test suite already does this per kernel;
   add a `transition_interp` parity test).
5. Because the old path *coexists* (Option flag), CI can run both and assert equality
   on every prove until we delete the old methods — zero-risk cutover.

### What I could not confirm
- Struct count (33 algebraic + 2 LogUp / 19 files) and per-struct kind counts are
  enumeration-verified; the ~600-800 LOC body total is an aggregate estimate (I read
  `mul.rs`/`dvrm.rs` in full; the rest share the `kind`-enum→`compute()` pattern).
- Whether any table reads periodic columns (none of the bodies I read did; the
  `Periodic` leaf is provided for completeness — `get_periodic_column_values`
  defaults to empty, `traits.rs:290`). `? INFERRED` periodic is unused by current
  tables.
- The cleanest `evaluate_transitions` seam for the batched GPU call (Risk 5).
- Keccak constraint body size/shape (didn't read it) — flagged for GPU register
  pressure (Risk 4).
