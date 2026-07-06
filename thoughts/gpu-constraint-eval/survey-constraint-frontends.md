# Survey: constraint front-ends across production STARK provers

How six production systems **define constraints once** and derive CPU-prover eval,
verifier eval, recursion-guest eval, and the GPU form from that single definition.
Compiled 2026-07-01 from the reference clones in `others/` (agent-verified file:line
refs are into those clones). Companion to `plan-generic-ir-fable.md`, which turns
these findings into our design.

Motivating question: lambda_vm currently hand-writes **two bodies per constraint**
(`evaluate` + `capture`). Is that ever necessary, and what should the single
source of truth look like?

## The matrix

| | Source of truth | CPU prover hot path | Verifier @ OOD | Recursion guest | GPU form | Dedup/CSE |
|---|---|---|---|---|---|---|
| **Plonky3** | one `Air::eval<AB>` body | re-run body per packed row (compiled folder) | re-run body, all-ext folder | — | — | none (tree = metadata only) |
| **OpenVM** | one `Air::eval<AB>` body | **interpret** captured DAG (self-documented as slower; AOT = future work) | interpret DAG | interpret DAG with circuit-var types | transpile DAG → 3-addr `u128` codec | Arc-pointer identity only |
| **SP1** | one `Air::eval<AB>` body | re-run body per packed row (compiled folder) | re-run body (folder) | re-run body, `Expr = SymbolicExt` DSL AST → staged straight-line circuit code | closed-source (moongate server) | — |
| **risc0** | Zirgen DSL (external tool) | generated straight-line C++/CUDA/Metal (old) or one shared C++ template (M3) | **interpret** compact SSA op-stream (`PolyExtStepDef`) | verifier compiled to ZKR bytecode on a micro-op VM | generated straight-line CUDA | generator's problem |
| **zisk** | PIL2 DSL | **interpret** bytecode, AVX-packed ×128 rows | interpret same bytecode, `domainSize=1`, all-ext | interpret (verify circuits are PIL airs) | interpret the **identical** bytecode in CUDA | compiler's problem |
| **airbender** | one imperative builder run | **interpret** deg-≤2 term-lists | generated straight-line Rust (checked-in, 9.5k lines) | generated straight-line (compiled) | flatten term-lists → metadata | none |

## Per-project notes

### Plonky3 (`others/Plonky3` — a fork; symbolic lives in `air/src/symbolic/`)
- `Air::eval(&mut AB)` is the one body (`air/src/air.rs:199`); associated types
  `Expr: Algebra<F> + Algebra<Var>`, `Var: Into<Expr> + Copy` with explicit
  `Add/Sub/Mul<F|Var|Expr>` bounds give **infix operators** (`air/src/builder.rs:12-43`).
- Folders: `ProverConstraintFolder` (`Expr = PackedVal`, SIMD, per quotient row,
  `uni-stark/src/folder.rs:113`), `VerifierConstraintFolder` (`Expr = Challenge`,
  once at ζ, Horner accumulate, `folder.rs:185,216`), `SymbolicAirBuilder`
  (`Expr = SymbolicExpression`, once at setup, `symbolic/builder.rs:277`),
  `DebugConstraintBuilder` (plain `F`, per trace row).
- Symbolic tree: `Arc<Self>` children, **no hash-consing, no CSE of any kind** —
  used only for constraint count / degree (`degree_multiple` cached per node,
  Mul sums, `symbolic/mod.rs:179`) / base-ext layout. Hot path never touches it.
- Selectors are builder methods (`is_first_row`/`is_transition`, `when_*` wraps a
  `FilteredAirBuilder` that multiplies the condition in, `filtered.rs:60`) — no
  per-constraint period/offset/exemptions metadata (ours is richer; keep ours).
- **Trap they document**: emission order is load-bearing — symbolic pass and
  folder pass must agree on constraint indexing (`folder.rs:99`).

### OpenVM (`others/openvm-stark-backend`)
- One body, run **once at keygen** by `SymbolicRapBuilder`; the captured DAG
  (`SymbolicExpressionDag`, `dag.rs:51`) is stored in the proving key. Production
  CPU quotient, verifier-at-OOD, and the CUDA transpiler are all **interpreters
  of that DAG** — the body never runs in production again (only the debug builder
  re-runs it). Their own README (`prover/cpu/quotient/README.md:13-28`) flags
  interpreter overhead vs p3's compiled folders; AOT-compile is listed future work.
- Dedup = `Arc::ptr_eq` identity only (`dag.rs:140-208`); structurally identical
  but separately built subtrees are NOT merged.
- Verifier folder is generic over `Var/Expr` precisely so a recursive verifier can
  interpret the same DAG with circuit types (`verifier/folder.rs:33`); explicit
  warning that the naive tree walk is exponential — use the linear DAG walk
  (`folder.rs:127`).
- Interactions declared in-body (`push_interaction`); the framework generates the
  LogUp constraints into the same constraint list (`interaction/rap.rs:28-43`) —
  same architecture as our `BusInteraction` + framework constraints.
- **Gotchas to avoid**: GPU rules are re-transpiled+re-encoded on *every prove*
  (`SymbolicRulesOnGpu::new` per call — cache per AIR instead); codec packs
  constants as 32-bit (`as_canonical_u32`, `codec.rs:101-139`) — hard-assumes a
  31-bit field, doesn't fit Goldilocks.

### SP1 (`others/sp1` = v6.2.1 hypercube, `others/sp1_4` = v4.2.1 FRI/quotient — mechanism identical)
- One `Air<AB>` body at the scale of hundreds of chips. CPU prover = compiled
  packed folder per row-group (`sp1_4/crates/stark/src/quotient.rs:57-160`);
  symbolic run happens once at chip construction for metadata only (degree is
  **measured**, not declared — `chip.rs:83`).
- **The recursion answer**: `GenericVerifierConstraintFolder<F, EF, PubVar, Var, Expr>`
  (`folder.rs:163`) instantiated with DSL types
  (`Expr = SymbolicExt` — a 3-variant AST with operator overloading,
  `recursion/compiler/src/ir/symbolic.rs:31`) so `chip.eval(&mut folder)` **stages
  straight-line circuit code**. Zero hashing, zero interpretation in-circuit
  (`recursion/circuit/src/constraints.rs:19-118`). v6 kept the exact pattern.
- **Ergonomics cost, visible at scale**: pervasive `.into()` / `.clone()` noise in
  bodies (Expr isn't Copy), and the generic folder's trait bounds are enormous —
  every `Add/Sub/Mul<Var|F|Expr>` combination spelled out per impl (`folder.rs:197-219`).
- No GPU constraint IR in public code (CUDA prover = closed gRPC server).

### risc0 (`others/risc0` — Zirgen NOT in repo, only generated artifacts)
- Old style emits the **same constraint DAG four times** (Rust verifier bytecode
  `poly_ext.rs` 923 KB + straight-line `poly_fp.cpp` 24.7k lines + `eval_check.cu`
  + `.metal`); rv32im's `poly_ext.rs` is 1.05 MB, keccak's is **18.9 MB** — all
  checked in; consistency rests on the external generator. M3 style collapses
  witgen+eval into one Context-parametrized C++ template body.
- Verifier-side representation worth mirroring: `PolyExtStepDef` — a compact SSA
  op-stream (`Const/Get/Add/Sub/Mul/AndEqz/AndCond` + taps metadata) interpreted
  at the OOD point (`zkp/src/adapter.rs:156-233`). Recursion runs the verifier as
  ZKR bytecode on a tiny micro-op VM (3 ops/cycle).
- Lesson: the codegen route costs an external toolchain, MB-scale generated files,
  FFI boundaries, and slow iteration; the compact interpreted op-list for the
  verifier is the part that aged well.

### zisk / pil2-proofman (`others/zisk`, `others/pil2-proofman`) — our exact field (Goldilocks + cubic ext, LogUp)
- One PIL2-compiled `.bin` of expression programs; **three interpreters of the
  identical artifact**: CPU (AVX2/AVX512, `NROWS_PACK=128`,
  `expressions_pack.hpp:351-483`), CUDA (same `ops/args/numbers` uploaded,
  same switch, shared-mem scratch, `expressions_gpu.cu:680-919`), verifier
  (`domainSize=1`, all-extension, `stark_verify.hpp:310-351`). Zero codegen,
  zero duplication; recursive verify circuits are themselves PIL airs.
- **Instruction encoding (production template for our device IR)**: 1 dim-signature
  byte (dest/src dims ∈ {(1,1,1),(3,3,1),(3,3,3)}) + 8 `u16` args
  `[arith_op, dest_pos, (type,pos,stride)×2]`, `u64` Goldilocks constant pool;
  4 arith ops (`add/sub/mul/sub_swap`). Rotations = per-operand `stride` index
  into the openings table. LogUp compiles to ordinary expressions — no special
  opcodes.
- Interpreter overhead is mitigated by 128-lane packing (dispatch amortized).

### airbender (`others/airbender`)
- One imperative `Circuit`/`BasicAssembly` run authors constraints AND registers
  witness resolvers in the same pass; lowered once to `CompiledCircuitArtifact`
  (degree-≤2 term-lists — quadratic enforced at authoring, `constraint.rs:513`).
  Four consumers derive from it: CPU prover interprets term-lists
  (`prover/src/prover_stages/stage3.rs:629-683`), GPU flattener
  (`stage_3_kernels.rs:102-172`), verifier **codegen** (checked-in 9,577-line
  unrolled `evaluate_quotient`), witness codegen (CPU Rust + GPU `.cuh`).
- Recursion guest = the generated straight-line verifier — compiled, no hashing.
- **Caveat that argues for trait-instantiation over codegen**: the generated
  verifier files are checked in and refreshed by a script (`recreate_verifiers.sh`)
  — freshness depends on CI discipline, not the compiler.
- Degree-≤2 term-lists don't fit our degree-3 op-DAG (known from the roadmap).

## Implications for lambda_vm

1. **Nobody hand-writes two bodies.** Every system has one source of truth; our
   `evaluate`+`capture` duplication is an anomaly with no precedent. It must go.
2. **The Rust-native one-body mechanism is the `Air<AB>` / builder-trait pattern**
   (p3, OpenVM, SP1). The DSL/codegen alternatives (risc0, zisk, airbender's
   verifier) buy the same single-source guarantee but cost external toolchains,
   MB-scale checked-in artifacts, or CI-enforced freshness. For a Rust codebase,
   trait instantiation gives the same guarantee compiler-enforced at every build.
3. **CPU hot path — compile it.** p3 and SP1 re-run the monomorphized body per
   (packed) row; OpenVM interprets and self-documents the cost; zisk interprets
   but amortizes over 128 SIMD lanes. We chase ~1% prover deltas ⇒ folder-style
   compiled eval. (Packed/SIMD folders are a future opportunity our design leaves
   open; our current scalar-per-row `evaluate` maps 1:1 onto a scalar folder.)
4. **Recursion guest — compile it too.** SP1 (staged DSL) and airbender (codegen)
   both evaluate constraints as straight-line compiled code in-guest; only zisk
   interprets. Our guest verifier is ordinary Rust compiled to RISC-V, so the
   eval folder instantiated at `FieldElement<E>` IS the compiled guest path —
   zero hashing, zero interpretation, no staging machinery needed.
5. **Capture stays out of the hot path and out of the guest.** Symbolic capture
   runs once at setup (keygen), host-side (p3, OpenVM, SP1 unanimously). CSE is
   *not* done during capture by anyone (p3: none; OpenVM: Arc-identity only);
   dedup belongs in the flatten/lowering step, where hashing is host-setup-only.
6. **GPU encoding template = zisk** (same field: 64-bit Goldilocks constants,
   3 dim-combos, stride-indexed rotations), with OpenVM's three-address register
   allocation as the alternative; OpenVM's 31-bit constant packing does not fit.
   Cache the lowered form per AIR (OpenVM forgets to — re-transpiles every prove).
7. **Emission order / constraint indexing is load-bearing** across every one-body
   system (p3 documents it; OpenVM's layout depends on it). Our explicit
   `constraint_idx` + indexed `emit` already handles this — keep it.
8. **Metadata**: p3/SP1 measure degree by running the symbolic builder instead of
   declaring it (one less hand-maintained number — we can measure per-root degree
   from the captured IR). Our per-constraint zerofier metadata
   (period/offset/exemptions) is richer than their `is_first/last/transition`
   selector trio — keep ours declared.
9. **LogUp architecture confirmed**: OpenVM/SP1 declare interactions in-body and
   let the framework generate the LogUp constraints. Our declarative
   `BusInteraction` + framework-generated lookup constraints is the same shape;
   the LogUp constraint bodies become single builder bodies like everything else.
