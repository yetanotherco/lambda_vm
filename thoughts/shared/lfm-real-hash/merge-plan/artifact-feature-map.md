# ConstraintArtifact feature map + OLD→NEW device-IR diff

READ-ONLY investigation. Branch `blake3-real-hash` @ `ed1b7785` vs `origin/main` @ `58160b6f`.
Working tree: `/Users/maurofab/workspace/lambda_vm-blake3-impl`.

## TL;DR

- **Recommendation: Approach A** — decouple the artifact's wire format from
  `device.rs`. The artifact should own a POD node type equal to the *OLD*
  `DeviceNode { op, a, b, dim }` (rkyv-derived, **node-index operands**, per-node
  `dim`), keep serializing that, and re-derive the device blob at read time via
  main's `DeviceProgram::lower(&self.program())`. This preserves the node-index
  operand model that `program()`, `validate_self()`, the census, **and the entire
  `prover/src/lfm/` recursion-machine lowering** are built on. Only two functions
  change materially (`capture`, `device_program`) plus import-path fixes.
- **Approach B (store main's slot-form `DeviceProgram`) is infeasible**, not just
  risky: main's `DeviceNode` has no rkyv derives, main's `lower()` is lossy
  (drops uniform leaves + dead nodes, slot-encodes operands), so `program()`
  cannot invert it — the round-trip's `prog.nodes == captured.nodes` assertion can
  never hold — and the LFM machine's per-node model has no meaning on the reduced
  slot graph.
- **THE key device-IR diff:** OLD `lower()` did **NOT** slot-encode operands; it
  produced a 1:1 image of `ConstraintProgram` with `a`/`b` as **raw node indices**
  and a per-node `dim`. NEW `lower()` runs a liveness slot allocator, encodes each
  operand as `kind << 29 | payload` (slot or uniform-table index), drops uniform
  leaves and dead nodes, replaces `dim` with a `res` slot word, and adds
  `num_base_slots`/`num_ext_slots`. `DeviceNode` also lost its rkyv derives.
- **Scope is bigger than the 3 named files.** `prover/src/lfm/constraints.rs`
  (`analyze`/`differential_program`/`ood_frame_words`), `constraint_tests.rs`,
  `join_tests.rs`, `epoch_verify*.rs`, and the `compute_constraint_artifacts`
  binary all consume `ConstraintArtifact`. All but one read it through
  `artifact.program()` (node-index `Op`), so Approach A leaves them untouched.

---

## 0. Where the feature lives, and why main breaks it

The `artifact` module is **branch-only**. `origin/main`'s
`crypto/stark/src/constraint_ir/mod.rs` does **not** declare `pub mod artifact;`
and does **not** re-export `AirShape/ArtifactError/ArtifactMeta/ConstraintArtifact`
(HEAD's mod.rs line 45 does; main's does not). So this is an additive feature that
was written against HEAD's `device.rs`; main independently rewrote `device.rs`.

The break is entirely at the `device.rs` seam:

| symbol the artifact imports from `device.rs` | HEAD | origin/main |
|---|---|---|
| `DeviceNode` fields | `{ op, a, b, dim }` | `{ op, a, b, res }` (✓ VERIFIED, new_device.rs:118) |
| `DeviceNode` rkyv derives | present (old_device.rs:76) | **absent** (`derive(Clone, Copy, Debug, PartialEq, Eq)`, new_device.rs:117) |
| `DIM_BASE` / `DIM_EXT` consts | present (old_device.rs:67,69) | **gone** (grep: none in new_device.rs) |
| operand model in `nodes` | raw node indices | slot-encoded `OPK_* << 29 \| payload` |
| `DeviceProgram` extra fields | — | `num_base_slots`, `num_ext_slots` (new_device.rs:169-171) |
| `roots` entries | node ids | `slot \| RES_EXT_BIT` (new_device.rs:162-164, 333-344) |

`ir.rs` is **byte-identical** between HEAD and main (`diff` = identical).
`ConstraintProgram`, `Op`, `Dim` are unchanged — and, importantly, **none of them
is rkyv-serializable** (plain `derive(Clone, Debug)` / `derive(..., Hash, Debug)`;
`ConstraintProgram` holds `FieldElement<F>`). That is *why* the artifact carries
its own POD projection rather than serializing `ConstraintProgram` directly.

---

## 1. What `ConstraintArtifact` serializes

Struct (`artifact.rs:226-243`), all fields rkyv:

```rust
#[derive(Clone, Debug, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct ConstraintArtifact {
    pub nodes: Vec<DeviceNode>,      // <-- imported from device.rs
    pub base_consts: Vec<u64>,
    pub ext_consts: Vec<[u64; 3]>,
    pub roots: Vec<u32>,
    pub num_base: u32,
    pub meta: Vec<ArtifactMeta>,     // {constraint_idx:u32, kind:u8, end_exemptions:u32}
    pub shape: AirShape,             // width/step/offsets/next_row_cols/... scalars
}
```

- **rkyv derives:** `ConstraintArtifact`, `ArtifactMeta` (artifact.rs:112),
  `AirShape` (artifact.rs:166) all derive `rkyv::{Archive, Serialize, Deserialize}`.
  The `nodes: Vec<DeviceNode>` field requires **`DeviceNode: rkyv::*`**, satisfied
  only by HEAD's `device.rs` (old_device.rs:76). On main this field would not
  compile — the first, hardest breakage.
- **`to_bytes`/`from_bytes`** (artifact.rs:656-668): `rkyv::to_bytes` /
  `rkyv::from_bytes` with `rancor::Error`; `from_bytes` runs `validate_self()`.
- **It stores the FLAT `DeviceProgram` form, not `ConstraintProgram`.** The five
  program fields (`nodes/base_consts/ext_consts/roots/num_base`) are copied
  straight out of `DeviceProgram::lower(prog)` in `capture()` (artifact.rs:302,
  333-338). BUT — and this is the load-bearing subtlety — HEAD's `DeviceProgram`
  is a **faithful 1:1 image** of `ConstraintProgram`: same node count, same order,
  per-node `dim`, and `a`/`b` = node indices. So "the flat DeviceProgram form" and
  "a serializable ConstraintProgram" are the *same bytes* on HEAD. That equivalence
  is exactly what main's slot-encoding `lower()` destroys.

---

## 2. Every dependency on the OLD (node-index) operand model

Each site below reads `n.a`/`n.b`/`roots` as **node indices** and/or reads `n.dim`.
Under main's slot encoding these words are `kind<<29|payload` and `res` slot words,
and many nodes are eliminated — so each site is a breakage point.

### 2a. `artifact.rs`

- **`validate_self()` operand check** (artifact.rs:476-514). The closure
  ```rust
  let check_id = |x: u32| if (x as usize) < i { Ok(()) } else { Err(... "references node {x}, which is not strictly earlier") };
  ...
  OP_ADD | OP_SUB | OP_MUL => { check_id(n.a)?; check_id(n.b)?; }
  OP_NEG | OP_EMBED => check_id(n.a)?,
  ```
  interprets `n.a`/`n.b` as **node ids** and enforces topological order
  (`id i references only < i`). Under slot encoding this is meaningless (operand is
  `OPK_* << 29 | slot`). Also reads `n.dim` against `DIM_BASE/DIM_EXT`
  (artifact.rs:471-472) — tags that no longer exist on main.
- **`program()` reconstruction** (artifact.rs:388-451). Rebuilds a
  `ConstraintProgram` by a linear walk that reads `n.a`/`n.b` as node ids
  (`OP_ADD => Op::Add(n.a, n.b)`, artifact.rs:413) and `n.dim` → `Dim`
  (artifact.rs:420-424). This is the **inverse of the OLD 1:1 lower** and is the
  method the LFM machine and the round-trip oracle both depend on.
- **`device_program()`** (artifact.rs:366-374). Cheap field copy that reconstructs
  a `DeviceProgram` from the stored fields — valid only because the stored form IS
  the device form on HEAD. On main a `DeviceProgram` also needs
  `num_base_slots/num_ext_slots`, which the artifact does not store.
- **`capture()`** (artifact.rs:302, 333-338). `let dev = DeviceProgram::lower(prog)`
  then copies `dev.nodes/roots/...`. On main this yields **slot-form** nodes and
  slot-encoded roots of a *different length* — the artifact would silently store
  the wrong thing even if it compiled.

### 2b. `constraint_artifact_tests.rs` (the round-trip + census)

- **`constraint_op_census`** (test lines 424-467). The `v_base` propagation reads
  operands as node ids:
  ```rust
  OP_NEG => (v_base[n.a as usize], true),
  _      => (v_base[n.a as usize], v_base[n.b as usize]),   // line 447
  v_base[i] = ba && bb && n.dim == DIM_BASE;                // line 451
  ```
  Both `n.a/n.b`-as-index and `n.dim` break on main. `DIM_BASE` is imported from
  `stark::constraint_ir::device` (test line 397) — gone on main.
- **`leg_instructions`** helper (test lines 925-955) — same pattern
  (`v_base[n.a as usize]`, `n.dim == DIM_BASE`), imports `DIM_BASE` from
  `device` (line 927).
- **`check_air_artifact`** (test lines 99-133) asserts `prog.nodes == captured.nodes`,
  `prog.dims == captured.dims`, `prog.roots == captured.roots` — i.e. `program()`
  must reproduce the captured `ConstraintProgram` exactly (see §4).
- The fusability/DCE pass in `constraint_op_census` (test lines 527-567) reads
  `uses[n.a as usize]`/`nodes[n.a as usize].op` as node ids.

### 2c. `crypto/stark/src/constraint_ir/artifact_tests.rs`

- `validate_self_rejects_a_forward_reference` (lines 196-207) sets
  `artifact.nodes[last].a = last` and expects rejection — depends on the node-index
  topological invariant.
- `validate_self_rejects_an_out_of_range_constant` (lines 220-235) reads
  `n.op == OP_CONST_BASE` and mutates `node.a` as a `base_consts` index.
- `lift_is_the_inverse_of_lower` (lines 138-166) asserts
  `artifact.program().nodes == air.constraint_program().nodes` (+ dims/roots/consts).

### 2d. `prover/src/lfm/` (the recursion machine — NOT in the task's file list, but the largest consumer)

- **`lfm/constraints.rs::analyze`** (constraints.rs:286-290) calls
  `artifact.program()` and works over `prog.nodes[i]` as `Op` with **node-index
  operands** (`Op::Add(a,b)` → `konst[a]`/`konst[b]`, fanout counting, DCE,
  MulAdd-fusion, `differential_program`). This whole subsystem consumes the
  *node-index `Op` form via `program()`* — it never touches the device slot blob.
  **Approach A leaves it untouched; Approach B would require rewriting all of it.**
- **`lfm/constraint_tests.rs:658`** is the one place that constructs a raw
  `DeviceNode { op, a, b, dim: DIM_EXT }` and pushes onto `injected.nodes` — a
  direct dependency on the OLD `DeviceNode` shape (has `dim`, node-index `a`/`b`,
  rkyv). This is a mechanical rename under Approach A.
- `join_tests.rs`, `epoch_verify.rs`/`epoch_verify_tests.rs`,
  `bin/compute_constraint_artifacts.rs` use `ConstraintArtifact::capture` /
  `from_bytes` / `program()` — all node-index / `program()`-mediated.

---

## 3. The OLD → NEW device-IR diff (precise)

### 3a. Did OLD `lower()` slot-encode operands? **NO — raw node indices.**

OLD `lower()` (old_device.rs:125-170) is a pure 1:1 map over `prog.nodes.zip(dims)`:

```rust
let (op, a, b) = match *op {
    Op::Add(a, b) => (OP_ADD, a, b),   // a,b are NODE IDS, passed through verbatim
    Op::Sub(a, b) => (OP_SUB, a, b),
    Op::Mul(a, b) => (OP_MUL, a, b),
    Op::Neg(a)    => (OP_NEG, a, 0),
    Op::Embed(a)  => (OP_EMBED, a, 0),
    ...
};
DeviceNode { op, a, b, dim }          // per-node dim carried
...
roots: prog.roots.clone(),            // roots = node ids, verbatim
num_base: prog.num_base as u32,
```

No slots, no liveness, no elimination. `nodes.len() == prog.nodes.len()`, order
preserved. This is what makes it a serializable mirror of `ConstraintProgram`.

### 3b. NEW `lower()` slot-encodes and eliminates (new_device.rs:195-358)

- **Liveness slot allocator** with per-class free lists (`free_base`/`free_ext`),
  `num_base_slots`/`num_ext_slots` counters, operand slots freed at last use, roots
  pinned (new_device.rs:246-344).
- **Operand encoding** `kind << OPK_SHIFT(29) | payload` (new_device.rs:86-105,
  263-274): `OPK_BASE_SLOT/EXT_SLOT/BASE_CONST/EXT_CONST/RAP/ALPHA/OFFSET`. An
  arithmetic operand is a **slot index or a uniform-table index**, never a node id.
- **Uniform-leaf propagation** (new_device.rs:174-227): `Op::ConstBase/ConstExt/
  RapChallenge/AlphaPow/TableOffset` are *not materialized as nodes* unless they
  are themselves roots; operands reference the uniform tables directly.
- **Dead-node elimination**: a node materializes only if `used[i]` (new_device.rs:225-227).
- So `nodes.len() < prog.nodes.len()` in general, order/indices no longer match
  `ConstraintProgram`, and lowering is **lossy** (uniforms/dead nodes gone).

### 3c. `DeviceNode` field diff

| | OLD | NEW |
|---|---|---|
| fields | `op, a, b, **dim**` | `op, a, b, **res**` |
| `dim` | `DIM_BASE`/`DIM_EXT` per node | removed |
| `res` | — | result slot; bit31 (`RES_EXT_BIT`) = ext class, low bits = slot |
| derives | `+ rkyv::{Archive,Serialize,Deserialize}` | **no rkyv** |

### 3d. `DeviceProgram` field diff

| | OLD | NEW |
|---|---|---|
| `nodes/base_consts/ext_consts/num_base` | yes | yes |
| `roots` | node ids | `slot \| RES_EXT_BIT` |
| `num_base_slots` | — | **added** (base `u64` slot-class size) |
| `num_ext_slots` | — | **added** (ext `[u64;3]` slot-class size) |
| derives | `Clone, Debug` | `Clone, Debug` (unchanged; neither is rkyv) |

### 3e. `eval_device_program` diff

OLD (old_device.rs:247-324): forward pass into a flat `Vec<Value>` indexed by node
id; `binop` reads `values[a]`/`values[b]`, dim-driven base/ext. NEW
(new_device.rs:390-497): two slot files (`base_slots`/`ext_slots`), decodes each
operand via `load_base`/`load_ext` on its `OPK_*` kind, writes `res` slot; roots
read back by slot. Semantically bit-identical, structurally different — and the
round-trip test calls `eval_device_program` on whatever `device_program()` returns,
so under Approach A it must return a **main**-lowered `DeviceProgram`.

### 3f. `ConstraintProgram`/`Op`/`Dim` (ir.rs): **identical** on both.
Not rkyv on either side (relevant to Approach A feasibility — see §5).

---

## 4. The round-trip contract (what the failing tests assert)

`all_table_artifacts_roundtrip_and_match_folders` → `check_air_artifact`
(constraint_artifact_tests.rs:72-264), for each of `NUM_PRODUCTION_AIRS` AIRs:

1. `capture` → `validate_against(air)` accepts.
2. `to_bytes` → `from_bytes` → `validate_against(air)` accepts (wire hop).
3. **Structural identity**: `prog = artifact.program()` must equal the AIR's own
   `air.constraint_program()` in `nodes`, `dims`, `roots`, `num_base`,
   `base_consts`, `ext_consts` (lines 101-124). ⇒ **`program()` must reconstruct
   the captured `ConstraintProgram` bit-for-bit.**
4. **Three evaluation oracles agree with the compiled folders** over 100 random
   trials:
   - `eval_program(&prog, ...)` (prover shape) vs `compute_transition_prover`.
   - `eval_device_program(&dev, ...)` with `dev = artifact.device_program()` (flat
     blob) vs the prover folder.
   - `eval_program_verifier(&prog, ...)` (OOD shape) vs `compute_transition`.

`production_airs_accept_a_precaptured_program` (lines 1237-1287): install
`artifact.program()` into a fresh AIR via `with_precaptured`, assert pointer
identity (no re-capture) and folder agreement.

`constraint_op_census` / `epoch_chunk_multiplier` /
`continuation_epoch_constraint_leg` / `continuation_epoch_chunk_counts_measured`:
walk `artifact.nodes` (node-index + `dim`) to count constraint-leg instructions;
assert a loose ceiling (`instr < 200_000`) and a fixed epoch sub-proof composition
(24 intermediate / 25 final).

**What must hold for all of these to pass:** (a) `nodes: Vec<DeviceNode>` must be
rkyv-serializable; (b) `program()` must be the exact inverse of the capture-time
lowering (`prog.nodes == captured.nodes`); (c) `device_program()` must produce a
`DeviceProgram` that `eval_device_program` evaluates to the folder result; (d) the
census must be able to read per-node `dim` and node-index operands. (b) and (d) are
**impossible from main's slot form**; they are trivially preserved by keeping the
OLD node-index form (Approach A).

---

## 5. Reconciliation — two approaches

### Approach A — artifact owns the node-index wire form; re-lower at read time ✅ RECOMMENDED

Keep the artifact storing a POD node array identical to the **OLD** `DeviceNode`
(`{ op, a, b, dim }`, rkyv, node-index operands), owned by the artifact module
instead of imported from `device.rs`. Derive the device blob on demand.

Is `ConstraintProgram`/`Op` rkyv on main? **No** (ir.rs derives are plain; it holds
`FieldElement`). So we cannot serialize `ConstraintProgram` directly — which is
fine, because the artifact already carries its own POD projection. Approach A =
*retain that projection* and stop piggy-backing it on `device.rs`'s type.

**Sites to change (concrete):**

1. **New owned node type in `artifact.rs`** — e.g. `ArtifactNode { op:u32, a:u32,
   b:u32, dim:u32 }` with `#[repr(C)]` + `rkyv::{Archive,Serialize,Deserialize}` +
   `Clone,Copy,Debug,PartialEq,Eq`. Verbatim copy of the OLD `DeviceNode`. Define
   `DIM_BASE`/`DIM_EXT` (u32) here too (gone from `device.rs`). Reuse main's still-
   exported `OP_*` tags and `pack_var`/`unpack_var` (unchanged on main), or re-home
   them alongside the node type for full decoupling.
2. **`ConstraintArtifact.nodes`** field: `Vec<ArtifactNode>` (was `Vec<DeviceNode>`).
3. **`capture()`**: replace `let dev = DeviceProgram::lower(prog)` + field copies
   with a **1:1 map** of `prog.nodes.zip(prog.dims)` into `ArtifactNode` (i.e. the
   OLD `lower` body, old_device.rs:126-158), and `roots = prog.roots.clone()`,
   `num_base = prog.num_base`. (The linearity/shape logic is unchanged.)
4. **`device_program()`**: return `DeviceProgram::lower(&self.program())` — re-lower
   through main's production lowering so the blob has correct
   slots/`res`/`num_base_slots`. (Now non-trivial instead of a field copy; still
   guest-safe: `program()` is a POD walk, `lower()` is a slot scan, no capture.)
5. **`program()`**: unchanged except imports (`DIM_BASE/DIM_EXT`, `OP_*` from the
   new home). Reads `n.a/n.b` as node ids, `n.dim` → `Dim`.
6. **`validate_self()`**: unchanged except imports. Node-index/topo check stays
   valid because the artifact's own form is node-index.
7. **Tests**: `constraint_artifact_tests.rs` and `lfm/constraint_tests.rs:658`
   swap `stark::constraint_ir::device::{DeviceNode, DIM_BASE, DIM_EXT}` for the
   artifact's node type / DIM tags. Census logic unchanged. `artifact_tests.rs`
   unchanged except the same import move.
8. **`lfm/constraints.rs` and the rest of `lfm/`**: **no change** — they consume
   `artifact.program()` (node-index `Op`), which is byte-identical to before.

**Soundness:** the device blob is produced by the *same* `DeviceProgram::lower`
the prover/GPU use, so `eval_device_program` agreement with the folder is inherited
from main's own device tests. The artifact's own form is validated by
`validate_self` (topo order, in-range consts/roots, dense meta) exactly as today.
No new trust surface: `program()` still lifts to the `ConstraintProgram` the folders
are pinned against, and `validate_against` still gates shape/metadata.

**Risk:** low. Two functions change behavior (`capture`, `device_program`); the rest
is renames. The node-index operand model — the thing the LFM machine, the census,
and `program()` all assume — is preserved verbatim.

### Approach B — store main's slot-form `DeviceProgram`; decode slots everywhere ❌ INFEASIBLE

1. **Serialization**: main's `DeviceNode` has no rkyv derives and `DeviceProgram`
   isn't rkyv either — would have to add rkyv to `device.rs` (and its `res` word,
   `num_base_slots/num_ext_slots`). Touches main's file.
2. **`program()` cannot be written**: main's `lower()` drops uniform leaves and dead
   nodes and slot-encodes operands. There is no function from the slot graph back to
   the original `ConstraintProgram.nodes`. ⇒ the round-trip's `prog.nodes ==
   captured.nodes` (and `lift_is_the_inverse_of_lower`) can never pass.
3. **The census / DCE / fanout / MulAdd-fusion** all count *per `ConstraintProgram`
   node* with node-index operands. On the reduced slot graph these quantities are
   different numbers (uniforms and dead nodes already removed) and the operand words
   are slot/uniform indices, not node ids — every one of §2b/§2d would need a
   semantic rewrite, and several have no slot-graph analogue.
4. **`lfm/constraints.rs`** is a node-index `Op` lowering fed by `program()`; without
   a working `program()` the entire recursion-machine constraint leg has no input.
5. **`validate_self`'s** topological/node-index invariant would be replaced by a
   slot-range check — losing the "references strictly earlier node" guarantee the
   falsification tests pin.

Approach B fails at step 2 alone.

---

## Appendix — file/line index

- Feature: `crypto/stark/src/constraint_ir/artifact.rs` (struct 226-243; capture
  297-362; device_program 366-374; program 388-451; validate_self 463-563;
  validate_against 578-652; to/from_bytes 656-668).
- Unit tests: `crypto/stark/src/constraint_ir/artifact_tests.rs`.
- Round-trip + census: `prover/src/tests/constraint_artifact_tests.rs`
  (check_air_artifact 72-264; all_table_...match_folders 279-295; census 394-606;
  leg_instructions 925-955).
- OLD device.rs (HEAD): `git show HEAD:crypto/stark/src/constraint_ir/device.rs`
  (lower 125-170 = node-index 1:1; DeviceNode 75-82 w/ `dim`+rkyv; DIM_* 67-69).
- NEW device.rs (main): `git show origin/main:...` (lower 195-358 = slot alloc;
  DeviceNode 116-123 w/ `res`, no rkyv; OPK_* 86-105; RES_EXT_BIT 109;
  DeviceProgram 149-172 w/ num_base_slots/num_ext_slots).
- ir.rs: identical HEAD vs main; `Op`/`Dim`/`ConstraintProgram` not rkyv.
- mod.rs: main omits `pub mod artifact;` and the artifact re-exports (branch-only).
- Extra consumers (all node-index / `program()`-mediated): `prover/src/lfm/
  constraints.rs` (analyze 286-290 calls `artifact.program()`), `lfm/
  constraint_tests.rs` (incl. raw `DeviceNode{...,dim}` at :658), `lfm/join_tests.rs`,
  `lfm/epoch_verify*.rs`, `prover/src/bin/compute_constraint_artifacts.rs`.
