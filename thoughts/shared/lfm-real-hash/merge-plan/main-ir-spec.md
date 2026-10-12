# main's constraint-IR device representation & operand model

Source of truth: `origin/main`, module `crypto/stark/src/constraint_ir/`.
Files read via `git show origin/main:<path>`: `device.rs`, `ir.rs`, `interp.rs`,
`mod.rs`, `builder.rs`, `gpu_interp.rs`. All line numbers below refer to those
files on `origin/main`.

This spec exists so a follow-up can adapt a build-time serialization feature to
main's current IR. The headline for that: **main has NO serde/rkyv derives on
any of these types** — see §5.

---

## 0. TL;DR

- **Two IR forms.** `ConstraintProgram` (`ir.rs`) is the high-level, field-generic
  node form: a topologically ordered `Vec<Op>` where each `Op` references its
  operands by **node-id** (`u32` index into `nodes`, id `i` only references `< i`).
  `DeviceProgram` (`device.rs`) is the flat, concrete-Goldilocks POD form: a
  `Vec<DeviceNode>` (16-byte `#[repr(C)]` structs) where operands are **slot-encoded
  words**, not node-ids — and where uniform leaves and dead nodes have been
  dropped entirely. `DeviceProgram::lower(&ConstraintProgram)` is the one-way map.

- **Serializability.** Neither `ConstraintProgram`, `Op`, `Dim`, `DeviceProgram`,
  nor `DeviceNode` derives `serde` or `rkyv` on main. `Op`/`Dim`/`DeviceNode` are
  `Copy + Eq + Hash`-friendly PODs (trivially serializable if a feature adds the
  derives); `ConstraintProgram`/`DeviceProgram` carry `FieldElement`/`[u64;3]`
  const tables, so serializing the high-level form directly requires deriving on
  `ConstraintProgram` + `Op` + `Dim` (and a field-element strategy), whereas the
  flat `DeviceProgram` is already all-`u64`/POD and is the cheaper serialization
  target. (§5)

- **Interpreter input forms.** `eval_program` / `eval_program_verifier` /
  `eval_program_base` (in `interp.rs`) all consume a **`ConstraintProgram`**
  (node-index walk). `eval_device_program` (in `device.rs`) and the whole GPU
  path (`gpu_interp.rs`) consume a **`DeviceProgram`** (slot walk). (§4)

---

## 1. THE OPERAND ENCODING (device.rs)

`DeviceNode` is the flat instruction (`device.rs:116-123`):

```rust
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceNode {
    pub op: u32,   // OP_* tag
    pub a: u32,    // operand word 0 (encoding depends on op)
    pub b: u32,    // operand word 1 (encoding depends on op)
    pub res: u32,  // result slot, RES_EXT_BIT selects class
}
```

16 bytes, `#[repr(C)]`, 1:1 device upload.

### 1a. OP_* tags (`device.rs:63-84`)

| Const | Value | Meaning | `a`/`b` meaning |
|---|---|---|---|
| `OP_CONST_BASE` | 0 | base literal (root-pinned uniform only) | `a` = raw `base_consts` index, `b`=0 |
| `OP_CONST_EXT` | 1 | ext literal (root-pinned uniform only) | `a` = raw `ext_consts` index, `b`=0 |
| `OP_VAR` | 2 | trace-cell read | `a`/`b` = packed `Op::Var` fields (§1d) |
| `OP_RAP_CHALLENGE` | 3 | RAP challenge (root-only) | `a` = raw `rap_challenges` index |
| `OP_ALPHA_POW` | 4 | LogUp alpha power (root-only) | `a` = raw `logup_alpha_powers` index |
| `OP_TABLE_OFFSET` | 5 | LogUp table offset (root-only) | no operands |
| `OP_ADD` | 6 | `a + b` | `a`,`b` = **OPK-encoded** operands (§1b) |
| `OP_SUB` | 7 | `a - b` | `a`,`b` = OPK-encoded operands |
| `OP_MUL` | 8 | `a * b` | `a`,`b` = OPK-encoded operands |
| `OP_NEG` | 9 | `-a` | `a` = OPK-encoded operand |
| `OP_EMBED` | 10 | base→ext embed | `a` = OPK-encoded operand |

Note the split-personality of `a`/`b`: for the arithmetic ops (6–10) they are
**OPK-encoded operand words**; for `OP_VAR` they are **packed var fields**; for the
root-pinned uniform leaves (0,1,3,4) `a` is a **raw table index** (not kind-tagged,
because the tag already tells the walker which table).

### 1b. OPK operand encoding — for arithmetic-op operand words `a`/`b`

Encoding scheme (`device.rs:86-105`): `enc = (kind << OPK_SHIFT) | payload`.

- `OPK_SHIFT = 29` (`device.rs:89`) — 3-bit kind occupies bits **29–31**.
- `OPK_PAYLOAD_MASK = (1 << OPK_SHIFT) - 1 = 0x1FFF_FFFF` (`device.rs:91`) — 29-bit
  payload occupies bits **0–28**.

Operand KINDs (`device.rs:92-105`):

| Const | Value | Kind | Payload = |
|---|---|---|---|
| `OPK_BASE_SLOT` | 0 | base (`u64`) scratch slot | slot index |
| `OPK_EXT_SLOT` | 1 | ext (`[u64;3]`) scratch slot | slot index |
| `OPK_BASE_CONST` | 2 | base constant | `base_consts` index |
| `OPK_EXT_CONST` | 3 | ext constant | `ext_consts` index |
| `OPK_RAP` | 4 | RAP challenge (uniform) | `rap_challenges` index |
| `OPK_ALPHA` | 5 | alpha power (uniform) | `logup_alpha_powers` index |
| `OPK_OFFSET` | 6 | LogUp table offset (uniform) | (payload unused) |

Decode (as done in `eval_device_program`, `device.rs:403-425`):
`kind = enc >> OPK_SHIFT` (29); `payload = enc & OPK_PAYLOAD_MASK`.

`load_base` accepts only `OPK_BASE_SLOT` / `OPK_BASE_CONST` (panics otherwise,
`device.rs:403-410`). `load_ext` accepts all seven kinds, embedding base slots/
consts into the extension (`device.rs:411-425`).

**Worked example — `0x4000_0001` as an OPERAND word:**
- `kind = 0x4000_0001 >> 29 = 0b010 = 2 = OPK_BASE_CONST`.
- `payload = 0x4000_0001 & 0x1FFF_FFFF = 0x1 = 1`.
- ⇒ this operand is `base_consts[1]`.

(Caution: the same 32-bit value means something different in a `res`/`roots`
word — see §1c. In a `res` word `0x4000_0001` has `RES_EXT_BIT` (bit 31) clear, so
it would be base slot index `0x4000_0001` — a distinct, non-OPK interpretation.)

### 1c. `res` word and `roots` entries — a DIFFERENT encoding

`RES_EXT_BIT = 1 << 31` (`device.rs:109`). Used in a node's `res` word **and** in
every `roots` entry:

- bit **31** set ⇒ ext (`[u64;3]`) slot class; clear ⇒ base (`u64`) slot class.
- low **31** bits (bits 0–30) = the slot index.

Built at `device.rs:326-329` (per-node `res`) and `device.rs:339-342` /
`device.rs:161-163` (roots): `res = slot` for `Dim::Base`, `res = slot | RES_EXT_BIT`
for `Dim::Ext`. Decoded at `device.rs:428-429` and `486-487`:
`res_slot = res & !RES_EXT_BIT`, `res_ext = (res & RES_EXT_BIT) != 0`.

**This is a 1-bit class tag at bit 31, NOT the 3-bit OPK kind at bits 29–31.** An
operand word and a `res`/`roots` word are decoded by two different schemes; do not
conflate them.

### 1d. `OP_VAR` field packing (`device.rs:125-143`)

`pack_var(main, offset, row, col) -> (a, b)`:
- `a = col as u32` (only low 16 bits are meaningful).
- `b = ((main as u32) << 16) | ((offset as u32) << 8) | (row as u32)`.

So in `b`: bit **16** = `main`; bits **8–15** = `offset` (u8); bits **0–7** = `row`
(u8). `a` bits **0–15** = `col` (u16).

`unpack_var(a, b) -> (main, offset, row, col)` (`device.rs:136-143`):
`col = (a & 0xFFFF)`, `main = (b >> 16) & 1`, `offset = (b >> 8) & 0xFF`,
`row = b & 0xFF`.

---

## 2. THE TWO IR FORMS

### 2a. `ConstraintProgram<F, E>` — high-level node form (`ir.rs:85-107`)

```rust
#[derive(Clone, Debug)]
pub struct ConstraintProgram<F: IsField = GoldilocksField, E: IsField = GoldilocksExtension> {
    pub nodes: Vec<Op>,                    // topologically ordered; id i refs only < i
    pub dims: Vec<Dim>,                    // per-node result dim, parallel to nodes
    pub base_consts: Vec<FieldElement<F>>, // base literals (indexed by Op::ConstBase)
    pub ext_consts: Vec<FieldElement<E>>,  // ext literals (indexed by Op::ConstExt)
    pub roots: Vec<u32>,                   // per-constraint root node-id
    pub num_base: usize,                   // # leading base-rooted constraints
}
```

Fields, all `pub`:
- `nodes: Vec<Op>` — the instruction arena.
- `dims: Vec<Dim>` — parallel to `nodes`, result dim of each node.
- `base_consts: Vec<FieldElement<F>>` — base-field literal table.
- `ext_consts: Vec<FieldElement<E>>` — extension-field literal table.
- `roots: Vec<u32>` — node-id of each constraint's value, indexed by `constraint_idx`.
- `num_base: usize` — count of leading base-`Dim`-rooted constraints (prover writes
  these to `base_evals`; the rest, always ext/LogUp, to `ext_evals`).

Methods (`ir.rs:109-...`): `len()`, `is_empty()`, `next_row_trace_reads(main_width)`
(tests/tooling only — derives the next-row read set from the captured IR).

The `Op` enum (`ir.rs:40-83`), `#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]`:

```rust
pub enum Op {
    ConstBase(u32),                                       // base_consts[idx]
    ConstExt(u32),                                        // ext_consts[idx]
    Var { main: bool, offset: u8, row: u8, col: u16 },    // trace cell read
    RapChallenge { idx: u16 },                            // rap_challenges[idx] (ext, uniform)
    AlphaPow { idx: u16 },                                // logup_alpha_powers[idx] (ext, uniform)
    TableOffset,                                          // LogUp L/N (ext, uniform)
    Add(u32, u32),                                        // nodes[a] + nodes[b]
    Sub(u32, u32),                                        // nodes[a] - nodes[b]
    Mul(u32, u32),                                        // nodes[a] * nodes[b]
    Neg(u32),                                             // -nodes[a]
    Embed(u32),                                           // base -> ext embed
}
```

**Operands are node-ids.** `Add/Sub/Mul(a,b)`, `Neg(a)`, `Embed(a)` carry `u32`
indices into `nodes` (id `i` references only `< i`). `ConstBase/ConstExt(idx)`
carry `u32` indices into the const side-tables (so `Op` stays field-free
`Copy + Eq + Hash`, per the module docs `ir.rs:11-17`).

The `Dim` enum (`ir.rs:26-34`), `#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]`:
`Base` (default) | `Ext`.

### 2b. `DeviceProgram` — flat POD form (`device.rs:149-172`)

```rust
#[derive(Clone, Debug)]
pub struct DeviceProgram {
    pub nodes: Vec<DeviceNode>,      // flat 16-byte ops; uniform leaves & dead nodes dropped
    pub base_consts: Vec<u64>,       // canonical raw base limbs
    pub ext_consts: Vec<[u64; 3]>,   // canonical raw ext limbs
    pub roots: Vec<u32>,             // per-constraint root slot (slot | RES_EXT_BIT)
    pub num_base: u32,               // # base-rooted constraints -> base_evals
    pub num_base_slots: u32,         // size of base (u64) slot class, per thread
    pub num_ext_slots: u32,          // size of ext ([u64;3]) slot class, per thread
}
```

Fields, all `pub`:
- `nodes: Vec<DeviceNode>` — flat instruction list; operands reference **slots**
  (or uniform tables), not node-ids. Uniform leaves and dead nodes are absent.
- `base_consts: Vec<u64>` — raw base limbs (`FieldElement::value()` copies, `device.rs:346`).
- `ext_consts: Vec<[u64; 3]>` — raw ext limbs (`encode_ext`, `device.rs:347,373-376`).
- `roots: Vec<u32>` — per-constraint root **slot** word, `slot | RES_EXT_BIT`.
- `num_base: u32` — same meaning as `ConstraintProgram::num_base`, narrowed to `u32`.
- `num_base_slots: u32` — count of `u64` scratch slots per thread.
- `num_ext_slots: u32` — count of `[u64;3]` scratch slots per thread.

### 2c. The difference, stated explicitly

`ConstraintProgram.nodes` is a node arena **indexed by node-id**, and every
arithmetic `Op` names its operands by those node-ids; the array is dense (every
captured node present, including uniform leaves) and field-generic
(`FieldElement<F>`/`<E>` const tables). `DeviceProgram.nodes` is a **compacted,
slot-addressed** array: `lower` drops uniform leaves (propagated into operand
words) and dead nodes, assigns each surviving node a reusable scratch **slot**
via liveness scan, and rewrites operands as slot-encoded words (`OPK_* << 29 |
payload`) pointing at slots or uniform tables — never at node positions. Constants
are demoted from `FieldElement` to raw `u64`/`[u64;3]` limbs. In short:
**node-id operands + generic field ⟶ slot-encoded operands + raw limbs, with
uniform/dead nodes removed.**

---

## 3. `DeviceProgram::lower(&ConstraintProgram) -> DeviceProgram` (device.rs:195-358)

Signature (`device.rs:200`):
`pub fn lower(prog: &ConstraintProgram<GoldilocksField, GoldilocksExtension>) -> Self`
— concrete Goldilocks only. **`dims` IS an input**: `ConstraintProgram` carries the
parallel `dims: Vec<Dim>`, and `lower` reads `prog.dims[i]` / `prog.dims[j]` for
slot-class decisions (`device.rs:260, 270, 304, 339`). Dims are not recomputed.

Algorithm:

1. **Bound check** (`device.rs:201-205`): `n = prog.nodes.len()` must be
   `<= OPK_PAYLOAD_MASK` (2^29−1), else panic — the 29-bit slot/payload space.

2. **Liveness pass** (`device.rs:208-220`): compute `used[j]` and `last_use[j]`
   (max consumer node-id) for every node by scanning `operands(op)` (the up-to-two
   operand node-ids, `device.rs:187-193`). Then mark `is_root[r]` and force
   `used[r]=true` for each root.

3. **Emit set** (`device.rs:225-227`): node `i` materializes iff
   `used[i] && (!is_uniform_leaf(nodes[i]) || is_root[i])`. `is_uniform_leaf`
   (`device.rs:175-184`) = `ConstBase|ConstExt|RapChallenge|AlphaPow|TableOffset`.
   Uniform leaves are propagated into operands and only kept as nodes when they are
   themselves constraint roots.

4. **Slot allocator — linear scan with per-class free lists** (`device.rs:246-331`).
   State: `slot_of[i]` (init `UNASSIGNED=u32::MAX`), `free_base: Vec<u32>`,
   `free_ext: Vec<u32>`, counters `num_base_slots`, `num_ext_slots`. For each
   emitted node `i` in order:
   - **Encode operands** while operand slots are still live (`enc_operand`,
     `device.rs:263-274`): if operand `j` is not emitted (a propagated uniform
     leaf) → `enc_uniform` (`device.rs:229-244`) emits `OPK_BASE_CONST/EXT_CONST/
     RAP/ALPHA/OFFSET << 29 | idx`. Else → `OPK_BASE_SLOT`/`OPK_EXT_SLOT << 29 |
     slot_of[j]`, class chosen by `prog.dims[j]`.
   - **Build `(tag, a, b)`** per op (`device.rs:276-296`): uniform-leaf & `OP_VAR`
     nodes stash raw indices / packed var fields; arithmetic ops store the encoded
     operand words.
   - **Free dead operand slots** (`device.rs:300-310`): for each operand `j`, if
     `emitted[j] && !is_root[j] && last_use[j]==i && slot_of[j]!=UNASSIGNED`, push
     `slot_of[j]` onto the matching free list and reset `slot_of[j]=UNASSIGNED`
     (the reset guards the `a==b` double-free). Roots are pinned (never freed).
   - **Allocate result slot** (`device.rs:314-324`): pop from the matching free
     list, else bump the class counter (`num_base_slots`/`num_ext_slots`). A slot
     freed this same node may be reused (kernel reads operands before writing res).
     Record `slot_of[i]`.
   - **Emit `DeviceNode`** (`device.rs:326-330`) with `res = slot` (base) or
     `slot | RES_EXT_BIT` (ext).

   ⇒ `num_base_slots` / `num_ext_slots` end as the **max-live-set per class**, not
   the node count (root pins excepted).

5. **Roots** (`device.rs:333-344`): map each `prog.roots[c]` node-id through
   `slot_of[..]` to `slot | (RES_EXT_BIT if Dim::Ext)`.

6. **Const tables** (`device.rs:346-347`): `base_consts` = raw `u64` via
   `c.value()`; `ext_consts` = `[u64;3]` via `encode_ext`.

7. **Assemble** (`device.rs:349-357`): `num_base = prog.num_base as u32`.

---

## 4. THE INTERPRETERS

### 4a. `interp.rs` — node-index walkers over `ConstraintProgram`

All three take a `&ConstraintProgram<F, E>` and walk `prog.nodes` by node-index
(shared `run`, `interp.rs:60-113`, which builds a parallel `Vec<Value>` indexed
1:1 with `nodes`).

- `eval_program_base` (`interp.rs:150-171`) — minimal single-root, main-only,
  base result, for the per-constraint diff test:
  ```rust
  pub fn eval_program_base<F, E>(
      prog: &ConstraintProgram<F, E>,
      constraint_idx: usize,
      main_row: &[FieldElement<F>],
  ) -> FieldElement<F>
  ```

- `eval_program` (`interp.rs:178-217`) — full **prover** entry; requires
  `TransitionEvaluationContext::Prover`; writes base-rooted → `base_evals`,
  ext-rooted → `ext_evals`:
  ```rust
  pub fn eval_program<F, E>(
      prog: &ConstraintProgram<F, E>,
      ctx: &TransitionEvaluationContext<F, E>,
      base_evals: &mut [FieldElement<F>],
      ext_evals: &mut [FieldElement<E>],
  )
  ```

- `eval_program_verifier` (`interp.rs:225-262`) — full **verifier** entry;
  requires `TransitionEvaluationContext::Verifier`; writes every constraint into
  `ext_evals` (base roots embedded):
  ```rust
  pub fn eval_program_verifier<F, E>(
      prog: &ConstraintProgram<F, E>,
      ctx: &TransitionEvaluationContext<F, E>,
      ext_evals: &mut [FieldElement<E>],
  )
  ```

### 4b. `device.rs` — flat slot walker over `DeviceProgram`

- `eval_device_program` (`device.rs:389-497`) — CPU model of the GPU kernel;
  consumes a `&DeviceProgram` and walks `dev.nodes` decoding slot-encoded
  operands (dim-split slot files `base_slots`/`ext_slots`), in raw limbs:
  ```rust
  pub fn eval_device_program(
      dev: &DeviceProgram,
      main: &[Vec<u64>],
      aux: &[Vec<[u64; 3]>],
      rap_challenges: &[[u64; 3]],
      alpha_powers: &[[u64; 3]],
      table_offset: [u64; 3],
      base_evals: &mut [u64],
      ext_evals: &mut [[u64; 3]],
  )
  ```

### 4c. GPU path (gpu_interp.rs) — consumes `DeviceProgram`

`#[cfg(feature = "cuda")]`. Both entry points — `try_eval_composition_gpu`
(`gpu_interp.rs`) and `try_eval_program_gpu` — take a **generic
`&ConstraintProgram<F, E>`**, but immediately funnel through `lower_and_pack`
(TypeId-gates the Goldilocks tower, `unsafe`-reinterprets to the concrete program,
then calls `DeviceProgram::lower`). Everything handed to the CUDA FFI is the
**lowered `DeviceProgram`** (via `pack_nodes` → 2×`u64` per node:
`op | a<<32`, `b | res<<32`; plus `dev.base_consts`, `flatten_ext3(dev.ext_consts)`,
`dev.roots`, `num_base_slots`, `num_ext_slots`). The lowering is cached
process-wide by content fingerprint (`lowering_cache`, `program_fingerprint`,
`program_eq`). **So the GPU consumes the `DeviceProgram` (slot) form**, produced
on demand from the `ConstraintProgram` at dispatch time.

---

## 5. SERIALIZABILITY — main derives NEITHER serde NOR rkyv

Verified by grepping the module for `rkyv|Archive|Serialize|Deserialize|serde`:
zero hits in `ir.rs`, `device.rs`, `builder.rs`, `interp.rs`, `mod.rs`,
`gpu_interp.rs`. The only derives present are the standard traits.

Exact derive lines:

- `Dim` (`ir.rs:26`): `#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]`
- `Op` (`ir.rs:40`): `#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]`
- `ConstraintProgram` (`ir.rs:85`): `#[derive(Clone, Debug)]`
- `DeviceNode` (`device.rs:117`): `#[derive(Clone, Copy, Debug, PartialEq, Eq)]`
- `DeviceProgram` (`device.rs:149`): `#[derive(Clone, Debug)]`
- `Expr` (`builder.rs:29`, builder handle, not part of the program): `#[derive(Clone, Copy, Debug)]`

Implications for a build-time serialization feature:

- **`DeviceProgram` is the cheap target.** `DeviceNode` is a 16-byte `#[repr(C)]`
  POD of four `u32`s; `DeviceProgram`'s other fields are `Vec<u64>` / `Vec<[u64;3]>`
  / `Vec<u32>` / `u32`. Adding `rkyv(Archive, Serialize, Deserialize)` (or serde)
  is mechanical — no field-element or generic-tower obstacle. This matches how the
  gpu path already treats it as flat `u64` blobs.

- **Serializing the high-level `ConstraintProgram` directly is more involved.** It
  is generic `<F, E>` and holds `Vec<FieldElement<F>>` / `Vec<FieldElement<E>>`
  const tables, so a derive must either (a) bound the field types with the
  serialization traits, or (b) fix the concrete Goldilocks tower and serialize the
  const tables as raw limbs (the same `to_raw`/`value()` trick `lower` and
  `program_fingerprint` use). `Op` and `Dim` themselves are trivially derivable
  (plain `u32`/enum payloads, already `Copy + Eq + Hash`).

- Consequence for the merge: if the incoming feature stores the **high-level**
  form, it needs derives on `ConstraintProgram + Op + Dim` plus a field-element
  serialization strategy; if it stores the **flat** form, it only needs derives on
  `DeviceProgram + DeviceNode`. Main provides neither today; both are additive.

---

## 6. mod.rs — public exports (mod.rs:29-45)

```rust
pub mod builder;
pub mod device;
#[cfg(feature = "cuda")]
pub mod gpu_interp;
pub mod interp;
pub mod ir;

#[cfg(test)]
mod tests;

pub use builder::{Expr, IrBuilder};
pub use device::{DeviceNode, DeviceProgram, eval_device_program};
pub use interp::{eval_program, eval_program_base, eval_program_verifier};
pub use ir::{ConstraintProgram, Dim, Op};
```

Re-exported from `constraint_ir`:
- from `builder`: `Expr`, `IrBuilder`
- from `device`: `DeviceNode`, `DeviceProgram`, `eval_device_program`
- from `interp`: `eval_program`, `eval_program_base`, `eval_program_verifier`
- from `ir`: `ConstraintProgram`, `Dim`, `Op`

**NOT re-exported (must be reached via `device::`):** the `OP_*` tag constants,
the `OPK_*` operand-kind constants, `OPK_SHIFT`, `OPK_PAYLOAD_MASK`, `RES_EXT_BIT`,
`pack_var` / `unpack_var`, and `DeviceProgram::lower`. The `cuda`-gated
`gpu_interp` (`try_eval_composition_gpu`, `try_eval_program_gpu`, the
`u64↔FieldElement` reinterpret helpers) is a `pub mod` but nothing is re-exported
at the `constraint_ir` root.

---

## Appendix: full constant table (device.rs)

| Name | Value | Role |
|---|---|---|
| `OP_CONST_BASE` | 0 | tag |
| `OP_CONST_EXT` | 1 | tag |
| `OP_VAR` | 2 | tag |
| `OP_RAP_CHALLENGE` | 3 | tag |
| `OP_ALPHA_POW` | 4 | tag |
| `OP_TABLE_OFFSET` | 5 | tag |
| `OP_ADD` | 6 | tag |
| `OP_SUB` | 7 | tag |
| `OP_MUL` | 8 | tag |
| `OP_NEG` | 9 | tag |
| `OP_EMBED` | 10 | tag |
| `OPK_SHIFT` | 29 | operand kind bit position |
| `OPK_PAYLOAD_MASK` | `0x1FFF_FFFF` | operand payload mask (bits 0–28) |
| `OPK_BASE_SLOT` | 0 | operand kind |
| `OPK_EXT_SLOT` | 1 | operand kind |
| `OPK_BASE_CONST` | 2 | operand kind |
| `OPK_EXT_CONST` | 3 | operand kind |
| `OPK_RAP` | 4 | operand kind |
| `OPK_ALPHA` | 5 | operand kind |
| `OPK_OFFSET` | 6 | operand kind |
| `RES_EXT_BIT` | `1 << 31` | `res`/`roots` ext-slot class bit |
