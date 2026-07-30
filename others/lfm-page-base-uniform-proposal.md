# Proposal: promote `page_base` (and `epoch_label`) to runtime uniforms

Status: **proposal only — no semantics touched.** Written by the phase0 agent
2026-07-30 against `feat/phase0-constraint-ir`. Decide before implementing.

## The problem, stated precisely

Four production AIRs fold a workload-dependent value into their captured
constraint IR as a literal constant:

| table | parameter | where it enters |
|---|---|---|
| `PAGE` | `page_base` | `tables/page.rs:533,539` — `LinearTerm::Constant(page_base_lo)`, `BusValue::constant(page_base_hi)` |
| `GLOBAL_MEMORY` | `page_base` | `continuation.rs:228` → `global_memory::bus_interactions(config.page_base)` |
| `L2G_GLOBAL` | `epoch_label` | `tables/local_to_global.rs:360` — `BusValue::constant(epoch_label)` |
| `L2G_MEMORY` | `epoch_label` | `tables/local_to_global.rs:447` — `LinearTerm::Constant(epoch_label as i64 - 1)` |

`BusValue::constant` / `LinearTerm::Constant` lower through
`ConstraintBuilder::const_base`, so the value becomes an `Op::ConstBase` leaf in
the captured program. A different parameter value is a different program.

**Why this is an identity problem, not a size problem.** Size is negligible
(measured below). The blocker is that LFM program identity is a registry-pinned
digest over the emitted program. If the program embeds constraint evaluation and
the constraints vary with the workload's page set, then registry entries become
workload-dependent — and page bases are arbitrary addresses, not a small
enumerable ladder. That breaks the premise the registry exists to uphold.

### Measured, at blowup 2 (from `constraint_artifact_tests`)

| table | nodes | bytes | two parameter values differ by |
|---|---|---|---|
| `PAGE` | 63 | 1,240 | 1 constant value; node count and roots stable |
| `GLOBAL_MEMORY` | 43 | 904 | 1 constant value; node count and roots stable |
| `L2G_GLOBAL` | 47 / 48 | 968 | +1 constant, +1 node, **roots move** |
| `L2G_MEMORY` | 93 / 95 | 1,768 | +1 constant, +2 nodes, **roots move** |

Two things worth pulling out of that table.

First, the variation is **not** confined to constant values, which is what one
would naively assume. The builder interns constants by value, so a parameter
whose value is already in the table costs no new node while a fresh one appends
— shifting every later node id and therefore the constraint ROOTS. `L2G_GLOBAL`
at `epoch_label = 1` reuses the existing `1`; at `epoch_label = 7` it appends.
**This kills the cheap patch.** "Emit one program and swap a constant per page"
is not available, because the programs are not even the same length.

Second, what IS invariant is the algebra: shape, metadata, `num_base`, and
constraint count are identical across parameter values (asserted by
`parameterized_airs_vary_per_parameter_value`). That invariance is exactly what
makes the uniform promotion viable — the parameter is genuinely a value, not a
structural choice.

### Scale (verified where marked)

- Page size `DEFAULT_PAGE_SIZE = 1 << 18` = 256 KiB (`tables/page.rs:50`) —
  VERIFIED.
- `local_to_global::MAX_EPOCHS = 1 << 20` (`tables/local_to_global.rs:83`), a
  hard cap from the `IsB20` range — VERIFIED. Real epoch counts are far smaller
  (the target-shape doc puts a small ethrex block at 1–2 epochs).
- Distinct page count for a realistic workload — **NOT MEASURED HERE.** I do not
  have a number I can point at code for, so I am not giving one.

## Proposed mechanism

### 1. A new IR leaf: base-field runtime uniform

```rust
// crypto/stark/src/constraint_ir/ir.rs
Op::BaseUniform { idx: u16 },   // Dim::Base
```

with device tag `OP_BASE_UNIFORM = 11` (the next free value; tags 0..10 keep
their meanings, so **every already-serialized artifact stays valid** and the
16-byte `DeviceNode` layout is untouched).

**It must be a BASE-field uniform, and that is the whole design constraint.**
Every uniform the IR has today — `RapChallenge`, `AlphaPow`, `TableOffset` — is
`Dim::Ext`. Reusing that machinery would be the obvious move and it is wrong:
`binop` promotes to the extension whenever either operand is `Ext`, so
`page_base_lo + OFFSET_column` would become an extension add. The values would
still agree (embedding is a ring homomorphism) but every downstream node's dim
flips, and `eval_program` would then hit `as_base()` on an extension value for a
base-rooted constraint — a panic, not a wrong answer. It would also silently move
the prover's hot path from base to extension arithmetic. So: a new leaf, base
dim, resolved against a `&[FieldElement<F>]`.

Degree is 0, same as a constant, so `max_degree` and the composition bound are
untouched — **no proof-format change**.

### 2. Cost to the two DeviceProgram consumers

The parity requirement (CUDA kernel and CPU walker consume `DeviceProgram`
bit-identically) is preserved by construction: the change is one additional tag,
handled the same way in both.

**CPU walker** (`eval_device_program`) — one match arm, structurally identical to
the existing `OP_RAP_CHALLENGE` arm but reading a `u64` table instead of a
`[u64;3]` one, plus one new `&[u64]` parameter:

```rust
OP_BASE_UNIFORM => Value::Base(FpE::from_raw(base_uniforms[node.a as usize])),
```

**CUDA kernel** — one `case` in the `switch (op)`, one extra `const uint64_t*`
kernel parameter, one small device allocation (a handful of `u64`s, uploaded
once per proof alongside the existing uniform buffers). No layout change, no new
divergence class beyond one more case in a switch that already has eleven.

This is the cheapest extension the IR admits. Anything that instead tried to
patch constants per-instance would require re-uploading the constant table per
page, which is strictly worse on the device.

### 3. Plumbing: on the AIR, NOT on the context

This is the part that needs a decision, because the obvious route is wrong.

The existing uniforms arrive via `TransitionEvaluationContext`, which is built
once per proof and shared across AIRs. `page_base` is **per-AIR** — a multi-proof
contains many PAGE AIRs with different bases — so it cannot ride that path
without being wrong.

Proposed instead:

```rust
// crypto/stark/src/traits.rs
fn base_uniforms(&self) -> &[FieldElement<Self::Field>] { &[] }
```

`AirWithBuses` stores the slice it was constructed with;
`compute_transition_prover` / `compute_transition` already have `&self`, so they
can hand it to the folder at construction. No signature change reaches the
prover or verifier driver.

Bus layer (the largest chunk of actual work, and the only semantics-adjacent
part): `BusValue::Uniform(idx)` and `LinearTerm::Uniform { coefficient, idx }`
alongside the existing `Constant` variants, lowering to `b.base_uniform(idx)`.
Then four call sites change — `page.rs`, `global_memory.rs`, and two in
`local_to_global.rs`.

### 4. SOUNDNESS OBLIGATION — the part I will not hand-wave

Today `page_base` is baked into the constraints, so a prover cannot lie about it.
Making it a supplied value moves it out of the program, and something must bind
it.

The argument that it is already bound: `page_base` is a **public,
verifier-derived** value. The verifier's page set comes from the ELF and the
declared page ranges, and each page's genesis commitment
(`page::compute_precomputed_commitment`) is recomputed by the verifier rather
than taken from the proof. So the verifier already knows every page base
independently of the prover.

**That argument is necessary but I have not verified it end-to-end, and it is
the single thing that must be checked before implementing.** The rule to hold to
is the one `trace_ood_next_row_columns` already states: the value must be
computed identically by prover and verifier and never read from the
prover-controlled proof. If any path lets the proof choose a base, this proposal
is unsound as written and the uniform must instead be bound by a constraint.

Same question, separately, for `epoch_label` — it is a counter the verifier
derives from the epoch chain, so the argument looks stronger there, but it is
still an argument that needs checking rather than asserting.

### 5. Effect on the artifact format

Small and additive:

- `AirShape` gains `num_base_uniforms: u32`.
- `validate_against` gains that one field comparison.
- `ConstraintArtifact::program()` gains the `OP_BASE_UNIFORM` decode arm.
- Values are **not** stored — they are supplied at verify time. That is the point.

Payoff, in the artifact's own terms: the four parameterized tables collapse from
"one artifact per parameter value" to one artifact each, and the node-count /
root-id instability measured above disappears (PAGE stays 63 nodes for every
base; `L2G_GLOBAL` stops oscillating between 47 and 48).

## Costs and risks, honestly

- **One extra runtime op for `L2G_MEMORY`.** `epoch_label - 1` is folded at
  capture time today; as a uniform it becomes a runtime subtraction on the
  prover's per-row path. Trivially avoidable by supplying `epoch_label - 1` as
  the uniform instead of `epoch_label` — mentioning it because it is the kind of
  detail that turns into a surprise regression otherwise.
- **No prover-hot-path regression from the zero-skip.**
  `ProverEvalFolder::fold_fingerprint_term` skips the multiply when the value is
  zero; that test is on the runtime `FieldElement`, so it behaves identically
  whether the value came from a constant or a uniform. (`page_base_hi` is 0 for
  every address below 2^32, so this was worth checking rather than assuming.)
- **`crypto/**` blast radius.** New `Op` variant, new device tag, new
  `ConstraintBuilder` method, two new bus-layer variants. All additive, but the
  new case has to be added in six places that must agree: `interp::run` and
  `DeviceProgram::lower` match `Op` exhaustively (so those two are compiler-
  enforced), while `eval_device_program`, `ConstraintArtifact::program`,
  `ConstraintArtifact::validate_self` and the CUDA kernel match the numeric tag
  and are **not** — a missing arm there is a runtime panic or, in the kernel, a
  silent wrong answer. The existing differential suites (28 AIRs × both folders ×
  the flat blob) are what would catch it, and they already exist; the CUDA side
  is covered only by `gpu_constraint_interp*` under the `cuda` feature.
- **Not in scope here:** whether the machine wants the uniform as a program
  constant per shape (registry ladder) or as an authenticated arena read. That
  is the shape-static question from the target-shape doc and it is the lead's
  call, not mine.

## What I recommend

Do it for `page_base` and `epoch_label` together — same mechanism, and
`epoch_label` is the one that actually demonstrated root instability, so fixing
only `page_base` would leave the sharper edge in place.

Sequence: (1) verify the soundness obligation in §4 — that is the gate; (2) IR
leaf + both consumers + artifact field, with the existing 28-AIR differential
suites as the safety net; (3) bus-layer variants and the four call sites; (4)
re-measure and confirm the four tables collapse to one artifact each.
