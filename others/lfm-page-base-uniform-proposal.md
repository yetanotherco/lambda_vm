# Proposal: promote `page_base` (and `epoch_label`) to runtime uniforms

Status: **GATE CLEARED 2026-07-30** by an independent read-only trace; proposal
revised accordingly. Still proposal only — no semantics touched.

Three things the gate trace changed, all of which made the proposal *safer* and
one of which retargets it:

1. **The constant was never a binding.** I argued the uniform would be sound
   *because* `page_base` is already bound by the preprocessed commitment. That
   premise was wrong — it is not bound by anything (§4). The conclusion survives
   and is stronger: there is nothing to break.
2. **On the continuation path the AIR that matters is GLOBAL_MEMORY, not PAGE**
   (§0.1). PAGE is never constructed for a continuation epoch.
3. **`epoch_label` is materially safer than `page_base`**, so my recommendation
   to move them together as equal-risk was wrong (§4.2).

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

### 0.1 SCOPE — on the continuation path, PAGE is never built

Continuation epochs pass `page_configs = &[]` (`continuation.rs:693`, `:797`,
enforced prover-side at `:677-681`), so `create_page_air` is **not called** for
an epoch proof. The page-base-as-constant AIR on the critical path is
**`GLOBAL_MEMORY`** — `global_memory::bus_interactions(page_base)`
(`tables/global_memory.rs:172-214`) via `global_memory_air`
(`continuation.rs:220`).

We recurse continuation epochs, so **GLOBAL_MEMORY is the target**; PAGE matters
only for monolithic proofs. The mechanism below is identical for both — the two
tables differ only in which constants they fold — but the priority is not, and
an implementation that fixed PAGE alone would leave the target path untouched.

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

### Scale

- Page size `DEFAULT_PAGE_SIZE = 1 << 18` = 256 KiB (`tables/page.rs:50`) —
  VERIFIED.
- `local_to_global::MAX_EPOCHS = 1 << 20` (`tables/local_to_global.rs:83`), a
  hard cap from the `IsB20` range — VERIFIED. Real epoch counts are far smaller
  (a small ethrex block is 1–2 epochs).
- **11 distinct ELF page bases** for the committed ethrex ELF, derived statically
  from its `PT_LOAD` headers (not file size, which overcounts). All carry
  `init_values`, so a monolithic ethrex `program_id` folds exactly 11 pairs.
  Plus 1 private-input page for every committed ethrex fixture. — from the gate
  trace.
- **Continuation touched-set size: NOT MEASURED and not statically derivable.**
  It is recorded nowhere. The design comments imply tens rather than thousands;
  that is INFERENCE, not measurement, and is labelled as such wherever it is used.

Either way this confirms size was never the issue: at 25 instructions per
GLOBAL_MEMORY sub-proof, even a four-figure page count is noise against a ~65K
constraint leg. The problem was only ever identity.

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

### 4. SOUNDNESS — gate cleared, and my premise was wrong in my favour

I argued the uniform would be sound *because* `page_base` is already bound by the
preprocessed commitment. **That premise is false.** The gate trace established:

- `page::compute_precomputed_commitment` covers only OFFSET and INIT.
  `page.rs:380-383` says the commitment "depends only on the blowup factor — not
  on page_base", pinned by `static_commitments_tests.rs:82`.
- `page_base` is **not absorbed into the transcript** — the verifier absorbs only
  preprocessed and trace roots.
- It reaches `program_id` only for ELF-backed data pages.

So the compile-time constant is a **verifier-side local, not a commitment**.
Removing it costs nothing, because it was never buying anything. The conclusion
survives and is stronger than the argument I made for it — but I had the reason
backwards, and a proposal resting on a false premise is one edit away from
resting on nothing.

#### 4.1 THE LOAD-BEARING INVARIANT

> **The uniform MUST be populated from the same verifier-side sources that
> produce the constant today: `page_configs` / `canonical_page_bases(
> bundle.touched_page_bases)`. It must NEVER be sourced from the proof or from
> the trace.**

This is not a note. It is the entire soundness content of the change, and it is
*more* critical precisely because §4 found no binding: if a prover-chosen base
ever reached this uniform, **nothing downstream would catch it**. No preprocessed
root covers it. No transcript absorb covers it. `program_id` is not a safety net
(it folds page bases only for ELF-backed data pages). The value would be
unconstrained, and the failure would be silent.

The rule is the one `trace_ood_next_row_columns` already states: computed
identically by prover and verifier, never read from the prover-controlled proof.

#### 4.2 `epoch_label` is NOT symmetric with `page_base`

I recommended moving them together as the same mechanism at the same risk. The
mechanism is the same; **the risk is not**, and the proposal should not have
flattened them.

`epoch_label` is **verifier-derived by construction**: it comes from the
verifier's own `enumerate()` position (`continuation.rs:1293-1295`,
`local_to_global::epoch_label(index) = index + 1`) and is never read from the
bundle. Prover and verifier compute it identically because neither has a choice —
it is a loop counter. There is no supply route to get wrong.

`page_base` has a real supply route (`bundle.touched_page_bases` →
`canonical_page_bases`), which is exactly where §4.1's invariant has to hold.

So `epoch_label` is the lower-risk promotion and a reasonable one to do first;
`page_base` is the one that needs the invariant enforced and reviewed.

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

## What I recommend (revised)

**Target GLOBAL_MEMORY first, not PAGE** — it is the one on the continuation
path (§0.1). Then `epoch_label`'s two L2G tables, which are lower-risk (§4.2) and
where the root instability actually showed up. PAGE last: it is monolithic-only,
and it gets the fix for free once the mechanism exists.

Sequence:

1. IR leaf (`Op::BaseUniform`, tag 11) + both `DeviceProgram` consumers +
   `AirShape::num_base_uniforms`, with the existing 28-AIR differential suites as
   the safety net.
2. Bus-layer `BusValue::Uniform` / `LinearTerm::Uniform`.
3. `GLOBAL_MEMORY` call site, **with §4.1's invariant enforced at the supply
   point and called out in review** — it is the only thing standing between this
   change and an unconstrained value.
4. The two L2G call sites, then PAGE.
5. Re-measure: the four tables should collapse to one artifact each, and
   `parameterized_airs_vary_per_parameter_value` should be able to be deleted —
   if it still passes afterwards, the promotion did not take.

That last point is worth stating as the acceptance test: the existing
characterization test becomes the falsifier for the fix.
