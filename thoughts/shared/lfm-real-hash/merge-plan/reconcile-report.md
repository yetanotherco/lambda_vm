# Artifact-feature reconciliation to main's constraint IR — report

Approach A, per `artifact-feature-map.md` §5. Worktree
`/Users/maurofab/workspace/lambda_vm-blake3-merge` (branch
`blake3-real-hash-mainmerge`, `git merge origin/main` still in progress,
everything left uncommitted).

**Status: GREEN for the artifact feature.** Round-trip suite 11/11. The 22 extra
`lfm::` failures are pre-existing main-drift, measured against the pre-merge
branch — see §5, they are the lead's, not this task's.

---

## 1. Diff summary

```
 crypto/stark/src/constraint_ir/artifact.rs    | 150 +++++++++++++++++++++-----
 crypto/stark/src/constraint_ir/mod.rs         |   2 +-
 prover/src/lfm/constraint_tests.rs            |   6 +-
 prover/src/tests/constraint_artifact_tests.rs |  35 ++++--
 4 files changed, 154 insertions(+), 39 deletions(-)
```

Line numbers are post-edit.

### `crypto/stark/src/constraint_ir/artifact.rs`

| site | change |
|---|---|
| :16-20 (module doc) | item 1 of the bundle no longer claims to be `DeviceProgram`'s form; points at `ArtifactNode` |
| :91 | `use super::device::{DeviceNode, DeviceProgram}` → `use super::device::DeviceProgram` |
| :105-146 | **NEW** `DIM_BASE: u32 = 0` (:110) / `DIM_EXT: u32 = 1` (:112) and `pub struct ArtifactNode { op, a, b, dim }` (:139), `#[repr(C)]` + `Clone, Copy, Debug, PartialEq, Eq, rkyv::{Archive, Serialize, Deserialize}` — verbatim shape and const values of the OLD `device::DeviceNode` / `DIM_*` (checked against `git show HEAD:crypto/stark/src/constraint_ir/device.rs`, old lines 67-69 and 75-82) |
| :278 | `ConstraintArtifact.nodes: Vec<DeviceNode>` → `Vec<ArtifactNode>` |
| :347-410 | `capture()` — `DeviceProgram::lower(prog)` removed; 1:1 node-index map transplanted at :360-392, const tables at :394-410 (body in §2) |
| :433-439 | `capture()` return — `nodes`/`base_consts`/`ext_consts` are the locals above, `roots`/`num_base` now come from `prog`, not `dev` |
| :474-476 | `device_program()` — field copy → `DeviceProgram::lower(&self.program())` |
| :493 / :568 | `program()` and `validate_self()` — dropped `DIM_BASE, DIM_EXT` from the `super::device::{…}` import lists; they now resolve to the module's own consts. **No logic change**: both still read `n.a`/`n.b` as node ids and `n.dim` as a `DIM_*` tag |

`OP_*` tags and `pack_var`/`unpack_var` are still imported from `device::` —
unchanged on main, and they mean the same thing in both forms; only the operand
encoding differs.

### `crypto/stark/src/constraint_ir/mod.rs`

`:45` — added `ArtifactNode` to the `pub use artifact::{…}` re-export list
(parallel to `device::DeviceNode` being re-exported at `:47`). `DIM_BASE` /
`DIM_EXT` are deliberately NOT re-exported, mirroring main's treatment of
`OP_*` / `RES_EXT_BIT` (reachable via `artifact::`).

### Tests — import moves only, node-index logic untouched

- `prover/src/tests/constraint_artifact_tests.rs:396` (`constraint_op_census`)
  and `:927` (`leg_instructions`): `DIM_BASE` now from
  `stark::constraint_ir::artifact`, the `OP_*` list still from
  `…::device`. The `v_base[n.a as usize]` / `n.dim == DIM_BASE` propagation is
  byte-identical.
- `prover/src/lfm/constraint_tests.rs:638,658,662`
  (`dead_nodes_are_eliminated`): `DeviceNode` → `ArtifactNode`, and
  `device::DIM_EXT` → `artifact::DIM_EXT`.
- `crypto/stark/src/constraint_ir/artifact_tests.rs`: **no change needed** — it
  never imported `DeviceNode` or `DIM_*`, only mutates `artifact.nodes[i].a` /
  reads `.op`, and those field names are identical on `ArtifactNode`. (The map
  predicted an import move here; there was none to make.)

---

## 2. The transplanted `capture()` body

```rust
use super::device::{
    OP_ADD, OP_ALPHA_POW, OP_CONST_BASE, OP_CONST_EXT, OP_EMBED, OP_MUL, OP_NEG,
    OP_RAP_CHALLENGE, OP_SUB, OP_TABLE_OFFSET, OP_VAR, pack_var,
};

let prog = air.constraint_program();

// A 1:1 projection of the captured program — same node count, same
// order, operands left as node ids. Deliberately NOT
// `DeviceProgram::lower`: that is the slot-allocating lowering, and its
// output cannot be lifted back (see `ArtifactNode`).
let nodes: Vec<ArtifactNode> = prog
    .nodes
    .iter()
    .zip(prog.dims.iter())
    .map(|(op, dim)| {
        let dim = match dim {
            Dim::Base => DIM_BASE,
            Dim::Ext => DIM_EXT,
        };
        let (op, a, b) = match *op {
            Op::ConstBase(idx) => (OP_CONST_BASE, idx, 0),
            Op::ConstExt(idx) => (OP_CONST_EXT, idx, 0),
            Op::Var { main, offset, row, col } => {
                let (a, b) = pack_var(main, offset, row, col);
                (OP_VAR, a, b)
            }
            Op::RapChallenge { idx } => (OP_RAP_CHALLENGE, idx as u32, 0),
            Op::AlphaPow { idx } => (OP_ALPHA_POW, idx as u32, 0),
            Op::TableOffset => (OP_TABLE_OFFSET, 0, 0),
            Op::Add(a, b) => (OP_ADD, a, b),
            Op::Sub(a, b) => (OP_SUB, a, b),
            Op::Mul(a, b) => (OP_MUL, a, b),
            Op::Neg(a) => (OP_NEG, a, 0),
            Op::Embed(a) => (OP_EMBED, a, 0),
        };
        ArtifactNode { op, a, b, dim }
    })
    .collect();

let base_consts: Vec<u64> = prog.base_consts.iter().map(|c| *c.value()).collect();
let ext_consts: Vec<[u64; 3]> = prog
    .ext_consts
    .iter()
    .map(|x| {
        let limbs = x.value();
        [*limbs[0].value(), *limbs[1].value(), *limbs[2].value()]
    })
    .collect();
```

and the return now reads

```rust
Self {
    nodes,
    base_consts,
    ext_consts,
    roots: prog.roots.clone(),
    num_base: prog.num_base as u32,
    meta: /* unchanged */,
    shape: /* unchanged */,
}
```

`device.rs`'s `encode_ext` is private to that module (`fn encode_ext`, not
`pub`), so the ext-limb encoding is inlined above rather than imported. It is the
same three-limb `value()` copy, and `program()`'s `FieldElement::from_raw` walk
is its exact inverse — pinned by the round-trip's `base_consts` / `ext_consts`
equality assertions.

`device_program()` is now:

```rust
pub fn device_program(&self) -> DeviceProgram {
    DeviceProgram::lower(&self.program())
}
```

so the slot encoding exists in exactly one place (main's `lower`) and cannot
drift from what the prover and the GPU path build.

---

## 3. Round-trip suite — the hard oracle

`cargo test --release -p lambda-vm-prover --lib constraint_artifact`

```
test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 913 filtered out
```

Every test the task named as previously failing now passes:

| test | result |
|---|---|
| `all_table_artifacts_roundtrip_and_match_folders` | ok |
| `production_airs_accept_a_precaptured_program` | ok |
| `constraint_op_census` | ok |
| `epoch_chunk_multiplier` | ok |
| `continuation_epoch_constraint_leg` | ok |
| `continuation_epoch_chunk_counts_measured` | ok |
| `global_memory_private_input_is_a_second_shape_not_a_second_program` | ok (after the §4 fix) |
| `artifacts_are_invariant_across_trace_length` | ok |
| `artifacts_are_invariant_across_proof_options` | ok |
| `parameterized_airs_vary_per_parameter_value` | ok |
| `an_artifact_does_not_validate_against_a_different_table` | ok |

This is the contract of `artifact-feature-map.md` §4 discharged in full:
`prog.nodes == captured.nodes` bit-for-bit (plus dims/roots/num_base/consts) for
every production AIR, and all three evaluation oracles — `eval_program`,
`eval_device_program` on the re-lowered blob, `eval_program_verifier` — agreeing
with the compiled folders over 100 random trials each.

Also green, and directly in scope:

`cargo test --release -p stark --lib constraint_ir` → **39 passed, 0 failed**
(includes `artifact_tests`: `lift_is_the_inverse_of_lower`,
`validate_self_rejects_a_forward_reference`,
`validate_self_rejects_an_out_of_range_constant`, `ExemptConstraints`).

`cargo check --release -p lambda-vm-prover` → clean.
`rustfmt --check` on all four touched files → clean.
`cargo clippy --release -p stark -p lambda-vm-prover --all-targets` → **no
errors**; the warnings are all the pre-existing `op_ref` class and none land in
a line this task touched.

### Fixture note (not a code change)

Three of these tests read ELFs from `executor/program_artifacts/asm/`, which is
gitignored build output and did not exist in this fresh worktree. I copied the
directory in from `/Users/maurofab/workspace/lambda_vm` so the tests would
actually run rather than fail fast on a missing file. `make compile-*` would
produce the same thing. It changed nothing about the `lfm::` numbers below
(measured both ways, §5).

---

## 4. Deviation from the map: one stale test assertion, fixed

`global_memory_private_input_is_a_second_shape_not_a_second_program`
(`prover/src/tests/constraint_artifact_tests.rs:1151`) failed after the
reconciliation, but **not because of it** — it got past the program-equality
assertions and died on a shape assertion about the AIR:

```
a private-input page is not preprocessed — the verifier never recomputes its
genesis column from the ELF
```

That is main's private-page OFFSET soundness fix landing on a branch-era
expectation. Verified directly:

- branch `HEAD:prover/src/continuation.rs:234` — `if config.is_private_input { return air; }` (no preprocessing at all)
- `origin/main:prover/src/continuation.rs:240` — returns
  `air.with_preprocessed(page::private_page_preprocessed_commitment(opts), page::NUM_PREPROCESSED_COLS_PRIVATE)`

So on main a private-input page **is** preprocessed; it commits OFFSET alone
(`NUM_PREPROCESSED_COLS_PRIVATE = 1`) while an ELF page commits OFFSET and INIT
(`global_memory::NUM_PREPROCESSED_COLS = 2`). INIT stays a main-trace column
because it is the private input; OFFSET must be committed because it is the
row's address and leaving it prover-chosen lets a genesis token name an
arbitrary address.

The test's thesis — *a second shape, not a second program* — is still exactly
right and still worth pinning, so I updated it to main's semantics rather than
deleting it: both variants assert `is_preprocessed`, the two
`num_precomputed_columns` are asserted against the two named constants, and the
"differ ONLY in the preprocessed fields" normalization now normalizes
`num_precomputed_columns` alone. Doc comment updated to match ("preprocess
OFFSET only" rather than "built non-preprocessed").

**This is a judgement call the lead should sanity-check** — it is a test
expectation changed to follow main, in a file otherwise touched only by import
moves.

No other deviations. No main-drift compile errors outside the artifact feature
turned up; the `lookup.rs` `precaptured_program` Clone the lead already added was
the only one.

---

## 5. `lfm::` — 41 failures, and why 22 of them are not this task's

`cargo test --release -p lambda-vm-prover --lib lfm::` in the merge worktree:

```
test result: FAILED. 284 passed; 41 failed; 7 ignored
```

That is **not** the expected 306/19. I measured the pre-merge baseline rather
than assume, running the same command in the branch's own worktree
`/Users/maurofab/workspace/lambda_vm-blake3-impl` @ `ed1b7785` (clean, and with
the identical fixture state — neither `asm/` nor `recursion/` present):

```
test result: FAILED. 306 passed; 19 failed; 7 ignored
```

Same 332 tests either side, so nothing was added or removed. Diffing the two
failure lists: **22 new, 0 fixed.** The 19 pre-existing are the
`recursion/fibonacci.elf` set (`run make compile-recursion-elfs`), exactly as
expected.

The 22 new ones:

```
lfm::constraint_tests::constraint_leg_instruction_census
lfm::constraint_tests::continuation_epoch_constraint_leg_cost
lfm::fri_tests::the_fri_leg_proves_and_verifies
lfm::join_tests::the_join_proves_and_verifies
lfm::keccak_probe::adapter_probe_proves_real_permutations
lfm::keccak_probe::duplicate_tag_output_swap_accepts_demonstrating_hazard
lfm::machine_tests::append_ext_proves_and_verifies
lfm::machine_tests::chunked_sponge_proves_and_verifies
lfm::machine_tests::chunking_does_not_change_what_is_proved
lfm::machine_tests::keccak_chain_proves_and_verifies
lfm::machine_tests::keccak_merkle_walk_authenticates_a_real_opening
lfm::machine_tests::keccak_sponge_proves_and_verifies
lfm::machine_tests::keccak_sponge_reference_lengths_prove_and_verify
lfm::machine_tests::machine_proves_the_sample_replay
lfm::machine_tests::permutations_may_be_reassigned_across_chunk_boundaries
lfm::machine_tests::preprocessed_tags_close_the_output_swap_hazard
lfm::machine_tests::program_id_folds_pages_in_the_production_layout
lfm::machine_tests::program_id_matches_production_on_the_real_fixture
lfm::machine_tests::splice_proves_and_verifies
lfm::machine_tests::statement_replay_proves_and_verifies
lfm::machine_tests::the_register_derivation_proves_and_verifies
lfm::machine_tests::transcript_replay_proves_and_verifies
```

### Why none of these is the reconciliation

The argument rests on a passing oracle, not on inspection:

1. **`lfm/` consumes exactly one thing from the artifact — `program()`** — and
   the round-trip suite asserts `program()` reproduces `air.constraint_program()`
   bit-for-bit for every production AIR. So the LFM machine's input is provably
   identical to what it was pre-merge.
2. **`device_program()` — the only other function whose output changed — has one
   caller in the entire tree**: `constraint_artifact_tests.rs:97`, which passes.
   `grep` over `prover/src` and `crypto/stark/src` finds no other call site, and
   none in `lfm/`.
3. **`keccak_probe.rs` contains zero occurrences of `artifact`**, yet two of its
   tests are in the new-failure list.

### What they actually are

Two are pinned design tables invalidated by main adding a table:

- `constraint_leg_instruction_census` dies on **`no design entry for HINT`**.
  HINT is new on main (`grep -c hint prover/src/tables/mod.rs`: 0 at branch
  `HEAD`, 1 in the merged tree). Note the census itself ran fine and printed a
  full, sane per-table node/leaf/fused/emitted table — the node-index walk over
  `artifact.nodes` works; it is the *design table* that lacks the new row.
- `continuation_epoch_constraint_leg_cost`: `the design's intermediate-epoch
  budget no longer reproduces, left: 62375 right: 63393` — same cause, the
  production AIR set and its constraint counts moved.

The other 20 are LFM machine proof/verify failures ("the machine proof of
sample() must verify", "the joined run must verify", "the registered keccak256
program must verify", …). `git log HEAD..origin/main -- prover/src/lfm/
crypto/stark/src/` shows main brought in **#909 "pin each trace-opening column
width to the AIR, not just their sum"** among others; the recursion machine
hand-builds the verifier it proves, so a verifier-side wire change plus a new
production table is the shape of drift that breaks this whole cluster at once.

**Recommendation:** treat the 22 as a separate reconciliation item for whoever
owns the LFM machine in this merge. The HINT design-table entry looks like the
cheapest first thread to pull — it is a known-missing row, and the budget number
downstream of it is a single pinned constant.
