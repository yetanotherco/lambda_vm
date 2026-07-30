# RESUME HERE — Phase R, keccak recursion

Written 2026-07-30 as a compaction-survival doc. If you are picking this up
with no memory of the session, read this file first, then
`lfm-standing-decisions.md`, then `lfm-target-shape.md`. Everything else is
reference.

## The goal, in one paragraph

Make the LFM (Lambda Field Machine — a straight-line, field-native recursion
machine living in `prover/src/lfm/`) verify a real Lambda VM **continuation
epoch proof**, using **keccak** as the hash. Keccak is explicitly the hash we
do NOT expect to ship; it is first because it needs zero changes to the inner
prover (post-#841 the verify path is keccak-only). Once the e2e works, the
same e2e becomes a **hash test matrix**: blake (most probable final choice)
and Poseidon behind the same socket, giving measured cells-per-verify per
candidate as input to the ecosystem hash decision.

## Where the code is

- Branch **`feat/lfm`**, worktree `/Users/maurofab/workspace/lambda_vm_3-lfm`.
  Never pushed. Based on `origin/main` (includes #841).
- Side branches, both merged into `feat/lfm` as of this writing:
  `feat/lfm-chunking`, `feat/phase0-constraint-ir`, `feat/lfm-constraint-emitter`.
- ⚠ **One worktree per agent, always.** Merging into a worktree an agent is
  live in produced a merge commit wearing that agent's message (`af5ea7c4`).
  Both agents were down when the latest consolidation happened; that is the
  only safe time.

## What works today (all committed, all green)

The machine proves and verifies, end to end, through the registry:

1. **Keccak family hosted unchanged** — production `KECCAK_RND`/`KECCAK_RC`/
   `BITWISE` AIRs driven by an LFM adapter chip speaking their bus contract.
   `KECCAK_RND` is chunkable (chunk count is program shape, in the digest).
2. **`keccak256` bit-exact** vs `PlatformKeccak256` at eight boundary lengths.
3. **Transcript replay** bit-exact vs the real post-#841 `DefaultTranscript`
   (squeeze buffer, absorb invalidation, canonicity guard, zero-rejection).
4. **Continuation-epoch statement + Phase A** → publishes the real `(z, α)`,
   verified against production's own `absorb_statement_with_digest`.
5. **A real Merkle opening authenticated under production keccak** — real
   proof, real committed root, real path.
6. **Constraint evaluation** — a serialized `ConstraintArtifact` lowered to
   machine instructions, all 28 AIRs vs the production evaluator, plus a
   real-proof composition check. 57,252 instructions/epoch, 9.7% under budget.
7. **DEEP slice 1** — composition-polynomial reconstruction at a query point,
   vs production's own reconstruction.
8. **Chaining (ii) and (iii)** — cross-epoch L2G root binding, and the
   attestation's `program_id` fold bit-exact vs production.

## What is left, in order

1. **Chaining (i)**: derive the next epoch's REGISTER preprocessed commitment
   from `reg_fini`. This is a full Merkle TREE build (255/511/1023
   permutations at blowup 2/4/8), not a path walk, plus 3 small FFTs.
   Predicted noise (0.018% of an epoch's hashing at blowup 2, 0.22% at 8 —
   both, they differ 10×). **Measure against 255/511/1023; a miss means the
   shape is not what we think.** Also: does it need a second hashing gadget
   distinct from `keccak_merkle_walk`? FRI will want a third.
2. **DEEP across a full sub-proof**, wired to R1f's Merkle authentication —
   this discharges the obligation the constraint leg deferred: the arena
   values these legs consume must be the ones the authentication leg
   authenticates. Until that join exists both legs are correct in isolation
   and neither proves anything about the other.
3. **FRI folding leg.**
4. **LogUp closure** (Σ L vs the recomputed expected bus balance).
5. **Assembly** into one epoch-verifier program. ⚠ Every per-epoch number so
   far is a COMPOSITION of per-AIR measurements, not a run. Assembly is what
   confirms or falsifies them.
6. **The wrap run** on the box (see `[[scaleway-box-idp]]` in memory:
   195.154.218.198, 124 GB, warm-built).

## Decisions already made — do not relitigate

- **Prove the inner proof at BLOWUP 8.** Two independent legs point there:
  DEEP scales with query count (73 vs 219 ⇒ ~3×), and the keccak bill does
  too (~460k vs ~1.4M permutations).
- **The REGISTER derivation IS the binding.** `VmAirs::new`'s
  `register_preprocessed` parameter looks like unfinished plumbing; it must
  stay unwired. Computing the commitment from `reg_fini` is what ties the
  values to it.
- **Shape-static values are program constants, never arena reads** — and
  next-row pruning likewise, because the verifier reconstructs an undeclared
  column as ZERO.
- **The uniform promotion (`epoch_label`/`page_base`) is PARKED** — off the
  critical path, design complete in `lfm-page-base-uniform-proposal.md`.
- **Zero-rejection transcript** is forced by straight-line shape, not a
  choice; completeness cost < 1e-6/proof.

## Open items needing the USER, not an agent

- **The prover determinism fix.** Root-caused: six dedup tables assign row
  indices by std `HashMap` iteration order (`lt.rs:163/168/177` and five
  siblings), plus grinding's `find_any`. Fix is a contained ~7-file change
  (insertion-ordered index map; `find_first`), no soundness risk, and it
  would restore byte-reproducible proofs and enable a decisive experiment on
  the long-standing ±100k recursion-bench noise. **Offered, not started.**
- **The `check_attestation` gap.** The consumer-side recompute that binds
  supplied roots to a trusted ELF has ZERO production call sites; there is a
  committed PoC (`prover/src/tests/recursion_soundness_gap_poc.rs`). Working
  as designed ("not self-enforcing"), but the design assumes a consumer who
  performs the ritual and nothing in the CLI does.

## How to work here

`lfm-standing-decisions.md` is binding: six method rules, the
pre-authorization list, and the always-stop list. The rules exist because
each one caught something. The highest-yield pattern of the phase, stated
generally:

> When all production instances share a degenerate parameter value, a
> differential over production data cannot distinguish implementations that
> differ only off that value. The synthetic case is the only witness.

Three members so far: next-row pruning, the DEEP coefficient stride, and the
`step_size = 1` collapse. Expect more.

Second-highest: **falsify your own test guards, not just the mechanism.**
Three separate agents found real holes that way — including a tamper suite
whose every vector hit byte 0, so a digest's second word was never checked.
