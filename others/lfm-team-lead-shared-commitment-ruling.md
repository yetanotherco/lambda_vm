# Team-lead ruling — shared/batched commitment lever

Written 2026-07-31, in answer to deep-join's slice-1 finding. Binding for the
rest of Phase R unless the user overrules.

## The finding being ruled on

deep-join measured the joined DEEP/authentication leg: authentication is
99.0% of its instructions, and the walk is charged PER MATRIX — a narrow
table pays ~88 of its ~92 permutations walking FOUR per-matrix trees to the
SAME index. A shared/batched commitment across a sub-proof's matrices would
collapse ~3/4 of that, the biggest single lever anywhere in the leg. Ruling
was requested before the FRI leg adds a fifth tree per query.

## Ruling

1. **PARKED for the Phase R e2e.** The charter is a keccak e2e with ZERO
   inner-prover changes. ✓ VERIFIED on feat/lfm: the shared-MMCS /
   batched-FRI restructuring (PR #768 line of work) is NOT on this branch's
   base — `BatchedMerkleTree` in `crypto/stark/src/config.rs:19` is merely
   the leaf-hash backend's name (`BatchKeccak256Backend`), and the
   commitment layer still builds one tree per committed matrix. The lever
   therefore requires landing inner-prover commitment restructuring (it
   exists unmerged on `feat/batched-fri-per-epoch`), which is exactly the
   class of change keccak-first exists to avoid. No mid-phase shape change.

2. **The FRI leg targets the CURRENT unbatched shape.** Warning for whoever
   writes that brief: the batched path does not only share trees — it also
   restructures folding (fold-to-scalar terminal, no early stop), so "add
   the shared commitment later" is not a tree-only edit; it changes the FRI
   leg's own shape. Building both shapes now means building FRI twice before
   any e2e exists. One shape, e2e first.

3. **RECORDED as a first-class input to the hash/batching decision.**
   Provenance: prior campaign measurements (sim/4, sim/36 — RV32 guest-side,
   NOT LFM) had batching cut permutations ~5× and left the verdict "gated on
   hash cost in the verifier." LFM's cost model is permutation-dominated
   (deep-join: hashing outweighs its own byteswapping 21.6×; authentication
   is 99% of the joined leg). This is the strongest evidence yet that the
   FINAL shape wants the batched inner proof. The hash-matrix phase after
   e2e must therefore include a batched-shape cell, measured, not argued.

4. **Prediction pinned for that future cell:** ~3/4 of opening-walk
   permutations collapse (deep-join's figure), i.e. per-epoch opening
   authentication ~213,744 → roughly 55–70k permutations at blowup 8,
   before FRI-tree effects. A measured miss means the shape model is wrong.

## What this ruling cannot see

The LFM instruction/permutation cost of leaf WIDENING under a shared tree:
wider leaves absorb more blocks per leaf, offsetting part of the walk
saving. The guest-side analogue was measured (+266M opening-hash when
sim/4's shared tree widened leaves); the LFM analogue is unmeasured and is
part of what the batched-shape cell must answer.
