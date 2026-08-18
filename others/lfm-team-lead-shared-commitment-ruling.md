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

4. **Prediction — CORRECTED 2026-07-31 after measurement** (deep-join,
   b728043c). My original pin (~3/4 collapse, 213,744 → 55–70k at blowup 8)
   was too optimistic by ~1.7×. The measured figure is **111,471 — a 48%
   collapse**. The reasoning was right about walks and wrong about their
   share: walks DO collapse 69% (1,958 → 616 permutations per query), but
   they are only two thirds of the bill; the other third is leaf absorbs,
   which sharing barely touches (absorbs scale with total bytes, walks with
   tree count, and only the tree count collapses). This is arithmetic over
   the shape — `ceil(leaf_bytes/136)` absorbs plus one permutation per
   level — under one assumption: one tree per sub-proof, leaf = the
   matrices' row pairs concatenated in matrix order.

## What this ruling could not see — CLOSED

The original version flagged leaf WIDENING under a shared tree as
unmeasured. Measured (deep-join, b728043c): widening is a small SAVING,
not a cost — absorbs go 970 → 911 per query, structurally, because total
leaf bytes do not change when matrices share a leaf; the only bytes that
move are the padding of the leaves that vanish. The sim/4 guest-side
+266M analogue does not transfer to LFM's permutation-count model.
