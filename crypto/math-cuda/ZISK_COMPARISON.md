# pil2-proofman (Zisk) vs lambda-vm CUDA kernels

Zisk reuses pil2-proofman's GPU prover, so this is a pil2-proofman
comparison. Same field (Goldilocks + deg-3 ext), same field
representations, similar STARK protocol shape. Kernel listings below
from `/workspace/references/pil2-proofman/pil2-stark/src/**/*.cu`
(HEAD sampled for this report).

## Kernels they have, mapped to ours

| Phase | pil2-proofman kernel | Our equivalent | Status |
|---|---|---|---|
| Goldilocks base arith | `gl64_tooling.cu` | `kernels/goldilocks.cuh` | ✓ parity |
| Cubic-ext arith | `goldilocks_cubic_extension.cuh` | `kernels/ext3.cuh` | ✓ parity |
| Base-field NTT | `ntt_goldilocks.cu` | `kernels/ntt.cu` (batched) | ✓ |
| Coset LDE | via NTT + `computeX_kernel` + `buildZHInv_kernel` | `lde::coset_lde_base` / `ext3` | ✓ |
| Hash (keccak) | external — uses rapidsnark path | `kernels/keccak.cu` | ✓ |
| Hash (poseidon2) | `poseidon2_goldilocks.cu` | **not ported** | — (we don't use poseidon2) |
| Merkle tree build | inside `proveQueries_inplace` | `kernels/fri.cu::fri_merkle_tree_*` | ✓ parity |
| LDE → leaf-hash → tree fuse | inline in `starks_gpu.cu` | `kernels/fri.cu::fri_fused_*` | ✓ |
| FRI fold | `fold` (starks_gpu.cu:604) | `kernels/fri.cu::fri_fold_ext3` | ✓ |
| FRI transpose | `transposeFRI` | n/a (different layout) | — |
| FRI proximity expression | `computeFRIExpression` (:1191) | part of our R4 `deep_composition_poly_evals` | ≈ |
| OOD eval (Lagrange) | `fillLEv_2d` + `computeEvals_v2` | `kernels/barycentric.cu` + `deep.cu` | ✓ |
| OOD reduction | `computeEvalsReduction` | barycentric kernel tail | ✓ |
| Constraint / expression evaluator | `computeExpressions_` (unified bytecode) | **partial: LogUp bytecode only (exp-7)** | ✗ |
| Insert trace col → aux_trace buffer | `insertTracePol` | **we D2H and re-allocate each time** | ✗ |
| Query trace extraction | `getTreeTracePols` / `getTreeTracePolsBlocks` | **CPU: `open_deep_composition_poly`** | ✗ |
| Merkle proof generation | `genMerkleProof` (:817) | **CPU: `fri::query_phase`** | ✗ |
| Query-position computation | `moduleQueries` | CPU (fast) | ≈ |
| Zerofier / domain `X` setup | `buildZHInv_kernel` / `computeX_kernel` | CPU, on host path | ≈ |
| Airgroup value reduce | `opAirgroupValue_` | n/a (different architecture) | — |
| Parallel scan | `prescan` / `prescan_correction` | `kernels/inverse.cu` (chunk scan) | ✓ parity |
| Trace unpack | `unpack` | `kernels/lde.cu` via extract | ✓ |
| Poseidon commit (BN128) | `poseidon_bn128.cu` | not used | — |

Legend: ✓ = we have it, ≈ = near-equivalent, ✗ = gap, — = intentionally
skipped (different architecture / not applicable).

## The three real gaps

### 1. Unified expression/constraint evaluator on device

pil2-proofman compiles every algebraic expression in the AIR — boundary
constraints, transition constraints, bus expressions, the whole lot —
into a single `(ops[], args[])` bytecode that `computeExpressions_`
interprets on GPU. One kernel launch evaluates arbitrarily many
expressions over the domain. All inputs (trace, public inputs,
airgroup values, challenges, evaluations) are pointers into a single
device-side `aux_trace` buffer.

We have a narrower version of this idea (exp-7: LogUp-only bytecode),
and our constraint evaluation stays on CPU. R2 evaluate aggregate is
~5.2 s, roughly ~250 ms wall — a unified bytecode evaluator would
eliminate both that wall time and the R1 aux-build H2D.

**Scope:** large. The PIL compiler emits their bytecode at build time;
we'd need a constraint → bytecode pass over our `Constraint` trait,
or a hand-written evaluator per AIR.

### 2. Query phase on device (`genMerkleProof` + `getTreeTracePols`)

Our `R4 queries & openings` is 3.88 s aggregate, wall 200–500 ms.
pil2-proofman's `proveQueries_inplace` launches two kernels per tree:
`getTreeTracePolsBlocks` (reads trace columns at query rows) and
`genMerkleProof` (walks the tree to build authentication paths). Both
are trivially parallel across queries.

**Scope:** small. We already keep the Merkle trees device-resident in
`FriCommitState` and elsewhere, so the authentication-path kernel is a
few hundred lines. The main-trace Merkle tree for the deep-poly
openings would need to stay on device too (currently it D2Hs after
R1). This is **the next obvious win** — ports cleanly, no architectural
rethink.

### 3. Device-resident trace with in-place writes (`insertTracePol`)

pil2-proofman keeps the whole trace layout as one contiguous device
buffer (`aux_trace`) with per-column offsets recorded in the AIR
metadata. Operations write into slots of that buffer in-place. We
allocate per-column `Vec<FieldElement<F>>` on host, D2H results, then
re-upload for the next operation.

This is the same idea as our exp-7 `DeviceMainCols` plus our
`LdeHandle` from experimental-lde-resident, but extended to the aux
trace too. Biggest latent win — eliminates nearly every H2D in Round 1
aux-build AND Round 2 composition-poly construction.

**Scope:** large. Touches `TraceTable`, the AIR builder, and Round 2.
This is "task E" in the current plan, tracked on cuda/exp-11.

## Minor differences

- **FRI arity.** They default to arity-4 folding; we default to arity-2
  but the stark crate has arity-4 commits landed (`3c03f1e6`). Their
  `fold` kernel handles both. Ours (`fri_fold_ext3`) is arity-2 only.
  Low priority — arity doesn't change the critical path much.
- **Query batching.** They process queries in `nQueries` parallelism
  per tree with a 32×32 thread tile. Whatever we port for query phase
  should mirror this layout.
- **`airgroupValues` / `airValues`.** Different architectural concept
  (their airgroup = our "table", partly). Not a direct port.

## Conclusion

Three gaps, two of them already surfaced in our own planning (E =
device-resident aux-trace, B = unified logup batch). The fresh one is
**query phase on device** — worth a checkpoint on its own. My hunch is
the order by yield/effort should be:

1. Query phase on device (task new — call it exp-13 or fold into exp-11)
2. Device-resident aux trace (task E / exp-11)
3. Unified expression evaluator (large, only worth it once ①+② land)
