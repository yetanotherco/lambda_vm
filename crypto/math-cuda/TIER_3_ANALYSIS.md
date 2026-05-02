# Tier 3 analysis

This branch (`cuda/exp-4-tier3`) was opened to pursue the tier-3
micro-optimisations identified at the end of tier 2, but after analysis
each item turned out to be too small relative to run-to-run variance
(≈ 0.4 s over 15 trials on fib_1M) to land safely. Starting state is
unchanged from the tier 2 end (`cuda/exp-3-tier2`, `2ba3af77`).

## Items investigated

### Stream overlap with `cudaEvent` dependencies (item 40)
The existing round-robin stream pool already gives per-table
concurrency. Within a single table, R2 can't usefully start until R1's
transcript root appends, and R3/R4 depend on R2's challenges — the
transcript is the serialisation point, not a stream barrier. Possible
saving: <50 ms wall. Deferred.

### Warp-level barycentric reduction (item 41)
Current `block_reduce_ext3` uses 3 × 256 u64 shmem + tree reduction
across 256 threads. A warp-shuffle-based approach would cut shmem to
3 × 32 u64 and save a few `__syncthreads` per block. Each barycentric
kernel call is already <5 ms on fib_1M's trace sizes, so the payoff
is well under 20 ms wall. Not shipped.

### GPU batch inverse for R4 DEEP denoms (item 42)
R4 DEEP computes `num_denoms = n × (1 + num_eval_points) ≈ 1M` ext3
elements on CPU (sequential `push` loop + `inplace_batch_inverse`).
Tried two approaches:

1. **Parallel `push` via rayon `par_iter`**: one ext3 subtract per
   task is finer-grained than rayon's overhead. Measured neutral to
   slightly slower. Reverted.

2. **Single-thread GPU Montgomery batch inverse**: 2M serial ext3
   muls on a single SM ≈ 20 ms per call. 7 tables running in
   parallel on GPU serialise on stream pool → ≈ 140 ms total GPU
   busy-time. Today's CPU version runs in ~20–30 ms *wall* thanks to
   7-way rayon parallelism across tables. **Net regression**, not
   shipped.

   A proper parallel Blelloch scan over ext3 would flip this
   (~5 ms GPU per call), but the implementation is ~300+ LoC with
   a delicate ext3-over-blocks primitive — too big for tier 3
   scope. Listed as tier-1 follow-up.

### Zisk's compact TILE layout for NTT (from item 31)
Their 256×4 tile layout for `batched_steps_blocks_par_dif_noBR_compact`
is a good trick, but we'd need to profile current NTT occupancy with
nsight-compute to know whether we're memory-bound enough to benefit.
Without that profile, re-writing 1700+ LoC of NTT kernels for
unclear gain is speculative.

## What would actually move the needle from here

See `NOTES.md`. The only remaining items with ≥0.3 s wall savings
require touching program-specific code (trace build, aux trace build,
constraint eval) or are architectural unlocks (constraint AST →
device bytecode interpreter). All tier-1 scope.

## Branch outcome

No code changes land on this branch. Performance stays at tier 2's
1.57× on fib_1M. Leaving `cuda/exp-4-tier3` pinned here so the
investigation is traceable.
