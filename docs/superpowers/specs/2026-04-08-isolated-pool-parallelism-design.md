# Isolated Rayon Pool Parallelism for multi_prove Rounds 2-4

## Problem

Lambda VM's proving time scales as `~5s + 5s × N_million_steps` on 32+ core
machines, with Rounds 2-4 accounting for ~70% of total time. The current
implementation processes table chunks in batches of `K = cores/3` using the
global Rayon pool. Each table's internal parallelism (FFT, constraint eval,
Merkle hashing) competes for the same threads, causing contention and limiting
throughput.

SP1 achieves sub-linear scaling (O(N^0.3) vs lambda VM's O(N^0.73)) by proving
independent shards on isolated worker pools. This design brings a similar
approach to lambda VM's Rounds 2-4 without changing the proof format or verifier.

## Goals

- Reduce Rounds 2-4 wall time by 3-4x on 32+ core machines
- Keep `MultiProof { proofs: Vec<StarkProof> }` format unchanged
- Keep the verifier unchanged
- No changes to `prove_rounds_2_to_4`, `reconstruct_round1`, or table code
- Change is isolated to the Rounds 2-4 section of `multi_prove` in `prover.rs`

## Non-Goals

- Changing the proof format or verification protocol
- Execution sharding or recursion
- Optimizing Phase A (main commits) or Phase C (aux commits)
- Changing the hash function

## Design

### Architecture

The change replaces the current "chunks of K on global Rayon pool" loop in
Rounds 2-4 with S isolated Rayon thread pools, each processing a partition of
tables on dedicated cores.

**Current flow (lines 1773-1903 of `crypto/stark/src/prover.rs`):**

```
for chunk_start in (0..num_airs).step_by(k):
    pool_sets[..chunk_size].par_iter_mut()      // global Rayon pool
        .map(|pool| reconstruct_round1 + prove_rounds_2_to_4)
    // collect results sequentially
```

**Proposed flow:**

```
let s = shard_parallelism();
let pools = create_isolated_rayon_pools(s, cores / s);
let partitions = partition_tables_by_cost(num_airs, s, &air_trace_pairs);

std::thread::scope(|scope| {
    for (partition, pool, pool_set) in zip(partitions, &pools, &mut pool_sets) {
        scope.spawn(|| {
            pool.install(|| {
                for table_idx in partition {
                    let round1 = reconstruct_round1(table_idx, pool_set);
                    // Append bus_public_inputs to forked transcript (Fiat-Shamir)
                    if let Some(ref bpi) = round1.bus_public_inputs {
                        table_transcripts[table_idx]
                            .append_field_element(&bpi.table_contribution);
                    }
                    prove_rounds_2_to_4(table_idx, &round1, table_transcripts[table_idx]);
                    // return pool buffers
                }
            });
        });
    }
});
// merge proofs in original table order
```

The inner loop body is identical to the current implementation (lines 1819-1867
of `prover.rs`). The `bus_public_inputs` transcript append between
`reconstruct_round1` and `prove_rounds_2_to_4` is mandatory for correct
Fiat-Shamir binding.

Phases A, B, and C remain unchanged — they continue using the global Rayon pool
with the existing chunks-of-K approach.

### Partitioning Strategy

Tables are assigned to partitions using greedy LPT (Longest Processing Time
first) scheduling to balance load:

1. Estimate each table's cost as `trace.num_rows() * (trace.num_main_columns + air.num_auxiliary_rap_columns())`
2. Sort tables by cost descending
3. Assign each table to the least-loaded partition

This ensures partitions finish at roughly the same time. For 50 table chunks
across 4 partitions, LPT gives near-optimal balance.

### Choosing S (number of partitions)

Default: `S = available_cores / 8`, capped at `num_airs / 2`, minimum 1.

Each partition needs enough threads for effective intra-table parallelism (FFT,
constraint eval benefit from 6-8 threads). On 32 cores: S=4 (8 threads each).
On 64 cores: S=8.

Override via `SHARD_PARALLELISM` environment variable, same pattern as the
existing `TABLE_PARALLELISM` env var.

When `S = 1`, the behavior is functionally equivalent to the current
implementation (tables proved sequentially with full intra-table parallelism).
The only overhead is ~1ms for isolated pool creation.

### Pool Buffers

Each partition gets one `PoolSet` (main + aux column buffers), reused across
tables within that partition. Total pool sets: S (was K). Since S <= K typically
(e.g., 4 vs 10 on 32 cores), the LDE column buffer memory decreases or stays
the same. Rayon's internal per-worker stack/deque memory is additional but
negligible (~64KB per thread).

Merkle trees in `Round1Metadata` are shared read-only across partitions via
`Arc` — no duplication.

Per-table forked transcripts are already independent (created at line 1665).
Each partition receives its slice of transcript references.

### Proof Collection

Each partition produces `Vec<(usize, StarkProof)>` — original table index paired
with its proof. After all partition threads join, results are merged and sorted
by original index to produce the final `Vec<StarkProof>` in the order the
verifier expects.

### Instruments Support

The `instruments` feature collects per-table timing data. Each partition
accumulates its own `Vec<(String, usize, Duration, TableSubOps)>` timing
vector within `pool.install`. After all partition threads join via
`std::thread::scope`, the main thread concatenates all timing vectors, sorts
by original table index, and calls `crate::instruments::store` on the main
thread. The `store` call must happen on the main thread (not inside a partition
closure) because it writes to thread-local storage that is read later by the
caller of `multi_prove`. Atomic counters like `R1_MAIN_LDE_US` accumulate
correctly across all pools without special handling.

## Files Changed

| File | Change |
|------|--------|
| `crypto/stark/src/prover.rs` | Replace Rounds 2-4 loop with isolated pool partitions |

No other files change. The verifier, proof structs, table code, and Phase A/B/C
logic are untouched.

## Expected Performance

### Timing at 8M steps, 32 cores

| Phase | Before | After |
|-------|--------|-------|
| Phase A (main commits) | ~8s | ~8s |
| Phase B (challenges) | <0.1s | <0.1s |
| Phase C (aux build + commit) | ~6s | ~6s |
| Rounds 2-4 | ~32s | ~8-10s |
| **Total** | **~46s** | **~22-24s** |

### Scaling

Marginal cost per million steps drops from ~5s/M to ~2-2.5s/M. The improvement
grows at larger scales because Rounds 2-4 dominate more as N increases.

### Future Bottleneck

After this change, Phase A+C become ~60% of total time. The same isolated-pool
technique could be applied to Phase A+C in a follow-up, but that requires
careful handling of the shared transcript ordering constraint.

## Risks

1. **Intra-table parallelism threshold**: If FFT/constraint eval don't scale
   well below 8 threads, S=4 with 8 threads each may underperform vs S=2 with
   16. Mitigated by the `SHARD_PARALLELISM` env var for empirical tuning.

2. **NUMA effects on 32+ core machines**: Isolated pools don't control core
   affinity. On multi-socket machines, a pool's threads may span NUMA domains.
   This is acceptable for now — affinity pinning can be added later if profiling
   shows NUMA penalties.

3. **Thread pool creation overhead**: Rayon `ThreadPool::build()` takes ~1ms per
   pool. For S=4, total overhead is ~4ms — negligible.

## Testing

- Existing `prove_elfs_tests` cover correctness (proof format unchanged)
- Benchmark with `prover/benches/vm_prover_benchmark.rs` at 1M, 4M, 8M steps
  comparing `SHARD_PARALLELISM=1` (baseline) vs default
- Verify identical `MultiProof` output with `SHARD_PARALLELISM=1` vs `S>1`
  (deterministic proofs — same transcript, same challenges, same proofs)
