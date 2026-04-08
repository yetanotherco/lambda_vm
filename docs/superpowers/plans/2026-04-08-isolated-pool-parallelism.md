# Isolated Pool Parallelism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the global Rayon pool in `multi_prove` Rounds 2-4 with S isolated thread pools to reduce wall time by 3-4x on 32+ core machines.

**Architecture:** Rounds 2-4 currently processes tables in sequential chunks of K on the global Rayon pool. This change partitions tables into S groups by estimated cost, creates S isolated Rayon `ThreadPool`s (each with `cores/S` threads), and processes all partitions simultaneously via `std::thread::scope`. The proof format and verifier are unchanged.

**Tech Stack:** Rust, rayon (`ThreadPoolBuilder`), `std::thread::scope`

**Spec:** `docs/superpowers/specs/2026-04-08-isolated-pool-parallelism-design.md`

---

### Task 1: Add `shard_parallelism()` helper function

**Files:**
- Modify: `crypto/stark/src/prover.rs:205-224`

- [ ] **Step 1: Add `shard_parallelism` function below `table_parallelism`**

Add the new function at line 226 (after `table_parallelism` closes):

```rust
/// Number of isolated thread pools for Rounds 2-4 in `multi_prove`.
/// Default: num_cores / 8 (each pool gets ~8 threads for effective intra-table parallelism).
/// Override with `SHARD_PARALLELISM` env var.
fn shard_parallelism() -> usize {
    #[cfg(feature = "parallel")]
    {
        std::env::var("SHARD_PARALLELISM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                let cores = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4);
                (cores / 8).max(1)
            })
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p stark`
Expected: compiles with no errors

- [ ] **Step 3: Commit**

```bash
git add crypto/stark/src/prover.rs
git commit -m "feat(stark): add shard_parallelism() helper for isolated pool config"
```

---

### Task 2: Add `partition_tables_by_cost` function

**Files:**
- Modify: `crypto/stark/src/prover.rs` (add function near `table_parallelism`)

- [ ] **Step 1: Add the partitioning function**

Add after `shard_parallelism`:

```rust
/// Partition tables into `num_partitions` groups using greedy LPT scheduling.
/// Each table's cost is estimated as `rows * (main_cols + aux_cols)`.
/// Returns `Vec<Vec<usize>>` — each inner Vec is a list of original table indices.
fn partition_tables_by_cost<Field: IsFFTField + IsSubFieldOf<FieldExtension>, FieldExtension: IsField, PI>(
    air_trace_pairs: &[AirTracePair<'_, Field, FieldExtension, PI>],
    num_partitions: usize,
) -> Vec<Vec<usize>> {
    let num_airs = air_trace_pairs.len();
    let costs: Vec<usize> = air_trace_pairs
        .iter()
        .map(|(air, trace, _)| {
            trace.num_rows() * (trace.num_main_columns + air.num_auxiliary_rap_columns())
        })
        .collect();

    // Sort indices by cost descending (LPT heuristic)
    let mut sorted_indices: Vec<usize> = (0..num_airs).collect();
    sorted_indices.sort_by(|a, b| costs[*b].cmp(&costs[*a]));

    // Greedy: assign each table to the least-loaded partition
    let mut partitions: Vec<Vec<usize>> = vec![Vec::new(); num_partitions];
    let mut loads = vec![0usize; num_partitions];

    for idx in sorted_indices {
        let min_partition = loads
            .iter()
            .enumerate()
            .min_by_key(|(_, load)| **load)
            .unwrap()
            .0;
        partitions[min_partition].push(idx);
        loads[min_partition] += costs[idx];
    }
    partitions
}
```

Note: `Field: IsFFTField` is required because `AirTracePair` contains
`&dyn AIR<Field=Field, ...>`, and the `AIR` trait has `Field: IsFFTField`
as a supertrait bound.

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p stark`
Expected: compiles (function is unused for now, may get a warning — that's fine)

- [ ] **Step 3: Commit**

```bash
git add crypto/stark/src/prover.rs
git commit -m "feat(stark): add partition_tables_by_cost LPT scheduler"
```

---

### Task 3: Replace Rounds 2-4 loop with isolated pool partitions

This is the main change. It replaces lines 1772-1886 of `prover.rs` (the
"Rounds 2-4: Parallel per-table proving in chunks of K" section).

**Files:**
- Modify: `crypto/stark/src/prover.rs:1772-1886`

**Critical implementation notes (from plan review):**

1. **Mutable transcript access:** The `table_transcripts` Vec needs mutable
   access from multiple partitions at disjoint indices. The borrow checker
   cannot prove disjointness. Use `Vec::as_mut_ptr()` before the scope to
   get a raw pointer, wrap in a `Send` newtype, and derive per-element
   `&mut` references via `ptr.add(idx)` inside each partition. Do NOT use
   `addr_of_mut!(vec[idx])` — it calls `IndexMut` which creates a `&mut Vec`
   data race.

2. **`table_timings` scope:** The `table_timings` vec must be declared in the
   outer scope (before the `#[cfg(feature = "parallel")]` block) so that
   `instruments::store` can access it after the block ends.

3. **`threads_per_pool` computation:** Compute after `s` is capped with
   `.min(num_airs)` to avoid division producing zero or wasting pools.

- [ ] **Step 1: Replace the Rounds 2-4 section**

Replace the entire block from the `// Rounds 2-4` comment (line 1772) through
the end of the `for chunk_start` loop (line 1886) with:

```rust
        // =====================================================================
        // Rounds 2-4: Isolated-pool parallel proving
        // =====================================================================
        // Partition tables into S groups. Each group runs on an isolated Rayon
        // ThreadPool with cores/S threads, eliminating cross-table contention.
        // Pool buffers are reused sequentially within each partition.

        #[cfg(feature = "instruments")]
        let phase_start = Instant::now();
        #[cfg(feature = "instruments")]
        let mut table_timings: Vec<(
            String,
            usize,
            std::time::Duration,
            crate::instruments::TableSubOps,
        )> = Vec::with_capacity(num_airs);

        let s = shard_parallelism().min(num_airs).max(1);
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let threads_per_pool = (cores / s).max(1);

        let partitions = partition_tables_by_cost(&air_trace_pairs, s);

        // Allocate S pool sets (one per partition) for Rounds 2-4.
        let mut r24_pool_sets: Vec<PoolSet<Field, FieldExtension>> = (0..s)
            .map(|_| PoolSet {
                main: (0..max_main_cols)
                    .map(|_| Vec::with_capacity(max_lde_size))
                    .collect(),
                aux: (0..max_aux_cols)
                    .map(|_| Vec::with_capacity(max_lde_size))
                    .collect(),
            })
            .collect();

        #[cfg(feature = "parallel")]
        let proofs = {
            // Raw pointer to transcript Vec elements for disjoint mutable access
            // across partition threads. SAFETY: partition_tables_by_cost guarantees
            // each table index appears in exactly one partition, so no two threads
            // access the same element.
            struct SendPtr<T>(*mut T);
            unsafe impl<T: Send> Send for SendPtr<T> {}
            unsafe impl<T: Send> Sync for SendPtr<T> {}
            let transcripts_ptr = SendPtr(table_transcripts.as_mut_ptr());
            // Create S isolated Rayon thread pools
            let pools: Vec<rayon::ThreadPool> = (0..s)
                .map(|_| {
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(threads_per_pool)
                        .build()
                        .expect("failed to create isolated Rayon thread pool")
                })
                .collect();

            // Each partition produces (original_index, proof, [timing]) tuples
            let partition_results: Vec<Vec<_>> =
                std::thread::scope(|scope| {
                    let handles: Vec<_> = partitions
                        .into_iter()
                        .zip(r24_pool_sets.iter_mut())
                        .zip(pools.iter())
                        .map(|((partition, pool_set), pool)| {
                            let air_trace_pairs = &air_trace_pairs;
                            let metadatas = &metadatas;
                            let domains = &domains;
                            let twiddle_caches = &twiddle_caches;
                            let t_ptr = SendPtr(transcripts_ptr.0);

                            scope.spawn(move || {
                                pool.install(|| {
                                    let mut results = Vec::with_capacity(partition.len());
                                    for &idx in &partition {
                                        let (air, trace, pub_inputs) = &air_trace_pairs[idx];
                                        let metadata = &metadatas[idx];
                                        let domain = &domains[idx];
                                        let twiddles = &twiddle_caches[idx];

                                        #[cfg(feature = "instruments")]
                                        let table_start = Instant::now();
                                        #[cfg(feature = "instruments")]
                                        let lde_start = Instant::now();

                                        let round_1_result = match Self::reconstruct_round1(
                                            *air,
                                            *trace,
                                            domain,
                                            metadata,
                                            twiddles,
                                            &mut pool_set.main,
                                            &mut pool_set.aux,
                                        ) {
                                            Ok(r) => r,
                                            Err(e) => {
                                                results.push(Err(e));
                                                continue;
                                            }
                                        };

                                        #[cfg(feature = "instruments")]
                                        let lde_dur = lde_start.elapsed();

                                        // SAFETY: partitions have disjoint indices, so
                                        // no two threads access the same transcript element.
                                        let table_transcript = unsafe {
                                            &mut *t_ptr.0.add(idx)
                                        };

                                        if let Some(ref bpi) = round_1_result.bus_public_inputs {
                                            table_transcript
                                                .append_field_element(&bpi.table_contribution);
                                        }

                                        let proof = match Self::prove_rounds_2_to_4(
                                            *air,
                                            *pub_inputs,
                                            &round_1_result,
                                            table_transcript,
                                            domain,
                                        ) {
                                            Ok(p) => p,
                                            Err(e) => {
                                                results.push(Err(e));
                                                continue;
                                            }
                                        };

                                        #[cfg(feature = "instruments")]
                                        let timing = {
                                            let mut sub_ops = crate::instruments::take_round_sub_ops()
                                                .unwrap_or_default();
                                            sub_ops.trace_lde += lde_dur;
                                            (
                                                air.name().to_string(),
                                                trace.num_rows(),
                                                table_start.elapsed(),
                                                sub_ops,
                                            )
                                        };

                                        // Return column Vecs to pool
                                        let (main_cols, aux_cols) =
                                            round_1_result.lde_trace.into_columns();
                                        for (slot, col) in pool_set.main.iter_mut().zip(main_cols) {
                                            *slot = col;
                                        }
                                        for (slot, col) in pool_set.aux.iter_mut().zip(aux_cols) {
                                            *slot = col;
                                        }

                                        #[cfg(feature = "instruments")]
                                        results.push(Ok((idx, proof, timing)));
                                        #[cfg(not(feature = "instruments"))]
                                        results.push(Ok((idx, proof)));
                                    }
                                    results
                                })
                            })
                        })
                        .collect();

                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                });

            // Flatten, extract proofs and timings, sort by original table index
            let mut indexed_proofs: Vec<(usize, StarkProof<Field, FieldExtension, PI>)> =
                Vec::with_capacity(num_airs);
            #[cfg(feature = "instruments")]
            let mut indexed_timings: Vec<(usize, _)> = Vec::with_capacity(num_airs);

            for partition_result in partition_results {
                for result in partition_result {
                    #[cfg(feature = "instruments")]
                    {
                        let (idx, proof, timing) = result?;
                        indexed_proofs.push((idx, proof));
                        indexed_timings.push((idx, timing));
                    }
                    #[cfg(not(feature = "instruments"))]
                    {
                        let (idx, proof) = result?;
                        indexed_proofs.push((idx, proof));
                    }
                }
            }

            indexed_proofs.sort_by_key(|(idx, _)| *idx);
            #[cfg(feature = "instruments")]
            {
                indexed_timings.sort_by_key(|(idx, _)| *idx);
                table_timings = indexed_timings.into_iter().map(|(_, t)| t).collect();
            }
            indexed_proofs.into_iter().map(|(_, p)| p).collect::<Vec<_>>()
        };

        #[cfg(not(feature = "parallel"))]
        let proofs = {
            // Sequential fallback: process tables one at a time
            let mut proofs = Vec::with_capacity(num_airs);

            for idx in 0..num_airs {
                let pool_set = &mut r24_pool_sets[0];
                let (air, trace, pub_inputs) = &air_trace_pairs[idx];
                let metadata = &metadatas[idx];
                let domain = &domains[idx];
                let twiddles = &twiddle_caches[idx];

                #[cfg(feature = "instruments")]
                let table_start = Instant::now();
                #[cfg(feature = "instruments")]
                let lde_start = Instant::now();

                let round_1_result = Self::reconstruct_round1(
                    *air, *trace, domain, metadata, twiddles,
                    &mut pool_set.main, &mut pool_set.aux,
                )?;

                #[cfg(feature = "instruments")]
                let lde_dur = lde_start.elapsed();

                if let Some(ref bpi) = round_1_result.bus_public_inputs {
                    table_transcripts[idx].append_field_element(&bpi.table_contribution);
                }

                let proof = Self::prove_rounds_2_to_4(
                    *air, *pub_inputs, &round_1_result,
                    &mut table_transcripts[idx], domain,
                )?;

                #[cfg(feature = "instruments")]
                {
                    let mut sub_ops = crate::instruments::take_round_sub_ops().unwrap_or_default();
                    sub_ops.trace_lde += lde_dur;
                    table_timings.push((
                        air.name().to_string(),
                        trace.num_rows(),
                        table_start.elapsed(),
                        sub_ops,
                    ));
                }

                let (main_cols, aux_cols) = round_1_result.lde_trace.into_columns();
                for (slot, col) in pool_set.main.iter_mut().zip(main_cols) {
                    *slot = col;
                }
                for (slot, col) in pool_set.aux.iter_mut().zip(aux_cols) {
                    *slot = col;
                }

                proofs.push(proof);
            }
            proofs
        };
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p stark`
Expected: compiles with no errors. May have warnings about unused `k` variable
(the old `table_parallelism()` call for Phase A/C still uses `k`).

- [ ] **Step 3: Run the existing test suite**

Run: `cargo test -p lambda_vm_prover -- --test-threads=1 test_prove_elfs`
Expected: all tests pass (proof format is unchanged)

- [ ] **Step 4: Commit**

```bash
git add crypto/stark/src/prover.rs
git commit -m "feat(stark): replace Rounds 2-4 with isolated Rayon pool partitions

Partitions tables into S groups using greedy LPT scheduling. Each group
runs on an isolated Rayon ThreadPool with cores/S threads, eliminating
cross-table contention. Proof format and verifier unchanged.

Default S = cores/8 (override via SHARD_PARALLELISM env var)."
```

---

### Task 4: Drop stale Phase A/C pool sets

**Files:**
- Modify: `crypto/stark/src/prover.rs`

After Task 3, there are two sets of pool sets: the original one used by
Phase A/C (`pool_sets`), and the new one for Rounds 2-4 (`r24_pool_sets`).
The old one should be dropped after the debug-checks block completes.

- [ ] **Step 1: Drop old pool sets after debug-checks block**

After the `#[cfg(feature = "debug-checks")]` block (around line 1770, AFTER
the closing brace of the debug-checks block), add:

```rust
        // Phase A/C pool buffers no longer needed — Rounds 2-4 allocates its own.
        drop(pool_sets);
```

This must be AFTER the debug-checks block (which uses `pool_sets[0]`), not
before it.

- [ ] **Step 2: Verify it compiles and tests pass**

Run: `cargo check -p stark && cargo test -p lambda_vm_prover -- --test-threads=1 test_cpu_only`
Expected: compiles, test passes

- [ ] **Step 3: Commit**

```bash
git add crypto/stark/src/prover.rs
git commit -m "refactor(stark): drop Phase A/C pool sets before Rounds 2-4 allocation"
```

---

### Task 5: Verify correctness with `SHARD_PARALLELISM=1` baseline

**Files:** No changes — verification only.

- [ ] **Step 1: Run full prove/verify test with S=1 (baseline)**

Run: `SHARD_PARALLELISM=1 cargo test -p lambda_vm_prover -- --test-threads=1 test_prove_elfs_sub_neg_result_fast`
Expected: PASS

- [ ] **Step 2: Run full prove/verify test with S=4**

Run: `SHARD_PARALLELISM=4 cargo test -p lambda_vm_prover -- --test-threads=1 test_prove_elfs_sub_neg_result_fast`
Expected: PASS

- [ ] **Step 3: Run a broader test to verify no regressions**

Run: `cargo test -p lambda_vm_prover -- --test-threads=1`
Expected: all tests pass

- [ ] **Step 4: Run with instruments feature to verify timing collection**

Run: `cargo test -p lambda_vm_prover --features instruments -- --test-threads=1 test_prove_elfs_sub_neg_result_fast`
Expected: PASS (instruments timing data collected without panics)

---

### Task 6: Add logging for partition diagnostics

**Files:**
- Modify: `crypto/stark/src/prover.rs`

- [ ] **Step 1: Add info logging after partitioning**

After the `partition_tables_by_cost` call in Rounds 2-4, add:

```rust
        info!(
            "Rounds 2-4: {} tables across {} partitions ({} threads/pool)",
            num_airs, s, threads_per_pool
        );
        for (p_idx, partition) in partitions.iter().enumerate() {
            let total_cost: usize = partition
                .iter()
                .map(|&idx| {
                    let (air, trace, _) = &air_trace_pairs[idx];
                    trace.num_rows()
                        * (trace.num_main_columns + air.num_auxiliary_rap_columns())
                })
                .sum();
            info!(
                "  Partition {}: {} tables, total_cost={}",
                p_idx,
                partition.len(),
                total_cost
            );
        }
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p stark`
Expected: compiles

- [ ] **Step 3: Commit**

```bash
git add crypto/stark/src/prover.rs
git commit -m "feat(stark): add partition diagnostic logging for Rounds 2-4"
```

---

### Task 7: Benchmark and tune

**Files:** No code changes — benchmarking only.

- [ ] **Step 1: Run benchmark at baseline (S=1)**

Run: `SHARD_PARALLELISM=1 cargo bench --bench vm_prover_benchmark`
Record the result.

- [ ] **Step 2: Run benchmark at default S**

Run: `cargo bench --bench vm_prover_benchmark`
Record the result. Compare wall time vs baseline.

- [ ] **Step 3: Try different S values if available cores > 16**

Run:
```bash
SHARD_PARALLELISM=2 cargo bench --bench vm_prover_benchmark
SHARD_PARALLELISM=4 cargo bench --bench vm_prover_benchmark
SHARD_PARALLELISM=8 cargo bench --bench vm_prover_benchmark
```

Record results and identify optimal S for the machine.

- [ ] **Step 4: Document results**

Add a brief note to the commit message or PR description with the benchmark
results showing the improvement.
