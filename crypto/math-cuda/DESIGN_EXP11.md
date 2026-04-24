# Design: device-resident main trace (exp-11)

Tracking the biggest remaining single win — eliminate redundant
main-trace host→device copies. Not yet implemented; this doc scopes
the work. Matches the pattern of exp-4 (tier-3 analysis), which
shipped as a checkpoint so the plan was preserved across context
windows.

## Current state

Fib_1M wall-time breakdown at exp-9 tip (15-trial mean 10.96 s):

```
Trace build             2.48 s   21.9%    CPU (user-supervised)
Round 1 Phase A         ~1.5 s   13%      Main commits (LDE+Merkle on GPU)
Round 1 Phase B         ~0 s              LogUp challenges
Round 1 Pass 1          ~2.0 s   18%      Aux-trace build (LogUp GPU, exp-9)
Round 1 Pass 2          ~1.0 s    9%      Aux commits (ext3 LDE+Merkle)
Rounds 2–4              ~4.0 s   36%
```

Two places currently H2D the same main-trace data per proof:

1. **Phase A** — `coset_lde_batch_base_into_with_merkle_tree_inner`
   copies each column (total ~240 MB/table) from pinned staging into
   a device buffer of size `m * lde_size`, then overwrites in place
   with the iNTT result. The pre-LDE main trace is on device for
   a few microseconds before the iNTT kernel starts.

2. **Pass 1** — `logup_gpu::try_compute_table_term_columns` calls
   `upload_main_cols` which does the exact same H2D again. Total
   wall cost per-table is ~20–40 ms on a 32 GB/s PCIe link; exp-9
   serializes them so the total is ~200–300 ms wall on fib_1M.

Both uploads carry identical bytes (the table's main columns); the
second one is pure waste.

## The fix in two steps

### Step 1 — preserve pre-LDE columns in the fused LDE kernel

Modify `coset_lde_batch_base_into_with_merkle_tree_inner` to
optionally preserve the uploaded trace before iNTT. In the current
code, after line 769 (`memcpy_htod` loop) the first `n` u64s of each
column-slab hold the trace. A device-to-device copy to a fresh
`m*n` buffer just before the iNTT kernel is basically free (VRAM
bandwidth ≈ 1 TB/s; 240 MB copy takes <0.3 ms).

Signature sketch:

```rust
pub fn coset_lde_batch_base_into_with_merkle_tree_keep_main(
    columns: &[&[u64]],
    blowup_factor: usize,
    weights: &[u64],
    outputs: &mut [&mut [u64]],
    merkle_nodes_out: &mut [u8],
) -> Result<(GpuLdeBase, Arc<logup::DeviceMainCols>)>
```

The returned `DeviceMainCols` owns a `CudaSlice<u64>` sized `m * n`
in column-major order — directly what
`logup::logup_pair_term_column_on_device` already expects.

### Step 2 — thread the handle to aux-build

`MainTraceCommitResult` already holds an optional `GpuLdeBase`
(`gpu_main` field, line 172 of prover.rs). Add a sibling
`gpu_main_pre_lde: Option<Arc<DeviceMainCols>>`. Prover's `multi_prove`
already stashes the main LDE handle per-table; reuse the same
lookup pattern for `gpu_main_pre_lde`.

Aux-build currently receives `&mut TraceTable` + `&[challenges]`. To
reach the per-table handle without changing trait signatures, add a
module-level `RwLock<HashMap<usize, Arc<DeviceMainCols>>>` in
`logup_gpu.rs` keyed by `trace as *const _ as usize`. Prover
populates after Phase A completes; aux-build consults; prover
clears after Pass 1.

```rust
// in logup_gpu.rs
static PRE_LDE_CACHE: RwLock<HashMap<usize, Arc<DeviceMainCols>>> =
    RwLock::new(HashMap::new());

pub fn store_pre_lde_main(trace_ptr: usize, handle: Arc<DeviceMainCols>);
pub fn take_pre_lde_main(trace_ptr: usize) -> Option<Arc<DeviceMainCols>>;
pub fn clear_pre_lde_cache();
```

Inside `try_compute_table_term_columns`, skip `upload_main_cols` if
the cache has a handle for this trace pointer; drop back to the
existing H2D path otherwise (keeps the function correct for tables
that went through the non-GPU Phase A path).

## Expected win

- Per-table H2D saved: ~20–40 ms
- Total saved on fib_1M (12 tables × exp-9 serialized): 200–300 ms wall
- Aux-trace-build wall is currently ~2.0 s, so this lands it at ~1.7 s
- Total fib_1M projected: ~10.6 s (vs 10.96 s today)

At larger sizes the gain scales:
- fib_4M: estimated 600–800 ms saved (same number of tables but more
  rows, so each H2D is bigger and takes longer absolutely)

## Risks / gotchas

- **CudaSlice Send/Sync.** `DeviceMainCols` must be `Send + Sync` to
  live in an `Arc` across rayon threads. cudarc 0.19 documents
  `CudaSlice<T>: Send + Sync where T: Send + Sync`, so u64 works.
  Verify at the compile-error level, don't trust docs.
- **Cache key stability.** `trace as *const _ as usize` only works
  while the `TraceTable` isn't moved. In Pass 1 the trace is behind
  `&mut` and never reallocates, so the key is stable — but if anyone
  later refactors the aux-build loop to move traces, the cache will
  silently miss or (worse) hit a stale entry. Add a debug-assert on
  length in `try_compute_table_term_columns` matching the cache's
  stored `n`.
- **Cache lifetime.** The cache must be cleared at the start of each
  prove so stale handles don't leak into the next proof. Simplest
  location: `multi_prove` preamble. Alternative: a drop guard tied
  to the outermost prover scope.
- **Phase-A CPU fallback.** When Phase A falls back to the CPU LDE
  path (trace below the GPU threshold), no handle is produced and
  aux-build correctly falls back to its existing H2D path. No
  special-casing required.
- **Memory pressure on 32 GB VRAM.** Each pre-LDE buffer is
  `num_cols * n * 8` bytes. For fib_4M's biggest table (MEMW_R ×
  3.1M rows × ~30 cols = 750 MB) multiplied by 3 MEMW_R instances =
  2.25 GB. Plus LDE buffers (4× larger), that's ~11 GB — still fits
  comfortably on an RTX 5090. If future work increases table count,
  consider a drop-when-aux-build-finishes policy rather than
  holding through Round 4.

## Why this ships as a design, not code

The plumbing touches:
- `crypto/math-cuda/src/lde.rs` (new fused-path variant)
- `crypto/math-cuda/src/logup.rs` (cache accessors)
- `crypto/stark/src/gpu_lde.rs` (wire through the keep variant)
- `crypto/stark/src/prover.rs` (populate cache, clear at prove start)
- `crypto/stark/src/logup_gpu.rs` (consult cache, fall back)

~600–900 lines. Doable in a focused day, but not within the time
budget of the current session. Checkpointing the plan so the next
pass can execute cleanly.

Estimated effort: one focused work session plus a parity + bench
run. Expected landing: fib_1M ~10.6 s, fib_4M ~32 s → ~30 s.
