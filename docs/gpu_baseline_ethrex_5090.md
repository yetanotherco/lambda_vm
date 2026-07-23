# GPU proving baseline — ethrex on RTX 5090 (2026-07-22)

Reference measurements taken with the instrumentation from this branch
(`feat/gpu-instrumentation`): NVTX + nsys (`make profile-gpu` +
`scripts/analyze_nsys.py`) and in-process CUDA-event timing
(`LAMBDA_VM_GPU_TIMELINE=1`). Use these numbers as the before-picture for GPU
optimizations; reproduce with the commands at the bottom.

**Environment**: Vast.ai RTX 5090 32 GB (driver 580.119.02), CUDA 13.1,
48-core host, rust 1.94.0. Guest ELF built from this tree (a stale ethrex ELF
executes ~4× more cycles and OOMs — always rebuild the guest when benching).
Workload: ethrex block fixtures, 5 transfers (base case) and 10 transfers.

Measurement overhead: `instruments`+`nvtx` build with collection off ≈ **0%**
(9.58s vs 9.57s baseline); `LAMBDA_VM_GPU_TIMELINE=1` ≈ **+4%**; under nsys
≈ +40% (use nsys for structure, not absolute walls).

## 5 tx, monolithic — E2E 9.61s, proving 6.01s

| phase | wall | % E2E | GPU util inside (nsys) |
|---|---|---|---|
| Execute | 0.21s | 2.2% | — |
| Trace build | 2.54s | 26.4% | — |
| AIR construction | 0.84s | 8.8% | — |
| Proving | 6.01s | 62.5% | see below |
| · R1 prepass | 0.05s | | 0% |
| · R1 main commit | 0.87s | | 51% |
| · R1 aux build (LogUp) | 0.72s | | **16%** |
| · R1 aux commit | 0.64s | | 59% |
| · Rounds 2–4 | 3.50s | | 58% |

Peak heap 11.8 GB; peak VRAM **25.7 GB** (monolithic 10tx OOMs — the 32 GB
limit sits between 5 and 10 tx; device-only paths hard-abort on OOM).

### GPU device view (window 5.4s, nsys)

| bucket | time | % |
|---|---|---|
| GPU busy (kernels ∪ memcpy) | 2.77s | 51.4% |
| · kernels (union) | 1.55s | 28.7% |
| · memcpy (union) | 1.72s | 31.8% |
| GPU idle | 2.62s | 48.6% |

Kernel concurrency: **0 kernels running 71.3%** of the window, exactly one
18.9%, ≥2 only ~10% — the 32-stream pool yields almost no overlap
(per-op H2D→kernels→D2H→sync pattern serializes).

### GPU time by phase (nsys attribution via launching thread's NVTX stack)

| phase | kernels | memcpy | launches |
|---|---|---|---|
| R2 composition | 985ms | 734ms | 2,423 |
| R1 aux commit | 446ms | 65ms | 3,474 |
| R1 main commit | 364ms | 156ms | 1,886 |
| R3 OOD | 325ms | 23ms | 776 |
| R1 aux build (logup) | 93ms | 34ms | 1,160 |
| R4 FRI | 89ms | **363ms** | 9,333 |
| R4 DEEP | 37ms | **574ms** | 29 |
| R3/R4 inv denoms | 41ms | 22ms | 612 |
| R4 queries | 7ms | 4ms | 690 |

DEEP is 15:1 copy:compute (downloads its full result; consumer is FRI, on
device). FRI is 4:1 (downloads full layer evals; the host query phase only
needs ~queries×layers values).

### Per-op device latency (CUDA events; enqueue→completion incl. stream stalls)

| op | dev ms | spans | host blocked in final sync |
|---|---|---|---|
| gpu:lde_row_major | 8,004 | 86 | 260ms |
| gpu:eval_composition_on_device | 3,510 | 29 | 0.4ms |
| gpu:lde_batch_ext3_into | 3,126 | 50 | 533ms |
| gpu:fri_layer | 1,273 | 604 | 107ms |
| gpu:deep_composition_ext3 | 1,161 | 29 | 0.6ms |
| gpu:compute_and_invert_denoms | 1,079 | 102 | — |
| gpu:barycentric_* (4 variants) | 1,181 | 244 | 3ms |
| gpu:gather_merkle_paths | 476 | 690 | 11ms |
| gpu:logup_aux_resident | 218 | 57 | 0.2ms |

Cross-read: eval_composition = 3.5s latency vs 0.86s pure kernel → ~2.6s of
stream stall (transfer/dependency water) inside the op. Same shape for
lde_row_major (8.0s vs ~0.8s kernels).

### Host side (sums across ~10 rayon threads; can exceed wall)

| API | total | calls |
|---|---|---|
| cuMemcpyDtoHAsync (host blocked) | **12.1s** | 2,635 |
| cuMemcpyHtoDAsync | 3.6s | 3,832 |
| cuMemHostAlloc + cuMemFreeHost (pinned churn) | **4.2s** | 55 |
| cuStreamSynchronize | 1.2s | 2,711 |
| cuLaunchKernel | 0.7s | 20,469 |

Transfers: 16.8 GB H2D, 7.3 GB D2H, 4.2 GB D2D, 53.5 GB memset
(alloc_zeros). CPU-only work inside proving: main-commit Merkle 2.4s
(thread-sum, tables off the GPU path), LogUp fingerprint+invert 0.8s.

### Top kernels (pure GPU time)

| kernel | total | launches | avg |
|---|---|---|---|
| constraint_composition_kernel | 861ms | 29 | 29.7ms |
| ntt_dit_level_row_major | 595ms | 3,296 | 0.18ms |
| barycentric_{ext3,base}_batched_strided | 311ms | 164 | ~1.9ms |
| keccak_merkle_level | 136ms | 10,087 | 14µs |
| keccak256_leaves_base_row_major_row_pair | 74ms | 86 | 0.86ms |

### GPU idle gaps > 1ms (by phase whose kernel ran next)

R1 aux commit 488ms · R1 main 411ms (largest single: 244ms) · logup 377ms ·
FRI 319ms · R2 301ms · queries 197ms · R3 190ms — spread out: structural
serialization, not one culprit.

## 10 tx, continuations (7 epochs of 2^20) — proving 23.3s

| | 5tx monolithic | 10tx continuations |
|---|---|---|
| Proving | 9.45s (wall 6.0s in-prove) | 23.3s |
| Peak VRAM | 25.7 GB | **8.6 GB (flat)** |
| GPU window / covered | 5.8s / **65%** | 21.9s / **29%** |

Per-op ranking matches 5tx (lde_row_major 7.3s, eval_composition 2.2s, …).
The dominant cost is epoch serialization: each epoch executes + builds its
trace on CPU before proving on GPU, so the GPU starves between epochs.

## Ranked optimization opportunities (from these numbers)

1. **Epoch pipelining (continuations)** — overlap epoch N+1 CPU
   execute/trace-build with epoch N GPU proving. GPU coverage 29%; est.
   23.3s → ~12-14s on 10tx. Host orchestration only.
2. **Break intra-prove serialization** — 49% GPU idle, 71% zero-kernels:
   async D2H (events, not blocking), per-table stream chains without
   intermediate syncs, overlap D2H with next compute.
3. **Eliminate D2H volume** — DEEP result stays on device (consumer is FRI);
   FRI query gathers the ~219×layers needed values on device instead of
   downloading full layers; R2 parts residency.
4. **Pinned slab cache** — pre-size + reuse: 4.2s of host alloc/free churn.
5. **constraint_composition_kernel** — 861ms, 56% of kernel time.
6. **Fuse NTT row-major levels** — 595ms over 3,296 one-level launches
   (precedent: ntt_dit_8_levels_batched).
7. **CPU remnants in proving** — CPU merkle for off-GPU tables (2.4s
   thread-sum), LogUp CPU fingerprint/invert (phase at 16% GPU util).
8. **VRAM robustness** — device-only paths hard-abort on OOM (no fallback);
   estimator misses transients; mempool retention unbounded. Blocks
   monolithic >5tx on 32 GB.
9. **Micro** — batch keccak_merkle_level launches (10k×14µs), skip memsets
   on fully-overwritten buffers, reduce total launch count (20.5k).

Also outside proving: trace build is 2.54s (26% of E2E) on CPU — second
largest E2E cost; epoch pipelining absorbs it for continuations.

## Reproduce

```bash
# full suite: warm-up (instruments + CUDA-event timing) + nsys + analyzer
OUT_DIR=/root/gpu_profile_5tx make profile-gpu           # TX_COUNT=10, CONTINUATIONS=1 to vary

# in-process timing only (any run, ~4% overhead)
LAMBDA_VM_GPU_TIMELINE=1 LAMBDA_VM_GPU_TIMELINE_JSON=t.json \
  cli prove ethrex.elf --private-input executor/tests/ethrex_5_transfers.bin -o p.bin --time
```

See `docs/gpu_profiling.md` for the tooling guide.
