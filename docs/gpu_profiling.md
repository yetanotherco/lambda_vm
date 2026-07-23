# GPU profiling (Nsight Systems + NVTX)

How to measure what actually happens on the GPU during proving: which kernels
run, on which streams, what overlaps, where the GPU sits idle, how long the
host blocks in `synchronize()`, and what the H2D/D2H transfers cost.

The measured window is the **proving section only** — from the point the trace
(built on CPU) is uploaded to the GPU through the final proof. CPU-side
execution and trace building are excluded via `cuProfilerStart/Stop`
(`prover/src/lib.rs`, around `multi_prove`).

## TL;DR — on a GPU box (Vast.ai, no sudo needed)

```bash
make profile-gpu                 # ethrex 5-tx block
TX_COUNT=10 make profile-gpu     # bigger block
```

This builds `cli` with `--features jemalloc-stats,nvtx,prover/cuda`, builds the
ethrex guest ELF + fixture if missing, does one untraced warm-up prove, then a
profiled prove under `nsys`, and drops everything in
`/tmp/gpu_profile/<timestamp>/`:

| file | what |
|---|---|
| `prove_ethrex_5tx.nsys-rep` | full timeline — copy to your machine, open with `nsys-ui` |
| `prove_ethrex_5tx.sqlite`   | exported DB for `scripts/analyze_nsys.py` |
| `stats_*.csv`               | per-NVTX-range / per-kernel / per-memcpy / per-API summaries |
| `timeline.json`             | host-side span tree (`instruments`) |
| `prove.log`                 | `=== PROVER TIMING ===` report from the warm-up run |

`scripts/profile_gpu.sh` handles the no-sudo environment: it finds `nsys` in
PATH/the CUDA toolkit, or downloads the `nsight-systems-cli` .deb and extracts
it into `$HOME` with `dpkg -x` (no root); if `perf_event_paranoid > 2` it
disables nsys CPU sampling (CUDA/NVTX tracing doesn't need perf events).

## What the instrumentation is

Two cargo features, both **off by default** (zero overhead when off):

* **`nvtx`** (on `math-cuda`, `stark`, `prover`, `cli`; implies `instruments`)
  — NVTX ranges that nsys overlays on the CUDA timeline. When no profiler is
  attached each range costs ~ns.
  * Every `stark::instruments::span()` also emits an NVTX range (hook in
    `crypto/stark/src/instruments.rs`), so the existing phase spans
    (`proving`, `r1_main_commit`, `rounds_2to4`, …) show up for free.
  * Per-table ranges in the rayon loops of `prover.rs`:
    `r1_main:<table>`, `r1_aux_build:<table>`, `r1_aux_commit:<table>`,
    `r2to4:<table>`, plus `round2`/`round3`/`round4` inside each table.
  * Per-dispatch ranges at each GPU entry point in
    `crypto/stark/src/gpu_lde.rs` / `logup_gpu.rs`: `gpu:commit_row_major`,
    `gpu:lde_batch_base`, `gpu:extend_halves_d2`, `gpu:parts_lde`,
    `gpu:comp_poly_tree`, `gpu:bary_base`, `gpu:bary_ext3`, `gpu:deep`,
    `gpu:inv_denoms`, `gpu:r3_prep`, `gpu:fri_commit`, `fri_layer:<k>`,
    `gpu:fri_query`, `gpu:gather_proofs`, `gpu:logup_aux`, `gpu:logup_terms`.
  * Sub-phase ranges inside `math-cuda` (`crate::nvtx_range!` from
    `crypto/math-cuda/src/profiling.rs`): `pack_pinned`, `h2d`, `twiddles`,
    `bit_rev`, `ntt`, `pointwise`, `keccak_leaves`, `tree_levels`, `d2h`,
    `sync`, and the LogUp phases (`h2d_main`, `fp`, `inv`, `desc_up`, `term`,
    `accum`, `l_dtoh`).
* **`instruments`** (pre-existing) — host wall-clock span tree + the
  `=== PROVER TIMING ===` report. Span records carry an epoch `start_ns` so
  they can be aligned with external samplers.

  Under `cuda`, `instruments` also compiles **in-process GPU event timing**
  (`crypto/math-cuda/src/timing.rs`): `crate::gpu_span!` records CUDA-event
  pairs per op/stream without synchronizing, and `timing::timed_sync` measures
  host blocking inside `synchronize()`. Collection is off unless
  `LAMBDA_VM_GPU_TIMELINE=1` (or `LAMBDA_VM_GPU_TIMELINE_JSON=<path>`) is set
  at runtime, so instrumented builds pay nothing by default. When enabled, the
  prover prints an `=== GPU TIMING (CUDA events) ===` section (device ms per
  op, window coverage, host-blocked-in-sync) and can write a combined
  host+device **Chrome trace** (open in Perfetto) — per-phase GPU numbers on
  every run, no nsys required. This is the tool for CI and for measuring
  optimizations run-over-run; nsys remains the tool for kernel-level structure.

## Reading the results

Start with the summary CSVs / `analyze_nsys.py` output, then open the
`.nsys-rep` in the GUI for anything surprising.

The questions this setup answers, and where to look:

* **Where does the wall time go?** `nvtx_sum` stats: per range, total time and
  instance count. Compare `r1_main_commit` vs `r1_aux_build` vs `rounds_2to4`.
* **Is the GPU busy or waiting?** In the GUI: the CUDA HW row under each
  NVTX phase. Gaps = CPU-bound stretches or serialization. The
  `cuda_api_sum` report shows total time in `cuStreamSynchronize` /
  `cuMemcpy*` — host blocking.
* **Does table-level parallelism actually overlap on device?** Each table
  binds one of the 32 pool streams (`math-cuda/src/device.rs`,
  `trace.rs::bound_stream`). In the GUI, count concurrently-active stream
  rows inside `r1_main_commit`. If one stream dominates, the rayon fan-out is
  not translating into GPU overlap.
* **What do transfers cost?** `cuda_gpu_mem_time_sum` / `mem_size_sum`:
  time + bytes per memcpy kind (pageable vs pinned matters — the base-field
  main-trace upload is currently pageable, `math-cuda/src/lde.rs`).
* **Which kernels dominate?** `cuda_gpu_kern_sum`.

## Caveats / environment notes

* **`ncu` (Nsight Compute) will usually NOT work on rented boxes**: per-kernel
  counters require `RmProfilingAdminOnly=0` in the *host* driver
  (`ERR_NVGPUCTRPERM` otherwise). The script reports this. Per-kernel deep
  dives need a box where you control the driver.
* The profiled run pays some tracing overhead; use the warm-up run's
  `Proving time` (in `prove.log`) for wall-time numbers, and the nsys run for
  structure/attribution.
* `NO_NSYS=1 scripts/profile_gpu.sh` runs the instrumented prove without any
  profiler (instruments report + timeline.json only).
* Force the GPU path on small traces with `LAMBDA_VM_GPU_LDE_THRESHOLD=8`
  (default threshold is `1<<19` rows). Other toggles:
  `LAMBDA_VM_DISABLE_GPU_COMPOSITION`, `LAMBDA_VM_DISABLE_DEVICE_ONLY`,
  `LAMBDA_VM_NO_GPU_LOGUP`, `LAMBDA_VM_VRAM_BUDGET_MB`.
* Server build reminders: run cargo with `--manifest-path prover/Cargo.toml`
  (not `-p prover`), guest Rust ELFs need the sysroot
  (`SYSROOT_DIR`, `make compile-programs-rust`).

## Workload policy

Benchmarks and profiles use the **ethrex** guest: a 5-transfer block
(`TX_COUNT=5`) as the small/base case, larger blocks (10/20 tx, optionally
`CONTINUATIONS=1`) to scale. `bench_single` (fib) remains only as a quick
smoke test.
