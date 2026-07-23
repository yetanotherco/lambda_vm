# GPU profiling toolkit

Tooling for profiling the CUDA prover on the dedicated RTX 5090 box. The full
methodology (what to measure, in what order, and how to read it) lives in
`thoughts/gpu-profiling/plan.md`; this directory is the executable part.

## One-time machine setup

```bash
scripts/profiling/setup_machine.sh   # apt tooling, nsight, perf/eBPF perms — then REBOOT
```

Prereqs handled elsewhere: NVIDIA driver, and the build toolchain + guest
programs per `scripts/SERVER_SETUP.md` (`make compile-programs-asm`, etc.).
On Blackwell (RTX 5090, sm_120) CUDA ≥ 12.8 and nsys/ncu ≥ 2025.1 are hard
requirements.

## Before every session

```bash
sudo scripts/profiling/bench_mode.sh on   # lock SM clocks, persistence, governor
make test-cuda-integration                # sanity: every GPU counter fires
```

`bench_mode.sh off` restores defaults. Numbers taken with floating clocks are
not comparable across sessions.

## The main entry point

```bash
# 3 instrumented runs + phase table with GPU util per phase:
scripts/profiling/run_profile.sh executor/program_artifacts/asm/fib_iterative_372k.elf

# the real workload, plus an nsys-traced run and the per-phase GPU-busy report:
scripts/profiling/run_profile.sh --nsys \
  executor/program_artifacts/rust/ethrex.elf \
  --private-input executor/tests/ethrex_5_transfers.bin

# big continuation run, nsys capture limited to one `proving` span (one epoch):
LAMBDA_VM_NSYS_CAPTURE_SPAN=proving scripts/profiling/run_profile.sh --nsys --continuations \
  executor/program_artifacts/rust/ethrex.elf \
  --private-input executor/tests/ethrex_20_transfers.bin
```

Each invocation produces a self-contained bundle under
`reports/<workload>_<sha>_<timestamp>/`:

| file | what |
|---|---|
| `env.json` | driver, clocks, sha, env — the context that makes numbers comparable |
| `phase_table.md` | warm-run phase tree: median ms, % of total, jitter, GPU util per phase |
| `phase_table_cold.md` | run 1 alone (module load, twiddle build, mempool growth) |
| `trace_perfetto.json` | span tree for ui.perfetto.dev |
| `nsys_report.nsys-rep` | open in the Nsight Systems GUI (scp to a laptop) |
| `nsys_stats.txt` | stock nsys summaries (kernels, memcpy, API, NVTX) |
| `phase_busy.md` | **the ranking input**: per-phase GPU busy %, memcpy, top kernels |

How to read `phase_busy.md`: a phase with low busy% is host-bound — fix
pipeline/overlap/syncs, don't touch its kernels. A phase with high busy% is
kernel-bound — take its top kernels to Nsight Compute.

## Continuations: per-epoch data for parallelization

`prove_continuation` is instrumented independently of the monolithic path
(`prover/src/continuation.rs`): a `prove_continuation_total` root, per-epoch
`epoch` → `epoch_execute` / `epoch_trace_build` / `epoch_prove` spans, a
`prove_global` span, and a dynamic `epoch[i=N]` NVTX range per epoch — and it
writes `LAMBDA_VM_TIMELINE_JSON` at the end (before this instrumentation,
continuation runs produced no timeline at all).

With `run_profile.sh --continuations` the reports gain the epoch-level view
that drives parallelization decisions:

- `phase_table.md` — per-epoch instances table: wall per epoch (uniformity —
  can epochs be scheduled symmetrically?), gap→next, and **GPU util inside the
  span vs inside the gap**. `epoch_execute`/`epoch_trace_build` rows with ~0
  GPU% while `epoch_prove` is high = the fraction of wall recoverable by
  overlapping epoch N+1's CPU work with epoch N's GPU proving.
- `phase_busy.md` — "Epochs (continuations)" section from the `epoch[i=N]`
  ranges: per-epoch GPU busy%, inter-epoch gaps and GPU work inside them
  (≈0 → idle gap = pipelining headroom, quantified in ms).
- In the Nsight GUI, epochs appear as named `epoch[i=N]` ranges, so
  cross-epoch overlap (or the lack of it) is visible at a glance; combine with
  `LAMBDA_VM_NSYS_CAPTURE_SPAN=epoch_prove` to capture a single epoch's prove.

## The `nvtx` cargo feature

`run_profile.sh` builds the cli with `--features nvtx,jemalloc-stats`. The
`nvtx` feature (cli → prover → stark → math-cuda) does three things:

1. every `instruments` span (prover phases) becomes an NVTX range, so Nsight
   timelines carry the same names as the phase table;
2. every `math-cuda` public entry point pushes a range with its shape, e.g.
   `lde_tree_base[n=1048576 m=48 bf=4]`;
3. `LAMBDA_VM_NSYS_CAPTURE_SPAN=<label>` brackets that span with
   cuProfilerStart/Stop, gating `nsys --capture-range=cudaProfilerApi` (used
   automatically by `run_profile.sh --nsys` when the env var is set).

The bindings dlopen `libnvToolsExt` at runtime (`crypto/math-cuda/src/nvtx.rs`);
without the library or without an attached profiler everything is a cheap no-op.
Without the feature it compiles to nothing.

CUDA-version caveat: toolkits ≥ 12.9 no longer ship `libnvToolsExt` (NVTX v2
was removed; only the v3 headers remain). `setup_machine.sh` handles it by
installing `cuda-nvtx-12-8` alongside whatever toolkit is present; the
`LAMBDA_VM_NVTX_LIB` env var overrides the library path if it lives somewhere
unusual. Verify ranges actually appear in the first nsys report — a missing
library no-ops silently by design.

## Nsight Compute (kernel deep dives)

Only for kernels that `phase_busy.md` proved dominant in a GPU-bound phase,
and never for timing (ncu replays kernels — durations are meaningless there):

```bash
# rebuild cubins with SASS→source line mapping (does not change codegen):
LAMBDA_VM_NVCC_LINEINFO=1 cargo build --release -p cli --features nvtx,jemalloc-stats

ncu --set full -k 'regex:ntt_dit|keccak' --launch-skip 20 --launch-count 10 \
  -o reports/ncu_ntt ./target/release/cli prove <elf> ...
```

## CPU flamegraphs

```bash
# on-CPU + off-CPU SVGs for one workload, in one command:
scripts/profiling/flamegraphs.sh executor/program_artifacts/asm/fib_iterative_372k.elf

scripts/profiling/flamegraphs.sh executor/program_artifacts/rust/ethrex.elf \
  --private-input executor/tests/ethrex_5_transfers.bin
```

Produces `oncpu.svg` (perf, 997 Hz, frame-pointer stacks: host work between
kernel launches — serialization, transposes, Fiat-Shamir, staging copies) and
`offcpu.svg` (offcputime-bpfcc: where threads *wait* — `cuStreamSynchronize`,
futexes; invisible in the on-CPU graph). Read them against `phase_busy.md`:
low GPU busy% + little on-CPU time in a phase = blocked time, and the off-CPU
graph names the stack. For an interactive view, `samply record
./target/release/cli prove <elf> ...` opens the Firefox profiler UI.

## Not yet built (next steps on the box)

- kernel microbenchmark harness with CUDA events (plan §10) — needs GPU
  iteration to be worth writing;
- nightly regression cron (plan §2.5) — depends on the microbench.
