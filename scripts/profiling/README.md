# GPU profiling toolkit

Tooling for profiling the CUDA prover on the dedicated RTX 5090 box: this
directory is the executable part of the profiling methodology (measure with
`run_profile.sh`, rank phases by `phase_busy.md`, then drill into kernels).

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

The ethrex transfer fixtures are generated, not checked in — build one first:

```bash
( cd tooling/ethrex-fixtures && cargo build --release )
tooling/ethrex-fixtures/target/release/ethrex-fixtures 5 executor/tests/ethrex_5_transfers.bin distinct
```

```bash
# 3 instrumented runs + phase table with GPU util per phase:
scripts/profiling/run_profile.sh executor/program_artifacts/rust/ethrex.elf \
  --private-input executor/tests/ethrex_5_transfers.bin

# the real workload, plus an nsys-traced run and the per-phase GPU-busy report:
scripts/profiling/run_profile.sh --nsys \
  executor/program_artifacts/rust/ethrex.elf \
  --private-input executor/tests/ethrex_5_transfers.bin

# big continuation run, nsys capture limited to one `epoch_prove` span (one epoch):
LAMBDA_VM_NSYS_CAPTURE_SPAN=epoch_prove scripts/profiling/run_profile.sh --nsys --continuations \
  executor/program_artifacts/rust/ethrex.elf \
  --private-input executor/tests/ethrex_10_transfers.bin
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

## What each column means

`phase_table.md` (from the instruments spans + the NVML sampler):

| column | meaning |
|---|---|
| `median` | wall-clock of the span, median across warm runs; repeated spans (per-epoch, per-table) are **summed within a run** first (`n/run` = instances) |
| `% of total` | share of the root span (`prove_total` / `prove_continuation_total`) |
| `cv%` | run-to-run spread (stdev/mean) — the noise floor an optimization claim must beat |
| `gpu% / mem%` | average `nvidia-smi` utilization **sampled at 10 Hz inside the span's wall window**. Coarse: SM-occupancy-ish, includes other phases' async kernels landing in the window |
| `vram MiB` | max VRAM sampled inside the window |
| instances tables | per-instance wall, `gap→next` (host time between consecutive instances), gpu% inside the span vs inside the gap |

`phase_busy.md` (from the nsys sqlite export; only exists with `--nsys`):

| column | meaning |
|---|---|
| `wall ms` | union of that phase's NVTX windows (merged if overlapping) |
| `kernel-sum ms` | sum of kernel durations **attributed by correlation ID to launches made inside the phase** — can exceed wall when streams overlap |
| `gpu-busy ms / busy%` | union coverage of kernel intervals clipped to the phase windows — the honest "GPU was doing *something*" number; the one to rank phases by |
| `h2d / d2h ms/MiB` | memcpy time and volume attributed to the phase |

The two GPU numbers answer different questions: `gpu%` (NVML) is a cheap
always-on sanity signal; `busy%` (nsys) is the precise one — trust it when
they disagree. Attribution is by *launch site*, so async kernels count toward
the phase that enqueued them even if they execute later.

## Reference: scripts and knobs

| script | what it does / flags |
|---|---|
| `run_profile.sh [opts] <elf> [--private-input <bin>]` | the bundle. `--runs N` (default 3; run 1 kept separately as cold), `--nsys`, `--gpu-metrics` (needs the counters permission), `--continuations`, `--out DIR`, `--no-build`. Env: `PROFILE_FEATURES` (default `nvtx,jemalloc-stats`), `EXTRA_PROVE_ARGS` |
| `flamegraphs.sh [opts] <elf> …` | on-CPU + off-CPU SVGs. `--offcpu-secs N` overrides the capture window, `--skip-offcpu`, `--no-build`, `--continuations` |
| `bench_mode.sh on [mhz] / off` | lock/unlock SM clocks (default 90% of max), persistence, governor |
| `capture_env.sh` | env JSON to stdout — attach to anything you measure by hand |
| `phase_table.py [--util u.csv]… tl.json…` | aggregate timelines; `--instances LABEL` adds per-instance tables for deeper repeated spans, `--min-pct X` hides noise rows |
| `nsys_phase_busy.py report.sqlite [--top N]` | the GPU busy report from `nsys export --type sqlite` |
| `nvml_sampler.py -o out.csv [-i 0.1]` | standalone 10 Hz GPU util sampler (epoch-ns timestamps, aligns with span `start_ns`) |
| `timeline_to_perfetto.py tl.json > trace.json` | span tree for ui.perfetto.dev |

Environment variables the tooling understands:

| var | effect |
|---|---|
| `LAMBDA_VM_TIMELINE_JSON=<path>` | prover writes the span timeline there (needs `instruments`) |
| `LAMBDA_VM_NSYS_CAPTURE_SPAN=<label>` | cuProfilerStart/Stop around that span → with `run_profile.sh --nsys`, capture only that window (`epoch_prove`, `rounds_2to4`, …) |
| `LAMBDA_VM_NVTX_LIB=<path>` | where to dlopen `libnvToolsExt.so.1` from |
| `LAMBDA_VM_NVCC_LINEINFO=1` | build cubins with SASS→source mapping for ncu |

Useful prover knobs for A/B experiments (pre-existing, see plan §11):
`LAMBDA_VM_DISABLE_GPU_COMPOSITION`, `LAMBDA_VM_NO_GPU_LOGUP`,
`LAMBDA_VM_DISABLE_DEVICE_ONLY`, `LAMBDA_VM_GPU_LDE_THRESHOLD`,
`LAMBDA_VM_GPU_BARY_THRESHOLD`, `LAMBDA_VM_VRAM_BUDGET_MB`,
`TABLE_PARALLELISM`.

| var | effect |
|---|---|
| `LAMBDA_VM_NO_GPU_GRIND=1` | force the round-4 proof-of-work nonce search onto the CPU (presence-based, like `LAMBDA_VM_NO_GPU_LOGUP`). The production escape hatch if the device search ever misbehaves; also the way to A/B the grind on its own. Below grinding factor 12 the GPU path declines regardless, so wrap and recursion proves (factor 1) never use it |

## Continuations: per-epoch data for parallelization

`prove_continuation` is instrumented independently of the monolithic path
(`prover/src/continuation.rs`): a `prove_continuation_total` root, per-epoch
`epoch_execute` / `epoch_collect` / `epoch_trace_build` / `epoch_prove` spans,
a `prove_global` span, and dynamic `epoch_collect[i=N]` /
`epoch_trace_build[i=N]` / `epoch_prove[i=N]` NVTX ranges per epoch and
pipeline stage — and it writes `LAMBDA_VM_TIMELINE_JSON` at the end (before
this instrumentation, continuation runs produced no timeline at all).

With `run_profile.sh --continuations` the reports gain the epoch-level view
that drives parallelization decisions:

- `phase_table.md` — per-epoch instances table: wall per epoch (uniformity —
  can epochs be scheduled symmetrically?), gap→next, and **GPU util inside the
  span vs inside the gap**. `epoch_execute`/`epoch_trace_build` rows with ~0
  GPU% while `epoch_prove` is high = the fraction of wall recoverable by
  overlapping epoch N+1's CPU work with epoch N's GPU proving.
- `phase_busy.md` — "Epochs (continuations)" section from the
  `epoch_{prove,trace_build,collect}[i=N]` ranges, grouped per stage:
  per-instance GPU busy%, gaps to the next instance of the same stage and GPU
  work inside them (a gap ≈0 on `epoch_prove` = the pipeline keeps the prover
  fed; a growing one = collect/build stopped hiding behind the proves).
- In the Nsight GUI, epochs appear as named `epoch_prove[i=N]` (and
  `epoch_trace_build[i=N]` / `epoch_collect[i=N]`) ranges, so cross-epoch
  overlap (or the lack of it) is visible at a glance; combine with
  `LAMBDA_VM_NSYS_CAPTURE_SPAN=epoch_prove` to capture a single epoch's prove.

## The `nvtx` cargo feature

`run_profile.sh` builds the cli with `--features nvtx,jemalloc-stats`. The
`nvtx` feature (cli → prover → stark → math-cuda) does three things:

1. every `instruments` span (prover phases) becomes an NVTX range, so Nsight
   timelines carry the same names as the phase table;
2. the finer `gpu_span!`/dynamic ranges some call sites emit (e.g.
   `epoch_prove[i=3]`) appear nested inside those phase ranges;
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

## Troubleshooting

- **No NVTX ranges in the nsys report** (`nsys_phase_busy.py` warns): the
  binary didn't find `libnvToolsExt.so.1`. Check `LAMBDA_VM_NVTX_LIB` /
  `~/nvtx/` — a missing library no-ops silently by design.
- **Numbers look great but the GPU path silently fell back to CPU**: every
  dataset must be gated on `make test-cuda-integration` (it asserts every
  `gpu_*_calls()` counter fired). A cubin/arch mismatch falls back without
  erroring.
- **Empty `offcpu.folded`**: rerun with `--offcpu-secs <prove seconds + 10>`;
  the bcc tool needs `linux-headers` for the running kernel.
- **`Proving time` jitter above ~1%**: clocks aren't locked (`bench_mode.sh
  on`) or something else is running on the box; check
  `nvidia-smi -q -d PERFORMANCE` for throttle reasons.
- **Spans from concurrent threads** (e.g. the epoch-pipeline branch): the
  tree in `phase_table.md` mis-nests spans recorded on different threads
  (order-based reconstruction, no tid in `SpanRecord`). The per-instance
  tables and everything nsys-side stay correct — use those.

## Not yet built (next steps on the box)

- kernel microbenchmark harness with CUDA events (plan §10) — needs GPU
  iteration to be worth writing;
- nightly regression cron (plan §2.5) — depends on the microbench.
