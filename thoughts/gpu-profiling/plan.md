# GPU Proving — Detailed Profiling Plan (RTX 5090)

> **Status (branch `gpu-profiling-tooling`)** — done before machine access:
> Layer 2 (NVTX) is implemented: `nvtx` cargo feature (cli → prover → stark →
> math-cuda), instruments spans mirrored as NVTX ranges, shape-annotated
> ranges on every math-cuda entry point, `LAMBDA_VM_NSYS_CAPTURE_SPAN`
> capture-range gating, `LAMBDA_VM_NVCC_LINEINFO=1` for ncu source mapping.
> The tooling in `scripts/profiling/` covers §2 (setup/bench-mode/env-capture),
> §4 (phase_table.py + nvml_sampler.py + perfetto export), §6.2
> (nsys_phase_busy.py, run_profile.sh) and §7 (flamegraphs.sh: on-CPU perf +
> off-CPU offcputime SVGs in one command). Analyzers are tested on synthetic
> data; everything GPU-touching still needs first contact with the real box.
> Continuations are first-class: `prove_continuation` now has its own span
> tree (per-epoch execute/trace_build/prove + `epoch[i=N]` NVTX ranges) and
> writes the timeline JSON (it previously produced no instruments output at
> all); phase_table.py reports per-epoch instances/gaps/GPU-util and
> nsys_phase_busy.py adds a per-epoch busy/gap section — the direct input for
> epoch-pipelining decisions.
> Pending: Layer 7 microbench + nightly cron (need GPU iteration), and the
> nvtx-overhead ABBA check (§5.2).

Goal: build a complete, reproducible picture of where wall-clock time goes during a
GPU prove, down to individual kernels, so optimization work is driven by data instead
of intuition. Target machine: **dedicated, owned bare-metal Linux box** with sudo +
RTX 5090 (Blackwell, sm_120, 32 GB GDDR7, ~1.79 TB/s peak DRAM bandwidth — the
roofline number that matters for our mostly memory-bound kernels). Being our own
persistent machine (not a rented container) means: full driver/module control, all
profiling permissions work, setup is one-time, and the box doubles as a long-term
regression-tracking rig (see §2.5).

The plan is layered top-down: each layer answers a question, and the answer decides
where to zoom in next. Layers 1–3 require no code changes (or one small NVTX patch);
layers 4–7 are targeted deep dives.

---

## 0. Key questions this profiling must answer

1. **GPU busy %**: of total proving wall-clock, what fraction is the GPU actually
   executing kernels? Idle gaps = host-side serialization, launch overhead, or syncs.
2. **Phase breakdown**: which round dominates? R1 (LDE+Merkle commit) vs R1 aux
   (LogUp) vs R2 (constraint eval + composition) vs R3 (OOD) vs R4 (DEEP+FRI+queries).
3. **Transfer cost**: how much time goes to H2D/D2H? Is the known GPU→host→GPU
   round-trip between trace expansion and R1 LDE (~250 ms/prove on ethrex-5tx)
   visible and quantified? Pinned vs pageable? Does copy overlap compute?
4. **Stream concurrency**: the 32-stream pool (`device.rs`) — do tables actually
   overlap on the GPU, or do mempool/twiddle-cache/sync dependencies serialize them?
5. **Sync points**: ~50 `stream.synchronize()` calls in `math-cuda` + more in
   `stark`. Which ones actually stall the pipeline?
6. **Kernel efficiency**: for the top kernels by total time (expected: `ntt_dit_*`,
   `keccak256_leaves_*`, `keccak_merkle_level`, `constraint_interp_kernel`,
   `logup_*`, batch-inverse scans), what % of peak DRAM bandwidth / SM occupancy do
   they reach? Where are they on the roofline?
7. **CPU-side cost**: what host work sits between GPU launches (data prep,
   transposes, Fiat-Shamir, serialization, rkyv)? On-CPU vs blocked-waiting-on-GPU.
8. **Memory behavior**: peak VRAM per workload, mempool hit rate vs real
   `cuMemAllocAsync`, host heap peak (jemalloc), VRAM-budget admission decisions.
9. **Scaling**: how do all of the above change with trace size (2^15 → 2^25 rows),
   `TABLE_PARALLELISM`, and continuations (`EPOCH_SIZE_LOG2`)?

---

## 1. What already exists (use it, don't rebuild it)

| Asset | Where | What it gives |
|---|---|---|
| `instruments` feature | `crypto/stark/src/instruments.rs` (352 L) | RAII wall-clock span tree: `r1_prepass`, `r1_main_commit`, `r1_aux_build`, `r1_aux_commit`, `rounds_2to4`, per-table `TableSubOps` (constraints, comp_commit, ood, deep, fri_commit, queries), `Round1SubOps`. Spans record `start_ns` epoch **specifically for aligning with external samplers**. |
| Timeline output | `prover/src/lib.rs:1194` | stdout tree + `LAMBDA_VM_TIMELINE_JSON=<path>` JSON dump. |
| GPU dispatch counters | `crypto/stark/src/gpu_lde.rs:60-135` | `gpu_*_calls()` atomics per GPU phase — the ground truth that a phase ran on GPU and didn't silently fall back to CPU. |
| Phase kill-switches | env vars | `LAMBDA_VM_DISABLE_GPU_COMPOSITION`, `LAMBDA_VM_NO_GPU_LOGUP`, `LAMBDA_VM_DISABLE_DEVICE_ONLY`, `LAMBDA_VM_GPU_LDE_THRESHOLD`, `LAMBDA_VM_GPU_BARY_THRESHOLD` — free A/B experiments per GPU phase. |
| Memory knobs | env vars | `LAMBDA_VM_VRAM_BUDGET_MB`, `LAMBDA_VM_MEMPOOL_RELEASE_MB`, `TABLE_PARALLELISM`. |
| Heap profile | `prover/src/instruments.rs` | jemalloc peak-heap report (`jemalloc-stats` feature). |
| ABBA harness | `scripts/bench_abba.sh` | paired statistical comparison — reuse for validating any optimization later. |
| Bench binaries | `prover/benches/profile_vm_prover.rs`, `crypto/stark/benches/` | plain (non-criterion) binaries made for profilers. |
| GPU integration test | `prover/tests/cuda_path_integration.rs` | asserts every GPU counter fired — sanity gate before any measurement session. |

What does **not** exist: NVTX ranges, `tracing` crate, any kernel-level timing.
That's the main instrumentation gap (Layer 2).

---

## 2. Machine setup (one-time, scripted)

Everything below goes into `scripts/profiling/setup_machine.sh`. On an owned box
this runs **once** — make every setting persistent (`/etc/sysctl.d/`,
`/etc/modprobe.d/`, systemd units) instead of re-applying per session.

### 2.1 Toolchain
- Driver ≥ 570, CUDA toolkit ≥ 12.8 (Blackwell floor; cudarc is pinned `cuda-12080`).
  Verify: `nvidia-smi --query-gpu=compute_cap --format=csv` → `12.0`; `nvcc --version`.
- **Nsight Systems ≥ 2025.1** (`nsys`) and **Nsight Compute ≥ 2025.1** (`ncu`) —
  older versions don't know sm_120. Install from NVIDIA apt repo, not Ubuntu's.
- `perf` (linux-tools matching kernel), `hyperfine`, flamegraph tooling
  (`cargo install flamegraph inferno`), `samply` (already the intended consumer of
  `profile_vm_prover.rs`), `heaptrack` (optional), `python3` + `sqlite3` (nsys export
  analysis), `tmux`.
- Repo build deps per `scripts/SERVER_SETUP.md` (clang-18, rust nightly, sysroot),
  then `make compile-programs-asm compile-programs-rust` + ethrex fixtures
  (`tooling/ethrex-fixtures`).

### 2.2 Permissions (one-time, persistent)
```bash
# perf: allow full sampling + kernel symbols (persist across reboots)
printf 'kernel.perf_event_paranoid=-1\nkernel.kptr_restrict=0\n' | \
  sudo tee /etc/sysctl.d/99-gpu-profiling.conf && sudo sysctl --system
# GPU perf counters for ncu / nsys --gpu-metrics without root:
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | \
  sudo tee /etc/modprobe.d/nvidia-profiling.conf
# reload the module once (reboot is simplest) and it's set forever
```
Bare metal + root means the full toolset works with no caveats: `ncu` as regular
user, `nsys --gpu-metrics-devices`, eBPF tools (`bpfcc-tools`/`bpftrace` for
off-CPU analysis), exact-matching `linux-tools` for perf. Install everything once.

### 2.3 Reproducibility controls (apply before every session)
```bash
sudo nvidia-smi -pm 1                     # persistence mode
sudo nvidia-smi -lgc <base>,<base>        # lock SM clock (pick a sustainable clock,
                                          #  e.g. ~90% of boost; -lmc may be rejected
                                          #  on GeForce — record clocks if so)
sudo cpupower frequency-set -g performance
```
On an owned box, wrap these in a systemd oneshot (`gpu-bench-mode.service`) so the
machine always boots into a known measurement state. Also silence background noise
once: disable `unattended-upgrades`, indexing daemons, and anything else that can
steal CPU/PCIe mid-run; nothing else should run during measurement sessions.
- Record in every result file: driver, clocks, temp, VRAM free, commit SHA, feature
  flags, env vars. A `scripts/profiling/capture_env.sh` that dumps this as JSON.
- **Warm vs cold**: first prove pays cubin module load, twiddle-cache build, mempool
  growth, page-cache for the ELF. Always run ≥1 warmup prove in-process is impossible
  (one prove per process) — so instead measure cold run separately and report both;
  for steady-state numbers use runs 2+ of the same binary back-to-back.
- Watch `nvidia-smi -q -d PERFORMANCE` for thermal/power throttling during sessions
  (5090 at 575 W will throttle in poorly cooled boxes — that's machine noise).

### 2.4 Build profile for profiling
Add to root `Cargo.toml`:
```toml
[profile.profiling]
inherits = "release"
debug = true            # line tables for perf/ncu source correlation
strip = false
```
plus `RUSTFLAGS="-Cforce-frame-pointers=yes"` for cheap, reliable stack unwinding
(dwarf unwinding at 997 Hz on a 3000-thread rayon prover drops samples).
`build.rs` already passes `-O3` to nvcc; add `-lineinfo` (feature- or env-gated,
e.g. `LAMBDA_VM_NVCC_LINEINFO=1`) so ncu can map SASS→CUDA source. `-lineinfo` does
not change codegen, unlike `-G`(never use `-G` for perf work).

### 2.5 Long-term rig (what a persistent box unlocks)

- **Archived artifacts live on the machine**: `~/profiling-reports/<workload>_<sha>_<date>/`
  with the `.nsys-rep`/`.ncu-rep` + env JSON. Reports accumulate → historical diffs
  are always possible. Analyze GUIs locally: `scp` the `.nsys-rep` to a laptop and
  open in the Nsight Systems GUI (the reports are self-contained).
- **Nightly regression tracking** (once Layer 7 exists): a cron/systemd timer that
  runs the kernel microbench + one instruments prove of `ethrex_5_transfers` on
  latest `main`, appending `{date, sha, phase table, per-kernel GB/s}` to a CSV.
  Catches GPU perf regressions the day they merge instead of during the next
  profiling session.
- **Optional**: register it as a self-hosted `bench` runner later (the
  `scripts/SERVER_SETUP.md` flow) so `/bench-gpu` ABBA runs stop depending on
  rented Vast.ai boxes with drifting drivers. Decision for the team, not required
  by this plan.
- **Stable identity for noise numbers**: the Layer-1 noise floor (jitter CV%,
  cold-vs-warm delta) is measured once and stays valid — on rented boxes it had to
  be re-established every session.

---

## 3. Workload matrix

Fixed set, smallest-that-shows-the-behavior first:

| Workload | Why |
|---|---|
| `fib_iterative_372k.elf` | small, fast iteration; existing default in `bench_prove.sh` |
| `fib_iterative_4M.elf` | mid-size, one big table dominates |
| `ethrex.elf` + `ethrex_5_transfers.bin` | the real benchmark (matches `/bench-gpu` CI) |
| `ethrex.elf` + `ethrex_20_transfers.bin` + `--continuations` | large realistic trace; where GPU residency matters; per-epoch profiling |
| `keccak.elf`, `matrix_multiply.elf` (bench programs) | different table-shape mix |

Each measurement run = `cli prove <elf> --private-input <in> -o /tmp/p.bin --time`
built with `--features jemalloc-stats,prover/cuda,instruments` (+ `nvtx` once it
exists). Always assert the GPU counters fired (or grep the instruments output) so a
silent CPU fallback never contaminates a dataset.

---

## 4. Layer 1 — Macro baseline (no code changes) — *first session on the box*

1. **Phase table**: for each workload, 5 runs with `instruments` +
   `LAMBDA_VM_TIMELINE_JSON`, aggregate to a table: phase × {median ms, % of prove,
   σ}. Script: `scripts/profiling/phase_table.py` (parses the JSON tree).
2. **GPU utilization overlay**: sample `nvidia-smi dmon -s pucm -d 1` (or NVML at
   ~10 Hz via a tiny sidecar) during the prove; align with span `start_ns` (that's
   what the field is for). Output: per-phase average SM util, mem util, VRAM.
   This alone answers question 1 at coarse grain: *which phases leave the GPU idle*.
3. **Cold vs warm** delta, and run-to-run jitter (CV%) per workload — establishes
   the noise floor that any future optimization claim must beat (feeds ABBA sizing).
4. **Scaling curve**: fib series 372k → 16M: phase times vs trace size (log-log).
   Phases that scale worse than O(n log n) are suspect.

Deliverable: `docs/` or PR-comment-ready markdown: baseline table + util-per-phase
chart + scaling curves. This is the reference every later change is diffed against.

---

## 5. Layer 2 — NVTX instrumentation (small code change, big payoff)

The one code investment worth making before touching profilers seriously.

### 5.1 Design
- New cargo feature `nvtx` in `math-cuda` (and forwarded from `stark`/`prover`),
  using the `nvtx` crate (nvtx-rs, wraps nvToolsExt; header-only, no runtime cost
  when no profiler is attached).
- **Hook into the existing span system**: `instruments::span(label)` is already an
  RAII guard around every interesting host phase — make `SpanGuard` also push/pop an
  NVTX range when the `nvtx` feature is on. One edit point instruments the whole
  prover (`r1_main_commit`, per-table sub-ops, etc.) for Nsight timelines.
- Additionally, thin NVTX ranges at each **public `math-cuda` entry point** (~15
  functions across `lde.rs`, `merkle.rs`, `logup.rs`, `barycentric.rs`, `inverse.rs`,
  `deep.rs`, `fri.rs`, `constraint_interp.rs`) with payload info in the range name:
  table id, rows, cols, stream index — e.g. `lde_base[t=cpu rows=2^20 cols=48 s=7]`.
  These ranges are what let a timeline answer "which table / which size was this
  kernel burst".
- Mark categories/colors: R1=green, LogUp=blue, R2=orange, R3=purple, R4=red,
  transfers=gray. Trivial with NVTX domains, makes timelines readable at a glance.
- Optionally `cudaProfilerStart/Stop` (cudarc exposes it) fenced by env var, so nsys
  `--capture-range=cudaProfilerApi` can profile *one epoch* of a continuations run
  instead of a 30-minute trace.

### 5.2 Acceptance
- `cargo build --features cuda` (no nvtx) → zero new deps, zero overhead.
- ABBA-check `nvtx` build vs plain: expected ≈0% (ranges are ~25 ns each when
  unattached); document the measurement.

---

## 6. Layer 3 — Nsight Systems: the global timeline (the core of the analysis)

```bash
nsys profile \
  -t cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --gpu-metrics-devices=0 --gpu-metrics-frequency=10000 \
  -o reports/nsys_${workload}_$(git rev-parse --short HEAD) \
  ./cli prove <elf> --private-input <in> -o /tmp/p.bin --time
```
(`--gpu-metrics-devices` adds hardware SM-occupancy/util timeline; drop it if the
box refuses the permission. `osrt` traces pthread/sync syscalls → shows rayon
workers blocking. For ethrex-20tx use `--capture-range=cudaProfilerApi` around one
epoch to keep the report tractable.)

### 6.1 What to read off the timeline (checklist)
- [ ] **GPU idle gaps** inside each NVTX phase: measure GPU busy % per phase (see
      6.2 script). A phase with 40% busy = host-bound, don't optimize its kernels.
- [ ] **Stream lanes**: do the 32 pool streams actually run concurrently during
      multi-table R1, or do twiddle-cache locks / mempool contention / the
      `util_stream` serialize them? Look for staircase patterns.
- [ ] **Sync stalls**: correlate `cuStreamSynchronize` API calls (CUDA API row) with
      GPU idle. Each `lde.rs` entry-point sync that's followed by host work then a
      new launch = pipeline bubble; the `_keep`/`_dev` no-sync variants should show
      back-to-back kernels.
- [ ] **Memcpy rows**: H2D/D2H volume, duration, pageable-vs-pinned (nsys labels
      them), and whether they overlap kernels (they should, given pinned staging +
      streams) or sit in gaps. Quantify the trace-upload and any GPU→host→GPU
      round-trips (the known ~250 ms one between execution/trace-build and R1).
- [ ] **Launch overhead**: kernels < ~10 µs launched in tight host loops (suspect:
      `keccak_merkle_level` per tree level, `ntt_dit_level` per stage,
      scan-phase kernels) — count launches/s; if API launch time ≈ kernel time,
      that's a fusion/graph candidate (CUDA graphs, or the existing
      `ntt_dit_8_levels` fusion applied more widely).
- [ ] **Mempool behavior**: `cuMemAllocAsync` that actually reserves (slow) vs pool
      hits (fast) — visible as API call duration spikes; check
      `--cuda-memory-usage` VRAM curve for growth mid-prove.
- [ ] **rayon threads**: with `osrt`, see whether `TABLE_PARALLELISM` workers are
      computing, blocked on GPU, or blocked on each other.

### 6.2 Quantitative extraction (don't eyeball — script it)
```bash
nsys stats -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_sum report.nsys-rep
nsys export --type sqlite report.nsys-rep
```
Write `scripts/profiling/nsys_phase_busy.py`: joins `CUPTI_ACTIVITY_KIND_KERNEL` /
`_MEMCPY` intervals against NVTX ranges in the sqlite export → per-phase table:
`{phase, wall_ms, gpu_busy_ms, busy_%, memcpy_ms, top-5 kernels}`. This table is
**the** deliverable of Layer 3 and directly ranks where optimization effort goes:
- busy % low → fix host pipeline / overlap / syncs (cheap wins first);
- busy % high → go to Layer 5 for those kernels.

---

## 7. Layer 4 — CPU-side flamegraphs (host work + wait analysis)

Run on the same workloads (GPU build; also one CPU-only build for contrast):

1. **On-CPU flamegraph**:
   ```bash
   perf record -F 997 -g --all-cpus -- ./cli prove ...      # fp stacks via RUSTFLAGS
   perf script | inferno-collapse-perf | inferno-flamegraph > oncpu.svg
   ```
   (or `cargo flamegraph -p cli --profile profiling --features ... -- prove ...`;
   `samply record ./cli prove ...` for an interactive Firefox-profiler view — the
   repo already ships `profile_vm_prover.rs` for exactly this.)
   Look for: serialization/rkyv, transposes/layout shuffles feeding H2D, Fiat-Shamir
   hashing, `memcpy` into pinned staging, allocator time, and anything single-threaded
   inside a phase whose GPU busy % was low.
2. **Off-CPU flamegraph** (where threads *wait* — invisible on-CPU):
   ```bash
   sudo /usr/share/bcc/tools/offcputime -df -p <pid> 30 | inferno-flamegraph --colors io > offcpu.svg
   ```
   Waiting in `cuStreamSynchronize` / futexes tells you which host thread is
   gating which phase. On-CPU + off-CPU together fully account for wall-clock.
3. Diff flamegraphs across workload sizes (inferno `--diff`) to see which host costs
   grow with trace size vs fixed overheads.

---

## 8. Layer 5 — Nsight Compute: per-kernel deep dives

Only for kernels that Layer 3 proved dominant **and** whose phase is GPU-bound.
ncu replays kernels — total-time numbers under ncu are meaningless; it's for *why is
this kernel slow*, never *how long is this kernel*.

```bash
sudo ncu --set full \
  -k 'regex:ntt_dit|keccak256_leaves|keccak_merkle_level|constraint_interp|logup_|batch_inverse|fri_fold' \
  --launch-skip 20 --launch-count 10 \
  -o reports/ncu_${workload} ./cli prove ...        # or better: the microbench (Layer 7)
```

Per-kernel-family checklist:
- **NTT (`ntt_dit_level`, `ntt_dit_8_levels`, row-major variants, transpose)** —
  expected memory-bound: DRAM throughput vs ~1.79 TB/s peak; L2 hit rate;
  shared-memory bank conflicts in the 8-level fused kernel; coalescing on strided
  row-major access (`matrix_transpose_strided` sector/byte efficiency); occupancy
  limiter (registers? shared mem?).
- **Keccak leaves/levels** — compute-heavier: register pressure (Keccak state = 25×
  u64), occupancy, ILP/pipe utilization (ALU vs LSU), whether `keccak_merkle_level`
  at small upper tree levels underfills the GPU (launch geometry per level).
- **`constraint_interp_kernel`** — the interpreter: warp divergence (branch
  efficiency), instruction-fetch stalls on the opcode dispatch, whether the
  `DeviceProgram` bytecode reads hit L1/constant cache, spills.
- **Scans (batch inverse, logup)** — multi-pass Hillis-Steele: total DRAM traffic vs
  the theoretical minimum bytes (measure "achieved bytes / useful bytes" ratio);
  candidates for single-pass decoupled-lookback later.
- **`fri_fold_ext3`, `deep_composition_ext3_row`, barycentric** — usually small;
  check they're not latency-bound at small sizes late in FRI (tiny grids).

Record for each: roofline position, top stall reason, occupancy + limiter,
achieved vs peak bandwidth. That fills the "estimated headroom" column of the final
bottleneck ranking.

---

## 9. Layer 6 — Memory profiling

1. **VRAM**: `--cuda-memory-usage` curve (Layer 3) + NVML sampler from Layer 1 →
   peak VRAM per workload/epoch; compare against `vram_budget_bytes()` admission
   decisions (`prover.rs:2403`) — is the budget leaving GPU capacity unused
   (tables needlessly falling to CPU) or overcommitting (mempool thrash)?
2. **Mempool efficacy**: sqlite export → histogram of `cuMemAllocAsync` durations;
   sweep `LAMBDA_VM_MEMPOOL_RELEASE_MB` to confirm the retain-all threshold wins.
3. **Host heap**: existing jemalloc peak report per phase; `heaptrack` on one run if
   allocation churn shows up in flamegraphs (temporary buffers feeding H2D are a
   classic).
4. **Pinned staging**: is `PinnedStaging` per-rayon-worker sized right? nsys shows
   pageable copies wherever it's bypassed/overflowed.

---

## 10. Layer 7 — Kernel microbenchmarks (CUDA events)

A small `math-cuda` bench binary (extend `make bench-math-cuda`) that runs each
kernel family standalone at the *real shapes* observed in Layer 3 (rows/cols per
table for each workload — dump shapes from the NVTX payloads), timed with CUDA
events (`cuEventElapsedTime`), 100 reps after warmup:

- Gives clean per-kernel numbers *without* profiler overhead → tracks regressions
  cheaply in CI-like fashion.
- Prints achieved GB/s and % of peak for memory-bound kernels — the number to move.
- Becomes the harness for iterating on kernel optimizations later (edit kernel →
  re-run microbench → confirm on full prove with ABBA).

---

## 11. Layer 8 — A/B experiments with existing toggles (cheap, high signal)

Using the phase table (Layer 1) as the metric, one variable at a time, ≥5 runs each:

| Experiment | Knob | Question |
|---|---|---|
| GPU phase contribution | `LAMBDA_VM_NO_GPU_LOGUP`, `LAMBDA_VM_DISABLE_GPU_COMPOSITION`, `LAMBDA_VM_DISABLE_DEVICE_ONLY` | actual end-to-end value of each GPU phase; interaction effects (does disabling one starve another of residency?) |
| Table parallelism | `TABLE_PARALLELISM` ∈ {1,2,4,8,…} | does multi-table overlap saturate the GPU or thrash streams/mempool? Find the knee. |
| Thresholds | `LAMBDA_VM_GPU_LDE_THRESHOLD`, `_BARY_THRESHOLD` sweep | are small tables better on CPU? Where's the real crossover on a 5090? |
| Continuations | `EPOCH_SIZE_LOG2` ∈ {18,20,22} on ethrex-20tx | epoch size vs GPU efficiency (bigger epochs = better GPU utilization vs VRAM/host peaks) |
| VRAM budget | `LAMBDA_VM_VRAM_BUDGET_MB` sweep | headroom of the admission controller on 32 GB |

---

## 12. Deliverables & report format

All under `reports/` (gitignored) + a committed summary doc:

1. **Baseline doc** (`thoughts/gpu-profiling/baseline-<sha>.md`):
   - phase × workload table (median, %, σ, GPU busy %, memcpy ms);
   - top-15 kernels by total GPU time (name, count, total ms, avg µs, % of GPU time);
   - transfer summary (volume, pinned %, overlap %);
   - flamegraph SVGs (on-CPU, off-CPU) linked;
   - scaling curves.
2. **Bottleneck ranking**: for each candidate, `{observed cost, mechanism (evidence:
   nsys/ncu screenshot or stat), estimated ceiling if fixed (Amdahl on the phase
   table), fix difficulty}` — sorted by (ceiling / difficulty). This is the input to
   the optimization roadmap.
3. **Tooling committed to the repo** (`scripts/profiling/`): `setup_machine.sh`,
   `capture_env.sh`, `run_profile.sh` (one command → instruments JSON + nsys report
   + stats + phase-busy table for a given workload), `phase_table.py`,
   `nsys_phase_busy.py`, `timeline_to_perfetto.py` (convert
   `LAMBDA_VM_TIMELINE_JSON` to Chrome-trace JSON so span trees are browsable in
   Perfetto next to nsys). Plus the NVTX patch (Layer 2) and the microbench
   (Layer 7).
4. **Archived artifacts**: `.nsys-rep`/`.ncu-rep` + env JSON per session, named
   `<workload>_<sha>_<date>` so future comparisons are possible.

---

## 13. Execution order & rough effort

| Step | Depends on | Effort |
|---|---|---|
| 1. `setup_machine.sh` + sanity (`make test-cuda-integration`) | box | ~1–2 h |
| 2. Layer 1 macro baseline + noise floor | 1 | ~half day |
| 3. NVTX patch (Layer 2) + overhead check | — (parallel with 2) | ~half day |
| 4. Layer 3 nsys timelines + `nsys_phase_busy.py` table | 2,3 | ~1 day |
| 5. Layer 4 flamegraphs (on/off-CPU) | 2 | ~half day |
| 6. Decision point: rank phases (host-bound vs GPU-bound) | 4,5 | — |
| 7. Layer 5 ncu on dominant kernels; Layer 6 memory; Layer 7 microbench | 6 | ~1–2 days |
| 8. Layer 8 knob sweeps | 2 | background (scriptable, runs alone) |
| 9. Baseline doc + bottleneck ranking | all | ~half day |

Total ≈ 4–5 focused days on the box; steps 3/5/8 parallelize.

---

## 14. Known gotchas (write them down before they bite)

- **Blackwell tooling floor**: nsys/ncu must be ≥ 2025.1 and CUDA ≥ 12.8; distro
  packages are too old — install from the NVIDIA apt repo. The repo already
  documents the cubin/PTX-JIT rationale (`README.md` §GPU tests,
  `math-cuda/build.rs:96-127`).
- **Driver updates change baselines**: on an owned box the driver only changes when
  *we* change it — pin it, and treat any driver/CUDA upgrade as a baseline reset
  (re-run Layer 1, new `baseline-<sha>.md`). Never compare numbers across driver
  versions.
- **ncu replay distortion**: never quote durations from ncu runs; kernel *time*
  comes from nsys or CUDA events only.
- **One prove per process**: no in-process warmup — treat run #1 (cold: module
  load, twiddles, mempool growth) as a separate "cold" datapoint.
- **Thermals**: 5090 will throttle in bad chassis; log clocks during sessions and
  lock them (`-lgc`); a throttled session's numbers are not comparable.
- **`instruments` overhead**: verify ≈0 with ABBA once; likewise the `nvtx` build.
- **GeForce clock locking**: `-lgc` works, `-lmc` may not — record actual clocks in
  `capture_env.sh` regardless.
- **Silent CPU fallback**: `math-cuda` falls back to CPU on any load/launch failure
  by design — every dataset must include the `gpu_*_calls()` counters (or the
  integration test as a pre-flight) to prove the GPU path actually ran.
