# scripts/profile: Nsight profiling of the WHIR block run

Scripts for a GPU box whose Nsight performance counters are unlocked. Everything they
profile is already on this branch (`whir/profile-rpx` = the WHIR pipeline's best
configuration, draft PR #1004, plus these scripts and the block fixtures). They add
nothing to the prover.

| script | what it answers | needs counters | time |
|---|---|---|---|
| `block_profile.sh` | **Run A** (Nsight Systems): where the block run's wall goes on the card, as GPU busy vs idle per stage (base, with its commit/prove/global; level 0; interior; root), per 5 s and per kernel. With counters, also SM active %, SM issue % and DRAM bandwidth % per 1 s and per stage. **Run B** (Nsight Compute): why the kernels that carry the time run as fast as they do, on the launch shapes the trace analysis picked (`thoughts/zf/prof/ANALYSIS-1GPU.md` §7), with the grind paired with its counted twin. Also `cuobjdump -res-usage` of every cubin. | run B and the GPU metrics only | ~1.5-2 h fresh |
| `whir_ncu.sh` | Nsight Compute on two micro-benches: the RPX permutation plus the grind (latency-bound or issue-bound?) and one block-scale WHIR sumcheck round (is the round kernel under-occupied?). | yes | ~10-40 min (less once the prover is built) |

The run is the record's, not an approximation of it. The test is
`lfm::per_table_aggregator_tests::the_whir_production_tree_composes_to_a_root`, with exactly
the record launcher's environment (`A-tree-whir.v10.sh` as `whir_tree17.sh` ran it). It is
built with `cargo test --release -p lambda-vm-prover --features cuda,nvtx --lib` and
`LAMBDA_VM_NVCC_LINEINFO=1`, as the trace analysis asks. That is the record's binary plus:
- NVTX ranges, which name the phases in the trace. The feature also turns on the
  `instruments` spans, which is host-side bookkeeping.
- SASS-to-source line tables in the cubins. The code itself is unchanged. The profiled
process gets only that environment plus a few system variables (`env -i`), so nothing else
from your shell reaches the run. The inputs are the record's guest ELF and block, in
`fixtures/`, verified by sha256.

## Secrets: what may leave the box

**`OUT/big` never leaves the box.** Its `.nsys-rep`, `.ncu-rep` and `.sqlite` files embed the
machine's environment: nsys and ncu store the process environment verbatim. A trace of
this run on our rented box carried that box's Jupyter token, container API key and a base64
SSH private key.
Never commit `OUT/big`, never share it, never upload it. Running under `env -i` narrows what
gets recorded, but that does not change the rule.

**Only `OUT/small` leaves the box, and only after the final bundle scan passes.** The scan
refuses the bundle in three cases:
- any file that is not plain text;
- any `.nsys-rep`, `.ncu-rep`, `.sqlite` or other Nsight capture, whatever its name;
- any line matching a credential marker: `TOKEN`, `KEY`, `SECRET`, `PASSWORD`,
  `BEGIN ...PRIVATE`, `ssh-`, `ghp_`, `github_pat_`, a base64-encoded PEM header, or an AWS
  key id.

On a hit the script writes no tarball, says there is nothing to send, and exits 4. The hits
(file, line, marker) are listed in `OUT/big/bundle_scan.txt`, which is not to be shared
either. Read those lines and remove them, then re-scan; the tarball is written only if the
scan passes:

```bash
bash -c '. scripts/profile/lib/common.sh && pf_scan_bundle OUT/small OUT/big/bundle_scan.txt && tar -czf OUT/small.tar.gz -C OUT small'
```

`OUT/small/env.txt` is an allowlist of run facts, never the environment:
- driver, CUDA, nvcc, nsys and ncu versions;
- the GPU's name, clocks and memory;
- the repo sha and the test binary's sha256;
- the knob variables the run sets (`LAMBDA_VM_*`, `LFM_*`, `TABLE_PARALLELISM`,
  `_RJEM_MALLOC_CONF`), with the checkout's path written as `<repo>`.

## For Mauro: the commands

```bash
# 1. the branch (an existing clone is fine)
git fetch origin whir/profile-rpx && git checkout whir/profile-rpx     # or: git clone -b whir/profile-rpx <repo url>
cd "$(git rev-parse --show-toplevel)"

# 2. preflight (~1 min): tools, GPU, counters proven on one real kernel, RAM, disk, toolchains,
#    fixtures. It provisions the guest sysroot when it is missing. Fix any FAIL line it prints.
bash scripts/profile/block_profile.sh --preflight-only

# 3. the block profile: build + run A (nsys, with GPU metrics) + run B (ncu, twelve passes)
bash scripts/profile/block_profile.sh

# 4. optional, afterwards: the two micro-benches
bash scripts/profile/whir_ncu.sh
```

Every step has a timeout, and `block_profile.sh` prints its own estimate before it starts.
Run it inside `tmux` or `screen`: a dropped ssh session would otherwise end it.

Useful options: `--gpu N` (another GPU); `--skip-build` (reuse this checkout's last build,
same HEAD); `--skip-nsys` / `--skip-ncu` (one run only, for example to redo run B:
`--skip-build --skip-nsys`); `--no-cpu-sampling`; `--out DIR`. `--help` lists everything,
including the env knobs (`NCU_PLAN`, `NCU_PASSES`, `NCU_METRICS`, `GPU_METRICS_FREQ`, the timeouts).

## Expected runtime (block_profile.sh)

| step | fresh checkout | rebuilt / `--skip-build` |
|---|---|---|
| preflight | ~1 min (+ sysroot download the first time) | ~1 min |
| guest ELFs (`make compile-programs-asm compile-programs-rust compile-recursion-elfs`) | ~15-25 min | ~3 min / 0 |
| prover build (release, CUDA) | ~15-25 min | a few min / 0 |
| run A: the block run under nsys (~2-4 min), export, stats, summary | ~10-15 min | same |
| run B: twelve ncu passes | ~40-60 min | same |

Run B is the uncertain one. Each pass reruns the block until its launches are reached, then
ncu ends it:
- seven passes aim at the base, 2-15 s into the run;
- four aim at the LFM wraps, which only start after the ~70 s base;
- `grind_pair` runs its own short test.

ncu replays each profiled launch once per metric pass (tens of passes for these sections),
saving and restoring the device memory the kernel can reach. The estimate the script prints
uses, per pass: 60 s + 2.5 × (the window's time into the run) + 20 s per launch.

## What comes back

`block_profile.sh` writes `scripts/profile/out/block-<UTC>/` (git-ignored):

- `small/`: a few MB of plain text, packed as `small.tar.gz` once the bundle scan passes.
  `small/INDEX.txt` says what each file is. Read `small/nsys/summary.txt` first (stages,
  busy/idle per 5 s, top kernels, copies, GPU metrics, NVTX), then
  `small/ncu/verify.tsv` (did each pass profile the shapes it was aimed at),
  `small/ncu/ncu_summary.txt` and `small/ncu/<pass>.details.txt`.
- `big/`: `blockA.nsys-rep`, `blockA.sqlite`, `ncu/*.ncu-rep`, the raw exports and the full
  build logs. It embeds the machine's environment: never commit, share or upload it.

The last lines the script prints say exactly what to send back. Once the scan passes, either
commit `small/`:

```bash
mkdir -p scripts/profile/results/<YYYYMMDD>-<gpu>
cp -R scripts/profile/out/block-<UTC>/small/. scripts/profile/results/<YYYYMMDD>-<gpu>/
git add scripts/profile/results/<YYYYMMDD>-<gpu>
git commit -m 'profile: WHIR block run on <gpu> (<date>)'
git push origin HEAD:whir/profile-rpx
```

or send `scripts/profile/out/block-<UTC>/small.tar.gz`. `whir_ncu.sh` writes the same way:
`OUT/small` (text, `small.tar.gz` once its scan passes) and `OUT/big` (its `.ncu-rep`,
`.nsys-rep` and `.sqlite`, which stay on the box).

## How to read the results

- **Stages** come from the run's own log, not from NVTX, so they work with or without
  `libnvToolsExt`. When NVTX ranges are present, `small/nsys/phase_busy.md` adds the
  in-repo per-NVTX-phase view (`scripts/profiling/nsys_phase_busy.py`). The harness prints
  Unix-time stamps (`BASE EPOCH n: stage Xs t=[a,b]`, `CARD HOLD ...`,
  `PROVE SPLIT ...`, `MARK AFTER ...`), and nsys stores the session's UTC start. The summary
  prints how far the log's last stamp sits from the last GPU activity: on the lead's
  trace it was 0.000 s. Phases partition the run: `pre`, `base`, `level0` (which
  includes the WHIR global child, run as task 0 of level 0's pool), `interior`, `root`,
  `post`. Stages are unions of the harness's intervals: `base/commit`, `base/prove`,
  `card/multi_prove` and so on.
- **busy** is the union of kernels, copies and memsets, so concurrent streams never count
  twice. `kern_sum` sums kernel durations and can exceed the wall.
- Timings under nsys or ncu are not block times. Take the walls from the untraced record.
  `runA.verdict.txt` re-reads the record launcher's gates (1 test passed, the COMPRESSED
  line, commit/device fallbacks 0, the grind banner, the base closure, the retention).
  SUSPECT there means the traced run was not the record's pipeline.
- ncu locks clocks at base, flushes caches between replay passes and replays each launch,
  so its durations are not wall times either. Read it for ratios: speed of light, issue
  %, occupancy against its limiter, stall reasons.

### Run B's plan

ncu has no grid-size filter: it selects a kernel's launches by their own index (`-k NAME -s
SKIP -c COUNT`). So each pass profiles a window of launches, chosen in the lead's nsys trace
of this code (RTX 5090) for the shapes the trace analysis asks for. The plan carries the exact
grid/block sequence each window holds, and two safeguards keep the index honest on another
box:
- Before run B, each window is re-anchored on this box's own run A trace
  (`lib/ncu_plan.py anchor`): the nearest index with exactly those shapes.
- After each pass, `verify` compares what ncu profiled with the plan and writes
  `small/ncu/verify.tsv`: `ok`, `partial` or `MISMATCH`.

Ordinals do not survive the tree stages, where sibling proofs launch concurrently: on 09-25
`merkle_wide`, re-anchored to launch 12044, profiled grids 32 and 16. A shape that lives there
is therefore a SELECT pass, whose shapes field is a predicate (`select:gridx>=8192,min=2`):
- ncu runs it with `--filter-mode per-launch-config`, so `-s`/`-c` apply to each launch
  configuration separately and every configuration of the kernel gets its launches, the wide
  ones included. There is no `--kill`, so the block runs to its end.
- The pass's exports are then filtered to the launches with gridDim.x ≥ N (the unfiltered ones
  stay in `big/ncu/<pass>.*.all`).
- `verify` passes it only with at least M of them. Otherwise run B logs `ERROR: SELECT PASS
  MISSED ITS SHAPES` and the run's verdict is FAIL.
- The preflight profiles the probe kernel with the select flags, so an ncu without
  per-launch-config fails there, not an hour in.

One pass alone, without run A: `NCU_PASSES=merkle_wide bash scripts/profile/block_profile.sh --skip-nsys`.

| pass | kernel | launches (shape in the trace) | what it settles |
|---|---|---|---|
| grind | rpx_grind_search | 5 from the base (1024 × 128, 64 regs) | why 4.46 ns per permutation against 2.77 in the leaf kernel |
| grind_pair | rpx_grind_search + rpx_grind_search_counted | the `rpx_grind_counted` test: warm-up and seed 0 at poll period 1, shipped then counted | the same, with a counted twin; the warm-up line prints the nonce (executed ≈ nonce + 131,072) |
| coset | rpx_leaves_base_coset | 2 at 16384 × 128 | the permutation's full-card ceiling |
| merkle_narrow | rpx_merkle_level | one tree's levels 512 → 2 (incl. grids 512, 32, 2) | latency |
| merkle_wide | rpx_merkle_level | SELECT: 2 launches of each configuration, kept if gridDim.x ≥ 8192 (base coset trees' 8192, the wraps' 16384) | throughput |
| tail | rpx_merkle_tail | 3 at 1 × 128 | per-level latency; barrier and local-memory stalls |
| sumcheck_2p21 | sumcheck_round_ext3 | one 2^21 sumcheck's rounds from 4096 × 256 (98 regs) down, then the two 1181 × 256 | slot-buffer traffic against capped occupancy |
| sumcheck_slow | sumcheck_round_ext3 | the slow 253 × 32 launches (162, 94, 59 ms) | the same, where it costs most |
| ntt_tile | ntt_dit_tile | one base LDE's four tile launches | base LDE throughput |
| rowpair_range | rpx_leaves_base_row_major_row_pair_range | 2 at 8192 × 128 | coalescing of row-major reads |
| ntt_level | ntt_dit_level_row_major | 3 (1 × 22796 and 1 × 65535 grids, 11 × 23 blocks) | memory throughput against peak |
| constraint | constraint_composition_kernel | 2 at 256 × 256 | occupancy |

The analysis asked for two 16384-wide launches. `merkle_wide` takes the 16384 level and
the 8192 level of the same tree instead, because the next 16384 launch comes about 90
launches later and a second window would cost another block run.

Every pass collects the eight sections plus the analysis's explicit metrics (`lib/common.sh`
`PF_NCU_METRICS`):
- `sm__throughput`, `gpu__compute_memory_throughput`, `dram__throughput`, `gpu__time_duration`;
- achieved occupancy, its register limit, registers per thread;
- issue activity, `smsp__inst_executed.sum` and `smsp__thread_inst_executed.sum`;
- the pipe table;
- the stall reasons (long/short scoreboard, wait, math-pipe throttle, LG throttle, not
  selected, barrier);
- local-memory sectors, global sectors per request (coalescing), the L2 hit rate.

The preflight keeps only the metrics this ncu and GPU can collect, because one unknown name
makes ncu profile nothing.

`NCU_PLAN=file.tsv` replaces the plan, and `NCU_PASSES="grind merkle_wide"` runs a subset.
To redo only run B: `--skip-build --skip-nsys`. Without run A in the same output directory,
nothing is re-anchored; the shape check still runs.

## Troubleshooting

- **`counters: COUNTERS locked (ERR_NVGPUCTRPERM)`**: as root, run
  `echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' > /etc/modprobe.d/nvidia-profiling.conf`,
  then `update-initramfs -u` and reboot. Or run the scripts as root. Without counters,
  `--no-counters` still gives run A without GPU metrics.
- **`nvcc: not at .../bin/nvcc`**: math-cuda's `build.rs` looks for nvcc only at
  `$CUDA_HOME/bin/nvcc` (else `$CUDA_PATH`, else `/usr/local/cuda`). Without it the build
  does not fail: it writes empty cubins and every kernel runs on the CPU. Set `CUDA_HOME`.
  The scripts refuse empty cubins after the build either way.
- **`ncu (Nsight Compute) not found` although `ncu` exists**: an `ncu` on PATH can be
  npm-check-updates. The scripts check every `ncu` on PATH by its banner, then
  `$CUDA_HOME/bin` and `/opt/nvidia/nsight-compute/*`, or `NCU=/path/to/ncu`. The same
  applies to `NSYS`.
- **CPU sampling off** (`--sample=none` in the preflight): nsys CPU sampling is optional
  and never fails a run. It is used where `nsys status --environment` offers it and the
  preflight probe succeeds with it. When a box refuses it (FAST's container does), the
  probe, and if need be run A itself, retry once without it. Run A still traces CUDA and the
  OS runtime. `--no-cpu-sampling` turns it off from the start.
- **`BLOCK_PROFILE VERDICT: REFUSED (bundle scan)`**: something in `small/` looks like a
  credential or is not plain text. See "Secrets" above. Nothing may be sent until the flagged
  lines are gone and the scan passes.
- **run B ends `partial`**: see `small/ncu/passes.tsv`, `small/ncu/verify.tsv` and
  `small/ncu/<pass>.prof.txt`.
  - A pass with no profile usually ran out of host memory in ncu's replay (the preflight
    warns below 64 GiB), or hit its timeout (`NCU_PASS_TIMEOUT`, default 2700 s).
  - A `MISMATCH` means the window drifted on this box and caught other shapes. The
    `.details.txt` still holds real launches. To aim again: copy `small/ncu/plan.tsv`, fix
    that pass's `skip`, then run
    `NCU_PLAN=<copy> NCU_PASSES="<pass>" bash scripts/profile/block_profile.sh --skip-build --skip-nsys`.
- **`nvtx: no libnvToolsExt`**: CUDA toolkits since 12.9 ship no `libnvToolsExt`, and the
  `nvtx` build then emits no ranges. Nothing fails; the trace just lacks named phases. Install
  `cuda-nvtx-12-8`, or point `LAMBDA_VM_NVTX_LIB` at a `libnvToolsExt.so.1`.
- **a GPU other than a 32 GB card**: the device layer sizes itself from the driver, so
  the schedule and any fallback differ from the record. The preflight warns, and
  `runA.verdict.txt` reports the fallbacks.

## For the lead: the dry run on a counter-locked box

```bash
bash scripts/profile/block_profile.sh --no-counters
```

This runs the preflight (the counter probe reports `locked`, as INFO), the build and run A
without GPU metrics, and skips run B. The last lines read
`RESULT: run A ok · run B skipped · bundle clean` and `BLOCK_PROFILE VERDICT: PASS`.

`lib/` holds the shared pieces: `common.sh` (tool discovery, probes, the clean environment,
the bundle scan),
`nsys_block_summary.py` and `ncu_summary.py` (post-processing, stdlib only, each with
`--selftest`), and `cargo_test_bin.py` (test binaries by cargo's own JSON).
