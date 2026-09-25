# scripts/profile: Nsight profiling of the WHIR block run

Scripts for a GPU box whose Nsight performance counters are unlocked. Everything they
profile is already on this branch (`whir/profile-rpx` = the WHIR pipeline's best
configuration, draft PR #1004, plus these scripts and the block fixtures). They add
nothing to the prover.

| script | what it answers | needs counters | time |
|---|---|---|---|
| `block_profile.sh` | **Run A** (Nsight Systems): where the block run's wall goes on the card, as GPU busy vs idle per stage (base, with its commit/prove/global; level 0; interior; root), per 5 s and per kernel. With counters, also SM active %, SM issue % and DRAM bandwidth % per 1 s and per stage. **Run B** (Nsight Compute): why the nine kernels that carry the time run as fast as they do, on real launches of that run. | run B and the GPU metrics only | ~1.5-2 h fresh |
| `whir_ncu.sh` | Nsight Compute on two micro-benches: the RPX permutation plus the grind (latency-bound or issue-bound?) and one block-scale WHIR sumcheck round (is the round kernel under-occupied?). | yes | ~10-40 min (less once the prover is built) |

The run is the record's, not an approximation of it. The test is
`lfm::per_table_aggregator_tests::the_whir_production_tree_composes_to_a_root`, from
`cargo test --release -p lambda-vm-prover --features cuda --lib`, with exactly the record
launcher's environment (`A-tree-whir.v10.sh` as `whir_tree17.sh` ran it). The profiled
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

# 3. the block profile: build + run A (nsys, with GPU metrics) + run B (ncu, nine kernels)
bash scripts/profile/block_profile.sh

# 4. optional, afterwards: the two micro-benches
bash scripts/profile/whir_ncu.sh
```

Every step has a timeout, and `block_profile.sh` prints its own estimate before it starts.
Run it inside `tmux` or `screen`: a dropped ssh session would otherwise end it.

Useful options: `--gpu N` (another GPU); `--skip-build` (reuse this checkout's last build,
same HEAD); `--skip-nsys` / `--skip-ncu` (one run only, for example to redo run B:
`--skip-build --skip-nsys`); `--no-cpu-sampling`; `--out DIR`. `--help` lists everything,
including the env knobs (`NCU_KERNELS`, `NCU_COUNT`, `GPU_METRICS_FREQ`, the timeouts).

## Expected runtime (block_profile.sh)

| step | fresh checkout | rebuilt / `--skip-build` |
|---|---|---|
| preflight | ~1 min (+ sysroot download the first time) | ~1 min |
| guest ELFs (`make compile-programs-asm compile-programs-rust compile-recursion-elfs`) | ~15-25 min | ~3 min / 0 |
| prover build (release, CUDA) | ~15-25 min | a few min / 0 |
| run A: the block run under nsys (~2-4 min), export, stats, summary | ~10-15 min | same |
| run B: nine ncu passes | ~40-60 min | same |

Run B is the uncertain one. Each pass reruns the block until its launches are reached: the
first five kernels are seconds into the base, the last four only run after the ~70 s base.
ncu then replays each profiled launch once per metric pass (tens of passes for these
sections), saving and restoring the device memory the kernel can reach. The estimate the script prints uses
60 s + 2.5 x (the launch's time into the run) + 45 s per launch, per pass.

## What comes back

`block_profile.sh` writes `scripts/profile/out/block-<UTC>/` (git-ignored):

- `small/`: a few MB of plain text, packed as `small.tar.gz` once the bundle scan passes.
  `small/INDEX.txt` says what each file is. Read `small/nsys/summary.txt` first (stages,
  busy/idle per 5 s, top kernels, copies, GPU metrics), then `small/ncu/ncu_summary.txt`
  and `small/ncu/<kernel>.details.txt`.
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

- **Stages** come from the run's own log, not from NVTX (the record build has none). The
  harness prints Unix-time stamps (`BASE EPOCH n: stage Xs t=[a,b]`, `CARD HOLD ...`,
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

Run B's default kernels (`NCU_KERNELS` overrides; `skip:count` index that kernel's own
launches), picked in the lead's nsys trace of this code on an RTX 5090 to land on the
launches that carry each kernel's time:

| kernel | skip:count | shape in that trace |
|---|---|---|
| rpx_grind_search | 16:3 | grid 1024 x 128, 7.7-11.7 ms (all launches share it) |
| rpx_merkle_level | 11:3 | grids 8192, 4096, 2048: the top of one tree |
| rpx_merkle_tail | 1:2 | one block of 128 threads, ~0.9 ms, 4692 launches in the run |
| rpx_leaves_base_coset | 1:3 | grid 16384, ~45 ms |
| sumcheck_round_ext3 | 340:3 | the three largest rounds of one 2^21 sumcheck (5.9, 3.1, 1.6 ms) |
| rpx_leaves_base_row_major_row_pair | 0:3 | grids 8192, 8192, 16384 |
| ntt_dit_level_row_major | 59:3 | grid 1 x 22796 / 65535 with an 11-thread block |
| rpx_leaves_base_row_major_row_pair_range | 1:4 | grids 16384, 8192, 16384, 4096 (up to 121 ms) |
| constraint_composition_kernel | 0:3 | grid 256 x 256 (30.5, 4.7, 6.0 ms) |

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
- **run B ends `partial`**: `small/ncu/passes.tsv` and `small/ncu/<kernel>.prof.txt` say
  which kernel got no profile. The usual causes are running out of host memory in ncu's
  replay (the preflight warns below 64 GiB) or a pass timeout (`NCU_PASS_TIMEOUT`,
  default 2700 s). Redo that kernel alone:
  `NCU_KERNELS="<name>:<skip>:<count>" bash scripts/profile/block_profile.sh --skip-build --skip-nsys`.
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
