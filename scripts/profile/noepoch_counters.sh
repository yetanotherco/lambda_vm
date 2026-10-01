#!/usr/bin/env bash
# noepoch_counters.sh: GPU counters on the no-epoch WHIR base and the no-epoch STARK recursion, for a
# machine whose counters are open (lane I-PROF2, 2026-10-01). Mauro runs it himself; nobody logs in.
#
# WHAT IT ANSWERS (block 25368371)
#   Q1  The no-epoch WHIR base (PR #1014, ~18 s of a ~22.3 s block): per phase (phase A: the streamed
#       build and commits; phase B: the argue and the openings) where the card's time goes: kernel time
#       by kernel family, card busy against idle, and whether the kernels that carry it sit at a hardware
#       roof (SM throughput, DRAM bandwidth, achieved occupancy, waves) in the block's larger tables.
#   Q2  The no-epoch STARK recursion (PR #1013, ~10.3 s of a ~31.8 s block, card-bound at every level):
#       each LFM proof's device phase (multi_prove ~0.45 s, an artifact build ~0.16 s, x15). Is the card
#       busy inside each hold, and do those kernels sit at a roof, or is there room for less card per
#       LFM proof?
#   Q3  For free: the profiled process's CPU (cores busy per stage, per thread role) on this machine's
#       CPU, from a /proc sampler beside the same runs.
#
# MODES  (one prover at a time; each step bounded by `timeout`; a deadline skips what would overrun)
#   --quick (the default)  ~15 min after the build, peak host ~46 GiB. Run A (Nsight Systems with GPU metrics
#       and the CPU sampler) on both workloads, then a few Nsight Compute passes on the top kernels: the
#       WHIR phase-B and phase-A leaders and the LFM multi_prove leaders. Answers Q1's busy/idle, families
#       and GPU-metric means per phase, Q2's per-hold busy and families, Q3, and a first roof reading.
#   --full  ~45-60 min after the build, same peak. Adds a reference run of each workload without a
#       profiler (its walls), and every Nsight Compute pass of the plan (one launch of each shape for
#       the kernels that carry each phase and each hold). Answers Q1 and Q2's roofs kernel by kernel.
#   The build is a few minutes in a fresh work directory (two checkouts, one target directory) and is
#   reused by a re-run.
#
# WHAT IT RUNS  (the structure of lane I-PROF's noepoch_counters.sh, 09-30, and mauro-ncu.sh, 09-28)
#   1. Checks; each failure refuses with exit 9 and says what to fix: Linux; an NVIDIA GPU with nothing
#      else on it; nvcc; Nsight Systems and Nsight Compute, GPU counters they can read and NVTX filtering
#      (one-kernel probes; ERR_NVGPUCTRPERM gets the fix printed); a libnvToolsExt; RAM (the WHIR run
#      peaks near 45 GiB on the host) and disk; the Rust toolchain.
#   2. Clones https://github.com/yetanotherco/lambda_vm (public, HTTPS) into its own work directory and
#      checks out two pins, each a prover branch plus instruments-only commits (NVTX ranges, profiling
#      builds only; a default build is unchanged):
#        whir   PIN_W = noepoch/whir @ 708fe273f (PR #1014; parallel concatenation on, one prepared stack
#               per group; the layout workers and LT streaming exist but are off by default) + process-wide NVTX ranges on phase A, phase B, their per-group steps, the
#               card holds and the test's base/tree/verify windows
#        stark  PIN_S = noepoch/stark @ 08ecc4310 (PR #1013; the pipeline, own-pool emission, parallel
#               concatenation) + the same card-hold ranges and the tree test's level windows
#   3. Fetches block 25368371 from the public release and the record's guest ELF from the public
#      repository, both checked by sha256. No guest is built.
#   4. Builds each pin's test binary with --features cuda,nvtx into one target directory (line tables
#      in the cubins, codegen unchanged).
#   5. (--full) Reference: each workload once with no profiler.
#   6. Run A, Nsight Systems: each workload once with CUDA API + NVTX tracing and GPU metrics (2 kHz),
#      nvidia-smi every 200 ms and the CPU sampler every 100 ms. Per stage: wall, card busy %, kernel
#      time by family, SM active/issue %, warps in flight, DRAM and PCIe throughput, VRAM, CPU cores; every
#      card hold with its busy % and kernels; CUDA API seconds per category, thread and stage.
#   7. Run B, Nsight Compute: short passes, each restricted to one process-wide NVTX window (WHIR phase A
#      or B, or the STARK tree's multi_prove holds) and one kernel family, with SpeedOfLight,
#      MemoryWorkloadAnalysis, ComputeWorkloadAnalysis, Occupancy and LaunchStats plus stall and pipe
#      metrics, at --clock-control base. A config pass profiles one launch of each launch shape of ONE
#      kernel; a window pass profiles the first N launches of a family in its window and ends the run.
#   8. Exports text only, scrubs this machine's paths, hostname and addresses out of it, self-checks it
#      and packs it.
#
# MEMORY  The WHIR run peaks near 45 GiB of host RAM, the STARK run near 43 (FAST); the checks require
#   NP_MIN_AVAIL_GIB (48) available before every run, and a watchdog ends a run whose host MemAvailable
#   falls under NP_MEM_FLOOR_MIB (2048 MiB) rather than letting the machine swap. VRAM: up to the card.
# DISK  about 20-30 GiB in the work directory: the clone, two worktrees and the cargo cache (~4 GiB), the
#   release CUDA builds (~10-15 GiB), run A's reports and exports (~3-10 GiB). The file to send back is a
#   few MB; keep/ can be deleted once it is sent.
#
# NEEDS  Linux; an NVIDIA GPU with nothing else running on it (the record's card is an RTX 5090, 32 GB);
#   CUDA toolkit >= 12.8 with nvcc at $CUDA_HOME/bin (default /usr/local/cuda); Nsight Systems and Nsight
#   Compute >= 2025.1; GPU performance counters open to this user; 48 GiB of RAM available; >= 30 GiB
#   free in the work directory; rustup with the toolchain 1.94.0; git, curl, python3 >= 3.8 with sqlite3;
#   internet (GitHub, crates.io, files.pythonhosted.org). The NVTX ranges need a libnvToolsExt (CUDA 12.9
#   and later ship none): LAMBDA_VM_NVTX_LIB=/path/to/libnvToolsExt.so.1 when you have one; otherwise the
#   script fetches the nvidia-nvtx-cu12 12.8.90 wheel and checks it by sha256.
#
# RUN  (inside tmux or screen, the GPU otherwise idle, big programs closed)
#   bash noepoch_counters.sh --preflight-only   # the checks alone, about two minutes
#   bash noepoch_counters.sh --quick            # ~15 min after the build
#   bash noepoch_counters.sh --full             # ~45-60 min after the build
#   The work directory is ./lambda-vm-prof2 (NP_WORKDIR=/path to change). The script writes only there:
#   the clone, the cargo cache (CARGO_HOME), temporary files (TMPDIR), the tools' config (HOME) and the
#   runs. (Nsight itself keeps lock and IPC files in /tmp.)
#
# SEND BACK  one file: <work dir>/send-back.tar.gz (a copy of runs/<UTC>/counters-<UTC>.tar.gz, the
#   latest run). It holds text only (CSV, TSV, markdown, log excerpts; SUMMARY.md first), with this
#   machine's paths, hostname and addresses replaced by <W>, <HOME>, <HOST>, <IP>, and it has passed the
#   self-check printed beside it. The .nsys-rep, .sqlite and .ncu-rep files stay in runs/<UTC>/keep/:
#   Nsight stores the profiled process's environment in them. If the self-check refuses, it names
#   file:line (never the text) and packs nothing; `--bundle-only <run dir>` re-checks and packs.
#
# OPTIONS  --quick · --full · --preflight-only · --bundle-only RUNDIR · --print-plan · --print-prereg ·
#          --print-tool · --print-commands (every command the run would execute, nothing run) · -h|--help
# KNOBS (environment, all optional)
#   NP_WORKDIR         work directory (default ./lambda-vm-prof2)
#   NP_MODE            quick or full (the flags set it; default quick)
#   NP_GPU             GPU index as nvidia-smi numbers it (default 0)
#   NP_WORKLOADS       the workloads to run, in order (default "whir stark")
#   NP_REFERENCE       1 runs the unprofiled reference runs (default: 1 in full, 0 in quick)
#   NP_RUN_A           1 (default) runs run A (Nsight Systems); 0 skips it
#   NP_RUN_B           1 (default) runs run B (Nsight Compute); 0 skips it
#   NP_PASSES          run only these run-B passes, e.g. "whir_b_sumcheck" (--print-plan lists them)
#   NP_PLAN            a plan file in --print-plan's format instead of the built-in one
#   NP_SECTIONS        ncu section identifiers (default: the five above)
#   NP_METRICS         explicit ncu metrics, comma-separated (default: the list below; the checks keep only
#                      what this ncu and GPU can collect, since one unknown name makes ncu profile nothing)
#   NP_CLOCK           ncu --clock-control: base (default, as on 09-25 and 09-28), boost or none
#   NP_GPU_METRICS_HZ  run A's GPU-metrics sampling rate (default 2000)
#   NP_NCU_VRAM_MB     run B's VRAM budget in MiB (default: each workload's own, as in run A)
#   NP_DEADLINE_MIN    minutes from the start after which nothing new starts (quick 35, full 80)
#   NP_PASS_TIMEOUT    seconds per run-B pass (900) · NP_RUN_TIMEOUT per reference / run-A run (900)
#   NP_BUILD_TIMEOUT   seconds for each build (2400)
#   NP_IDLE_MIB        the GPU counts as idle below this many MiB in use with no compute process (500)
#   NP_MIN_AVAIL_GIB   host MemAvailable required before every run (default 48)
#   NP_MEM_FLOOR_MIB   the watchdog ends a run whose host MemAvailable falls under this (default 2048; 0: off)
#   NP_FETCH_NVTX      1 (default) fetches the NVTX library when none is found; 0 does not
#   NP_WHIR_KNOBS / NP_STARK_KNOBS   extra NAME=VALUE knobs for one workload (blank-separated)
#   NP_SCAN_ALLOW      environment variable NAMES whose values may appear in the bundle (after a look)
#   NP_CLEAN           1 deletes the build and keep/'s traces after packing (the lead's box; default 0)
#   NCU / NSYS         explicit tool paths
#   NP_REPO_URL / NP_INPUT_URL   mirrors; the content stays pinned by commit sha and by sha256
#   NP_DRY=1           the lead's validation on a box whose counters are closed: every check that needs no
#                      counters, the build, the reference runs, run A without GPU metrics, and every run-B
#                      pass's selection evaluated against run A's traces instead of running ncu
# EXIT  0 done · 2 usage · 3 finished, something failed (the VERDICT line says what) · 4 the self-check
#       refused the bundle · 5 clone, fixture or build failed · 9 a check refused (the message says what)
set -euo pipefail
umask 022

SCRIPT_VERSION=iprof2-2026-10-01a
REPO_URL_DEFAULT=https://github.com/yetanotherco/lambda_vm
PIN_W=2cf177497c24154d04939b05cb314856fddaee1b             # prof/whir-counters: noepoch/whir + NVTX ranges
PIN_S=5432a5e7298740e00b6212bad0c8d1e9a7a2e4f1             # prof/stark-counters: noepoch/stark + NVTX ranges
WHIR_SHA=708fe273fe1ab7ad3339d488cc994bda753e3a4d          # noepoch/whir (PR #1014), PIN_W's base
STARK_SHA=08ecc43108fa253f1841fb115e5133b769968cdd         # noepoch/stark (PR #1013), PIN_S's base
PIN_W_BRANCH=prof/whir-counters
PIN_S_BRANCH=prof/stark-counters
ELF_COMMIT=da2a423b137b87e04213ecafbfd5d16e98a1f4a3         # whir/profile-rpx, where the record ELF is committed
ELF_REPO_PATH=scripts/profile/fixtures/ethrex_8f826601.elf
ELF_SHA256=8f826601776d4085cbb6fbf0302fe8d8d5d1be7940ac1aaca24899c6244ec80a
INPUT_URL_DEFAULT=https://github.com/yetanotherco/lambda_vm/releases/download/bench-fixtures-v1/ethrex_mainnet_25368371_797df554.bin
INPUT_SHA256=573004e62e3680a00d3cdbae19dc4897e2ec60d6ec0c1d05d9ef118cb8aef17f
RUST_STABLE=1.94.0             # rust-toolchain.toml at both pins
FEATURES=cuda,nvtx             # nvtx implies cuda and instruments: the spans and the NVTX ranges exist
MIN_CUDA=12.8
MIN_NSIGHT=2025.1              # Blackwell-capable nsys and ncu
WHIR_TEST=lfm::whir_block_tests::the_whir_block_tree_on_a_real_block
STARK_TEST=lfm::block_tree_tests::the_block_tree_composes_to_a_top_node
NVTX_WHEEL_URL=https://files.pythonhosted.org/packages/a2/eb/86626c1bbc2edb86323022371c39aa48df6fd8b0a1647bc274577f72e90b/nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
NVTX_WHEEL_SHA256=5b17e2001cc0d751a5bc2c6ec6d26ad95913324a4adb86788c944f8ce9ba441f
NVTX_LIB_SHA256=c498fcbab0202886c27a0adeac44abf233ade03d30680ffa2d2abe93ab88d913   # nvidia/nvtx/lib/libnvToolsExt.so.1 in it
STARK_VRAM_MB=24000            # the STARK record's budget (FAST 456); the WHIR record sets none (80 % of the card)

DEFAULT_SECTIONS="SpeedOfLight MemoryWorkloadAnalysis ComputeWorkloadAnalysis Occupancy LaunchStats"
# On top of the sections: DRAM bytes (bytes per element), L2 bytes and throughput, issue and instruction
# counts, the occupancy limits, the busiest pipes by instructions and by active cycles, and the stall
# reasons (both only in charts otherwise), local memory and global coalescing. The 09-28 run on this
# machine collected all 40.
DEFAULT_METRICS="gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,\
lts__t_sector_hit_rate.pct,lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum,smsp__thread_inst_executed.sum,\
launch__registers_per_thread,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,\
launch__occupancy_limit_warps,launch__occupancy_limit_blocks,\
sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_uniform.avg.pct_of_peak_sustained_active,\
sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"

# Run B's pass plan. Columns: pass, workload, modes (quick,full or full), mode (config|window), skip,
# count, nvtx (the process-wide NVTX range ncu filters on, or -), kernels (a regex BODY: the tools anchor it
# as ^(...)$), family, note. A config pass names exactly ONE kernel, because ncu's per-launch-config key is
# the grid, block and shared memory, not the kernel name; it profiles the first launch of every shape in
# its window. A window pass profiles the first `count` matching launches in its window and ends the run.
# PLAN_ROWS_BEGIN
plan_rows() {
  printf '%s\n' \
    "whir_b_sumcheck	whir	quick,full	config	0	1	blk_phase_b	sumcheck_round_ext3	sumcheck	phase B, the zerocheck/GKR sumcheck rounds: one launch of each shape" \
    "whir_a_leaves	whir	quick,full	config	0	1	blk_phase_a	rpx_leaves_base_coset	leaves	phase A, the round-0 coset leaves of the group commits: one launch of each shape" \
    "whir_a_merkle	whir	full	config	0	1	blk_phase_a	rpx_merkle_level	merkle	phase A, Merkle internal levels: one launch of each width" \
    "whir_a_lde	whir	full	window	0	12	blk_phase_a	ntt_cm_di[ft]_k[4-8]|mobius_(tile|low_levels|level)	lde	phase A, the encoding (column-major NTT, Mobius): the first 12" \
    "whir_b_argue	whir	full	window	0	24	blk_phase_b	sumcheck_fold_ext3|fraction_fold(_padded)?_ext3|eq_expand_level(_shares)?_ext3|eq_seed_shares_ext3|factors_from_columns_ext3|mle_lift_base_ext3|mle_fold_base_ext3(_many)?|sum_partials_ext3	sumcheck	phase B, the GKR layers and folds: the first 24" \
    "whir_b_program_map	whir	full	config	0	1	blk_phase_b	program_map_ext3	sumcheck	phase B, the zerocheck's constraint program map: one launch of each shape" \
    "whir_b_open	whir	full	window	0	16	blk_phase_b	whir_lean_(round|materialize|colmap)|whir_fold_(k_)?(base_)?ext3|gather_cosets|rpx_leaves_ext3_coset	whir-fold	phase B, the opening (lean rounds, k-folds, folded leaves): the first 16" \
    "whir_b_encode	whir	full	window	0	12	blk_b_encode	ntt_cm_di[ft]_k[4-8]|mobius_(tile|low_levels|level)|rpx_leaves_base_coset	lde	phase B, the re-encode before each opening: the first 12" \
    "whir_b_merkle	whir	full	config	0	1	blk_phase_b	rpx_merkle_level	merkle	phase B, Merkle internal levels: one launch of each width" \
    "whir_b_grind	whir	full	window	0	5	blk_phase_b	rpx_grind_search_queue	grind	phase B, grinding: the first 5" \
    "stark_lfm_leaves	stark	quick,full	config	0	1	cardhold_multi_prove	rpx_leaves_base_row_pair_batched	leaves	LFM multi_prove, main-trace leaves: one launch of each shape" \
    "stark_lfm_quotient	stark	quick,full	window	0	24	cardhold_multi_prove	ccomp_[0-9a-f]+|constraint_composition_kernel	quotient	LFM multi_prove, the compiled constraint kernels: the first 24" \
    "stark_lfm_merkle	stark	full	config	0	1	cardhold_multi_prove	rpx_merkle_level	merkle	LFM multi_prove, Merkle internal levels: one launch of each width" \
    "stark_lfm_ntt7	stark	full	config	0	1	cardhold_multi_prove	ntt_cm_dit_k7	lde	LFM multi_prove, the column-major NTT k7: one launch of each shape" \
    "stark_lfm_ntt8	stark	full	config	0	1	cardhold_multi_prove	ntt_cm_dit_k8	lde	LFM multi_prove, the column-major NTT k8: one launch of each shape" \
    "stark_lfm_ext3_leaves	stark	full	config	0	1	cardhold_multi_prove	rpx_leaves_ext3_batched	leaves	LFM multi_prove, aux-trace leaves: one launch of each shape" \
    "stark_lfm_comp_poly	stark	full	config	0	1	cardhold_multi_prove	rpx_comp_poly_leaves_ext3	leaves	LFM multi_prove, composition-polynomial leaves: one launch of each shape" \
    "stark_lfm_deep	stark	full	config	0	1	cardhold_multi_prove	deep_composition_ext3_fused_m3	deep	LFM multi_prove, DEEP composition: one launch of each shape" \
    "stark_lfm_fri	stark	full	config	0	1	cardhold_multi_prove	rpx_fri_group_leaves_ext3	leaves	LFM multi_prove, FRI layer leaves: one launch of each shape" \
    "stark_lfm_logup	stark	full	window	0	8	cardhold_multi_prove	logup_[a-z0-9_]+	logup	LFM multi_prove, LogUp aux columns: the first 8" \
    "stark_lfm_grind	stark	full	window	0	5	cardhold_multi_prove	rpx_grind_search_queue	grind	LFM multi_prove, grinding: the first 5" \
    "stark_art_commit	stark	full	window	0	16	cardhold_build_artifacts	ntt_cm_di[ft]_k[4-8]|rpx_leaves_[a-z0-9_]+|rpx_merkle_level(_warp)?	lde	LFM artifact builds, the preprocessed commits: the first 16"
}
# PLAN_ROWS_END
default_plan() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' pass workload modes mode skip count nvtx kernels family note
  plan_rows
}

prereg_text() {
  cat <<'PREREG'
Pre-registered 2026-10-01 (lane I-PROF2), before any run of this script on the counter machine. Sources:
FAST 419 (no-epoch WHIR, whole 22.30 s, base 18.13-18.55, phase A 10.37-10.73, recursion 4.12-4.16),
FAST 456 (no-epoch STARK, whole 31.78 s: base 21.28, level 0 5.18, interior 4.88), FAST 454 (R4: idle
inside the recursion's holds 1.2-1.8 s of 9.5 s held), FAST 457 (the STARK recursion card-bound at
every level), G5-NCU (09-28: the epoch pipelines' base kernels at their roofs, RPX fmaheavy-bound) and this
script's FAST dry runs (490+). Mauro's machine is not FAST (another CPU, 60 GiB), so walls get wide bands;
the structural checks are exact.

Gates (a miss makes the VERDICT PARTIAL or FAILED):
  - every reference and run-A process passes its test (the WHIR tree verify ACCEPTED; the STARK tree
    reaches its final check);
  - run A's stages come from NVTX ranges (needs a libnvToolsExt), and the trace holds exactly as many
    cardhold_* ranges as the log's CARD HOLD lines (LFM_CARD_TRACE=1), at least one per workload;
  - every run-B pass that runs profiles at least one launch.
Expected (reported, not gated):
  - walls: WHIR base 16-22 s, whole 20-27 s; STARK base 19-25 s, recursion 9-13 s, whole 29-37 s;
    nsys run A within +15 % of the reference; host peak WHIR 40-48 GiB, STARK 36-46 GiB;
  - WHIR phase A 9-12 s, phase B 6-9 s; phase B card busy below phase A's (the argue's host glue, memory
    argue-idle-is-host-glue); the top phase-A kernels (RPX leaves, Merkle levels) at a compute roof
    (fmaheavy >= 80 %), as on 09-28;
  - STARK recursion: about 15 multi_prove holds of 0.3-0.7 s; inside them kernel busy 60-90 % (R4: idle
    13-19 % of held time); the LFM kernels' launches smaller than the base's, so more of them under one
    wave; RPX kernels still fmaheavy-bound where they fill the card.
PREREG
}

# ---------------------------------------------------------------------------------------------------
# small helpers

log() { printf 'NP %s %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
MODE_WORD="COUNTERS2"
die() { # die RC MESSAGE
  local rc="$1"
  shift
  log "FATAL: $*"
  echo "VERDICT: $MODE_WORD FAILED rc=$rc — $*"
  exit "$rc"
}
usage() { sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }
ver_ge() { # ver_ge HAVE WANT: dotted versions, true when HAVE >= WANT
  awk -v a="$1" -v b="$2" 'BEGIN {
    n = split(a, x, "."); m = split(b, y, "."); k = (n > m) ? n : m
    for (i = 1; i <= k; i++) {
      xi = (i <= n) ? x[i] + 0 : 0; yi = (i <= m) ? y[i] + 0 : 0
      if (xi > yi) exit 0
      if (xi < yi) exit 1
    }
    exit 0 }'
}
num_ge() { awk -v a="$1" -v b="$2" 'BEGIN { exit !((a + 0) >= (b + 0)) }'; }
first_match() { awk -v re="$1" '!done && match($0, re) { print substr($0, RSTART, RLENGTH); done = 1 }'; }
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{ print $1 }'
  else shasum -a 256 "$1" | awk '{ print $1 }'
  fi
}
md5_of() {
  if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | awk '{ print $1 }'
  else md5 -q "$1"
  fi
}
cuda_home() { printf '%s\n' "${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}"; }
gpu_field() { # gpu_field FIELD: one nvidia-smi --query-gpu field of GPU $NP_GPU, trimmed
  { nvidia-smi -i "$NP_GPU" --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null || true; } |
    awk 'NR == 1 { gsub(/^ +| +$/, ""); print }'
}
gpu_reading() { # "N MiB in use, K compute process(es)"; returns 0 when idle
  local used apps
  used="$(gpu_field memory.used)"
  apps="$({ nvidia-smi -i "$NP_GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true; } | awk 'NF { n++ } END { print n + 0 }')"
  echo "gpu $NP_GPU: ${used:-?} MiB in use, $apps compute process(es)"
  case "$used" in ''|*[!0-9]*) return 1 ;; esac
  if [ "$used" -lt "$NP_IDLE_MIB" ] && [ "$apps" -eq 0 ]; then return 0; fi
  return 1
}
wait_idle() { # a run that has just ended must release the card before the next one starts
  local reading="" i
  for i in $(seq 1 30); do
    if reading="$(gpu_reading)"; then return 0; fi
    if [ "$i" -lt 30 ]; then sleep 2; fi
  done
  die 9 "the GPU is not idle ($reading): another process holds it; wait for it to finish, then run again"
}
host_os() { uname -s; }
mem_total_gib() { awk '/^MemTotal:/ { printf "%.1f", $2 / 1048576 }' /proc/meminfo 2>/dev/null || true; }
mem_avail_gib() { awk '/^MemAvailable:/ { printf "%.1f", $2 / 1048576 }' /proc/meminfo 2>/dev/null || true; }
cgroup_limit_gib() {
  local v=""
  if [ -r /sys/fs/cgroup/memory.max ]; then v="$(cat /sys/fs/cgroup/memory.max 2>/dev/null || true)"
  elif [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then v="$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || true)"
  fi
  case "$v" in ''|max|*[!0-9]*) return 0 ;; esac
  awk -v b="$v" 'BEGIN { if (b < 2 ^ 50) printf "%.1f", b / 2 ^ 30 }'
}
free_gib() { df -Pk "$1" 2>/dev/null | awk 'NR == 2 { printf "%.0f", $4 / 1048576 }'; }
cpu_model() { awk -F': ' '/^model name/ { print $2; exit }' /proc/cpuinfo 2>/dev/null || true; }
local_ips() { # this machine's addresses (IPv4 and IPv6), for the scrub and the self-check
  { hostname -I 2>/dev/null || true; } | tr ' ' '\n'
  printf '%s %s\n' "${SSH_CONNECTION:-}" "${SSH_CLIENT:-}" | tr ' ' '\n' | awk '/[.:]/'
}

find_tool() { # find_tool ncu|nsys: the real Nsight binary, recognised by its banner (an `ncu` on PATH can be npm-check-updates)
  local tool="$1" banner c out
  local -a cands=()
  case "$tool" in
    ncu) banner='Nsight Compute'; cands=("${NCU:-}") ;;
    nsys) banner='Nsight Systems'; cands=("${NSYS:-}") ;;
    *) return 2 ;;
  esac
  while IFS= read -r c; do cands+=("$c"); done < <(type -ap "$tool" 2>/dev/null || true)
  cands+=("$(cuda_home)/bin/$tool")
  if [ "$tool" = ncu ]; then
    # shellcheck disable=SC2012 # version directories, newest first
    while IFS= read -r c; do cands+=("$c"); done < <(ls -d /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | sort -rV || true)
  else
    # shellcheck disable=SC2012 # as above
    while IFS= read -r c; do cands+=("$c"); done < <(ls -d /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | sort -rV || true)
  fi
  for c in "${cands[@]}"; do
    if [ -z "$c" ] || [ ! -x "$c" ]; then continue; fi
    out="$("$c" --version 2>/dev/null || true)"
    case "$out" in *"$banner"*) printf '%s\n' "$c"; return 0 ;; esac
  done
  return 1
}
tool_version() { { "$1" --version 2>/dev/null || true; } | first_match '20[0-9][0-9][.][0-9]+([.][0-9]+)*'; }
find_nvtx_lib() { # the libnvToolsExt a --features nvtx binary can dlopen (crypto/math-cuda/src/nvtx.rs), or nothing
  local c d
  local -a cands=("${LAMBDA_VM_NVTX_LIB:-}")
  while IFS= read -r c; do cands+=("$c"); done < <({ ldconfig -p 2>/dev/null || true; } | awk '/libnvToolsExt\.so/ { print $NF }')
  local IFS_SAVE="$IFS"
  IFS=:
  for d in ${LD_LIBRARY_PATH:-}; do cands+=("$d/libnvToolsExt.so.1" "$d/libnvToolsExt.so"); done
  IFS="$IFS_SAVE"
  cands+=("$(cuda_home)/lib64/libnvToolsExt.so.1" "$(cuda_home)/lib64/libnvToolsExt.so" "$REAL_HOME/nvtx/libnvToolsExt.so.1")
  while IFS= read -r c; do cands+=("$c"); done < <(ls -d /usr/local/cuda*/lib64/libnvToolsExt.so* /usr/lib/x86_64-linux-gnu/libnvToolsExt.so* 2>/dev/null || true)
  for c in "${cands[@]}"; do
    if [ -n "$c" ] && [ -f "$c" ]; then printf '%s\n' "$c"; return 0; fi
  done
  return 1
}
fetch_nvtx_lib() { # the nvidia-nvtx-cu12 12.8.90 wheel's libnvToolsExt.so.1 into $W/nvtx, both checked by sha256
  local d="$W/nvtx" whl lib
  lib="$d/libnvToolsExt.so.1" whl="$d/nvidia_nvtx_cu12-12.8.90.whl"
  if [ -f "$lib" ] && [ "$(sha256_of "$lib")" = "$NVTX_LIB_SHA256" ]; then printf '%s\n' "$lib"; return 0; fi
  mkdir -p "$d"
  curl -fsSL --retry 3 -o "$whl" "$NVTX_WHEEL_URL" > "$d/curl.log" 2>&1 || return 1
  [ "$(sha256_of "$whl")" = "$NVTX_WHEEL_SHA256" ] || return 1
  python3 - "$whl" "$lib" <<'PY' >> "$d/curl.log" 2>&1 || return 1
import sys, zipfile
with open(sys.argv[2], "wb") as f:
    f.write(zipfile.ZipFile(sys.argv[1]).read("nvidia/nvtx/lib/libnvToolsExt.so.1"))
PY
  [ "$(sha256_of "$lib")" = "$NVTX_LIB_SHA256" ] || return 1
  printf '%s\n' "$lib"
}

# ---------------------------------------------------------------------------------------------------
# knobs, work directory, run directory

REAL_HOME="${HOME:-}"
RUSTUP_HOME_REAL="${RUSTUP_HOME:-$REAL_HOME/.rustup}"
knob_defaults() {
  local v w
  NP_WORKDIR="${NP_WORKDIR:-$PWD/lambda-vm-prof2}"
  NP_MODE="${NP_MODE:-quick}"
  case "$NP_MODE" in quick|full) ;; *) echo "NP_MODE must be quick or full, got '$NP_MODE'" >&2; exit 2 ;; esac
  NP_GPU="${NP_GPU:-0}"
  NP_DRY="${NP_DRY:-0}"
  NP_WORKLOADS="${NP_WORKLOADS:-whir stark}"
  if [ "$NP_MODE" = full ]; then NP_REFERENCE="${NP_REFERENCE:-1}"; else NP_REFERENCE="${NP_REFERENCE:-0}"; fi
  NP_RUN_A="${NP_RUN_A:-1}"
  NP_RUN_B="${NP_RUN_B:-1}"
  NP_PASSES="${NP_PASSES:-}"
  NP_PLAN="${NP_PLAN:-}"
  NP_SECTIONS="${NP_SECTIONS:-$DEFAULT_SECTIONS}"
  NP_METRICS="${NP_METRICS:-$DEFAULT_METRICS}"
  NP_CLOCK="${NP_CLOCK:-base}"
  NP_GPU_METRICS_HZ="${NP_GPU_METRICS_HZ:-2000}"
  NP_NCU_VRAM_MB="${NP_NCU_VRAM_MB:-}"
  if [ "$NP_MODE" = full ]; then NP_DEADLINE_MIN="${NP_DEADLINE_MIN:-80}"; else NP_DEADLINE_MIN="${NP_DEADLINE_MIN:-35}"; fi
  NP_PASS_TIMEOUT="${NP_PASS_TIMEOUT:-900}"
  NP_RUN_TIMEOUT="${NP_RUN_TIMEOUT:-900}"
  NP_BUILD_TIMEOUT="${NP_BUILD_TIMEOUT:-2400}"
  NP_IDLE_MIB="${NP_IDLE_MIB:-500}"
  NP_MIN_AVAIL_GIB="${NP_MIN_AVAIL_GIB:-48}"
  NP_MEM_FLOOR_MIB="${NP_MEM_FLOOR_MIB:-2048}"
  NP_FETCH_NVTX="${NP_FETCH_NVTX:-1}"
  NP_WHIR_KNOBS="${NP_WHIR_KNOBS:-}"
  NP_STARK_KNOBS="${NP_STARK_KNOBS:-}"
  NP_CLEAN="${NP_CLEAN:-0}"
  NP_REPO_URL="${NP_REPO_URL:-$REPO_URL_DEFAULT}"
  NP_INPUT_URL="${NP_INPUT_URL:-$INPUT_URL_DEFAULT}"
  NP_SCAN_ALLOW="${NP_SCAN_ALLOW:-}"
  case "$NP_GPU" in ''|*[!0-9]*) echo "NP_GPU must be a number, got '$NP_GPU'" >&2; exit 2 ;; esac
  for v in NP_DRY NP_REFERENCE NP_RUN_A NP_RUN_B NP_FETCH_NVTX NP_CLEAN; do
    case "${!v}" in 0|1) ;; *) echo "$v must be 0 or 1, got '${!v}'" >&2; exit 2 ;; esac
  done
  case "$NP_CLOCK" in base|boost|none) ;; *) echo "NP_CLOCK must be base, boost or none, got '$NP_CLOCK'" >&2; exit 2 ;; esac
  for v in NP_PASS_TIMEOUT NP_RUN_TIMEOUT NP_BUILD_TIMEOUT NP_IDLE_MIB NP_GPU_METRICS_HZ NP_MIN_AVAIL_GIB NP_MEM_FLOOR_MIB NP_DEADLINE_MIN; do
    case "${!v}" in ''|*[!0-9]*) echo "$v must be a number, got '${!v}'" >&2; exit 2 ;; esac
  done
  case "$NP_NCU_VRAM_MB" in ''|*[!0-9]*) [ -z "$NP_NCU_VRAM_MB" ] || { echo "NP_NCU_VRAM_MB must be a number, got '$NP_NCU_VRAM_MB'" >&2; exit 2; } ;; esac
  [ -n "$NP_WORKLOADS" ] || { echo "NP_WORKLOADS is empty" >&2; exit 2; }
  for v in $NP_WHIR_KNOBS $NP_STARK_KNOBS; do
    [[ "$v" =~ ^(LAMBDA_VM|LFM|BLOCK_WHIR|NOEPOCH|W3)_[A-Z0-9_]+=[A-Za-z0-9_.,:-]*$ ]] || { echo "NP_*_KNOBS: '$v' is not a LAMBDA_VM_/LFM_/BLOCK_WHIR_/NOEPOCH_/W3_ NAME=VALUE" >&2; exit 2; }
  done
  for w in $NP_WORKLOADS; do
    case "$w" in whir|stark) ;; *) echo "NP_WORKLOADS: unknown workload '$w' (whir, stark)" >&2; exit 2 ;; esac
  done
  if [ "$NP_DRY" = 1 ]; then MODE_WORD="COUNTERS2 DRY-RUN"; fi
  MODE_WORD="$MODE_WORD $NP_MODE"
}

setup_workdir() {
  mkdir -p "$NP_WORKDIR"
  W="$(cd "$NP_WORKDIR" && pwd -P)"
  REPO="$W/lambda_vm"
  REPO_S="$W/lambda_vm-stark"
  TARGET="$W/target"
  TOOL="$W/tools/noepoch_counters_summary.py"
  mkdir -p "$W/tools" "$W/tmp" "$W/home" "$W/cargo-home" "$W/fixtures" "$W/probe" "$W/runs"
  write_tool
}

TEE_PID=""
setup_run() { # a fresh run directory; everything printed from here on is also in keep/driver.log
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  RUN="$W/runs/$STAMP"
  if [ -e "$RUN" ]; then sleep 1; STAMP="$(date -u +%Y%m%dT%H%M%SZ)"; RUN="$W/runs/$STAMP"; fi
  SEND="$RUN/send"
  KEEP="$RUN/keep"
  mkdir -p "$SEND/ncu" "$SEND/logs" "$SEND/summary" "$SEND/runa" "$SEND/reference" "$SEND/cpu" "$KEEP/ncu"
  exec > >(tee -a "$KEEP/driver.log") 2>&1
  TEE_PID=$!
  printf 'step\tstart_utc\tseconds\trc\n' > "$SEND/steps.tsv"
}

STEP_NAME="" STEP_T0=0
step_begin() { STEP_NAME="$1"; STEP_T0="$(date +%s)"; log "== $1"; }
step_end() {
  local rc="${1:-0}" dt
  dt=$(( $(date +%s) - STEP_T0 ))
  printf '%s\t%s\t%s\t%s\n' "$STEP_NAME" "$(date -u +%FT%TZ)" "$dt" "$rc" >> "$SEND/steps.tsv"
  log "== done (rc=$rc, ${dt} s): $STEP_NAME"
}

write_tool() { # the Python side (stages, holds, CPU, ncu summary, scrub, self-check): noepoch_counters_summary.py, verbatim
  cat > "$TOOL" <<'NOEPOCH_COUNTERS_SUMMARY_PY_EOF'
#!/usr/bin/env python3
"""noepoch_counters_summary.py: the text side of noepoch_counters.sh (lane I-PROF2, 2026-10-01).

Standard library only (python >= 3.8; `runa` and `dry` also need the sqlite3 module).
noepoch_counters.sh carries a byte-identical copy of this file and writes it into its
work directory; the copy next to the script is the one to read and edit. It descends from
mauro_ncu_summary.py (lane G5-NCU, 2026-09-28) and lane I-PROF's version (09-30): the ncu
parsing, the scrub and the self-check are theirs; the workloads, the stage windows, the card
holds, the kernel families per stage, the CPU sampler and the NVTX-filtered passes are new.

    runa           --sqlite S --log L --workload W --out O [--smi F] [--cpu PREFIX] [--bin-ms N]
                                                   run A: per stage the card's busy time, GPU metrics, VRAM,
                                                   the kernels and their families, the host's CPU (cores busy,
                                                   per thread role); every card hold (an LFM proof's device
                                                   phase); the kernels each NVTX label launched; CUDA API time
                                                   per category, thread and stage; a time series
    summary        --plan P --ncu-dir D --out O    run B: one row per launch, per kernel and per stage
    stages         --log L --nvtx N                run B: the stage of each profiled launch (the pass's NVTX
                                                   window, or `run`)
    dry            --plan P --workload W --sqlite S --log L --out O [--modes M]
                                                   what each ncu pass would profile, against run A's trace
    cpu-sample     --exe BIN --out PREFIX [--interval-ms N] [--mem-floor-mib N]
                                                   the profiled process's CPU ticks per thread and the host's
                                                   MemAvailable every N ms, and a memory watchdog that ends the
                                                   process below the floor (PREFIX.watchdog says so)
    cpu-total      --cpu PREFIX                    one line: CPU seconds, mean cores busy, peak RSS of a run
    loghead        --log L --workload W            one line: the run's log gates (ok|bad), test result, card
                                                   holds, headline
    report         --send D                        SUMMARY.md: the text summary of a run, from the files above
    plan-check     --plan P                        the pass plan is well formed
    cargo-artifact (--exe NAME | --outdir PKG)     one path from cargo's JSON messages on stdin
    scrub          --dir D                         literal replacements (stdin: OLD<TAB>NEW lines), and the
                                                   "Host Name" column dropped from ncu CSVs
    check          --dir D --report R              the bundle self-check (stdin: LABEL<TAB>VALUE records,
                                                   NUL-separated): plain text only, no Nsight report or
                                                   database, no credential marker, none of the values
    selftest                                       every subcommand on synthetic inputs

Workloads. `whir`: the no-epoch WHIR block and its tree (lfm::whir_block_tests::
the_whir_block_tree_on_a_real_block, PR #1014). `stark`: the no-epoch STARK block and its tree
(lfm::block_tree_tests::the_block_tree_composes_to_a_top_node, PR #1013).

Stages (run A). The build carries process-wide NVTX ranges (start/end, on no thread's stack) for
the windows, and the prover's instruments spans as push/pop ranges on their threads:
  whir   setup (trace start to the base) · phase_a (blk_phase_a: the streamed build, the uploads and
         the commits) · prepared (blk_prepared) · phase_b (blk_phase_b: the argue and the openings) ·
         base_other (the rest of blk_base: the statement, the absorb, the glue) · recursion (lfm_tree)
         · verify (harness_verify to the end) · other. Overlapping rows: a_upload, a_commit, a_retire,
         b_upload, b_argue, b_encode, b_open (one range per group each) and the card holds.
  stark  setup · base_prepass, base_main_commit, base_fused (the multi_prove spans r1_prepass,
         r1_main_commit, rounds_2to4, inside blk_base) · base_other · harvest (tree_harvest) · level0
         (tree_level0) · interior (tree_interior) · verify · other. Overlapping rows: precommit,
         recommit and the card holds.
A card hold is a range cardhold_<phase> (multi_prove or build_artifacts): one LFM proof's device
phase under the tree's card permit, which admits one holder at a time, so every kernel inside it is
that proof's. Without NVTX ranges there is one window, `whole`.

What `summary` reports, per profiled launch and per kernel (duration-weighted over its launches):
  time      ncu's Duration, at the clock ncu held (launches.tsv's sm_ghz; env.txt names the
            --clock-control mode). Under `base` an RTX 5090 runs its SMs near 2.0 GHz where the block
            runs near 2.76 GHz, so a compute-bound kernel's duration here is ~1.4x its in-block time.
            The percentages are against the peak at the clock ncu ran.
  DRAM %    dram throughput, % of peak (Speed Of Light)
  SM %      Compute (SM) throughput, % of peak (Speed Of Light)
  L2 %      L2 throughput, % of peak (Speed Of Light); L2 hit % from Memory Workload Analysis
  occ %     achieved occupancy (theoretical beside it, and the block limit that sets it)
  waves     waves per SM (under 1: the launch cannot fill the card)
  B/elem    DRAM bytes (read + write) per element: a field element for the column-major NTT (16 per
            thread), a leaf for the leaf kernels, a parent for the Merkle levels (a half-warp per
            parent in rpx_merkle_level_warp), a thread for every other kernel
  roof      "DRAM", "compute" or "L2" roof when its % of peak is >= 60, "mid" between 30 and 60,
            "below" when all three are under 30 (latency, occupancy or launch bound)

ncu's --csv prints base units (nsecond, byte, byte/second, cycle/second) and, in some versions and
columns, thousands separators; every value goes through num() and a unit table, and a value that
does not parse is left empty rather than guessed.
"""
import argparse
import bisect
import csv
import io
import json
import os
import re
import signal
import sys
import tempfile
import threading
import time
from datetime import datetime

PLAN_COLS = ["pass", "workload", "modes", "mode", "skip", "count", "nvtx", "kernels", "family", "note"]
WORKLOADS = ("whir", "stark")
MODES = ("window", "config")
RUN_MODES = ("quick", "full")

# ---------------------------------------------------------------------------------------------
# small helpers


def num(v):
    """ncu's printed number -> float. Thousands separators ("4,194,304") are dropped; anything
    else that is not a number (n/a, empty) is None."""
    if v is None:
        return None
    v = v.strip()
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", v):
        v = v.replace(",", "")
    try:
        return float(v)
    except ValueError:
        return None


TIME_US = {"ns": 1e-3, "nsecond": 1e-3, "nseconds": 1e-3, "usecond": 1.0, "useconds": 1.0, "us": 1.0,
           "msecond": 1e3, "mseconds": 1e3, "ms": 1e3, "second": 1e6, "seconds": 1e6, "s": 1e6}
BYTE_PREFIX = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "Ki": 1024.0, "Mi": 1024.0 ** 2,
               "Gi": 1024.0 ** 3, "Ti": 1024.0 ** 4}
GHZ = {"hz": 1e-9, "khz": 1e-6, "mhz": 1e-3, "ghz": 1.0, "cycle/second": 1e-9, "cycle/s": 1e-9,
       "cycle/msecond": 1e-6, "cycle/ms": 1e-6, "cycle/usecond": 1e-3, "cycle/us": 1e-3,
       "cycle/nsecond": 1.0, "cycle/ns": 1.0}


def to_us(value, unit):
    f = TIME_US.get((unit or "").strip().lower())
    return None if value is None or f is None else value * f


def to_bytes(value, unit):
    """byte, Kbyte, Mbyte, ... (ncu's prefixes are decimal: a 16,384-byte carveout prints 16.38 Kbyte)."""
    m = re.fullmatch(r"([KMGT]i?)?(bytes?|B)", (unit or "").strip())
    if value is None or not m:
        return None
    return value * BYTE_PREFIX[m.group(1) or ""]


def to_bytes_per_s(value, unit):
    u = (unit or "").strip()
    m = re.fullmatch(r"(.+?)/(s|second)", u)
    return None if not m else to_bytes(value, m.group(1))


def to_ghz(value, unit):
    f = GHZ.get((unit or "").strip().lower())
    return None if value is None or f is None else value * f


def dims(s):
    """"(128, 1, 1)" or "128, 1, 1" -> (128, 1, 1); anything else -> None."""
    m = re.fullmatch(r"\(?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)?", (s or "").strip())
    return tuple(int(x) for x in m.groups()) if m else None


def prod(t):
    p = 1
    for x in t:
        p *= x
    return p


def fmt(x, nd=1):
    if x is None:
        return "-"
    if isinstance(x, float) and abs(x) >= 1e5:
        return f"{x:.3g}"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def shape(grid, block):
    g = "x".join(str(v) for v in grid) if grid else "?"
    b = "x".join(str(v) for v in block) if block else "?"
    return f"{g}/{b}"


def weighted(items, key):
    """Duration-weighted mean of key over launches that have both."""
    num_, den = 0.0, 0.0
    for it in items:
        v, d = it.get(key), it.get("dur_us")
        if v is not None and d:
            num_ += v * d
            den += d
    return num_ / den if den else None


def write_tsv(path, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f, delimiter="\t", lineterminator="\n").writerows(rows)


def read_tsv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# ---------------------------------------------------------------------------------------------
# the plan


def read_plan(path):
    rows = []
    with open(path, newline="") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "pass":
                if parts != PLAN_COLS:
                    raise SystemExit(f"plan {path}: header must be {'<TAB>'.join(PLAN_COLS)}")
                continue
            if len(parts) != len(PLAN_COLS):
                raise SystemExit(f"plan {path}:{ln}: {len(parts)} fields, want {len(PLAN_COLS)}")
            r = dict(zip(PLAN_COLS, parts))
            r["line"] = ln
            rows.append(r)
    return rows


def check_plan(rows):
    errs, seen = [], set()
    for r in rows:
        where = f"line {r['line']} ({r['pass']})"
        if not re.fullmatch(r"[a-z0-9_]+", r["pass"]):
            errs.append(f"{where}: pass name must be [a-z0-9_]+")
        if r["pass"] in seen:
            errs.append(f"{where}: duplicate pass name")
        seen.add(r["pass"])
        if r["workload"] not in WORKLOADS:
            errs.append(f"{where}: workload must be one of {WORKLOADS}")
        if r["mode"] not in MODES:
            errs.append(f"{where}: mode must be one of {MODES}")
        if not r["modes"] or any(m not in RUN_MODES for m in r["modes"].split(",")):
            errs.append(f"{where}: modes is a comma-separated subset of {RUN_MODES}")
        if r["nvtx"] != "-" and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", r["nvtx"]):
            errs.append(f"{where}: nvtx is - or one process-wide NVTX range name ([A-Za-z][A-Za-z0-9_]*)")
        if not r["skip"].isdigit():
            errs.append(f"{where}: skip must be an integer >= 0")
        if not r["count"].isdigit() or int(r["count"]) < 1:
            errs.append(f"{where}: count must be an integer >= 1")
        if any(c in r["kernels"] for c in "^$\t "):
            errs.append(f"{where}: kernels is the regex BODY (anchored as ^(...)$ by the tools), no ^ $ or blanks")
        try:
            re.compile(r["kernels"])
        except re.error as e:
            errs.append(f"{where}: kernels does not compile: {e}")
        # ncu's per-launch-config key is grid, block and shared memory, not the kernel: a config
        # pass over two kernels would skip the second one's launches whose shape the first used.
        if r["mode"] == "config" and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", r["kernels"]):
            errs.append(f"{where}: a config pass names exactly one kernel (ncu's per-launch-config key "
                        "does not include the kernel name)")
        for c in ("family", "note"):
            if not r[c].strip():
                errs.append(f"{where}: {c} is empty")
    return errs


def kernel_re(row):
    """The pass's kernel regex; callers fullmatch it, as ncu is handed ^(...)$."""
    return re.compile("(?:" + row["kernels"] + ")")


def matches(row, name):
    return kernel_re(row).fullmatch(name) is not None


# ---------------------------------------------------------------------------------------------
# summary: ncu details CSV -> one row per launch, one row per kernel

# (kernel regex, element name, elements per thread); first match wins, default one thread
ELEMENTS = [
    (r"ntt_cm_di[ft]_k[4-8]", "felt", 16.0),
    (r"rpx_merkle_level_warp", "parent", 1.0 / 16.0),
    (r"rpx_merkle_level", "parent", 1.0),
    (r"rpx_(leaves_[a-z0-9_]+|comp_poly_leaves_ext3|fri_leaves_ext3|fri_group_leaves_ext3)", "leaf", 1.0),
]

FAMILIES = [  # the summary's family of a kernel, by name
    (r"ntt_[a-z0-9_]+|mobius_[a-z_]+|lift_spread|matrix_transpose_strided|bit_reverse_(permute|row_major)[a-z_]*"
     r"|pointwise_mul[a-z_]*|scalar_mul[a-z_]*", "lde"),
    (r"rpx_merkle_[a-z_]+", "merkle"),
    (r"rpx_grind_[a-z_]+", "grind"),
    (r"rpx_[a-z0-9_]*leaves[a-z0-9_]*", "leaves"),
    (r"sumcheck_[a-z0-9_]+|program_map_ext3|eq_[a-z_]+|fraction_fold[a-z_]*|mle_[a-z0-9_]+|factors_from_columns_ext3"
     r"|sum_partials_ext3|add_scaled_ext3|fill_ext3", "sumcheck"),
    (r"constraint_[a-z_]+|ccomp_[0-9a-f]+|comp_h_to_slabs_ext3|decompose_d2_ext3", "quotient"),
    (r"logup_[a-z0-9_]+", "logup"),
    (r"deep_[a-z0-9_]+|bit_reverse_ext3_interleaved|invert_[a-z0-9_]+|compute_denoms_ext3|batch_inverse_[a-z0-9_]+"
     r"|block_inclusive_scan_[a-z0-9_]+|apply_block_offsets_[a-z0-9_]+|barycentric_[a-z0-9_]+|gather_rows_[a-z0-9_]+",
     "deep"),
    (r"fri_[a-z0-9_]+|gather_ext3_at|merkle_gather_paths", "fri"),
    (r"whir_[a-z0-9_]+|gather_cosets", "whir-fold"),
]

STALL_RE = re.compile(r"smsp__average_warps?_issue_stalled_(\w+?)_per_issue_active\.ratio")
PIPE_RE = re.compile(r"sm__inst_executed_pipe_(\w+?)\.avg\.pct_of_peak_sustained_active")
PIPE_CYC_RE = re.compile(r"sm__pipe_(\w+?)_cycles_active\.avg\.pct_of_peak_sustained_active")


def element_of(kernel):
    for pat, name, per in ELEMENTS:
        if re.fullmatch(pat, kernel):
            return name, per
    return "thread", 1.0


def family_of(kernel):
    for pat, fam in FAMILIES:
        if re.fullmatch(pat, kernel):
            return fam
    return "other"


def read_details(path):
    """{id: {"kernel", "grid", "block", "m": {(section, metric): (unit, value)}}}, in ID order."""
    with open(path, newline="", errors="replace") as f:
        lines = [ln for ln in f if not ln.startswith("==")]
    rdr = csv.reader(lines)
    hdr = None
    launches = {}
    for row in rdr:
        if hdr is None:
            if "ID" in row and "Metric Name" in row and "Kernel Name" in row:
                hdr = {c: i for i, c in enumerate(row)}
            continue
        if len(row) < len(hdr) - 6:  # rule columns may be missing on short rows
            continue

        def col(c):
            i = hdr.get(c)
            return row[i] if i is not None and i < len(row) else ""

        lid = col("ID")
        if not lid.isdigit():
            continue
        lid = int(lid)
        rec = launches.setdefault(lid, {"kernel": col("Kernel Name"), "grid": dims(col("Grid Size")),
                                        "block": dims(col("Block Size")), "m": {}})
        name = col("Metric Name")
        if name:
            rec["m"][(col("Section Name"), name)] = (col("Metric Unit"), col("Metric Value"))
    return [dict(v, id=k) for k, v in sorted(launches.items())]


def metric(rec, names):
    """First present (section, metric) pair of `names` -> (value, unit); section None = any."""
    for sec, name in names:
        for (s, n), (u, v) in rec["m"].items():
            if n == name and (sec is None or s == sec):
                x = num(v)
                if x is not None:
                    return x, u
    return None, None


SOL = "GPU Speed Of Light Throughput"
MWA = "Memory Workload Analysis"
CWA = "Compute Workload Analysis"
OCC = "Occupancy"
LST = "Launch Statistics"


def launch_row(rec):
    kernel = rec["kernel"]
    out = {"id": rec["id"], "kernel": kernel, "grid": rec["grid"], "block": rec["block"]}
    grid, block = rec["grid"], rec["block"]
    threads = prod(grid) * prod(block) if grid and block else None
    if threads is None:
        t, _ = metric(rec, [(LST, "Threads")])
        threads = int(t) if t else None
    out["threads"] = threads
    v, u = metric(rec, [(SOL, "Duration"), (None, "gpu__time_duration.sum")])
    out["dur_us"] = to_us(v, u)
    v, u = metric(rec, [(SOL, "SM Frequency")])
    out["sm_ghz"] = to_ghz(v, u)
    out["dram_pct"], _ = metric(rec, [(SOL, "DRAM Throughput"),
                                      (None, "dram__throughput.avg.pct_of_peak_sustained_elapsed")])
    out["sm_pct"], _ = metric(rec, [(SOL, "Compute (SM) Throughput"),
                                    (None, "sm__throughput.avg.pct_of_peak_sustained_elapsed")])
    out["mem_pct"], _ = metric(rec, [(SOL, "Memory Throughput"),
                                     (None, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")])
    out["l2_pct"], _ = metric(rec, [(SOL, "L2 Cache Throughput"),
                                    (None, "lts__throughput.avg.pct_of_peak_sustained_elapsed")])
    out["l1_pct"], _ = metric(rec, [(SOL, "L1/TEX Cache Throughput")])
    out["l2_hit"], _ = metric(rec, [(MWA, "L2 Hit Rate"), (None, "lts__t_sector_hit_rate.pct")])
    out["occ"], _ = metric(rec, [(OCC, "Achieved Occupancy"),
                                 (None, "sm__warps_active.avg.pct_of_peak_sustained_active")])
    out["occ_theo"], _ = metric(rec, [(OCC, "Theoretical Occupancy")])
    out["regs"], _ = metric(rec, [(LST, "Registers Per Thread"), (None, "launch__registers_per_thread")])
    out["waves"], _ = metric(rec, [(LST, "Waves Per SM")])
    out["issue_pct"], _ = metric(rec, [(CWA, "Issue Slots Busy"),
                                       (None, "smsp__issue_active.avg.pct_of_peak_sustained_active")])
    out["ipc"], _ = metric(rec, [(CWA, "Executed Ipc Active")])
    limits = {}
    for lim in ("Registers", "Shared Mem", "Warps", "SM", "Barriers"):
        x, _ = metric(rec, [(OCC, "Block Limit " + lim)])
        if x is not None:
            limits[lim] = x
    out["occ_limit"] = min(limits, key=lambda k: limits[k]) if limits else None
    # DRAM bytes: the explicit counters when collected, else the throughput x the duration
    r, ru = metric(rec, [(None, "dram__bytes_read.sum")])
    w, wu = metric(rec, [(None, "dram__bytes_write.sum")])
    rb, wb = to_bytes(r, ru), to_bytes(w, wu)
    if rb is not None and wb is not None:
        out["dram_bytes"], out["bytes_src"] = rb + wb, "counters"
    else:
        v, u = metric(rec, [(MWA, "Memory Throughput")])
        bps = to_bytes_per_s(v, u)
        if bps is not None and out["dur_us"] is not None:
            out["dram_bytes"], out["bytes_src"] = bps * out["dur_us"] * 1e-6, "throughput x duration"
        else:
            out["dram_bytes"], out["bytes_src"] = None, None
    elem, per = element_of(kernel)
    out["elem"] = elem
    out["elems"] = threads * per if threads else None
    out["b_per_elem"] = (out["dram_bytes"] / out["elems"]
                         if out["dram_bytes"] is not None and out["elems"] else None)
    stalls, pipes, cycles = {}, {}, {}
    for (_s, n), (_u, v) in rec["m"].items():
        x = num(v)
        if x is None:
            continue
        for rx, into in ((STALL_RE, stalls), (PIPE_RE, pipes), (PIPE_CYC_RE, cycles)):
            m = rx.fullmatch(n)
            if m:
                into[m.group(1)] = x
    tot = sum(stalls.values())
    out["stalls"] = ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in
                              sorted(stalls.items(), key=lambda kv: -kv[1])[:3]) if tot > 0 else ""
    out["pipes"] = ", ".join(f"{k} {v:.0f}%" for k, v in sorted(pipes.items(), key=lambda kv: -kv[1])[:2])
    out["pipe_cycles"] = ", ".join(f"{k} {v:.0f}%" for k, v in sorted(cycles.items(), key=lambda kv: -kv[1])[:2])
    out["roof"] = roof(out["dram_pct"], out["sm_pct"], out["l2_pct"])
    out["bound"] = bound_of(out["dram_pct"], out["sm_pct"], out["l2_pct"])
    return out


def roof(dram, sm, l2):
    cand = [(v, n) for v, n in ((dram, "DRAM"), (sm, "compute"), (l2, "L2")) if v is not None]
    if not cand:
        return "?"
    top, name = max(cand)
    if top >= 60:
        return f"{name} roof {top:.0f}%"
    if top >= 30:
        return f"mid ({name} {top:.0f}%)"
    return f"below ({name} {top:.0f}%)"


def bound_of(dram, sm, l2):
    """One word: compute, memory (DRAM or L2) or latency (every roof under 30 %)."""
    cand = [(v, n) for v, n in ((dram, "memory"), (sm, "compute"), (l2, "memory")) if v is not None]
    if not cand:
        return "?"
    top, name = max(cand)
    return name if top >= 30 else "latency"


def read_stages(path):
    st = {}
    if path and os.path.exists(path):
        with open(path) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row.get("id", "").isdigit():
                    st[int(row["id"])] = row.get("stage", "?")
    return st


LAUNCH_COLS = ["workload", "pass", "family", "stage", "id", "kernel", "shape", "threads", "regs", "waves",
               "dur_us", "sm_ghz", "dram_pct", "sm_pct", "l2_pct", "mem_pct", "l1_pct", "l2_hit", "occ",
               "occ_theo", "occ_limit", "issue_pct", "ipc", "dram_bytes", "bytes_src", "elem", "elems",
               "b_per_elem", "roof", "bound", "stalls", "pipes", "pipe_cycles"]


def cell(v):
    return fmt(v, 2) if isinstance(v, float) else ("" if v is None else v)


def cmd_summary(a):
    plan = {r["pass"]: r for r in read_plan(a.plan)} if a.plan else {}
    launches = []
    names = sorted(f[:-len(".details.csv")] for f in os.listdir(a.ncu_dir) if f.endswith(".details.csv"))
    for p in names:
        recs = read_details(os.path.join(a.ncu_dir, p + ".details.csv"))
        stages = read_stages(os.path.join(a.ncu_dir, p + ".stages.tsv"))
        pr = plan.get(p, {})
        for rec in recs:
            row = launch_row(rec)
            row["pass"] = p
            row["workload"] = pr.get("workload", "?")
            row["family"] = family_of(row["kernel"])
            row["stage"] = stages.get(rec["id"], "?")
            row["shape"] = shape(row["grid"], row["block"])
            launches.append(row)
    os.makedirs(a.out, exist_ok=True)
    write_tsv(os.path.join(a.out, "launches.tsv"), [LAUNCH_COLS] + [[cell(r.get(c)) for c in LAUNCH_COLS]
                                                                     for r in launches])
    kern = aggregate(launches, ("workload", "kernel"))
    kstage = aggregate(launches, ("workload", "kernel", "stage"))
    fam = aggregate(launches, ("workload", "stage", "family"))
    md = render(kern, kstage, fam, launches, names)
    with open(os.path.join(a.out, "kernels.md"), "w") as f:
        f.write(md)
    cols = ["workload", "kernel", "family", "stages", "launches", "configs", "dur_us_sum", "dram_pct",
            "sm_pct", "l2_pct", "l2_hit", "occ", "occ_theo", "issue_pct", "waves", "b_per_elem", "elem", "roof", "bound",
            "largest_shape", "largest_dur_us", "largest_dram_pct", "largest_sm_pct", "largest_occ", "passes"]
    write_tsv(os.path.join(a.out, "kernels.tsv"), [cols] + [[cell(k.get(c)) for c in cols] for k in kern])
    scols = ["workload", "kernel", "stage", "launches", "configs", "dur_us_sum", "dram_pct", "sm_pct", "l2_pct",
             "occ", "roof", "bound", "largest_shape"]
    write_tsv(os.path.join(a.out, "kernels_by_stage.tsv"), [scols] + [[cell(k.get(c)) for c in scols]
                                                                      for k in kstage])
    sys.stdout.write(md)
    if not launches:
        sys.stderr.write("summary: no profiled launch in any details CSV\n")
        return 2
    return 0


def aggregate(launches, key):
    groups = {}
    for r in launches:
        groups.setdefault(tuple(r[k] for k in key), []).append(r)
    out = []
    for gk, rs in groups.items():
        d = dict(zip(key, gk))
        d["launches"] = len(rs)
        d["configs"] = len({(r["kernel"], r["shape"]) for r in rs})
        d["dur_us_sum"] = sum(r["dur_us"] or 0.0 for r in rs)
        for c in ("dram_pct", "sm_pct", "l2_pct", "l2_hit", "occ", "occ_theo", "issue_pct", "waves"):
            d[c] = weighted(rs, c)
        byt = [r for r in rs if r.get("dram_bytes") is not None and r.get("elems")]
        d["b_per_elem"] = (sum(r["dram_bytes"] for r in byt) / sum(r["elems"] for r in byt)) if byt else None
        d["elem"] = "/".join(sorted({r["elem"] for r in rs}))
        if "family" not in d:
            d["family"] = rs[0]["family"]
        d["stages"] = "/".join(sorted({r["stage"] for r in rs}))
        d["passes"] = "/".join(sorted({r["pass"] for r in rs}))
        big = max(rs, key=lambda r: ((r["threads"] or 0), (r["dur_us"] or 0.0)))
        d["largest_shape"], d["largest_dur_us"] = big["shape"], big["dur_us"]
        d["largest_dram_pct"], d["largest_sm_pct"], d["largest_occ"] = big["dram_pct"], big["sm_pct"], big["occ"]
        d["largest_threads"] = big["threads"]
        d["roof"] = roof(d["dram_pct"], d["sm_pct"], d["l2_pct"])
        d["bound"] = bound_of(d["dram_pct"], d["sm_pct"], d["l2_pct"])
        out.append(d)
    out.sort(key=lambda d: (d.get("workload", ""), -d["dur_us_sum"]))
    return out


def render(kern, kstage, fam, launches, passes):
    o = io.StringIO()
    o.write("# Nsight Compute summary (noepoch_counters.sh, run B)\n\n")
    o.write(f"{len(launches)} profiled launches from {len(passes)} pass export(s). Duration is ncu's, at the "
            "clock ncu held (the sm_ghz column of launches.tsv); percentages are of peak at that clock. "
            "Values per kernel are duration-weighted over its profiled launches (one per launch shape in a "
            "config pass, the first N in a window pass). B/elem = DRAM bytes (read + write) per element (the "
            "elem column). roof: the highest of DRAM/compute/L2 % of peak, a roof at >= 60 %; bound: compute, "
            "memory, or latency when every roof is under 30 %. Σ time adds the profiled launches only: it "
            "ranks nothing, run A's stage tables do.\n\n")
    by = {}
    for k in kern:
        by.setdefault(k["kernel"], {})[k.get("workload", "?")] = k
    o.write("## per kernel, both workloads (the roofs)\n\n")
    o.write("Largest = the profiled launch with the most threads (the biggest instance each workload ran). "
            "waves = waves per SM, duration-weighted (under 1: the launch cannot fill the card).\n\n")
    o.write("| kernel | family | workload | configs | SM % | DRAM % | L2 % | occ % (theo) | waves | bound | largest: shape, "
            "µs, SM %, DRAM %, occ % |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    for name in sorted(by, key=lambda n: -max(v["dur_us_sum"] for v in by[n].values())):
        for wl in WORKLOADS + tuple(sorted(set(by[name]) - set(WORKLOADS))):
            k = by[name].get(wl)
            if k is None:
                continue
            o.write(f"| {name} | {k['family']} | {wl} | {k['configs']} | {fmt(k['sm_pct'])} | {fmt(k['dram_pct'])} | "
                    f"{fmt(k['l2_pct'])} | {fmt(k['occ'])} ({fmt(k['occ_theo'], 0)}) | {fmt(k['waves'], 2)} | {k['bound']} | "
                    f"{k['largest_shape']}, {fmt(k['largest_dur_us'], 0)}, {fmt(k['largest_sm_pct'])}, "
                    f"{fmt(k['largest_dram_pct'])}, {fmt(k['largest_occ'])} |\n")
    o.write("\n")
    for wl in sorted({k.get("workload", "?") for k in kern}):
        o.write(f"## {wl}: one row per kernel\n\n")
        o.write("| kernel | family | stage | launches (configs) | Σ time ms | DRAM % | SM % | L2 % | L2 hit % | "
                "occ % (theo) | issue % | B/elem | roof | largest launch: shape, µs, DRAM %, SM % |\n")
        o.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for k in kern:
            if k.get("workload", "?") != wl:
                continue
            o.write(f"| {k['kernel']} | {k['family']} | {k['stages']} | {k['launches']} ({k['configs']}) | "
                    f"{fmt(k['dur_us_sum'] / 1e3, 2)} | {fmt(k['dram_pct'])} | {fmt(k['sm_pct'])} | {fmt(k['l2_pct'])} | "
                    f"{fmt(k['l2_hit'])} | {fmt(k['occ'])} ({fmt(k['occ_theo'], 0)}) | {fmt(k['issue_pct'])} | "
                    f"{fmt(k['b_per_elem'], 2)} {k['elem']} | {k['roof']} | {k['largest_shape']}, "
                    f"{fmt(k['largest_dur_us'], 0)}, {fmt(k['largest_dram_pct'])}, {fmt(k['largest_sm_pct'])} |\n")
        o.write("\n")
    o.write("## per workload, stage and kernel (stage = the pass's NVTX window: blk_phase_a / blk_phase_b on "
            "the WHIR base, cardhold_multi_prove / cardhold_build_artifacts in the STARK tree)\n\n")
    o.write("| workload | stage | kernel | launches (configs) | SM % | DRAM % | occ % | bound | largest shape |\n"
            "|---|---|---|---|---|---|---|---|---|\n")
    for k in sorted(kstage, key=lambda d: (d["workload"], d["stage"], -d["dur_us_sum"])):
        o.write(f"| {k['workload']} | {k['stage']} | {k['kernel']} | {k['launches']} ({k['configs']}) | "
                f"{fmt(k['sm_pct'])} | {fmt(k['dram_pct'])} | {fmt(k['occ'])} | {k['bound']} | {k['largest_shape']} |\n")
    o.write("\n## per stage and family (duration-weighted over every profiled launch of the family in that stage)\n\n")
    o.write("| workload | stage | family | launches | Σ time ms | DRAM % | SM % | L2 % | occ % | B/elem | roof |\n")
    o.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
    for f in sorted(fam, key=lambda d: (d["workload"], d["stage"], -d["dur_us_sum"])):
        o.write(f"| {f['workload']} | {f['stage']} | {f['family']} | {f['launches']} | {fmt(f['dur_us_sum'] / 1e3, 2)} | "
                f"{fmt(f['dram_pct'])} | {fmt(f['sm_pct'])} | {fmt(f['l2_pct'])} | {fmt(f['occ'])} | "
                f"{fmt(f['b_per_elem'], 2)} {f['elem']} | {f['roof']} |\n")
    return o.getvalue()


# ---------------------------------------------------------------------------------------------
# stages: which stage each profiled launch fell in, from the order of one pass's output lines

PROF_RE = re.compile(r'^==PROF== Profiling "([^"]+)"(?:\s*-\s*(\d+))?:?.*?(?:-\s*(\d+) passes)?\s*$')


def stages_from_lines(lines, nvtx="-"):
    """[(id, kernel, stage, passes)]. The k-th `==PROF== Profiling` line is report ID k-1 (ncu
    numbers results in the order it profiles them). A pass filtered by a process-wide NVTX range
    profiles only the launches inside it, so the range is the stage; an unfiltered pass reads `run`."""
    stage = nvtx if nvtx and nvtx != "-" else "run"
    out, n = [], 0
    for line in lines:
        m = PROF_RE.match(line.rstrip("\n"))
        if m:
            out.append((n, m.group(1), stage, m.group(3) or ""))
            n += 1
    return out


def cmd_stages(a):
    with open(a.log, errors="replace") as f:
        rows = stages_from_lines(f, a.nvtx)
    w = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    w.writerow(["id", "kernel", "stage", "passes"])
    for r in rows:
        w.writerow(r)
    return 0


# ---------------------------------------------------------------------------------------------
# intervals: sorted, disjoint [start, end) lists in trace nanoseconds


def union(ivs):
    out = []
    for s, e in sorted(ivs):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def subtract(a, b):
    """a minus b, both unions."""
    out, j = [], 0
    for s, e in a:
        cur = s
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            if b[k][0] > cur:
                out.append((cur, b[k][0]))
            cur = max(cur, b[k][1])
            k += 1
        if cur < e:
            out.append((cur, e))
    return out


def total(ivs):
    return sum(e - s for s, e in ivs)


def covered(merged, a, b):
    """ns of [a, b) covered by the merged intervals (sorted, disjoint)."""
    tot = 0
    i = bisect.bisect_right(merged, (a, float("inf"))) - 1
    i = max(i, 0)
    while i < len(merged):
        s, e = merged[i]
        if s >= b:
            break
        if e > a:
            tot += min(e, b) - max(s, a)
        i += 1
    return tot


def covered_ivs(merged, ivs):
    return sum(covered(merged, s, e) for s, e in ivs)


class StepFn:
    """A piecewise-constant count over time (how many ranges are open), for time averages."""

    def __init__(self, ranges):
        ev = {}
        for s, e in ranges:
            if e > s:
                ev[s] = ev.get(s, 0) + 1
                ev[e] = ev.get(e, 0) - 1
        self.t, self.v, c = [], [], 0
        for t in sorted(ev):
            c += ev[t]
            self.t.append(t)
            self.v.append(c)

    def integral(self, a, b):
        """∫ count dt over [a, b), in count x ns."""
        if not self.t or b <= a:
            return 0.0
        i = bisect.bisect_right(self.t, a) - 1
        tot, cur = 0.0, a
        while cur < b:
            v = self.v[i] if i >= 0 else 0
            nxt = self.t[i + 1] if i + 1 < len(self.t) else b
            nxt = min(nxt, b)
            tot += v * (nxt - cur)
            cur = nxt
            i += 1
        return tot

    def mean(self, ivs):
        w = total(ivs)
        return sum(self.integral(s, e) for s, e in ivs) / w if w else None


# ---------------------------------------------------------------------------------------------
# the trace: tables, kernels, copies, NVTX ranges, API calls, GPU metrics


def tables(db):
    return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}


def columns(db, table):
    return [r[1] for r in db.execute(f"PRAGMA table_info({table})")]


def session_start_ns(db, tbls):
    if "TARGET_INFO_SESSION_START_TIME" in tbls and "utcEpochNs" in columns(db, "TARGET_INFO_SESSION_START_TIME"):
        r = db.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()
        if r and r[0]:
            return int(r[0])
    return None


def open_ro(path):
    import sqlite3
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def load_kernels(db, tbls):
    """[(start, end, name, api_start or None, globalTid or None)] sorted by start: the launch API
    call joined by correlationId gives the launching thread and the launch time."""
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tbls:
        return []
    kcols = columns(db, "CUPTI_ACTIVITY_KIND_KERNEL")
    name_col = next((c for c in ("shortName", "demangledName", "mangledName") if c in kcols), None)
    name_sql = f"s.value" if name_col and "StringIds" in tbls else "'?'"
    join_s = f"LEFT JOIN StringIds s ON s.id = k.{name_col}" if name_col and "StringIds" in tbls else ""
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tbls:
        q = (f"SELECT k.start, k.end, {name_sql}, r.start, r.globalTid FROM CUPTI_ACTIVITY_KIND_KERNEL k {join_s} "
             f"LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId")
    else:
        q = f"SELECT k.start, k.end, {name_sql}, NULL, NULL FROM CUPTI_ACTIVITY_KIND_KERNEL k {join_s}"
    return sorted((s, e, n or "?", a, t) for s, e, n, a, t in db.execute(q))


def load_copies(db, tbls):
    out = []
    for t in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if t in tbls:
            out += [(s, e) for s, e in db.execute(f"SELECT start, end FROM {t}")]
    return out


def norm_label(lab):
    """An NVTX text to its label: the domain colon and a `[i=3]` instance suffix dropped."""
    lab = (lab or "").strip().lstrip(":")
    return re.sub(r"\[.*\]$", "", lab)


def load_nvtx(db, tbls):
    """[(start, end, label, globalTid, process_wide)] of the closed NVTX ranges; process_wide is a
    start/end range (eventType 60), which sits on no thread's stack."""
    if "NVTX_EVENTS" not in tbls:
        return []
    cols = columns(db, "NVTX_EVENTS")
    txt = "e.text" if "text" in cols else "NULL"
    tid = "e.globalTid" if "globalTid" in cols else "0"
    et = "e.eventType" if "eventType" in cols else "59"
    if "textId" in cols and "StringIds" in tbls:
        q = (f"SELECT e.start, e.end, COALESCE({txt}, s.value), {tid}, {et} FROM NVTX_EVENTS e "
             f"LEFT JOIN StringIds s ON s.id = e.textId WHERE e.end IS NOT NULL AND e.end > e.start")
    else:
        q = (f"SELECT e.start, e.end, {txt}, {tid}, {et} FROM NVTX_EVENTS e "
             f"WHERE e.end IS NOT NULL AND e.end > e.start")
    return [(s, e, norm_label(lab), t, et_ == 60) for s, e, lab, t, et_ in db.execute(q) if lab]


def load_api(db, tbls):
    """[(start, end, name, globalTid)] of every traced CUDA runtime and driver API call."""
    if "CUPTI_ACTIVITY_KIND_RUNTIME" not in tbls or "StringIds" not in tbls:
        return []
    q = ("SELECT r.start, r.end, s.value, r.globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
         "LEFT JOIN StringIds s ON s.id = r.nameId")
    return [(s, e, n or "?", t) for s, e, n, t in db.execute(q)]


def thread_names(db, tbls):
    out = {}
    if "ThreadNames" in tbls and "StringIds" in tbls:
        cols = columns(db, "ThreadNames")
        if "globalTid" in cols and "nameId" in cols:
            for t, n in db.execute("SELECT t.globalTid, s.value FROM ThreadNames t LEFT JOIN StringIds s "
                                   "ON s.id = t.nameId"):
                if n:
                    out[t] = n
    return out


def tid_of(gtid):
    return None if gtid is None else int(gtid) & 0xFFFFFF


# Key GPU-metric series, matched against the names nsys stores (they differ a little between GPU
# generations, so by pattern, first match wins); every series is in the metrics TSV whatever its name.
KEY_METRICS = [
    ("sm_active", re.compile(r"\bSMs? Active\b.*%", re.I)),
    ("sm_issue", re.compile(r"\bSM Issue\b.*%", re.I)),
    ("warps_in_flight", re.compile(r"\bCompute Warps in Flight \[Throughput %\]", re.I)),
    ("dram_read", re.compile(r"\bDRAM Read\b.*%", re.I)),
    ("dram_write", re.compile(r"\bDRAM Write\b.*%", re.I)),
    ("pcie_rx", re.compile(r"\bPCIe (RX|Read)\b.*%", re.I)),
    ("pcie_tx", re.compile(r"\bPCIe (TX|Write)\b.*%", re.I)),
    ("gr_active", re.compile(r"\bGR Active\b.*%", re.I)),
]


class Metrics:
    """Every GPU-metric series, sorted by time, with prefix sums for window means."""

    def __init__(self, db, tbls):
        self.series, self.why = {}, None
        if "GPU_METRICS" not in tbls:
            self.why = "no GPU_METRICS table (no GPU metrics were collected)"
            return
        cols = columns(db, "GPU_METRICS")
        ts = next((c for c in ("timestamp", "start") if c in cols), None)
        if ts is None or "metricId" not in cols or "value" not in cols:
            self.why = f"GPU_METRICS has unrecognised columns {cols}"
            return
        tcol = "typeId" if "typeId" in cols else "0"
        names = {}
        if "TARGET_INFO_GPU_METRICS" in tbls:
            icols = columns(db, "TARGET_INFO_GPU_METRICS")
            if "metricId" in icols and "metricName" in icols:
                tsel = "typeId" if "typeId" in icols else "0"
                for t, m, n in db.execute(f"SELECT DISTINCT {tsel}, metricId, metricName FROM TARGET_INFO_GPU_METRICS"):
                    names[(t, m)] = n
        raw = {}
        for t, m, when, v in db.execute(f"SELECT {tcol}, metricId, {ts}, value FROM GPU_METRICS ORDER BY {ts}"):
            raw.setdefault(names.get((t, m), f"metric {m}"), []).append((when, float(v or 0)))
        for name, pts in raw.items():
            tt = [p[0] for p in pts]
            pre = [0.0]
            for _, v in pts:
                pre.append(pre[-1] + v)
            self.series[name] = (tt, pre)
        if not self.series:
            self.why = "GPU_METRICS is empty"

    def name_of(self, key):
        pat = dict(KEY_METRICS)[key]
        return next((n for n in sorted(self.series) if pat.search(n)), None)

    def mean(self, name, ivs):
        if name not in self.series:
            return None
        tt, pre = self.series[name]
        s = c = 0.0
        for a, b in ivs:
            i, j = bisect.bisect_left(tt, a), bisect.bisect_left(tt, b)
            s += pre[j] - pre[i]
            c += j - i
        return s / c if c else None

    def samples(self, name, ivs):
        if name not in self.series:
            return 0
        tt, _ = self.series[name]
        return sum(bisect.bisect_left(tt, b) - bisect.bisect_left(tt, a) for a, b in ivs)


def read_smi(path, t0_ns):
    """nvidia-smi samples (timestamp, memory.used MiB, utilization.gpu %) -> [(trace ns, mib, util)].
    The timestamp is the local wall clock nvidia-smi prints ("2026/09/30 18:00:00.123"); this
    process runs on the same machine and in the same time zone as the sampler did."""
    out = []
    if not path or not os.path.exists(path) or t0_ns is None:
        return out
    with open(path, errors="replace") as f:
        for line in f:
            p = [x.strip() for x in line.split(",")]
            if len(p) < 3:
                continue
            try:
                ts = datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                out.append((int(ts * 1e9) - t0_ns, float(p[1]), float(p[2])))
            except ValueError:
                continue
    return sorted(out)


def smi_in(smi, ivs):
    """(max MiB, mean util %) over the samples in the intervals."""
    ts = [x[0] for x in smi]
    mx, us = None, []
    for a, b in ivs:
        for x in smi[bisect.bisect_left(ts, a):bisect.bisect_left(ts, b)]:
            mx = x[1] if mx is None or x[1] > mx else mx
            us.append(x[2])
    return mx, (sum(us) / len(us) if us else None)


# ---------------------------------------------------------------------------------------------
# the harness log


SPLIT_RE = re.compile(r"PROVE SPLIT #(\d+)[^:]*: airs (\d+) · rows (\d+) · wall ([0-9.]+)s · "
                      r"t=\[([0-9.]+),([0-9.]+)\] · prepass ([0-9.]+) · main_commit ([0-9.]+) · absorb ([0-9.]+) · "
                      r"fused ([0-9.]+) · other ([0-9.]+)")
RECOMMIT_SUM_RE = re.compile(r"recommit\[Σ\] ([0-9.]+)")
TL_RE = re.compile(r"TABLE TL (\S+) idx=(\d+) (.*?) est=([0-9.]+)GiB claim=([0-9.]+) start=([0-9.]+) end=([0-9.]+)")
HOLD_RE = re.compile(r"CARD HOLD #(\d+) (\S+): waited ([0-9.]+)s · held ([0-9.]+)s · t=\[([0-9.]+),([0-9.]+)\]")
# the whir workload (lfm::whir_block_tests::the_whir_block_tree_on_a_real_block)
W3_BASE_RE = re.compile(r"^W3 BASE: ([0-9.]+)s")
W3_REC_RE = re.compile(r"^W3 RECURSION: ([0-9.]+)s after the base \(tree ([0-9.]+)s\) · whole block ([0-9.]+)s")
W3_VERIFY_RE = re.compile(r"^W3 TREE VERIFY: ([A-Z]+)")
W3_LEVEL_RE = re.compile(r"^W3 LEVEL (\d+): ([0-9.]+)s wall")
PHASES_RE = re.compile(r"^BLOCK PHASES: execute ([0-9.]+) · build ([0-9.]+) · prep ([0-9.]+) · A ([0-9.]+) "
                       r"\(wait ([0-9.]+) upload ([0-9.]+) commit ([0-9.]+) retire ([0-9.]+)\) · B ([0-9.]+) "
                       r"\(argue ([0-9.]+) open ([0-9.]+) tax ([0-9.]+) = upload ([0-9.]+) \+ encode ([0-9.]+)\)")
PHASE_A_END_RE = re.compile(r"phase A ended at ([0-9.]+)s")
# the stark workload (lfm::block_tree_tests::the_block_tree_composes_to_a_top_node)
NOEPOCH_BLOCK_RE = re.compile(r"★★★ NO-EPOCH BLOCK: base ([0-9.]+)s · harvest ([0-9.]+)s · level 0 ([0-9.]+)s "
                              r"\((\d+) leaves\) · interior ([0-9.]+)s \(levels ([^)]*)\) · recursion ([0-9.]+)s · "
                              r"whole ([0-9.]+)s")
STARK_BASE_RE = re.compile(r"^\s*base: (\d+) sub-proofs in ([0-9.]+)s \(execute ([0-9.]+) · build ([0-9.]+) · "
                           r"setup ([0-9.]+) · prove ([0-9.]+)\)")
LFM_PROVE_RE = re.compile(r"(BLOCK L\d+[^:]*?) LFM PROVE: execute ([0-9.]+)s · fill ([0-9.]+)s · "
                          r"multi_prove ([0-9.]+)s")
CHILD_RE = re.compile(r"CHILD VERIFIES beside the timed path: (\d+) accepted")
PEAK_RE = re.compile(r"★★★ WHOLE RUN: host peak ([0-9.]+) GiB")


def read_log(path):
    lg = {"splits": [], "tl": [], "holds": [], "recommit_sum": 0.0, "test_result": None, "packing": False,
          "w3_base": None, "w3_rec": None, "w3_verify": None, "w3_levels": [], "phases": None, "phase_a_end": None,
          "block": None, "stark_base": None, "lfm_proves": [], "child_verifies": None, "peak": None,
          "final_check": False}
    if not path or not os.path.exists(path):
        return lg
    with open(path, errors="replace") as f:
        for line in f:
            m = SPLIT_RE.search(line)
            if m:
                g = m.groups()
                lg["splits"].append({"seq": int(g[0]), "airs": int(g[1]), "rows": int(g[2]), "wall": float(g[3]),
                                     "t0": float(g[4]), "t1": float(g[5]), "prepass": float(g[6]),
                                     "main_commit": float(g[7]), "absorb": float(g[8]), "fused": float(g[9]),
                                     "other": float(g[10])})
                r = RECOMMIT_SUM_RE.search(line)
                if r:
                    lg["recommit_sum"] += float(r.group(1))
            m = TL_RE.search(line)
            if m:
                g = m.groups()
                lg["tl"].append({"phase": g[0], "idx": int(g[1]), "label": g[2], "est_gib": float(g[3]),
                                 "claim": float(g[4]), "start": float(g[5]), "end": float(g[6])})
            m = HOLD_RE.search(line)
            if m:
                lg["holds"].append({"seq": int(m.group(1)), "phase": m.group(2), "waited": float(m.group(3)),
                                    "held": float(m.group(4)), "t0": float(m.group(5)), "t1": float(m.group(6))})
            m = W3_BASE_RE.search(line)
            if m:
                lg["w3_base"] = float(m.group(1))
            m = W3_REC_RE.search(line)
            if m:
                lg["w3_rec"] = {"recursion": float(m.group(1)), "tree": float(m.group(2)), "whole": float(m.group(3))}
            m = W3_VERIFY_RE.search(line)
            if m:
                lg["w3_verify"] = m.group(1)
            m = W3_LEVEL_RE.search(line)
            if m:
                lg["w3_levels"].append((int(m.group(1)), float(m.group(2))))
            m = PHASES_RE.search(line)
            if m:
                k = ("execute", "build", "prep", "a", "a_wait", "a_upload", "a_commit", "a_retire", "b", "b_argue",
                     "b_open", "b_tax", "b_upload", "b_encode")
                lg["phases"] = dict(zip(k, (float(x) for x in m.groups())))
            m = PHASE_A_END_RE.search(line)
            if m:
                lg["phase_a_end"] = float(m.group(1))
            m = NOEPOCH_BLOCK_RE.search(line)
            if m:
                g = m.groups()
                lg["block"] = {"base": float(g[0]), "harvest": float(g[1]), "level0": float(g[2]), "leaves": int(g[3]),
                               "interior": float(g[4]), "levels": g[5], "recursion": float(g[6]), "whole": float(g[7])}
            m = STARK_BASE_RE.search(line)
            if m and lg["stark_base"] is None:
                g = m.groups()
                lg["stark_base"] = {"subs": int(g[0]), "base": float(g[1]), "execute": float(g[2]),
                                    "build": float(g[3]), "setup": float(g[4]), "prove": float(g[5])}
            m = LFM_PROVE_RE.search(line)
            if m:
                lg["lfm_proves"].append({"label": m.group(1), "execute": float(m.group(2)), "fill": float(m.group(3)),
                                         "multi_prove": float(m.group(4))})
            m = CHILD_RE.search(line)
            if m:
                lg["child_verifies"] = int(m.group(1))
            m = PEAK_RE.search(line)
            if m:
                lg["peak"] = float(m.group(1))
            if "BLOCK FINAL CHECK (harness)" in line:
                lg["final_check"] = True
            if line.startswith("test result:"):
                lg["test_result"] = line.strip()
            if "packing admission (LAMBDA_VM_GATE_PACKING=1)" in line:
                lg["packing"] = True
    return lg


def base_line(lg):
    """The run's headline from its log, either workload."""
    parts = []
    if lg["w3_base"] is not None:
        parts.append(f"whir base {lg['w3_base']:.2f} s")
        if lg["w3_rec"]:
            r = lg["w3_rec"]
            parts.append(f"recursion {r['recursion']:.2f} s (tree {r['tree']:.2f}) · whole {r['whole']:.2f} s")
        if lg["phases"]:
            p = lg["phases"]
            parts.append(f"phases: execute {p['execute']:.2f} · build {p['build']:.2f} · A {p['a']:.2f} (upload "
                         f"{p['a_upload']:.2f} commit {p['a_commit']:.2f}) · B {p['b']:.2f} (argue {p['b_argue']:.2f} "
                         f"open {p['b_open']:.2f} tax {p['b_tax']:.2f})")
        parts.append(f"tree verify {lg['w3_verify'] or '-'}")
    if lg["block"]:
        b = lg["block"]
        parts.append(f"stark base {b['base']:.2f} s · harvest {b['harvest']:.2f} · level 0 {b['level0']:.2f} "
                     f"({b['leaves']} leaves) · interior {b['interior']:.2f} (levels {b['levels']}) · recursion "
                     f"{b['recursion']:.2f} · whole {b['whole']:.2f} s")
        if lg["lfm_proves"]:
            mp = [x["multi_prove"] for x in lg["lfm_proves"]]
            parts.append(f"{len(mp)} LFM PROVE lines, multi_prove Σ {sum(mp):.2f} s (mean {sum(mp) / len(mp):.2f})")
        parts.append(f"child verifies {lg['child_verifies'] if lg['child_verifies'] is not None else '-'}")
    if lg["holds"]:
        parts.append(f"{len(lg['holds'])} card holds, held Σ {sum(h['held'] for h in lg['holds']):.2f} s")
    if lg["peak"] is not None:
        parts.append(f"host peak {lg['peak']:.1f} GiB")
    return " · ".join(parts) if parts else "no W3 BASE / NO-EPOCH BLOCK line in the log"


# ---------------------------------------------------------------------------------------------
# the stage windows

STAGES_BY = {
    "whir": ("setup", "phase_a", "prepared", "phase_b", "base_other", "recursion", "verify", "other"),
    "stark": ("setup", "base_prepass", "base_main_commit", "base_fused", "base_other", "harvest", "level0",
              "interior", "verify", "other"),
}
OVERLAP_BY = {
    "whir": (("a_upload", "blk_a_upload"), ("a_commit", "blk_a_commit"), ("a_retire", "blk_a_retire"),
             ("b_upload", "blk_b_upload"), ("b_argue", "blk_b_argue"), ("b_encode", "blk_b_encode"),
             ("b_open", "blk_b_open")),
    "stark": (("precommit", "r1_precommit_table"), ("recommit", "r1_main_recommit_table")),
}
HOLD_PREFIX = "cardhold_"
RECOMMIT_LABEL = "r1_main_recommit_table"
TASK_LABELS = ("r1_main_recommit_table", "r1_aux_build_table", "r1_aux_commit_table", "rounds_2to4_table")
# Thread roles: a push/pop range on a thread names what that thread is (the WHIR block's threads).
ROLE_LABELS = ("blk_executor", "blk_walker", "blk_builder", "blk_layout", "blk_b_upload_next")


def intersect(a, b):
    """a and b, both unions."""
    return subtract(a, subtract(a, b))


def trace_windows(nvtx, end_ns, workload):
    """({stage: intervals} over the workload's disjoint stages, its overlapping rows, one row per
    card-hold phase (`hold:<phase>`) and `whole`; source note)."""
    by = {}
    for s, e, lab, *_ in nvtx:
        by.setdefault(lab, []).append((s, e))
    u = {lab: union(v) for lab, v in by.items()}
    whole = [(0, end_ns)]
    base = u.get("blk_base", [])
    if not base:
        return {"whole": whole}, "no blk_base NVTX range (no libnvToolsExt, or another build): one window only"
    claims = []
    if workload == "whir":
        claims = [("phase_a", u.get("blk_phase_a", [])), ("prepared", u.get("blk_prepared", [])),
                  ("phase_b", u.get("blk_phase_b", [])), ("base_other", base)]
    else:
        pre, mc, fu = (intersect(u.get(k, []), base) for k in ("r1_prepass", "r1_main_commit", "rounds_2to4"))
        claims = [("base_prepass", pre), ("base_main_commit", mc), ("base_fused", fu), ("base_other", base),
                  ("harvest", u.get("tree_harvest", [])), ("level0", u.get("tree_level0", [])),
                  ("interior", u.get("tree_interior", []))]
    if workload == "whir":
        claims.append(("recursion", u.get("lfm_tree", [])))
    ver = u.get("harness_verify", [])
    if ver:
        claims.append(("verify", [(ver[0][0], max(end_ns, ver[-1][1]))]))
    claims.append(("setup", [(0, base[0][0])]))
    w, taken = {}, []
    for name, ivs in claims:
        own = subtract(intersect(union(ivs), whole), union(taken))
        if own:
            w[name] = own
            taken = union(taken + own)
    rest = subtract(whole, taken)
    if rest:
        w["other"] = rest
    for name, lab in OVERLAP_BY[workload]:
        if u.get(lab):
            w[name] = u[lab]
    for lab in sorted(u):
        if lab.startswith(HOLD_PREFIX):
            w["hold:" + lab[len(HOLD_PREFIX):]] = u[lab]
    w["whole"] = whole
    return w, "NVTX ranges"


def partition(w, workload):
    """Sorted disjoint (start, end, stage) segments of the workload's disjoint stages."""
    names = STAGES_BY.get(workload, ())
    return sorted((s, e, st) for st in names for s, e in w.get(st, []))


def stage_at(segs, starts, t):
    i = bisect.bisect_right(starts, t) - 1
    if i >= 0 and segs[i][0] <= t < segs[i][1]:
        return segs[i][2]
    return "other"


# ---------------------------------------------------------------------------------------------
# CUDA API categories (driver and runtime names; first match wins)

API_CATS = [
    ("sync", r"cu(StreamSynchronize|CtxSynchronize|EventSynchronize)(_v\d+)?(_ptsz)?"
             r"|cuda(StreamSynchronize|DeviceSynchronize|EventSynchronize|ThreadSynchronize)(_v\d+)?(_ptsz)?"),
    ("copy_async", r"cu(Memcpy\w*Async|MemsetD\w*Async|MemPrefetchAsync)(_v\d+)?(_ptsz|_ptds)?"
                   r"|cuda(Memcpy\w*Async|Memset\w*Async|MemPrefetchAsync)(_v\d+)?(_ptsz|_ptds)?"),
    ("copy_sync", r"cu(Memcpy(HtoD|DtoH|DtoD|HtoA|AtoH|AtoD|DtoA|AtoA|2D|2DUnaligned|3D|3DPeer|Peer)?"
                  r"|MemsetD(8|16|32|2D8|2D16|2D32))(_v\d+)?(_ptds)?"
                  r"|cuda(Memcpy(2D|3D|Peer|ToSymbol|FromSymbol)?|Memset(2D|3D)?)(_v\d+)?(_ptds)?"),
    ("alloc", r"cu(MemAlloc(Pitch|Managed|Async|FromPoolAsync)?|MemCreate|MemMap|MemAddressReserve"
              r"|MemPoolCreate)(_v\d+)?(_ptsz)?"
              r"|cuda(Malloc(Async|Managed|Pitch|3D|FromPoolAsync)?)(_v\d+)?(_ptsz)?"),
    ("free", r"cu(MemFree(Async)?|MemRelease|MemUnmap|MemAddressFree|MemPoolDestroy|MemPoolTrimTo)(_v\d+)?(_ptsz)?"
             r"|cuda(Free(Async)?|MemPoolTrimTo)(_v\d+)?(_ptsz)?"),
    ("host_pinned", r"cu(MemHostAlloc|MemAllocHost|MemFreeHost|MemHostRegister|MemHostUnregister)(_v\d+)?"
                    r"|cuda(HostAlloc|MallocHost|FreeHost|HostRegister|HostUnregister)(_v\d+)?"),
    ("launch", r"cu(LaunchKernel(Ex)?|LaunchCooperativeKernel)(_ptsz)?"
               r"|cuda(LaunchKernel(ExC)?|LaunchCooperativeKernel)(_v\d+)?(_ptsz)?"),
    ("event_stream", r"cu(Event\w+|Stream\w+)(_v\d+)?(_ptsz)?|cuda(Event\w+|Stream\w+)(_v\d+)?(_ptsz)?"),
    ("module", r"cu(Module\w+|Library\w+|Func\w+|OccupancyMax\w+|Kernel\w+)(_v\d+)?|cuda(Func\w+|Occupancy\w+)"),
    ("context", r"cu(Init|Ctx\w+|DevicePrimaryCtx\w+|Device\w+|MemGetInfo|DriverGetVersion|GetExportTable|"
                r"GetProcAddress|PointerGetAttributes?)(_v\d+)?|cuda(SetDevice|GetDevice\w*|MemGetInfo|DeviceGet\w+)"),
]
API_CAT_RE = [(c, re.compile(p)) for c, p in API_CATS]
API_NOTE = ("sync = stream/context/event synchronize; copy_async = the *Async copies and memsets, which still "
            "block the calling thread when the host side is pageable (the driver stages through its own pinned "
            "buffer); copy_sync = the synchronous copies and memsets; alloc/free = device memory (cuMemAlloc*, "
            "cuMemFree*, pools, VMM); host_pinned = page-locked host memory (cuMemHostAlloc, cuMemAllocHost, "
            "cuMemHostRegister and their frees).")


def api_cat(name):
    for c, rx in API_CAT_RE:
        if rx.fullmatch(name):
            return c
    return "other"


# ---------------------------------------------------------------------------------------------
# the host's CPU: the sampler's files (cpu-sample) read back on the trace's clock


def read_cpu(prefix, t0_ns):
    """{"clk", "proc": [(t, ticks, rss_kb, threads, avail_mib)], "tid": {tid: [(t, ticks)]}, "comm": {tid: comm},
    "watchdog"} with t in trace nanoseconds (the sampler stamps the wall clock, as nsys's session start is);
    None without the files."""
    if not prefix or t0_ns is None or not os.path.exists(prefix + ".proc.tsv"):
        return None
    clk, proc, tid, comm = 100, [], {}, {}
    with open(prefix + ".proc.tsv", errors="replace") as f:
        for line in f:
            if line.startswith("# clk_tck "):
                clk = int(line.split()[2])
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 6 or not p[0].isdigit():
                continue
            proc.append((int(p[0]) - t0_ns, int(p[2]), int(p[3]), int(p[4]), int(p[5])))
    if os.path.exists(prefix + ".tid.tsv"):
        with open(prefix + ".tid.tsv", errors="replace") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 4 or not p[0].isdigit() or not p[1].isdigit():
                    continue
                t = int(p[1])
                comm[t] = p[2]
                tid.setdefault(t, []).append((int(p[0]) - t0_ns, int(p[3])))
    wd = None
    if os.path.exists(prefix + ".watchdog"):
        wd = open(prefix + ".watchdog", errors="replace").read().strip()
    return {"clk": clk, "proc": sorted(proc), "tid": tid, "comm": comm, "watchdog": wd}


def cpu_deltas(series):
    """[(t_mid, dt_ns, dticks)] between consecutive cumulative samples."""
    out = []
    for (ta, ka, *_), (tb, kb, *_) in zip(series, series[1:]):
        if tb > ta and kb >= ka:
            out.append(((ta + tb) // 2, tb - ta, kb - ka))
    return out


def comm_family(comm):
    return re.sub(r"[-_ ]?\d+$", "", comm or "?") or "?"


def cpu_by_stage(cpu, segs, starts):
    """({stage: (cpu_s, covered_s)}, {(role, stage): cpu_s}) from the process and per-thread series."""
    st, roles = {}, {}
    if not cpu:
        return st, roles
    for t, dt, dk in cpu_deltas(cpu["proc"]):
        name = stage_at(segs, starts, t) if segs else "whole"
        d = st.setdefault(name, [0.0, 0.0])
        d[0] += dk / cpu["clk"]
        d[1] += dt / 1e9
    for tid, series in cpu["tid"].items():
        role = cpu.get("roles", {}).get(tid) or comm_family(cpu["comm"].get(tid))
        for t, _dt, dk in cpu_deltas(series):
            name = stage_at(segs, starts, t) if segs else "whole"
            roles[(role, name)] = roles.get((role, name), 0.0) + dk / cpu["clk"]
    return st, roles


# ---------------------------------------------------------------------------------------------
# runa: run A's tables from an nsys trace


def cmd_runa(a):
    db = open_ro(a.sqlite)
    tbls = tables(db)
    t0 = session_start_ns(db, tbls)
    kernels = load_kernels(db, tbls)
    copies = load_copies(db, tbls)
    nvtx = load_nvtx(db, tbls)
    api = load_api(db, tbls)
    tnames = thread_names(db, tbls)
    mets = Metrics(db, tbls)
    lg = read_log(a.log)
    smi = read_smi(a.smi, t0)
    cpu = read_cpu(a.cpu, t0)
    wl = a.workload
    end = max([e for _, e, *_ in kernels] + [e for _, e in copies] + [e for _, e, *_ in api] + [0])
    w, src = trace_windows(nvtx, end, wl)
    segs = partition(w, wl)
    sstarts = [x[0] for x in segs]
    kmerged = union([(s, e) for s, e, *_ in kernels])
    cmerged = union(copies)
    amerged = union([(s, e) for s, e, *_ in kernels] + copies)
    kstarts = [k[0] for k in kernels]
    maxdur = max([e - s for s, e, *_ in kernels] + [0])
    tasks = StepFn([(s, e) for s, e, lab, *_ in nvtx if lab in TASK_LABELS])
    holdsfn = StepFn([(s, e) for s, e, lab, *_ in nvtx if lab.startswith(HOLD_PREFIX)])
    keys = {k: mets.name_of(k) for k, _ in KEY_METRICS}
    if cpu is not None:
        cpu["roles"] = {}
        for _s, _e, lab, gt, pw in nvtx:
            if not pw and lab in ROLE_LABELS and gt is not None:
                cpu["roles"].setdefault(tid_of(gt), lab[len("blk_"):])
    cpu_st, cpu_roles = cpu_by_stage(cpu, segs, sstarts)
    os.makedirs(a.out, exist_ok=True)

    def kernels_in(ivs):
        ksum, nl, kk = 0.0, 0, {}
        for s0, e0 in ivs:
            lo = bisect.bisect_left(kstarts, s0 - maxdur)
            hi = bisect.bisect_left(kstarts, e0)
            for s, e, k, _a, _t in kernels[lo:hi]:
                ov = min(e, e0) - max(s, s0)
                if ov > 0:
                    ksum += ov / 1e9
                    d = kk.setdefault(k, [0, 0.0])
                    d[1] += ov / 1e9
                    if s0 <= s < e0:
                        d[0] += 1
                        nl += 1
        return ksum, nl, kk

    def mmean(key, ivs):
        return mets.mean(keys[key], ivs) if keys[key] else None

    disjoint = [x for x in STAGES_BY[wl] if x in w]
    overlap = [n for n, _ in OVERLAP_BY[wl] if n in w] + sorted(n for n in w if n.startswith("hold:"))
    order = disjoint + overlap + ["whole"] if "whole" in w else disjoint + overlap
    stages, per_kernel, per_family = [], {}, {}
    for name in order:
        ivs = w[name]
        wall = total(ivs) / 1e9
        ksum, nl, kk = kernels_in(ivs)
        per_kernel[name] = kk
        fam = {}
        for k, (_n, ks) in kk.items():
            fam[family_of(k)] = fam.get(family_of(k), 0.0) + ks
        per_family[name] = fam
        vram, util = smi_in(smi, ivs)
        row = {"stage": name, "wall_s": wall, "ranges": len(ivs),
               "busy_pct": 100.0 * covered_ivs(amerged, ivs) / max(total(ivs), 1),
               "kernel_busy_pct": 100.0 * covered_ivs(kmerged, ivs) / max(total(ivs), 1),
               "copy_busy_pct": 100.0 * covered_ivs(cmerged, ivs) / max(total(ivs), 1),
               "kernel_sum_s": ksum, "launches": nl, "tasks_open": tasks.mean(ivs), "holds_open": holdsfn.mean(ivs),
               "vram_max_mib": vram, "smi_util_pct": util}
        c = cpu_st.get(name)
        row["cpu_cores"] = (c[0] / c[1]) if c and c[1] > 0 else None
        row["cpu_s"] = c[0] if c else None
        for key, _ in KEY_METRICS:
            row[key] = mmean(key, ivs)
        top = sorted(kk.items(), key=lambda kv: -kv[1][1])[:5]
        row["top"] = " ; ".join(f"{k} {v[1]:.2f}s ({100 * v[1] / ksum:.0f}%)" for k, v in top) if ksum else ""
        stages.append(row)
    cols = ["stage", "wall_s", "ranges", "busy_pct", "kernel_busy_pct", "copy_busy_pct", "kernel_sum_s", "launches",
            "tasks_open", "holds_open", "vram_max_mib", "smi_util_pct", "cpu_cores", "cpu_s"] + \
        [k for k, _ in KEY_METRICS] + ["top"]

    def scell(c, v):
        return f"{v:.4f}" if c.endswith("_s") and isinstance(v, float) else cell(v)

    write_tsv(os.path.join(a.out, f"runa-{wl}-stages.tsv"), [cols] + [[scell(c, r.get(c)) for c in cols]
                                                                     for r in stages])
    rows = [["stage", "kernel", "family", "launches", "sum_s", "share_of_stage_kernel_s"]]
    for st in stages:
        for k, (n, sm) in sorted(per_kernel.get(st["stage"], {}).items(), key=lambda kv: -kv[1][1]):
            if sm >= 0.001:
                rows.append([st["stage"], k, family_of(k), n, f"{sm:.4f}",
                             f"{100 * sm / st['kernel_sum_s']:.2f}" if st["kernel_sum_s"] else ""])
    write_tsv(os.path.join(a.out, f"runa-{wl}-stage-kernels.tsv"), rows)
    fams = sorted({f for d in per_family.values() for f in d}, key=lambda f: -sum(d.get(f, 0.0) for n, d in
                                                                                  per_family.items() if n in disjoint))
    write_tsv(os.path.join(a.out, f"runa-{wl}-stage-families.tsv"),
              [["stage", "kernel_sum_s"] + [f"{f}_s" for f in fams]] +
              [[st["stage"], f"{st['kernel_sum_s']:.4f}"] + [f"{per_family[st['stage']].get(f, 0.0):.4f}" for f in fams]
               for st in stages])
    if mets.series:
        mrows = [["stage", "metric", "mean", "samples"]]
        for st in order:
            for n in sorted(mets.series):
                v = mets.mean(n, w[st])
                mrows.append([st, n, "" if v is None else f"{v:.3f}", mets.samples(n, w[st])])
        write_tsv(os.path.join(a.out, f"runa-{wl}-gpu-metrics.tsv"), mrows)

    # every card hold: one LFM proof's device phase (the permit admits one holder at a time)
    holds = sorted((s, e, lab[len(HOLD_PREFIX):]) for s, e, lab, *_ in nvtx if lab.startswith(HOLD_PREFIX))
    hrows = [["idx", "phase", "stage", "start_s", "wall_s", "kernel_busy_pct", "kernel_s", "launches", "sm_active",
              "warps_in_flight", "dram_read", "top_kernels"]]
    hsum = {}
    for i, (s0, e0, ph) in enumerate(holds):
        ksum, nl, kk = kernels_in([(s0, e0)])
        busy = 100.0 * covered(kmerged, s0, e0) / max(e0 - s0, 1)
        stg = stage_at(segs, sstarts, s0) if segs else "whole"
        top = sorted(kk.items(), key=lambda kv: -kv[1][1])[:3]
        hrows.append([i, ph, stg, f"{s0 / 1e9:.3f}", f"{(e0 - s0) / 1e9:.4f}", fmt(busy), f"{ksum:.4f}", nl,
                      fmt(mmean("sm_active", [(s0, e0)])), fmt(mmean("warps_in_flight", [(s0, e0)])),
                      fmt(mmean("dram_read", [(s0, e0)])),
                      " ; ".join(f"{k} {v[1] * 1e3:.1f}ms" for k, v in top)])
        d = hsum.setdefault((ph, stg), {"n": 0, "wall": 0.0, "busy_ns": 0.0, "k": 0.0, "busy": [], "kk": {}})
        d["n"] += 1
        d["wall"] += (e0 - s0) / 1e9
        d["busy_ns"] += covered(kmerged, s0, e0)
        d["k"] += ksum
        d["busy"].append(busy)
        for k, (n, sm) in kk.items():
            x = d["kk"].setdefault(k, [0, 0.0])
            x[0] += n
            x[1] += sm
    write_tsv(os.path.join(a.out, f"runa-{wl}-holds.tsv"), hrows)
    hkrows = [["phase", "stage", "kernel", "family", "launches", "kernel_s", "share_of_held_kernel_s"]]
    for (ph, stg), d in sorted(hsum.items()):
        for k, (n, sm) in sorted(d["kk"].items(), key=lambda kv: -kv[1][1]):
            if sm >= 0.0005:
                hkrows.append([ph, stg, k, family_of(k), n, f"{sm:.4f}", f"{100 * sm / d['k']:.2f}" if d["k"] else ""])
    write_tsv(os.path.join(a.out, f"runa-{wl}-hold-kernels.tsv"), hkrows)

    # the kernels each NVTX label launched: push/pop = the launch call on the label's thread inside its
    # range; process-wide = any launch call inside its window
    by_lab, pw_lab = {}, {}
    for s, e, lab, t, pw in nvtx:
        if pw:
            pw_lab.setdefault(lab, []).append((s, e))
        else:
            by_lab.setdefault(lab, {}).setdefault(t, []).append((s, e))
    for lab in by_lab:
        for t in by_lab[lab]:
            by_lab[lab][t].sort()
    attr = {}
    for lab, per_t in by_lab.items():
        d = {"kind": "thread", "ranges": sum(len(v) for v in per_t.values()),
             "range_s": sum(total(union(v)) for v in per_t.values()) / 1e9,
             "threads": len(per_t), "launches": 0, "kernel_s": 0.0, "k": {}}
        starts = {t: [r[0] for r in v] for t, v in per_t.items()}
        for s, e, k, api_s, gt in kernels:
            if api_s is None or gt not in per_t:
                continue
            i = bisect.bisect_right(starts[gt], api_s) - 1
            if i >= 0 and per_t[gt][i][0] <= api_s < per_t[gt][i][1]:
                d["launches"] += 1
                d["kernel_s"] += (e - s) / 1e9
                kd = d["k"].setdefault(k, [0, 0.0])
                kd[0] += 1
                kd[1] += (e - s) / 1e9
        attr[lab] = d
    for lab, ivs in pw_lab.items():
        u = union(ivs)
        us = [x[0] for x in u]
        d = {"kind": "process", "ranges": len(ivs), "range_s": total(u) / 1e9, "threads": "-", "launches": 0,
             "kernel_s": 0.0, "k": {}}
        for s, e, k, api_s, _gt in kernels:
            if api_s is None:
                continue
            i = bisect.bisect_right(us, api_s) - 1
            if i >= 0 and u[i][0] <= api_s < u[i][1]:
                d["launches"] += 1
                d["kernel_s"] += (e - s) / 1e9
                kd = d["k"].setdefault(k, [0, 0.0])
                kd[0] += 1
                kd[1] += (e - s) / 1e9
        attr[lab + ("" if lab not in attr else " (process)")] = d
    arows = [["label", "kind", "ranges", "threads", "range_s", "launches", "kernel_s", "top_kernels"]]
    krows = [["label", "kernel", "launches", "kernel_s"]]
    for lab, d in sorted(attr.items(), key=lambda kv: -kv[1]["kernel_s"]):
        top = sorted(d["k"].items(), key=lambda kv: -kv[1][1])
        arows.append([lab, d["kind"], d["ranges"], d["threads"], f"{d['range_s']:.3f}", d["launches"],
                      f"{d['kernel_s']:.3f}", " ; ".join(f"{k} {v[1]:.2f}s" for k, v in top[:5])])
        for k, (n, sm) in top:
            krows.append([lab, k, n, f"{sm:.4f}"])
    write_tsv(os.path.join(a.out, f"runa-{wl}-nvtx-kernels.tsv"), arows)
    write_tsv(os.path.join(a.out, f"runa-{wl}-nvtx-label-kernels.tsv"), krows)

    # CUDA API time per category, thread and stage (by the call's start)
    cat_st, thr, names = {}, {}, {}
    for s, e, n, gt in api:
        c = api_cat(n)
        st = stage_at(segs, sstarts, s) if segs else "whole"
        dur = (e - s) / 1e9
        d = cat_st.setdefault((c, st), [0, 0.0, 0.0])
        d[0] += 1
        d[1] += dur
        d[2] = max(d[2], dur)
        t = thr.setdefault((gt, c), {"calls": 0, "s": 0.0, "max": 0.0, "st": {}})
        t["calls"] += 1
        t["s"] += dur
        t["max"] = max(t["max"], dur)
        t["st"][st] = t["st"].get(st, 0.0) + dur
        nd = names.setdefault(n, {"cat": c, "calls": 0, "s": 0.0, "max": 0.0, "st": {}})
        nd["calls"] += 1
        nd["s"] += dur
        nd["max"] = max(nd["max"], dur)
        nd["st"][st] = nd["st"].get(st, 0.0) + dur
    stl = disjoint or ["whole"]
    cats = [c for c, _ in API_CATS] + ["other"]
    write_tsv(os.path.join(a.out, f"api-{wl}-by-category.tsv"),
              [["category", "stage", "calls", "sum_s", "max_ms"]] +
              [[c, st, v[0], f"{v[1]:.4f}", f"{v[2] * 1e3:.3f}"] for (c, st), v in
               sorted(cat_st.items(), key=lambda kv: (cats.index(kv[0][0]), stl.index(kv[0][1])
                                                      if kv[0][1] in stl else 99))])

    def tlabel(gt):
        return f"tid {tid_of(gt)}" + (f" ({tnames[gt]})" if gt in tnames else "")

    write_tsv(os.path.join(a.out, f"api-{wl}-by-thread.tsv"),
              [["thread", "category", "calls", "sum_s", "max_ms"] + [f"{s}_s" for s in stl]] +
              [[tlabel(gt), c, v["calls"], f"{v['s']:.4f}", f"{v['max'] * 1e3:.3f}"] +
               [f"{v['st'].get(s, 0.0):.4f}" for s in stl]
               for (gt, c), v in sorted(thr.items(), key=lambda kv: -kv[1]["s"])])
    write_tsv(os.path.join(a.out, f"api-{wl}-by-call.tsv"),
              [["call", "category", "calls", "sum_s", "max_ms"] + [f"{s}_s" for s in stl]] +
              [[n, v["cat"], v["calls"], f"{v['s']:.4f}", f"{v['max'] * 1e3:.3f}"] +
               [f"{v['st'].get(s, 0.0):.4f}" for s in stl]
               for n, v in sorted(names.items(), key=lambda kv: -kv[1]["s"])])

    # the host's CPU per stage and thread role
    if cpu is not None:
        role_names = sorted({r for r, _ in cpu_roles}, key=lambda r: -sum(v for (rr, _), v in cpu_roles.items()
                                                                          if rr == r))
        write_tsv(os.path.join(a.out, f"cpu-{wl}-by-stage.tsv"),
                  [["stage", "wall_s", "cpu_s", "cores_busy"] + [f"{r}_s" for r in role_names]] +
                  [[st, f"{total(w[st]) / 1e9:.3f}", f"{cpu_st.get(st, [0.0, 0.0])[0]:.3f}",
                    fmt(cpu_st[st][0] / cpu_st[st][1], 2) if st in cpu_st and cpu_st[st][1] else "-"] +
                   [f"{cpu_roles.get((r, st), 0.0):.3f}" for r in role_names] for st in disjoint])

    # the time series
    binn = max(int(a.bin_ms * 1e6), 1_000_000)
    ts_cols = ["t_s", "stage", "kernel_busy_pct", "copy_busy_pct", "tasks_open", "holds_open", "vram_mib",
               "smi_util_pct", "cpu_cores"] + [k for k, _ in KEY_METRICS]
    ts_rows = [ts_cols]
    smi_t = [x[0] for x in smi]
    cdel = cpu_deltas(cpu["proc"]) if cpu else []
    ct = [x[0] for x in cdel]
    for b0 in range(0, int(end) + 1, binn):
        b1 = b0 + binn
        iv = [(b0, b1)]
        cnt = {}
        for s, e, st in segs[max(bisect.bisect_right(sstarts, b0) - 1, 0):bisect.bisect_left(sstarts, b1)]:
            ov = min(e, b1) - max(s, b0)
            if ov > 0:
                cnt[st] = cnt.get(st, 0) + ov
        st = max(cnt, key=cnt.get) if cnt else "-"
        j = bisect.bisect_left(smi_t, b1) - 1
        vr = smi[j][1] if j >= 0 and smi else None
        ut = smi[j][2] if j >= 0 and smi else None
        cs = cdel[bisect.bisect_left(ct, b0):bisect.bisect_left(ct, b1)]
        cores = (sum(x[2] for x in cs) / cpu["clk"]) / (sum(x[1] for x in cs) / 1e9) if cs and sum(x[1] for x in cs) \
            else None
        r = [f"{b0 / 1e9:.3f}", st, fmt(100.0 * covered(kmerged, b0, b1) / binn), fmt(100.0 * covered(cmerged, b0, b1) / binn),
             fmt(tasks.mean(iv), 2), fmt(holdsfn.mean(iv), 2), fmt(vr, 0), fmt(ut, 0), fmt(cores, 2)]
        r += [fmt(mets.mean(keys[k], iv)) if keys[k] else "-" for k, _ in KEY_METRICS]
        ts_rows.append(r)
    write_tsv(os.path.join(a.out, f"runa-{wl}-timeseries.tsv"), ts_rows)

    # the markdown
    o = io.StringIO()
    o.write(f"# Run A, {wl}: the workload under Nsight Systems\n\n")
    o.write(f"{base_line(lg)}.\n\n")
    o.write(f"trace: {len(kernels)} kernel launches, {len(copies)} copies/memsets, {len(api)} CUDA API calls, "
            f"{len(nvtx)} NVTX ranges, {end / 1e9:.1f} s from the session start; {len(holds)} card holds in the trace, "
            f"{len(lg['holds'])} CARD HOLD lines in the log. Stages from {src}. CPU: "
            f"{'sampled' if cpu else 'no sampler file'}"
            f"{' · WATCHDOG: ' + cpu['watchdog'] if cpu and cpu.get('watchdog') else ''}.\n\n")
    o.write("busy % = any kernel or copy on the card; kernel/copy busy % = any kernel / any copy or memset. holds = "
            "the mean number of card holds open (0 or 1: the permit admits one). cores = the profiled process's CPU "
            "seconds per wall second (the sampler's 100 ms ticks). VRAM = the nvidia-smi maximum in the stage (200 ms "
            f"samples). GPU metrics are nsys's samples averaged over the stage ({mets.why or 'collected'}): warps = "
            "Compute Warps in Flight, % of the card's warp slots (an occupancy proxy over time). The rows after the "
            "disjoint stages overlap them (one range per group, or per hold); `whole` is the run.\n\n")
    o.write("| stage | wall s | busy % | kernel busy % | copy busy % | Σ kernel s | launches | holds | cores | "
            "VRAM MiB | SM active % | SM issue % | warps % | DRAM rd % | DRAM wr % | PCIe rx % | PCIe tx % | "
            "top kernels (Σ s in the stage) |\n")
    o.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in stages:
        o.write(f"| {r['stage']} | {fmt(r['wall_s'], 2)} | {fmt(r['busy_pct'])} | {fmt(r['kernel_busy_pct'])} | "
                f"{fmt(r['copy_busy_pct'])} | {fmt(r['kernel_sum_s'], 2)} | {r['launches']} | {fmt(r['holds_open'], 2)} | "
                f"{fmt(r['cpu_cores'], 1)} | {fmt(r['vram_max_mib'], 0)} | {fmt(r['sm_active'])} | "
                f"{fmt(r['sm_issue'])} | {fmt(r['warps_in_flight'])} | {fmt(r['dram_read'])} | {fmt(r['dram_write'])} | "
                f"{fmt(r['pcie_rx'])} | {fmt(r['pcie_tx'])} | {r['top']} |\n")
    if fams:
        o.write("\n## kernel seconds by family and stage\n\n")
        o.write("| stage | Σ kernel s | " + " | ".join(fams) + " |\n|---|---|" + "---|" * len(fams) + "\n")
        for st in stages:
            o.write(f"| {st['stage']} | {st['kernel_sum_s']:.2f} | " +
                    " | ".join(f"{per_family[st['stage']].get(f, 0.0):.2f}" for f in fams) + " |\n")
    if hsum:
        o.write("\n## card holds (each an LFM proof's device phase; one holder at a time)\n\n")
        o.write("busy % = kernel time covered / held time; idle s = held time with no kernel on the card. Per hold "
                f"in runa-{wl}-holds.tsv, its kernels in runa-{wl}-hold-kernels.tsv.\n\n")
        o.write("| phase | stage | holds | held s | Σ kernel s | busy % | idle s | busy % min / median / max | "
                "top kernels (Σ s) |\n|---|---|---|---|---|---|---|---|---|\n")
        for (ph, stg), d in sorted(hsum.items(), key=lambda kv: -kv[1]["wall"]):
            b = sorted(d["busy"])
            top = sorted(d["kk"].items(), key=lambda kv: -kv[1][1])[:4]
            o.write(f"| {ph} | {stg} | {d['n']} | {d['wall']:.2f} | {d['k']:.2f} | "
                    f"{100 * d['busy_ns'] / 1e9 / d['wall'] if d['wall'] else 0:.1f} | "
                    f"{d['wall'] - d['busy_ns'] / 1e9:.2f} | {b[0]:.0f} / {b[len(b) // 2]:.0f} / {b[-1]:.0f} | "
                    + " ; ".join(f"{k} {v[1]:.2f}" for k, v in top) + " |\n")
    if cpu is not None and cpu_st:
        o.write("\n## the host's CPU per stage (cores busy = CPU seconds / wall second) and by thread role\n\n")
        top_roles = sorted({r for r, _ in cpu_roles}, key=lambda r: -sum(v for (rr, _), v in cpu_roles.items()
                                                                         if rr == r))[:8]
        o.write("A role is the thread's NVTX range (executor, walker, builder, layout on the WHIR block) or its "
                "name less a trailing number; an unnamed thread carries its creator's name.\n\n")
        o.write("| stage | wall s | CPU s | cores | " + " | ".join(top_roles) + " |\n|---|---|---|---|" +
                "---|" * len(top_roles) + "\n")
        for st in disjoint:
            c = cpu_st.get(st)
            o.write(f"| {st} | {total(w[st]) / 1e9:.2f} | {c[0] if c else 0:.2f} | "
                    f"{fmt(c[0] / c[1], 2) if c and c[1] else '-'} | " +
                    " | ".join(f"{cpu_roles.get((r, st), 0.0):.2f}" for r in top_roles) + " |\n")
    if attr:
        o.write("\n## kernels by the NVTX label that launched them\n\n")
        o.write("thread: a kernel belongs to a push/pop label when its launch call ran on the label's thread inside "
                "one of its ranges. process: a launch call inside a process-wide range's window, from any thread.\n\n")
        o.write("| label | kind | ranges | threads | range s | launches | Σ kernel s | top kernels |\n"
                "|---|---|---|---|---|---|---|---|\n")
        for r in arows[1:]:
            o.write("| " + " | ".join(str(x) for x in r) + " |\n")
    if api:
        o.write("\n## CUDA API time (calls x duration on the calling thread, by the call's start)\n\n")
        o.write(API_NOTE + "\n\n")
        o.write("| category | " + " | ".join(stl) + " | total s | calls | max ms |\n|---|" + "---|" * (len(stl) + 3) + "\n")
        for c in cats:
            vals = [cat_st.get((c, s), [0, 0.0, 0.0]) for s in stl]
            tot_s = sum(v[1] for v in vals)
            if not tot_s:
                continue
            o.write(f"| {c} | " + " | ".join(f"{v[1]:.2f}" for v in vals) +
                    f" | {tot_s:.2f} | {sum(v[0] for v in vals)} | {max(v[2] for v in vals) * 1e3:.1f} |\n")
        o.write("\ntop threads (thread, category), by API seconds:\n\n")
        o.write("| thread | category | calls | total s | max ms | " + " | ".join(stl) + " |\n|---|---|---|---|---|" +
                "---|" * len(stl) + "\n")
        for (gt, c), v in sorted(thr.items(), key=lambda kv: -kv[1]["s"])[:20]:
            if c == "launch" and v["s"] < 0.5:
                continue
            o.write(f"| {tlabel(gt)} | {c} | {v['calls']} | {v['s']:.2f} | {v['max'] * 1e3:.1f} | " +
                    " | ".join(f"{v['st'].get(s, 0.0):.2f}" for s in stl) + " |\n")
        o.write("\ntop calls:\n\n| call | category | calls | total s | max ms | " + " | ".join(stl) +
                " |\n|---|---|---|---|---|" + "---|" * len(stl) + "\n")
        for n, v in sorted(names.items(), key=lambda kv: -kv[1]["s"])[:20]:
            o.write(f"| {n} | {v['cat']} | {v['calls']} | {v['s']:.2f} | {v['max'] * 1e3:.1f} | " +
                    " | ".join(f"{v['st'].get(s, 0.0):.2f}" for s in stl) + " |\n")
    with open(os.path.join(a.out, f"runa-{wl}.md"), "w") as f:
        f.write(o.getvalue())
    sys.stdout.write(o.getvalue())
    nrec = attr.get(RECOMMIT_LABEL, {}).get("ranges", 0)
    print(f"RUNA {wl}: stages from {src}; card holds {len(holds)} (log {len(lg['holds'])}); recommit ranges {nrec}; "
          f"cpu {'yes' if cpu else 'no'}")
    return 0 if kernels else 2


# ---------------------------------------------------------------------------------------------
# dry: evaluate the plan's passes against an nsys trace of a workload


def load_trace(path):
    """Every kernel launch, in the order ncu sees them (the launch API call's start)."""
    db = open_ro(path)
    q = """SELECT s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ,
                  k.staticSharedMemory + k.dynamicSharedMemory, k.registersPerThread,
                  k.start, k.end, r.start
           FROM CUPTI_ACTIVITY_KIND_KERNEL k
           JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId
           JOIN StringIds s ON s.id = k.shortName
           ORDER BY r.start, k.correlationId"""
    out = []
    for (name, gx, gy, gz, bx, by, bz, smem, regs, ks, ke, api) in db.execute(q):
        out.append({"kernel": name, "grid": (gx, gy, gz), "block": (bx, by, bz), "smem": smem, "regs": regs,
                    "dur_us": (ke - ks) / 1e3, "api_ns": api, "k_start_ns": ks})
    first = out[0]["api_ns"] if out else 0
    for i, r in enumerate(out):
        r["idx"] = i
        r["t_run_s"] = (r["api_ns"] - first) / 1e9  # seconds since the first launch
    return out


def in_window(row, windows, api_ns):
    """The launch call is inside the pass's process-wide NVTX range (ncu --nvtx-include "<name>"); a pass
    without one takes every launch."""
    if row["nvtx"] == "-":
        return True
    u = windows.get(row["nvtx"], [])
    i = bisect.bisect_right(u, (api_ns, float("inf"))) - 1
    return i >= 0 and u[i][0] <= api_ns < u[i][1]


def select(row, launches, windows=None):
    """(matched, profiled): the launches the pass's regex matches inside its NVTX window, and the ones
    ncu would profile.
    window: ncu's default filter: skip `skip` matching launches, profile the next `count` (then
            --kill ends the run).
    config: --filter-mode per-launch-config, whose key is the launch's grid, block and shared
            memory: skip/count per key. A config pass names one kernel, so the key never mixes two."""
    rx = kernel_re(row)
    windows = windows or {}
    m = [r for r in launches if rx.fullmatch(r["kernel"]) and in_window(row, windows, r["api_ns"])]
    skip, count = int(row["skip"]), int(row["count"])
    if row["mode"] == "window":
        return m, m[skip:skip + count]
    seen, picked = {}, []
    for r in m:
        k = (r["grid"], r["block"], r["smem"])
        n = seen.get(k, 0)
        if skip <= n < skip + count:
            picked.append(r)
        seen[k] = n + 1
    return m, picked


def est_seconds(row, picked, launches, wall_s):
    """A guide for the counters box, not a bound: 15 s to start, the workload up to the last
    profiled launch at 1.3x (window: --kill ends it there; config: the whole run, `wall_s`), and
    per profiled launch 1.5 s plus ~40 replays of its own duration."""
    if not launches:
        return 0.0
    end = wall_s
    if row["mode"] == "window":
        end = picked[-1]["t_run_s"] if picked else launches[-1]["t_run_s"]
    return 15.0 + 1.3 * end + sum(1.5 + 40 * r["dur_us"] * 1e-6 for r in picked)


def plan_for(path, workload, modes):
    return [r for r in read_plan(path) if r["workload"] == workload and modes in r["modes"].split(",")]


def cmd_dry(a):
    plan = plan_for(a.plan, a.workload, a.modes)
    launches = load_trace(a.sqlite)
    db = open_ro(a.sqlite)
    tbls = tables(db)
    nvtx = load_nvtx(db, tbls)
    end = max([r["k_start_ns"] + r["dur_us"] * 1e3 for r in launches] + [0])
    w, src = trace_windows(nvtx, end, a.workload)
    segs = partition(w, a.workload)
    sstarts = [s[0] for s in segs]
    windows = {}
    for s, e, lab, _t, pw in nvtx:
        if pw:
            windows.setdefault(lab, []).append((s, e))
    windows = {k: union(v) for k, v in windows.items()}
    for r in launches:
        r["stage"] = stage_at(segs, sstarts, r["api_ns"]) if segs else "whole"
    wall = end / 1e9
    os.makedirs(a.out, exist_ok=True)
    tot = sum(r["dur_us"] for r in launches) or 1.0
    per_kernel = {}
    for r in launches:
        k = per_kernel.setdefault(r["kernel"], {"n": 0, "us": 0.0, "cfg": set(), "stages": {}})
        k["n"] += 1
        k["us"] += r["dur_us"]
        k["cfg"].add((r["grid"], r["block"], r["smem"]))
        k["stages"][r["stage"]] = k["stages"].get(r["stage"], 0.0) + r["dur_us"]
    tag = a.tag or a.workload
    o = io.StringIO()
    o.write(f"# The pass plan against a trace: {tag} ({a.modes} mode)\n\n")
    o.write(f"trace: {len(launches)} kernel launches, {len(per_kernel)} kernels, Σ kernel time {tot / 1e6:.2f} s, "
            f"{wall:.1f} s of trace; stages from {src}; process-wide NVTX ranges: "
            f"{', '.join(f'{k} x{len(v)}' for k, v in sorted(windows.items())) or 'none'}.\n\n")
    o.write("## per pass: what ncu would profile\n\n")
    o.write("| pass | mode | nvtx | skip/count | matched launches | kernels matched | configs | would profile | "
            "stages of those | shapes of those | est. s | verdict |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    tsv = [["pass", "mode", "nvtx", "skip", "count", "matched", "kernels", "configs", "profile", "est_s", "verdict"]]
    detail = io.StringIO()
    rc = 0
    est_total = 0.0
    for row in plan:
        m, pa = select(row, launches, windows)
        names = {}
        for r in m:
            names[r["kernel"]] = names.get(r["kernel"], 0) + 1
        cfgs = len({(r["kernel"], r["grid"], r["block"], r["smem"]) for r in m})
        st, shp = {}, {}
        for r in pa:
            st[r["stage"]] = st.get(r["stage"], 0) + 1
            sh = shape(r["grid"], r["block"])
            shp[sh] = shp.get(sh, 0) + 1
        if row["nvtx"] != "-" and row["nvtx"] not in windows:
            verdict = "NO NVTX RANGE"
        else:
            verdict = "ok" if pa else ("NO MATCH" if not m else "NOTHING SELECTED")
        if verdict != "ok":
            rc = 1
        est = est_seconds(row, pa, launches, wall)
        est_total += est
        shapes_s = ", ".join(f"{k}{' x' + str(v) if v > 1 else ''}" for k, v in
                             sorted(shp.items(), key=lambda kv: -kv[1])[:4]) + (" …" if len(shp) > 4 else "")
        o.write(f"| {row['pass']} | {row['mode']} | {row['nvtx']} | {row['skip']}/{row['count']} | {len(m)} | "
                f"{len(names)} | {cfgs} | {len(pa)} | {', '.join(f'{k} {v}' for k, v in sorted(st.items()))} | "
                f"{shapes_s} | {est:.0f} | {verdict} |\n")
        tsv.append([row["pass"], row["mode"], row["nvtx"], row["skip"], row["count"], len(m), len(names), cfgs,
                    len(pa), f"{est:.0f}", verdict])
        detail.write(f"\n### {row['pass']} ({row['mode']}, nvtx {row['nvtx']}, kernels `{row['kernels']}`)\n\n")
        detail.write("matched: " + (", ".join(f"{k} x{v}" for k, v in sorted(names.items(), key=lambda kv: -kv[1]))
                                    or "NOTHING") + "\n\n")
        if pa:
            detail.write("| # in pass | launch idx | t s | stage | kernel | grid/block | smem | regs | µs |\n"
                         "|---|---|---|---|---|---|---|---|---|\n")
            for i, r in enumerate(pa[:60]):
                detail.write(f"| {i} | {r['idx']} | {r['t_run_s']:.2f} | {r['stage']} | {r['kernel']} | "
                             f"{shape(r['grid'], r['block'])} | {r['smem']} | {r['regs']} | {r['dur_us']:.1f} |\n")
            if len(pa) > 60:
                detail.write(f"| … | {len(pa) - 60} more | | | | | | | |\n")
    o.write(f"\nconfigs = distinct (kernel, grid, block, shared memory) among the matched. est. s = a guide for the "
            f"counters box (see est_seconds); Σ {est_total:.0f} s for these {len(plan)} passes.\n")
    o.write("\n## kernels by time, and which pass covers each\n\n| kernel | family | launches | configs | Σ s | % | "
            "stages (Σ s) | passes |\n|---|---|---|---|---|---|---|---|\n")
    uncovered = []
    for name, k in sorted(per_kernel.items(), key=lambda kv: -kv[1]["us"]):
        cov = [row["pass"] for row in plan if matches(row, name)]
        share = 100.0 * k["us"] / tot
        if share >= 1.0 and not cov:
            uncovered.append(f"{name} ({share:.1f} %)")
        if share >= 0.1 or cov:
            sts = ", ".join(f"{s} {v / 1e6:.2f}" for s, v in sorted(k["stages"].items(), key=lambda kv: -kv[1])[:4])
            o.write(f"| {name} | {family_of(name)} | {k['n']} | {len(k['cfg'])} | {k['us'] / 1e6:.3f} | {share:.1f} | "
                    f"{sts} | {', '.join(cov) or '-'} |\n")
    o.write(f"\nkernels with >= 1 % of kernel time that no pass covers: {', '.join(uncovered) or 'none'}\n")
    o.write(detail.getvalue())
    with open(os.path.join(a.out, f"dry-{tag}.md"), "w") as f:
        f.write(o.getvalue())
    write_tsv(os.path.join(a.out, f"dry-{tag}.tsv"), tsv)
    for t in tsv[1:]:
        print(f"DRY {tag} {t[0]}: nvtx {t[2]}; matched {t[5]} launches ({t[6]} kernels, {t[7]} configs); would "
              f"profile {t[8]}; est {t[9]} s; {t[10]}")
    print(f"DRY {tag}: {len(plan)} passes, est Σ {est_total:.0f} s; uncovered kernels >= 1 %: "
          f"{', '.join(uncovered) or 'none'}")
    return rc


# ---------------------------------------------------------------------------------------------
# report: SUMMARY.md from a send directory


def md_section(path, start, stop="\n## "):
    """The text of a markdown file from the line starting with `start` to the next `stop` heading."""
    if not os.path.exists(path):
        return None
    t = open(path, errors="replace").read()
    i = t.find(start)
    if i < 0:
        return None
    j = t.find(stop, i + len(start))
    return t[i:j if j >= 0 else len(t)].rstrip() + "\n"


def log_ok(lg, workload):
    """The pre-registered log gates of one run: the test passed and its headline line is there (the WHIR
    tree's verify accepted it; the STARK tree reached its final check and every child verify)."""
    if not (lg["test_result"] or "").startswith("test result: ok. 1 passed"):
        return False
    if workload == "whir":
        return lg["w3_base"] is not None and lg["w3_rec"] is not None and lg["w3_verify"] == "ACCEPTED"
    return lg["block"] is not None and lg["final_check"]


def cmd_loghead(a):
    """One TSV line for the driver: ok|bad, the test result, CARD HOLD lines, the headline."""
    lg = read_log(a.log)
    print("\t".join([("ok" if log_ok(lg, a.workload) else "bad"), lg["test_result"] or "<none>",
                     str(len(lg["holds"])), base_line(lg)[:600]]))
    return 0


def cmd_report(a):
    s = a.send
    o = io.StringIO()
    o.write("# noepoch_counters.sh: summary\n\n")
    env = os.path.join(s, "env.txt")
    if os.path.exists(env):
        keep = ("script=", "repo=", "whir_", "stark_", "gpu_name=", "driver=", "ncu=", "nsys=", "cpu=", "nvtx_")
        o.write("```\n" + "".join(l for l in open(env) if l.startswith(keep)) + "```\n\n")
    ref = read_tsv(os.path.join(s, "reference", "runs.tsv"))
    if ref:
        o.write("## reference runs (no profiler)\n\n| workload | rc | seconds | headline | CPU | VRAM max MiB |\n"
                "|---|---|---|---|---|---|\n")
        for r in ref:
            o.write(f"| {r['workload']} | {r['rc']} | {r['seconds']} | {r['headline']} | {r.get('cpu', '-')} | "
                    f"{r['vram_max_mib']} |\n")
        o.write("\n")
    for wl in WORKLOADS:
        p = os.path.join(s, "runa", wl, f"runa-{wl}.md")
        if not os.path.exists(p):
            continue
        t = open(p, errors="replace").read()
        o.write(f"## run A, {wl}\n\n")
        o.write(t.split("\n", 2)[2] if t.count("\n") > 2 else t)
        o.write("\n")
    k = md_section(os.path.join(s, "summary", "kernels.md"), "## per kernel, both workloads")
    if k:
        o.write("## run B: " + k.split(" ", 1)[1] + "\n")
    passes = read_tsv(os.path.join(s, "passes.tsv"))
    if passes:
        o.write("## run B passes\n\n| pass | workload | nvtx | mode | profiled | seconds | verdict |\n"
                "|---|---|---|---|---|---|---|\n")
        for r in passes:
            o.write(f"| {r['pass']} | {r['workload']} | {r['nvtx']} | {r['mode']} | {r['profiled']} | {r['seconds']} | "
                    f"{r['verdict']} |\n")
        o.write("\n")
    for wl in WORKLOADS:
        d = md_section(os.path.join(s, "runa", wl, f"dry-runa-{wl}.md"), "## per pass")
        if d:
            o.write(f"## the plan against run A's {wl} trace\n\n" + d.split("\n", 1)[1] + "\n")
    with open(os.path.join(s, "SUMMARY.md"), "w") as f:
        f.write(o.getvalue())
    print(f"report: {os.path.join(s, 'SUMMARY.md')} ({len(o.getvalue())} bytes)")
    return 0


# ---------------------------------------------------------------------------------------------
# plan-check, cargo-artifact


def cmd_plan_check(a):
    rows = read_plan(a.plan)
    errs = check_plan(rows)
    if not rows:
        errs.append("the plan has no pass")
    for e in errs:
        print("PLAN ERROR: " + e)
    if not errs:
        by = {}
        for r in rows:
            by[r["workload"]] = by.get(r["workload"], 0) + 1
        print("plan ok: " + ", ".join(f"{k} {v} pass(es)" for k, v in sorted(by.items())))
    return 1 if errs else 0


def cmd_cargo_artifact(a):
    found = None
    pkg = re.compile(r"(^|[/ ])" + re.escape(a.outdir or "\0") + r"([ #@]|$)")
    for line in sys.stdin:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if a.exe and msg.get("reason") == "compiler-artifact":
            tgt = msg.get("target", {})
            if (tgt.get("name") == a.exe and msg.get("executable") and msg.get("profile", {}).get("test")
                    and "lib" in tgt.get("kind", [])):
                found = msg["executable"]
        elif a.outdir and msg.get("reason") == "build-script-executed":
            if pkg.search(msg.get("package_id", "")) and msg.get("out_dir"):
                found = msg["out_dir"]
    if not found:
        sys.stderr.write("cargo-artifact: nothing matching in cargo's messages\n")
        return 1
    print(found)
    return 0


# ---------------------------------------------------------------------------------------------
# cpu-sample and cpu-total: the profiled process's CPU and the host's memory, beside a run


def proc_stat(path):
    """(comm, utime + stime ticks, threads, rss pages, state) from a /proc stat file."""
    with open(path) as f:
        t = f.read()
    left, right = t.index("("), t.rindex(")")
    rest = t[right + 2:].split()
    return t[left + 1:right], int(rest[11]) + int(rest[12]), int(rest[17]), int(rest[21]), rest[0]


def mem_avail_mib():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def find_pid(exe, me):
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            if os.path.realpath(f"/proc/{d}/exe") == exe:
                return int(d)
        except OSError:
            continue
    return None


def cmd_cpu_sample(a):
    """Waits for the process running --exe (under nsys or ncu it is their child), then every interval
    writes its cumulative CPU ticks, RSS, thread count and the host's MemAvailable (PREFIX.proc.tsv) and
    each thread whose ticks moved (PREFIX.tid.tsv), until it exits. Below --mem-floor-mib of MemAvailable
    it sends the process SIGTERM, then SIGKILL 10 s later, and says so in PREFIX.watchdog: a run that
    would push the machine into swap or the OOM killer ends instead."""
    exe = os.path.realpath(a.exe)
    clk = os.sysconf("SC_CLK_TCK")
    page_kb = os.sysconf("SC_PAGE_SIZE") // 1024
    me = os.getpid()
    deadline = time.time() + a.wait_s
    pid = None
    while pid is None:
        pid = find_pid(exe, me)
        if pid is None:
            if time.time() > deadline:
                sys.stderr.write(f"cpu-sample: no process ran {exe} within {a.wait_s} s\n")
                return 1
            time.sleep(0.05)
    term_at, killed = None, False
    with open(a.out + ".proc.tsv", "w") as fp, open(a.out + ".tid.tsv", "w") as ft:
        fp.write(f"# clk_tck {clk}\nepoch_ns\tpid\tticks\trss_kb\tthreads\tmem_avail_mib\n")
        ft.write("epoch_ns\ttid\tcomm\tticks\n")
        last = {}
        while True:
            now = time.time_ns()
            try:
                _comm, ticks, nth, rss, state = proc_stat(f"/proc/{pid}/stat")
            except (OSError, ValueError, IndexError):
                break
            if state in ("Z", "X"):
                break  # exited, not yet reaped by its parent
            avail = mem_avail_mib()
            fp.write(f"{now}\t{pid}\t{ticks}\t{rss * page_kb}\t{nth}\t{avail}\n")
            try:
                tids = os.listdir(f"/proc/{pid}/task")
            except OSError:
                tids = []
            for t in tids:
                try:
                    tc, tt, _, _, _ = proc_stat(f"/proc/{pid}/task/{t}/stat")
                except (OSError, ValueError, IndexError):
                    continue
                if last.get(t) != tt:
                    ft.write(f"{now}\t{t}\t{tc}\t{tt}\n")
                    last[t] = tt
            fp.flush()
            ft.flush()
            if a.mem_floor_mib and 0 <= avail < a.mem_floor_mib and term_at is None:
                with open(a.out + ".watchdog", "w") as f:
                    f.write(f"MemAvailable {avail} MiB under the floor {a.mem_floor_mib} MiB at epoch_ns {now}: "
                            "the profiled process was sent SIGTERM (SIGKILL 10 s later if still running)\n")
                sys.stderr.write(f"cpu-sample: WATCHDOG: MemAvailable {avail} MiB < {a.mem_floor_mib} MiB; "
                                 "ending the profiled process\n")
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
                term_at = time.time()
            elif term_at is not None and not killed and time.time() - term_at > 10:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
                killed = True
            time.sleep(a.interval_ms / 1000.0)
    return 0


def cpu_total_line(prefix):
    cpu = read_cpu(prefix, 0)
    if not cpu or len(cpu["proc"]) < 2:
        return "no CPU samples"
    p = cpu["proc"]
    secs = (p[-1][1] - p[0][1]) / cpu["clk"]
    wall = (p[-1][0] - p[0][0]) / 1e9
    peak = max(x[2] for x in p) / 1048576
    avail = [x[4] for x in p if x[4] >= 0]
    line = (f"CPU {secs:.1f} s over {wall:.1f} s ({secs / wall if wall else 0:.1f} cores), peak RSS {peak:.1f} GiB, "
            f"min MemAvailable {min(avail) / 1024 if avail else -1:.1f} GiB")
    if cpu.get("watchdog"):
        line += " · WATCHDOG: " + cpu["watchdog"]
    return line


def cmd_cpu_total(a):
    print(cpu_total_line(a.cpu))
    return 0


# ---------------------------------------------------------------------------------------------
# scrub and check: what may leave the machine

FORBIDDEN_EXT = (".ncu-rep", ".nsys-rep", ".qdrep", ".qdstrm", ".sqlite", ".sqlite3", ".db", ".arrow", ".parquet")
# case-sensitive credential shapes; a hit refuses the bundle (the report names file:line:marker only)
MARKERS = re.compile(r"API_KEY|APIKEY|SECRET|TOKEN|PASSWORD|PASSWD|PRIVATE KEY|BEGIN [A-Z ]*PRIVATE|"
                     r"ssh-(rsa|ed25519|dss|ecdsa)|ghp_[A-Za-z0-9]|gho_[A-Za-z0-9]|github_pat_|xox[bp]-|"
                     r"LS0tLS1CRUdJT|AKIA[0-9A-Z]{16}")


def text_of(path):
    """The file's text, or None when it is not plain UTF-8 text without NUL bytes."""
    with open(path, "rb") as f:
        data = f.read()
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def drop_host_column(text):
    rows = list(csv.reader(io.StringIO(text)))
    hdr_i = next((i for i, r in enumerate(rows) if "Host Name" in r and "ID" in r), None)
    if hdr_i is None:
        return text, False
    j = rows[hdr_i].index("Host Name")
    o = io.StringIO()
    w = csv.writer(o, quoting=csv.QUOTE_ALL, lineterminator="\n")
    for i, r in enumerate(rows):
        if i >= hdr_i and len(r) > j:
            r = r[:j] + r[j + 1:]
        w.writerow(r)
    return o.getvalue(), True


def cmd_scrub(a):
    pairs = []
    for line in sys.stdin:
        line = line.rstrip("\n")
        if "\t" in line:
            old, new = line.split("\t", 1)
            if len(old) >= 3:
                pairs.append((old, new))
    pairs.sort(key=lambda p: -len(p[0]))
    counts, dropped = {}, 0
    for root, _dirs, files in os.walk(a.dir):
        for fn in files:
            p = os.path.join(root, fn)
            t = text_of(p)
            if t is None:
                continue
            new = t
            if fn.endswith(".csv"):
                new, d = drop_host_column(new)
                dropped += int(d)
            for old, rep in pairs:
                n = new.count(old)
                if n:
                    counts[rep] = counts.get(rep, 0) + n
                    new = new.replace(old, rep)
            if new != t:
                with open(p, "w", encoding="utf-8") as f:
                    f.write(new)
    print("scrub: " + (", ".join(f"{k} x{v}" for k, v in sorted(counts.items())) or "nothing to replace")
          + f"; Host Name column dropped from {dropped} CSV file(s)")
    return 0


def cmd_check(a):
    raw = sys.stdin.buffer.read().decode("utf-8", "replace")
    values = []
    for rec in raw.split("\0"):
        if "\t" in rec:
            label, val = rec.split("\t", 1)
            if len(val) >= 4:
                values.append((label, val))
    hits, bad, nfiles = [], [], 0
    for root, _dirs, files in os.walk(a.dir):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, a.dir)
            nfiles += 1
            if fn.lower().endswith(FORBIDDEN_EXT):
                bad.append(f"{rel}: an Nsight report, trace or database")
                continue
            if os.path.getsize(p) > 50 * 1024 * 1024:
                bad.append(f"{rel}: over 50 MB")
                continue
            t = text_of(p)
            if t is None:
                bad.append(f"{rel}: not plain UTF-8 text")
                continue
            if t.startswith("SQLite format 3"):
                bad.append(f"{rel}: a sqlite database")
                continue
            for ln, line in enumerate(t.split("\n"), 1):
                m = MARKERS.search(line)
                if m:
                    hits.append(f"{rel}:{ln}: credential marker '{m.group(0)[:4]}...'")
                for label, val in values:
                    if val in line:
                        hits.append(f"{rel}:{ln}: {label}")
    with open(a.report, "w") as f:
        for x in bad + hits:
            f.write(x + "\n")
    if bad or hits:
        files = sorted({x.split(":", 1)[0] for x in bad + hits})
        print(f"self-check: REFUSED: {len(hits)} hit(s) and {len(bad)} forbidden file(s) in {len(files)} file(s): "
              f"{' '.join(files[:12])}{' …' if len(files) > 12 else ''} (file:line:what in {a.report}; "
              "the matched text itself is never printed)")
        return 1
    print(f"self-check: clean: {nfiles} plain-text files, no Nsight report or database, no credential marker, "
          f"none of {len(values)} machine-specific values (hostname, home path, IP addresses, environment values)")
    return 0


# ---------------------------------------------------------------------------------------------
# selftest


def synth_details_csv(host="127.0.0.1", base_units=False):
    hdr = ["ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context", "Stream", "Block Size",
           "Grid Size", "Device", "CC", "Section Name", "Metric Name", "Metric Unit", "Metric Value", "Rule Name",
           "Rule Type", "Rule Description", "Estimated Speedup Type", "Estimated Speedup"]
    rows = []

    def add(i, k, blk, grd, sec, name, unit, val):
        rows.append([str(i), "4242", "lambda_vm_prover-0123", host, k, "1", "13", blk, grd, "0", "12.0", sec, name,
                     unit, val, "", "", "", "", ""])

    # launch 0: a leaf kernel, compute-bound, bytes from the explicit counters
    k0, b0, g0 = "rpx_leaves_base_row_pair_batched", "(128, 1, 1)", "(16384, 1, 1)"
    dur, freq = (("nsecond", "62,360,000"), ("cycle/second", "2,010,000,000")) if base_units else \
        (("ms", "62.36"), ("Ghz", "2.01"))
    rd, wr = (("byte", "1,048,580,000"), ("byte", "67,110,000")) if base_units else \
        (("Mbyte", "1,048.58"), ("Mbyte", "67.11"))
    for sec, n, u, v in [(SOL, "Duration") + dur, (SOL, "SM Frequency") + freq,
                         (SOL, "DRAM Throughput", "%", "2.10"), (SOL, "Compute (SM) Throughput", "%", "92.70"),
                         (SOL, "Memory Throughput", "%", "33.15"), (SOL, "L2 Cache Throughput", "%", "20.00"),
                         (MWA, "L2 Hit Rate", "%", "98.10"), (OCC, "Achieved Occupancy", "%", "72.30"),
                         (OCC, "Theoretical Occupancy", "%", "75.00"), (OCC, "Block Limit Registers", "block", "9"),
                         (OCC, "Block Limit Warps", "block", "12"), (LST, "Threads", "thread", "2,097,152"),
                         (LST, "Registers Per Thread", "register/thread", "56"),
                         ("Command line profiler metrics", "dram__bytes_read.sum") + rd,
                         ("Command line profiler metrics", "dram__bytes_write.sum") + wr,
                         ("Command line profiler metrics",
                          "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio", "inst", "1.2"),
                         ("Command line profiler metrics",
                          "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio", "inst", "0.4"),
                         ("Command line profiler metrics",
                          "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active", "%", "53.6"),
                         ("Command line profiler metrics",
                          "sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active", "%", "88.0")]:
        add(0, k0, b0, g0, sec, n, u, v)
    # launch 1: an NTT pass, DRAM-bound, bytes from throughput x duration (no explicit counters)
    k1, b1, g1 = "ntt_cm_dit_k8", "(256, 1, 1)", "(2048, 16, 1)"
    tp = ("byte/second", "1,490,000,000,000") if base_units else ("Tbyte/s", "1.49")
    for sec, n, u, v in [(SOL, "Duration", "us", "140.50"), (SOL, "DRAM Throughput", "%", "84.00"),
                         (SOL, "Compute (SM) Throughput", "%", "20.00"), (SOL, "L2 Cache Throughput", "%", "40.00"),
                         (MWA, "Memory Throughput") + tp, (MWA, "L2 Hit Rate", "%", "45.0"),
                         (OCC, "Achieved Occupancy", "%", "30.0"), (OCC, "Theoretical Occupancy", "%", "33.3")]:
        add(1, k1, b1, g1, sec, n, u, v)
    # launch 2: a second NTT launch of another shape
    for sec, n, u, v in [(SOL, "Duration", "us", "59.50"), (SOL, "DRAM Throughput", "%", "70.00"),
                         (SOL, "Compute (SM) Throughput", "%", "18.00"), (SOL, "L2 Cache Throughput", "%", "30.00"),
                         (MWA, "Memory Throughput", "Gbyte/s", "1,200.00")]:
        add(2, k1, b1, "(1024, 16, 1)", sec, n, u, v)
    rows.append(["2", "4242", "lambda_vm_prover-0123", host, k1, "1", "13", b1, "(1024, 16, 1)", "0", "12.0",
                 "SpeedOfLight", "", "", "", "SOLBottleneck", "OPT", "rule text, no metric", "", ""])
    o = io.StringIO()
    o.write("==PROF== Connected to process 4242\n")
    w = csv.writer(o, quoting=csv.QUOTE_ALL, lineterminator="\n")
    w.writerow(hdr)
    w.writerows(rows)
    return o.getvalue()


T0_SYNTH = 1_800_000_000 * 10 ** 9
MAIN_TID, DRV_TID = (7 << 24) | 100, (7 << 24) | 101


def build_synth_sqlite(path):
    """A small nsys-shaped trace of one no-epoch WHIR block and its tree, session start T0_SYNTH (trace
    seconds): setup [0, 1), blk_base [1, 9) with blk_phase_a [1.5, 5), blk_prepared [5, 5.2) and
    blk_phase_b [5.5, 8.5); lfm_tree [9.2, 9.8) holding one cardhold_multi_prove [9.3, 9.6);
    harness_verify [9.8, 10). A blk_walker push/pop range on the driver thread names it."""
    import sqlite3
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE TARGET_INFO_SESSION_START_TIME (utcEpochNs INTEGER, utcTime TEXT, localTime TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, eventClass INTEGER, globalTid INTEGER,
            correlationId INTEGER, nameId INTEGER, returnValue INTEGER, callchainId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, deviceId INTEGER, correlationId INTEGER,
            shortName INTEGER, gridX INTEGER, gridY INTEGER, gridZ INTEGER, blockX INTEGER, blockY INTEGER,
            blockZ INTEGER, staticSharedMemory INTEGER, dynamicSharedMemory INTEGER, registersPerThread INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, copyKind INTEGER, bytes INTEGER);
        CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, eventType INTEGER, rangeId INTEGER, category INTEGER,
            color INTEGER, text TEXT, globalTid INTEGER, endGlobalTid INTEGER, textId INTEGER, domainId INTEGER);
        CREATE TABLE ThreadNames (nameId INTEGER, priority INTEGER, globalTid INTEGER);
        CREATE TABLE GPU_METRICS (rawTimestamp INTEGER, timestamp INTEGER, typeId INTEGER, metricId INTEGER,
            value INTEGER);
        CREATE TABLE TARGET_INFO_GPU_METRICS (typeId INTEGER, sourceId INTEGER, typeName TEXT, metricId INTEGER,
            metricName TEXT);
    """)
    names = ["ntt_cm_dit_k8", "rpx_merkle_level", "rpx_leaves_base_coset", "sumcheck_round_ext3",
             "cuLaunchKernel", "cuStreamSynchronize", "cuMemAlloc_v2", "cuMemcpyDtoHAsync_v2", "cuMemHostAlloc",
             "driver-3", "blk_phase_b", "whir_fold_ext3", "ccomp_0b8d15837e1e77a3"]
    for i, n in enumerate(names, 1):
        db.execute("INSERT INTO StringIds VALUES (?, ?)", (i, n))
    db.execute("INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (?, 'x', 'x')", (T0_SYNTH,))
    db.execute("INSERT INTO ThreadNames VALUES (10, 0, ?)", (DRV_TID,))
    sec = 10 ** 9

    def rng(a, b, text, tid, et=60, text_id=None):
        db.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, 0, 0, 0, ?, ?, ?, ?, 0)",
                   (int(a * sec), int(b * sec), et, text, tid, tid, text_id))

    rng(1.0, 9.0, "blk_base", MAIN_TID)
    rng(1.5, 5.0, "blk_phase_a", MAIN_TID)
    rng(3.0, 4.5, "blk_a_commit", MAIN_TID)
    rng(5.0, 5.2, "blk_prepared", MAIN_TID)
    rng(5.5, 8.5, None, MAIN_TID, 60, 11)          # a registered string: the text through StringIds
    rng(5.5, 7.0, "blk_b_argue", MAIN_TID)
    rng(9.2, 9.8, "lfm_tree", MAIN_TID)
    rng(9.3, 9.6, "cardhold_multi_prove", DRV_TID)
    rng(9.8, 10.0, "harness_verify", MAIN_TID)
    rng(1.5, 5.0, "blk_walker", DRV_TID, 59)       # a thread's role, push/pop
    # (kernel, grid, t launch, duration ms, thread): the launch call starts 1 µs before the kernel
    launches = [("ntt_cm_dit_k8", 1024, 3.0, 100.0, MAIN_TID), ("rpx_leaves_base_coset", 4096, 3.5, 200.0, MAIN_TID),
                ("sumcheck_round_ext3", 253, 6.0, 300.0, MAIN_TID), ("whir_fold_ext3", 512, 7.0, 400.0, DRV_TID),
                ("ntt_cm_dit_k8", 2048, 7.6, 50.0, MAIN_TID),
                ("ccomp_0b8d15837e1e77a3", 512, 9.4, 100.0, DRV_TID), ("rpx_merkle_level", 256, 9.85, 10.0, MAIN_TID)]
    for c, (k, gx, ts, dms, tid) in enumerate(launches, 1):
        s = int(ts * sec)
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 0, ?, ?, 5, 0, 0)", (s - 1000, s, tid, c))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 0, ?, ?, ?, 1, 1, 128, 1, 1, 0, 0, 40)",
                   (s, s + int(dms * 1e6), c, names.index(k) + 1, gx))
    # API calls without kernels: a 0.3 s sync in phase A, a 0.2 s alloc and a 0.1 s copy in phase B, a pinned
    # alloc in the setup, and the last call at 9.95 s (the trace's end)
    for c, (nm, ts, dur, tid) in enumerate([("cuStreamSynchronize", 4.0, 0.3, MAIN_TID),
                                            ("cuMemAlloc_v2", 5.6, 0.2, DRV_TID),
                                            ("cuMemcpyDtoHAsync_v2", 7.5, 0.1, DRV_TID),
                                            ("cuMemHostAlloc", 0.5, 0.05, MAIN_TID),
                                            ("cuStreamSynchronize", 9.9, 0.05, MAIN_TID)], 100):
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 0, ?, ?, ?, 0, 0)",
                   (int(ts * sec), int((ts + dur) * sec), tid, c, names.index(nm) + 1))
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, 1, 1000)", (int(1.6 * sec), int(2.1 * sec)))
    db.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (7, 0, 'syn', 1, 'SMs Active [Throughput %]')")
    db.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (7, 0, 'syn', 2, 'Compute Warps in Flight [Throughput %]')")
    for tenth in range(0, 100):               # 10 Hz: SM active 80 in phase A, 30 in phase B, 0 elsewhere
        ts = tenth * 10 ** 8
        v = 80 if 15 <= tenth < 50 else (30 if 55 <= tenth < 85 else 0)
        db.execute("INSERT INTO GPU_METRICS VALUES (?, ?, 7, 1, ?)", (ts, ts, v))
        db.execute("INSERT INTO GPU_METRICS VALUES (?, ?, 7, 2, ?)", (ts, ts, v // 2))
    db.commit()
    db.close()


def write_synth_cpu(prefix):
    """The sampler's files for the synthetic run: every 0.1 s, 20 ticks (2 cores at 100 Hz) inside phase A
    [1.5, 5), 5 ticks elsewhere; the driver thread (tid 101, the walker) 10 ticks a sample in phase A."""
    with open(prefix + ".proc.tsv", "w") as fp, open(prefix + ".tid.tsv", "w") as ft:
        fp.write("# clk_tck 100\nepoch_ns\tpid\tticks\trss_kb\tthreads\tmem_avail_mib\n")
        ft.write("epoch_ns\ttid\tcomm\tticks\n")
        ticks = tt = 0
        for tenth in range(0, 101):
            t = T0_SYNTH + tenth * 10 ** 8
            fp.write(f"{t}\t4242\t{ticks}\t{1048576 * (1 + tenth // 50)}\t40\t30000\n")
            ft.write(f"{t}\t101\tlfm::whir_bloc\t{tt}\n")
            ft.write(f"{t}\t102\telf-beside-3\t{tenth}\n")
            mid = tenth + 0.5
            ticks += 20 if 15 <= mid < 50 else 5
            tt += 10 if 15 <= mid < 50 else 0


def selftest():
    fails = []

    def ok(cond, what):
        if not cond:
            fails.append(what)

    def fnum(s):
        try:
            return float(s)
        except (TypeError, ValueError):
            return float("nan")

    def run(fn, ns, stdin=None):
        so, si = sys.stdout, sys.stdin
        sys.stdout = io.StringIO()
        if stdin is not None:
            sys.stdin = stdin
        try:
            rc = fn(ns)
            return rc, sys.stdout.getvalue()
        finally:
            sys.stdout, sys.stdin = so, si

    ok(num("4,194,304") == 4194304.0 and num("62.36") == 62.36 and num("n/a") is None and num("1,2") is None,
       "num() parses ncu numbers")
    ok(to_bytes(1.5, "Kbyte") == 1500.0 and to_bytes(2, "Gbyte") == 2e9 and to_bytes(3, "%") is None
       and to_bytes(7, "byte") == 7.0, "to_bytes")
    ok(abs(to_bytes_per_s(1.49, "Tbyte/s") - 1.49e12) < 1 and to_bytes_per_s(5, "byte/second") == 5.0
       and to_us(62.36, "ms") == 62360.0 and to_us(62360000, "nsecond") == 62360.0, "rates and times")
    ok(abs(to_ghz(2.01, "Ghz") - 2.01) < 1e-9 and abs(to_ghz(2.01e9, "cycle/second") - 2.01) < 1e-9
       and to_ghz(2, "%") is None, "clock units")
    ok(dims("(128, 1, 1)") == (128, 1, 1) and dims("128, 2, 1") == (128, 2, 1) and dims("n/a") is None, "dims")
    ok(union([(5, 7), (1, 3), (2, 4), (7, 8)]) == [(1, 4), (5, 8)], "union")
    ok(subtract([(0, 10)], [(2, 3), (5, 7)]) == [(0, 2), (3, 5), (7, 10)] and subtract([(0, 4)], [(0, 4)]) == [],
       "subtract")
    ok(intersect([(0, 10)], [(2, 3), (5, 12)]) == [(2, 3), (5, 10)], "intersect")
    sf = StepFn([(0, 10), (5, 15)])
    ok(sf.integral(0, 20) == 20.0 and abs(sf.mean([(5, 10)]) - 2.0) < 1e-9, "the open-range step function")
    ok(api_cat("cuStreamSynchronize") == "sync" and api_cat("cuMemcpyDtoHAsync_v2") == "copy_async"
       and api_cat("cuMemcpyHtoD_v2") == "copy_sync" and api_cat("cuMemAlloc_v2") == "alloc"
       and api_cat("cuMemAllocAsync") == "alloc" and api_cat("cuMemFreeAsync") == "free"
       and api_cat("cudaMallocHost") == "host_pinned" and api_cat("cuMemHostRegister_v2") == "host_pinned"
       and api_cat("cuEventSynchronize") == "sync" and api_cat("cuEventRecord") == "event_stream"
       and api_cat("cuLaunchKernel") == "launch" and api_cat("cuMemcpyBatchAsync") == "copy_async"
       and api_cat("cuModuleLoadData") == "module" and api_cat("cuWeird") == "other", "API categories")
    ok(norm_label(":rounds_2to4_table") == "rounds_2to4_table" and norm_label("epoch_prove[i=3]") == "epoch_prove",
       "NVTX labels")
    ok(family_of("sumcheck_round_ext3") == "sumcheck" and family_of("whir_fold_ext3") == "whir-fold"
       and family_of("ccomp_0b8d15837e1e77a3") == "quotient" and family_of("rpx_leaves_base_coset") == "leaves",
       "kernel families")
    ok(comm_family("elf-beside-3") == "elf-beside" and comm_family("rayon-worker-12") == "rayon-worker",
       "thread-name families")
    hl = HOLD_RE.search("CARD HOLD #3 multi_prove: waited 0.120s · held 0.450s · t=[1800000010.000,1800000010.450]")
    ok(hl is not None and hl.group(2) == "multi_prove" and hl.group(4) == "0.450", "the CARD HOLD line")
    ok(LFM_PROVE_RE.search("   BLOCK L1N0 (4 children) LFM PROVE: execute 0.21s · fill 0.30s · multi_prove 0.45s")
       is not None and LFM_PROVE_RE.search("   BLOCK L0 leaf 3 LFM PROVE: execute 0.21s · fill 0.30s · "
                                           "multi_prove 0.45s").group(1) == "BLOCK L0 leaf 3", "the LFM PROVE line")
    with tempfile.TemporaryDirectory() as d:
        plan = os.path.join(d, "plan.tsv")
        with open(plan, "w") as f:
            f.write("\t".join(PLAN_COLS) + "\n")
            f.write("whir_b_sum\twhir\tquick,full\tconfig\t0\t1\tblk_phase_b\tsumcheck_round_ext3\tsumcheck\tone per shape\n")
            f.write("whir_a_ntt\twhir\tfull\twindow\t0\t4\tblk_phase_a\tntt_cm_di[ft]_k[4-8]\tlde\tphase A only\n")
            f.write("whir_all_ntt\twhir\tfull\twindow\t0\t4\t-\tntt_cm_di[ft]_k[4-8]\tlde\tno window\n")
            f.write("whir_b_leaves\twhir\tfull\twindow\t0\t1\tblk_phase_b\trpx_leaves_base_coset\tleaves\tmust match 0\n")
            f.write("whir_none\twhir\tfull\twindow\t0\t1\tno_such_range\tntt_cm_dit_k8\tlde\tmust say NO NVTX RANGE\n")
            f.write("stark_lfm\tstark\tquick,full\twindow\t0\t4\tcardhold_multi_prove\tccomp_[0-9a-f]+\tquotient\tq\n")
        ok(check_plan(read_plan(plan)) == [], f"the synthetic plan is well formed ({check_plan(read_plan(plan))})")
        badplan = os.path.join(d, "bad.tsv")
        with open(badplan, "w") as f:
            f.write("\t".join(PLAN_COLS) + "\n")
            f.write("x\twhir\tfull\twindow\t0\t0\t-\t^rpx$\tf\tn\n")
            f.write("x\tzisk\tsome\tsome\t-1\t1\tbad range!\trpx(\tf\tn\n")
            f.write("y\tstark\tfull\tconfig\t0\t1\t-\tfri_fold_ext3|logup_[a-z0-9_]+\tf\tn\n")
        errs = check_plan(read_plan(badplan))
        ok(len(errs) >= 9 and any("exactly one kernel" in e for e in errs) and any("nvtx is" in e for e in errs)
           and any("modes is" in e for e in errs), f"a malformed plan is refused ({len(errs)} errors: {errs})")
        # summary, in both unit styles, one export per workload
        for base_units in (False, True):
            nd = os.path.join(d, f"ncu{int(base_units)}")
            os.makedirs(nd)
            for p in ("whir_b_sum", "stark_lfm"):
                with open(os.path.join(nd, f"{p}.details.csv"), "w") as f:
                    f.write(synth_details_csv(base_units=base_units))
                with open(os.path.join(nd, f"{p}.stages.tsv"), "w") as f:
                    lab = "blk_phase_b" if p.startswith("whir") else "cardhold_multi_prove"
                    f.write(f"id\tkernel\tstage\tpasses\n0\trpx_leaves_base_row_pair_batched\t{lab}\t18\n"
                            f"1\tntt_cm_dit_k8\t{lab}\t18\n2\tntt_cm_dit_k8\t{lab}\t18\n")
            out = os.path.join(d, f"sum{int(base_units)}")
            rc, _ = run(cmd_summary, argparse.Namespace(plan=plan, ncu_dir=nd, out=out))
            tag = "base units" if base_units else "auto units"
            ok(rc == 0, f"summary exits 0 ({tag})")
            rows = [r for r in read_tsv(os.path.join(out, "launches.tsv")) if r["pass"] == "whir_b_sum"]
            ok(len(rows) == 3, f"three launches ({len(rows)}, {tag})")
            r0 = rows[0]
            ok(r0["kernel"] == "rpx_leaves_base_row_pair_batched" and r0["stage"] == "blk_phase_b"
               and r0["elem"] == "leaf", f"launch 0 fields ({tag})")
            ok(abs(fnum(r0["dur_us"]) - 62360.0) < 0.5 and abs(fnum(r0["sm_ghz"]) - 2.01) < 1e-6,
               f"duration and SM clock ({r0['dur_us']}, {r0['sm_ghz']}, {tag})")
            ok(r0["family"] == "leaves" and r0["workload"] == "whir" and r0["bound"] == "compute",
               "the family comes from the kernel, the workload from the pass")
            ok(abs(fnum(r0["b_per_elem"]) - (1048.58e6 + 67.11e6) / 2097152) < 0.01 and r0["bytes_src"] == "counters",
               f"leaf bytes/elem from the counters ({tag})")
            ok(r0["roof"].startswith("compute roof 93"), f"launch 0 roof ({r0['roof']})")
            ok(r0["occ_limit"] == "Registers" and "math_pipe_throttle 75%" in r0["stalls"] and "alu 54%" in r0["pipes"]
               and "fmaheavy 88%" in r0["pipe_cycles"], "occupancy limiter, stalls, pipes and pipe cycles")
            r1 = rows[1]
            want = 1.49e12 * 140.5e-6 / (2048 * 16 * 256 * 16)
            ok(abs(fnum(r1["b_per_elem"]) - want) < 0.01 and r1["bytes_src"] == "throughput x duration",
               f"NTT bytes/elem from throughput x duration ({r1['b_per_elem']} vs {want:.2f}, {tag})")
            ok(r1["roof"].startswith("DRAM roof 84") and r1["bound"] == "memory", "NTT roof and bound")
            kn = {(k["workload"], k["kernel"]): k for k in read_tsv(os.path.join(out, "kernels.tsv"))}
            ok(set(kn) == {(w, k) for w in ("whir", "stark") for k in ("rpx_leaves_base_row_pair_batched",
                                                                        "ntt_cm_dit_k8")}, "one row per workload and kernel")
            ntt = kn[("whir", "ntt_cm_dit_k8")]
            wdram = (84.0 * 140.5 + 70.0 * 59.5) / 200.0
            ok(ntt["launches"] == "2" and ntt["configs"] == "2" and abs(fnum(ntt["dram_pct"]) - wdram) < 0.01,
               f"duration-weighted DRAM % ({ntt['dram_pct']} vs {wdram:.2f})")
            ok(ntt["largest_shape"] == "2048x16x1/256x1x1" and ntt["stages"] == "blk_phase_b", "largest launch, stages")
            md = open(os.path.join(out, "kernels.md")).read()
            ok("## per kernel, both workloads (the roofs)" in md and "| ntt_cm_dit_k8 | lde | whir |" in md
               and "| ntt_cm_dit_k8 | lde | stark |" in md and "## whir: one row per kernel" in md,
               "kernels.md: the comparison and the per-workload tables")
            ks = read_tsv(os.path.join(out, "kernels_by_stage.tsv"))
            ok(len(ks) == 4 and any(r["stage"] == "cardhold_multi_prove" and r["kernel"] == "ntt_cm_dit_k8" for r in ks),
               f"kernels by stage ({len(ks)})")
        # stages of a pass = its NVTX window
        log = ['==PROF== Connected to process 1 (<W>/bin)', '==PROF== Profiling "ntt_cm_dit_k8": 0%....100% - 18 passes',
               'W3 BASE: 18.20s', '==PROF== Profiling "ntt_cm_dit_k8" - 1: 0%....100% - 17 passes']
        st = stages_from_lines(log, "blk_phase_b")
        ok([s[2] for s in st] == ["blk_phase_b", "blk_phase_b"] and st[1][3] == "17" and st[1][0] == 1
           and stages_from_lines(log)[0][2] == "run", f"stages from the pass's window ({st})")
        # runa on the synthetic trace, with the sampler's files
        db = os.path.join(d, "t.sqlite")
        build_synth_sqlite(db)
        rlog = os.path.join(d, "a.log")
        with open(rlog, "w") as f:
            f.write("BLOCK PHASES: execute 3.10 · build 7.60 · prep 1.20 · A 10.42 (wait 2.00 upload 1.00 commit 6.00 "
                    "retire 0.40) · B 7.70 (argue 4.00 open 2.50 tax 1.20 = upload 0.40 + encode 0.80)\n"
                    "W3 BASE: 18.18s · statement at 10.50s · plan + 3 leaves emitted by 12.00s (inside the base)\n"
                    "CARD HOLD #0 multi_prove: waited 0.000s · held 0.300s · t=[1800000009.300,1800000009.600]\n"
                    "W3 LEVEL 0: 2.62s wall · per program build+prove 0.10+0.80\n"
                    "W3 RECURSION: 4.16s after the base (tree 4.10s) · whole block 22.34s\n"
                    "W3 TREE VERIFY: ACCEPTED in 3.00s (derives every program and its artifacts)\n"
                    "test result: ok. 1 passed; 0 failed\n")
        smi = os.path.join(d, "smi.csv")
        with open(smi, "w") as f:
            for tenth in range(0, 100, 2):
                ts = datetime.fromtimestamp(T0_SYNTH / 1e9 + tenth / 10).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
                f.write(f"{ts}, {30000 if 55 <= tenth < 85 else 1000}, 50\n")
        cpre = os.path.join(d, "cpu")
        write_synth_cpu(cpre)
        rout = os.path.join(d, "runa")
        rc, printed = run(cmd_runa, argparse.Namespace(sqlite=db, log=rlog, workload="whir", out=rout, smi=smi,
                                                       cpu=cpre, bin_ms=500))
        ok(rc == 0 and "| phase_b | 3.00 |" in printed and "Stages from NVTX ranges" in printed
           and "whir base 18.18 s" in printed, f"runa exits 0 with a 3.0 s phase B ({printed[:700]!r})")
        srows = {r["stage"]: r for r in read_tsv(os.path.join(rout, "runa-whir-stages.tsv"))}
        ok(set(srows) == {"setup", "phase_a", "prepared", "phase_b", "base_other", "recursion", "verify", "other",
                          "a_commit", "b_argue", "hold:multi_prove", "whole"}, f"runa's stages ({sorted(srows)})")
        walls = {k: fnum(v["wall_s"]) for k, v in srows.items()}
        ok(abs(walls["setup"] - 1.0) < 1e-6 and abs(walls["phase_a"] - 3.5) < 1e-6 and abs(walls["prepared"] - 0.2) < 1e-6
           and abs(walls["base_other"] - 1.3) < 1e-6 and abs(walls["recursion"] - 0.6) < 1e-6
           and abs(walls["other"] - 0.2) < 1e-6 and abs(walls["verify"] - 0.15) < 1e-6
           and abs(walls["hold:multi_prove"] - 0.3) < 1e-6, f"runa's stage walls ({walls})")
        ok(abs(fnum(srows["phase_a"]["kernel_sum_s"]) - 0.3) < 1e-6 and srows["phase_b"]["launches"] == "3"
           and abs(fnum(srows["phase_a"]["copy_busy_pct"]) - 100 * 0.5 / 3.5) < 0.01,
           f"runa's kernels per stage ({srows['phase_a']['kernel_sum_s']}, {srows['phase_b']['launches']})")
        ok(abs(fnum(srows["phase_a"]["sm_active"]) - 80.0) < 1e-6 and abs(fnum(srows["phase_b"]["sm_active"]) - 30.0) < 1e-6
           and abs(fnum(srows["phase_b"]["warps_in_flight"]) - 15.0) < 1e-6, "runa's GPU metric means per stage")
        ok(abs(fnum(srows["phase_b"]["vram_max_mib"]) - 30000) < 1e-6 and abs(fnum(srows["setup"]["vram_max_mib"]) - 1000) < 1e-6,
           f"runa's VRAM per stage ({srows['phase_b']['vram_max_mib']})")
        ok(abs(fnum(srows["phase_a"]["cpu_cores"]) - 2.0) < 1e-6 and abs(fnum(srows["setup"]["cpu_cores"]) - 0.5) < 1e-6,
           f"runa's cores per stage ({srows['phase_a']['cpu_cores']}, {srows['setup']['cpu_cores']})")
        fam = {r["stage"]: r for r in read_tsv(os.path.join(rout, "runa-whir-stage-families.tsv"))}
        ok(abs(fnum(fam["phase_b"]["sumcheck_s"]) - 0.3) < 1e-6 and abs(fnum(fam["phase_b"]["whir-fold_s"]) - 0.4) < 1e-6
           and abs(fnum(fam["phase_b"]["lde_s"]) - 0.05) < 1e-6, f"kernel families per stage ({fam.get('phase_b')})")
        hold = read_tsv(os.path.join(rout, "runa-whir-holds.tsv"))
        ok(len(hold) == 1 and hold[0]["phase"] == "multi_prove" and hold[0]["stage"] == "recursion"
           and abs(fnum(hold[0]["kernel_busy_pct"]) - 100 / 3) < 0.1 and hold[0]["launches"] == "1",
           f"the card hold ({hold})")
        ok("## card holds" in printed and "| multi_prove | recursion | 1 | 0.30 | 0.10 | 33.3 | 0.20 |" in printed,
           "the card holds' table")
        cst = {r["stage"]: r for r in read_tsv(os.path.join(rout, "cpu-whir-by-stage.tsv"))}
        ok(abs(fnum(cst["phase_a"]["walker_s"]) - 3.5) < 1e-6 and abs(fnum(cst["phase_a"]["cores_busy"]) - 2.0) < 1e-6
           and "elf-beside_s" in cst["phase_a"], f"CPU by stage and role ({cst.get('phase_a')})")
        att = {r["label"]: r for r in read_tsv(os.path.join(rout, "runa-whir-nvtx-kernels.tsv"))}
        ok(att.get("blk_phase_b", {}).get("launches") == "3" and att["blk_phase_b"]["kind"] == "process"
           and abs(fnum(att["cardhold_multi_prove"]["kernel_s"]) - 0.1) < 1e-6
           and att.get("blk_walker", {}).get("kind") == "thread",
           f"kernels by NVTX label ({att})")
        cat = {(r["category"], r["stage"]): r for r in read_tsv(os.path.join(rout, "api-whir-by-category.tsv"))}
        ok(abs(fnum(cat[("sync", "phase_a")]["sum_s"]) - 0.3) < 1e-6 and abs(fnum(cat[("alloc", "phase_b")]["sum_s"]) - 0.2) < 1e-6
           and abs(fnum(cat[("copy_async", "phase_b")]["sum_s"]) - 0.1) < 1e-6 and ("host_pinned", "setup") in cat
           and cat[("launch", "phase_b")]["calls"] == "3", f"API seconds per category and stage ({sorted(cat)})")
        thr = read_tsv(os.path.join(rout, "api-whir-by-thread.tsv"))
        ok(any(r["thread"] == "tid 101 (driver-3)" and r["category"] == "alloc" for r in thr), "API per named thread")
        ts = read_tsv(os.path.join(rout, "runa-whir-timeseries.tsv"))
        ok(len(ts) == 20 and ts[6]["stage"] == "phase_a" and ts[12]["stage"] == "phase_b"
           and abs(fnum(ts[6]["cpu_cores"]) - 2.0) < 1e-6, f"the time series ({len(ts)}, {ts[6] if len(ts) > 6 else ''})")
        ok("RUNA whir: stages from NVTX ranges; card holds 1 (log 1)" in printed, "runa's closing line")
        rc, lh = run(cmd_loghead, argparse.Namespace(log=rlog, workload="whir"))
        ok(lh.startswith("ok\ttest result: ok. 1 passed; 0 failed\t1\twhir base 18.18 s"), f"loghead ({lh!r})")
        rc, lh = run(cmd_loghead, argparse.Namespace(log=rlog, workload="stark"))
        ok(lh.startswith("bad\t"), "loghead refuses a log without its workload's headline")
        ok(cpu_total_line(cpre).startswith("CPU 10.2 s over 10.0 s (1.0 cores), peak RSS 3.0 GiB"),
           f"cpu-total ({cpu_total_line(cpre)})")
        # no NVTX table: one window
        import sqlite3
        db2 = os.path.join(d, "t2.sqlite")
        build_synth_sqlite(db2)
        c2 = sqlite3.connect(db2)
        c2.execute("DROP TABLE NVTX_EVENTS")
        c2.commit()
        c2.close()
        rc, printed = run(cmd_runa, argparse.Namespace(sqlite=db2, log=rlog, workload="whir", out=os.path.join(d, "r2"),
                                                       smi=None, cpu=None, bin_ms=1000))
        s2 = {r["stage"]: r for r in read_tsv(os.path.join(d, "r2", "runa-whir-stages.tsv"))}
        ok(rc == 0 and "one window only" in printed and set(s2) == {"whole"}, f"runa without NVTX ({sorted(s2)})")
        # dry against the synthetic trace
        dout = os.path.join(d, "dry")
        rc, printed = run(cmd_dry, argparse.Namespace(plan=plan, workload="whir", sqlite=db, log=rlog, out=dout,
                                                      tag=None, modes="full"))
        ok(rc == 1 and "whir_none: nvtx no_such_range; matched 0 launches" in printed and "NO NVTX RANGE" in printed,
           "dry exits 1 on a pass whose range is not in the trace")
        drows = {r["pass"]: r for r in read_tsv(os.path.join(dout, "dry-whir.tsv"))}
        ok(set(drows) == {"whir_b_sum", "whir_a_ntt", "whir_all_ntt", "whir_b_leaves", "whir_none"},
           "dry evaluates this workload's full-mode passes only")
        ok(drows["whir_b_sum"]["profile"] == "1" and drows["whir_a_ntt"]["matched"] == "1"
           and drows["whir_all_ntt"]["matched"] == "2" and drows["whir_b_leaves"]["verdict"] == "NO MATCH",
           f"dry selections inside the NVTX windows ({drows})")
        rc, printed = run(cmd_dry, argparse.Namespace(plan=plan, workload="whir", sqlite=db, log=rlog, out=dout,
                                                      tag="q", modes="quick"))
        ok(rc == 0 and set(r["pass"] for r in read_tsv(os.path.join(dout, "dry-q.tsv"))) == {"whir_b_sum"},
           "dry --modes quick takes the quick passes")
        rc, printed = run(cmd_dry, argparse.Namespace(plan=plan, workload="stark", sqlite=db, log=rlog, out=dout,
                                                      tag="s", modes="quick"))
        ok(rc == 0 and "stark_lfm: nvtx cardhold_multi_prove; matched 1 launches" in printed,
           f"a hold-filtered pass selects the hold's launch ({printed[:300]!r})")
        # report
        send = os.path.join(d, "send")
        os.makedirs(os.path.join(send, "runa", "whir"))
        os.makedirs(os.path.join(send, "summary"))
        os.makedirs(os.path.join(send, "reference"))
        for fn in os.listdir(rout):
            with open(os.path.join(rout, fn)) as src, open(os.path.join(send, "runa", "whir", fn), "w") as dst:
                dst.write(src.read())
        with open(os.path.join(send, "summary", "kernels.md"), "w") as f:
            f.write(open(os.path.join(d, "sum0", "kernels.md")).read())
        with open(os.path.join(send, "reference", "runs.tsv"), "w") as f:
            f.write("workload\trc\tseconds\theadline\tcpu\tvram_max_mib\nwhir\t0\t60\twhir base 18.20 s\tCPU 1 s\t32110\n")
        rc, _ = run(cmd_report, argparse.Namespace(send=send))
        rep = open(os.path.join(send, "SUMMARY.md")).read()
        ok(rc == 0 and "## run A, whir" in rep and "## run B: per kernel, both workloads" in rep
           and "## reference runs" in rep and "| phase_b | 3.00 |" in rep, "report assembles SUMMARY.md")
        # scrub + check
        bd = os.path.join(d, "bundle")
        os.makedirs(os.path.join(bd, "ncu"))
        with open(os.path.join(bd, "ncu", "p.details.csv"), "w") as f:
            f.write(synth_details_csv(host="myhost.example"))
        with open(os.path.join(bd, "log.txt"), "w") as f:
            f.write("==PROF== Report: /home/alice/np-work/runs/x/keep/p.ncu-rep\nhost myhost.example ok\n")
        rc, scrub_out = run(cmd_scrub, argparse.Namespace(dir=bd),
                            stdin=io.StringIO("/home/alice/np-work\t<W>\n/home/alice\t<HOME>\nmyhost.example\t<HOST>\n"))
        txt = open(os.path.join(bd, "log.txt")).read()
        csvt = open(os.path.join(bd, "ncu", "p.details.csv")).read()
        ok("<W>/runs/x" in txt and "alice" not in txt and "<HOST> ok" in txt, "scrub replaces the literals")
        ok("Host Name" not in csvt and "myhost" not in csvt and "Host Name column dropped from 1" in scrub_out,
           "scrub drops the Host Name column")
        ok(len(read_details(os.path.join(bd, "ncu", "p.details.csv"))) == 3, "a scrubbed CSV still parses")

        def run_check(values):
            return run(cmd_check, argparse.Namespace(dir=bd, report=os.path.join(d, "report.txt")),
                       stdin=io.TextIOWrapper(io.BytesIO("\0".join(values).encode())))

        vals = ["hostname\tmyhost.example", "home\t/home/alice", "env:SOMEVAR\tvalue-of-somevar-123"]
        rc, msg = run_check(vals)
        ok(rc == 0 and msg.startswith("self-check: clean"), f"a scrubbed bundle is clean ({msg.strip()})")
        with open(os.path.join(bd, "log.txt"), "a") as f:
            f.write("leak value-of-somevar-123 here\n" + "gh" + "p_" + "A" * 36 + "\n")
        rc, msg = run_check(vals)
        rep = open(os.path.join(d, "report.txt")).read()
        ok(rc == 1 and "log.txt:3: env:SOMEVAR" in rep and "log.txt:4: credential marker" in rep
           and "value-of-somevar" not in msg + rep, "check refuses a planted value and marker, without printing them")
        with open(os.path.join(bd, "x.ncu-rep"), "wb") as f:
            f.write(b"\0\1\2")
        rc, msg = run_check([])
        ok(rc == 1 and "x.ncu-rep: an Nsight report" in open(os.path.join(d, "report.txt")).read(),
           "check refuses an Nsight report")
        # cargo-artifact
        msgs = [json.dumps({"reason": "compiler-artifact", "target": {"name": "lambda_vm_prover", "kind": ["lib"]},
                            "profile": {"test": True}, "executable": "/w/target/release/deps/lambda_vm_prover-1"}),
                json.dumps({"reason": "compiler-artifact", "target": {"name": "lambda_vm_prover", "kind": ["lib"]},
                            "profile": {"test": False}, "executable": None}),
                json.dumps({"reason": "build-script-executed",
                            "package_id": "path+file:///w/crypto/math-cuda#0.1.0", "out_dir": "/w/target/out"})]
        for want, ns in (("/w/target/release/deps/lambda_vm_prover-1", argparse.Namespace(exe="lambda_vm_prover", outdir=None)),
                         ("/w/target/out", argparse.Namespace(exe=None, outdir="math-cuda"))):
            rc, got = run(cmd_cargo_artifact, ns, stdin=io.StringIO("\n".join(msgs) + "\n"))
            ok(rc == 0 and got.strip() == want, f"cargo-artifact {want} ({got.strip()})")
        # cpu-sample on a real short process (Linux only: it reads /proc)
        if os.path.isdir("/proc/self/task"):
            import shutil
            import subprocess
            sl = shutil.which("sleep")
            if sl:
                exe = os.path.join(d, "np-selftest-sleep")
                shutil.copy(sl, exe)
                child = subprocess.Popen([exe, "1.2"])
                pre = os.path.join(d, "live")
                # the child stays a zombie until wait() below: the sampler must take that as an exit, not hang
                guard = threading.Timer(30.0, lambda: os.kill(os.getpid(), signal.SIGALRM))
                guard.start()
                try:
                    rc, _ = run(cmd_cpu_sample, argparse.Namespace(exe=exe, out=pre, interval_ms=100.0,
                                                                   mem_floor_mib=0, wait_s=10.0))
                finally:
                    guard.cancel()
                child.wait()
                live = read_cpu(pre, 0)
                ok(rc == 0 and live is not None and len(live["proc"]) >= 3 and live["tid"],
                   f"cpu-sample follows a live process ({rc}, {len(live['proc']) if live else 0} samples)")
    for f in fails:
        print("SELFTEST FAIL: " + f)
    print("SELFTEST " + ("GREEN" if not fails else f"RED ({len(fails)} failure(s))"))
    return 1 if fails else 0


# ---------------------------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--plan")
    s.add_argument("--ncu-dir", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("stages")
    s.add_argument("--log", required=True)
    s.add_argument("--nvtx", default="-", help="the pass's process-wide NVTX range, or -")
    s = sub.add_parser("runa")
    s.add_argument("--sqlite", required=True)
    s.add_argument("--log", required=True)
    s.add_argument("--workload", required=True, choices=WORKLOADS)
    s.add_argument("--out", required=True)
    s.add_argument("--smi", help="nvidia-smi samples: timestamp, memory.used, utilization.gpu (csv, noheader)")
    s.add_argument("--cpu", help="the cpu-sample prefix of this run")
    s.add_argument("--bin-ms", type=float, default=100.0, help="the time series' bin (ms, >= 1)")
    s = sub.add_parser("dry")
    s.add_argument("--plan", required=True)
    s.add_argument("--workload", required=True, choices=WORKLOADS)
    s.add_argument("--sqlite", required=True)
    s.add_argument("--log", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--tag", help="names the outputs dry-<tag>.md/.tsv (default: the workload)")
    s.add_argument("--modes", default="full", choices=RUN_MODES, help="the passes of this run mode")
    s = sub.add_parser("cpu-sample")
    s.add_argument("--exe", required=True)
    s.add_argument("--out", required=True, help="the files' prefix: PREFIX.proc.tsv, PREFIX.tid.tsv, PREFIX.watchdog")
    s.add_argument("--interval-ms", type=float, default=100.0)
    s.add_argument("--mem-floor-mib", type=int, default=0, help="0: no watchdog")
    s.add_argument("--wait-s", type=float, default=300.0, help="how long to wait for the process to appear")
    s = sub.add_parser("cpu-total")
    s.add_argument("--cpu", required=True)
    s = sub.add_parser("loghead")
    s.add_argument("--log", required=True)
    s.add_argument("--workload", required=True, choices=WORKLOADS)
    s = sub.add_parser("report")
    s.add_argument("--send", required=True)
    s = sub.add_parser("plan-check")
    s.add_argument("--plan", required=True)
    s = sub.add_parser("cargo-artifact")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--exe")
    g.add_argument("--outdir")
    s = sub.add_parser("scrub")
    s.add_argument("--dir", required=True)
    s = sub.add_parser("check")
    s.add_argument("--dir", required=True)
    s.add_argument("--report", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    return {"summary": cmd_summary, "stages": cmd_stages, "runa": cmd_runa, "dry": cmd_dry, "report": cmd_report,
            "cpu-sample": cmd_cpu_sample, "cpu-total": cmd_cpu_total, "loghead": cmd_loghead,
            "plan-check": cmd_plan_check, "cargo-artifact": cmd_cargo_artifact, "scrub": cmd_scrub,
            "check": cmd_check, "selftest": lambda _a: selftest()}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
NOEPOCH_COUNTERS_SUMMARY_PY_EOF
}

print_tool() { # the embedded tool, byte for byte (compare it with noepoch_counters_summary.py)
  # shellcheck disable=SC2016 # a literal $TOOL in the pattern
  sed -n '/^  cat > "\$TOOL" <<.NOEPOCH_COUNTERS_SUMMARY_PY_EOF.$/,/^NOEPOCH_COUNTERS_SUMMARY_PY_EOF$/p' "$0" | sed '1d;$d'
}

plan_source() { if [ -n "$NP_PLAN" ]; then cat "$NP_PLAN"; else default_plan; fi; }
PLAN_FILE=""
load_plan() {
  PLAN_FILE="$W/tmp/plan.tsv"
  plan_source > "$PLAN_FILE"
}

# ---------------------------------------------------------------------------------------------------
# environments: the build's and the profiled process's are explicit (env -i + an allowlist), so no knob,
# wrapper or target directory from the caller's shell reaches them, and Nsight (which stores the profiled
# process's environment in its reports) records only these.

BUILD_ENV=() RUN_ENV=() WL_ENV=() WL_TEST="" WL_BIN="" WL_REPO="" ARCH_ENV=() NVTX_LIB=""
BIN_W="" BIN_S=""
build_env() {
  BUILD_ENV=("PATH=$PATH" "HOME=$W/home" "RUSTUP_HOME=$RUSTUP_HOME_REAL" "RUSTUP_AUTO_INSTALL=0"
    "CARGO_HOME=$W/cargo-home" "CARGO_TARGET_DIR=$TARGET" "TMPDIR=$W/tmp" "LC_ALL=C" "LANG=C" "RUSTC_WRAPPER="
    "RUSTC_WORKSPACE_WRAPPER=" "CARGO_TERM_COLOR=never" "CARGO_TERM_PROGRESS_WHEN=never" "LAMBDA_VM_NVCC_LINEINFO=1")
  local v
  for v in CUDA_HOME CUDA_PATH LD_LIBRARY_PATH; do
    if [ -n "${!v:-}" ]; then BUILD_ENV+=("$v=${!v}"); fi
  done
}
run_env() { # run_env [nvtx]: the profiled process's system variables (+ the NVTX library)
  RUN_ENV=("PATH=$PATH" "HOME=$W/home" "TMPDIR=$W/tmp" "LC_ALL=C" "LANG=C"
    "CUDA_DEVICE_ORDER=PCI_BUS_ID" "CUDA_VISIBLE_DEVICES=$NP_GPU")
  local v
  for v in CUDA_HOME CUDA_PATH LD_LIBRARY_PATH; do
    if [ -n "${!v:-}" ]; then RUN_ENV+=("$v=${!v}"); fi
  done
  if [ "${1:-}" = nvtx ] && [ -n "$NVTX_LIB" ]; then RUN_ENV+=("LAMBDA_VM_NVTX_LIB=$NVTX_LIB"); fi
}
wl_env() { # wl_env whir|stark record|runb: the workload's knobs, as its FAST record arms run it, or as run B runs it
  local elf="$W/fixtures/ethrex_8f826601.elf" input="$W/fixtures/ethrex_mainnet_25368371_573004e6.bin"
  local jem="_RJEM_MALLOC_CONF=dirty_decay_ms:-1,muzzy_decay_ms:-1"
  case "$1" in
    whir)
      # w4c-block-box.sh's env (FAST 419, the 22.30 s record), LFM_CARD_TRACE for the hold lines
      WL_ENV=("BLOCK_WHIR_ELF=$elf" "BLOCK_WHIR_INPUT=$input" "LAMBDA_VM_MAX_ROWS_LOG2=21"
        "LAMBDA_VM_MEMPOOL_RELEASE_MB=0" "LFM_WHIR_RETENTION=1" "LAMBDA_VM_WHIR_HASH=rpx" "TABLE_PARALLELISM=4"
        "LFM_PRECOMPUTED_TREE_CACHE_CAP=64" "LFM_EXEC_PARALLEL=1" "LAMBDA_VM_BASE_SPLIT=1"
        "LAMBDA_VM_GRIND_SCAN_FACTOR=8" "LAMBDA_VM_GRIND_GRID=1024" "$jem" "LFM_CARD_TRACE=1"
        "CARGO_MANIFEST_DIR=$REPO/prover")
      if [ "$2" = runb ] && [ -n "$NP_NCU_VRAM_MB" ]; then WL_ENV+=("LAMBDA_VM_VRAM_BUDGET_MB=$NP_NCU_VRAM_MB"); fi
      # shellcheck disable=SC2206 # blank-separated NAME=VALUE words, validated in knob_defaults
      WL_ENV+=($NP_WHIR_KNOBS)
      WL_TEST="$WHIR_TEST" WL_BIN="$BIN_W" WL_REPO="$REPO" ;;
    stark)
      # s3-cj.sh's tree arm env (FAST 456, the 31.78 s record): BLOCK_ENV + B2, LFM_CARD_TRACE for the hold lines
      local budget="$STARK_VRAM_MB"
      if [ "$2" = runb ] && [ -n "$NP_NCU_VRAM_MB" ]; then budget="$NP_NCU_VRAM_MB"; fi
      WL_ENV=("NOEPOCH_ELF=$elf" "NOEPOCH_INPUT=$input" "TABLE_PARALLELISM=8" "LAMBDA_VM_VRAM_BUDGET_MB=$budget"
        "LAMBDA_VM_MAX_ROWS_LOG2=21" "LFM_PROVE_SPLIT=1" "LAMBDA_VM_BASE_SPLIT=1" "LFM_EXEC_PARALLEL=1"
        "LFM_PRECOMPUTED_TREE_CACHE_CAP=64" "$jem" "LFM_TREE_SIBLINGS_L0=8" "LFM_TREE_SIBLINGS=4"
        "LAMBDA_VM_GATE_PACKING=1" "LFM_CARD_TRACE=1" "CARGO_MANIFEST_DIR=$REPO_S/prover")
      # shellcheck disable=SC2206 # as above
      WL_ENV+=($NP_STARK_KNOBS)
      WL_TEST="$STARK_TEST" WL_BIN="$BIN_S" WL_REPO="$REPO_S" ;;
    *) die 2 "unknown workload '$1'" ;;
  esac
}

# ---------------------------------------------------------------------------------------------------
# preflight

PF_FAILS=0 PF_WARNS=0
chk() { # chk PASS|WARN|FAIL|INFO "what: detail"
  printf '%-4s %s\n' "$1" "$2" | tee -a "$KEEP/preflight.txt"
  case "$1" in FAIL) PF_FAILS=$((PF_FAILS + 1)) ;; WARN) PF_WARNS=$((PF_WARNS + 1)) ;; esac
}

NCU_BIN="" NSYS_BIN="" NVCC="" GPU_NAME="gpu" GPU_CC="" GPU_VRAM=0 GPU_DRIVER="" DRIVER_CUDA="" NVCC_REL=""
NCU_VER="" NSYS_VER="" NCU_HAS_KILL=0 NCU_METRICS_OK="" NSYS_METRICS_FLAG="" KILL_PROBE="not probed"
NGPUS=0 WANT_NSYS=0 WANT_METRICS=0 WANT_NCU=0 NVTX_FILTER="not probed" NVTX_FILTER_OK=1
preflight() {
  local n t got reading mt cg eff av fr tl sev

  # -- host and plain tools
  if [ "$(host_os)" = Linux ]; then
    chk PASS "host: $(host_os) · $(awk -F= '$1 == "PRETTY_NAME" { gsub(/"/, "", $2); print $2 }' /etc/os-release 2>/dev/null || true)"
  else
    chk FAIL "host: $(host_os): this needs Linux (the GPU runs and the Nsight tools)"
  fi
  chk INFO "cpu: $(cpu_model) · $(nproc 2>/dev/null || echo '?') threads"
  if [ "${BASH_VERSINFO[0]}" -ge 5 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -ge 4 ]; }; then
    chk PASS "bash: $BASH_VERSION"
  else chk FAIL "bash $BASH_VERSION: 4.4 or newer is required"; fi
  for t in git curl tar awk sed timeout python3 df nvidia-smi env; do
    if ! command -v "$t" >/dev/null 2>&1; then chk FAIL "tool: $t not found"; fi
  done
  if ! command -v sha256sum >/dev/null 2>&1 && ! command -v shasum >/dev/null 2>&1; then chk FAIL "tool: neither sha256sum nor shasum"; fi
  if python3 -c 'import sqlite3, sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
    if python3 "$TOOL" selftest > "$KEEP/tool-selftest.log" 2>&1; then chk PASS "python3: $(python3 --version 2>&1) with sqlite3; the summary tool's selftest is GREEN"
    else chk FAIL "python3: the summary tool's selftest is RED here: see $KEEP/tool-selftest.log"; fi
  else
    chk FAIL "python3 >= 3.8 with the sqlite3 module is required (the summaries read the Nsight exports)"
  fi
  if python3 "$TOOL" plan-check --plan "$PLAN_FILE" > "$KEEP/plan-check.log" 2>&1; then chk PASS "plan: $(tail -1 "$KEEP/plan-check.log")"
  else chk FAIL "plan: $(grep -c 'PLAN ERROR' "$KEEP/plan-check.log" || true) error(s): see $KEEP/plan-check.log"; fi

  # -- the GPU
  if command -v nvidia-smi >/dev/null 2>&1; then
    NGPUS="$({ nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true; } | awk 'NF { n++ } END { print n + 0 }')"
    GPU_NAME="$(gpu_field name)"
    if [ -z "$GPU_NAME" ]; then
      chk FAIL "gpu: no NVIDIA GPU with index $NP_GPU ($NGPUS visible to nvidia-smi)"
      GPU_NAME="gpu"
    else
      GPU_VRAM="$(gpu_field memory.total)"; case "$GPU_VRAM" in ''|*[!0-9]*) GPU_VRAM=0 ;; esac
      GPU_CC="$(gpu_field compute_cap)"; GPU_DRIVER="$(gpu_field driver_version)"
      DRIVER_CUDA="$({ nvidia-smi 2>/dev/null || true; } | first_match 'CUDA Version: [0-9]+[.][0-9]+' | awk '{ print $3 }')"
      chk PASS "gpu $NP_GPU of $NGPUS: $GPU_NAME · compute capability $GPU_CC · $GPU_VRAM MiB · driver $GPU_DRIVER"
      if [ -n "$DRIVER_CUDA" ] && ver_ge "$DRIVER_CUDA" "$MIN_CUDA"; then chk PASS "driver: CUDA $DRIVER_CUDA >= $MIN_CUDA"
      else chk FAIL "driver: CUDA '${DRIVER_CUDA:-?}' < $MIN_CUDA (the prover's CUDA bindings need the 12.8 driver API)"; fi
      if [ "$GPU_VRAM" -lt 30000 ]; then
        chk WARN "gpu: $GPU_VRAM MiB, less than the record's 32 GB RTX 5090: the runs may not fit, and what runs on the card differs from the record"
      fi
      if [ "$NGPUS" -gt 1 ]; then
        ARCH_ENV=("CUDARC_NVCC_ARCH=sm_${GPU_CC//./}")
        chk INFO "build: $NGPUS GPUs; the kernels are compiled for GPU $NP_GPU (sm_${GPU_CC//./})"
      fi
    fi
    if reading="$(gpu_reading)"; then chk PASS "idle: $reading"
    else chk FAIL "idle: $reading: something else holds the GPU (under $NP_IDLE_MIB MiB and no compute process counts as idle); stop it and run again"; fi
  else
    chk FAIL "gpu: nvidia-smi not found (no NVIDIA driver?)"
  fi

  # -- CUDA toolkit: math-cuda's build.rs looks for nvcc in exactly one place
  NVCC="$(cuda_home)/bin/nvcc"
  if [ -x "$NVCC" ]; then
    NVCC_REL="$({ "$NVCC" --version 2>/dev/null || true; } | first_match 'release [0-9]+[.][0-9]+' | awk '{ print $2 }')"
    if [ -n "$NVCC_REL" ] && ver_ge "$NVCC_REL" "$MIN_CUDA"; then chk PASS "nvcc: $NVCC (release $NVCC_REL)"
    else chk FAIL "nvcc: $NVCC is release '${NVCC_REL:-?}', < $MIN_CUDA"; fi
  else
    chk FAIL "nvcc: not at $NVCC. math-cuda's build.rs looks ONLY there (CUDA_HOME, else CUDA_PATH, else /usr/local/cuda), and without it the build writes EMPTY kernels and runs everything on the CPU. Install the CUDA toolkit or set CUDA_HOME"
  fi

  # -- Nsight Systems (run A), with the GPU-metrics option it spells
  WANT_NSYS="$NP_RUN_A"; WANT_METRICS=0; WANT_NCU=0
  if [ "$NP_RUN_A" = 1 ] && [ "$NP_DRY" = 0 ]; then WANT_METRICS=1; fi
  if [ "$NP_RUN_B" = 1 ] && [ "$NP_DRY" = 0 ]; then WANT_NCU=1; fi
  NSYS_BIN="$(find_tool nsys || true)"
  if [ -n "$NSYS_BIN" ]; then
    NSYS_VER="$(tool_version "$NSYS_BIN")"
    if [ -n "$NSYS_VER" ] && ver_ge "$NSYS_VER" "$MIN_NSIGHT"; then chk PASS "nsys: $NSYS_BIN ($NSYS_VER)"
    elif [ "$WANT_NSYS" = 1 ]; then chk FAIL "nsys: $NSYS_BIN is ${NSYS_VER:-?}, < $MIN_NSIGHT (Blackwell needs 2025.1 or newer)"
    else chk INFO "nsys: $NSYS_BIN is ${NSYS_VER:-?} (not used: NP_RUN_A=0)"; fi
    got="$({ "$NSYS_BIN" profile --help 2>&1 || true; })"
    case "$got" in
      *--gpu-metrics-devices*) NSYS_METRICS_FLAG="--gpu-metrics-devices" ;;
      *--gpu-metrics-device*) NSYS_METRICS_FLAG="--gpu-metrics-device" ;;
      *) NSYS_METRICS_FLAG="" ;;
    esac
    if [ "$WANT_METRICS" = 1 ] && [ -z "$NSYS_METRICS_FLAG" ]; then chk FAIL "nsys: this nsys has no GPU-metrics option (run A samples SM, DRAM and PCIe throughput with it)"; fi
  elif [ "$WANT_NSYS" = 1 ]; then
    chk FAIL "nsys (Nsight Systems) not found on PATH, in $(cuda_home)/bin or /opt/nvidia/nsight-systems/*; install it or set NSYS=/path/to/nsys"
  fi
  NVTX_LIB="$(find_nvtx_lib || true)"
  if [ -n "$NVTX_LIB" ]; then chk PASS "nvtx: $NVTX_LIB (handed to the runs as LAMBDA_VM_NVTX_LIB: the phases and the card holds are named ranges)"
  elif [ "$NP_FETCH_NVTX" = 1 ] && NVTX_LIB="$(fetch_nvtx_lib)"; then
    chk PASS "nvtx: $NVTX_LIB, fetched (nvidia-nvtx-cu12 12.8.90, sha256-checked) and handed to the runs as LAMBDA_VM_NVTX_LIB"
  else
    NVTX_LIB=""
    chk FAIL "nvtx: no libnvToolsExt found (CUDA 12.9 and later ship none)$([ "$NP_FETCH_NVTX" = 1 ] && echo " and the fetch failed (see $W/nvtx/curl.log)"). Every stage, card hold and run-B window is an NVTX range, so this run needs one: export LAMBDA_VM_NVTX_LIB=/path/to/libnvToolsExt.so.1"
  fi

  # -- Nsight Compute, recognised by its banner; counters proven on one real kernel
  NCU_BIN="$(find_tool ncu || true)"
  sev=INFO
  if [ "$WANT_NCU" = 1 ]; then sev=FAIL; fi
  if [ -n "$NCU_BIN" ]; then
    NCU_VER="$(tool_version "$NCU_BIN")"
    if [ -n "$NCU_VER" ] && ver_ge "$NCU_VER" "$MIN_NSIGHT"; then chk PASS "ncu: $NCU_BIN ($NCU_VER); clock control for run B: $NP_CLOCK"
    else chk "$sev" "ncu: $NCU_BIN is ${NCU_VER:-?}, < $MIN_NSIGHT (Blackwell needs 2025.1 or newer)"; fi
    got="$({ "$NCU_BIN" --help 2>&1 || true; })"
    case "$got" in *--kill*) NCU_HAS_KILL=1 ;; *) NCU_HAS_KILL=0 ;; esac
  else
    chk "$sev" "ncu (Nsight Compute) not found on PATH, in $(cuda_home)/bin or /opt/nvidia/nsight-compute/*; install it or set NCU=/path/to/ncu (an 'ncu' that is npm-check-updates does not count)"
  fi
  if [ -x "$NVCC" ] && [ -n "$GPU_CC" ]; then
    if build_probe; then
      probe_counters
      probe_nsys
    else
      chk FAIL "probe: nvcc could not build the one-kernel CUDA probe: see $W/probe/build.log"
    fi
  fi

  # -- host memory and disk
  mt="$(mem_total_gib)"; cg="$(cgroup_limit_gib)"; eff="$mt"; av="$(mem_avail_gib)"
  if [ -n "$cg" ] && { [ -z "$mt" ] || ! num_ge "$cg" "$mt"; }; then eff="$cg"; fi
  if [ -z "$eff" ] || [ -z "$av" ]; then chk FAIL "ram: cannot read /proc/meminfo"
  elif ! num_ge "$eff" 50; then chk FAIL "ram: $eff GiB usable (MemTotal ${mt:-?}, cgroup limit ${cg:-none}) < 50 GiB: the WHIR run peaks near 45 GiB on the host"
  elif ! num_ge "$av" "$NP_MIN_AVAIL_GIB"; then chk FAIL "ram: $av GiB available now (MemTotal $mt GiB), < $NP_MIN_AVAIL_GIB: the WHIR run peaks near 45 GiB on the host; close big programs (browsers, IDEs, VMs) and run again"
  elif ! num_ge "$av" 52; then chk WARN "ram: $av GiB available (MemTotal $mt GiB): enough for the 45 GiB peak, with little margin; keep other programs closed during the run (the watchdog ends a run under $NP_MEM_FLOOR_MIB MiB available)"
  else chk PASS "ram: $av GiB available (MemTotal $mt GiB, cgroup limit ${cg:-none}); need >= $NP_MIN_AVAIL_GIB before every run"; fi
  fr="$(free_gib "$W")"
  if [ -z "$fr" ]; then chk WARN "disk: cannot read the free space in $W"
  elif ! num_ge "$fr" 20; then chk FAIL "disk: $fr GiB free in $W; need >= 30 (clone, cargo cache, two release CUDA builds, the traces)"
  elif ! num_ge "$fr" 30; then chk WARN "disk: $fr GiB free in $W; 30 recommended"
  else chk PASS "disk: $fr GiB free in $W"; fi

  # -- Rust: the prover's toolchain, checked in the build's own environment (no guest is built)
  build_env
  if command -v rustup >/dev/null 2>&1 && command -v cargo >/dev/null 2>&1; then
    tl="$(env -i "${BUILD_ENV[@]}" rustup toolchain list 2>/dev/null || true)"
    case "$tl" in
      *"$RUST_STABLE"*) chk PASS "rust: toolchain $RUST_STABLE (the prover's rust-toolchain.toml) installed" ;;
      *) chk FAIL "rust: toolchain $RUST_STABLE missing. Install it: rustup toolchain install $RUST_STABLE --profile default" ;;
    esac
  else
    chk FAIL "rust: rustup/cargo not found on PATH (https://rustup.rs), needed for $RUST_STABLE"
  fi

  # -- what this shell would have leaked into the runs (names only; none of it is passed on)
  n="$(env | awk -F= '/^(LAMBDA_VM_|LFM_|ZF_|NOEPOCH_|BLOCK_WHIR_|W3_|A_BUNDLE|A_CACHE|TABLE_PARALLELISM=|RAYON_|_RJEM_|MALLOC_CONF=|RUSTFLAGS=|CARGO_BUILD_|CARGO_ENCODED_RUSTFLAGS=|CARGO_TARGET_DIR=|RUSTC_WRAPPER=)/ { printf "%s ", $1 }')"
  if [ -n "$n" ]; then chk INFO "env: set in this shell and NOT passed to the build or the runs (they get an explicit environment; an NVTX library is handed over by path): $n"
  else chk PASS "env: no prover, cargo or allocator knob set in this shell"; fi

  if [ "$PF_FAILS" -eq 0 ]; then chk INFO "PREFLIGHT: PASS ($PF_WARNS warning(s))"
  else chk INFO "PREFLIGHT: FAIL ($PF_FAILS failure(s), $PF_WARNS warning(s)): fix the FAIL lines above and run again"; fi
}

build_probe() { # the CUDA probe: two kernels, two shapes, and one launch inside a process-wide NVTX range, from another thread
  local cc="${GPU_CC//./}"
  cat > "$W/probe/ncuprobe.cu" <<'CU'
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <thread>
extern "C" __global__ void ncuprobe_a(float *x) { x[threadIdx.x] += 1.0f; }
extern "C" __global__ void ncuprobe_b(float *x) { x[threadIdx.x] += 2.0f; }
typedef uint64_t (*range_start_t)(const char *);
typedef void (*range_end_t)(uint64_t);
int main() {
  // The prover's way: dlopen the NVTX library named by LAMBDA_VM_NVTX_LIB (a profiler injects through it).
  range_start_t start = nullptr;
  range_end_t end = nullptr;
  const char *lib = std::getenv("LAMBDA_VM_NVTX_LIB");
  if (lib) {
    void *h = dlopen(lib, RTLD_NOW);
    if (h) {
      start = (range_start_t)dlsym(h, "nvtxRangeStartA");
      end = (range_end_t)dlsym(h, "nvtxRangeEnd");
    }
  }
  float *d = nullptr;
  if (cudaMalloc(&d, 1024 * sizeof(float)) != cudaSuccess) { std::printf("ncuprobe: cudaMalloc failed\n"); return 2; }
  cudaMemset(d, 0, 1024 * sizeof(float));
  ncuprobe_a<<<1, 64>>>(d);
  // The range opens on this thread and the launch inside it comes from another, as a card hold's kernels
  // come from the holder's table threads.
  uint64_t id = (start && end) ? start("np_probe_range") : 0;
  std::thread inside([d]() {
    ncuprobe_b<<<1, 64>>>(d);
    cudaDeviceSynchronize();
  });
  inside.join();
  if (start && end) end(id);
  ncuprobe_a<<<2, 64>>>(d);
  ncuprobe_a<<<1, 64>>>(d);
  cudaError_t e = cudaDeviceSynchronize();
  std::printf("ncuprobe: done (%s) nvtx=%s\n", cudaGetErrorString(e), (start && end) ? "yes" : "no");
  return e == cudaSuccess ? 0 : 3;
}
CU
  build_env
  env -i "${BUILD_ENV[@]}" timeout 300 "$NVCC" -O2 -std=c++17 -arch="sm_$cc" -o "$W/probe/ncuprobe" "$W/probe/ncuprobe.cu" -ldl -lpthread \
    > "$W/probe/build.log" 2>&1
}

ncu_probe() { # ncu_probe LABEL NCU-ARGS...: profile the probe; prints unlocked | locked | unknown (rc N); log in probe/LABEL.log
  local label="$1" rc=0 out
  shift
  run_env nvtx
  env -i "${RUN_ENV[@]}" timeout 300 "$NCU_BIN" "$@" "$W/probe/ncuprobe" > "$W/probe/$label.log" 2>&1 || rc=$?
  out="$(cat "$W/probe/$label.log")"
  case "$out" in
    *ERR_NVGPUCTRPERM*) echo "locked" ;;
    *'==PROF== Profiling "ncuprobe_'*) if [ "$rc" -eq 0 ]; then echo "unlocked"; else echo "unknown (ncu rc=$rc after profiling)"; fi ;;
    *) echo "unknown (ncu rc=$rc, no probe kernel profiled)" ;;
  esac
}

probe_counters() {
  local res fix m ok="" rc all n na nb dropped=""
  fix="Fix: run this script as root (sudo -E bash noepoch_counters.sh), or open the counters to all users: as root, echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' > /etc/modprobe.d/nvidia-profiling.conf, then update-initramfs -u (dracut -f on Fedora/RHEL) and reboot"
  if [ -z "$NCU_BIN" ]; then
    if [ "$WANT_NCU" = 1 ]; then chk FAIL "counters: not probed (no ncu)"; fi
    return 0
  fi
  if [ "$WANT_NCU" = 0 ]; then
    # run B's full flag sets (every metric, unvalidated), on the probe: ncu parses every flag before it
    # touches the counters, so on a locked box `locked` still says this ncu accepts the passes' command lines
    NCU_METRICS_OK="$NP_METRICS"
    ncu_args window 0 1 np_probe_range 'ncuprobe_[ab]' "$W/probe/flags_window"
    res="$(ncu_probe flags_window "${NCU_ARGS[@]}")"
    chk INFO "ncu flags, window passes with an NVTX window: $res (a box with closed counters: locked = flags accepted, then counters refused)"
    ncu_args config 0 1 - 'ncuprobe_a' "$W/probe/flags_config"
    res="$(ncu_probe flags_config "${NCU_ARGS[@]}")"
    chk INFO "ncu flags, config passes: $res (as above)"
    NVTX_FILTER="not probed (counters closed)"
    return 0
  fi
  res="$(ncu_probe counters --section SpeedOfLight -k 'regex:^ncuprobe_a$' -c 1)"
  if [ "$res" != unlocked ]; then
    if [ "$res" = locked ]; then chk FAIL "counters: ncu cannot read this GPU's performance counters (ERR_NVGPUCTRPERM). $fix"
    else chk FAIL "counters: $res: see $W/probe/counters.log"; fi
    return 0
  fi
  chk PASS "counters: readable (ncu profiled the probe kernel)"
  # the explicit metrics, narrowed to what this ncu and GPU collect (one unknown name makes ncu profile nothing)
  all="$NP_METRICS"
  rc=0
  run_env nvtx
  env -i "${RUN_ENV[@]}" timeout 300 "$NCU_BIN" --metrics "$all" -k 'regex:^ncuprobe_a$' -c 1 "$W/probe/ncuprobe" \
    > "$W/probe/metrics_all.log" 2>&1 || rc=$?
  if [ "$rc" -eq 0 ] && ! grep -q '==ERROR==' "$W/probe/metrics_all.log"; then
    for m in ${all//,/ }; do
      if grep -qF -- "$m" "$W/probe/metrics_all.log"; then ok="${ok:+$ok,}$m"; else dropped="$dropped $m"; fi
    done
  else
    for m in ${all//,/ }; do
      rc=0
      env -i "${RUN_ENV[@]}" timeout 120 "$NCU_BIN" --metrics "$m" -k 'regex:^ncuprobe_a$' -c 1 "$W/probe/ncuprobe" \
        > "$W/probe/metric.log" 2>&1 || rc=$?
      if [ "$rc" -eq 0 ] && ! grep -q '==ERROR==' "$W/probe/metric.log" && grep -qF -- "$m" "$W/probe/metric.log"; then
        ok="${ok:+$ok,}$m"
      else dropped="$dropped $m"; fi
    done
  fi
  NCU_METRICS_OK="$ok"
  n="$(printf '%s' "$ok" | awk -F, '{ print (length($0) ? NF : 0) }')"
  chk INFO "ncu metrics: $n of $(printf '%s' "$all" | awk -F, '{ print NF }') collectable here; dropped:${dropped:- none}"
  # the passes' exact flags, on the probe: a window pass filtered by the probe's NVTX range must profile
  # ncuprobe_b (inside it) and never ncuprobe_a (outside it)
  ncu_args window 0 4 np_probe_range 'ncuprobe_[ab]' "$W/probe/window"
  res="$(ncu_probe window "${NCU_ARGS[@]}")"
  na="$(grep -c '^==PROF== Profiling "ncuprobe_a' "$W/probe/window.log" || true)"
  nb="$(grep -c '^==PROF== Profiling "ncuprobe_b' "$W/probe/window.log" || true)"
  if [ "$res" = unlocked ] && [ -s "$W/probe/window.ncu-rep" ] && [ "$nb" = 1 ] && [ "$na" = 0 ]; then
    NVTX_FILTER="works (only the launch inside the range was profiled)"
    chk PASS "ncu: the window passes' exact flags (--nvtx --nvtx-include <range>, --clock-control $NP_CLOCK) profile exactly the probe launch inside its process-wide NVTX range, made from another thread than the one that opened it"
  else
    NVTX_FILTER="FAILED ($res; inside $nb, outside $na)"
    NVTX_FILTER_OK=0
    chk WARN "ncu: NVTX filtering on the probe: $res, $nb launch(es) inside the range and $na outside profiled (want 1 and 0): run B's NVTX-windowed passes will be skipped (run A still answers the stage and hold questions); see $W/probe/window.log"
  fi
  ncu_args config 0 1 - 'ncuprobe_a' "$W/probe/config"
  res="$(ncu_probe config "${NCU_ARGS[@]}")"
  n="$(grep -c '^==PROF== Profiling' "$W/probe/config.log" || true)"
  if [ "$res" = unlocked ] && [ "$n" = 2 ]; then chk PASS "ncu per-launch-config: the config passes' exact flags profile one launch of each of the probe's 2 shapes"
  elif [ "$res" = unlocked ]; then chk WARN "ncu per-launch-config: $n probe launches profiled, expected 2 (one per shape of ncuprobe_a): see $W/probe/config.log"
  else chk FAIL "ncu: the config passes' flags (--filter-mode per-launch-config) failed on the probe ($res): see $W/probe/config.log"; fi
  if [ "$NCU_HAS_KILL" = 1 ]; then
    if grep -q 'ncuprobe: done' "$W/probe/window.log"; then KILL_PROBE="accepted, the probe still finished"
    else KILL_PROBE="works (the probe was ended after its window)"; fi
    chk INFO "ncu --kill: $KILL_PROBE"
  else
    KILL_PROBE="not supported"
    chk INFO "ncu --kill: not in this ncu's --help: window passes run their workload to the end (slower)"
  fi
}

probe_nsys() { # run A's exact nsys flags on the probe: the process-wide NVTX range, and GPU metrics when wanted, reach the export
  local rc=0 res
  if [ "$WANT_NSYS" = 0 ] || [ -z "$NSYS_BIN" ]; then return 0; fi
  if [ "$WANT_METRICS" = 1 ] && [ -z "$NSYS_METRICS_FLAG" ]; then return 0; fi
  nsys_args "$W/probe/nsysprobe" "$WANT_METRICS"
  run_env nvtx
  env -i "${RUN_ENV[@]}" timeout 300 "$NSYS_BIN" "${NSYS_ARGS[@]}" "$W/probe/ncuprobe" > "$W/probe/nsys.log" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ] || [ ! -s "$W/probe/nsysprobe.nsys-rep" ]; then
    if grep -q 'ERR_NVGPUCTRPERM\|insufficient privilege\|permission' "$W/probe/nsys.log" 2>/dev/null && [ "$WANT_METRICS" = 1 ]; then
      chk FAIL "nsys: GPU metrics need the same counter access as ncu (see the counters line above): see $W/probe/nsys.log"
    else chk FAIL "nsys: run A's flags failed on the probe (rc=$rc): see $W/probe/nsys.log"; fi
    return 0
  fi
  rc=0
  timeout 300 "$NSYS_BIN" export --type sqlite --force-overwrite true --output "$W/probe/nsysprobe.sqlite" \
    "$W/probe/nsysprobe.nsys-rep" >> "$W/probe/nsys.log" 2>&1 || rc=$?
  res="$(python3 - "$W/probe/nsysprobe.sqlite" <<'PY' 2>>"$W/probe/nsys.log" || true
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
def q(sql):
    try:
        return db.execute(sql).fetchone()[0]
    except sqlite3.Error:
        return 0
m = q("SELECT count(*) FROM GPU_METRICS")
r = q("SELECT count(*) FROM NVTX_EVENTS e LEFT JOIN StringIds s ON s.id = e.textId "
      "WHERE COALESCE(e.text, s.value) = 'np_probe_range' AND e.eventType = 60")
print(f"{m} {r}")
PY
)"
  local nm="${res% *}" nr="${res#* }"
  if [ "$rc" -ne 0 ] || [ "${nr:-0}" != 1 ]; then
    chk FAIL "nsys: the probe's process-wide NVTX range is not in its export (rc=$rc, ranges ${nr:-0}; the library: ${NVTX_LIB:-none}): see $W/probe/nsys.log"
    return 0
  fi
  if [ "$WANT_METRICS" = 0 ]; then chk PASS "nsys: run A's flags trace the probe and its process-wide NVTX range (no GPU metrics in this mode)"; return 0; fi
  if [ "${nm:-0}" -gt 0 ]; then chk PASS "nsys: run A's flags trace the probe with its NVTX range and GPU metrics ($nm samples at $NP_GPU_METRICS_HZ Hz)"
  else chk FAIL "nsys: GPU metrics requested, ${nm:-0} samples in the probe's export: see $W/probe/nsys.log"; fi
}

ncu_args() { # ncu_args window|config SKIP COUNT NVTX KERNEL-REGEX-BODY REPORT-BASE -> NCU_ARGS
  local s
  NCU_ARGS=(--target-processes all --kernel-name "regex:^(${5})\$" --launch-skip "$2" --launch-count "$3")
  if [ "$4" != - ]; then NCU_ARGS+=(--nvtx --nvtx-include "$4"); fi
  if [ "$1" = config ]; then NCU_ARGS+=(--filter-mode per-launch-config)
  elif [ "$NCU_HAS_KILL" = 1 ]; then NCU_ARGS+=(--kill yes); fi
  for s in $NP_SECTIONS; do NCU_ARGS+=(--section "$s"); done
  if [ -n "$NCU_METRICS_OK" ]; then NCU_ARGS+=(--metrics "$NCU_METRICS_OK"); fi
  NCU_ARGS+=(--clock-control "$NP_CLOCK" --export "$6" --force-overwrite)
}

nsys_args() { # nsys_args REPORT-BASE WITH_METRICS(0|1) -> NSYS_ARGS: run A's flags
  NSYS_ARGS=(profile "--trace=cuda,nvtx" --sample=none --cpuctxsw=none)
  if [ "$2" = 1 ]; then NSYS_ARGS+=("$NSYS_METRICS_FLAG=all" "--gpu-metrics-frequency=$NP_GPU_METRICS_HZ"); fi
  NSYS_ARGS+=(--stats=false --force-overwrite=true "--output=$1")
}

selected() { # selected PASS WORKLOAD MODES: the workload in NP_WORKLOADS, this mode in MODES, the pass in NP_PASSES (or NP_PASSES empty)
  case " $NP_WORKLOADS " in *" $2 "*) ;; *) return 1 ;; esac
  case ",$3," in *",$NP_MODE,"*) ;; *) return 1 ;; esac
  if [ -z "$NP_PASSES" ]; then return 0; fi
  case " $NP_PASSES " in *" $1 "*) return 0 ;; esac
  return 1
}

pass_est() { # pass_est WORKLOAD MODE COUNT: seconds, a guide (the FAST dry run's walls under ncu's overhead)
  if [ "$2" = window ]; then echo $((90 + 4 * $3)); else echo 180; fi
}

estimate() { # a guide, printed before the long steps
  local b r=0 a=0 p=0 pass wl modes mode skip count nw
  if [ -d "$TARGET/release/deps" ]; then b=2; else b=6; fi
  # shellcheck disable=SC2086 # the workload list, split on blanks
  nw="$(printf '%s\n' $NP_WORKLOADS | awk 'NF { n++ } END { print n + 0 }')"
  if [ "$NP_REFERENCE" = 1 ]; then r=$((nw * 80)); fi
  if [ "$NP_RUN_A" = 1 ]; then a=$((nw * (100 + 200))); fi # a traced run plus its export and tables
  if [ "$NP_RUN_B" = 1 ] && [ "$NP_DRY" = 0 ]; then
    while IFS=$'\t' read -r pass wl modes mode skip count _; do
      if [ "$pass" = pass ] || ! selected "$pass" "$wl" "$modes"; then continue; fi
      p=$((p + $(pass_est "$wl" "$mode" "$count")))
    done < "$PLAN_FILE"
  fi
  log "ESTIMATE ($NP_MODE; a guide, not a bound: every step has its own timeout and nothing new starts after the ${NP_DEADLINE_MIN}-minute deadline): build ~$b min (fresh: 4-8), reference ~$((r / 60)) min, run A ~$((a / 60)) min, run B ~$((p / 60)) min, total ~$((b + (r + a + p) / 60 + 3)) min"
}

DEADLINE_EPOCH=0
time_left() { echo $((DEADLINE_EPOCH - $(date +%s))); }

# ---------------------------------------------------------------------------------------------------
# the checkouts, the fixtures, the builds

restore_lockfiles() { # restore_lockfiles REPO: put back the Cargo.lock files a build rewrote; refuse other edits
  local mod
  mod="$(git -C "$1" status --porcelain --untracked-files=no | awk '{ print $2 }')"
  if [ -z "$mod" ]; then return 0; fi
  if printf '%s\n' "$mod" | awk '!/(^|\/)Cargo[.]lock$/ { bad = 1 } END { exit bad }'; then
    log "restoring $(printf '%s\n' "$mod" | wc -l | tr -d ' ') lockfile(s) a build rewrote in $1"
    # shellcheck disable=SC2086 # file names without spaces, from git
    git -C "$1" checkout --quiet -- $mod
  else
    die 9 "the checkout $1 has modified tracked files ($(printf '%s' "$mod" | tr '\n' ' ')): this directory is the script's; delete $W (or move it away) and run again"
  fi
}

fetch_repo() {
  local pin br fresh=0
  step_begin "clone $NP_REPO_URL; check out $PIN_W (whir) and $PIN_S (stark)"
  if [ ! -e "$REPO/.git" ]; then
    git clone --quiet --no-checkout "$NP_REPO_URL" "$REPO" > "$KEEP/git-clone.log" 2>&1 || die 5 "git clone $NP_REPO_URL failed: see $KEEP/git-clone.log"
    fresh=1
  fi
  for pin in "$PIN_W:$PIN_W_BRANCH" "$PIN_S:$PIN_S_BRANCH"; do
    br="${pin#*:}" pin="${pin%%:*}"
    if ! git -C "$REPO" cat-file -e "$pin^{commit}" 2>/dev/null; then
      git -C "$REPO" fetch --quiet origin "$pin" >> "$KEEP/git-fetch.log" 2>&1 \
        || git -C "$REPO" fetch --quiet origin "$br" >> "$KEEP/git-fetch.log" 2>&1 || true
    fi
    git -C "$REPO" cat-file -e "$pin^{commit}" 2>/dev/null || die 5 "commit $pin is not on $NP_REPO_URL (branch $br): see $KEEP/git-fetch.log"
  done
  # a fresh --no-checkout clone has no working tree yet; an earlier run's checkout may hold a rewritten lockfile
  if [ "$fresh" = 0 ]; then restore_lockfiles "$REPO"; fi
  git -C "$REPO" -c advice.detachedHead=false checkout --quiet --detach "$PIN_W" > "$KEEP/git-checkout.log" 2>&1 \
    || die 5 "git checkout $PIN_W failed: see $KEEP/git-checkout.log"
  if [ -e "$REPO_S/.git" ]; then
    restore_lockfiles "$REPO_S"
    git -C "$REPO_S" -c advice.detachedHead=false checkout --quiet --detach "$PIN_S" >> "$KEEP/git-checkout.log" 2>&1 \
      || die 5 "git checkout $PIN_S in $REPO_S failed: see $KEEP/git-checkout.log"
  else
    git -C "$REPO" worktree prune > /dev/null 2>&1 || true
    git -C "$REPO" worktree add --quiet --detach "$REPO_S" "$PIN_S" >> "$KEEP/git-checkout.log" 2>&1 \
      || die 5 "git worktree add $REPO_S $PIN_S failed: see $KEEP/git-checkout.log"
  fi
  [ "$(git -C "$REPO" rev-parse HEAD)" = "$PIN_W" ] || die 5 "HEAD of $REPO is $(git -C "$REPO" rev-parse HEAD), expected $PIN_W"
  [ "$(git -C "$REPO_S" rev-parse HEAD)" = "$PIN_S" ] || die 5 "HEAD of $REPO_S is $(git -C "$REPO_S" rev-parse HEAD), expected $PIN_S"
  git -C "$REPO" merge-base --is-ancestor "$WHIR_SHA" "$PIN_W" || die 5 "$PIN_W does not descend from noepoch/whir @ $WHIR_SHA"
  git -C "$REPO" merge-base --is-ancestor "$STARK_SHA" "$PIN_S" || die 5 "$PIN_S does not descend from noepoch/stark @ $STARK_SHA"
  log "whir HEAD $PIN_W (descends from noepoch/whir $WHIR_SHA) · stark HEAD $PIN_S (descends from noepoch/stark $STARK_SHA); both asserted"
  step_end 0
}

fetch_fixtures() {
  local elf="$W/fixtures/ethrex_8f826601.elf" input="$W/fixtures/ethrex_mainnet_25368371_573004e6.bin" got mk
  step_begin "fixtures: block 25368371 (release asset) and the record's guest ELF (git), by sha256"
  mk="$(awk -F':= *' '/^ETHREX_REAL_BLOCK_FIXTURE_SHA256/ { print $2; exit }' "$REPO/Makefile" | tr -d ' ')"
  if [ "$mk" != "$INPUT_SHA256" ]; then die 5 "the Makefile's ETHREX_REAL_BLOCK_FIXTURE_SHA256 at $PIN_W is '$mk', this script pins $INPUT_SHA256"; fi
  if [ ! -s "$input" ] || [ "$(sha256_of "$input")" != "$INPUT_SHA256" ]; then
    curl -fsSL --retry 3 -o "$input.part" "$NP_INPUT_URL" > "$KEEP/curl-input.log" 2>&1 \
      || die 5 "downloading $NP_INPUT_URL failed: see $KEEP/curl-input.log"
    mv "$input.part" "$input"
  fi
  got="$(sha256_of "$input")"
  [ "$got" = "$INPUT_SHA256" ] || die 5 "block input sha256 $got, want $INPUT_SHA256"
  if [ ! -s "$elf" ] || [ "$(sha256_of "$elf")" != "$ELF_SHA256" ]; then
    if ! git -C "$REPO" cat-file -e "$ELF_COMMIT:$ELF_REPO_PATH" 2>/dev/null; then
      git -C "$REPO" fetch --quiet origin "$ELF_COMMIT" > "$KEEP/git-fetch-elf.log" 2>&1 \
        || git -C "$REPO" fetch --quiet origin whir/profile-rpx >> "$KEEP/git-fetch-elf.log" 2>&1 || true
    fi
    if git -C "$REPO" cat-file -e "$ELF_COMMIT:$ELF_REPO_PATH" 2>/dev/null; then
      git -C "$REPO" cat-file blob "$ELF_COMMIT:$ELF_REPO_PATH" > "$elf.part"
    else
      curl -fsSL --retry 3 -o "$elf.part" \
        "https://raw.githubusercontent.com/yetanotherco/lambda_vm/$ELF_COMMIT/$ELF_REPO_PATH" > "$KEEP/curl-elf.log" 2>&1 \
        || die 5 "the record ELF is neither in the clone nor at raw.githubusercontent.com: see $KEEP/curl-elf.log"
    fi
    mv "$elf.part" "$elf"
  fi
  got="$(sha256_of "$elf")"
  [ "$got" = "$ELF_SHA256" ] || die 5 "guest ELF sha256 $got, want $ELF_SHA256"
  log "block input $(wc -c < "$input" | tr -d ' ') B sha256 $INPUT_SHA256 · guest ELF $(wc -c < "$elf" | tr -d ' ') B sha256 $ELF_SHA256"
  step_end 0
}

CUBIN_NOTE=""
build_one() { # build_one whir|stark: that pin's test binary into the shared target directory -> BUILT_BIN
  local wl="$1" repo rc f n=0 empty=0 li=0 cdir test
  if [ "$wl" = whir ]; then repo="$REPO" test="$WHIR_TEST"; else repo="$REPO_S" test="$STARK_TEST"; fi
  step_begin "build $wl: cargo test --release -p lambda-vm-prover --features $FEATURES --lib --no-run (in $repo, lineinfo cubins)"
  rc=0
  (cd "$repo" && exec env -i "${BUILD_ENV[@]}" ${ARCH_ENV[@]+"${ARCH_ENV[@]}"} timeout "$NP_BUILD_TIMEOUT" \
    cargo test --release -p lambda-vm-prover --features "$FEATURES" --lib --no-run --message-format=json-render-diagnostics) \
    > "$KEEP/cargo-$wl.json" 2> "$KEEP/build-$wl.log" || rc=$?
  tail -n 40 "$KEEP/build-$wl.log" > "$SEND/logs/build-$wl.tail.txt" || true
  step_end "$rc"
  if [ "$rc" -ne 0 ]; then die 5 "the $wl build failed (rc=$rc): see $KEEP/build-$wl.log"; fi
  restore_lockfiles "$repo"
  BUILT_BIN="$(python3 "$TOOL" cargo-artifact --exe lambda_vm_prover < "$KEEP/cargo-$wl.json")" \
    || die 5 "cargo reported no lambda_vm_prover test binary for $wl (see $KEEP/cargo-$wl.json)"
  cdir="$(python3 "$TOOL" cargo-artifact --outdir math-cuda < "$KEEP/cargo-$wl.json")" \
    || die 5 "cargo reported no math-cuda build-script output for $wl (see $KEEP/cargo-$wl.json)"
  for f in "$cdir"/*.cubin; do
    [ -e "$f" ] || continue
    n=$((n + 1))
    if [ ! -s "$f" ]; then empty=$((empty + 1)); fi
    if LC_ALL=C grep -q '[.]debug_line' "$f"; then li=$((li + 1)); fi
  done
  log "$wl cubins: $n present, $empty empty; with SASS-to-source line tables: $li"
  if [ "$n" -lt 12 ] || [ "$empty" -ne 0 ]; then
    die 5 "$wl: $n cubins ($empty empty) in $cdir: build.rs did not find nvcc, so every kernel would run on the CPU"
  fi
  CUBIN_NOTE="${CUBIN_NOTE:+$CUBIN_NOTE · }$wl $li of $n with line tables"
  n="$({ "$BUILT_BIN" --list 2>/dev/null || true; } | awk -v t="$test: test" '$0 == t { n++ } END { print n + 0 }')"
  [ "$n" = 1 ] || die 5 "the $wl test binary does not list $test"
  if [ -x "$(cuda_home)/bin/cuobjdump" ]; then
    for f in "$cdir"/*.cubin; do
      echo "== $(basename "$f")"
      "$(cuda_home)/bin/cuobjdump" -res-usage "$f" 2>&1 | sed "s#$cdir/##g" || true
    done > "$SEND/cubin_res_usage-$wl.txt"
  fi
  log "$wl test binary: $BUILT_BIN (sha256 $(sha256_of "$BUILT_BIN"))"
}

build_all() {
  local w
  build_env
  for w in $NP_WORKLOADS; do
    BUILT_BIN=""
    build_one "$w"
    if [ "$w" = whir ]; then BIN_W="$BUILT_BIN"; else BIN_S="$BUILT_BIN"; fi
  done
}

write_env_facts() { # an allowlist of run facts (this file leaves the machine; the environment itself never does)
  local e w po f="$SEND/env.txt"
  {
    echo "# noepoch_counters.sh run facts: an allowlist. The environment is never recorded here."
    echo "script=$SCRIPT_VERSION md5 $(md5_of "$0") mode=$NP_MODE dry=$NP_DRY workloads=$NP_WORKLOADS reference=$NP_REFERENCE run_a=$NP_RUN_A run_b=$NP_RUN_B deadline_min=$NP_DEADLINE_MIN whir_knobs=${NP_WHIR_KNOBS:-none} stark_knobs=${NP_STARK_KNOBS:-none}"
    echo "repo=$NP_REPO_URL whir_head=$PIN_W ($PIN_W_BRANCH) stark_head=$PIN_S ($PIN_S_BRANCH)"
    echo "whir_base=$WHIR_SHA (noepoch/whir, PR #1014) test $WHIR_TEST"
    echo "stark_base=$STARK_SHA (noepoch/stark, PR #1013) test $STARK_TEST"
    echo "tracked_files_modified_after_build=whir $(git -C "$REPO" status --porcelain --untracked-files=no | awk 'END { print NR }') stark $(git -C "$REPO_S" status --porcelain --untracked-files=no 2>/dev/null | awk 'END { print NR }')"
    echo "guest_elf=ethrex_8f826601.elf sha256 $ELF_SHA256 (from $ELF_COMMIT:$ELF_REPO_PATH)"
    echo "block_input=ethrex_mainnet_25368371 sha256 $INPUT_SHA256"
    echo "build=cargo test --release -p lambda-vm-prover --features $FEATURES --lib, LAMBDA_VM_NVCC_LINEINFO=1, one target directory"
    echo "test_binary_sha256=whir ${BIN_W:+$(sha256_of "$BIN_W")} stark ${BIN_S:+$(sha256_of "$BIN_S")}"
    echo "cubins=${CUBIN_NOTE:-?}"
    echo "gpu_index=$NP_GPU of $NGPUS"
    echo "gpu_name=$GPU_NAME"
    echo "gpu_compute_capability=$GPU_CC"
    echo "gpu_memory_total_mib=$GPU_VRAM"
    echo "gpu_clocks_max_sm_mhz=$(gpu_field clocks.max.sm) mem_mhz=$(gpu_field clocks.max.mem)"
    echo "gpu_power_limit_w=$(gpu_field power.limit)"
    echo "driver=$GPU_DRIVER cuda=$DRIVER_CUDA nvcc=$NVCC_REL"
    echo "ncu=${NCU_VER:-none} clock_control=$NP_CLOCK kill=$KILL_PROBE nvtx_filter=$NVTX_FILTER run_b_vram_mb=${NP_NCU_VRAM_MB:-the workload default}"
    echo "nsys=${NSYS_VER:-none} gpu_metrics=$([ "$WANT_METRICS" = 1 ] && echo "$NP_GPU_METRICS_HZ Hz" || echo off) nvtx_library=$([ -n "$NVTX_LIB" ] && echo found || echo none)"
    echo "cpu=$(cpu_model) · $(nproc 2>/dev/null || echo '?') threads · ram_gib=$(mem_total_gib) · available_at_start_gib=$(mem_avail_gib)"
    echo "rustc=$(cd "$REPO" && env -i "${BUILD_ENV[@]}" rustc --version 2>/dev/null || echo '?')"
    echo "ncu_sections=$NP_SECTIONS"
    echo "ncu_metrics=${NCU_METRICS_OK:-<none validated: dry run or none collectable>}"
    echo "# The profiled process's environment (env -i): PATH, HOME=<W>/home, TMPDIR=<W>/tmp, LC_ALL=C, LANG=C,"
    echo "# CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES, CUDA_HOME/CUDA_PATH/LD_LIBRARY_PATH when set,"
    echo "# LAMBDA_VM_NVTX_LIB, and the workload's knobs:"
    for w in whir stark; do
      for po in record runb; do
        wl_env "$w" "$po"
        echo "[$w $po] test=$WL_TEST"
        for e in "${WL_ENV[@]}"; do printf '[%s %s] %s\n' "$w" "$po" "$e"; done
      done
    done
  } > "$f"
}

# ---------------------------------------------------------------------------------------------------
# the runs: the reference, run A, run B

GATES_RE='^W3 |^BLOCK |CARD HOLD|★★★|TREE PIPE|TREE AHEAD|LFM PROVE|CHILD VERIFIES|harvest|LEVEL|PROVE SPLIT|packing admission|card permit|test result:|panicked|FAILED|^error|==WARNING==|==ERROR=='
gates() { awk -v re="$GATES_RE" '$0 ~ re { print substr($0, 1, 1200) }' "$1"; }

CPU_PID="" SMI_PID=""
start_cpu() { # start_cpu PREFIX BIN: the CPU sampler and memory watchdog, waiting for BIN's process
  python3 "$TOOL" cpu-sample --exe "$2" --out "$1" --interval-ms 100 --mem-floor-mib "$NP_MEM_FLOOR_MIB" \
    --wait-s 600 > "$1.sampler.log" 2>&1 &
  CPU_PID=$!
}
stop_cpu() { # the sampler ends with its process; give it a moment, then stop it
  local i
  if [ -z "$CPU_PID" ]; then return 0; fi
  for i in $(seq 1 20); do
    if ! kill -0 "$CPU_PID" 2>/dev/null; then break; fi
    sleep 1
  done
  kill "$CPU_PID" 2>/dev/null || true
  wait "$CPU_PID" 2>/dev/null || true
  CPU_PID=""
}
start_smi() { # start_smi FILE: VRAM and utilisation every 200 ms, with nvidia-smi's local timestamp
  nvidia-smi -i "$NP_GPU" --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 200 \
    > "$1" 2>/dev/null &
  SMI_PID=$!
}
stop_smi() {
  if [ -n "$SMI_PID" ]; then kill "$SMI_PID" 2>/dev/null || true; wait "$SMI_PID" 2>/dev/null || true; SMI_PID=""; fi
}
smi_max() { awk -F', *' 'NF >= 3 && $2 + 0 > m { m = $2 + 0 } END { print m + 0 }' "$1" 2>/dev/null || echo 0; }

mem_gate() { # mem_gate WHAT: MemAvailable >= NP_MIN_AVAIL_GIB before a run (a minute for the page cache to drain)
  local i av
  for i in $(seq 1 30); do
    av="$(mem_avail_gib)"
    if [ -n "$av" ] && num_ge "$av" "$NP_MIN_AVAIL_GIB"; then return 0; fi
    sleep 2
  done
  log "SKIPPED $1: host MemAvailable ${av:-?} GiB < $NP_MIN_AVAIL_GIB GiB (close other programs; the run would not fit)"
  return 1
}

launch() { # launch plain|nsys|ncu WORKLOAD record|runb LOG CPU-PREFIX [REPORT-BASE] -> rc; the workload's test, once
  local kind="$1" wl="$2" po="$3" lg="$4" cp="$5" rep="${6:-}" rc=0 to="$NP_RUN_TIMEOUT"
  wl_env "$wl" "$po"
  run_env nvtx
  local -a pre=()
  case "$kind" in
    nsys) nsys_args "$rep" "$([ "$WANT_METRICS" = 1 ] && echo 1 || echo 0)"; pre=("$NSYS_BIN" "${NSYS_ARGS[@]}") ;;
    ncu) pre=("$NCU_BIN" "${NCU_ARGS[@]}"); to="$NP_PASS_TIMEOUT" ;;
  esac
  start_cpu "$cp" "$WL_BIN"
  (cd "$WL_REPO/prover" && exec timeout --signal=INT --kill-after=120 "$to" \
    env -i "${RUN_ENV[@]}" "${WL_ENV[@]}" ${pre[@]+"${pre[@]}"} "$WL_BIN" "$WL_TEST" --ignored --exact --nocapture --test-threads=1) \
    < /dev/null > "$lg" 2>&1 || rc=$?
  stop_cpu
  return "$rc"
}

REF_OK=1 REF_LINE=""
reference_runs() {
  local w lg smi rc t0 t1 lh ok vr ct
  printf 'workload\trc\tseconds\tlog_ok\ttest_result\tcard_holds\theadline\tcpu\tvram_max_mib\n' > "$SEND/reference/runs.tsv"
  for w in $NP_WORKLOADS; do
    if [ "$(time_left)" -le 0 ]; then log "reference $w: SKIPPED (the deadline passed)"; REF_OK=0; continue; fi
    if ! mem_gate "reference $w"; then REF_OK=0; continue; fi
    wait_idle
    lg="$KEEP/ref-$w.log" smi="$KEEP/ref-$w.smi"
    step_begin "reference: the $w workload, no profiler"
    start_smi "$smi"
    t0="$(date +%s)"
    rc=0
    launch plain "$w" record "$lg" "$KEEP/ref-$w.cpu" || rc=$?
    t1="$(date +%s)"
    stop_smi
    step_end "$rc"
    gates "$lg" > "$SEND/logs/ref-$w.gates.txt"
    lh="$(python3 "$TOOL" loghead --log "$lg" --workload "$w")"
    ok="$(printf '%s' "$lh" | cut -f1)"
    ct="$(python3 "$TOOL" cpu-total --cpu "$KEEP/ref-$w.cpu")"
    vr="$(smi_max "$smi")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$w" "$rc" "$((t1 - t0))" "$lh" "$ct" "$vr" >> "$SEND/reference/runs.tsv"
    cp "$KEEP/ref-$w.cpu.proc.tsv" "$SEND/cpu/ref-$w.proc.tsv" 2>/dev/null || true
    log "reference $w: rc=$rc in $((t1 - t0)) s · $ok · $(printf '%s' "$lh" | cut -f4) · $ct · VRAM max $vr MiB"
    if [ "$rc" -ne 0 ] || [ "$ok" != ok ]; then REF_OK=0; fi
    REF_LINE="${REF_LINE:+$REF_LINE · }$w rc $rc $((t1 - t0)) s"
  done
}

export_trace() { # export_trace REP: REP.sqlite from REP.nsys-rep; returns 1 when there is none
  if [ ! -s "$1.nsys-rep" ]; then return 1; fi
  timeout 1800 "$NSYS_BIN" export --type sqlite --force-overwrite true --output "$1.sqlite" "$1.nsys-rep" \
    > "$1.export.log" 2>&1 || true
  [ -s "$1.sqlite" ]
}

RUNA_OK=1 RUNA_LINE=""
run_a() { # each workload under nsys (GPU metrics unless dry); stage, hold, CPU and API tables; the plan against the trace
  local w rep lg smi rc t0 t1 lh ok arc drc dfrc out r rl src th lgh vr ct reports
  printf 'workload\trc\tseconds\tlog_ok\ttest_result\tlog_card_holds\theadline\tcpu\trunA_rc\tplan_rc\tstages\ttrace_card_holds\tvram_max_mib\n' > "$SEND/runa/runs.tsv"
  reports="cuda_gpu_kern_sum nvtx_sum"
  if [ "$NP_MODE" = full ] || [ "$NP_DRY" = 1 ]; then
    reports="$reports cuda_gpu_mem_time_sum cuda_gpu_mem_size_sum cuda_api_sum nvtx_gpu_proj_sum"
  fi
  for w in $NP_WORKLOADS; do
    if [ "$(time_left)" -le 0 ]; then log "run A $w: SKIPPED (the deadline passed)"; RUNA_OK=0; continue; fi
    if ! mem_gate "run A $w"; then RUNA_OK=0; continue; fi
    wait_idle
    rep="$KEEP/runa-$w" lg="$KEEP/runa-$w.log" smi="$KEEP/runa-$w.smi" out="$SEND/runa/$w"
    mkdir -p "$out"
    step_begin "run A: the $w workload under Nsight Systems ($([ "$WANT_METRICS" = 1 ] && echo "GPU metrics at $NP_GPU_METRICS_HZ Hz" || echo 'no GPU metrics'))"
    start_smi "$smi"
    t0="$(date +%s)"
    rc=0
    launch nsys "$w" record "$lg" "$KEEP/runa-$w.cpu" "$rep" || rc=$?
    t1="$(date +%s)"
    stop_smi
    step_end "$rc"
    gates "$lg" > "$SEND/logs/runa-$w.gates.txt"
    lh="$(python3 "$TOOL" loghead --log "$lg" --workload "$w")"
    ok="$(printf '%s' "$lh" | cut -f1)" lgh="$(printf '%s' "$lh" | cut -f3)"
    ct="$(python3 "$TOOL" cpu-total --cpu "$KEEP/runa-$w.cpu")"
    vr="$(smi_max "$smi")"
    cp "$KEEP/runa-$w.cpu.proc.tsv" "$SEND/cpu/runa-$w.proc.tsv" 2>/dev/null || true
    arc=9 drc=9 src="-" th="-"
    if export_trace "$rep"; then
      step_begin "run A: $w export, nsys stats, the stage, hold, CPU and API tables"
      for r in $reports; do
        timeout 900 "$NSYS_BIN" stats --report "$r" --format csv --output "$out/nsys" "$rep.sqlite" \
          >> "$KEEP/runa-$w.stats.log" 2>&1 || true
      done
      arc=0
      python3 "$TOOL" runa --sqlite "$rep.sqlite" --log "$lg" --workload "$w" --smi "$smi" --cpu "$KEEP/runa-$w.cpu" \
        --out "$out" > "$KEEP/runa-$w.summary.log" 2>&1 || arc=$?
      drc=0
      python3 "$TOOL" dry --plan "$PLAN_FILE" --workload "$w" --sqlite "$rep.sqlite" --log "$lg" --out "$out" \
        --tag "runa-$w" --modes "$NP_MODE" > "$KEEP/runa-$w.plan.log" 2>&1 || drc=$?
      if [ "$NP_DRY" = 1 ] && [ "$NP_MODE" = full ]; then
        dfrc=0
        python3 "$TOOL" dry --plan "$PLAN_FILE" --workload "$w" --sqlite "$rep.sqlite" --log "$lg" --out "$out" \
          --tag "runa-$w-quick" --modes quick > "$KEEP/runa-$w.plan-quick.log" 2>&1 || dfrc=$?
        grep -E '^DRY ' "$KEEP/runa-$w.plan-quick.log" || true
        if [ "$dfrc" -ne 0 ]; then drc="$dfrc"; fi
      fi
      step_end "$arc"
      rl="$(grep '^RUNA ' "$KEEP/runa-$w.summary.log" | tail -1 || true)"
      case "$rl" in *"stages from NVTX ranges"*) src=nvtx ;; *) src=none ;; esac
      th="$(printf '%s' "$rl" | sed -n 's/.*card holds \([0-9]*\) (log.*/\1/p')"
      if [ "$arc" -eq 0 ]; then sed -n '/^| stage /,/^$/p' "$out/runa-$w.md"; sed -n '/^## card holds/,/^$/p;/^| phase | stage | holds/,/^$/p' "$out/runa-$w.md" | head -12; fi
      grep -E '^DRY ' "$KEEP/runa-$w.plan.log" || true
      if [ "$NP_CLEAN" = 1 ]; then rm -f "$rep.sqlite"; fi
    else
      log "run A $w: nsys wrote no report: see $lg"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$w" "$rc" "$((t1 - t0))" "$lh" "$ct" "$arc" "$drc" "$src" "${th:--}" "$vr" >> "$SEND/runa/runs.tsv"
    log "run A $w: rc=$rc in $((t1 - t0)) s · $ok · tables rc=$arc · plan rc=$drc · stages from $src · card holds ${th:--} in the trace, $lgh in the log · $ct · VRAM max $vr MiB"
    if [ "$rc" -ne 0 ] || [ "$arc" -ne 0 ] || [ "$ok" != ok ] || [ "$src" != nvtx ]; then RUNA_OK=0; fi
    if [ "${th:-x}" != "$lgh" ] || [ "${lgh:-0}" -lt 1 ]; then RUNA_OK=0; log "run A $w: card holds in the trace (${th:--}) != CARD HOLD lines ($lgh), or none"; fi
    if [ "$NP_DRY" = 1 ] && [ "$drc" -ne 0 ]; then RUNA_OK=0; fi
    RUNA_LINE="${RUNA_LINE:+$RUNA_LINE · }$w rc $rc $((t1 - t0)) s, stages $src, holds ${th:--}/$lgh"
  done
}

PASSES_RUN=0 PASSES_OK=0 PASSES_SKIPPED=0 LAUNCHES_TOTAL=0
run_passes() { # run B under ncu
  local pass wl modes mode skip count nvtx kernels note lg rc t0 t1 profiled verdict est ct
  printf 'pass\tworkload\tnvtx\tmode\tskip\tcount\tkernels\trc\tprofiled\tseconds\tcpu\tverdict\n' > "$SEND/passes.tsv"
  while IFS=$'\t' read -r pass wl modes mode skip count nvtx kernels _ note; do
    if [ "$pass" = pass ] || ! selected "$pass" "$wl" "$modes"; then continue; fi
    est="$(pass_est "$wl" "$mode" "$count")"
    verdict=""
    if [ "$nvtx" != - ] && [ "$NVTX_FILTER_OK" = 0 ]; then verdict="skipped: NVTX filtering failed on the probe"
    elif [ "$(time_left)" -lt "$est" ]; then verdict="skipped: the deadline (${est} s estimated, $(time_left) s left)"
    elif ! mem_gate "pass $pass"; then verdict="skipped: host memory"; fi
    if [ -n "$verdict" ]; then
      PASSES_SKIPPED=$((PASSES_SKIPPED + 1))
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t-\t0\t0\t-\t%s\n' "$pass" "$wl" "$nvtx" "$mode" "$skip" "$count" "$kernels" "$verdict" >> "$SEND/passes.tsv"
      log "pass $pass: $verdict"
      continue
    fi
    PASSES_RUN=$((PASSES_RUN + 1))
    wait_idle
    ncu_args "$mode" "$skip" "$count" "$nvtx" "$kernels" "$KEEP/ncu/$pass"
    lg="$KEEP/ncu/$pass.log"
    step_begin "run B: pass $pass ($wl, $mode, nvtx $nvtx, skip $skip count $count): $note"
    t0="$(date +%s)"
    rc=0
    launch ncu "$wl" runb "$lg" "$KEEP/ncu/$pass.cpu" || rc=$?
    t1="$(date +%s)"
    profiled="$(grep -c '^==PROF== Profiling' "$lg" || true)"
    if [ -s "$KEEP/ncu/$pass.ncu-rep" ]; then
      run_env nvtx
      env -i "${RUN_ENV[@]}" timeout 900 "$NCU_BIN" --import "$KEEP/ncu/$pass.ncu-rep" --page details --csv \
        > "$SEND/ncu/$pass.details.csv" 2> "$KEEP/ncu/$pass.import.log" || log "ncu --import (details) failed: see $KEEP/ncu/$pass.import.log"
    fi
    python3 "$TOOL" stages --log "$lg" --nvtx "$nvtx" > "$SEND/ncu/$pass.stages.tsv" 2>> "$KEEP/ncu/$pass.import.log" || true
    { grep -E '^==(PROF|WARNING|ERROR)==' "$lg" || true; } > "$SEND/ncu/$pass.prof.txt"
    gates "$lg" | grep -v 'CARD HOLD' > "$SEND/logs/$pass.gates.txt" || true
    ct="$(python3 "$TOOL" cpu-total --cpu "$KEEP/ncu/$pass.cpu")"
    if [ "$profiled" -ge 1 ] && [ -s "$SEND/ncu/$pass.details.csv" ]; then
      verdict=ok
      if [ "$mode" = window ] && [ "$profiled" -lt "$count" ]; then verdict="partial ($profiled of $count)"; fi
    else verdict="FAILED (rc=$rc, $profiled profiled; see keep/ncu/$pass.log)"; fi
    if [ "$verdict" = ok ] || [ "${verdict#partial}" != "$verdict" ]; then PASSES_OK=$((PASSES_OK + 1)); fi
    LAUNCHES_TOTAL=$((LAUNCHES_TOTAL + profiled))
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$pass" "$wl" "$nvtx" "$mode" "$skip" "$count" "$kernels" \
      "$rc" "$profiled" "$((t1 - t0))" "$ct" "$verdict" >> "$SEND/passes.tsv"
    log "pass $pass: $profiled launch(es) profiled in $((t1 - t0)) s, $verdict · $ct"
    step_end "$rc"
  done < "$PLAN_FILE"
}

# ---------------------------------------------------------------------------------------------------
# --print-commands: every command a run would execute, with placeholders, nothing run (the laptop check)

print_commands() {
  local w pass wl modes mode skip count nvtx kernels
  W="$NP_WORKDIR" REPO="$NP_WORKDIR/lambda_vm" REPO_S="$NP_WORKDIR/lambda_vm-stark" TARGET="$NP_WORKDIR/target"
  BIN_W="<W>/target/release/deps/lambda_vm_prover-<hash-whir>" BIN_S="<W>/target/release/deps/lambda_vm_prover-<hash-stark>"
  NSYS_BIN=nsys NCU_BIN=ncu NSYS_METRICS_FLAG=--gpu-metrics-devices NCU_HAS_KILL=1 NCU_METRICS_OK="<the metrics this ncu can collect>"
  WANT_METRICS=$([ "$NP_DRY" = 1 ] && echo 0 || echo 1)
  NVTX_LIB="$(find_nvtx_lib || true)"
  if [ -z "$NVTX_LIB" ]; then NVTX_LIB="<LAMBDA_VM_NVTX_LIB>"; fi
  build_env
  echo "# noepoch_counters.sh $SCRIPT_VERSION --print-commands ($NP_MODE): nothing is run. Work directory $W"
  echo "# checkout: git clone $NP_REPO_URL $REPO; git checkout --detach $PIN_W; git worktree add --detach $REPO_S $PIN_S"
  echo "# fixtures: $NP_INPUT_URL (sha256 $INPUT_SHA256); $ELF_COMMIT:$ELF_REPO_PATH (sha256 $ELF_SHA256)"
  echo "build whir: (cd $REPO && env -i ${BUILD_ENV[*]} cargo test --release -p lambda-vm-prover --features $FEATURES --lib --no-run)"
  echo "build stark: (cd $REPO_S && env -i ${BUILD_ENV[*]} cargo test --release -p lambda-vm-prover --features $FEATURES --lib --no-run)"
  for w in $NP_WORKLOADS; do
    wl_env "$w" record
    run_env nvtx
    if [ "$NP_REFERENCE" = 1 ]; then
      echo "reference $w: (cd $WL_REPO/prover && env -i ${RUN_ENV[*]} ${WL_ENV[*]} $WL_BIN $WL_TEST --ignored --exact --nocapture --test-threads=1)"
    fi
    if [ "$NP_RUN_A" = 1 ]; then
      nsys_args "<W>/runs/<UTC>/keep/runa-$w" "$WANT_METRICS"
      echo "run A $w: (cd $WL_REPO/prover && env -i ${RUN_ENV[*]} ${WL_ENV[*]} $NSYS_BIN ${NSYS_ARGS[*]} $WL_BIN $WL_TEST --ignored --exact --nocapture --test-threads=1)"
    fi
  done
  if [ "$NP_RUN_B" = 1 ] && [ "$NP_DRY" = 0 ]; then
    while IFS=$'\t' read -r pass wl modes mode skip count nvtx kernels _; do
      if [ "$pass" = pass ] || ! selected "$pass" "$wl" "$modes"; then continue; fi
      wl_env "$wl" runb
      run_env nvtx
      ncu_args "$mode" "$skip" "$count" "$nvtx" "$kernels" "<W>/runs/<UTC>/keep/ncu/$pass"
      echo "run B $pass: (cd $WL_REPO/prover && env -i ${RUN_ENV[*]} ${WL_ENV[*]} $NCU_BIN ${NCU_ARGS[*]} $WL_BIN $WL_TEST --ignored --exact --nocapture --test-threads=1)"
    done < <(plan_source)
  fi
}

# ---------------------------------------------------------------------------------------------------
# the bundle: scrub, self-check, pack

BENIGN_ENV='^(PATH|PWD|OLDPWD|SHELL|SHLVL|TERM|TERM_PROGRAM|TERM_PROGRAM_VERSION|COLORTERM|LANG|LANGUAGE|LC_[A-Z]+|HOME|USER|LOGNAME|MAIL|_|HOSTTYPE|OSTYPE|MACHTYPE|CUDA_HOME|CUDA_PATH|LD_LIBRARY_PATH|MANPATH|INFOPATH|XDG_[A-Z_]+|DISPLAY|WAYLAND_DISPLAY|EDITOR|VISUAL|PAGER|LESS|LESSOPEN|LESSCLOSE|LS_COLORS|TMUX|TMUX_PANE|STY|WINDOW|DBUS_SESSION_BUS_ADDRESS|NP_[A-Z_]+|NCU|NSYS|SSH_TTY|MOTD_SHOWN|COLUMNS|LINES)$'
ENV_NAMES=()
snapshot_env() { # the names of the caller's environment, taken at start; values are read at check time
  local n
  while IFS= read -r n; do ENV_NAMES+=("$n"); done < <(compgen -e)
}

sensitive_pairs() { # OLD<TAB>NEW for the scrub, longest first is the tool's job
  local h ip
  printf '%s\t<W>\n' "$W"
  if [ -n "$NP_WORKDIR" ] && [ "$NP_WORKDIR" != "$W" ]; then printf '%s\t<W>\n' "$NP_WORKDIR"; fi
  if [ -n "$REAL_HOME" ] && [ "$REAL_HOME" != / ]; then printf '%s\t<HOME>\n' "$REAL_HOME"; fi
  for h in "$(hostname 2>/dev/null || true)" "$(hostname -f 2>/dev/null || true)" "$(cat /etc/hostname 2>/dev/null || true)"; do
    if [ "${#h}" -ge 5 ]; then printf '%s\t<HOST>\n' "$h"; fi
  done
  while IFS= read -r ip; do
    case "$ip" in ''|127.0.0.1|::1) ;; *) printf '%s\t<IP>\n' "$ip" ;; esac
  done < <(local_ips)
}

sensitive_values() { # LABEL<TAB>VALUE records, NUL-separated, for the self-check (never printed)
  local n v h ip allow=" $NP_SCAN_ALLOW "
  printf 'work directory\t%s\0' "$W"
  if [ -n "$REAL_HOME" ] && [ "$REAL_HOME" != / ]; then printf 'home directory\t%s\0' "$REAL_HOME"; fi
  if [ -n "${USER:-}" ] && [ "${#USER}" -ge 3 ]; then printf 'user path\t/home/%s/\0' "$USER"; fi
  for h in "$(hostname 2>/dev/null || true)" "$(hostname -f 2>/dev/null || true)"; do
    if [ "${#h}" -ge 5 ]; then printf 'hostname\t%s\0' "$h"; fi
  done
  while IFS= read -r ip; do
    case "$ip" in ''|127.0.0.1|::1) ;; *) printf 'ip address\t%s\0' "$ip" ;; esac
  done < <(local_ips)
  for n in "${ENV_NAMES[@]}"; do
    if [[ "$n" =~ $BENIGN_ENV ]]; then continue; fi
    case "$allow" in *" $n "*) continue ;; esac
    v="${!n:-}"
    if [ "${#v}" -ge 8 ]; then printf 'env:%s\t%s\0' "$n" "$v"; fi
  done
}

BUNDLE_LINE=""
make_bundle() {
  local scan rc=0 tarball size
  cp "$KEEP/preflight.txt" "$SEND/preflight.txt" 2>/dev/null || true
  prereg_text > "$SEND/prereg.txt"
  cp "$KEEP/driver.log" "$SEND/driver.log" 2>/dev/null || true
  python3 "$TOOL" report --send "$SEND" > "$KEEP/report.log" 2>&1 || log "the text summary failed: see $KEEP/report.log"
  {
    echo "noepoch_counters.sh bundle ($SCRIPT_VERSION, $NP_MODE): text only. Read SUMMARY.md first."
    echo "  SUMMARY.md                          the whole run in one file: reference walls, run A per workload, run B, the plan"
    echo "  reference/runs.tsv                  the unprofiled runs (--full): rc, seconds, log gates, headline, CPU, VRAM max"
    echo "  runa/runs.tsv                       run A per workload: rc, seconds, headline, stage source, card holds trace/log"
    echo "  runa/<w>/runa-<w>.md, *-stages.tsv  run A per stage: wall, busy %, kernels, families, cores, SM/DRAM/PCIe %, VRAM"
    echo "  runa/<w>/*-stage-families.tsv       kernel seconds per stage and kernel family · *-stage-kernels.tsv per kernel"
    echo "  runa/<w>/*-holds.tsv, *-hold-kernels.tsv   every card hold (an LFM proof's device phase): busy %, kernels"
    echo "  runa/<w>/cpu-<w>-by-stage.tsv       the profiled process's CPU seconds per stage and thread role"
    echo "  runa/<w>/*-timeseries.tsv           run A in 100 ms bins: stage, kernel/copy busy %, holds, VRAM, cores, GPU metrics"
    echo "  runa/<w>/*-nvtx-kernels.tsv         the kernels each NVTX label launched (thread or process-wide window)"
    echo "  runa/<w>/api-<w>-by-*.tsv           CUDA API seconds per category x stage, per thread, per call"
    echo "  runa/<w>/*-gpu-metrics.tsv          every GPU-metric series per stage · nsys_*.csv nsys stats"
    echo "  runa/<w>/dry-runa-<w>*.md, .tsv     run B's plan against run A's trace: what each pass targets, and coverage"
    echo "  summary/kernels.md, kernels.tsv     run B, one row per kernel and workload (the roofs)"
    echo "  summary/kernels_by_stage.tsv        run B per workload, kernel and NVTX window · summary/launches.tsv every launch"
    echo "  ncu/<pass>.details.csv              ncu --import --page details --csv (the Host Name column removed)"
    echo "  ncu/<pass>.stages.tsv, .prof.txt    each profiled launch's window · ncu's own messages"
    echo "  cpu/*.proc.tsv                      the CPU sampler per run: cumulative ticks, RSS, threads, MemAvailable (100 ms)"
    echo "  logs/<run>.gates.txt                the harness's lines (W3, BLOCK, CARD HOLD, NO-EPOCH BLOCK, LFM PROVE, test result)"
    echo "  passes.tsv                          run B per pass: rc, launches profiled, seconds, CPU, verdict"
    echo "  env.txt · preflight.txt · prereg.txt · steps.tsv · driver.log · cubin_res_usage-<w>.txt"
    echo "Paths, hostname and addresses of the machine are replaced by <W>, <HOME>, <HOST>, <IP>."
  } > "$SEND/INDEX.txt"
  sensitive_pairs | python3 "$TOOL" scrub --dir "$SEND"
  scan="$(sensitive_values | python3 "$TOOL" check --dir "$SEND" --report "$KEEP/selfcheck.txt")" || rc=$?
  log "$scan"
  tarball="$RUN/counters-$STAMP.tar.gz"
  rm -f "$tarball"
  if [ "$rc" -ne 0 ]; then
    echo "==================== NOTHING TO SEND: THE SELF-CHECK REFUSED ===================="
    echo "$scan"
    echo "Read the lines named in $KEEP/selfcheck.txt (file:line:what, never the text). When they are"
    echo "harmless, allow those environment variable names (NP_SCAN_ALLOW=\"NAME ...\") or edit the lines, then:"
    echo "  bash $0 --bundle-only $RUN"
    return 4
  fi
  tar -czf "$tarball" -C "$RUN" send
  cp "$tarball" "$W/send-back.tar.gz"
  size="$(du -h "$tarball" | awk '{ print $1 }')"
  echo "==================== SEND BACK ===================="
  echo "ONE file: $W/send-back.tar.gz ($size; the same as $tarball)"
  echo "$scan"
  echo "It holds send/ (INDEX.txt says what each file is; SUMMARY.md alone answers the questions)."
  echo "Do NOT send anything else from $W: keep/ holds the .nsys-rep, .sqlite and .ncu-rep files, which store this machine's environment."
  BUNDLE_LINE="bundle $W/send-back.tar.gz ($size) · self-check clean"
  return 0
}

# ---------------------------------------------------------------------------------------------------
# main

cleanup() { # stop the samplers; then let tee write the last lines before the shell exits
  if [ -n "$CPU_PID" ]; then kill "$CPU_PID" 2>/dev/null || true; fi
  stop_smi
  if [ -n "$TEE_PID" ]; then exec 1>&- 2>&-; wait "$TEE_PID" 2>/dev/null || true; fi
}
main() {
  local preflight_only=0 bundle_only="" print_cmds=0 rc t_start
  t_start="$(date +%s)"
  while [ $# -gt 0 ]; do
    case "$1" in
      --quick) NP_MODE=quick ;;
      --full) NP_MODE=full ;;
      --preflight-only) preflight_only=1 ;;
      --bundle-only) bundle_only="${2:?--bundle-only needs a run directory}"; shift ;;
      --print-plan) NP_PLAN="${NP_PLAN:-}"; plan_source; exit 0 ;;
      --print-prereg) prereg_text; exit 0 ;;
      --print-tool) print_tool; exit 0 ;;
      --print-commands) print_cmds=1 ;;
      -h|--help) usage; exit 0 ;;
      *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
  done
  knob_defaults
  DEADLINE_EPOCH=$((t_start + NP_DEADLINE_MIN * 60))
  if [ "$print_cmds" = 1 ]; then
    print_commands | while IFS= read -r l; do printf '%s\n' "${l//"PATH=$PATH"/PATH=<PATH>}"; done
    exit 0
  fi
  snapshot_env
  trap cleanup EXIT
  trap 'log "interrupted"; exit 130' INT TERM
  if [ -n "$bundle_only" ]; then
    W="$(cd "$NP_WORKDIR" && pwd -P)"; TOOL="$W/tools/noepoch_counters_summary.py"
    RUN="$(cd "$bundle_only" && pwd -P)"; SEND="$RUN/send"; KEEP="$RUN/keep"; STAMP="$(basename "$RUN")"
    [ -d "$SEND" ] && [ -d "$KEEP" ] || die 2 "$RUN is not a run directory (no send/ and keep/)"
    make_bundle
    exit $?
  fi
  setup_workdir
  setup_run
  load_plan
  log "noepoch_counters.sh $SCRIPT_VERSION · mode $NP_MODE$([ "$NP_DRY" = 1 ] && echo ' (DRY: no counters)') · work directory $W · run $RUN · workloads $NP_WORKLOADS · reference $NP_REFERENCE · run A $NP_RUN_A · run B $NP_RUN_B · deadline ${NP_DEADLINE_MIN} min"
  step_begin "preflight"
  preflight
  step_end "$PF_FAILS"
  if [ "$PF_FAILS" -ne 0 ]; then
    cp "$KEEP/preflight.txt" "$SEND/preflight.txt"
    die 9 "$PF_FAILS check(s) failed (see the FAIL lines above, also in $KEEP/preflight.txt)"
  fi
  if [ "$preflight_only" = 1 ]; then log "--preflight-only: done (all checks passed)"; exit 0; fi
  estimate
  fetch_repo
  fetch_fixtures
  build_all
  write_env_facts
  if [ "$NP_REFERENCE" = 1 ]; then reference_runs; fi
  if [ "$NP_RUN_A" = 1 ]; then run_a; fi
  if [ "$NP_RUN_B" = 1 ] && [ "$NP_DRY" = 0 ]; then
    run_passes
    step_begin "summary: run B, one row per kernel"
    rc=0
    python3 "$TOOL" summary --plan "$PLAN_FILE" --ncu-dir "$SEND/ncu" --out "$SEND/summary" > "$KEEP/summary.out" 2>&1 || rc=$?
    step_end "$rc"
    if [ "$rc" -eq 0 ]; then sed -n '/^## per kernel, both workloads/,/^## [a-z]*: one row/p' "$SEND/summary/kernels.md" | sed '$d'
    else log "the summary failed (rc=$rc): see $KEEP/summary.out"; fi
  fi
  rc=0
  make_bundle || rc=$?
  if [ "$NP_CLEAN" = 1 ]; then
    log "NP_CLEAN=1: deleting the build and keep/'s traces"
    rm -rf "$TARGET" "$KEEP"/*.sqlite "$KEEP"/*.nsys-rep "$KEEP"/ncu/*.ncu-rep 2>/dev/null || true
  fi
  if [ "$rc" -ne 0 ]; then echo "VERDICT: $MODE_WORD FAILED rc=4 — the self-check refused the bundle (nothing packed)"; exit 4; fi
  local line="reference: ${REF_LINE:-skipped} (ok=$REF_OK) · run A: ${RUNA_LINE:-skipped} (ok=$RUNA_OK)"
  if [ "$NP_RUN_B" = 1 ] && [ "$NP_DRY" = 0 ]; then line="$line · run B: $PASSES_OK/$PASSES_RUN passes profiled ($LAUNCHES_TOTAL launches), $PASSES_SKIPPED skipped"; fi
  line="$line · $(( ($(date +%s) - t_start) / 60 )) min"
  if [ "$REF_OK" = 1 ] && [ "$RUNA_OK" = 1 ] && { [ "$NP_RUN_B" = 0 ] || [ "$NP_DRY" = 1 ] || { [ "$PASSES_OK" -eq "$PASSES_RUN" ] && [ "$PASSES_RUN" -gt 0 ] && [ "$PASSES_SKIPPED" -eq 0 ]; }; }; then
    echo "VERDICT: $MODE_WORD DONE — $line · $BUNDLE_LINE"
    exit 0
  fi
  echo "VERDICT: $MODE_WORD PARTIAL rc=3 — $line (reference/runs.tsv, runa/runs.tsv, passes.tsv say which) · $BUNDLE_LINE"
  exit 3
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
