#!/usr/bin/env bash
# noepoch_counters.sh: GPU counters on the no-epoch block proof against the epoch base, for a machine
# whose counters are open (lane I-PROF, 2026-09-30). Mauro runs it himself; nobody logs in.
#
# WHAT IT ANSWERS
#   Why the no-epoch STARK base (the whole block 25368371 as ONE proof, prove_block, PR #1013) runs
#   ~31 s where #1009's epoch base runs ~26 s on the same card (FAST, after packing admission, KECCAK_RND
#   chunking and device page roots). Both run on one binary, so every difference is the schedule and the instance sizes:
#   (a) is the card saturated or idle in each stage (head, main commit, fused, the recommit)?
#   (b) are the kernels that carry the fused and recommit time compute-, memory- or latency-bound, and
#       do the block-wide instances run them worse than the epoch-sized ones?
#   (c) how much time the host threads spend in CUDA allocation and synchronisation calls, per thread
#       and per stage.
#
# WHAT IT RUNS  (the structure of mauro-ncu.sh, lane G5-NCU, which ran on this machine on 2026-09-28)
#   1. Checks; each failure refuses with exit 9 and says what to fix: Linux; an NVIDIA GPU with nothing
#      else on it; nvcc; Nsight Systems and Nsight Compute, and GPU counters they can read (a one-kernel
#      probe; ERR_NVGPUCTRPERM gets the fix printed); RAM (the no-epoch run peaks at 43.9 GiB on the
#      host) and disk; the Rust toolchain.
#   2. Clones https://github.com/yetanotherco/lambda_vm (public, HTTPS) into its own work directory and
#      checks out PIN_SHA below: noepoch/stark @ fe3f6bf05 (the no-epoch prover: packing admission
#      behind a knob; KECCAK_RND chunked to 2^16 and the data pages' roots derived on the device, both
#      default-on) plus one instruments-only commit, an NVTX range around the device recommit. That
#      branch sits 7 commits on #1009's head (stark-recursion-rpx @ cc411aa2c) and carries #1009's base
#      as a control test, so ONE build runs both workloads:
#        epoch    tests::noepoch_block_tests::noepoch_epoch_base_reference   (#1009's base: 15 epochs of
#                 2^21 rows and the global proof, continuation::prove_continuation; no recursion, no verify)
#        noepoch  tests::noepoch_block_tests::noepoch_block_prove_and_verify (prove_block with
#                 LAMBDA_VM_GATE_PACKING=1, then the monolithic verifier)
#      both in the FAST 351/352 posture (i-noepoch's nm-ab.sh): TABLE_PARALLELISM=4, a 24000 MiB VRAM
#      budget, 2^21-row caps, the PROVE SPLIT line and the per-table timeline on.
#   3. Fetches block 25368371 from the public release (the Makefile's ETHREX_REAL_BLOCK_FIXTURE_URL) and
#      the record's guest ELF from the public repository; both are checked by sha256. No guest is built.
#   4. Builds the prover's test binary with --features cuda,nvtx. math-cuda's build.rs compiles the
#      kernels for this GPU's arch, here with LAMBDA_VM_NVCC_LINEINFO=1 (line tables, codegen unchanged).
#   5. Reference: each workload once with no profiler (its wall on this machine, and the control for
#      the profilers' overhead).
#   6. Run A, Nsight Systems: each workload once with CUDA API + NVTX tracing and GPU metrics sampled at
#      2 kHz, plus nvidia-smi VRAM every 200 ms. Per stage (from the prover's NVTX ranges): wall, card
#      busy %, SM active/issue %, warps in flight %, DRAM and PCIe throughput, VRAM, open fused tasks
#      and the top kernels; the fused stage second by second; the kernels each NVTX label launched (the
#      recommit's own); the CUDA API seconds per category (alloc, free, pinned host, sync, copies),
#      per thread and per stage; a 100 ms time series.
#   7. Run B, Nsight Compute: 32 short passes, 16 per workload, on the kernels that carry the main
#      commit, the fused rounds and the recommit (RPX leaves and Merkle levels, the NTT, the compiled
#      constraint kernels, DEEP, FRI leaves, grinding), with the sections SpeedOfLight, MemoryWorkloadAnalysis,
#      ComputeWorkloadAnalysis, Occupancy and LaunchStats plus stall and pipe metrics, at --clock-control
#      base. A config pass profiles one launch of every launch shape of ONE kernel, so each instance size
#      either workload ran is measured; a window pass profiles the first N launches of a family and ends
#      the run (--kill yes). Run B's workloads run at a 16000 MiB VRAM budget (run A: 24000), so the card
#      keeps room for ncu's replay copies.
#   8. Exports text only (nsys stats CSVs, the stage tables, ncu --import --page details --csv, one
#      summary row per kernel, SUMMARY.md), scrubs this machine's paths, hostname and addresses out of
#      it, self-checks it and packs it.
#
# EXPECTED RUNTIME  about 60-90 min on a Ryzen 9 9950X3D + RTX 5090: the build 5-30 min in a fresh work
#   directory (a re-run reuses it), the reference runs 2-3 min, run A 8-12 min, run B 40-50 min. The
#   script prints its own estimate after the checks. Run it inside tmux or screen.
# DISK  about 25-35 GiB in the work directory: the clone and cargo cache (~3 GiB), the release CUDA build
#   (~12-16 GiB), run A's two reports and their sqlite exports (~5-12 GiB), run B's reports (~1 GiB).
#   The file to send back is a few MB. keep/ can be deleted once the bundle is sent.
#
# NEEDS  Linux; an NVIDIA GPU with nothing else running on it (the record's card is an RTX 5090, 32 GB);
#   CUDA toolkit >= 12.8 with nvcc at $CUDA_HOME/bin (default /usr/local/cuda); Nsight Systems and Nsight
#   Compute >= 2025.1 (the toolkit's own are fine); GPU performance counters open to this user; RAM: 47 GiB
#   available when the script starts (the no-epoch run peaks at 43.9 GiB on the host; close big
#   programs first); >= 40 GiB free in the work directory; rustup with the toolchain 1.94.0; git, curl,
#   python3 >= 3.8 with sqlite3; internet (GitHub, crates.io, files.pythonhosted.org). Named NVTX ranges
#   need a libnvToolsExt (CUDA 12.9 and later ship none): LAMBDA_VM_NVTX_LIB=/path/to/libnvToolsExt.so.1
#   when you have one (as on 2026-09-28); otherwise the script fetches the nvidia-nvtx-cu12 12.8.90 wheel
#   (the last release that ships the library) and checks it by sha256. Without any, run A takes its
#   stages from the harness log and has no recommit row.
#
# RUN
#   bash noepoch_counters.sh --preflight-only   # the checks alone, about two minutes
#   bash noepoch_counters.sh                    # everything
#   The work directory is ./lambda-vm-noepoch-prof (NP_WORKDIR=/path to change). The script writes only
#   there: the clone, the cargo cache (CARGO_HOME), temporary files (TMPDIR), the tools' config (HOME)
#   and the runs. (Nsight itself keeps lock and IPC files in /tmp.)
#
# SEND BACK  one file, whose path the script prints at the end:
#   <work dir>/runs/<UTC>/noepoch-counters-<UTC>.tar.gz
#   It holds text only (CSV, TSV, markdown and log excerpts; SUMMARY.md first), with this machine's
#   paths, hostname and addresses replaced by <W>, <HOME>, <HOST>, <IP>, and it has passed the self-check
#   printed beside it (no Nsight report or database, no credential marker, no hostname, home path, IP
#   address or environment value). If you prefer to send less, SUMMARY.md alone answers the question.
#   The .nsys-rep, .sqlite and .ncu-rep files stay in <work dir>/runs/<UTC>/keep/ on this machine:
#   Nsight stores the profiled process's environment in them. If the self-check refuses, it names
#   file:line (never the text) and packs nothing; `--bundle-only <run dir>` re-checks and packs after a
#   look.
#
# OPTIONS  --preflight-only · --bundle-only RUNDIR · --print-plan · --print-prereg · --print-tool ·
#          --print-commands (every command the run would execute, nothing run) · -h|--help
# KNOBS (environment, all optional)
#   NP_WORKDIR         work directory (default ./lambda-vm-noepoch-prof)
#   NP_GPU             GPU index as nvidia-smi numbers it (default 0)
#   NP_WORKLOADS       the workloads to run, in order (default "epoch noepoch")
#   NP_REFERENCE       1 (default) runs the unprofiled reference runs; 0 skips them
#   NP_RUN_A           1 (default) runs run A (Nsight Systems); 0 skips it
#   NP_RUN_B           1 (default) runs run B (Nsight Compute); 0 skips it (run A alone: ~15 min)
#   NP_PASSES          run only these run-B passes, e.g. "noepoch_merkle epoch_merkle" (--print-plan lists them)
#   NP_PLAN            a plan file in --print-plan's format instead of the built-in one
#   NP_SECTIONS        ncu section identifiers (default: the five above)
#   NP_METRICS         explicit ncu metrics, comma-separated (default: the list below; the checks keep only
#                      what this ncu and GPU can collect, since one unknown name makes ncu profile nothing)
#   NP_CLOCK           ncu --clock-control: base (default, as on 09-25 and 09-28), boost or none
#   NP_GPU_METRICS_HZ  run A's GPU-metrics sampling rate (default 2000)
#   NP_NCU_VRAM_MB     run B's VRAM budget in MiB (default 16000; run A and the reference: 24000)
#   NP_PASS_TIMEOUT    seconds per run-B pass (2400) · NP_RUN_TIMEOUT per reference / run-A run (1800)
#   NP_BUILD_TIMEOUT   seconds for the build (7200)
#   NP_IDLE_MIB        the GPU counts as idle below this many MiB in use with no compute process (500)
#   NP_MIN_AVAIL_GIB   host MemAvailable the checks require at start (default 47)
#   NP_FETCH_NVTX      1 (default) fetches the NVTX library when none is found; 0 does not
#   NP_SCAN_ALLOW      environment variable NAMES whose values may appear in the bundle (after a look)
#   NCU / NSYS         explicit tool paths
#   NP_REPO_URL / NP_INPUT_URL   mirrors; the content stays pinned by commit sha and by sha256
#   NP_DRY=1           the lead's validation on a box whose counters are closed: every check that needs no
#                      counters, the build, the reference runs, run A without GPU metrics, and every run-B
#                      pass's selection evaluated against run A's traces instead of running ncu
# EXIT  0 done · 2 usage · 3 finished, something failed (the VERDICT line says what) · 4 the self-check
#       refused the bundle · 5 clone, fixture or build failed · 9 a check refused (the message says what)
set -euo pipefail
umask 022

SCRIPT_VERSION=iprof-2026-09-30c
REPO_URL_DEFAULT=https://github.com/yetanotherco/lambda_vm
PIN_SHA=80e6aa89f841950c46a76d66c6b87f83e44290b8            # profile/noepoch-counters: noepoch/stark + the recommit span
NOEPOCH_SHA=fe3f6bf0538936ad8c1b4dc015d42ff383c90e2c        # noepoch/stark (PR #1013), PIN_SHA's parent
EPOCH_SHA=cc411aa2c7a779464b577b5751eb846cf755b3d6          # stark-recursion-rpx (#1009), an ancestor of both
ELF_COMMIT=da2a423b137b87e04213ecafbfd5d16e98a1f4a3         # whir/profile-rpx, where the record ELF is committed
ELF_REPO_PATH=scripts/profile/fixtures/ethrex_8f826601.elf
ELF_SHA256=8f826601776d4085cbb6fbf0302fe8d8d5d1be7940ac1aaca24899c6244ec80a
INPUT_URL_DEFAULT=https://github.com/yetanotherco/lambda_vm/releases/download/bench-fixtures-v1/ethrex_mainnet_25368371_797df554.bin
INPUT_SHA256=573004e62e3680a00d3cdbae19dc4897e2ec60d6ec0c1d05d9ef118cb8aef17f
RUST_STABLE=1.94.0             # rust-toolchain.toml at the pin
FEATURES=cuda,nvtx             # nvtx implies cuda and instruments: the prover's phase spans become NVTX ranges
MIN_CUDA=12.8
MIN_NSIGHT=2025.1              # Blackwell-capable nsys and ncu
EPOCH_TEST=tests::noepoch_block_tests::noepoch_epoch_base_reference
NOEPOCH_TEST=tests::noepoch_block_tests::noepoch_block_prove_and_verify
NVTX_WHEEL_URL=https://files.pythonhosted.org/packages/a2/eb/86626c1bbc2edb86323022371c39aa48df6fd8b0a1647bc274577f72e90b/nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
NVTX_WHEEL_SHA256=5b17e2001cc0d751a5bc2c6ec6d26ad95913324a4adb86788c944f8ce9ba441f
NVTX_LIB_SHA256=c498fcbab0202886c27a0adeac44abf233ade03d30680ffa2d2abe93ab88d913   # nvidia/nvtx/lib/libnvToolsExt.so.1 in it
RUN_VRAM_MB=24000              # the record's budget (FAST 351/352)
CUBINS="arith ntt ntt_cm keccak barycentric deep fri inverse rpx_v0 rpx_v5 logup constraint_interp constraint_compiled blake3 sumcheck whir_fold"

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

# Run B's pass plan: the same 16 passes for each workload, the kernels that carried the STARK base on
# 09-28 (lane G1's ledger, NCU-5090.md): RPX leaves and Merkle levels (main commit, recommit, aux and
# composition trees), the NTT (LDE), the compiled constraint kernels (ccomp_<hash>, one per constraint
# program since cc411aa2c, so a window rather than a config pass), DEEP and the FRI leaves. Kernels are
# a regex BODY: the tools anchor it as ^(...)$. A config pass names exactly ONE kernel, because ncu's
# per-launch-config key is the grid, block and shared memory, not the kernel name; it profiles the first
# launch of every shape, i.e. every instance size the workload ran. A window pass profiles the first
# `count` matching launches (--launch-skip/--launch-count) and ends the run (--kill yes).
plan_rows() { # plan_rows WORKLOAD
  local w="$1"
  printf '%s\n' \
    "${w}_rowpair	$w	config	0	1	rpx_leaves_base_row_pair_batched	leaves	main-trace leaves, row pairs (the main commit; in no-epoch also the recommit), one launch of each shape" \
    "${w}_merkle	$w	config	0	1	rpx_merkle_level	merkle	Merkle internal levels, every tree, one launch of each width" \
    "${w}_batched	$w	config	0	1	rpx_leaves_base_batched	leaves	main-trace leaves, batched (the main commit and the recommit), one launch of each shape" \
    "${w}_rowrange	$w	config	0	1	rpx_leaves_base_row_major_row_range	leaves	main-trace leaves, row-major ranges, one launch of each shape" \
    "${w}_rowpair_range	$w	config	0	1	rpx_leaves_base_row_major_row_pair_range	leaves	main-trace leaves, row-major row-pair ranges, one launch of each shape" \
    "${w}_ext3_leaves	$w	config	0	1	rpx_leaves_ext3_batched	leaves	aux-trace leaves (ext3, the fused aux commit), one launch of each shape" \
    "${w}_comp_poly	$w	config	0	1	rpx_comp_poly_leaves_ext3	leaves	composition-polynomial leaves (fused round 2), one launch of each shape" \
    "${w}_fri_leaves	$w	config	0	1	rpx_fri_group_leaves_ext3	leaves	FRI layer leaves (fused round 4), one launch of each shape" \
    "${w}_ntt_dit7	$w	config	0	1	ntt_cm_dit_k7	lde	LDE, the column-major engine's busiest kernel, one launch of each shape" \
    "${w}_ntt_dit8	$w	config	0	1	ntt_cm_dit_k8	lde	LDE, the column-major engine's k8 kernel, one launch of each shape" \
    "${w}_deep	$w	config	0	1	deep_composition_ext3_fused_m3	deep	DEEP composition, the fused kernel (round 4), one launch of each shape" \
    "${w}_quotient	$w	window	0	32	ccomp_[0-9a-f]+|constraint_composition_kernel	quotient	constraint composition (round 2): the compiled per-program kernels, the first 32 launches (largest tables first in the walk)" \
    "${w}_ntt_other	$w	window	0	24	ntt_cm_dif_k[4-8]|ntt_cm_dit_k[456]|ntt_dit_level_row_major|ntt_dit_8_levels_row_major|matrix_transpose_strided|bit_reverse_row_major	lde	LDE, the rest (dif kernels, legacy row-major passes, transposes): the first 24" \
    "${w}_merkle_warp	$w	window	0	16	rpx_merkle_(level_warp|tail_warp)	merkle	narrow Merkle levels, the half-warp kernels: one tree's first 16" \
    "${w}_logup	$w	window	0	8	logup_[a-z0-9_]+	logup	LogUp aux columns (fused aux build): the first 8" \
    "${w}_grind	$w	window	0	5	rpx_grind_search_queue	grind	grinding, the queue kernel (round 4): the first 5"
}
default_plan() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' pass workload mode skip count kernels family note
  plan_rows noepoch
  plan_rows epoch
}

prereg_text() {
  cat <<'PREREG'
Pre-registered 2026-09-30 (lane I-PROF), before any run of this script. Sources: FAST 351 (ds1002, i-noepoch
I-NOEPOCH.md §5), FAST 352 (ds1003, packing: -0.64 s) and the later FAST arms at fe3f6bf05 (KECCAK_RND
chunking -3.23 s, device page roots -1.43 s: base 30.9 s, per the lead). Mauro's machine is not FAST (another CPU), so the
walls get wide bands; the structural checks are exact.

Gates (a miss makes the VERDICT PARTIAL or FAILED):
  - every reference and run-A process passes its test (the no-epoch proof verifies);
  - the no-epoch log carries the packing banner (LAMBDA_VM_GATE_PACKING=1) and a PROVE SPLIT line with a
    recommit[Σ] field; the epoch log carries PROVE SPLIT lines (16 on FAST 351: 15 epochs and the global proof);
  - run A's stages come from NVTX ranges (needs LAMBDA_VM_NVTX_LIB), and the no-epoch trace has exactly as
    many r1_main_recommit_table ranges as the log's `device recommits` (127 on FAST 351, before chunking); the epoch trace 0;
  - every run-B pass profiles at least one launch.
Expected (reported, not gated):
  - reference walls: epoch base 23-30 s (FAST 26.05), no-epoch base 27-36 s (FAST 30.9 at fe3f6bf05);
    nsys run A within +15 % of the reference; host peak of the no-epoch run 35-50 GiB (FAST 43.9 at ae11e232e);
  - run A, no-epoch fused stage 13-24 s (FAST 22.7 at ae11e232e, before packing and chunking), with recommit
    ranges covering 3-12 s of it;
    VRAM max near the card's 32 GB in fused (FAST 32,110 MiB);
  - if the fused stage is card-bound, its SM active % is within 10 points of the epoch base's fused stage; a
    no-epoch fused SM active % 15 or more points lower, with fewer than 2.5 open tasks on average, reads as
    host/admission starvation, not a kernel problem.
PREREG
}

# ---------------------------------------------------------------------------------------------------
# small helpers

log() { printf 'NP %s %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
MODE_WORD="NOEPOCH-COUNTERS"
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
mem_avail_mib() { awk '/^MemAvailable:/ { printf "%d", $2 / 1024 }' /proc/meminfo 2>/dev/null || true; }
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
  NP_WORKDIR="${NP_WORKDIR:-$PWD/lambda-vm-noepoch-prof}"
  NP_GPU="${NP_GPU:-0}"
  NP_DRY="${NP_DRY:-0}"
  NP_WORKLOADS="${NP_WORKLOADS:-epoch noepoch}"
  NP_REFERENCE="${NP_REFERENCE:-1}"
  NP_RUN_A="${NP_RUN_A:-1}"
  NP_RUN_B="${NP_RUN_B:-1}"
  NP_PASSES="${NP_PASSES:-}"
  NP_PLAN="${NP_PLAN:-}"
  NP_SECTIONS="${NP_SECTIONS:-$DEFAULT_SECTIONS}"
  NP_METRICS="${NP_METRICS:-$DEFAULT_METRICS}"
  NP_CLOCK="${NP_CLOCK:-base}"
  NP_GPU_METRICS_HZ="${NP_GPU_METRICS_HZ:-2000}"
  NP_NCU_VRAM_MB="${NP_NCU_VRAM_MB:-16000}"
  NP_PASS_TIMEOUT="${NP_PASS_TIMEOUT:-2400}"
  NP_RUN_TIMEOUT="${NP_RUN_TIMEOUT:-1800}"
  NP_BUILD_TIMEOUT="${NP_BUILD_TIMEOUT:-7200}"
  NP_IDLE_MIB="${NP_IDLE_MIB:-500}"
  NP_MIN_AVAIL_GIB="${NP_MIN_AVAIL_GIB:-47}"
  NP_FETCH_NVTX="${NP_FETCH_NVTX:-1}"
  NP_REPO_URL="${NP_REPO_URL:-$REPO_URL_DEFAULT}"
  NP_INPUT_URL="${NP_INPUT_URL:-$INPUT_URL_DEFAULT}"
  NP_SCAN_ALLOW="${NP_SCAN_ALLOW:-}"
  case "$NP_GPU" in ''|*[!0-9]*) echo "NP_GPU must be a number, got '$NP_GPU'" >&2; exit 2 ;; esac
  for v in NP_DRY NP_REFERENCE NP_RUN_A NP_RUN_B NP_FETCH_NVTX; do
    case "${!v}" in 0|1) ;; *) echo "$v must be 0 or 1, got '${!v}'" >&2; exit 2 ;; esac
  done
  case "$NP_CLOCK" in base|boost|none) ;; *) echo "NP_CLOCK must be base, boost or none, got '$NP_CLOCK'" >&2; exit 2 ;; esac
  for v in NP_NCU_VRAM_MB NP_PASS_TIMEOUT NP_RUN_TIMEOUT NP_BUILD_TIMEOUT NP_IDLE_MIB NP_GPU_METRICS_HZ NP_MIN_AVAIL_GIB; do
    case "${!v}" in ''|*[!0-9]*) echo "$v must be a number, got '${!v}'" >&2; exit 2 ;; esac
  done
  [ -n "$NP_WORKLOADS" ] || { echo "NP_WORKLOADS is empty" >&2; exit 2; }
  for w in $NP_WORKLOADS; do
    case "$w" in epoch|noepoch) ;; *) echo "NP_WORKLOADS: unknown workload '$w' (epoch, noepoch)" >&2; exit 2 ;; esac
  done
  if [ "$NP_DRY" = 1 ]; then MODE_WORD="NOEPOCH-COUNTERS DRY-RUN"; fi
}

setup_workdir() {
  mkdir -p "$NP_WORKDIR"
  W="$(cd "$NP_WORKDIR" && pwd -P)"
  REPO="$W/lambda_vm"
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
  mkdir -p "$SEND/ncu" "$SEND/logs" "$SEND/summary" "$SEND/runa" "$SEND/reference" "$KEEP/ncu"
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

write_tool() { # the Python side (stages, API tables, ncu summary, scrub, self-check): noepoch_counters_summary.py, verbatim
  cat > "$TOOL" <<'NOEPOCH_COUNTERS_SUMMARY_PY_EOF'
#!/usr/bin/env python3
"""noepoch_counters_summary.py: the text side of noepoch_counters.sh (lane I-PROF, 2026-09-30).

Standard library only (python >= 3.8; `runa` and `dry` also need the sqlite3 module).
noepoch_counters.sh carries a byte-identical copy of this file and writes it into its
work directory; the copy next to the script is the one to read and edit. It is a fork of
mauro_ncu_summary.py (lane G5-NCU, 2026-09-28): the ncu parsing, the scrub and the
self-check are that file's; the stages, the CUDA API tables and the time series are new.

    runa           --sqlite S --log L --workload W --out O [--smi F] [--bin-ms N]
                                                   run A: per stage (head, prepass, main_commit, between,
                                                   fused, tail; recommit as an overlapping row) the card's busy
                                                   time, GPU metrics, VRAM and the kernels; the kernels each NVTX
                                                   label launched; CUDA API time per category, thread and stage;
                                                   a time series; from an nsys trace and the harness log
    summary        --plan P --ncu-dir D --out O    run B: one row per kernel, and epoch against no-epoch
    stages         --log L                         run B: the stage of each profiled launch, from one pass's output
    dry            --plan P --workload W --sqlite S --log L --out O
                                                   what each ncu pass would profile, against run A's trace
    report         --send D                        SUMMARY.md: the text summary of a run, from the files above
    plan-check     --plan P                        the pass plan is well formed
    cargo-artifact (--exe NAME | --outdir PKG)     one path from cargo's JSON messages on stdin
    scrub          --dir D                         literal replacements (stdin: OLD<TAB>NEW lines), and the
                                                   "Host Name" column dropped from ncu CSVs
    check          --dir D --report R              the bundle self-check (stdin: LABEL<TAB>VALUE records,
                                                   NUL-separated): plain text only, no Nsight report or
                                                   database, no credential marker, none of the values
    selftest                                       every subcommand on synthetic inputs

Stages (run A). The prover's instruments spans are NVTX ranges in a --features nvtx build:
r1_prepass, r1_main_commit and rounds_2to4 once per multi_prove on the calling thread, and per
table on the driver threads r1_main_recommit_table (the no-epoch device recommit), r1_aux_build_table,
r1_aux_commit_table and rounds_2to4_table. The disjoint stages are: head = the trace start to the
first prepass; prepass, main_commit, fused = the unions of those ranges; between = the rest of
[first prepass, last fused end] (the absorb, and in the epoch base the waits between epochs);
tail = after the last fused range (the no-epoch arm's verify). recommit = the union of the
recommit ranges, a part of fused, reported as its own row. Without NVTX ranges (no libnvToolsExt)
the stages come from the log's PROVE SPLIT lines (t=[..] plus the phase walls, the `other`
remainder unplaced) and there is no recommit row.

What `summary` reports, per profiled launch and per kernel (duration-weighted over its launches):
  time      ncu's Duration, at the clock ncu held (launches.tsv's sm_ghz; env.txt names the
            --clock-control mode). Under `base` an RTX 5090 runs its SMs near 2.0 GHz where the block
            runs near 2.76 GHz, so a compute-bound kernel's duration here is ~1.4x its in-block time.
            The percentages are against the peak at the clock ncu ran.
  DRAM %    dram throughput, % of peak (Speed Of Light)
  SM %      Compute (SM) throughput, % of peak (Speed Of Light)
  L2 %      L2 throughput, % of peak (Speed Of Light); L2 hit % from Memory Workload Analysis
  occ %     achieved occupancy (theoretical beside it, and the block limit that sets it)
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
import sys
import tempfile
from datetime import datetime

PLAN_COLS = ["pass", "workload", "mode", "skip", "count", "kernels", "family", "note"]
WORKLOADS = ("epoch", "noepoch")
MODES = ("window", "config")

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
            "sm_pct", "l2_pct", "l2_hit", "occ", "occ_theo", "issue_pct", "b_per_elem", "elem", "roof", "bound",
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
        for c in ("dram_pct", "sm_pct", "l2_pct", "l2_hit", "occ", "occ_theo", "issue_pct"):
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
    o.write("## epoch base against no-epoch, per kernel\n\n")
    o.write("Largest = the profiled launch with the most threads (the biggest instance each workload ran).\n\n")
    o.write("| kernel | family | workload | configs | SM % | DRAM % | L2 % | occ % (theo) | bound | largest: shape, "
            "µs, SM %, DRAM %, occ % |\n|---|---|---|---|---|---|---|---|---|---|\n")
    for name in sorted(by, key=lambda n: -max(v["dur_us_sum"] for v in by[n].values())):
        for wl in WORKLOADS + tuple(sorted(set(by[name]) - set(WORKLOADS))):
            k = by[name].get(wl)
            if k is None:
                continue
            o.write(f"| {name} | {k['family']} | {wl} | {k['configs']} | {fmt(k['sm_pct'])} | {fmt(k['dram_pct'])} | "
                    f"{fmt(k['l2_pct'])} | {fmt(k['occ'])} ({fmt(k['occ_theo'], 0)}) | {k['bound']} | "
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
    o.write("## per workload, stage and kernel (a main-commit kernel in the no-epoch fused stage is the recommit)\n\n")
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
R1_WALK_RE = re.compile(r"\[prover\] table walk R1\b")
FUSED_WALK_RE = re.compile(r"\[prover\] table walk rounds 2-4\b")
SPLIT_LINE_RE = re.compile(r"PROVE SPLIT #\d+")


def stages_from_lines(lines):
    """[(id, kernel, stage, passes)]. The k-th `==PROF== Profiling` line is report ID k-1 (ncu
    numbers results in the order it profiles them). The stage is set by the last prover line seen
    before it: `head` until the first `table walk R1` line, `main_commit` from it, `fused` from
    `table walk rounds 2-4`, `between` from the `PROVE SPLIT` line that closes the prove (the next
    prove's R1 line opens `main_commit` again). The prover prints those lines to stderr unbuffered
    and ncu prints its line while the launch is held, into the same file, so the order in the file
    follows the events. ncu cannot see the recommit: its launches read as `fused`."""
    stage, out, n = "head", [], 0
    for line in lines:
        line = line.rstrip("\n")
        m = PROF_RE.match(line)
        if m:
            out.append((n, m.group(1), stage, m.group(3) or ""))
            n += 1
            continue
        if R1_WALK_RE.search(line):
            stage = "main_commit"
        elif FUSED_WALK_RE.search(line):
            stage = "fused"
        elif SPLIT_LINE_RE.search(line):
            stage = "between"
    return out


def cmd_stages(a):
    with open(a.log, errors="replace") as f:
        rows = stages_from_lines(f)
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
    """[(start, end, label, globalTid)] of the closed NVTX ranges."""
    if "NVTX_EVENTS" not in tbls:
        return []
    cols = columns(db, "NVTX_EVENTS")
    txt = "e.text" if "text" in cols else "NULL"
    tid = "e.globalTid" if "globalTid" in cols else "0"
    if "textId" in cols and "StringIds" in tbls:
        q = (f"SELECT e.start, e.end, COALESCE({txt}, s.value), {tid} FROM NVTX_EVENTS e "
             f"LEFT JOIN StringIds s ON s.id = e.textId WHERE e.end IS NOT NULL AND e.end > e.start")
    else:
        q = f"SELECT e.start, e.end, {txt}, {tid} FROM NVTX_EVENTS e WHERE e.end IS NOT NULL AND e.end > e.start"
    return [(s, e, norm_label(lab), t) for s, e, lab, t in db.execute(q) if lab]


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
RESULT_RE = re.compile(r"NOEPOCH RESULT: verified=(\w+) · sub-proofs (\d+) · base ([0-9.]+) s \(execute ([0-9.]+) · "
                       r"build ([0-9.]+) · setup ([0-9.]+) · prove ([0-9.]+)\) · verify ([0-9.]+) s · proof (\d+) B · "
                       r"host peak ([0-9.]+ GiB|unknown) · device recommits (\d+)")
REFERENCE_RE = re.compile(r"NOEPOCH REFERENCE: epoch base ([0-9.]+) s · (\d+) epochs · host peak ([0-9.]+ GiB|unknown)")
TL_RE = re.compile(r"TABLE TL (\S+) idx=(\d+) (.*?) est=([0-9.]+)GiB claim=([0-9.]+) start=([0-9.]+) end=([0-9.]+)")


def read_log(path):
    lg = {"splits": [], "result": None, "reference": None, "tl": [], "recommit_sum": 0.0, "test_result": None,
          "packing": False, "compiled": None}
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
            m = RESULT_RE.search(line)
            if m and lg["result"] is None:
                g = m.groups()
                lg["result"] = {"verified": g[0], "subs": int(g[1]), "base": float(g[2]), "execute": float(g[3]),
                                "build": float(g[4]), "setup": float(g[5]), "prove": float(g[6]),
                                "verify": float(g[7]), "size": int(g[8]), "peak": g[9], "recommits": int(g[10])}
            m = REFERENCE_RE.search(line)
            if m and lg["reference"] is None:
                lg["reference"] = {"base": float(m.group(1)), "epochs": int(m.group(2)), "peak": m.group(3)}
            m = TL_RE.search(line)
            if m:
                g = m.groups()
                lg["tl"].append({"phase": g[0], "idx": int(g[1]), "label": g[2], "est_gib": float(g[3]),
                                 "claim": float(g[4]), "start": float(g[5]), "end": float(g[6])})
            if line.startswith("test result:"):
                lg["test_result"] = line.strip()
            if "packing admission (LAMBDA_VM_GATE_PACKING=1)" in line:
                lg["packing"] = True
    return lg


def base_line(lg):
    if lg["result"]:
        r = lg["result"]
        return (f"no-epoch base {r['base']:.2f} s (execute {r['execute']:.2f} · build {r['build']:.2f} · setup "
                f"{r['setup']:.2f} · prove {r['prove']:.2f}) · {r['subs']} sub-proofs · verified={r['verified']} · "
                f"verify {r['verify']:.2f} s · host peak {r['peak']} · device recommits {r['recommits']}")
    if lg["reference"]:
        r = lg["reference"]
        return f"epoch base {r['base']:.2f} s · {r['epochs']} epochs · host peak {r['peak']}"
    return "no NOEPOCH RESULT/REFERENCE line in the log"


# ---------------------------------------------------------------------------------------------
# the stage windows

STAGES = ("head", "prepass", "main_commit", "between", "fused", "tail")
RECOMMIT_LABEL = "r1_main_recommit_table"
TASK_LABELS = ("r1_main_recommit_table", "r1_aux_build_table", "r1_aux_commit_table", "rounds_2to4_table")


def trace_windows(nvtx, lg, t0_ns, end_ns):
    """({stage: intervals} over the disjoint STAGES plus `recommit` and `whole`, source note)."""
    by = {}
    for s, e, lab, _t in nvtx:
        by.setdefault(lab, []).append((s, e))
    pre, mc, fu = union(by.get("r1_prepass", [])), union(by.get("r1_main_commit", [])), union(by.get("rounds_2to4", []))
    rc = union(by.get(RECOMMIT_LABEL, []))
    src = "NVTX ranges"
    if not (pre or mc or fu) and lg["splits"] and t0_ns is not None:
        src = "the log's PROVE SPLIT lines (no NVTX ranges; `other` unplaced, no recommit row)"
        for sp in lg["splits"]:
            a = sp["t0"] * 1e9 - t0_ns
            b = sp["t1"] * 1e9 - t0_ns
            pre.append((a, a + sp["prepass"] * 1e9))
            mc.append((a + sp["prepass"] * 1e9, a + (sp["prepass"] + sp["main_commit"]) * 1e9))
            fu.append((b - sp["fused"] * 1e9, b))
        pre, mc, fu = union(pre), union(mc), union(fu)
        rc = []
    if not (pre or mc or fu):
        return {"whole": [(0, end_ns)]}, "no NVTX ranges and no PROVE SPLIT line: one window only"
    allp = union(pre + mc + fu)
    first, last = allp[0][0], max(e for _, e in fu) if fu else allp[-1][1]
    mc = subtract(mc, pre)
    fu = subtract(fu, union(pre + mc))
    w = {"head": [(0, first)] if first > 0 else [], "prepass": pre, "main_commit": mc, "fused": fu,
         "between": subtract([(first, last)], union(pre + mc + fu)),
         "tail": [(last, end_ns)] if end_ns > last else [], "recommit": rc, "whole": [(0, end_ns)]}
    if not rc:
        w.pop("recommit")
    return w, src


def partition(w):
    """Sorted disjoint (start, end, stage) segments of the disjoint stages."""
    segs = sorted((s, e, st) for st in STAGES for s, e in w.get(st, []))
    return segs


def stage_at(segs, starts, t):
    i = bisect.bisect_right(starts, t) - 1
    if i >= 0 and segs[i][0] <= t < segs[i][1]:
        return segs[i][2]
    return "tail" if segs and t >= segs[-1][1] else "head"


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
    end = max([e for _, e, *_ in kernels] + [e for _, e in copies] + [e for _, e, *_ in api] + [0])
    w, src = trace_windows(nvtx, lg, t0, end)
    kmerged = union([(s, e) for s, e, *_ in kernels])
    cmerged = union(copies)
    amerged = union([(s, e) for s, e, *_ in kernels] + copies)
    kstarts = [k[0] for k in kernels]
    maxdur = max([e - s for s, e, *_ in kernels] + [0])
    tasks = StepFn([(s, e) for s, e, lab, _ in nvtx if lab in TASK_LABELS])
    recs = StepFn([(s, e) for s, e, lab, _ in nvtx if lab == RECOMMIT_LABEL])
    keys = {k: mets.name_of(k) for k, _ in KEY_METRICS}
    os.makedirs(a.out, exist_ok=True)
    wl = a.workload

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

    order = [s for s in STAGES if s in w] + [s for s in ("recommit", "whole") if s in w]
    stages, per_kernel = [], {}
    for name in order:
        ivs = w[name]
        wall = total(ivs) / 1e9
        ksum, nl, kk = kernels_in(ivs)
        per_kernel[name] = kk
        vram, util = smi_in(smi, ivs)
        row = {"stage": name, "wall_s": wall, "ranges": len(ivs),
               "busy_pct": 100.0 * covered_ivs(amerged, ivs) / max(total(ivs), 1),
               "kernel_busy_pct": 100.0 * covered_ivs(kmerged, ivs) / max(total(ivs), 1),
               "copy_busy_pct": 100.0 * covered_ivs(cmerged, ivs) / max(total(ivs), 1),
               "kernel_sum_s": ksum, "launches": nl, "tasks_open": tasks.mean(ivs), "recommits_open": recs.mean(ivs),
               "vram_max_mib": vram, "smi_util_pct": util}
        for key, _ in KEY_METRICS:
            row[key] = mets.mean(keys[key], ivs) if keys[key] else None
        top = sorted(kk.items(), key=lambda kv: -kv[1][1])[:5]
        row["top"] = " ; ".join(f"{k} {v[1]:.2f}s ({100 * v[1] / ksum:.0f}%)" for k, v in top) if ksum else ""
        stages.append(row)
    cols = ["stage", "wall_s", "ranges", "busy_pct", "kernel_busy_pct", "copy_busy_pct", "kernel_sum_s", "launches",
            "tasks_open", "recommits_open", "vram_max_mib", "smi_util_pct"] + [k for k, _ in KEY_METRICS] + ["top"]

    def scell(c, v):
        return f"{v:.4f}" if c.endswith("_s") and isinstance(v, float) else cell(v)

    write_tsv(os.path.join(a.out, f"runa-{wl}-stages.tsv"), [cols] + [[scell(c, r.get(c)) for c in cols]
                                                                     for r in stages])
    rows = [["stage", "kernel", "launches", "sum_s", "share_of_stage_kernel_s"]]
    for st in stages:
        for k, (n, s) in sorted(per_kernel.get(st["stage"], {}).items(), key=lambda kv: -kv[1][1]):
            if s >= 0.001:
                rows.append([st["stage"], k, n, f"{s:.4f}",
                             f"{100 * s / st['kernel_sum_s']:.2f}" if st["kernel_sum_s"] else ""])
    write_tsv(os.path.join(a.out, f"runa-{wl}-stage-kernels.tsv"), rows)
    if mets.series:
        mrows = [["stage", "metric", "mean", "samples"]]
        for st in order:
            for n in sorted(mets.series):
                v = mets.mean(n, w[st])
                mrows.append([st, n, "" if v is None else f"{v:.3f}", mets.samples(n, w[st])])
        write_tsv(os.path.join(a.out, f"runa-{wl}-gpu-metrics.tsv"), mrows)

    # the kernels each NVTX label launched: the launch call on the label's thread, inside its range
    by_lab = {}
    for s, e, lab, t in nvtx:
        by_lab.setdefault(lab, {}).setdefault(t, []).append((s, e))
    for lab in by_lab:
        for t in by_lab[lab]:
            by_lab[lab][t].sort()
    attr = {}
    for lab, per_t in by_lab.items():
        d = {"ranges": sum(len(v) for v in per_t.values()), "range_s": sum(total(union(v)) for v in per_t.values()) / 1e9,
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
    arows = [["label", "ranges", "threads", "range_s_per_thread_union", "launches", "kernel_s", "top_kernels"]]
    krows = [["label", "kernel", "launches", "kernel_s"]]
    for lab, d in sorted(attr.items(), key=lambda kv: -kv[1]["kernel_s"]):
        top = sorted(d["k"].items(), key=lambda kv: -kv[1][1])
        arows.append([lab, d["ranges"], d["threads"], f"{d['range_s']:.3f}", d["launches"], f"{d['kernel_s']:.3f}",
                      " ; ".join(f"{k} {v[1]:.2f}s" for k, v in top[:5])])
        for k, (n, s) in top:
            krows.append([lab, k, n, f"{s:.4f}"])
    write_tsv(os.path.join(a.out, f"runa-{wl}-nvtx-kernels.tsv"), arows)
    write_tsv(os.path.join(a.out, f"runa-{wl}-nvtx-label-kernels.tsv"), krows)

    # CUDA API time per category, thread and stage (by the call's start)
    segs = partition(w)
    sstarts = [s[0] for s in segs]
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
    stl = [s for s in STAGES if s in w] or ["whole"]
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

    # the time series
    binn = max(int(a.bin_ms * 1e6), 1_000_000)
    ts_cols = ["t_s", "stage", "kernel_busy_pct", "copy_busy_pct", "tasks_open", "recommits_open", "vram_mib",
               "smi_util_pct"] + [k for k, _ in KEY_METRICS]
    ts_rows = [ts_cols]
    smi_t = [x[0] for x in smi]
    fused_sec = []
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
        r = [f"{b0 / 1e9:.3f}", st, fmt(100.0 * covered(kmerged, b0, b1) / binn), fmt(100.0 * covered(cmerged, b0, b1) / binn),
             fmt(tasks.mean(iv), 2), fmt(recs.mean(iv), 2), fmt(vr, 0), fmt(ut, 0)]
        r += [fmt(mets.mean(keys[k], iv)) if keys[k] else "-" for k, _ in KEY_METRICS]
        ts_rows.append(r)
    write_tsv(os.path.join(a.out, f"runa-{wl}-timeseries.tsv"), ts_rows)
    if "fused" in w and w["fused"]:
        f0, f1 = w["fused"][0][0], w["fused"][-1][1]
        step = 1_000_000_000
        for b0 in range(int(f0), int(f1), step):
            iv = subtract([(b0, min(b0 + step, f1))], subtract([(f0, f1)], w["fused"]))
            if not total(iv):
                continue
            vr, ut = smi_in(smi, iv)
            fused_sec.append((b0 - f0, iv, vr))
            if len(fused_sec) >= 120:
                break

    # the markdown
    o = io.StringIO()
    o.write(f"# Run A, {wl}: the workload under Nsight Systems\n\n")
    o.write(f"{base_line(lg)}.\n\n")
    o.write(f"trace: {len(kernels)} kernel launches, {len(copies)} copies/memsets, {len(api)} CUDA API calls, "
            f"{len(nvtx)} NVTX ranges, {end / 1e9:.1f} s from the session start; {len(lg['splits'])} PROVE SPLIT "
            f"line(s); TABLE TL lines {len(lg['tl'])}; packing admission {'on' if lg['packing'] else 'off'}. "
            f"Stages from {src}.\n\n")
    o.write("busy % = any kernel or copy on the card; kernel/copy busy % = any kernel / any copy or memset. tasks = "
            "the mean number of fused-task NVTX ranges open (recommit, aux build, aux commit, rounds 2-4, one per "
            "driver thread at a time); recommits = the mean open recommit ranges. VRAM = the nvidia-smi maximum in "
            "the stage (200 ms samples). GPU metrics are nsys's samples averaged over the stage "
            f"({mets.why or 'collected'}): warps = Compute Warps in Flight, % of the card's warp slots (an "
            "occupancy proxy over time). `recommit` overlaps `fused`; `whole` is the run.\n\n")
    o.write("| stage | wall s | busy % | kernel busy % | copy busy % | Σ kernel s | launches | tasks | recommits | "
            "VRAM MiB | SM active % | SM issue % | warps % | DRAM rd % | DRAM wr % | PCIe rx % | PCIe tx % | "
            "top kernels (Σ s in the stage) |\n")
    o.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in stages:
        o.write(f"| {r['stage']} | {fmt(r['wall_s'], 2)} | {fmt(r['busy_pct'])} | {fmt(r['kernel_busy_pct'])} | "
                f"{fmt(r['copy_busy_pct'])} | {fmt(r['kernel_sum_s'], 2)} | {r['launches']} | {fmt(r['tasks_open'], 2)} | "
                f"{fmt(r['recommits_open'], 2)} | {fmt(r['vram_max_mib'], 0)} | {fmt(r['sm_active'])} | "
                f"{fmt(r['sm_issue'])} | {fmt(r['warps_in_flight'])} | {fmt(r['dram_read'])} | {fmt(r['dram_write'])} | "
                f"{fmt(r['pcie_rx'])} | {fmt(r['pcie_tx'])} | {r['top']} |\n")
    if fused_sec:
        o.write("\n## the fused stage, second by second\n\n")
        o.write("| s into fused | kernel busy % | tasks | recommits | VRAM MiB | SM active % | warps % | DRAM rd % | "
                "DRAM wr % | PCIe rx % | PCIe tx % |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
        for off, iv, vr in fused_sec:
            o.write(f"| {off / 1e9:.0f} | {fmt(100.0 * covered_ivs(kmerged, iv) / max(total(iv), 1))} | "
                    f"{fmt(tasks.mean(iv), 2)} | {fmt(recs.mean(iv), 2)} | {fmt(vr, 0)} | "
                    f"{fmt(mets.mean(keys['sm_active'], iv)) if keys['sm_active'] else '-'} | "
                    f"{fmt(mets.mean(keys['warps_in_flight'], iv)) if keys['warps_in_flight'] else '-'} | "
                    f"{fmt(mets.mean(keys['dram_read'], iv)) if keys['dram_read'] else '-'} | "
                    f"{fmt(mets.mean(keys['dram_write'], iv)) if keys['dram_write'] else '-'} | "
                    f"{fmt(mets.mean(keys['pcie_rx'], iv)) if keys['pcie_rx'] else '-'} | "
                    f"{fmt(mets.mean(keys['pcie_tx'], iv)) if keys['pcie_tx'] else '-'} |\n")
    if attr:
        o.write("\n## kernels by the NVTX label that launched them\n\n")
        o.write("A kernel belongs to a label when its launch call ran on the label's thread inside one of its "
                "ranges (a launch from another thread, e.g. a rayon worker, is not attributed). range s = the "
                "ranges' time summed over threads.\n\n")
        o.write("| label | ranges | threads | range s | launches | Σ kernel s | top kernels |\n"
                "|---|---|---|---|---|---|---|\n")
        for r in arows[1:]:
            o.write("| " + " | ".join(str(x) for x in r) + " |\n")
        unattr = sum((e - s) for s, e, *_ in kernels) / 1e9 - (attr.get("r1_main_commit", {}).get("kernel_s", 0.0)
                                                               + sum(attr.get(l, {}).get("kernel_s", 0.0)
                                                                     for l in TASK_LABELS))
        o.write(f"\nΣ kernel time not launched from r1_main_commit or a fused-task range: {unattr:.2f} s "
                "(the main commits' own drivers, the head, rayon workers).\n")
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
    print(f"RUNA {wl}: stages from {src}; recommit ranges {nrec}; device recommits "
          f"{lg['result']['recommits'] if lg['result'] else '-'}")
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


def select(row, launches):
    """(matched, profiled): the launches the pass's regex matches, and the ones ncu would profile.
    window: ncu's default filter: skip `skip` matching launches, profile the next `count` (then
            --kill ends the run).
    config: --filter-mode per-launch-config, whose key is the launch's grid, block and shared
            memory: skip/count per key. A config pass names one kernel, so the key never mixes two."""
    rx = kernel_re(row)
    m = [r for r in launches if rx.fullmatch(r["kernel"])]
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


def cmd_dry(a):
    plan = [r for r in read_plan(a.plan) if r["workload"] == a.workload]
    launches = load_trace(a.sqlite)
    db = open_ro(a.sqlite)
    tbls = tables(db)
    t0 = session_start_ns(db, tbls)
    nvtx = load_nvtx(db, tbls)
    lg = read_log(a.log)
    end = max([r["k_start_ns"] + r["dur_us"] * 1e3 for r in launches] + [0])
    w, src = trace_windows(nvtx, lg, t0, end)
    segs = partition(w)
    sstarts = [s[0] for s in segs]
    rc_ivs = w.get("recommit", [])
    rc_starts = [s for s, _ in rc_ivs]
    for r in launches:
        r["stage"] = stage_at(segs, sstarts, r["api_ns"]) if segs else "whole"
        i = bisect.bisect_right(rc_starts, r["api_ns"]) - 1
        if r["stage"] == "fused" and i >= 0 and rc_ivs[i][0] <= r["api_ns"] < rc_ivs[i][1]:
            r["stage"] = "fused(recommit window)"
    wall = end / 1e9
    os.makedirs(a.out, exist_ok=True)
    tot = sum(r["dur_us"] for r in launches) or 1.0
    per_kernel = {}
    for r in launches:
        k = per_kernel.setdefault(r["kernel"], {"n": 0, "us": 0.0, "cfg": set(), "stages": set()})
        k["n"] += 1
        k["us"] += r["dur_us"]
        k["cfg"].add((r["grid"], r["block"], r["smem"]))
        k["stages"].add(r["stage"].split("(")[0])
    tag = a.tag or a.workload
    o = io.StringIO()
    o.write(f"# The pass plan against a trace: {tag}\n\n")
    o.write(f"trace: {len(launches)} kernel launches, {len(per_kernel)} kernels, Σ kernel time {tot / 1e6:.2f} s, "
            f"{wall:.1f} s of trace; stages from {src}. A launch whose call started inside a recommit range reads "
            "`fused(recommit window)` (other tables' launches in the same window read so too).\n\n")
    o.write("## per pass: what ncu would profile\n\n")
    o.write("| pass | mode | skip/count | matched launches | kernels matched | configs | would profile | "
            "stages of those | shapes of those | est. s | verdict |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    tsv = [["pass", "mode", "skip", "count", "matched", "kernels", "configs", "profile", "est_s", "verdict"]]
    detail = io.StringIO()
    rc = 0
    for row in plan:
        m, pa = select(row, launches)
        names = {}
        for r in m:
            names[r["kernel"]] = names.get(r["kernel"], 0) + 1
        cfgs = len({(r["kernel"], r["grid"], r["block"], r["smem"]) for r in m})
        st, shp = {}, {}
        for r in pa:
            st[r["stage"]] = st.get(r["stage"], 0) + 1
            s = shape(r["grid"], r["block"])
            shp[s] = shp.get(s, 0) + 1
        verdict = "ok" if pa else ("NO MATCH" if not m else "NOTHING SELECTED")
        if not pa:
            rc = 1
        est = est_seconds(row, pa, launches, wall)
        shapes_s = ", ".join(f"{k}{' x' + str(v) if v > 1 else ''}" for k, v in
                             sorted(shp.items(), key=lambda kv: -kv[1])[:4]) + (" …" if len(shp) > 4 else "")
        o.write(f"| {row['pass']} | {row['mode']} | {row['skip']}/{row['count']} | {len(m)} | {len(names)} | {cfgs} | "
                f"{len(pa)} | {', '.join(f'{k} {v}' for k, v in sorted(st.items()))} | {shapes_s} | {est:.0f} | "
                f"{verdict} |\n")
        tsv.append([row["pass"], row["mode"], row["skip"], row["count"], len(m), len(names), cfgs, len(pa),
                    f"{est:.0f}", verdict])
        detail.write(f"\n### {row['pass']} ({row['mode']}, kernels `{row['kernels']}`)\n\n")
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
    o.write("\nconfigs = distinct (kernel, grid, block, shared memory) among the matched. est. s = a guide for the "
            "counters box (see est_seconds).\n")
    o.write("\n## kernels by time, and which pass covers each\n\n| kernel | launches | configs | Σ s | % | stages | "
            "passes |\n|---|---|---|---|---|---|---|\n")
    uncovered = []
    for name, k in sorted(per_kernel.items(), key=lambda kv: -kv[1]["us"]):
        cov = [row["pass"] for row in plan if matches(row, name)]
        share = 100.0 * k["us"] / tot
        if share >= 1.0 and not cov:
            uncovered.append(f"{name} ({share:.1f} %)")
        if share >= 0.1 or cov:
            o.write(f"| {name} | {k['n']} | {len(k['cfg'])} | {k['us'] / 1e6:.3f} | {share:.1f} | "
                    f"{'/'.join(sorted(k['stages']))} | {', '.join(cov) or '-'} |\n")
    o.write(f"\nkernels with >= 1 % of kernel time that no pass covers: {', '.join(uncovered) or 'none'}\n")
    o.write(detail.getvalue())
    with open(os.path.join(a.out, f"dry-{tag}.md"), "w") as f:
        f.write(o.getvalue())
    write_tsv(os.path.join(a.out, f"dry-{tag}.tsv"), tsv)
    for t in tsv[1:]:
        print(f"DRY {tag} {t[0]}: matched {t[4]} launches ({t[5]} kernels, {t[6]} configs); would profile {t[7]}; "
              f"est {t[8]} s; {t[9]}")
    print(f"DRY {tag}: uncovered kernels >= 1 %: {', '.join(uncovered) or 'none'}")
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


def cmd_report(a):
    s = a.send
    o = io.StringIO()
    o.write("# noepoch_counters.sh: summary\n\n")
    env = os.path.join(s, "env.txt")
    if os.path.exists(env):
        keep = ("script=", "repo=", "noepoch_", "gpu_name=", "driver=", "ncu=", "nsys=", "cpu=")
        o.write("```\n" + "".join(l for l in open(env) if l.startswith(keep)) + "```\n\n")
    ref = read_tsv(os.path.join(s, "reference", "runs.tsv"))
    if ref:
        o.write("## reference runs (no profiler)\n\n| workload | rc | seconds | base line | VRAM max MiB |\n"
                "|---|---|---|---|---|\n")
        for r in ref:
            o.write(f"| {r['workload']} | {r['rc']} | {r['seconds']} | {r['base']} | {r['vram_max_mib']} |\n")
        o.write("\n")
    for wl in WORKLOADS:
        p = os.path.join(s, "runa", wl, f"runa-{wl}.md")
        if not os.path.exists(p):
            continue
        t = open(p, errors="replace").read()
        o.write(f"## run A, {wl}\n\n")
        o.write(t.split("\n", 2)[2] if t.count("\n") > 2 else t)
        o.write("\n")
    k = md_section(os.path.join(s, "summary", "kernels.md"), "## epoch base against no-epoch")
    if k:
        o.write("## run B: " + k.split(" ", 1)[1] + "\n")
    passes = read_tsv(os.path.join(s, "passes.tsv"))
    if passes:
        o.write("## run B passes\n\n| pass | workload | mode | profiled | seconds | verdict |\n|---|---|---|---|---|---|\n")
        for r in passes:
            o.write(f"| {r['pass']} | {r['workload']} | {r['mode']} | {r['profiled']} | {r['seconds']} | {r['verdict']} |\n")
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
    """A small nsys-shaped trace of one no-epoch prove, session start T0_SYNTH (trace seconds):
    head [0, 2), prepass [2, 2.5), main_commit [2.5, 5), absorb [5, 5.2), fused [5.2, 9) with one
    recommit range [5.2, 6) and a rounds_2to4_table range [6.5, 8.5) on a driver thread, tail [9, 10)."""
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
    names = ["ntt_cm_dit_k8", "rpx_merkle_level", "rpx_leaves_base_row_pair_batched", "ccomp_0b8d15837e1e77a3",
             "cuLaunchKernel", "cuStreamSynchronize", "cuMemAlloc_v2", "cuMemcpyDtoHAsync_v2", "cuMemHostAlloc",
             "driver-3", "r1_main_commit"]
    for i, n in enumerate(names, 1):
        db.execute("INSERT INTO StringIds VALUES (?, ?)", (i, n))
    db.execute("INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (?, 'x', 'x')", (T0_SYNTH,))
    db.execute("INSERT INTO ThreadNames VALUES (10, 0, ?)", (DRV_TID,))
    sec = 10 ** 9

    def rng(a, b, text, tid, text_id=None):
        db.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, 59, 0, 0, 0, ?, ?, ?, ?, 0)",
                   (int(a * sec), int(b * sec), text, tid, tid, text_id))

    rng(2.0, 2.5, "r1_prepass", MAIN_TID)
    rng(2.5, 5.0, None, MAIN_TID, 11)        # a registered string: the text through StringIds
    rng(5.2, 9.0, "rounds_2to4", MAIN_TID)
    rng(5.2, 6.0, "r1_main_recommit_table", DRV_TID)
    rng(6.5, 8.5, ":rounds_2to4_table", DRV_TID)
    # (kernel, grid, t launch, duration ms, thread): the launch call starts 1 µs before the kernel
    launches = [("ntt_cm_dit_k8", 1024, 3.0, 100.0, MAIN_TID), ("rpx_leaves_base_row_pair_batched", 4096, 3.5, 200.0,
                                                                MAIN_TID),
                ("ntt_cm_dit_k8", 2048, 5.3, 50.0, DRV_TID), ("rpx_leaves_base_row_pair_batched", 8192, 5.5, 300.0,
                                                              DRV_TID),
                ("ccomp_0b8d15837e1e77a3", 512, 7.0, 400.0, DRV_TID), ("rpx_merkle_level", 256, 9.5, 10.0, MAIN_TID)]
    for c, (k, gx, ts, dms, tid) in enumerate(launches, 1):
        s = int(ts * sec)
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 0, ?, ?, 5, 0, 0)", (s - 1000, s, tid, c))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 0, ?, ?, ?, 1, 1, 128, 1, 1, 0, 0, 40)",
                   (s, s + int(dms * 1e6), c, names.index(k) + 1, gx))
    # API calls without kernels: a 0.3 s sync in main_commit, a 0.2 s alloc and a 0.1 s pageable copy in fused
    for c, (nm, ts, dur, tid) in enumerate([("cuStreamSynchronize", 4.0, 0.3, MAIN_TID),
                                            ("cuMemAlloc_v2", 5.25, 0.2, DRV_TID),
                                            ("cuMemcpyDtoHAsync_v2", 7.5, 0.1, DRV_TID),
                                            ("cuMemHostAlloc", 1.0, 0.05, MAIN_TID)], 100):
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 0, ?, ?, ?, 0, 0)",
                   (int(ts * sec), int((ts + dur) * sec), tid, c, names.index(nm) + 1))
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, 1, 1000)", (1 * sec, int(1.5 * sec)))
    db.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (7, 0, 'syn', 1, 'SMs Active [Throughput %]')")
    db.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (7, 0, 'syn', 2, 'Compute Warps in Flight [Throughput %]')")
    for tenth in range(0, 100):               # 10 Hz: SM active 80 in main_commit, 30 in fused, 0 elsewhere
        ts = tenth * 10 ** 8
        v = 80 if 25 <= tenth < 50 else (30 if 52 <= tenth < 90 else 0)
        db.execute("INSERT INTO GPU_METRICS VALUES (?, ?, 7, 1, ?)", (ts, ts, v))
        db.execute("INSERT INTO GPU_METRICS VALUES (?, ?, 7, 2, ?)", (ts, ts, v // 2))
    db.commit()
    db.close()


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
    with tempfile.TemporaryDirectory() as d:
        plan = os.path.join(d, "plan.tsv")
        with open(plan, "w") as f:
            f.write("\t".join(PLAN_COLS) + "\n")
            f.write("noepoch_rowpair\tnoepoch\tconfig\t0\t1\trpx_leaves_base_row_pair_batched\tleaves\tone per shape\n")
            f.write("noepoch_ntt\tnoepoch\twindow\t1\t1\tntt_cm_di[ft]_k[4-8]\tlde\tncu window\n")
            f.write("noepoch_quot\tnoepoch\twindow\t0\t8\tccomp_[0-9a-f]+|constraint_composition_kernel\tquotient\tq\n")
            f.write("noepoch_none\tnoepoch\twindow\t0\t1\tno_such_kernel\tnone\tmust report NO MATCH\n")
            f.write("epoch_ntt\tepoch\twindow\t0\t4\tntt_cm_di[ft]_k[4-8]\tlde\tother workload\n")
        ok(check_plan(read_plan(plan)) == [], f"the synthetic plan is well formed ({check_plan(read_plan(plan))})")
        badplan = os.path.join(d, "bad.tsv")
        with open(badplan, "w") as f:
            f.write("\t".join(PLAN_COLS) + "\n")
            f.write("x\tepoch\twindow\t0\t0\t^rpx$\tf\tn\n")
            f.write("x\tzisk\tsome\t-1\t1\trpx(\tf\tn\n")
            f.write("y\tnoepoch\tconfig\t0\t1\tfri_fold_ext3|logup_[a-z0-9_]+\tf\tn\n")
        errs = check_plan(read_plan(badplan))
        ok(len(errs) >= 7 and any("exactly one kernel" in e for e in errs),
           f"a malformed plan is refused, a two-kernel config pass included ({len(errs)} errors)")
        # summary, in both unit styles, with an epoch export beside the no-epoch one
        for base_units in (False, True):
            nd = os.path.join(d, f"ncu{int(base_units)}")
            os.makedirs(nd)
            for p in ("noepoch_ntt", "epoch_ntt"):
                with open(os.path.join(nd, f"{p}.details.csv"), "w") as f:
                    f.write(synth_details_csv(base_units=base_units))
                with open(os.path.join(nd, f"{p}.stages.tsv"), "w") as f:
                    f.write("id\tkernel\tstage\tpasses\n0\trpx_leaves_base_row_pair_batched\tfused\t18\n"
                            "1\tntt_cm_dit_k8\tmain_commit\t18\n2\tntt_cm_dit_k8\tfused\t18\n")
            out = os.path.join(d, f"sum{int(base_units)}")
            rc, _ = run(cmd_summary, argparse.Namespace(plan=plan, ncu_dir=nd, out=out))
            tag = "base units" if base_units else "auto units"
            ok(rc == 0, f"summary exits 0 ({tag})")
            rows = [r for r in read_tsv(os.path.join(out, "launches.tsv")) if r["pass"] == "noepoch_ntt"]
            ok(len(rows) == 3, f"three launches ({len(rows)}, {tag})")
            r0 = rows[0]
            ok(r0["kernel"] == "rpx_leaves_base_row_pair_batched" and r0["stage"] == "fused" and r0["elem"] == "leaf",
               f"launch 0 fields ({tag})")
            ok(abs(fnum(r0["dur_us"]) - 62360.0) < 0.5 and abs(fnum(r0["sm_ghz"]) - 2.01) < 1e-6,
               f"duration and SM clock ({r0['dur_us']}, {r0['sm_ghz']}, {tag})")
            ok(r0["family"] == "leaves" and r0["workload"] == "noepoch" and r0["bound"] == "compute",
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
            ok(set(kn) == {(w, k) for w in ("epoch", "noepoch") for k in ("rpx_leaves_base_row_pair_batched",
                                                                           "ntt_cm_dit_k8")}, "one row per workload and kernel")
            ntt = kn[("noepoch", "ntt_cm_dit_k8")]
            wdram = (84.0 * 140.5 + 70.0 * 59.5) / 200.0
            ok(ntt["launches"] == "2" and ntt["configs"] == "2" and abs(fnum(ntt["dram_pct"]) - wdram) < 0.01,
               f"duration-weighted DRAM % ({ntt['dram_pct']} vs {wdram:.2f})")
            ok(ntt["largest_shape"] == "2048x16x1/256x1x1" and ntt["stages"] == "fused/main_commit",
               "largest launch, stages")
            md = open(os.path.join(out, "kernels.md")).read()
            ok("## epoch base against no-epoch, per kernel" in md and "| ntt_cm_dit_k8 | lde | epoch |" in md
               and "| ntt_cm_dit_k8 | lde | noepoch |" in md and "## noepoch: one row per kernel" in md,
               "kernels.md: the comparison and the per-workload tables")
            ks = read_tsv(os.path.join(out, "kernels_by_stage.tsv"))
            ok(len(ks) == 6 and any(r["stage"] == "fused" and r["kernel"] == "ntt_cm_dit_k8" for r in ks),
               f"kernels by stage ({len(ks)})")
        # stages from a pass's output order
        log = ['==PROF== Connected to process 1 (<W>/bin)', '==PROF== Profiling "ntt_cm_dit_k8": 0%....100% - 18 passes',
               '[prover] table walk R1 (walk weight, largest first): CPU[0] 3.1', '==PROF== Profiling "ntt_cm_dit_k8" - 1: 0%....100% - 17 passes',
               '[prover] table walk rounds 2-4 (walk weight, largest first): x', '==PROF== Profiling "rpx_merkle_level": 0%....100% - 17 passes',
               'PROVE SPLIT #0: airs 134 · rows 1 · wall 30.0s', '==PROF== Profiling "fri_fold_ext3": 0%....100% - 9 passes']
        st = stages_from_lines(log)
        ok([s[2] for s in st] == ["head", "main_commit", "fused", "between"] and st[1][3] == "17" and st[3][0] == 3,
           f"stages from the output order ({st})")
        # runa on the synthetic trace
        db = os.path.join(d, "t.sqlite")
        build_synth_sqlite(db)
        rlog = os.path.join(d, "a.log")
        with open(rlog, "w") as f:
            f.write("[prover] VRAM gate: packing admission (LAMBDA_VM_GATE_PACKING=1)\n"
                    f"PROVE SPLIT #0: airs 134 · rows 99 · wall 7.00s · t=[{T0_SYNTH / 1e9 + 2:.3f},{T0_SYNTH / 1e9 + 9:.3f}]"
                    " · prepass 0.50 · main_commit 2.50 · absorb 0.200 · fused 3.80 · other 0.00 || tables[Σ] aux_build 1"
                    " || recommit[Σ] 0.80\n"
                    "TABLE TL fused idx=3 CPU[0] est=10.50GiB claim=1.000 start=1.100 end=2.000\n"
                    "NOEPOCH RESULT: verified=true · sub-proofs 134 · base 9.00 s (execute 1.00 · build 1.00 · setup "
                    "0.00 · prove 7.00) · verify 1.00 s · proof 73400000 B · host peak 43.90 GiB · device recommits 1\n"
                    "test result: ok. 1 passed; 0 failed\n")
        smi = os.path.join(d, "smi.csv")
        with open(smi, "w") as f:
            for tenth in range(0, 100, 2):
                ts = datetime.fromtimestamp(T0_SYNTH / 1e9 + tenth / 10).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
                f.write(f"{ts}, {30000 if 52 <= tenth < 90 else 1000}, 50\n")
        rout = os.path.join(d, "runa")
        rc, printed = run(cmd_runa, argparse.Namespace(sqlite=db, log=rlog, workload="noepoch", out=rout, smi=smi,
                                                       bin_ms=500))
        ok(rc == 0 and "| fused | 3.80 |" in printed and "Stages from NVTX ranges" in printed,
           f"runa exits 0 with a 3.8 s fused stage ({printed[:600]!r})")
        srows = {r["stage"]: r for r in read_tsv(os.path.join(rout, "runa-noepoch-stages.tsv"))}
        ok(set(srows) == {"head", "prepass", "main_commit", "between", "fused", "tail", "recommit", "whole"},
           f"runa's stages ({sorted(srows)})")
        ok(abs(fnum(srows["main_commit"]["wall_s"]) - 2.5) < 1e-6 and abs(fnum(srows["between"]["wall_s"]) - 0.2) < 1e-6
           and abs(fnum(srows["recommit"]["wall_s"]) - 0.8) < 1e-6 and abs(fnum(srows["head"]["wall_s"]) - 2.0) < 1e-6,
           "runa's stage walls")
        ok(abs(fnum(srows["main_commit"]["kernel_sum_s"]) - 0.3) < 1e-6 and srows["fused"]["launches"] == "3",
           f"runa's kernels per stage ({srows['main_commit']['kernel_sum_s']}, {srows['fused']['launches']})")
        ok(abs(fnum(srows["main_commit"]["sm_active"]) - 80.0) < 1e-6 and abs(fnum(srows["fused"]["sm_active"]) - 30.0) < 1e-6
           and abs(fnum(srows["fused"]["warps_in_flight"]) - 15.0) < 1e-6, "runa's GPU metric means per stage")
        ok(abs(fnum(srows["fused"]["vram_max_mib"]) - 30000) < 1e-6 and abs(fnum(srows["head"]["vram_max_mib"]) - 1000) < 1e-6,
           f"runa's VRAM per stage ({srows['fused']['vram_max_mib']})")
        ok(abs(fnum(srows["fused"]["tasks_open"]) - 2.8 / 3.8) < 0.01 and abs(fnum(srows["recommit"]["recommits_open"]) - 1.0) < 1e-6,
           f"runa's open fused tasks ({srows['fused']['tasks_open']})")
        att = {r["label"]: r for r in read_tsv(os.path.join(rout, "runa-noepoch-nvtx-kernels.tsv"))}
        ok(att.get("r1_main_recommit_table", {}).get("launches") == "2"
           and abs(fnum(att["r1_main_recommit_table"]["kernel_s"]) - 0.35) < 1e-6
           and att.get("rounds_2to4_table", {}).get("launches") == "1" and att.get("r1_main_commit", {}).get("launches") == "2",
           f"kernels by launching NVTX label ({att})")
        cat = {(r["category"], r["stage"]): r for r in read_tsv(os.path.join(rout, "api-noepoch-by-category.tsv"))}
        ok(abs(fnum(cat[("sync", "main_commit")]["sum_s"]) - 0.3) < 1e-6 and abs(fnum(cat[("alloc", "fused")]["sum_s"]) - 0.2) < 1e-6
           and abs(fnum(cat[("copy_async", "fused")]["sum_s"]) - 0.1) < 1e-6 and ("host_pinned", "head") in cat
           and cat[("launch", "fused")]["calls"] == "3", f"API seconds per category and stage ({sorted(cat)})")
        thr = read_tsv(os.path.join(rout, "api-noepoch-by-thread.tsv"))
        ok(any(r["thread"] == "tid 101 (driver-3)" and r["category"] == "alloc" for r in thr), "API per named thread")
        ts = read_tsv(os.path.join(rout, "runa-noepoch-timeseries.tsv"))
        ok(len(ts) == 20 and ts[6]["stage"] == "main_commit" and ts[12]["stage"] == "fused", f"the time series ({len(ts)})")
        ok("## the fused stage, second by second" in printed and "RUNA noepoch: stages from NVTX ranges; recommit ranges 1;"
           " device recommits 1" in printed, "runa's fused table and its closing line")
        # the log fallback: no NVTX table
        import sqlite3
        db2 = os.path.join(d, "t2.sqlite")
        build_synth_sqlite(db2)
        c2 = sqlite3.connect(db2)
        c2.execute("DROP TABLE NVTX_EVENTS")
        c2.commit()
        c2.close()
        rc, printed = run(cmd_runa, argparse.Namespace(sqlite=db2, log=rlog, workload="noepoch", out=os.path.join(d, "r2"),
                                                       smi=None, bin_ms=1000))
        s2 = {r["stage"]: r for r in read_tsv(os.path.join(d, "r2", "runa-noepoch-stages.tsv"))}
        ok(rc == 0 and "PROVE SPLIT lines" in printed and "recommit" not in s2
           and abs(fnum(s2["fused"]["wall_s"]) - 3.8) < 1e-3 and abs(fnum(s2["main_commit"]["wall_s"]) - 2.5) < 1e-3,
           f"runa's log fallback ({sorted(s2)})")
        # dry against the synthetic trace
        dout = os.path.join(d, "dry")
        rc, printed = run(cmd_dry, argparse.Namespace(plan=plan, workload="noepoch", sqlite=db, log=rlog, out=dout,
                                                      tag=None))
        ok(rc == 1 and "noepoch_none: matched 0 launches" in printed and "NO MATCH" in printed,
           "dry exits 1 on a pass that matches nothing")
        drows = {r["pass"]: r for r in read_tsv(os.path.join(dout, "dry-noepoch.tsv"))}
        ok(set(drows) == {"noepoch_rowpair", "noepoch_ntt", "noepoch_quot", "noepoch_none"},
           "dry evaluates this workload's passes only")
        ok(drows["noepoch_rowpair"]["profile"] == "2" and drows["noepoch_ntt"]["profile"] == "1"
           and drows["noepoch_quot"]["matched"] == "1", f"dry selections ({drows})")
        dmd = open(os.path.join(dout, "dry-noepoch.md")).read()
        ok("fused(recommit window) 1" in dmd and "main_commit 1" in dmd, "dry stages, the recommit window included")
        # report
        send = os.path.join(d, "send")
        os.makedirs(os.path.join(send, "runa", "noepoch"))
        os.makedirs(os.path.join(send, "summary"))
        os.makedirs(os.path.join(send, "reference"))
        for fn in os.listdir(rout):
            with open(os.path.join(rout, fn)) as src, open(os.path.join(send, "runa", "noepoch", fn), "w") as dst:
                dst.write(src.read())
        with open(os.path.join(send, "summary", "kernels.md"), "w") as f:
            f.write(open(os.path.join(d, "sum0", "kernels.md")).read())
        with open(os.path.join(send, "reference", "runs.tsv"), "w") as f:
            f.write("workload\trc\tseconds\tbase\tvram_max_mib\nnoepoch\t0\t60\tno-epoch base 38.00 s\t32110\n")
        rc, _ = run(cmd_report, argparse.Namespace(send=send))
        rep = open(os.path.join(send, "SUMMARY.md")).read()
        ok(rc == 0 and "## run A, noepoch" in rep and "## run B: epoch base against no-epoch" in rep
           and "## reference runs" in rep and "| fused | 3.80 |" in rep, "report assembles SUMMARY.md")
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
    s = sub.add_parser("runa")
    s.add_argument("--sqlite", required=True)
    s.add_argument("--log", required=True)
    s.add_argument("--workload", required=True, choices=WORKLOADS)
    s.add_argument("--out", required=True)
    s.add_argument("--smi", help="nvidia-smi samples: timestamp, memory.used, utilization.gpu (csv, noheader)")
    s.add_argument("--bin-ms", type=float, default=100.0, help="the time series' bin (ms, >= 1)")
    s = sub.add_parser("dry")
    s.add_argument("--plan", required=True)
    s.add_argument("--workload", required=True, choices=WORKLOADS)
    s.add_argument("--sqlite", required=True)
    s.add_argument("--log", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--tag", help="names the outputs dry-<tag>.md/.tsv (default: the workload)")
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

BUILD_ENV=() RUN_ENV=() WL_ENV=() WL_TEST="" ARCH_ENV=() NVTX_LIB=""
build_env() {
  BUILD_ENV=("PATH=$PATH" "HOME=$W/home" "RUSTUP_HOME=$RUSTUP_HOME_REAL" "RUSTUP_AUTO_INSTALL=0"
    "CARGO_HOME=$W/cargo-home" "TMPDIR=$W/tmp" "LC_ALL=C" "LANG=C" "RUSTC_WRAPPER=" "RUSTC_WORKSPACE_WRAPPER="
    "CARGO_TERM_COLOR=never" "CARGO_TERM_PROGRESS_WHEN=never" "LAMBDA_VM_NVCC_LINEINFO=1")
  local v
  for v in CUDA_HOME CUDA_PATH LD_LIBRARY_PATH; do
    if [ -n "${!v:-}" ]; then BUILD_ENV+=("$v=${!v}"); fi
  done
}
run_env() { # run_env [nvtx]: the profiled process's system variables (+ the NVTX library)
  RUN_ENV=("PATH=$PATH" "HOME=$W/home" "TMPDIR=$W/tmp" "LC_ALL=C" "LANG=C"
    "CUDA_DEVICE_ORDER=PCI_BUS_ID" "CUDA_VISIBLE_DEVICES=$NP_GPU" "CARGO_MANIFEST_DIR=$REPO/prover")
  local v
  for v in CUDA_HOME CUDA_PATH LD_LIBRARY_PATH; do
    if [ -n "${!v:-}" ]; then RUN_ENV+=("$v=${!v}"); fi
  done
  if [ "${1:-}" = nvtx ] && [ -n "$NVTX_LIB" ]; then RUN_ENV+=("LAMBDA_VM_NVTX_LIB=$NVTX_LIB"); fi
}
wl_env() { # wl_env epoch|noepoch record|runb: the workload's knobs, as i-noepoch's FAST arms run it or as run B runs it
  local budget="$RUN_VRAM_MB"
  if [ "$2" = runb ]; then budget="$NP_NCU_VRAM_MB"; fi
  WL_ENV=("NOEPOCH_ELF=$W/fixtures/ethrex_8f826601.elf" "NOEPOCH_INPUT=$W/fixtures/ethrex_mainnet_25368371_573004e6.bin"
    "TABLE_PARALLELISM=4" "LAMBDA_VM_VRAM_BUDGET_MB=$budget" "LAMBDA_VM_MAX_ROWS_LOG2=21" "LFM_PROVE_SPLIT=1"
    "LAMBDA_VM_BASE_SPLIT=1" "LFM_EXEC_PARALLEL=1" "LFM_PRECOMPUTED_TREE_CACHE_CAP=64"
    "_RJEM_MALLOC_CONF=dirty_decay_ms:-1,muzzy_decay_ms:-1" "LAMBDA_VM_TABLE_TIMELINE=1")
  case "$1" in
    epoch) WL_TEST="$EPOCH_TEST" ;;
    noepoch) WL_ENV+=("LAMBDA_VM_GATE_PACKING=1"); WL_TEST="$NOEPOCH_TEST" ;;
    *) die 2 "unknown workload '$1' in the plan" ;;
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
NGPUS=0 WANT_NSYS=0 WANT_METRICS=0 WANT_NCU=0
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
        chk WARN "gpu: $GPU_VRAM MiB, less than the record's 32 GB RTX 5090: the no-epoch run filled 32 GB on FAST, so it may not fit, and what runs on the card differs from the record"
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
  if [ -n "$NVTX_LIB" ]; then chk PASS "nvtx: $NVTX_LIB (handed to the runs as LAMBDA_VM_NVTX_LIB: the prover's phases and the recommit are named ranges)"
  elif [ "$NP_FETCH_NVTX" = 1 ] && NVTX_LIB="$(fetch_nvtx_lib)"; then
    chk PASS "nvtx: $NVTX_LIB, fetched (nvidia-nvtx-cu12 12.8.90, sha256-checked) and handed to the runs as LAMBDA_VM_NVTX_LIB"
  else
    NVTX_LIB=""
    chk WARN "nvtx: no libnvToolsExt found (CUDA 12.9 and later ship none)$([ "$NP_FETCH_NVTX" = 1 ] && echo " and the fetch failed (see $W/nvtx/curl.log)"), so the NVTX ranges are silent no-ops: run A's stages come from the harness log and it has no recommit row. If you have one: export LAMBDA_VM_NVTX_LIB=/path/to/libnvToolsExt.so.1"
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
  elif ! num_ge "$eff" 48; then chk FAIL "ram: $eff GiB usable (MemTotal ${mt:-?}, cgroup limit ${cg:-none}) < 48 GiB: the no-epoch run peaks at 43.9 GiB on the host"
  elif ! num_ge "$av" "$NP_MIN_AVAIL_GIB"; then chk FAIL "ram: $av GiB available now (MemTotal $mt GiB), < $NP_MIN_AVAIL_GIB: the no-epoch run peaks at 43.9 GiB on the host; close big programs (browsers, IDEs, VMs) and run again"
  elif ! num_ge "$av" 52; then chk WARN "ram: $av GiB available (MemTotal $mt GiB): enough for the no-epoch run's 43.9 GiB peak, with little margin for ncu's replay copies; keep other programs closed during the run"
  else chk PASS "ram: $av GiB available (MemTotal $mt GiB, cgroup limit ${cg:-none}); need >= $NP_MIN_AVAIL_GIB"; fi
  fr="$(free_gib "$W")"
  if [ -z "$fr" ]; then chk WARN "disk: cannot read the free space in $W"
  elif ! num_ge "$fr" 30; then chk FAIL "disk: $fr GiB free in $W; need >= 40 (clone, cargo cache, a release CUDA build, two traces with GPU metrics, the reports)"
  elif ! num_ge "$fr" 40; then chk WARN "disk: $fr GiB free in $W; 40 recommended"
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
  n="$(env | awk -F= '/^(LAMBDA_VM_|LFM_|ZF_|NOEPOCH_|A_BUNDLE|A_CACHE|TABLE_PARALLELISM=|RAYON_|_RJEM_|MALLOC_CONF=|RUSTFLAGS=|CARGO_BUILD_|CARGO_ENCODED_RUSTFLAGS=|CARGO_TARGET_DIR=|RUSTC_WRAPPER=)/ { printf "%s ", $1 }')"
  if [ -n "$n" ]; then chk INFO "env: set in this shell and NOT passed to the build or the runs (they get an explicit environment; an NVTX library is handed over by path): $n"
  else chk PASS "env: no prover, cargo or allocator knob set in this shell"; fi

  if [ "$PF_FAILS" -eq 0 ]; then chk INFO "PREFLIGHT: PASS ($PF_WARNS warning(s))"
  else chk INFO "PREFLIGHT: FAIL ($PF_FAILS failure(s), $PF_WARNS warning(s)): fix the FAIL lines above and run again"; fi
}

build_probe() { # the one-kernel CUDA probe (two kernels and two shapes, for the config passes' flags)
  local cc="${GPU_CC//./}"
  cat > "$W/probe/ncuprobe.cu" <<'CU'
#include <cstdio>
extern "C" __global__ void ncuprobe_a(float *x) { x[threadIdx.x] += 1.0f; }
extern "C" __global__ void ncuprobe_b(float *x) { x[threadIdx.x] += 2.0f; }
int main() {
  float *d = nullptr;
  if (cudaMalloc(&d, 1024 * sizeof(float)) != cudaSuccess) { std::printf("ncuprobe: cudaMalloc failed\n"); return 2; }
  cudaMemset(d, 0, 1024 * sizeof(float));
  ncuprobe_a<<<1, 64>>>(d);
  ncuprobe_b<<<1, 64>>>(d);
  ncuprobe_a<<<2, 64>>>(d);
  ncuprobe_a<<<1, 64>>>(d);
  cudaError_t e = cudaDeviceSynchronize();
  std::printf("ncuprobe: done (%s)\n", cudaGetErrorString(e));
  return e == cudaSuccess ? 0 : 3;
}
CU
  build_env
  env -i "${BUILD_ENV[@]}" timeout 300 "$NVCC" -O2 -arch="sm_$cc" -o "$W/probe/ncuprobe" "$W/probe/ncuprobe.cu" \
    > "$W/probe/build.log" 2>&1
}

ncu_probe() { # ncu_probe LABEL NCU-ARGS...: profile the probe; prints unlocked | locked | unknown (rc N); log in probe/LABEL.log
  local label="$1" rc=0 out
  shift
  run_env
  env -i "${RUN_ENV[@]}" timeout 300 "$NCU_BIN" "$@" "$W/probe/ncuprobe" > "$W/probe/$label.log" 2>&1 || rc=$?
  out="$(cat "$W/probe/$label.log")"
  case "$out" in
    *ERR_NVGPUCTRPERM*) echo "locked" ;;
    *'==PROF== Profiling "ncuprobe_'*) if [ "$rc" -eq 0 ]; then echo "unlocked"; else echo "unknown (ncu rc=$rc after profiling)"; fi ;;
    *) echo "unknown (ncu rc=$rc, no probe kernel profiled)" ;;
  esac
}

probe_counters() {
  local res fix m ok="" rc all n dropped=""
  fix="Fix: run this script as root (sudo -E bash noepoch_counters.sh), or open the counters to all users: as root, echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' > /etc/modprobe.d/nvidia-profiling.conf, then update-initramfs -u (dracut -f on Fedora/RHEL) and reboot"
  if [ -z "$NCU_BIN" ]; then
    if [ "$WANT_NCU" = 1 ]; then chk FAIL "counters: not probed (no ncu)"; fi
    return 0
  fi
  if [ "$WANT_NCU" = 0 ]; then
    # run B's full flag sets (every metric, unvalidated), on the probe: ncu parses every flag before it
    # touches the counters, so on a locked box `locked` still says this ncu accepts the passes' command lines
    NCU_METRICS_OK="$NP_METRICS"
    ncu_args window 0 1 'ncuprobe_a' "$W/probe/flags_window"
    res="$(ncu_probe flags_window "${NCU_ARGS[@]}")"
    chk INFO "ncu flags, window passes: $res (a box with closed counters: locked = flags accepted, then counters refused)"
    ncu_args config 0 1 'ncuprobe_a' "$W/probe/flags_config"
    res="$(ncu_probe flags_config "${NCU_ARGS[@]}")"
    chk INFO "ncu flags, config passes: $res (as above)"
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
  run_env
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
  # the passes' exact flags, on the probe
  ncu_args window 0 1 'ncuprobe_a' "$W/probe/window"
  res="$(ncu_probe window "${NCU_ARGS[@]}")"
  if [ "$res" = unlocked ] && [ -s "$W/probe/window.ncu-rep" ]; then chk PASS "ncu: the window passes' exact flags (--clock-control $NP_CLOCK) profile the probe kernel"
  else chk FAIL "ncu: the window passes' flags failed on the probe kernel ($res): see $W/probe/window.log (NP_CLOCK=base works on every ncu since 2025.1)"; fi
  ncu_args config 0 1 'ncuprobe_a' "$W/probe/config"
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

probe_nsys() { # run A's exact nsys flags on the probe; with GPU metrics, the samples must reach the export
  local rc=0 n
  if [ "$WANT_NSYS" = 0 ] || [ -z "$NSYS_BIN" ]; then return 0; fi
  if [ "$WANT_METRICS" = 1 ] && [ -z "$NSYS_METRICS_FLAG" ]; then return 0; fi
  nsys_args "$W/probe/nsysprobe" "$WANT_METRICS"
  run_env
  env -i "${RUN_ENV[@]}" timeout 300 "$NSYS_BIN" "${NSYS_ARGS[@]}" "$W/probe/ncuprobe" > "$W/probe/nsys.log" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ] || [ ! -s "$W/probe/nsysprobe.nsys-rep" ]; then
    if grep -q 'ERR_NVGPUCTRPERM\|insufficient privilege\|permission' "$W/probe/nsys.log" 2>/dev/null && [ "$WANT_METRICS" = 1 ]; then
      chk FAIL "nsys: GPU metrics need the same counter access as ncu (see the counters line above): see $W/probe/nsys.log"
    else chk FAIL "nsys: run A's flags failed on the probe (rc=$rc): see $W/probe/nsys.log"; fi
    return 0
  fi
  if [ "$WANT_METRICS" = 0 ]; then chk PASS "nsys: run A's flags trace the probe (no GPU metrics in this mode)"; return 0; fi
  rc=0
  timeout 300 "$NSYS_BIN" export --type sqlite --force-overwrite true --output "$W/probe/nsysprobe.sqlite" \
    "$W/probe/nsysprobe.nsys-rep" >> "$W/probe/nsys.log" 2>&1 || rc=$?
  n="$(python3 - "$W/probe/nsysprobe.sqlite" <<'PY' 2>>"$W/probe/nsys.log" || true
import sqlite3, sys
try:
    print(sqlite3.connect(sys.argv[1]).execute("SELECT count(*) FROM GPU_METRICS").fetchone()[0])
except sqlite3.Error:
    print(0)
PY
)"
  if [ "$rc" -eq 0 ] && [ "${n:-0}" -gt 0 ]; then chk PASS "nsys: run A's flags trace the probe with GPU metrics ($n samples at $NP_GPU_METRICS_HZ Hz)"
  else chk FAIL "nsys: GPU metrics requested, ${n:-0} samples in the probe's export (rc=$rc): see $W/probe/nsys.log"; fi
}

ncu_args() { # ncu_args window|config SKIP COUNT KERNEL-REGEX-BODY REPORT-BASE -> NCU_ARGS
  local s
  NCU_ARGS=(--target-processes all --kernel-name "regex:^(${4})\$" --launch-skip "$2" --launch-count "$3")
  if [ "$1" = config ]; then NCU_ARGS+=(--filter-mode per-launch-config)
  elif [ "$NCU_HAS_KILL" = 1 ]; then NCU_ARGS+=(--kill yes); fi
  for s in $NP_SECTIONS; do NCU_ARGS+=(--section "$s"); done
  if [ -n "$NCU_METRICS_OK" ]; then NCU_ARGS+=(--metrics "$NCU_METRICS_OK"); fi
  NCU_ARGS+=(--clock-control "$NP_CLOCK" --export "$5" --force-overwrite)
}

nsys_args() { # nsys_args REPORT-BASE WITH_METRICS(0|1) -> NSYS_ARGS: run A's flags
  NSYS_ARGS=(profile "--trace=cuda,nvtx" --sample=none --cpuctxsw=none)
  if [ "$2" = 1 ]; then NSYS_ARGS+=("$NSYS_METRICS_FLAG=all" "--gpu-metrics-frequency=$NP_GPU_METRICS_HZ"); fi
  NSYS_ARGS+=(--stats=false --force-overwrite=true "--output=$1")
}

estimate() { # a guide, printed before the long steps
  local b r=0 a=0 p=0 pass wl mode skip count nw
  if [ -d "$REPO/target/release/deps" ]; then b=3; else b=20; fi
  # shellcheck disable=SC2086 # the workload list, split on blanks
  nw="$(printf '%s\n' $NP_WORKLOADS | awk 'NF { n++ } END { print n + 0 }')"
  if [ "$NP_REFERENCE" = 1 ]; then r=$((nw * 70)); fi
  if [ "$NP_RUN_A" = 1 ]; then a=$((nw * (90 + 240))); fi # a traced run plus its export and tables
  if [ "$NP_RUN_B" = 1 ] && [ "$NP_DRY" = 0 ]; then
    while IFS=$'\t' read -r pass wl mode skip count _; do
      if [ "$pass" = pass ] || ! selected "$pass" "$wl"; then continue; fi
      if [ "$mode" = window ]; then p=$((p + 40 + 3 * count + skip / 100))
      elif [ "$wl" = noepoch ]; then p=$((p + 130)); else p=$((p + 90)); fi
    done < "$PLAN_FILE"
  fi
  log "ESTIMATE (a guide, not a bound; every step has its own timeout): build ~$b min (a fresh work directory: 5-30), reference ~$((r / 60)) min, run A ~$((a / 60)) min, run B ~$((p / 60)) min, total ~$((b + (r + a + p) / 60 + 3)) min"
}

selected() { # selected PASS WORKLOAD: in NP_PASSES (or NP_PASSES empty), and the workload in NP_WORKLOADS
  case " $NP_WORKLOADS " in *" $2 "*) ;; *) return 1 ;; esac
  if [ -z "$NP_PASSES" ]; then return 0; fi
  case " $NP_PASSES " in *" $1 "*) return 0 ;; esac
  return 1
}

# ---------------------------------------------------------------------------------------------------
# the checkout, the fixtures, the build

fetch_repo() {
  local mod
  step_begin "clone $NP_REPO_URL and check out $PIN_SHA"
  if [ -e "$REPO/.git" ]; then
    if ! git -C "$REPO" cat-file -e "$PIN_SHA^{commit}" 2>/dev/null; then
      git -C "$REPO" fetch --quiet origin > "$KEEP/git-fetch.log" 2>&1 || die 5 "git fetch failed in $REPO: see $KEEP/git-fetch.log"
    fi
    mod="$(git -C "$REPO" status --porcelain --untracked-files=no | awk '{ print $2 }')"
    if [ -n "$mod" ]; then
      if printf '%s\n' "$mod" | awk '!/(^|\/)Cargo[.]lock$/ { bad = 1 } END { exit bad }'; then
        log "restoring $(printf '%s\n' "$mod" | wc -l | tr -d ' ') lockfile(s) a build rewrote"
        # shellcheck disable=SC2086 # file names without spaces, from git
        git -C "$REPO" checkout --quiet -- $mod
      else
        die 9 "the clone $REPO has modified tracked files ($(printf '%s' "$mod" | tr '\n' ' ')): this directory is the script's; delete $REPO (or move it away) and run again"
      fi
    fi
  else
    git clone --quiet --no-checkout "$NP_REPO_URL" "$REPO" > "$KEEP/git-clone.log" 2>&1 || die 5 "git clone $NP_REPO_URL failed: see $KEEP/git-clone.log"
  fi
  if ! git -C "$REPO" cat-file -e "$PIN_SHA^{commit}" 2>/dev/null; then
    git -C "$REPO" fetch --quiet origin "$PIN_SHA" >> "$KEEP/git-fetch.log" 2>&1 \
      || git -C "$REPO" fetch --quiet origin profile/noepoch-counters >> "$KEEP/git-fetch.log" 2>&1 || true
  fi
  git -C "$REPO" -c advice.detachedHead=false checkout --quiet --detach "$PIN_SHA" > "$KEEP/git-checkout.log" 2>&1 \
    || die 5 "git checkout $PIN_SHA failed (is the commit on $NP_REPO_URL?): see $KEEP/git-checkout.log"
  [ "$(git -C "$REPO" rev-parse HEAD)" = "$PIN_SHA" ] || die 5 "HEAD is $(git -C "$REPO" rev-parse HEAD), expected $PIN_SHA"
  git -C "$REPO" merge-base --is-ancestor "$NOEPOCH_SHA" "$PIN_SHA" || die 5 "$PIN_SHA does not descend from noepoch/stark @ $NOEPOCH_SHA"
  git -C "$REPO" merge-base --is-ancestor "$EPOCH_SHA" "$PIN_SHA" || die 5 "$PIN_SHA does not descend from #1009 @ $EPOCH_SHA"
  log "HEAD $PIN_SHA (asserted; descends from noepoch/stark $NOEPOCH_SHA and #1009 $EPOCH_SHA)"
  step_end 0
}

fetch_fixtures() {
  local elf="$W/fixtures/ethrex_8f826601.elf" input="$W/fixtures/ethrex_mainnet_25368371_573004e6.bin" got mk
  step_begin "fixtures: block 25368371 (release asset) and the record's guest ELF (git), by sha256"
  mk="$(awk -F':= *' '/^ETHREX_REAL_BLOCK_FIXTURE_SHA256/ { print $2; exit }' "$REPO/Makefile" | tr -d ' ')"
  if [ "$mk" != "$INPUT_SHA256" ]; then die 5 "the Makefile's ETHREX_REAL_BLOCK_FIXTURE_SHA256 at $PIN_SHA is '$mk', this script pins $INPUT_SHA256"; fi
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

BIN="" CUBIN_DIR="" CUBIN_LINEINFO=""
build_all() {
  local rc f n=0 empty=0 li=0 missing=""
  build_env
  step_begin "build: the prover's test binary (cargo test --release -p lambda-vm-prover --features $FEATURES --lib --no-run, lineinfo cubins)"
  rc=0
  (cd "$REPO" && exec env -i "${BUILD_ENV[@]}" ${ARCH_ENV[@]+"${ARCH_ENV[@]}"} timeout "$NP_BUILD_TIMEOUT" \
    cargo test --release -p lambda-vm-prover --features "$FEATURES" --lib --no-run --message-format=json-render-diagnostics) \
    > "$KEEP/cargo-prover.json" 2> "$KEEP/build-prover.log" || rc=$?
  tail -n 40 "$KEEP/build-prover.log" > "$SEND/logs/build-prover.tail.txt" || true
  step_end "$rc"
  if [ "$rc" -ne 0 ]; then die 5 "the prover build failed (rc=$rc): see $KEEP/build-prover.log"; fi
  BIN="$(python3 "$TOOL" cargo-artifact --exe lambda_vm_prover < "$KEEP/cargo-prover.json")" \
    || die 5 "cargo reported no lambda_vm_prover test binary (see $KEEP/cargo-prover.json)"
  CUBIN_DIR="$(python3 "$TOOL" cargo-artifact --outdir math-cuda < "$KEEP/cargo-prover.json")" \
    || die 5 "cargo reported no math-cuda build-script output (see $KEEP/cargo-prover.json)"
  for f in $CUBINS; do
    if [ ! -e "$CUBIN_DIR/$f.cubin" ]; then missing="$missing $f"; continue; fi
    n=$((n + 1))
    if [ ! -s "$CUBIN_DIR/$f.cubin" ]; then empty=$((empty + 1)); fi
    if LC_ALL=C grep -q '[.]debug_line' "$CUBIN_DIR/$f.cubin"; then li=$((li + 1)); fi
  done
  log "cubins: $n present, $empty empty, missing:${missing:- none}; with SASS-to-source line tables: $li"
  if [ -n "$missing" ] || [ "$empty" -ne 0 ]; then
    die 5 "kernels missing or empty in $CUBIN_DIR: build.rs did not find nvcc, so every kernel would run on the CPU"
  fi
  CUBIN_LINEINFO="$li of $n"
  for f in "$EPOCH_TEST" "$NOEPOCH_TEST"; do
    n="$({ "$BIN" --list 2>/dev/null || true; } | awk -v t="$f: test" '$0 == t { n++ } END { print n + 0 }')"
    [ "$n" = 1 ] || die 5 "the test binary does not list $f"
  done
  if [ -x "$(cuda_home)/bin/cuobjdump" ]; then
    for f in $CUBINS; do
      echo "== $f.cubin"
      "$(cuda_home)/bin/cuobjdump" -res-usage "$CUBIN_DIR/$f.cubin" 2>&1 | sed "s#$CUBIN_DIR/##g" || true
    done > "$SEND/cubin_res_usage.txt"
  fi
  log "test binary: $BIN (sha256 $(sha256_of "$BIN"))"
}

write_env_facts() { # an allowlist of run facts (this file leaves the machine; the environment itself never does)
  local e w po f="$SEND/env.txt"
  {
    echo "# noepoch_counters.sh run facts: an allowlist. The environment is never recorded here."
    echo "script=$SCRIPT_VERSION md5 $(md5_of "$0") mode=$([ "$NP_DRY" = 1 ] && echo dry || echo counters) workloads=$NP_WORKLOADS reference=$NP_REFERENCE run_a=$NP_RUN_A run_b=$NP_RUN_B"
    echo "repo=$NP_REPO_URL head=$(git -C "$REPO" rev-parse HEAD) (asserted = $PIN_SHA, profile/noepoch-counters)"
    echo "noepoch_base=$NOEPOCH_SHA (noepoch/stark, PR #1013) · epoch_base=$EPOCH_SHA (#1009; its base runs as $EPOCH_TEST on this binary)"
    echo "tracked_files_modified_after_build=$(git -C "$REPO" status --porcelain --untracked-files=no | awk 'END { print NR }')"
    echo "guest_elf=ethrex_8f826601.elf sha256 $ELF_SHA256 (from $ELF_COMMIT:$ELF_REPO_PATH)"
    echo "block_input=ethrex_mainnet_25368371 sha256 $INPUT_SHA256"
    echo "build=cargo test --release -p lambda-vm-prover --features $FEATURES --lib, LAMBDA_VM_NVCC_LINEINFO=1"
    echo "test_binary_sha256=$(sha256_of "$BIN")"
    echo "cubins_with_lineinfo=${CUBIN_LINEINFO:-?}"
    echo "gpu_index=$NP_GPU of $NGPUS"
    echo "gpu_name=$GPU_NAME"
    echo "gpu_compute_capability=$GPU_CC"
    echo "gpu_memory_total_mib=$GPU_VRAM"
    echo "gpu_clocks_max_sm_mhz=$(gpu_field clocks.max.sm) mem_mhz=$(gpu_field clocks.max.mem)"
    echo "gpu_power_limit_w=$(gpu_field power.limit)"
    echo "driver=$GPU_DRIVER cuda=$DRIVER_CUDA nvcc=$NVCC_REL"
    echo "ncu=${NCU_VER:-none} clock_control=$NP_CLOCK kill=$KILL_PROBE run_b_vram_mb=$NP_NCU_VRAM_MB"
    echo "nsys=${NSYS_VER:-none} gpu_metrics=$([ "$WANT_METRICS" = 1 ] && echo "$NP_GPU_METRICS_HZ Hz" || echo off) nvtx_library=$([ -n "$NVTX_LIB" ] && echo found || echo none)"
    echo "cpu=$(cpu_model) · $(nproc 2>/dev/null || echo '?') threads · ram_gib=$(mem_total_gib) · available_at_start_gib=$(mem_avail_gib)"
    echo "rustc=$(cd "$REPO" && env -i "${BUILD_ENV[@]}" rustc --version 2>/dev/null || echo '?')"
    echo "ncu_sections=$NP_SECTIONS"
    echo "ncu_metrics=${NCU_METRICS_OK:-<none validated: dry run or none collectable>}"
    echo "# The profiled process's environment (env -i): PATH, HOME=<W>/home, TMPDIR=<W>/tmp, LC_ALL=C, LANG=C,"
    echo "# CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES, CARGO_MANIFEST_DIR, CUDA_HOME/CUDA_PATH/LD_LIBRARY_PATH"
    echo "# when set, LAMBDA_VM_NVTX_LIB when a library was found, and the workload's knobs:"
    for w in epoch noepoch; do
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

GATES_RE='NOEPOCH|PROVE SPLIT|packing admission|table walk|VRAM gate|TABLE TL|device recommit|RecomputedCommitmentMismatch|test result:|panicked|FAILED|^error|==WARNING==|==ERROR=='
gates() { awk -v re="$GATES_RE" '$0 ~ re { print substr($0, 1, 1200) }' "$1"; }

run_readback() { # run_readback WORKLOAD LOG: "test result; packing; split lines; recommit fields; base line; recommits"
  local tr pk sp rcf base recs
  tr="$(awk '/^test result:/ { r = $0 } END { print r }' "$2")"
  pk="$(grep -c 'packing admission (LAMBDA_VM_GATE_PACKING=1)' "$2" || true)"
  sp="$(grep -c 'PROVE SPLIT #' "$2" || true)"
  rcf="$(grep 'PROVE SPLIT #' "$2" | grep -c 'recommit\[' || true)"
  base="$(awk '/NOEPOCH (RESULT|REFERENCE):/ { sub(/^.*NOEPOCH /, "NOEPOCH "); print substr($0, 1, 400); exit }' "$2")"
  recs="$(awk '/NOEPOCH RESULT:/ { if (match($0, /device recommits [0-9]+/)) { s = substr($0, RSTART, RLENGTH); sub(/.* /, "", s); print s; exit } }' "$2")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${tr:-<none>}" "$pk" "$sp" "$rcf" "${base:-<none>}" "${recs:--}"
}
readback_ok() { # readback_ok WORKLOAD "READBACK": the pre-registered log gates
  local tr pk sp rcf
  IFS=$'\t' read -r tr pk sp rcf _ _ <<< "$2"
  case "$tr" in *"1 passed"*) ;; *) return 1 ;; esac
  if [ "$sp" -lt 1 ]; then return 1; fi
  if [ "$1" = noepoch ]; then [ "$pk" -ge 1 ] && [ "$rcf" -ge 1 ]; else [ "$pk" = 0 ]; fi
}

SAMPLER_PID="" SMI_PID=""
start_sampler() { # host MemAvailable and GPU memory every 2 s, for the per-run minima/maxima
  local f="$KEEP/mem.tsv"
  if [ ! -s "$f" ]; then printf 'epoch_s\tmem_avail_mib\tgpu_used_mib\n' > "$f"; fi
  (
    while :; do
      printf '%s\t%s\t%s\n' "$(date +%s)" "$(mem_avail_mib)" "$(gpu_field memory.used)" >> "$f"
      sleep 2
    done
  ) &
  SAMPLER_PID=$!
}
stop_sampler() {
  if [ -n "$SAMPLER_PID" ]; then kill "$SAMPLER_PID" 2>/dev/null || true; wait "$SAMPLER_PID" 2>/dev/null || true; SAMPLER_PID=""; fi
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
mem_window() { # mem_window T0 T1 -> "min_avail_mib<TAB>max_gpu_mib" over the samples in [T0, T1]
  awk -F'\t' -v a="$1" -v b="$2" 'NR > 1 && $1 >= a && $1 <= b {
      if ($2 != "" && (mn == "" || $2 + 0 < mn)) mn = $2 + 0
      if ($3 != "" && (mx == "" || $3 + 0 > mx)) mx = $3 + 0 }
    END { printf "%s\t%s\n", (mn == "" ? "-" : mn), (mx == "" ? "-" : mx) }' "$KEEP/mem.tsv"
}

plain_run() { # plain_run WORKLOAD LOG: the workload's test, no profiler, into LOG
  local rc=0
  wl_env "$1" record
  run_env nvtx
  (cd "$REPO/prover" && exec timeout --signal=INT --kill-after=120 "$NP_RUN_TIMEOUT" \
    env -i "${RUN_ENV[@]}" "${WL_ENV[@]}" "$BIN" "$WL_TEST" --ignored --exact --nocapture --test-threads=1) \
    < /dev/null > "$2" 2>&1 || rc=$?
  return "$rc"
}

traced_run() { # traced_run WORKLOAD REP LOG WITH_METRICS: the workload's test under nsys, into LOG
  local rc=0
  wl_env "$1" record
  run_env nvtx
  nsys_args "$2" "$4"
  (cd "$REPO/prover" && exec timeout --signal=INT --kill-after=300 "$NP_RUN_TIMEOUT" \
    env -i "${RUN_ENV[@]}" "${WL_ENV[@]}" "$NSYS_BIN" "${NSYS_ARGS[@]}" \
    "$BIN" "$WL_TEST" --ignored --exact --nocapture --test-threads=1) < /dev/null > "$3" 2>&1 || rc=$?
  return "$rc"
}

export_trace() { # export_trace REP: REP.sqlite from REP.nsys-rep; returns 1 when there is none
  if [ ! -s "$1.nsys-rep" ]; then return 1; fi
  timeout 1800 "$NSYS_BIN" export --type sqlite --force-overwrite true --output "$1.sqlite" "$1.nsys-rep" \
    > "$1.export.log" 2>&1 || true
  [ -s "$1.sqlite" ]
}

REF_OK=1 REF_LINE=""
reference_runs() {
  local w log smi rc t0 t1 rb mw vr
  printf 'workload\trc\tseconds\ttest_result\tpacking_lines\tsplit_lines\trecommit_fields\tbase\tdevice_recommits\tvram_max_mib\tmin_mem_avail_mib\tmax_gpu_mib\n' > "$SEND/reference/runs.tsv"
  start_sampler
  for w in $NP_WORKLOADS; do
    wait_idle
    log="$KEEP/ref-$w.log" smi="$KEEP/ref-$w.smi"
    step_begin "reference: the $w workload, no profiler"
    start_smi "$smi"
    t0="$(date +%s)"
    rc=0
    plain_run "$w" "$log" || rc=$?
    t1="$(date +%s)"
    stop_smi
    step_end "$rc"
    gates "$log" > "$SEND/logs/ref-$w.gates.txt"
    rb="$(run_readback "$w" "$log")"
    mw="$(mem_window "$t0" "$t1")"
    vr="$(smi_max "$smi")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$w" "$rc" "$((t1 - t0))" "$rb" "$vr" "$mw" >> "$SEND/reference/runs.tsv"
    log "reference $w: rc=$rc in $((t1 - t0)) s · $(printf '%s' "$rb" | cut -f5) · VRAM max $vr MiB · min host MemAvailable / max GPU MiB: ${mw//$'\t'/ / }"
    if [ "$rc" -ne 0 ] || ! readback_ok "$w" "$rb"; then REF_OK=0; fi
    REF_LINE="${REF_LINE:+$REF_LINE · }$w rc $rc $((t1 - t0)) s"
  done
  stop_sampler
}

RUNA_OK=1 RUNA_LINE=""
run_a() { # each workload under nsys (GPU metrics unless dry); per-stage tables, API tables, the plan against the trace
  local w rep log smi rc t0 t1 rb mw arc drc out r metrics=0 rl src nrec drec vr
  if [ "$WANT_METRICS" = 1 ]; then metrics=1; fi
  printf 'workload\trc\tseconds\ttest_result\tpacking_lines\tsplit_lines\trecommit_fields\tbase\tdevice_recommits\trunA_rc\tplan_rc\tstages\trecommit_ranges\tvram_max_mib\tmin_mem_avail_mib\tmax_gpu_mib\n' > "$SEND/runa/runs.tsv"
  start_sampler
  for w in $NP_WORKLOADS; do
    wait_idle
    rep="$KEEP/runa-$w" log="$KEEP/runa-$w.log" smi="$KEEP/runa-$w.smi" out="$SEND/runa/$w"
    mkdir -p "$out"
    step_begin "run A: the $w workload under Nsight Systems ($([ "$metrics" = 1 ] && echo "GPU metrics at $NP_GPU_METRICS_HZ Hz" || echo 'no GPU metrics'))"
    start_smi "$smi"
    t0="$(date +%s)"
    rc=0
    traced_run "$w" "$rep" "$log" "$metrics" || rc=$?
    t1="$(date +%s)"
    stop_smi
    step_end "$rc"
    gates "$log" > "$SEND/logs/runa-$w.gates.txt"
    rb="$(run_readback "$w" "$log")"
    mw="$(mem_window "$t0" "$t1")"
    vr="$(smi_max "$smi")"
    arc=9 drc=9 src="-" nrec="-"
    drec="$(printf '%s' "$rb" | cut -f6)"
    if export_trace "$rep"; then
      step_begin "run A: $w export, nsys stats, the stage and API tables"
      for r in cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_gpu_mem_size_sum cuda_api_sum nvtx_sum nvtx_gpu_proj_sum; do
        timeout 900 "$NSYS_BIN" stats --report "$r" --format csv --output "$out/nsys" "$rep.sqlite" \
          >> "$KEEP/runa-$w.stats.log" 2>&1 || true
      done
      arc=0
      python3 "$TOOL" runa --sqlite "$rep.sqlite" --log "$log" --workload "$w" --smi "$smi" --out "$out" \
        > "$KEEP/runa-$w.summary.log" 2>&1 || arc=$?
      drc=0
      python3 "$TOOL" dry --plan "$PLAN_FILE" --workload "$w" --sqlite "$rep.sqlite" --log "$log" --out "$out" \
        --tag "runa-$w" > "$KEEP/runa-$w.plan.log" 2>&1 || drc=$?
      step_end "$arc"
      rl="$(grep '^RUNA ' "$KEEP/runa-$w.summary.log" | tail -1 || true)"
      case "$rl" in *"stages from NVTX ranges"*) src=nvtx ;; *"PROVE SPLIT"*) src=log ;; *) src=none ;; esac
      nrec="$(printf '%s' "$rl" | sed -n 's/.*recommit ranges \([0-9]*\).*/\1/p')"
      if [ "$arc" -eq 0 ]; then sed -n '/^| stage /,/^$/p' "$out/runa-$w.md"; fi
      grep -E '^DRY ' "$KEEP/runa-$w.plan.log" || true
    else
      log "run A $w: nsys wrote no report: see $log"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$w" "$rc" "$((t1 - t0))" "$rb" "$arc" "$drc" "$src" "${nrec:--}" "$vr" "$mw" >> "$SEND/runa/runs.tsv"
    log "run A $w: rc=$rc in $((t1 - t0)) s · $(printf '%s' "$rb" | cut -f5) · tables rc=$arc · plan rc=$drc · stages from $src · recommit ranges ${nrec:--} (device recommits $drec) · VRAM max $vr MiB"
    if [ "$rc" -ne 0 ] || [ "$arc" -ne 0 ] || ! readback_ok "$w" "$rb"; then RUNA_OK=0; fi
    if [ "$src" != nvtx ]; then RUNA_OK=0; fi
    if [ "$w" = noepoch ] && [ "${nrec:-x}" != "$drec" ]; then RUNA_OK=0; log "run A noepoch: recommit ranges ${nrec:--} != device recommits $drec"; fi
    if [ "$w" = epoch ] && [ "${nrec:-0}" != 0 ]; then RUNA_OK=0; fi
    if [ "$NP_DRY" = 1 ] && [ "$drc" -ne 0 ]; then RUNA_OK=0; fi
    RUNA_LINE="${RUNA_LINE:+$RUNA_LINE · }$w rc $rc $((t1 - t0)) s, stages $src, recommits ${nrec:--}/$drec"
  done
  stop_sampler
}

PASSES_RUN=0 PASSES_OK=0 LAUNCHES_TOTAL=0
run_passes() { # run B under ncu
  local pass wl mode skip count kernels note log rc t0 t1 profiled verdict mw
  printf 'pass\tworkload\tmode\tskip\tcount\tkernels\trc\tprofiled\tseconds\tmin_mem_avail_mib\tmax_gpu_mib\tverdict\n' > "$SEND/passes.tsv"
  start_sampler
  while IFS=$'\t' read -r pass wl mode skip count kernels _ note; do
    if [ "$pass" = pass ] || ! selected "$pass" "$wl"; then continue; fi
    PASSES_RUN=$((PASSES_RUN + 1))
    wait_idle
    wl_env "$wl" runb
    run_env nvtx
    ncu_args "$mode" "$skip" "$count" "$kernels" "$KEEP/ncu/$pass"
    log="$KEEP/ncu/$pass.log"
    step_begin "run B: pass $pass ($wl, $mode, skip $skip count $count): $note"
    t0="$(date +%s)"
    rc=0
    (cd "$REPO/prover" && exec timeout --signal=INT --kill-after=120 "$NP_PASS_TIMEOUT" \
      env -i "${RUN_ENV[@]}" "${WL_ENV[@]}" "$NCU_BIN" "${NCU_ARGS[@]}" \
      "$BIN" "$WL_TEST" --ignored --exact --nocapture --test-threads=1) < /dev/null > "$log" 2>&1 || rc=$?
    t1="$(date +%s)"
    profiled="$(grep -c '^==PROF== Profiling' "$log" || true)"
    if [ -s "$KEEP/ncu/$pass.ncu-rep" ]; then
      env -i "${RUN_ENV[@]}" timeout 900 "$NCU_BIN" --import "$KEEP/ncu/$pass.ncu-rep" --page details --csv \
        > "$SEND/ncu/$pass.details.csv" 2> "$KEEP/ncu/$pass.import.log" || log "ncu --import (details) failed: see $KEEP/ncu/$pass.import.log"
    fi
    python3 "$TOOL" stages --log "$log" > "$SEND/ncu/$pass.stages.tsv" 2>> "$KEEP/ncu/$pass.import.log" || true
    { grep -E '^==(PROF|WARNING|ERROR)==' "$log" || true; } > "$SEND/ncu/$pass.prof.txt"
    gates "$log" | grep -v 'TABLE TL' > "$SEND/logs/$pass.gates.txt" || true
    mw="$(mem_window "$t0" "$t1")"
    if [ "$profiled" -ge 1 ] && [ -s "$SEND/ncu/$pass.details.csv" ]; then
      verdict=ok
      if [ "$mode" = window ] && [ "$profiled" -lt "$count" ]; then verdict="partial ($profiled of $count)"; fi
    else verdict="FAILED (rc=$rc, $profiled profiled; see keep/ncu/$pass.log)"; fi
    if [ "$verdict" = ok ] || [ "${verdict#partial}" != "$verdict" ]; then PASSES_OK=$((PASSES_OK + 1)); fi
    LAUNCHES_TOTAL=$((LAUNCHES_TOTAL + profiled))
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$pass" "$wl" "$mode" "$skip" "$count" "$kernels" \
      "$rc" "$profiled" "$((t1 - t0))" "$mw" "$verdict" >> "$SEND/passes.tsv"
    log "pass $pass: $profiled launch(es) profiled in $((t1 - t0)) s, $verdict"
    step_end "$rc"
  done < "$PLAN_FILE"
  stop_sampler
}

# ---------------------------------------------------------------------------------------------------
# --print-commands: every command a run would execute, with placeholders, nothing run (the laptop check)

print_commands() {
  local w pass wl mode skip count kernels
  W="$NP_WORKDIR" REPO="$NP_WORKDIR/lambda_vm" BIN="<W>/lambda_vm/target/release/deps/lambda_vm_prover-<hash>"
  NSYS_BIN=nsys NCU_BIN=ncu NSYS_METRICS_FLAG=--gpu-metrics-devices NCU_HAS_KILL=1 NCU_METRICS_OK="<the metrics this ncu can collect>"
  NVTX_LIB="$(find_nvtx_lib || true)"
  if [ -z "$NVTX_LIB" ]; then NVTX_LIB="<LAMBDA_VM_NVTX_LIB>"; fi
  build_env
  echo "# noepoch_counters.sh $SCRIPT_VERSION --print-commands: nothing is run. Work directory $W"
  echo "# checkout: git clone $NP_REPO_URL $REPO; git checkout --detach $PIN_SHA"
  echo "# fixtures: $NP_INPUT_URL (sha256 $INPUT_SHA256); $ELF_COMMIT:$ELF_REPO_PATH (sha256 $ELF_SHA256)"
  echo "build: (cd $REPO && env -i ${BUILD_ENV[*]} cargo test --release -p lambda-vm-prover --features $FEATURES --lib --no-run)"
  for w in $NP_WORKLOADS; do
    wl_env "$w" record
    run_env nvtx
    if [ "$NP_REFERENCE" = 1 ]; then
      echo "reference $w: (cd $REPO/prover && env -i ${RUN_ENV[*]} ${WL_ENV[*]} $BIN $WL_TEST --ignored --exact --nocapture --test-threads=1)"
    fi
    if [ "$NP_RUN_A" = 1 ]; then
      nsys_args "<W>/runs/<UTC>/keep/runa-$w" "$([ "$NP_DRY" = 1 ] && echo 0 || echo 1)"
      echo "run A $w: (cd $REPO/prover && env -i ${RUN_ENV[*]} ${WL_ENV[*]} $NSYS_BIN ${NSYS_ARGS[*]} $BIN $WL_TEST --ignored --exact --nocapture --test-threads=1)"
    fi
  done
  if [ "$NP_RUN_B" = 1 ] && [ "$NP_DRY" = 0 ]; then
    while IFS=$'\t' read -r pass wl mode skip count kernels _; do
      if [ "$pass" = pass ] || ! selected "$pass" "$wl"; then continue; fi
      wl_env "$wl" runb
      run_env nvtx
      ncu_args "$mode" "$skip" "$count" "$kernels" "<W>/runs/<UTC>/keep/ncu/$pass"
      echo "run B $pass: (cd $REPO/prover && env -i ${RUN_ENV[*]} ${WL_ENV[*]} $NCU_BIN ${NCU_ARGS[*]} $BIN $WL_TEST --ignored --exact --nocapture --test-threads=1)"
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
    echo "noepoch_counters.sh bundle ($SCRIPT_VERSION): text only. Read SUMMARY.md first."
    echo "  SUMMARY.md                          the whole run in one file: reference walls, run A per workload, run B epoch vs no-epoch"
    echo "  reference/runs.tsv                  the unprofiled runs: rc, seconds, the NOEPOCH line, VRAM max, host memory"
    echo "  runa/runs.tsv                       run A per workload: rc, seconds, stage source, recommit ranges vs device recommits"
    echo "  runa/<w>/runa-<w>.md, *-stages.tsv  run A per stage: wall, busy %, SM active/issue %, warps, DRAM/PCIe %, VRAM, tasks, kernels"
    echo "  runa/<w>/*-timeseries.tsv           run A in 100 ms bins: stage, kernel/copy busy %, open tasks, VRAM, GPU metrics"
    echo "  runa/<w>/*-nvtx-kernels.tsv         the kernels each NVTX label launched (r1_main_recommit_table = the recommit)"
    echo "  runa/<w>/api-<w>-by-*.tsv           CUDA API seconds per category x stage, per thread, per call"
    echo "  runa/<w>/*-stage-kernels.tsv        every kernel's time per stage · *-gpu-metrics.tsv every GPU-metric series per stage"
    echo "  runa/<w>/nsys_*.csv                 nsys stats: kernel, copy, API, NVTX and NVTX-projected sums"
    echo "  runa/<w>/dry-runa-<w>.md, .tsv      run B's plan against run A's trace: what each pass targets, and coverage"
    echo "  summary/kernels.md, kernels.tsv     run B, one row per kernel and workload, and epoch against no-epoch"
    echo "  summary/kernels_by_stage.tsv        run B per workload, kernel and stage · summary/launches.tsv every profiled launch"
    echo "  ncu/<pass>.details.csv              ncu --import --page details --csv (the Host Name column removed)"
    echo "  ncu/<pass>.stages.tsv, .prof.txt    each profiled launch's stage (from the output order) · ncu's own messages"
    echo "  logs/<run>.gates.txt                the harness's lines (NOEPOCH, PROVE SPLIT, table walks, TABLE TL, test result)"
    echo "  passes.tsv                          run B per pass: rc, launches profiled, seconds, host/GPU memory, verdict"
    echo "  env.txt · preflight.txt · prereg.txt · steps.tsv · driver.log · cubin_res_usage.txt"
    echo "Paths, hostname and addresses of the machine are replaced by <W>, <HOME>, <HOST>, <IP>."
  } > "$SEND/INDEX.txt"
  sensitive_pairs | python3 "$TOOL" scrub --dir "$SEND"
  scan="$(sensitive_values | python3 "$TOOL" check --dir "$SEND" --report "$KEEP/selfcheck.txt")" || rc=$?
  log "$scan"
  tarball="$RUN/noepoch-counters-$STAMP.tar.gz"
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
  size="$(du -h "$tarball" | awk '{ print $1 }')"
  echo "==================== SEND BACK ===================="
  echo "ONE file: $tarball ($size)"
  echo "$scan"
  echo "It holds send/ (INDEX.txt says what each file is; SUMMARY.md alone answers the question)."
  echo "Do NOT send anything else from $W: keep/ holds the .nsys-rep, .sqlite and .ncu-rep files, which store this machine's environment."
  BUNDLE_LINE="bundle $tarball ($size) · self-check clean"
  return 0
}

# ---------------------------------------------------------------------------------------------------
# main

cleanup() { # stop the samplers; then let tee write the last lines before the shell exits
  stop_sampler
  stop_smi
  if [ -n "$TEE_PID" ]; then exec 1>&- 2>&-; wait "$TEE_PID" 2>/dev/null || true; fi
}
main() {
  local preflight_only=0 bundle_only="" print_cmds=0 rc
  while [ $# -gt 0 ]; do
    case "$1" in
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
  log "noepoch_counters.sh $SCRIPT_VERSION · work directory $W · run $RUN · mode $([ "$NP_DRY" = 1 ] && echo 'DRY (no counters)' || echo counters) · workloads $NP_WORKLOADS · reference $NP_REFERENCE · run A $NP_RUN_A · run B $NP_RUN_B"
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
    if [ "$rc" -eq 0 ]; then sed -n '/^## epoch base against no-epoch/,/^## [a-z]*: one row/p' "$SEND/summary/kernels.md" | sed '$d'
    else log "the summary failed (rc=$rc): see $KEEP/summary.out"; fi
  fi
  rc=0
  make_bundle || rc=$?
  if [ "$rc" -ne 0 ]; then echo "VERDICT: $MODE_WORD FAILED rc=4 — the self-check refused the bundle (nothing packed)"; exit 4; fi
  local line="reference: ${REF_LINE:-skipped} (ok=$REF_OK) · run A: ${RUNA_LINE:-skipped} (ok=$RUNA_OK)"
  if [ "$NP_RUN_B" = 1 ] && [ "$NP_DRY" = 0 ]; then line="$line · run B: $PASSES_OK/$PASSES_RUN passes profiled ($LAUNCHES_TOTAL launches)"; fi
  if [ "$REF_OK" = 1 ] && [ "$RUNA_OK" = 1 ] && { [ "$NP_RUN_B" = 0 ] || [ "$NP_DRY" = 1 ] || { [ "$PASSES_OK" -eq "$PASSES_RUN" ] && [ "$PASSES_RUN" -gt 0 ]; }; }; then
    echo "VERDICT: $MODE_WORD DONE — $line · $BUNDLE_LINE"
    exit 0
  fi
  echo "VERDICT: $MODE_WORD PARTIAL rc=3 — $line (reference/runs.tsv, runa/runs.tsv, passes.tsv say which) · $BUNDLE_LINE"
  exit 3
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
