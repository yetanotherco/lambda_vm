#!/usr/bin/env bash
# block_profile.sh — where the card and the host spend PR #1014's median and large block, per stage, on one
# counter-enabled GPU (an RTX 5090). The machine's owner runs it; nobody logs in. It builds the pinned head, proves the
# blocks with and without Nsight Systems, profiles a few kernels with Nsight Compute, and writes one text-only tarball
# to send back.
#
#   bash block_profile.sh                    preflight, setup, build, the runs, the ncu passes, the summary, the tarball
#   bash block_profile.sh --quick            median control + 1 capture, no large block, 3 ncu passes (≈ 25 min less)
#   bash block_profile.sh --preflight-only   the checks alone (≈ 2 min); nothing is built or run
#   options: --dir DIR (work directory, default ~/mauro/lambda-vm-profile) · --gpu N (default 0) · --skip-build (reuse
#            the build of an earlier run in the same DIR) · --no-large (skip the large block) · --no-ncu (skip Nsight
#            Compute) · --keep-raw (keep the .nsys-rep / .ncu-rep / .sqlite files in DIR/runs/*/raw) · --dry-run (print
#            what would run; no GPU work)
#
# Target (pinned by commit and by tree; the script refuses anything else): origin/noepoch/whir 85eb596e0 + the three
#   group-cap commits = 5d1b28030 (tree 81731331), with PR #1014's defaults (padding-only Keccak-round tables left out, the openings' first rounds from one
#   pass, the recursion's program budget reading what a purge could not return).
# Blocks: median 25475471 and large 25481021, and the small 25368371 for the ncu passes (fixtures in this branch,
#   sha256-checked). The large runs at LAMBDA_VM_BLOCK_SPILL_TARGET_GIB=50 so it fits 60 GiB, and only with ≥ 54 GiB free.
# Settings: the production ones (21-row-log tables, the Poseidon1 base, four tables proved at once, a cache of 64
#   precomputed trees, the recursion's WHIR codewords kept, the allocator never returning freed pages), plus
#   LFM_CARD_TRACE=1, which prints one line each time a recursion proof takes the card.
# Runs:
#   full:  01-Mw median warm-up (not read) · 02-Mc median control · 03-Mn 04-Mn median captures · 05-Pc large control ·
#          06-Pn large capture · 07-Sn small capture (only to plan the ncu passes) · ncu passes on the small block
#   quick: 01-Mw · 02-Mc · 03-Mn · 07-Sn · 3 ncu passes
# Captures: Nsight Systems `-t cuda,osrt --sample=none --cpuctxsw=none`, no GPU metrics, so capture ÷ control measures
#   nsys's own cost. ncu: base clocks, one launch per kernel (the first-round tile pass, the commit's leaf hashing, the
#   lookup argument's layer rounds, and in full mode five more).
# Rows (tools/profile_read.py): V1 (rc 0, both verifiers, device counters 0, shape median 941/83/30.767 G, large
#   1880/167/62.706 G), V2 (capture ÷ control 0.99–1.04), V4 (nothing forbidden in the tarball) gate; E1–E5 expectations.
#
# What leaves the machine: only DIR/profile-send-<stamp>.tar.gz, plain text, scrubbed of DIR and HOME and scanned for
# credential markers before it is written. The .nsys-rep / .ncu-rep / .sqlite files embed the process environment:
# they stay in DIR (deleted after their summary unless --keep-raw). The profiled processes run under env -i.
# Exit: 0 done (or preflight passed) · 1 usage · 2 preflight failed · 3 setup or build failed · 4 the bundle scan
# refused (nothing to send; see the message) · 5 deadline
set -euo pipefail

# ------------------------------------------------------------------------------------------------ pins
REPO_URL="${PROFILE_REPO_URL:-https://github.com/yetanotherco/lambda_vm.git}"
PROFILE_BRANCH=prof/block-profile-5090
TARGET_SHA=5d1b28030debe4251b4a5101b00424fdfa32de25
TARGET_TREE=8173133151b080afe2be7a6d4c5c6c982e48d1aa
UNION_SHA=85eb596e0768cb928abdcb79a323e9c5e0272a7e
ELF_NAME=ethrex_8f826601.elf
ELF_SHA256=8f826601776d4085cbb6fbf0302fe8d8d5d1be7940ac1aaca24899c6244ec80a
SMALL_NAME=ethrex_mainnet_25368371_573004e6.bin
SMALL_SHA256=573004e62e3680a00d3cdbae19dc4897e2ec60d6ec0c1d05d9ef118cb8aef17f
MED_NAME=ethrex_mainnet_25475471_8140dd6d.bin
MED_SHA256=8140dd6dc606759310d1fc4b0655d426ef6eab9467654468d71e41f27704bbd7
LARGE_NAME=ethrex_mainnet_25481021_53fe47f3.bin
LARGE_SHA256=53fe47f3f74393f33f03945a827924ae1db7bda24f757ab9133d1f77fb93c2e2
RUST_TOOLCHAIN=1.94.0
TEST=lfm::whir_block_tests::the_whir_block_tree_on_a_real_block
SHAPE_JSON='{"M": [941, 83, 30.767], "P": [1880, 167, 62.706]}'   # the BLOCK SHAPE the two blocks print at this head
LARGE_TARGET_GIB=50
TOOLS_LIST="capture_summary.py host_sampler.py tools/profile_read.py tools/ncu_plan.py tools/run_read.py tools/nsys_extract.py \
tools/nsys_tables.py tools/stamp.py tools/stage_windows.py tools/run_summary.py tools/probe.cu"
NCU_KERNELS_FULL="whir_tile_contract_wide|whir_tile_contract,whir_tile_round,whir_tile_materialize,p1s_leaves_base_coset,gkr_round_gruen,fraction_fold_ext3,zc_round_gruen,ntt_cm_dit_k7"
NCU_KERNELS_QUICK="whir_tile_contract_wide|whir_tile_contract,p1s_leaves_base_coset,gkr_round_gruen"
NCU_SECTIONS="SpeedOfLight MemoryWorkloadAnalysis ComputeWorkloadAnalysis Occupancy LaunchStats"
# 40 ncu metrics (collectable with ncu 2025.3.0 on an RTX 5090); the preflight keeps them only if the probe can collect them
NCU_METRICS="gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,lts__t_sector_hit_rate.pct,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum,\
smsp__thread_inst_executed.sum,launch__registers_per_thread,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,\
launch__occupancy_limit_warps,launch__occupancy_limit_blocks,sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_uniform.avg.pct_of_peak_sustained_active,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"
SECRET_MARKERS='TOKEN=|_TOKEN|API_KEY|_KEY=|SECRET|PASSWORD|BEGIN .*PRIVATE|ssh-(rsa|ed25519|dss)|ghp_|github_pat_|LS0tLS1CRUdJT|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{20}'

# ------------------------------------------------------------------------------------------------ args
MODE=full; PREFLIGHT_ONLY=0; SKIP_BUILD=0; KEEP_RAW=0; DRY=0; GPU=0; LARGE=1; NCU_ON=1
W="${PROFILE_DIR:-$HOME/mauro/lambda-vm-profile}"
usage() { sed -n '2,37p' "${BASH_SOURCE[0]}"; exit 1; }
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) MODE=quick ;;
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --keep-raw) KEEP_RAW=1 ;;
    --dry-run) DRY=1 ;;
    --no-large) LARGE=0 ;;
    --no-ncu) NCU_ON=0 ;;
    --gpu) GPU="${2:?}"; shift ;;
    --dir) W="${2:?}"; shift ;;
    -h|--help) usage ;;
    *) echo "block_profile: unknown argument $1"; usage ;;
  esac
  shift
done
[[ "$GPU" =~ ^[0-9]+$ ]] || usage
[ "$MODE" = full ] || LARGE=0
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$W"
W="$(cd "$W" && pwd)"
RUN="$W/runs/$STAMP"; SEND="$RUN/send"; RAWD="$RUN/raw"; TAGD="$RAWD/tag"
mkdir -p "$SEND" "$RAWD" "$TAGD" "$W/home" "$W/tmp" "$W/cargo-home" "$W/fixtures" "$W/tools" "$W/probe"
LOG="$SEND/profile.log"
T_START=$(date +%s)
DEADLINE_MIN="${PROFILE_DEADLINE_MIN:-150}"
say() { echo "PROFILE $(date -u +%H:%M:%SZ) $*" | tee -a "$LOG"; }
die() { local rc=$1; shift; say "STOP ($rc): $*"; echo "block_profile: stopped: $*  (log: $LOG)"; exit "$rc"; }
step() { printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "${2:-}" >> "$SEND/steps.tsv"; say "== $1 ${2:-}"; }
left_min() { echo $(( DEADLINE_MIN - ($(date +%s) - T_START) / 60 )); }

# ------------------------------------------------------------------------------------------------ tools
find_tool() {  # find_tool NAME → path or empty: PATH, then CUDA_HOME/bin, /usr/local/cuda/bin, the newest Nsight install
  local n="$1" p
  p="$(command -v "$n" 2>/dev/null || true)"
  [ -n "$p" ] && { echo "$p"; return; }
  for d in "${CUDA_HOME:-}" "${CUDA_PATH:-}" /usr/local/cuda; do
    [ -n "$d" ] && [ -x "$d/bin/$n" ] && { echo "$d/bin/$n"; return; }
  done
  p="$(ls -d /opt/nvidia/nsight-*/*/bin/"$n" /opt/nvidia/nsight-*/bin/"$n" /usr/local/cuda*/bin/"$n" 2>/dev/null | sort -V | tail -1 || true)"
  [ -z "$p" ] || echo "$p"
  return 0
}
NVSMI="$(find_tool nvidia-smi)"; NVCC="$(find_tool nvcc)"; NSYS="$(find_tool nsys)"; NCU="$(find_tool ncu)"
CARGO_BIN="$(dirname "$(command -v cargo 2>/dev/null || echo "$HOME/.cargo/bin/cargo")")"
RUSTUP_HOME_REAL="${RUSTUP_HOME:-$HOME/.rustup}"
CUDA_DIR="$(dirname "$(dirname "${NVCC:-/usr/local/cuda/bin/nvcc}")")"
BASE_PATH="$CARGO_BIN:$CUDA_DIR/bin:$(dirname "${NSYS:-/usr/bin/nsys}"):$(dirname "${NCU:-/usr/bin/ncu}"):/usr/local/bin:/usr/bin:/bin"
RUN_ENV=("PATH=$BASE_PATH" "HOME=$W/home" "TMPDIR=$W/tmp" "LC_ALL=C" "LANG=C" "CUDA_DEVICE_ORDER=PCI_BUS_ID" "CUDA_VISIBLE_DEVICES=$GPU")
[ -z "${LD_LIBRARY_PATH:-}" ] || RUN_ENV+=("LD_LIBRARY_PATH=$LD_LIBRARY_PATH")
BUILD_ENV=("PATH=$BASE_PATH" "HOME=$W/home" "RUSTUP_HOME=$RUSTUP_HOME_REAL" "RUSTUP_TOOLCHAIN=$RUST_TOOLCHAIN" "CARGO_HOME=$W/cargo-home"
           "TMPDIR=$W/tmp" "LC_ALL=C" "LANG=C" "CUDA_HOME=$CUDA_DIR" "CUDA_VISIBLE_DEVICES=$GPU")
T="$W/tools"
SUMMARY_TOOL="$T/capture_summary.py"; SAMPLER="$T/host_sampler.py"; READ="$T/profile_read.py"; PLAN="$T/ncu_plan.py"
STAMPER="$T/stamp.py"; EXTRACT="$T/nsys_extract.py"

text_of() { "$@" 2>&1 || true; }                       # a command's whole output, whatever its status
first_of() { grep -oE -- "$1" | head -1 || true; }      # the first match of an ERE on stdin, or nothing
gpu_q() { "$NVSMI" -i "$GPU" --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null | head -1 | sed 's/^ *//; s/ *$//'; }
gpu_idle() {  # used MiB below the floor and no compute process on this GPU
  local used apps
  used="$(gpu_q memory.used)"
  apps="$("$NVSMI" -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
  [ "${used:-99999}" -lt "${PROFILE_IDLE_MIB:-1024}" ] && [ "${apps:-1}" -eq 0 ]
}
avail_gib() { awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo; }
wait_ready() {  # wait_ready LABEL MIN_GIB: the GPU idle and ≥ MIN_GIB available, up to 5 min
  local t=0 av
  while :; do
    av="$(avail_gib)"
    if gpu_idle && [ "${av:-0}" -ge "$2" ]; then return 0; fi
    [ "$t" -lt 300 ] || { say "$1: not ready after 5 min (GPU used $(gpu_q memory.used) MiB, $av GiB available, $2 wanted)"; return 1; }
    sleep 10; t=$((t + 10))
  done
}
bounded() {  # bounded SECS OUT cmd...: own process group; past SECS: INT, 60 s, KILL to the group; status or 124. With
             # ABORT_LOG set: once that log shows a device ABORT, the group gets 60 s to exit on its own, then the same stop (125)
  local secs="$1" out="$2" pid t=0 rc ab=-1
  shift 2
  set -m
  "$@" < /dev/null > "$out" 2>&1 &
  pid=$!
  set +m
  while kill -0 "$pid" 2>/dev/null; do
    if [ -n "${ABORT_LOG:-}" ] && [ "$ab" -lt 0 ] && [ $((t % 5)) -eq 0 ] && grep '\[gpu\] ABORT' "$ABORT_LOG" > /dev/null 2>&1; then
      ab=$t; say "a device ABORT in $(basename "$(dirname "$ABORT_LOG")"): the run gets 60 s to exit, then it is stopped"
    fi
    if [ "$t" -ge "$secs" ] || { [ "$ab" -ge 0 ] && [ $((t - ab)) -ge 60 ]; }; then
      say "$( [ "$t" -ge "$secs" ] && echo "timeout after ${secs} s" || echo "hung after a device ABORT"): stopping process group $pid"
      kill -INT -- "-$pid" 2>/dev/null || true
      for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -KILL -- "-$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      [ "$ab" -ge 0 ] && return 125
      return 124
    fi
    sleep 1; t=$((t + 1))
  done
  rc=0; wait "$pid" || rc=$?
  return "$rc"
}

# ------------------------------------------------------------------------------------------------ flags
NSYS_FLAGS=()
nsys_flags() {  # nsys_flags OUT-PREFIX → NSYS_FLAGS: CUDA and OS-runtime trace, no GPU metrics
  NSYS_FLAGS=("$NSYS" profile -t cuda,osrt --sample=none --cpuctxsw=none --cuda-memory-usage=false --stats=false
              --force-overwrite=true -o "$1" --osrt-threshold=10000)
}
NCU_KILL=1; NCU_METRICS_OK="$NCU_METRICS"; CC_ARCH=sm_120; HAVE_NUMPY=1
ncu_flags() {  # ncu_flags KERNEL SKIP COUNT REPORT → NCU_FLAGS: one launch, base clocks (the preflight probe uses these too)
  NCU_FLAGS=(--target-processes all --kernel-name-base function --kernel-name "regex:^${1}\$" --launch-skip "$2" --launch-count "$3")
  [ "$NCU_KILL" = 0 ] || NCU_FLAGS+=(--kill yes)
  for s in $NCU_SECTIONS; do NCU_FLAGS+=(--section "$s"); done
  [ -z "$NCU_METRICS_OK" ] || NCU_FLAGS+=(--metrics "$NCU_METRICS_OK")
  NCU_FLAGS+=(--clock-control base --export "$4" --force-overwrite)
}

# ------------------------------------------------------------------------------------------------ preflight
NFAIL=0; NWARN=0
chk() { printf '%s %s\n' "$1" "$2" | tee -a "$SEND/preflight.txt" >> "$LOG"; echo "  $1 $2"
        case "$1" in FAIL) NFAIL=$((NFAIL + 1)) ;; WARN) NWARN=$((NWARN + 1)) ;; esac; }
preflight() {
  step preflight
  : > "$SEND/preflight.txt"
  [ "$(uname -s)" = Linux ] && chk PASS "host: Linux · $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-?}")" || chk FAIL "host: Linux needed"
  chk INFO "cpu: $(awk -F': ' '/^model name/ {print $2; exit}' /proc/cpuinfo) · $(nproc) threads"
  [ "${BASH_VERSINFO[0]}" -ge 5 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -ge 4 ]; } && chk PASS "bash ${BASH_VERSION}" || chk FAIL "bash >= 4.4 needed"
  for t in git tar timeout awk sed python3 sha256sum top pgrep; do command -v "$t" > /dev/null || chk FAIL "missing tool: $t"; done
  if python3 -c 'import sqlite3, sys; assert sys.version_info >= (3, 8)' 2>/dev/null; then chk PASS "python3 $(python3 -c 'import sys; print(sys.version.split()[0])') with sqlite3"
  else chk FAIL "python3 >= 3.8 with the sqlite3 module needed"; fi
  if python3 -c 'import numpy' 2>/dev/null; then HAVE_NUMPY=1; chk PASS "python3 numpy: the captures' per-bin tables (x/) are extracted"
  else HAVE_NUMPY=0; chk WARN "python3 has no numpy: the captures' per-bin tables (x/) and two expectation rows are skipped; the stage tables come from capture_summary.py (standard library). To get them: apt install python3-numpy"; fi
  for v in NVSMI NVCC NSYS; do [ -n "${!v}" ] && [ -x "${!v}" ] || chk FAIL "missing: $(echo "$v" | tr 'A-Z' 'a-z' | sed 's/nvsmi/nvidia-smi/') (CUDA toolkit / Nsight Systems)"; done
  if [ "$NCU_ON" = 1 ] && { [ -z "$NCU" ] || [ ! -x "$NCU" ]; }; then chk FAIL "missing: ncu (Nsight Compute); or pass --no-ncu"; fi
  [ "$NFAIL" -eq 0 ] || return 0
  local name cc mem drv used plim
  name="$(gpu_q name)"; cc="$(gpu_q compute_cap)"; mem="$(gpu_q memory.total)"; drv="$(gpu_q driver_version)"; plim="$(gpu_q power.limit)"
  [ -n "$name" ] && chk PASS "gpu $GPU: $name · compute capability $cc · $mem MiB · driver $drv · power limit $plim W" || chk FAIL "gpu $GPU: nvidia-smi sees none"
  [ "$cc" = "12.0" ] || chk WARN "compute capability $cc: the plan was made on an RTX 5090 (12.0)"
  CC_ARCH="sm_${cc//./}"
  used="$(gpu_q memory.used)"
  gpu_idle && chk PASS "idle: gpu $GPU uses $used MiB, no compute process" || chk FAIL "gpu $GPU is busy ($used MiB used or a compute process): close it and re-run"
  case "$drv" in 58[0-9].*) chk WARN "driver $drv: on 580.65 freed device memory was not reused; 595.71 is the tested driver" ;; esac
  NVCC_VER="$(text_of "$NVCC" --version | first_of 'release [0-9.]+')"; NSYS_VER="$(text_of "$NSYS" --version | first_of '20[0-9]{2}\.[0-9.]+')"
  NCU_VER="-"; [ "$NCU_ON" = 0 ] || NCU_VER="$(text_of "$NCU" --version | first_of '20[0-9]{2}\.[0-9.]+')"
  chk PASS "nvcc $NVCC_VER · nsys $NSYS_VER · ncu $NCU_VER"
  local mt av fr
  mt="$(awk '/^MemTotal:/ {printf "%.1f", $2 / 1048576}' /proc/meminfo)"; av="$(awk '/^MemAvailable:/ {printf "%.1f", $2 / 1048576}' /proc/meminfo)"
  awk -v m="$mt" 'BEGIN { exit !(m >= 56) }' && chk PASS "ram: MemTotal $mt GiB, $av available (the prover sizes its spill and program budget to MemTotal - 10 GiB)" \
    || chk FAIL "ram: MemTotal $mt GiB; the median block needs >= 56 GiB"
  awk -v a="$av" 'BEGIN { exit !(a >= 46) }' || chk FAIL "ram: only $av GiB available; close other programs (>= 46 needed before each median run)"
  if [ "$LARGE" = 1 ]; then
    awk -v m="$mt" 'BEGIN { exit !(m >= 58) }' || { LARGE=0; chk WARN "ram: MemTotal $mt GiB < 58: the large block is skipped"; }
    [ "$LARGE" = 0 ] || chk INFO "the large block runs at a spill target of $LARGE_TARGET_GIB GiB, and only with >= 54 GiB available just before each of its runs"
  fi
  fr="$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9)"
  [ "${fr:-0}" -ge 60 ] && chk PASS "disk: $fr GiB free in the work directory" || chk FAIL "disk: $fr GiB free in $W; need >= 60 (build ≈ 15, captures ≈ 15, a possible spill)"
  if text_of env RUSTUP_HOME="$RUSTUP_HOME_REAL" "$CARGO_BIN/rustup" toolchain list | grep "^$RUST_TOOLCHAIN" > /dev/null; then chk PASS "rust: toolchain $RUST_TOOLCHAIN installed"
  else chk FAIL "rust: toolchain $RUST_TOOLCHAIN missing: rustup toolchain install $RUST_TOOLCHAIN"; fi
  # the capture path end to end on a probe: nvcc, nsys with the run flags, the export, the text extraction
  if ! env -i "${BUILD_ENV[@]}" timeout 300 "$NVCC" -O2 -arch="$CC_ARCH" -o "$W/probe/probe" "$T/probe.cu" > "$W/probe/nvcc.log" 2>&1; then
    chk FAIL "nvcc could not build the probe for $CC_ARCH: see $W/probe/nvcc.log"; return 0; fi
  rm -rf "$W/probe/x" "$W/probe/cap.nsys-rep" "$W/probe/cap.sqlite"
  nsys_flags "$W/probe/cap"
  local pok=0
  : > "$W/probe/empty.log"
  if env -i "${RUN_ENV[@]}" timeout 300 "${NSYS_FLAGS[@]}" "$W/probe/probe" > "$W/probe/nsys.log" 2>&1 \
     && env -i "${RUN_ENV[@]}" timeout 300 "$NSYS" export --type=sqlite --force-overwrite=true --output="$W/probe/cap.sqlite" "$W/probe/cap.nsys-rep" >> "$W/probe/nsys.log" 2>&1; then
    python3 "$SUMMARY_TOOL" runa --sqlite "$W/probe/cap.sqlite" --log "$W/probe/empty.log" --kind W --out "$W/probe/runa" >> "$W/probe/nsys.log" 2>&1 || true
    [ "$(grep -oE 'kernels [0-9]+' "$W/probe/nsys.log" | tail -1 | grep -oE '[0-9]+' || echo 0)" = 50 ] && pok=1
    if [ "$HAVE_NUMPY" = 1 ]; then
      python3 "$EXTRACT" "$W/probe/cap.sqlite" "$W/probe/x" 0.1 >> "$W/probe/nsys.log" 2>&1 \
        && grep -q $'^kernels\t50$' "$W/probe/x/summary.tsv" && grep -q $'^memcpys\t100$' "$W/probe/x/summary.tsv" \
        && grep -q $'^memsets\t1$' "$W/probe/x/summary.tsv" || pok=0
    fi
  fi
  [ "$pok" = 1 ] && chk PASS "nsys: the run flags trace the probe (50 kernels); export + the stage tool$( [ "$HAVE_NUMPY" = 1 ] && echo ' + the per-bin extraction (100 copies, 1 memset)') read them" \
    || chk FAIL "nsys: the probe's capture / export / extraction failed: see $W/probe/nsys.log"
  rm -f "$W/probe/cap.nsys-rep" "$W/probe/cap.sqlite"
  [ "$NCU_ON" = 1 ] || return 0
  case "$(text_of "$NCU" --help)" in *--kill*) NCU_KILL=1 ;; *) NCU_KILL=0; chk WARN "ncu has no --kill: each pass runs its workload to the end (slower)" ;; esac
  NCU_METRICS_OK=""
  if env -i "${RUN_ENV[@]}" timeout 300 "$NCU" --metrics "$NCU_METRICS" -k regex:^profile_probe -c 1 "$W/probe/probe" > "$W/probe/ncu-metrics.log" 2>&1 \
     && grep -q "sm__throughput" "$W/probe/ncu-metrics.log"; then
    NCU_METRICS_OK="$NCU_METRICS"; chk PASS "counters: readable; the ncu metric list is collectable"
  elif env -i "${RUN_ENV[@]}" timeout 300 "$NCU" --section SpeedOfLight -c 1 "$W/probe/probe" > "$W/probe/ncu.log" 2>&1 \
     && grep -q "Compute (SM) Throughput" "$W/probe/ncu.log"; then
    chk WARN "counters: readable, but the metric list failed (see $W/probe/ncu-metrics.log): the ncu passes use the sections only"
  else chk FAIL "counters: ncu could not profile the probe (ERR_NVGPUCTRPERM? see $W/probe/ncu.log); or pass --no-ncu"; return 0; fi
  local pk
  pk="$(grep -oE '__global__ void [A-Za-z0-9_]+' "$T/probe.cu" | head -1 | awk '{print $3}')"
  rm -f "$W/probe/ncuprobe.ncu-rep"
  ncu_flags "${pk:-probe}" 1 1 "$W/probe/ncuprobe"
  env -i "${RUN_ENV[@]}" timeout 300 "$NCU" "${NCU_FLAGS[@]}" "$W/probe/probe" > "$W/probe/ncu-pass.log" 2>&1 || true
  if [ -f "$W/probe/ncuprobe.ncu-rep" ] \
     && env -i "${RUN_ENV[@]}" timeout 300 "$NCU" --import "$W/probe/ncuprobe.ncu-rep" --page details --csv > "$W/probe/ncuprobe.csv" 2>> "$W/probe/ncu-pass.log" \
     && python3 "$SUMMARY_TOOL" ncu --csv "$W/probe/ncuprobe.csv" --pass probe --out "$W/probe/ncuprobe.tsv" >> "$W/probe/ncu-pass.log" 2>&1 \
     && [ "$(awk -F'\t' 'NR == 2 && $8 != "-" { print "ok" }' "$W/probe/ncuprobe.tsv")" = ok ]; then
    chk PASS "ncu: the pass flags (launch-skip, --kill $NCU_KILL, base clocks) profile one probe launch ($pk); import + parser read its SM %"
  else chk FAIL "ncu: the pass flags failed on the probe: see $W/probe/ncu-pass.log; or pass --no-ncu"; fi
  rm -f "$W/probe/ncuprobe.ncu-rep"
}

# ------------------------------------------------------------------------------------------------ setup and build
setup_repo() {
  step setup "clone / fetch, the helpers and fixtures"
  local R="$W/repo" rev f
  if [ ! -d "$R/.git" ]; then git clone -q --no-checkout --filter=blob:none "$REPO_URL" "$R" || die 3 "git clone $REPO_URL failed"; fi
  git -C "$R" fetch -q origin "$PROFILE_BRANCH" || die 3 "git fetch of $PROFILE_BRANCH failed"
  git -C "$R" cat-file -e "$TARGET_SHA^{commit}" 2>/dev/null || die 3 "the target $TARGET_SHA is not on origin's $PROFILE_BRANCH"
  [ "$(git -C "$R" rev-parse "$TARGET_SHA~3")" = "$UNION_SHA" ] || die 3 "the target's third parent is not $UNION_SHA"
  [ "$(git -C "$R" rev-parse "$TARGET_SHA^{tree}")" = "$TARGET_TREE" ] || die 3 "the target's tree is not $TARGET_TREE"
  # the helpers and fixtures come from the commit this script came from: PROFILE_REV, else the branch's tip, and that
  # commit's block_profile.sh must be this very file
  rev="${PROFILE_REV:-$(git -C "$R" rev-parse "origin/$PROFILE_BRANCH")}"
  git -C "$R" cat-file -e "$rev:scripts/profile/block_profile.sh" 2>/dev/null || die 3 "no scripts/profile/block_profile.sh at $rev"
  [ "$(git -C "$R" show "$rev:scripts/profile/block_profile.sh" | sha256sum | cut -c1-64)" = "$(sha256sum "$SCRIPT" | cut -c1-64)" ] \
    || die 3 "this block_profile.sh differs from the one at $rev ($PROFILE_BRANCH): download the script again, or set PROFILE_REV to the commit it came from"
  git -C "$R" merge-base --is-ancestor "$TARGET_SHA" "$rev" || die 3 "$rev does not contain the target $TARGET_SHA"
  PROFILE_REV="$rev"
  for f in $TOOLS_LIST; do git -C "$R" show "$rev:scripts/profile/$f" > "$T/$(basename "$f")" || die 3 "helper $f missing at $rev"; done
  for f in "$ELF_NAME:$ELF_SHA256" "$SMALL_NAME:$SMALL_SHA256" "$MED_NAME:$MED_SHA256" "$LARGE_NAME:$LARGE_SHA256"; do
    git -C "$R" show "$rev:scripts/profile/fixtures/${f%%:*}" > "$W/fixtures/${f%%:*}"
    [ "$(sha256sum "$W/fixtures/${f%%:*}" | cut -c1-64)" = "${f##*:}" ] || die 3 "fixture ${f%%:*}: sha256 mismatch"
  done
  python3 "$SUMMARY_TOOL" selftest > "$SEND/selftest.txt" 2>&1 || die 3 "the summary tool's selftest is RED: see $SEND/selftest.txt"
  say "helpers and fixtures from $rev; the summary tool's selftest is GREEN"
}
setup_worktree() {
  step worktree "$TARGET_SHA"
  local R="$W/repo" wt="$W/wt"
  if [ -e "$wt/.git" ]; then git -C "$wt" checkout -q --detach "$TARGET_SHA"; else git -C "$R" worktree add -q --detach "$wt" "$TARGET_SHA"; fi
  [ "$(git -C "$wt" rev-parse HEAD)" = "$TARGET_SHA" ] || die 3 "worktree $wt is not at $TARGET_SHA"
  [ "$(git -C "$wt" rev-parse "HEAD^{tree}")" = "$TARGET_TREE" ] || die 3 "worktree $wt does not hold the tree $TARGET_TREE"
  [ -z "$(git -C "$wt" status --porcelain --untracked-files=no)" ] || die 3 "worktree $wt has local changes"
}
BIN=""
build() {
  local wt="$W/wt" rel=""
  step build "cargo test --release -p lambda-vm-prover --features cuda --lib --no-run"
  if [ "$SKIP_BUILD" = 0 ]; then
    (cd "$wt" && bounded 5400 "$SEND/build.log" env -i "${BUILD_ENV[@]}" "CARGO_TARGET_DIR=$W/target" "CUDARC_NVCC_ARCH=$CC_ARCH" \
       cargo test --release -p lambda-vm-prover --features cuda --lib --no-run) || { tail -30 "$SEND/build.log"; die 3 "build failed: $SEND/build.log"; }
  fi
  [ ! -f "$SEND/build.log" ] || rel="$({ sed -n 's/.*Executable unittests src\/lib.rs (\(.*lambda_vm_prover-[0-9a-f]*\)).*/\1/p' "$SEND/build.log" || true; } | tail -1)"
  if [ -z "$rel" ]; then
    local f
    for f in "$W/target"/release/deps/lambda_vm_prover-*; do
      [ -f "$f" ] && [ -x "$f" ] || continue
      if [ -z "$rel" ] || [ "$f" -nt "$rel" ]; then rel="$f"; fi
    done
  fi
  case "$rel" in /*|"") ;; *) rel="$wt/$rel" ;; esac
  [ -n "$rel" ] && [ -f "$rel" ] && [ -x "$rel" ] || die 3 "no test binary$( [ "$SKIP_BUILD" = 0 ] || echo " in $W/target (--skip-build needs an earlier build in this work directory)")"
  [ -z "$(git -C "$wt" status --porcelain --untracked-files=no)" ] || die 3 "worktree $wt changed during the build"
  BIN="$rel"
  say "binary: $(basename "$BIN") sha256 $(sha256sum "$BIN" | cut -c1-16)…"
}

# ------------------------------------------------------------------------------------------------ the runs
posture() {  # posture small|median|large → the production settings, this machine's fixtures
  local fix="$W/fixtures/$SMALL_NAME" extra=""
  case "$1" in median) fix="$W/fixtures/$MED_NAME" ;; large) fix="$W/fixtures/$LARGE_NAME"; extra=" LAMBDA_VM_BLOCK_SPILL_TARGET_GIB=$LARGE_TARGET_GIB" ;; esac
  echo "LAMBDA_VM_MAX_ROWS_LOG2=21 LFM_WHIR_RETENTION=1 LAMBDA_VM_WHIR_HASH=rpx LFM_PRECOMPUTED_TREE_CACHE_CAP=64 LFM_EXEC_PARALLEL=1 LAMBDA_VM_BASE_SPLIT=1 LAMBDA_VM_GRIND_SCAN_FACTOR=8 LAMBDA_VM_GRIND_GRID=1024 _RJEM_MALLOC_CONF=dirty_decay_ms:-1,muzzy_decay_ms:-1 BLOCK_WHIR_ELF=$W/fixtures/$ELF_NAME BLOCK_WHIR_INPUT=$fix TABLE_PARALLELISM=4 BLOCK_WHIR_BASE=p1w$extra"
}
env_size() {  # env_size WORDS: the environment's size, in bytes, of a process started the way the runs start
  local pid n
  # shellcheck disable=SC2086
  ( cd "$W/wt/prover" && exec env -i "${RUN_ENV[@]}" $1 sleep 3 ) < /dev/null > /dev/null 2>&1 &
  pid=$!
  sleep 1
  n="$(wc -c < "/proc/$pid/environ" | tr -d ' ')"
  wait "$pid" || true
  echo "$n"
}
env_size_nsys() {  # the same under nsys with the run flags: what a profiled prover's environment weighs
  local pd="$W/probe" rc
  rm -f "$pd/envn.txt"
  nsys_flags "$pd/envcap"
  {
    echo "cd $(printf '%q' "$W/wt/prover")"
    # shellcheck disable=SC2086
    echo "exec env -i $(printf '%q ' "${RUN_ENV[@]}")$1 $(printf '%q ' "${NSYS_FLAGS[@]}" bash -c "wc -c < /proc/self/environ > $pd/envn.txt; exec $pd/probe")"
  } > "$pd/envn.sh"
  set +e
  bounded 300 "$pd/envn.log" bash "$pd/envn.sh"
  rc=$?
  set -e
  rm -f "$pd/envcap.nsys-rep"
  [ -s "$pd/envn.txt" ] || return 1
  tr -d ' \n' < "$pd/envn.txt"
}
pad_by() {  # pad_by WORDS N: N more x on the last word (the pad word)
  local w="$1" n="$2" i=0
  while [ "$i" -lt "$n" ]; do w="${w}x"; i=$((i + 1)); done
  echo "$w"
}
device_abort() { grep -qE '\[gpu\] ABORT|CUDA_ERROR_OUT_OF_MEMORY|out of memory|DriverError' "$1" 2>/dev/null; }

# run_one DIR KIND WORDS SECS MIN_GIB [ncu args file]: one proving run; KIND w|c (plain), n (nsys), ncu → RC
run_one() {
  local rd="$1" kind="$2" words="$3" secs="$4" mingib="$5" upid spid wpid t0
  mkdir -p "$rd"
  local pre=()
  case "$kind" in
    n) nsys_flags "$rd/cap"; pre=("${NSYS_FLAGS[@]}") ;;
    ncu) mapfile -t pre < "$6"; pre=("$NCU" "${pre[@]}") ;;
  esac
  printf '%s\n' "$words" | sed "s#$W#<W>#g" > "$rd/env.txt"
  {
    echo "set -o pipefail"
    echo "cd $(printf '%q' "$W/wt/prover")"
    # shellcheck disable=SC2086
    echo "env -i $(printf '%q ' "${RUN_ENV[@]}")$words $(printf '%q ' ${pre[@]+"${pre[@]}"} "$BIN" --ignored --exact --nocapture --test-threads=1 "$TEST")2>&1 | python3 -u $(printf '%q' "$STAMPER") > $(printf '%q' "$rd/out.log")"
  } > "$rd/cmd.sh"
  if [ "$DRY" = 1 ]; then say "DRY $(basename "$rd"): $(tail -1 "$rd/cmd.sh" | sed "s#$W#<W>#g" | cut -c1-300)"; bash -n "$rd/cmd.sh"; RC=0; return 0; fi
  wait_ready "$(basename "$rd")" "$mingib" || { RC=97; return 0; }
  date +%z > "$rd/tz.txt"
  "$NVSMI" -i "$GPU" --query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,power.draw,temperature.gpu --format=csv,noheader,nounits -lms 200 > "$rd/smi.csv" 2>/dev/null &
  upid=$!
  python3 "$SAMPLER" "$BIN" "$rd/host.tsv" "$rd/watchdog.txt" 0.5 1536 &
  spid=$!
  ( set +e   # the prover found by its executable: its environment's size once, then `top -H` on it once a second
    pid=""
    for _ in $(seq 1 600); do
      for p in $(pgrep -x lambda_vm_prove 2>/dev/null); do
        if [ "$(readlink "/proc/$p/exe" 2>/dev/null)" = "$BIN" ]; then pid="$p"; break; fi
      done
      [ -z "$pid" ] || break
      sleep 0.2
    done
    [ -n "$pid" ] || exit 0
    wc -c < "/proc/$pid/environ" 2>/dev/null | tr -d ' ' > "$rd/environ.bytes"
    exec top -H -b -d 1 -w 512 -p "$pid" > "$rd/top.txt" 2>/dev/null
  ) &
  wpid=$!
  t0=$(date +%s)
  set +e
  ABORT_LOG="$rd/out.log" bounded "$secs" "$rd/bounded.out" bash "$rd/cmd.sh"
  RC=$?
  set -e
  kill "$upid" "$spid" "$wpid" 2>/dev/null || true; wait "$upid" "$spid" "$wpid" 2>/dev/null || true
  echo "$RC" > "$rd/rc.txt"
  say "$(basename "$rd"): rc $RC · $(( $(date +%s) - t0 )) s · $(grep -hoE 'W3 RECURSION: .*whole block [0-9.]+s' "$rd/out.log" 2>/dev/null | head -1 | cut -c1-120)$( [ -s "$rd/watchdog.txt" ] && echo " · MEMORY WATCHDOG: $(cat "$rd/watchdog.txt")")$(device_abort "$rd/out.log" && echo ' · DEVICE ABORT' || true)"
}
extract() {  # extract RD: nsys export + stats + the text extraction (+ the ncu plan for the small capture); the raw files go unless --keep-raw
  local rd="$1" erc=0
  [ -f "$rd/cap.nsys-rep" ] || { say "$(basename "$rd"): no report written"; return 1; }
  set +e
  bounded 3600 "$rd/export.log" env -i "${RUN_ENV[@]}" "$NSYS" export --type=sqlite --force-overwrite=true --output="$rd/cap.sqlite" "$rd/cap.nsys-rep"
  erc=$?
  if [ "$erc" -eq 0 ]; then
    mkdir -p "$rd/stats"
    bounded 1800 "$rd/stats.log" env -i "${RUN_ENV[@]}" "$NSYS" stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_api_sum,osrt_sum \
      --format csv --force-overwrite=true --output "$rd/stats/s" "$rd/cap.sqlite" || say "$(basename "$rd"): nsys stats rc $? (non-fatal)"
    python3 "$SUMMARY_TOOL" runa --sqlite "$rd/cap.sqlite" --log "$rd/out.log" --kind W --out "$rd/runa" --host "$rd/host.tsv" \
      --smi "$rd/smi.csv" --tz "$(cat "$rd/tz.txt" 2>/dev/null || echo +0000)" > "$rd/runa.log" 2>&1 || say "$(basename "$rd"): the stage tool failed (non-fatal)"
    if [ "$HAVE_NUMPY" = 1 ]; then bounded 3600 "$rd/extract.log" python3 "$EXTRACT" "$rd/cap.sqlite" "$rd/x" 0.1; erc=$?; fi
    if [ -n "${2:-}" ]; then
      python3 "$PLAN" --sqlite "$rd/cap.sqlite" --out "$2" --kernels "$3" --prefix a 2>&1 | tee -a "$LOG" || say "the ncu plan failed"
    fi
  fi
  set -e
  [ "$KEEP_RAW" = 1 ] || rm -f "$rd/cap.nsys-rep" "$rd/cap.sqlite"
  say "$(basename "$rd"): export + extraction rc $erc$( [ "$KEEP_RAW" = 1 ] && echo ' · raw kept (--keep-raw)' || echo ' · raw removed')"
  return "$erc"
}

# The median and large runs: control and capture of a size carry environments of one size (the pad word)
NRUN=0; NOK=0; NCAP=0; NX=0
measure() {  # measure NAME BLOCK KIND
  local name="$1" blk="$2" kind="$3" words secs mingib
  [ "$(left_min)" -gt 12 ] || { say "$name: skipped (deadline: $(left_min) min left)"; return 0; }
  case "$blk" in M) words="$C_M"; [ "$kind" = n ] && words="$N_M"; secs=1800; mingib=46 ;;
                 P) words="$C_P"; [ "$kind" = n ] && words="$N_P"; secs=2700; mingib=54 ;; esac
  step "run $name" "$( [ "$blk" = M ] && echo median || echo large) $(case "$kind" in w) echo warm-up ;; c) echo control ;; n) echo capture ;; esac)"
  run_one "$TAGD/$name" "$kind" "$words" "$secs" "$mingib"
  [ "$DRY" = 1 ] && return 0
  NRUN=$((NRUN + 1)); [ "$RC" -ne 0 ] || NOK=$((NOK + 1))
  if [ "$RC" = 97 ]; then say "$name: not run (memory or GPU not ready)"; rm -rf "${TAGD:?}/$name"; return 0; fi
  if [ "$kind" = n ]; then NCAP=$((NCAP + 1)); extract "$TAGD/$name" && NX=$((NX + 1)) || true; fi
  python3 "$READ" run "$TAGD/$name" 2>&1 | tail -3 | tee -a "$LOG" || say "$name: read-out failed (non-fatal)"
}

# The ncu passes: one per plan row, on the small block, the launch picked by its order among that kernel's launches
RUNB_OK=0; RUNB_N=0
runb() {
  local plan="$RAWD/plan.tsv" pass kern skip cnt rest
  [ -s "$plan" ] || { say "ncu: no plan (the small capture failed?)"; return 0; }
  cp "$plan" "$SEND/plan.tsv"
  mkdir -p "$SEND/ncu"
  while IFS=$'\t' read -r pass kern skip cnt rest; do
    [ "$pass" = pass ] && continue
    RUNB_N=$((RUNB_N + 1))
    [ "$(left_min)" -gt 6 ] || { say "ncu $pass: skipped (deadline)"; continue; }
    step "ncu $pass" "$kern launch $skip"
    ncu_flags "$kern" "$skip" "$cnt" "$RAWD/ncu-$pass"
    printf '%s\n' "${NCU_FLAGS[@]}" > "$RAWD/ncu-$pass.args"
    run_one "$RAWD/ncu-$pass" ncu "$(posture small)" 900 20 "$RAWD/ncu-$pass.args"
    [ "$DRY" = 1 ] && continue
    if [ -f "$RAWD/ncu-$pass.ncu-rep" ]; then
      set +e
      env -i "${RUN_ENV[@]}" timeout 900 "$NCU" --import "$RAWD/ncu-$pass.ncu-rep" --page details --csv > "$RAWD/ncu-$pass.csv" 2> "$RAWD/ncu-$pass.import.log"
      python3 "$SUMMARY_TOOL" ncu --csv "$RAWD/ncu-$pass.csv" --pass "$pass" --out "$SEND/ncu/$pass.launches.tsv" 2>&1 | tee -a "$LOG"
      set -e
      sed "s#$W#<W>#g; s#$HOME#<HOME>#g" "$RAWD/ncu-$pass.csv" > "$SEND/ncu/$pass.details.csv"
      [ -s "$SEND/ncu/$pass.launches.tsv" ] && [ "$(wc -l < "$SEND/ncu/$pass.launches.tsv")" -gt 1 ] && RUNB_OK=$((RUNB_OK + 1))
      [ "$KEEP_RAW" = 1 ] || rm -f "$RAWD/ncu-$pass.ncu-rep"
    else
      say "ncu $pass: no report (rc $RC): $(grep -m2 -E '==PROF==|==ERROR==|==WARNING==' "$RAWD/ncu-$pass/out.log" 2>/dev/null | tr '\n' ' ' | cut -c1-200)"
    fi
  done < "$plan"
}

# ------------------------------------------------------------------------------------------------ results and bundle
collect() {  # the derived text of every read run into SEND/runs/<name>/ (no raw report, no environment)
  local rd n f
  for rd in "$TAGD"/[0-9][0-9]-*; do
    [ -d "$rd" ] || continue
    n="$(basename "$rd")"; mkdir -p "$SEND/runs/$n"
    for f in env.txt tz.txt smi.csv host.tsv top.txt environ.bytes rc.txt watchdog.txt stage.json win.tsv fam.tsv top.tsv gaps.tsv \
             rate.tsv idle.tsv holds.tsv l0split.json run.md; do
      [ ! -f "$rd/$f" ] || cp "$rd/$f" "$SEND/runs/$n/"
    done
    [ ! -f "$rd/out.log" ] || cp "$rd/out.log" "$SEND/runs/$n/out.log"
    [ ! -d "$rd/stats" ] || cp -r "$rd/stats" "$SEND/runs/$n/"
    [ ! -d "$rd/x" ] || cp -r "$rd/x" "$SEND/runs/$n/"
    [ ! -d "$rd/runa" ] || cp -r "$rd/runa" "$SEND/runs/$n/"
  done
}
scan() {  # scan → nonzero when SEND holds anything that must not leave the machine; findings in RAWD/bundle-scan.txt
  local f
  : > "$RAWD/bundle-scan.txt"
  while IFS= read -r -d '' f; do sed -i "s#$W#<W>#g; s#$HOME#<HOME>#g" "$f"; done < <(find "$SEND" -type f -print0)
  while IFS= read -r -d '' f; do
    case "$f" in *.nsys-rep|*.qdrep|*.qdstrm|*.ncu-rep|*.sqlite|*.db|*.arrow) echo "forbidden file type: ${f#"$SEND"/}" >> "$RAWD/bundle-scan.txt"; continue ;; esac
    [ "$(head -c 15 "$f" | tr -d '\0')" = "SQLite format 3" ] && echo "forbidden file type: ${f#"$SEND"/} (sqlite)" >> "$RAWD/bundle-scan.txt"
    { [ ! -s "$f" ] || LC_ALL=C grep -qI . "$f"; } || echo "forbidden file type: ${f#"$SEND"/} (not text)" >> "$RAWD/bundle-scan.txt"
  done < <(find "$SEND" -type f -print0)
  { LC_ALL=C grep -rnoE -- "$SECRET_MARKERS" "$SEND" 2>/dev/null || true; } | sed "s#^$SEND/##" >> "$RAWD/bundle-scan.txt"
  [ ! -s "$RAWD/bundle-scan.txt" ]
}
facts() {
  {
    echo "block_profile: script sha256 $(sha256sum "$SCRIPT" | cut -c1-16)… from ${PROFILE_REV:-?} · mode $MODE · large $LARGE · ncu $NCU_ON · started $STAMP · deadline $DEADLINE_MIN min"
    echo "target: $TARGET_SHA, tree $TARGET_TREE (origin/noepoch/whir $UNION_SHA + the three group-cap commits)"
    [ -z "$NVSMI" ] || echo "gpu $GPU: $(gpu_q name) · cc $(gpu_q compute_cap) · $(gpu_q memory.total) MiB · driver $(gpu_q driver_version) · power limit $(gpu_q power.limit) W · max SM $(gpu_q clocks.max.sm) MHz"
    echo "tools: nvcc ${NVCC_VER:-?} · nsys ${NSYS_VER:-?} (-t cuda,osrt, no GPU metrics) · ncu ${NCU_VER:-?} (--clock-control base, kill $NCU_KILL) · numpy $HAVE_NUMPY"
    echo "host: $(awk -F': ' '/^model name/ {print $2; exit}' /proc/cpuinfo 2>/dev/null) · $(nproc 2>/dev/null) threads · MemTotal $(awk '/^MemTotal:/ {printf "%.1f", $2 / 1048576}' /proc/meminfo 2>/dev/null) GiB · $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-?}")"
    echo "binary: $(sha256sum "${BIN:-/dev/null}" 2>/dev/null | cut -c1-16)… (cargo test --release -p lambda-vm-prover --features cuda --lib, CUDARC_NVCC_ARCH=${CC_ARCH:-?})"
    echo "fixtures: $ELF_NAME, $SMALL_NAME, $MED_NAME, $LARGE_NAME (sha256-checked); the large at LAMBDA_VM_BLOCK_SPILL_TARGET_GIB=$LARGE_TARGET_GIB"
    echo "runs: $NOK of $NRUN rc 0 · captures extracted $NX of $NCAP · ncu passes summarised $RUNB_OK of $RUNB_N · $(( ($(date +%s) - T_START) / 60 )) min"
  } > "$SEND/facts.txt"
}
bundle() {
  step bundle "read-out, scrub, scan, tar"
  local out="$W/profile-send-$STAMP.tar.gz"
  collect
  if scan; then echo ok > "$TAGD/hygiene.txt"; else echo FAIL > "$TAGD/hygiene.txt"; fi
  set +e
  python3 "$READ" tag "$TAGD" > "$SEND/readout.txt" 2>&1
  set -e
  cp "$TAGD/expect.json" "$SEND/" 2>/dev/null || true
  facts
  python3 "$SUMMARY_TOOL" report --dir "$SEND" > /dev/null 2>&1 || true
  { echo; echo "# rows (tools/profile_read.py)"; echo; sed 's/^/    /' "$SEND/readout.txt"; } >> "$SEND/SUMMARY.md"
  if ! scan; then
    say "BUNDLE REFUSED: $(wc -l < "$RAWD/bundle-scan.txt") finding(s); see $RAWD/bundle-scan.txt (do not send it). Remove the lines it names from $SEND, then: tar -czf $out -C $RUN send"
    return 4
  fi
  tar -czf "$out" -C "$RUN" send
  say "bundle: $out ($(du -h "$out" | cut -f1), $(find "$SEND" -type f | wc -l) plain-text files, scan clean)"
  say "$(grep '^VERDICT:' "$SEND/readout.txt" | cut -c1-200)"
  echo
  echo "=== DONE. Send back this one file (text only): $out"
}

# ------------------------------------------------------------------------------------------------ main
say "block_profile $MODE · work dir $W · gpu $GPU · run $STAMP · large $LARGE · ncu $NCU_ON"
setup_repo
if [ "$DRY" = 1 ]; then say "DRY RUN: no GPU work; the preflight's GPU checks are skipped"
else
  preflight
  say "PREFLIGHT: $( [ "$NFAIL" -eq 0 ] && echo PASS || echo FAIL ) ($NFAIL failure(s), $NWARN warning(s)); $SEND/preflight.txt"
  [ "$NFAIL" -eq 0 ] || exit 2
fi
[ "$PREFLIGHT_ONLY" = 0 ] || exit 0
if [ "$MODE" = quick ]; then est="≈ 30–45 min (build ≈ 10–20, runs ≈ 15, ncu ≈ 5)"
else est="≈ 65–95 min (build ≈ 10–20, median runs ≈ 20, large runs ≈ 20, ncu ≈ 12)$( [ "$LARGE" = 1 ] || echo ' less ≈ 20 without the large block')"; fi
say "estimate: $est; deadline $DEADLINE_MIN min. Leave the GPU and most of the RAM to it."
setup_worktree
if [ "$DRY" = 1 ]; then BIN="$W/target/release/deps/lambda_vm_prover-0000"; else build; fi

W_M="$(posture median) LFM_CARD_TRACE=1 LAMBDA_VM_ND3_PAD=x"
W_P="$(posture large) LFM_CARD_TRACE=1 LAMBDA_VM_ND3_PAD=x"
C_M="$W_M"; C_P="$W_P"; N_M="$W_M"; N_P="$W_P"
if [ "$DRY" = 0 ]; then
  E_M="$(env_size "$W_M")"; E_N="$(env_size_nsys "$W_M")" || E_N=""
  if [[ "$E_M" =~ ^[0-9]+$ ]] && [[ "$E_N" =~ ^[0-9]+$ ]]; then
    D=$((E_N - E_M))
    if [ "$D" -ge 0 ]; then C_M="$(pad_by "$W_M" "$D")"; C_P="$(pad_by "$W_P" "$D")"
    else N_M="$(pad_by "$W_M" $((-D)))"; N_P="$(pad_by "$W_P" $((-D)))"; fi
    say "environment: plain $E_M B · under nsys $E_N B (nsys adds $D B): the shorter arm's pad word is lengthened by $(( D < 0 ? -D : D ))"
  else say "WARNING: the environment's size under nsys could not be read ($W/probe/envn.log): control and capture keep the words as they are"; fi
fi
printf '{"mode": "capture", "why": "nsys %s", "shape": %s, "reference": null}\n' "${NSYS_VER:-?}" "$SHAPE_JSON" > "$TAGD/expect.json"

measure 01-Mw M w
measure 02-Mc M c
measure 03-Mn M n
[ "$MODE" = quick ] || measure 04-Mn M n
if [ "$LARGE" = 1 ]; then measure 05-Pc P c; measure 06-Pn P n; fi
if [ "$NCU_ON" = 1 ]; then
  step "run 07-Sn" "small capture, only to plan the ncu passes"
  kern="$NCU_KERNELS_FULL"; [ "$MODE" = quick ] && kern="$NCU_KERNELS_QUICK"
  run_one "$RAWD/07-Sn" n "$(posture small)" 900 20
  if [ "$DRY" = 1 ]; then printf 'pass\tkernel\tlaunch_skip\tlaunch_count\na01\twhir_tile_contract_wide\t3\t1\na02\tp1s_leaves_base_coset\t10\t1\n' > "$RAWD/plan.tsv"
  else extract "$RAWD/07-Sn" "$RAWD/plan.tsv" "$kern" || true; fi
  runb
fi
[ "$DRY" = 0 ] || { say "DRY RUN done: the commands above would run"; exit 0; }
set +e
bundle; brc=$?
set -e
[ "$brc" -eq 0 ] || exit 4
[ "$(left_min)" -gt 0 ] || exit 5
exit 0
