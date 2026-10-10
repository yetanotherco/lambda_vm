#!/usr/bin/env bash
# prof4.sh — where the block prover's time goes on one counter-enabled GPU (lane I-PROF4, for Mauro's RTX 5090).
#
#   bash prof4.sh                    preflight, setup, builds, run A, run B, the summary, the send-back tarball
#   bash prof4.sh --quick            run A on #1014 only (Poseidon1 bench + median) and 5 ncu passes (≈ 35 min less)
#   bash prof4.sh --preflight-only   the checks alone (≈ 1 min); nothing is built or run
#   options: --dir DIR (work directory, default ./lambda-vm-prof4) · --gpu N (default 0) · --skip-build (reuse the
#            builds of an earlier run in the same DIR) · --keep-raw (keep the .nsys-rep / .ncu-rep files in DIR/runs/*/big)
#            · --dry-run (print what would run; no GPU work)
#
# Targets (pinned; the script refuses any other sha):
#   W = #1014 no-epoch WHIR with the Poseidon1 base, whirp1/s6b-box 4082b0df3 (the same-hash bench's build). Arms on one
#       binary: BLOCK_WHIR_BASE=p1w (Poseidon1, the main arm) and =rpx (reference).
#   S = #1013 no-epoch STARK, noepoch/stark 4841cf5a9, with the Poseidon1 base (NOEPOCH_BASE=p1, cap 1).
#   Blocks: bench 25368371 and median 25475471 (fixtures in this branch, checked by sha256). Not the p90: it needs
#   ≈ 77–83 GiB at the box posture; the box ladder (ULTRA) covers it.
# Run A: each workload once under Nsight Systems with GPU metrics (2 kHz), windowed by the prover's own phase lines.
# Run B: Nsight Compute (base clocks) on one representative launch of each kernel that carries the bench run's time,
#        chosen from run A's bench captures.
#
# What leaves the machine: only DIR/prof4-send-<stamp>.tar.gz, plain text, scrubbed of DIR and HOME and scanned for
# credential markers before it is written. The .nsys-rep / .ncu-rep / .sqlite files embed the process environment:
# they stay in DIR (and are deleted after their summary unless --keep-raw). The profiled processes run under env -i.
# Exit: 0 done (or preflight passed) · 1 usage · 2 preflight failed · 3 setup or build failed · 4 the bundle scan
# refused (nothing to send; see the message) · 5 deadline
set -euo pipefail

# ------------------------------------------------------------------------------------------------ pins
REPO_URL="${PROF4_REPO_URL:-https://github.com/yetanotherco/lambda_vm.git}"
LANE_BRANCH=prof/iprof4-5090
W_BRANCH=whirp1/s6b-box
W_SHA=4082b0df36c81f9002617d4dc4c4826c6f2efc19
S_BRANCH=noepoch/stark
S_SHA=4841cf5a9fa6dc9df33a65972228c175723693f4
ELF_NAME=ethrex_8f826601.elf
ELF_SHA256=8f826601776d4085cbb6fbf0302fe8d8d5d1be7940ac1aaca24899c6244ec80a
BENCH_NAME=ethrex_mainnet_25368371_573004e6.bin
BENCH_SHA256=573004e62e3680a00d3cdbae19dc4897e2ec60d6ec0c1d05d9ef118cb8aef17f
MED_NAME=ethrex_mainnet_25475471_8140dd6d.bin
MED_SHA256=8140dd6dc606759310d1fc4b0655d426ef6eab9467654468d71e41f27704bbd7
RUST_TOOLCHAIN=1.94.0
NCU_SECTIONS="SpeedOfLight MemoryWorkloadAnalysis ComputeWorkloadAnalysis Occupancy LaunchStats"
# The 40 metrics I-PROF2's run collected on this machine (ncu 2025.3.0); the preflight keeps the collectable ones.
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
MODE=full; PREFLIGHT_ONLY=0; SKIP_BUILD=0; KEEP_RAW=0; DRY=0; GPU=0; W="${PROF4_DIR:-$PWD/lambda-vm-prof4}"
usage() { sed -n '2,27p' "${BASH_SOURCE[0]}"; exit 1; }
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) MODE=quick ;;
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --keep-raw) KEEP_RAW=1 ;;
    --dry-run) DRY=1 ;;
    --gpu) GPU="${2:?}"; shift ;;
    --dir) W="${2:?}"; shift ;;
    -h|--help) usage ;;
    *) echo "prof4: unknown argument $1"; usage ;;
  esac
  shift
done
[[ "$GPU" =~ ^[0-9]+$ ]] || usage
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$W"
W="$(cd "$W" && pwd)"
RUN="$W/runs/$STAMP"; SEND="$RUN/send"; BIG="$RUN/big"
mkdir -p "$SEND" "$BIG" "$W/home" "$W/tmp" "$W/cargo-home" "$W/fixtures" "$W/tools" "$W/probe"
LOG="$SEND/prof4.log"
T_START=$(date +%s)
DEADLINE_MIN="${PROF4_DEADLINE_MIN:-170}"
say() { echo "PROF4 $(date -u +%H:%M:%SZ) $*" | tee -a "$LOG"; }
die() { local rc=$1; shift; say "STOP ($rc): $*"; echo "prof4: stopped: $*  (log: $LOG)"; exit "$rc"; }
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
TOOL="$W/tools/prof4_summary.py"; SAMPLER="$W/tools/prof4_sampler.py"

text_of() { "$@" 2>&1 || true; }                       # a command's whole output, whatever its status
first_of() { grep -oE -- "$1" | head -1 || true; }      # the first match of an ERE on stdin, or nothing
gpu_q() { "$NVSMI" -i "$GPU" --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null | head -1 | sed 's/^ *//; s/ *$//'; }
gpu_idle() {  # used MiB below the floor and no compute process on this GPU
  local used apps
  used="$(gpu_q memory.used)"
  apps="$("$NVSMI" -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
  [ "${used:-99999}" -lt "${PROF4_IDLE_MIB:-1024}" ] && [ "${apps:-1}" -eq 0 ]
}
wait_ready() {  # wait_ready LABEL: the GPU idle and ≥ 46 GiB available, up to 5 min
  local t=0 av
  while :; do
    av="$(awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo)"
    if gpu_idle && [ "${av:-0}" -ge "${PROF4_MIN_AVAIL_GIB:-46}" ]; then return 0; fi
    [ "$t" -lt 300 ] || { say "$1: not ready after 5 min (GPU used $(gpu_q memory.used) MiB, $av GiB available)"; return 1; }
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
  for t in git tar timeout awk sed python3 sha256sum; do command -v "$t" > /dev/null || chk FAIL "missing tool: $t"; done
  if python3 -c 'import sqlite3, sys; assert sys.version_info >= (3, 8)' 2>/dev/null; then chk PASS "python3 $(python3 -c 'import sys; print(sys.version.split()[0])') with sqlite3"
  else chk FAIL "python3 >= 3.8 with the sqlite3 module needed"; fi
  for v in NVSMI NVCC NSYS NCU; do [ -n "${!v}" ] && [ -x "${!v}" ] || chk FAIL "missing: $(echo "$v" | tr 'A-Z' 'a-z' | sed 's/nvsmi/nvidia-smi/') (CUDA toolkit / Nsight Systems / Nsight Compute)"; done
  [ "$NFAIL" -eq 0 ] || return 0
  local name cc mem drv used
  name="$(gpu_q name)"; cc="$(gpu_q compute_cap)"; mem="$(gpu_q memory.total)"; drv="$(gpu_q driver_version)"
  [ -n "$name" ] && chk PASS "gpu $GPU: $name · compute capability $cc · $mem MiB · driver $drv" || chk FAIL "gpu $GPU: nvidia-smi sees none"
  [ "$cc" = "12.0" ] || chk WARN "compute capability $cc: the plan was made on an RTX 5090 (12.0)"
  CC_ARCH="sm_${cc//./}"
  used="$(gpu_q memory.used)"
  gpu_idle && chk PASS "idle: gpu $GPU uses $used MiB, no compute process" || chk FAIL "gpu $GPU is busy ($used MiB used or a compute process): close it and re-run"
  case "$drv" in 58[0-9].*) chk WARN "driver $drv: on 580.65 the 10-01 run saw freed device memory not reused (VRAM climbs per group); a run that aborts on the device is retried once with a smaller VRAM budget (16000 MiB, pool released at each sync) and says so" ;; esac
  NVCC_VER="$(text_of "$NVCC" --version | first_of 'release [0-9.]+')"; NSYS_VER="$(text_of "$NSYS" --version | first_of '20[0-9]{2}\.[0-9.]+')"
  NCU_VER="$(text_of "$NCU" --version | first_of '20[0-9]{2}\.[0-9.]+')"
  chk PASS "nvcc $NVCC_VER · nsys $NSYS_VER · ncu $NCU_VER"
  case "$(text_of "$NSYS" profile --help)" in *--gpu-metrics-devices*) GM_FLAG=--gpu-metrics-devices ;; *) GM_FLAG=--gpu-metrics-device ;; esac
  case "$(text_of "$NCU" --help)" in *--kill*) NCU_KILL=1 ;; *) NCU_KILL=0; chk WARN "ncu has no --kill: each run-B pass runs its workload to the end (slower)" ;; esac
  local mt av fr
  mt="$(awk '/^MemTotal:/ {printf "%.1f", $2 / 1048576}' /proc/meminfo)"; av="$(awk '/^MemAvailable:/ {printf "%.1f", $2 / 1048576}' /proc/meminfo)"
  awk -v m="$mt" 'BEGIN { exit !(m >= 56) }' && chk PASS "ram: MemTotal $mt GiB, $av available (the prover sizes its spill and program budget to MemTotal - 10 GiB)" \
    || chk FAIL "ram: MemTotal $mt GiB; the median block needs >= 56 GiB"
  awk -v a="$av" 'BEGIN { exit !(a >= 46) }' || chk FAIL "ram: only $av GiB available; close other programs (>= 46 needed before each run)"
  fr="$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9)"
  [ "${fr:-0}" -ge 80 ] && chk PASS "disk: $fr GiB free in the work directory" || chk FAIL "disk: $fr GiB free in $W; need >= 80 (builds ≈ 15, captures ≈ 10, a possible spill ≈ 30)"
  if text_of env RUSTUP_HOME="$RUSTUP_HOME_REAL" "$CARGO_BIN/rustup" toolchain list | grep "^$RUST_TOOLCHAIN" > /dev/null; then chk PASS "rust: toolchain $RUST_TOOLCHAIN installed"
  else chk FAIL "rust: toolchain $RUST_TOOLCHAIN missing: rustup toolchain install $RUST_TOOLCHAIN"; fi
  # counters, the nsys GPU-metrics flags and the export on a 3-kernel probe, with the exact run flags
  cat > "$W/probe/probe.cu" <<'CU'
#include <cstdio>
__global__ void prof4_probe(float *x, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] = x[i] * 1.0001f + 1.0f; }
int main() { float *x; cudaMalloc(&x, 1 << 24); for (int k = 0; k < 3; k++) prof4_probe<<<16384, 256>>>(x, 1 << 22);
             cudaDeviceSynchronize(); cudaFree(x); printf("probe ok\n"); return 0; }
CU
  if ! env -i "${BUILD_ENV[@]}" timeout 300 "$NVCC" -O2 -arch="$CC_ARCH" -o "$W/probe/probe" "$W/probe/probe.cu" > "$W/probe/nvcc.log" 2>&1; then
    chk FAIL "nvcc could not build the probe for $CC_ARCH: see $W/probe/nvcc.log"; return 0; fi
  NCU_METRICS_OK=""
  if env -i "${RUN_ENV[@]}" timeout 300 "$NCU" --metrics "$NCU_METRICS" -k regex:^prof4_probe$ -c 1 "$W/probe/probe" > "$W/probe/ncu-metrics.log" 2>&1 \
     && grep -q "sm__throughput" "$W/probe/ncu-metrics.log"; then
    NCU_METRICS_OK="$NCU_METRICS"; chk PASS "counters: readable; all $(echo "$NCU_METRICS" | tr ',' '\n' | grep -c .) ncu metrics collectable"
  elif env -i "${RUN_ENV[@]}" timeout 300 "$NCU" --section SpeedOfLight -k regex:^prof4_probe$ -c 1 "$W/probe/probe" > "$W/probe/ncu.log" 2>&1 \
     && grep -q "Compute (SM) Throughput" "$W/probe/ncu.log"; then
    chk WARN "counters: readable, but the metric list failed (see $W/probe/ncu-metrics.log): run B uses the sections only"
  else chk FAIL "counters: ncu could not profile the probe (ERR_NVGPUCTRPERM? see $W/probe/ncu.log)"; return 0; fi
  # run B end to end on the probe: the pass flags (launch 2 of 3, --kill), the import, the parser
  rm -f "$W/probe/ncuprobe.ncu-rep"
  ncu_flags prof4_probe 1 1 "$W/probe/ncuprobe"
  env -i "${RUN_ENV[@]}" timeout 300 "$NCU" "${NCU_FLAGS[@]}" "$W/probe/probe" > "$W/probe/ncu-pass.log" 2>&1 || true
  if [ -f "$W/probe/ncuprobe.ncu-rep" ] \
     && env -i "${RUN_ENV[@]}" timeout 300 "$NCU" --import "$W/probe/ncuprobe.ncu-rep" --page details --csv > "$W/probe/ncuprobe.csv" 2>> "$W/probe/ncu-pass.log" \
     && python3 "$TOOL" ncu --csv "$W/probe/ncuprobe.csv" --pass probe --out "$W/probe/ncuprobe.tsv" >> "$W/probe/ncu-pass.log" 2>&1 \
     && [ "$(awk -F'\t' 'NR == 2 && $8 != "-" { print "ok" }' "$W/probe/ncuprobe.tsv")" = ok ]; then
    chk PASS "ncu: run B's pass flags (launch-skip, --kill $NCU_KILL, base clocks) profile one probe launch; import + parser read its SM %"
  else chk FAIL "ncu: run B's pass flags failed on the probe: see $W/probe/ncu-pass.log"; fi
  # run A end to end on the probe: the exact nsys flags with GPU metrics, the export, the summary tool
  rm -f "$W/probe/nsysprobe.nsys-rep" "$W/probe/nsysprobe.sqlite"
  nsys_flags "$W/probe/nsysprobe"
  if ! env -i "${RUN_ENV[@]}" timeout 300 "${NSYS_FLAGS[@]}" "$W/probe/probe" > "$W/probe/nsys-osrt.log" 2>&1; then
    NSYS_OSRT=0; nsys_flags "$W/probe/nsysprobe"
    chk WARN "nsys: the OS-runtime trace flags failed on the probe (see $W/probe/nsys-osrt.log): run A traces CUDA only"
  fi
  if env -i "${RUN_ENV[@]}" timeout 300 "${NSYS_FLAGS[@]}" "$W/probe/probe" > "$W/probe/nsys.log" 2>&1 \
     && env -i "${RUN_ENV[@]}" timeout 300 "$NSYS" export --type=sqlite --force-overwrite=true --output="$W/probe/nsysprobe.sqlite" "$W/probe/nsysprobe.nsys-rep" >> "$W/probe/nsys.log" 2>&1; then
    : > "$W/probe/empty.log"
    python3 "$TOOL" runa --sqlite "$W/probe/nsysprobe.sqlite" --log "$W/probe/empty.log" --kind W --out "$W/probe/runa" >> "$W/probe/nsys.log" 2>&1 || true
    local nk nm
    nk="$(grep -oE 'kernels [0-9]+' "$W/probe/nsys.log" | tail -1 | grep -oE '[0-9]+' || echo 0)"
    nm="$(grep -oE 'GPU-metric names [0-9]+' "$W/probe/nsys.log" | tail -1 | grep -oE '[0-9]+' || echo 0)"
    [ "$nk" = 3 ] && [ "${nm:-0}" -gt 0 ] && chk PASS "nsys: run A's flags trace the probe's 3 kernels with $nm GPU metrics ($GM_FLAG 2 kHz); export + summary read them" \
      || chk FAIL "nsys: the probe's summary saw $nk kernels and ${nm:-0} GPU metrics (want 3 and > 0): see $W/probe/nsys.log"
  else chk FAIL "nsys: run A's flags failed on the probe: see $W/probe/nsys.log"; fi
  rm -f "$W/probe/nsysprobe.nsys-rep" "$W/probe/nsysprobe.sqlite" "$W/probe/ncuprobe.ncu-rep"
}

# ------------------------------------------------------------------------------------------------ setup and builds
setup_repo() {
  step setup "clone / fetch, the helpers and fixtures"
  local R="$W/repo" rev f
  if [ ! -d "$R/.git" ]; then git clone -q --no-checkout --filter=blob:none "$REPO_URL" "$R" || die 3 "git clone $REPO_URL failed"; fi
  git -C "$R" fetch -q origin "$LANE_BRANCH" "$W_BRANCH" "$S_BRANCH" || die 3 "git fetch of the pins failed"
  for p in "$W_SHA" "$S_SHA"; do git -C "$R" cat-file -e "$p^{commit}" 2>/dev/null || die 3 "pin $p is not on origin"; done
  # the helpers and fixtures come from the commit this script came from: PROF4_REV, else the lane branch's tip, and that
  # commit's prof4.sh must be this very file
  rev="${PROF4_REV:-$(git -C "$R" rev-parse "origin/$LANE_BRANCH")}"
  git -C "$R" cat-file -e "$rev:scripts/profile/prof4.sh" 2>/dev/null || die 3 "no scripts/profile/prof4.sh at $rev"
  [ "$(git -C "$R" show "$rev:scripts/profile/prof4.sh" | sha256sum | cut -c1-64)" = "$(sha256sum "$SCRIPT" | cut -c1-64)" ] \
    || die 3 "this prof4.sh differs from the one at $rev ($LANE_BRANCH): download the script again, or set PROF4_REV to the commit it came from"
  PROF4_REV="$rev"
  for f in prof4_summary.py prof4_sampler.py; do git -C "$R" show "$rev:scripts/profile/$f" > "$W/tools/$f"; done
  for f in "$ELF_NAME:$ELF_SHA256" "$BENCH_NAME:$BENCH_SHA256" "$MED_NAME:$MED_SHA256"; do
    git -C "$R" show "$rev:scripts/profile/fixtures/${f%%:*}" > "$W/fixtures/${f%%:*}"
    [ "$(sha256sum "$W/fixtures/${f%%:*}" | cut -c1-64)" = "${f##*:}" ] || die 3 "fixture ${f%%:*}: sha256 mismatch"
  done
  python3 "$TOOL" selftest > "$SEND/selftest.txt" 2>&1 || die 3 "the summary tool's selftest is RED: see $SEND/selftest.txt"
  say "helpers and fixtures from $rev; the summary tool's selftest is GREEN"
}
setup_worktrees() {
  step worktrees "W $W_SHA · S $S_SHA"
  local R="$W/repo"
  for x in w:"$W_SHA" s:"$S_SHA"; do
    local wt="$W/wt-${x%%:*}" sha="${x#*:}"
    if [ -e "$wt/.git" ]; then git -C "$wt" checkout -q --detach "$sha"; else git -C "$R" worktree add -q --detach "$wt" "$sha"; fi
    [ "$(git -C "$wt" rev-parse HEAD)" = "$sha" ] || die 3 "worktree $wt is not at $sha"
    [ -z "$(git -C "$wt" status --porcelain --untracked-files=no)" ] || die 3 "worktree $wt has local changes"
  done
  say "pins: W $W_SHA ($W_BRANCH) · S $S_SHA ($S_BRANCH)"
}
build() {  # build w|s → BIN_w / BIN_s
  local k="$1" wt="$W/wt-$1" rel
  step "build $k" "cargo test --release -p lambda-vm-prover --features cuda --lib --no-run"
  if [ "$SKIP_BUILD" = 0 ]; then
    (cd "$wt" && bounded 5400 "$SEND/build-$k.log" env -i "${BUILD_ENV[@]}" "CARGO_TARGET_DIR=$W/target-$k" "CUDARC_NVCC_ARCH=$CC_ARCH" \
       cargo test --release -p lambda-vm-prover --features cuda --lib --no-run) || { tail -30 "$SEND/build-$k.log"; die 3 "build $k failed: $SEND/build-$k.log"; }
  fi
  # Under --skip-build this run directory has no build log: the binary is the newest test executable in the target dir.
  # Each lookup tolerates a missing file (under set -e + pipefail a failing sed or find would end the script silently).
  rel=""
  [ ! -f "$SEND/build-$k.log" ] || rel="$({ sed -n 's/.*Executable unittests src\/lib.rs (\(.*lambda_vm_prover-[0-9a-f]*\)).*/\1/p' "$SEND/build-$k.log" || true; } | tail -1)"
  if [ -z "$rel" ]; then
    local f
    for f in "$W/target-$k"/release/deps/lambda_vm_prover-*; do
      [ -f "$f" ] && [ -x "$f" ] || continue
      if [ -z "$rel" ] || [ "$f" -nt "$rel" ]; then rel="$f"; fi
    done
  fi
  case "$rel" in /*|"") ;; *) rel="$wt/$rel" ;; esac
  [ -n "$rel" ] && [ -f "$rel" ] && [ -x "$rel" ] || die 3 "no test binary for $k$( [ "$SKIP_BUILD" = 0 ] || echo " in $W/target-$k (--skip-build needs an earlier build in this work directory)")"
  [ -z "$(git -C "$wt" status --porcelain --untracked-files=no)" ] || die 3 "worktree $wt changed during the build"
  printf -v "BIN_$k" '%s' "$rel"
  say "binary $k: $(basename "$rel") sha256 $(sha256sum "$rel" | cut -c1-16)…"
}

# ------------------------------------------------------------------------------------------------ the workloads
posture() {  # posture w|s P|R bench|median → the knob words (I-COMP's postures, as the ULTRA ladder 124/125)
  local fix="$W/fixtures/$BENCH_NAME"; [ "$3" = median ] && fix="$W/fixtures/$MED_NAME"
  if [ "$1" = w ]; then
    local b=rpx; [ "$2" = P ] && b=p1w
    echo "LAMBDA_VM_MAX_ROWS_LOG2=21 LFM_WHIR_RETENTION=1 LAMBDA_VM_WHIR_HASH=rpx LFM_PRECOMPUTED_TREE_CACHE_CAP=64 LFM_EXEC_PARALLEL=1 LAMBDA_VM_BASE_SPLIT=1 LAMBDA_VM_GRIND_SCAN_FACTOR=8 LAMBDA_VM_GRIND_GRID=1024 _RJEM_MALLOC_CONF=dirty_decay_ms:-1,muzzy_decay_ms:-1 BLOCK_WHIR_ELF=$W/fixtures/$ELF_NAME BLOCK_WHIR_INPUT=$fix TABLE_PARALLELISM=4 BLOCK_WHIR_BASE=$b"
  else
    local b="NOEPOCH_BASE=rpx NOEPOCH_P1_CAP=1"; [ "$2" = P ] && b="NOEPOCH_BASE=p1 NOEPOCH_P1_CAP=01"
    echo "_RJEM_MALLOC_CONF=dirty_decay_ms:-1,muzzy_decay_ms:-1 LAMBDA_VM_VRAM_BUDGET_MB=24000 LAMBDA_VM_MAX_ROWS_LOG2=21 LFM_PROVE_SPLIT=1 LAMBDA_VM_BASE_SPLIT=1 LFM_EXEC_PARALLEL=1 LFM_PRECOMPUTED_TREE_CACHE_CAP=64 LAMBDA_VM_GATE_PACKING=1 LFM_TREE_SIBLINGS_L0=8 LFM_TREE_SIBLINGS=4 NOEPOCH_ELF=$W/fixtures/$ELF_NAME NOEPOCH_INPUT=$fix TABLE_PARALLELISM=8 $b"
  fi
}
SAFE_VRAM="LAMBDA_VM_VRAM_BUDGET_MB=16000 LAMBDA_VM_MEMPOOL_RELEASE_MB=0"
NSYS_OSRT=1
nsys_flags() {  # nsys_flags OUT-PREFIX → NSYS_FLAGS: run A's exact flags (the preflight probe uses these too)
  if [ "$NSYS_OSRT" = 1 ]; then NSYS_FLAGS=("$NSYS" profile -t cuda,osrt --osrt-threshold=10000)
  else NSYS_FLAGS=("$NSYS" profile -t cuda); fi
  NSYS_FLAGS+=(--sample=none --cpuctxsw=none --cuda-memory-usage=false "$GM_FLAG=$GPU" --gpu-metrics-frequency=2000
               --stats=false --force-overwrite=true -o "$1")
}
ncu_flags() {  # ncu_flags KERNEL SKIP COUNT REPORT → NCU_FLAGS: run B's exact flags (the preflight probe uses these too)
  NCU_FLAGS=(--target-processes all --kernel-name-base function --kernel-name "regex:^${1}\$" --launch-skip "$2" --launch-count "$3")
  [ "$NCU_KILL" = 0 ] || NCU_FLAGS+=(--kill yes)
  for s in $NCU_SECTIONS; do NCU_FLAGS+=(--section "$s"); done
  [ -z "$NCU_METRICS_OK" ] || NCU_FLAGS+=(--metrics "$NCU_METRICS_OK")
  NCU_FLAGS+=(--clock-control base --export "$4" --force-overwrite)
}
test_of() { [ "$1" = w ] && echo lfm::whir_block_tests::the_whir_block_tree_on_a_real_block || echo lfm::block_tree_tests::the_block_tree_composes_to_a_top_node; }
device_abort() { grep -qE '\[gpu\] ABORT|CUDA_ERROR_OUT_OF_MEMORY|out of memory|DriverError' "$1" 2>/dev/null; }


# run_one NAME w|s P|R bench|median nsys|ncu:<args file> [extra knob words] → RC, RD (its raw directory)
run_one() {
  local name="$1" k="$2" arm="$3" blk="$4" prof="$5" extra="${6:-}" bin words to spid upid t0
  RD="$BIG/$name"; mkdir -p "$RD"
  bin="BIN_$k"; bin="${!bin}"
  words="$(posture "$k" "$arm" "$blk")${extra:+ $extra}"
  to=900; [ "$blk" = median ] && to=1800
  local pre=()
  case "$prof" in
    nsys) nsys_flags "$RD/cap"; pre=("${NSYS_FLAGS[@]}") ;;
    ncu:*) mapfile -t pre < "${prof#ncu:}"; pre=("$NCU" "${pre[@]}"); to=600 ;;
  esac
  printf '%s\n' "$words" | sed "s#$W#<W>#g" > "$RD/knobs.txt"
  {
    echo "set -o pipefail"
    echo "cd $(printf '%q' "$W/wt-$k/prover")"
    # shellcheck disable=SC2086
    echo "env -i $(printf '%q ' "${RUN_ENV[@]}")$words $(printf '%q ' "${pre[@]}" "$bin" --ignored --exact --nocapture --test-threads=1 "$(test_of "$k")")2>&1 | python3 -u $(printf '%q' "$TOOL") stamp > $(printf '%q' "$RD/out.log")"
  } > "$RD/cmd.sh"
  if [ "$DRY" = 1 ]; then say "DRY $name: $(tail -1 "$RD/cmd.sh" | sed "s#$W#<W>#g" | cut -c1-400)"; bash -n "$RD/cmd.sh"; RC=0; return 0; fi
  wait_ready "$name" || { RC=97; return 0; }
  date +%z > "$RD/tz.txt"
  "$NVSMI" -i "$GPU" --query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,power.draw,temperature.gpu --format=csv,noheader,nounits -lms 200 > "$RD/smi.csv" 2>/dev/null &
  upid=$!
  python3 "$SAMPLER" "$bin" "$RD/host.tsv" "$RD/watchdog.txt" 0.5 1536 &
  spid=$!
  t0=$(date +%s)
  set +e
  ABORT_LOG="$RD/out.log" bounded "$to" "$RD/bounded.out" bash "$RD/cmd.sh"
  RC=$?
  set -e
  kill "$upid" "$spid" 2>/dev/null || true; wait "$upid" "$spid" 2>/dev/null || true
  say "$name: rc $RC · $(( $(date +%s) - t0 )) s · $(grep -hoE 'W3 RECURSION: .*whole block [0-9.]+s|NO-EPOCH BLOCK: base .*whole [0-9.]+s' "$RD/out.log" 2>/dev/null | head -1 | cut -c1-150)$( [ -s "$RD/watchdog.txt" ] && echo " · MEMORY WATCHDOG: $(cat "$RD/watchdog.txt")")"
}
verified() {  # verified w|s LOG: the run proved and its harness verified the block
  grep -q "test result: ok. 1 passed" "$2" 2>/dev/null || return 1
  if [ "$1" = w ]; then grep -q "W3 TREE VERIFY: ACCEPTED" "$2"; else grep -q "BLOCK VERIFIER: verify_block_tree" "$2"; fi
}

# runa NAME w|s P|R bench|median [PREFIX MAX]: one capture (and, with PREFIX MAX, the run-B plan from it), retried once on a device abort with the small VRAM budget
RUNA_OK=0; RUNA_N=0
runa() {
  local name="$1" k="$2" arm="$3" blk="$4" note=""
  RUNA_N=$((RUNA_N + 1))
  [ "$(left_min)" -gt 15 ] || { say "$name: skipped (deadline: $(left_min) min left)"; return 0; }
  step "run A $name" "$k $arm $blk under nsys + GPU metrics"
  run_one "$name" "$k" "$arm" "$blk" nsys
  [ "$DRY" = 1 ] && return 0
  if [ "$RC" -ne 0 ] && [ ! -s "$RD/watchdog.txt" ] && device_abort "$RD/out.log"; then
    say "$name: the device aborted (driver-side memory?): one retry with $SAFE_VRAM"
    mv "$RD" "$RD.aborted"; rm -f "$RD.aborted/cap.nsys-rep"
    local safe="$SAFE_VRAM"; [ "$k" = w ] || safe="$SAFE_VRAM TABLE_PARALLELISM=4"
    run_one "$name" "$k" "$arm" "$blk" nsys "$safe"; note="retried with $safe"
  fi
  local out="$SEND/runa/$name"; mkdir -p "$out"
  cp "$RD/knobs.txt" "$RD/tz.txt" "$RD/smi.csv" "$RD/host.tsv" "$out/" 2>/dev/null || true
  [ ! -s "$RD/watchdog.txt" ] || cp "$RD/watchdog.txt" "$out/"
  sed "s#$W#<W>#g; s#$HOME#<HOME>#g" "$RD/out.log" > "$out/out.log" 2>/dev/null || true
  local v=no; verified "$k" "$RD/out.log" && v=yes
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$k" "$arm" "$blk" "$RC" "$v" "$note" >> "$SEND/runa/runs.tsv"
  if [ ! -f "$RD/cap.nsys-rep" ]; then say "$name: no report written"; return 0; fi
  set +e
  bounded 3600 "$RD/export.log" env -i "${RUN_ENV[@]}" "$NSYS" export --type=sqlite --force-overwrite=true --output="$RD/cap.sqlite" "$RD/cap.nsys-rep"
  local erc=$?
  [ "$erc" -ne 0 ] || python3 "$TOOL" runa --sqlite "$RD/cap.sqlite" --log "$RD/out.log" --kind "$(echo "$k" | tr ws WS)" --out "$out" \
      --host "$RD/host.tsv" --smi "$RD/smi.csv" --tz "$(cat "$RD/tz.txt")" 2>&1 | tee -a "$LOG"
  [ "$erc" -ne 0 ] || bounded 1800 "$RD/stats.log" env -i "${RUN_ENV[@]}" "$NSYS" stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,osrt_sum \
      --format csv --force-overwrite=true --output "$out/nsys" "$RD/cap.sqlite"
  set -e
  [ "$erc" -eq 0 ] && [ -f "$out/stages.tsv" ] && [ "$v" = yes ] && RUNA_OK=$((RUNA_OK + 1))
  if [ -n "${6:-}" ] && [ -f "$RD/cap.sqlite" ]; then
    python3 "$TOOL" plan --sqlite "$RD/cap.sqlite" --log "$RD/out.log" --kind "$(echo "$k" | tr ws WS)" --out "$SEND/plan-$k.tsv" \
      --prefix "$5" --max "$6" 2>&1 | tee -a "$LOG" || say "$name: the plan failed"
  fi
  [ "$KEEP_RAW" = 1 ] || rm -f "$RD/cap.nsys-rep" "$RD/cap.sqlite"
}

# run B: one ncu pass per plan row, on the bench block, the launch picked by its order among that kernel's launches
RUNB_OK=0; RUNB_N=0
runb() {
  local k="$1" plan="$SEND/plan-$1.tsv" pass kern skip cnt grid block rest
  [ -f "$plan" ] || { say "run B $k: no plan (its bench capture failed)"; return 0; }
  mkdir -p "$SEND/ncu"
  while IFS=$'\t' read -r pass kern skip cnt grid block rest; do
    [ "$pass" = pass ] && continue
    RUNB_N=$((RUNB_N + 1))
    [ "$(left_min)" -gt 6 ] || { say "ncu $pass: skipped (deadline)"; continue; }
    step "run B $pass" "$kern skip $skip ($grid / $block)"
    local args="$BIG/ncu-$pass.args"
    ncu_flags "$kern" "$skip" "$cnt" "$BIG/ncu-$pass"
    printf '%s\n' "${NCU_FLAGS[@]}" > "$args"
    run_one "ncu-$pass" "$k" P bench "ncu:$args"
    [ "$DRY" = 1 ] && continue
    if [ -f "$BIG/ncu-$pass.ncu-rep" ]; then
      set +e
      env -i "${RUN_ENV[@]}" timeout 900 "$NCU" --import "$BIG/ncu-$pass.ncu-rep" --page details --csv > "$BIG/ncu-$pass.csv" 2> "$BIG/ncu-$pass.import.log"
      python3 "$TOOL" ncu --csv "$BIG/ncu-$pass.csv" --pass "$pass" --out "$SEND/ncu/$pass.launches.tsv" 2>&1 | tee -a "$LOG"
      set -e
      sed "s#$W#<W>#g; s#$HOME#<HOME>#g" "$BIG/ncu-$pass.csv" > "$SEND/ncu/$pass.details.csv"
      [ -s "$SEND/ncu/$pass.launches.tsv" ] && [ "$(wc -l < "$SEND/ncu/$pass.launches.tsv")" -gt 1 ] && RUNB_OK=$((RUNB_OK + 1))
      [ "$KEEP_RAW" = 1 ] || rm -f "$BIG/ncu-$pass.ncu-rep"
    else
      say "ncu $pass: no report (rc $RC): the launch was not reached? $(grep -m2 -E '==PROF==|==ERROR==|==WARNING==' "$BIG/ncu-$pass/out.log" 2>/dev/null | tr '\n' ' ' | cut -c1-200)"
    fi
  done < "$plan"
}

# ------------------------------------------------------------------------------------------------ facts, bundle
facts() {
  {
    echo "prof4: script sha256 $(sha256sum "$SCRIPT" | cut -c1-16)… from ${PROF4_REV:-?} · mode $MODE · started $STAMP · deadline $DEADLINE_MIN min"
    echo "pins: W $W_SHA ($W_BRANCH, arms BLOCK_WHIR_BASE=p1w|rpx) · S $S_SHA ($S_BRANCH, NOEPOCH_BASE=p1 cap 1)"
    [ -z "$NVSMI" ] || echo "gpu $GPU: $(gpu_q name) · cc $(gpu_q compute_cap) · $(gpu_q memory.total) MiB · driver $(gpu_q driver_version) · power limit $(gpu_q power.limit) W · max SM $(gpu_q clocks.max.sm) MHz"
    echo "tools: nvcc ${NVCC_VER:-?} · nsys ${NSYS_VER:-?} ($GM_FLAG 2 kHz, osrt $NSYS_OSRT) · ncu ${NCU_VER:-?} (--clock-control base, kill $NCU_KILL)"
    echo "host: $(awk -F': ' '/^model name/ {print $2; exit}' /proc/cpuinfo 2>/dev/null) · $(nproc) threads · MemTotal $(awk '/^MemTotal:/ {printf "%.1f", $2 / 1048576}' /proc/meminfo 2>/dev/null) GiB · $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-?}")"
    echo "binaries: w $(sha256sum "${BIN_w:-/dev/null}" 2>/dev/null | cut -c1-16)… s $(sha256sum "${BIN_s:-/dev/null}" 2>/dev/null | cut -c1-16)… (cargo test --release -p lambda-vm-prover --features cuda --lib, CUDARC_NVCC_ARCH=${CC_ARCH:-?})"
    echo "fixtures: $ELF_NAME, $BENCH_NAME, $MED_NAME (sha256-checked)"
    echo "run A: $RUNA_OK of $RUNA_N captures verified and summarised · run B: $RUNB_OK of $RUNB_N ncu passes summarised · $(( ($(date +%s) - T_START) / 60 )) min"
  } > "$SEND/facts.txt"
}
bundle() {
  step bundle "scrub, scan, tar"
  local f bad=0 out="$W/prof4-send-$STAMP.tar.gz"
  python3 "$TOOL" report --dir "$SEND" | tee -a "$LOG"
  # scrub: the work directory and HOME, in every text file (the profiled runs printed paths)
  while IFS= read -r -d '' f; do sed -i "s#$W#<W>#g; s#$HOME#<HOME>#g" "$f"; done < <(find "$SEND" -type f -print0)
  : > "$BIG/bundle-scan.txt"
  while IFS= read -r -d '' f; do
    case "$f" in *.nsys-rep|*.qdrep|*.qdstrm|*.ncu-rep|*.sqlite|*.db|*.arrow) echo "forbidden file type: ${f#"$SEND"/}" >> "$BIG/bundle-scan.txt"; continue ;; esac
    [ "$(head -c 15 "$f" | tr -d '\0')" = "SQLite format 3" ] && echo "forbidden file type: ${f#"$SEND"/} (sqlite)" >> "$BIG/bundle-scan.txt"
    { [ ! -s "$f" ] || LC_ALL=C grep -qI . "$f"; } || echo "forbidden file type: ${f#"$SEND"/} (not text)" >> "$BIG/bundle-scan.txt"
  done < <(find "$SEND" -type f -print0)
  { LC_ALL=C grep -rnoE -- "$SECRET_MARKERS" "$SEND" 2>/dev/null || true; } | sed "s#^$SEND/##" >> "$BIG/bundle-scan.txt"
  [ ! -s "$BIG/bundle-scan.txt" ] || bad=1
  if [ "$bad" = 1 ]; then
    say "BUNDLE REFUSED: $(wc -l < "$BIG/bundle-scan.txt") finding(s); see $BIG/bundle-scan.txt (do not send it). Remove the lines it names from $SEND, then: tar -czf $out -C $RUN send"
    return 4
  fi
  tar -czf "$out" -C "$RUN" send
  say "bundle: $out ($(du -h "$out" | cut -f1), $(find "$SEND" -type f | wc -l) plain-text files, scan clean)"
  echo
  echo "=== DONE. Send back this one file (text only): $out"
}

# ------------------------------------------------------------------------------------------------ main
say "prof4 $MODE · work dir $W · gpu $GPU · run $STAMP"
GM_FLAG=--gpu-metrics-devices; NCU_KILL=1; NCU_METRICS_OK="$NCU_METRICS"; CC_ARCH=sm_120
setup_repo
if [ "$DRY" = 1 ]; then say "DRY RUN: no GPU work; the preflight's GPU checks are skipped"
else
  preflight
  say "PREFLIGHT: $( [ "$NFAIL" -eq 0 ] && echo PASS || echo FAIL ) ($NFAIL failure(s), $NWARN warning(s)); $SEND/preflight.txt"
  [ "$NFAIL" -eq 0 ] || exit 2
fi
[ "$PREFLIGHT_ONLY" = 0 ] || exit 0
if [ "$MODE" = quick ]; then est="≈ 40–55 min (builds ≈ 10–20, run A ≈ 10, run B ≈ 15)"; else est="≈ 75–110 min (builds ≈ 10–20, run A ≈ 25–35, run B ≈ 35–45)"; fi
say "estimate: $est; deadline $DEADLINE_MIN min. Leave the GPU and most of the RAM to it."
setup_worktrees
if [ "$DRY" = 1 ]; then BIN_w="$W/target-w/release/deps/lambda_vm_prover-0000"; BIN_s="$W/target-s/release/deps/lambda_vm_prover-0000"
else build w; [ "$MODE" = quick ] || build s; fi
mkdir -p "$SEND/runa"; printf 'run\tpin\tarm\tblock\trc\tverified\tnote\n' > "$SEND/runa/runs.tsv"
runa a1-w-p1-bench w P bench w 10
if [ "$MODE" = full ]; then
  runa a2-s-p1-bench s P bench s 8
  runa a3-w-p1-median w P median
  runa a4-w-rpx-median w R median
  runa a5-s-p1-median s P median
else
  runa a3-w-p1-median w P median
  [ ! -f "$SEND/plan-w.tsv" ] || { head -6 "$SEND/plan-w.tsv" > "$SEND/plan-w.tsv.q" && mv "$SEND/plan-w.tsv.q" "$SEND/plan-w.tsv"; }
fi
if [ "$DRY" = 1 ]; then
  for k in w s; do printf 'pass\tkernel\tlaunch_skip\tlaunch_count\tgrid\tblock\n%s01\tp1w16_zleaves_base_coset_v2\t3\t1\t65536x1x1\t128x1x1\n%s02\tgkr_round_gruen\t1200\t1\t2048x1x1\t256x1x1\n' "$k" "$k" > "$SEND/plan-$k.tsv"; done
fi
runb w
[ "$MODE" = quick ] || runb s
facts
set +e
bundle; brc=$?
set -e
[ "$brc" -eq 0 ] || exit 4
[ "$(left_min)" -gt 0 ] || exit 5
exit 0
