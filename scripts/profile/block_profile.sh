#!/usr/bin/env bash
# block_profile.sh — Nsight profile of the WHIR block run (block 25368371) on ONE GPU.
#
# WHAT IT ANSWERS
#   Run A (Nsight Systems): where the block run's wall goes on the card: GPU busy vs idle
#     per stage (base; the base's commit / prove / global; level 0; interior; root), per
#     5 s, and per kernel; with counters also SM active %, SM issue % and DRAM bandwidth %
#     per 1 s and per stage (nsys GPU metrics).
#   Run B (Nsight Compute, counters only): why the kernels that carry the time run as fast as
#     they do, on the launch shapes the trace analysis picked (thoughts/zf/prof/ANALYSIS-1GPU.md
#     §7): speed of light, stall reasons, occupancy and its limiter, issue and instruction
#     counts, pipes, local memory, coalescing, L2 hit rate; plus the grind paired with its
#     counted twin. Also cuobjdump -res-usage of every cubin.
#
# THE RUN IS THE RECORD'S
#   Test  lfm::per_table_aggregator_tests::the_whir_production_tree_composes_to_a_root
#   Build cargo test --release -p lambda-vm-prover --features cuda,nvtx --lib, with
#         LAMBDA_VM_NVCC_LINEINFO=1: the record's test, plus NVTX ranges (named phases in the
#         trace; the feature also turns on the instruments spans, host-side bookkeeping) and
#         SASS-to-source line tables in the cubins (codegen unchanged), both per the analysis.
#         The binary runs directly, from prover/ as cargo would, so no cargo process sits in
#         the trace. The rpx_grind_counted test is built the same way for the grind pairing.
#   Env   A-tree-whir.v10.sh's, as whir_tree17.sh launched it (ROOT_OPTION=A,
#         EXPECT_RETENTION=yes, SIBLINGS=4, VRAM_BUDGET_MB=query): see record_env below.
#         The process gets ONLY that plus a few system variables (env -i): no LAMBDA_VM_ZF_*,
#         no stray knob from this shell, and no token from it in the reports either (nsys
#         and ncu store the target's environment verbatim).
#   Input scripts/profile/fixtures/: the record's guest ELF and block, verified by sha256.
#
# USAGE (from anywhere inside a checkout of whir/profile-rpx)
#   bash scripts/profile/block_profile.sh --preflight-only   # the checks alone, ~1 min
#   bash scripts/profile/block_profile.sh                    # build + run A + run B
#   bash scripts/profile/block_profile.sh --no-counters      # build + run A, no GPU metrics, no run B
#
# OPTIONS
#   --no-counters     no GPU metrics in run A and no run B (a box whose counters are locked)
#   --skip-build      reuse this checkout's last build (same HEAD): no make, no cargo
#   --skip-nsys       no run A                --skip-ncu   no run B
#   --out DIR         output directory (default scripts/profile/out/block-<UTC>); never reused
#   --gpu N           GPU index as nvidia-smi numbers it (default 0)
#   --allow-busy-gpu  run although the card is not idle (the numbers then describe a shared card)
#   --no-cpu-sampling no nsys CPU sampling. It is otherwise on where the box allows it, and a box
#                     that refuses it (a container without perf events) runs without: never a failure
#   --preflight-only  stop after the checks
#
# ENV KNOBS (all optional)
#   SYSROOT_DIR        guest C sysroot (default $HOME/.lambda-vm-sysroot; provisioned when missing)
#   GPU_METRICS_FREQ   nsys GPU-metrics sampling rate in Hz (default 2000)
#   NCU_PLAN           run B's plan, a TSV in lib/ncu_plan.py's format (default: default_plan below)
#   NCU_PASSES         run only these passes of the plan, e.g. "grind merkle_wide"
#   NCU_SECTIONS       ncu section identifiers (default: the eight below)
#   NCU_METRICS        explicit ncu metrics (default: lib/common.sh PF_NCU_METRICS, the analysis's
#                      list), narrowed by the preflight to what this ncu and GPU can collect
#   BUILD_TIMEOUT RUN_A_TIMEOUT EXPORT_TIMEOUT NCU_PASS_TIMEOUT   step bounds, seconds
#   IDLE_MIB           the card is idle below this many MiB in use with no compute process (500)
#   NSYS NCU           explicit tool paths
#
# OUTPUT
#   OUT/small/  CSVs, text summaries, logs and an allowlist of run facts, a few MB. The only part
#               that leaves the box, and only after the final bundle scan passes (plain text, no
#               credential marker, no Nsight report or database); otherwise no tarball is written.
#   OUT/big/    blockA.nsys-rep, blockA.sqlite, ncu/*.ncu-rep, raw exports, full build logs. These
#               EMBED THE MACHINE'S ENVIRONMENT: never commit, share or upload them.
#   The last lines printed say exactly what to send back; scripts/profile/README.md has more.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
. "$HERE/lib/common.sh"
REPO="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null)" || pf_die "not inside a git checkout: $HERE"

TEST=lfm::per_table_aggregator_tests::the_whir_production_tree_composes_to_a_root
ELF="$HERE/fixtures/ethrex_8f826601.elf"
ELF_SHA=8f826601776d4085cbb6fbf0302fe8d8d5d1be7940ac1aaca24899c6244ec80a
INPUT="$HERE/fixtures/ethrex_mainnet_25368371_573004e6.bin"
INPUT_SHA=573004e62e3680a00d3cdbae19dc4897e2ec60d6ec0c1d05d9ef118cb8aef17f
EPOCH_LOG2=21
MIN_CUDA=12.8     # cudarc is pinned to the 12.8 driver-API symbol set; sm_120 needs nvcc 12.8
MIN_NSIGHT=2025.1 # Blackwell-capable nsys / ncu
MIN_RAM_GIB=48
RECORD_VRAM_MIB=30000 # the record's card: RTX 5090, 32 GB (31.4 GiB usable)

# Run B's default plan (lib/ncu_plan.py's format): one ncu pass per row, the window
# [skip, skip+count) of that kernel's OWN launches (ncu counts only launches matching -k), and
# the exact grid/block shapes the window holds. Derived from the lead's wt90 nsys trace
# (cdf0238f1, prover library = 169b66831's, RTX 5090) with `ncu_plan.py derive`, for the shapes
# the trace analysis asks for (ANALYSIS-1GPU.md §7). ncu has no grid-size filter, so run B
# re-anchors each window on this box's own run A trace (the nearest index with exactly these
# shapes) and verifies after each pass that the launches it profiled have them.
default_plan() {
  cat <<'PLAN'
pass	kernel	skip	count	t_ref_s	shapes	note	target
grind	rpx_grind_search	16	5	4.7	1024,1,1/128,1,1;1024,1,1/128,1,1;1024,1,1/128,1,1;1024,1,1/128,1,1;1024,1,1/128,1,1	5 grinds from the base: why 4.46 ns per permutation against 2.77 in the leaf kernel (with grind_pair)	block
coset	rpx_leaves_base_coset	1	2	3.2	16384,1,1/128,1,1;16384,1,1/128,1,1	the permutation's full-card ceiling	block
merkle_narrow	rpx_merkle_level	2	9	1.6	512,1,1/128,1,1;256,1,1/128,1,1;128,1,1/128,1,1;64,1,1/128,1,1;32,1,1/128,1,1;16,1,1/128,1,1;8,1,1/128,1,1;4,1,1/128,1,1;2,1,1/128,1,1	one tree's narrow levels, 512 down to 2: latency	block
merkle_wide	rpx_merkle_level	12493	2	71.5	16384,1,1/128,1,1;8192,1,1/128,1,1	the two widest levels of one wrap tree: throughput	block
tail	rpx_merkle_tail	1	3	3.4	1,1,1/128,1,1;1,1,1/128,1,1;1,1,1/128,1,1	per-level latency; barrier and local-memory stalls	block
sumcheck_2p21	sumcheck_round_ext3	340	18	3.9	4096,1,1/256,1,1;4096,1,1/256,1,1;4096,1,1/256,1,1;4096,1,1/256,1,1;4096,1,1/256,1,1;2048,2,1/256,1,1;1024,3,1/256,1,1;512,3,1/256,1,1;512,3,1/128,1,1;512,3,1/64,1,1;512,3,1/32,1,1;256,3,1/32,1,1;128,3,1/32,1,1;64,3,1/32,1,1;32,3,1/32,1,1;16,3,1/32,1,1;1181,1,1/256,1,1;1181,1,1/256,1,1	one 2^21 sumcheck's rounds from 4096x256 down, then the two 1181x256	block
sumcheck_slow	sumcheck_round_ext3	6929	3	15.3	253,1,1/32,1,1;253,1,1/32,1,1;253,1,1/32,1,1	the slow 253x32 launches: slot-buffer traffic against capped occupancy	block
ntt_tile	ntt_dit_tile	4	4	3.1	8,16384,1/32,32,1;256,512,1/32,32,1;8192,16,1/32,32,1;262144,1,1/32,16,1	one base LDE's four tile launches: base LDE throughput	block
rowpair_range	rpx_leaves_base_row_major_row_pair_range	6	2	70.8	8192,1,1/128,1,1;8192,1,1/128,1,1	row-major leaf reads: coalescing	block
ntt_level	ntt_dit_level_row_major	59	3	69.6	1,22796,1/11,23,1;1,65535,1/11,23,1;1,65535,1/11,23,1	memory throughput against peak	block
constraint	constraint_composition_kernel	0	2	71.1	256,1,1/256,1,1;256,1,1/256,1,1	occupancy	block
grind_pair	rpx_grind_search(_counted)?	0	4	10	1024,1,1/128,1,1;1024,1,1/128,1,1;1024,1,1/128,1,1;1024,1,1/128,1,1	the rpx_grind_counted test: warm-up then seed 0 at poll period 1, each the shipped kernel then its counted twin; the warm-up line prints the nonce	grind_counted
PLAN
}
GRIND_TEST=what_lowering_the_grind_poll_rate_does_to_the_overrun
FEATURES="cuda,nvtx"

usage() { sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

COUNTERS=1 SKIP_BUILD=0 SKIP_NSYS=0 SKIP_NCU=0 ALLOW_BUSY=0 PREFLIGHT_ONLY=0 CPU_SAMPLING=1 GPU=0 OUT=""
NO_SAMPLING="--sample=none --cpuctxsw=none"
parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --no-counters) COUNTERS=0 ;;
      --skip-build) SKIP_BUILD=1 ;;
      --skip-nsys) SKIP_NSYS=1 ;;
      --skip-ncu) SKIP_NCU=1 ;;
      --allow-busy-gpu) ALLOW_BUSY=1 ;;
      --no-cpu-sampling) CPU_SAMPLING=0 ;;
      --preflight-only) PREFLIGHT_ONLY=1 ;;
      --gpu) GPU="${2:?--gpu needs an index}"; shift ;;
      --out) OUT="${2:?--out needs a directory}"; shift ;;
      -h|--help) usage; exit 0 ;;
      *) echo "unknown option: $1 (see --help)" >&2; exit 64 ;;
    esac
    shift
  done
  case "$GPU" in ''|*[!0-9]*) echo "--gpu takes a number, got '$GPU'" >&2; exit 64 ;; esac
  if [ "$COUNTERS" = 0 ]; then SKIP_NCU=1; fi
}

knob_defaults() {
  SYSROOT_DIR="${SYSROOT_DIR:-$HOME/.lambda-vm-sysroot}"
  GPU_METRICS_FREQ="${GPU_METRICS_FREQ:-2000}"
  NCU_METRICS_OK="${NCU_METRICS:-$PF_NCU_METRICS}"
  NCU_SECTIONS="${NCU_SECTIONS:-SpeedOfLight WarpStateStats Occupancy SchedulerStats InstructionStats ComputeWorkloadAnalysis MemoryWorkloadAnalysis LaunchStats}"
  BUILD_TIMEOUT="${BUILD_TIMEOUT:-7200}"
  RUN_A_TIMEOUT="${RUN_A_TIMEOUT:-3600}"
  EXPORT_TIMEOUT="${EXPORT_TIMEOUT:-3600}"
  NCU_PASS_TIMEOUT="${NCU_PASS_TIMEOUT:-2700}"
  IDLE_MIB="${IDLE_MIB:-500}"
  # One GPU, the same one for nvidia-smi and for CUDA: CUDA's default order is fastest-first.
  PF_EXTRA_ENV=("CUDA_DEVICE_ORDER=PCI_BUS_ID" "CUDA_VISIBLE_DEVICES=$GPU")
}

setup_out() { # the output directory, never reused; everything printed from here on is also logged
  OUT="${OUT:-$HERE/out/block-$(date -u +%Y%m%dT%H%M%SZ)}"
  if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    echo "REFUSING: $OUT exists and is not empty; an output directory is never reused" >&2; exit 64
  fi
  mkdir -p "$OUT/small" "$OUT/big/tmp"
  OUT="$(cd "$OUT" && pwd)"
  SMALL="$OUT/small" BIG="$OUT/big" TMP="$OUT/big/tmp"
  exec > >(tee -a "$SMALL/driver.log") 2>&1
  printf 'step\tstart_utc\tseconds\trc\n' > "$SMALL/steps.tsv"
}

SAMPLER=""
cleanup() {
  if [ -n "$SAMPLER" ]; then kill "$SAMPLER" 2>/dev/null || true; fi
}

step_begin() { STEP_NAME="$1"; STEP_T0="$(date +%s)"; pf_log "== $1"; }
step_end() {
  local rc="${1:-0}" dt
  dt=$(( $(date +%s) - STEP_T0 ))
  printf '%s\t%s\t%s\t%s\n' "$STEP_NAME" "$(date -u -d "@$STEP_T0" +%FT%TZ 2>/dev/null || echo "$STEP_T0")" "$dt" "$rc" >> "$SMALL/steps.tsv"
  pf_log "== done (rc=$rc, ${dt} s): $STEP_NAME"
}

# ======================================================================================
# PREFLIGHT: every check runs and reports; the verdict comes at the end.
# ======================================================================================
PF_FAILS=0 PF_WARNS=0
chk() { # chk PASS|WARN|FAIL|INFO "what: detail"
  printf '%-4s %s\n' "$1" "$2" | tee -a "$SMALL/preflight.txt"
  case "$1" in FAIL) PF_FAILS=$((PF_FAILS + 1)) ;; WARN) PF_WARNS=$((PF_WARNS + 1)) ;; esac
}
num_ge() { awk -v a="$1" -v b="$2" 'BEGIN { exit !((a + 0) >= (b + 0)) }'; }
cgroup_limit_gib() { # this process's cgroup memory limit in GiB, when one is set
  local v=""
  if [ -r /sys/fs/cgroup/memory.max ]; then v="$(cat /sys/fs/cgroup/memory.max 2>/dev/null || true)"
  elif [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then v="$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || true)"
  fi
  case "$v" in ''|max|*[!0-9]*) return 0 ;; esac
  awk -v b="$v" 'BEGIN { if (b < 2 ^ 50) printf "%.1f", b / 2 ^ 30 }'
}
free_gib() { df -Pk "$1" 2>/dev/null | awk 'NR == 2 { printf "%.0f", $4 / 1048576 }'; }

NSYS_BIN="" NCU_BIN="" SAMPLE_FLAGS="$NO_SAMPLING" METRICS_FLAG="" NCU_HAS_KILL=0 GPU_NAME="gpu" GPU_CC="" NVTX_LIB=""
DRIVER_CUDA="" NVCC_REL=""
preflight() {
  local sev_build v t got what path want pair reading n tl comps chan nightly probe res mt cg eff fr

  # -- host, checkout -------------------------------------------------------------------
  if [ "$(uname -s)" = Linux ]; then
    chk PASS "host: $(uname -srm) · $(awk -F= '$1 == "PRETTY_NAME" { gsub(/"/, "", $2); print $2 }' /etc/os-release 2>/dev/null || true)"
  else
    chk FAIL "host: $(uname -s): the GPU runs need Linux"
  fi
  chk INFO "cpu: $(awk -F': ' '/^model name/ { print $2; exit }' /proc/cpuinfo 2>/dev/null || true) · $(nproc 2>/dev/null || echo '?') threads"
  chk INFO "git: HEAD $(git -C "$REPO" rev-parse --short=9 HEAD) on $(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
  n="$(git -C "$REPO" status --porcelain --untracked-files=no | awk 'END { print NR }')"
  if [ "$n" -eq 0 ]; then chk PASS "git: no tracked file modified"
  else chk WARN "git: $n tracked file(s) modified: the profile describes this working tree, not HEAD"; fi

  # -- plain tools ----------------------------------------------------------------------
  if [ "${BASH_VERSINFO[0]}" -ge 4 ]; then chk PASS "bash: $BASH_VERSION"; else chk FAIL "bash $BASH_VERSION: 4 or newer is required"; fi
  for t in git make awk sed tar timeout curl python3 df env; do
    if ! command -v "$t" >/dev/null 2>&1; then chk FAIL "tool: $t not found"; fi
  done
  if command -v sha256sum >/dev/null 2>&1 || command -v shasum >/dev/null 2>&1; then :; else chk FAIL "tool: neither sha256sum nor shasum"; fi
  if python3 -c 'import sqlite3, sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
    if python3 "$HERE/lib/nsys_block_summary.py" --selftest > "$TMP/selftest-nsys.log" 2>&1 \
       && python3 "$HERE/lib/ncu_summary.py" --selftest > "$TMP/selftest-ncu.log" 2>&1; then
      chk PASS "python3: $(python3 --version 2>&1) with sqlite3; post-processing selftests pass"
    else
      chk FAIL "python3: the post-processing selftests fail here: see $TMP/selftest-*.log"
    fi
  else
    chk FAIL "python3 >= 3.8 with the sqlite3 module is required (post-processing)"
  fi

  # -- the GPU --------------------------------------------------------------------------
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    chk FAIL "gpu: nvidia-smi not found"
  else
    n="$({ nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true; } | awk 'NF { n++ } END { print n + 0 }')"
    GPU_NAME="$(pf_gpu_field "$GPU" name)"
    if [ -z "$GPU_NAME" ]; then
      chk FAIL "gpu: no GPU with index $GPU ($n visible)"; GPU_NAME="gpu"
    else
      local vram drv cudadrv
      vram="$(pf_gpu_field "$GPU" memory.total)"; case "$vram" in ''|*[!0-9]*) vram=0 ;; esac
      GPU_CC="$(pf_gpu_field "$GPU" compute_cap)"; drv="$(pf_gpu_field "$GPU" driver_version)"
      cudadrv="$({ nvidia-smi 2>/dev/null || true; } | pf_first_match 'CUDA Version: [0-9]+[.][0-9]+' | awk '{ print $3 }')"
      DRIVER_CUDA="$cudadrv"
      chk PASS "gpu $GPU of $n: $GPU_NAME · $vram MiB · compute $GPU_CC · driver $drv"
      if [ -n "$cudadrv" ] && pf_ver_ge "$cudadrv" "$MIN_CUDA"; then chk PASS "driver: CUDA $cudadrv >= $MIN_CUDA"
      else chk FAIL "driver: CUDA '${cudadrv:-?}' < $MIN_CUDA (cudarc is pinned to the 12.8 driver API)"; fi
      if [ "$vram" -lt "$RECORD_VRAM_MIB" ]; then
        chk WARN "gpu: $vram MiB, less than the record's 32 GB card: the device layer sizes itself from the driver, so the schedule (and any fallback) differs from the record"
      fi
      if [ "$n" -gt 1 ] && [ -z "${CUDARC_NVCC_ARCH:-}" ]; then BUILD_ARCH="sm_${GPU_CC//./}"; chk INFO "build: $n GPUs, cubins built for GPU $GPU ($BUILD_ARCH)"; fi
    fi
    if reading="$(pf_gpu_idle "$GPU" "$IDLE_MIB")"; then chk PASS "idle: $reading"
    elif [ "$ALLOW_BUSY" = 1 ]; then chk WARN "idle: $reading (--allow-busy-gpu)"
    else chk FAIL "idle: $reading: another process holds the card (wait, or --allow-busy-gpu)"; fi
  fi

  # -- CUDA toolkit: build.rs looks for nvcc in exactly one place -------------------------
  v="$(pf_cuda_home)/bin/nvcc"
  if [ -x "$v" ]; then
    got="$({ "$v" --version 2>/dev/null || true; } | pf_first_match 'release [0-9]+[.][0-9]+' | awk '{ print $2 }')"
    NVCC_REL="$got"
    if [ -n "$got" ] && pf_ver_ge "$got" "$MIN_CUDA"; then chk PASS "nvcc: $v (release $got)"
    else chk FAIL "nvcc: $v is release '${got:-?}', < $MIN_CUDA"; fi
  else
    chk FAIL "nvcc: not at $v. math-cuda's build.rs looks ONLY there (CUDA_HOME, else CUDA_PATH, else /usr/local/cuda) and without it writes EMPTY cubins, so every kernel silently runs on the CPU. Set CUDA_HOME."
  fi

  # -- Nsight tools, identified by their banners -----------------------------------------
  NSYS_BIN="$(pf_find_tool nsys || true)"
  if [ -n "$NSYS_BIN" ]; then
    v="$(pf_tool_version "$NSYS_BIN")"
    if [ -n "$v" ] && pf_ver_ge "$v" "$MIN_NSIGHT"; then chk PASS "nsys: $NSYS_BIN ($v)"; else chk FAIL "nsys: $NSYS_BIN is ${v:-?}, < $MIN_NSIGHT"; fi
    if [ "$CPU_SAMPLING" = 1 ]; then SAMPLE_FLAGS="$(pf_sampling_flags "$NSYS_BIN")"; else SAMPLE_FLAGS="$NO_SAMPLING"; fi
    METRICS_FLAG="$(pf_metrics_flag "$NSYS_BIN")"
    chk INFO "nsys: CPU sampling: $SAMPLE_FLAGS$([ "$CPU_SAMPLING" = 0 ] && echo ' (--no-cpu-sampling)') · GPU-metrics option: ${METRICS_FLAG:-<none>}"
    { "$NSYS_BIN" status --environment 2>&1 || true; } > "$SMALL/nsys_status.txt"
    if [ "$SKIP_NSYS" = 0 ]; then
      NVTX_LIB="$(pf_find_nvtx_lib || true)"
      if [ -n "$NVTX_LIB" ]; then chk PASS "nvtx: $NVTX_LIB (handed to the run as LAMBDA_VM_NVTX_LIB)"
      else chk WARN "nvtx: no libnvToolsExt on this box (toolkits since CUDA 12.9 ship none), so the nvtx build's ranges are silent no-ops; run A still traces kernels, API calls and the OS runtime. For named phases: install cuda-nvtx-12-8 or set LAMBDA_VM_NVTX_LIB"; fi
    fi
  elif [ "$SKIP_NSYS" = 1 ]; then chk INFO "nsys: not found (run A skipped)"
  else chk FAIL "nsys (Nsight Systems) not found on PATH, in \$CUDA_HOME/bin or /opt/nvidia/nsight-systems/*; set NSYS=/path/to/nsys"; fi
  NCU_BIN="$(pf_find_tool ncu || true)"
  if [ -n "$NCU_BIN" ]; then
    v="$(pf_tool_version "$NCU_BIN")"
    if [ -n "$v" ] && pf_ver_ge "$v" "$MIN_NSIGHT"; then chk PASS "ncu: $NCU_BIN ($v)"
    elif [ "$SKIP_NCU" = 1 ]; then chk INFO "ncu: $NCU_BIN is ${v:-?} (run B skipped)"
    else chk FAIL "ncu: $NCU_BIN is ${v:-?}, < $MIN_NSIGHT"; fi
    got="$({ "$NCU_BIN" --help 2>&1 || true; })"
    case "$got" in *--kill*) NCU_HAS_KILL=1 ;; *) NCU_HAS_KILL=0 ;; esac
  elif [ "$SKIP_NCU" = 1 ]; then chk INFO "ncu: not found (run B skipped)"
  else chk FAIL "ncu (Nsight Compute) not found on PATH, in \$CUDA_HOME/bin or /opt/nvidia/nsight-compute/*; set NCU=/path/to/ncu (an 'ncu' that is npm-check-updates does not count)"; fi

  # -- counters, proven on one real kernel; nsys proven the same way ---------------------
  probe=""
  if [ -x "$(pf_cuda_home)/bin/nvcc" ] && [ -n "$GPU_CC" ]; then
    probe="$(pf_build_probe "$TMP/probe" "$GPU" || true)"
    if [ -z "$probe" ]; then chk FAIL "probe: could not build the one-kernel CUDA probe: see $TMP/probe/pf_probe.build.log"; fi
  fi
  if [ -n "$probe" ] && [ -n "$NCU_BIN" ]; then
    if [ "$COUNTERS" = 0 ]; then
      # run B's full flag set: ncu parses it before it touches the counters, so even a locked
      # box shows whether this ncu accepts it ('locked' = accepted, then refused)
      ncu_args pf_probe_kernel 0 1 "$TMP/probe/pf_probe_ncu"
      res="$(pf_counter_probe "$NCU_BIN" "$probe" "$TMP/probe/pf_probe.ncu.log" "${NCU_ARGS[@]}")"
      chk INFO "counters: $res (not needed with --no-counters; 'locked' also means ncu accepted run B's flags)"
    else
      res="$(pf_counter_probe "$NCU_BIN" "$probe" "$TMP/probe/pf_probe.ncu.log")"
      if [ "$res" != "COUNTERS unlocked" ]; then
        chk FAIL "counters: $res. Unlock (as root: echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' > /etc/modprobe.d/nvidia-profiling.conf, update-initramfs -u, reboot), or run as root, or pass --no-counters"
      elif [ "$SKIP_NCU" = 1 ]; then
        chk PASS "counters: $res (ncu profiled the probe kernel)"
      else
        # the analysis's metrics, narrowed to what this ncu and GPU collect; then run B's exact flags
        got="$NCU_METRICS_OK"
        NCU_METRICS_OK="$(pf_validate_metrics "$NCU_BIN" "$probe" "$TMP/probe" "$got")"
        n="$(printf '%s' "$NCU_METRICS_OK" | awk -F, '{ print (length($0) ? NF : 0) }')"
        v="$(printf '%s\n' "${got//,/$'\n'}" | while IFS= read -r t; do case ",$NCU_METRICS_OK," in *",$t,"*) ;; *) printf '%s ' "$t" ;; esac; done)"
        if [ "$n" -gt 0 ]; then chk INFO "ncu metrics: $n of $(printf '%s' "$got" | awk -F, '{ print NF }') collectable here; dropped (usually the other spelling of a stall metric): ${v:-none}"
        else chk WARN "ncu metrics: none of the explicit metrics is collectable here; run B keeps the sections only"; fi
        ncu_args pf_probe_kernel 0 1 "$TMP/probe/pf_probe_ncu"
        res="$(pf_counter_probe "$NCU_BIN" "$probe" "$TMP/probe/pf_probe_runb.ncu.log" "${NCU_ARGS[@]}")"
        if [ "$res" = "COUNTERS unlocked" ]; then chk PASS "counters: unlocked; ncu profiled the probe kernel with run B's exact flags"
        else chk FAIL "counters are unlocked but ncu refused run B's flags on the probe kernel: $res"; fi
      fi
    fi
  elif [ "$COUNTERS" = 1 ]; then
    chk FAIL "counters: not probed (no ncu or no probe binary); counters mode needs the proof"
  fi
  if [ -n "$probe" ] && [ -n "$NSYS_BIN" ] && [ "$SKIP_NSYS" = 0 ]; then
    if [ "$COUNTERS" = 1 ] && [ -z "$METRICS_FLAG" ]; then chk FAIL "nsys: this nsys has no GPU-metrics option"
    else
      nsys_args "$TMP/probe/pf_probe_nsys"
      res="$(pf_nsys_probe "$NSYS_BIN" "$probe" "$TMP/probe/pf_probe_nsys" "$COUNTERS" "${NSYS_ARGS[@]}")"
      if [ "${res#NSYS ok}" = "$res" ] && [ "$SAMPLE_FLAGS" != "$NO_SAMPLING" ]; then
        # CPU sampling is optional: a box that refuses it runs without it, it is never a failure
        chk INFO "nsys: the probe failed with CPU sampling on ($res); retrying without it"
        SAMPLE_FLAGS="$NO_SAMPLING"
        nsys_args "$TMP/probe/pf_probe_nsys_nosample"
        res="$(pf_nsys_probe "$NSYS_BIN" "$probe" "$TMP/probe/pf_probe_nsys_nosample" "$COUNTERS" "${NSYS_ARGS[@]}")"
      fi
      case "$res" in "NSYS ok"*) chk PASS "nsys probe (run A's flags, $SAMPLE_FLAGS): $res" ;; *) chk FAIL "nsys probe (run A's flags): $res" ;; esac
    fi
  fi

  # -- host memory and disk ---------------------------------------------------------------
  mt="$(awk '/^MemTotal:/ { printf "%.1f", $2 / 1048576 }' /proc/meminfo 2>/dev/null || true)"
  cg="$(cgroup_limit_gib)"
  eff="$mt"
  if [ -n "$cg" ] && { [ -z "$mt" ] || ! num_ge "$cg" "$mt"; }; then eff="$cg"; fi
  if [ -z "$eff" ]; then chk FAIL "ram: cannot read /proc/meminfo"
  elif num_ge "$eff" "$MIN_RAM_GIB"; then chk PASS "ram: $eff GiB usable (MemTotal ${mt:-?} GiB, cgroup limit ${cg:-none})"
  else chk FAIL "ram: $eff GiB usable (MemTotal ${mt:-?}, cgroup ${cg:-none}) < $MIN_RAM_GIB GiB: the run peaks near 24 GiB on the host"; fi
  if [ -n "$eff" ] && [ "$SKIP_NCU" = 0 ] && ! num_ge "$eff" 64; then
    chk WARN "ram: under 64 GiB: ncu's kernel replay saves the card's allocations to the host when the card is full, so run B's later passes may run out of memory"
  fi
  fr="$(free_gib "$REPO")"
  if [ "$SKIP_BUILD" = 1 ]; then chk INFO "disk: ${fr:-?} GiB free for the checkout"
  elif [ -z "$fr" ]; then chk WARN "disk: cannot read the free space under $REPO"
  elif ! num_ge "$fr" 10; then chk FAIL "disk: $fr GiB free under $REPO; a release CUDA build plus the guests needs ~20-40 GiB"
  elif ! num_ge "$fr" 40; then chk WARN "disk: $fr GiB free under $REPO; a fresh release CUDA build plus the guests can take ~20-40 GiB"
  else chk PASS "disk: $fr GiB free for the checkout"; fi
  fr="$(free_gib "$OUT")"
  if [ -z "$fr" ]; then chk WARN "disk: cannot read the free space under $OUT"
  elif ! num_ge "$fr" 5; then chk FAIL "disk: $fr GiB free for the output; the trace and its sqlite need several GiB"
  elif ! num_ge "$fr" 20; then chk WARN "disk: $fr GiB free for the output; with GPU metrics the trace + sqlite can reach ~5-10 GiB"
  else chk PASS "disk: $fr GiB free for the output"; fi

  # -- build toolchain: needed unless --skip-build -----------------------------------------
  sev_build=FAIL; if [ "$SKIP_BUILD" = 1 ]; then sev_build=INFO; fi
  chan="$(awk -F'"' '/^channel/ { print $2 }' "$REPO/rust-toolchain.toml" 2>/dev/null || true)"
  nightly="$({ grep -o 'rustup run nightly-[0-9-]*' "$REPO/Makefile" 2>/dev/null || true; } | awk 'NR == 1 { print $3 }')"
  if command -v rustup >/dev/null 2>&1 && command -v cargo >/dev/null 2>&1; then
    tl="$(rustup toolchain list 2>/dev/null || true)"
    case "$tl" in
      *"$chan"*) chk PASS "rust: toolchain $chan (rust-toolchain.toml) installed" ;;
      *) chk "$sev_build" "rust: toolchain $chan missing: rustup toolchain install $chan --profile default" ;;
    esac
    case "$tl" in
      *"$nightly"*)
        comps="$(rustup component list --toolchain "$nightly" --installed 2>/dev/null || true)"
        case "$comps" in
          *rust-src*) chk PASS "rust: $nightly with rust-src (the guests build std)" ;;
          *) chk "$sev_build" "rust: $nightly lacks rust-src: rustup component add rust-src --toolchain $nightly" ;;
        esac ;;
      *) chk "$sev_build" "rust: $nightly missing (the guests build std): rustup toolchain install $nightly --component rust-src" ;;
    esac
  else
    chk "$sev_build" "rust: rustup/cargo not found (https://rustup.rs)"
  fi
  if command -v clang >/dev/null 2>&1; then
    if printf 'int main(void) { return 0; }\n' | clang --target=riscv64 -march=rv64im -mabi=lp64 -x c -c - -o "$TMP/pf_rv.o" >/dev/null 2>&1; then
      chk PASS "clang: $(clang --version 2>/dev/null | awk 'NR == 1') (targets riscv64)"
    else chk "$sev_build" "clang cannot target riscv64 (the guests): install LLVM, scripts/SERVER_SETUP.md"; fi
  else chk "$sev_build" "clang not found (the guests): scripts/SERVER_SETUP.md"; fi
  if command -v ld.lld >/dev/null 2>&1 || command -v lld >/dev/null 2>&1; then chk PASS "lld: $(command -v ld.lld || command -v lld)"
  else chk "$sev_build" "lld not found (the guests link with -fuse-ld=lld): scripts/SERVER_SETUP.md"; fi
  if [ -f "$SYSROOT_DIR/include/stdlib.h" ] && [ -d "$SYSROOT_DIR/lib" ]; then chk PASS "sysroot: $SYSROOT_DIR"
  elif [ "$SKIP_BUILD" = 1 ]; then chk INFO "sysroot: missing at $SYSROOT_DIR (not needed with --skip-build)"
  else
    pf_log "provisioning the guest sysroot: make prepare-sysroot SYSROOT_DIR=$SYSROOT_DIR"
    if (cd "$REPO" && timeout 1800 make prepare-sysroot SYSROOT_DIR="$SYSROOT_DIR") > "$BIG/prepare-sysroot.log" 2>&1; then
      chk PASS "sysroot: provisioned at $SYSROOT_DIR"
    else chk FAIL "sysroot: make prepare-sysroot failed: see $BIG/prepare-sysroot.log"; fi
  fi

  # -- the fixtures, by content ------------------------------------------------------------
  for pair in "block ELF|$ELF|$ELF_SHA" "block input|$INPUT|$INPUT_SHA"; do
    IFS='|' read -r what path want <<< "$pair"
    if [ ! -r "$path" ]; then chk FAIL "fixture: the $what is missing: $path"; continue; fi
    got="$(pf_sha256 "$path" || true)"
    if [ "$got" = "$want" ]; then chk PASS "fixture: $what $(wc -c < "$path" | tr -d ' ') B, sha256 $got"
    else chk FAIL "fixture: $what sha256 ${got:-?}, want $want: same name, different file"; fi
  done

  # -- what this shell would have leaked into the run (names only) -------------------------
  n="$(env | awk -F= '/^(LAMBDA_VM_|LFM_|ZF_|A_BUNDLE|A_CACHE|TABLE_PARALLELISM=|RAYON_|_RJEM_|MALLOC_CONF=)/ { printf "%s ", $1 }')"
  if [ -n "$n" ]; then chk INFO "env: set in this shell and NOT passed to the profiled process: $n"
  else chk PASS "env: no prover knob is set in this shell"; fi
  n="$(env | awk -F= '/^(RUSTFLAGS|CARGO_BUILD_RUSTFLAGS|CARGO_ENCODED_RUSTFLAGS|LAMBDA_VM_RPX_MAXRREGCOUNT)=/ { printf "%s ", $1 }')"
  if [ -n "$n" ]; then chk WARN "env: $n set here; unset for the build so the binary and cubins are the production ones"; fi

  if [ "$PF_FAILS" -eq 0 ]; then chk INFO "PREFLIGHT: PASS ($PF_WARNS warning(s))"
  else chk INFO "PREFLIGHT: FAIL ($PF_FAILS failure(s), $PF_WARNS warning(s)); fix the FAIL lines above"; fi
}

# ======================================================================================
# the environment of the profiled process
# ======================================================================================
record_env() {
  pf_base_env
  RUN_ENV=("${PF_ENV[@]}"
    "CARGO_MANIFEST_DIR=$REPO/prover"
    "LFM_CENSUS_ELF=$ELF"
    "LFM_CENSUS_INPUT=$INPUT"
    "LFM_CENSUS_EPOCH_LOG2=$EPOCH_LOG2"
    "LAMBDA_VM_MAX_ROWS_LOG2=$EPOCH_LOG2"
    "LAMBDA_VM_MEMPOOL_RELEASE_MB=0"
    "LFM_TREE_PROVE_ROOT=1"
    "LFM_TREE_ROOT_OPTION=A"
    "LFM_WHIR_RETENTION=1"
    "LAMBDA_VM_WHIR_HASH=rpx"
    "TABLE_PARALLELISM=4"
    "LFM_TREE_SIBLINGS_L0=6"
    "LFM_TREE_SIBLINGS=4"
    "LFM_TREE_LEVEL_POOL=1"
    "LFM_PRECOMPUTED_TREE_CACHE_CAP=64"
    "LFM_EXEC_PARALLEL=1"
    "LFM_PROVE_SPLIT=1"
    "LAMBDA_VM_BASE_SPLIT=1"
    "LFM_CARD_TRACE=1"
    "LAMBDA_VM_GRIND_SCAN_FACTOR=8"
    "LAMBDA_VM_GRIND_GRID=1024"
    "_RJEM_MALLOC_CONF=dirty_decay_ms:-1,muzzy_decay_ms:-1")
  if [ -n "$NVTX_LIB" ]; then RUN_ENV+=("LAMBDA_VM_NVTX_LIB=$NVTX_LIB"); fi
}

write_env_capture() { # an ALLOWLIST of run facts, never the environment (this file leaves the box)
  local f="$SMALL/env.txt" e name
  {
    echo "# Run facts, an allowlist. The environment itself is never recorded in OUT/small: nsys and"
    echo "# ncu store it in their reports, which is why those stay in OUT/big."
    echo "repo_sha=$(git -C "$REPO" rev-parse HEAD)"
    echo "repo_branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
    echo "tracked_files_modified=$(git -C "$REPO" status --porcelain --untracked-files=no | awk 'END { print NR }')"
    echo "test_binary_sha256=$(pf_sha256 "$BIN" || echo '?')"
    echo "gpu_index=$GPU"
    echo "gpu_name=$GPU_NAME"
    echo "gpu_compute_cap=$GPU_CC"
    echo "gpu_memory_total_mib=$(pf_gpu_field "$GPU" memory.total)"
    echo "gpu_clocks_max_sm_mhz=$(pf_gpu_field "$GPU" clocks.max.sm)"
    echo "gpu_clocks_max_mem_mhz=$(pf_gpu_field "$GPU" clocks.max.mem)"
    echo "gpu_power_limit_w=$(pf_gpu_field "$GPU" power.limit)"
    echo "driver_version=$(pf_gpu_field "$GPU" driver_version)"
    echo "driver_cuda_version=${DRIVER_CUDA:-?}"
    echo "nvcc_release=${NVCC_REL:-?}"
    echo "nsys_version=$(if [ -n "$NSYS_BIN" ]; then pf_tool_version "$NSYS_BIN"; else echo none; fi)"
    echo "ncu_version=$(if [ -n "$NCU_BIN" ]; then pf_tool_version "$NCU_BIN"; else echo none; fi)"
    echo "rustc=$(cd "$REPO" && rustc --version 2>/dev/null || echo '?')"
    echo "cpu=$(awk -F': ' '/^model name/ { print $2; exit }' /proc/cpuinfo 2>/dev/null || true) · $(nproc 2>/dev/null || echo '?') threads"
    echo "options=counters:$COUNTERS skip_build:$SKIP_BUILD skip_nsys:$SKIP_NSYS skip_ncu:$SKIP_NCU gpu_metrics_hz:$GPU_METRICS_FREQ cpu_sampling:'$SAMPLE_FLAGS'"
    echo "# The knob variables the profiled process ran with. It runs under env -i with these and the"
    echo "# system basics PATH HOME USER LOGNAME LANG LC_ALL TERM TMPDIR XDG_RUNTIME_DIR LD_LIBRARY_PATH"
    echo "# CUDA_HOME CUDA_PATH (each only when set), CUDA_DEVICE_ORDER, CUDA_VISIBLE_DEVICES and"
    echo "# CARGO_MANIFEST_DIR, whose values are not recorded. <repo> = this checkout."
    for e in "${RUN_ENV[@]}"; do
      name="${e%%=*}"
      case "$name" in
        LAMBDA_VM_*|LFM_*|TABLE_PARALLELISM|_RJEM_MALLOC_CONF|ZF_*) printf '%s\n' "${e//"$REPO"/<repo>}" ;;
      esac
    done
    echo "# LAMBDA_VM_ZF_* (the proof-format knobs): none set, i.e. the default format."
    echo "# Deliberately not set, as in the record: LAMBDA_VM_VRAM_BUDGET_MB (VRAM_BUDGET_MB=query: the"
    echo "# device layer asks the driver), LFM_TREE_LEVELS (refused under PROVE_ROOT), LFM_TREE_TOP_OVERLAP"
    echo "# (its default), LFM_WHIR_PREFETCH, LAMBDA_VM_TREE_BUSY_PROBE, LFM_CENSUS_FAN_IN, A_BUNDLE*, A_CACHE_DIR."
    echo "# Run: (cd <repo>/prover && <test binary> $TEST --ignored --exact --nocapture --test-threads=1)"
  } > "$f"
}

# ======================================================================================
# build
# ======================================================================================
BUILD_ARCH="" BIN="" CUBIN_DIR="" GRIND_BIN=""
build_env() { # profiling builds: no ad-hoc rustflags and no register cap; lineinfo ON, which adds
  # SASS-to-source line tables to the cubins and leaves the code itself unchanged (build.rs)
  env -u RUSTFLAGS -u CARGO_BUILD_RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS -u LAMBDA_VM_RPX_MAXRREGCOUNT \
      LAMBDA_VM_NVCC_LINEINFO=1 ${BUILD_ARCH:+"CUDARC_NVCC_ARCH=$BUILD_ARCH"} "$@"
}

check_artifacts() { # the guest ELFs the Makefile enumerates, all present (the record launcher's probe)
  local probe arts f expected=0 missing=0
  probe="$TMP/artifacts.mk"
  # shellcheck disable=SC2016 # make variables, expanded by make
  printf '__probe: ; @echo $(ASM_ARTIFACTS) $(RUST_ARTIFACTS) $(RECURSION_ARTIFACTS) $(RECURSION_VERIFIER_ARTIFACTS)\n' > "$probe"
  arts="$(cd "$REPO" && make --no-print-directory -f Makefile -f "$probe" __probe 2>"$TMP/artifacts.err" || true)"
  for f in $arts; do
    expected=$((expected + 1))
    if [ ! -r "$REPO/$f" ]; then
      missing=$((missing + 1))
      if [ "$missing" -le 5 ]; then pf_log "  MISSING: $f"; fi
    fi
  done
  pf_log "guest artifacts: expected=$expected missing=$missing"
  if [ "$expected" -gt 0 ] && [ "$missing" -eq 0 ]; then return 0; fi
  return 1
}

check_cubins() { # an empty cubin is build.rs's no-nvcc stub: the kernels would run on the CPU
  local f n=0 empty=0
  for f in "$CUBIN_DIR"/*.cubin; do
    if [ ! -e "$f" ]; then continue; fi
    n=$((n + 1))
    if [ ! -s "$f" ]; then empty=$((empty + 1)); fi
  done
  pf_log "cubins: $n in $CUBIN_DIR, $empty empty"
  if [ "$n" -gt 0 ] && [ "$empty" -eq 0 ]; then return 0; fi
  return 1
}

check_test_listed() { # check_test_listed BINARY TEST
  local listed
  listed="$({ "$1" --list 2>/dev/null || true; } | awk -v t="$2: test" '$0 == t { n++ } END { print n + 0 }')"
  pf_log "$(basename "$1") lists $2: $listed time(s)"
  if [ "$listed" = 1 ]; then return 0; fi
  return 1
}

res_usage() { # cuobjdump -res-usage of every cubin: registers, stack, shared and local memory per kernel
  local cuobjdump f
  cuobjdump="$(pf_cuda_home)/bin/cuobjdump"
  if [ ! -x "$cuobjdump" ]; then pf_log "no cuobjdump at $cuobjdump: cubin_res_usage.txt skipped"; return 0; fi
  for f in "$CUBIN_DIR"/*.cubin; do
    if [ ! -e "$f" ]; then continue; fi
    echo "== $(basename "$f")"
    "$cuobjdump" -res-usage "$f" 2>&1 | sed "s#$CUBIN_DIR/##g" || true
  done > "$SMALL/cubin_res_usage.txt"
  pf_log "cuobjdump -res-usage: $(grep -c 'Function ' "$SMALL/cubin_res_usage.txt" || true) kernel entries -> small/cubin_res_usage.txt"
}

build() {
  local rc locks_clean=0
  local -a locks=(executor/programs/rust/hint_min/Cargo.lock executor/programs/rust/hint_multi/Cargo.lock)
  if git -C "$REPO" diff --quiet -- "${locks[@]}" 2>/dev/null; then locks_clean=1; fi
  step_begin "build: guest ELFs (make compile-programs-asm compile-programs-rust compile-recursion-elfs)"
  rc=0
  (cd "$REPO" && build_env SYSROOT_DIR="$SYSROOT_DIR" timeout "$BUILD_TIMEOUT" \
     make compile-programs-asm compile-programs-rust compile-recursion-elfs) > "$BIG/build-guests.log" 2>&1 || rc=$?
  tail -n 40 "$BIG/build-guests.log" > "$SMALL/build-guests.tail.txt" || true
  # the guest build rewrites these two lockfiles (the record launcher restores them too)
  if [ "$locks_clean" = 1 ]; then git -C "$REPO" checkout -- "${locks[@]}" 2>/dev/null || true; fi
  step_end "$rc"
  if [ "$rc" -ne 0 ]; then pf_die "guest ELF build failed (rc=$rc): see $BIG/build-guests.log"; fi
  check_artifacts || pf_die "guest ELFs missing after the build: see $BIG/build-guests.log"

  step_begin "build: prover lib tests (cargo test --release -p lambda-vm-prover --features $FEATURES --lib --no-run, lineinfo)"
  rc=0
  (cd "$REPO" && build_env timeout "$BUILD_TIMEOUT" cargo test --release -p lambda-vm-prover --features "$FEATURES" --lib \
     --no-run --message-format=json-render-diagnostics) > "$TMP/cargo-prover.json" 2> "$BIG/build-prover.log" || rc=$?
  tail -n 40 "$BIG/build-prover.log" > "$SMALL/build-prover.tail.txt" || true
  step_end "$rc"
  if [ "$rc" -ne 0 ]; then pf_die "prover build failed (rc=$rc): see $BIG/build-prover.log"; fi
  BIN="$(python3 "$HERE/lib/cargo_test_bin.py" exe --name lambda_vm_prover --kind lib < "$TMP/cargo-prover.json")" \
    || pf_die "cargo reported no lambda_vm_prover unit-test binary (see $TMP/cargo-prover.json)"
  CUBIN_DIR="$(python3 "$HERE/lib/cargo_test_bin.py" outdir --package math-cuda < "$TMP/cargo-prover.json")" \
    || pf_die "cargo reported no math-cuda build-script output (see $TMP/cargo-prover.json)"

  step_begin "build: the rpx_grind_counted test (run B's grind pairing), same features"
  rc=0
  (cd "$REPO" && build_env timeout "$BUILD_TIMEOUT" cargo test --release -p lambda-vm-prover --features "$FEATURES" \
     --test rpx_grind_counted --no-run --message-format=json-render-diagnostics) > "$TMP/cargo-grind.json" 2> "$BIG/build-grind.log" || rc=$?
  step_end "$rc"
  if [ "$rc" -ne 0 ]; then pf_die "rpx_grind_counted build failed (rc=$rc): see $BIG/build-grind.log"; fi
  GRIND_BIN="$(python3 "$HERE/lib/cargo_test_bin.py" exe --name rpx_grind_counted --kind test < "$TMP/cargo-grind.json")" \
    || pf_die "cargo reported no rpx_grind_counted binary (see $TMP/cargo-grind.json)"

  check_cubins || pf_die "the cubins are missing or empty: nvcc was not found by build.rs, so the GPU path is a CPU path"
  check_test_listed "$BIN" "$TEST" || pf_die "the built binary does not contain $TEST"
  check_test_listed "$GRIND_BIN" "$GRIND_TEST" || pf_die "the rpx_grind_counted binary does not contain $GRIND_TEST"
  res_usage
  printf 'HEAD=%s\nFEATURES=%s\nLINEINFO=1\nBIN=%s\nGRIND_BIN=%s\nCUBIN_DIR=%s\nBUILT=%s\n' \
    "$(git -C "$REPO" rev-parse HEAD)" "$FEATURES" "$BIN" "$GRIND_BIN" "$CUBIN_DIR" "$(date -u +%FT%TZ)" \
    > "$REPO/target/block_profile.last-build"
  pf_log "test binaries: $BIN ; $GRIND_BIN"
}

reuse_build() {
  local crumb="$REPO/target/block_profile.last-build" head feat
  if [ ! -r "$crumb" ]; then pf_die "--skip-build: no $crumb; run once without --skip-build"; fi
  head="$(sed -n 's/^HEAD=//p' "$crumb")"
  feat="$(sed -n 's/^FEATURES=//p' "$crumb")"
  BIN="$(sed -n 's/^BIN=//p' "$crumb")"
  GRIND_BIN="$(sed -n 's/^GRIND_BIN=//p' "$crumb")"
  CUBIN_DIR="$(sed -n 's/^CUBIN_DIR=//p' "$crumb")"
  if [ "$head" != "$(git -C "$REPO" rev-parse HEAD)" ]; then pf_die "--skip-build: the last build is of $head, HEAD has moved; rebuild"; fi
  if [ "$feat" != "$FEATURES" ] || [ "$(sed -n 's/^LINEINFO=//p' "$crumb")" != 1 ]; then
    pf_die "--skip-build: the last build is --features ${feat:-?} without the lineinfo cubins this script wants; rebuild"
  fi
  if [ ! -x "$BIN" ] || [ ! -x "$GRIND_BIN" ]; then pf_die "--skip-build: a test binary of the last build is gone; rebuild"; fi
  check_artifacts || pf_die "--skip-build: guest ELFs are missing; rebuild"
  check_cubins || pf_die "--skip-build: the cubins are missing or empty; rebuild"
  check_test_listed "$BIN" "$TEST" || pf_die "--skip-build: $BIN does not contain $TEST"
  check_test_listed "$GRIND_BIN" "$GRIND_TEST" || pf_die "--skip-build: $GRIND_BIN does not contain $GRIND_TEST"
  res_usage
  pf_log "reusing the build of $head: $BIN"
}

# ======================================================================================
# run A: Nsight Systems
# ======================================================================================
start_sampler() { # device memory at 10 Hz, UTC stamps, for the per-stage VRAM peak
  TZ=UTC nvidia-smi -i "$GPU" --query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,power.draw \
    --format=csv,noheader,nounits -lms 100 > "$SMALL/nvsmi_100ms.csv" 2>/dev/null &
  SAMPLER=$!
}
stop_sampler() {
  if [ -n "$SAMPLER" ]; then kill "$SAMPLER" 2>/dev/null || true; wait "$SAMPLER" 2>/dev/null || true; SAMPLER=""; fi
}

nsys_block_run() { # the traced run itself, into big/runA.raw.log
  (cd "$REPO/prover" && exec timeout --signal=INT --kill-after=300 "$RUN_A_TIMEOUT" \
     env -i "${RUN_ENV[@]}" "$NSYS_BIN" "${NSYS_ARGS[@]}" \
     "$BIN" "$TEST" --ignored --exact --nocapture --test-threads=1) > "$BIG/runA.raw.log" 2>&1
}

nsys_args() { # nsys_args REP_BASE — NSYS_ARGS=(profile ...): run A's flags, shared with the preflight probe
  local -a sflags
  NSYS_ARGS=(profile --trace="cuda,osrt,nvtx")
  read -r -a sflags <<< "$SAMPLE_FLAGS"
  NSYS_ARGS+=("${sflags[@]}")
  if [ "$COUNTERS" = 1 ]; then NSYS_ARGS+=("$METRICS_FLAG=all" "--gpu-metrics-frequency=$GPU_METRICS_FREQ"); fi
  NSYS_ARGS+=(--stats=false -f true -o "$1")
}

ncu_args() { # ncu_args KERNEL SKIP COUNT REP_BASE — NCU_ARGS=(...): run B's flags, shared with the probe
  local s
  NCU_ARGS=(--target-processes all -k "regex:^${1}\$" -s "$2" -c "$3")
  if [ "$NCU_HAS_KILL" = 1 ]; then NCU_ARGS+=(--kill yes); fi
  for s in $NCU_SECTIONS; do NCU_ARGS+=(--section "$s"); done
  if [ -n "$NCU_METRICS_OK" ]; then NCU_ARGS+=(--metrics "$NCU_METRICS_OK"); fi
  NCU_ARGS+=(-f -o "$4")
}

RUN_A_VERDICT="skipped"
readback_a() { # the record launcher's gates, reported rather than enforced: the trace is kept either way
  local log="$1" rc="$2" passed compressed cfb dfb knobs closure admitted issues=""
  passed="$(sed -n 's/^test result:.*[^0-9]\([0-9][0-9]*\) passed.*/\1/p' "$log" | awk 'END { print }')"
  compressed="$(grep -c '^★★★ THE BLOCK IS COMPRESSED UNDER WHIR' "$log" || true)"
  cfb="$(sed -n 's/.*commit fallbacks \([0-9][0-9]*\).*/\1/p' "$log" | awk 'NR == 1')"
  dfb="$(sed -n 's/.*device fallbacks \([0-9][0-9]*\).*/\1/p' "$log" | awk 'NR == 1')"
  knobs="$(grep -c 'GRIND KNOBS:' "$log" || true)"
  closure="$(grep -c 'WHIR BASE SPLIT: closure GREEN' "$log" || true)"
  admitted="$(sed -n 's/.*retention\[leaf layers\] admitted \([0-9][0-9]*\).*/\1/p' "$log" | awk 'NR == 1')"
  if [ "$rc" -ne 0 ]; then issues="$issues nsys/test rc=$rc;"; fi
  if [ "${passed:-0}" -lt 1 ]; then issues="$issues no passing test;"; fi
  if [ "$compressed" -ne 1 ]; then issues="$issues COMPRESSED line x$compressed (want 1);"; fi
  if [ "${cfb:-x}" != 0 ]; then issues="$issues commit fallbacks '${cfb:-absent}';"; fi
  if [ "${dfb:-x}" != 0 ]; then issues="$issues device fallbacks '${dfb:-absent}';"; fi
  if [ "$knobs" -lt 1 ]; then issues="$issues no GRIND KNOBS banner (no device grind);"; fi
  if [ "$closure" -ne 1 ]; then issues="$issues no base closure GREEN;"; fi
  if [ "${admitted:-0}" -lt 1 ]; then issues="$issues retention admitted '${admitted:-absent}';"; fi
  {
    echo "tests passed: ${passed:-<none>} · COMPRESSED x$compressed · commit fallbacks ${cfb:-<absent>} · device fallbacks ${dfb:-<absent>}"
    echo "GRIND KNOBS x$knobs · base closure GREEN x$closure · retention admitted ${admitted:-<absent>}"
    grep -E 'ZF FORMAT:|WHIR HASH:|base \(WHIR\): [0-9]+ epochs in|level 0: [0-9]+ WHIR wraps in|WHOLE RUN: host peak' "$log" || true
    if [ -z "$issues" ]; then echo "RUN A: OK (the record's gates hold: this is the record's pipeline, under tracing)"
    else echo "RUN A: SUSPECT:$issues"; fi
  } > "$SMALL/runA.verdict.txt"
  cat "$SMALL/runA.verdict.txt"
  if [ -z "$issues" ]; then RUN_A_VERDICT="ok"; else RUN_A_VERDICT="suspect"; fi
}

run_a() {
  local rc reading n v
  record_env
  write_env_capture
  if ! reading="$(pf_gpu_idle "$GPU" "$IDLE_MIB")"; then
    if [ "$ALLOW_BUSY" = 1 ]; then pf_log "WARN: $reading (--allow-busy-gpu)"; else pf_die "card not idle before run A: $reading"; fi
  fi
  nsys_args "$BIG/blockA"
  step_begin "run A: the WHIR block run under Nsight Systems"
  pf_log "nsys ${NSYS_ARGS[*]} $BIN $TEST --ignored --exact --nocapture --test-threads=1"
  start_sampler
  rc=0
  nsys_block_run || rc=$?
  stop_sampler
  if [ ! -s "$BIG/blockA.nsys-rep" ] && [ "$SAMPLE_FLAGS" != "$NO_SAMPLING" ]; then
    # CPU sampling is optional: a refusal the preflight did not catch costs one retry, not the run
    pf_log "run A wrote no report with CPU sampling on (rc=$rc); retrying once without it"
    mv "$BIG/runA.raw.log" "$BIG/runA.raw.try1.log"
    SAMPLE_FLAGS="$NO_SAMPLING"
    write_env_capture
    nsys_args "$BIG/blockA"
    start_sampler
    rc=0
    nsys_block_run || rc=$?
    stop_sampler
  fi
  { tr '\r' '\n' < "$BIG/runA.raw.log" | grep -vE '^\[[0-9]+/[0-9]+\] +\[' || true; } > "$SMALL/runA.test.log"
  step_end "$rc"
  readback_a "$SMALL/runA.test.log" "$rc"
  if [ ! -s "$BIG/blockA.nsys-rep" ]; then pf_log "run A wrote no report: see $BIG/runA.raw.log"; RUN_A_VERDICT="failed"; return 0; fi

  step_begin "run A: nsys export --type sqlite"
  rc=0
  timeout "$EXPORT_TIMEOUT" "$NSYS_BIN" export --type sqlite --force-overwrite true --output "$BIG/blockA.sqlite" \
    "$BIG/blockA.nsys-rep" > "$BIG/nsys-export.log" 2>&1 || rc=$?
  step_end "$rc"
  if [ ! -s "$BIG/blockA.sqlite" ]; then pf_log "no sqlite export: see $BIG/nsys-export.log"; RUN_A_VERDICT="failed"; return 0; fi

  step_begin "run A: nsys stats (the stock CSVs)"
  mkdir -p "$SMALL/nsys_stats"
  rc=0
  timeout "$EXPORT_TIMEOUT" "$NSYS_BIN" stats \
    --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,osrt_sum \
    --format csv --output "$SMALL/nsys_stats/blockA" "$BIG/blockA.sqlite" > "$BIG/nsys-stats.log" 2>&1 || rc=$?
  step_end "$rc"

  step_begin "run A: stage / timeline / GPU-metrics summary (lib/nsys_block_summary.py)"
  rc=0
  timeout "$EXPORT_TIMEOUT" python3 "$HERE/lib/nsys_block_summary.py" --sqlite "$BIG/blockA.sqlite" \
    --log "$SMALL/runA.test.log" --nvsmi "$SMALL/nvsmi_100ms.csv" --out "$SMALL/nsys" --quiet \
    > "$BIG/summary.log" 2>&1 || rc=$?
  step_end "$rc"
  if [ "$rc" -ne 0 ]; then pf_log "the summary failed: see $BIG/summary.log"; RUN_A_VERDICT="${RUN_A_VERDICT}+nosummary"; fi
  n="$(python3 - "$BIG/blockA.sqlite" <<'PY' 2>/dev/null || echo 0
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
try:
    print(db.execute("SELECT count(*) FROM NVTX_EVENTS WHERE end > start").fetchone()[0])
except sqlite3.Error:
    print(0)
PY
)"
  v="NVTX ranges in the trace: $n$([ "${n:-0}" -eq 0 ] && echo ' (none: no libnvToolsExt reached, see the preflight)')"
  pf_log "$v"
  # just above the verdict line, which stays the file's last
  awk -v ins="$v" 'NR > 1 { print prev } { prev = $0 } END { print ins; print prev }' "$SMALL/runA.verdict.txt" > "$TMP/verdict.txt" \
    && mv "$TMP/verdict.txt" "$SMALL/runA.verdict.txt"
  if [ "${n:-0}" -gt 0 ] && [ -r "$REPO/scripts/profiling/nsys_phase_busy.py" ]; then
    timeout "$EXPORT_TIMEOUT" python3 "$REPO/scripts/profiling/nsys_phase_busy.py" "$BIG/blockA.sqlite" --top 15 \
      > "$SMALL/nsys/phase_busy.md" 2> "$BIG/phase_busy.log" || pf_log "nsys_phase_busy.py failed: see $BIG/phase_busy.log"
  fi
}

# ======================================================================================
# run B: Nsight Compute, one pass per kernel
# ======================================================================================
RUN_B_VERDICT="skipped"
wait_idle() { # a killed pass must have released the card before the next one starts
  local reading=""
  for _ in $(seq 1 30); do
    if reading="$(pf_gpu_idle "$GPU" "$IDLE_MIB")"; then return 0; fi
    sleep 2
  done
  if [ "$ALLOW_BUSY" = 1 ]; then pf_log "WARN: $reading (--allow-busy-gpu)"; return 0; fi
  pf_die "card not idle after 60 s: $reading"
}

run_b() {
  local pass kernel skip count target bin test rc profiled t0 nbad=0 nshape=0 total=0
  record_env
  if [ ! -s "$SMALL/env.txt" ]; then write_env_capture; fi
  mkdir -p "$BIG/ncu" "$SMALL/ncu"
  if [ -n "${NCU_PLAN:-}" ]; then cp "$NCU_PLAN" "$TMP/ncu_plan.in.tsv"; else default_plan > "$TMP/ncu_plan.in.tsv"; fi
  if [ -s "$BIG/blockA.sqlite" ]; then
    python3 "$HERE/lib/ncu_plan.py" anchor "$TMP/ncu_plan.in.tsv" "$BIG/blockA.sqlite" > "$SMALL/ncu/plan.tsv" 2> "$BIG/ncu/anchor.log" \
      || cp "$TMP/ncu_plan.in.tsv" "$SMALL/ncu/plan.tsv"
    pf_log "run B: plan re-anchored on this box's run A trace: $(grep -c 'anchor: skip [0-9]* ->' "$SMALL/ncu/plan.tsv" || true) window(s) moved, $(grep -c 'not found' "$SMALL/ncu/plan.tsv" || true) not found"
  else
    cp "$TMP/ncu_plan.in.tsv" "$SMALL/ncu/plan.tsv"
    pf_log "run B: no run A trace in this OUT, so the windows are the reference trace's (verified after each pass)"
  fi
  printf 'pass\tkernel\tskip\tcount\trc\tprofiled\tseconds\n' > "$SMALL/ncu/passes.tsv"
  while IFS=$'\t' read -r pass kernel skip count _ _ _ target; do
    if [ "$pass" = pass ]; then continue; fi
    if [ -n "${NCU_PASSES:-}" ] && [[ " $NCU_PASSES " != *" $pass "* ]]; then continue; fi
    case "$target" in
      block) bin="$BIN"; test="$TEST" ;;
      grind_counted) bin="$GRIND_BIN"; test="$GRIND_TEST" ;;
      *) pf_log "run B: pass $pass has an unknown target '$target': skipped"; continue ;;
    esac
    total=$((total + 1))
    wait_idle
    step_begin "run B: ncu $pass ($kernel, its launches $skip..$((skip + count - 1)))"
    ncu_args "$kernel" "$skip" "$count" "$BIG/ncu/$pass"
    t0="$(date +%s)"
    rc=0
    (cd "$REPO/prover" && exec timeout --signal=INT --kill-after=120 "$NCU_PASS_TIMEOUT" \
       env -i "${RUN_ENV[@]}" "$NCU_BIN" "${NCU_ARGS[@]}" \
       "$bin" "$test" --ignored --exact --nocapture --test-threads=1) < /dev/null > "$BIG/ncu/$pass.stdout" 2>&1 || rc=$?
    profiled="$(grep -c '^==PROF== Profiling' "$BIG/ncu/$pass.stdout" || true)"
    { grep -E '^==(PROF|ERROR|WARNING)==' "$BIG/ncu/$pass.stdout" || true; } > "$SMALL/ncu/$pass.prof.txt"
    if [ "$target" != block ]; then # a micro test's own lines (the grind pairing prints the nonce)
      { tr '\r' '\n' < "$BIG/ncu/$pass.stdout" | grep -vE '^==(PROF|ERROR|WARNING)==' || true; } > "$SMALL/ncu/$pass.test.log"
    fi
    if [ -s "$BIG/ncu/$pass.ncu-rep" ]; then
      timeout 600 "$NCU_BIN" --import "$BIG/ncu/$pass.ncu-rep" --page details > "$SMALL/ncu/$pass.details.txt" 2>&1 || true
      timeout 600 "$NCU_BIN" --import "$BIG/ncu/$pass.ncu-rep" --page details --csv > "$SMALL/ncu/$pass.details.csv" 2>/dev/null || true
      timeout 600 "$NCU_BIN" --import "$BIG/ncu/$pass.ncu-rep" --page raw --csv > "$BIG/ncu/$pass.raw.csv" 2>/dev/null || true
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$pass" "$kernel" "$skip" "$count" "$rc" "$profiled" "$(( $(date +%s) - t0 ))" >> "$SMALL/ncu/passes.tsv"
    # with --kill ncu ends the run itself once the launches are in, so rc alone says little
    if [ "$profiled" -lt 1 ] || [ ! -s "$BIG/ncu/$pass.ncu-rep" ]; then nbad=$((nbad + 1)); fi
    pf_log "ncu $pass: $profiled launch(es) profiled, rc=$rc"
    step_end "$rc"
  done < "$SMALL/ncu/plan.tsv"
  python3 "$HERE/lib/ncu_plan.py" verify "$SMALL/ncu/plan.tsv" "$BIG/ncu" > "$SMALL/ncu/verify.tsv" 2> "$BIG/ncu/verify.log" || true
  if [ -n "${NCU_PASSES:-}" ]; then # verify reports every pass of the plan; keep the ones that ran
    awk -F'\t' -v keep=" $NCU_PASSES " 'NR == 1 || index(keep, " " $1 " ")' "$SMALL/ncu/verify.tsv" > "$TMP/verify.tsv" && mv "$TMP/verify.tsv" "$SMALL/ncu/verify.tsv"
  fi
  nshape="$(awk -F'\t' 'NR > 1 && $3 != "ok" { n++ } END { print n + 0 }' "$SMALL/ncu/verify.tsv")"
  pf_log "run B: shape check: $(awk -F'\t' 'NR > 1 && $3 == "ok" { n++ } END { print n + 0 }' "$SMALL/ncu/verify.tsv") pass(es) profiled exactly the planned shapes, $nshape did not (small/ncu/verify.tsv)"
  if compgen -G "$BIG/ncu/*.raw.csv" > /dev/null; then
    python3 "$HERE/lib/ncu_summary.py" --out "$SMALL/ncu" "$BIG"/ncu/*.raw.csv > "$BIG/ncu/summary.log" 2>&1 || true
  fi
  if [ "$nbad" -eq 0 ] && [ "$nshape" -eq 0 ]; then RUN_B_VERDICT="ok ($total passes, shapes as planned)"
  else RUN_B_VERDICT="partial ($nbad of $total passes without a profile, $nshape off the planned shapes)"; fi
}

# ======================================================================================
# plan, estimate, finish
# ======================================================================================
estimate() {
  local g=0 p=0 a=0 b=0 pass kernel skip count tref rest
  if [ "$SKIP_BUILD" = 0 ]; then
    if [ -r "$REPO/executor/program_artifacts/rust/ethrex.elf" ]; then g=3; else g=20; fi
    if [ -d "$REPO/target/release/deps" ]; then p=10; else p=25; fi
  fi
  if [ "$SKIP_NSYS" = 0 ]; then a=10; if [ "$COUNTERS" = 1 ]; then a=15; fi; fi
  pf_log "ESTIMATE (a guide, not a bound; every step has its own timeout):"
  pf_log "  guest ELFs      ~${g} min   (fresh ~20; up to date ~3: make re-runs cargo per guest)"
  pf_log "  prover builds   ~${p} min   (fresh release CUDA build + the grind test ~25; incremental less)"
  pf_log "  run A           ~${a} min   (~2-4 min of block run under nsys, then export + stats + summary)"
  if [ "$SKIP_NCU" = 0 ]; then
    while IFS=$'\t' read -r pass kernel skip count tref rest; do
      if [ "$pass" = pass ]; then continue; fi
      if [ -n "${NCU_PASSES:-}" ] && [[ " $NCU_PASSES " != *" $pass "* ]]; then continue; fi
      case "$tref" in ''|-|*[!0-9.]*) tref=110 ;; esac
      b="$(awk -v b="$b" -v t="$tref" -v c="$count" 'BEGIN { printf "%.0f", b + 60 + 2.5 * t + 20 * c }')"
      pf_log "  run B  $(printf '%-14s' "$pass") ~$(awk -v t="$tref" -v c="$count" 'BEGIN { printf "%.1f", (60 + 2.5 * t + 20 * c) / 60 }') min"
    done < <(if [ -n "${NCU_PLAN:-}" ]; then cat "$NCU_PLAN"; else default_plan; fi)
    pf_log "  run B total     ~$((b / 60)) min   (per pass: 60 s start + 2.5 x the window's time into the run + 20 s per launch)"
  fi
  pf_log "  TOTAL           ~$(( g + p + a + b / 60 )) min"
}

send_back() { # send_back SAFE(0|1) VERDICT SCAN_LINE — SEND-BACK.txt, marker-free by construction
  local safe="$1" verdict="$2" scan="$3" slug day
  slug="$(printf '%s' "$GPU_NAME" | tr '[:upper:]' '[:lower:]' | sed -e 's/nvidia//; s/geforce//; s/[^a-z0-9]\{1,\}/-/g; s/^-//; s/-$//')"
  day="$(date -u +%Y%m%d)"
  {
    if [ "$safe" = 1 ]; then
      echo "==================== SEND BACK ===================="
      echo "1. $SMALL ($(du -sh "$SMALL" | awk '{ print $1 }'); packed as $OUT/small.tar.gz)"
      echo "   passed the bundle scan: plain text only, no credential marker, no Nsight report or database."
      echo "   Commit it on this branch:"
      echo "     cd $REPO"
      echo "     mkdir -p scripts/profile/results/$day-$slug"
      echo "     cp -R $SMALL/. scripts/profile/results/$day-$slug/"
      echo "     git add scripts/profile/results/$day-$slug"
      echo "     git commit -m 'profile: WHIR block run on $GPU_NAME ($(date -u +%Y-%m-%d))'"
      echo "     git push origin HEAD:whir/profile-rpx"
      echo "   or send $OUT/small.tar.gz."
    else
      echo "==================== NOTHING TO SEND: THE BUNDLE SCAN REFUSED ===================="
      echo "1. $scan"
      echo "   Nothing in $OUT may be committed or sent until those lines are read and removed; the"
      echo "   findings list is in OUT/big and is not to be shared either. No small.tar.gz was written."
    fi
    echo "2. NEVER commit, share or upload $BIG: the .nsys-rep, .ncu-rep and .sqlite embed this"
    echo "   machine's environment (nsys and ncu store it verbatim). They stay on this box."
    echo "RESULT: run A $RUN_A_VERDICT · run B $RUN_B_VERDICT · bundle $([ "$safe" = 1 ] && echo clean || echo refused)"
    echo "BLOCK_PROFILE VERDICT: $verdict"
  } > "$SMALL/SEND-BACK.txt"
}

finish() {
  local verdict safe=0 scan
  {
    echo "OUT/small: what each file is (the only part of a run that leaves the box)"
    echo "  preflight.txt         every check, PASS/WARN/FAIL/INFO"
    echo "  env.txt               run facts, an allowlist: versions, GPU, repo sha, the knob variables set"
    echo "  steps.tsv             every step: start, seconds, rc"
    echo "  driver.log            this script's own output"
    echo "  build-*.tail.txt      the last lines of each build (full logs in OUT/big)"
    echo "  runA.test.log         the test's stdout under nsys (the harness's own stage lines)"
    echo "  runA.verdict.txt      the record's gates, read back from that log"
    echo "  nvsmi_100ms.csv       nvidia-smi at 10 Hz: memory.used MiB, util %, SM clock, power (UTC)"
    echo "  nsys_status.txt       nsys status --environment: what this box allows (CPU sampling, perf events)"
    echo "  cubin_res_usage.txt   cuobjdump -res-usage of every cubin: registers, stack, shared, local per kernel"
    echo "  nsys/phase_busy.md    (NVTX ranges present) GPU busy per NVTX phase (scripts/profiling/nsys_phase_busy.py)"
    echo "  nsys_stats/*.csv      nsys stats: cuda_gpu_kern_sum, cuda_api_sum, cuda_gpu_mem_{time,size}_sum, osrt_sum"
    echo "  nsys/summary.txt      READ FIRST: stages, busy/idle per 5 s, top kernels, copies, GPU metrics"
    echo "  nsys/stages.csv       per stage: wall, busy (kernel+copy union), kernel sum, VRAM peak, metrics, top kernels"
    echo "  nsys/busy_5s.csv      per 5 s: busy/idle, top kernel, VRAM peak (+ key GPU metrics)"
    echo "  nsys/kernels.csv      every kernel: launches, time, and its split by phase"
    echo "  nsys/gpu_metrics_*    (counters only) SM active / SM issue / DRAM % per 1 s, per stage, and every metric"
    echo "  ncu/ncu_summary.txt   (counters only) one line per profiled launch: SOL, issue, occupancy, stalls, pipes"
    echo "  ncu/<pass>.details.txt|csv   ncu's own per-launch report (the authority, with its rule messages)"
    echo "  ncu/plan.tsv          run B's plan as run: each pass's kernel window, its expected shapes and why"
    echo "  ncu/verify.tsv        per pass: did ncu profile the planned shapes (ok / partial / MISMATCH)"
    echo "  ncu/passes.tsv, ncu/<pass>.prof.txt   each pass's outcome and ncu's own messages"
    echo "  ncu/grind_pair.test.log   the counted-grind test's own lines (the warm-up line prints the nonce)"
    echo "  SEND-BACK.txt         what to send back, and the bundle scan's verdict"
    echo
    echo "files present:"
    (cd "$SMALL" && find . -type f | sort | sed 's/^\.\//  /')
  } > "$SMALL/INDEX.txt"
  {
    echo "OUT/big (never committed, shared or uploaded: it embeds this machine's environment):"
    (cd "$BIG" && find . -type f -size +0 ! -path './tmp/*' -exec ls -l {} + 2>/dev/null | awk '{ printf "  %12d  %s\n", $5, $NF }')
  } > "$SMALL/manifest.txt"

  verdict="PASS"
  if [ "$SKIP_NSYS" = 0 ] && [ "$RUN_A_VERDICT" != ok ]; then verdict="PARTIAL"; fi
  if [ "$SKIP_NCU" = 0 ] && [ "${RUN_B_VERDICT%% *}" != ok ]; then verdict="PARTIAL"; fi
  # Scan, write SEND-BACK, and scan again: the second scan is the authority, over the final files.
  rm -f "$OUT/small.tar.gz"
  if scan="$(pf_scan_bundle "$SMALL" "$BIG/bundle_scan.txt")"; then safe=1; fi
  send_back "$safe" "$([ "$safe" = 1 ] && echo "$verdict" || echo 'REFUSED (bundle scan)')" "$scan"
  if scan="$(pf_scan_bundle "$SMALL" "$BIG/bundle_scan.txt")"; then safe=1; else safe=0; fi
  if [ "$safe" = 1 ]; then
    tar -czf "$OUT/small.tar.gz" -C "$OUT" small
  else
    send_back 0 'REFUSED (bundle scan)' "$scan"
  fi
  pf_log "$scan"
  cat "$SMALL/SEND-BACK.txt"
  if [ "$safe" != 1 ]; then return 4; fi
  if [ "$verdict" != PASS ]; then return 3; fi
  return 0
}

# ======================================================================================
# main (sourcing this file defines the functions and runs nothing)
# ======================================================================================
main() {
  parse_args "$@"
  knob_defaults
  setup_out
  trap cleanup EXIT
  trap 'pf_log "interrupted"; exit 130' INT TERM
  pf_log "block_profile.sh: repo $REPO @ $(git -C "$REPO" rev-parse --short=9 HEAD) · out $OUT"
  pf_log "mode: counters=$COUNTERS skip_build=$SKIP_BUILD skip_nsys=$SKIP_NSYS skip_ncu=$SKIP_NCU gpu=$GPU"

  step_begin "preflight"
  preflight
  step_end "$PF_FAILS"
  if [ "$PF_FAILS" -ne 0 ]; then pf_log "stopping: preflight failed (see $SMALL/preflight.txt)"; exit 2; fi
  if [ "$PREFLIGHT_ONLY" = 1 ]; then pf_log "--preflight-only: done ($SMALL/preflight.txt)"; exit 0; fi
  if [ "$COUNTERS" = 1 ] && [ "$SKIP_NSYS" = 0 ] && [ -z "$METRICS_FLAG" ]; then pf_die "counters mode needs nsys GPU metrics"; fi

  estimate
  if [ "$SKIP_BUILD" = 1 ]; then reuse_build; else build; fi
  if [ "$SKIP_NSYS" = 0 ]; then run_a; fi
  if [ "$SKIP_NCU" = 0 ]; then run_b; fi
  finish
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
