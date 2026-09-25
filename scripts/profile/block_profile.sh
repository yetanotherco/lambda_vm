#!/usr/bin/env bash
# block_profile.sh — Nsight profile of the WHIR block run (block 25368371) on ONE GPU.
#
# WHAT IT ANSWERS
#   Run A (Nsight Systems): where the block run's wall goes on the card: GPU busy vs idle
#     per stage (base; the base's commit / prove / global; level 0; interior; root), per
#     5 s, and per kernel; with counters also SM active %, SM issue % and DRAM bandwidth %
#     per 1 s and per stage (nsys GPU metrics).
#   Run B (Nsight Compute, counters only): why the nine kernels that carry the time run as
#     fast as they do (speed of light, warp-stall reasons, occupancy and its limiter,
#     scheduler issue, instruction mix, pipes, memory) on real launches of this run.
#
# THE RUN IS THE RECORD'S
#   Test  lfm::per_table_aggregator_tests::the_whir_production_tree_composes_to_a_root
#   Build cargo test --release -p lambda-vm-prover --features cuda --lib (the binary is run
#         directly, from prover/ as cargo would, so no cargo process sits in the trace)
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
#   --preflight-only  stop after the checks
#
# ENV KNOBS (all optional)
#   SYSROOT_DIR        guest C sysroot (default $HOME/.lambda-vm-sysroot; provisioned when missing)
#   GPU_METRICS_FREQ   nsys GPU-metrics sampling rate in Hz (default 2000)
#   NCU_KERNELS        run B's kernels, "name:skip:count[:t_ref_s] ..." (default: the table below)
#   NCU_COUNT          one launch count for every kernel of run B
#   NCU_SECTIONS       ncu section identifiers (default: the eight below)
#   BUILD_TIMEOUT RUN_A_TIMEOUT EXPORT_TIMEOUT NCU_PASS_TIMEOUT   step bounds, seconds
#   IDLE_MIB           the card is idle below this many MiB in use with no compute process (500)
#   NSYS NCU           explicit tool paths
#
# OUTPUT
#   OUT/small/  CSVs, text summaries, logs, the exact environment: a few MB, safe to commit
#   OUT/big/    blockA.nsys-rep, blockA.sqlite, ncu/*.ncu-rep, raw exports, full build logs
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

# Run B's kernels, name:skip:count:t_ref. skip/count index that kernel's OWN launches (ncu
# counts only launches matching -k), chosen in the lead's wt90 trace (cdf0238f1, whose prover
# library is 169b66831's; RTX 5090) to land on the launches that carry each kernel's time:
#   rpx_grind_search          grid 1024/128, 7.7-11.7 ms (every launch has this shape)
#   rpx_merkle_level          grids 8192, 4096, 2048: the top of one tree (2.9, 1.5, 0.8 ms)
#   rpx_merkle_tail           one block of 128 threads, ~0.9 ms each, 4692 launches
#   rpx_leaves_base_coset     grid 16384, ~45 ms (the base's LDE leaves)
#   sumcheck_round_ext3       the three biggest rounds of one 2^21 sumcheck (5.9, 3.1, 1.6 ms)
#   rpx_leaves_base_row_major_row_pair        grids 8192, 8192, 16384 (5.7-16.5 ms)
#   ntt_dit_level_row_major   grid 1 x 22796 / 65535 with an 11-thread block (0.5-0.9 ms)
#   rpx_leaves_base_row_major_row_pair_range  grids 16384, 8192, 16384, 4096 (17.7-121 ms)
#   constraint_composition_kernel             grid 256/256 (30.5, 4.7, 6.0 ms)
# t_ref = when the last of them ran in that trace, which the runtime estimate uses. Kernels
# of the WHIR base come first: their passes end seconds into the run; the last four only run
# in the LFM wraps, so each of those passes first sits through ~70 s of base under ncu.
NCU_KERNELS_DEFAULT="
rpx_grind_search:16:3:4.7
rpx_merkle_level:11:3:3.2
rpx_merkle_tail:1:2:3.2
rpx_leaves_base_coset:1:3:3.3
sumcheck_round_ext3:340:3:3.9
rpx_leaves_base_row_major_row_pair:0:3:69.6
ntt_dit_level_row_major:59:3:69.6
rpx_leaves_base_row_major_row_pair_range:1:4:70.6
constraint_composition_kernel:0:3:71.1
"

usage() { sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

COUNTERS=1 SKIP_BUILD=0 SKIP_NSYS=0 SKIP_NCU=0 ALLOW_BUSY=0 PREFLIGHT_ONLY=0 GPU=0 OUT=""
parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --no-counters) COUNTERS=0 ;;
      --skip-build) SKIP_BUILD=1 ;;
      --skip-nsys) SKIP_NSYS=1 ;;
      --skip-ncu) SKIP_NCU=1 ;;
      --allow-busy-gpu) ALLOW_BUSY=1 ;;
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
  NCU_KERNELS="${NCU_KERNELS:-$NCU_KERNELS_DEFAULT}"
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

NSYS_BIN="" NCU_BIN="" SAMPLE_FLAGS="--sample=none --cpuctxsw=none" METRICS_FLAG="" NCU_HAS_KILL=0 GPU_NAME="gpu" GPU_CC=""
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
    SAMPLE_FLAGS="$(pf_sampling_flags "$NSYS_BIN")"
    METRICS_FLAG="$(pf_metrics_flag "$NSYS_BIN")"
    chk INFO "nsys: CPU sampling on this box: $SAMPLE_FLAGS · GPU-metrics option: ${METRICS_FLAG:-<none>}"
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
    # with run B's own flags in every mode: ncu parses them before it touches the counters, so
    # even a locked box shows whether this ncu accepts them ('locked' = accepted, then refused)
    ncu_args pf_probe_kernel 0 1 "$TMP/probe/pf_probe_ncu"
    res="$(pf_counter_probe "$NCU_BIN" "$probe" "$TMP/probe/pf_probe.ncu.log" "${NCU_ARGS[@]}")"
    if [ "$SKIP_NCU" = 1 ] && [ "$COUNTERS" = 0 ]; then chk INFO "counters: $res (not needed with --no-counters; 'locked' also means ncu accepted run B's flags)"
    elif [ "$res" = "COUNTERS unlocked" ]; then chk PASS "counters: $res (ncu profiled the probe kernel)"
    else
      chk FAIL "counters: $res. Unlock (as root: echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' > /etc/modprobe.d/nvidia-profiling.conf, update-initramfs -u, reboot), or run as root, or pass --no-counters"
    fi
  elif [ "$COUNTERS" = 1 ]; then
    chk FAIL "counters: not probed (no ncu or no probe binary); counters mode needs the proof"
  fi
  if [ -n "$probe" ] && [ -n "$NSYS_BIN" ] && [ "$SKIP_NSYS" = 0 ]; then
    if [ "$COUNTERS" = 1 ] && [ -z "$METRICS_FLAG" ]; then chk FAIL "nsys: this nsys has no GPU-metrics option"
    else
      nsys_args "$TMP/probe/pf_probe_nsys"
      res="$(pf_nsys_probe "$NSYS_BIN" "$probe" "$TMP/probe/pf_probe_nsys" "$COUNTERS" "${NSYS_ARGS[@]}")"
      case "$res" in "NSYS ok"*) chk PASS "nsys probe (run A's flags): $res" ;; *) chk FAIL "nsys probe (run A's flags): $res" ;; esac
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
  n="$(env | awk -F= '/^(RUSTFLAGS|CARGO_BUILD_RUSTFLAGS|CARGO_ENCODED_RUSTFLAGS|LAMBDA_VM_RPX_MAXRREGCOUNT|LAMBDA_VM_NVCC_LINEINFO)=/ { printf "%s ", $1 }')"
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
}

write_env_capture() {
  local f="$SMALL/env.txt" e
  {
    echo "# The profiled process ran under env -i with exactly these variables:"
    for e in "${RUN_ENV[@]}"; do printf '%s\n' "$e"; done
    echo "# Deliberately NOT set (the record's choices): LAMBDA_VM_VRAM_BUDGET_MB (VRAM_BUDGET_MB=query:"
    echo "#   the device layer asks the driver), LFM_TREE_LEVELS (refused under PROVE_ROOT), LFM_TREE_TOP_OVERLAP"
    echo "#   (left at its default), every LAMBDA_VM_ZF_* (the default proof format), LFM_WHIR_PREFETCH,"
    echo "#   LAMBDA_VM_TREE_BUSY_PROBE, LFM_CENSUS_FAN_IN, A_BUNDLE*, A_CACHE_DIR."
    echo "# Run: (cd $REPO/prover && <bin> $TEST --ignored --exact --nocapture --test-threads=1)"
    echo "repo_head=$(git -C "$REPO" rev-parse HEAD)"
    echo "repo_branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
    echo "tracked_files_modified=$(git -C "$REPO" status --porcelain --untracked-files=no | awk 'END { print NR }')"
    echo "test_binary=$BIN"
    echo "test_binary_sha256=$(pf_sha256 "$BIN" || echo '?')"
    echo "cubin_dir=$CUBIN_DIR"
    echo "gpu=$GPU $GPU_NAME (compute $GPU_CC)"
    echo "gpu_query=$(nvidia-smi -i "$GPU" --query-gpu=name,memory.total,driver_version,pci.bus_id,clocks.max.sm,power.limit --format=csv,noheader 2>/dev/null || echo '?')"
    echo "nvcc=$({ "$(pf_cuda_home)/bin/nvcc" --version 2>/dev/null || true; } | awk 'END { print }')"
    echo "nsys=${NSYS_BIN:-none} $( [ -n "$NSYS_BIN" ] && pf_tool_version "$NSYS_BIN" )"
    echo "ncu=${NCU_BIN:-none} $( [ -n "$NCU_BIN" ] && pf_tool_version "$NCU_BIN" )"
    echo "rustc=$(cd "$REPO" && rustc --version 2>/dev/null || echo '?')"
    echo "host=$(uname -srm) cpu=$(awk -F': ' '/^model name/ { print $2; exit }' /proc/cpuinfo 2>/dev/null || true) threads=$(nproc 2>/dev/null || echo '?')"
    echo "options: counters=$COUNTERS skip_build=$SKIP_BUILD skip_nsys=$SKIP_NSYS skip_ncu=$SKIP_NCU gpu_metrics_freq=$GPU_METRICS_FREQ sampling='$SAMPLE_FLAGS'"
  } > "$f"
}

# ======================================================================================
# build
# ======================================================================================
BUILD_ARCH="" BIN="" CUBIN_DIR=""
build_env() { # production builds: no ad-hoc rustflags, no register cap, no lineinfo
  env -u RUSTFLAGS -u CARGO_BUILD_RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS \
      -u LAMBDA_VM_RPX_MAXRREGCOUNT -u LAMBDA_VM_NVCC_LINEINFO \
      ${BUILD_ARCH:+"CUDARC_NVCC_ARCH=$BUILD_ARCH"} "$@"
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

check_test_listed() {
  local listed
  listed="$({ "$BIN" --list 2>/dev/null || true; } | awk -v t="$TEST: test" '$0 == t { n++ } END { print n + 0 }')"
  pf_log "test binary lists $TEST: $listed time(s)"
  if [ "$listed" = 1 ]; then return 0; fi
  return 1
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

  step_begin "build: prover lib tests (cargo test --release -p lambda-vm-prover --features cuda --lib --no-run)"
  rc=0
  (cd "$REPO" && build_env timeout "$BUILD_TIMEOUT" cargo test --release -p lambda-vm-prover --features cuda --lib \
     --no-run --message-format=json-render-diagnostics) > "$TMP/cargo-prover.json" 2> "$BIG/build-prover.log" || rc=$?
  tail -n 40 "$BIG/build-prover.log" > "$SMALL/build-prover.tail.txt" || true
  step_end "$rc"
  if [ "$rc" -ne 0 ]; then pf_die "prover build failed (rc=$rc): see $BIG/build-prover.log"; fi
  BIN="$(python3 "$HERE/lib/cargo_test_bin.py" exe --name lambda_vm_prover --kind lib < "$TMP/cargo-prover.json")" \
    || pf_die "cargo reported no lambda_vm_prover unit-test binary (see $TMP/cargo-prover.json)"
  CUBIN_DIR="$(python3 "$HERE/lib/cargo_test_bin.py" outdir --package math-cuda < "$TMP/cargo-prover.json")" \
    || pf_die "cargo reported no math-cuda build-script output (see $TMP/cargo-prover.json)"
  check_cubins || pf_die "the cubins are missing or empty: nvcc was not found by build.rs, so the GPU path is a CPU path"
  check_test_listed || pf_die "the built binary does not contain $TEST"
  printf 'HEAD=%s\nBIN=%s\nCUBIN_DIR=%s\nBUILT=%s\n' "$(git -C "$REPO" rev-parse HEAD)" "$BIN" "$CUBIN_DIR" \
    "$(date -u +%FT%TZ)" > "$REPO/target/block_profile.last-build"
  pf_log "test binary: $BIN"
}

reuse_build() {
  local crumb="$REPO/target/block_profile.last-build" head
  if [ ! -r "$crumb" ]; then pf_die "--skip-build: no $crumb; run once without --skip-build"; fi
  head="$(sed -n 's/^HEAD=//p' "$crumb")"
  BIN="$(sed -n 's/^BIN=//p' "$crumb")"
  CUBIN_DIR="$(sed -n 's/^CUBIN_DIR=//p' "$crumb")"
  if [ "$head" != "$(git -C "$REPO" rev-parse HEAD)" ]; then pf_die "--skip-build: the last build is of $head, HEAD has moved; rebuild"; fi
  if [ ! -x "$BIN" ]; then pf_die "--skip-build: $BIN is gone; rebuild"; fi
  check_artifacts || pf_die "--skip-build: guest ELFs are missing; rebuild"
  check_cubins || pf_die "--skip-build: the cubins are missing or empty; rebuild"
  check_test_listed || pf_die "--skip-build: $BIN does not contain $TEST"
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
  local rc reading
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
  (cd "$REPO/prover" && exec timeout --signal=INT --kill-after=300 "$RUN_A_TIMEOUT" \
     env -i "${RUN_ENV[@]}" "$NSYS_BIN" "${NSYS_ARGS[@]}" \
     "$BIN" "$TEST" --ignored --exact --nocapture --test-threads=1) > "$BIG/runA.raw.log" 2>&1 || rc=$?
  stop_sampler
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
  local spec k skip count tref rc profiled t0 bad=0 total=0
  record_env
  if [ ! -s "$SMALL/env.txt" ]; then write_env_capture; fi
  mkdir -p "$BIG/ncu" "$SMALL/ncu"
  printf 'kernel\tskip\tcount\trc\tprofiled\tseconds\n' > "$SMALL/ncu/passes.tsv"
  for spec in $NCU_KERNELS; do
    IFS=: read -r k skip count tref <<< "$spec"
    count="${NCU_COUNT:-$count}"
    total=$((total + 1))
    wait_idle
    step_begin "run B: ncu $k (launches $skip..$((skip + count - 1)) of that kernel)"
    t0="$(date +%s)"
    ncu_args "$k" "$skip" "$count" "$BIG/ncu/$k"
    rc=0
    (cd "$REPO/prover" && exec timeout --signal=INT --kill-after=120 "$NCU_PASS_TIMEOUT" \
       env -i "${RUN_ENV[@]}" "$NCU_BIN" "${NCU_ARGS[@]}" \
       "$BIN" "$TEST" --ignored --exact --nocapture --test-threads=1) > "$BIG/ncu/$k.stdout" 2>&1 || rc=$?
    profiled="$(grep -c '^==PROF== Profiling' "$BIG/ncu/$k.stdout" || true)"
    { grep -E '^==(PROF|ERROR|WARNING)==' "$BIG/ncu/$k.stdout" || true; } > "$SMALL/ncu/$k.prof.txt"
    if [ -s "$BIG/ncu/$k.ncu-rep" ]; then
      timeout 600 "$NCU_BIN" --import "$BIG/ncu/$k.ncu-rep" --page details > "$SMALL/ncu/$k.details.txt" 2>&1 || true
      timeout 600 "$NCU_BIN" --import "$BIG/ncu/$k.ncu-rep" --page details --csv > "$SMALL/ncu/$k.details.csv" 2>/dev/null || true
      timeout 600 "$NCU_BIN" --import "$BIG/ncu/$k.ncu-rep" --page raw --csv > "$BIG/ncu/$k.raw.csv" 2>/dev/null || true
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$k" "$skip" "$count" "$rc" "$profiled" "$(( $(date +%s) - t0 ))" >> "$SMALL/ncu/passes.tsv"
    # with --kill ncu ends the run itself once the launches are in, so rc alone says little
    if [ "$profiled" -lt 1 ] || [ ! -s "$BIG/ncu/$k.ncu-rep" ]; then bad=$((bad + 1)); fi
    pf_log "ncu $k: $profiled launch(es) profiled, rc=$rc"
    step_end "$rc"
  done
  if compgen -G "$BIG/ncu/*.raw.csv" > /dev/null; then
    python3 "$HERE/lib/ncu_summary.py" --out "$SMALL/ncu" "$BIG"/ncu/*.raw.csv > "$BIG/ncu/summary.log" 2>&1 || true
  fi
  if [ "$bad" -eq 0 ]; then RUN_B_VERDICT="ok ($total kernels)"; else RUN_B_VERDICT="partial ($bad of $total kernels without a profile)"; fi
}

# ======================================================================================
# plan, estimate, finish
# ======================================================================================
estimate() {
  local g=0 p=0 a=0 b=0 spec k skip count tref
  if [ "$SKIP_BUILD" = 0 ]; then
    if [ -r "$REPO/executor/program_artifacts/rust/ethrex.elf" ]; then g=3; else g=20; fi
    if [ -d "$REPO/target/release/deps" ]; then p=8; else p=20; fi
  fi
  if [ "$SKIP_NSYS" = 0 ]; then a=10; if [ "$COUNTERS" = 1 ]; then a=15; fi; fi
  pf_log "ESTIMATE (a guide, not a bound; every step has its own timeout):"
  pf_log "  guest ELFs      ~${g} min   (fresh ~20; up to date ~3: make re-runs cargo per guest)"
  pf_log "  prover build    ~${p} min   (fresh release CUDA build ~20; incremental less)"
  pf_log "  run A           ~${a} min   (~2-4 min of block run under nsys, then export + stats + summary)"
  if [ "$SKIP_NCU" = 0 ]; then
    for spec in $NCU_KERNELS; do
      IFS=: read -r k skip count tref <<< "$spec"
      count="${NCU_COUNT:-$count}"
      b="$(awk -v b="$b" -v t="${tref:-110}" -v c="$count" 'BEGIN { printf "%.0f", b + 60 + 2.5 * t + 45 * c }')"
      pf_log "  run B  $(printf '%-42s' "$k") ~$(awk -v t="${tref:-110}" -v c="$count" 'BEGIN { printf "%.1f", (60 + 2.5 * t + 45 * c) / 60 }') min"
    done
    pf_log "  run B total     ~$((b / 60)) min   (per pass: 60 s start + 2.5 x the launch's time into the run + 45 s per launch)"
  fi
  pf_log "  TOTAL           ~$(( g + p + a + b / 60 )) min"
}

finish() {
  local slug verdict rc=0 f
  slug="$(printf '%s' "$GPU_NAME" | tr '[:upper:]' '[:lower:]' | sed -e 's/nvidia//; s/geforce//; s/[^a-z0-9]\{1,\}/-/g; s/^-//; s/-$//')"
  {
    echo "OUT/small: what each file is"
    echo "  preflight.txt         every check, PASS/WARN/FAIL/INFO"
    echo "  env.txt               the exact environment of the profiled process, tools, HEAD, binary sha256"
    echo "  steps.tsv             every step: start, seconds, rc"
    echo "  driver.log            this script's own output"
    echo "  build-*.tail.txt      the last lines of each build (full logs in OUT/big)"
    echo "  runA.test.log         the test's stdout under nsys (the harness's own stage lines)"
    echo "  runA.verdict.txt      the record's gates, read back from that log"
    echo "  nvsmi_100ms.csv       nvidia-smi at 10 Hz: memory.used MiB, util %, SM clock, power (UTC)"
    echo "  nsys_stats/*.csv      nsys stats: cuda_gpu_kern_sum, cuda_api_sum, cuda_gpu_mem_{time,size}_sum, osrt_sum"
    echo "  nsys/summary.txt      READ FIRST: stages, busy/idle per 5 s, top kernels, copies, GPU metrics"
    echo "  nsys/stages.csv       per stage: wall, busy (kernel+copy union), kernel sum, VRAM peak, metrics, top kernels"
    echo "  nsys/busy_5s.csv      per 5 s: busy/idle, top kernel, VRAM peak (+ key GPU metrics)"
    echo "  nsys/kernels.csv      every kernel: launches, time, and its split by phase"
    echo "  nsys/gpu_metrics_*    (counters only) SM active / SM issue / DRAM % per 1 s, per stage, and every metric"
    echo "  ncu/ncu_summary.txt   (counters only) one line per profiled launch: SOL, issue, occupancy, stalls, pipes"
    echo "  ncu/<kernel>.details.txt|csv   ncu's own per-launch report (the authority, with its rule messages)"
    echo "  ncu/passes.tsv, ncu/<kernel>.prof.txt   each pass's outcome and ncu's own messages"
    echo
    echo "files present:"
    (cd "$SMALL" && find . -type f | sort | sed 's/^\.\//  /')
  } > "$SMALL/INDEX.txt"
  {
    echo "OUT/big (NOT for git):"
    (cd "$BIG" && find . -type f -size +0 ! -path './tmp/*' -exec ls -l {} + 2>/dev/null | awk '{ printf "  %12d  %s\n", $5, $NF }')
  } > "$SMALL/manifest.txt"

  verdict="PASS"
  if [ "$SKIP_NSYS" = 0 ] && [ "$RUN_A_VERDICT" != ok ]; then verdict="PARTIAL"; fi
  if [ "$SKIP_NCU" = 0 ] && [ "${RUN_B_VERDICT%% *}" != ok ]; then verdict="PARTIAL"; fi
  {
    echo "==================== SEND BACK ===================="
    echo "1. ALWAYS: $SMALL  ($(du -sh "$SMALL" | awk '{ print $1 }'); also packed as $OUT/small.tar.gz)"
    echo "   It holds no secret: the only environment in it is env.txt's explicit list."
    echo "   Either commit it on this branch:"
    echo "     cd $REPO"
    echo "     mkdir -p scripts/profile/results/$(date -u +%Y%m%d)-$slug"
    echo "     cp -R $SMALL/. scripts/profile/results/$(date -u +%Y%m%d)-$slug/"
    echo "     git add scripts/profile/results/$(date -u +%Y%m%d)-$slug"
    echo "     git commit -m 'profile: WHIR block run on $GPU_NAME ($(date -u +%Y-%m-%d))'"
    echo "     git push origin HEAD:whir/profile-rpx"
    echo "   or send $OUT/small.tar.gz."
    echo "2. ONLY IF ASKED (large; never into git): $BIG/blockA.nsys-rep (Nsight Systems GUI),"
    echo "   $BIG/ncu/*.ncu-rep (Nsight Compute GUI). Sizes in small/manifest.txt."
    echo "RESULT: run A $RUN_A_VERDICT · run B $RUN_B_VERDICT"
    echo "BLOCK_PROFILE VERDICT: $verdict"
  } > "$SMALL/SEND-BACK.txt"
  tar -czf "$OUT/small.tar.gz" -C "$OUT" small || rc=$?
  cat "$SMALL/SEND-BACK.txt"
  if [ "$verdict" != PASS ]; then return 3; fi
  return "$rc"
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
