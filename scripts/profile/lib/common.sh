# shellcheck shell=bash
# Shared helpers for scripts/profile/*.sh. Sourced (after `set -euo pipefail`), never run.
#
# Everything here serves one purpose: a profile taken on SOMEONE ELSE'S box must be
# trustworthy without anyone watching it run.
#   * Tools are identified by what they print, not by their name: an `ncu` on PATH can
#     be npm-check-updates.
#   * "Counters are unlocked" is proven by profiling one real kernel, not inferred.
#   * The profiled process runs under an explicit environment (env -i + an allowlist):
#     no knob from the caller's shell reaches it, and — because nsys and ncu store the
#     target's environment verbatim in their reports — no token or key from that shell
#     ends up in a file that gets sent around.
# Written for `set -euo pipefail`: no `cmd | grep -q` (SIGPIPE under pipefail), no
# `| head`, and no `test && action` as the last statement of a function.

pf_log() { printf 'PROF %s %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
pf_die() { pf_log "FATAL: $*"; exit 1; }

# pf_ver_ge HAVE WANT — true when the dotted version HAVE >= WANT (2025.4.1.172 >= 2025.1).
pf_ver_ge() {
  awk -v a="$1" -v b="$2" 'BEGIN {
    n = split(a, x, "."); m = split(b, y, "."); k = (n > m) ? n : m
    for (i = 1; i <= k; i++) {
      xi = (i <= n) ? x[i] + 0 : 0; yi = (i <= m) ? y[i] + 0 : 0
      if (xi > yi) exit 0
      if (xi < yi) exit 1
    }
    exit 0 }'
}

# The toolkit math-cuda's build.rs compiles with: CUDA_HOME, else CUDA_PATH, else /usr/local/cuda.
# ⚠ When $(pf_cuda_home)/bin/nvcc is missing the build does NOT fail: it writes EMPTY cubin
# stubs and every kernel silently falls back to the CPU. Hence the checks on nvcc and on the
# cubins' sizes in the drivers.
pf_cuda_home() { printf '%s\n' "${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}"; }

# pf_first_match REGEX — the first substring of stdin matching an extended regex (awk reads
# to EOF, so no SIGPIPE upstream); prints nothing when absent. ⚠ awk -v processes escapes, so
# write a literal dot as [.], never \. (which arrives as "any character").
pf_first_match() {
  awk -v re="$1" '!done && match($0, re) { print substr($0, RSTART, RLENGTH); done = 1 }'
}

# pf_find_tool nsys|ncu — the path of the real Nsight binary, verified by its banner.
pf_find_tool() {
  local tool="$1" banner c out
  local -a cands=()
  case "$tool" in
    nsys)
      banner='Nsight Systems'
      cands=("${NSYS:-}")
      while IFS= read -r c; do cands+=("$c"); done < <(type -ap nsys 2>/dev/null || true)
      cands+=("$(pf_cuda_home)/bin/nsys")
      # shellcheck disable=SC2012 # version directories, newest first: sort -V orders 2025.10 after 2025.4
      while IFS= read -r c; do cands+=("$c"); done < <(ls -d /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | sort -rV || true)
      ;;
    ncu)
      banner='Nsight Compute'
      cands=("${NCU:-}")
      while IFS= read -r c; do cands+=("$c"); done < <(type -ap ncu 2>/dev/null || true)
      cands+=("$(pf_cuda_home)/bin/ncu")
      # shellcheck disable=SC2012 # as above
      while IFS= read -r c; do cands+=("$c"); done < <(ls -d /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | sort -rV || true)
      ;;
    *) return 2 ;;
  esac
  for c in "${cands[@]}"; do
    if [ -z "$c" ] || [ ! -x "$c" ]; then continue; fi
    out="$("$c" --version 2>/dev/null || true)"
    case "$out" in
      *"$banner"*) printf '%s\n' "$c"; return 0 ;;
    esac
  done
  return 1
}

# pf_tool_version PATH — the YYYY.N[.N...] release an Nsight binary reports.
pf_tool_version() { { "$1" --version 2>/dev/null || true; } | pf_first_match '20[0-9][0-9][.][0-9]+([.][0-9]+)*'; }

# pf_gpu_field INDEX FIELD — one `nvidia-smi --query-gpu` field for one GPU, trimmed.
pf_gpu_field() {
  { nvidia-smi -i "$1" --query-gpu="$2" --format=csv,noheader,nounits 2>/dev/null || true; } |
    awk 'NR == 1 { gsub(/^ +| +$/, ""); print }'
}

# pf_gpu_idle INDEX MAX_MIB — 0 when the GPU has < MAX_MIB in use and no compute process.
# Prints the reading either way (a guard silent on its accepting path reads like one never run).
pf_gpu_idle() {
  local used apps
  used="$(pf_gpu_field "$1" memory.used)"
  apps="$({ nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true; } | awk 'NF { n++ } END { print n + 0 }')"
  echo "gpu $1: ${used:-?} MiB in use, ${apps} compute process(es)"
  if [ -n "$used" ] && [ "$used" -lt "$2" ] && [ "$apps" -eq 0 ]; then return 0; fi
  return 1
}

# pf_base_env — sets PF_ENV=(NAME=value ...): the system part of the environment every
# profiled process gets, plus the driver's PF_EXTRA_ENV (its GPU pin), and nothing else
# from the caller's shell.
pf_base_env() {
  local v
  PF_ENV=("PATH=$PATH")
  for v in HOME USER LOGNAME LANG LC_ALL TERM TMPDIR XDG_RUNTIME_DIR LD_LIBRARY_PATH CUDA_HOME CUDA_PATH; do
    if [ -n "${!v:-}" ]; then PF_ENV+=("$v=${!v}"); fi
  done
  PF_ENV+=(${PF_EXTRA_ENV[@]+"${PF_EXTRA_ENV[@]}"})
}

# pf_sha256 FILE — hex digest (sha256sum, else shasum: the Makefile's own idiom).
pf_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{ print $1 }'
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{ print $1 }'
  else return 1
  fi
}

# pf_build_probe DIR GPU_INDEX — compile the one-kernel CUDA probe the permission checks
# profile; prints its path, or returns 1 (the reason is in DIR/pf_probe.build.log).
pf_build_probe() {
  local dir="$1" cc
  cc="$(pf_gpu_field "$2" compute_cap | tr -d .)"
  mkdir -p "$dir"
  cat > "$dir/pf_probe.cu" <<'CU'
#include <cstdio>
extern "C" __global__ void pf_probe_kernel(float *x) { x[threadIdx.x] = 2.0f * threadIdx.x; }
int main() {
  float *d = nullptr;
  if (cudaMalloc(&d, 256 * sizeof(float)) != cudaSuccess) { std::printf("pf_probe: cudaMalloc failed\n"); return 2; }
  pf_probe_kernel<<<1, 256>>>(d);
  cudaError_t e = cudaDeviceSynchronize();
  std::printf("pf_probe: %s\n", cudaGetErrorString(e));
  return e == cudaSuccess ? 0 : 3;
}
CU
  if [ -z "$cc" ]; then echo "no compute capability for GPU $2" > "$dir/pf_probe.build.log"; return 1; fi
  if ! timeout 300 "$(pf_cuda_home)/bin/nvcc" -O2 -arch="sm_${cc}" -o "$dir/pf_probe" "$dir/pf_probe.cu" \
       > "$dir/pf_probe.build.log" 2>&1; then
    return 1
  fi
  printf '%s\n' "$dir/pf_probe"
}

# pf_counter_probe NCU PROBE_BIN LOG [NCU ARGS...] — profile the probe's one kernel under ncu,
# in the same clean environment the real runs use, with the caller's ncu arguments (pass the
# real runs' own, so a flag this ncu rejects fails here and not an hour later; they must
# include SpeedOfLight). Default: --section SpeedOfLight -c 1. Prints exactly one of:
#   COUNTERS unlocked | COUNTERS locked (ERR_NVGPUCTRPERM) | COUNTERS unknown (<why>)
pf_counter_probe() {
  local ncu="$1" bin="$2" log="$3" rc=0 out
  shift 3
  if [ $# -eq 0 ]; then set -- --section SpeedOfLight -c 1; fi
  pf_base_env
  out="$(env -i "${PF_ENV[@]}" timeout 300 "$ncu" "$@" "$bin" 2>&1)" || rc=$?
  printf '%s\n' "$out" > "$log"
  case "$out" in
    *ERR_NVGPUCTRPERM*) echo "COUNTERS locked (ERR_NVGPUCTRPERM)" ;;
    *"pf_probe_kernel"*Duration*)
      if [ "$rc" -eq 0 ]; then echo "COUNTERS unlocked"; else echo "COUNTERS unknown (ncu rc=$rc after profiling; see $log)"; fi ;;
    *) echo "COUNTERS unknown (ncu rc=$rc and no profile of the probe kernel; see $log)" ;;
  esac
}

# pf_nsys_probe NSYS PROBE_BIN REP WANT_METRICS NSYS_ARGS... — run the probe under nsys with the
# caller's `profile ... -o REP` arguments (pass the real run's own, so a flag this nsys rejects
# fails here), and, when WANT_METRICS is 1, export it and count its GPU-metric samples. Prints:
#   NSYS ok (<n> GPU-metric samples) | NSYS ok (no GPU metrics requested) | NSYS fail (<why>)
pf_nsys_probe() {
  local nsys="$1" bin="$2" rep="$3" want="$4" rc=0 n
  shift 4
  pf_base_env
  env -i "${PF_ENV[@]}" timeout 300 "$nsys" "$@" "$bin" > "$rep.log" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ] || [ ! -s "$rep.nsys-rep" ]; then
    echo "NSYS fail (nsys profile rc=$rc; see $rep.log)"; return 0
  fi
  if [ "$want" != 1 ]; then echo "NSYS ok (no GPU metrics requested)"; return 0; fi
  rc=0
  timeout 300 "$nsys" export --type sqlite --force-overwrite true --output "$rep.sqlite" \
    "$rep.nsys-rep" >> "$rep.log" 2>&1 || rc=$?
  n="$(python3 - "$rep.sqlite" <<'PY' 2>>"$rep.log" || true
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
try:
    print(db.execute("SELECT count(*) FROM GPU_METRICS").fetchone()[0])
except sqlite3.Error:
    print(0)
PY
)"
  if [ "$rc" -eq 0 ] && [ "${n:-0}" -gt 0 ]; then echo "NSYS ok ($n GPU-metric samples)"
  else echo "NSYS fail (GPU metrics requested, ${n:-0} samples exported, export rc=$rc; see $rep.log)"
  fi
}

# pf_sampling_flags NSYS — nsys CPU-sampling flags this box supports: process-tree sampling
# when `nsys status --environment` says so, else none (a container without perf events).
pf_sampling_flags() {
  local out
  out="$({ "$1" status --environment 2>&1 || true; } | tr -d '\r')"
  case "$out" in
    *"CPU Profiling Environment (process-tree): OK"*) echo "--sample=process-tree --cpuctxsw=process-tree" ;;
    *) echo "--sample=none --cpuctxsw=none" ;;
  esac
}

# pf_metrics_flag NSYS — the GPU-metrics device option this nsys spells (plural since 2024.x).
pf_metrics_flag() {
  local out
  out="$({ "$1" profile --help 2>&1 || true; })"
  case "$out" in
    *--gpu-metrics-devices*) echo "--gpu-metrics-devices" ;;
    *--gpu-metrics-device*) echo "--gpu-metrics-device" ;;
    *) echo "" ;;
  esac
}
