#!/usr/bin/env bash
# whir_ncu.sh — Nsight Compute on two micro-benches of the WHIR pipeline's hot kernels, on a
# box whose GPU performance counters are unlocked (NVreg_RestrictProfilingToAdminUsers=0, or
# root). Smaller and quicker than block_profile.sh's run B, and a different question: the
# kernels in isolation at controlled shapes, not inside the block run. Adds nothing to the
# prover: both benches are #[ignore] tests already on this branch.
#
# THE BENCHES
#   1. prover/tests/rpx_occupancy_sweep.rs :: how_occupancy_moves_the_rpx_permutation
#      the RPX permutation probe (rpx_permute_probe: 2^20 states x 256 launches) and the
#      production grind (rpx_grind_search: factor 20, scan 8, grid 1024)
#   2. crypto/math-cuda/tests/sumcheck.rs :: ncu_sumcheck_round_block_scale
#      six block-scale rounds (num_vars 21, width 4, degree 5): sumcheck_round_ext3,
#      sum_partials_ext3, sumcheck_fold_ext3
#   3. optional (nsys present and the ethrex guest built): the pure WHIR prover's timeline on
#      the in-repo ethrex_10_transfers input (multilinear_bench_tests::continuations, backend whir)
#
# WHAT IT ANSWERS (each a question a counter-locked box cannot)
#   1. The RPX permutation: rpx_grind_search is the largest kernel of the block run (17.2 s of
#      84.9 s of kernel time in the lead's wt90 nsys trace) and the two hashing passes run the
#      same permutation. Our register-cap occupancy sweep was flat (2.81-2.84 ns/perm from 8 to
#      12 blocks/SM), read as "issue-bound at its arithmetic floor". ncu tells that apart from
#      LATENCY-bound (the inverse S-box is a 72-multiply dependent chain, little ILP): SOL SM %
#      against issue-slot use, and the WarpStateStats stall reasons (long/short scoreboard and
#      wait, against math-pipe throttle). Latency-bound => interleaving 2-4 independent
#      permutations per thread is a lever; issue-bound => the floor stands.
#   2. The WHIR sumcheck round kernel (sumcheck_round_ext3, 7.0 s of kernel time in wt90): is
#      the kernel itself under-occupied, or is the argue stage's idle only the per-round host
#      sync? ncu on block-scale round launches answers the first half; nsys the second.
#
# USAGE (repo checkout of whir/profile-rpx; needs cargo, CUDA with nvcc, Nsight Compute)
#   bash scripts/profile/whir_ncu.sh [out-dir]      # default scripts/profile/out/whir-ncu-<UTC>
# ENV: GPU (index, default 0) · NCU / NSYS (tool paths) · STEP_TIMEOUT (seconds, default 1800)
# Leaves OUT/small (text: read ncu_summary.txt, then *.details.txt), packed as OUT/small.tar.gz once
# the bundle scan passes, and OUT/big (the .ncu-rep/.nsys-rep/.sqlite, raw exports, build logs).
# OUT/big EMBEDS THE MACHINE'S ENVIRONMENT: never commit, share or upload it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
. "$HERE/lib/common.sh"
REPO="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null)" || pf_die "not inside a git checkout: $HERE"
case "${1:-}" in -h|--help) sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;; esac

GPU="${GPU:-0}"
STEP_TIMEOUT="${STEP_TIMEOUT:-1800}"
PF_EXTRA_ENV=("CUDA_DEVICE_ORDER=PCI_BUS_ID" "CUDA_VISIBLE_DEVICES=$GPU")
OUT="${1:-$HERE/out/whir-ncu-$(date -u +%Y%m%dT%H%M%SZ)}"
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then pf_die "$OUT exists and is not empty"; fi
mkdir -p "$OUT/small" "$OUT/big"
OUT="$(cd "$OUT" && pwd)"
SMALL="$OUT/small" BIG="$OUT/big" # small leaves the box once the bundle scan passes; big never does
exec > >(tee -a "$SMALL/run.log") 2>&1
cd "$REPO"

# --- 0. tools, and counters proven on one real kernel -------------------------------------
NVCC="$(pf_cuda_home)/bin/nvcc"
[ -x "$NVCC" ] || pf_die "no nvcc at $NVCC: math-cuda's build.rs would write EMPTY cubins and every kernel would run on the CPU. Set CUDA_HOME."
NCU_BIN="$(pf_find_tool ncu)" || pf_die "Nsight Compute not found (PATH, \$CUDA_HOME/bin, /opt/nvidia/nsight-compute/*); set NCU=/path/to/ncu"
NSYS_BIN="$(pf_find_tool nsys || true)"
for t in cargo python3 timeout nvidia-smi; do command -v "$t" >/dev/null 2>&1 || pf_die "$t not found"; done
pf_log "repo HEAD $(git rev-parse --short=9 HEAD) · gpu $GPU: $(pf_gpu_field "$GPU" name), $(pf_gpu_field "$GPU" memory.total) MiB, driver $(pf_gpu_field "$GPU" driver_version)"
pf_log "ncu $NCU_BIN ($(pf_tool_version "$NCU_BIN")) · nsys ${NSYS_BIN:-<none>} · nvcc $NVCC"
SECTIONS=(--section SpeedOfLight --section SpeedOfLight_RooflineChart --section ComputeWorkloadAnalysis
          --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats --section Occupancy
          --section LaunchStats --section InstructionStats --section SourceCounters)
PROBE="$(pf_build_probe "$BIG/probe" "$GPU")" || pf_die "could not build the one-kernel probe: see $BIG/probe/pf_probe.build.log"
# the passes' own flags, so a section or option this ncu rejects fails here, before the builds
COUNTERS="$(pf_counter_probe "$NCU_BIN" "$PROBE" "$BIG/probe/pf_probe.ncu.log" "${SECTIONS[@]}" \
  -k 'regex:^pf_probe_kernel$' -c 1 -f -o "$BIG/probe/pf_probe_ncu")"
pf_log "$COUNTERS"
if [ "$COUNTERS" != "COUNTERS unlocked" ]; then
  pf_log "ncu cannot read the counters here: unlock them (README) or run as root, then re-run"; exit 2
fi
if ! reading="$(pf_gpu_idle "$GPU" 500)"; then pf_die "card not idle: $reading"; fi

# --- 1. build the bench binaries (production cubins: no register cap, no lineinfo) --------
build_bench() { # build_bench <tag> <cargo test args...> ; leaves $BIG/cargo-<tag>.json
  local tag="$1" rc=0
  shift
  pf_log "build: cargo test $* --no-run"
  env -u RUSTFLAGS -u CARGO_BUILD_RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS -u LAMBDA_VM_RPX_MAXRREGCOUNT -u LAMBDA_VM_NVCC_LINEINFO \
    timeout 7200 cargo test "$@" --no-run --message-format=json-render-diagnostics \
    > "$BIG/cargo-$tag.json" 2> "$BIG/build-$tag.log" || rc=$?
  if [ "$rc" -ne 0 ]; then tail -20 "$BIG/build-$tag.log"; pf_die "build failed (rc=$rc): $BIG/build-$tag.log"; fi
}
check_cubins() { # an empty cubin is build.rs's no-nvcc stub
  local dir f n=0 empty=0
  dir="$(python3 "$HERE/lib/cargo_test_bin.py" outdir --package math-cuda < "$1")" || pf_die "no math-cuda build output in $1"
  for f in "$dir"/*.cubin; do
    if [ ! -e "$f" ]; then continue; fi
    n=$((n + 1))
    if [ ! -s "$f" ]; then empty=$((empty + 1)); fi
  done
  pf_log "cubins: $n in $dir, $empty empty"
  if [ "$n" -eq 0 ] || [ "$empty" -ne 0 ]; then pf_die "missing or empty cubins: the GPU path would be a CPU path"; fi
}
build_bench rpx -p lambda-vm-prover --release --features cuda --test rpx_occupancy_sweep
RPX_BIN="$(python3 "$HERE/lib/cargo_test_bin.py" exe --name rpx_occupancy_sweep --kind test < "$BIG/cargo-rpx.json")" || pf_die "no rpx_occupancy_sweep binary"
check_cubins "$BIG/cargo-rpx.json"
build_bench sumcheck -p math-cuda --release --test sumcheck
SC_BIN="$(python3 "$HERE/lib/cargo_test_bin.py" exe --name sumcheck --kind test < "$BIG/cargo-sumcheck.json")" || pf_die "no sumcheck test binary"
check_cubins "$BIG/cargo-sumcheck.json"
pf_log "binaries: $RPX_BIN ; $SC_BIN"

# --- 2./3. ncu: one pass per question, full sections, in the clean environment ------------
pf_base_env
ncu_pass() { # ncu_pass <name> <kernel regex> <launch count> <binary> <test name>
  local name="$1" regex="$2" count="$3" bin="$4" test="$5" rc=0
  pf_log "ncu $name: -k regex:$regex -c $count ($test)"
  env -i "${PF_ENV[@]}" timeout --signal=INT --kill-after=60 "$STEP_TIMEOUT" "$NCU_BIN" "${SECTIONS[@]}" \
    -k "regex:$regex" -c "$count" -f -o "$BIG/$name" "$bin" "$test" --exact --ignored --nocapture \
    > "$BIG/$name.stdout" 2>&1 || rc=$?
  { grep -E '^==(PROF|ERROR|WARNING)==' "$BIG/$name.stdout" || true; } > "$SMALL/$name.prof.txt"
  pf_log "ncu $name: rc=$rc, $(grep -c '^==PROF== Profiling' "$BIG/$name.stdout" || true) launch(es) profiled"
}
ncu_pass rpx-permute '^rpx_permute_probe$' 2 "$RPX_BIN" how_occupancy_moves_the_rpx_permutation
ncu_pass rpx-grind '^rpx_grind_search$' 2 "$RPX_BIN" how_occupancy_moves_the_rpx_permutation
ncu_pass sumcheck-round '^(sumcheck_round_ext3|sum_partials_ext3|sumcheck_fold_ext3)$' 3 "$SC_BIN" ncu_sumcheck_round_block_scale

# --- 4. exports: ncu's own text first, then one line per launch ----------------------------
for r in "$BIG"/*.ncu-rep; do
  if [ ! -e "$r" ]; then continue; fi
  n="$(basename "${r%.ncu-rep}")"
  timeout 600 "$NCU_BIN" --import "$r" --page details > "$SMALL/$n.details.txt" 2>&1 || true
  timeout 600 "$NCU_BIN" --import "$r" --page details --csv > "$SMALL/$n.details.csv" 2>/dev/null || true
  timeout 600 "$NCU_BIN" --import "$r" --page raw --csv > "$BIG/$n.raw.csv" 2>/dev/null || true
done
if compgen -G "$BIG/*.raw.csv" > /dev/null; then
  python3 "$HERE/lib/ncu_summary.py" --out "$SMALL" "$BIG"/*.raw.csv > /dev/null 2>&1 || pf_log "ncu_summary.py failed (the .details.txt files stand)"
fi

# --- 5. optional nsys timeline of the pure WHIR prover on the in-repo workload ---------------
if [ -z "$NSYS_BIN" ]; then
  pf_log "nsys not found: timeline stage skipped"
elif [ ! -r executor/program_artifacts/rust/ethrex.elf ]; then
  pf_log "the ethrex guest is not built: timeline stage skipped (make compile-programs-rust SYSROOT_DIR=\$HOME/.lambda-vm-sysroot)"
else
  build_bench prover -p lambda-vm-prover --release --features cuda --lib
  PROVER_BIN="$(python3 "$HERE/lib/cargo_test_bin.py" exe --name lambda_vm_prover --kind lib < "$BIG/cargo-prover.json")" || pf_die "no prover test binary"
  pf_log "nsys: pure WHIR prover, ethrex_10_transfers (in-repo), backend whir"
  SAMPLING="$(pf_sampling_flags "$NSYS_BIN")"
  timeline() { # one traced run with the given CPU-sampling flags
    local -a sflags
    read -r -a sflags <<< "$1"
    (cd "$REPO/prover" && exec env -i "${PF_ENV[@]}" LAMBDA_VM_BENCH_BACKEND=whir LAMBDA_VM_BENCH_ELF=ethrex \
       LAMBDA_VM_BENCH_INPUT=ethrex_10_transfers timeout --signal=INT --kill-after=120 "$STEP_TIMEOUT" \
       "$NSYS_BIN" profile --trace="cuda,nvtx,osrt" "${sflags[@]}" --stats=false -f true -o "$BIG/whir-prover-timeline" \
       "$PROVER_BIN" tests::multilinear_bench_tests::continuations --exact --ignored --nocapture) > "$BIG/nsys.stdout" 2>&1
  }
  rc=0
  timeline "$SAMPLING" || rc=$?
  if [ ! -s "$BIG/whir-prover-timeline.nsys-rep" ] && [ "$SAMPLING" != "--sample=none --cpuctxsw=none" ]; then
    pf_log "nsys wrote no report with CPU sampling on (rc=$rc); CPU sampling is optional: retrying without it"
    rc=0
    timeline "--sample=none --cpuctxsw=none" || rc=$?
  fi
  pf_log "nsys: rc=$rc"
  { tr '\r' '\n' < "$BIG/nsys.stdout" | grep -vE '^\[[0-9]+/[0-9]+\] +\[' || true; } > "$SMALL/nsys.test.log"
  if [ -s "$BIG/whir-prover-timeline.nsys-rep" ]; then
    timeout "$STEP_TIMEOUT" "$NSYS_BIN" export --type sqlite --force-overwrite true --output "$BIG/whir-prover-timeline.sqlite" \
      "$BIG/whir-prover-timeline.nsys-rep" > /dev/null 2>&1 || true
    timeout "$STEP_TIMEOUT" "$NSYS_BIN" stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv \
      --output "$SMALL/whir-prover-timeline" "$BIG/whir-prover-timeline.sqlite" > /dev/null 2>&1 || true
    python3 "$HERE/lib/nsys_block_summary.py" --sqlite "$BIG/whir-prover-timeline.sqlite" --log "$SMALL/nsys.test.log" \
      --out "$SMALL/whir-prover-timeline.summary" --quiet > /dev/null 2>&1 || pf_log "nsys summary failed"
  fi
fi

# --- 6. the bundle: only OUT/small leaves the box, and only when the scan passes -------------
rm -f "$OUT/small.tar.gz"
if scan="$(pf_scan_bundle "$SMALL" "$BIG/bundle_scan.txt")"; then
  tar -czf "$OUT/small.tar.gz" -C "$OUT" small
  pf_log "$scan"
  pf_log "DONE: send back $OUT/small.tar.gz ($(du -h "$OUT/small.tar.gz" | cut -f1)). Read first: ncu_summary.txt, then *.details.txt"
  pf_log "NEVER commit, share or upload $BIG: its .ncu-rep/.nsys-rep/.sqlite embed this machine's environment"
else
  pf_log "$scan"
  pf_log "NOTHING TO SEND: read and remove the flagged lines first (the findings list is in OUT/big and is not to be shared); no tarball written"
  exit 4
fi
