#!/usr/bin/env bash
# whir_ncu.sh — GPU profiling of the WHIR-recursion hot kernels on a profiler-capable box (ncu needs perf counters unlocked:
# NVreg_RestrictProfilingToAdminUsers=0 or root). Everything it profiles already exists on this branch; it adds nothing to the prover.
#
# Run from the repo root of a checkout of whir/profile-rpx (contains the two #[ignore] benches from whir/lfm-o6-grind @ 7f319cc18):
#   bash scripts/profile/whir_ncu.sh [out-dir]
# Needs: cargo, nvcc, ncu (Nsight Compute CLI); nsys optional. Builds with --features cuda. Leaves a tarball to send back.
#
# WHAT IT ANSWERS (each is a question our own box could not, its counters being driver-locked):
#   1. RPX permutation — the single largest cost of the block run (~45 s of 131.6 s: the grind + two hashing passes). Our occupancy
#      sweep was FLAT (2.81–2.84 ns/perm across 8→12 blocks/SM), read as "compute/issue-bound at its arithmetic floor". ncu decides
#      between that and per-thread LATENCY-bound (the inverse S-box is a 72-multiply dependent chain; low ILP): SpeedOfLight SM% vs
#      issue-slot utilisation, WarpStateStats stall reasons (long-scoreboard/wait vs math-pipe-throttle), pipe utilisation.
#      If latency-bound: interleaving 2–4 independent permutations per thread is a lever; if issue-bound: the floor stands.
#   2. WHIR sumcheck round kernel (argue, ~16 s of the base) — is the kernel itself under-occupied, or is the idle only the per-round
#      host sync (which the CUDA-event probe already measured at ~9.5 s of 16.6 s)? ncu on one block-scale round launch.
#   3. (optional, nsys) the pure WHIR prover timeline on the in-repo ethrex_10_transfers workload: kernel gaps per epoch.
set -euo pipefail
OUT="${1:-profile-out/$(date -u +%Y%m%dT%H%M%SZ)}"; mkdir -p "$OUT"
log(){ echo "PROF $(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/run.log"; }
need(){ command -v "$1" >/dev/null 2>&1 || { log "MISSING: $1"; return 1; }; }
need cargo; need nvcc; need ncu
HAVE_NSYS=0; command -v nsys >/dev/null 2>&1 && HAVE_NSYS=1
log "repo HEAD $(git rev-parse --short=9 HEAD) ; gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)"
log "ncu $(ncu --version | tail -1)"
# --- 0. can ncu read counters here? (our box could not: ERR_NVGPUCTRPERM) ---
if ! ncu --query-metrics >/dev/null 2>"$OUT/ncu-query.err"; then
  log "ncu cannot query metrics: $(head -2 "$OUT/ncu-query.err" | tr '\n' ' ') — if ERR_NVGPUCTRPERM, unlock perf counters (admin) and re-run"; exit 2
fi

# --- 1. build the two bench binaries (no run) ---
log "build: cargo test -p lambda-vm-prover --release --features cuda --test rpx_occupancy_sweep --no-run"
cargo test -p lambda-vm-prover --release --features cuda --test rpx_occupancy_sweep --no-run 2>&1 | tail -3 | tee -a "$OUT/run.log"
RPX_BIN=$(ls -t target/release/deps/rpx_occupancy_sweep-* | grep -v '\.d$' | head -1)
log "build: cargo test -p math-cuda --release --test sumcheck --no-run"
cargo test -p math-cuda --release --test sumcheck --no-run 2>&1 | tail -3 | tee -a "$OUT/run.log"
SC_BIN=$(ls -t target/release/deps/sumcheck-* | grep -v '\.d$' | head -1)
log "binaries: $RPX_BIN ; $SC_BIN"

# --- 2. RPX: the permutation probe + the grind, one launch each, full sections ---
SECTIONS="--section SpeedOfLight --section SpeedOfLight_RooflineChart --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats --section Occupancy --section LaunchStats --section InstructionStats --section SourceCounters"
log "ncu 2a: RPX permutation probe kernel(s) (regex 'rpx.*(permute|probe|leaves)'), 2 launches"
ncu $SECTIONS -k 'regex:rpx.*(permute|probe|leaves)' -c 2 -f -o "$OUT/rpx-permute" \
  "$RPX_BIN" how_occupancy_moves_the_rpx_permutation --exact --ignored --nocapture > "$OUT/rpx-permute.stdout" 2>&1 || log "ncu 2a rc=$? (see rpx-permute.stdout)"
log "ncu 2b: RPX grind kernel (regex 'rpx_grind_search'), 2 launches"
ncu $SECTIONS -k 'regex:rpx_grind_search' -c 2 -f -o "$OUT/rpx-grind" \
  "$RPX_BIN" how_occupancy_moves_the_rpx_permutation --exact --ignored --nocapture > "$OUT/rpx-grind.stdout" 2>&1 || log "ncu 2b rc=$? (see rpx-grind.stdout)"

# --- 3. sumcheck round kernel at block scale (O6's micro-bench), 1 launch each of the three kernels ---
log "ncu 3: sumcheck_round_ext3 / sum_partials_ext3 / sumcheck_fold_ext3 at num_vars=21"
ncu $SECTIONS -k 'regex:sumcheck_round_ext3|sum_partials_ext3|sumcheck_fold_ext3' -c 3 -f -o "$OUT/sumcheck-round" \
  "$SC_BIN" ncu_sumcheck_round_block_scale --exact --ignored --nocapture > "$OUT/sumcheck-round.stdout" 2>&1 || log "ncu 3 rc=$? (see sumcheck-round.stdout)"

# --- 4. text exports of every report (what to read first) ---
for r in "$OUT"/*.ncu-rep; do
  ncu --import "$r" --page details --csv > "${r%.ncu-rep}.details.csv" 2>/dev/null || true
  ncu --import "$r" --page raw --csv > "${r%.ncu-rep}.raw.csv" 2>/dev/null || true
  ncu --import "$r" --page details 2>/dev/null | grep -E 'Kernel Name|Duration|SM Busy|Issue Slot|Compute \(SM\) Throughput|Memory Throughput|Achieved Occupancy|Theoretical Occupancy|Registers Per Thread|Stall|No Eligible|Eligible Warps|Warp Cycles Per Issued' | head -60 > "${r%.ncu-rep}.summary.txt" || true
done

# --- 5. optional nsys timeline of the pure WHIR prover on the in-repo reference workload ---
if [ "$HAVE_NSYS" = 1 ]; then
  log "nsys: pure WHIR prover, ethrex_10_transfers (in-repo), backend=whir"
  cargo test -p lambda-vm-prover --release --features cuda --lib tests::multilinear_bench_tests::continuations --no-run 2>&1 | tail -1 | tee -a "$OUT/run.log"
  PROVER_BIN=$(ls -t target/release/deps/lambda_vm_prover-* | grep -v '\.d$' | head -1)
  LAMBDA_VM_BENCH_BACKEND=whir LAMBDA_VM_BENCH_ELF=ethrex LAMBDA_VM_BENCH_INPUT=ethrex_10_transfers \
    nsys profile -t cuda,nvtx,osrt --cuda-memory-usage=true -f true -o "$OUT/whir-prover-timeline" \
    "$PROVER_BIN" tests::multilinear_bench_tests::continuations --exact --ignored --nocapture > "$OUT/nsys.stdout" 2>&1 || log "nsys rc=$? (see nsys.stdout)"
  nsys stats --report cuda_gpu_kern_sum,cuda_gpu_trace --format csv -o "$OUT/whir-prover-timeline" "$OUT/whir-prover-timeline.nsys-rep" >/dev/null 2>&1 || true
else
  log "nsys not found — timeline stage skipped"
fi

# --- 6. bundle ---
TAR="${OUT%/}.tar.gz"; tar -czf "$TAR" -C "$(dirname "$OUT")" "$(basename "$OUT")"
log "DONE — send back: $TAR ($(du -h "$TAR" | cut -f1)). Read first: *.summary.txt"
