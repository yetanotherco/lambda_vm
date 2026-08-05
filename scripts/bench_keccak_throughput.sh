#!/usr/bin/env bash
#
# bench_keccak_throughput.sh — how many keccak-f[1600] permutations per second
# does the prover actually prove?
#
# This is an ABSOLUTE-throughput bench, not an A/B one. Use scripts/bench_abba.sh
# when you want to detect a ~1% regression between two refs; use this when you
# want the standalone number ("we do N keccaks/sec on this box") and how it
# scales with workload size.
#
# Method
# ------
# For each N in the sweep it generates a guest that does exactly N keccak-f
# permutations (scripts/gen_keccak_bench.sh), then:
#
#   1. GATE — runs `cli execute --cycles` and asserts the executor's own
#      `Keccak calls` equals N. The throughput number divides by this count, so
#      it is read back from the VM rather than trusted from the loop constant;
#      a miscompiled or mis-unrolled guest fails the bench instead of silently
#      inflating the result.
#   2. SIZE — runs `cli count-elements` for the committed trace size, and
#      converts to base cell-equivalents as `main + 3*aux`. `count_elements`
#      returns aux as committed EXTENSION-field columns x rows
#      (prover/src/lib.rs), and the extension is cubic, so each aux element is
#      3 base cells. This gives a cells/sec reading alongside keccaks/sec —
#      they should track each other across the sweep, and a divergence means
#      something other than keccak started dominating.
#
#      That total includes the FIXED preprocessed tables (the ~2^20-row shared
#      BITWISE table and friends), which do not scale with N — about 12.5M cell-
#      equiv of floor. So `cells/perm` is a whole-trace average that falls as N
#      grows, not the marginal cost of one permutation. The report therefore also
#      differences adjacent sweep points to recover the MARGINAL cells/perm,
#      which is the number to compare against the hand-derived model
#      (24 rows x (1,480 main + 3*ceil(1031/2) aux) = 72,672 for KECCAK_RND after
#      #889, plus the ~750 of keccak.rs sponge wrapper).
#   3. TIME — one discarded warm-up prove, then RUNS timed proves, median
#      reported. Warm-up matters: the first prove in a process pays page-cache
#      and allocator warm-up worth several percent.
#
# Padding note: KECCAK_RND commits 24 rows per permutation and the trace is
# padded to a power of two, so throughput is not flat in N. The default sweep
# uses N values where 24*N lands just under a power of two (1365 -> 32,760 rows
# vs 2^15; 5461 -> 131,064 vs 2^17; 21845 -> 524,280 vs 2^19), which measures
# peak throughput. N=5000 is included because it is the historical figure the
# HWSL-inline bench (#889) was recorded at — it wastes ~8% of the table to
# padding, so expect it to read slightly slower than 5461 despite being smaller.
#
# Usage:
#   scripts/bench_keccak_throughput.sh [options]
#
#   -n "N1 N2 ..."     permutation counts to sweep
#                        CPU default: "1365 5000 5461 21845"
#                        GPU default: "10922 21845 43690" (see the threshold note)
#   -r RUNS            timed runs per point, median reported (default: 5)
#   --blowup B         FRI blowup factor (default: 2, the prover default)
#   --epoch-size-log2 K   continuation epoch size (default: CLI default, 20)
#   --monolithic       prove without continuations (default: single-epoch continuation,
#                      matching how the keccak numbers were historically recorded)
#   --cuda | --gpu     build and measure the CUDA prover path
#   --features LIST    explicit cargo feature list (overrides --cuda's default)
#   --cli PATH         reuse an already-built cli (e.g. bench_abba.sh's
#                      /tmp/abba_run/cli_A) instead of building one
#   --csv FILE         also append machine-readable results to FILE
#   --skip-build       reuse an existing target/release/cli
#   -h, --help
#
# GPU THRESHOLD — the trap this script refuses to walk into. The GPU LDE path
# only engages once lde_size = padded_rows * blowup clears 2^19
# (DEFAULT_GPU_LDE_THRESHOLD, crypto/stark/src/gpu_lde.rs). KECCAK_RND commits
# 24 rows per permutation, so at blowup 2 nothing reaches the GPU until 24*N
# pads to 2^18 rows, i.e. N >= 10922. N=5461 — the best CPU point, because it
# packs 131,064 rows just under 2^17 — sits a full factor of two below that, so
# `--cuda -n 5461` would run entirely on the CPU and report it as a GPU number.
# Rather than warn, the script exits: a plausible-looking wrong number is worse
# than no number. Lower the bar with LAMBDA_VM_GPU_LDE_THRESHOLD if you
# genuinely want to measure the small-trace fallback.
#
# Expect ~10 minutes for the CPU sweep; the GPU sweep is larger per point.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMP_DIR="${TMPDIR:-/tmp}/bench_keccak_throughput"

BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

N_LIST=""
RUNS=5
BLOWUP=2
EPOCH_LOG2=""
CONTINUATIONS=true
CSV=""
SKIP_BUILD=false
CUDA=false
FEATURES=""
CLI_OVERRIDE=""

while [ $# -gt 0 ]; do
    case "$1" in
        -n) N_LIST="${2:?-n needs a value}"; shift 2 ;;
        -r) RUNS="${2:?-r needs a value}"; shift 2 ;;
        --blowup) BLOWUP="${2:?--blowup needs a value}"; shift 2 ;;
        --epoch-size-log2) EPOCH_LOG2="${2:?--epoch-size-log2 needs a value}"; shift 2 ;;
        --monolithic) CONTINUATIONS=false; shift ;;
        --csv) CSV="${2:?--csv needs a value}"; shift 2 ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --cuda|--gpu) CUDA=true; shift ;;
        --features) FEATURES="${2:?--features needs a value}"; shift 2 ;;
        --cli) CLI_OVERRIDE="${2:?--cli needs a value}"; shift 2 ;;
        -h|--help) sed -n '2,75p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1 (try --help)" >&2; exit 1 ;;
    esac
done

[ -z "$FEATURES" ] && { FEATURES="jemalloc-stats"; [ "$CUDA" = true ] && FEATURES="jemalloc-stats,prover/cuda"; }

# The GPU LDE path only engages when lde_size = padded_rows * blowup clears
# DEFAULT_GPU_LDE_THRESHOLD = 2^19 (crypto/stark/src/gpu_lde.rs). KECCAK_RND
# commits 24 rows per permutation, so at blowup 2 the GPU does not fire until
# 24*N pads to 2^18 rows — N=5461 (the CPU sweet spot) is a full factor of two
# BELOW that and would silently measure the CPU path on a GPU box. The GPU
# default sweep therefore starts where the GPU actually engages.
GPU_LDE_THRESHOLD="${LAMBDA_VM_GPU_LDE_THRESHOLD:-524288}"
if [ -z "$N_LIST" ]; then
    if [ "$CUDA" = true ]; then
        N_LIST="10922 21845 43690"
    else
        N_LIST="1365 5000 5461 21845"
    fi
fi

if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [ "$RUNS" -lt 1 ]; then
    echo "bench_keccak_throughput: -r must be a positive integer" >&2
    exit 1
fi

rm -rf "$TMP_DIR" && mkdir -p "$TMP_DIR"
RESULTS="$TMP_DIR/results.tsv"
: > "$RESULTS"

# --- Environment -----------------------------------------------------------
# Recorded because throughput is meaningless without it, and because on sliced
# cloud boxes `nproc` reports the HOST's cores while the container is capped far
# lower by the cgroup — which silently oversubscribes rayon and inflates variance.

echo -e "${BOLD}=== Environment ===${NC}"
NPROC="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo '?')"
echo "nproc:        $NPROC"
CPU_QUOTA="n/a"
if [ -r /sys/fs/cgroup/cpu.max ]; then
    CPU_QUOTA="$(cat /sys/fs/cgroup/cpu.max)"
    quota_n="${CPU_QUOTA%% *}"; period_n="${CPU_QUOTA##* }"
    if [ "$quota_n" != "max" ] && [ -n "$period_n" ]; then
        effective=$(( quota_n / period_n ))
        echo "cgroup cpu.max: $CPU_QUOTA  -> ~${effective} effective cores"
        if [ "$effective" -lt "$NPROC" ]; then
            echo -e "${YELLOW}  WARNING: cgroup caps this box below nproc.${NC}"
            echo -e "${YELLOW}  rayon sizes its pool from nproc -> oversubscription, noisy timings.${NC}"
        fi
    else
        echo "cgroup cpu.max: $CPU_QUOTA (uncapped)"
    fi
elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
    CPU_QUOTA="$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)/$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)"
    echo "cgroup (v1):  $CPU_QUOTA"
fi
if [ -r /proc/meminfo ]; then
    echo "memory:       $(awk '/MemTotal/ {printf "%.0f GB", $2/1048576}' /proc/meminfo)"
fi
GIT_REV="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
GIT_BRANCH="$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
echo "git:          $GIT_BRANCH @ $GIT_REV"
if ! git -C "$ROOT_DIR" diff --quiet 2>/dev/null; then
    echo -e "${YELLOW}  WARNING: working tree is dirty — the number is not attributable to $GIT_REV.${NC}"
fi
MODE="continuations"; [ "$CONTINUATIONS" = false ] && MODE="monolithic"
BACKEND="cpu"; [ "$CUDA" = true ] && BACKEND="gpu"
echo "backend:      $BACKEND (features: $FEATURES)"
echo "mode:         $MODE, blowup=$BLOWUP, runs=$RUNS, sweep=[$N_LIST]"
if [ "$CUDA" = true ]; then
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total,driver_version \
        --format=csv,noheader 2>/dev/null | sed 's/^/gpu:          /'
    # Refuse to report a "GPU" number the GPU never touched.
    for N in $N_LIST; do
        rows=$(( N * 24 )); padded=1
        while [ "$padded" -lt "$rows" ]; do padded=$(( padded * 2 )); done
        if [ $(( padded * BLOWUP )) -lt "$GPU_LDE_THRESHOLD" ]; then
            echo -e "${RED}ERROR: N=$N gives lde_size=$(( padded * BLOWUP )) < GPU threshold $GPU_LDE_THRESHOLD.${NC}" >&2
            echo -e "${RED}The GPU LDE path would not fire and this would silently be a CPU number.${NC}" >&2
            echo -e "${RED}Use N >= 10922 at blowup 2, or set LAMBDA_VM_GPU_LDE_THRESHOLD to lower the bar.${NC}" >&2
            exit 1
        fi
    done
fi
echo

# --- Build -----------------------------------------------------------------

CLI="$ROOT_DIR/target/release/cli"
if [ -n "$CLI_OVERRIDE" ]; then
    # Reuse a binary someone else already built — e.g. bench_abba.sh's
    # /tmp/abba_run/cli_A — so a throughput reading costs no extra build.
    [ -x "$CLI_OVERRIDE" ] || { echo -e "${RED}--cli $CLI_OVERRIDE is not executable${NC}" >&2; exit 1; }
    CLI="$CLI_OVERRIDE"
    echo -e "${YELLOW}Using prebuilt $CLI (--cli); its feature set is NOT verified here.${NC}"
elif [ "$SKIP_BUILD" = true ]; then
    [ -x "$CLI" ] || { echo -e "${RED}--skip-build given but $CLI is missing${NC}" >&2; exit 1; }
    echo -e "${YELLOW}Reusing existing $CLI (--skip-build)${NC}"
else
    echo -e "${GREEN}Building release CLI (features: $FEATURES)...${NC}"
    cargo build --release -p cli --features "$FEATURES" \
        --manifest-path "$ROOT_DIR/Cargo.toml" 2>&1 | tail -2
fi
echo

# --- Sweep -----------------------------------------------------------------

for N in $N_LIST; do
    echo -e "${BOLD}=== N = $N permutations ===${NC}"
    ELF="$TMP_DIR/keccak_$N.elf"
    "$SCRIPT_DIR/gen_keccak_bench.sh" "$N" "$ELF" >/dev/null

    # 1. GATE: the executor's own permutation count, not the loop constant.
    exec_out="$("$CLI" execute "$ELF" --cycles 2>/dev/null)"
    keccak_calls="$(echo "$exec_out" | awk '/^Keccak calls:/ {print $3}')"
    cycles="$(echo "$exec_out" | awk '/^Cycles:/ {print $2}')"
    if [ -z "$keccak_calls" ]; then
        echo -e "${RED}FAIL: CLI printed no 'Keccak calls' line — cannot attribute time.${NC}" >&2
        exit 1
    fi
    if [ "$keccak_calls" -ne "$N" ]; then
        echo -e "${RED}FAIL: executor counted $keccak_calls permutations, guest asked for $N.${NC}" >&2
        exit 1
    fi
    echo "  guest:  $cycles cycles, $keccak_calls permutations (verified)"

    # 2. SIZE: committed trace, converted to base cell-equivalents.
    elem_out="$("$CLI" count-elements "$ELF" 2>/dev/null)"
    main_elems="$(echo "$elem_out" | awk '/^Elements:/ {print $2}')"
    aux_elems="$(echo "$elem_out" | awk '/^Aux elements/ {print $4}')"
    total_cells=$(( main_elems + 3 * aux_elems ))
    echo "  trace:  $main_elems main + $aux_elems aux-EF => $total_cells base cell-equiv"

    # 3. TIME.
    prove_args=(prove "$ELF" -o "$TMP_DIR/proof_$N.bin" --blowup "$BLOWUP" --time)
    if [ "$CONTINUATIONS" = true ]; then
        prove_args+=(--continuations)
        [ -n "$EPOCH_LOG2" ] && prove_args+=(--epoch-size-log2 "$EPOCH_LOG2")
    fi

    echo -n "  warm-up... "
    warm_out="$("$CLI" "${prove_args[@]}" 2>/dev/null)"
    echo "done"

    epochs="$(echo "$warm_out" | awk '/^Epochs:/ {print $2}')"
    if [ "$CONTINUATIONS" = true ] && [ -n "$epochs" ] && [ "$epochs" -ne 1 ]; then
        echo -e "${YELLOW}  NOTE: $epochs epochs — per-epoch overhead is now in the number.${NC}"
    fi

    times=""
    peak=""
    for run in $(seq 1 "$RUNS"); do
        out="$("$CLI" "${prove_args[@]}" 2>/dev/null)"
        t="$(echo "$out" | awk '/^Proving time:/ {gsub(/s$/,"",$3); print $3}')"
        p="$(echo "$out" | awk '/^Peak heap:/ {print $3}')"
        if [ -z "$t" ]; then
            echo -e "${RED}FAIL: no 'Proving time' line from run $run.${NC}" >&2
            exit 1
        fi
        times="$times $t"
        [ -n "$p" ] && peak="$p"
        echo "  run $run:  ${t}s"
    done

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$N" "$cycles" "$total_cells" "${peak:-0}" "${epochs:-1}" "$times" >> "$RESULTS"
    echo
done

# --- Report ----------------------------------------------------------------

MODE="$MODE" GIT_REV="$GIT_REV" GIT_BRANCH="$GIT_BRANCH" NPROC="$NPROC" \
BLOWUP="$BLOWUP" CSV="$CSV" BACKEND="$BACKEND" python3 - "$RESULTS" <<'PY'
import os, statistics, sys

rows = []
with open(sys.argv[1]) as fh:
    for line in fh:
        if not line.strip():
            continue
        n, cycles, cells, peak, epochs, times = line.rstrip("\n").split("\t")
        ts = [float(x) for x in times.split()]
        rows.append({
            "n": int(n), "cycles": int(cycles), "cells": int(cells),
            "peak": int(peak), "epochs": int(epochs), "times": ts,
            "median": statistics.median(ts), "min": min(ts), "max": max(ts),
        })

def cv(ts):
    if len(ts) < 2:
        return 0.0
    return statistics.stdev(ts) / statistics.mean(ts) * 100

print("\033[1m=== keccak-f[1600] prover throughput ===\033[0m")
print(f"{os.environ['GIT_BRANCH']} @ {os.environ['GIT_REV']}  |  {os.environ['BACKEND'].upper()}  |  "
      f"{os.environ['MODE']}, blowup={os.environ['BLOWUP']}, {os.environ['NPROC']} cores\n")

def padded(rows_used):
    """Trace tables are padded to a power of two."""
    p = 1
    while p < rows_used:
        p *= 2
    return p

hdr = (f"{'N perms':>9} {'rnd rows':>10} {'pad':>6} {'median s':>9} {'spread':>7} "
       f"{'keccak/s':>9} {'Mcell/s':>8} {'cells/perm':>11} {'peak MB':>8}")
print(hdr)
print("-" * len(hdr))
by_n = sorted(rows, key=lambda r: r["n"])
for r in by_n:
    used = r["n"] * 24
    waste = (padded(used) - used) / padded(used) * 100
    print(f"{r['n']:>9,} {used:>10,} {waste:>5.1f}% {r['median']:>9.3f} "
          f"{cv(r['times']):>6.1f}% {r['n'] / r['median']:>9,.0f} "
          f"{r['cells'] / r['median'] / 1e6:>8.1f} {r['cells'] // r['n']:>11,} {r['peak']:>8,}")

best = max(rows, key=lambda r: r["n"] / r["median"])
print(f"\n\033[1mPeak: {best['n'] / best['median']:,.0f} keccak-f permutations/sec\033[0m "
      f"(at N={best['n']:,}) = {best['n'] / best['median'] * 136 / 1024:,.0f} KB/s absorbed "
      f"(136 B rate/perm).")

print("\n'pad' is the share of KECCAK_RND rows burned on power-of-two padding — it is why")
print("throughput is not monotone in N. 'cells/perm' is the whole-trace average and")
print("includes the fixed preprocessed floor, so it falls as N grows.")

# Marginal cost with the fixed floor differenced out. Taken across the widest
# lever arm available; adjacent points can share a padding bucket, which makes a
# per-row difference meaningless (cells barely move while N grows).
if len(by_n) >= 2:
    lo, hi = by_n[0], by_n[-1]
    dn = hi["n"] - lo["n"]
    if dn > 0:
        print(f"\nMarginal cost, N={lo['n']:,} -> {hi['n']:,}: "
              f"\033[1m{(hi['cells'] - lo['cells']) / dn:,.0f} cell-equiv per permutation\033[0m")
        print("Model for comparison: 72,672 (KECCAK_RND, 24 rows x (1,480 main + 1,548 aux))")
        print("plus ~750 for the keccak.rs sponge row. Padding inflates the measured value;")
        print("it is exact only when both endpoints sit flush against a power of two.")

noisy = [r for r in rows if cv(r["times"]) > 3]
if noisy:
    print(f"\n\033[1;33mWARNING: run-to-run spread >3% at N={[r['n'] for r in noisy]}.\033[0m")
    print("\033[1;33mA contended or CPU-capped box; treat these medians as soft.\033[0m")

csv = os.environ.get("CSV")
if csv:
    new = not os.path.exists(csv)
    with open(csv, "a") as fh:
        if new:
            fh.write("git_rev,backend,mode,blowup,cores,n_perms,cycles,cells,epochs,"
                     "median_s,min_s,max_s,cv_pct,keccaks_per_s,peak_mb\n")
        for r in rows:
            fh.write(f"{os.environ['GIT_REV']},{os.environ['BACKEND']},{os.environ['MODE']},{os.environ['BLOWUP']},"
                     f"{os.environ['NPROC']},{r['n']},{r['cycles']},{r['cells']},{r['epochs']},"
                     f"{r['median']:.3f},{r['min']:.3f},{r['max']:.3f},{cv(r['times']):.2f},"
                     f"{r['n'] / r['median']:.1f},{r['peak']}\n")
    print(f"\nAppended {len(rows)} rows to {csv}")
PY

echo
echo "Artifacts in $TMP_DIR (guest ELFs + proofs)."
