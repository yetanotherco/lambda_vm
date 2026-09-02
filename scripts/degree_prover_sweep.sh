#!/usr/bin/env bash
#
# degree_prover_sweep.sh — DEGREE-LANE EXPERIMENT (temporary, not for merge).
#
# Prover-side arms for the constraint-degree cost model: peak RSS and wall
# clock as a function of composition part count and blowup, at real scale.
#
# The verifier half of the model is already measured host-side and exactly
# (see thoughts/shared/alt-protocol/DEGREE-COST-MODEL.md §3) and needs no box
# time. This script covers only what genuinely needs a big idle machine:
# memory and time.
#
# Discipline this script enforces:
#   * ONE ARM PER PROCESS. Peak RSS is a high-water mark, so two configurations
#     in one process would only ever measure the larger. Every arm is a fresh
#     `/usr/bin/time -v` invocation of the test binary.
#   * PAIRED ABBA over the fast-path flag, with replicates, because wall clock
#     reproduces far worse than peak RSS.
#   * VM_MAX_DEGREE is a compile-time constant read by BOTH the prover and the
#     guest verifier, so each degree needs its own build. The script rebuilds
#     and restores the source on exit.
#
# Usage:
#   scripts/degree_prover_sweep.sh [ELF_NAME] [REPS]
#     ELF_NAME  asm artifact name (default all_instructions_64), or set
#               LVM_DEGREE_ELF_PATH to point at an arbitrary ELF.
#     REPS      ABBA cycles per arm (default 3 → 6 timing samples per config).
#
# Output: one TSV line per arm on stdout, full logs under the run directory.

set -uo pipefail

ELF_NAME="${1:-all_instructions_64}"
REPS="${2:-3}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB="$REPO/prover/src/lib.rs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$REPO/degree_sweep_$STAMP"
mkdir -p "$OUT"
RESULTS="$OUT/results.tsv"

# Restore the source no matter how we exit — the constant must not be left
# edited in the working tree.
ORIG_DEGREE="$(grep -oE '^pub const VM_MAX_DEGREE: usize = [0-9]+;' "$LIB" | grep -oE '[0-9]+')"
restore() { set_degree "$ORIG_DEGREE"; }
trap restore EXIT INT TERM

set_degree() {
  # Portable in-place edit (GNU sed and BSD sed disagree on -i).
  local d="$1" tmp
  tmp="$(mktemp)"
  sed "s/^pub const VM_MAX_DEGREE: usize = [0-9]*;/pub const VM_MAX_DEGREE: usize = $d;/" "$LIB" > "$tmp"
  mv "$tmp" "$LIB"
}

# Path of the built test binary for the current feature set.
test_bin() {
  cargo test -p lambda-vm-prover --release --features instruments --no-run \
    --message-format=json 2>/dev/null |
  python3 -c '
import sys, json
for line in sys.stdin:
    try: m = json.loads(line)
    except Exception: continue
    if (m.get("reason") == "compiler-artifact"
            and m.get("target", {}).get("name") == "lambda_vm_prover"
            and m.get("profile", {}).get("test")):
        print(m["executable"])
'
}

# Pick a working time(1): GNU on the box, BSD locally.
if /usr/bin/time -v true >/dev/null 2>&1; then
  TIME_BIN=(/usr/bin/time -v); TIME_STYLE=gnu
elif /usr/bin/time -l true >/dev/null 2>&1; then
  TIME_BIN=(/usr/bin/time -l); TIME_STYLE=bsd
else
  echo "no usable /usr/bin/time; peak RSS cannot be measured" >&2
  exit 1
fi
echo "time(1) style: $TIME_STYLE"

printf 'degree\tparts\tblowup\tforce_generic\trep\twall_s\tpeak_rss_kb\tdecompose_s\tcomp_commit_s\tconstraints_s\n' > "$RESULTS"

run_arm() {
  local degree="$1" blowup="$2" force="$3" rep="$4" bin="$5"
  local log="$OUT/d${degree}_b${blowup}_g${force}_r${rep}.log"

  # /usr/bin/time writes to stderr; the instruments report does too.
  # GNU time (-v, the box) reports "Maximum resident set size" in KB; BSD time
  # (-l, macOS, used only to smoke-test this script) reports it in BYTES.
  LVM_DEGREE_ELF="$ELF_NAME" \
  LVM_DEGREE_BLOWUP="$blowup" \
  LVM_FORCE_GENERIC_PARTS="$force" \
  "${TIME_BIN[@]}" "$bin" degree_prove_instrumented --ignored --nocapture \
    > "$log" 2>&1

  local wall rss dec commit cons
  wall=$(grep -oE 'total_secs=[0-9.]+' "$log" | head -1 | cut -d= -f2)
  if [[ "$TIME_STYLE" == gnu ]]; then
    rss=$(grep -E 'Maximum resident set size' "$log" | grep -oE '[0-9]+$')
  else
    # BSD: "<bytes>  maximum resident set size" → convert to KB to match GNU.
    local rss_b
    rss_b=$(grep -E 'maximum resident set size' "$log" | grep -oE '^[[:space:]]*[0-9]+' | tr -d ' ')
    [[ -n "$rss_b" ]] && rss=$((rss_b / 1024)) || rss=""
  fi
  dec=$(grep -E 'decompose_and_extend_d2' "$log" | grep -oE '[0-9]+\.[0-9]+s' | head -1 | tr -d 's')
  commit=$(grep -E 'commit_bit_reversed \(comp' "$log" | grep -oE '[0-9]+\.[0-9]+s' | head -1 | tr -d 's')
  cons=$(grep -E 'R2  evaluate' "$log" | grep -oE '[0-9]+\.[0-9]+s' | head -1 | tr -d 's')

  # A grid of NA is not a result — it is a silent failure that looks like data.
  # If the very first arm produced no wall time, every later one will fail the
  # same way, so stop and say why instead of burning the box for an hour.
  if [ -z "${wall:-}" ]; then
    echo "FATAL: arm d=$degree b=$blowup g=$force produced no measurement." >&2
    echo "       The prove almost certainly panicked; first 20 lines of $log:" >&2
    head -20 "$log" >&2
    exit 1
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$degree" "$((degree - 1))" "$blowup" "$force" "$rep" \
    "$wall" "${rss:-NA}" "${dec:-NA}" "${commit:-NA}" "${cons:-NA}" \
    | tee -a "$RESULTS"
}

# The VM arms read a compiled ASM artifact, and `executor/program_artifacts/` is
# NOT tracked in git — a fresh clone has none. Without this, every arm panics on
# the missing ELF in milliseconds and the sweep records a full grid of NA that
# looks like data. Build them, then verify, then fail fast if still absent.
ASM_ELF="$REPO/executor/program_artifacts/asm/${ELF_NAME}.elf"
if [ ! -f "$ASM_ELF" ]; then
  echo "ASM artifact missing ($ASM_ELF) — running make compile-programs-asm"
  if ! make -C "$REPO" compile-programs-asm > "$OUT/build_asm.log" 2>&1; then
    echo "FATAL: make compile-programs-asm failed — see $OUT/build_asm.log" >&2
    exit 1
  fi
fi
if [ ! -f "$ASM_ELF" ]; then
  echo "FATAL: $ASM_ELF still missing after compile-programs-asm." >&2
  echo "       Check that '$ELF_NAME' is a real program under executor/programs/asm/." >&2
  exit 1
fi
echo "ASM artifact OK: $ASM_ELF"

echo "degree-lane prover sweep: elf=$ELF_NAME reps=$REPS out=$OUT"

for degree in 3 5 7; do
  set_degree "$degree"
  echo "=== building VM_MAX_DEGREE=$degree (parts=$((degree - 1))) ==="
  if ! cargo build -p lambda-vm-prover --release --features instruments --tests \
        > "$OUT/build_d${degree}.log" 2>&1; then
    echo "BUILD FAILED for degree $degree — see $OUT/build_d${degree}.log" >&2
    continue
  fi
  BIN="$(test_bin)"
  if [[ -z "$BIN" ]]; then
    echo "could not locate test binary for degree $degree" >&2
    continue
  fi

  for blowup in 4 8; do
    # d=7 needs blowup 8 for a genuinely degree-7 AIR. Here the VM's TRUE
    # degree is still 3, so every (degree, blowup) pair is representable and
    # blowup 4 stays legal — that is exactly what de-confounds the two axes.
    for rep in $(seq 1 "$REPS"); do
      # ABBA over the fast-path flag, only where it is meaningful (parts == 2).
      if [[ "$degree" == "3" ]]; then
        run_arm "$degree" "$blowup" 0 "$rep" "$BIN"
        run_arm "$degree" "$blowup" 1 "$rep" "$BIN"
        run_arm "$degree" "$blowup" 1 "$rep" "$BIN"
        run_arm "$degree" "$blowup" 0 "$rep" "$BIN"
      else
        run_arm "$degree" "$blowup" 0 "$rep" "$BIN"
      fi
    done
  done
done

echo
echo "=== results: $RESULTS ==="
column -t -s $'\t' "$RESULTS" 2>/dev/null || cat "$RESULTS"
