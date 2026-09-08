#!/usr/bin/env bash
# Runs the whole ECSM-affine campaign and writes one transcript to gate.log.
#
# Order matters: the anchors run BEFORE the lemmas, because an UNSAT is only worth reading
# once the model has been shown to describe the right function (oracle) and the right columns
# (real-witness anchor). The transcription audit runs first of all, since a premise that no
# longer holds invalidates whatever the lemmas concluded from it.
#
# Usage:
#   ./run_gate.sh              # everything
#   ./run_gate.sh --quick      # skip the Rust harness rebuild (reuse the existing dump)
#   ./run_gate.sh --check-log  # run to a temp file and fail if gate.log is stale
#
# Dependencies: python3 with z3-solver + sympy (+ ecdsa for the optional third-party anchor).
# A local venv at .venv is used when present; otherwise `python3` from PATH.

set -uo pipefail
cd "$(dirname "$0")"

QUICK=0
CHECK=0
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    --check-log) CHECK=1 ;;
    *) echo "usage: $0 [--quick] [--check-log]"; exit 2 ;;
  esac
done

if [[ -x .venv/bin/python ]]; then
  PY="$PWD/.venv/bin/python"
else
  PY="$(command -v python3)" || { echo "no python3 on PATH"; exit 1; }
fi

COMMITTED=gate.log
if ((CHECK)); then
  LOG=$(mktemp -t ecsm-affine-gate)
else
  LOG=$COMMITTED
fi
: > "$LOG"
FAILED=()

run() {                     # run <label> <script>
  local label="$1"; shift
  printf '\n=== %s ===\n' "$label" | tee -a "$LOG"
  if "$PY" "$@" 2>&1 | tee -a "$LOG"; then
    :
  else
    FAILED+=("$label")
  fi
}

# Solver timings, cargo's build chatter and the harness stage itself are the only things
# allowed to differ between two runs of the same scripts (`--quick` skips the harness, so its
# section is normalised away too); everything else is a verdict.
normalise() {
  grep -vE '^ *(Compiling|Finished|Updating|Locking|Adding|Downloaded|Downloading) ' "$1" \
    | grep -vxF '=== harness (cargo) ===' \
    | grep -v '^[[:space:]]*$' \
    | sed -E 's/; [0-9]+\.[0-9]+s$//'
}

# 0. transcription audit — are the gate's premises still true of the code?
run audit_transcription audit_transcription.py

# 1. oracle anchors — is the modelled FUNCTION the right one?
run oracle_anchors test_oracle.py
run small_y_point small_y_point.py

# 2. real-witness anchor — are the modelled COLUMNS the right ones?
if ((!QUICK)); then
  printf '\n=== harness (cargo) ===\n' | tee -a "$LOG"
  if cargo build --release 2>&1 | tee -a "$LOG"; then
    ./target/release/ecsm-affine-harness > real_witnesses.jsonl
  else
    FAILED+=(harness_build)
  fi
fi
run a6_real_witness a6_real_witness.py

# 3. the lemmas
run a1_selector a1_selector.py
run a2_yr_lt_p a2_yr_lt_p.py
run a3_parity_binding a3_parity_binding.py
run a4_addressing a4_addressing.py

printf '\n========================================\n' | tee -a "$LOG"
if ((${#FAILED[@]})); then
  printf 'GATE: FAILED — %s\n' "${FAILED[*]}" | tee -a "$LOG"
  exit 1
fi

printf 'GATE: all stages green (transcript in %s)\n' "$COMMITTED" | tee -a "$LOG"

if ((CHECK)); then
  if diff -u <(normalise "$COMMITTED") <(normalise "$LOG"); then
    printf 'GATE: %s matches this run\n' "$COMMITTED"
  else
    printf 'GATE: %s is STALE — re-run ./run_gate.sh and commit it\n' "$COMMITTED"
    exit 1
  fi
  rm -f "$LOG"
fi
