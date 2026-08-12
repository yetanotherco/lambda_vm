#!/usr/bin/env bash
# Runs the whole ECSM-affine campaign and writes each stage's output to gate/logs/.
#
# Order matters: the anchors run BEFORE the lemmas, because an UNSAT is only worth reading
# once the model has been shown to describe the right function (oracle) and the right columns
# (real-witness anchor). The transcription audit runs first of all, since a premise that no
# longer holds invalidates whatever the lemmas concluded from it.
#
# Usage:
#   ./run_gate.sh            # everything
#   ./run_gate.sh --quick    # skip the Rust harness rebuild (reuse the existing dump)
#
# Dependencies: python3 with z3-solver + sympy (+ ecdsa for the optional third-party anchor).
# A local venv at .venv is used when present; otherwise `python3` from PATH.

set -uo pipefail
cd "$(dirname "$0")"

# Absolute, so the per-stage subshells can `cd` freely.
if [[ -x .venv/bin/python ]]; then
  PY="$PWD/.venv/bin/python"
else
  PY="$(command -v python3)" || { echo "no python3 on PATH"; exit 1; }
fi
ROOT="$PWD"

mkdir -p gate/logs
FAILED=()

run() {                     # run <log-name> <dir> <script...>
  local name="$1" dir="$2"; shift 2
  printf '\n=== %s ===\n' "$name"
  if (cd "$dir" && "$PY" "$@" 2>&1) | tee "$ROOT/gate/logs/${name}.log"; then
    :
  else
    FAILED+=("$name")
  fi
}

# 0. transcription audit — are the gate's premises still true of the code?
run audit_transcription gate audit_transcription.py

# 1. oracle anchors — is the modelled FUNCTION the right one?
run oracle_anchors oracle test_oracle.py
run small_y_point oracle small_y_point.py

# 2. real-witness anchor — are the modelled COLUMNS the right ones?
if [[ "${1:-}" != "--quick" ]]; then
  printf '\n=== harness (cargo) ===\n'
  if (cd harness && cargo build --release 2>&1) | tee gate/logs/harness_build.log; then
    ./harness/target/release/ecsm-affine-harness > gate/logs/real_witnesses.jsonl
  else
    FAILED+=(harness_build)
  fi
fi
run a6_real_witness gate a6_real_witness.py

# 3. the lemmas
run a1_selector gate a1_selector.py
run a2_yr_lt_p gate a2_yr_lt_p.py
run a3_parity_binding gate a3_parity_binding.py
run a4_addressing gate a4_addressing.py

printf '\n========================================\n'
if ((${#FAILED[@]})); then
  printf 'GATE: FAILED — %s\n' "${FAILED[*]}"
  exit 1
fi
printf 'GATE: all stages green (logs in gate/logs/)\n'
