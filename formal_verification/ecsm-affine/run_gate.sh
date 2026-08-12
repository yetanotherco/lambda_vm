#!/usr/bin/env bash
# Runs the whole ECSM-affine campaign and writes one transcript to gate.log.
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

if [[ -x .venv/bin/python ]]; then
  PY="$PWD/.venv/bin/python"
else
  PY="$(command -v python3)" || { echo "no python3 on PATH"; exit 1; }
fi

LOG=gate.log
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

# 0. transcription audit — are the gate's premises still true of the code?
run audit_transcription audit_transcription.py

# 1. oracle anchors — is the modelled FUNCTION the right one?
run oracle_anchors test_oracle.py
run small_y_point small_y_point.py

# 2. real-witness anchor — are the modelled COLUMNS the right ones?
if [[ "${1:-}" != "--quick" ]]; then
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
printf 'GATE: all stages green (transcript in %s)\n' "$LOG" | tee -a "$LOG"
