#!/usr/bin/env bash
# Regenerates the Lean RV64D model from the commits in PINS.
#
# Needs opam, git, cmake, gmp, z3 and pkg-config (on macOS: brew install opam gmp z3 pkgconf cmake).
# Sail is installed in its own opam switch (default `sail-riscv-lean`), so nothing else is touched.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=../PINS
source "$ROOT/PINS"
WORK="${SAIL_RISCV_LEAN_WORK:-$ROOT/_work}"
SWITCH="${OPAM_SWITCH:-sail-riscv-lean}"

# 1. The Sail compiler at SAIL_COMMIT.
if ! opam switch list --short | grep -qx "$SWITCH"; then
  opam switch create "$SWITCH" ocaml-base-compiler.5.2.1 -y --no-switch
fi
eval "$(opam env --switch="$SWITCH" --set-switch)"
opam pin add -y -n "$SAIL_REPO#$SAIL_COMMIT"
opam install -y sail

# 2. The spec at SAIL_RISCV_COMMIT.
rm -rf "$WORK/sail-riscv"
mkdir -p "$WORK"
git clone -q "$SAIL_RISCV_REPO" "$WORK/sail-riscv"
git -C "$WORK/sail-riscv" checkout -q "$SAIL_RISCV_COMMIT"

# 3. Translate the spec to Lean.
cmake -S "$WORK/sail-riscv" -B "$WORK/sail-riscv/build" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$WORK/sail-riscv/build" --target generated_lean_rv64d

# 4. Replace the model in this repo with the generated one.
OUT="$WORK/sail-riscv/build/model/Lean_RV64D"
rm -rf "$ROOT/LeanRV64D"
cp -R "$OUT/LeanRV64D" "$ROOT/LeanRV64D"
cp "$OUT/LeanRV64D.lean" "$OUT/lakefile.toml" "$OUT/lean-toolchain" "$ROOT/"
# The model is BSD-2-Clause: its licence travels with it.
cp "$WORK/sail-riscv/LICENCE" "$ROOT/LICENCE"

echo "Generated from sail $SAIL_COMMIT and sail-riscv $SAIL_RISCV_COMMIT."
echo "Next: step 3 of README.md (lake update here and in formal-riscv, then build formal-riscv)."
