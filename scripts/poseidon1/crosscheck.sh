#!/usr/bin/env bash
# Derive and cross-check the Poseidon1 Goldilocks W16 instance.
#   PLONKY3=<Plonky3 checkout> scripts/poseidon1/crosscheck.sh <scratch dir>
# 1. our generator reproduces Plonky3's W8/W12 constants and KATs;
# 2. the round-number inequalities at t = 16;
# 3. Plonky3's generator (width whitelist lifted) gives the same 480 W16 constants;
# 4. Plonky3's subspace-trail checks on the circulant MDS rows, and its reference
#    permutation reproduces our W16 KATs;
# 5. exhaustive MDS check of the W16 circulant, with a negative control;
# 6. Plonky3's Rust Poseidon1 (sparse partial rounds) reproduces our W16 KATs.
# 7. ZisK's own Rust (pil2-proofman `proofman-fields`, MIT/Apache) regenerates the base
#    STARK's known answers (W16/W8 permutations, leaf, 4-ary tree and path, transcript,
#    grinding) byte for byte.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PLONKY3=${PLONKY3:-/Users/maurofab/workspace/Plonky3}
OUT=${1:?scratch dir}
mkdir -p "$OUT"
RUN=${ZF_RUN:-/Users/maurofab/workspace/zf-tools/zf-run}

python3 "$HERE/p1_params.py" selftest --plonky3 "$PLONKY3/goldilocks/src/poseidon1.rs"
python3 "$HERE/p1_params.py" rounds --width 16
python3 "$HERE/p1_params.py" consts --width 16 > "$OUT/mine_rc16.txt"
python3 "$HERE/p1_params.py" kat --width 16 > "$OUT/mine_kat16.txt"

sed 's/"valid_widths": \[8, 12\]/"valid_widths": [8, 12, 16]/' \
  "$PLONKY3/poseidon1/generate_constants.py" > "$OUT/gen16.py"
python3 "$OUT/gen16.py" --field goldilocks --width 16 --format json --skip-mds > "$OUT/p3_rc16.json"
python3 - "$OUT" <<'PY'
import json, sys
d = sys.argv[1]
j = json.load(open(f"{d}/p3_rc16.json"))
theirs = [int(x, 16) for row in j["round_constants"] for x in row]
mine = [int(x, 16) for l in open(f"{d}/mine_rc16.txt") for x in l.split()]
assert (j["R_F"], j["R_P"]) == (8, 22), (j["R_F"], j["R_P"])
print(f"W16 round constants: ours {len(mine)}, Plonky3 generator {len(theirs)}:",
      "MATCH" if mine == theirs else "MISMATCH")
sys.exit(0 if mine == theirs else 1)
PY

python3 "$HERE/p3_subspace_checks.py" "$OUT" "$OUT/p3_rc16.json" > "$OUT/p3_checks.txt"
grep '^#' "$OUT/p3_checks.txt"
grep -v '^#' "$OUT/p3_checks.txt" | diff - "$OUT/mine_kat16.txt" \
  && echo "W16 KATs: Plonky3 generator's reference permutation MATCH"

"$RUN" rustc -O "$HERE/mds_check.rs" -o "$OUT/mds_check"
"$RUN" "$OUT/mds_check" 16 1,1,51,1,11,17,2,1,101,63,15,2,67,22,13,3 | tail -1
echo "negative control (last entry 3 -> 1):"
"$RUN" "$OUT/mds_check" 16 1,1,51,1,11,17,2,1,101,63,15,2,67,22,13,1 | tail -1

(cd "$HERE/p3_crosscheck" && CARGO_TARGET_DIR="$OUT/p3_target" "$RUN" cargo run --release -q) \
  > "$OUT/p3_rust_kat16.txt"
diff "$OUT/p3_rust_kat16.txt" "$OUT/mine_kat16.txt" && echo "W16 KATs: Plonky3 Rust Poseidon1 MATCH"

(cd "$HERE/zisk_crosscheck" && CARGO_TARGET_DIR="$OUT/zisk_target" "$RUN" cargo run --release -q) \
  > "$OUT/zisk_kat.rs"
rustfmt --edition 2024 "$OUT/zisk_kat.rs"
diff "$OUT/zisk_kat.rs" "$HERE/../../crypto/crypto/src/hash/poseidon1_stark/zisk_kat.rs" \
  && echo "base-STARK KATs: ZisK proofman-fields MATCH"
