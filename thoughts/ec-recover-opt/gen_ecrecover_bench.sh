#!/bin/bash
# Phase-H bench driver for the lincomb2 ecrecover path.
#
# Companion to gen_ec_bench.sh (which builds a SYNTHETIC ECSM ladder, written for
# the AreBytes-pairing work). This one drives the REAL shipping path — guest
# ecrecover -> ecsm_lincomb2 -> ECSM2/ECDAS2 -> keccak(pk) — through the bench
# guest at executor/programs/bench/ecrecover/.
#
# ONE ELF serves every configuration: the workload is chosen by private input,
# so an A/B never compares two different binaries.
#
# *** RUN THIS ON THE BENCH SERVER, NOT LOCALLY. *** A local run on a box that
# has already OOMed on ethrex-20tx produces a misleading number. See BENCH.md.
#
# Subcommands:
#   build                       compile the bench guest ELF
#   input   <case> <n> <file>   write a private-input file (case: mean|worst)
#   cells   <case> <n>          count committed cells (no proving) — the primary
#                               measurement; prints main, aux and total base cells
#   slope   <case> <n1> <n2>    cells per ecrecover, from the two-point slope
#                               (this is the number that answers phase H)
#   share   <block.elf> <input> EC share of a real block: total cells, and the EC
#                               contribution implied by the measured slope
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ELF="$ROOT/executor/program_artifacts/bench/ecrecover.elf"
CLI=(cargo run --release -q -p cli --)

usage() { sed -n '2,28p' "${BASH_SOURCE[0]}"; exit 1; }

case_byte() {
  case "$1" in
    mean)  echo 0 ;;
    worst) echo 1 ;;
    *) echo "case must be 'mean' or 'worst'" >&2; exit 1 ;;
  esac
}

# <case> <n> <outfile>
write_input() {
  local c n out
  c="$(case_byte "$1")"; n="$2"; out="$3"
  if [ "$n" -lt 1 ] || [ "$n" -gt 65535 ]; then
    echo "count must be in [1, 65535] (u16 in the guest ABI)" >&2; exit 1
  fi
  printf "$(printf '\\x%02x\\x%02x\\x%02x' \
    "$c" "$((n & 0xFF))" "$(((n >> 8) & 0xFF))")" > "$out"
}

# Total committed BASE field elements. `count-elements` reports main elements and
# aux EXTENSION columns; one extension element is 3 base elements, which is the
# same 1.5-base-cells-per-interaction rule the cost model uses (LogUp packs two
# interactions per aux column).
cells_for() {
  local inp out main aux
  inp="$(mktemp)"; trap 'rm -f "$inp"' RETURN
  write_input "$1" "$2" "$inp"
  out="$("${CLI[@]}" count-elements "$ELF" --private-input "$inp")"
  main="$(echo "$out" | awk '/^Elements:/ {print $2}')"
  aux="$(echo "$out" | awk '/^Aux elements/ {print $4}')"
  echo "$main $aux $((main + 3 * aux))"
}

cmd="${1:-}"; shift || usage
case "$cmd" in
  build)
    cd "$ROOT" && make compile-bench
    ls -l "$ELF"
    ;;

  input)
    [ $# -eq 3 ] || usage
    write_input "$1" "$2" "$3"
    echo "wrote $3 (case=$1, n=$2)"
    ;;

  cells)
    [ $# -eq 2 ] || usage
    read -r main aux total <<<"$(cells_for "$1" "$2")"
    printf 'case=%s n=%s  main=%s  aux_ef_cols=%s  total_base_cells=%s\n' \
      "$1" "$2" "$main" "$aux" "$total"
    ;;

  slope)
    [ $# -eq 3 ] || usage
    c="$1"; n1="$2"; n2="$3"
    read -r _ _ t1 <<<"$(cells_for "$c" "$n1")"
    read -r _ _ t2 <<<"$(cells_for "$c" "$n2")"
    if [ "$n2" -le "$n1" ]; then echo "n2 must exceed n1" >&2; exit 1; fi
    python3 - "$t1" "$t2" "$n1" "$n2" "$c" <<'PY'
import sys
t1, t2, n1, n2, c = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
slope = (t2 - t1) / (n2 - n1)
base = 1_467_000  # post-pairing 4x-ecsm_mul baseline, cells/ecrecover
print(f"case={c}  n={n1}->{n2}  cells {t1} -> {t2}")
print(f"  cells per ecrecover : {slope:,.0f}  ({slope/1e6:.3f}M)")
print(f"  vs 1.467M baseline  : {100*(slope-base)/base:+.1f}%")
print(f"  fixed overhead      : {t1 - slope*n1:,.0f} cells "
      f"(EC_T0 + preprocessed + CPU/keccak floor)")
PY
    ;;

  share)
    [ $# -eq 2 ] || usage
    blk="$1"; blk_in="$2"
    out="$("${CLI[@]}" count-elements "$blk" --private-input "$blk_in")"
    main="$(echo "$out" | awk '/^Elements:/ {print $2}')"
    aux="$(echo "$out" | awk '/^Aux elements/ {print $4}')"
    echo "block total base cells: $((main + 3 * aux))  (main=$main aux_ef_cols=$aux)"
    echo
    echo "EC share = n_ecrecover * <cells-per-ecrecover from \`slope\`> / total."
    echo "n_ecrecover for an ethrex block = one per transaction with a signature;"
    echo "read it from the fixture rather than assuming it equals the tx count."
    ;;

  *) usage ;;
esac
