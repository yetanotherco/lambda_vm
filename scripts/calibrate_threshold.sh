#!/bin/bash
# Calibrate the auto-disk-spill threshold: actual RSS / predicted_peak_bytes.
#
# Usage: calibrate_threshold.sh elf1.elf [elf2.elf ...]
#
# Builds CLI with jemalloc-stats, runs each ELF under `/usr/bin/time -v`,
# and prints predicted vs measured peak. The max of rss/pred is r_max;
# set the threshold in select_storage_mode to ~1/r_max minus a small margin.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT="/tmp/calibrate_threshold"

mkdir -p "$OUT"
rm -f "$OUT"/*.txt

echo "Building CLI with jemalloc-stats and disk-spill..."
cargo build --release -p cli --features jemalloc-stats,disk-spill --manifest-path "$ROOT_DIR/Cargo.toml" 2>&1 | tail -1

CLI="$ROOT_DIR/target/release/cli"

printf "\n%-55s %10s %10s %10s %10s %10s\n" \
    "ELF" "pred(MB)" "heap(MB)" "rss(MB)" "rss/pred" "heap/pred"
printf '%.0s-' {1..110}
printf '\n'

for elf in "$@"; do
    name=$(basename "$elf")
    RUST_LOG=info /usr/bin/time -v "$CLI" prove "$elf" -o "$OUT/proof.bin" \
        > "$OUT/out.txt" 2> "$OUT/err.txt" || {
            echo "FAIL: $name"
            tail -5 "$OUT/err.txt"
            continue
        }

    pred=$(grep -o 'predicted_peak_bytes: [0-9]*' "$OUT/err.txt" | awk '{print $2}')
    heap_mb=$(grep -o 'Peak heap: [0-9]*' "$OUT/out.txt" | awk '{print $3}')
    rss_kb=$(grep "Maximum resident set size" "$OUT/err.txt" | awk '{print $NF}')

    awk -v name="$name" -v p="$pred" -v h="$heap_mb" -v r="$rss_kb" 'BEGIN {
        pred_mb = p / 1024 / 1024
        rss_mb  = r / 1024
        printf "%-55s %10.0f %10.0f %10.0f %10.2f %10.2f\n",
            name, pred_mb, h, rss_mb, rss_mb/pred_mb, h/pred_mb
    }'

    rm -f "$OUT/proof.bin"
done

echo ""
echo "Take the max rss/pred across runs as r_max."
echo "Set threshold in select_storage_mode to ~1/r_max minus margin (e.g. 0.05)."
