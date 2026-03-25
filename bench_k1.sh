#!/usr/bin/env bash
# Compares prove time (TABLE_PARALLELISM=1) between current branch and main.
# Prints row counts for MEMW and MEMW_A, prove time, and speedup.
#
# Usage: ./bench_k1.sh
#
# Requirements: cargo, jemalloc-stats feature, clang for ELF compilation.

set -euo pipefail

ELF="executor/program_artifacts/asm/fib_iterative_2M.elf"
PR_BRANCH=$(git rev-parse --abbrev-ref HEAD)
WORKTREE_PATH="/tmp/lambda_vm_main_bench_$$"

# Compile ELF if missing
if [ ! -f "$ELF" ]; then
    echo ">>> Compiling ELF..."
    mkdir -p executor/program_artifacts/asm
    clang --target=riscv64 -march=rv64im -fuse-ld=lld -nostdlib -Wl,-e,main \
        executor/programs/asm/fib_iterative_2M.s -o "$ELF"
fi

cleanup() {
    git worktree remove --force "$WORKTREE_PATH" 2>/dev/null || true
}
trap cleanup EXIT

echo "======================================================="
echo "  k=1 bench: $PR_BRANCH vs main"
echo "  Program: fib_iterative_2M"
echo "======================================================="
echo ""

# ── Step 1: Build and run PR branch ──────────────────────

echo ">>> Building PR branch ($PR_BRANCH)..."
cargo build --release -p cli --features jemalloc-stats 2>&1 | tail -2

echo ">>> Running PR branch (TABLE_PARALLELISM=1)..."
PR_OUTPUT=$(TABLE_PARALLELISM=1 ./target/release/cli prove "$ELF" \
    -o /tmp/proof_pr_$$.bin --time 2>&1)
rm -f /tmp/proof_pr_$$.bin
echo "$PR_OUTPUT" | grep -E "^MEMW|Proving time|Peak heap"

PR_TIME=$(echo "$PR_OUTPUT"   | grep -o 'Proving time: [0-9.]*' | awk '{print $3}')
PR_HEAP=$(echo "$PR_OUTPUT"   | grep -o 'Peak heap: [0-9]*'     | awk '{print $3}')
PR_MEMW_ROWS=$(echo "$PR_OUTPUT"   | grep '^MEMW '  | grep -o 'rows=[0-9]*'   | cut -d= -f2)
PR_MEMWA_ROWS=$(echo "$PR_OUTPUT"  | grep '^MEMW_A' | grep -o 'rows=[0-9]*'   | cut -d= -f2)
PR_MEMW_CHUNKS=$(echo "$PR_OUTPUT" | grep '^MEMW '  | grep -o 'chunks=[0-9]*' | cut -d= -f2)
PR_MEMWA_CHUNKS=$(echo "$PR_OUTPUT"| grep '^MEMW_A' | grep -o 'chunks=[0-9]*' | cut -d= -f2)
echo ""

# ── Step 2: Build and run main in a worktree ─────────────

echo ">>> Setting up main worktree..."
git worktree add "$WORKTREE_PATH" main

MAIN_LIB="$WORKTREE_PATH/prover/src/lib.rs"
PATCH_LINE=$(grep -n "Traces::from_elf_and_logs" "$MAIN_LIB" | head -1 | cut -d: -f1)

# Patch main's lib.rs to print MEMW row counts
awk -v line="$PATCH_LINE" '
NR == line {
    print
    print ""
    print "    {"
    print "        let memw_rows: usize = traces.memws.iter().map(|t| t.num_rows()).sum();"
    print "        eprintln!(\"MEMW   chunks={}, rows={}\", traces.memws.len(), memw_rows);"
    print "    }"
    next
}
{ print }
' "$MAIN_LIB" > "${MAIN_LIB}.tmp" && mv "${MAIN_LIB}.tmp" "$MAIN_LIB"

echo ">>> Building main..."
(cd "$WORKTREE_PATH" && cargo build --release -p cli --features jemalloc-stats 2>&1 | tail -2)

# Copy ELF into worktree if needed
mkdir -p "$WORKTREE_PATH/executor/program_artifacts/asm"
cp "$ELF" "$WORKTREE_PATH/executor/program_artifacts/asm/"

echo ">>> Running main (TABLE_PARALLELISM=1)..."
MAIN_OUTPUT=$(cd "$WORKTREE_PATH" && \
    TABLE_PARALLELISM=1 ./target/release/cli prove \
    executor/program_artifacts/asm/fib_iterative_2M.elf \
    -o /tmp/proof_main_$$.bin --time 2>&1)
rm -f /tmp/proof_main_$$.bin
echo "$MAIN_OUTPUT" | grep -E "^MEMW|Proving time|Peak heap"

MAIN_TIME=$(echo "$MAIN_OUTPUT"      | grep -o 'Proving time: [0-9.]*' | awk '{print $3}')
MAIN_HEAP=$(echo "$MAIN_OUTPUT"      | grep -o 'Peak heap: [0-9]*'     | awk '{print $3}')
MAIN_MEMW_ROWS=$(echo "$MAIN_OUTPUT" | grep '^MEMW '  | grep -o 'rows=[0-9]*'   | cut -d= -f2)
MAIN_MEMW_CHUNKS=$(echo "$MAIN_OUTPUT"| grep '^MEMW ' | grep -o 'chunks=[0-9]*' | cut -d= -f2)
echo ""

# ── Step 3: Print comparison ─────────────────────────────

TIME_SPEEDUP=$(awk "BEGIN { printf \"%.1f\", (($MAIN_TIME - $PR_TIME) / $MAIN_TIME) * 100 }")
HEAP_DELTA=$(awk  "BEGIN { printf \"+%.1f\", (($PR_HEAP - $MAIN_HEAP) / $MAIN_HEAP) * 100 }")

echo "======================================================="
echo "  Results"
echo "======================================================="
echo ""
echo "Row counts:"
printf "| %-7s | %-12s | %-12s |\n" "Table"  "main"              "PR"
printf "| %-7s | %-12s | %-12s |\n" "-------" "------------"     "------------"
printf "| %-7s | %-12s | %-12s |\n" "MEMW"   "$MAIN_MEMW_ROWS"  "$PR_MEMW_ROWS"
printf "| %-7s | %-12s | %-12s |\n" "MEMW_A" "—"                "$PR_MEMWA_ROWS"
echo ""
echo "Prove time (k=1):"
printf "| %-8s | %-10s | %-12s | %-10s |\n" "Branch" "Time (s)"  "Heap (MB)"  "Speedup"
printf "| %-8s | %-10s | %-12s | %-10s |\n" "--------" "----------" "------------" "----------"
printf "| %-8s | %-10s | %-12s | %-10s |\n" "main"   "${MAIN_TIME}s" "${MAIN_HEAP} MB" "baseline"
printf "| %-8s | %-10s | %-12s | %-10s |\n" "PR"     "${PR_TIME}s"  "${PR_HEAP} MB"   "${TIME_SPEEDUP}% (heap: ${HEAP_DELTA}%)"
echo ""
