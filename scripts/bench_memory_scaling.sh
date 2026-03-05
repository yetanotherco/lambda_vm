#!/bin/bash
# Benchmark memory scaling: how peak heap grows with program size.
#
# Generates fib programs of various sizes, proves each, and reports
# peak heap (jemalloc) and proving time in a table.
#
# Usage:
#   scripts/bench_memory_scaling.sh [options]
#
# Options:
#   --sizes "160 372 700 1400"    Space-separated list of program sizes in thousands of cycles
#   --max-rows-log2 15            Power of 2 for max rows per table (default: use production defaults)
#   --runs 1                      Number of runs per size (median is reported if >1)
#   --compare <commit>            Also benchmark a comparison commit (e.g. ea254b8, main~3)
#   --output <dir>                Directory for results and artifacts (default: /tmp/bench_scaling)
#
# Examples:
#   # Quick local test
#   scripts/bench_memory_scaling.sh --sizes "160 372 700" --max-rows-log2 15
#
#   # Full scaling sweep on a server
#   scripts/bench_memory_scaling.sh --sizes "160 372 700 1400 2800 5600" --max-rows-log2 14 --runs 3
#
#   # Compare current branch against pre-PR commit
#   scripts/bench_memory_scaling.sh --sizes "160 700 1400" --max-rows-log2 15 --compare ea254b8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

# --- Defaults -----------------------------------------------------------------

SIZES="160 372 700 1400"
MAX_ROWS_LOG2=""
RUNS=1
COMPARE_REF=""
OUTPUT_DIR="/tmp/bench_scaling"

# --- Parse args ---------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case $1 in
        --sizes)        SIZES="$2"; shift 2 ;;
        --max-rows-log2) MAX_ROWS_LOG2="$2"; shift 2 ;;
        --runs)         RUNS="$2"; shift 2 ;;
        --compare)      COMPARE_REF="$2"; shift 2 ;;
        --output)       OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            head -25 "$0" | tail -23
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

CURRENT_BRANCH=$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD)
CURRENT_SHA=$(git -C "$ROOT_DIR" rev-parse --short HEAD)

# Restore branch + stash on exit if we checked out something else
if [ -n "$COMPARE_REF" ]; then
    trap 'git -C "$ROOT_DIR" checkout -- "$ROOT_DIR/Cargo.lock" 2>/dev/null; git -C "$ROOT_DIR" checkout "$CURRENT_BRANCH" --quiet 2>/dev/null; git -C "$ROOT_DIR" checkout -- "$ROOT_DIR/Cargo.lock" 2>/dev/null; git -C "$ROOT_DIR" stash pop --quiet 2>/dev/null || true' EXIT
fi

# --- Helpers ------------------------------------------------------------------

median() {
    sort -n "$1" | awk '{a[NR]=$1} END {
        if (NR%2==1) print a[(NR+1)/2];
        else print (a[NR/2]+a[NR/2+1])/2
    }'
}

# Generate a fib iterative .s file for a given number of thousands of cycles.
# 5 instructions per loop iteration, so iterations = cycles_k * 1000 / 5.
generate_fib_elf() {
    local cycles_k=$1
    local elf_path=$2
    local iterations=$(( cycles_k * 1000 / 5 ))
    local asm_path="${elf_path%.elf}.s"

    cat > "$asm_path" << EOF
	.attribute	5, "rv64i2p1_m2p0"
	.globl	main
main:
	li	t0, 0
	li	t1, 1
	li	a0, ${iterations}
.loop:
	add	t2, t0, t1
	mv	t0, t1
	mv	t1, t2
	addi	a0, a0, -1
	bnez	a0, .loop
	mv	a0, t1
	li	a7, 5
	ecall
EOF
    clang --target=riscv64 -fuse-ld=lld -nostdlib -Wl,-e,main "$asm_path" -o "$elf_path" 2>/dev/null
}

# Cross-platform sed -i helper (macOS vs Linux).
sed_i() {
    if [[ "$(uname)" == "Darwin" ]]; then
        sed -i '' "$@"
    else
        sed -i "$@"
    fi
}

# Patch max_rows constants in the source tree to use a uniform 2^N for all tables.
# This is needed for compare builds where the CLI doesn't have --max-rows-log2.
patch_max_rows() {
    local log2=$1
    local mod_rs="$ROOT_DIR/prover/src/tables/mod.rs"
    if [ ! -f "$mod_rs" ]; then
        echo -e "${RED}Warning: $mod_rs not found, skipping max_rows patch${NC}"
        return
    fi
    for table in CPU MEMW DVRM MUL LT LOAD BRANCH; do
        sed_i -E "s/(pub const ${table}: usize = 1 << )[0-9]+/\1${log2}/" "$mod_rs"
    done
    echo -e "${GREEN}  Patched max_rows to 2^${log2} in mod.rs${NC}"
}

# Inject memory-trace feature + checkpoints into an older codebase that lacks them.
# Patches Cargo.toml files, injects mem_checkpoint function and calls into prover.rs,
# and adds trace-gen checkpoints to prover/src/lib.rs.
patch_memory_trace() {
    local stark_toml="$ROOT_DIR/crypto/stark/Cargo.toml"
    local prover_toml="$ROOT_DIR/prover/Cargo.toml"
    local cli_toml="$ROOT_DIR/bin/cli/Cargo.toml"
    local prover_rs="$ROOT_DIR/crypto/stark/src/prover.rs"
    local lib_rs="$ROOT_DIR/prover/src/lib.rs"

    # Skip if memory-trace already exists (current branch has it)
    if grep -q 'memory-trace' "$stark_toml" 2>/dev/null; then
        echo -e "${GREEN}  memory-trace already present, skipping patch${NC}"
        return
    fi

    echo -e "${GREEN}  Patching memory-trace into source...${NC}"

    # --- Cargo.toml patches (using awk for reliable cross-platform newlines) ---

    # stark/Cargo.toml: add feature after debug-checks + dep after rayon
    awk '
    /^debug-checks = / { print; print "memory-trace = [\"dep:tikv-jemalloc-ctl\"]"; next }
    /^rayon = / { print; print "tikv-jemalloc-ctl = { version = \"0.6\", features = [\"stats\"], optional = true }"; next }
    { print }
    ' "$stark_toml" > "${stark_toml}.tmp" && mv "${stark_toml}.tmp" "$stark_toml"

    # prover/Cargo.toml: add memory-trace feature after debug-checks
    awk '
    /^debug-checks = / { print; print "memory-trace = [\"stark/memory-trace\"]"; next }
    { print }
    ' "$prover_toml" > "${prover_toml}.tmp" && mv "${prover_toml}.tmp" "$prover_toml"

    # cli/Cargo.toml: add memory-trace feature after jemalloc-stats
    awk '
    /^jemalloc-stats = / { print; print "memory-trace = [\"prover/memory-trace\"]"; next }
    { print }
    ' "$cli_toml" > "${cli_toml}.tmp" && mv "${cli_toml}.tmp" "$cli_toml"

    # --- prover.rs: inject mem_checkpoint function + checkpoint calls ---

    awk '
    BEGIN { in_r1_build = 0; fn_inserted = 0 }

    # Insert mem_checkpoint function before first "pub struct"
    /^pub struct/ && fn_inserted == 0 {
        print "#[cfg(feature = \"memory-trace\")]"
        print "pub fn mem_checkpoint(label: &str) {"
        print "    tikv_jemalloc_ctl::epoch::advance().ok();"
        print "    if let Ok(allocated) = tikv_jemalloc_ctl::stats::allocated::read() {"
        print "        let mb = allocated / (1024 * 1024);"
        print "        eprintln!(\"[mem] {label}: {mb} MB\");"
        print "    }"
        print "}"
        print ""
        fn_inserted = 1
    }

    # After "let num_airs = air_trace_pairs.len();" -> start checkpoint
    /let num_airs = air_trace_pairs\.len\(\);/ {
        print
        print ""
        print "        #[cfg(feature = \"memory-trace\")]"
        print "        mem_checkpoint(&format!(\"multi_prove start ({num_airs} tables)\"));"
        next
    }

    # After main_commits.push(MainCommitData { ... }); -> Phase A checkpoint (a0035c1+ code)
    /main_commits\.push\(/ { in_main_push = 1 }
    in_main_push == 1 && /\}\);/ {
        in_main_push = 0
        print
        print ""
        print "            #[cfg(feature = \"memory-trace\")]"
        print "            mem_checkpoint(&format!(\"Phase A: committed ({}/{})\", main_commits.len(), num_airs));"
        next
    }

    # Before "let lookup_challenges" -> Phase A done
    /let lookup_challenges: Vec/ {
        print "        #[cfg(feature = \"memory-trace\")]"
        print "        mem_checkpoint(\"Phase A done (all main traces committed)\");"
        print ""
    }

    # Old code (92666dc): track round_1_build_auxiliary_trace call to insert after its )?;
    /round_1_build_auxiliary_trace\(/ { in_r1_build = 1 }
    in_r1_build == 1 && /\)\?;/ {
        in_r1_build = 0
        print
        print ""
        print "            #[cfg(feature = \"memory-trace\")]"
        print "            mem_checkpoint(&format!(\"aux built + committed ({}/{})\", idx + 1, num_airs));"
        next
    }

    # a0035c1 code: after table_transcripts.push -> aux committed checkpoint
    /table_transcripts\.push\(table_transcript\);/ {
        print
        print ""
        print "            #[cfg(feature = \"memory-trace\")]"
        print "            mem_checkpoint(&format!(\"aux built + committed ({}/{})\", metadatas.len(), num_airs));"
        next
    }

    # After "proofs.push(proof);" -> checkpoint (no drop — round_1_result may be used after)
    /proofs\.push\(proof\);/ {
        print
        print ""
        print "            #[cfg(feature = \"memory-trace\")]"
        print "            mem_checkpoint(&format!(\"table done ({}/{})\", proofs.len(), num_airs));"
        next
    }

    { print }
    ' "$prover_rs" > "${prover_rs}.tmp" && mv "${prover_rs}.tmp" "$prover_rs"

    # --- prover/src/lib.rs: inject trace-gen checkpoints ---
    if [ -f "$lib_rs" ]; then
        awk '
        # After executor .run() ... )?; -> execution done
        /\.run\(\)/ { saw_run = 1 }
        saw_run == 1 && /\)\?;/ {
            saw_run = 0
            print
            print ""
            print "    #[cfg(feature = \"memory-trace\")]"
            print "    stark::prover::mem_checkpoint(\"execution done\");"
            next
        }

        # After from_elf_and_logs ... ?; -> traces generated
        /from_elf_and_logs/ { saw_traces = 1 }
        saw_traces == 1 && /\?;/ {
            saw_traces = 0
            print
            print ""
            print "    #[cfg(feature = \"memory-trace\")]"
            print "    stark::prover::mem_checkpoint(\"traces generated\");"
            next
        }

        { print }
        ' "$lib_rs" > "${lib_rs}.tmp" && mv "${lib_rs}.tmp" "$lib_rs"
    fi

    echo -e "${GREEN}  Patched memory-trace into source${NC}"
}

# Restore ALL patched files to their git state.
restore_all_patches() {
    git -C "$ROOT_DIR" checkout -- \
        "$ROOT_DIR/prover/src/tables/mod.rs" \
        "$ROOT_DIR/crypto/stark/Cargo.toml" \
        "$ROOT_DIR/crypto/stark/src/prover.rs" \
        "$ROOT_DIR/prover/Cargo.toml" \
        "$ROOT_DIR/prover/src/lib.rs" \
        "$ROOT_DIR/bin/cli/Cargo.toml" \
        2>/dev/null || true
}

# Build the CLI binary for a given label. If compare_ref is set, checks out that ref first.
# For compare builds: stashes local changes, checks out the ref, patches, builds, restores.
build_cli() {
    local label=$1
    local ref=${2:-}
    if [ -n "$ref" ]; then
        # Save current branch to restore later
        ORIG_BRANCH=$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
        # Stash any local modifications so checkout is clean
        echo -e "${GREEN}[$label] Stashing local changes...${NC}"
        git -C "$ROOT_DIR" stash --quiet 2>/dev/null || true
        echo -e "${GREEN}[$label] Checking out $ref...${NC}"
        git -C "$ROOT_DIR" checkout "$ref" --quiet 2>/dev/null
    fi
    # For compare builds, patch max_rows and memory-trace into old source
    if [ "$label" = "compare" ]; then
        [ -n "$MAX_ROWS_LOG2" ] && patch_max_rows "$MAX_ROWS_LOG2"
        patch_memory_trace
    fi
    echo -e "${GREEN}[$label] Building CLI (release + jemalloc-stats)...${NC}"
    if ! cargo build --release -p cli --features jemalloc-stats,memory-trace --manifest-path "$ROOT_DIR/Cargo.toml" 2>&1; then
        echo -e "${RED}[$label] Build FAILED${NC}"
        # Restore before aborting
        if [ "$label" = "compare" ]; then
            restore_all_patches
            git -C "$ROOT_DIR" checkout -- Cargo.lock 2>/dev/null || true
            git -C "$ROOT_DIR" checkout "${ORIG_BRANCH:-main}" --quiet 2>/dev/null
            git -C "$ROOT_DIR" stash pop --quiet 2>/dev/null || true
        fi
        exit 1
    fi
    cp "$ROOT_DIR/target/release/cli" "$OUTPUT_DIR/cli-$label"
    echo -e "${GREEN}[$label] Binary ready.${NC}"
    # Restore patched files and go back to original branch
    if [ "$label" = "compare" ]; then
        restore_all_patches
        git -C "$ROOT_DIR" checkout -- Cargo.lock 2>/dev/null || true
        git -C "$ROOT_DIR" checkout "${ORIG_BRANCH:-main}" --quiet 2>/dev/null
        git -C "$ROOT_DIR" stash pop --quiet 2>/dev/null || true
    fi
}

# Run prove for one size, one binary, collecting results.
bench_size() {
    local cli=$1
    local label=$2
    local cycles_k=$3
    local elf="$OUTPUT_DIR/elfs/fib_${cycles_k}k.elf"

    # Only pass --max-rows-log2 for the current build (which has the flag).
    # Compare builds have max_rows patched at compile time instead.
    local max_rows_flag=""
    if [ -n "$MAX_ROWS_LOG2" ] && [ "$label" = "current" ]; then
        max_rows_flag="--max-rows-log2 $MAX_ROWS_LOG2"
    fi

    local heap_file="$OUTPUT_DIR/results/${label}_${cycles_k}k_heap.txt"
    local time_file="$OUTPUT_DIR/results/${label}_${cycles_k}k_time.txt"
    rm -f "$heap_file" "$time_file"

    for i in $(seq 1 "$RUNS"); do
        local run_label="[$label] ${cycles_k}k"
        if [ "$RUNS" -gt 1 ]; then
            run_label="[$label] ${cycles_k}k run $i/$RUNS"
        fi
        echo -ne "  ${YELLOW}${run_label}: proving...${NC}"

        local stdout_tmp="$OUTPUT_DIR/tmp_stdout.txt"
        local stderr_tmp="$OUTPUT_DIR/tmp_stderr.txt"
        # shellcheck disable=SC2086
        if ! "$cli" prove "$elf" -o "$OUTPUT_DIR/tmp_proof.bin" --time $max_rows_flag \
            > "$stdout_tmp" 2>"$stderr_tmp"; then
            echo -e " ${RED}FAILED${NC}"
            cat "$stderr_tmp"
            rm -f "$OUTPUT_DIR/tmp_proof.bin" "$stdout_tmp" "$stderr_tmp"
            continue
        fi

        local heap_mb
        heap_mb=$(grep -o 'Peak heap: [0-9]*' "$stdout_tmp" | awk '{print $3}')
        local time_s
        time_s=$(grep -o 'Proving time: [0-9.]*' "$stdout_tmp" | awk '{print $3}')

        [ -n "$heap_mb" ] && echo "$heap_mb" >> "$heap_file"
        [ -n "$time_s" ] && echo "$time_s" >> "$time_file"

        echo -e "\r  ${GREEN}${run_label}: ${time_s}s, ${heap_mb:-?} MB${NC}          "

        # Save memory trace if present
        local mem_trace_file="$OUTPUT_DIR/results/${label}_${cycles_k}k_memtrace.txt"
        if grep -q '^\[mem\]' "$stderr_tmp" 2>/dev/null; then
            grep '^\[mem\]' "$stderr_tmp" > "$mem_trace_file"
        fi

        rm -f "$OUTPUT_DIR/tmp_proof.bin" "$stdout_tmp" "$stderr_tmp"
    done
}

# --- Setup --------------------------------------------------------------------

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/elfs" "$OUTPUT_DIR/results"

echo -e "${BOLD}=== Memory Scaling Benchmark ===${NC}"
echo "  Sizes (k cycles): $SIZES"
echo "  Max rows log2:    ${MAX_ROWS_LOG2:-default}"
echo "  Runs per size:    $RUNS"
echo "  Compare ref:      ${COMPARE_REF:-none}"
echo ""

# --- Generate ELFs ------------------------------------------------------------

echo -e "${GREEN}Generating fib ELFs...${NC}"
for k in $SIZES; do
    generate_fib_elf "$k" "$OUTPUT_DIR/elfs/fib_${k}k.elf"
done
echo ""

# --- Build binaries -----------------------------------------------------------

build_cli "current"
if [ -n "$COMPARE_REF" ]; then
    build_cli "compare" "$COMPARE_REF"
    # Reset Cargo.lock (modified by cargo build on old ref) so stash pop won't conflict
    git -C "$ROOT_DIR" checkout -- "$ROOT_DIR/Cargo.lock" 2>/dev/null || true
    git -C "$ROOT_DIR" checkout "$CURRENT_BRANCH" --quiet 2>/dev/null
    git -C "$ROOT_DIR" checkout -- "$ROOT_DIR/Cargo.lock" 2>/dev/null || true
    git -C "$ROOT_DIR" stash pop --quiet 2>/dev/null || true
fi
echo ""

# --- Run benchmarks -----------------------------------------------------------

run_all_sizes() {
    local cli=$1
    local label=$2
    echo -e "${BOLD}--- Benchmarking: $label ---${NC}"
    for k in $SIZES; do
        bench_size "$cli" "$label" "$k"
    done
}

run_all_sizes "$OUTPUT_DIR/cli-current" "current"
if [ -n "$COMPARE_REF" ]; then
    run_all_sizes "$OUTPUT_DIR/cli-compare" "compare"
fi

# --- Print results ------------------------------------------------------------

print_table() {
    local label=$1
    echo -e "\n${BOLD}  $label${NC}"
    printf "  %-12s %12s %12s %12s\n" "Program" "Time (s)" "Heap (MB)" "Delta (MB)"
    printf "  %-12s %12s %12s %12s\n" "-------" "--------" "---------" "----------"

    local prev_heap=0
    for k in $SIZES; do
        local heap_file="$OUTPUT_DIR/results/${label}_${k}k_heap.txt"
        local time_file="$OUTPUT_DIR/results/${label}_${k}k_time.txt"

        local heap_val="N/A"
        local time_val="N/A"
        local delta="—"

        if [ -f "$heap_file" ]; then
            if [ "$RUNS" -gt 1 ]; then
                heap_val=$(median "$heap_file")
            else
                heap_val=$(cat "$heap_file")
            fi
        fi
        if [ -f "$time_file" ]; then
            if [ "$RUNS" -gt 1 ]; then
                time_val=$(median "$time_file")
            else
                time_val=$(cat "$time_file")
            fi
        fi

        if [ "$prev_heap" -gt 0 ] 2>/dev/null && [ "$heap_val" != "N/A" ]; then
            delta="+$(( heap_val - prev_heap ))"
        fi

        printf "  %-12s %12s %12s %12s\n" "${k}k" "$time_val" "$heap_val" "$delta"

        if [ "$heap_val" != "N/A" ]; then
            prev_heap=$heap_val
        fi
    done
}

echo ""
echo -e "${BOLD}=== Results ===${NC}"
if [ -n "$MAX_ROWS_LOG2" ]; then
    echo "  Max rows: 2^${MAX_ROWS_LOG2} = $(( 1 << MAX_ROWS_LOG2 ))"
else
    echo "  Max rows: production defaults"
fi
echo "  Runs:     $RUNS"

print_table "current"
if [ -n "$COMPARE_REF" ]; then
    print_table "compare"

    # Summary: growth rate comparison
    echo ""
    echo -e "${BOLD}  Scaling comparison:${NC}"
    read -ra SIZES_ARR <<< "$SIZES"
    FIRST_K=${SIZES_ARR[0]}
    LAST_K=${SIZES_ARR[${#SIZES_ARR[@]}-1]}

    cur_first=$(cat "$OUTPUT_DIR/results/current_${FIRST_K}k_heap.txt" 2>/dev/null || echo 0)
    cur_last=$(cat "$OUTPUT_DIR/results/current_${LAST_K}k_heap.txt" 2>/dev/null || echo 0)
    cmp_first=$(cat "$OUTPUT_DIR/results/compare_${FIRST_K}k_heap.txt" 2>/dev/null || echo 0)
    cmp_last=$(cat "$OUTPUT_DIR/results/compare_${LAST_K}k_heap.txt" 2>/dev/null || echo 0)

    if [ "$cur_first" -gt 0 ] && [ "$cmp_first" -gt 0 ] && [ "$FIRST_K" != "$LAST_K" ]; then
        cur_growth=$((cur_last - cur_first))
        cmp_growth=$((cmp_last - cmp_first))
        cycle_range=$(( (LAST_K - FIRST_K) ))
        cur_rate=$(awk "BEGIN {printf \"%.1f\", $cur_growth / $cycle_range}")
        cmp_rate=$(awk "BEGIN {printf \"%.1f\", $cmp_growth / $cycle_range}")
        echo "    current: ${FIRST_K}k→${LAST_K}k = +${cur_growth} MB  (${cur_rate} MB/k-cycle)"
        echo "    compare: ${FIRST_K}k→${LAST_K}k = +${cmp_growth} MB  (${cmp_rate} MB/k-cycle)"
    fi
fi

# --- Memory traces ------------------------------------------------------------

# Summarize a memory trace file into a compact phase breakdown.
# Extracts key milestones, computes deltas, and shows a simple bar chart.
summarize_mem_trace() {
    local trace_file=$1
    [ -f "$trace_file" ] || return

    # Extract key milestones from [mem] lines, compute deltas, show bar chart.
    # Handles both new code (Phase A/C/Rounds) and old code (per-table loop) labels.
    awk '
    {
        if (!/^\[mem\]/) next
        nf = NF
        if ($nf != "MB") next
        mb = $(nf - 1) + 0

        line = $0
        sub(/^\[mem\] /, "", line)
        sub(/: [0-9]+ MB$/, "", line)

        keep = 0
        if (line ~ /execution done/)            { tag = "exec done";      keep = 1 }
        if (line ~ /traces generated/)           { tag = "trace gen";      keep = 1 }
        if (line ~ /multi_prove start/)          { tag = "multi_prove";    keep = 1 }
        if (line ~ /pools allocated/)            { tag = "pools alloc";    keep = 1 }
        if (line ~ /Phase A done/)               { tag = "Phase A done";   keep = 1 }
        if (line ~ /Phase C pass 1 done/)        { tag = "aux traces";     keep = 1 }
        if (line ~ /Phase C done/)               { tag = "Phase C done";   keep = 1 }

        # New code (merged loop): per-table proving (last entry = final)
        if (line ~ /Rounds 2-4: finished/ || line ~ /: proved/) {
            last_r24_mb = mb
            r24_count++
            if (r24_count == 1) first_r24_mb = mb
        }

        # Old code: per-table loop (aux built, table done)
        if (line ~ /aux built/) {
            last_aux_mb = mb
            aux_count++
            if (aux_count == 1) first_aux_mb = mb
        }
        if (line ~ /table done/) {
            last_table_mb = mb
            table_count++
            if (table_count == 1) first_table_mb = mb
        }

        if (keep) {
            names[n] = tag
            vals[n] = mb
            n++
        }
    }
    END {
        # Append per-table loop summary (works for both old and new code)
        if (r24_count > 0) {
            # New code: show first/last Rounds 2-4
            names[n] = "prove 1st"; vals[n] = first_r24_mb; n++
            if (r24_count > 1) {
                names[n] = "prove last"; vals[n] = last_r24_mb; n++
            }
        } else if (table_count > 0) {
            # Old code: show first/last table done (after drop)
            names[n] = "1st tbl done"; vals[n] = first_table_mb; n++
            if (table_count > 1) {
                names[n] = "last tbl done"; vals[n] = last_table_mb; n++
            }
        }

        max_mb = 0
        for (i = 0; i < n; i++) if (vals[i] > max_mb) max_mb = vals[i]
        bar_w = 35

        for (i = 0; i < n; i++) {
            d = (i == 0) ? 0 : vals[i] - vals[i-1]
            ds = (i == 0) ? "     —" : sprintf("%+6d", d)

            bl = (max_mb > 0) ? int(vals[i] * bar_w / max_mb) : 0
            bar = ""
            for (j = 0; j < bl; j++) bar = bar "#"

            printf "    %-14s %6d MB  %s MB  %s\n", names[i], vals[i], ds, bar
        }

        # Show per-table loop trend
        if (table_count > 1) {
            diff = last_table_mb - first_table_mb
            printf "\n    Per-table loop: %d tables, 1st=%d MB, last=%d MB, delta=%+d MB\n", table_count, first_table_mb, last_table_mb, diff
            if (diff < 50 && diff > -50)
                print "    -> Memory FLAT during per-table proving (good)"
            else if (diff < 0)
                print "    -> Memory DECREASED during per-table proving (good)"
            else
                print "    -> Memory GREW during per-table proving (bad)"
        }
        if (r24_count > 1) {
            diff = last_r24_mb - first_r24_mb
            printf "\n    Per-table loop: %d tables, 1st=%d MB, last=%d MB, delta=%+d MB\n", r24_count, first_r24_mb, last_r24_mb, diff
            if (diff < 50 && diff > -50)
                print "    -> Memory FLAT during per-table proving (good)"
            else if (diff < 0)
                print "    -> Memory DECREASED during per-table proving (good)"
            else
                print "    -> Memory GREW during per-table proving (bad)"
        }
    }
    ' "$trace_file"
}

print_mem_traces() {
    local label=$1
    local has_traces=false
    for k in $SIZES; do
        local trace_file="$OUTPUT_DIR/results/${label}_${k}k_memtrace.txt"
        if [ -f "$trace_file" ]; then
            has_traces=true
            break
        fi
    done

    if ! $has_traces; then
        return
    fi

    echo ""
    echo -e "${BOLD}  Memory breakdown: $label${NC}"
    printf "    %-14s %9s %11s\n" "Phase" "Heap" "Delta"
    printf "    %-14s %9s %11s\n" "-----" "----" "-----"

    for k in $SIZES; do
        local trace_file="$OUTPUT_DIR/results/${label}_${k}k_memtrace.txt"
        if [ -f "$trace_file" ]; then
            echo ""
            echo -e "    ${YELLOW}${k}k cycles:${NC}"
            summarize_mem_trace "$trace_file"
        fi
    done
}

print_mem_traces "current"
if [ -n "$COMPARE_REF" ]; then
    print_mem_traces "compare"
fi

echo ""
echo "Raw data in $OUTPUT_DIR/results/"
