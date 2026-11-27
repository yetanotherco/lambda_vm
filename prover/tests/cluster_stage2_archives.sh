#!/usr/bin/env bash
# On the CUDA box, pass the CUDA-enabled prover lib test binary and the
# ethrex ELF used by the stage 1 gate. The optional third argument supplies
# independently captured batch 10 archives for a direct byte comparison.
# Build with test-security and run from the workspace root.
set -euo pipefail

test_binary=$(realpath "${1:?prover lib test binary}")
ethrex=$(realpath "${2:?ethrex.elf from the stage 1 gate}")
reference=
if [[ -n ${3:-} ]]; then
    reference=$(realpath "$3")
fi
output=$(mktemp -d /tmp/lambda-vm-cluster-archives.XXXXXX)
echo "Cluster archive logs and files in $output"

export LAMBDA_VM_DETERMINISTIC_GRIND=1 LAMBDA_VM_FIXED_TRACE_HASH=1
unset LAMBDA_VM_CLUSTER LAMBDA_VM_CARDS LAMBDA_VM_CARD_ALIASES
unset LAMBDA_VM_TWO_CARD_SPLIT LAMBDA_VM_TWO_CARD_LEVELS LAMBDA_VM_TWO_CARD_OPENINGS
export CUDA_VISIBLE_DEVICES=0 RAYON_NUM_THREADS=8

# adc08bf8a, built with cuda,test-security, using the settings below.
cat > "$output/test_commit_4.sha256" <<'SHA'
d095e74da057f953a3ea05165ab11bf3c0048ea91eccb1d58d28520fd36aefbd  base
4bee7e8d308417b7490e1922c227a6b4ab662cf53c79c484ef29bf8e5e1ce087  level-1
b123c6e54713e31649021920f535ecfbc924a6e4b214b114350e54fd5c5b0ac6  level-2
d92f54d12bd6950c962d1e9574f1e7526e364e0ece826d7f0d964882db2f0cf7  shipped
SHA
cat > "$output/ethrex_simple_tx.sha256" <<'SHA'
e285ce7951229751a4d9e347a1f141a7443fdb65592a6e6b10e1a6d5a5f235e2  base
f244daecc4088b084a28ddd8c5ee81b0f540e38272217b6fff32505d9857ee64  level-1
60ff76de3b9c4bc37bd716c396bae918573d238127c57d92ab78ad2ed49520d6  level-2
7f8294fedaa611b60660bd18795162ee908d98641f18ec0a4ea14698f8f114b3  shipped
SHA

run_fixture() {
    local fixture=$1 test_name=$2 mode dir
    shift 2
    for mode in local 1 2 4; do
        dir="$output/$fixture-$mode"
        mkdir "$dir"
        if [[ "$mode" != local ]]; then
            export LAMBDA_VM_CLUSTER="inproc:$mode"
        fi
        # Each invocation gets a fresh process for the cached proof settings.
        if ! env LAMBDA_VM_TEST_PROOF_DIR="$dir" "$@" \
            "$test_binary" "$test_name" --exact --ignored --nocapture --test-threads=1 \
            > "$dir.log" 2>&1; then
            tail -80 "$dir.log"
            return 1
        fi
        echo "$fixture ($mode)"
        grep 'test result: ok. 1 passed' "$dir.log"
        if [[ "$mode" == local ]]; then
            grep 'CLUSTER ARCHIVE LOCAL PHASE A:' "$dir.log"
            (cd "$dir" && sha256sum -c "$output/$fixture.sha256")
            if [[ -n $reference ]]; then
                for part in base level-1 level-2 shipped; do
                    cmp "$dir/$part" "$reference/$fixture/$part"
                done
            fi
        else
            local workers
            workers=$(grep 'BLOCK CLUSTER INPROC: completed groups by worker' "$dir.log")
            echo "$workers"
            grep 'BLOCK CLUSTER HANDOUT:' "$dir.log"
            grep 'BLOCK CLUSTER REPLICA ' "$dir.log"
            test "$(grep -c 'BLOCK CLUSTER REPLICA ' "$dir.log")" -eq "$mode"
            grep -Eq 'BLOCK CLUSTER HANDOUT: .* · [1-9][0-9]* terminal' "$dir.log"
            python3 - "$fixture" "$mode" "${workers##*worker }" <<'PYTHON'
import json
import sys

fixture, count, groups = sys.argv[1], int(sys.argv[2]), json.loads(sys.argv[3])
assert len(groups) == count and sum(groups) > 0, groups
# The archived Ethereum fixture has a single canonical group.
if fixture == "test_commit_4":
    assert all(groups), groups
PYTHON
            for part in base level-1 level-2 shipped; do
                cmp "$dir/$part" "$output/$fixture-local/$part"
            done
        fi
    done
    unset LAMBDA_VM_CLUSTER
}

run_fixture test_commit_4 lfm::levels_tests::cluster_stage2_test_commit_4
# Use the stage 1 Ethereum proving and storage profile for raw archive parity.
run_fixture ethrex_simple_tx lfm::levels_tests::rta_a_real_block_through_its_levels \
    BLOCK_WHIR_ELF="$ethrex" BLOCK_WHIR_INPUT="$PWD/executor/tests/ethrex_simple_tx.bin" \
    RTA_POSTURE=test RTA_FAMILY=b3 RTA_LEVELS=2 RTA_BESIDE=1 RTA_BUILD_BESIDE=1 RTA_RUNS=1 \
    RAYON_NUM_THREADS=4 LAMBDA_VM_MAX_ROWS_LOG2=21 LFM_WHIR_RETENTION=1 LAMBDA_VM_WHIR_HASH=rpx \
    LFM_PRECOMPUTED_TREE_CACHE_CAP=64 LFM_EXEC_PARALLEL=1 LAMBDA_VM_BASE_SPLIT=1 \
    LAMBDA_VM_GRIND_SCAN_FACTOR=8 LAMBDA_VM_GRIND_GRID=1024 TABLE_PARALLELISM=4 \
    LAMBDA_VM_BACKEND_BEFORE_PROVE=1 LAMBDA_VM_BLOCK_COMPRESS=0 LAMBDA_VM_CLAIMS_TILED=1 \
    BLOCK_WHIR_MEMORY=words LAMBDA_VM_PINNED_POOL_GIB=32 LAMBDA_VM_POOL_EXEC_LOGS=1 \
    LAMBDA_VM_POOL_ARENA=class LAMBDA_VM_B3L_RECORDS=1 LAMBDA_VM_WLFM_CHIP_RECORDS=1 \
    LAMBDA_VM_ZF_WHIR_FOLDS=first6 LAMBDA_VM_B3_BASE_ROUNDS=1 \
    LAMBDA_VM_CLOSE_BEFORE_BITWISE=0 LAMBDA_VM_COMMIT_STREAMED_LEAVES=1 \
    _RJEM_MALLOC_CONF=dirty_decay_ms:-1,muzzy_decay_ms:-1,thp:always,metadata_thp:always
