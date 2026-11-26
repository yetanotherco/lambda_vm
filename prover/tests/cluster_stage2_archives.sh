#!/usr/bin/env bash
# On the CUDA box, pass the CUDA-enabled prover lib test binary and the
# ethrex ELF used by the stage 1 gate. Run from the workspace root.
set -euo pipefail

test_binary=$(realpath "${1:?prover lib test binary}")
ethrex=$(realpath "${2:?ethrex.elf from the stage 1 gate}")
output=$(mktemp -d /tmp/lambda-vm-cluster-archives.XXXXXX)
echo "Cluster archive logs and files in $output"

export LAMBDA_VM_DETERMINISTIC_GRIND=1 LAMBDA_VM_FIXED_TRACE_HASH=1
unset LAMBDA_VM_CLUSTER LAMBDA_VM_CARDS LAMBDA_VM_CARD_ALIASES
unset LAMBDA_VM_TWO_CARD_SPLIT LAMBDA_VM_TWO_CARD_LEVELS LAMBDA_VM_TWO_CARD_OPENINGS
export CUDA_VISIBLE_DEVICES=0 RAYON_NUM_THREADS=8

# The stage 1 raw archives matched both 717a57ec5 and e2bb5f79f.
cat > "$output/test_commit_4.sha256" <<'SHA'
4e72550ad5998c7d86f5768d36fc422f5a444ae0780acbec57260de62de8398c  base
45a0f7b78e017779e581d5bdbb13ea1d381088088a4215465873647998904975  level-1
9aa19f039c80ddaeb1cd23cca82cddb640a66790ccd4ed7b523154bbee59b9dd  level-2
4253a3e1e79f54b929b40159e1b1582977e476830a94ebb0965aaae08e824292  shipped
SHA
cat > "$output/ethrex_simple_tx.sha256" <<'SHA'
4ccc483bf0c1e91921b073ad43bac4c819f3cad4f078c85e02f63752bbffaf1a  base
5ceed2741bcb1d29b1408769a54fa29eb268079472a4168c25c8bd26d0ed9986  level-1
4903b2fc89f6f0c0978700747c7cedd4bbf937a4bbe6157c287778e4a4d588a7  level-2
4986f04daaae3f68ddf136bd7e156926cb91dcea1ef123b8c61d07250675370d  shipped
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
            (cd "$dir" && sha256sum -c "$output/$fixture.sha256")
        else
            local workers
            workers=$(grep 'BLOCK CLUSTER INPROC: completed groups by worker' "$dir.log")
            echo "$workers"
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
    RTA_FAMILY=b3 RTA_LEVELS=2 RTA_BESIDE=1 RTA_BUILD_BESIDE=1 RTA_RUNS=1 \
    RAYON_NUM_THREADS=4 LAMBDA_VM_MAX_ROWS_LOG2=21 LFM_WHIR_RETENTION=1 LAMBDA_VM_WHIR_HASH=rpx \
    LFM_PRECOMPUTED_TREE_CACHE_CAP=64 LFM_EXEC_PARALLEL=1 LAMBDA_VM_BASE_SPLIT=1 \
    LAMBDA_VM_GRIND_SCAN_FACTOR=8 LAMBDA_VM_GRIND_GRID=1024 TABLE_PARALLELISM=4 \
    LAMBDA_VM_BACKEND_BEFORE_PROVE=1 LAMBDA_VM_BLOCK_COMPRESS=0 LAMBDA_VM_CLAIMS_TILED=1 \
    BLOCK_WHIR_MEMORY=words LAMBDA_VM_PINNED_POOL_GIB=32 LAMBDA_VM_POOL_EXEC_LOGS=1 \
    LAMBDA_VM_POOL_ARENA=class LAMBDA_VM_B3L_RECORDS=1 LAMBDA_VM_WLFM_CHIP_RECORDS=1 \
    LAMBDA_VM_ZF_WHIR_FOLDS=first6 LAMBDA_VM_B3_BASE_ROUNDS=1 \
    LAMBDA_VM_CLOSE_BEFORE_BITWISE=0 LAMBDA_VM_COMMIT_STREAMED_LEAVES=1 \
    _RJEM_MALLOC_CONF=dirty_decay_ms:-1,muzzy_decay_ms:-1,thp:always,metadata_thp:always
