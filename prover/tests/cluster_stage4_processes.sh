#!/usr/bin/env bash
# Run on the CUDA box with the CUDA-enabled prover lib test executable.
set -euo pipefail
binary=$(realpath "${1:?CUDA-enabled prover lib test binary}")
logs=$(mktemp -d /tmp/lambda-vm-cluster-tcp.XXXXXX)
echo "Cluster process logs and proof archives in $logs"

export CUDA_VISIBLE_DEVICES=0 RAYON_NUM_THREADS=4
export LAMBDA_VM_DETERMINISTIC_GRIND=1 LAMBDA_VM_FIXED_TRACE_HASH=1
unset LAMBDA_VM_CLUSTER LAMBDA_VM_CARDS LAMBDA_VM_CARD_ALIASES
unset LAMBDA_VM_TWO_CARD_SPLIT LAMBDA_VM_TWO_CARD_LEVELS LAMBDA_VM_TWO_CARD_OPENINGS
parts=(base)
if [[ -z ${LAMBDA_VM_TEST_CLUSTER_ETHREX_ELF:-} ]]; then
    parts+=(level-1 level-2 shipped)
fi
coordinator=lfm::levels_tests::cluster_stage2_test_commit_4
worker=lfm::levels_tests::cluster_stage4_worker
pids=()
stop_workers() {
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    pids=()
}
trap stop_workers EXIT

topology() {
    python3 - "$1" "$2" "$3" <<'PYTHON'
import pathlib
import secrets
import socket
import sys

directory, workers, commit_ms = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
ports = [socket.socket() for _ in range(workers + 1)]
for s in ports:
    s.bind(("127.0.0.1", 0))
addresses = [f"127.0.0.1:{s.getsockname()[1]}" for s in ports]
tokens = [secrets.token_hex(32) for _ in range(workers)]

def write(name, own=None, wrong=False):
    with (directory / name).open("w") as f:
        print("coordinator", addresses[0], file=f)
        for i, (address, token) in enumerate(zip(addresses[1:], tokens)):
            if own is not None and own != i:
                token = "-"
            elif wrong:
                token = secrets.token_hex(32)
            print("worker", address, "0", token, file=f)
        print("deadlines-ms", 10000, commit_ms, 12000, file=f)

write("topology")
for i in range(workers):
    write(f"worker-{i}.conf", i)
write("wrong.conf", 1, True)
PYTHON
}

start_workers() {
    local dir=$1 count=$2 fault=$3 id config mode
    for ((id=0; id<count; id++)); do
        config="$dir/worker-$id.conf"
        mode=""
        if [[ $fault == global-balance || $id == 1 ]]; then
            if [[ $fault == wrong-token ]]; then
                config="$dir/wrong.conf"
            else
                mode=$fault
            fi
        fi
        env RAYON_NUM_THREADS=2 LAMBDA_VM_CARDS=0 LAMBDA_VM_CARD_ALIASES=1 \
            LAMBDA_VM_VRAM_BUDGET_MB=4096 LAMBDA_VM_TEST_CLUSTER_TOPOLOGY="$config" \
            LAMBDA_VM_TEST_CLUSTER_WORKER="$id" LAMBDA_VM_CLUSTER_TEST_FAULT="$mode" \
            LAMBDA_VM_BLOCK_WALKERS=ignored LAMBDA_VM_FIXED_TRACE_HASH=1 \
            "$binary" "$worker" --exact --ignored --nocapture --test-threads=1 \
            > "$dir/worker-$id.log" 2>&1 &
        pids+=("$!")
    done
}

run_coordinator() {
    local dir=$1
    shift
    env LAMBDA_VM_TEST_PROOF_DIR="$dir/proofs" "$@" \
        timeout 40s "$binary" "$coordinator" --exact --ignored --nocapture --test-threads=1 \
        > "$dir/coordinator.log" 2>&1
}

ok() {
    local dir=$1
    grep -E 'CLUSTER PROVE |test result:' "$dir/coordinator.log"
    grep -q 'test result: ok. 1 passed' "$dir/coordinator.log"
    if [[ ${LAMBDA_VM_FIXED_TRACE_HASH:-} == 1 && -d "$logs/local/proofs" && "$dir" != "$logs/local" ]]; then
        for part in "${parts[@]}"; do
            cmp "$dir/proofs/$part" "$logs/local/proofs/$part"
        done
    fi
}

show_fail() {
    local dir=$1
    for log in "$dir"/*.log; do
        echo "$log"
        tail -40 "$log"
    done
}

baseline() {
    local dir="$logs/$1"
    mkdir "$dir"
    shift
    if ! run_coordinator "$dir" "$@"; then
        show_fail "$dir"
        return 1
    fi
    echo "${dir##*/}"
    ok "$dir"
}

success() {
    local label=$1 count=$2 fault=${3:-} dir="$logs/$1"
    mkdir "$dir"
    topology "$dir" "$count" 12000
    start_workers "$dir" "$count" "$fault"
    if ! run_coordinator "$dir" LAMBDA_VM_TEST_CLUSTER_TOPOLOGY="$dir/topology"; then
        show_fail "$dir"
        return 1
    fi
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            show_fail "$dir"
            return 1
        fi
    done
    pids=()
    echo "$label"
    ok "$dir"
    python3 - "$dir/coordinator.log" "$count" <<'PYTHON'
import json
import pathlib
import sys

log = pathlib.Path(sys.argv[1]).read_text()
line = next(line for line in log.splitlines() if "completed groups by worker " in line)
groups = json.loads(line.rsplit("worker ", 1)[1])
assert len(groups) == int(sys.argv[2]) and all(groups), groups
print("Groups by worker", groups)
PYTHON
    if [[ $fault == stale ]]; then
        grep 'dropped .* stale frames' "$dir/coordinator.log"
    fi
}

failure() {
    local fault=$1 pattern=$2 dir="$logs/$1" ctl waited=0 killed=0 rc=0 commit_ms=12000
    mkdir "$dir"
    if [[ $fault == timeout ]]; then commit_ms=2500; fi
    topology "$dir" 2 "$commit_ms"
    start_workers "$dir" 2 "$fault"
    run_coordinator "$dir" LAMBDA_VM_TEST_CLUSTER_TOPOLOGY="$dir/topology" &
    ctl=$!
    if [[ $fault == commit || $fault == bind ]]; then
        until grep -q "CLUSTER TEST READY $fault" "$dir/worker-1.log"; do
            if ! kill -0 "$ctl" 2>/dev/null || [[ $waited == 400 ]]; then
                stop_workers
                wait "$ctl" || true
                show_fail "$dir"
                return 1
            fi
            sleep 0.05
            waited=$((waited + 1))
        done
        killed=$(date +%s%N)
        kill -KILL "${pids[1]}"
    fi
    wait "$ctl" || rc=$?
    local finished
    finished=$(date +%s%N)
    stop_workers
    if [[ $rc == 0 || $rc == 124 || -e "$dir/proofs" ]] || ! grep -qi "$pattern" "$dir/coordinator.log"; then
        show_fail "$dir"
        return 1
    fi
    echo "$fault"
    grep 'CLUSTER PROVE ' "$dir/coordinator.log"
    if [[ $killed != 0 ]]; then
        python3 - "$killed" "$finished" <<'PYTHON'
import sys
elapsed = (int(sys.argv[2]) - int(sys.argv[1])) / 1e9
print(f"Cancellation after SIGKILL {elapsed:.3f}s")
assert elapsed < 3, elapsed
PYTHON
    fi
}

baseline local
if [[ -n ${LAMBDA_VM_TEST_CLUSTER_ETHREX_ELF:-} ]]; then
    success ethrex 1
    exit
fi
baseline inproc2 LAMBDA_VM_CLUSTER=inproc:2
success tcp2 2
baseline inproc4 LAMBDA_VM_CLUSTER=inproc:4
success tcp4 4
failure wrong-token 'authentication failed'
failure duplicate 'duplicate commit reply'
failure group-index 'authenticated assignment'
failure receipt-shape 'derived roots, digests or counts'
failure witness-reply 'replica witness digest disagrees'
failure group-shape 'sealed plan or proof shape'
failure divergence 'disagrees with coordinator.*BITWISE lookup counts'
failure group-verification 'group .* verification'
failure global-balance 'failed verification.*BusImbalance'
failure commit 'commit: .*end of file'
failure bind 'open: .*end of file'
failure timeout 'commit.*deadline\|commit: .*timed out\|commit: .*temporarily unavailable'
success fresh2 2 stale


unset LAMBDA_VM_FIXED_TRACE_HASH
success unfixed2 2
success unfixed4 4

export LAMBDA_VM_FIXED_TRACE_HASH=1
mkdir "$logs/fallback"
topology "$logs/fallback" 2 12000
echo 'fallback local' >> "$logs/fallback/topology"
start_workers "$logs/fallback" 2 duplicate
if ! run_coordinator "$logs/fallback" LAMBDA_VM_TEST_CLUSTER_TOPOLOGY="$logs/fallback/topology"; then
    show_fail "$logs/fallback"
    exit 1
fi
stop_workers
echo fallback
grep 'starting a fresh local attempt' "$logs/fallback/coordinator.log"
ok "$logs/fallback"
