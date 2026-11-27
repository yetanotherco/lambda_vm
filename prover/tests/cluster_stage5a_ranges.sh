#!/usr/bin/env bash
# Run on a CUDA host with the prover library tests built with cuda,test-security.
# Workers use independent TCP processes and share the visible card on a one-card host.
set -euo pipefail
binary=$(realpath "${1:?prover lib test binary}")
logs=$(mktemp -d /tmp/lambda-vm-cluster-ranges.XXXXXX)
echo "Cluster range logs in $logs"

export CUDA_VISIBLE_DEVICES=0 RAYON_NUM_THREADS=4
export LAMBDA_VM_DETERMINISTIC_GRIND=1 LAMBDA_VM_FIXED_TRACE_HASH=1
export LAMBDA_VM_BLOCK_WALKERS=2
unset LAMBDA_VM_CLUSTER LAMBDA_VM_CLUSTER_WINDOWS LAMBDA_VM_CARDS LAMBDA_VM_CARD_ALIASES
unset LAMBDA_VM_TEST_CLUSTER_TOPOLOGY LAMBDA_VM_TEST_CLUSTER_ETHREX_ELF
unset LAMBDA_VM_TEST_CLUSTER_BASE_ONLY LAMBDA_VM_TEST_CLUSTER_BOUNDARY_WORDS
unset LAMBDA_VM_TWO_CARD_SPLIT LAMBDA_VM_TWO_CARD_LEVELS LAMBDA_VM_TWO_CARD_OPENINGS
coordinator=lfm::levels_tests::cluster_stage2_test_commit_4
worker=lfm::levels_tests::cluster_stage4_worker
parts=(base level-1 level-2 shipped)
pids=()

stop_workers() {
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    pids=()
}
trap stop_workers EXIT

run() {
    local name=$1
    shift
    mkdir -p "$logs/$name"
    env LAMBDA_VM_TEST_PROOF_DIR="$logs/$name" "$@" \
        timeout 40s "$binary" "$coordinator" --exact --ignored --nocapture --test-threads=1 \
        > "$logs/$name/prove.log" 2>&1
}

report() {
    local name=$1
    echo "$name"
    grep -E 'BLOCK CLUSTER RANGE |CLUSTER ARCHIVE (LOCAL PHASE A|FILE VERIFY)|CLUSTER PROVE |test result:' "$logs/$name/"*.log
}

compare() {
    local a=$1 b=$2 part
    for part in "${parts[@]}"; do
        cmp "$logs/$a/$part" "$logs/$b/$part"
    done
}

topology() {
    python3 - "$1" "$2" "${3:-}" <<'PY'
import pathlib
import secrets
import socket
import sys

directory, cuts, fallback = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
workers = len(cuts.split(","))
ports = [socket.socket() for _ in range(workers + 1)]
for sock in ports:
    sock.bind(("127.0.0.1", 0))
addresses = [f"127.0.0.1:{sock.getsockname()[1]}" for sock in ports]
tokens = [secrets.token_hex(32) for _ in range(workers)]

for owner in range(-1, workers):
    path = directory / ("topology" if owner == -1 else f"worker-{owner}.conf")
    with path.open("w") as out:
        print("coordinator", addresses[0], file=out)
        for index, (address, token) in enumerate(zip(addresses[1:], tokens)):
            print("worker", address, "0", token if owner in (-1, index) else "-", file=out)
        print("windows", cuts, file=out)
        print("deadlines-ms 8000 20000 18000", file=out)
        if fallback:
            print("fallback local", file=out)
PY
}

tcp() {
    local name=$1 cuts=$2 count=$3 seed=$4 fault=${5:-} reason=${6:-} faulty_worker=${7:-0}
    local id injected
    mkdir -p "$logs/$name"
    topology "$logs/$name" "$cuts"
    for ((id=0; id<count; id++)); do
        injected=
        if ((id == faulty_worker)); then injected=$fault; fi
        env RAYON_NUM_THREADS=2 LAMBDA_VM_CARDS=0 \
            LAMBDA_VM_FIXED_TRACE_HASH="$seed" LAMBDA_VM_VRAM_BUDGET_MB="${WORKER_VRAM_MB:-4096}" \
            LAMBDA_VM_TEST_CLUSTER_TOPOLOGY="$logs/$name/worker-$id.conf" \
            LAMBDA_VM_TEST_CLUSTER_WORKER="$id" LAMBDA_VM_CLUSTER_TEST_FAULT="$injected" \
            timeout 40s "$binary" "$worker" --exact --ignored --nocapture --test-threads=1 \
            > "$logs/$name/worker-$id.log" 2>&1 &
        pids+=("$!")
    done
    if run "$name" LAMBDA_VM_FIXED_TRACE_HASH="$seed" LAMBDA_VM_TEST_CLUSTER_TOPOLOGY="$logs/$name/topology"; then
        [[ -z $fault ]] || return 1
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                tail -n 60 "$logs/$name/"*.log
                return 1
            fi
        done
        pids=()
        report "$name"
    elif [[ -n $fault ]]; then
        if ! grep -E "$reason" "$logs/$name/prove.log"; then
            tail -n 60 "$logs/$name/"*.log
            return 1
        fi
        grep 'test result:' "$logs/$name/prove.log"
        test ! -e "$logs/$name/base" && test ! -e "$logs/$name/shipped"
        stop_workers
    else
        tail -n 60 "$logs/$name/"*.log
        return 1
    fi
}

refuses_windows() {
    local name=$1 word=$2 reason=$3
    if run "$name" LAMBDA_VM_CLUSTER_WINDOWS="$word"; then
        echo "$name unexpectedly proved"
        return 1
    fi
    grep -F "$reason" "$logs/$name/prove.log"
    test ! -e "$logs/$name/shipped"
}

killed_worker_falls_back() {
    local name=killed-worker i coordinator_pid worker_pid
    mkdir -p "$logs/$name"
    topology "$logs/$name" 1 local
    env RAYON_NUM_THREADS=2 LAMBDA_VM_CARDS=0 \
        LAMBDA_VM_TEST_CLUSTER_TOPOLOGY="$logs/$name/worker-0.conf" \
        LAMBDA_VM_TEST_CLUSTER_WORKER=0 LAMBDA_VM_CLUSTER_TEST_FAULT=range-build \
        timeout 40s "$binary" "$worker" --exact --ignored --nocapture --test-threads=1 \
        > "$logs/$name/worker-0.log" 2>&1 &
    worker_pid=$!
    pids+=("$worker_pid")
    run "$name" LAMBDA_VM_TEST_CLUSTER_TOPOLOGY="$logs/$name/topology" &
    coordinator_pid=$!
    for ((i=0; i<300; i++)); do
        if grep -q 'CLUSTER TEST READY range-build' "$logs/$name/worker-0.log"; then
            break
        fi
        sleep 0.05
    done
    if ((i == 300)); then
        tail -n 60 "$logs/$name/"*.log
        return 1
    fi
    kill -KILL "$(pgrep -P "$worker_pid")"
    local status=0
    wait "$worker_pid" || status=$?
    test "$status" -eq 137
    pids=()
    if ! wait "$coordinator_pid"; then
        tail -n 60 "$logs/$name/"*.log
        return 1
    fi
    grep -F 'starting a fresh local attempt' "$logs/$name/prove.log"
    compare fallback-reference "$name"
    report "$name"
}

if [[ ${2:-} == ethrex ]]; then
    export LAMBDA_VM_TEST_CLUSTER_ETHREX_ELF="${3:?ethrex.elf}"
    export LAMBDA_VM_TEST_CLUSTER_ETHREX_WORDS=1
    export LAMBDA_VM_TEST_CLUSTER_ETHREX_WINDOW_LOG2=${WINDOW_LOG2:-12}
    export LAMBDA_VM_BLOCK_WALKERS=${WALKERS:-4}
    if [[ -n ${4:-} ]]; then
        export LAMBDA_VM_TEST_CLUSTER_ETHREX_INPUT="$4"
    fi
    if [[ ${ETHREX_LEVELS:-0} == 1 ]]; then
        export LAMBDA_VM_TEST_CLUSTER_ETHREX_LEVELS=1
    else
        parts=(base)
    fi
    if ! run single; then tail -n 60 "$logs/single/prove.log"; exit 1; fi
    report single
    cycles=$(awk '/CLUSTER ARCHIVE LOCAL PHASE A:/ {print $(NF-1); exit}' "$logs/single/prove.log")
    windows=$((cycles / (1 << LAMBDA_VM_TEST_CLUSTER_ETHREX_WINDOW_LOG2)))
    if ((windows < 4)); then
        echo "Ethereum fixture too short for this window size"
        exit 1
    fi
    if [[ ${ETHREX_LEVELS:-0} == 1 ]]; then
        run reference-1 LAMBDA_VM_CLUSTER_WINDOWS=-
        run inproc-1 LAMBDA_VM_CLUSTER_WINDOWS=- LAMBDA_VM_CLUSTER=inproc:1
        compare reference-1 inproc-1
        report reference-1
        report inproc-1
        topologies=(2 4)
    else
        topologies=(2)
    fi
    for ranges in "${topologies[@]}"; do
        cuts=$((windows / ranges))
        for ((id=2; id<ranges; id++)); do
            cuts+=",$((windows * id / ranges))"
        done
        tcp "tcp-$ranges" "$cuts" "$((ranges - 1))" 1
        if ! run "reference-$ranges" LAMBDA_VM_CLUSTER_WINDOWS="$cuts"; then
            tail -n 60 "$logs/reference-$ranges/prove.log"
            exit 1
        fi
        report "reference-$ranges"
        compare "reference-$ranges" "tcp-$ranges"
    done
    exit 0
fi

if ! run default; then tail -n 60 "$logs/default/prove.log"; exit 1; fi
report default
(cd "$logs/default" && sha256sum -c <<'SHA'
d095e74da057f953a3ea05165ab11bf3c0048ea91eccb1d58d28520fd36aefbd  base
4bee7e8d308417b7490e1922c227a6b4ab662cf53c79c484ef29bf8e5e1ce087  level-1
b123c6e54713e31649021920f535ecfbc924a6e4b214b114350e54fd5c5b0ac6  level-2
d92f54d12bd6950c962d1e9574f1e7526e364e0ece826d7f0d964882db2f0cf7  shipped
SHA
)
run reference-1 LAMBDA_VM_CLUSTER_WINDOWS=- || { tail -n 70 "$logs/reference-1/prove.log"; exit 1; }
run inproc-1 LAMBDA_VM_CLUSTER_WINDOWS=- LAMBDA_VM_CLUSTER=inproc:1 || { tail -n 70 "$logs/inproc-1/prove.log"; exit 1; }
grep -q 'BLOCK CLUSTER RANGE 0: cycles 0..18' "$logs/reference-1/prove.log"
grep -q 'BLOCK CLUSTER RANGE 0: cycles 0..18' "$logs/inproc-1/prove.log"
# One range runs the range builder, whose layout differs from the default
# path's; its own bytes are pinned here.
(cd "$logs/reference-1" && sha256sum -c <<'SHA'
ee1a0d9c0de968d30ae7922a9fa8ba5513a5869bb16d7044f340110219a6626b  base
baf722cc8d56a111f197e2e2914b8fbfd8cdce6aaee4c4435f0760f1ca119cbb  level-1
5983e10b10c75c8e65b8dd378ad337c643c8e854d3cd2d805cc89313bbf9e189  level-2
e2737016401f3745b71fd3953aa2d3c6aedffd298d03f1566aa4738564064699  shipped
SHA
)
if cmp -s "$logs/default/base" "$logs/reference-1/base"; then
    echo "one range took the default path"
    exit 1
fi
compare reference-1 inproc-1
report reference-1
report inproc-1
refuses_windows empty-windows "" 'empty window list'
refuses_windows invalid-windows 1,no 'invalid window index'
refuses_windows non-increasing-windows 2,2 'invalid window ranges or CPU chunk size'

export LAMBDA_VM_TEST_CLUSTER_RANGE_FIXTURE=1
for case in '2 1' '4 1,2,4'; do
    read -r ranges cuts <<< "$case"
    run "reference-$ranges" LAMBDA_VM_CLUSTER_WINDOWS="$cuts" || {
        tail -n 70 "$logs/reference-$ranges/prove.log"
        exit 1
    }
    run "inproc-$ranges" LAMBDA_VM_CLUSTER_WINDOWS="$cuts" LAMBDA_VM_CLUSTER="inproc:$ranges" || {
        tail -n 70 "$logs/inproc-$ranges/prove.log"
        exit 1
    }
    compare "reference-$ranges" "inproc-$ranges"
    report "reference-$ranges"
    report "inproc-$ranges"
    tcp "tcp-$ranges-fixed" "$cuts" "$((ranges - 1))" 1
    compare "reference-$ranges" "tcp-$ranges-fixed"
    tcp "tcp-$ranges-random" "$cuts" "$((ranges - 1))" 0
done

# The cut after window 4 leaves the halting window with the coordinator.
grep -q 'BLOCK CLUSTER RANGE 3: cycles 16..18' "$logs/tcp-4-fixed/prove.log"
tcp wrong-index-base 1 1 1 wrong-index-base 'group [0-9]+ verification'
tcp wrong-stream-base 2,4 2 1 wrong-stream-base 'group [0-9]+ verification' 1
grep -Eq 'streamed group [1-9][0-9]* committed at' "$logs/wrong-stream-base/worker-1.log"
tcp missing-range 1 1 1 missing-range 'injected missing range'
tcp stream-count-mismatch 1 1 1 stream-count-mismatch 'streamed group count disagrees with range receipt'
tcp stream-count-cap 2,4 2 1 stream-count-cap 'groups — a block takes at most'
for fault in bitwise-increment bitwise-multiple-rows decode-increment keccak-increment; do
    tcp "$fault" 1 1 1 "$fault" 'cluster base proof failed verification: Proof\(BusImbalance\)'
done
tcp bitwise-short 1 1 1 bitwise-short 'invalid range BITWISE counts'
tcp bitwise-field-limit 1 1 1 bitwise-field-limit 'invalid range BITWISE counts'
tcp decode-unordered 1 1 1 decode-unordered 'unordered DECODE count rows'
tcp decode-out-of-range 1 1 1 decode-out-of-range 'invalid DECODE row'
export LAMBDA_VM_TEST_CLUSTER_BOUNDARY_WORDS=1
for fault in boundary-register boundary-memory boundary-drop-cycle; do
    tcp "$fault" 1,2 2 1 "$fault" 'cluster base proof failed verification: Proof\(BusImbalance\)' 1
done
unset LAMBDA_VM_TEST_CLUSTER_BOUNDARY_WORDS

export LAMBDA_VM_TEST_CLUSTER_BASE_ONLY=1
run overestimated LAMBDA_VM_CLUSTER_WINDOWS=1,10,20
grep -q 'BLOCK CLUSTER RANGE 2: cycles 0..0' "$logs/overestimated/prove.log"
grep -q 'BLOCK CLUSTER RANGE 3: cycles 16..18' "$logs/overestimated/prove.log"
report overestimated
tcp empty-worker 1,10,20 3 1
grep -q 'BLOCK CLUSTER RANGE 2: cycles 0..0' "$logs/empty-worker/worker-2.log"
cmp "$logs/overestimated/base" "$logs/empty-worker/base"
unset LAMBDA_VM_TEST_CLUSTER_BASE_ONLY
run fallback-reference
killed_worker_falls_back
