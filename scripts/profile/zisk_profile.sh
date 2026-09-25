#!/usr/bin/env bash
# zisk_profile.sh — Nsight Compute on ZisK's hot kernels, the counterpart of block_profile.sh's
# run B for OUR kernels: same sections, same explicit metrics, same env allowlist, same bundle
# rule. For a box whose GPU performance counters are unlocked. Adds nothing to our prover.
#
# WHY (GAP-W3/K0, thoughts/zf/gap/W3-KERNELS.md): per unit of work our NTT is ~10x, our
# constraint evaluation ~10x, our Merkle ~4x slower than ZisK's (Poseidon1 key) on the same
# block. The same-input microbenchmarks on FAST (no counters) measured the times; this run
# measures the bytes, passes, occupancy and instruction counts behind them, on ZisK's side.
#
# STAGE 1 — same-input microbenchmark (always; ~10 min): ZisK's production LDE and Merkle commit
#   (NTTGoldilocksGPU::LDE + PoseidonGoldilocksGPU<16>::merkletree arity 4 / BLAKE3 arity 2, both
#   column-major, the calls of starks_gpu.cu extendAndMerkelize_inplace) on one Goldilocks matrix:
#     wide   2^22 rows x 245 cols, blowup 2 (ZisK's Main shape), Poseidon1 and BLAKE3
#     narrow 2^22 rows x 32 cols,  blowup 4, Poseidon1
#   built from the PUBLISHED proofman-starks-src 1.3.0-alpha crate (crates.io, sha256-pinned; the
#   source cargo-zisk 1.3.0-alpha links), only the files the harness needs (no nasm/MPI/bn128).
#   Kernels: nttDifColMajorKernel, nttDitColMajorKernel, linearHashTiledKernel_pos1,
#   merkleNodeKernel_pos1, merkleNodeWarpKernel_pos1, b3_linearHashKernel, b3_merkleNodeKernel —
#   one launch of EVERY launch configuration (--filter-mode per-launch-config -c 1).
# STAGE 2 — ZisK's own prove (ZP_PROVE=1, the default; first run ~40-90 min, mostly the key):
#   installs the public ZisK 1.3.0-alpha release (no Docker) and its Poseidon1 proving key under
#   $ZP_HOME, then proves a committed zisk-eth-client workload (reth guest, block 25431013 — its ELF
#   and input are in that repo, so no guest build; AIR shapes are fixed per AIR, so the kernel
#   launches are the ones of any block) once plainly and once under ncu, filtered to the
#   expression kernels of the largest AIR (gen_basic_a0_0_b22_*), computeFRIExpressionFolded,
#   fold_reg, computeEvals_v2 and transposeFRI (NTT and Merkle are stage 1's).
#   Emulator mode (no --asm): the GPU kernels are the same, and no ASM services are spawned.
#
# USAGE (repo checkout of whir/profile-rpx; needs CUDA with nvcc, Nsight Compute, curl, git)
#   bash scripts/profile/zisk_profile.sh [out-dir]      # default scripts/profile/out/zisk-ncu-<UTC>
# ENV: GPU (index, default 0) · NCU (tool path) · STEP_TIMEOUT (seconds per ncu pass, default 5400)
#      ZP_PROVE (1 = stage 2, 0 = stage 1 only) · ZP_HOME (install dir, default $HOME/.zisk-prof;
#      ~30 GB free needed for stage 2) · ZP_NO_NCU=1 (plumbing check on a counter-locked box: runs
#      both stages without ncu and profiles nothing)
# Leaves OUT/small (text: read zisk_summary.txt, ncu_summary.txt, then *.details.txt), packed as
# OUT/small.tar.gz once the bundle scan passes, and OUT/big (.ncu-rep, raw exports, builds, logs).
# OUT/big EMBEDS THE MACHINE'S ENVIRONMENT: never commit, share or upload it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
. "$HERE/lib/common.sh"
case "${1:-}" in -h|--help) sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;; esac

GPU="${GPU:-0}"
STEP_TIMEOUT="${STEP_TIMEOUT:-5400}"
ZP_PROVE="${ZP_PROVE:-1}"
ZP_HOME="${ZP_HOME:-$HOME/.zisk-prof}"
ZP_NO_NCU="${ZP_NO_NCU:-0}"
PF_EXTRA_ENV=("CUDA_DEVICE_ORDER=PCI_BUS_ID" "CUDA_VISIBLE_DEVICES=$GPU")
OUT="${1:-$HERE/out/zisk-ncu-$(date -u +%Y%m%dT%H%M%SZ)}"
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then pf_die "$OUT exists and is not empty"; fi
mkdir -p "$OUT/small" "$OUT/big"
OUT="$(cd "$OUT" && pwd)"
SMALL="$OUT/small" BIG="$OUT/big"
exec > >(tee -a "$SMALL/run.log") 2>&1

# ---- pinned public inputs (read 2026-09-25) ------------------------------------------------
ZISK_VERSION="1.3.0-alpha"
CRATE_URL="https://static.crates.io/crates/proofman-starks-src/proofman-starks-src-${ZISK_VERSION}.crate"
CRATE_SHA256="3a7ebe850e6ba9c8cfbdae2d68c196e5453ff4548e91c4ce01b91ffc2e727384"
REL_URL="https://github.com/0xPolygonHermez/zisk/releases/download/v${ZISK_VERSION}/cargo_zisk_linux_amd64.tar.gz"
REL_SHA256="39e87f6c28eabc6de1226b1106054d61e6c03188bcdd94ac1d067c58b40145ca"
KEY_URL="https://storage.googleapis.com/zisk-setup/zisk-provingkey-${ZISK_VERSION}.tar.gz"
KEY_MD5="b3d31643958407c9f932e2226d4225e9"
KEY_SHA256="1e867238e6933c5cc37896d88a344379d29a73af1975c7c741c47be821dfbac2"
KEY_BYTES=5180527793
ZEC_URL="https://github.com/0xPolygonHermez/zisk-eth-client.git"
ZEC_SHA="1a58023c19780671d5f704e68263bbff04dd49c4"
ELF_REL="bin/guests/stateless-validator-reth/elf/zec-reth.elf"
ELF_SHA256="04f367f24e62dd0d757a6c820e55b4edb8c467cf46e567c028cc13d012daaaa7"
IN_REL="bin/guests/stateless-validator-reth/inputs/mainnet_25431013_172_16_zec_reth.bin"
IN_SHA256="2a16730874018a58ccc6c7fbb95d5d5d2817dbde7598f1101c67847eb7792b1b"

# ---- 0. tools, counters proven on one real kernel, card idle --------------------------------
NVCC="$(pf_cuda_home)/bin/nvcc"
[ -x "$NVCC" ] || pf_die "no nvcc at $NVCC; set CUDA_HOME"
for t in curl git tar python3 timeout nvidia-smi g++ md5sum; do command -v "$t" >/dev/null 2>&1 || pf_die "$t not found"; done
# ZisK's logger links GMP: apt-get install libgmp-dev (the ZisK bench kit's own dependency list has it)
compgen -G "/usr/include/gmp.h" > /dev/null || compgen -G "/usr/include/*/gmp.h" > /dev/null || pf_die "gmp.h not found: apt-get install libgmp-dev"
SM="$(pf_gpu_field "$GPU" compute_cap | tr -d .)"
[ -n "$SM" ] || pf_die "no compute capability for GPU $GPU"
pf_log "gpu $GPU: $(pf_gpu_field "$GPU" name), sm_$SM, $(pf_gpu_field "$GPU" memory.total) MiB, driver $(pf_gpu_field "$GPU" driver_version)"
SECTIONS=(--section SpeedOfLight --section SpeedOfLight_RooflineChart --section ComputeWorkloadAnalysis
          --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats --section Occupancy
          --section LaunchStats --section InstructionStats)
METRICS=""
if [ "$ZP_NO_NCU" = 1 ]; then
  pf_log "ZP_NO_NCU=1: plumbing check only, nothing is profiled"
else
  NCU_BIN="$(pf_find_tool ncu)" || pf_die "Nsight Compute not found; set NCU=/path/to/ncu"
  pf_log "ncu $NCU_BIN ($(pf_tool_version "$NCU_BIN"))"
  PROBE="$(pf_build_probe "$BIG/probe" "$GPU")" || pf_die "could not build the probe: see $BIG/probe/pf_probe.build.log"
  COUNTERS="$(pf_counter_probe "$NCU_BIN" "$PROBE" "$BIG/probe/pf_probe.ncu.log" "${SECTIONS[@]}" \
    --filter-mode per-launch-config -k 'regex:^pf_probe_kernel$' -c 1 -f -o "$BIG/probe/pf_probe_ncu")"
  pf_log "$COUNTERS"
  if [ "$COUNTERS" != "COUNTERS unlocked" ]; then
    pf_log "ncu cannot read the counters here: unlock them (README) or run as root, then re-run"; exit 2
  fi
  # the kit's explicit list plus DRAM bytes: bytes per butterfly / per leaf is the question here
  METRICS="$(pf_validate_metrics "$NCU_BIN" "$PROBE" "$BIG/probe" "${NCU_METRICS:-$PF_NCU_METRICS,dram__bytes_read.sum,dram__bytes_write.sum}")"
  pf_log "ncu metrics: $(printf '%s' "$METRICS" | awk -F, '{ print (length($0) ? NF : 0) }') of the explicit list collectable here"
fi
if ! reading="$(pf_gpu_idle "$GPU" 500)"; then pf_die "card not idle: $reading"; fi
pf_log "$reading"

fetch() { # fetch URL DEST SHA256 — download once, verify, keep
  local url="$1" dest="$2" want="$3" got
  if [ ! -s "$dest" ] || [ "$(pf_sha256 "$dest")" != "$want" ]; then
    pf_log "download $url"
    timeout 7200 curl -fL --retry 5 --retry-delay 10 -sS -C - -o "$dest.part" "$url" || pf_die "download failed: $url"
    mv -f "$dest.part" "$dest"
  fi
  got="$(pf_sha256 "$dest")"
  [ "$got" = "$want" ] || pf_die "sha256 mismatch for $dest: $got != $want"
}

pf_base_env
ncu_pass() { # ncu_pass <name> <kernel regex> <cmd...> : one launch of every configuration
  local name="$1" regex="$2" rc=0
  shift 2
  local -a mflag=()
  if [ -n "$METRICS" ]; then mflag=(--metrics "$METRICS"); fi
  pf_log "ncu $name: -k regex:$regex --filter-mode per-launch-config -c 1"
  env -i "${PF_ENV[@]}" ${ZP_ENV[@]+"${ZP_ENV[@]}"} timeout --signal=INT --kill-after=120 "$STEP_TIMEOUT" "$NCU_BIN" \
    "${SECTIONS[@]}" ${mflag[@]+"${mflag[@]}"} --target-processes all --filter-mode per-launch-config \
    -k "regex:$regex" -c 1 -f -o "$BIG/$name" "$@" > "$BIG/$name.stdout" 2>&1 || rc=$?
  { grep -E '^==(PROF|ERROR|WARNING)==' "$BIG/$name.stdout" || true; } > "$SMALL/$name.prof.txt"
  pf_log "ncu $name: rc=$rc, $(grep -c '^==PROF== Profiling' "$BIG/$name.stdout" || true) launch(es) profiled"
}
run_plain() { # run_plain <name> <cmd...> : the same command without ncu, output kept in SMALL
  local name="$1" rc=0
  shift
  env -i "${PF_ENV[@]}" ${ZP_ENV[@]+"${ZP_ENV[@]}"} timeout "$STEP_TIMEOUT" "$@" > "$BIG/$name.plain.log" 2>&1 || rc=$?
  pf_log "plain $name: rc=$rc"
  return "$rc"
}

# ---- 1. same-input microbenchmark: ZisK's LDE + Merkle commit --------------------------------
mkdir -p "$BIG/crate" "$BIG/zk"
fetch "$CRATE_URL" "$BIG/crate/src.crate" "$CRATE_SHA256"
tar -xzf "$BIG/crate/src.crate" -C "$BIG/crate"
S="$BIG/crate/proofman-starks-src-$ZISK_VERSION"
[ -d "$S/src/goldilocks/src" ] || pf_die "unexpected crate layout under $S"
INC=()
while IFS= read -r d; do INC+=("-I$d"); done < <(find "$S/src" -type d)
NVFLAGS=(--expt-relaxed-constexpr -std=c++17 -Xcompiler -fPIC -Xcompiler -mavx2 -Xcompiler -fopenmp -O3
  -gencode "arch=compute_${SM},code=sm_${SM}" -DGL64_PARTIALLY_REDUCED -D__AVX2__ -D__USE_ASSEMBLY__ -D__ADX__
  "-I$S/external/sppark/ff" "-I$S/external/sppark" "-I$S/external/sppark/util" "${INC[@]}")
G="$S/src/goldilocks"
objs=()
pf_log "build zk_k0 (sm_$SM) from the published crate"
for f in "$HERE/zisk/zk_k0.cu" "$G/src/ntt_goldilocks.cu" "$G/src/poseidon_goldilocks.cu" "$G/src/poseidon_goldilocks.cpp" \
         "$G/src/blake3_goldilocks.cu" "$G/src/blake3_goldilocks.cpp" "$G/src/goldilocks_tooling.cu" \
         "$G/src/goldilocks_base_field.cpp" "$G/utils/cuda_utils.cu" "$S/src/utils/zklog.cpp" \
         "$S/src/utils/exit_process.cpp" "$S/src/utils/utils.cpp" "$S/src/rapidsnark/logger.cpp"; do
  o="$BIG/zk/$(basename "$f").o"
  lang="c++"; case "$f" in *.cu) lang=cu ;; esac
  timeout 1800 "$NVCC" "${NVFLAGS[@]}" -x "$lang" -c "$f" -o "$o" >> "$BIG/zk/build.log" 2>&1 \
    || { tail -20 "$BIG/zk/build.log"; pf_die "build failed on $f (see $BIG/zk/build.log)"; }
  objs+=("$o")
done
timeout 600 "$NVCC" "${NVFLAGS[@]}" "${objs[@]}" -o "$BIG/zk/zk_k0" -lgmp -lgomp -lpthread >> "$BIG/zk/build.log" 2>&1 \
  || { tail -20 "$BIG/zk/build.log"; pf_die "link failed (see $BIG/zk/build.log)"; }
ZK="$BIG/zk/zk_k0"
# plain timings first (3 timed iterations after a warm-up), the numbers the summary quotes
for a in "22 1 245 p1" "22 1 245 b3" "22 2 32 p1" "21 2 245 p1"; do
  read -r nb bb m h <<< "$a"
  run_plain "micro-$h-$nb-$bb-$m" "$ZK" "$nb" "$bb" "$m" "$h" 3 || pf_die "zk_k0 $a failed (see $BIG/micro-$h-$nb-$bb-$m.plain.log)"
  grep '^K0 ' "$BIG/micro-$h-$nb-$bb-$m.plain.log" >> "$SMALL/micro_times.txt"
done
if [ "$ZP_NO_NCU" != 1 ]; then
  MICRO_RE='^(nttDitColMajorKernel|nttDifColMajorKernel|linearHashTiledKernel_pos1|merkleNodeKernel_pos1|merkleNodeWarpKernel_pos1|b3_linearHashKernel|b3_merkleNodeKernel)$'
  ncu_pass micro-p1-wide "$MICRO_RE" "$ZK" 22 1 245 p1 1
  ncu_pass micro-b3-wide "$MICRO_RE" "$ZK" 22 1 245 b3 1
  ncu_pass micro-p1-narrow "$MICRO_RE" "$ZK" 22 2 32 p1 1
fi

# ---- 2. ZisK's own prove: install, key, workload, one plain run, one ncu run ------------------
if [ "$ZP_PROVE" = 1 ]; then
  mkdir -p "$ZP_HOME/dl" "$ZP_HOME/zisk-home" "$ZP_HOME/keys"
  ZH="$ZP_HOME/zisk-home"
  if [ ! -x "$ZH/bin/cargo-zisk" ]; then
    fetch "$REL_URL" "$ZP_HOME/dl/cargo_zisk_linux_amd64.tar.gz" "$REL_SHA256"
    tar --no-same-owner -xzf "$ZP_HOME/dl/cargo_zisk_linux_amd64.tar.gz" -C "$ZH"
    # the release tarball carries bin/ (as ziskup installs it); accept a flat layout too
    if [ ! -x "$ZH/bin/cargo-zisk" ] && [ -x "$ZH/cargo-zisk" ]; then mkdir -p "$ZH/bin"; mv "$ZH"/cargo-zisk* "$ZH"/zisk* "$ZH"/lib*.a "$ZH/bin/" 2>/dev/null || true; fi
  fi
  [ -x "$ZH/bin/cargo-zisk" ] || pf_die "cargo-zisk not found under $ZH after extracting the release"
  PK="$ZP_HOME/keys/poseidon1/provingKey"
  if [ ! -d "$PK" ]; then
    df_gb="$(df -Pk "$ZP_HOME" | awk 'NR == 2 { print int($4 / 1048576) }')"
    [ "$df_gb" -ge 30 ] || pf_die "need ~30 GB free under $ZP_HOME for the proving key (have $df_gb GB)"
    kt="$ZP_HOME/dl/zisk-provingkey-$ZISK_VERSION.tar.gz"
    fetch "$KEY_URL" "$kt" "$KEY_SHA256"
    [ "$(wc -c < "$kt" | tr -d ' ')" = "$KEY_BYTES" ] || pf_die "proving key size differs from $KEY_BYTES"
    [ "$(md5sum "$kt" | awk '{ print $1 }')" = "$KEY_MD5" ] || pf_die "proving key md5 differs"
    mkdir -p "$ZP_HOME/keys/poseidon1"
    tar --no-same-owner -xzf "$kt" -C "$ZP_HOME/keys/poseidon1"
    [ -d "$PK" ] || pf_die "no provingKey directory in the key tarball"
  fi
  ZP_ENV=("ZISK_HOME=$ZH" "PATH=$ZH/bin:$PATH" "NO_COLOR=1" "OMPI_ALLOW_RUN_AS_ROOT=1" "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
  if [ ! -e "$ZP_HOME/check-setup.ok" ]; then
    pf_log "check-setup (const trees for the GPU; one time)"
    run_plain check-setup "$ZH/bin/cargo-zisk-dev" check-setup -k "$PK" --gpu || pf_die "check-setup failed (see $BIG/check-setup.plain.log)"
    touch "$ZP_HOME/check-setup.ok"
  fi
  W="$ZP_HOME/zisk-eth-client"
  if [ ! -s "$W/$ELF_REL" ] || [ "$(pf_sha256 "$W/$ELF_REL")" != "$ELF_SHA256" ]; then
    rm -rf "$W"; mkdir -p "$W"
    git -C "$W" init -q
    git -C "$W" remote add origin "$ZEC_URL"
    timeout 1800 git -C "$W" fetch -q --depth 1 --filter=blob:none origin "$ZEC_SHA" || pf_die "git fetch $ZEC_URL@$ZEC_SHA failed"
    timeout 1800 git -C "$W" checkout -q "$ZEC_SHA" -- "$ELF_REL" "$IN_REL" || pf_die "checkout of the workload files failed"
  fi
  [ "$(pf_sha256 "$W/$ELF_REL")" = "$ELF_SHA256" ] || pf_die "workload ELF sha256 differs"
  [ "$(pf_sha256 "$W/$IN_REL")" = "$IN_SHA256" ] || pf_die "workload input sha256 differs"
  PROVE=("$ZH/bin/cargo-zisk" prove -e "$W/$ELF_REL" -i "$W/$IN_REL" -k "$PK" --gpu)
  pf_log "plain prove (first run also builds the ROM setup for this ELF)"
  run_plain prove "${PROVE[@]}" -o "$BIG/proof-plain.bin" || pf_die "plain prove failed (see $BIG/prove.plain.log)"
  { grep -E 'Proof generated|steps' "$BIG/prove.plain.log" || true; } > "$SMALL/prove_plain.txt"
  if [ "$ZP_NO_NCU" != 1 ]; then
    # NTT and Merkle are stage 1's (same kernels, the same shapes); this pass takes what only a prove has
    PROVE_RE='^(gen_basic_a0_0_b22_.*|computeFRIExpressionFolded|fold_reg|computeEvals_v2|transposeFRI)$'
    ncu_pass prove "$PROVE_RE" "${PROVE[@]}" -o "$BIG/proof-ncu.bin"
    { grep -E 'Proof generated|steps' "$BIG/prove.stdout" || true; } > "$SMALL/prove_ncu.txt"
  fi
fi

# ---- 3. exports: ncu's own text first, then one line per launch -----------------------------
for r in "$BIG"/*.ncu-rep; do
  if [ ! -e "$r" ]; then continue; fi
  n="$(basename "${r%.ncu-rep}")"
  timeout 600 "$NCU_BIN" --import "$r" --page details > "$SMALL/$n.details.txt" 2>&1 || true
  timeout 600 "$NCU_BIN" --import "$r" --page details --csv > "$SMALL/$n.details.csv" 2>/dev/null || true
  timeout 600 "$NCU_BIN" --import "$r" --page raw --csv > "$BIG/$n.raw.csv" 2>/dev/null || true
done
if compgen -G "$BIG/*.raw.csv" > /dev/null; then
  python3 "$HERE/lib/ncu_summary.py" --out "$SMALL" "$BIG"/*.raw.csv > /dev/null 2>&1 || pf_log "ncu_summary.py failed (the .details.txt files stand)"
  python3 - "$SMALL/zisk_summary.txt" "$BIG"/*.raw.csv <<'PY' || pf_log "zisk summary failed (the .details.txt files stand)"
# One line per profiled launch: kernel, grid, block, duration, DRAM bytes read/written, L2 hit, SM %, occupancy.
import csv, sys
out = open(sys.argv[1], "w")
want = ["Kernel Name", "Grid Size", "Block Size", "gpu__time_duration.sum", "dram__bytes_read.sum",
        "dram__bytes_write.sum", "lts__t_sector_hit_rate.pct", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active", "smsp__thread_inst_executed.sum"]
for f in sys.argv[2:]:
    rows = list(csv.reader(open(f, errors="replace")))
    if len(rows) < 3: continue
    h = rows[0]; units = rows[1]
    idx = {w: h.index(w) for w in want if w in h}
    out.write(f"## {f.rsplit('/', 1)[-1]}\n" + "\t".join(f"{w} [{units[idx[w]]}]" for w in idx) + "\n")
    for r in rows[2:]:
        out.write("\t".join(r[idx[w]] for w in idx) + "\n")
PY
fi

# ---- 4. the bundle: only OUT/small leaves the box, and only when the scan passes -------------
rm -f "$OUT/small.tar.gz"
if scan="$(pf_scan_bundle "$SMALL" "$BIG/bundle_scan.txt")"; then
  tar -czf "$OUT/small.tar.gz" -C "$OUT" small
  pf_log "$scan"
  pf_log "DONE: send back $OUT/small.tar.gz ($(du -h "$OUT/small.tar.gz" | cut -f1)). Read first: zisk_summary.txt, micro_times.txt, then *.details.txt"
  pf_log "NEVER commit, share or upload $BIG: its .ncu-rep files embed this machine's environment"
else
  pf_log "$scan"
  pf_log "NOTHING TO SEND: read and remove the flagged lines first (the findings list is in OUT/big and is not to be shared); no tarball written"
  exit 4
fi
