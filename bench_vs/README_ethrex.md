# Ethrex Block Benchmarks

Benchmarks Lambda VM proving a stateless **ethrex** block (Ethereum state
execution) inside the zkVM. The same ethrex guest ELF is proven against
different block inputs:

| Block | Input fixture | ~Instructions | Mode |
|-------|---------------|---------------|------|
| empty block | `executor/tests/ethrex_empty_block.bin` | ~995k | monolithic |
| 1 transaction (plain ETH transfer) | `executor/tests/ethrex_simple_tx.bin` | ~1.6M | monolithic |
| 100 transfers | `executor/tests/ethrex_100_transfers.bin` (generated) | ~41M | `--continuations` |

Instruction counts measured 2026-07-27; they move when guest accelerators change
(the 1-tx block dropped from ~4.4M once keccak and ecrecover became ecalls, which also
put a 100-transfer block at ~410k cycles/transfer instead of the ~4M it used to cost).
The authoritative numbers are in the latest nightly artifact.

Each input is a serialized `ProgramInput` (the block + its execution witness,
rkyv-encoded) for the ethrex commit pinned (as `rev`) in
`executor/programs/rust/ethrex/Cargo.toml`. The guest reads it via
`get_private_input()` and runs ethrex's `execution_program`.

Proving is timed **end to end** (ELF load + execution + trace build + AIR
construction + STARK prove) and **excludes** verification; verification is timed
separately, right after, on the proof just produced.

The large block is proved with `--continuations`, which splits execution into
fixed-size epochs so peak memory stays flat instead of growing with the trace.
This is the production proving mode: monolithically, a block this size does not
fit in memory.

---

## 1. Running the benchmark locally

Prereqs: Rust stable + `nightly-2026-02-01`, and the RV64 sysroot (see
[§2](#2-generating-the-ethrex-elf)). The script builds the CLI and reuses an
existing `ethrex.elf` if present, otherwise builds it.

```bash
# Prove + verify every block in the script's BLOCKS list, print a summary table:
./bench_vs/run_ethrex.sh

# Cheaper run: size the continuation block down from 100 transfers to 10.
./bench_vs/run_ethrex.sh --cont-txs 10

# Write machine-readable reports (markdown + key=value metrics + raw stdout/stderr):
./bench_vs/run_ethrex.sh --report-dir bench_artifacts --no-color

# Bench the CUDA prover path (and enable `Peak heap:` reporting):
BENCH_FEATURES=jemalloc-stats,prover/cuda ./bench_vs/run_ethrex.sh
```

Output (measured 2026-07-27 at `e039384f` on an RTX 5090 + Ryzen 9 7950X,
`BENCH_FEATURES=jemalloc-stats,prover/cuda`). These predate #866 (7.7x faster ECSM witness
generation), which ethrex hits through ecrecover, so current prove times are lower:

```
  Program                    Prove (s)    Verify (s)          Cycles    Epochs   Heap (MB)
  ----------------------     ---------    ----------  --------------    ------   ---------
  ethrex empty block            7.566s        1.592s          760868       n/a        4904
  ethrex 1 tx                   8.574s        1.609s         1292897       n/a        5929
  ethrex 100tx cont           106.368s       10.106s        41272829        10       11091
```

Peak heap is `n/a` unless the CLI is built with the `jemalloc-stats` feature. Note the
continuation block's flat memory: 32x the cycles of the 1-tx block at ~2x the peak heap.

With `--report-dir DIR` it also writes:
- `DIR/ethrex_summary.md` — markdown table
- `DIR/ethrex_metrics.txt` — per block: `<slug>_time_s`, `<slug>_cycles`,
  `<slug>_verify_s`, `<slug>_epochs`, `<slug>_peak_heap_mb`, `<slug>_proof_bytes`;
  plus `bench_features` and `cont_epoch_size_log2`
- `DIR/raw/<slug>.stdout` / `.verify.stdout` / `.stderr`

### Adding more blocks
Append one line to the `BLOCKS` array in `bench_vs/run_ethrex.sh`. Fields are
`label|input_basename|continuations|epoch_size_log2` (`-` for the epoch size when
monolithic). A fixture named `ethrex_<N>_transfers.bin` is generated on demand via
`tooling/ethrex-fixtures`; anything else must already exist in `executor/tests/`.

```bash
BLOCKS=(
    "ethrex empty block|ethrex_empty_block.bin|0|-"
    "ethrex 1 tx|ethrex_simple_tx.bin|0|-"
    "ethrex 50tx cont|ethrex_50_transfers.bin|1|20"   # <-- new, generated on demand
)
```

### Daily runs
Two nightlies call this script:

- `.github/workflows/bench-vs-nightly.yml` — CPU, self-hosted bench runner. Also
  runs the fibonacci-vs-SP1 series.
- `.github/workflows/bench-gpu-nightly.yml` — rents a Vast.ai RTX 5090 and runs this
  script **twice on that box**, with and without `prover/cuda`, so the GPU/CPU ratio
  is measured on one host. Absolute times aren't comparable across nights (each is a
  different rented machine); the ratio is.

Both post to Slack via `.github/scripts/publish_bench_vs.sh`. Because the script is
data-driven, a block added to `BLOCKS` is picked up automatically, and the GPU
section renders every block it finds. To add a block to the CPU nightly's Slack
message, add an `ethrex_line` call in `publish_bench_vs.sh`.

---

## 2. Generating the ethrex ELF

`ethrex.elf` is **gitignored** (`executor/.gitignore`) and built on demand. The
fixtures (`*.bin`) are small and committed.

```bash
# One-time: fetch the RV64 sysroot used by the guest build.
make prepare-sysroot SYSROOT_DIR=$HOME/.lambda-vm-sysroot

# Build just the ethrex guest ELF (or `make compile-programs-rust` for all):
make executor/program_artifacts/rust/ethrex.elf SYSROOT_DIR=$HOME/.lambda-vm-sysroot
```

What the build needs:
- **Toolchains:** `1.94.0` stable (workspace) + `nightly-2026-02-01` with
  `rust-src` (the Makefile pins it; builds the guest via `-Z build-std`).
- **clang + lld** for ethrex's C dependencies.
- **Network**, the first time: cargo fetches `ethrex-guest-program` from
  `github.com/lambdaclass/ethrex.git` (commit pinned as `rev` in the guest `Cargo.toml`).
- **`SYSROOT_DIR` must match** between `prepare-sysroot` and the build.

The guest source is `executor/programs/rust/ethrex/` (a small `main.rs` that
reads the private input, calls `execution_program`, and commits the output).
