# Ethrex Block Benchmarks

Benchmarks Lambda VM proving a stateless **ethrex** block (Ethereum state
execution) inside the zkVM. The same ethrex guest ELF is proven against
different block inputs:

| Block | Input fixture | ~Instructions |
|-------|---------------|---------------|
| empty block | `executor/tests/ethrex_empty_block.bin` | ~184k |
| 1 transaction (plain ETH transfer) | `executor/tests/ethrex_simple_tx.bin` | ~4.4M |

Each input is a serialized `ProgramInput` (the block + its execution witness,
rkyv-encoded) for the ethrex commit pinned in
`executor/programs/rust/ethrex/Cargo.lock`. The guest reads it via
`get_private_input()` and runs ethrex's `execution_program`.

The timing window is **single-shot end-to-end prove** (ELF load + execution +
trace build + AIR construction + STARK prove); it **excludes** verification.

---

## 1. Running the benchmark locally

Prereqs: Rust stable + `nightly-2026-02-01`, and the RV64 sysroot (see
[§2](#2-generating-the-ethrex-elf)). The script builds the CLI and reuses an
existing `ethrex.elf` if present, otherwise builds it.

```bash
# Prove every block in the script's BLOCKS list, print a summary table:
./bench_vs/run_ethrex.sh

# Write machine-readable reports (markdown + key=value metrics + raw stdout/stderr):
./bench_vs/run_ethrex.sh --report-dir bench_artifacts --no-color
```

Output (example):

```
  Program                     Lambda (s)   Lambda cycles
  ----------------------      ----------   -------------
  ethrex empty block             11.549s          183931
  ethrex 1 tx                    47.302s         4392951
```

With `--report-dir DIR` it also writes:
- `DIR/ethrex_summary.md` — markdown table
- `DIR/ethrex_metrics.txt` — `<slug>_time_s=` / `<slug>_cycles=` per block
- `DIR/raw/<slug>.stdout` / `.stderr`

### Adding more blocks
Append one line to the `BLOCKS` array in `bench_vs/run_ethrex.sh` and drop the
fixture into `executor/tests/`:

```bash
BLOCKS=(
    "ethrex empty block|ethrex_empty_block.bin"
    "ethrex 1 tx|ethrex_simple_tx.bin"
    "ethrex 5 txs|ethrex_5_txs.bin"      # <-- new
)
```

### Daily run
The nightly workflow `.github/workflows/bench-vs-nightly.yml` calls
`run_ethrex.sh` and posts results to Slack via
`.github/scripts/publish_bench_vs.sh`. Because the script is data-driven, any
block added to `BLOCKS` is picked up automatically; to also show it in the
Slack post, add a line in `publish_bench_vs.sh` (see the `ethrex_line` helper).

> Note: the nightly currently copies a **cached** `ethrex.elf` onto the runner
> (a temporary step until the sysroot is provisioned there). Refresh that cache
> when the guest or its ethrex dependency changes.

---

## 2. Generating the ethrex ELF

`ethrex.elf` is **gitignored** (`executor/.gitignore`) and built on demand. The
fixtures (`*.bin`) are small and committed.

```bash
# One-time: fetch the RV64 sysroot (needed for ethrex's C deps: c-kzg, etc.)
make prepare-sysroot SYSROOT_DIR=$HOME/.lambda-vm-sysroot

# Build just the ethrex guest ELF (or `make compile-programs-rust` for all):
make executor/program_artifacts/rust/ethrex.elf SYSROOT_DIR=$HOME/.lambda-vm-sysroot
```

What the build needs:
- **Toolchains:** `1.94.0` stable (workspace) + `nightly-2026-02-01` with
  `rust-src` (the Makefile pins it; builds the guest via `-Z build-std`).
- **clang + lld** for ethrex's C dependencies.
- **Network**, the first time: cargo fetches `guest_program` from
  `github.com/lambdaclass/ethrex.git` (commit pinned in the guest `Cargo.lock`).
- **`SYSROOT_DIR` must match** between `prepare-sysroot` and the build.

The guest source is `executor/programs/rust/ethrex/` (an 11-line `main.rs` that
reads the private input, calls `execution_program`, and commits the output).
