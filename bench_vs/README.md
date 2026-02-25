# Lambda VM vs SP1 v6 Benchmark

Compares proving time for an identical u64 wrapping Fibonacci computation.

## Prerequisites

1. **Lambda VM CLI** (built from this repo):
   ```bash
   cargo build --release -p cli
   ```

2. **SP1 toolchain** (Succinct's prover):
   ```bash
   curl -L https://sp1up.succinct.xyz | bash
   sp1up
   ```

3. **RISC-V assembler** — Homebrew clang + ld.lld (macOS):
   ```bash
   brew install llvm
   ```

## Usage

```bash
# Default series: 1k, 10k, 100k, 300k iterations
./bench_vs/run.sh

# Custom series
./bench_vs/run.sh -n 1000 50000

# Run only one prover
./bench_vs/run.sh --lambda-only
./bench_vs/run.sh --sp1-only
```

## What it measures

Both provers execute the same program: iterative Fibonacci with `u64::wrapping_add`.
Only **proving time** is compared (wall-clock, no recursion/compression on either side).

- **Lambda VM**: Generates RISC-V assembly at runtime, assembles to ELF, proves via the CLI.
- **SP1 v6**: Compiles a Rust guest program to RISC-V, proves via `sp1-sdk` core mode.

## Output

```
=== Summary ===
Program: Fibonacci (u64 wrapping)

  n           Lambda VM       SP1 v6     Ratio
  ---         ---------       ------     -----
  1000          13.3s         12.4s      0.9x
  10000         22.4s         12.9s      0.6x
  100000       116.4s         14.7s      0.1x
  300000          ...           ...       ...

Green ratio = Lambda VM faster, Red = SP1 faster
```
