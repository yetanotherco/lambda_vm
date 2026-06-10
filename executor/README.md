# Lambda VM Executor

RISC-V (RV64IM) emulator for the Lambda VM. Loads ELF binaries, runs them against an in-memory VM state, and emits the per-instruction execution logs that the [prover](../prover) turns into a STARK trace.

Published as `executor`. Used directly by the CLI and the prover; you can also drive it from Rust.

## Usage

```rust
use executor::elf::Elf;
use executor::vm::execution::Executor;

let elf_bytes = std::fs::read("program.elf")?;
let program = Elf::load(&elf_bytes)?;
let executor = Executor::new(&program, /* private input */ vec![])?;
let result = executor.run()?;

println!("Executed {} instructions", result.logs.len());
```

For chunked execution (useful when you don't want to hold all logs in memory), drive the executor via `executor.resume()` in a loop until it yields `None`, then call `executor.finish()`. See [`bin/cli/src/main.rs`](../bin/cli/src/main.rs) for an example.

## Example programs

The repo ships ready-to-use guest programs in three flavours, all compiled by Makefile targets at the repo root:

- [`programs/asm/`](./programs/asm/) — raw RISC-V assembly. Built with `make compile-programs-asm` into `program_artifacts/asm/`.
- [`programs/rust/`](./programs/rust/) — Rust guest projects (`fibonacci`, `keccak`, `hashmap`, …). Built with `make compile-programs-rust` into `program_artifacts/rust/`. Requires the pinned nightly toolchain and sysroot — see the root [`README.md`](../README.md).
- [`programs/bench/`](./programs/bench/) — benchmark programs. Built with `make compile-bench`.

The custom RISC-V target spec used for Rust guests lives at [`programs/riscv64im-lambda-vm-elf.json`](./programs/riscv64im-lambda-vm-elf.json).

## Tests

```sh
# Compile all programs and run executor tests
make test-executor

# Just the asm tests
make test-asm

# Just the Rust tests
make test-rust
```

To add a new test:

- **ASM**: add a `.s` file under [`programs/asm/`](./programs/asm/) and a matching entry in [`tests/asm.rs`](./tests/asm.rs).
- **Rust**: add a cargo project under [`programs/rust/<name>/`](./programs/rust/) (the directory and the `Cargo.toml` package name must match) and a matching entry in [`tests/rust.rs`](./tests/rust.rs).

## Flamegraphs

The executor includes a flamegraph generator (`executor::flamegraph::FlamegraphGenerator`) that produces folded-stack output by instruction count. Drive it via the CLI: `cli execute <elf> --flamegraph stacks.txt`. See [`bin/cli/README.md`](../bin/cli/README.md) for details.
