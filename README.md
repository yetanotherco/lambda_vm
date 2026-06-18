# Lambda VM

A verifiable virtual machine developed in collaboration with [Lambdaclass](https://lambdaclass.com/) and [3MI Labs](https://www.3milabs.tech/).

This open-source zkVM lets users prove the correct execution of a program over a given input stream. The current implementation generates base proofs using STARKs over the Goldilocks field, with 128 bits of security and LogUp as the lookup argument linking tables.

Proof accelerators, GPU support, and proof compression are under development; see the [Roadmap](#roadmap) for the full plan and how the pieces fit together.

> ⚠️ **This project is under active development and experimentation — do not use in production.**

## Roadmap

The **[public roadmap](https://yetanotherco.github.io/lambda_vm_roadmap/)** lays out what's in progress and what's planned, along with the ordering and dependencies between those workstreams.

## Getting Started

### Dependencies

- Rust nightly with `rust-src` component
- Clang with RISC-V target support and LLD linker (used by `make compile-programs-asm`)
  - **macOS**: `brew install llvm` (the Homebrew LLVM includes `clang` and `lld` with RISC-V support)
  - **Linux**: use LLVM 21+ from apt.llvm.org or your distribution; older distro clang packages may reject the assembly fixtures' RISC-V ISA attributes

### Dev dependencies

- Samply (To create flamegraphs) [Installation](https://github.com/mstange/samply?tab=readme-ov-file#installation)

### Setup executor

#### Install Rust

Install Rust using [rustup](https://rustup.rs/):

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Add the `rust-src` component to the pinned nightly toolchain used to build guest programs (required for building `std` for the custom RISC-V target — `make compile-programs-rust` will auto-fetch the toolchain itself):

```sh
rustup component add rust-src --toolchain nightly-2026-02-01
```

#### Compile sysroot

Some of the tests require linking with C libraries.

##### Download pre-installed C libraries

The easiest way is to let `make` do it:

```sh
SYSROOT_DIR=$HOME/.lambda-vm-sysroot make prepare-sysroot  # recommended: user-writable, no sudo
make prepare-sysroot                                       # installs to /opt (uses sudo)
```

Or do it manually:

```sh
wget https://lambda.alignedlayer.com/lambda-vm-sysroot-rv64im.tar.gz
echo "420e394a096f3859235e3a8121a8d5a10f995ac48e636e8d700f17d50803a0e7  lambda-vm-sysroot-rv64im.tar.gz" | sha256sum -c -
sudo mkdir -p /opt && sudo tar -xzf lambda-vm-sysroot-rv64im.tar.gz -C /opt
```

##### Compile them directly
    
```sh                                                
   sudo apt-get install -y autoconf automake autotools-dev curl python3 \                                           
   libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison flex \                                   
   texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev             
```    

```sh                                                                      
  git clone https://github.com/riscv/riscv-gnu-toolchain /tmp/riscv-gnu-toolchain
  cd /tmp/riscv-gnu-toolchain                                                                      
  ./configure --prefix=/opt/riscv64-newlib --with-arch=rv64im --with-abi=lp64 --disable-gdb 
  make -j$(nproc)                                                                               
```     

```sh                                                                                                                  
  mkdir -p /opt/lambda-vm-sysroot                                                     
  cp -r /opt/riscv64-newlib/riscv64-unknown-elf/include /opt/lambda-vm-sysroot/         
  cp -r /opt/riscv64-newlib/riscv64-unknown-elf/lib /opt/lambda-vm-sysroot/                
```

Then, you can check that the executor works by running:

```sh
make test-executor
```

### Using the CLI

The `cli` binary lets you execute, prove, and verify RISC-V ELF programs. Build it once with:

```sh
cargo build --release -p cli
```

The binary will be available at `target/release/cli`.

To get a sample program to work with, compile the bundled assembly tests:

```sh
make compile-programs-asm
```

This emits ELF files under `executor/program_artifacts/asm/`. With those in place, you can run the three core commands:

#### Execute

Run a program without generating a proof. Useful for sanity checks and debugging:

```sh
cargo run -p cli --release -- execute executor/program_artifacts/asm/add.elf
```

#### Prove

Generate a STARK proof of the execution:

```sh
cargo run -p cli --release -- prove executor/program_artifacts/asm/add.elf -o /tmp/proof.bin
```

#### Verify

Verify a proof against the ELF it was generated from. The command exits `0` on success and `1` on failure:

```sh
cargo run -p cli --release -- verify /tmp/proof.bin executor/program_artifacts/asm/add.elf
```

For the full CLI reference — including private inputs, blowup factor tuning, timing, and flamegraph profiling — see [`bin/cli/README.md`](./bin/cli/README.md).

### Writing a guest program

Guest programs are written in Rust (or RISC-V assembly) and cross-compiled to the custom RV64IM target. The guest SDK [`lambda-vm-syscalls`](./syscalls/README.md) provides the syscalls a program uses to read private input, commit public output, halt, and call precompiles like Keccak. The [`executor`](./executor/README.md) crate is what loads your compiled ELF and emits the per-instruction logs the prover consumes.

To add a new Rust guest, drop a project under `executor/programs/rust/<name>/` and run `make compile-programs-rust`. See [`executor/programs/rust/`](./executor/programs/rust/) for examples (`fibonacci`, `keccak`, `hashmap`, …).

## Design choices

- The Instruction Set Architecture is RISCV64IM
- The proof system is transparent (no trusted setup) and post-quantum secure (hash-based)
- The security is over 100 bits of provable security (not conjectured)
- The codebase of the whole project must be simple and minimalistic

## Design principles

Following [ethrex](https://github.com/lambdaclass/ethrex):

- Ensure effortless setup and execution across all target environments.
- Be vertically integrated. Have the minimal amount of dependencies.
- Have a simple type system. Avoid generics leaking over the codebase.
- Have few abstractions. Do not generalize until you absolutely need it. Repeating code two or three times can be fine.
- Prioritize code readability and maintainability over premature optimizations.

## Documentation

High-level documentation lives in [`docs/`](./docs/):

- [Overview of VM flow](./docs/general_flow.md) — the pipeline from source code to proof
- [Proof system overview](./docs/cryptography/proof_system.md) — design goals and primitives
- [Lookup arguments](./docs/cryptography/lookup.md) — how tables are linked via LogUp
- [Recommended reading](./docs/other_resources.md) — papers and tutorials

### Specification

A formal specification of the VM is written in [Typst](https://typst.app/) under [`spec/`](./spec/) and rendered as a browsable wiki (HTML) or PDF using [`shiroa`](https://myriad-dreamin.github.io/shiroa/). With both tools installed, run `shiroa serve` from `spec/` to host the wiki locally.

See [`spec/README.md`](./spec/README.md) for full setup instructions.

## Testing

### Quick Reference

| Command | Description |
|---------|-------------|
| `make test` | Run all tests (compiles programs first) |
| `make test-fast` | Fast tests for prover, stark, and executor (skips slow tests) |
| `make test-prover` | Prover tests only |
| `make test-prover-all` | Prover tests including slow ones |
| `make test-prover-debug` | Prover tests with bus balance report (`debug-checks` feature) |
| `make test-asm` | Compile and run ASM tests |
| `make test-rust` | Compile and run Rust tests |
| `make test-executor` | Compile all programs and run executor tests |
| `make test-math-cuda` | math-cuda parity tests (requires NVIDIA GPU + nvcc) |
| `make build` | Build all workspace crates |
| `make check` | Check all crates (faster than build, no codegen) |
| `make clippy` | Run clippy on all crates |
| `make fmt` | Format all code |
| `make lint` | Run fmt check + clippy |

### Full Test Suite

To run all tests across the project use

`make test`

### ASM Tests

In order to add a new asm test you should add the `.s` file under `executor/programs/asm`
Then add the corresponding test under `executor/tests/asm.rs`

To run them you can use

`make test-asm`

This will compile them and run the tests

### Rust Tests

In order to add a new rust test you should add the cargo project under `executor/programs/rust` as a new directory.
The folder should have the same name as the `Cargo.toml` program name.
Then add the corresponding test under `executor/tests/rust.rs`

You can run it with

`make test-rust`

## Benchmarking & Profiling

You can create a flamegraph for proof generation using the following target:

```sh
make flamegraph-prover
```

This profiles the synthetic `fibonacci_multi_column` STARK example in `crypto/stark` (i.e. the STARK engine itself, not a real guest ELF). To profile the VM prover end-to-end on a real ELF, use the dedicated bench in the `prover` crate:

```sh
samply record cargo bench --bench profile_vm_prover --features parallel
```

For a quick GPU microbench (requires an NVIDIA GPU + `nvcc`):

```sh
make bench-math-cuda
```

## Debug Checks

The `debug-checks` cargo feature enables extra runtime diagnostics for the proof system:

- **Bus balance report**: After proving, prints whether each LogUp bus is balanced (sum of all interactions equals zero).
- **Bus mismatch analysis**: When a bus is imbalanced, identifies which rows/tables have orphan or mismatched interactions.
- **Constraint validation**: Runs `validate_trace` to check that all AIR constraints evaluate to zero on the trace.

To run prover tests with debug output:

```sh
make test-prover-debug
```

Or enable it directly:

```sh
cargo test --release -p lambda-vm-prover --features debug-checks -- --nocapture
```

The feature is defined in `crypto/stark/Cargo.toml` and forwarded through `prover/Cargo.toml`. It has zero overhead when disabled.

## Acknowledgements

This project would not be possible without the contributions made by various teams who developed the core cryptographic primitives and designs and we have learnt and drawn inspiration from them.

- [Starkware](https://starkware.co/)
- [Cairo](https://eprint.iacr.org/2021/1063)
- [Miden](https://github.com/0xMiden/miden-vm)
- [Zisk](https://github.com/0xPolygonHermez/zisk/tree/main)
- [Plonky3](https://github.com/Plonky3/Plonky3)
- [Polygon](https://polygon.technology/)
- [Lean Ethereum](https://leanroadmap.org/)
- [Risc0](https://github.com/risc0/risc0)
- [SP1](https://github.com/succinctlabs/sp1)
- [Valida](https://github.com/valida-xyz/valida)
- [Pico](https://github.com/brevis-network/pico)
- [AirBender](https://github.com/matter-labs/zksync-airbender)
- [Constantine](https://github.com/mratsim/constantine)
- [Jolt](https://github.com/a16z/jolt)
- [Neptune - TritonVM](https://github.com/TritonVM/triton-vm)
- [Winterfell](https://github.com/facebook/winterfell)
- [Stwo](https://github.com/starkware-libs/stwo)
- [Aztec](https://github.com/AztecProtocol)

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
