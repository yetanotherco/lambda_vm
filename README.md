# Lambda VM

Verifiable VM made in collaboration with [Lambdaclass](https://lambdaclass.com/) and [3MI Labs](https://www.3milabs.tech/)

We are developing an open-source verifiable virtual machine that allows users to prove the correctness of the execution of a given program with an input stream.

Right now, this is a project under development and experimentation and must not be used in production!

## Getting Started

### Dependencies

- Rust nightly with `rust-src` component

### Dev dependencies

- Samply (To create flamegraphs) [Installation](https://github.com/mstange/samply?tab=readme-ov-file#installation)

### Setup executor

#### Install Rust

Install Rust using [rustup](https://rustup.rs/):

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then install the nightly toolchain with the `rust-src` component (required for building `std` for the custom RISC-V target):

```sh
rustup toolchain install nightly
rustup component add rust-src --toolchain nightly
```

#### Compile sysroot

Some of the tests require linking with C libraries.

##### Download pre-installed C libraries

```sh
wget https://lambda.alignedlayer.com/lambda-vm-sysroot-rv64im.tar.gz
sudo mkdir -p /opt && sudo tar -xzf lambda-vm-sysroot-rv64im.tar.gz -C /opt
```

##### Compile it directly             
    
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

#### Install the dependencies

```sh
make deps
```

**Note:** At the moment, `make deps` only works on macOS.

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
- [Lookup arguments](./docs/cryptography/lookup.md) — how tables are linked
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

```
  make flamegraph-prover
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

## Roadmap for the virtual machine

This project is under active development. Our primary objective is to have a first working version for the virtual machine. Priorities and features might change as we continue developing.

### Version Milestones

| Version | Description | Key Deliverables |
|---------|-------------|------------------|
| **v0** | Prove fibonacci with lookups | CPU table, decoder, basic constraints |
| **v1** | Prove any program + CLI + SDK | Full instruction set, CLI tools, native verifier, benchmarks, Phase 1 audit |
| **v1.5** | Production hardening | Segmented execution, trace compression, debugging tools |
| **v2** | Coprocessors + recursion + Solidity verifier | All syscall tables, recursion, on-chain verification, Phase 2 audit |
| **v3** | GPU acceleration | CUDA/Metal support, parallel proving, full audit |

### CLI

| Feature | Description | Status | Duration | Version |
|---------|-------------|--------|----------|---------|
| `run` command | Execute ELF files with verbose mode, step limits, memory limits | In progress | 1 week | v0 |
| `prove` command | Generate proof from execution trace | Planned | 2 weeks | v1 |
| `verify` command | Verify a proof locally | Planned | 1 week | v1 |
| `build` command | Compile guest programs to ELF | Planned | 2 weeks | v1 |
| `new` command | Scaffold new guest project | Planned | 1 week | v1 |
| `inspect` command | Inspect proof contents, trace stats | Planned | 1 week | v2 |

### SDK / Guest Library

| Feature | Description | Status | Duration | Version |
|---------|-------------|--------|----------|---------|
| Basic syscalls | Print, Panic, Commit, GetPrivateInputs, Halt | Done | - | v0 |
| Structured I/O | Typed input/output with serde support | Planned | 2 weeks | v1 |
| Logging macros | `info!`, `debug!` macros for guest programs | Planned | 3 days | v1 |
| Assertions | `assert_eq!` that reports to host | Planned | 3 days | v1 |
| Environment variables | Read env vars from host | Planned | 3 days | v1 |
| Guest library docs | Comprehensive SDK documentation | Planned | 1 week | v1 |
| Hint/Advice mechanism | Non-deterministic hints for optimization | Planned | 1 week | v2 |

### Executor

| Feature | Description | Status | Duration | Version |
|---------|-------------|--------|----------|---------|
| Documentation | Explain how the executor works | In progress | 1 week | all |
| 32-bit CPU | CPU with all base operations | Done | - | v0 |
| Cycle counter | Track instruction cycles for profiling | Planned | 3 days | v1 |
| Memory statistics | Track memory usage, peak allocation | Planned | 3 days | v1 |
| Execution limits | Max cycles, max memory configuration | Planned | 3 days | v1 |
| Public/Private Inputs | Support for public and private inputs in the VM | Planned | 2 weeks | v1 |
| STD Support | Implement all STD operations | Planned | 3 weeks | v1 |
| System Instructions | `ecall`, `ebreak` | Planned | 1 week | v1 |
| 64-bit lookup variants | Lookup arguments for 64-bit ops | Done | - | v1 |
| 64-bit executor | RV64IM instruction execution | In progress | 1 week | v1 |
| 64-bit memory model | 64-bit address space | In progress | 1 week | v1 |
| Segmented execution | Split execution into chunks for large programs | Planned | 2 weeks | v1.5 |
| Checkpoint/Resume | Save and restore execution state | Planned | 2 weeks | v1.5 |
| CPU with Coprocessors | Add coprocessors for cryptographic operations | Planned | 1 week | v2 |
| Big Integer Arithmetic | Big integer arithmetic syscall | Planned | 3 days | v2 |
| Elliptic Curve Addition | EC operations syscall | Planned | 3 days | v2 |
| Poseidon Hash syscall | Poseidon hash syscall | Planned | 3 days | v2 |
| Keccak Hash syscall | Keccak hash syscall | Planned | 3 days | v2 |
| SHA256 syscall | SHA256 syscall | Planned | 3 days | v2 |
| Pairing syscall | Table for pairings | Planned | 3 days | v2 |

### Trace and Constraints Generator

| Feature | Description | Status | Duration | Version |
|---------|-------------|--------|----------|---------|
| Documentation | Document trace generation and constraints | In progress | 2 weeks | all |
| CPU Table with Basic Constraints | Implement CPU table with constraints | In progress | 4 weeks | v0 |
| Decoder Table | Implement decoder table | Done | - | v0 |
| Link Decoder and CPU Tables | Use lookup to connect tables | In progress | 1 week | v0 |
| PC Update Constraints | Implement constraints for updating pc | In progress | 1 week | v0 |
| Trace serialization | Serialize/deserialize execution traces | Planned | 1 week | v1 |
| Constraint debugging | Tools to identify failing constraints | Planned | 1 week | v1 |
| Memory | Implement memory table with constraints | In progress | 2 weeks | v1 |
| ALU  | All ALU operations | Planned | 2 weeks | v1 |
| MEMW chip | Memory word read/write operations | Planned | 2 weeks | v1 |
| 64-bit ALU constraints | Constraints for 64-bit arithmetic | In progress | 2 weeks | v1 |
| Trace compression | Compress trace for storage efficiency | Planned | 1 week | v1.5 |
| Syscall - Big Integer | Table for big integer arithmetic | Planned | 3 weeks | v2 |
| Syscall - Elliptic Curve | Table for EC operations | Planned | 3 weeks | v2 |
| Syscall - Poseidon Hash | Table for Poseidon hash | Planned | 3 weeks | v2 |
| Syscall - Keccak Hash | Table for Keccak hash | Planned | 3 weeks | v2 |
| Syscall - SHA256 | Table for SHA256 | Planned | 1 week | v2 |
| Syscall - Pairing | Table for pairings | Planned | 3 weeks | v2 |

### Table Infrastructure

| Feature | Description | Status | Duration | Version |
|---------|-------------|--------|----------|---------|
| Preprocessed columns | Generate constant/precomputed columns | Planned | 1 week | v0 |
| Padding utilities | Consistent power-of-two padding across tables | Planned | 1 week | v0 |
| Column count optimization | Track and minimize total column counts | Planned | 2 weeks | v1 |
| Range check tables | IS_B8, IS_B16, IS_B20 lookup tables | Planned | 2 weeks | v1 |
| Bitwise lookup tables | AND, OR, XOR, MSB precomputed tables | Planned | 2 weeks | v1 |
| Shift lookup tables | HWSL, HWSLC for half-word shifts | Planned | 2 weeks | v1 |

### Memory Argument

| Feature | Description | Status | Duration | Version |
|---------|-------------|--------|----------|---------|
| PAGE tables | Paged memory initialization/finalization | Planned | 2 weeks | v1 |
| Timestamp ordering | Unique timestamps for memory access ordering | Planned | 1 week | v1 |
| Token balancing | LogUp emission/consumption balance | Planned | 1 week | v1 |
| ELF binary verification | Verifier checks initial memory from ELF | Planned | 1 week | v1 |
| Register init state | Initial register values (x0-x31) | Planned | 3 days | v1 |
| Private input pages | PAGE tables for prover private inputs | Planned | 1 week | v1 |

### Proof System

| Feature | Description | Status | Duration | Version |
|---------|-------------|--------|----------|---------|
| Documentation | Prepare comprehensive documentation on proof system | In progress | 4 weeks | all |
| Lookup arguments | Linking tables via lookup arguments | Done | - | v0 |
| Lookup - Multi-table | Accept multitables | Done | - | v0 |
| Lookup - Constraints | Perform argument with constraints | Done | - | v0 |
| Public Input | Add public input using Lookup | Done | - | v0 |
| Multi-table Merkle Trees (MTMT) | Merkle tree for polynomials of various sizes | In progress | 2 weeks | v2 |
| Multi-FRI | Perform FRI using MTMT | Planned | 2 weeks | v2 |
| Adjust parameters | Adjust parameters for 128 bits of security | Planned | 1 week | v2 |
| Single-layer Recursion | Verify one proof inside another | Planned | 4 weeks | v2 |
| Proof Aggregation | Combine N proofs into one | Planned | 2 weeks | v2 |
| Recursive Verifier Circuit | AIR constraints for in-circuit verification | Planned | 3 weeks | v2 |
| More Efficient Lookups | Experiment with lookup arguments | Planned | 4 weeks | v3 |

### Verifier

| Feature | Description | Status | Duration | Version |
|---------|-------------|--------|----------|---------|
| Native Rust verifier | Verify proofs locally | Done | - | v1 |
| Proof serialization | Standard proof format (bincode/JSON) | Planned | 1 week | v1 |
| CPU Table Ethereum Verifier | Solidity verifier for the vm | Planned | 2 weeks | v2 |
| Batch verification | Verify multiple proofs efficiently | In progress | 2 weeks | v2 |
| Browser Verifier | Verifier using wasm in javascript | Planned | 1 week | v2 |
| Optimize Ethereum Verifier | Optimize gas cost for verifier | Planned | 2 weeks | v2 |
| Multi Table Ethereum Verifier | Solidity verifier for multi-table vm | Planned | 2 weeks | v2 |

### Testing & Quality

| Feature | Description | Status | Duration | Version |
|---------|-------------|--------|----------|---------|
| ASM test suite | Assembly test programs | Done | - | v0 |
| Rust test programs | Rust guest program tests | Done | - | v0 |
| Benchmarking suite | Systematic performance benchmarks | In progress | 2 weeks | v1 |
| Fuzzing infrastructure | Continuous fuzzing for executor/prover | Planned | 2 weeks | v1 |
| RISC-V compliance tests | Official RISC-V test suite | Planned | 1 week | v1 |
| Property-based tests | Proptest for constraint correctness | Planned | 1 week | v1 |
| Integration tests | Full pipeline tests (compile→prove→verify) | Planned | 2 weeks | v1 |

### Security & Audits

| Feature | Description | Status | Version |
|---------|-------------|--------|---------|
| Constraint audit (Phase 1) | Audit core CPU/memory constraints | Planned | v1 |
| Crypto audit | Audit field arithmetic, hash functions | Planned | v1 |
| Prover audit (Phase 2) | Audit STARK prover, FRI, lookups | Planned | v2 |
| Full system audit | End-to-end security audit | Planned | v3 |

### Integration & Examples

| Feature | Description | Status | Duration | Version |
|---------|-------------|--------|----------|---------|
| End-to-end examples | Complete prove/verify workflows | Planned | 1 week | v1 |
| Project templates | Starter templates for common use cases | Planned | 1 week | v1 |

### GPU and Performance

| Feature | Description | Status | Duration | Version |
|---------|-------------|--------|----------|---------|
| Fields (ASM/SIMD) | Improve field performance using assembly/SIMD | In progress | 4 weeks | v3 |
| CUDA NTT/FFT | Number theoretic transform on CUDA | Planned | 4 weeks | v3 |
| Metal NTT/FFT | NTT for Apple Silicon | Planned | 3 weeks | v3 |
| GPU Merkle hashing | Parallel Merkle tree on GPU | Planned | 3 weeks | v3 |
| GPU witness generation | Parallel trace generation | Planned | 4 weeks | v3 |
| GPU FRI | FRI folding on GPU | Planned | 4 weeks | v3 |
| Multi-GPU support | Distribute across multiple GPUs | Planned | 3 weeks | v3 |
  
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
