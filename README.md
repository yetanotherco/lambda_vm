# Lambda VM

Verifiable VM made in collaboration with [Lambdaclass](https://lambdaclass.com/) and [3MI Labs](https://www.3milabs.tech/)

We are developing an open-source verifiable virtual machine that allows users to prove the correctness of the execution of a given program with an input stream.

Right now, this is a project under development and experimentation and must not be used in production!

## Getting Started

### Dependencies

- Our Rust fork with support for our riscv target
- Risc-V toolchain (To run executor tests)

### Setup executor

#### Install Our Rust Fork

First remove rust if you already have it installed

```sh
rustup self uninstall
```

```sh
git clone https://github.com/yetanotherco/rust.git
cd rust
```
Add `bootstrap.toml` file:

```toml
profile = "dist"
change-id = 149355
rust.lld = true
```

Export the directory where you want rust to be installed

```sh
export DESTDIR=<Your_rust_destiny_dir>
```

Run the rust installation
```sh
./x.py build && ./x.py install
```

Add the rust directory to your path

```sh
export PATH="/<your_rust_path>/usr/local/bin:$PATH"
source ~/.zshrc
```

#### Install the dependencies

```sh
cd executor
make deps
```

**Note:** At the moment, `make deps` only works on macOS.

Then, you can check that the executor works by running:

```sh
make test
```

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

Full documentation can be found in [docs](./docs/). It is currently a work in progress, we expect that as more features and components become ready, they will be included in the docs.

## Testing

### ASM Tests

In order to add a new asm test you should add the `.s` file under `programs/asm`
Then add the corresponding test under `tests/asm.rs`

To run them you can use

`make test`

This will compile them and run the tests

### Rust Tests

In order to add a new rust test you should add the cargo project under `programs/rust` as a new directory.
The folder should have the same name as the `Cargo.toml` program name.
Then add the corresponding test under `tests/rust.rs`

You can run it with

`make test`

## Roadmap for the virtual machine

This project is under active development. Our primary objective is to have a first working version for the virtual machine. The first roadmap for the project can be found [here](./docs/roadmap.md). Priorities and features might change as we continue developing.

#### Milestones

- v0: VM that can prove fibonacci with lookups
- v1: VM that can prove any program, with public inputs, and has at least one co-processor
- v2: VM with all co-processors, recursion and solidity verifiers
- v3: GPU + Engineering upgrades

Notice this roadmap doesn't fully take in account the migration to 64 bits, leaving aside the executor

Total weeks: 80

### Executor

#### Engineering
- Mauro
- Gianluca

**Total estimated duration:** 16 weeks

| Feature                     | Description                        | Status       | Duration | Version |
|---------------------------- |------------------------------------|--------------| ---------| ------- |
| Documentation               | Explain how the executor works     | In progress  |  1 weeks | all versions |
| 32-bits CPU                 | CPU with all operations | Done | - | v0 |
| Public Inputs / Private Inputs | Support for public and private inputs in the VM | planned | 2 weeks | v1 |
| STD Support                 | Implement all STD operations, compile get_rand | Planned | 3 weeks | v1 |
| System instructions | `ecall`, `ebreak` | Planned | 1 week | v1 |
| CPU with coprocessors       | Add coprocessors for special cryptographic operations   | Planned | 1 week | v2 |
| Big Integer arithmetic | Big integer arithmetic syscall | Planned | 3 days | v2 |
| Elliptic curve addition | EC operations syscall | Planned | 3 days  | v2 |
| Poseidon hash | Poseidon hash syscall | Planned | 3 days | v2 |
| Keccak hash | Keccak hash syscall | Planned | 3 days | v2 |
| SHA256 | SHA 256 syscall | Planned | 3 days | v2 |
| Pairing | Table for pairings | Planned | 3 days | v2 |
| Recursion Experiments | Try naive recursion, and explore how it behaves | Planned | 1 week | v2 |
| Perf tools for guest programs | Flamegraphs, cycle counts, and tools to optimize guests | Planned | 2 weeks | v2 |
| RISCV64IM CPU               | Migration to 64 bits | Planned | 1 week | ??? |

### Trace and Constraints generator

#### Engineering
- Mauro
- Federica
- Colo

**Total estimated duration:** 37 weeks

*This includes the linking with the executor*

| Feature                     | Description                       | Status       | Duration | Version |
|---------------------------- |-----------------------------------|--------------| -------- | ------- |
| Documentation               | Document trace generation and constraints | In progress | 2 weeks | all versions |
| CPU Table with basic constraints  | Implement CPU table with constraints | In progress | 4 weeks | v0 |
| Decoder table | Implement decoder table | Planned | 1 week | v0 |
| Link decoder table and CPU | Use lookup to connect tables | Planned | 1 week | v0 |
| Constraints for updating pc | Implement constraints for updating pc | In progress | 1 week | v0 |
| ALU - Range checkers | Implement rangecheck for u16 and u8 | Planned | 2 week | v0 |
| Memory                      | Implement memory table with constraints | Planned | 2 weeks | v0 |
| ALU - Bitwise operations (and, xor, or) | Implement tables for u8 bitwise operations | Planned | 2 week | v1 |
| ALU - Shift operations | Implement tables for shift operations | Planned | 2 week | v1 |
| ALU - Multiplication table | Implement table for multiplication table | Planned | 2 week | v1 |
| ALU - Division and remainder | Implement table for integer division operations | Planned | 2 week | v1 |
| Syscall - Initial - Big Integer arithmetic | Table for big integer arithmetic | Planned | 3 week |v2|
| Syscall - Elliptic curve addition | Table for EC operations | Planned | 3 weeks  | v2|
| Syscall - Poseidon hash | Table for Poseidon hash | Planned | 3 weeks | v2 |
| Syscall -Keccak hash | Table for Keccak hash | Planned | 3 weeks | v2 |
| Syscall - SHA256 | Table for SHA256 | Planned | 1 weeks | v2 |
| Syscall - Pairing / FP | Table for pairings | Planned | 3 weeks | v2 |

### Core Proof system

#### Theory

- Diego
- Manuel
- 3MILabs (Cyprien, Robin, and Erik)

#### Implementation
- Mauro
- Juan
- Nicole

**Total estimated duration:** 18 weeks

| Feature                     | Description                       | Status       | Duration | Version |
|---------------------------- |-----------------------------------|--------------| -------- | ------- |
| Documentation               | Prepare comprehensive documentation on proof system | In progress   | 4 weeks | all versions |
| Lookup arguments            | Linking tables via lookup arguments | In progress | 2 weeks | v0 |
| Lookup - I | Accept multitables | In progress | 1 week | v0 |
| Lookup - II | Perform argument with constraints | In progress | 1 week | v0 |
| Public input | Add public input using Lookup | Planned | 1 week | v0 |
| Poseidon hash               | Adapt Poseidon for Goldilocks      | Planned      | 3 days   | v1 |
| Multi-table Merkle trees (MTMT)   | Merkle tree that can be used to commit to polynomials of various sizes | In progress | 2 weeks | v2 |
| Multi-FRI                   | Perform FRI using MTMT | Planned | 2 weeks | v2 |
| Adjust parameters           | Adjust parameters for 128 bits of security | Planned | 1 week | v2 |
| Recursion | Allow for n-1 recursion tree to compress proof size | Planned | 4 weeks | v2 |
| More efficient lookups | Experiment with lookup arguments | Planned | 4 weeks | v3 |

### Verifier

**Total estimated duration:** 7 weeks

| Feature | Description | Status | Duration | Version |
| ------ | -------- |--------| -----------| ------- |
| CPU Table Ethereum verifier | Solidity verifier for the vm | Planned | 2 weeks | v1 |
| Browser Verifier | Verifier for the vm using wasm in javascript | Planned | 1 weeks | v2 |
|Optimize Ethereum verifier | Optimize gas cost for verifier | Planned | 2 weeks | v2 |
| Multi Table Ethereum verifier | Solidity verifier for the vm | Planned | 2 weeks | v2 |

### GPU and performance

**Total estimated duration:** 24 weeks

| Feature                     | Description                       | Status       | Version |
|---------------------------- |-----------------------------------|--------------| ------ |
| Fields                      | Improve field performance using assembly | Planned | v3 |
| GPU-Fast-Fourier transform      | Implement GPU version of FFT | Planned | v3 |
| GPU-Merkle tree                 | Implement GPU version for Merkle trees | Planned | v3 |
| Parallel witness generation   | Use GPU for fast witness generation | Planned | v3 |
| GPU-FRI | Perform FRI on GPU | Planned | v3 |
  
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
