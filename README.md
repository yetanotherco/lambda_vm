# Lambda VM

Verifiable VM made in collaboration with [Lambdaclass](https://lambdaclass.com/) and [3MI Labs](https://www.3milabs.tech/)

We are developing an open-source verifiable virtual machine that allows users to prove the correctness of the execution of a given program with an input stream.

Right now, this is a project under development and experimentation and must not be used in production!

## Getting Started

### Dependencies

- Rust 1.90.0
- Risc-V toolchain (To run executor tests)

### Setup executor

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

# Roadmap for the virtual machine

This project is under active development. Our primary objective is to have a first working version for the virtual machine. The first roadmap for the project can be found [here](./docs/roadmap.md). Priorities and features might change as we continue developing.

## Teams

### Theory
- Diego
- Manuel
- 3MILabs (Cyprien, Robin y Erik)

### Applied cryptography
- Colo
- Juan
- Nicole

### Engineering
- Mauro
- Federica
- Gianluca

### Milestones

- v0: Minimal CPU: able to prove simple computations, but not all operations supported
- v1: RV64IM vm: prove general RV64IM code 
- v2: Full vm: supports coprocessors for expensive operations
- v3: GPU vm: leverages GPU for fast proving

## Basic building blocks

The first version is going to use the primitives contained in [lambdaworks](https://github.com/lambdaclass/lambdaworks)

**Total estimated duration:** 5 weeks

| Feature                     | Description                        | Status       | Duration | Version |
|---------------------------- |------------------------------------|--------------| ---------| ------- |
| Documentation               | Explain how everything works       | In progress  | 4 weeks | all versions |
| Field                       | Basic field type                   | ✔️  | 1 week | v0 |
| Poseidon hash               | Implement Poseidon-2 hash          | Planned      | 1 week | v1 |
| Keccak                      | Implement Keccak hash function     | Planned      | 1 week | v1 |
| CPU FFT                     | Radix-2 Fast-Fourier transform in CPU | ✔️ | 1 week | v0 |
| Basic Merkle commitment     | Merkle tree                        | ✔️  | 1 week | v0 |
| Basic FRI                   | FRI proximity test                 | ✔️  | 1 week  | v0 |
| Basic constraints           | Simple API for defining constraints for AIR | ✔️  | 1 week | v0 |
| Basic AIR                   | Algebraic intermediate representation for computations | ✔️  | 1 week | v0 |      

## Executor

**Total estimated duration:** 24 weeks

| Feature                     | Description                        | Status       | Duration | Version |
|---------------------------- |------------------------------------|--------------| ---------| ------- |
| Documentation               | Explain how the executor works     | In progress  |  4 weeks | all versions |
| Minimal CPU                 | Minimal CPU that can perform basic operations | In progress | 6 weeks | v0 |
| Fibonacci operations I         | Operations needed to run Fibonacci part I (`addi`, `sw`, `beq`, `jal`, `jalr`) | ✔️  | 1 week | v0 |
| Fibonacci operations II         | Operations needed to run Fibonacci part II (`auipc`, `bltu`, `lui`, `sb`, `slli`) | ✔️  | 1 week | v0 |
| Compute decoding table | Decoding table indexed by pc | Planned | 1 week | v0 |
| Basic logging        | Basic logs for minimal opcodes | ✔️  | 2 weeks | v0 |
| RISCV64 CPU                 | Minimal version of the CPU with 52 RISCV instructions* | Planned | 8 weeks | v1 |
| Control flow opcodes | Implement remaining control flow operations (`bne`, `blt`, `bge`, `bgeu`) | ✔️ | 1 week | v1 |
| Store operations | Remaining store operations (`sh`, `sw`, `sd`) | ✔️  | 1 week | v1 |
| Load operations | Remaining load operations (`lb`, `lh`, `lw`, `ld`, `lbu`, `lhu`, `lwu`) | ✔️  | 1 week | v1 |
| Integer arithmetic | Operations (`add`, `sub`, `sll`, `slt`, `sltu`, `xor`, `srl`, `sra`, `or`, `and`) | ✔️  | 1 week | v1 |
| Integer arithmetic - immediate | Operations (`addi`, `subi`, `slli`, `slti`, `sltui`, `xori`, `srli`, `srai`, `ori`, `andi`) | ✔️  | 1 week | v1 |
| 32-bit word operations | RV64 32-bit operations (`addw`, `subw`, `sllw`, `srlw`, `sraw`, `addiw`, `slliw`, `srliw`, `sraiw`) | Planned | 1 week | v1 |
| Input/Output | Pass input and output | Planned | 1 week | v1 |
| RISCV64IM CPU               | Working executor for RV64 virtual machine with 65 RISCV instructions | Planned | | v1 |
| Multiplication operations | Operations related to multiplication (`mul`, `mulh`, `mulhsu`, `mulhu`) | ✔️ | 1 week | v1 |
| Division operations | Operations related to division (`div`, `divu`, `rem`, `remu`) | ✔️ | 1 week | v1 |
| CPU with coprocessors       | Add coprocessors for special cryptographic operations   | Planned | 10 weeks | v2 |
| System instructions | `ecall`, `ebreak` | Planned | 1 week | v2 |
| Big Integer arithmetic | Big integer arithmetic syscall | Planned | 1 week | v2 |
| Elliptic curve addition | EC operations syscall | Planned | 2 weeks  | v2 |
| Poseidon hash | Poseidon hash syscall | Planned | 3 weeks | v2 |
| Keccak hash | Keccak hash syscall | Planned | 3 weeks | v2 |
| SHA256 | SHA 256 syscall | Planned 2 weeks | v2 |
| Pairing | Table for pairings | Planned? | ? | v2 |

*few operations remain to be implemented

## Trace generator

**Total estimated duration:** 24 weeks

| Feature                     | Description                       | Status       | Duration | Version |
|---------------------------- |-----------------------------------|--------------| -------- | ------- |
| Documentation               | Document trace generation and constraints | In progress | 8 weeks | all versions |
| CPU                         | Implement CPU table with constraints | In progress | 5 weeks | v0 |
| Define basic CPU constraints | Add basic type constraints for CPU | In progress | 1 week| v0 |
| Decoder table | Implement decoder table | Planned | 1 week | v0 |
| Link decoder table and CPU | Use lookup to connect tables | Planned | 1 week | v0 |
| Constraints for updating pc | Implement constraints for updating pc | In progress | 1 week | v0 |
| ALU                         | Implement ALU tables with constraints| Not started | 6 weeks | v1 |
| Range checkers | Implement rangecheck for u16 and u8 | Planned | 1 week | v0 |
| Bitwise operations (and, xor, or) | Implement tables for u8 bitwise operations | Planned | 1 week | v1 |
| Shift operations | Implement tables for shift operations | Planned | 1 week | v1 |
| Multiplication table | Implement table for multiplication table | Planned | 1 week | v1 |
| Division and remainder | Implement table for integer division operations | 1 week | v1 |
| Memory                      | Implement memory table with constraints | Planned | 2 weeks | v1 |
| Syscalls                    | Tables for coprocessors | Planned | | v2|
| Big Integer arithmetic | Table for big integer arithmetic | Planned | 1 week |v2|
| Elliptic curve addition | Table for EC operations | Planned | 2 weeks  | v2|
| Poseidon hash | Table for Poseidon hash | Planned | 3 weeks | v2 |
| Keccak hash | Table for Keccak hash | Planned | 3 weeks | v2 |
| SHA256 | Table for SHA256 | Planned | 3 weeks | v2 |
| Pairing | Table for pairings | Planned? | | v2 |

## Proof system

**Total estimated duration:** 18 weeks

| Feature                     | Description                       | Status       | Duration | Version |
|---------------------------- |-----------------------------------|--------------| -------- | ------- |
| Documentation               | Prepare comprehensive documentation on proof system | In progress   | 4 weeks | all versions |
| Lookup arguments            | Linking tables via lookup arguments | In progress | 2 weeks | v0 |
| Lookup - I | Accept multitables | In progress | 1 week | v0 |
| Lookup - II | Perform argument with constraints | In progress | 1 week | v0 |
| Public input | Add public input using Lookup | 1 week | v1 |
| Multi-table Merkle trees (MTMT)   | Merkle tree that can be used to commit to polynomials of various sizes | In progress | 2 weeks | v1 |
| Multi-FRI                   | Perform FRI using MTMT | Planned | 2 weeks | v1 |
| Adjust parameters           | Adjust parameters for 128 bits of security | Planned | 1 week | v1 |
| Recursion | Allow for n-1 recursion tree to compress proof size | Planned | 4 weeks | v2 |
| More efficient lookups | Implement better lookup arguments | 4 weeks | v2 |

## Verifier

**Total estimated duration:** 6 weeks

| Feature | Description | Status | Duration | Version |
| ------ | -------- |--------| -----------| ------- |
|Ethereum verifier | Solidity verifier for the vm | Planned | 2 weeks | v2 |
|Verifier | Verifier for the vm | Planned | 2 weeks | v2 |
|Optimize Ethereum verifier | Optimize gas cost for verifier | Planned | 2 weeks | v2 |

## GPU and performance

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
