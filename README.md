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

## Basic building blocks

The first version is going to use the primitives contained in [lambdaworks](https://github.com/lambdaclass/lambdaworks)

| Feature                     | Description                        | Status       | Duration |
|---------------------------- |------------------------------------|--------------| ---------|
| Documentation               | Explain how everything works       | In progress  | 4 weeks |
| Field                       | Basic field type                   | ✔️  | 1 week |
| Poseidon hash               | Implement Poseidon-2 hash          | Planned      | 1 week |
| Keccak                      | Implement Keccak hash function     | Planned      | 1 week |
| CPU FFT                     | Radix-2 Fast-Fourier transform in CPU | ✔️ | 1 week |
| Basic Merkle commitment     | Merkle tree                        | ✔️  | 1 week |
| Basic FRI                   | FRI proximity test                 | ✔️  | 1 week  |
| Basic constraints           | Simple API for defining constraints for AIR | ✔️  | 1 week |
| Basic AIR                   | Algebraic intermediate representation for computations | ✔️  | 1 week |       

## Executor

| Feature                     | Description                        | Status       | Duration |
|---------------------------- |------------------------------------|--------------| ---------|
| Documentation               | Explain how the executor works     | In progress  |  4 weeks |
| Minimal CPU                 | Minimal CPU that can perform basic operations | In progress | 6 weeks |
| Fibonacci operations I         | Operations needed to run Fibonacci part I (`addi`, `sw`, `beq`, `jal`, `jalr`) | ✔️  | 1 week |
| Fibonacci operations II         | Operations needed to run Fibonacci part II (`auipc`, `bltu`, `lui`, `sb`, `slli`) | ✔️  | 1 week |
| Compute decoding table | Decoding table indexed by pc | 1 week |
| Basic logging        | Basic logs for minimal opcodes | ✔️  | 2 weeks |
| RISCV64 CPU                 | Minimal version of the CPU with 52 RISCV instructions | Planned | 8 weeks |
| Control flow opcodes | Implement remaining control flow operations (`bne`, `blt`, `bge`, `bgeu`) | ✔️ | 1 week |
| Store operations | Remaining store operations (`sh`, `sw`, `sd`) | ✔️  | 1 week |
| Load operations | Remaining load operations (`lb`, `lh`, `lw`, `ld`, `lbu`, `lhu`, `lwu`) | ✔️  | 1 week |
| Integer arithmetic | Operations (`add`, `sub`, `sll`, `slt`, `sltu`, `xor`, `srl`, `sra`, `or`, `and`) | ✔️  | 1 week |
| Integer arithmetic - immediate | Operations (`addi`, `subi`, `slli`, `slti`, `sltui`, `xori`, `srli`, `srai`, `ori`, `andi`) | ✔️  | 1 week |
| 32-bit word operations | RV64 32-bit operations (`addw`, `subw`, `sllw`, `srlw`, `sraw`, `addiw`, `slliw`, `srliw`, `sraiw`) | Planned | 1 week |
| RISCV64IM CPU               | Working executor for RV64 virtual machine with 65 RISCV instructions | Planned | |
| Multiplication operations | Operations related to multiplication (`mul`, `mulh`, `mulhsu`, `mulhu`) | Planned | 1 week |
| Division operations | Operations related to division (`div`, `divu`, `rem`, `remu`) | Planned | 1 week |
| CPU with coprocessors       | Add coprocessors for special cryptographic operations   | Planned | 10 weeks |
| System instructions | `ecall`, `ebreak` | Planned | 1 week |
| Big Integer arithmetic | Big integer arithmetic syscall | Planned | 1 week |
| Elliptic curve addition | EC operations syscall | Planned | 2 weeks  |
| Poseidon hash | Poseidon hash syscall | Planned | 3 weeks |
| Keccak hash | Keccak hash syscall | Planned | 3 weeks |
| SHA256 | SHA 256 syscall | Planned 2 weeks |
| Pairing | Table for pairings | Planned? | ? |

## Trace generator

| Feature                     | Description                       | Status       | Duration |
|---------------------------- |-----------------------------------|--------------| -------- |
| Documentation               | Document trace generation and constraints | In progress | 8 weeks |
| CPU                         | Implement CPU table with constraints | In progress | 5 weeks |
| Define basic CPU constraints | Add basic type constraints for CPU | In progress | 1 week|
| Decoder table | Implement decoder table | Planned | 1 week |
| Link decoder table and CPU | Use lookup to connect tables | Planned | 1 week |
| Constraints for updating pc | Implement constraints for updating pc | In progress | 1 week |
| ALU                         | Implement ALU tables with constraints| Not started | 6 weeks |
| Range checkers | Implement rangecheck for u16 and u8 | Planned | 1 week |
| Bitwise operations (and, xor, or) | Implement tables for u8 bitwise operations | Planned | 1 week |
| Shift operations | Implement tables for shift operations | Planned | 1 week |
| Multiplication table | Implement table for multiplication table | Planned | 1 week |
| Division and remainder | Implement table for integer division operations | 1 week |
| Memory                      | Implement memory table with constraints | Planned | 2 weeks |
| Syscalls                    | Tables for coprocessors | Planned | |
| Big Integer arithmetic | Table for big integer arithmetic | Planned | 1 week |
| Elliptic curve addition | Table for EC operations | Planned | 2 weeks  |
| Poseidon hash | Table for Poseidon hash | Planned | 3 weeks |
| Keccak hash | Table for Keccak hash | Planned | 3 weeks |
| SHA256 | Table for SHA256 | Planned | 3 weeks |
| Pairing | Table for pairings | Planned? | |

## Proof system

| Feature                     | Description                       | Status       | Duration |
|---------------------------- |-----------------------------------|--------------| -------- |
| Documentation               | Prepare comprehensive documentation on proof system | In progress   | 4 weeks |
| Lookup arguments            | Linking tables via lookup arguments | In progress | 2 weeks |
| Lookup - I | Accept multitables | In progress | 1 week |
| Lookup - II | Perform argument with constraints | In progress | 1 week |
| Multi-table Merkle trees (MTMT)   | Merkle tree that can be used to commit to polynomials of various sizes | In progress | 2 weeks |
| Multi-FRI                   | Perform FRI using MTMT | Planned | 2 weeks |
| Adjust parameters           | Adjust parameters for 128 bits of security | Planned | 1 week |
| Recursion | Allow for n-1 recursion tree to compress proof size | Planned | 4 weeks |

## Verifier

| Feature | Description | Status | Duration |
| ------ | -------- |--------| -----------|
|Ethereum verifier | Solidity verifier for the vm | Planned | 2 weeks |
|Verifier | Verifier for the vm | Planned | 2 weeks |
|Optimize Ethereum verifier | Optimize gas cost for verifier | Planned | 2 weeks |

## GPU and performance

| Feature                     | Description                       | Status       |
|---------------------------- |-----------------------------------|--------------|
| Fields                      | Improve field performance using assembly | Planned |
| GPU-Fast-Fourier transform      | Implement GPU version of FFT | Planned |
| GPU-Merkle tree                 | Implement GPU version for Merkle trees | Planned |
| Parallel witness generation   | Use GPU for fast witness generation | Planned |
| GPU-FRI | Perform FRI on GPU | Planned |

### Milestones

- Minimal CPU: able to prove simple computations, but not all operations supported
- RV64IM vm: prove general RV64IM code 
- Full vm: supports coprocessors for expensive operations
- GPU vm: leverages GPU for fast proving
  
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
