# Roadmap for the virtual machine

This is a tentative list of features that are going to be implemented in the near future. Soon we will provide rough estimates on the time each task should take to implement. This may change according to the project's needs.

## Basic building blocks

The first version is going to use the primitives contained in [lambdaworks](https://github.com/lambdaclass/lambdaworks)

| Feature                     | Description                        | Status       |
|---------------------------- |------------------------------------|--------------|
| Documentation               | Explain how everything works       | In progress  |
| Field                       | Basic field type                   | First version|
| Poseidon hash               | Implement Poseidon-2 hash          | Planned      |
| Keccak                      | Implement Keccak hash function     | Planned      |
| CPU FFT                     | Radix-2 Fast-Fourier transform in CPU | First version |
| Basic Merkle commitment     | Merkle tree                        | First version |
| Basic FRI                   | FRI proximity test                 | First version |
| Basic constraints           | Simple API for defining constraints for AIR | First version |
| Basic AIR                   | Algebraic intermediate representation for computations | First version |       

## Executor

| Feature                     | Description                        | Status       |
|---------------------------- |------------------------------------|--------------|
| Documentation               | Explain how the executor works     | In progress  |
| Minimal CPU                 | Minimal CPU that can perform basic operations | In progress |
| RISCV64 CPU                 | Minimal version of the CPU with 52 RISCV instructions | Planned |
| RISCV64IM CPU               | Working executor for RV64 virtual machine | Planned |
| CPU with coprocessors       | Add coprocessors for special cryptographic operations   | Planned |


## Trace generator

| Feature                     | Description                       | Status       |
|---------------------------- |-----------------------------------|--------------|
| Documentation               | Document trace generation and constraints | In progress |
| CPU                         | Implement CPU table with constraints | In progress |
| ALU                         | Implement ALU tables with constraints| Not started |
| Memory                      | Implement memory table with constraints | Not started |
| Syscalls                    | Tables for coprocessors | Planned |


## Proof system

| Feature                     | Description                       | Status       |
|---------------------------- |-----------------------------------|--------------|
| Documentation               | Prepare comprehensive documentation on proof system | In progress   |
| Lookup arguments            | Linking tables via lookup arguments | In progress |
| Multi-table Merkle trees (MTMT)   | Merkle tree that can be used to commit to polynomials of various sizes | In progress
| Multi-FRI                   | Perform FRI using MTMT | Planned |
| Adjust parameters           | Adjust parameters for 128 bits of security | Planned |

## Prover Optimizations

Based on profiling at 2^20 trace (1M rows), here are the bottlenecks and optimization priorities:

### Profiling Results (10.66s total prove time)

| Component | Time | % | Priority |
|-----------|------|---|----------|
| Merkle tree construction (total) | 2.76s | 25.9% | P1 |
| FRI protocol | 1.77s | 16.6% | P2 |
| LDE evaluations (total) | 1.65s | 15.5% | P3 |
| Constraint evaluation | 1.65s | 15.5% | P4 |
| Composition poly interpolation | 852ms | 8.0% | P5 |

Verifier is highly optimized: ~232μs (constant time regardless of trace size).

### CPU Optimizations

| Feature | Description | Status |
|---------|-------------|--------|
| Lightweight verifier domain | Avoid O(n) domain allocation in verifier | Done |
| Mathematical OOD sampling | O(log n) membership check vs O(n) search | Done |
| Pre-computed FFT twiddles | Reuse twiddles across polynomial operations | Done |
| Parallel constraint evaluation | Use rayon for constraint evaluation | Done |
| Vec capacity pre-allocation | Reduce allocations in hot paths | Done |
| Batch domain element computation | Incremental multiplication (not beneficial) | Tested |

### Optimization Priorities

| Priority | Feature | Description | Expected Impact | Status |
|----------|---------|-------------|-----------------|--------|
| P1 | Parallel Merkle hashing | Parallelize leaf hashing in Merkle tree construction | ~20% total speedup | Planned |
| P2 | FRI optimization | Profile and optimize FRI inner loop (FFTs, hashing) | ~15% total speedup | Planned |
| P3 | SIMD field operations | Use AVX2/AVX-512 for field arithmetic | ~10-20% speedup | Planned |
| P4 | Better hash function | Replace SHA256 with Poseidon or BLAKE3 | Faster Merkle trees | Planned |
| P5 | Circle STARK | Consider Circle STARK for FFT-friendly fields | Architecture change | Research |

## GPU and Performance

| Feature                     | Description                       | Status       |
|---------------------------- |-----------------------------------|--------------|
| Fields                      | Improve field performance using assembly | Planned |
| GPU-Fast-Fourier transform      | Implement GPU version of FFT | Planned |
| GPU-Merkle tree                 | Implement GPU version for Merkle trees (highest impact) | Planned |
| Parallel trace generation   | Use GPU for fast trace generation | Planned |
| GPU-FRI | Perform FRI on GPU | Planned |
  