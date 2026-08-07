# Overview of VM flow

The Lambda VM proves correct execution of a RISC-V (RV64IM) program against an input stream. The pipeline has five artifacts and four transformations.

## Artifacts

1. **Source code** — high-level Rust (using [`syscalls/`](../syscalls/) for guest-host I/O) or RISC-V assembly.
2. **ELF binary** — the program in the VM's ISA, ready to load.
3. **Execution record** — per-instruction logs emitted by running the ELF on the VM.
4. **Witness** — a set of trace tables (CPU, decode, MEMW, LOAD, bitwise, branch, LT, shift, MUL, DVRM, page, register, halt, commit, keccak, …) derived from the execution record. Each table is an AIR (Algebraic Intermediate Representation); tables are linked by LogUp lookup arguments.
5. **Proof** — a multi-table STARK proof (transparent, hash-based, post-quantum secure) that the witness satisfies all AIR constraints and lookup arguments. Low-degree of the witness polynomials is verified via FRI.

## Transformations

1. **Compiler** — `rustc` cross-compiles to the custom RISC-V target spec ([`executor/programs/riscv64im-lambda-vm-elf.json`](../executor/programs/riscv64im-lambda-vm-elf.json)) and produces the ELF. The `lambda-vm-syscalls` crate exposes guest-side syscalls (`commit`, `get_private_input`, `print_string`, `keccak_permute`, `sys_halt`).
2. **Executor** ([`executor/`](../executor/)) — loads the ELF, runs the program against the VM's memory and register state, handles syscalls and precompiles (e.g. Keccak), and emits the per-instruction logs.
3. **Witness generator** ([`prover/src/tables/`](../prover/src/tables/)) — turns the logs into trace tables, populates AIR columns, and computes the LogUp auxiliary columns that connect tables.
4. **Proof system** ([`crypto/stark/`](../crypto/stark/)) — commits to each table's trace via Merkle trees, samples challenges via Fiat-Shamir, and runs FRI for the low-degree test. Produces a `MultiProof`; the verifier replays the transcript and checks all AIR and lookup constraints.

For a deeper dive into each component see the [proof system overview](./cryptography/proof_system.md).

## Accelerated memory operations

`memcpy` is accelerated: `lambda-vm-syscalls` exports it under its standard, unmangled C name, so both explicit calls and the copies the compiler emits implicitly (struct moves, slice copies, `Vec` growth) reach the DMA ecall with no guest source changes. Behaviour is identical to the C function for every input, including `n == 0` and any alignment of `dest`, `src` or `n`. `memmove`, `memset` and `memcmp` are not accelerated and fall back to the toolchain's `compiler-builtins` definitions.

**Observability.** `cli execute <elf> --cycles` reports `Dma calls`, `Dma bytes` and `Dma rows`. The call line confirms the accelerator is engaged at all; the byte and row lines are the cost, since the guest stub chunks one `memcpy` into as many ecalls as it needs and each ecall adds one DMA row per eight-byte chunk, one per tail byte, and a terminal row.

**Symbol resolution.** `compiler-builtins` defines `memcpy` *weakly*, and a linker extracts a static-archive member only to satisfy an *undefined* symbol — a weak definition already satisfies the reference, so a strong definition that lives in a member nothing else pulls in is dropped silently, with no duplicate-symbol diagnostic. Lambda VM therefore defines `memcpy` in [`syscalls/src/entrypoint.rs`](../syscalls/src/entrypoint.rs), the same object that defines `_start`, which every guest links unconditionally. That object is always extracted, so the strong definition is in the link graph from the start and overrides the weak one. No `--whole-archive` and no guest link flag is required, and resolution does not depend on archive order.
