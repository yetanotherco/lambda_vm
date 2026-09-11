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

**Observability.** `cli execute <elf> --cycles` reports `Dma calls`, `Dma bytes` and `Dma rows`. The call line confirms the accelerator is engaged at all; the byte and row lines are the cost, since the guest stub chunks one `memcpy` into as many ecalls as it needs and each ecall adds one DMA row per eight-byte chunk, one per tail byte, and a terminal row. `Dma rows` is the raw row count the copies contribute, before the DMA trace is padded to a power-of-two height — for a guest with few copies the padded table is larger than the reported figure.

**Aligned vs misaligned.** The chunk width comes from the bytes remaining, not from the alignment of `dest` or `src`, so the DMA table's own row count is the same either way — but the cost is not. Each eight-byte chunk emits two width-8 memory operations, one reading the source and one writing the destination, each at its address as given, and the memory argument routes each one by that address: an 8-aligned window sharing one old timestamp reaches MEMW_A (29 columns, one ALU `LT` range check), and anything else falls to the general MEMW table (49 columns, eight `LT` rows). The two sides are independent, so a copy can take the fast path on one end and not the other; and because the width is chosen from the bytes remaining alone, a side that starts misaligned stays misaligned for every chunk. A misaligned copy therefore commits strictly more cells than an aligned copy of the same length, which is what makes the aligned/misaligned split the standard recommends informative here. It is not reported: the accelerator statistics are derived from `Log`, whose two operand slots are already taken (`src2_val = src`, `dst_val = n`, and `n` is what yields the byte and row figures), so reporting the split needs those statistics to move into the executor. Left as a follow-up, and stated here rather than claimed as done.

**Symbol resolution.** `compiler-builtins` defines `memcpy` *weakly*, and a linker extracts a static-archive member only to satisfy an *undefined* symbol — a weak definition already satisfies the reference, so a strong definition that lives in a member nothing else pulls in is dropped silently, with no duplicate-symbol diagnostic. Lambda VM therefore defines `memcpy` in [`syscalls/src/entrypoint.rs`](../syscalls/src/entrypoint.rs), the same object that defines `_start`, which every guest links unconditionally. That object is always extracted, so the strong definition is in the link graph from the start and overrides the weak one. No `--whole-archive` and no guest link flag is required, and resolution does not depend on archive order.

**Deviation from the standard's scope clause.** The standard says the accelerated symbols "are exported from the vendor static library defined by the Static Library and Linker Script standard". Lambda VM has no such library: the guest interface is a Rust rlib (`lambda-vm-syscalls`), and `memcpy` is exported from its always-linked entrypoint object. The linking clause above is satisfied by mechanism (1); the packaging the scope clause assumes is not, and adopting it is a repo-wide decision rather than one this accelerator can make.

Placing `memcpy` beside `_start` is insurance rather than a repair: defined in `syscalls.rs` it also resolved correctly, and not by luck — `_start` calls `sys_halt` from that module and it is not `#[inline]`, so every guest carries an undefined reference that forces the object out of the archive, whatever the guest itself names. What the move removes is the two things that guarantee rested on: `_start` continuing to call into `syscalls.rs`, and rustc's codegen-unit merging keeping the two modules together. Co-locating with `_start` — the one symbol the linker is obliged to resolve — makes the guarantee local instead, and `test_dma_memcpy_compiler_emitted_copies` (a guest that never names `memcpy`, asserting the DMA ecall count stays above zero) is what detects a regression, since a guest that falls back to the weak definition still produces correct output.
