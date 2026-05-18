# Lambda VM Syscalls (Guest SDK)

Guest-side library for programs that run inside the Lambda VM. Provides the syscalls and the default entry point that let a Rust program interact with the host: read private input, commit public output, halt, and invoke precompiles.

Published as `lambda-vm-syscalls`. Intended to be used from RISC-V (RV64IM) guest binaries cross-compiled with the toolchain described in the root [`README.md`](../README.md).

## What it provides

| Function | Purpose |
|---|---|
| `commit(bytes: &[u8])` | Append bytes to the **public output** that the verifier checks. |
| `get_private_input() -> Vec<u8>` | Read the host-supplied private input bytes (memory-mapped at `0xFF000000`). |
| `sys_halt() -> !` | Terminate execution cleanly. Called automatically after `main` by the default entry point. |
| `keccak_permute(state: &mut [u64; 25])` | Keccak-f[1600] permutation precompile. |

The crate also provides a default `_start` that initialises the allocator, calls `main`, and halts.

> **Note:** the `print_string` syscall is temporarily unavailable — calling it in a guest will cause proof verification to fail. Tracked as a follow-up.

## Example

A minimal guest that reads private input and commits a (non-secret) summary of it:

```rust
use lambda_vm_syscalls::syscalls;

pub fn main() {
    let input = syscalls::get_private_input();

    // Anything passed to `commit` becomes part of the proof's public output.
    // Don't echo private input here — commit a derived value instead.
    let len = (input.len() as u32).to_le_bytes();
    syscalls::commit(&len);
}
```

See [`executor/programs/rust/`](../executor/programs/rust/) for more example guests (`fibonacci`, `keccak`, `hashmap`, …).

## Building a guest

Guests are compiled with a pinned nightly toolchain for the custom RISC-V target `riscv64im-lambda-vm-elf.json`. The simplest path is to drop a new project under `executor/programs/rust/<name>/` and run `make compile-programs-rust` from the repo root.

See the root [`README.md`](../README.md) for the full toolchain setup (sysroot, nightly pin, target spec).

## Unsupported `std` functions

The following functions are stubbed and **panic at runtime** if called — Lambda VM does not provide stdin or command-line arguments:

- `sys_read` (so `io::Read` for `Stdin` is not available)
- `sys_argc`, `sys_argv`

To pass data into a guest, use `get_private_input()` instead.
