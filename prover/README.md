# Lambda VM Prover

STARK prover for the Lambda VM. Proves correct execution of RISC-V ELF binaries by generating a multi-table STARK proof (CPU, decode, bitwise, branch, LT, shift, MUL, DVRM, MEMW, LOAD, page, register, halt, commit, keccak) and provides the matching native verifier.

Published as `lambda-vm-prover`. Consumed by [`bin/cli`](../bin/cli) and by the benchmarks; you can also use it directly from Rust.

## Usage

```rust
use lambda_vm_prover as prover;

let elf_bytes = std::fs::read("program.elf").unwrap();

let proof = prover::prove(&elf_bytes).unwrap();
assert!(prover::verify(&proof, &elf_bytes).unwrap());
```

With private inputs:

```rust
let private_inputs = std::fs::read("input.bin").unwrap();
let proof = prover::prove_with_inputs(&elf_bytes, &private_inputs).unwrap();
```

## Public API

| Function | Description |
|---|---|
| `prove(elf)` | Generate a proof with default options (blowup = 2). |
| `prove_with_inputs(elf, private)` | Same, with private input bytes. |
| `prove_with_options(elf, opts, max_rows)` | Custom proof options and max-rows config. |
| `prove_with_options_and_inputs(...)` | Most general entry point. |
| `verify(proof, elf)` | Verify a proof with default options. |
| `verify_with_options(proof, elf, opts)` | Verify with caller-chosen options (the verifier enforces its own security parameters, not the prover's). |
| `prove_and_verify(elf)` | Prove + verify in one call (convenience). |
| `count_elements(elf, private)` | Build traces and return `(main, aux)` field-element counts without running the proof step. |

The proof bundle type is `VmProof`, containing the multi-table STARK proof, the public output bytes committed by the guest, and the metadata the verifier needs to reconstruct the AIR configuration (table chunk counts, runtime page ranges, number of private-input pages).

## Features

| Feature | Description |
|---|---|
| `parallel` (default) | Rayon parallelism across tables and FFTs. |
| `debug-checks` | Runs `validate_trace` and prints a per-bus LogUp balance report after proving. Forwarded to `crypto/stark`. |
| `instruments` | Per-phase timing and heap-usage report (execute, trace build, AIR construction, prove). |

To run the test suite with debug output:

```sh
cargo test --release -p lambda-vm-prover --features debug-checks -- --nocapture
```

See the root [`README.md`](../README.md) for the end-to-end workflow (compiling guest programs, the CLI wrapper, benchmarks).
