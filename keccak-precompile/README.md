# keccak-precompile

A drop-in shim for the [`keccak`](https://crates.io/crates/keccak) crate
(version 0.1.5) that routes the Keccak-f[1600] / Keccak-p[1600, 24]
permutation to the lambda-vm RISC-V **Keccak precompile** when compiled for
the `riscv64` guest.

## What it is

`src/lib.rs` is a **verbatim copy** of upstream `keccak` 0.1.5, with a single
targeted change: the non-asm `p1600` / `f1600` functions delegate the full
24-round permutation to the VM precompile via an `ecall`:

```rust
#[cfg(target_arch = "riscv64")]
if round_count == 24 {
    // ABI: a0 = state ptr, a7 = usize::MAX - 1
    unsafe { core::arch::asm!("ecall", in("a0") state.as_mut_ptr(), in("a7") usize::MAX - 1); }
    return;
}
```

Every other code path (reduced-round `p1600`, the `LaneSize` trait, `keccak_p`,
the round constants, the `u8`/`u16`/`u32` lane sizes, the `simd` module, and the
aarch64 `asm` variants) is unchanged, so host builds behave exactly like
upstream.

## Why it works

The STARK verifier hashes via `sha3::Keccak256`. `sha3` 0.10.x performs its
permutation by calling `keccak::p1600(&mut state, 24)`. By overriding that
single function and patching the `keccak` crate, **all** of `sha3`'s usage —
the Merkle tree backend and the Fiat-Shamir transcript — transparently routes
to the precompile on `riscv64`, while reusing `sha3`'s correct sponge and
padding.

Correctness is guaranteed: the precompile computes exactly Keccak-f[1600]
(= 24-round `p1600`) in place over the `[u64; 25]` state, so hashes are
byte-identical to the software implementation.

## How the recursion guest enables it

Add to the **guest's root `Cargo.toml`**:

```toml
[patch.crates-io]
keccak = { path = "<relative path to>/keccak-precompile" }
```

This replaces the upstream `keccak` dependency pulled in transitively by
`sha3`. On `riscv64` the permutation hits the precompile; on host targets the
build is unchanged.

## Testing

The riscv64 precompile path cannot run on the host. The included unit test
(`cargo test`) verifies the copied software permutation is faithful by:

1. Asserting `f1600` of the all-zero state matches the standard Keccak-f[1600]
   test vector, and that `p1600(_, 24)` equals `f1600`.
2. Comparing `crate::f1600` / `crate::p1600` against the upstream `keccak`
   crate (pulled in as a renamed dev-dependency `dev-dep-keccak` to avoid a
   self-patch cycle) over many pseudo-random states and reduced round counts.

## License

Apache-2.0 OR MIT, matching upstream `keccak`.
