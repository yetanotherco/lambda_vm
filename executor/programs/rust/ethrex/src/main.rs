#[cfg(target_arch = "riscv64")]
use std::sync::Arc;

#[cfg(target_arch = "riscv64")]
use ethrex_guest_program::l1::execution_program;
#[cfg(target_arch = "riscv64")]
use lambda_vm_ethrex_crypto::LambdaVmEcsmCrypto;
#[cfg_attr(not(target_arch = "riscv64"), allow(dead_code))]
mod input;
#[cfg_attr(not(target_arch = "riscv64"), allow(dead_code))]
mod shared_table;

#[cfg(target_arch = "riscv64")]
pub fn main() {
    // Zero-copy private input: borrow the memory-mapped input region in place
    // (the host pre-loads it before execution) so rkyv deserializes straight
    // out of it. `get_private_input()` is this same slice plus a `to_vec()` —
    // a full extra copy and one large allocation (~50k cycles on a 20-tx
    // block).
    let input = lambda_vm_syscalls::syscalls::get_private_input_slice();
    let input = input::decode_input(input).unwrap();
    // LambdaVM crypto provider, defined in the lambda_vm repo and injected here
    // (so crypto changes don't require an ethrex PR — see `crypto/ethrex-crypto`).
    // It accelerates trait-routed `keccak256` (via the keccak_permute precompile)
    // and `secp256k1_ecrecover` (via the ECSM precompile); everything else uses
    // ethrex's pure-Rust trait defaults. ethrex's trie/RLP keccak that goes
    // through the free `keccak_hash` fn is still software. Unsupported, no backend
    // linked: KZG 0x0a (reverts) and BLS12-381 0x0b-0x11 (aborts on the unwrap below).
    let crypto = Arc::new(LambdaVmEcsmCrypto);
    let output = execution_program(input, crypto).unwrap();
    lambda_vm_syscalls::syscalls::commit(&output.encode());
}

// On the host this crate builds only for the unit tests of its input decoding.
#[cfg(not(target_arch = "riscv64"))]
pub fn main() {}
