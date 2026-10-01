use std::sync::Arc;

use ethrex_guest_program::l1::{ProgramInput, execution_program};
use lambda_vm_ethrex_crypto::LambdaVmEcsmCrypto;
use rkyv::rancor::Error;

pub fn main() {
    // Zero-copy private input: borrow the memory-mapped input region in place
    // (the host pre-loads it before execution) so rkyv deserializes straight
    // out of it. `get_private_input()` is this same slice plus a `to_vec()` —
    // a full extra copy and one large allocation (~50k cycles on a 20-tx
    // block).
    let input = lambda_vm_syscalls::syscalls::get_private_input_slice();
    let input = rkyv::from_bytes::<ProgramInput, Error>(input).unwrap();
    // LambdaVM crypto provider, defined in the lambda_vm repo and injected here
    // (so crypto changes don't require an ethrex PR — see `crypto/ethrex-crypto`).
    // It accelerates trait-routed `keccak256` (via the keccak_permute precompile)
    // and `secp256k1_ecrecover` (via the ECSM precompile); everything else uses
    // ethrex's pure-Rust trait defaults. ethrex's trie/RLP keccak that goes
    // through the free `keccak_hash` fn is still software, and KZG (0x0a) is
    // unsupported under the `lambdavm` feature (blob txs execute; a point-eval
    // precompile call reverts).
    let crypto = Arc::new(LambdaVmEcsmCrypto);
    let output = execution_program(input, crypto).unwrap();
    lambda_vm_syscalls::syscalls::commit(&output.encode());
}
