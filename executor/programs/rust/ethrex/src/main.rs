use std::sync::Arc;

use ethrex_guest_program::crypto::lambdavm::LambdaVmCrypto;
use ethrex_guest_program::l1::{ProgramInput, execution_program};
use rkyv::rancor::Error;

pub fn main() {
    let input = lambda_vm_syscalls::syscalls::get_private_input();
    let input = rkyv::from_bytes::<ProgramInput, Error>(&input).unwrap();
    // LambdaVM crypto provider. Only `Crypto::keccak256` routes to our
    // keccak_permute precompile — ethrex's trie/RLP keccak goes through the free
    // `ethrex_crypto::keccak::keccak_hash` fn, which still runs software keccak on
    // riscv64, so the precompile only covers trait-routed keccak today. ECDSA and
    // BN254 use pure-Rust crates; KZG is unimplemented under the `lambdavm`
    // feature, so blob (EIP-4844) transactions are not supported.
    let crypto = Arc::new(LambdaVmCrypto);
    let output = execution_program(input, crypto).unwrap();
    lambda_vm_syscalls::syscalls::commit(&output.encode());
}
