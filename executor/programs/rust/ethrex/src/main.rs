use std::sync::Arc;

use ethrex_guest_program::crypto::lambdavm::LambdaVmCrypto;
use ethrex_guest_program::l1::{ProgramInput, execution_program};
use rkyv::rancor::Error;

pub fn main() {
    let input = lambda_vm_syscalls::syscalls::get_private_input();
    let input = rkyv::from_bytes::<ProgramInput, Error>(&input).unwrap();
    // LambdaVM crypto provider: keccak256 routes to our keccak_permute precompile;
    // ECDSA/BN254/KZG/etc. fall back to pure-Rust trait defaults for now.
    let crypto = Arc::new(LambdaVmCrypto);
    let output = execution_program(input, crypto).unwrap();
    lambda_vm_syscalls::syscalls::commit(&output.encode());
}
