use std::sync::Arc;

use ethrex_guest_program::l1::run_stateless_guest;
use lambda_vm_ethrex_crypto::LambdaVmEcsmCrypto;

pub fn main() {
    let input = lambda_vm_syscalls::syscalls::get_private_input_slice();
    let crypto = Arc::new(LambdaVmEcsmCrypto);
    let output = run_stateless_guest(input, crypto);
    lambda_vm_syscalls::syscalls::commit(&output);
}
