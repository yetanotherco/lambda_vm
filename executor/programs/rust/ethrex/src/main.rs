use std::sync::Arc;

use ethrex_guest_program::l1::run_stateless_guest;
use lambda_vm_ethrex_crypto::LambdaVmEcsmCrypto;

pub fn main() {
    let input = lambda_vm_syscalls::syscalls::get_private_input_slice();
    // LambdaVM crypto provider, defined in this repo and injected here (so crypto
    // changes don't require an ethrex PR — see `crypto/ethrex-crypto`). It
    // accelerates trait-routed `keccak256` (via the keccak_permute precompile) and
    // `secp256k1_ecrecover` (via the ECSM precompile); every other `Crypto` method
    // inherits ethrex's pure-Rust trait default. No KZG backend is linked, so the
    // point-evaluation precompile (0x0a) is unsupported — `no_kzg_backend_linked`
    // in tooling/ethrex-tests pins that.
    let crypto = Arc::new(LambdaVmEcsmCrypto);
    let output = run_stateless_guest(input, crypto);
    lambda_vm_syscalls::syscalls::commit(&output);
}
