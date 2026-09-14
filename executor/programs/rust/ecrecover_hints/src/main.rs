//! ecrecover measurement guest for the hint arena: N secp256k1 recoveries via
//! the LambdaVM crypto provider, whose inverses/sqrts come from the
//! private-input hint arena (no hint ecall).
//!
//! Private input layout: `[u32 LE count]` then `count` records of
//! `sig(64) || recid(1) || msg(32)`. The recovered addresses are XOR-folded and
//! committed. Hint consumption is positional: per recovery the guest requests
//! sqrt (decompress), the scalar inverse, and the batched field inverse
//! (lincomb), in that order — the host's arena must follow the same order.

use ethrex_crypto::Crypto;
use lambda_vm_ethrex_crypto::LambdaVmEcsmCrypto;
use lambda_vm_syscalls as syscalls;

pub fn main() {
    let input = syscalls::syscalls::get_private_input();
    assert!(input.len() >= 4, "input too short for count");
    let count = u32::from_le_bytes(input[0..4].try_into().unwrap()) as usize;
    assert_eq!(input.len(), 4 + count * 97, "input length mismatch");

    let crypto = LambdaVmEcsmCrypto;
    let mut acc = [0u8; 32];
    let mut off = 4;
    for _ in 0..count {
        let sig: &[u8; 64] = input[off..off + 64].try_into().unwrap();
        let recid = input[off + 64];
        let msg: &[u8; 32] = input[off + 65..off + 97].try_into().unwrap();
        let addr = crypto
            .secp256k1_ecrecover(sig, recid, msg)
            .expect("ecrecover failed");
        for i in 0..32 {
            acc[i] ^= addr[i];
        }
        off += 97;
    }

    syscalls::syscalls::commit(&acc);
}
