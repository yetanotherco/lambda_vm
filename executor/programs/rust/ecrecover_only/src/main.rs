//! Minimal ecrecover reproducer for debugging LogUp bus imbalance.
//! Signs a message then recovers — exercises the full k256 stack.
use k256::ecdsa::{SigningKey, VerifyingKey, signature::hazmat::PrehashSigner};
use lambda_vm_syscalls as syscalls;
use sha3::{Digest, Keccak256};

pub fn main() {
    // Fixed private key (classic test key, Hardhat account #0)
    let key_bytes: [u8; 32] = [
        0xac, 0x09, 0x74, 0xbe, 0xc3, 0x9a, 0x17, 0xe3, 0x6b, 0xa4, 0xa6, 0xb4, 0xd2, 0x38, 0xff,
        0x94, 0x4b, 0xac, 0xb4, 0x78, 0xcb, 0xed, 0x5e, 0xfc, 0xae, 0x78, 0x4d, 0x7b, 0xf4, 0xf2,
        0xff, 0x80,
    ];
    let signing_key = SigningKey::from_slice(&key_bytes).expect("valid key");

    // Hash a fixed message
    let msg_hash: [u8; 32] = {
        let mut h = Keccak256::new();
        h.update(b"lambda_vm test");
        h.finalize().into()
    };

    // Sign with k256 (real signing exercises full field ops)
    let (signature, recovery_id) = signing_key
        .sign_prehash(&msg_hash)
        .expect("sign succeeds");

    // Recover the verifying key (full point mul, field inverse, etc.)
    let recovered = VerifyingKey::recover_from_prehash(&msg_hash, &signature, recovery_id)
        .expect("recover succeeds");

    // Commit first byte of the recovered key as a public output
    let bytes = recovered.to_encoded_point(true);
    let out = [bytes.as_bytes()[1]];
    syscalls::syscalls::commit(&out);
}
