//! A W-LFM proof's identity and bytes, printed: the two-build check that a
//! change to the W-LFM path leaves today's proofs byte for byte. The same file
//! is compiled into the base's build and the head's; the two printed lines
//! must be equal. Needs `LAMBDA_VM_DETERMINISTIC_GRIND=1` (the grind may take
//! any valid nonce otherwise).

use crypto::hash::blake3::Blake3Std;
use digest::Digest;

use super::wlfm::{WlfmRate, prove_wlfm, verify_wlfm};
use super::wlfm_tests::{arenas, build, honest};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The fixture program at the production config (Poseidon1, blowup 2,
/// grinding on): its identity and its proof's bytes, as `Blake3Std` digests.
#[test]
#[ignore = "two-build pin: run with LAMBDA_VM_DETERMINISTIC_GRIND=1 in both builds"]
fn wlfm_pin_print() {
    assert!(
        std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some(),
        "the pin needs the deterministic grind"
    );
    let program = honest();
    let build = build(&program, WlfmRate::Two);
    let proof = prove_wlfm(&program, &build, &arenas()).expect("the fixture proves");
    verify_wlfm(&build.artifacts, &proof, &proof.public_words).expect("and verifies");
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof).expect("serializes");
    let digest: [u8; 32] = <Blake3Std as Digest>::digest(&bytes).into();
    println!(
        "WLFM PIN: id {} · proof {} bytes · proof digest {}",
        hex(&build.artifacts.program_id),
        bytes.len(),
        hex(&digest)
    );
}
