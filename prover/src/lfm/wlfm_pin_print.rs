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

/// ★ The fixture program's W-LFM identity at the production config, pinned:
/// the identity it had before the WHIR round form existed (the round form is
/// hashed only when a chain takes it). Grind-free: no proof is made.
#[cfg(not(feature = "test-security"))]
#[test]
fn the_default_wlfm_program_id_is_unmoved() {
    let program = honest();
    let build = build(&program, WlfmRate::Two);
    let id = hex(&build.artifacts.program_id);
    println!("WLFM ID: {id}");
    assert_eq!(
        id,
        "75c808decaf786f67e9a0d7105843270a2c612a32c86b47183a9ed320e26b670"
    );
    // The text the identity hashes names no round form, under either form:
    // the round form is hashed on its own, and only when a chain takes it.
    let fri = build.artifacts.config.format;
    let text = super::wlfm::format_identity_text(&fri);
    assert!(!text.contains("rounds"), "{text}");
    let whir = build
        .artifacts
        .config
        .with_whir_rounds(3, 128, 20)
        .expect("a round form")
        .format;
    assert_eq!(super::wlfm::format_identity_text(&whir), text);
}

/// ★ The chain format's identity text under a batched argue on today's
/// one-layer GKR ladder is the text it was before the ladder existed (the
/// literal below, printed by the format's `Debug` then, less its round form);
/// a two-layer ladder names itself in the text, so it moves the identity.
#[test]
fn the_identity_text_of_a_one_layer_ladder_is_unmoved() {
    use multilinear::whir_chain::{ArgueFormat, ChainFormat, GkrLadder, GkrPack};
    let format = |argue: ArgueFormat| ChainFormat {
        argue,
        ..ChainFormat::DEFAULT
    };
    assert_eq!(
        super::wlfm::format_identity_text(&format(ArgueFormat::BATCHED)),
        "ChainFormat { cap: Off, folds: Uniform, stack: StackVars(25), nonces: Three, argue: \
         Batched { bin_log_cells: 26, pack: Chunks { tree_log_cells: 22 } }, arity4_cap: Off, \
         group: PerPoly, leaves: Natural, round0: Each, final_poly: Off }"
    );
    let off = ArgueFormat::Batched {
        bin_log_cells: 25,
        pack: GkrPack::Off,
        ladder: GkrLadder::Single,
    };
    assert_eq!(
        super::wlfm::format_identity_text(&format(off)),
        "ChainFormat { cap: Off, folds: Uniform, stack: StackVars(25), nonces: Three, argue: \
         Batched { bin_log_cells: 25, pack: Off }, arity4_cap: Off, group: PerPoly, leaves: \
         Natural, round0: Each, final_poly: Off }"
    );
    let double = super::wlfm::format_identity_text(&format(
        ArgueFormat::BATCHED.with_ladder(GkrLadder::Double),
    ));
    assert!(double.contains("ladder: Double"), "{double}");
    assert_ne!(
        double,
        super::wlfm::format_identity_text(&format(ArgueFormat::BATCHED))
    );
}
