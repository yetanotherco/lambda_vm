use crate::config::{GrindingDigest, KeccakStarkHash};
use crate::grinding::is_valid_nonce;

/// The default configuration's grinding hash. These vectors were computed
/// against keccak-256, so they name it rather than following any alias.
type Keccak = GrindingDigest<KeccakStarkHash>;

#[test]
fn test_invalid_nonce_grinding_factor_6() {
    // This setting produces a hash with 5 leading zeros, therefore not enough for grinding
    // factor 6.
    let seed = [
        174, 187, 26, 134, 6, 43, 222, 151, 140, 48, 52, 67, 69, 181, 177, 165, 111, 222, 148, 92,
        130, 241, 171, 2, 62, 34, 95, 159, 37, 116, 155, 217,
    ];
    let nonce = 4;
    let grinding_factor = 6;
    assert!(!is_valid_nonce::<Keccak>(&seed, nonce, grinding_factor));
}

#[test]
fn test_invalid_nonce_grinding_factor_9() {
    // This setting produces a hash with 8 leading zeros, therefore not enough for grinding
    // factor 9.
    let seed = [
        174, 187, 26, 134, 6, 43, 222, 151, 140, 48, 52, 67, 69, 181, 177, 165, 111, 222, 148, 92,
        130, 241, 171, 2, 62, 34, 95, 159, 37, 116, 155, 217,
    ];
    let nonce = 287;
    let grinding_factor = 9;
    assert!(!is_valid_nonce::<Keccak>(&seed, nonce, grinding_factor));
}

#[test]
fn test_is_valid_nonce_grinding_factor_10() {
    let seed = [
        37, 68, 26, 150, 139, 142, 66, 175, 33, 47, 199, 160, 9, 109, 79, 234, 135, 254, 39, 11,
        225, 219, 206, 108, 224, 165, 25, 72, 189, 96, 218, 95,
    ];
    let nonce = 0x5ba;
    let grinding_factor = 10;
    assert!(is_valid_nonce::<Keccak>(&seed, nonce, grinding_factor));
}

#[test]
fn test_is_valid_nonce_grinding_factor_20() {
    let seed = [
        37, 68, 26, 150, 139, 142, 66, 175, 33, 47, 199, 160, 9, 109, 79, 234, 135, 254, 39, 11,
        225, 219, 206, 108, 224, 165, 25, 72, 189, 96, 218, 95,
    ];
    let nonce = 0x2c5db8;
    let grinding_factor = 20;
    assert!(is_valid_nonce::<Keccak>(&seed, nonce, grinding_factor));
}

#[test]
fn test_invalid_nonce_grinding_factor_19() {
    // This setting would pass for grinding factor 20 instead of 19. The nonce is invalid
    // here because the grinding factor is part of the inner hash, changing the outer hash
    // and the resulting number of leading zeros.
    let seed = [
        37, 68, 26, 150, 139, 142, 66, 175, 33, 47, 199, 160, 9, 109, 79, 234, 135, 254, 39, 11,
        225, 219, 206, 108, 224, 165, 25, 72, 189, 96, 218, 95,
    ];
    let nonce = 0x2c5db8;
    let grinding_factor = 19;
    assert!(!is_valid_nonce::<Keccak>(&seed, nonce, grinding_factor));
}

#[test]
fn test_is_valid_nonce_grinding_factor_30() {
    let seed = [
        37, 68, 26, 150, 139, 142, 66, 175, 33, 47, 199, 160, 9, 109, 79, 234, 135, 254, 39, 11,
        225, 219, 206, 108, 224, 165, 25, 72, 189, 96, 218, 95,
    ];
    let nonce = 0x1ae839e1;
    let grinding_factor = 30;
    assert!(is_valid_nonce::<Keccak>(&seed, nonce, grinding_factor));
}

#[test]
fn test_is_valid_nonce_grinding_factor_33() {
    let seed = [
        37, 68, 26, 150, 139, 142, 66, 175, 33, 47, 199, 160, 9, 109, 79, 234, 135, 254, 39, 11,
        225, 219, 206, 108, 224, 165, 25, 72, 189, 96, 218, 95,
    ];
    let nonce = 0x4cc3123f;
    let grinding_factor = 33;
    assert!(is_valid_nonce::<Keccak>(&seed, nonce, grinding_factor));
}

// =========================================================================
// The BLAKE3 configuration's proof of work.
//
// Grinding is a type substitution — same two one-block hashes, same 32-byte
// seed and digest — so what needs pinning is that the substitution actually
// happened and that the two configurations do not accept each other's work.
// =========================================================================

/// The digest the BLAKE3 configuration grinds over. Gated with the tests below
/// because `Blake3StarkHash` does not exist under `cuda` — the device kernels
/// are keccak-only, so there is no second configuration to name there.
#[cfg(not(feature = "cuda"))]
type Blake3 = GrindingDigest<crate::config::Blake3StarkHash>;

/// ★ Honest path: a nonce ground under BLAKE3 satisfies the BLAKE3 check.
///
/// At `grinding_factor = 20` a nonce passes by chance with probability 2⁻²⁰, so
/// the cross-hash rejection below is a real control rather than a coin flip.
#[cfg(not(feature = "cuda"))]
#[test]
fn a_blake3_ground_nonce_satisfies_the_blake3_check() {
    let seed = [0x5au8; 32];
    let factor = 20;

    let nonce = crate::grinding::generate_nonce::<Blake3>(&seed, factor)
        .expect("a nonce exists at this factor");
    assert!(
        is_valid_nonce::<Blake3>(&seed, nonce, factor),
        "the nonce grinding found must satisfy the check it was ground against"
    );

    // FALSIFICATION: work done against one hash is not work against the other.
    // Without this, `generate_nonce::<Blake3>` could still be computing keccak
    // and every assertion above would hold.
    assert!(
        !is_valid_nonce::<Keccak>(&seed, nonce, factor),
        "a BLAKE3-ground nonce must not satisfy the keccak check"
    );
}

/// The two configurations compute different work on identical inputs.
#[cfg(not(feature = "cuda"))]
#[test]
fn the_two_configurations_grind_different_work() {
    let seed = [0x11u8; 32];
    let factor = 16;

    let blake3 = crate::grinding::generate_nonce::<Blake3>(&seed, factor).expect("blake3 nonce");
    let keccak = crate::grinding::generate_nonce::<Keccak>(&seed, factor).expect("keccak nonce");
    assert_ne!(
        blake3, keccak,
        "the same seed under two hashes must not grind to the same nonce"
    );
}
