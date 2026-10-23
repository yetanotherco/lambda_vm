//! ★ The IDENTITY line for a WHIR proof — the byte gate the hash seam is
//! measured against.
//!
//! # What it is for
//!
//! The control for every arm of the hash work is "with the parameter unset, the
//! proof is the one PR #988 produced". Hashing the serialized proof would be
//! the obvious way to say that, and it does not work: **grinding nonces are
//! nondeterministic.** `crypto::grinding::generate_nonce` searches with rayon's
//! `find_any` under the `parallel` feature and returns whichever valid nonce a
//! worker reached first; on the device arm the kernel returns the smallest in
//! the range it scanned. Neither is a contract — the verifier accepts any nonce
//! passing `is_valid_nonce` — so two honest runs of the same prover on the same
//! input produce different proof BYTES, and a raw digest of them reports a
//! difference that means nothing.
//!
//! So the digest below is taken over the proof with **every grinding nonce
//! zeroed**, and over nothing else that has been excluded. Everything a hash
//! swap actually moves — every Merkle root, every out-of-domain value, every
//! sumcheck coefficient, every opened codeword value and every authentication
//! path — is inside it.
//!
//! # What it therefore does NOT cover
//!
//! - **The nonces themselves.** A defect that produced a valid-but-wrong nonce
//!   would not show here. It is not invisible: `check_grind` rejects an invalid
//!   nonce at verify time, and that is the gate for this property.
//! - **Anything outside the `MultiProof`.** Table heights, page ranges, public
//!   output and the epoch bookends live in the enclosing proof structs; the
//!   caller hashes those separately if it wants them bound.
//! - **Serialized LENGTH is checked separately** by
//!   [`serialized_len`], because that is the sharper of the two statements: a
//!   hash swap must leave the length equal to the byte (32-byte digests either
//!   way, no proof struct gains a field), while the digest is *expected* to
//!   change under a different hash.

use multilinear::whir_chain::{ChainProof, RoundNonces};

use crate::test_utils::{E, F};

/// The keccak the identity line itself is taken with.
///
/// Deliberately fixed, and deliberately NOT the proof's own hash: this is a
/// measuring instrument, not part of the protocol. If it followed the
/// configuration then the keccak arm and the RPX arm would be hashed by
/// different functions, and "the digests differ" would no longer distinguish a
/// changed proof from a changed instrument.
type Line = crypto::hash::platform_keccak::PlatformKeccak256;

/// A proof with every grinding nonce zeroed — the form the identity line is
/// taken over.
fn without_nonces(proof: &MultiProof) -> MultiProof {
    let mut out = proof.clone();
    for stacked in &mut out.columns {
        for chain in &mut stacked.polys {
            zero_nonces(chain);
        }
    }
    out
}

fn zero_nonces(chain: &mut ChainProof<F, E>) {
    for round in &mut chain.rounds {
        round.nonces = RoundNonces::default();
    }
}

/// The `MultiProof` this VM's multilinear path produces.
pub type MultiProof = stark::multilinear_table::MultiProof<F, E>;

/// ★ The identity line: keccak-256 over the rkyv bytes of the proof with every
/// grinding nonce zeroed.
///
/// Two runs of the same prover on the same input must produce the same line.
/// Two runs under different hash configurations must produce different ones —
/// see [`crate::tests::whir_identity_tests`], where both halves are asserted,
/// because a digest that could not change is not a gate.
pub fn identity_line(proof: &MultiProof) -> Result<[u8; 32], String> {
    use digest::Digest;

    let normalised = without_nonces(proof);
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&normalised)
        .map_err(|e| format!("the proof did not serialize: {e}"))?;
    let digest = Line::digest(bytes.as_ref());
    Ok(digest.into())
}

/// The identity line as lowercase hex — what a box run prints and a coordinator
/// diffs.
pub fn identity_hex(proof: &MultiProof) -> Result<String, String> {
    Ok(identity_line(proof)?
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// The serialized length of the proof, in bytes.
///
/// ★ Checked SEPARATELY from the digest and held to a stricter standard: the
/// digest is expected to move when the hash moves, the length is not. Every
/// [`multilinear::whir_hash::WhirHash`] has a 32-byte commitment and no proof
/// struct gains a field, so a length difference between two hash arms is a
/// defect in the seam rather than a property of the hash.
pub fn serialized_len(proof: &MultiProof) -> Result<usize, String> {
    Ok(rkyv::to_bytes::<rkyv::rancor::Error>(proof)
        .map_err(|e| format!("the proof did not serialize: {e}"))?
        .len())
}
