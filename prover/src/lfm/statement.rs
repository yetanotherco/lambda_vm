//! LFM program identity and statement binding.
//!
//! `lfm_program_id` binds the instruction column groups (roots + heights),
//! the machine version and the preset — it is the digest the registry pins
//! and the consumer's attestation folds. Keccak today; `_V2` rides the
//! ecosystem hash migration (a host/consumer-side artifact).
//!
//! The statement absorb seeds the Fiat–Shamir transcript before
//! `multi_prove` / `multi_verify_views`, exactly like the RV64 VM's
//! `statement.rs`: any divergence in the absorbed bytes changes every derived
//! challenge and verification rejects.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::hash::platform_keccak::PlatformKeccak256 as Keccak256;
use digest::Digest;
use math::field::traits::IsPrimeField;
use stark::config::Commitment;

use crate::tables::types::{GoldilocksExtension, GoldilocksField};

use super::airs::NUM_LFM_CHIPS;
use super::hash::HasherKind;
use super::word::LfmWord;

type E = GoldilocksExtension;

pub const LFM_MACHINE_VERSION: u32 = 1;
/// Single preset in v0; becomes the preset ladder tag later.
pub const LFM_PRESET_TAG: u32 = 0;

const LFM_PROGRAM_TAG: &[u8] = b"LAMBDAVM_LFM_PROGRAM_V1";
const LFM_STATEMENT_TAG: &[u8] = b"LAMBDAVM_LFM_STATEMENT_V1";

/// The program digest over the frozen chip order.
///
/// `keccak_rnd_chunks` is bound alongside the roots and heights because it is
/// program shape too: it decides how many `KECCAK_RND` instances the verifier
/// builds. Binding it here is what makes the registry entry — rather than the
/// proof — the authority on that shape.
///
/// `hasher` is bound for the same reason and is the one piece of program shape
/// the roots cannot carry: `LFM_HASH`'s preprocessed width is 11 under every
/// candidate, so the commitments are hasher-independent by construction
/// (`airs.rs`). Without this tag the only thing separating one permutation's
/// machine from another's would be a main-trace width coincidence, which a
/// third candidate could collide with. The tag is what makes two hashers two
/// programs.
pub fn lfm_program_id(
    roots: &[Commitment; NUM_LFM_CHIPS],
    log_heights: &[u8; NUM_LFM_CHIPS],
    keccak_rnd_chunks: usize,
    hasher: HasherKind,
) -> Commitment {
    let mut h = Keccak256::new();
    h.update(LFM_PROGRAM_TAG);
    h.update(LFM_MACHINE_VERSION.to_le_bytes());
    h.update(LFM_PRESET_TAG.to_le_bytes());
    h.update([hasher.as_tag()]);
    for i in 0..NUM_LFM_CHIPS {
        h.update([i as u8]);
        h.update(roots[i]);
        h.update([log_heights[i]]);
    }
    h.update((keccak_rnd_chunks as u64).to_le_bytes());
    h.finalize().into()
}

/// Binds the LFM statement: program identity, machine version, the claimed
/// public words and the FRI terminal degree. Exhaustive by construction —
/// extending the statement means extending this function, in one place.
pub fn absorb_lfm_statement(
    transcript: &mut DefaultTranscript<E>,
    program_id: &Commitment,
    public_words: &[(u32, LfmWord)],
    fri_final_poly_log_degree: u8,
) {
    transcript.append_bytes(LFM_STATEMENT_TAG);
    transcript.append_bytes(program_id);
    transcript.append_bytes(&LFM_MACHINE_VERSION.to_le_bytes());
    transcript.append_bytes(&(public_words.len() as u64).to_le_bytes());
    for (index, word) in public_words {
        transcript.append_bytes(&index.to_le_bytes());
        for lane in word {
            transcript.append_bytes(&GoldilocksField::canonical(lane.value()).to_le_bytes());
        }
    }
    transcript.append_bytes(&[fri_final_poly_log_degree]);
}
