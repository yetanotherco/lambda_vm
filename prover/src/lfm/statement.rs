//! LFM program identity and statement binding.
//!
//! `lfm_program_id` binds the instruction column groups (roots + heights),
//! the machine version, the preset and the hash those roots were committed
//! under — it is the digest the registry pins and the consumer's attestation
//! folds. The digest FUNCTION here is keccak regardless of what the roots were
//! committed with: it is a host-side program identity, not a commitment, and
//! `_V2` rides the ecosystem hash migration.
//!
//! The statement absorb seeds the Fiat–Shamir transcript before
//! `multi_prove` / `multi_verify_views`, exactly like the RV64 VM's
//! `statement.rs`: any divergence in the absorbed bytes changes every derived
//! challenge and verification rejects.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::hash::platform_keccak::PlatformKeccak256 as Keccak256;
use digest::Digest;
use math::field::traits::IsPrimeField;
use stark::config::{Commitment, CommitmentHash};

use crate::tables::types::{GoldilocksExtension, GoldilocksField};

use super::airs::{ChipSet, NUM_LFM_CHIPS};
use super::hash::HasherKind;
use super::word::LfmWord;

type E = GoldilocksExtension;

pub const LFM_MACHINE_VERSION: u32 = 1;
/// Single preset in v0; becomes the preset ladder tag later.
pub const LFM_PRESET_TAG: u32 = 0;

const LFM_PROGRAM_TAG: &[u8] = b"LAMBDAVM_LFM_PROGRAM_V1";
/// `pub(super)`: the aggregation layer's emitted verifier replays
/// [`absorb_lfm_statement`] byte for byte and needs the same tag bytes.
pub(super) const LFM_STATEMENT_TAG: &[u8] = b"LAMBDAVM_LFM_STATEMENT_V1";

/// The byte that names a commitment hash inside [`lfm_program_id`].
///
/// Exhaustive on purpose, and that is the whole of what remains of the tripwire
/// `build_artifacts_with_hasher` used to carry: a third commitment hash cannot
/// be added without choosing a tag for it here, and choosing a tag is the act of
/// deciding what program identity says about it. The old guard made that
/// decision unskippable by refusing to compile; this makes it unskippable by
/// having no default.
///
/// Tags are frozen. Changing one re-blesses every `LFM_REGISTRY` entry.
const fn commitment_hash_tag(hash: CommitmentHash) -> u8 {
    match hash {
        CommitmentHash::Keccak256 => 0,
        CommitmentHash::Blake3 => 1,
    }
}

/// The program digest over the frozen chip order.
///
/// `keccak_rnd_chunks` is bound alongside the roots and heights because it is
/// program shape too: it decides how many `KECCAK_RND` instances the verifier
/// builds. Binding it here is what makes the registry entry — rather than the
/// proof — the authority on that shape.
///
/// `hasher` is bound for the same reason and is the one piece of program shape
/// the roots cannot carry: `LFM_HASH`'s preprocessed group is its INSTRUCTION
/// group — addresses, mode selectors and multiplicities — which no candidate
/// changes, so every hasher commits the same width (13 since `MODE_L`) and the
/// commitments are hasher-independent by construction (`airs.rs`). Without this
/// tag the only thing separating one permutation's machine from another's would
/// be a main-trace width coincidence, which a third candidate could collide
/// with. The tag is what makes two hashers two programs.
///
/// ★ **The COMMITMENT hash is bound too, and it is a different axis from
/// `hasher`.** `hasher` names the `LFM_HASH` chip the machine RUNS;
/// [`commitment_hash_tag`] names the hash the `roots` above were BUILT with. The
/// two were separate axes with only the first one named, which is what
/// `build_artifacts_with_hasher`'s guard existed to force a decision about. The
/// decision is here: a build committing under a different hash is a different
/// program identity by name, not merely by value.
///
/// Binding it by value alone would not have been enough. The roots do move when
/// the commitment hash moves, so the digest already changed — but "changed" and
/// "says which" are different properties, and only the second one lets a
/// mismatch be reported as *what it is* rather than as an unrecognised root.
///
/// It is read from the global rather than taken as a parameter because the three
/// commit helpers in `registry.rs` are hard-wired to `stark`'s default aliases,
/// which is exactly what `stark::config::COMMITMENT_HASH` names. Should those
/// helpers ever become generic over `H`, this read moves with them.
pub fn lfm_program_id(
    roots: &[Commitment; NUM_LFM_CHIPS],
    log_heights: &[u8; NUM_LFM_CHIPS],
    keccak_rnd_chunks: usize,
    hasher: HasherKind,
    chip_set: ChipSet,
) -> Commitment {
    let mut h = Keccak256::new();
    h.update(LFM_PROGRAM_TAG);
    h.update(LFM_MACHINE_VERSION.to_le_bytes());
    h.update(LFM_PRESET_TAG.to_le_bytes());
    h.update([hasher.as_tag()]);
    h.update([commitment_hash_tag(stark::config::COMMITMENT_HASH)]);
    // ★ The chip set is program shape and is bound by NAME, for the reason the
    // commitment hash is: the roots of an absent family are still in the array
    // (a hole, like KECCAK_RND's), so nothing else in this digest distinguishes
    // a program that instantiates a family from one that omits it. Without this
    // byte a verifier resolving the wrong mask would build a different AIR set
    // and report an unrecognised shape rather than the mismatch it is.
    h.update([chip_set.as_tag()]);
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
///
/// Generic over the transcript because the statement bind is hash-agnostic: it
/// only absorbs, so it is the same sequence of `append_bytes` calls whichever
/// sponge the proof runs on. Pinning it to `DefaultTranscript` would have made
/// the machine's own transcript a fork of this function rather than a caller of
/// it, and two copies of a statement encoding is exactly the drift the
/// "exhaustive by construction" note above exists to prevent.
pub fn absorb_lfm_statement(
    transcript: &mut impl IsTranscript<E>,
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
