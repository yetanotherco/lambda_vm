//! Host-side arena filler: real proof BYTES → LFM arena words.
//!
//! Input is the guest's wire-format blob, not an in-memory bundle — see
//! [`super::proof_fixture`] for why that fidelity matters. Everything here reads
//! the archived view in place, exactly as the recursion guest does.
//!
//! ## The packing rule this module exists to enforce
//!
//! An arena is a vector of `u32` words, NOT a byte stream. Every field must be
//! packed into its OWN halves; concatenating fields and packing afterwards lets
//! a field whose length is not a multiple of four shift every field behind it.
//! That bug cost real debugging time in R1e and it is silent — the halves count
//! still comes out right, only the values are wrong.

use math::field::element::FieldElement;
use stark::config::Commitment;

use crate::tables::types::GoldilocksField;

use super::keccak_host::pack_stream;
use super::proof_fixture::FixtureArchive;
use super::word::{LfmWord, base_word};

type FE = FieldElement<GoldilocksField>;

/// Halves in one 32-byte commitment.
pub const ROOT_HALVES: usize = 8;

/// The main-trace Merkle roots an epoch's sub-proofs commit to, in air order.
///
/// These are the roots Phase A absorbs and, more importantly for R1f, the roots
/// a Merkle opening is authenticated AGAINST. They come straight off the proof.
///
/// NOTE for the Phase-A leg: the verifier also absorbs each air's PREPROCESSED
/// commitment, and that one does NOT live in the proof — it comes from the AIR
/// set (`air.precomputed_commitment()`), which means replaying Phase A over a
/// real proof needs the epoch's AIRs rebuilt, not just its bytes. Out of scope
/// here and flagged rather than papered over.
pub fn epoch_main_roots(archive: &FixtureArchive, epoch: usize) -> Vec<Commitment> {
    let bundle = &archive.guest_input().bundle;
    assert!(
        epoch < bundle.num_epochs(),
        "epoch {epoch} out of range ({} epochs)",
        bundle.num_epochs()
    );
    let proofs = bundle.epoch_proof(epoch);
    (0..proofs.len())
        .map(|i| *proofs.get(i).lde_trace_main_merkle_root())
        .collect()
}

/// Number of sub-proofs (tables) in an epoch.
pub fn epoch_num_tables(archive: &FixtureArchive, epoch: usize) -> usize {
    archive.guest_input().bundle.epoch_proof(epoch).len()
}

pub fn num_epochs(archive: &FixtureArchive) -> usize {
    archive.guest_input().bundle.num_epochs()
}

/// Bytes epoch `epoch` committed — the statement's `public_output` field.
pub fn epoch_public_output(archive: &FixtureArchive, epoch: usize) -> &[u8] {
    archive.guest_input().bundle.epoch_public_output(epoch)
}

/// Packs commitments into arena halves, each root into its OWN eight halves.
pub fn roots_to_halves(roots: &[Commitment]) -> Vec<FE> {
    let mut out = Vec::with_capacity(roots.len() * ROOT_HALVES);
    for root in roots {
        let halves = pack_stream(root);
        debug_assert_eq!(halves.len(), ROOT_HALVES);
        out.extend(halves);
    }
    out
}

/// Wraps packed halves as arena words.
pub fn halves_to_arena(halves: Vec<FE>) -> Vec<LfmWord> {
    halves.into_iter().map(base_word).collect()
}
