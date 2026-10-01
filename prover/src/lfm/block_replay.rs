//! The no-epoch BLOCK's statement and Phase A, replayed in the machine — the
//! shared front every block leaf runs before its own forks.
//!
//! A no-epoch block is ONE monolithic `VmProof` (`crate::block`): the
//! `StatementKind::Monolithic` statement, then every instance's roots in AIR
//! order, one LogUp challenge pair, and a transcript fork per instance. Its
//! recursion splits the instances across leaves (D-NOEPOCH §12), and every leaf
//! needs the SAME post-challenge transcript state to fork from. Each leaf
//! therefore replays the whole front itself — the statement plus Phase A over
//! all the block's roots — rather than receiving a transcript state from
//! another proof: about two hundred permutations per leaf, against ≈ 260 k for
//! its forks, and no new "transcript from a hinted state" primitive.
//!
//! What the leaves must then agree on is checked ABOVE them, by equality:
//! [`BlockFront::state`] is the transcript's digest right after `z, α`
//! (`DefaultTranscript::state()`, which does not advance it), so two leaves
//! publishing the same state forked every instance from the same statement and
//! the same roots.
//!
//! # What is constant and what is arena
//!
//! Every field of the statement except the public output is program SHAPE or
//! verifier-side identity, and is emitted as a constant: the table counts and
//! the page ranges fix how many roots Phase A absorbs, the private-page count
//! fixes the PAGE layout, the FRI byte fixes every fork's schedule, and the ELF
//! digest is the trusted ELF's — the block leaves attest host-side (the STARK
//! wraps' default), so a leaf is a function of the ELF by design. The public
//! output is the one per-proof value: arena halves, published by the leaf so
//! the tree carries the block's claim to its top.

use crate::statement::DOMAIN_TAG;

use super::builder::{Ext, Felt, LfmBuilder};
use super::edsl::WrapDigest;
use super::keccak_host::BYTES_PER_HALF;
use super::statement_replay::{NUM_TABLE_COUNTS, PhaseATable, replay_phase_a};
use super::transcript_replay::TranscriptReplay;

/// The block statement's fields, in `absorb_statement_with_digest`'s order —
/// everything but the ELF digest and the public output's bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockStatementShape {
    /// Length of the public output in bytes; fixes how many arena halves the
    /// leaf reads.
    pub public_output_len: usize,
    /// The absorbed counts, in `TableCounts` declaration order
    /// (`statement::table_count_values`).
    pub table_counts: [u64; NUM_TABLE_COUNTS],
    pub num_private_input_pages: u64,
    pub fri_final_poly_log_degree: u8,
    /// `(base, count)` per runtime page range.
    pub page_ranges: Vec<(u64, u64)>,
}

impl BlockStatementShape {
    /// Total bytes the statement absorbs.
    pub fn byte_len(&self) -> usize {
        DOMAIN_TAG.len()
            + 32
            + 8
            + self.public_output_len
            + 8 * NUM_TABLE_COUNTS
            + 8
            + 1
            + 8
            + 16 * self.page_ranges.len()
    }

    /// Arena halves the public output occupies.
    pub fn out_halves(&self) -> usize {
        self.public_output_len.div_ceil(BYTES_PER_HALF)
    }
}

/// Emits `absorb_statement_with_digest(StatementKind::Monolithic, …)` call for
/// call: the tag, the ELF digest, the output's length and bytes, every count,
/// the private-page count, the FRI byte, and the page ranges. Monolithic
/// statements carry no epoch label and no role byte.
///
/// ⚠ ONE APPEND PER HOST CALL. An algebraic transcript length-prefixes every
/// call, so coalescing two constants that the host appends separately is a
/// different transcript (`transcript_replay::Append`).
pub fn absorb_block_statement(
    t: &mut TranscriptReplay,
    shape: &BlockStatementShape,
    elf_digest: &[u8; 32],
    public_output: &[Felt],
) {
    assert_eq!(
        public_output.len(),
        shape.out_halves(),
        "public_output halves must match the declared length"
    );
    t.append_const_bytes(DOMAIN_TAG);
    t.append_const_bytes(elf_digest);
    t.append_const_bytes(&(shape.public_output_len as u64).to_le_bytes());
    t.append_bytes_misaligned(public_output, shape.public_output_len);
    for count in shape.table_counts {
        t.append_const_bytes(&count.to_le_bytes());
    }
    t.append_const_bytes(&shape.num_private_input_pages.to_le_bytes());
    t.append_const_bytes(&[shape.fri_final_poly_log_degree]);
    t.append_const_bytes(&(shape.page_ranges.len() as u64).to_le_bytes());
    for (base, count) in &shape.page_ranges {
        t.append_const_bytes(&base.to_le_bytes());
        t.append_const_bytes(&count.to_le_bytes());
    }
}

/// The shared front's outputs.
pub struct BlockFront {
    /// The post-challenge transcript every instance forks from
    /// (`epoch::fork_table`).
    pub transcript: TranscriptReplay,
    pub z: Ext,
    pub alpha: Ext,
    /// The transcript's digest right after `z, α` — what a leaf publishes and
    /// what the nodes above compare across leaves.
    pub state: WrapDigest,
}

/// The statement, then Phase A over EVERY instance of the block, then `z, α`
/// and the state digest.
///
/// `tables` is the whole block in AIR order, not a leaf's subset: a leaf that
/// absorbed only its own roots would draw challenges no other leaf, and no host
/// verifier, ever drew.
pub fn replay_block_front(
    b: &mut LfmBuilder,
    shape: &BlockStatementShape,
    elf_digest: &[u8; 32],
    public_output: &[Felt],
    tables: &[PhaseATable],
) -> BlockFront {
    let mut t = TranscriptReplay::new(&[]);
    absorb_block_statement(&mut t, shape, elf_digest, public_output);
    let (z, alpha) = replay_phase_a(&mut t, b, tables);
    // On a clone: `state` observes without advancing on both arms, and the
    // clone keeps that a property of this call rather than of the arm.
    let state = t.clone().state(b);
    BlockFront {
        transcript: t,
        z,
        alpha,
        state,
    }
}
