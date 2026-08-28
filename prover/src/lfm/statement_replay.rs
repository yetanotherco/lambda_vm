//! The continuation-epoch statement and Phase A, replayed in the machine.
//!
//! This is the first leg of a REAL verifier: everything a `multi_verify` does to
//! its transcript before the per-table forks. Two pieces, in order:
//!
//! 1. `absorb_statement(StatementKind::ContinuationEpoch { .. })` — the
//!    canonical, domain-separated statement encoding from `crate::statement`;
//! 2. the Phase-A commitment absorbs from `crate::replay_transcript_phase_a_view`
//!    — per air an optional preprocessed root then the main trace root — followed
//!    by the two shared LogUp challenges `z` and `α`.
//!
//! The target is a continuation EPOCH, not a monolithic proof (see
//! `others/lfm-target-shape.md`), so the tag is `LAMBDAVM_CONTINUATION_EPOCH_V2`
//! and the encoding carries a trailing `epoch_label` the monolithic variant
//! lacks.
//!
//! ## Why this leg is misaligned end to end
//!
//! The tag is 30 bytes, `≡ 2 (mod 4)`, so the ELF digest immediately after it
//! straddles half boundaries; the one-byte `fri_final_poly_log_degree` later
//! moves the cursor again. The whole statement is
//! `215 + public_output_len + 16·page_ranges` bytes, which is `≡ 3 (mod 4)`
//! whenever `public_output_len ≡ 0 (mod 4)` — so **every Phase-A root absorb is
//! spliced at shift 3 too**, at about one `BitDec` and 34 `BALU` rows per half.
//! A single pad byte at the end of the statement encoding would make all of
//! Phase A free; that is a production-encoding change and is not taken here.
//!
//! ## Which fields are program constants
//!
//! Shape-static fields are emitted as constants, not read from an arena, because
//! they DETERMINE the program's shape: the table counts and the page-range list
//! fix how many sub-proofs Phase A absorbs, and `num_private_input_pages` fixes
//! the AIR layout. A program that read them from an arena would be claiming to
//! verify a shape it was not compiled for. Only the genuinely per-proof
//! values — the ELF digest, the public output and the epoch label — come from
//! the arena.

use crate::statement::CONTINUATION_EPOCH_TAG;

use super::builder::{Ext, Felt, LfmBuilder};
use super::keccak_host::BYTES_PER_HALF;
use super::transcript_replay::TranscriptReplay;

/// Counts `TableCounts` absorbs: fourteen split-table families plus the
/// 0-or-1 BLAKE3 presence count. The guest must absorb exactly what the host's
/// `statement::absorb_statement_with_digest` does — one count too few and every
/// challenge downstream diverges, so this tracks that encoding, not a
/// structural property of the machine.
pub const NUM_TABLE_COUNTS: usize = 15;

/// The shape-static half of the statement — emitted as program constants.
#[derive(Debug, Clone)]
pub struct EpochStatementShape {
    /// Length of the public output in bytes. Shape-static: it fixes how many
    /// arena halves the program reads.
    pub public_output_len: usize,
    /// The absorbed counts, in `TableCounts` declaration order: the fourteen
    /// split-table chunk counts, then BLAKE3's 0-or-1.
    pub table_counts: [u64; NUM_TABLE_COUNTS],
    pub num_private_input_pages: u64,
    pub fri_final_poly_log_degree: u8,
    /// `(base, count)` per runtime page range.
    pub page_ranges: Vec<(u64, u64)>,
}

impl EpochStatementShape {
    /// Total bytes the statement absorbs — the emitter's own accounting, so a
    /// test can pin the resulting misalignment instead of trusting prose.
    pub fn byte_len(&self) -> usize {
        CONTINUATION_EPOCH_TAG.len()
            + 32
            + 8
            + self.public_output_len
            + 8 * NUM_TABLE_COUNTS
            + 8
            + 1
            + 8
            + 16 * self.page_ranges.len()
            + 8
    }
}

/// The per-proof half of the statement — arena halves, four bytes each,
/// little-endian, in absorb order.
pub struct EpochStatementVars<'a> {
    /// The 32-byte ELF digest: 8 halves.
    pub elf_digest: &'a [Felt],
    /// `public_output_len / 4` halves.
    pub public_output: &'a [Felt],
    /// The `u64` epoch label, little-endian: `[low32, high32]`.
    pub epoch_label: &'a [Felt],
}

/// Emits `absorb_statement(ContinuationEpoch)` byte for byte — and, since the
/// algebraic arm landed, **call for call**.
///
/// ⚠ The call sequence below tracks
/// `statement::absorb_statement_with_digest`'s `append_bytes` calls one for one.
/// That is a real obligation rather than tidiness: a byte transcript concatenates
/// and cannot see the boundaries, an algebraic one length-prefixes every call and
/// sees nothing else.
///
/// Every multi-byte field in this encoding is LITTLE-endian (`to_le_bytes`),
/// unlike `append_field_element`'s big-endian rendering — so a `u64` carried as
/// `[low32, high32]` halves needs no byte manipulation at all, and the only cost
/// here is the misalignment splice.
pub fn absorb_epoch_statement(
    t: &mut TranscriptReplay,
    shape: &EpochStatementShape,
    vars: &EpochStatementVars,
) {
    assert_eq!(
        vars.public_output.len(),
        shape.public_output_len.div_ceil(BYTES_PER_HALF),
        "public_output halves must match the declared length"
    );
    assert_eq!(vars.elf_digest.len(), 8, "the ELF digest is 32 bytes");
    assert_eq!(vars.epoch_label.len(), 2, "the epoch label is one u64");

    t.append_const_bytes(CONTINUATION_EPOCH_TAG);
    t.append_halves_misaligned(vars.elf_digest);
    t.append_const_bytes(&(shape.public_output_len as u64).to_le_bytes());
    // Byte-granular on purpose. `public_output` is collected one byte per COMMIT
    // operation (`trace_builder`), so an epoch's length is whatever the workload
    // produced — nothing aligns it, and the trailing half must be masked rather
    // than absorbed whole.
    t.append_bytes_misaligned(vars.public_output, shape.public_output_len);

    // ⚠ ONE APPEND PER HOST CALL, not one run. The counts, the page total, the
    // FRI byte and the range list are all shape-static, so a byte transcript
    // cannot tell a single concatenated run from this sequence — the packer
    // chunks consecutive constants together either way, and the emitted halves
    // are identical. An ALGEBRAIC transcript can: it prefixes every
    // `append_bytes` call with that call's LENGTH, so coalescing here would
    // absorb one long field where the host absorbed twenty short ones, and
    // every challenge downstream would diverge. See `transcript_replay::Append`
    // for the general statement of this.
    for count in shape.table_counts {
        t.append_const_bytes(&count.to_le_bytes());
    }
    t.append_const_bytes(&shape.num_private_input_pages.to_le_bytes());
    // A single byte, no endianness concern — and its own call.
    t.append_const_bytes(&[shape.fri_final_poly_log_degree]);
    t.append_const_bytes(&(shape.page_ranges.len() as u64).to_le_bytes());
    for (base, count) in &shape.page_ranges {
        t.append_const_bytes(&base.to_le_bytes());
        t.append_const_bytes(&count.to_le_bytes());
    }

    // Continuation epochs bind their position last (replay protection).
    t.append_halves_misaligned(vars.epoch_label);
}

/// A preprocessed commitment as Phase A absorbs it — and the distinction is
/// which SOURCE the root has, not how it is encoded.
///
/// Production reads every one of these from the AIR and never from the proof
/// (`verifier.rs:1187`), so what the machine must reproduce is the root's
/// provenance: a commitment that is a function of the proof options alone is
/// program text and absorbs as literal bytes; one that is a function of per-proof
/// data (an ELF, a register boundary) is cells, and something else in the program
/// owes their binding (assembly ledger entry 7).
pub enum PhaseAPreprocessed<'a> {
    /// Program text — 32 literal bytes, absorbed with no arithmetic at all.
    Constant(&'a [u8; 32]),
    /// Cells: eight `u32` halves, derived in-machine or read from the arena.
    Cells(&'a [Felt]),
}

/// One sub-proof's Phase-A commitments, as arena halves (8 per 32-byte root).
pub struct PhaseATable<'a> {
    /// Present exactly when the air is preprocessed — the verifier absorbs the
    /// precomputed commitment only then.
    pub preprocessed_root: Option<PhaseAPreprocessed<'a>>,
    pub main_root: &'a [Felt],
}

/// Absorb one root supplied as the configuration's ROOT FELTS — eight `u32`
/// halves on a byte hash, the digest's four felts on an algebraic one.
///
/// ⚠ The byte arm's emission is deliberately unchanged: the felts go straight to
/// `append_halves_misaligned` with no packing, because `statement_replay` is a
/// REGISTRY program and one extra instruction would drift every blessed
/// `program_id`. The algebraic arm packs the four felts into the one digest cell
/// they already are, which is what `RootCells::absorb` does for a root the
/// emitter holds as cells rather than as arena felts.
fn absorb_root_felts(b: &mut LfmBuilder, t: &mut TranscriptReplay, felts: &[Felt]) {
    // Four felts per digest cell: two `u32` halves each on a byte hash, the four
    // Goldilocks felts themselves on an algebraic one.
    let words = super::edsl::digest_words(b);
    assert_eq!(
        felts.len(),
        4 * words as usize,
        "a commitment is one root's felts"
    );
    if words == 1 {
        let cell = b.pack_word([felts[0], felts[1], felts[2], felts[3]]);
        t.append_root_cells(b, &[cell]);
    } else {
        t.append_halves_misaligned(felts);
    }
}

/// Replays Phase A: the commitment absorbs, then the two shared LogUp
/// challenges.
///
/// Mirrors `crate::replay_transcript_phase_a_view` — for each air, the
/// preprocessed commitment when it has one, then the main trace root, and
/// finally `z` and `α` sampled as cubic-extension elements in that order.
///
/// The absorbs use the misaligned path because the statement leaves the cursor
/// at `≡ 3 (mod 4)`; nothing about a 32-byte root is itself misaligned.
pub fn replay_phase_a(
    t: &mut TranscriptReplay,
    b: &mut LfmBuilder,
    tables: &[PhaseATable],
) -> (Ext, Ext) {
    for table in tables {
        match &table.preprocessed_root {
            Some(PhaseAPreprocessed::Constant(bytes)) => t.append_const_bytes(&bytes[..]),
            Some(PhaseAPreprocessed::Cells(prep)) => absorb_root_felts(b, t, prep),
            None => {}
        }
        absorb_root_felts(b, t, table.main_root);
    }
    let z = t.sample_ext(b);
    let alpha = t.sample_ext(b);
    (z, alpha)
}
