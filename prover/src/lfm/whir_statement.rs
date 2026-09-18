//! The continuation statements a WHIR verifier binds before any challenge.
//!
//! `absorb_epoch` (`multilinear_continuation.rs:397`) and `absorb_global`
//! (`:560`) write a byte stream into the transcript and then pad it up to a
//! field-element boundary. This module writes the same two streams into a
//! [`WhirTranscript`].
//!
//! # ★ It costs no operation rows, and that is a property, not a shortcut
//!
//! Every field of a continuation statement is a property of the epoch the
//! program is compiled FOR — the L0 design builds one program per epoch from a
//! harvested `WhirRealEpoch` — so the whole statement is a run of PROGRAM
//! CONSTANT bytes. [`WhirTranscript::absorb_const_bytes`] takes such a run at
//! any length and packs it into felts at emit time by `sponge_leaf_bytes`' own
//! rule, so the statement's whole cost is the DISTINCT felt values it interns.
//! A program that read these from an arena would be claiming to verify a shape
//! it was not compiled for, and — because the fields sit at offsets that are not
//! multiples of eight — could not absorb them without a byte shift the
//! transcript refuses by design.
//!
//! # The pad, and the formula that has to come from the code
//!
//! `statement_padding(len) = (8 − len mod 8) mod 8` (`statement.rs:142`) brings
//! the statement up to a felt boundary so that what follows it — the commitment
//! roots, which ARE runtime values — starts aligned.
//!
//! Counting the epoch's fixed part field by field: the 42-byte tag, the 32-byte
//! ELF digest, the 8-byte epoch label, the output's 8-byte length prefix,
//! `statement::NUM_TABLE_KINDS · 8 = 120` bytes of counts, the num-vars' length
//! prefix, three 8-byte config words and the 3-byte grind trailer — **245**,
//! which is `≡ 5 (mod 8)`. So
//!
//! ```text
//!     epoch pad = (3 − |public_output| − |table_num_vars|) mod 8
//! ```
//!
//! The global's is the same derivation: `absorb_global`'s fixed part is the
//! 43-byte tag + 32 + 8 + 8 + 8 + 8 + 24 + 3 = **134**, `≡ 6 (mod 8)`, and the
//! page BASES are eight bytes each so they never move the alignment. So
//!
//! ```text
//!     global pad = (2 − |table_num_vars|) mod 8
//! ```
//!
//! which is 0 at the block's `page_bases = 35, table_num_vars = 50` — the
//! measured value.
//!
//! ⚠ **The campaign's `(2 − epochs − pages) mod 8` is the SAME formula, and I
//! briefly claimed it was not.** `table_num_vars` is one byte per table, and the
//! cross-epoch proof's tables are every epoch's bookend plus the global-memory
//! tables (`multilinear_continuation.rs:556`), so `|table_num_vars| = epochs +
//! pages` — 15 + 35 = 50 on the block — and the two forms agree identically, not
//! coincidentally. The mistake that produced the "correction" was reading the
//! two numbers a `WHIR-PAD` line prints (`page_bases` and `table_num_vars`) as
//! the recorded formula's two variables (epochs and pages). A shape line names
//! what the pad is a function of; it does not name a formula's arguments. Kept
//! here because the identity is the load-bearing part: the recorded form is only
//! right while the global's table set stays one bookend per epoch plus one table
//! per page, and the byte-stream form stays right either way.

use multilinear::whir_chain::ChainConfig;

use crate::TableCounts;
use crate::statement::{statement_padding, table_count_values};

use super::whir_transcript::{CANDIDATES_PER_SQUEEZE, SpongeEntry, WhirTranscript};
use super::word::{LfmWord, base_word};

use crate::tables::types::FE;

/// Bytes one felt occupies in the sponge's stream (`statement::FELT_BYTES`).
const BYTES_PER_FELT: usize = 8;

/// Everything an epoch's statement binds, in `absorb_epoch`'s own order.
pub struct EpochStatement<'a> {
    pub elf_digest: &'a [u8; 32],
    pub epoch_label: u64,
    pub public_output: &'a [u8],
    /// ⚠ The struct, not an array of its values. `table_count_values` is the
    /// host's own exhaustive destructure of it (`statement.rs:243`), so a field
    /// added to `TableCounts` moves the emitter and the host together — which is
    /// the whole reason that function exists.
    pub table_counts: &'a TableCounts,
    pub table_num_vars: &'a [u8],
    pub config: &'a ChainConfig,
}

/// Everything the cross-epoch statement binds, in `absorb_global`'s own order.
pub struct GlobalStatement<'a> {
    pub elf_digest: &'a [u8; 32],
    pub num_epochs: u64,
    pub num_private_input_pages: u64,
    pub page_bases: &'a [u64],
    pub table_num_vars: &'a [u8],
    pub config: &'a ChainConfig,
}

/// The tail both statements share: three config words then the grind trailer.
fn push_config(bytes: &mut Vec<u8>, config: &ChainConfig) {
    let &ChainConfig {
        log_blowup,
        log_folding,
        num_queries,
        grind,
    } = config;
    for value in [log_blowup as u64, log_folding as u64, num_queries as u64] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(&[grind.folding, grind.ood, grind.query]);
}

/// The epoch statement's bytes, BEFORE the pad — `absorb_epoch`'s stream.
///
/// One source for the layout: the emitter writes these, the length form counts
/// them and the constant form groups them, so none of the three can drift from
/// the other two.
pub fn epoch_statement_bytes(statement: &EpochStatement<'_>) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(crate::multilinear_continuation::MULTILINEAR_EPOCH_TAG);
    bytes.extend_from_slice(statement.elf_digest);
    bytes.extend_from_slice(&statement.epoch_label.to_le_bytes());

    bytes.extend_from_slice(&(statement.public_output.len() as u64).to_le_bytes());
    bytes.extend_from_slice(statement.public_output);

    for count in table_count_values(statement.table_counts) {
        bytes.extend_from_slice(&count.to_le_bytes());
    }

    bytes.extend_from_slice(&(statement.table_num_vars.len() as u64).to_le_bytes());
    bytes.extend_from_slice(statement.table_num_vars);

    push_config(&mut bytes, statement.config);
    bytes
}

/// The global statement's bytes, BEFORE the pad — `absorb_global`'s stream.
pub fn global_statement_bytes(statement: &GlobalStatement<'_>) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(crate::multilinear_continuation::MULTILINEAR_GLOBAL_TAG);
    bytes.extend_from_slice(statement.elf_digest);
    bytes.extend_from_slice(&statement.num_epochs.to_le_bytes());
    bytes.extend_from_slice(&statement.num_private_input_pages.to_le_bytes());

    bytes.extend_from_slice(&(statement.page_bases.len() as u64).to_le_bytes());
    for base in statement.page_bases {
        bytes.extend_from_slice(&base.to_le_bytes());
    }

    bytes.extend_from_slice(&(statement.table_num_vars.len() as u64).to_le_bytes());
    bytes.extend_from_slice(statement.table_num_vars);

    push_config(&mut bytes, statement.config);
    bytes
}

/// What a statement costs, and what it leaves the sponge holding.
///
/// ⚠ `operations` is ZERO by construction and is carried anyway, so a census
/// that sums legs does not have to special-case this one — and so a change that
/// made the statement emit an operation would have somewhere to show up.
pub struct StatementCost {
    /// The stream's length before the pad.
    pub len: usize,
    pub pad: usize,
    /// Felts the sponge is left holding: `(len + pad) / 8`.
    pub felts: usize,
    /// The DISTINCT `LFM_CONST` words the run interns, by value.
    pub constants: Vec<LfmWord>,
}

impl StatementCost {
    pub const fn operations(&self) -> usize {
        0
    }

    /// INSTRUCTIONS the statement costs: its interned constants and nothing
    /// else.
    pub fn rows(&self) -> usize {
        self.constants.len()
    }

    /// Where the sponge is left for the first leg after the statement.
    ///
    /// Every absorb invalidates the output buffer
    /// (`default_transcript.rs:205-210`), so nothing is in hand.
    pub fn entry(&self) -> SpongeEntry {
        SpongeEntry {
            buffered_felts: self.felts,
            out_pos: CANDIDATES_PER_SQUEEZE,
        }
    }
}

/// The cost of absorbing `bytes` as a padded constant run.
///
/// The constants are the DISTINCT 8-byte BIG-endian groups the packer reads
/// (`whir_transcript::pack_const_bytes`), keyed the way the builder's pool keys
/// them — so a group that repeats, and the zero group that most statements are
/// full of, costs one row between them all.
///
/// ⚠ **A NON-MUTATION, recorded rather than deleted (instance 52).** Dropping
/// the `resize` below — counting the groups of the UNPADDED stream — passes
/// every gate, and it is a rewrite and not a fudge: the pad is zeros and the
/// packer zero-extends a trailing partial group on the low side, so the last
/// group is the same word either way. The pad changes the FELT COUNT, which
/// `felts` carries and a mutation of it does fail, and it changes no constant at
/// all. Kept explicit so the next reader does not spend a run discovering it.
pub fn statement_cost(bytes: &[u8]) -> StatementCost {
    let len = bytes.len();
    let pad = statement_padding(len);
    let mut padded = bytes.to_vec();
    padded.resize(len + pad, 0u8);

    let mut constants: Vec<LfmWord> = Vec::new();
    for group in padded.chunks(BYTES_PER_FELT) {
        let mut whole = [0u8; BYTES_PER_FELT];
        whole[..group.len()].copy_from_slice(group);
        let word = base_word(FE::from(u64::from_be_bytes(whole)));
        if !constants.contains(&word) {
            constants.push(word);
        }
    }

    StatementCost {
        len,
        pad,
        felts: (len + pad) / BYTES_PER_FELT,
        constants,
    }
}

/// ★ `absorb_epoch`, emitted — the stream and the pad that closes it.
///
/// Takes no builder because it emits no instruction: see the module header. The
/// constants it interns appear when the first RUNTIME value after it flushes the
/// pending run, which is exactly the absorb the pad exists to align — so a test
/// that wants to SEE those rows absorbs the root the verifier absorbs next,
/// rather than a dummy felt that would move the transcript.
pub fn emit_epoch_statement(
    transcript: &mut WhirTranscript,
    statement: &EpochStatement<'_>,
) -> StatementCost {
    absorb_padded(transcript, &epoch_statement_bytes(statement))
}

/// ★ `absorb_global`, emitted.
pub fn emit_global_statement(
    transcript: &mut WhirTranscript,
    statement: &GlobalStatement<'_>,
) -> StatementCost {
    absorb_padded(transcript, &global_statement_bytes(statement))
}

/// Absorbs a statement's bytes and then its pad, as
/// `statement::absorb_statement_padding` does.
fn absorb_padded(transcript: &mut WhirTranscript, bytes: &[u8]) -> StatementCost {
    let cost = statement_cost(bytes);
    transcript.absorb_const_bytes(bytes);
    transcript.absorb_const_bytes(&[0u8; BYTES_PER_FELT - 1][..cost.pad]);
    cost
}
