//! Statement absorbed into the Fiat-Shamir transcript before Phase A.
//!
//! Streams a canonical, domain-separated, length-prefixed encoding directly
//! into the transcript. The transcript is itself a Keccak256 absorber
//! (`DefaultTranscript`), so a single hash suffices — no external digest
//! needed beyond the ELF.
//!
//! Both call sites (prove, verify) must absorb identical bytes; the bus-balance
//! replay inherits the post-absorb transcript via clone(). Any divergence makes
//! every derived challenge differ and verification reject.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::hash::platform_keccak::PlatformKeccak256 as Keccak256;
use digest::Digest;

use crate::test_utils::E;
use crate::{RuntimePageRange, TableCounts};

/// Domain-separation tag. Bump the suffix (`_V2`, ...) on any encoding change.
/// V4 appends `TableCounts::blake3`, which made the BLAKE3 table conditional.
/// [`CONTINUATION_EPOCH_TAG`] moved to V3 in the same change and for the same
/// reason: the count loop below is SHARED, so a continuation epoch absorbs the
/// new u64 too. Bumping only the monolithic tag would have left continuation
/// proofs from two encodings sharing a transcript prefix.
pub(crate) const DOMAIN_TAG: &[u8] = b"LAMBDAVM_STARK_STATEMENT_V4";

/// Canonical full-ELF identity digest — exactly what [`absorb_statement`] binds
/// into the transcript. The recursion attestation folds the same digest into
/// `program_id` (see the `recursion` module), sharing one pass over the ELF.
pub(crate) fn elf_digest(elf: &[u8]) -> [u8; 32] {
    let mut h = Keccak256::new();
    h.update(elf);
    h.finalize().into()
}

/// Which statement is being bound. Selects the leading domain tag and whether an
/// epoch label is appended, so monolithic and continuation-epoch proofs share one
/// function while each starts with its own tag. `Monolithic` reproduces the
/// original encoding byte-for-byte (no label), so existing proofs are unaffected.
#[derive(Clone, Copy)]
pub(crate) enum StatementKind {
    /// Whole-program (monolithic) proof.
    Monolithic,
    /// One continuation epoch proof, pinned to its position by `epoch_label`.
    ContinuationEpoch { epoch_label: u64 },
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn absorb_statement(
    t: &mut impl IsTranscript<E>,
    kind: StatementKind,
    elf_bytes: &[u8],
    public_output: &[u8],
    table_counts: &TableCounts,
    num_private_input_pages: usize,
    runtime_page_ranges: &[RuntimePageRange],
    fri_final_poly_log_degree: u8,
) {
    absorb_statement_with_digest(
        t,
        kind,
        &elf_digest(elf_bytes),
        public_output,
        table_counts,
        num_private_input_pages,
        runtime_page_ranges,
        fri_final_poly_log_degree,
    )
}

/// [`absorb_statement`] with the ELF digest precomputed. Callers that already
/// hold the digest reuse it instead of a second full-ELF Keccak pass — the
/// recursion attestation path shares one digest between the transcript absorb
/// and the `program_id` fold (a full-ELF hash is expensive in-guest).
#[allow(clippy::too_many_arguments)]
pub(crate) fn absorb_statement_with_digest(
    t: &mut impl IsTranscript<E>,
    kind: StatementKind,
    elf_digest: &[u8; 32],
    public_output: &[u8],
    table_counts: &TableCounts,
    num_private_input_pages: usize,
    runtime_page_ranges: &[RuntimePageRange],
    fri_final_poly_log_degree: u8,
) {
    // Leading domain tag — distinct per statement kind, so a monolithic proof and
    // a continuation epoch proof can never share a transcript prefix.
    let domain_tag = match kind {
        StatementKind::Monolithic => DOMAIN_TAG,
        StatementKind::ContinuationEpoch { .. } => CONTINUATION_EPOCH_TAG,
    };
    t.append_bytes(domain_tag);

    // ELF: fixed 32-byte digest — no length prefix needed.
    t.append_bytes(elf_digest);

    // public_output: variable length → length-prefix to prevent boundary collisions.
    t.append_bytes(&(public_output.len() as u64).to_le_bytes());
    t.append_bytes(public_output);

    // table_counts: fixed-width u64s in declared order. The exhaustive
    // destructure makes any field added to TableCounts a compile error here —
    // that's the signal to extend the loop below and bump DOMAIN_TAG.
    let &TableCounts {
        cpu,
        lt,
        memw,
        memw_aligned,
        load,
        mul,
        dvrm,
        shift,
        branch,
        memw_register,
        eq,
        bytewise,
        store,
        cpu32,
        blake3,
    } = table_counts;
    for count in [
        cpu,
        lt,
        memw,
        memw_aligned,
        load,
        mul,
        dvrm,
        shift,
        branch,
        memw_register,
        eq,
        bytewise,
        store,
        cpu32,
        // 0 or 1, and the one count the verifier cannot derive for itself —
        // binding it is what stops prover and verifier building different AIR
        // sets from the same bytes (see `TableCounts::blake3`).
        blake3,
    ] {
        t.append_bytes(&(count as u64).to_le_bytes());
    }

    t.append_bytes(&(num_private_input_pages as u64).to_le_bytes());

    // fri_final_poly_log_degree: single byte, no endianness concern.
    t.append_bytes(&[fri_final_poly_log_degree]);

    // runtime_page_ranges: count-prefixed; each entry fixed width.
    t.append_bytes(&(runtime_page_ranges.len() as u64).to_le_bytes());
    for r in runtime_page_ranges {
        // Exhaustive destructure: any field added to RuntimePageRange becomes
        // a compile error here.
        let &RuntimePageRange { base, count } = r;
        t.append_bytes(&base.to_le_bytes());
        t.append_bytes(&count.to_le_bytes());
    }

    // Continuation epochs additionally bind their position (replay protection).
    // Monolithic proofs append nothing here, so their encoding is unchanged.
    if let StatementKind::ContinuationEpoch { epoch_label } = kind {
        t.append_bytes(&epoch_label.to_le_bytes());
    }
}

/// Continuation domain tags. Distinct from the monolithic `DOMAIN_TAG` so a
/// monolithic proof and a continuation proof can never share a transcript prefix.
/// `pub(crate)` so the LFM statement replay emits the identical tag instead of
/// duplicating the literal: a second copy would drift silently on a version
/// bump, and the tag existing at all depends on both sides agreeing on it.
pub(crate) const CONTINUATION_EPOCH_TAG: &[u8] = b"LAMBDAVM_CONTINUATION_EPOCH_V3";
const CONTINUATION_GLOBAL_TAG: &[u8] = b"LAMBDAVM_CONTINUATION_GLOBAL_V2";

/// Statement bound into the cross-epoch **global** proof's transcript before
/// Phase A: the ELF (so the global proof is program-bound), the epoch count (so a
/// global proof from a run with a different number of epochs cannot be spliced in),
/// the private-input page count (so the global proof's AIR layout — which touched pages
/// are built non-preprocessed — is canonically pinned, like the monolithic path's
/// `absorb_statement`), `fri_final_poly_log_degree` (which sets the FRI transcript
/// shape, exactly as the monolithic and epoch statements bind it), and the touched
/// page-base set (which GLOBAL_MEMORY tables exist).
/// Prove and verify must call this with identical arguments.
pub(crate) fn absorb_continuation_global_statement(
    t: &mut impl IsTranscript<E>,
    elf_bytes: &[u8],
    num_epochs: usize,
    num_private_input_pages: usize,
    fri_final_poly_log_degree: u8,
    touched_page_bases: &[u64],
) {
    t.append_bytes(CONTINUATION_GLOBAL_TAG);
    t.append_bytes(&elf_digest(elf_bytes));
    t.append_bytes(&(num_epochs as u64).to_le_bytes());
    t.append_bytes(&(num_private_input_pages as u64).to_le_bytes());

    // fri_final_poly_log_degree: single byte, no endianness concern.
    t.append_bytes(&[fri_final_poly_log_degree]);

    // Touched page-base set: count-prefixed, each fixed-width u64. Binds the exact set
    // (and order) of GLOBAL_MEMORY tables the verifier rebuilds, so a tampered list
    // diverges the challenges. Prover and verifier pass the identical canonical
    // (ascending, deduped) list.
    t.append_bytes(&(touched_page_bases.len() as u64).to_le_bytes());
    for base in touched_page_bases {
        t.append_bytes(&base.to_le_bytes());
    }
}
