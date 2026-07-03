//! Statement absorbed into the Fiat-Shamir transcript before Phase A.
//!
//! Streams a canonical, domain-separated, length-prefixed encoding directly
//! into the transcript. The transcript is itself a Keccak256 absorber
//! (`DefaultTranscript`), so a single hash suffices — no external digest
//! needed beyond the ELF.
//!
//! All three call sites (prove, verify, bus-balance replay) must absorb
//! identical bytes; any divergence makes every derived challenge differ and
//! verification reject.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use sha3::{Digest, Keccak256};

use crate::test_utils::E;
use crate::{RuntimePageRange, TableCounts};

/// Domain-separation tag. Bump the suffix (`_V3`, ...) on any encoding change.
const DOMAIN_TAG: &[u8] = b"LAMBDAVM_STARK_STATEMENT_V3";

/// Keccak256 of the raw ELF bytes — the program identity bound into the
/// statement and committed by the recursion guest.
pub fn elf_digest(elf: &[u8]) -> [u8; 32] {
    let mut h = Keccak256::new();
    h.update(elf);
    h.finalize().into()
}

/// Which statement is being bound. Selects the leading domain tag and the
/// kind-specific fields, so monolithic and continuation-epoch proofs share one
/// function while each starts with its own tag.
#[derive(Clone, Copy)]
pub(crate) enum StatementKind {
    /// Whole-program (monolithic) proof. Carries the digest of the
    /// [`crate::VmVerifyingKey`] (preprocessed commitments + proof options)
    /// so every challenge depends on which vkey the proof was made against.
    Monolithic { vk_digest: [u8; 32] },
    /// One continuation epoch proof, pinned to its position by `epoch_label`.
    ContinuationEpoch { epoch_label: u64 },
}

pub(crate) fn absorb_statement(
    t: &mut impl IsTranscript<E>,
    kind: StatementKind,
    elf_bytes: &[u8],
    public_output: &[u8],
    table_counts: &TableCounts,
    num_private_input_pages: usize,
    runtime_page_ranges: &[RuntimePageRange],
) {
    // Leading domain tag — distinct per statement kind, so a monolithic proof and
    // a continuation epoch proof can never share a transcript prefix.
    let domain_tag = match kind {
        StatementKind::Monolithic { .. } => DOMAIN_TAG,
        StatementKind::ContinuationEpoch { .. } => CONTINUATION_EPOCH_TAG,
    };
    t.append_bytes(domain_tag);

    // Fixed 32 bytes, no length prefix needed; the per-kind tags keep
    // kind-specific fields unambiguous.
    if let StatementKind::Monolithic { vk_digest } = kind {
        t.append_bytes(&vk_digest);
    }

    // ELF: fixed 32-byte digest — no length prefix needed.
    t.append_bytes(&elf_digest(elf_bytes));

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
    ] {
        t.append_bytes(&(count as u64).to_le_bytes());
    }

    t.append_bytes(&(num_private_input_pages as u64).to_le_bytes());

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
const CONTINUATION_EPOCH_TAG: &[u8] = b"LAMBDAVM_CONTINUATION_EPOCH_V1";
const CONTINUATION_GLOBAL_TAG: &[u8] = b"LAMBDAVM_CONTINUATION_GLOBAL_V1";

/// Statement bound into the cross-epoch **global** proof's transcript before
/// Phase A: the ELF (so the global proof is program-bound), the epoch count (so a
/// global proof from a run with a different number of epochs cannot be spliced in),
/// the private-input page count (so the global proof's AIR layout — which touched pages
/// are built non-preprocessed — is canonically pinned, like the monolithic path's
/// `absorb_statement`), and the touched page-base set (which GLOBAL_MEMORY tables exist).
/// Prove and verify must call this with identical arguments.
pub(crate) fn absorb_continuation_global_statement(
    t: &mut impl IsTranscript<E>,
    elf_bytes: &[u8],
    num_epochs: usize,
    num_private_input_pages: usize,
    touched_page_bases: &[u64],
) {
    t.append_bytes(CONTINUATION_GLOBAL_TAG);
    t.append_bytes(&elf_digest(elf_bytes));
    t.append_bytes(&(num_epochs as u64).to_le_bytes());
    t.append_bytes(&(num_private_input_pages as u64).to_le_bytes());

    // Touched page-base set: count-prefixed, each fixed-width u64. Binds the exact set
    // (and order) of GLOBAL_MEMORY tables the verifier rebuilds, so a tampered list
    // diverges the challenges. Prover and verifier pass the identical canonical
    // (ascending, deduped) list.
    t.append_bytes(&(touched_page_bases.len() as u64).to_le_bytes());
    for base in touched_page_bases {
        t.append_bytes(&base.to_le_bytes());
    }
}
