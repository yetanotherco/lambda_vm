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
///
/// ⛔ **This statement is NOT padded to a field element boundary**, unlike the
/// three WHIR ones — see [`absorb_statement_padding`]. Its transcript is
/// keccak, which absorbs a byte stream and never re-slices it into field
/// elements, so there is no straddling value to prevent; and its bytes are the
/// univariate pipeline's, which the block identity lines pin. Padding here
/// would move a record for nothing.
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

    // The width is for callers that pad from it; this one does not.
    let _ = absorb_table_counts(t, table_counts);

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

/// Bytes per field element, which is the granularity the algebraic transcript
/// slices its absorbed buffer at (`rpx::sponge_leaf_bytes`).
pub(crate) const FELT_BYTES: usize = 8;

/// Zeroes for [`absorb_statement_padding`]. A pad is at most `FELT_BYTES - 1`.
const PAD_ZEROS: [u8; FELT_BYTES - 1] = [0u8; FELT_BYTES - 1];

/// Zero bytes that bring a statement of `len` bytes up to a multiple of
/// [`FELT_BYTES`].
pub(crate) const fn statement_padding(len: usize) -> usize {
    (FELT_BYTES - len % FELT_BYTES) % FELT_BYTES
}

/// Closes a WHIR statement so that whatever is absorbed next starts on a field
/// element boundary, and returns the number of bytes it added.
///
/// # Why a statement is padded at all
///
/// The transcript hashes BYTES, and the algebraic configuration's sponge
/// re-slices everything absorbed since the last squeeze into field elements
/// every [`FELT_BYTES`] bytes (`crypto::hash::rpx::sponge_leaf_bytes`). A value
/// absorbed at an offset that is not a multiple of 8 therefore STRADDLES two
/// field elements, and a field-machine verifier replaying the transcript has to
/// bit-decompose it to reproduce the absorb.
///
/// Every window after the first is already aligned: a squeeze leaves the buffer
/// holding its own 32-byte output, and everything absorbed afterwards — roots
/// 32, extension elements 24, grind nonces 8, final values 24 — is a multiple of
/// 8. The FIRST window is the exception, because it opens with the statement,
/// and the roots that follow it land wherever the statement ended.
///
/// # Why the pad is COMPUTED and not a constant
///
/// A statement's roots do not sit at the end of its fixed prefix: two
/// variable-length fields sit in between (the public output and the per-table
/// heights). Padding the fixed prefix to a multiple of 8 would leave the roots
/// at `(|public_output| + |table_num_vars|) mod 8` — 2 mod 8 at the shape this
/// system runs — so it would align nothing while moving every pinned constant.
/// The length is accumulated beside the absorbs that produce it, and the pad
/// follows from that length.
///
/// # Why it is called even when the pad is empty
///
/// So that "one padding absorb per statement" holds for every shape. The
/// transcript counts an empty `append_bytes` as an absorb, so an absorb count
/// stays a function of the statement's FIELDS rather than of its lengths, and
/// the pinned pair moves by exactly one per statement instead of by a number
/// nobody can predict without the shapes.
pub(crate) fn absorb_statement_padding(
    t: &mut impl IsTranscript<E>,
    kind: &str,
    len: usize,
    shape: &[(&str, usize)],
) -> usize {
    #[cfg(not(feature = "hash-metrics"))]
    let _ = (kind, shape);

    let pad = statement_padding(len);
    t.append_bytes(&PAD_ZEROS[..pad]);

    // A diagnostic, not a gate: it prints every variable length the pad is a
    // function of beside the pad itself, so a run that reports a total number of
    // padding bytes can be checked against the shapes that produced it instead
    // of against a premise nobody measured.
    #[cfg(feature = "hash-metrics")]
    {
        // ⚠ Its own label, not `WHIR`: the bench's per-arm lines already start
        // with that, and a box launcher counting `WHIR` lines would silently
        // pick these up as well.
        let mut line = format!("{:<12} statement {kind}", "WHIR-PAD");
        for (name, value) in shape {
            line.push_str(&format!(" {name}={value}"));
        }
        println!("{line} len={len} pad={pad}");
    }

    pad
}

/// How many per-table counts a statement binds — one `u64` each.
///
/// ★ Read, never written as a literal by a caller. The transcript pin's
/// expected absorb counts are `base + epochs * NUM_TABLE_KINDS`, because the
/// per-table campaign added a count to this list (`TableCounts::blake3`) and a
/// pin carrying the total would be a constant describing one branch while
/// claiming to describe the protocol.
///
/// Two compile errors guard it together, and neither alone is enough: the
/// exhaustive destructure in [`table_count_values`] fails when a field is added
/// to [`TableCounts`], and that function's return type fails when the new field
/// is pushed into the array without bumping this constant.
pub(crate) const NUM_TABLE_KINDS: usize = 15;

/// Every per-table count, in declared order.
///
/// The exhaustive destructure makes any field added to [`TableCounts`] a
/// compile error here — that's the signal to extend the array and bump the
/// domain tag of every statement that absorbs it.
pub(crate) fn table_count_values(table_counts: &TableCounts) -> [u64; NUM_TABLE_KINDS] {
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
    [
        cpu as u64,
        lt as u64,
        memw as u64,
        memw_aligned as u64,
        load as u64,
        mul as u64,
        dvrm as u64,
        shift as u64,
        branch as u64,
        memw_register as u64,
        eq as u64,
        bytewise as u64,
        store as u64,
        cpu32 as u64,
        // 0 or 1, and the one count the verifier cannot derive for itself —
        // binding it is what stops prover and verifier building different AIR
        // sets from the same bytes (see `TableCounts::blake3`).
        blake3 as u64,
    ]
}

/// The table layout, as fixed-width u64s in declared order.
///
/// Returns the number of BYTES it absorbed, so a caller accumulating a
/// statement's length does not have to know — or track — how many counts there
/// are. A caller that does not need the length (the univariate path) ignores it.
#[must_use]
pub(crate) fn absorb_table_counts(
    t: &mut impl IsTranscript<E>,
    table_counts: &TableCounts,
) -> usize {
    let counts = table_count_values(table_counts);
    for count in counts {
        t.append_bytes(&count.to_le_bytes());
    }
    counts.len() * size_of::<u64>()
}

/// Domain tag for the multilinear path. A WHIR proof and a FRI proof must
/// never share a transcript prefix.
pub(crate) const MULTILINEAR_TAG: &[u8] = b"LAMBDAVM_MULTILINEAR_STATEMENT_V1";

/// Continuation domain tags. Distinct from the monolithic `DOMAIN_TAG` so a
/// monolithic proof and a continuation proof can never share a transcript prefix.
/// `pub(crate)` so the LFM statement replay emits the identical tag instead of
/// duplicating the literal: a second copy would drift silently on a version
/// bump, and the tag existing at all depends on both sides agreeing on it.
pub(crate) const CONTINUATION_EPOCH_TAG: &[u8] = b"LAMBDAVM_CONTINUATION_EPOCH_V3";
pub(crate) const CONTINUATION_GLOBAL_TAG: &[u8] = b"LAMBDAVM_CONTINUATION_GLOBAL_V2";

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
