//! Host and guest API for the naive (single-step) recursion pipeline.
//!
//! The recursion verifier guest (`bench_vs/lambda/recursion`) verifies an
//! inner lambda-vm proof in-VM. Its private input (a [`crate::GuestInput`],
//! built host-side by [`encode_guest_input`]) carries the inner program's
//! precomputed DECODE/ELF-data-page roots so the guest skips the in-VM
//! FFT + Merkle rebuild. `verify_with_options` uses supplied roots verbatim —
//! it does NOT bind them to the inner ELF — so on success the guest commits
//! an attestation that folds them into the identity instead:
//! `program_id || inner_public_output` (see [`verify_and_attest_blob`]).
//!
//! Trust model: the attestation is NOT self-enforcing. A consumer of the
//! outer proof MUST recompute the id from the inner ELF it trusts and
//! compare — that is [`check_attestation`]. A substituted root yields an id
//! that differs from the honest recompute ([`expected_program_id`]), so the
//! substitution `verify_with_options` cannot see is rejected here. The
//! recompute is an expensive native FFT + Merkle pass, done once at the top
//! level, never in-VM. `prover/src/tests/recursion_soundness_gap_poc.rs`
//! demonstrates the attack this compare defeats.
//!
//! [`program_id`] deliberately does not fold the `ProofOptions`: the security
//! level is pinned by which verifier guest the outer proof is checked against
//! (`recursion-min.elf` vs `recursion-blowup2.elf`/`recursion-blowup8.elf`,
//! fixed at build time — see [`Preset`]). A consumer must pin that outer ELF
//! too, or a 1-query `min` attestation is indistinguishable from a 128-bit one.

use crypto::hash::platform_keccak::PlatformKeccak256 as Keccak256;
use digest::Digest;
use executor::elf::Elf;

use crate::statement::elf_digest;
use crate::tables::trace_builder::Traces;
use crate::{Commitment, Error, ProofOptions, VmProof};

/// Smallest possible proof options (blowup=2, 1 query). Intentionally
/// insecure — for cheap diagnostics, not soundness. The single source for the
/// `recursion-min` guest build and the host tests that must match it.
pub const MIN_PROOF_OPTIONS: ProofOptions = ProofOptions {
    blowup_factor: 2,
    fri_number_of_queries: 1,
    coset_offset: 3,
    grinding_factor: 1,
    fri_final_poly_log_degree: 7,
};

/// The recursion verifier's build presets. Each fixes the guest's
/// `ProofOptions` at build time (a Cargo feature — private input could
/// otherwise downgrade the security level) and names the ELF artifact
/// `make compile-recursion-elfs` produces. Deriving both from one value keeps
/// a host from proving the inner under options the guest wasn't built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    /// Blowup=2, 1 query ([`MIN_PROOF_OPTIONS`]) — insecure, diagnostics only.
    Min,
    /// Blowup=2, full 128-bit query count (219 queries at 20 grinding bits) —
    /// the realistic base-layer shape: production pipelines prove the base
    /// proof at low blowup (2/4) and reserve high blowup for the final wrap.
    Blowup2,
    /// Blowup=4, 110 queries — the other realistic base-layer point (e.g.
    /// Zisk's compressor layer): 2× the prover LDE of blowup=2 for half the
    /// queries to verify.
    Blowup4,
    /// Blowup=8, multi-query (73 queries) — 128-bit security at final-wrap-style
    /// parameters: more prover work per row, far fewer queries to verify.
    Blowup8,
}

impl Preset {
    /// Every preset, for name→preset lookups (e.g. the blob-dump test's
    /// `RECURSION_DUMP_PRESET`). Keep in sync with the enum.
    pub const ALL: [Preset; 4] = [
        Preset::Min,
        Preset::Blowup2,
        Preset::Blowup4,
        Preset::Blowup8,
    ];

    /// The fixed `ProofOptions` this preset's guest verifies with.
    pub fn options(&self) -> ProofOptions {
        match self {
            Preset::Min => MIN_PROOF_OPTIONS,
            Preset::Blowup2 => crate::GoldilocksCubicProofOptions::with_blowup(2)
                .expect("blowup=2 is always valid"),
            Preset::Blowup4 => crate::GoldilocksCubicProofOptions::with_blowup(4)
                .expect("blowup=4 is always valid"),
            Preset::Blowup8 => crate::GoldilocksCubicProofOptions::with_blowup(8)
                .expect("blowup=8 is always valid"),
        }
    }

    /// Artifact stem under `executor/program_artifacts/recursion/`
    /// (`<stem>.elf`), matching the Makefile's preset rules.
    pub fn artifact_stem(&self) -> &'static str {
        match self {
            Preset::Min => "recursion-min",
            Preset::Blowup2 => "recursion-blowup2",
            Preset::Blowup4 => "recursion-blowup4",
            Preset::Blowup8 => "recursion-blowup8",
        }
    }

    /// Short preset name (the Cargo feature that selects it).
    pub fn name(&self) -> &'static str {
        match self {
            Preset::Min => "min",
            Preset::Blowup2 => "blowup2",
            Preset::Blowup4 => "blowup4",
            Preset::Blowup8 => "blowup8",
        }
    }
}

/// Precompute the DECODE and ELF-data-page preprocessed roots for `elf_bytes`
/// under `opts` — the values the guest receives via private input instead of
/// recomputing in-VM, and the values [`expected_program_id`] recomputes
/// natively. Selection matches the verifier's page construction: every ELF
/// data page (`init_values.is_some()`), keyed by `page_base`; zero-init pages
/// use a compile-time constant and are never listed.
pub fn precomputed_commitments(
    elf_bytes: &[u8],
    opts: &ProofOptions,
) -> Result<(Commitment, Vec<(u64, Commitment)>), Error> {
    let elf = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let decode_commitment = crate::tables::decode::commitment_from_elf(&elf, opts)
        .map_err(|e| Error::Recursion(format!("DECODE commitment from ELF: {e}")))?;
    let page_commitments: Vec<(u64, Commitment)> = Traces::page_configs_from_elf(&elf)
        .iter()
        .filter(|c| c.init_values.is_some())
        .map(|c| {
            (
                c.page_base,
                crate::tables::page::compute_precomputed_commitment(c, opts),
            )
        })
        .collect();
    Ok((decode_commitment, page_commitments))
}

/// Build the guest's private-input blob for `inner_proof` of `inner_elf`:
/// precomputes the roots and rkyv-encodes a [`crate::GuestInput`] (see
/// [`crate::encode_recursion_input`]).
pub fn encode_guest_input(
    inner_proof: &VmProof,
    inner_elf: &[u8],
    opts: &ProofOptions,
) -> Result<Vec<u8>, Error> {
    let (decode_commitment, page_commitments) = precomputed_commitments(inner_elf, opts)?;
    crate::encode_recursion_input(&crate::GuestInput {
        vm_proof: inner_proof.clone(),
        inner_elf: inner_elf.to_vec(),
        decode_commitment,
        page_commitments,
    })
}

/// Build the v2 (`attest-commitment-id`) monolithic guest blob for `inner_proof`
/// of `inner_elf`: precomputes the roots host-side, plus the entry point and
/// full-ELF digest the guest would otherwise derive itself (host-side these are
/// cheap; the point is the guest no longer pays for them).
///
/// `embed_elf` controls whether the ELF bytes ride along in `inner_elf` — they
/// are unused on the v2 verify path, so pass `false` for the smaller
/// production/measurement blob, or `true` to isolate the in-guest cycle saving
/// (skip parse+hash) from the blob-size saving on the same-sized wire.
pub fn encode_guest_input_v2(
    inner_proof: &VmProof,
    inner_elf: &[u8],
    opts: &ProofOptions,
    embed_elf: bool,
) -> Result<Vec<u8>, Error> {
    let (decode_commitment, page_commitments) = precomputed_commitments(inner_elf, opts)?;
    let elf = Elf::load(inner_elf).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    crate::encode_recursion_input_v2(&crate::GuestInputV2 {
        vm_proof: inner_proof.clone(),
        inner_elf: if embed_elf {
            inner_elf.to_vec()
        } else {
            Vec::new()
        },
        elf_digest: elf_digest(inner_elf),
        entry_point: elf.entry_point,
        decode_commitment,
        page_commitments,
    })
}

/// Domain tag for [`program_id`].
const PROGRAM_ID_TAG: &[u8] = b"LAMBDAVM_PROGRAM_ID_V1";

/// [`program_id`] from a precomputed ELF digest and entry point — the guest
/// path, sharing one full-ELF Keccak pass with the verify-side statement
/// absorb (see [`verify_and_attest_blob`]).
pub fn program_id_from_digest(
    elf_digest: &[u8; 32],
    pc_start: u64,
    decode_commitment: &Commitment,
    page_commitments: &[(u64, Commitment)],
) -> [u8; 32] {
    let mut pages = page_commitments.to_vec();
    pages.sort_by_key(|(base, _)| *base);

    let mut h = Keccak256::new();
    h.update(PROGRAM_ID_TAG);
    h.update(elf_digest);
    h.update(pc_start.to_le_bytes());
    h.update(decode_commitment);
    h.update((pages.len() as u64).to_le_bytes());
    for (base, c) in &pages {
        h.update(base.to_le_bytes());
        h.update(c);
    }
    h.finalize().into()
}

/// Canonical program identity: a fold of the full ELF digest, entry point, and
/// the supplied DECODE / ELF-data-page roots (folded in ascending `page_base`
/// order). Folding the roots in makes a supplied-root substitution yield a
/// different id than an honest native recompute — the binding is that compare
/// ([`check_attestation`]).
///
/// `decode_commitment` needs no length prefix: [`Commitment`] is a fixed-size
/// `[u8; COMMITMENT_SIZE]`, so its boundary in the hash input is unambiguous.
/// Pages are self-delimiting too (count-prefixed, each entry a fixed
/// `u64` base + fixed-size `Commitment`).
pub fn program_id(
    elf_bytes: &[u8],
    pc_start: u64,
    decode_commitment: &Commitment,
    page_commitments: &[(u64, Commitment)],
) -> [u8; 32] {
    program_id_from_digest(
        &elf_digest(elf_bytes),
        pc_start,
        decode_commitment,
        page_commitments,
    )
}

/// [`program_id`] with `pc_start` taken from `elf_bytes`' entry point.
pub fn program_id_from_elf(
    elf_bytes: &[u8],
    decode_commitment: &Commitment,
    page_commitments: &[(u64, Commitment)],
) -> Result<[u8; 32], Error> {
    let elf = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    Ok(program_id(
        elf_bytes,
        elf.entry_point,
        decode_commitment,
        page_commitments,
    ))
}

// ============================================================================
// Commitment-derived program identity (v2 — the `attest-commitment-id` scheme)
// ============================================================================
//
// The v1 [`program_id`] folds the full-ELF Keccak digest. Deriving that digest
// in-guest costs a full-ELF parse (`Elf::load`) plus a full-ELF Keccak — and,
// on the supplied-roots path (#844), it buys nothing: the inner proof already
// binds the program's entire executable image through commitments the guest
// verifies against, so the digest is redundant with what the proof enforces.
//
// v2 rebinds identity to exactly those commitments (see [`program_id_v2`]), so
// the guest needs neither the ELF bytes nor the hash. The guest still consumes
// a *supplied* digest for the statement absorb (the inner proof was produced
// with it in the transcript), but that digest is NOT part of the identity —
// folding a prover-supplied digest would bind nothing, since nothing
// cross-checks it against the code or pages.

/// Domain tag for [`program_id_v2`]. Distinct from [`PROGRAM_ID_TAG`] so a v1
/// and a v2 attestation can never be confused for one another.
const PROGRAM_ID_TAG_V2: &[u8] = b"LAMBDAVM_PROGRAM_ID_V2";

/// Commitment-derived program identity (`attest-commitment-id`): a fold of the
/// entry point and the DECODE / page-genesis roots — WITHOUT the full-ELF
/// digest. These three inputs are each verified by the inner proof and so bind
/// the whole semantic program:
///
/// * `decode_commitment` — the DECODE preprocessed root over the program's
///   executable instructions (the code).
/// * `page_commitments` — the per-page genesis (INIT-column) roots over the
///   program's entire initial memory image. `Elf::load` materializes every
///   PT_LOAD segment across its full `p_memsz` (zero-filling `.bss`), so every
///   loadable page — `.bss` included — is a committed init-data page here;
///   there is no un-committed "zero-init ELF page" left to bind.
/// * `entry_point` — bound by the REGISTER preprocessed commitment
///   (`register::register_init_from_entry_point` places it at register word
///   addresses 510/511).
///
/// SEMANTIC CHANGE vs [`program_id`]: identity is now "same semantic program",
/// not "same binary". Two ELFs that differ only in non-loaded metadata — symbol
/// tables, section ordering, debug info, anything outside the PT_LOAD image,
/// the entry point and the executable code — share a `program_id_v2`. A
/// consumer that needs byte-exact binary identity must layer its own hash of
/// the ELF on top; the recursion attestation deliberately does not.
///
/// `decode_commitment` needs no length prefix ([`Commitment`] is fixed-size);
/// pages are count-prefixed and folded in ascending `page_base` order, so the
/// encoding is unambiguous.
pub fn program_id_v2(
    entry_point: u64,
    decode_commitment: &Commitment,
    page_commitments: &[(u64, Commitment)],
) -> [u8; 32] {
    let mut pages = page_commitments.to_vec();
    pages.sort_by_key(|(base, _)| *base);

    let mut h = Keccak256::new();
    h.update(PROGRAM_ID_TAG_V2);
    h.update(entry_point.to_le_bytes());
    h.update(decode_commitment);
    h.update((pages.len() as u64).to_le_bytes());
    for (base, c) in &pages {
        h.update(base.to_le_bytes());
        h.update(c);
    }
    h.finalize().into()
}

/// The honest [`program_id_v2`] for a trusted inner ELF under `opts` (monolithic
/// path): recomputes the DECODE/page roots natively (the expensive FFT + Merkle
/// pass) and folds them with the ELF's entry point. Unlike [`expected_program_id`]
/// it does no full-ELF Keccak. Compute once per (ELF, opts) and reuse.
pub fn expected_program_id_v2(
    trusted_elf_bytes: &[u8],
    opts: &ProofOptions,
) -> Result<[u8; 32], Error> {
    let (decode_commitment, page_commitments) = precomputed_commitments(trusted_elf_bytes, opts)?;
    let elf = Elf::load(trusted_elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    Ok(program_id_v2(
        elf.entry_point,
        &decode_commitment,
        &page_commitments,
    ))
}

/// [`check_attestation`] for the v2 (commitment-derived) identity: split the
/// guest's committed bytes, recompute the id from the ELF the *consumer* trusts
/// via [`expected_program_id_v2`], and compare. Semantics otherwise identical to
/// [`check_attestation`] (see its docs and the [`program_id_v2`] note on what
/// "same identity" now means).
pub fn check_attestation_v2(
    committed: &[u8],
    trusted_elf_bytes: &[u8],
    opts: &ProofOptions,
) -> Result<Option<Vec<u8>>, Error> {
    let Some((id, inner_public_output)) = split_attestation(committed) else {
        return Ok(None);
    };
    if id != expected_program_id_v2(trusted_elf_bytes, opts)? {
        return Ok(None);
    }
    Ok(Some(inner_public_output.to_vec()))
}

/// [`check_attestation_v2`] gated on [`Preset`] instead of a raw [`ProofOptions`],
/// refusing `Preset::Min` for the same reason as [`check_attestation_for_preset`].
pub fn check_attestation_for_preset_v2(
    committed: &[u8],
    trusted_elf_bytes: &[u8],
    preset: Preset,
) -> Result<Option<Vec<u8>>, Error> {
    if preset == Preset::Min {
        return Err(Error::Recursion(
            "Preset::Min (recursion-min.elf) is insecure (blowup=2, 1 query) and must not be \
             used as a production consumer's trust gate; call check_attestation_v2 directly with \
             MIN_PROOF_OPTIONS for diagnostics"
                .to_string(),
        ));
    }
    check_attestation_v2(committed, trusted_elf_bytes, &preset.options())
}

/// Verify the guest's private-input blob ([`encode_guest_input`]) in place and,
/// on success, produce the attestation bytes the recursion guest commits:
/// `program_id(elf, roots) || inner_public_output`. `Ok(None)` means the
/// proof did not verify. This is the guest's whole job in one call; it does a
/// single `Elf::load` and a single full-ELF Keccak (inside
/// [`crate::verify_recursion_blob`]), shared between the statement absorb and
/// the `program_id` fold — no deserialization pass over the inner proof.
///
/// The attestation binds identity only for a consumer that recomputes the id
/// from a trusted ELF ([`check_attestation`]) — see the module docs.
pub fn verify_and_attest_blob(
    blob: &[u8],
    proof_options: &ProofOptions,
) -> Result<Option<Vec<u8>>, Error> {
    let verification = crate::verify_recursion_blob(blob, proof_options)?;
    if !verification.ok {
        return Ok(None);
    }
    let id = program_id_from_digest(
        &verification.elf_digest,
        verification.entry_point,
        &verification.decode_commitment,
        &verification.page_commitments,
    );
    let mut attestation = id.to_vec();
    attestation.extend_from_slice(verification.public_output);
    Ok(Some(attestation))
}

/// [`verify_and_attest_blob`] for the v2 (`attest-commitment-id`) monolithic
/// path: verifies a [`crate::GuestInputV2`] blob in place and, on success,
/// commits `program_id_v2(entry, decode, pages) || inner_public_output`. The
/// guest never parses or hashes the inner ELF (see [`crate::verify_recursion_blob_v2`]
/// and [`program_id_v2`]).
pub fn verify_and_attest_blob_v2(
    blob: &[u8],
    proof_options: &ProofOptions,
) -> Result<Option<Vec<u8>>, Error> {
    let verification = crate::verify_recursion_blob_v2(blob, proof_options)?;
    if !verification.ok {
        return Ok(None);
    }
    let id = program_id_v2(
        verification.entry_point,
        &verification.decode_commitment,
        &verification.page_commitments,
    );
    let mut attestation = id.to_vec();
    attestation.extend_from_slice(verification.public_output);
    Ok(Some(attestation))
}

/// The continuation guest's private-input layout (the `continuation` guest
/// feature). Mirrors [`crate::GuestInput`] with the monolithic proof replaced
/// by the bundle and the PAGE roots replaced by the global-memory genesis
/// roots (see [`crate::continuation::continuation_precomputed_commitments`]).
/// Rkyv-archived on the same magic-prefixed wire format as the monolithic
/// blob ([`crate::encode_recursion_input`]); the guest is feature-pinned to
/// one layout, and a blob of the other kind fails the bytecheck validation.
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct ContinuationGuestInput {
    pub bundle: crate::continuation::ContinuationProof,
    pub inner_elf: Vec<u8>,
    pub decode_commitment: Commitment,
    pub page_commitments: Vec<(u64, Commitment)>,
}

/// Build the continuation guest's private-input blob for `bundle` of
/// `inner_elf`: precomputes the roots and rkyv-encodes a
/// [`ContinuationGuestInput`] behind the standard aligning prefix. Takes the
/// bundle by value (it is large; the encoder is its last consumer).
pub fn encode_continuation_guest_input(
    bundle: crate::continuation::ContinuationProof,
    inner_elf: &[u8],
    opts: &ProofOptions,
) -> Result<Vec<u8>, Error> {
    let (decode_commitment, page_commitments) =
        crate::continuation::continuation_precomputed_commitments(inner_elf, &bundle, opts)?;
    let input = ContinuationGuestInput {
        bundle,
        inner_elf: inner_elf.to_vec(),
        decode_commitment,
        page_commitments,
    };
    let archive = rkyv::to_bytes::<rkyv::rancor::Error>(&input)
        .map_err(|e| Error::Execution(format!("rkyv encode failed: {e}")))?;
    let mut blob = Vec::with_capacity(crate::RECURSION_INPUT_PREFIX_LEN + archive.len());
    blob.extend_from_slice(&crate::RECURSION_INPUT_MAGIC);
    blob.extend_from_slice(&crate::RECURSION_INPUT_VERSION.to_le_bytes());
    blob.extend_from_slice(&[0u8; 4]); // reserved
    debug_assert_eq!(blob.len(), crate::RECURSION_INPUT_PREFIX_LEN);
    blob.extend_from_slice(&archive);
    Ok(blob)
}

/// [`verify_and_attest_blob`]'s logic for a continuation bundle: takes the
/// wire-format blob ([`encode_continuation_guest_input`]) and does the
/// intended `continuation` guest's whole job in one call — verify every
/// epoch + the global memory proof against the supplied roots, then attest
/// `program_id(elf, roots) || public_output`. Uses the same [`program_id`] as
/// the monolithic path over the continuation's root set (DECODE + touched
/// data-page genesis roots), so a consumer re-binds with
/// [`crate::continuation::continuation_precomputed_commitments`] over the
/// bundle it holds — the touched-page set is bundle-dependent, unlike the
/// monolithic path's ELF-only page set. The archive is bytecheck-validated,
/// then verified via [`crate::continuation::verify_continuation_archived`], which
/// reads the batched bundle IN PLACE through `ContinuationProofView::Archived` —
/// no proof (and no per-query batched opening) is deserialized, only the tiny
/// per-epoch metadata.
pub fn verify_continuation_and_attest(
    blob: &[u8],
    proof_options: &ProofOptions,
) -> Result<Option<Vec<u8>>, Error> {
    use rkyv::rancor::Error as RkyvError;

    let archive_bytes = crate::recursion_archive_bytes(blob).ok_or_else(|| {
        Error::Execution(String::from(
            "continuation recursion blob: bad magic or version",
        ))
    })?;
    // Host callers' Vec<u8> carries no alignment guarantee; the guest slice is
    // aligned by construction (same prefix arithmetic as the monolithic blob).
    let mut aligned_fallback = rkyv::util::AlignedVec::<{ crate::RECURSION_INPUT_ALIGN }>::new();
    let archive: &[u8] =
        if (archive_bytes.as_ptr() as usize).is_multiple_of(crate::RECURSION_INPUT_ALIGN) {
            archive_bytes
        } else {
            aligned_fallback.extend_from_slice(archive_bytes);
            &aligned_fallback
        };
    let archived = rkyv::access::<ArchivedContinuationGuestInput, RkyvError>(archive)
        .map_err(|e| Error::Execution(format!("continuation blob validation failed: {e}")))?;

    // Only small metadata deserialized here; the (large) bundle is read in place
    // inside `verify_continuation_archived` via `ContinuationProofView::Archived`
    // (zero-copy — no per-query batched opening is materialized).
    let page_commitments: Vec<(u64, Commitment)> = rkyv::deserialize::<
        Vec<(u64, Commitment)>,
        RkyvError,
    >(&archived.page_commitments)
    .map_err(|e| Error::Execution(format!("rkyv deserialize page commitments failed: {e}")))?;
    let decode_commitment: Commitment = archived.decode_commitment;
    let inner_elf: &[u8] = archived.inner_elf.as_slice();

    let Some((public_output, entry_point)) = crate::continuation::verify_continuation_archived(
        &archived.bundle,
        inner_elf,
        proof_options,
        decode_commitment,
        &page_commitments,
    )?
    else {
        return Ok(None);
    };

    // Avoids a second `Elf::load` (already done by `verify_continuation_archived`).
    let digest = elf_digest(inner_elf);
    let id = program_id_from_digest(&digest, entry_point, &decode_commitment, &page_commitments);
    let mut attestation = id.to_vec();
    attestation.extend_from_slice(&public_output);
    Ok(Some(attestation))
}

/// The v2 (`attest-commitment-id`) continuation guest layout: like
/// [`ContinuationGuestInput`] but the guest never parses or hashes the inner ELF
/// — it receives the entry point and the full-ELF digest directly (the digest
/// only for the per-epoch/global statement absorbs, not the identity). See
/// [`crate::GuestInputV2`] for the `inner_elf`-may-be-empty rationale.
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct ContinuationGuestInputV2 {
    pub bundle: crate::continuation::ContinuationProof,
    /// Unused on the v2 path; may be empty.
    pub inner_elf: Vec<u8>,
    pub elf_digest: [u8; 32],
    pub entry_point: u64,
    pub decode_commitment: Commitment,
    pub page_commitments: Vec<(u64, Commitment)>,
}

/// [`encode_continuation_guest_input`] for the v2 (`attest-commitment-id`) guest:
/// additionally precomputes the entry point and full-ELF digest host-side.
/// `embed_elf` keeps the ELF bytes on the wire (unused by the v2 guest — only for
/// isolating the cycle vs. blob-size saving; see [`encode_guest_input_v2`]).
pub fn encode_continuation_guest_input_v2(
    bundle: crate::continuation::ContinuationProof,
    inner_elf: &[u8],
    opts: &ProofOptions,
    embed_elf: bool,
) -> Result<Vec<u8>, Error> {
    let (decode_commitment, page_commitments) =
        crate::continuation::continuation_precomputed_commitments(inner_elf, &bundle, opts)?;
    let elf = Elf::load(inner_elf).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let input = ContinuationGuestInputV2 {
        bundle,
        inner_elf: if embed_elf {
            inner_elf.to_vec()
        } else {
            Vec::new()
        },
        elf_digest: elf_digest(inner_elf),
        entry_point: elf.entry_point,
        decode_commitment,
        page_commitments,
    };
    crate::encode_recursion_archive(&input, crate::RECURSION_INPUT_VERSION_V2)
}

/// [`verify_continuation_and_attest`] for the v2 (`attest-commitment-id`) guest:
/// verifies the bundle against the supplied roots using the supplied entry
/// point + digest (no in-VM ELF parse or hash), then attests
/// `program_id_v2(entry, decode, pages) || public_output`.
pub fn verify_continuation_and_attest_v2(
    blob: &[u8],
    proof_options: &ProofOptions,
) -> Result<Option<Vec<u8>>, Error> {
    use rkyv::rancor::Error as RkyvError;

    let archive_bytes =
        crate::recursion_archive_bytes_for_version(blob, crate::RECURSION_INPUT_VERSION_V2)
            .ok_or_else(|| {
                Error::Execution(String::from(
                    "continuation recursion blob (v2): bad magic or version",
                ))
            })?;
    let mut aligned_fallback = rkyv::util::AlignedVec::<{ crate::RECURSION_INPUT_ALIGN }>::new();
    let archive: &[u8] =
        if (archive_bytes.as_ptr() as usize).is_multiple_of(crate::RECURSION_INPUT_ALIGN) {
            archive_bytes
        } else {
            aligned_fallback.extend_from_slice(archive_bytes);
            &aligned_fallback
        };
    // sim/27 R9 SIM_FIELD_PAGES: model the future commitment-checked (fext_page)
    // typed-input form — the proof's field-element payloads arrive already bound by
    // a page/leaf commitment, so the guest skips the rkyv bytecheck pass. Guest-side
    // omission (no ecall). SOUND-SHAPE: the guest still reads the SAME blob bytes;
    // a value tamper flows straight to the verifier's own checks and rejects, and a
    // structural tamper faults/mismatches downstream (the model assumes the numeric
    // arrays are commitment-bound, so structural validation moves out of rkyv).
    // MEASUREMENT-ONLY — never prove this build.
    #[cfg(all(target_arch = "riscv64", feature = "sim-field-pages"))]
    let archived: &ArchivedContinuationGuestInputV2 = unsafe { rkyv::access_unchecked(archive) };
    #[cfg(not(all(target_arch = "riscv64", feature = "sim-field-pages")))]
    let archived = rkyv::access::<ArchivedContinuationGuestInputV2, RkyvError>(archive)
        .map_err(|e| Error::Execution(format!("continuation blob (v2) validation failed: {e}")))?;

    let page_commitments: Vec<(u64, Commitment)> = rkyv::deserialize::<
        Vec<(u64, Commitment)>,
        RkyvError,
    >(&archived.page_commitments)
    .map_err(|e| Error::Execution(format!("rkyv deserialize page commitments failed: {e}")))?;
    let decode_commitment: Commitment = archived.decode_commitment;
    let elf_digest: [u8; 32] = archived.elf_digest;
    let entry_point: u64 = archived.entry_point.to_native();

    let Some(public_output) = crate::continuation::verify_continuation_archived_v2(
        &archived.bundle,
        entry_point,
        elf_digest,
        proof_options,
        decode_commitment,
        &page_commitments,
    )?
    else {
        return Ok(None);
    };

    let id = program_id_v2(entry_point, &decode_commitment, &page_commitments);
    let mut attestation = id.to_vec();
    attestation.extend_from_slice(&public_output);
    Ok(Some(attestation))
}

/// The honest v2 continuation identity for a trusted `bundle` of `trusted_elf`:
/// recomputes the DECODE/genesis roots and the entry point natively and folds
/// them via [`program_id_v2`]. The continuation analog of [`expected_program_id_v2`]
/// (the touched-page set is bundle-dependent, so it takes the bundle).
pub fn expected_continuation_program_id_v2(
    trusted_elf_bytes: &[u8],
    bundle: &crate::continuation::ContinuationProof,
    opts: &ProofOptions,
) -> Result<[u8; 32], Error> {
    let (decode_commitment, page_commitments) =
        crate::continuation::continuation_precomputed_commitments(trusted_elf_bytes, bundle, opts)?;
    let elf = Elf::load(trusted_elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    Ok(program_id_v2(
        elf.entry_point,
        &decode_commitment,
        &page_commitments,
    ))
}

/// Split committed attestation bytes into `(program_id, inner_public_output)`.
/// `None` if too short to contain an id.
pub fn split_attestation(committed: &[u8]) -> Option<([u8; 32], &[u8])> {
    if committed.len() < 32 {
        return None;
    }
    let id: [u8; 32] = committed[..32].try_into().ok()?;
    Some((id, &committed[32..]))
}

/// The honest `program_id` for a trusted inner ELF under `opts`: recomputes
/// the DECODE/page roots natively (the expensive FFT + Merkle pass) and folds
/// them. Compute once per (ELF, opts) and reuse across proofs.
pub fn expected_program_id(
    trusted_elf_bytes: &[u8],
    opts: &ProofOptions,
) -> Result<[u8; 32], Error> {
    let (decode_commitment, page_commitments) = precomputed_commitments(trusted_elf_bytes, opts)?;
    program_id_from_elf(trusted_elf_bytes, &decode_commitment, &page_commitments)
}

/// The mandatory consumer-side binding check (see the module docs): split the
/// guest's committed bytes, recompute the id from the ELF the *consumer*
/// trusts, and compare. `Ok(Some(inner_public_output))` on match; `Ok(None)`
/// if the bytes are malformed or attest a different (ELF, roots) identity —
/// i.e. the inner proof was not for `trusted_elf_bytes` as the consumer knows
/// it. The caller must also have verified the outer proof against the pinned
/// `recursion-<preset>.elf` with `opts = preset.options()`.
///
/// This is the low-level primitive: it accepts any `opts`, including
/// [`MIN_PROOF_OPTIONS`], and is meant for diagnostics/tests that need that
/// escape hatch. Production consumers should go through
/// [`check_attestation_for_preset`] instead, which refuses `Preset::Min`.
pub fn check_attestation(
    committed: &[u8],
    trusted_elf_bytes: &[u8],
    opts: &ProofOptions,
) -> Result<Option<Vec<u8>>, Error> {
    let Some((id, inner_public_output)) = split_attestation(committed) else {
        return Ok(None);
    };
    if id != expected_program_id(trusted_elf_bytes, opts)? {
        return Ok(None);
    }
    Ok(Some(inner_public_output.to_vec()))
}

/// [`check_attestation`] gated on [`Preset`] instead of a raw [`ProofOptions`],
/// refusing `Preset::Min`: that preset (blowup=2, 1 query — [`MIN_PROOF_OPTIONS`])
/// is intentionally insecure and exists only for cheap diagnostics. A real
/// consumer pinning its trust to `recursion-min.elf` would accept a 1-query
/// attestation as if it had 128-bit security. Diagnostics/benches that
/// legitimately need `Preset::Min` call [`check_attestation`] directly with
/// [`MIN_PROOF_OPTIONS`].
pub fn check_attestation_for_preset(
    committed: &[u8],
    trusted_elf_bytes: &[u8],
    preset: Preset,
) -> Result<Option<Vec<u8>>, Error> {
    if preset == Preset::Min {
        return Err(Error::Recursion(
            "Preset::Min (recursion-min.elf) is insecure (blowup=2, 1 query) and must not be \
             used as a production consumer's trust gate; call check_attestation directly with \
             MIN_PROOF_OPTIONS for diagnostics"
                .to_string(),
        ));
    }
    check_attestation(committed, trusted_elf_bytes, &preset.options())
}
