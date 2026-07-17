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
//! (`recursion-min.elf` vs `recursion-blowup2.elf`/`recursion-blowup4.elf`/
//! `recursion-blowup8.elf`, fixed at build time — see [`Preset`]). A consumer
//! must pin that outer ELF too, or a 1-query `min` attestation is
//! indistinguishable from a 128-bit one.

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
/// then verified zero-copy via
/// [`crate::continuation::verify_continuation_archived`] — no owned
/// deserialize of the (large) bundle, same as [`crate::verify_recursion_blob`]
/// for the monolithic proof.
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

    // Only small metadata here; the bundle's proofs stay in the archive (read
    // in place by `verify_continuation_archived`).
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
