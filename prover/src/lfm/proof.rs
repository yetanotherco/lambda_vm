//! LFM prove / verify entry points.
//!
//! Prove: execute → traces → statement-bound transcript → the same generic
//! `multi_prove` the RV64 VM uses. Verify: registry-resolve the program's
//! roots (hard error on a miss — no fallback), rebuild the AIR set, replay
//! Phase A on a forked transcript to recover the shared LogUp challenges,
//! compute the expected `LfmPublic` balance from the *claimed* public words
//! (the COMMIT-bus pattern), and run `multi_verify_views`.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::traits::IsPrimeField;
use stark::batched::proof::BatchedMultiProof;
use stark::batched::prover::multi_prove_batched;
use stark::batched::shape::{EpochShape, PinnedPrep};
use stark::batched::verifier::{multi_verify_batched, replay_epoch_transcript};
use stark::config::Commitment;
use stark::proof::options::ProofOptions;
use stark::proof::stark::MultiProof;
use stark::proof::view::MultiProofView;
use stark::prover::{IsStarkProver, Prover, ProvingError};
use stark::residency_mode::ResidencyMode;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::tables::types::{BusId, GoldilocksExtension, GoldilocksField};

use super::airs::{LfmAirs, NUM_LFM_CHIPS, num_lfm_airs};
use super::compiler::LfmProgram;
use super::executor::{LfmExecError, execute};
use super::hash::HasherKind;
use super::registry::{LfmArtifacts, LfmProgramKind, LfmRegistryError, resolve};
use super::statement::absorb_lfm_statement;
use super::trace::{LfmTraces, build_traces_with_hasher};
use super::word::LfmWord;

type F = GoldilocksField;
type E = GoldilocksExtension;

pub struct LfmProof {
    pub proof: MultiProof<F, E, ()>,
    /// The public output the execution produced, in emission order.
    pub public_words: Vec<(u32, LfmWord)>,
}

#[derive(Debug)]
pub enum LfmProveError {
    Exec(LfmExecError),
    Prover(ProvingError),
}

/// Proves under the permutation `artifacts` was built for.
///
/// The hasher comes from the artifacts rather than from a default, because
/// `artifacts.program_id` is derived from it: taking it from anywhere else
/// would let the statement claim one permutation while the AIRs prove another.
pub fn lfm_prove(
    program: &LfmProgram,
    artifacts: &LfmArtifacts,
    arenas: &[Vec<LfmWord>],
    options: &ProofOptions,
) -> Result<LfmProof, LfmProveError> {
    lfm_prove_with_hasher(program, artifacts, arenas, options, artifacts.hasher)
}

/// [`lfm_prove`] with the `LFM_HASH` permutation named explicitly at the call
/// site instead of read off `artifacts`.
///
/// The chips bake their hasher's constants into their constraints, so execution
/// must use the same hasher — this function is the single place that holds them
/// together, passing one `hasher` to the executor, the trace filler and the AIR
/// set. Verification needs the same value ([`verify_against`]).
///
/// # Panics
///
/// If `hasher` is not the one `artifacts` was built for. The two are not
/// independent: `artifacts.program_id` binds the hasher, so a mismatch would
/// produce a proof whose statement names a permutation the trace does not use —
/// unverifiable everywhere, and confusing at exactly the point (registry
/// regeneration) where it would be introduced. The agreement is a caller bug,
/// not a proof outcome, so it is asserted rather than returned.
pub fn lfm_prove_with_hasher(
    program: &LfmProgram,
    artifacts: &LfmArtifacts,
    arenas: &[Vec<LfmWord>],
    options: &ProofOptions,
    hasher: HasherKind,
) -> Result<LfmProof, LfmProveError> {
    assert_eq!(
        artifacts.hasher, hasher,
        "artifacts were built for {:?} but proving was asked for {hasher:?}; \
         program_id binds the hasher, so the two must agree",
        artifacts.hasher
    );
    lfm_prove_with_residency(
        program,
        artifacts,
        arenas,
        options,
        hasher,
        decide_lfm_residency(),
    )
}

/// [`lfm_prove_with_hasher`] with the residency mode supplied instead of read
/// from the environment, so a test can prove the same program under both modes
/// in one process without touching global state.
pub(crate) fn lfm_prove_with_residency(
    program: &LfmProgram,
    artifacts: &LfmArtifacts,
    arenas: &[Vec<LfmWord>],
    options: &ProofOptions,
    hasher: HasherKind,
    residency: ResidencyMode,
) -> Result<LfmProof, LfmProveError> {
    let exec = execute(program, arenas, &hasher).map_err(LfmProveError::Exec)?;
    let mut traces = build_traces_with_hasher(program, &exec.records, hasher);
    let proof = prove_traces_with_hasher(
        artifacts,
        &mut traces,
        &exec.public_words,
        options,
        hasher,
        residency,
    )
    .map_err(LfmProveError::Prover)?;

    Ok(LfmProof {
        proof,
        public_words: exec.public_words,
    })
}

/// Proves an already-built trace set against `artifacts`.
///
/// Split out of [`lfm_prove`] so callers that need to inspect or corrupt a
/// trace between generation and proving (the tamper tests) share this
/// transcript setup instead of reimplementing it. `lfm_prove` itself goes
/// through [`prove_traces_with_hasher`], so this artifacts-hasher form has only
/// test callers.
#[cfg(test)]
pub(crate) fn prove_traces(
    artifacts: &LfmArtifacts,
    traces: &mut LfmTraces,
    public_words: &[(u32, LfmWord)],
    options: &ProofOptions,
) -> Result<MultiProof<F, E, ()>, ProvingError> {
    prove_traces_with_hasher(
        artifacts,
        traces,
        public_words,
        options,
        artifacts.hasher,
        decide_lfm_residency(),
    )
}

/// [`prove_traces`] against an AIR set built for `hasher`. The traces must have
/// been built with the same one.
///
/// Storage mode comes from [`crate::auto_storage::decide_lfm`] and residency
/// mode from [`decide_lfm_residency`] rather than parameters: both are resource
/// decisions, invisible to the proof — spilling changes where a trace lives and
/// recompute changes how long an LDE lives, never a byte the transcript absorbs
/// — so threading them through the prove signature would put knobs with no wire
/// meaning in front of every caller.
pub(crate) fn prove_traces_with_hasher(
    artifacts: &LfmArtifacts,
    traces: &mut LfmTraces,
    public_words: &[(u32, LfmWord)],
    options: &ProofOptions,
    hasher: HasherKind,
    residency: ResidencyMode,
) -> Result<MultiProof<F, E, ()>, ProvingError> {
    let airs = LfmAirs::new_with_hasher(
        &artifacts.roots,
        options,
        artifacts.keccak_rnd_chunks,
        hasher,
    );
    let mut transcript = DefaultTranscript::<E>::new(&[]);
    absorb_lfm_statement(
        &mut transcript,
        &artifacts.program_id,
        public_words,
        options.fri_final_poly_log_degree,
    );
    Prover::multi_prove(
        airs.air_trace_pairs(traces),
        &mut transcript,
        #[cfg(feature = "disk-spill")]
        crate::auto_storage::decide_lfm(),
        residency,
    )
}

/// The LFM wrap's [`ResidencyMode`]: `RecomputeLde` when `LAMBDA_VM_RESIDENCY`
/// is set to `recompute`, else `Retain`.
///
/// An explicit knob for the same reason the storage mode is one: the wrap has
/// no calibrated peak estimate to decide from, and the trade this mode makes —
/// one extra forward NTT per table against dropping the `O(N)` main-LDE
/// retention — is only worth taking when `N` is large. The fixture wrap has one
/// or two `KECCAK_RND` chunks and would just pay the NTT.
///
/// `RecomputeLde` also releases each table's aux columns once its proof exists,
/// so callers that read the traces after proving must leave this unset. Nothing
/// on the wrap path does.
pub(crate) fn decide_lfm_residency() -> ResidencyMode {
    match std::env::var("LAMBDA_VM_RESIDENCY").as_deref() {
        Ok("recompute") => {
            log::info!("lfm residency_mode: RecomputeLde (LAMBDA_VM_RESIDENCY=recompute)");
            ResidencyMode::RecomputeLde
        }
        _ => ResidencyMode::Retain,
    }
}

/// `Err` = registry miss (the hard, no-fallback path). `Ok(false)` = invalid
/// proof or claimed-public mismatch.
pub fn lfm_verify(
    kind: LfmProgramKind,
    proof: &MultiProof<F, E, ()>,
    claimed_public: &[(u32, LfmWord)],
    options: &ProofOptions,
) -> Result<bool, LfmRegistryError> {
    let entry = resolve(kind, options.blowup_factor)?;
    Ok(verify_against(
        &entry.roots,
        &entry.program_id,
        entry.keccak_rnd_chunks,
        proof,
        claimed_public,
        options,
        entry.hasher,
    ))
}

/// [`verify_against`] driven by a whole [`LfmArtifacts`].
///
/// # Why this exists
///
/// `verify_against` takes seven separate pieces of program shape, so every new
/// thing the registry pins would change its signature and every call site with
/// it. Taking the struct means a field added to `LfmArtifacts` reaches the
/// verifier without moving anyone — `prep_root` and `prep_widths` (M-6) were the
/// first, and `prover/tests/d0_king_gate.rs` compiles unchanged across their
/// arrival because of it.
///
/// # ⚠ What it does NOT do yet
///
/// It does not check `prep_root`. The LFM machine proves and verifies a
/// per-table [`MultiProof`], whose openings are authenticated against the
/// per-slot `roots`; the batched preprocessed round is a commitment to the same
/// matrices that only a verifier reading a `BatchedMultiProof` can use
/// (`stark::batched::verifier`). Until the machine switches paths, this is
/// plumbing ahead of its consumer, and saying otherwise would overstate what a
/// passing verification means.
///
/// The shape that consumer will need is [`LfmArtifacts::prep_round_shape`].
pub fn verify_against_artifacts(
    artifacts: &LfmArtifacts,
    proof: &MultiProof<F, E, ()>,
    claimed_public: &[(u32, LfmWord)],
    options: &ProofOptions,
) -> bool {
    verify_against(
        &artifacts.roots,
        &artifacts.program_id,
        artifacts.keccak_rnd_chunks,
        proof,
        claimed_public,
        options,
        artifacts.hasher,
    )
}

/// Verifies against a supplied root vector, program digest, `KECCAK_RND` chunk
/// count and hasher instead of a registry entry.
///
/// The registry lookup in [`lfm_verify`] is the soundness argument's first
/// premise and has no off-switch; this is not one. It exists for callers that
/// legitimately hold freshly built artifacts — the registry regeneration path,
/// and tests covering program shapes that are not (and need not be) registered,
/// such as the per-length keccak256 programs.
///
/// Every piece is supplied for the same reason: it is program shape the
/// verifier must know to build the AIR set, and none of it is ever read off the
/// proof. That includes the hasher — which a caller holding artifacts should
/// pass as `artifacts.hasher`, since the digest it is paired with was derived
/// from exactly that value. There is deliberately no defaulting form: a
/// verifier that silently assumed a permutation would be assuming the one thing
/// the roots cannot tell it.
#[allow(clippy::too_many_arguments)]
pub fn verify_against(
    roots: &[Commitment; NUM_LFM_CHIPS],
    program_id: &Commitment,
    keccak_rnd_chunks: usize,
    proof: &MultiProof<F, E, ()>,
    claimed_public: &[(u32, LfmWord)],
    options: &ProofOptions,
    hasher: HasherKind,
) -> bool {
    // A zero chunk count would drop KECCAK_RND — and its constraints — from
    // the set entirely. Reject the shape rather than build it.
    if keccak_rnd_chunks == 0 {
        return false;
    }
    let view = MultiProofView::Owned(proof);
    if view.len() != num_lfm_airs(keccak_rnd_chunks) {
        return false;
    }

    let airs = LfmAirs::new_with_hasher(roots, options, keccak_rnd_chunks, hasher);
    let refs = airs.air_refs();

    let mut transcript = DefaultTranscript::<E>::new(&[]);
    absorb_lfm_statement(
        &mut transcript,
        program_id,
        claimed_public,
        options.fri_final_poly_log_degree,
    );

    // Fork the statement-bound state and replay Phase A to recover the shared
    // LogUp challenges; the expected balance is the LfmPublic sum recomputed
    // from the claimed words (all other LFM buses balance to zero internally).
    let mut replay = transcript.clone();
    let (z, alpha) = crate::replay_transcript_phase_a_view(&refs, view, &mut replay);
    let Some(expected) = expected_public_balance(claimed_public, &z, &alpha) else {
        return false;
    };

    Verifier::multi_verify_views(&refs, view, &mut transcript, &expected)
}

/// `Σ_i 1/(z − (LfmPublic + index_i·α + Σ_l v_l·α^{2+l}))` — the fingerprint
/// layout matches the `LFM_PUBLIC` sender token `(index, v0..v3)`.
fn expected_public_balance(
    words: &[(u32, LfmWord)],
    z: &FieldElement<E>,
    alpha: &FieldElement<E>,
) -> Option<FieldElement<E>> {
    let bus = FieldElement::<E>::from(BusId::LfmPublic as u64);
    let mut powers = [FieldElement::<E>::zero(); 5];
    powers[0] = *alpha;
    for i in 1..5 {
        powers[i] = &powers[i - 1] * alpha;
    }
    let mut fingerprints: Vec<FieldElement<E>> = words
        .iter()
        .map(|(index, word)| {
            let mut acc = &bus + FieldElement::<E>::from(*index as u64) * &powers[0];
            for (l, lane) in word.iter().enumerate() {
                let v = GoldilocksField::canonical(lane.value());
                acc += FieldElement::<E>::from(v) * &powers[1 + l];
            }
            z - acc
        })
        .collect();
    // A zero fingerprint (a collision with z) is a failure, like COMMIT's.
    FieldElement::inplace_batch_inverse(&mut fingerprints).ok()?;
    Some(
        fingerprints
            .iter()
            .fold(FieldElement::<E>::zero(), |acc, t| acc + t),
    )
}

// ===========================================================================
// The batched path (M-7)
// ===========================================================================
//
// The per-table entry points above stay the default everywhere: `lfm_prove` /
// `lfm_verify` go through `multi_prove` / `multi_verify_views` under keccak, and
// nothing below changes that. These are siblings, not a mode switch, because a
// batched epoch proof is a DIFFERENT wire type (`BatchedMultiProof`) rather than
// the same proof verified differently — an `Option` on the existing signatures
// would have been a lie about what varies.

/// Proves an LFM program as ONE batched epoch.
///
/// `expected_prep` is the registry's preprocessed pin, threaded straight to
/// `multi_prove_batched`. `None` is permitted here and is how a root is
/// generated in the first place; see `stark::batched::shape::PinnedPrep` for why
/// the verifier's disposition of `None` is the opposite.
pub fn lfm_prove_batched(
    program: &LfmProgram,
    artifacts: &LfmArtifacts,
    arenas: &[Vec<LfmWord>],
    options: &ProofOptions,
    expected_prep: Option<PinnedPrep<'_>>,
) -> Result<BatchedLfmProof, LfmProveError> {
    let hasher = artifacts.hasher;
    let exec = execute(program, arenas, &hasher).map_err(LfmProveError::Exec)?;
    let mut traces = build_traces_with_hasher(program, &exec.records, hasher);

    let airs = LfmAirs::new_with_hasher(
        &artifacts.roots,
        options,
        artifacts.keccak_rnd_chunks,
        hasher,
    );
    let mut transcript = DefaultTranscript::<E>::new(&[]);
    absorb_lfm_statement(
        &mut transcript,
        &artifacts.program_id,
        &exec.public_words,
        options.fri_final_poly_log_degree,
    );

    let (proof, _stats) =
        multi_prove_batched::<F, E, (), stark::config::KeccakStarkHash, Prover<F, E, ()>>(
            airs.air_trace_pairs(&mut traces),
            &mut transcript,
            expected_prep,
            #[cfg(feature = "disk-spill")]
            crate::auto_storage::decide_lfm(),
            decide_lfm_residency(),
        )
        .map_err(LfmProveError::Prover)?;

    Ok(BatchedLfmProof {
        proof,
        public_words: exec.public_words,
    })
}

/// An LFM epoch proved through the batched commitment path.
pub struct BatchedLfmProof {
    pub proof: BatchedMultiProof<F, E, ()>,
    pub public_words: Vec<(u32, LfmWord)>,
}

/// [`lfm_verify`] for a batched epoch proof.
///
/// `Err` = registry miss (the hard, no-fallback path, same as `lfm_verify`).
/// `Ok(false)` = invalid proof, claimed-public mismatch, **or a preprocessed
/// round this program's pin does not cover** — see [`verify_against_batched`],
/// which is where that last case is decided and why it is currently the
/// answer for every real LFM epoch.
pub fn lfm_verify_batched(
    kind: LfmProgramKind,
    proof: &BatchedMultiProof<F, E, ()>,
    claimed_public: &[(u32, LfmWord)],
    options: &ProofOptions,
) -> Result<bool, LfmRegistryError> {
    let entry = resolve(kind, options.blowup_factor)?;
    Ok(verify_against_batched(
        &LfmArtifacts {
            roots: entry.roots,
            log_heights: entry.log_heights,
            keccak_rnd_chunks: entry.keccak_rnd_chunks,
            hasher: entry.hasher,
            program_id: entry.program_id,
            prep_root: entry.prep_root,
            prep_widths: entry.prep_widths,
        },
        proof,
        claimed_public,
        options,
    ))
}

/// Verifies a batched epoch against supplied artifacts.
///
/// # The preprocessed pin, and why this refuses today
///
/// `multi_verify_batched` FAILS CLOSED on an unpinned preprocessed round: an
/// epoch whose AIR set has preprocessed tables and no pinned root is rejected,
/// because the only root left to compare against would be the prover's own.
/// The LFM AIR set has fourteen preprocessed matrices — the twelve program
/// groups plus `KECCAK_RC` and `BITWISE` — while `PREP_ROUND_SLOTS` covers
/// twelve, so `pinned_prep_widths` returns `None` and this returns `false`.
///
/// That is deliberate and it is the correct answer. Passing `None` through would
/// mean verifying the preprocessed content against the prover's claim, and
/// passing a twelve-entry slice would describe a different round than
/// `prep_root` commits. Widening the round is M-8's prerequisite; until then a
/// batched LFM epoch is provable but not verifiable, and
/// `a_batched_lfm_epoch_is_refused_for_the_round_coverage_gap` pins exactly that
/// with the cause separated from the symptom.
pub fn verify_against_batched(
    artifacts: &LfmArtifacts,
    proof: &BatchedMultiProof<F, E, ()>,
    claimed_public: &[(u32, LfmWord)],
    options: &ProofOptions,
) -> bool {
    if artifacts.keccak_rnd_chunks == 0 {
        return false;
    }
    let airs = LfmAirs::new_with_hasher(
        &artifacts.roots,
        options,
        artifacts.keccak_rnd_chunks,
        artifacts.hasher,
    );
    let refs = airs.air_refs();
    if refs.len() != num_lfm_airs(artifacts.keccak_rnd_chunks) {
        return false;
    }

    let trace_lengths: Vec<usize> = proof.tables.iter().map(|t| t.trace_length).collect();
    let Ok((shape, _params)) = EpochShape::derive(&refs, &trace_lengths) else {
        return false;
    };
    let Some(widths) = artifacts.pinned_prep_widths(&shape.prep) else {
        return false;
    };
    let pin = PinnedPrep {
        root: &artifacts.prep_root,
        widths: &widths,
    };

    let mut transcript = DefaultTranscript::<E>::new(&[]);
    absorb_lfm_statement(
        &mut transcript,
        &artifacts.program_id,
        claimed_public,
        options.fri_final_poly_log_degree,
    );
    // The batched transcript draws the shared LogUp challenges itself, after the
    // shape histogram and the prep/main roots, so they are recovered by
    // replaying the EPOCH on a fork — not by the per-table Phase A walk, which
    // absorbs per-table roots this path never sends. `LOGUP_NUM_CHALLENGES == 2`
    // and they are `(z, alpha)`, the same pair the per-table path samples.
    let mut replay = transcript.clone();
    let Some((_, _, challenges)) = replay_epoch_transcript(&refs, proof, &mut replay) else {
        return false;
    };
    let [z, alpha] = challenges.lookup.as_slice() else {
        return false;
    };
    let Some(expected) = expected_public_balance(claimed_public, z, alpha) else {
        return false;
    };

    multi_verify_batched::<F, E, (), stark::config::KeccakStarkHash, Verifier<F, E, ()>, _>(
        &refs,
        proof,
        &mut transcript,
        &expected,
        Some(pin),
    )
}
