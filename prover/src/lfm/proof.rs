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
use stark::config::Commitment;
use stark::proof::options::ProofOptions;
use stark::proof::stark::MultiProof;
use stark::proof::view::MultiProofView;
use stark::prover::{IsStarkProver, Prover, ProvingError};
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::tables::types::{BusId, GoldilocksExtension, GoldilocksField};

use super::airs::{LfmAirs, NUM_LFM_CHIPS};
use super::compiler::LfmProgram;
use super::executor::{LfmExecError, execute};
use super::hash::TestPermutation;
use super::registry::{LfmArtifacts, LfmProgramKind, LfmRegistryError, resolve};
use super::statement::absorb_lfm_statement;
use super::trace::{LfmTraces, build_traces};
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

pub fn lfm_prove(
    program: &LfmProgram,
    artifacts: &LfmArtifacts,
    arenas: &[Vec<LfmWord>],
    options: &ProofOptions,
) -> Result<LfmProof, LfmProveError> {
    // The chips bake `TestPermutation`'s constants into their constraints,
    // so execution must use the same hasher (the swap surface swaps both).
    let exec = execute(program, arenas, &TestPermutation).map_err(LfmProveError::Exec)?;
    let mut traces = build_traces(program, &exec.records);
    let proof = prove_traces(artifacts, &mut traces, &exec.public_words, options)
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
/// transcript setup instead of reimplementing it.
pub(crate) fn prove_traces(
    artifacts: &LfmArtifacts,
    traces: &mut LfmTraces,
    public_words: &[(u32, LfmWord)],
    options: &ProofOptions,
) -> Result<MultiProof<F, E, ()>, ProvingError> {
    let airs = LfmAirs::new(&artifacts.roots, options);
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
        Default::default(),
    )
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
        proof,
        claimed_public,
        options,
    ))
}

/// Verifies against a supplied root vector and program digest instead of a
/// registry entry.
///
/// The registry lookup in [`lfm_verify`] is the soundness argument's first
/// premise and has no off-switch; this is not one. It exists for callers that
/// legitimately hold freshly built artifacts — the registry regeneration path,
/// and tests covering program shapes that are not (and need not be) registered,
/// such as the per-length keccak256 programs.
pub fn verify_against(
    roots: &[Commitment; NUM_LFM_CHIPS],
    program_id: &Commitment,
    proof: &MultiProof<F, E, ()>,
    claimed_public: &[(u32, LfmWord)],
    options: &ProofOptions,
) -> bool {
    let view = MultiProofView::Owned(proof);
    if view.len() != NUM_LFM_CHIPS {
        return false;
    }

    let airs = LfmAirs::new(roots, options);
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
