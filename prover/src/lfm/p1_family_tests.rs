//! An LFM proof under the block's proof family (`hash_pin::P1Family`): the
//! machine's trivial program proved and verified on ZisK's Poseidon1 —
//! artifacts committed under `P1StarkHash`, ZisK's transcript, the family's
//! statement tag — and refused across the families, either way and with
//! either family's artifacts. The tag is load-bearing: a family that differs
//! from P1's only by keeping today's `LAMBDAVM_LFM_STATEMENT_V1` tag accepts
//! nothing P1 proves, and P1 accepts nothing it proves.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use stark::config::StarkHash;
use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};

use crate::hash_pin::{LegacyFamily, P1Family, ProofFamily};
use math::field::traits::IsPrimeField;

use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};

use super::p1_commit::P1StarkHash;
use super::programs::trivial_program;
use super::proof::{LfmProof, lfm_prove_under, verify_against_artifacts_under};
use super::registry::{LfmArtifacts, build_artifacts_under};
use super::statement::{LFM_STATEMENT_TAG, absorb_lfm_statement, absorb_lfm_statement_tagged};
use super::word::LfmWord;

fn options() -> ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(2).expect("options")
}

fn arenas() -> Vec<Vec<LfmWord>> {
    vec![
        (0..4u64)
            .map(|i| core::array::from_fn(|j| FE::from(1_000 * (i + 1) + j as u64)))
            .collect(),
    ]
}

/// P1 in everything but the statement tag, which is today's
/// `LAMBDAVM_LFM_STATEMENT_V1`: the mutant the tag must refuse.
struct P1Untagged;

impl ProofFamily for P1Untagged {
    type Stark = <P1Family as ProofFamily>::Stark;
    type Transcript = <P1Family as ProofFamily>::Transcript;
    const STATEMENT_TAG: &'static [u8] = LFM_STATEMENT_TAG;
    const WRAP: super::edsl::WrapHash = P1Family::WRAP;
    const SOCKET: super::hash::HasherKind = P1Family::SOCKET;
    fn transcript(seed: &[u8]) -> Self::Transcript {
        P1Family::transcript(seed)
    }
}

fn artifacts_under<Fam: ProofFamily>(opts: &ProofOptions) -> LfmArtifacts {
    let program = trivial_program();
    let hasher = super::registry::build_artifacts(&program, opts).hasher;
    build_artifacts_under::<Fam::Stark>(&program, opts, hasher)
}

fn prove<Fam: ProofFamily>(artifacts: &LfmArtifacts, opts: &ProofOptions) -> LfmProof {
    lfm_prove_under::<Fam>(&trivial_program(), artifacts, &arenas(), opts).expect("prove")
}

fn verifies<Fam: ProofFamily>(
    artifacts: &LfmArtifacts,
    proof: &LfmProof,
    opts: &ProofOptions,
) -> bool {
    verify_against_artifacts_under::<Fam>(artifacts, &proof.proof, &proof.public_words, opts)
}

/// The artifacts under P1 are another program identity: every committed root
/// is P1's (4-ary, ZisK's leaf), and the id names Poseidon1. The legacy
/// family's are `build_artifacts`' byte for byte.
#[test]
fn p1_artifacts_are_their_own_identity() {
    let opts = options();
    let program = trivial_program();
    let legacy = super::registry::build_artifacts(&program, &opts);
    assert_eq!(
        artifacts_under::<LegacyFamily>(&opts).program_id,
        legacy.program_id,
        "the legacy family's artifacts are today's"
    );
    let p1 = artifacts_under::<P1Family>(&opts);
    assert_ne!(p1.program_id, legacy.program_id);
    for slot in [0usize, 10, 13, 14] {
        assert_ne!(p1.roots[slot], legacy.roots[slot], "slot {slot}");
    }
    assert_eq!(p1.log_heights, legacy.log_heights);
    assert_eq!(
        artifacts_under::<P1Family>(&opts).program_id,
        p1.program_id,
        "deterministic"
    );
    assert_eq!(
        <P1StarkHash as StarkHash>::COMMITMENT_HASH,
        stark::config::CommitmentHash::Poseidon1
    );
}

/// The trivial program proves and verifies under P1, and the families do not
/// cross: a legacy verifier refuses the P1 proof with either family's
/// artifacts, a P1 verifier refuses the legacy proof with either, and neither
/// verifier accepts its own family's proof against the other's artifacts.
#[test]
fn a_p1_lfm_proof_verifies_and_the_families_do_not_cross() {
    let opts = options();
    let p1_art = artifacts_under::<P1Family>(&opts);
    let rpx_art = artifacts_under::<LegacyFamily>(&opts);
    let p1 = prove::<P1Family>(&p1_art, &opts);
    let rpx = prove::<LegacyFamily>(&rpx_art, &opts);
    assert!(verifies::<P1Family>(&p1_art, &p1, &opts), "honest P1 proof");
    assert!(
        verifies::<LegacyFamily>(&rpx_art, &rpx, &opts),
        "honest legacy proof"
    );
    assert_eq!(p1.public_words, rpx.public_words, "one execution");
    for art in [&p1_art, &rpx_art] {
        assert!(
            !verifies::<LegacyFamily>(art, &p1, &opts),
            "P1 proof, legacy verifier"
        );
        assert!(
            !verifies::<P1Family>(art, &rpx, &opts),
            "legacy proof, P1 verifier"
        );
    }
    assert!(
        !verifies::<P1Family>(&rpx_art, &p1, &opts),
        "P1 proof against legacy artifacts"
    );
    // A claimed word the execution did not publish is refused under P1.
    let mut claimed = p1.public_words.clone();
    claimed[0].1[0] = &claimed[0].1[0] + FE::from(1u64);
    assert!(!verify_against_artifacts_under::<P1Family>(
        &p1_art, &p1.proof, &claimed, &opts
    ));
}

/// The family's statement tag is load-bearing: P1 under today's tag accepts
/// no P1 proof, and P1 accepts none of its proofs; each accepts its own.
#[test]
fn the_p1_statement_tag_is_load_bearing() {
    let opts = options();
    let art = artifacts_under::<P1Family>(&opts);
    let tagged = prove::<P1Family>(&art, &opts);
    let untagged = prove::<P1Untagged>(&art, &opts);
    assert!(verifies::<P1Family>(&art, &tagged, &opts));
    assert!(verifies::<P1Untagged>(&art, &untagged, &opts));
    assert!(
        !verifies::<P1Untagged>(&art, &tagged, &opts),
        "a tagged proof under the untagged verifier"
    );
    assert!(
        !verifies::<P1Family>(&art, &untagged, &opts),
        "an untagged proof under the tagged verifier"
    );
}

/// The family's tag REPLACES today's: the statement is the tag as its first
/// call, then today's calls after `LAMBDAVM_LFM_STATEMENT_V1` unchanged. The
/// legacy family's tag is `LAMBDAVM_LFM_STATEMENT_V1` itself, so its statement
/// is today's bytes.
#[test]
fn the_tag_replaces_todays_first_call() {
    type E = GoldilocksExtension;
    let id = [7u8; 32];
    let words: Vec<(u32, LfmWord)> = vec![
        (0, core::array::from_fn(|j| FE::from(j as u64 + 1))),
        (3, core::array::from_fn(|j| FE::from(u64::MAX - j as u64))),
    ];
    let state = |tag: &[u8]| {
        let mut t = P1Family::transcript(&[]);
        absorb_lfm_statement_tagged(&mut t, tag, &id, &words, 8);
        <_ as IsTranscript<E>>::state(&t)
    };
    // Today's statement, call by call, under `first` as its first call.
    let by_hand = |first: &[u8]| {
        let mut t = P1Family::transcript(&[]);
        let mut append = |bytes: &[u8]| <_ as IsTranscript<E>>::append_bytes(&mut t, bytes);
        append(first);
        append(&id);
        append(&super::statement::LFM_MACHINE_VERSION.to_le_bytes());
        append(&(words.len() as u64).to_le_bytes());
        for (index, word) in &words {
            append(&index.to_le_bytes());
            for lane in word {
                append(&GoldilocksField::canonical(lane.value()).to_le_bytes());
            }
        }
        append(&[8]);
        <_ as IsTranscript<E>>::state(&t)
    };
    let mut today = P1Family::transcript(&[]);
    absorb_lfm_statement(&mut today, &id, &words, 8);
    let today = <_ as IsTranscript<E>>::state(&today);
    assert_eq!(state(LegacyFamily::STATEMENT_TAG), today);
    assert_eq!(by_hand(b"LAMBDAVM_LFM_STATEMENT_V1"), today);
    assert_eq!(
        state(P1Family::STATEMENT_TAG),
        by_hand(crate::hash_pin::P1_LFM_STATEMENT_TAG)
    );
    assert_ne!(state(P1Family::STATEMENT_TAG), today);
    // Replaced, not prepended: the P1 tag then today's tag is another state.
    let mut prepended = P1Family::transcript(&[]);
    <_ as IsTranscript<E>>::append_bytes(&mut prepended, P1Family::STATEMENT_TAG);
    absorb_lfm_statement(&mut prepended, &id, &words, 8);
    assert_ne!(
        state(P1Family::STATEMENT_TAG),
        <_ as IsTranscript<E>>::state(&prepended)
    );
    assert!(P1Family::STATEMENT_TAG.starts_with(LFM_STATEMENT_TAG));
}
