//! Milestone B end-to-end: the machine proves a trivial program; valid
//! accepts, tampered variants reject, and the registry drift test pins the
//! program's identity (recompute-and-compare, the static-commitments policy).

use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};

use crate::tables::types::FE;

use super::executor::LfmExecError;
use super::fixture::{self, bump_lane0, fixture_prove};
use super::programs::{fri_toy_program, trivial_program, trivial_program_source};
use super::proof::{LfmProveError, lfm_prove, lfm_verify};
use super::registry::{LfmProgramKind, LfmRegistryError, build_artifacts, resolve};
use super::validator::validate;
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

#[test]
fn trivial_program_is_admissible() {
    let program = trivial_program();
    validate(&program).expect("the registered program must pass admission");
}

#[test]
fn trivial_program_proves_and_verifies() {
    let opts = options();
    let program = trivial_program();
    let artifacts = build_artifacts(&program, &opts);
    let proved = lfm_prove(&program, &artifacts, &arenas(), &opts).expect("prove");
    let ok = lfm_verify(
        LfmProgramKind::TrivialV0,
        &proved.proof,
        &proved.public_words,
        &opts,
    )
    .expect("registry entry exists");
    assert!(ok, "honest machine proof must verify");
}

#[test]
fn tampered_claimed_public_word_rejects() {
    let opts = options();
    let program = trivial_program();
    let artifacts = build_artifacts(&program, &opts);
    let proved = lfm_prove(&program, &artifacts, &arenas(), &opts).expect("prove");

    let mut claimed = proved.public_words.clone();
    claimed[0].1[0] = &claimed[0].1[0] + FE::from(1u64);
    let ok = lfm_verify(LfmProgramKind::TrivialV0, &proved.proof, &claimed, &opts)
        .expect("registry entry exists");
    assert!(!ok, "a tampered claimed public word must reject");
}

#[test]
fn different_arena_values_change_the_public_output_not_the_program() {
    // Same program identity, different hints: proves and verifies against its
    // own (different) public words.
    let opts = options();
    let program = trivial_program();
    let artifacts = build_artifacts(&program, &opts);

    let other: Vec<Vec<LfmWord>> = vec![
        (0..4u64)
            .map(|i| core::array::from_fn(|j| FE::from(7_777 * (i + 1) + j as u64)))
            .collect(),
    ];
    let a = lfm_prove(&program, &artifacts, &arenas(), &opts).expect("prove a");
    let b = lfm_prove(&program, &artifacts, &other, &opts).expect("prove b");
    assert_ne!(a.public_words, b.public_words);
    assert!(
        lfm_verify(LfmProgramKind::TrivialV0, &b.proof, &b.public_words, &opts).expect("entry")
    );
    // Cross-claiming rejects: proof b with proof a's public words.
    assert!(
        !lfm_verify(LfmProgramKind::TrivialV0, &b.proof, &a.public_words, &opts).expect("entry")
    );
}

#[test]
fn registry_miss_is_a_hard_error() {
    let opts = GoldilocksCubicProofOptions::with_blowup(8).expect("options");
    let program = trivial_program();
    let artifacts = build_artifacts(&program, &opts);
    let proved = lfm_prove(&program, &artifacts, &arenas(), &opts).expect("prove");
    let err = lfm_verify(
        LfmProgramKind::TrivialV0,
        &proved.proof,
        &proved.public_words,
        &opts,
    )
    .unwrap_err();
    assert_eq!(
        err,
        LfmRegistryError::UnknownProgram {
            kind: LfmProgramKind::TrivialV0,
            blowup_factor: 8
        },
        "no registry entry ⇒ hard error, never a fallback"
    );
}

/// The registry drift test — the LFM analogue of
/// `static_commitments_tests.rs`. A failure here means the trivial program,
/// a chip layout, the commit pipeline or the digest changed: investigate,
/// never re-bless.
#[test]
fn registry_drift_trivial_v0_blowup2() {
    let opts = options();
    let program = trivial_program();
    let artifacts = build_artifacts(&program, &opts);
    let entry = resolve(LfmProgramKind::TrivialV0, 2).expect("TrivialV0@2 must be registered");
    assert_eq!(entry.roots, artifacts.roots, "group roots drifted");
    assert_eq!(
        entry.log_heights, artifacts.log_heights,
        "group heights drifted"
    );
    assert_eq!(entry.program_id, artifacts.program_id, "program_id drifted");
}

#[test]
fn trivial_program_source_is_deterministic() {
    let a = trivial_program_source();
    let b = trivial_program_source();
    assert_eq!(a.num_addrs, b.num_addrs);
    assert_eq!(a.instrs.len(), b.instrs.len());
}

// ======================= Milestone C: the FRI verifier =======================

fn fri_arenas(proof: &fixture::FriToyProof) -> Vec<Vec<LfmWord>> {
    vec![proof.commitments.clone(), proof.openings.clone()]
}

#[test]
fn fri_toy_program_is_admissible() {
    validate(&fri_toy_program()).expect("the FRI verifier program must pass admission");
}

/// The Milestone-C headline: the machine verifies a structurally real FRI
/// commitment-opening proof (sponge transcript, Merkle-authenticated
/// openings, α-combination, two unnormalized folds, terminal check) and the
/// resulting machine proof verifies against the registry.
#[test]
fn machine_verifies_fixture_fri_proof_end_to_end() {
    let opts = options();
    let program = fri_toy_program();
    let artifacts = build_artifacts(&program, &opts);
    let inner = fixture_prove();
    let proved =
        lfm_prove(&program, &artifacts, &fri_arenas(&inner), &opts).expect("machine accepts");
    // The attested public output is the inner proof's identity: both roots.
    assert_eq!(proved.public_words[0].1, inner.commitments[0]);
    assert_eq!(proved.public_words[1].1, inner.commitments[1]);
    assert!(
        lfm_verify(
            LfmProgramKind::FriToyV0,
            &proved.proof,
            &proved.public_words,
            &opts,
        )
        .expect("FriToyV0 is registered"),
        "the machine proof of FRI verification must verify"
    );
}

/// Every tamper vector must make the verification program *unprovable* (the
/// executor hits the same failed assert the AIR's division constraint makes
/// unsatisfiable).
#[test]
fn machine_rejects_tampered_fri_proofs() {
    let opts = options();
    let program = fri_toy_program();
    let artifacts = build_artifacts(&program, &opts);
    let honest = fixture_prove();

    let expect_reject = |arenas: Vec<Vec<LfmWord>>, what: &str| match lfm_prove(
        &program, &artifacts, &arenas, &opts,
    ) {
        Err(LfmProveError::Exec(LfmExecError::DivByZero { .. })) => {}
        other => panic!(
            "{what}: expected a failed in-machine assert, got {:?}",
            other.map(|_| "accepted")
        ),
    };

    // (a) a tampered opened row value breaks its Merkle path.
    let mut t = fri_arenas(&honest);
    t[1][0] = bump_lane0(&t[1][0]);
    expect_reject(t, "tampered opened row");

    // (b) a tampered sibling digest breaks the walk.
    let mut t = fri_arenas(&honest);
    t[1][2] = bump_lane0(&t[1][2]);
    expect_reject(t, "tampered sibling");

    // (c) a tampered main root diverges the transcript: different queries,
    // openings no longer match.
    let mut t = fri_arenas(&honest);
    t[0][0] = bump_lane0(&t[0][0]);
    expect_reject(t, "tampered main root");

    // (d) a tampered terminal coefficient fails the terminal check.
    let mut t = fri_arenas(&honest);
    t[0][2] = bump_lane0(&t[0][2]);
    expect_reject(t, "tampered terminal coefficient");

    // (e) a tampered L1 opened value fails fold-consistency or its path.
    let mut t = fri_arenas(&honest);
    t[1][12] = bump_lane0(&t[1][12]);
    expect_reject(t, "tampered layer-1 opening");
}

#[test]
fn registry_drift_fri_toy_v0_blowup2() {
    let opts = options();
    let program = fri_toy_program();
    let artifacts = build_artifacts(&program, &opts);
    let entry = resolve(LfmProgramKind::FriToyV0, 2).expect("FriToyV0@2 must be registered");
    assert_eq!(entry.roots, artifacts.roots, "group roots drifted");
    assert_eq!(
        entry.log_heights, artifacts.log_heights,
        "group heights drifted"
    );
    assert_eq!(entry.program_id, artifacts.program_id, "program_id drifted");
}

/// The kill-risk-3 instrument on the first real verification program.
#[test]
fn fri_toy_cell_counts() {
    let program = fri_toy_program();
    let (main, aux) = super::airs::lfm_cell_counts(&program);
    println!(
        "FriToyV0: {} instructions, {} addresses, {} main value cells, {} aux ext elements",
        program.instrs.len(),
        program.num_addrs,
        main,
        aux
    );
    assert!(main > 0 && aux > 0);
}

// ===================== R1b: keccak-f[1600] in the machine =====================

use super::compiler::LfmProgram;
use super::keccak_adapter;
use super::layout::keccak as klayout;
use super::programs::{keccak_chain_program, keccak_chain_program_source};
use super::proof::prove_traces;
use super::registry::LfmArtifacts;
use super::trace::{LfmTraces, build_traces};
use super::validator::LfmViolation;
use crate::lfm::chips::keccak as kchip;
use crate::tables::types::VmTable;
use stark::prover::ProvingError;

/// A keccak state derived from `seed`, in the machine's word form.
fn keccak_state(seed: u64) -> [u64; 25] {
    core::array::from_fn(|i| {
        seed.wrapping_mul(i as u64 + 1)
            .wrapping_add(0x9E37_79B9_7F4A_7C15)
            ^ 0x0123_4567_89AB_CDEF
    })
}

fn keccak_arenas(seed: u64) -> Vec<Vec<LfmWord>> {
    vec![keccak_adapter::state_to_words(&keccak_state(seed)).to_vec()]
}

type KeccakChainProof = stark::proof::stark::MultiProof<
    crate::tables::types::GoldilocksField,
    crate::tables::types::GoldilocksExtension,
    (),
>;

/// Execute + build traces, let the caller corrupt them, then prove.
fn prove_keccak_chain_with_tamper(
    program: &LfmProgram,
    artifacts: &LfmArtifacts,
    seed: u64,
    mutate: impl FnOnce(&mut LfmTraces),
) -> Result<(KeccakChainProof, Vec<(u32, LfmWord)>), ProvingError> {
    let opts = options();
    let exec =
        super::executor::execute(program, &keccak_arenas(seed), &super::hash::TestPermutation)
            .expect("honest execution");
    let mut traces = build_traces(program, &exec.records);
    mutate(&mut traces);
    let proof = prove_traces(artifacts, &mut traces, &exec.public_words, &opts)?;
    Ok((proof, exec.public_words))
}

#[test]
fn keccak_chain_program_is_admissible() {
    validate(&keccak_chain_program()).expect("the keccak-chain program must pass admission");
}

#[test]
fn keccak_chain_source_is_deterministic() {
    let a = keccak_chain_program_source();
    let b = keccak_chain_program_source();
    assert_eq!(a.num_addrs, b.num_addrs);
    assert_eq!(a.instrs.len(), b.instrs.len());
}

/// The R1b headline: the machine proves two *chained* real `keccak-f[1600]`
/// permutations, with the state bound to `LfmMem` words and the permutation
/// itself discharged by the unchanged production `KECCAK_RND` / `KECCAK_RC` /
/// `BITWISE` chips.
#[test]
fn keccak_chain_proves_and_verifies() {
    let opts = options();
    let program = keccak_chain_program();
    let artifacts = build_artifacts(&program, &opts);
    let proved = lfm_prove(&program, &artifacts, &keccak_arenas(7), &opts).expect("machine proves");

    // Host-side reference: the same two permutations, same word convention.
    let once = keccak_adapter::permute(keccak_state(7));
    let twice = keccak_adapter::permute(once);
    let once_words = keccak_adapter::state_to_words(&once);
    let twice_words = keccak_adapter::state_to_words(&twice);
    assert_eq!(proved.public_words[0].1, once_words[0], "first permutation");
    assert_eq!(proved.public_words[1].1, twice_words[0], "second, word 0");
    assert_eq!(proved.public_words[2].1, twice_words[1], "second, word 1");

    assert!(
        lfm_verify(
            LfmProgramKind::KeccakChainV0,
            &proved.proof,
            &proved.public_words,
            &opts,
        )
        .expect("KeccakChainV0 is registered"),
        "the machine proof of two chained keccak permutations must verify"
    );
}

/// Flipping one output byte — i.e. one quarter of one `u32` half — must break
/// the proof. The byte columns feed both the `Keccak` reply token and, through
/// the `Linear` half recomposition, the `LfmMem` word the next instruction
/// reads, so either bus catches it.
#[test]
fn tampered_keccak_output_half_rejects() {
    let opts = options();
    let program = keccak_chain_program();
    let artifacts = build_artifacts(&program, &opts);
    let (proof, public) = prove_keccak_chain_with_tamper(&program, &artifacts, 7, |t| {
        let col = kchip::cols::out_byte(3, 2);
        let old = t.keccak.main_table.get_row(0)[col];
        t.keccak.main_table.set_fe(0, col, old + FE::from(1u64));
    })
    .expect("the adapter has no constraints, so the prover accepts");

    assert!(
        !lfm_verify(LfmProgramKind::KeccakChainV0, &proof, &public, &opts).expect("registered"),
        "a flipped output byte must reject"
    );
}

/// Flipping an input byte likewise rejects: the request token no longer matches
/// the round chip's first receive.
#[test]
fn tampered_keccak_input_half_rejects() {
    let opts = options();
    let program = keccak_chain_program();
    let artifacts = build_artifacts(&program, &opts);
    let (proof, public) = prove_keccak_chain_with_tamper(&program, &artifacts, 7, |t| {
        let col = kchip::cols::state_byte(11, 5);
        let old = t.keccak.main_table.get_row(1)[col];
        t.keccak.main_table.set_fe(1, col, old + FE::from(1u64));
    })
    .expect("locally consistent");

    assert!(
        !lfm_verify(LfmProgramKind::KeccakChainV0, &proof, &public, &opts).expect("registered"),
        "a flipped input byte must reject"
    );
}

/// CLOSES THE R1a HAZARD.
///
/// `keccak_probe::duplicate_tag_output_swap_accepts_demonstrating_hazard`
/// exhibits a live forgery against the raw keccak family: given two rows
/// sharing a tag, swapping their output states leaves the `Keccak` bus
/// balanced, so the verifier accepts two permutations neither of which is
/// genuine. Nothing but the tag binds a request token to its reply.
///
/// Moving the tag into the preprocessed column group closes it, in three legs
/// asserted below:
///   1. the compiled program's keccak rows carry *distinct* tags;
///   2. swapping two rows' output states now REJECTS (it accepted in R1a);
///   3. the prover cannot repair leg 2 by colliding the tags, because they are
///      preprocessed — editing them fails the recommit before a proof exists.
#[test]
fn preprocessed_tags_close_the_output_swap_hazard() {
    let opts = options();
    let program = keccak_chain_program();
    let artifacts = build_artifacts(&program, &opts);
    let group = &program.groups.keccak;
    assert_eq!(group.real_rows, 2, "the chain program has two keccak rows");

    // Leg 1: distinct tags, assigned by the compiler as row ordinals.
    let tag = |row: usize| {
        (
            *group.at(row, klayout::TAG_LO),
            *group.at(row, klayout::TAG_HI),
        )
    };
    assert_ne!(tag(0), tag(1), "keccak tags must be distinct");

    // Leg 2: the R1a forgery, replayed — swap the two rows' 200 output bytes.
    let (proof, public) = prove_keccak_chain_with_tamper(&program, &artifacts, 7, |t| {
        for col in kchip::cols::OUT..kchip::cols::NUM_COLUMNS {
            let a = t.keccak.main_table.get_row(0)[col];
            let b = t.keccak.main_table.get_row(1)[col];
            t.keccak.main_table.set_fe(0, col, b);
            t.keccak.main_table.set_fe(1, col, a);
        }
    })
    .expect("locally consistent");
    assert!(
        !lfm_verify(LfmProgramKind::KeccakChainV0, &proof, &public, &opts).expect("registered"),
        "with distinct tags the swapped outputs must no longer balance"
    );

    // Leg 3: colliding the tags is not available to the prover. Copying row 0's
    // tag over row 1's makes the trace's leading columns disagree with the
    // committed group, and the prover refuses before producing anything.
    let err = prove_keccak_chain_with_tamper(&program, &artifacts, 7, |t| {
        let lo = t.keccak.main_table.get_row(0)[klayout::TAG_LO];
        let hi = t.keccak.main_table.get_row(0)[klayout::TAG_HI];
        t.keccak.main_table.set_fe(1, klayout::TAG_LO, lo);
        t.keccak.main_table.set_fe(1, klayout::TAG_HI, hi);
    })
    .expect_err("preprocessed tags cannot be rewritten");
    assert!(
        matches!(err, ProvingError::PrecomputedCommitmentMismatch),
        "expected a preprocessed recommit failure, got {err:?}"
    );
}

/// The registrar's independent gate on the same obligation: even if a future
/// compiler change stopped assigning distinct tags, admission would catch it.
#[test]
fn duplicate_keccak_tags_fail_admission() {
    let mut program = keccak_chain_program();
    let lo = *program.groups.keccak.at(0, klayout::TAG_LO);
    let hi = *program.groups.keccak.at(0, klayout::TAG_HI);
    program.groups.keccak.set(1, klayout::TAG_LO, lo);
    program.groups.keccak.set(1, klayout::TAG_HI, hi);
    assert_eq!(
        validate(&program),
        Err(LfmViolation::DuplicateKeccakTag { tag: (1, 0) }),
        "duplicate keccak tags must fail admission"
    );
}

/// A keccak lane is a `u64`, but a felt lane carrying a half must be a `u32`.
/// A hinted word above that bound is caught by the executor — on the AIR side
/// no such value exists, since each half is a fixed combination of four
/// BITWISE-constrained bytes.
#[test]
fn keccak_rejects_non_u32_half() {
    let program = keccak_chain_program();
    let mut arenas = keccak_arenas(7);
    arenas[0][0][0] = FE::from(1u64 << 32);
    assert_eq!(
        super::executor::execute(&program, &arenas, &super::hash::TestPermutation).unwrap_err(),
        LfmExecError::NotU32Half { addr: 0, lane: 0 }
    );
}

/// The state is 50 halves in 52 word slots; the two spare slots are pinned to
/// zero as bus tuple constants, so a nonzero one is unprovable.
#[test]
fn keccak_rejects_nonzero_spare_lane() {
    let program = keccak_chain_program();
    let mut arenas = keccak_arenas(7);
    let last = klayout::NUM_WORDS - 1;
    arenas[0][last][2] = FE::from(1u64);
    assert_eq!(
        super::executor::execute(&program, &arenas, &super::hash::TestPermutation).unwrap_err(),
        LfmExecError::KeccakSpareLaneNonZero {
            addr: last as u64,
            lane: 2
        }
    );
}

#[test]
fn registry_drift_keccak_chain_v0_blowup2() {
    let opts = options();
    let program = keccak_chain_program();
    let artifacts = build_artifacts(&program, &opts);
    let entry =
        resolve(LfmProgramKind::KeccakChainV0, 2).expect("KeccakChainV0@2 must be registered");
    assert_eq!(entry.roots, artifacts.roots, "group roots drifted");
    assert_eq!(
        entry.log_heights, artifacts.log_heights,
        "group heights drifted"
    );
    assert_eq!(entry.program_id, artifacts.program_id, "program_id drifted");
}

/// The kill-risk-3 instrument with the keccak family in the set.
#[test]
fn keccak_chain_cell_counts() {
    let program = keccak_chain_program();
    let (main, aux) = super::airs::lfm_cell_counts(&program);
    println!(
        "KeccakChainV0: {} instructions, {} main value cells, {} aux ext elements",
        program.instrs.len(),
        main,
        aux
    );
    assert!(main > 0 && aux > 0);
}

// ==================== R1c: keccak256 over byte streams ====================

use super::keccak_host;
use super::programs::{KECCAK_SPONGE_LEN, keccak_sponge_program};
use super::proof::verify_against;

/// Reference messages. Together they cover: the empty string (padding only),
/// a short message, the exact rate boundary, one byte either side of it, and
/// two lengths whose final `u32` half mixes message bytes with padding.
fn reference_messages() -> Vec<Vec<u8>> {
    let lens = [0usize, 1, 4, 135, 136, 137, KECCAK_SPONGE_LEN, 272];
    lens.iter()
        .map(|&n| {
            (0..n)
                .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
                .collect()
        })
        .collect()
}

fn sponge_arenas(msg: &[u8]) -> Vec<Vec<LfmWord>> {
    let halves = keccak_host::pack_stream(msg);
    keccak_host::assert_high_bytes_zero(&halves, msg.len());
    vec![halves.into_iter().map(super::word::base_word).collect()]
}

/// The 32-byte digest from the two public words: byte `j` is byte `j % 4` of
/// half `j / 4`, and half `h` is lane `h % 4` of word `h / 4`.
fn digest_bytes(public: &[(u32, LfmWord)]) -> [u8; 32] {
    use math::field::traits::IsPrimeField;
    let mut out = [0u8; 32];
    for h in 0..8 {
        let lane = public[h / 4].1[h % 4];
        let half = crate::tables::types::GoldilocksField::canonical(lane.value()) as u32;
        out[4 * h..4 * h + 4].copy_from_slice(&half.to_le_bytes());
    }
    out
}

/// Bit-exactness against the production hasher, execute-only (fast): every
/// reference length must reproduce `PlatformKeccak256` byte for byte.
#[test]
fn keccak256_matches_platform_hasher() {
    for msg in reference_messages() {
        let program = keccak_sponge_program(msg.len());
        let exec = super::executor::execute(
            &program,
            &sponge_arenas(&msg),
            &super::hash::TestPermutation,
        )
        .unwrap_or_else(|e| panic!("len {}: execution failed: {e:?}", msg.len()));
        assert_eq!(
            digest_bytes(&exec.public_words),
            keccak_host::keccak256(&msg),
            "keccak256 mismatch at len {}",
            msg.len()
        );
    }
}

#[test]
fn keccak_sponge_program_is_admissible() {
    for msg in reference_messages() {
        validate(&keccak_sponge_program(msg.len()))
            .unwrap_or_else(|e| panic!("len {} must pass admission: {e:?}", msg.len()));
    }
}

/// The R1c headline: the machine PROVES keccak256 of real byte streams and the
/// proofs verify, with the digest matching `PlatformKeccak256` byte for byte.
///
/// The four lengths cover the shapes that differ structurally: padding-only
/// (empty), a single block whose last half mixes message and padding bytes,
/// a multi-block message crossing the rate boundary, and an exact multiple of
/// the rate — which `pad10*1` grows by a whole extra block.
#[test]
fn keccak_sponge_reference_lengths_prove_and_verify() {
    let opts = options();
    for len in [0usize, 135, KECCAK_SPONGE_LEN, 272] {
        let msg: Vec<u8> = (0..len)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
            .collect();
        let program = keccak_sponge_program(len);
        let artifacts = build_artifacts(&program, &opts);
        let proved = lfm_prove(&program, &artifacts, &sponge_arenas(&msg), &opts)
            .unwrap_or_else(|e| panic!("len {len}: prove failed: {e:?}"));
        assert_eq!(
            digest_bytes(&proved.public_words),
            keccak_host::keccak256(&msg),
            "len {len}: digest must match the production hasher"
        );
        assert!(
            verify_against(
                &artifacts.roots,
                &artifacts.program_id,
                &proved.proof,
                &proved.public_words,
                &opts,
            ),
            "len {len}: the machine proof of keccak256 must verify"
        );
    }
}

/// The registered length, through the full registry-resolving verify path.
#[test]
fn keccak_sponge_proves_and_verifies() {
    let opts = options();
    let msg: Vec<u8> = (0..KECCAK_SPONGE_LEN)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
        .collect();
    let program = keccak_sponge_program(KECCAK_SPONGE_LEN);
    let artifacts = build_artifacts(&program, &opts);
    let proved = lfm_prove(&program, &artifacts, &sponge_arenas(&msg), &opts).expect("prove");
    assert_eq!(
        digest_bytes(&proved.public_words),
        keccak_host::keccak256(&msg)
    );
    assert!(
        lfm_verify(
            LfmProgramKind::KeccakSpongeV0,
            &proved.proof,
            &proved.public_words,
            &opts,
        )
        .expect("KeccakSpongeV0 is registered"),
        "the registered keccak256 program must verify"
    );
}

/// Claiming the honest digest for a message whose stream was altered must
/// reject: the absorbed block differs, so the sponge produces a different
/// digest and the claimed public words no longer match the proof.
#[test]
fn tampered_stream_half_rejects() {
    let opts = options();
    let msg: Vec<u8> = (0..KECCAK_SPONGE_LEN)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
        .collect();
    let program = keccak_sponge_program(KECCAK_SPONGE_LEN);
    let artifacts = build_artifacts(&program, &opts);
    let honest = lfm_prove(&program, &artifacts, &sponge_arenas(&msg), &opts).expect("prove");

    let mut tampered = sponge_arenas(&msg);
    tampered[0][3][0] = &tampered[0][3][0] + FE::from(1u64);
    let forged = lfm_prove(&program, &artifacts, &tampered, &opts).expect("prove");

    assert_ne!(
        forged.public_words, honest.public_words,
        "a changed stream half must change the digest"
    );
    assert!(
        !verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            &forged.proof,
            &honest.public_words,
            &opts,
        ),
        "claiming the honest digest for a tampered stream must reject"
    );
}

/// An absorb row's XOR is pinned by BITWISE lookups, so corrupting the
/// permutation input the family sees — without touching the state read from
/// memory — must reject.
#[test]
fn tampered_absorb_xor_rejects() {
    let opts = options();
    let msg: Vec<u8> = (0..KECCAK_SPONGE_LEN)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
        .collect();
    let program = keccak_sponge_program(KECCAK_SPONGE_LEN);
    let artifacts = build_artifacts(&program, &opts);
    let exec = super::executor::execute(
        &program,
        &sponge_arenas(&msg),
        &super::hash::TestPermutation,
    )
    .expect("honest execution");
    let mut traces = build_traces(&program, &exec.records);
    // Rate byte 5 of the first absorb row: XOR(state, block) no longer holds.
    let col = kchip::cols::PERM_IN + 5;
    let old = traces.keccak.main_table.get_row(0)[col];
    traces
        .keccak
        .main_table
        .set_fe(0, col, old + FE::from(1u64));

    let proof =
        prove_traces(&artifacts, &mut traces, &exec.public_words, &opts).expect("prover accepts");
    assert!(
        !verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            &proof,
            &exec.public_words,
            &opts,
        ),
        "a broken absorb XOR must reject"
    );
}

#[test]
fn registry_drift_keccak_sponge_v0_blowup2() {
    let opts = options();
    let program = keccak_sponge_program(KECCAK_SPONGE_LEN);
    let artifacts = build_artifacts(&program, &opts);
    let entry =
        resolve(LfmProgramKind::KeccakSpongeV0, 2).expect("KeccakSpongeV0@2 must be registered");
    assert_eq!(entry.roots, artifacts.roots, "group roots drifted");
    assert_eq!(
        entry.log_heights, artifacts.log_heights,
        "group heights drifted"
    );
    assert_eq!(entry.program_id, artifacts.program_id, "program_id drifted");
}

#[test]
fn keccak_sponge_cell_counts() {
    let program = keccak_sponge_program(KECCAK_SPONGE_LEN);
    let (main, aux) = super::airs::lfm_cell_counts(&program);
    println!(
        "KeccakSpongeV0 ({} bytes): {} instructions, {} main value cells, {} aux ext elements",
        KECCAK_SPONGE_LEN,
        program.instrs.len(),
        main,
        aux
    );
    assert!(main > 0 && aux > 0);
}

/// Isolates the rate-region pass-through constraint
/// `MODE_PERM · (PERM_IN − STATE) = 0`.
///
/// Absorb rows get `PERM_IN` pinned by the BYTE_ALU[XOR] lookups; permute rows
/// have no lookups, so without this constraint a prover could feed the keccak
/// family a permutation input unrelated to the state it read from memory. Trace
/// tampering alone does not reach that hole — it desynchronises the round chip
/// and the bus catches it first. So this builds the *coordinated* forgery: the
/// last keccak row's `perm_in` is replaced BEFORE trace generation, so the
/// KECCAK_RND rows, the BITWISE multiplicities, the reply token and the output
/// words are all internally consistent with the forged input, and the claimed
/// public words are recomputed to match. Every bus balances. The only thing
/// standing between this and an accepted proof is the constraint.
#[test]
fn permute_row_cannot_substitute_the_permuted_state() {
    let opts = options();
    let program = keccak_chain_program();
    let artifacts = build_artifacts(&program, &opts);
    let mut exec =
        super::executor::execute(&program, &keccak_arenas(7), &super::hash::TestPermutation)
            .expect("honest execution");

    // Forge the second (last) permutation's input, and make everything
    // downstream of it consistent.
    let last = exec.records.keccak.len() - 1;
    let mut forged = exec.records.keccak[last].perm_in;
    forged[0] ^= 1;
    let output = keccak_adapter::permute(forged);
    exec.records.keccak[last].perm_in = forged;
    exec.records.keccak[last].output = output;

    // The chain program publics are once[0], twice[0], twice[1]; the last two
    // come from this row, so claim the values the forged run actually produces.
    let words = keccak_adapter::state_to_words(&output);
    exec.public_words[1].1 = words[0];
    exec.public_words[2].1 = words[1];
    exec.records.public[1] = words[0];
    exec.records.public[2] = words[1];

    let mut traces = build_traces(&program, &exec.records);
    let proof = prove_traces(&artifacts, &mut traces, &exec.public_words, &opts)
        .expect("the prover has no constraint checks, so it accepts");
    assert!(
        !verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            &proof,
            &exec.public_words,
            &opts,
        ),
        "a permute row whose PERM_IN differs from the state it read must reject"
    );
}

// ============ R1d groundwork: DefaultTranscript::sample() replay ============

/// The machine's reversed digest must equal the production transcript's
/// `sample()` byte for byte.
///
/// This is a REAL bit-exactness check against `DefaultTranscript`, not a
/// reimplementation: `sample()` — finalize, reverse the 32 bytes, absorb the
/// reversed bytes, return them — is identical before and after #841, so it can
/// be verified even though this worktree predates that change. The buffered
/// candidate machinery that #841 introduced is what is blocked, not this.
#[test]
fn machine_reversed_digest_matches_default_transcript_sample() {
    use crate::tables::types::GoldilocksExtension;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;

    for len in [0usize, 1, 135, KECCAK_SPONGE_LEN] {
        let msg: Vec<u8> = (0..len)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
            .collect();
        let program = super::programs::keccak_sample_program(len);
        let exec = super::executor::execute(
            &program,
            &sponge_arenas(&msg),
            &super::hash::TestPermutation,
        )
        .unwrap_or_else(|e| panic!("len {len}: execution failed: {e:?}"));

        let mut host = DefaultTranscript::<GoldilocksExtension>::new(&msg);
        let expected = host.sample();
        assert_eq!(
            digest_bytes(&exec.public_words),
            expected,
            "len {len}: reversed digest must match DefaultTranscript::sample()"
        );
    }
}

/// The reversed-digest send must actually reverse: the machine's own
/// non-reversed digest and its reversed digest are byte-reverses of each other.
#[test]
fn reversed_digest_is_the_reverse_of_the_digest() {
    let msg: Vec<u8> = (0..KECCAK_SPONGE_LEN)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
        .collect();
    let plain = super::executor::execute(
        &keccak_sponge_program(msg.len()),
        &sponge_arenas(&msg),
        &super::hash::TestPermutation,
    )
    .expect("exec");
    let reversed = super::executor::execute(
        &super::programs::keccak_sample_program(msg.len()),
        &sponge_arenas(&msg),
        &super::hash::TestPermutation,
    )
    .expect("exec");

    let mut want = digest_bytes(&plain.public_words);
    want.reverse();
    assert_eq!(digest_bytes(&reversed.public_words), want);
}

/// PROVES the `sample()` replay, which the execute-only test above does NOT.
///
/// This distinction bit me: `execute` writes the reversed words from the host
/// mirror (`keccak_adapter::reversed_digest_words`), so an execute-only test
/// passes no matter what the CHIP's reversed-coefficient `Linear` says. The two
/// have to agree, and only a proof checks that — if the bus send recomposes the
/// bytes in any other order, the words it sends differ from the ones the
/// executor wrote to memory and the `LfmMem` bus stops balancing. Neutralising
/// the reversal in the chip leaves the execute-only test green and makes THIS
/// one fail, which is how it should be.
#[test]
fn machine_proves_the_sample_replay() {
    use crate::tables::types::GoldilocksExtension;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;

    let opts = options();
    for len in [0usize, 135, KECCAK_SPONGE_LEN] {
        let msg: Vec<u8> = (0..len)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
            .collect();
        let program = super::programs::keccak_sample_program(len);
        let artifacts = build_artifacts(&program, &opts);
        let proved = lfm_prove(&program, &artifacts, &sponge_arenas(&msg), &opts)
            .unwrap_or_else(|e| panic!("len {len}: prove failed: {e:?}"));

        let mut host = DefaultTranscript::<GoldilocksExtension>::new(&msg);
        assert_eq!(
            digest_bytes(&proved.public_words),
            host.sample(),
            "len {len}: proved sample() must match DefaultTranscript"
        );
        assert!(
            verify_against(
                &artifacts.roots,
                &artifacts.program_id,
                &proved.proof,
                &proved.public_words,
                &opts,
            ),
            "len {len}: the machine proof of sample() must verify"
        );
    }
}

// ============ R1d: DefaultTranscript model + candidate identity ============

/// The host model must track the real post-#841 `DefaultTranscript` exactly,
/// across an interleaving that exercises buffer refill AND absorb invalidation.
#[test]
fn transcript_model_matches_default_transcript() {
    use crate::tables::types::GoldilocksExtension;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use crypto::fiat_shamir::is_transcript::IsTranscript;

    let mut host = DefaultTranscript::<GoldilocksExtension>::new(b"seed");
    let mut model = keccak_host::TranscriptModel::new(b"seed");

    // `sample_u64(2^n)` has threshold 0, so it consumes exactly one candidate
    // and returns its low n bits. Compare the model's raw candidate masked the
    // same way; the raw 32-byte squeezes are compared exactly further down.
    const MASK: u64 = (1u64 << 63) - 1;
    // Drain a full squeeze (4 candidates) and force a refill on the 5th.
    for i in 0..5 {
        assert_eq!(
            model.next_u64() & MASK,
            host.sample_u64(1 << 63),
            "candidate {i}"
        );
    }
    // Absorb mid-buffer: both must drop the remaining squeezed bytes.
    host.append_bytes(b"abc");
    model.append(b"abc");
    for i in 0..3 {
        assert_eq!(
            model.next_u64() & MASK,
            host.sample_u64(1 << 63),
            "post-absorb {i}"
        );
    }
    // A raw sample() also invalidates.
    assert_eq!(model.sample(), host.sample(), "raw sample");
    for i in 0..2 {
        assert_eq!(
            model.next_u64() & MASK,
            host.sample_u64(1 << 63),
            "post-sample {i}"
        );
    }
    // Absorbs of several lengths, including one crossing the keccak rate.
    for len in [1usize, 135, 136, 200] {
        let msg: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(7)).collect();
        host.append_bytes(&msg);
        model.append(&msg);
        assert_eq!(
            model.next_u64() & MASK,
            host.sample_u64(1 << 63),
            "len {len}"
        );
    }
}

/// THE IDENTITY THE EMITTER RESTS ON: the four big-endian candidates carved out
/// of a reversed digest are the ORIGINAL digest's `u64` lanes 3, 2, 1, 0.
///
/// If this holds, the machine reads candidates straight off the plain digest
/// words — already `u32` halves on the bus — and never reverses anything to
/// sample. The big-endian read and the byte reversal cancel.
#[test]
fn be_candidates_are_plain_state_lanes() {
    for len in [0usize, 1, 135, 202] {
        let msg: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(13)).collect();

        // The state whose first 32 bytes are the digest.
        let digest = keccak_host::keccak256(&msg);
        let mut state = [0u64; 25];
        for (lane, chunk) in state[..4].iter_mut().zip(digest.chunks_exact(8)) {
            let mut b = [0u8; 8];
            b.copy_from_slice(chunk);
            *lane = u64::from_le_bytes(b);
        }

        let mut model = keccak_host::TranscriptModel::new(&msg);
        for i in 0..4 {
            assert_eq!(
                model.next_u64(),
                keccak_host::candidate_from_state(&state, i),
                "len {len}, candidate {i} must be state lane {}",
                3 - i
            );
        }
    }
}

// ================= R1d: the TranscriptReplay emitter =================

use super::programs::{
    TRANSCRIPT_ABSORB_A, TRANSCRIPT_ABSORB_B, TRANSCRIPT_ARENA_HALVES, TRANSCRIPT_QUERY_BITS,
    TRANSCRIPT_SEED, canonicity_guard_program, transcript_replay_program,
    transcript_replay_program_source,
};

/// Goldilocks: `p = 2^64 − 2^32 + 1`.
const P: u64 = 0xFFFF_FFFF_0000_0001;

// ---------------------------- oracle scrutiny ----------------------------

/// The assumption that lets a BASE-field `DefaultTranscript` be the oracle for a
/// machine script containing an EXTENSION draw: an ext3 element is three
/// consecutive base draws, in coordinate order 0, 1, 2.
///
/// Read off `Degree3GoldilocksExtensionField::sample_field_element_from`, which
/// is `from_fn(|_| GoldilocksField::sample_field_element_from(&mut next_u64))`.
/// `from_fn` evaluating in index order is the load-bearing part, so it is pinned
/// here against the real thing rather than trusted.
#[test]
fn ext_draw_is_three_base_draws_in_coordinate_order() {
    use crate::tables::types::{GoldilocksExtension, GoldilocksField};
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use crypto::fiat_shamir::is_transcript::IsTranscript;

    let mut ext = DefaultTranscript::<GoldilocksExtension>::new(TRANSCRIPT_SEED);
    let mut base = DefaultTranscript::<GoldilocksField>::new(TRANSCRIPT_SEED);
    // Four draws = twelve candidates, so this spans three refills and cannot be
    // satisfied by a coincidence inside one squeeze.
    for draw in 0..4 {
        let e = ext.sample_field_element();
        let coords: [FE; 3] = core::array::from_fn(|_| base.sample_field_element());
        assert_eq!(*e.value(), coords, "ext draw {draw}");
    }
}

/// The guard's predicate, host-side: a candidate is out of range exactly when
/// `hi = 2^32 − 1 ∧ lo ≠ 0`.
fn machine_accepts(lo: u64, hi: u64) -> bool {
    !(hi == 0xFFFF_FFFF && lo != 0)
}

/// THE DERIVATION, checked against the production sampler rather than against
/// itself: for every candidate, the machine's one-instruction predicate agrees
/// with `GoldilocksField::sample_field_element_from` on whether the FIRST draw
/// is accepted, and on the value when it is.
///
/// The production sampler is probed by feeding it the candidate under test and
/// then zeros: it took a second draw exactly when it rejected the first.
#[test]
fn canonicity_predicate_matches_production_sampler() {
    use math::field::traits::HasDefaultTranscript;

    let production = |candidate: u64| -> Option<FE> {
        let mut draws = 0usize;
        let v = crate::tables::types::GoldilocksField::sample_field_element_from(|| {
            draws += 1;
            if draws == 1 { candidate } else { 0 }
        });
        (draws == 1).then_some(v)
    };

    let mut candidates: Vec<u64> = vec![0, 1, 1 << 32, 0xFFFF_FFFF, u64::MAX];
    // Dense coverage of the boundary itself.
    for d in 0..40u64 {
        candidates.push(P.wrapping_sub(20).wrapping_add(d));
    }
    // The whole of the reject region's shape: hi pinned at 2^32 − 1.
    for lo in [0u64, 1, 2, 3, 0x7FFF_FFFF, 0xFFFF_FFFE, 0xFFFF_FFFF] {
        candidates.push((0xFFFF_FFFFu64 << 32) | lo);
        candidates.push((0xFFFF_FFFEu64 << 32) | lo);
    }
    // A broad deterministic sweep.
    let mut x = 0x1234_5678_9abc_def0u64;
    for _ in 0..5000 {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        candidates.push(x);
    }

    let mut rejects = 0usize;
    for c in candidates {
        let (lo, hi) = (c & 0xFFFF_FFFF, c >> 32);
        let accepted = production(c);
        assert_eq!(
            machine_accepts(lo, hi),
            accepted.is_some(),
            "candidate {c:#018x}: guard and production sampler disagree"
        );
        match accepted {
            Some(v) => {
                let recomposed = &(&FE::from(hi) * &FE::from(1u64 << 32)) + &FE::from(lo);
                assert_eq!(recomposed, v, "candidate {c:#018x}: value");
            }
            None => rejects += 1,
        }
    }
    assert!(
        rejects >= 20,
        "the sweep must actually exercise the reject branch, saw {rejects}"
    );
}

// -------------------------- the canonicity guard --------------------------

fn guard_arenas(lo: u64, hi: u64) -> Vec<Vec<LfmWord>> {
    vec![vec![
        super::word::base_word(FE::from(lo)),
        super::word::base_word(FE::from(hi)),
    ]]
}

/// The machine's guard at the boundary, which the replay itself cannot reach:
/// producing a digest whose candidate is ≥ p by search costs about 2^32 keccaks.
#[test]
fn machine_canonicity_guard_accepts_and_rejects_at_the_boundary() {
    let program = canonicity_guard_program();
    validate(&program).expect("the guard harness must pass admission");
    let run = |c: u64| {
        super::executor::execute(
            &program,
            &guard_arenas(c & 0xFFFF_FFFF, c >> 32),
            &super::hash::TestPermutation,
        )
    };

    for c in [0u64, 1, 12345, 1 << 32, P - 2, P - 1] {
        let exec = run(c).unwrap_or_else(|e| panic!("{c:#018x} is canonical: {e:?}"));
        assert_eq!(
            exec.public_words[0].1[0],
            FE::from(c),
            "{c:#018x}: recomposed value"
        );
    }
    for c in [P, P + 1, P + 12345, u64::MAX] {
        match run(c) {
            Err(LfmExecError::DivByZero { .. }) => {}
            other => panic!(
                "{c:#018x} is ≥ p and must fail the guard, got {:?}",
                other.map(|_| "accepted")
            ),
        }
    }
}

/// The guard has to hold against a prover, not just against the executor.
///
/// Trace tampering cannot show this — changing the arena makes the executor
/// refuse, and changing one trace cell desynchronises the memory bus, which
/// rejects for the wrong reason. So this is the coherent forgery (§ the
/// permute-row precedent): start from candidate `p − 1`, whose guard row is
/// `div(lo = 0, g = 0)`, and forge `lo = 1` — i.e. candidate `p` — in EVERY row
/// that touches that cell, recomputing the published value to what the forged
/// halves really give ((2^32 − 1)·2^32 + 1 = p ≡ 0). The hint's send, both
/// receives, the mul-add's own constraint and the public output are then all
/// internally consistent and every bus balances. The single division constraint
/// `SEL_DIV·(B·OUT − A) = 0`, which now reads `0·1 − 1 ≠ 0`, is the only thing
/// left standing between this and an accepted proof.
#[test]
fn canonicity_guard_rejects_an_out_of_range_candidate_in_the_proof() {
    let opts = options();
    let program = canonicity_guard_program();
    let artifacts = build_artifacts(&program, &opts);
    let mut exec = super::executor::execute(
        &program,
        &guard_arenas(0, 0xFFFF_FFFF),
        &super::hash::TestPermutation,
    )
    .expect("p − 1 is canonical");

    let one = FE::one();
    exec.records.hint[0] = super::word::base_word(one);
    // BALU rows in emission order: sub (g = 2^32 − 1 − hi), div (the guard),
    // mul-add (the value). Only the guard's numerator and the value move.
    exec.records.balu[1].a = one;
    exec.records.balu[2].c = one;
    exec.records.balu[2].out = FE::zero();
    exec.records.public[0] = super::word::base_word(FE::zero());
    exec.public_words[0].1 = super::word::base_word(FE::zero());

    let mut traces = build_traces(&program, &exec.records);
    let proof = prove_traces(&artifacts, &mut traces, &exec.public_words, &opts)
        .expect("the prover has no constraint checks, so it accepts");
    assert!(
        !verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            &proof,
            &exec.public_words,
            &opts,
        ),
        "a candidate at p must fail the canonicity guard"
    );
}

// ---------------------------- the replay itself ----------------------------

/// The two absorbed blobs. Both lengths are multiples of four, so packing their
/// CONCATENATION into halves gives each blob its own whole halves — which is
/// also the property `append_halves` relies on.
fn transcript_absorbs() -> (Vec<u8>, Vec<u8>) {
    let a = (0..TRANSCRIPT_ABSORB_A)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
        .collect();
    let b = (0..TRANSCRIPT_ABSORB_B)
        .map(|i| (i as u8).wrapping_mul(17).wrapping_add(3))
        .collect();
    (a, b)
}

fn transcript_arenas() -> Vec<Vec<LfmWord>> {
    let (a, b) = transcript_absorbs();
    let mut bytes = a;
    bytes.extend_from_slice(&b);
    let halves = keccak_host::pack_stream(&bytes);
    assert_eq!(halves.len(), TRANSCRIPT_ARENA_HALVES as usize);
    vec![halves.into_iter().map(super::word::base_word).collect()]
}

struct ReplayExpectation {
    f0: FE,
    f1: FE,
    e: [FE; 3],
    q: u64,
    f2: FE,
    s: [u8; 32],
    f3: FE,
}

/// The oracle: the REAL `DefaultTranscript`, driven through the same script.
///
/// Instantiated over the base field so that `sample_field_element` is one draw,
/// matching the machine's `sample_felt`; the extension draw in the middle is
/// three consecutive base draws, which
/// `ext_draw_is_three_base_draws_in_coordinate_order` pins against the real ext
/// sampler independently.
fn host_expectation() -> ReplayExpectation {
    use crate::tables::types::GoldilocksField;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use crypto::fiat_shamir::is_transcript::IsTranscript;

    let (a, b) = transcript_absorbs();
    let mut h = DefaultTranscript::<GoldilocksField>::new(TRANSCRIPT_SEED);
    h.append_bytes(&a);
    let f0 = h.sample_field_element();
    let f1 = h.sample_field_element();
    let e: [FE; 3] = core::array::from_fn(|_| h.sample_field_element());
    h.append_bytes(&b);
    let q = h.sample_u64(1 << TRANSCRIPT_QUERY_BITS);
    let f2 = h.sample_field_element();
    let s = h.sample();
    let f3 = h.sample_field_element();
    ReplayExpectation {
        f0,
        f1,
        e,
        q,
        f2,
        s,
        f3,
    }
}

fn check_replay_publics(public: &[(u32, LfmWord)], what: &str) {
    let x = host_expectation();
    assert_eq!(public.len(), 8, "{what}: public word count");
    assert_eq!(public[0].1[0], x.f0, "{what}: first base challenge");
    assert_eq!(public[1].1[0], x.f1, "{what}: second base challenge");
    for i in 0..3 {
        assert_eq!(public[2].1[i], x.e[i], "{what}: ext coordinate {i}");
    }
    assert_eq!(public[3].1[0], FE::from(x.q), "{what}: sample_u64 draw");
    assert_eq!(public[4].1[0], x.f2, "{what}: post-absorb challenge");
    assert_eq!(digest_bytes(&public[5..7]), x.s, "{what}: raw sample()");
    assert_eq!(public[7].1[0], x.f3, "{what}: post-sample challenge");
}

#[test]
fn transcript_replay_program_is_admissible() {
    validate(&transcript_replay_program()).expect("the replay must pass admission");
}

#[test]
fn transcript_replay_source_is_deterministic() {
    let a = transcript_replay_program_source();
    let b = transcript_replay_program_source();
    assert_eq!(a.instrs.len(), b.instrs.len());
    assert_eq!(a.num_addrs, b.num_addrs);
    assert_eq!(format!("{:?}", a.instrs), format!("{:?}", b.instrs));
}

/// Bit-exactness against the real transcript, execute-only (fast). Validates the
/// EMITTER — the consumption schedule, the invalidation rules, the candidate
/// lane mapping — against `DefaultTranscript` itself.
#[test]
fn transcript_replay_matches_default_transcript() {
    let exec = super::executor::execute(
        &transcript_replay_program(),
        &transcript_arenas(),
        &super::hash::TestPermutation,
    )
    .expect("the replay must execute");
    check_replay_publics(&exec.public_words, "execute");
}

/// The R1d headline: the machine PROVES a scripted `DefaultTranscript`
/// interleaving and the proof verifies through the registry, with every sampled
/// value identical to the real transcript's.
///
/// The proving half is not redundant with the execute-only test above. Per the
/// R1c lesson, `execute` fills the keccak rows from the host mirror, so an
/// execute-only test says nothing about whether the CHIP agrees — and this
/// program leans on the chip's reversed-digest send (the re-absorb), on `Unpack`
/// of keccak output words (the candidates), and on the BALU division that
/// enforces canonicity.
#[test]
fn transcript_replay_proves_and_verifies() {
    let opts = options();
    let program = transcript_replay_program();
    let artifacts = build_artifacts(&program, &opts);
    let proved = lfm_prove(&program, &artifacts, &transcript_arenas(), &opts).expect("prove");
    check_replay_publics(&proved.public_words, "prove");
    assert!(
        lfm_verify(
            LfmProgramKind::TranscriptReplayV0,
            &proved.proof,
            &proved.public_words,
            &opts,
        )
        .expect("TranscriptReplayV0 is registered"),
        "the registered transcript replay must verify"
    );
}

/// Flipping one absorbed half must reject: the absorb feeds a squeeze, so every
/// later challenge moves, and claiming the honest ones no longer matches.
///
/// Both blobs are covered — the first is absorbed before any squeeze, the second
/// invalidates a buffer mid-flight, and they reach the sponge by different
/// paths.
#[test]
fn tampered_transcript_absorb_half_rejects() {
    let opts = options();
    let program = transcript_replay_program();
    let artifacts = build_artifacts(&program, &opts);
    let honest = lfm_prove(&program, &artifacts, &transcript_arenas(), &opts).expect("prove");
    check_replay_publics(&honest.public_words, "honest");

    for (half, what) in [(5u32, "first absorb"), (20, "second absorb")] {
        let mut tampered = transcript_arenas();
        tampered[0][half as usize][0] = &tampered[0][half as usize][0] + FE::from(1u64);
        let forged = lfm_prove(&program, &artifacts, &tampered, &opts).expect("prove");
        assert_ne!(
            forged.public_words, honest.public_words,
            "{what}: a changed half must change the challenges"
        );
        assert!(
            !verify_against(
                &artifacts.roots,
                &artifacts.program_id,
                &forged.proof,
                &honest.public_words,
                &opts,
            ),
            "{what}: claiming the honest challenges for a tampered absorb must reject"
        );
    }
}

#[test]
fn registry_drift_transcript_replay_v0_blowup2() {
    let opts = options();
    let program = transcript_replay_program();
    let artifacts = build_artifacts(&program, &opts);
    let entry = resolve(LfmProgramKind::TranscriptReplayV0, 2)
        .expect("TranscriptReplayV0@2 must be registered");
    assert_eq!(entry.roots, artifacts.roots, "group roots drifted");
    assert_eq!(
        entry.log_heights, artifacts.log_heights,
        "group heights drifted"
    );
    assert_eq!(entry.program_id, artifacts.program_id, "program_id drifted");
}

/// Pins the emitted SHAPE, which the value tests would only catch indirectly:
/// the script's five squeezes span six rate blocks (segment #3 is 168 bytes and
/// takes two), so the program must hold exactly six keccak rows. An extra or
/// missing squeeze — the classic invalidation-rule bug — moves this number.
#[test]
fn transcript_replay_cell_counts() {
    let program = transcript_replay_program();
    let (main, aux) = super::airs::lfm_cell_counts(&program);
    println!(
        "TranscriptReplayV0: {} instructions, {} addresses, {} main value cells, {} aux ext elements",
        program.instrs.len(),
        program.num_addrs,
        main,
        aux
    );
    assert_eq!(
        program.groups.keccak.real_rows, 6,
        "five squeezes over six rate blocks"
    );
    assert!(main > 0 && aux > 0);
}

// ------------------------- emitter-contract guards -------------------------

#[test]
#[should_panic(expected = "nbits must be in 1..=32")]
fn sample_u64_pow2_rejects_more_than_32_bits() {
    use super::transcript_replay::TranscriptReplay;
    let mut b = super::builder::LfmBuilder::new();
    let mut t = TranscriptReplay::new(TRANSCRIPT_SEED);
    let _ = t.sample_u64_pow2(&mut b, 33);
}

/// The packing obligation, made unmissable: a constant whose length is not a
/// multiple of four leaves the segment byte-misaligned, and machine data
/// appended after one would straddle a half boundary — which needs the
/// byte-level splice the statement-absorb leg will build, not a silent
/// miscoding here.
#[test]
#[should_panic(expected = "must start on a 4-byte boundary")]
fn machine_data_after_a_misaligned_constant_is_rejected() {
    use super::transcript_replay::TranscriptReplay;
    let mut b = super::builder::LfmBuilder::new();
    let mut t = TranscriptReplay::new(b"abc");
    let z = b.felt_const(FE::zero());
    t.append_halves(&[z]);
}

/// Pins the completeness figures `SOUNDNESS.md` §6.3 quotes, so a doc number
/// cannot drift away from the arithmetic behind it.
#[test]
fn zero_rejection_completeness_bound() {
    use super::transcript_replay::{
        reject_probability_per_candidate, reject_probability_per_proof,
    };

    // q = (2^32 − 1)/2^64: just under 2^−32, and within a hair of it.
    let q = reject_probability_per_candidate();
    assert!(q < 2f64.powi(-32), "q must be strictly below 2^-32");
    assert!(q > 2f64.powi(-32) * (1.0 - 1e-9), "q ≈ 2^-32");

    // The verified schedule: E = 4 + T·(3 + L_t) extension draws per proof, each
    // three base candidates. L_t = 12 with tables at their 2^19 row cap.
    let ext_draws = |tables: usize, fold_challenges: usize| 4 + tables * (3 + fold_challenges);
    assert_eq!(ext_draws(24, 12), 364, "T = 24 (the structural minimum)");
    assert_eq!(ext_draws(60, 12), 904, "T ≈ 60 (realistic)");

    let p_min = reject_probability_per_proof(3 * ext_draws(24, 12));
    let p_real = reject_probability_per_proof(3 * ext_draws(60, 12));
    assert!(
        (p_min - 2.54e-7).abs() < 0.01e-7,
        "the T = 24 bound moved: {p_min:e}"
    );
    assert!(
        (p_real - 6.31e-7).abs() < 0.01e-7,
        "the T = 60 bound moved: {p_real:e}"
    );
    assert!(
        p_real < 1e-6,
        "the headline claim is < 1e-6 at production shapes"
    );

    // Per-table growth is 15 extension draws, NOT one: the per-draw figure is
    // ~7e-10 and the per-table figure is 15x that. Conflating them was a real
    // error in an earlier draft of §6.3, so both are pinned.
    let per_draw = reject_probability_per_proof(3);
    let per_table = (p_real - p_min) / 36.0;
    assert!((per_draw - 6.98e-10).abs() < 0.01e-10, "per extension draw");
    assert!(
        (per_table - 1.048e-8).abs() < 0.01e-8,
        "per additional table"
    );
    assert!(
        (per_table / per_draw - 15.0).abs() < 1e-6,
        "a table is 3 + L_t = 15 extension draws"
    );

    // Where it stops being negligible: ~4.3e7 base candidates for 1%, 2^31 for 50%.
    assert!(reject_probability_per_proof(43_000_000) > 0.01);
    assert!(reject_probability_per_proof(42_000_000) < 0.01);
    assert!((reject_probability_per_proof(1 << 31) - 0.5).abs() < 1e-6);
}

/// Pins `append_digest`'s word-to-halves byte order: a machine-computed keccak
/// digest absorbed into the replay must reach the sponge as the same 32 bytes
/// `DefaultTranscript::append_bytes` sees.
///
/// This is the absorb path a real verifier uses for every commitment root, and
/// the acceptance script above does not reach it (it absorbs arena halves
/// directly). Reversing the lane order inside `append_word` fails this test and
/// nothing else.
#[test]
fn absorbed_machine_digest_matches_default_transcript() {
    use crate::tables::types::GoldilocksField;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use crypto::fiat_shamir::is_transcript::IsTranscript;

    for len in [0usize, 135, KECCAK_SPONGE_LEN] {
        let msg: Vec<u8> = (0..len)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
            .collect();
        let program = super::programs::transcript_absorb_digest_program(len);
        validate(&program).unwrap_or_else(|e| panic!("len {len}: admission: {e:?}"));
        let exec = super::executor::execute(
            &program,
            &sponge_arenas(&msg),
            &super::hash::TestPermutation,
        )
        .unwrap_or_else(|e| panic!("len {len}: execution failed: {e:?}"));

        let mut h = DefaultTranscript::<GoldilocksField>::new(TRANSCRIPT_SEED);
        h.append_bytes(&keccak_host::keccak256(&msg));
        assert_eq!(
            exec.public_words[0].1[0],
            h.sample_field_element(),
            "len {len}: challenge after absorbing a machine-computed digest"
        );
    }
}

/// Makes the buffer-position table in `transcript_replay_program_source`'s doc
/// comment executable, so the documented interleaving cannot drift away from the
/// emitted one.
///
/// The value tests would catch a schedule change too, but only as "the numbers
/// moved". This says which step moved.
#[test]
fn transcript_replay_schedule_matches_the_documented_table() {
    use super::transcript_replay::TranscriptReplay;

    let halves_a = TRANSCRIPT_ABSORB_A / keccak_host::BYTES_PER_HALF;
    let mut b = super::builder::LfmBuilder::new();
    let arena = b.declare_arena(TRANSCRIPT_ARENA_HALVES);
    let halves: Vec<_> = (0..TRANSCRIPT_ARENA_HALVES)
        .map(|i| b.hint_felt(arena, i))
        .collect();
    let (absorb_a, absorb_b) = halves.split_at(halves_a);

    let mut t = TranscriptReplay::new(TRANSCRIPT_SEED);
    t.append_halves(absorb_a);
    assert_eq!(
        (t.out_pos(), t.segment_len()),
        (32, TRANSCRIPT_SEED.len() + TRANSCRIPT_ABSORB_A),
        "after absorb A: buffer empty, segment is seed ‖ A"
    );

    let _ = t.sample_felt(&mut b);
    assert_eq!(
        (t.out_pos(), t.segment_len()),
        (8, 32),
        "squeeze #1, then one candidate; the segment becomes the reversed digest"
    );
    let _ = t.sample_felt(&mut b);
    assert_eq!(t.out_pos(), 16, "second candidate, no squeeze");
    let _ = t.sample_ext(&mut b);
    assert_eq!(
        t.out_pos(),
        8,
        "three more candidates: squeeze #2 lands INSIDE the extension draw"
    );

    t.append_halves(absorb_b);
    assert_eq!(
        (t.out_pos(), t.segment_len()),
        (32, 32 + TRANSCRIPT_ABSORB_B),
        "absorb B invalidates a buffer with 24 live bytes; 168 bytes = two blocks"
    );
    let _ = t.sample_u64_pow2(&mut b, TRANSCRIPT_QUERY_BITS);
    assert_eq!(t.out_pos(), 8, "squeeze #3");
    let _ = t.sample_felt(&mut b);
    assert_eq!(t.out_pos(), 16, "no squeeze");
    let _ = t.sample(&mut b);
    assert_eq!(
        t.out_pos(),
        32,
        "raw sample #4 invalidates a buffer with 16 live bytes"
    );
    let _ = t.sample_felt(&mut b);
    assert_eq!(t.out_pos(), 8, "squeeze #5");
}

/// The segment-level packing rule, made executable: consecutive constant appends
/// are ONE byte run, chunked into halves only at the squeeze.
///
/// `"abc"` then `"de"` must hash as the five-byte string `"abcde"` — two halves
/// — not as two independently packed pieces (which would give `"abc\0de\0\0"`).
/// Per-append packing cannot pass this.
#[test]
fn constant_appends_concatenate_across_append_boundaries() {
    use super::transcript_replay::TranscriptReplay;
    use crate::tables::types::GoldilocksField;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use crypto::fiat_shamir::is_transcript::IsTranscript;

    for pieces in [
        vec![&b"abc"[..], &b"de"[..]],
        vec![&b"a"[..], &b"b"[..], &b"c"[..], &b"d"[..], &b"e"[..]],
        vec![&b""[..], &b"abcde"[..]],
        vec![&b"abcde"[..]],
    ] {
        let mut b = super::builder::LfmBuilder::new();
        let mut t = TranscriptReplay::new(pieces[0]);
        for p in &pieces[1..] {
            t.append_const_bytes(p);
        }
        let f = t.sample_felt(&mut b);
        b.public(f.as_cell());
        let program = super::compiler::compile(b.finish());
        let exec =
            super::executor::execute(&program, &[], &super::hash::TestPermutation).expect("exec");

        let mut h = DefaultTranscript::<GoldilocksField>::new(b"abcde");
        assert_eq!(
            exec.public_words[0].1[0],
            h.sample_field_element(),
            "{pieces:?} must absorb as the concatenation \"abcde\""
        );
    }
}

/// The alignment rule is about the SEGMENT's length, not about whether some
/// earlier append happened to be misaligned: a 3-byte constant followed by a
/// 1-byte constant leaves the segment 4-byte aligned, so machine data may follow.
#[test]
fn machine_data_may_follow_constants_that_together_align() {
    use super::transcript_replay::TranscriptReplay;

    let mut b = super::builder::LfmBuilder::new();
    let mut t = TranscriptReplay::new(b"abc");
    t.append_const_bytes(b"d");
    let z = b.felt_const(FE::zero());
    t.append_halves(&[z]);
    assert_eq!(t.segment_len(), 8, "4 constant bytes plus one machine half");
}

/// Cross-check of the emitter's squeeze economics against the verified
/// production draw schedule.
///
/// Per table the verifier draws β, z_OOD, γ and `L` FRI fold challenges, each
/// preceded by a root absorb that invalidates the buffer — so each extension
/// draw costs one fresh squeeze and uses three of its four candidates. The `Q`
/// query indices are then drawn back to back, costing `⌈Q/4⌉`.
///
/// Keccak ROWS equal squeezes here because every segment stays inside one
/// 136-byte rate block: 32 reversed-digest bytes plus a 32-byte root is 64.
#[test]
fn squeeze_economics_match_the_verified_draw_schedule() {
    use super::transcript_replay::TranscriptReplay;

    const L: usize = 12; // fold challenges: log2(trace) − 7, tables at the 2^19 cap
    const Q: usize = 219; // Preset::Blowup2 query count
    let ext_draws = 3 + L; // β, z_OOD, γ, then L fold challenges
    let halves_per_root = 8;

    let mut b = super::builder::LfmBuilder::new();
    let arena = b.declare_arena(((ext_draws + 1) * halves_per_root) as u32);
    let mut t = TranscriptReplay::new(TRANSCRIPT_SEED);
    for r in 0..=ext_draws {
        let root: Vec<_> = (0..halves_per_root)
            .map(|i| b.hint_felt(arena, (r * halves_per_root + i) as u32))
            .collect();
        t.append_halves(&root);
        // The last absorb stands for the grinding/final-poly absorb that precedes
        // query sampling; it draws nothing.
        if r < ext_draws {
            let _ = t.sample_ext(&mut b);
        }
    }
    for _ in 0..Q {
        let _ = t.sample_u64_pow2(&mut b, 20);
    }
    let program = super::compiler::compile(b.finish());

    let expected = ext_draws + Q.div_ceil(4);
    assert_eq!(
        program.groups.keccak.real_rows,
        expected,
        "expected {ext_draws} squeezes for the extension draws (one each, the \
         preceding absorb having invalidated the buffer) plus ⌈{Q}/4⌉ = {} for the \
         query draws",
        Q.div_ceil(4)
    );
    assert_eq!(expected, 70, "15 extension squeezes + 55 query squeezes");
}

// ============ R1e slice a: field elements on the wire (big-endian) ============

/// Byte patterns that make an endianness or permutation error impossible to
/// miss: every byte of the first value is distinct, and the boundary values pin
/// the canonical range `bit_dec` enforces.
fn be_reference_felts() -> Vec<u64> {
    vec![
        0,
        1,
        0x0123_4567_89ab_cdef,
        0xfedc_ba98_7654_3210,
        0xff,
        0xff00_0000,
        1 << 32,
        P - 1,
        P - 2,
    ]
}

/// A base field element must reach the sponge as the same 8 big-endian bytes
/// `append_field_element` streams.
///
/// The program publishes the raw 32-byte squeeze rather than a sampled
/// challenge, so a mismatch localises to the absorbed bytes.
#[test]
fn append_felt_matches_default_transcript() {
    use crate::tables::types::GoldilocksField;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use crypto::fiat_shamir::is_transcript::IsTranscript;

    let program = super::programs::append_felt_program();
    validate(&program).expect("admission");
    for v in be_reference_felts() {
        let arenas = vec![vec![super::word::base_word(FE::from(v))]];
        let exec = super::executor::execute(&program, &arenas, &super::hash::TestPermutation)
            .unwrap_or_else(|e| panic!("{v:#018x}: execution failed: {e:?}"));

        let mut h = DefaultTranscript::<GoldilocksField>::new(TRANSCRIPT_SEED);
        h.append_field_element(&FE::from(v));
        assert_eq!(
            digest_bytes(&exec.public_words),
            h.sample(),
            "{v:#018x}: absorbed bytes must match append_field_element"
        );
    }
}

/// The same for a cubic-extension element — 24 bytes, coordinates 0, 1, 2.
///
/// The three coordinates are deliberately distinct, so a reversed coordinate
/// order (the other byte order this file offers, which belongs to the raw
/// `[FpE; 3]` type) fails rather than coincidentally passing.
#[test]
fn append_ext_matches_default_transcript() {
    use crate::tables::types::GoldilocksExtension;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use math::field::element::FieldElement;

    let program = super::programs::append_ext_program();
    validate(&program).expect("admission");
    for coords in [
        [0u64, 1, 2],
        [0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210, 0xff],
        [P - 1, 0, 1 << 32],
    ] {
        let arenas = vec![
            coords
                .iter()
                .map(|&c| super::word::base_word(FE::from(c)))
                .collect::<Vec<_>>(),
        ];
        let exec = super::executor::execute(&program, &arenas, &super::hash::TestPermutation)
            .unwrap_or_else(|e| panic!("{coords:?}: execution failed: {e:?}"));

        let e =
            FieldElement::<GoldilocksExtension>::new(core::array::from_fn(|i| FE::from(coords[i])));
        let mut h = DefaultTranscript::<GoldilocksExtension>::new(TRANSCRIPT_SEED);
        h.append_field_element(&e);
        assert_eq!(
            digest_bytes(&exec.public_words),
            h.sample(),
            "{coords:?}: absorbed bytes must match append_field_element"
        );
    }
}

/// The byteswap gadget PROVED, not just executed.
///
/// `felt_be_halves` is the first thing in this emitter that leans on `LFM_BITDEC`
/// for a value rather than for index bits, and on a 32-term `MulAdd` chain whose
/// weights carry the byte permutation. Execution alone would not catch a
/// chip-vs-executor disagreement in either.
#[test]
fn append_ext_proves_and_verifies() {
    let opts = options();
    let program = super::programs::append_ext_program();
    let artifacts = build_artifacts(&program, &opts);
    let coords = [0x0123_4567_89ab_cdefu64, 0xfedc_ba98_7654_3210, P - 1];
    let arenas = vec![
        coords
            .iter()
            .map(|&c| super::word::base_word(FE::from(c)))
            .collect::<Vec<_>>(),
    ];
    let proved = lfm_prove(&program, &artifacts, &arenas, &opts).expect("prove");
    assert!(
        verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            &proved.proof,
            &proved.public_words,
            &opts,
        ),
        "the big-endian absorb must verify"
    );
}

/// Pins the gadget's cost, which is the reason `append_field_element` was
/// deferred out of R1d: one `BitDec` plus 64 `BALU` rows per felt.
#[test]
fn felt_be_halves_cost() {
    let program = super::programs::append_felt_program();
    println!(
        "append_felt: {} instructions, bitdec {}, balu {}",
        program.instrs.len(),
        program.groups.bitdec.real_rows,
        program.groups.balu.real_rows
    );
    assert_eq!(
        program.groups.bitdec.real_rows, 1,
        "one decomposition per felt"
    );
    assert_eq!(
        program.groups.balu.real_rows, 64,
        "two accumulators, each 1 Mul + 31 MulAdd over its 32 bits"
    );
}

// ==================== R1e slice b: the byte-level splice ====================

use super::programs::{
    SPLICE_ALT_DIGEST_HALVES, SPLICE_ALT_FIELD_HALVES, SPLICE_ALT_TAG, splice_alternating_program,
    splice_dynamic, splice_prefix, splice_program,
};

fn splice_arenas(byte_len: usize) -> Vec<Vec<LfmWord>> {
    vec![
        keccak_host::pack_stream(&splice_dynamic(byte_len))
            .into_iter()
            .map(super::word::base_word)
            .collect(),
    ]
}

/// The splice at every shift, against the REAL transcript.
///
/// The oracle is `DefaultTranscript` over the concatenated byte string, which is
/// the definition of what the machine must reproduce: append boundaries leave no
/// trace in the digest input, so the whole segment is one byte string and the
/// machine's job is to hash exactly it.
///
/// Shift 0 is included as the control — it takes the aligned fast path, so if
/// the splice were silently applied there it would show up here.
#[test]
fn splice_matches_default_transcript_at_every_shift() {
    use crate::tables::types::GoldilocksField;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;

    const DYN_BYTES: usize = 32;
    let halves = (DYN_BYTES / keccak_host::BYTES_PER_HALF) as u32;
    for prefix_len in [0usize, 1, 2, 3, 4, 5, 6, 7, 29, 30, 31, 32] {
        let program = splice_program(prefix_len, halves);
        validate(&program).unwrap_or_else(|e| panic!("prefix {prefix_len}: admission: {e:?}"));
        let exec = super::executor::execute(
            &program,
            &splice_arenas(DYN_BYTES),
            &super::hash::TestPermutation,
        )
        .unwrap_or_else(|e| panic!("prefix {prefix_len}: execution failed: {e:?}"));

        let mut bytes = splice_prefix(prefix_len);
        bytes.extend_from_slice(&splice_dynamic(DYN_BYTES));
        let mut h = DefaultTranscript::<GoldilocksField>::new(&bytes);
        assert_eq!(
            digest_bytes(&exec.public_words),
            h.sample(),
            "prefix {prefix_len} (shift {}): spliced bytes must equal the concatenation",
            prefix_len % keccak_host::BYTES_PER_HALF
        );
    }
}

/// The statement's real shape: alternating constant and dynamic runs where a
/// one-byte field moves the shift from 2 to 3 partway through.
#[test]
fn splice_alternating_runs_match_default_transcript() {
    use crate::tables::types::GoldilocksField;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;

    let d = SPLICE_ALT_DIGEST_HALVES as usize * keccak_host::BYTES_PER_HALF;
    let f = SPLICE_ALT_FIELD_HALVES as usize * keccak_host::BYTES_PER_HALF;
    let program = splice_alternating_program();
    validate(&program).expect("admission");
    let exec = super::executor::execute(
        &program,
        &splice_arenas(d + 2 * f),
        &super::hash::TestPermutation,
    )
    .expect("execution");

    // The same byte string, built independently in absorb order.
    let dynamic = splice_dynamic(d + 2 * f);
    let mut bytes = splice_prefix(SPLICE_ALT_TAG);
    bytes.extend_from_slice(&dynamic[..d]);
    bytes.extend_from_slice(&splice_prefix(8));
    bytes.extend_from_slice(&dynamic[d..d + f]);
    bytes.extend_from_slice(&splice_prefix(1));
    bytes.extend_from_slice(&dynamic[d + f..]);

    let mut h = DefaultTranscript::<GoldilocksField>::new(&bytes);
    assert_eq!(
        digest_bytes(&exec.public_words),
        h.sample(),
        "alternating const/dynamic runs across a shift change must match"
    );
}

/// The splice PROVED, not just executed: it leans on `BitDec` plus a weighted
/// sum plus the recomposition assert, and only a proof sees the chips.
#[test]
fn splice_proves_and_verifies() {
    let opts = options();
    let program = splice_program(30, 8);
    let artifacts = build_artifacts(&program, &opts);
    let proved = lfm_prove(&program, &artifacts, &splice_arenas(32), &opts).expect("prove");
    assert!(
        verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            &proved.proof,
            &proved.public_words,
            &opts,
        ),
        "the spliced absorb must verify"
    );
}

/// A half at or above `2^32` has no four-byte rendering, so the splice must
/// refuse it rather than silently absorb the wrong bytes.
///
/// `bit_dec` alone bounds its input by `p`, not by `2^32`; the recomposition
/// assert inside `split_half` is what closes the gap, and this is the test that
/// fails if it is removed.
#[test]
fn splice_rejects_a_non_u32_half() {
    let program = splice_program(2, 8);
    let mut arenas = splice_arenas(32);
    arenas[0][0][0] = FE::from(1u64 << 32);
    match super::executor::execute(&program, &arenas, &super::hash::TestPermutation) {
        Err(LfmExecError::DivByZero { .. }) => {}
        other => panic!(
            "a half at 2^32 must fail the splice's recomposition assert, got {:?}",
            other.map(|_| "accepted")
        ),
    }
}

/// Pins the splice's cost, and that the ALIGNED path is still free.
#[test]
fn splice_cost() {
    let spliced = splice_program(2, 8);
    let aligned = splice_program(4, 8);
    println!(
        "splice 8 halves @shift2: bitdec {}, balu {} | aligned: bitdec {}, balu {}",
        spliced.groups.bitdec.real_rows,
        spliced.groups.balu.real_rows,
        aligned.groups.bitdec.real_rows,
        aligned.groups.balu.real_rows,
    );
    assert_eq!(
        spliced.groups.bitdec.real_rows, 8,
        "one decomposition per spliced half"
    );
    assert_eq!(
        aligned.groups.bitdec.real_rows, 0,
        "the aligned path must emit no splice at all"
    );
    assert_eq!(
        aligned.groups.balu.real_rows, 0,
        "the aligned path must stay instruction-free"
    );
}
