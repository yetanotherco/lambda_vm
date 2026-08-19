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
    assert_eq!(
        entry.keccak_rnd_chunks, artifacts.keccak_rnd_chunks,
        "KECCAK_RND chunk count drifted"
    );
    assert_eq!(entry.hasher, artifacts.hasher, "hasher drifted");
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
    assert_eq!(
        entry.keccak_rnd_chunks, artifacts.keccak_rnd_chunks,
        "KECCAK_RND chunk count drifted"
    );
    assert_eq!(entry.hasher, artifacts.hasher, "hasher drifted");
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

    // Leg 3: colliding the tags is not available to the prover. The TAG columns
    // are PREPROCESSED — pinned by the group's precomputed commitment — so a
    // prover cannot obtain a VERIFYING proof with the tags rewritten. Copying
    // row 0's tag over row 1's makes the trace's leading columns disagree with
    // the committed group.
    //
    // Where the failure lands is precomputed-tree-cache-dependent (the cache is
    // keyed by the expected root): on a cache MISS the prover recomputes the
    // preprocessed tree from the tampered columns and refuses with
    // `PrecomputedCommitmentMismatch`; on a warm cache HIT it commits the pinned
    // tree without re-checking (so it does NOT refuse at prove time) and the
    // tampered proof is caught at VERIFY instead. Either way the forgery fails —
    // which is the obligation this leg exists to pin.
    match prove_keccak_chain_with_tamper(&program, &artifacts, 7, |t| {
        let lo = t.keccak.main_table.get_row(0)[klayout::TAG_LO];
        let hi = t.keccak.main_table.get_row(0)[klayout::TAG_HI];
        t.keccak.main_table.set_fe(1, klayout::TAG_LO, lo);
        t.keccak.main_table.set_fe(1, klayout::TAG_HI, hi);
    }) {
        Err(err) => assert!(
            matches!(err, ProvingError::PrecomputedCommitmentMismatch),
            "expected a preprocessed recommit failure, got {err:?}"
        ),
        Ok((proof, public)) => assert!(
            !lfm_verify(LfmProgramKind::KeccakChainV0, &proof, &public, &opts).expect("registered"),
            "a tag-rewrite tamper must not yield a verifying proof"
        ),
    }
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
    assert_eq!(
        entry.keccak_rnd_chunks, artifacts.keccak_rnd_chunks,
        "KECCAK_RND chunk count drifted"
    );
    assert_eq!(entry.hasher, artifacts.hasher, "hasher drifted");
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
                artifacts.keccak_rnd_chunks,
                &proved.proof,
                &proved.public_words,
                &opts,
                artifacts.hasher,
                artifacts.chip_set,
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
            artifacts.keccak_rnd_chunks,
            &forged.proof,
            &honest.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
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
            artifacts.keccak_rnd_chunks,
            &proof,
            &exec.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
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
    assert_eq!(
        entry.keccak_rnd_chunks, artifacts.keccak_rnd_chunks,
        "KECCAK_RND chunk count drifted"
    );
    assert_eq!(entry.hasher, artifacts.hasher, "hasher drifted");
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
            artifacts.keccak_rnd_chunks,
            &proof,
            &exec.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
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
                artifacts.keccak_rnd_chunks,
                &proved.proof,
                &proved.public_words,
                &opts,
                artifacts.hasher,
                artifacts.chip_set,
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
            artifacts.keccak_rnd_chunks,
            &proof,
            &exec.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
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

    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use stark::config::DefaultStarkTranscript;

    let (a, b) = transcript_absorbs();
    let mut h = DefaultStarkTranscript::<GoldilocksField>::new(TRANSCRIPT_SEED);
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
                artifacts.keccak_rnd_chunks,
                &forged.proof,
                &honest.public_words,
                &opts,
                artifacts.hasher,
                artifacts.chip_set,
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
    assert_eq!(
        entry.keccak_rnd_chunks, artifacts.keccak_rnd_chunks,
        "KECCAK_RND chunk count drifted"
    );
    assert_eq!(entry.hasher, artifacts.hasher, "hasher drifted");
    assert_eq!(entry.program_id, artifacts.program_id, "program_id drifted");
}

/// Pins the emitted SHAPE, which the value tests would only catch indirectly:
/// the script's five squeezes span six rate blocks (segment #3 is 168 bytes and
/// takes two), so the program must hold exactly six keccak rows. An extra or
/// missing squeeze — the classic invalidation-rule bug — moves this number.
#[test]
fn transcript_replay_cell_counts() {
    /// Compressions the replay emits under the CONFIGURED wrap hash. It was 6
    /// under keccak — "five squeezes over six rate blocks" — and a 136-byte
    /// sponge rate does not divide this script the way a 64-byte BLAKE3 block
    /// does, so the number is re-derived rather than carried over.
    const TRANSCRIPT_REPLAY_COMPRESSIONS: usize = 8;

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
        wrap_hash_rows(&program),
        TRANSCRIPT_REPLAY_COMPRESSIONS,
        "the replay's compression count"
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

    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use stark::config::DefaultStarkTranscript;

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

        let mut h = DefaultStarkTranscript::<GoldilocksField>::new(TRANSCRIPT_SEED);
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

    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use stark::config::DefaultStarkTranscript;

    let program = super::programs::append_felt_program();
    validate(&program).expect("admission");
    for v in be_reference_felts() {
        let arenas = vec![vec![super::word::base_word(FE::from(v))]];
        let exec = super::executor::execute(&program, &arenas, &super::hash::TestPermutation)
            .unwrap_or_else(|e| panic!("{v:#018x}: execution failed: {e:?}"));

        let mut h = DefaultStarkTranscript::<GoldilocksField>::new(TRANSCRIPT_SEED);
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

    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use math::field::element::FieldElement;
    use stark::config::DefaultStarkTranscript;

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
        let mut h = DefaultStarkTranscript::<GoldilocksExtension>::new(TRANSCRIPT_SEED);
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
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &proved.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
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

    use stark::config::DefaultStarkTranscript;

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
        let mut h = DefaultStarkTranscript::<GoldilocksField>::new(&bytes);
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

    use stark::config::DefaultStarkTranscript;

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

    let mut h = DefaultStarkTranscript::<GoldilocksField>::new(&bytes);
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
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &proved.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
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

// ========== R1e slices c+d: the epoch statement and Phase A ==========

use super::programs::{
    STMT_PREPROCESSED, STMT_PUBLIC_OUTPUT_LEN, epoch_statement_shape, statement_replay_program,
    stmt_arena_halves,
};

/// The per-proof statement values and the Phase-A roots, as bytes. The machine
/// arena and the host oracle are both built from this, so they cannot drift.
struct StatementFixture {
    elf_digest: [u8; 32],
    public_output: Vec<u8>,
    epoch_label: u64,
    /// `(preprocessed_root, main_root)` per sub-proof, in air order.
    roots: Vec<(Option<[u8; 32]>, [u8; 32])>,
}

fn statement_fixture() -> StatementFixture {
    let root = |seed: u8| -> [u8; 32] {
        core::array::from_fn(|i| (i as u8).wrapping_mul(seed).wrapping_add(seed))
    };
    StatementFixture {
        elf_digest: root(7),
        public_output: (0..STMT_PUBLIC_OUTPUT_LEN)
            .map(|i| (i as u8).wrapping_mul(19).wrapping_add(5))
            .collect(),
        epoch_label: 0x0123_4567_89ab_cdef,
        roots: STMT_PREPROCESSED
            .iter()
            .enumerate()
            .map(|(i, &prep)| {
                let p = prep.then(|| root(11 + 2 * i as u8));
                (p, root(31 + 2 * i as u8))
            })
            .collect(),
    }
}

/// Each field gets its OWN halves. The arena is a vector of `u32` words, not a
/// byte stream, so concatenating first and packing after would let a field whose
/// length is not a multiple of four shift every field behind it — which is
/// exactly what an unaligned `public_output` does. `pack_stream` zeroes the
/// trailing half's unused high bytes, the property the machine's mask pins.
fn statement_arenas(f: &StatementFixture) -> Vec<Vec<LfmWord>> {
    let mut halves = keccak_host::pack_stream(&f.elf_digest);
    halves.extend(keccak_host::pack_stream(&f.public_output));
    halves.extend(keccak_host::pack_stream(&f.epoch_label.to_le_bytes()));
    for (prep, main) in &f.roots {
        if let Some(p) = prep {
            halves.extend(keccak_host::pack_stream(p));
        }
        halves.extend(keccak_host::pack_stream(main));
    }
    assert_eq!(halves.len(), stmt_arena_halves() as usize);
    vec![halves.into_iter().map(super::word::base_word).collect()]
}

/// The host reference: the REAL `absorb_statement_with_digest`, then Phase A.
///
/// The statement half of this is production code, not a reimplementation — which
/// matters, because that encoding has ten fields and is exactly where a replay
/// would go wrong. The Phase-A half is a four-line transcription of
/// `crate::replay_transcript_phase_a_view` (`lib.rs`: for each air, the
/// precomputed commitment when `is_preprocessed()`, then
/// `lde_trace_main_merkle_root()`, then `z` and `α`); calling the helper itself
/// would mean synthesising `dyn AIR`s and proof views for three fake tables,
/// which would test the fakes rather than the replay.
type ExtFE = math::field::element::FieldElement<crate::tables::types::GoldilocksExtension>;

fn host_statement_challenges(f: &StatementFixture) -> (ExtFE, ExtFE) {
    use crate::statement::{StatementKind, absorb_statement_with_digest};
    use crate::tables::types::GoldilocksExtension;
    use crate::{RuntimePageRange, TableCounts};

    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use stark::config::DefaultStarkTranscript;

    let shape = epoch_statement_shape();
    let c = shape.table_counts.map(|v| v as usize);
    let counts = TableCounts {
        cpu: c[0],
        lt: c[1],
        memw: c[2],
        memw_aligned: c[3],
        load: c[4],
        mul: c[5],
        dvrm: c[6],
        shift: c[7],
        branch: c[8],
        memw_register: c[9],
        eq: c[10],
        bytewise: c[11],
        store: c[12],
        cpu32: c[13],
    };
    let ranges: Vec<RuntimePageRange> = shape
        .page_ranges
        .iter()
        .map(|&(base, count)| RuntimePageRange { base, count })
        .collect();

    let mut t = DefaultStarkTranscript::<GoldilocksExtension>::new(&[]);
    absorb_statement_with_digest(
        &mut t,
        StatementKind::ContinuationEpoch {
            epoch_label: f.epoch_label,
        },
        &f.elf_digest,
        &f.public_output,
        &counts,
        shape.num_private_input_pages as usize,
        &ranges,
        shape.fri_final_poly_log_degree,
    );
    for (prep, main) in &f.roots {
        if let Some(p) = prep {
            t.append_bytes(p);
        }
        t.append_bytes(main);
    }
    (t.sample_field_element(), t.sample_field_element())
}

fn assert_challenges_match(public: &[(u32, LfmWord)], f: &StatementFixture, what: &str) {
    let (z, alpha) = host_statement_challenges(f);
    assert_eq!(public.len(), 2, "{what}: z and alpha");
    for (i, (name, want)) in [("z", z), ("alpha", alpha)].iter().enumerate() {
        for lane in 0..3 {
            assert_eq!(
                public[i].1[lane],
                want.value()[lane],
                "{what}: {name} coordinate {lane}"
            );
        }
    }
}

/// Pins where the epoch statement leaves the byte cursor, which decides whether
/// Phase A is spliced and at what shift.
///
/// CORRECTION to an earlier claim of mine: the statement is NOT unconditionally
/// 3 bytes past a boundary. Its length is `207 + L + 16R`, so the shift Phase A
/// inherits is `(3 + L) mod 4` — it is 3 only when the public output happens to
/// be a multiple of four, and it is ZERO (Phase A entirely unspliced) whenever
/// `L ≡ 1 (mod 4)`. Since `L` is one byte per COMMIT op and therefore workload-
/// determined, the Phase-A splice cost is workload-dependent and free for about
/// one workload in four.
#[test]
fn epoch_statement_cursor_is_three_plus_output_len() {
    let shape = epoch_statement_shape();
    let r = shape.page_ranges.len();
    assert_eq!(shape.byte_len(), 207 + STMT_PUBLIC_OUTPUT_LEN + 16 * r);
    for l in 0..8usize {
        let total = 207 + l + 16 * r;
        assert_eq!(
            total % keccak_host::BYTES_PER_HALF,
            (3 + l) % keccak_host::BYTES_PER_HALF,
            "Phase A inherits shift (3 + L) mod 4"
        );
    }
    // The acceptance shape is chosen to exercise BOTH new paths at once: an
    // unaligned public output (so the trailing half is masked) and a nonzero
    // inherited shift (so Phase A is spliced).
    assert_ne!(STMT_PUBLIC_OUTPUT_LEN % keccak_host::BYTES_PER_HALF, 0);
    assert_ne!(shape.byte_len() % keccak_host::BYTES_PER_HALF, 0);
}

#[test]
fn statement_replay_program_is_admissible() {
    validate(&statement_replay_program()).expect("admission");
}

/// R1e's acceptance: the machine's `(z, α)` must equal what the REAL statement
/// absorb plus Phase A produce.
#[test]
fn statement_replay_matches_the_host_challenges() {
    let f = statement_fixture();
    let exec = super::executor::execute(
        &statement_replay_program(),
        &statement_arenas(&f),
        &super::hash::TestPermutation,
    )
    .expect("execution");
    assert_challenges_match(&exec.public_words, &f, "execute");
}

/// The same, PROVED and verified through the registry.
#[test]
fn statement_replay_proves_and_verifies() {
    let opts = options();
    let f = statement_fixture();
    let program = statement_replay_program();
    let artifacts = build_artifacts(&program, &opts);
    let proved = lfm_prove(&program, &artifacts, &statement_arenas(&f), &opts).expect("prove");
    assert_challenges_match(&proved.public_words, &f, "prove");
    assert!(
        lfm_verify(
            LfmProgramKind::StatementReplayV0,
            &proved.proof,
            &proved.public_words,
            &opts,
        )
        .expect("StatementReplayV0 is registered"),
        "the registered statement replay must verify"
    );
}

/// Both tamper vectors: a flipped Phase-A root half and a flipped statement
/// byte. Each must move the challenges, and claiming the honest ones must reject.
#[test]
fn tampered_statement_or_root_rejects() {
    let opts = options();
    let f = statement_fixture();
    let program = statement_replay_program();
    let artifacts = build_artifacts(&program, &opts);
    let honest = lfm_prove(&program, &artifacts, &statement_arenas(&f), &opts).expect("prove");

    // Half 0 is the ELF digest (statement); half 14 is inside the first
    // sub-proof's preprocessed root (Phase A).
    for (half, what) in [(0usize, "statement byte"), (14, "Phase-A root half")] {
        let mut arenas = statement_arenas(&f);
        arenas[0][half][0] = &arenas[0][half][0] + FE::from(1u64);
        let forged = lfm_prove(&program, &artifacts, &arenas, &opts).expect("prove");
        assert_ne!(
            forged.public_words, honest.public_words,
            "{what}: a flip must move z or alpha"
        );
        assert!(
            !verify_against(
                &artifacts.roots,
                &artifacts.program_id,
                artifacts.keccak_rnd_chunks,
                &forged.proof,
                &honest.public_words,
                &opts,
                artifacts.hasher,
                artifacts.chip_set,
            ),
            "{what}: claiming the honest challenges must reject"
        );
    }
}

#[test]
fn registry_drift_statement_replay_v0_blowup2() {
    let opts = options();
    let artifacts = build_artifacts(&statement_replay_program(), &opts);
    let entry = resolve(LfmProgramKind::StatementReplayV0, 2)
        .expect("StatementReplayV0@2 must be registered");
    assert_eq!(entry.roots, artifacts.roots, "group roots drifted");
    assert_eq!(
        entry.log_heights, artifacts.log_heights,
        "group heights drifted"
    );
    assert_eq!(
        entry.keccak_rnd_chunks, artifacts.keccak_rnd_chunks,
        "KECCAK_RND chunk count drifted"
    );
    assert_eq!(entry.hasher, artifacts.hasher, "hasher drifted");
    assert_eq!(entry.program_id, artifacts.program_id, "program_id drifted");
}

#[test]
fn statement_replay_cell_counts() {
    let program = statement_replay_program();
    let (main, aux) = super::airs::lfm_cell_counts(&program);
    println!(
        "StatementReplayV0: {} instructions, keccak {}, bitdec {}, balu {}, {main} main cells, {aux} aux",
        program.instrs.len(),
        program.groups.keccak.real_rows,
        program.groups.bitdec.real_rows,
        program.groups.balu.real_rows,
    );
    assert!(main > 0 && aux > 0);
}

/// The masked trailing half's soundness obligation: bytes PAST the encoded
/// length must not reach the sponge.
///
/// `public_output` is length-prefixed, so its final arena half has live bytes
/// only up to `len % 4`. The high bytes of that felt are arena data and
/// otherwise unconstrained — without the zero-pin in `Packer::push_masked` a
/// prover could put anything there and change the absorbed byte string while the
/// length prefix said otherwise. Here byte 3 of the trailing half is past the
/// 14-byte length, so the program must refuse to execute rather than absorb it.
#[test]
fn statement_rejects_garbage_past_the_public_output_length() {
    let f = statement_fixture();
    let mut arenas = statement_arenas(&f);
    // Halves: elf 0..8, public_output 8..12 (14 bytes = 3 whole + 2 live),
    // so half 11's top two bytes are past the length.
    arenas[0][11][0] = &arenas[0][11][0] + FE::from(1u64 << 24);
    match super::executor::execute(
        &statement_replay_program(),
        &arenas,
        &super::hash::TestPermutation,
    ) {
        Err(LfmExecError::DivByZero { .. }) => {}
        other => panic!(
            "bytes past the public-output length must be pinned to zero, got {:?}",
            other.map(|_| "accepted")
        ),
    }
}
// ======================= KECCAK_RND chunking =======================
//
// `KECCAK_RND` costs 24 rows per permutation, so one instance cannot hold the
// ~460k permutations a real proof wrap needs. These tests cover the split: the
// shape it produces, that a multi-chunk program proves and verifies, and the
// two ways the split itself can be wrong (a corrupted chunk, a dropped
// permutation).

use super::airs::{keccak_rnd_chunk_permutations, keccak_rnd_chunk_rows, num_lfm_airs};
use super::chunking::KeccakChunking;
use crate::tables::keccak_rnd;

/// A 3-permutation sponge: `pad10*1` grows 280 bytes to 3 rate blocks, so at
/// two permutations per chunk it splits unevenly (2 + 1) — the partial-final
/// chunk is the case a uniform split would miss.
const CHUNKED_SPONGE_LEN: usize = 280;

/// Two permutations per chunk. Small enough that the multi-chunk tests prove in
/// seconds instead of the 21,845 permutations the default policy would need.
fn test_chunking() -> KeccakChunking {
    KeccakChunking::from_permutations(2)
}

fn chunked_sponge_program() -> LfmProgram {
    keccak_sponge_program(CHUNKED_SPONGE_LEN).with_keccak_chunking(test_chunking())
}

fn chunked_sponge_msg() -> Vec<u8> {
    (0..CHUNKED_SPONGE_LEN)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
        .collect()
}

/// The permutation-level round operations a program's execution produces —
/// the same list `build_traces` chunks, rebuilt here so the tamper tests can
/// re-chunk it by hand.
fn round_ops_of(
    program: &LfmProgram,
    arenas: &[Vec<LfmWord>],
) -> Vec<crate::tables::keccak_rnd::KeccakRoundOperation> {
    let exec = super::executor::execute(program, arenas, &super::hash::TestPermutation)
        .expect("honest execution");
    let ops: Vec<_> = exec
        .records
        .keccak
        .iter()
        .enumerate()
        .map(|(row, r)| keccak_adapter::KeccakAdapterOperation {
            tag: klayout::tag_for_row(row),
            input: r.perm_in,
        })
        .collect();
    keccak_adapter::round_operations(&ops)
}

/// Every registered program that HAS the keccak family is single-chunk under
/// the default policy, so the production path is unchanged by this feature —
/// chunking is dormant until a program exceeds 21,845 permutations.
///
/// A program without the family carries zero instances, not one. The policy's
/// own floor of one exists to keep an unused chip present, and `ChipSet` is
/// what retired that: post-flip the authenticating programs hash with BLAKE3
/// and have no keccak work at all, so a KECCAK_RND instance would be a whole
/// AIR — commitment, tree, opening, FRI — carrying nothing.
#[test]
fn registered_programs_are_single_chunk() {
    for entry in super::registry::LFM_REGISTRY {
        let expected = usize::from(entry.chip_set.keccak);
        assert_eq!(
            entry.keccak_rnd_chunks, expected,
            "{:?} is registered with a chunk count its chip set does not imply",
            entry.kind
        );
    }
}

/// ★ The registry is six rows and every `(kind, blowup_factor)` is unique.
///
/// `resolve` keys on that pair alone, so a seventh row for a pair that already
/// exists would be unreachable data — and `compute_lfm_registry`'s doc promises
/// exactly that shape ("a second hasher becomes additional rows"). `resolve`
/// now rejects the ambiguity rather than taking the first match; this is the
/// half that makes ADDING a row a deliberate act instead of a silent widening,
/// since nothing else in the crate pinned the table's length.
#[test]
fn the_registry_is_six_unambiguous_rows() {
    use super::registry::{LFM_REGISTRY, LfmRegistryError, resolve};

    assert_eq!(
        LFM_REGISTRY.len(),
        6,
        "one entry per LfmProgramKind at the single registered blowup; adding \
         rows is a re-blessing decision, not a regeneration side effect"
    );

    for (i, entry) in LFM_REGISTRY.iter().enumerate() {
        for other in LFM_REGISTRY.iter().skip(i + 1) {
            assert!(
                entry.kind != other.kind || entry.blowup_factor != other.blowup_factor,
                "{:?} at blowup {} appears twice — `resolve` keys on this pair",
                entry.kind,
                entry.blowup_factor
            );
        }
        // The honest control: every row is reachable through the real lookup.
        assert!(
            resolve(entry.kind, entry.blowup_factor).is_ok(),
            "{:?} must resolve",
            entry.kind
        );
    }

    // And the miss is still a hard error, not a fallback.
    assert!(matches!(
        resolve(super::registry::LfmProgramKind::TrivialV0, 8),
        Err(LfmRegistryError::UnknownProgram {
            blowup_factor: 8,
            ..
        })
    ));
}

/// Every `HasherKind` there is. Not derived — a new candidate must be added
/// here by hand, which is the point: the two tests below are what say a new
/// hasher gets its own program identity rather than sharing one.
const ALL_HASHERS: [super::hash::HasherKind; 3] = [
    super::hash::HasherKind::Test,
    super::hash::HasherKind::Poseidon,
    super::hash::HasherKind::Blake3,
];

/// ★ Each registered entry's digest is bound to the hasher the entry names,
/// and to no other.
///
/// The first assertion is the honest control: the stored `program_id` really is
/// what the stored `(roots, heights, chunks, hasher)` derive, so the table is
/// internally consistent. The second is the property — swapping *only* the
/// hasher, with every root held fixed, must move the digest. Held together they
/// say a Test-backed and a Poseidon-backed machine of the same program are two
/// programs, which is what stops a verifier from pairing one hasher's digest
/// with another hasher's AIR set.
#[test]
fn every_registry_entry_binds_its_hasher_into_its_digest() {
    for entry in super::registry::LFM_REGISTRY {
        assert_eq!(
            super::statement::lfm_program_id(
                &entry.roots,
                &entry.log_heights,
                entry.keccak_rnd_chunks,
                entry.hasher,
                entry.chip_set,
            ),
            entry.program_id,
            "{:?}: the stored digest must be what the stored shape derives",
            entry.kind
        );
        for other in ALL_HASHERS {
            if other == entry.hasher {
                continue;
            }
            assert_ne!(
                super::statement::lfm_program_id(
                    &entry.roots,
                    &entry.log_heights,
                    entry.keccak_rnd_chunks,
                    other,
                    entry.chip_set,
                ),
                entry.program_id,
                "{:?}: {other:?} must not share {:?}'s program identity",
                entry.kind,
                entry.hasher
            );
        }
    }
}

/// The registry-path consequence: `lfm_verify` builds its AIR set from the
/// entry's hasher, so an honest proof of a registered program verifies, and the
/// same proof against the same entry's roots and digest under any *other*
/// hasher does not.
///
/// The accept half is not decoration. A verifier that rejected everything would
/// pass the reject half on its own, so the pair is what makes this a binding
/// test rather than a "does it say no" test.
#[test]
fn the_registry_hasher_is_what_verify_builds() {
    let opts = options();
    let program = trivial_program();
    let entry = resolve(LfmProgramKind::TrivialV0, opts.blowup_factor).expect("registered");
    let artifacts = build_artifacts(&program, &opts);
    let proved = lfm_prove(&program, &artifacts, &arenas(), &opts).expect("prove");

    assert!(
        lfm_verify(
            LfmProgramKind::TrivialV0,
            &proved.proof,
            &proved.public_words,
            &opts
        )
        .expect("registered"),
        "the honest proof must verify under the hasher the entry names"
    );

    for other in ALL_HASHERS {
        if other == entry.hasher {
            continue;
        }
        assert!(
            !verify_against(
                &entry.roots,
                &entry.program_id,
                entry.keccak_rnd_chunks,
                &proved.proof,
                &proved.public_words,
                &opts,
                other,
                entry.chip_set,
            ),
            "the entry's own proof must not verify under {other:?}"
        );
    }
}

/// The split's shape: chunk count, per-chunk permutation counts, per-chunk
/// trace heights, AIR count and trace count all agree.
#[test]
fn chunking_splits_the_sponge_into_two_uneven_chunks() {
    let program = chunked_sponge_program();
    assert_eq!(
        program.groups.keccak.real_rows, 3,
        "a {CHUNKED_SPONGE_LEN}-byte message must be 3 rate blocks"
    );

    assert_eq!(keccak_rnd_chunk_permutations(&program), vec![2, 1]);
    // 2 permutations = 48 rows → 64; 1 permutation = 24 rows → 32.
    assert_eq!(keccak_rnd_chunk_rows(&program), vec![64, 32]);

    let artifacts = build_artifacts(&program, &options());
    assert_eq!(artifacts.keccak_rnd_chunks, 2);
    assert_eq!(num_lfm_airs(2), super::NUM_LFM_CHIPS + 1);

    let exec = super::executor::execute(
        &program,
        &sponge_arenas(&chunked_sponge_msg()),
        &super::hash::TestPermutation,
    )
    .expect("honest execution");
    let traces = build_traces(&program, &exec.records);
    assert_eq!(traces.keccak_rnd.len(), 2, "one KECCAK_RND trace per chunk");
    assert_eq!(
        traces
            .keccak_rnd
            .iter()
            .map(|t| t.num_rows())
            .collect::<Vec<_>>(),
        vec![64, 32],
        "chunk traces must match the heights the artifacts predict"
    );
}

/// ★ The acceptance test: a program needing more than one `KECCAK_RND` chunk
/// proves and verifies end to end, and its digest still matches the production
/// hasher.
#[test]
fn chunked_sponge_proves_and_verifies() {
    let opts = options();
    let msg = chunked_sponge_msg();
    let program = chunked_sponge_program();
    let artifacts = build_artifacts(&program, &opts);
    assert_eq!(artifacts.keccak_rnd_chunks, 2, "this test needs 2 chunks");

    let proved = lfm_prove(&program, &artifacts, &sponge_arenas(&msg), &opts).expect("prove");
    assert_eq!(
        digest_bytes(&proved.public_words),
        keccak_host::keccak256(&msg),
        "a chunked proof must hash the same as the production hasher"
    );
    assert_eq!(
        stark::proof::view::MultiProofView::Owned(&proved.proof).len(),
        artifacts.chip_set.num_airs(2),
        "the proof must carry one sub-proof per AIR instance"
    );
    assert!(
        verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &proved.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
        ),
        "a two-chunk KECCAK_RND proof must verify"
    );
}

/// Chunking is a prover-side layout choice, not a semantic one: the same
/// message proved at 1 and at 2 chunks yields the same public output. (The
/// program *identity* does differ — the chunk count is bound into the digest —
/// which is exactly why the two need different artifacts.)
#[test]
fn chunking_does_not_change_what_is_proved() {
    let opts = options();
    let msg = chunked_sponge_msg();

    let one = keccak_sponge_program(CHUNKED_SPONGE_LEN);
    let one_artifacts = build_artifacts(&one, &opts);
    assert_eq!(one_artifacts.keccak_rnd_chunks, 1);
    let one_proof = lfm_prove(&one, &one_artifacts, &sponge_arenas(&msg), &opts).expect("prove");

    let two = chunked_sponge_program();
    let two_artifacts = build_artifacts(&two, &opts);
    let two_proof = lfm_prove(&two, &two_artifacts, &sponge_arenas(&msg), &opts).expect("prove");

    assert_eq!(
        one_proof.public_words, two_proof.public_words,
        "chunking must not change the program's output"
    );
    assert_eq!(
        one_artifacts.roots, two_artifacts.roots,
        "chunking must not move any preprocessed root"
    );
    assert_ne!(
        one_artifacts.program_id, two_artifacts.program_id,
        "the chunk count is program shape and must be bound into the digest"
    );
    for (artifacts, proof) in [(&one_artifacts, &one_proof), (&two_artifacts, &two_proof)] {
        assert!(
            verify_against(
                &artifacts.roots,
                &artifacts.program_id,
                artifacts.keccak_rnd_chunks,
                &proof.proof,
                &proof.public_words,
                &opts,
                artifacts.hasher,
                artifacts.chip_set,
            ),
            "both chunkings must verify against their own artifacts"
        );
    }
}

/// ★ Tamper: corrupting a permutation that lives in the *second* chunk must
/// reject. The first chunk is untouched, so this only rejects if chunk 1's
/// rows are really part of the proof's bus balance.
#[test]
fn tampered_second_chunk_permutation_rejects() {
    let opts = options();
    let msg = chunked_sponge_msg();
    let program = chunked_sponge_program();
    let artifacts = build_artifacts(&program, &opts);
    let exec = super::executor::execute(
        &program,
        &sponge_arenas(&msg),
        &super::hash::TestPermutation,
    )
    .expect("honest execution");

    let mut traces = build_traces(&program, &exec.records);
    assert_eq!(traces.keccak_rnd.len(), 2);
    // Byte 0 of lane (0,0) on the second chunk's first row: the `Keccak`
    // receive token no longer matches the send that fed it.
    let col = keccak_rnd::cols::start(0, 0, 0);
    let old = traces.keccak_rnd[1].main_table.get_row(0)[col];
    traces.keccak_rnd[1]
        .main_table
        .set_fe(0, col, old + FE::from(1u64));

    let proof =
        prove_traces(&artifacts, &mut traces, &exec.public_words, &opts).expect("prover accepts");
    assert!(
        !verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proof,
            &exec.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
        ),
        "a corrupted permutation in the second chunk must reject"
    );
}

/// ★ Falsifies the split itself: drop the permutation the second chunk holds.
/// The `LFM_KECCAK` chip still sends its request token, so the `Keccak` bus is
/// left with a send that nothing receives. If this ever accepts, chunks are
/// not actually contributing their rows to the balance.
#[test]
fn dropping_the_second_chunks_permutation_rejects() {
    let opts = options();
    let msg = chunked_sponge_msg();
    let program = chunked_sponge_program();
    let artifacts = build_artifacts(&program, &opts);
    let exec = super::executor::execute(
        &program,
        &sponge_arenas(&msg),
        &super::hash::TestPermutation,
    )
    .expect("honest execution");

    let mut traces = build_traces(&program, &exec.records);
    // Same chunk COUNT — so the AIR set and the digest still match — but the
    // last chunk is now empty.
    traces.keccak_rnd[1] = keccak_rnd::generate_keccak_rnd_trace(&[]);

    let proof =
        prove_traces(&artifacts, &mut traces, &exec.public_words, &opts).expect("prover accepts");
    assert!(
        !verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proof,
            &exec.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
        ),
        "a chunk missing its permutation must reject"
    );
}

/// The mechanism's foundation, stated positively: which chunk a permutation
/// lands in is free. `KECCAK_RND` has no row-to-row constraints and its rounds
/// are linked by `Keccak` bus tokens rather than row adjacency, so LogUp
/// cannot tell a 2+1 split from a 1+2 one. This is why chunking needs no
/// pairing logic — and if it ever fails, the round chip has grown a
/// cross-row dependency that chunking would silently break.
#[test]
fn permutations_may_be_reassigned_across_chunk_boundaries() {
    let opts = options();
    let msg = chunked_sponge_msg();
    let program = chunked_sponge_program();
    let artifacts = build_artifacts(&program, &opts);
    let exec = super::executor::execute(
        &program,
        &sponge_arenas(&msg),
        &super::hash::TestPermutation,
    )
    .expect("honest execution");

    let round_ops = round_ops_of(&program, &sponge_arenas(&msg));
    assert_eq!(round_ops.len(), 3);

    let mut traces = build_traces(&program, &exec.records);
    // Canonical split is 2 + 1; re-split as 1 + 2.
    traces.keccak_rnd[0] = keccak_rnd::generate_keccak_rnd_trace(&round_ops[..1]);
    traces.keccak_rnd[1] = keccak_rnd::generate_keccak_rnd_trace(&round_ops[1..]);

    let proof =
        prove_traces(&artifacts, &mut traces, &exec.public_words, &opts).expect("prover accepts");
    assert!(
        verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proof,
            &exec.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
        ),
        "chunk assignment is free — a 1+2 split proves the same statement as 2+1"
    );
}

/// The verifier builds its AIR set from the supplied chunk count, so a count
/// that disagrees with the proof's shape must be rejected — including zero,
/// which would drop `KECCAK_RND` and its constraints from the set.
///
/// Two layers enforce this and the test does not distinguish them: the
/// explicit length check in `verify_against`, and the framework's own
/// AIR-count handling. Measured: deleting the explicit check leaves this test
/// green, so it pins the *behaviour*, not that particular guard. The guard
/// stays because it makes the shape contract local and legible, not because
/// this test would catch its removal.
#[test]
fn verify_rejects_a_chunk_count_that_does_not_match_the_proof() {
    let opts = options();
    let msg = chunked_sponge_msg();
    let program = chunked_sponge_program();
    let artifacts = build_artifacts(&program, &opts);
    let proved = lfm_prove(&program, &artifacts, &sponge_arenas(&msg), &opts).expect("prove");

    for wrong in [0usize, 1, 3, 14] {
        assert!(
            !verify_against(
                &artifacts.roots,
                &artifacts.program_id,
                wrong,
                &proved.proof,
                &proved.public_words,
                &opts,
                artifacts.hasher,
                artifacts.chip_set,
            ),
            "chunk count {wrong} must not verify a 2-chunk proof"
        );
    }
}

/// What chunking costs: `KECCAK_RND` pads each chunk to its own power of two,
/// so the only overhead is padding, and splitting can even reduce it.
#[test]
fn chunking_cell_cost() {
    let one = keccak_sponge_program(CHUNKED_SPONGE_LEN);
    let two = chunked_sponge_program();
    let perms = one.groups.keccak.real_rows as u64;

    let (main_one, aux_one) = super::airs::lfm_cell_counts(&one);
    let (main_two, aux_two) = super::airs::lfm_cell_counts(&two);
    println!(
        "{CHUNKED_SPONGE_LEN}-byte sponge, {perms} permutations:\n  \
         1 chunk  rows {:?} main {main_one} aux {aux_one}\n  \
         2 chunks rows {:?} main {main_two} aux {aux_two}\n  \
         delta main {} aux {}",
        keccak_rnd_chunk_rows(&one),
        keccak_rnd_chunk_rows(&two),
        main_two as i64 - main_one as i64,
        aux_two as i64 - aux_one as i64,
    );

    // KECCAK_RND rows are the only thing chunking moves; here 128 padded rows
    // in one chunk versus 64 + 32 in two.
    assert_eq!(keccak_rnd_chunk_rows(&one).iter().sum::<usize>(), 128);
    assert_eq!(keccak_rnd_chunk_rows(&two).iter().sum::<usize>(), 96);
    assert!(
        main_two < main_one,
        "this split lands on tighter power-of-two boundaries, so it is cheaper"
    );
}

/// At the default policy's geometry chunking does not cost rows, it saves
/// them. A single table must pad to one power of two for the whole program; N
/// chunks each pad to their own, and every full chunk is within 8 rows of its
/// power of two by construction.
#[test]
fn default_policy_beats_a_single_table_at_wrap_scale() {
    let c = KeccakChunking::default();
    let per = c.permutations_per_chunk();
    let full_chunk_rows = (per * 24).next_power_of_two();
    assert_eq!(full_chunk_rows, 1 << 19);
    assert_eq!(
        full_chunk_rows - per * 24,
        8,
        "a full chunk wastes 8 rows of 524,288"
    );

    // The proof wrap this feature exists for.
    const WRAP_PERMUTATIONS: usize = 460_000;
    let chunks = c.chunk_count(WRAP_PERMUTATIONS);
    assert_eq!(chunks, 22, "21 full chunks plus a partial one");

    let chunked_rows: usize = (0..chunks)
        .map(|i| {
            let perms = WRAP_PERMUTATIONS.saturating_sub(i * per).min(per);
            (perms * 24).next_power_of_two().max(4)
        })
        .sum();
    let single_table_rows = (WRAP_PERMUTATIONS * 24).next_power_of_two();

    println!(
        "{WRAP_PERMUTATIONS} permutations: {chunks} chunks = {chunked_rows} rows, \
         single table = {single_table_rows} rows ({:.1}% saved)",
        100.0 * (1.0 - chunked_rows as f64 / single_table_rows as f64),
    );
    assert!(
        chunked_rows < single_table_rows,
        "chunking must not cost more rows than one table would"
    );
    // 2^24 rows at 1480 columns is also far past what one table can hold.
    assert_eq!(single_table_rows, 1 << 24);
}

// ================= R1f slice b: real continuation-proof bytes =================

use super::proof_fixture;

/// Cache path for the fixture blob. Outside the repository on purpose: a
/// checked-in binary can drift from the encoder silently, so the generation path
/// is what a cold run exercises.
///
/// ★ **Keyed on the configuration, not only on the name.** `load_or_generate`
/// hands back whatever sits at this path if the file exists, so a fixed name
/// meant the first run after any fixture change read the PREVIOUS
/// configuration's proof — four tests below failing on epoch count and passing
/// on a second run, which reads as flakiness rather than as a stale blob. Both
/// inputs that decide the blob's shape are in the name, so a change to either
/// simply misses the cache.
fn fixture_cache() -> std::path::PathBuf {
    let elf = proof_fixture::FIXTURE_INNER_ELF.replace(['/', '\\', '.'], "_");
    // The format key keeps branches that encode differently off each other's
    // blob — see `proof_fixture::cache_format_key`.
    let fmt = proof_fixture::cache_format_key().replace(['/', '\\', '.'], "_");
    std::env::temp_dir().join(format!(
        "lfm-r1f-continuation-fixture-{elf}-log2_{}-{fmt}.bin",
        proof_fixture::FIXTURE_EPOCH_LOG2
    ))
}

/// R1f(b): the machine's fixture is a REAL two-epoch continuation proof, encoded
/// by the same function that builds the recursion guest's private input.
#[test]
fn continuation_fixture_generates_two_epochs() {
    let (blob, num_epochs) = proof_fixture::generate();
    println!(
        "R1f fixture: inner={} epoch_log2={} epochs={} blob={} bytes",
        proof_fixture::FIXTURE_INNER_ELF,
        proof_fixture::FIXTURE_EPOCH_LOG2,
        num_epochs,
        blob.len()
    );
    assert!(
        proof_fixture::has_recursion_prefix(&blob),
        "the blob must carry the recursion input wire format's magic prefix"
    );
    assert!(
        num_epochs >= 2,
        "a CONTINUATION fixture needs more than one epoch, got {num_epochs} — \
         lower FIXTURE_EPOCH_LOG2"
    );
    // Cache it for the slices that consume it. Atomic: this test runs in
    // parallel with readers of the same path, and blobs are not reproducible,
    // so a torn write would hand someone a truncated proof.
    proof_fixture::write_cache(&fixture_cache(), &blob);
}

/// R1f(a): the arena filler reads a REAL proof's committed roots out of the
/// guest's wire-format blob, in place, exactly as the recursion guest would.
#[test]
fn arena_filler_reads_real_committed_roots() {
    use super::proof_arena;
    use super::proof_fixture::FixtureArchive;

    let blob = proof_fixture::load_or_generate(&fixture_cache());
    let archive = FixtureArchive::open(&blob);

    let epochs = proof_arena::num_epochs(&archive);
    assert_eq!(epochs, 2, "the fixture is a two-epoch continuation");

    for epoch in 0..epochs {
        let tables = proof_arena::epoch_num_tables(&archive, epoch);
        let roots = proof_arena::epoch_main_roots(&archive, epoch);
        assert_eq!(roots.len(), tables, "one main root per sub-proof");
        assert!(tables > 0, "epoch {epoch} must have sub-proofs");
        // Real commitments, not defaults: an all-zero root would mean the reader
        // is looking at the wrong bytes rather than at the proof.
        assert!(
            roots.iter().all(|r| *r != [0u8; 32]),
            "epoch {epoch}: every committed root must be nonzero"
        );
        let halves = proof_arena::roots_to_halves(&roots);
        assert_eq!(halves.len(), tables * proof_arena::ROOT_HALVES);
        println!(
            "R1f arena: epoch {epoch} -> {tables} sub-proofs, {} arena halves, output {} bytes",
            halves.len(),
            proof_arena::epoch_public_output(&archive, epoch).len()
        );
    }
}

/// Verifies the team lead's ruling premise directly against the blob: the
/// SUPPLIED preprocessed roots really are embedded, so replaying Phase A does
/// not need `build_epoch_airs` reachable.
///
/// Checked here rather than taken on trust, because the whole leg's shape
/// depends on it.
#[test]
fn supplied_preprocessed_roots_are_embedded_in_the_blob() {
    use super::proof_fixture::FixtureArchive;

    let blob = proof_fixture::load_or_generate(&fixture_cache());
    let archive = FixtureArchive::open(&blob);
    let gi = archive.guest_input();

    // DECODE: one commitment, directly in the guest input.
    assert_ne!(
        gi.decode_commitment, [0u8; 32],
        "the DECODE root must be embedded and nonzero"
    );
    // Per-page genesis roots: (base, commitment) pairs, also directly embedded.
    println!(
        "R1f supplied roots: decode present, {} page commitments",
        gi.page_commitments.len()
    );
    for pair in gi.page_commitments.iter() {
        assert_ne!(pair.1, [0u8; 32], "page genesis roots must be nonzero");
    }
}

// ============ R1f (c)+(d): a REAL Merkle opening, in the machine ============
//
// Everything up to here ran on data this machine produced. This is the first
// leg that authenticates production-committed data: one FRI query's main-trace
// opening from a real two-epoch continuation proof, walked under the production
// keccak Merkle conventions, against that proof's own committed root.
//
// The oracle is the proof's root. Nothing here recomputes an expected answer
// with a local model and compares the machine against itself.

use super::programs::{MerkleOpeningShape, keccak_merkle_opening_program, merkle_opening_program};
use super::proof_arena::MainTraceOpening;

/// Which opening the leg authenticates.
///
/// Epoch 0's first sub-proof, chosen on measured grounds. Two things make it
/// the right target, and only the first is stable across blobs.
///
/// **Depth.** It is one of exactly two depth-20 trees in the fixture (the other
/// is epoch 1's table 0); everything else is depth 7 or less, and half the
/// sub-proofs are depth 2. Depth is SHAPE, so it does not move when the blob
/// does — see `fixture_generation_is_not_reproducible`.
///
/// **A unique leaf index.** Measured on one blob, 24 of the 49 sub-proofs have
/// exactly one index that verifies and 25 have several: a table whose trace is
/// mostly padding commits identical rows, so identical leaves sit under
/// identical subtrees and every index checks out. On one of those, "flip an
/// index bit" is not a tamper at all and the (d) vector would pass while
/// testing nothing. That split is blob-dependent, so it is NOT pinned as a
/// constant — `real_opening_is_a_usable_tamper_target` asserts uniqueness at
/// run time on whatever blob it is handed.
const R1F_EPOCH: usize = 0;
const R1F_TABLE: usize = 0;
const R1F_QUERY: usize = 0;

/// The pinned shape, asserted against the real proof rather than read from it —
/// program shape is compile-time by construction, so if the fixture ever moves,
/// this must fail loudly rather than quietly recompile to a new program.
const R1F_SHAPE: MerkleOpeningShape = MerkleOpeningShape {
    leaf_values: 20,
    depth: 20,
};

/// The opening and its recovered leaf index, resolved once per test binary.
///
/// The index costs a `2^depth` sweep (~4 s at depth 20) because `iota` is a
/// transcript challenge and is not in the proof; see
/// [`MainTraceOpening::indices_that_verify`]. Sharing it across the tests that
/// need it keeps that to one sweep.
fn r1f_opening() -> &'static (MainTraceOpening, usize) {
    use std::sync::OnceLock;
    static CELL: OnceLock<(MainTraceOpening, usize)> = OnceLock::new();
    CELL.get_or_init(|| {
        let blob = proof_fixture::load_or_generate(&fixture_cache());
        let archive = super::proof_fixture::FixtureArchive::open(&blob);
        let opening = MainTraceOpening::extract(&archive, R1F_EPOCH, R1F_TABLE, R1F_QUERY);
        let hits = opening.indices_that_verify();
        assert_eq!(
            hits.len(),
            1,
            "the authenticated opening must sit at exactly one index, else the \
             index-tamper vector tests nothing; got {hits:?}"
        );
        (opening, hits[0])
    })
}

fn merkle_arenas(opening: &MainTraceOpening, index: usize) -> Vec<Vec<LfmWord>> {
    vec![
        opening.leaf_arena(),
        opening.sibling_arena(),
        vec![super::word::base_word(FE::from(index as u64))],
        opening.root_arena(),
    ]
}

/// Scrutinises the oracle before anything is built on it: the opening really is
/// what the leg assumes, and PRODUCTION's own path check accepts it.
#[test]
fn real_opening_is_a_usable_tamper_target() {
    let (opening, index) = r1f_opening();
    assert_eq!(
        opening.depth(),
        R1F_SHAPE.depth,
        "the fixture's tree depth moved; R1F_SHAPE is program shape and must be updated deliberately"
    );
    assert_eq!(
        opening.values.len(),
        R1F_SHAPE.leaf_values,
        "the fixture's column count moved"
    );
    assert_eq!(opening.num_columns, R1F_SHAPE.columns());
    assert!(
        opening.verifies_at(*index),
        "production's own checker must accept the opening we are about to \
         authenticate in the machine"
    );
    assert!(
        !opening.verifies_at(index ^ 1),
        "flipping the low index bit must break production's check"
    );
    println!(
        "R1f target: epoch {R1F_EPOCH} table {R1F_TABLE} query {R1F_QUERY} — \
         {} columns, row pair = {} values, depth {}, index {index}",
        opening.num_columns,
        opening.values.len(),
        opening.depth()
    );
}

/// ★ The headline: the machine walks a real opening to a real committed root,
/// PROVED and verified.
///
/// Two independent things are checked. The published root equals the root the
/// proof committed to — that is the authentication, and its oracle is the proof
/// itself. And the machine proof verifies against those published words — that
/// is what makes it a proof rather than an execution, which per method rule 2
/// is the only thing that says anything about the chips.
#[test]
fn the_merkle_walk_authenticates_a_real_opening() {
    let opts = options();
    let (opening, index) = r1f_opening();
    // The PRODUCTION twin: the opening is real, so the walk has to re-derive a
    // root the host actually built. The keccak instrument cannot, and the name
    // moved with the hash rather than outliving it.
    let program = merkle_opening_program(R1F_SHAPE);
    let artifacts = build_artifacts(&program, &opts);
    let proved = lfm_prove(&program, &artifacts, &merkle_arenas(opening, *index), &opts)
        .expect("the honest opening must execute and prove");

    assert_eq!(
        digest_bytes(&proved.public_words),
        opening.root,
        "the walked root must be the root the proof committed to"
    );
    assert!(
        verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &proved.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
        ),
        "the authenticated opening must verify"
    );
}

/// One tamper vector: corrupted arenas, plus the root those arenas really fold
/// to — which is what lets the same vector be run both incoherently (claiming
/// the real root) and coherently (claiming its own).
struct TamperVector {
    what: &'static str,
    arenas: Vec<Vec<LfmWord>>,
    root: [u8; 32],
}

/// ★ (d) Tamper, both ways round, for all three inputs the walk consumes.
///
/// INCOHERENT: change one input and leave the claimed root alone. The
/// in-machine root assert makes the program unexecutable — the earliest and
/// loudest failure, and the one that shows the assert is load-bearing.
///
/// COHERENT (method rule 4): change the input AND supply the root that input
/// really folds to, so every value in the run is consistent with every other,
/// nothing asserts, and a proof comes out. The forgery then fails on the one
/// thing it cannot fake — the published root is not the root the proof
/// committed to, so a verifier claiming the real one rejects.
#[test]
fn tampered_merkle_opening_rejects() {
    let opts = options();
    let (opening, index) = r1f_opening();
    // Same production twin as the honest-path test above — a tamper control is
    // only a control over the walk the honest path uses.
    let program = merkle_opening_program(R1F_SHAPE);
    let artifacts = build_artifacts(&program, &opts);
    let honest = lfm_prove(&program, &artifacts, &merkle_arenas(opening, *index), &opts)
        .expect("honest prove");

    let mut vectors: Vec<TamperVector> = Vec::new();

    // 1. A wrong sibling at the leaf level.
    {
        let mut siblings = opening.siblings.clone();
        siblings[0][0] ^= 1;
        let mut arenas = merkle_arenas(opening, *index);
        arenas[1] = siblings
            .iter()
            .flat_map(super::proof_arena::commitment_words)
            .collect();
        let root = super::proof_arena::walk_to_root(opening.leaf_hash(), *index, &siblings);
        vectors.push(TamperVector {
            what: "wrong sibling",
            arenas,
            root,
        });
    }

    // 2. Wrong index bits: the same leaf and the same path, walked in the other
    //    order at level 0.
    {
        let bad = index ^ 1;
        let arenas = merkle_arenas(opening, bad);
        let root = super::proof_arena::walk_to_root(opening.leaf_hash(), bad, &opening.siblings);
        vectors.push(TamperVector {
            what: "wrong index bits",
            arenas,
            root,
        });
    }

    // 3. A wrong opened value: one field element of the row pair.
    {
        let mut tampered = MainTraceOpening {
            root: opening.root,
            values: opening.values.clone(),
            num_columns: opening.num_columns,
            siblings: opening.siblings.clone(),
        };
        tampered.values[0] = &tampered.values[0] + FE::from(1u64);
        let mut arenas = merkle_arenas(opening, *index);
        arenas[0] = tampered.leaf_arena();
        let root =
            super::proof_arena::walk_to_root(tampered.leaf_hash(), *index, &tampered.siblings);
        vectors.push(TamperVector {
            what: "wrong leaf value",
            arenas,
            root,
        });
    }

    for TamperVector {
        what,
        arenas,
        root: forged_root,
    } in vectors
    {
        assert_ne!(
            forged_root, opening.root,
            "{what}: the tamper must actually move the root, or the vector is vacuous"
        );

        // Incoherent: still claiming the real root.
        let err = super::executor::execute(&program, &arenas, &super::hash::TestPermutation)
            .err()
            .unwrap_or_else(|| panic!("{what}: claiming the real root must not execute"));
        println!("R1f tamper {what}: incoherent run rejected with {err:?}");

        // Coherent: claim the root the tampered inputs really reach.
        let mut coherent = arenas;
        coherent[3] = super::proof_arena::commitment_words(&forged_root).to_vec();
        let proved = lfm_prove(&program, &artifacts, &coherent, &opts)
            .unwrap_or_else(|e| panic!("{what}: the coherent forgery must prove: {e:?}"));
        assert_eq!(
            digest_bytes(&proved.public_words),
            forged_root,
            "{what}: the coherent forgery must publish its own root"
        );
        assert_ne!(
            proved.public_words, honest.public_words,
            "{what}: the forgery must not publish the honest root"
        );
        assert!(
            !verify_against(
                &artifacts.roots,
                &artifacts.program_id,
                artifacts.keccak_rnd_chunks,
                &proved.proof,
                &honest.public_words,
                &opts,
                artifacts.hasher,
                artifacts.chip_set,
            ),
            "{what}: claiming the real committed root for a forged walk must reject"
        );
    }
}

/// Main-trace cells one byteswap costs: one `LFM_BITDEC` row and 64 `LFM_BALU`
/// rows, each at its chip's non-preprocessed width — the same accounting
/// [`super::airs::lfm_cell_counts`] uses.
pub(super) fn byteswap_cells() -> u64 {
    use super::chips::{balu, bitdec};
    use super::layout;
    let bitdec_w = (bitdec::cols::NUM_COLUMNS - layout::bitdec::PREP_WIDTH) as u64;
    let balu_w = (balu::cols::NUM_COLUMNS - layout::balu::PREP_WIDTH) as u64;
    bitdec_w + 64 * balu_w
}

/// ★ Rows the CONFIGURED wrap hash occupies in a program.
///
/// Structural counts — "a Merkle tree over `n` leaves costs `2n − 1` hashes" —
/// are claims about tree shape, not about a hash, so they must be counted
/// against whichever chip group the configured hash fills. Reading
/// `groups.keccak` directly made them silently read ZERO the moment production
/// moved to BLAKE3, which reports a true structural claim as a failed one.
pub(super) fn wrap_hash_rows(program: &super::compiler::LfmProgram) -> usize {
    match super::edsl::WrapHash::production() {
        super::edsl::WrapHash::Keccak => program.groups.keccak.real_rows,
        super::edsl::WrapHash::Blake3 => program.groups.blake3.real_rows,
    }
}

/// [`wrap_hash_rows`]'s instruction-stream twin: emitted compressions of the
/// configured wrap hash.
pub(super) fn wrap_hash_instrs(program: &super::compiler::LfmProgram) -> usize {
    use super::instr::Instr;
    program
        .instrs
        .iter()
        .filter(|i| match super::edsl::WrapHash::production() {
            super::edsl::WrapHash::Keccak => matches!(i, Instr::KeccakF(_)),
            super::edsl::WrapHash::Blake3 => matches!(i, Instr::Blake3(_)),
        })
        .count()
}

/// Main-trace cells one keccak permutation costs: the `LFM_KECCAK` row that
/// requests it, plus the 24 `KECCAK_RND` rounds that carry it.
pub(super) fn permutation_cells() -> u64 {
    use super::chips::keccak;
    use super::chunking::KECCAK_RND_ROWS_PER_PERMUTATION as ROUNDS;
    use super::layout;
    use crate::tables::keccak_rnd;
    let keccak_w = (keccak::cols::NUM_COLUMNS - layout::keccak::PREP_WIDTH) as u64;
    keccak_w + ROUNDS as u64 * keccak_rnd::cols::NUM_COLUMNS as u64
}

/// Pins the widths the cost model above is built on.
///
/// Not ceremony: the inline `// 52 / 252 / 388 / 588 / 788` comments on
/// `chips::keccak::cols` were stale by 4 (R1d widened `PREP_WIDTH` for the
/// reversed-digest columns and they were not updated), and reading them instead
/// of the constants is what produced a wrong per-permutation figure on the first
/// pass through this measurement. A wrong width silently rescales every cell
/// number in `keccak_merkle_opening_cost`, so the widths get an assertion of
/// their own rather than a comment.
#[test]
fn cost_model_widths_are_what_the_chips_declare() {
    use super::chips::{balu, bitdec, keccak};
    use super::layout;
    use crate::tables::keccak_rnd;
    assert_eq!(keccak::cols::NUM_COLUMNS, 792, "LFM_KECCAK total width");
    assert_eq!(layout::keccak::PREP_WIDTH, 56, "LFM_KECCAK preprocessed");
    assert_eq!(balu::cols::NUM_COLUMNS - layout::balu::PREP_WIDTH, 4);
    assert_eq!(bitdec::cols::NUM_COLUMNS - layout::bitdec::PREP_WIDTH, 66);
    assert_eq!(keccak_rnd::cols::NUM_COLUMNS, 1480);
    assert_eq!(byteswap_cells(), 322, "66 + 64 x 4");
    assert_eq!(permutation_cells(), 36_256, "736 + 24 x 1480");
}

/// ★ The leg's headline measurement — and it REFUTES the prediction it was set
/// up to confirm.
///
/// The R1f handoff predicted that byteswapping the opened values would dominate
/// the leaf, "not the hashing", on the strength of the row counts: a 10-column
/// table pays 20 `LFM_BITDEC` + 1280 `LFM_BALU` rows of byteswapping against
/// only 22 permutations. Those row counts are right. The conclusion drawn from
/// them is wrong, because rows of different chips are not comparable units.
///
/// A byteswap's rows are narrow — `LFM_BALU` carries 4 non-preprocessed columns
/// — while a permutation expands into 24 `KECCAK_RND` rounds at 1480 columns
/// each. Priced in main-trace cells, the unit the proof actually pays in, the
/// measured figures are **322 cells per byteswap against 36,256 per
/// permutation, a factor of 113**. Hashing then dominates at every width in the
/// fixture: 124× at the 10-column table this leg authenticates, 8.9× at 511
/// columns, 7.4× at 1480. The crossover this test was written to find does not
/// exist. Both terms are linear in the column count — `2c` byteswaps against
/// `≈16c/136` rate blocks — so the ratio flattens near 6.6× rather than
/// inverting.
///
/// This is why a byteswap chiplet is NOT the lever it looked like, and the
/// measurement rather than the intuition is what says so. The attribution is
/// MARGINAL (real rows, not padded), so it answers "what does one more column
/// cost" and not "what does this proof cost"; the whole-program figure is
/// printed alongside because the fixed floor — `BITWISE` is 2^20 rows whatever
/// the program does — dwarfs both terms at these sizes.
#[test]
fn keccak_merkle_opening_cost() {
    let (opening, _) = r1f_opening();
    println!(
        "one byteswap = {} main cells; one permutation = {} main cells ({:.0}x)",
        byteswap_cells(),
        permutation_cells(),
        permutation_cells() as f64 / byteswap_cells() as f64,
    );
    println!("shape                 instrs  keccak  bitdec    balu  select   lanes");
    let mut shapes = vec![R1F_SHAPE];
    // Two wider tables from the same fixture, to show the scaling rather than
    // assert a single point. 511 and 1480 columns are real widths in it.
    for columns in [511usize, 1480] {
        shapes.push(MerkleOpeningShape {
            leaf_values: 2 * columns,
            depth: R1F_SHAPE.depth,
        });
    }
    for shape in &shapes {
        let program = keccak_merkle_opening_program(*shape);
        println!(
            "{:>4} cols d={:<3}  {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}",
            shape.columns(),
            shape.depth,
            program.instrs.len(),
            program.groups.keccak.real_rows,
            program.groups.bitdec.real_rows,
            program.groups.balu.real_rows,
            program.groups.select.real_rows,
            program.groups.lanes.real_rows,
        );
    }

    // The same three shapes priced in main-trace cells, which is where the
    // prediction inverts. `swap` counts only the byteswapping; `hash` counts
    // every permutation (leaf blocks and walk levels alike).
    println!("shape                    swap cells   hash cells   hash/swap   whole program");
    for shape in &shapes {
        let program = keccak_merkle_opening_program(*shape);
        let swap = shape.leaf_values as u64 * byteswap_cells();
        let hash = program.groups.keccak.real_rows as u64 * permutation_cells();
        let (main, _aux) = super::airs::lfm_cell_counts(&program);
        println!(
            "{:>4} cols d={:<3} {:>12} {:>12} {:>11.1} {:>15}",
            shape.columns(),
            shape.depth,
            swap,
            hash,
            hash as f64 / swap as f64,
            main,
        );
        assert!(
            hash > swap,
            "{} columns: hashing must dominate — if this ever flips, the \
             byteswap-chiplet argument becomes live and the docs above are stale",
            shape.columns()
        );
    }

    // Pin the real shape's decomposition, so a regression in either half shows.
    let program = keccak_merkle_opening_program(R1F_SHAPE);
    let leaf_bytes = 8 * R1F_SHAPE.leaf_values;
    let leaf_perms = super::keccak_host::num_blocks(leaf_bytes);
    assert_eq!(
        program.groups.keccak.real_rows,
        leaf_perms + R1F_SHAPE.depth,
        "one permutation per rate block of the leaf, plus one per level"
    );
    assert_eq!(
        program.groups.bitdec.real_rows,
        R1F_SHAPE.leaf_values + 1,
        "one decomposition per opened value, plus one for the index"
    );
    assert_eq!(
        program.groups.balu.real_rows,
        64 * R1F_SHAPE.leaf_values + 8 * 2,
        "64 rows per byteswap, plus the two root asserts (4 sub + 4 div each)"
    );
    assert_eq!(
        program.groups.select.real_rows,
        2 * R1F_SHAPE.depth,
        "two selects per level: a digest is two words and both swap together"
    );
    println!(
        "R1f leaf: {} values -> {leaf_bytes} bytes -> {leaf_perms} permutations, \
         against {} bitdec + {} balu rows of byteswapping",
        R1F_SHAPE.leaf_values,
        R1F_SHAPE.leaf_values,
        64 * R1F_SHAPE.leaf_values,
    );
    // The fixed floor, for scale: BITWISE alone is 2^20 rows regardless of what
    // the program does, so nothing above is a claim about total proof cost.
    let (main, aux) = super::airs::lfm_cell_counts(&program);
    println!("R1f whole program: {main} main cells, {aux} aux cells");
    assert_eq!(opening.values.len(), R1F_SHAPE.leaf_values);
}

/// ⚠ STANDING EVIDENCE that the fixture is not reproducible.
///
/// Two `generate()` calls on identical inputs — same ELF, same empty input,
/// same epoch size, same options — produce different blobs, and the difference
/// is semantic rather than archive padding: sub-proof roots move, which moves
/// the Fiat-Shamir challenges, which opens different leaves. Measured at ~65k
/// of 587k bytes differing.
///
/// This is why R1f pins SHAPE (`R1F_SHAPE`) and RECOVERS per-blob values
/// (`r1f_opening`) instead of pinning a query index or a root. A pinned index
/// would pass for exactly as long as the cache survived and then fail on the
/// next cold run, which is the worst possible failure mode for a fixture.
///
/// `#[ignore]`d because it costs two full continuation proofs (~17 s) and
/// asserts nothing the rest of the suite depends on. Run it with
/// `--ignored` when the question comes up again; it is here so the claim has
/// evidence attached rather than being folklore in a status log.
#[test]
#[ignore]
fn fixture_generation_is_not_reproducible() {
    let (a, _) = proof_fixture::generate();
    let (b, _) = proof_fixture::generate();
    let differing = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
    println!(
        "two fixture generations: {} vs {} bytes, {differing} differing",
        a.len(),
        b.len()
    );
    assert_ne!(
        a, b,
        "if this ever passes, the prover became reproducible \
                      and the no-pinning rule in proof_fixture can be relaxed"
    );

    // The difference reaches the committed data, so it is not archive padding.
    let (aa, bb) = (
        super::proof_fixture::FixtureArchive::open(&a),
        super::proof_fixture::FixtureArchive::open(&b),
    );
    assert_ne!(
        super::proof_arena::epoch_main_roots(&aa, 0),
        super::proof_arena::epoch_main_roots(&bb, 0),
        "the divergence must be semantic — if the roots match, this is padding \
         and the no-pinning rule is too strong"
    );

    // The tree this leg authenticates keeps its SHAPE across runs, which is the
    // property R1F_SHAPE relies on, while its opened values do not.
    let oa = MainTraceOpening::extract(&aa, 0, 0, 0);
    let ob = MainTraceOpening::extract(&bb, 0, 0, 0);
    assert_eq!(
        oa.depth(),
        ob.depth(),
        "tree depth is shape and must be stable"
    );
    assert_eq!(
        oa.num_columns, ob.num_columns,
        "column count is shape and must be stable"
    );
    println!(
        "table 0 across runs: depth {} stable, same root = {}, same opened values = {}",
        oa.depth(),
        oa.root == ob.root,
        oa.values == ob.values
    );
}

// ============ R1g (ii): the cross-epoch L2G commitment binding ============
//
// The first obligation of the chaining leg, and the first time the machine
// reads ACROSS structures: each epoch's own committed L2G root against the
// corresponding sub-proof of the one global proof. R1f stayed inside a single
// epoch's own sub-proof.

use super::programs::l2g_binding_program;

/// Epochs the binding program is compiled for. SHAPE, not a blob-derived
/// constant: the epoch count follows from the inner ELF and
/// `FIXTURE_EPOCH_LOG2`, not from anything the prover chooses per run. Asserted
/// against the real bundle rather than read from it, so a fixture change is
/// loud — the same discipline `R1F_SHAPE` uses.
const R1G_EPOCHS: usize = 2;

/// The `i`-th 32-byte root in a program's published words.
fn published_root(public: &[(u32, LfmWord)], i: usize) -> [u8; 32] {
    use math::field::traits::IsPrimeField;
    let mut out = [0u8; 32];
    for h in 0..8 {
        let lane = public[2 * i + h / 4].1[h % 4];
        let half = crate::tables::types::GoldilocksField::canonical(lane.value()) as u32;
        out[4 * h..4 * h + 4].copy_from_slice(&half.to_le_bytes());
    }
    out
}

fn l2g_arenas(
    epoch: &[stark::config::Commitment],
    global: &[stark::config::Commitment],
) -> Vec<Vec<LfmWord>> {
    use super::proof_arena::commitments_to_arena;
    vec![commitments_to_arena(epoch), commitments_to_arena(global)]
}

/// The real bundle's L2G roots, resolved once.
fn r1g_l2g_roots() -> &'static (
    Vec<stark::config::Commitment>,
    Vec<stark::config::Commitment>,
) {
    use std::sync::OnceLock;
    static CELL: OnceLock<(
        Vec<stark::config::Commitment>,
        Vec<stark::config::Commitment>,
    )> = OnceLock::new();
    CELL.get_or_init(|| {
        let blob = proof_fixture::load_or_generate(&fixture_cache());
        let archive = super::proof_fixture::FixtureArchive::open(&blob);
        let epoch = super::proof_arena::epoch_l2g_roots(&archive);
        let global = super::proof_arena::global_l2g_roots(&archive, epoch.len());
        (epoch, global)
    })
}

/// Scrutinises the oracle before building on it: production's binding really
/// does hold on the real bundle, and — the part that matters for (d) — the
/// per-epoch roots are DISTINCT, so swapping two of them is a real tamper.
///
/// Without that second check the position-swap vector would pass while testing
/// nothing, exactly as an index-bit flip would have on a degenerate tree in R1f.
#[test]
fn l2g_binding_holds_on_the_real_bundle() {
    let (epoch, global) = r1g_l2g_roots();
    assert_eq!(
        epoch.len(),
        R1G_EPOCHS,
        "the fixture's epoch count moved; R1G_EPOCHS is program shape"
    );
    assert_eq!(epoch, global, "production's own L2G binding must hold");
    assert!(
        epoch.iter().all(|r| *r != [0u8; 32]),
        "every L2G root must be nonzero"
    );
    for i in 0..epoch.len() {
        for j in (i + 1)..epoch.len() {
            assert_ne!(
                epoch[i], epoch[j],
                "epochs {i} and {j} share an L2G root, so swapping them is not a tamper"
            );
        }
    }
    println!(
        "R1g(ii): {} epochs, binding holds, roots pairwise distinct",
        epoch.len()
    );
}

/// ★ The binding, emitted and PROVED against the real bundle.
#[test]
fn l2g_binding_proves_and_verifies() {
    let opts = options();
    let (epoch, global) = r1g_l2g_roots();
    let program = l2g_binding_program(R1G_EPOCHS);
    let artifacts = build_artifacts(&program, &opts);
    let proved = lfm_prove(&program, &artifacts, &l2g_arenas(epoch, global), &opts)
        .expect("the honest binding must execute and prove");

    for (i, root) in epoch.iter().enumerate().take(R1G_EPOCHS) {
        assert_eq!(
            published_root(&proved.public_words, i),
            *root,
            "published root {i} must be epoch {i}'s committed L2G root"
        );
    }
    assert!(
        verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &proved.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
        ),
        "the L2G binding must verify"
    );
}

/// ★ (d) Tamper. Same two-mode structure as R1f: incoherent trips the assert,
/// coherent satisfies every assert and then fails on the published roots.
///
/// The SWAP vector is the one that matters — it is what "position-sensitive"
/// means. Epoch `i`'s root must meet global sub-proof `i` and no other, so a
/// bundle whose L2G roots are correct as a SET but wrong in ORDER must reject.
#[test]
fn tampered_l2g_binding_rejects() {
    let opts = options();
    let (epoch, global) = r1g_l2g_roots();
    let program = l2g_binding_program(R1G_EPOCHS);
    let artifacts = build_artifacts(&program, &opts);
    let honest =
        lfm_prove(&program, &artifacts, &l2g_arenas(epoch, global), &opts).expect("honest prove");

    // Incoherent: one side changed, the other left honest — the assert must fire.
    //
    // The two byte positions are chosen, not arbitrary. A digest spans TWO
    // machine words (bytes 0-15 and 16-31) and each needs its own
    // `assert_word_eq`; vectors that all land in byte 0 would leave an emitter
    // that compares only the first word completely uncaught. Byte 31 covers the
    // second word. Falsification F32 confirmed the gap was real before this.
    let mut bad_epoch = epoch.clone();
    bad_epoch[0][0] ^= 1;
    let mut bad_global = global.clone();
    bad_global[1][31] ^= 1;
    let swapped_one_side = {
        let mut s = epoch.clone();
        s.swap(0, 1);
        s
    };
    for (what, arenas) in [
        ("wrong epoch root", l2g_arenas(&bad_epoch, global)),
        ("wrong global root", l2g_arenas(epoch, &bad_global)),
        (
            "epoch roots swapped on one side",
            l2g_arenas(&swapped_one_side, global),
        ),
    ] {
        let err = super::executor::execute(&program, &arenas, &super::hash::TestPermutation)
            .err()
            .unwrap_or_else(|| panic!("{what}: must not execute"));
        println!("R1g tamper {what}: rejected with {err:?}");
    }

    // Coherent: BOTH sides swapped consistently. Every assert passes — the
    // bundle's roots are the right set — but the order is wrong, so the
    // published roots are not the ones the real bundle commits to.
    let mut swapped = epoch.clone();
    swapped.swap(0, 1);
    let proved = lfm_prove(&program, &artifacts, &l2g_arenas(&swapped, &swapped), &opts)
        .expect("the coherent swap must prove — every assert is satisfied");
    assert_eq!(
        published_root(&proved.public_words, 0),
        epoch[1],
        "the coherent forgery publishes the swapped order"
    );
    assert_ne!(
        proved.public_words, honest.public_words,
        "a reordered binding must not publish the honest roots"
    );
    assert!(
        !verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &honest.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
        ),
        "claiming the real per-epoch roots for a reordered binding must reject"
    );
}

// ============ R1g (iii): the attestation's program id ============

use super::programs::{ProgramIdShape, program_id_program};

/// Arena for a `program_id` fold, each field in its own halves.
fn program_id_arenas(
    elf_digest: &[u8; 32],
    pc_start: u64,
    decode: &stark::config::Commitment,
    pages: &[(u64, stark::config::Commitment)],
) -> Vec<Vec<LfmWord>> {
    let mut halves = keccak_host::pack_stream(elf_digest);
    halves.extend(keccak_host::pack_stream(&pc_start.to_le_bytes()));
    halves.extend(keccak_host::pack_stream(decode));
    for (base, c) in pages {
        halves.extend(keccak_host::pack_stream(&base.to_le_bytes()));
        halves.extend(keccak_host::pack_stream(c));
    }
    vec![halves.into_iter().map(super::word::base_word).collect()]
}

/// The real fixture's program-id inputs.
fn r1g_program_id_inputs() -> (
    [u8; 32],
    u64,
    stark::config::Commitment,
    Vec<(u64, stark::config::Commitment)>,
) {
    use super::proof_arena;
    let blob = proof_fixture::load_or_generate(&fixture_cache());
    let archive = super::proof_fixture::FixtureArchive::open(&blob);
    let elf_bytes = proof_arena::inner_elf(&archive).to_vec();
    let elf = executor::elf::Elf::load(&elf_bytes).expect("the fixture's inner ELF must load");
    (
        crate::statement::elf_digest(&elf_bytes),
        elf.entry_point,
        proof_arena::decode_commitment(&archive),
        proof_arena::page_commitments(&archive),
    )
}

/// ★ The fold, PROVED, bit-exact against production's own `program_id_from_digest`.
///
/// The oracle is the production function, not a local re-implementation.
#[test]
fn program_id_matches_production_on_the_real_fixture() {
    let opts = options();
    let (elf_digest, pc_start, decode, pages) = r1g_program_id_inputs();
    assert!(
        pages.is_empty(),
        "the fibonacci fixture is expected to touch no data pages; if this \
         changes, the shape below must change with it"
    );
    let shape = ProgramIdShape {
        num_pages: pages.len(),
    };
    let program = program_id_program(shape);
    let artifacts = build_artifacts(&program, &opts);
    let arenas = program_id_arenas(&elf_digest, pc_start, &decode, &pages);
    let proved = lfm_prove(&program, &artifacts, &arenas, &opts).expect("prove");

    let expected = crate::recursion::program_id_from_digest(&elf_digest, pc_start, &decode, &pages);
    assert_eq!(
        digest_bytes(&proved.public_words),
        expected,
        "the machine's program id must equal production's"
    );
    assert!(
        verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &proved.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
        ),
        "the program-id fold must verify"
    );
    println!(
        "R1g(iii): {} pages, {} bytes hashed, tag {} bytes (shift {})",
        shape.num_pages,
        shape.byte_len(),
        crate::recursion::PROGRAM_ID_TAG.len(),
        crate::recursion::PROGRAM_ID_TAG.len() % 4,
    );
}

/// ★ The page loop, exercised. The fixture has ZERO page commitments, so the
/// sorted-page path is present-but-untested on real data — the caveat the team
/// lead flagged for the supplied roots applies here too. This drives it with a
/// synthetic shape against the same production oracle, so "it compiles" is not
/// mistaken for "it is covered".
///
/// Proved, not just executed: the fold's byte length changes with the page
/// count, which moves every padding position, and only a proof sees the keccak
/// chip agree with the executor about that.
#[test]
fn program_id_folds_pages_in_the_production_layout() {
    let opts = options();
    let (elf_digest, pc_start, decode, _) = r1g_program_id_inputs();
    for num_pages in [1usize, 3] {
        let pages: Vec<(u64, stark::config::Commitment)> = (0..num_pages)
            .map(|i| {
                let base = 0x1000u64 * (i as u64 + 1);
                let mut c = [0u8; 32];
                for (j, b) in c.iter_mut().enumerate() {
                    *b = (17 * i + j) as u8;
                }
                (base, c)
            })
            .collect();
        let shape = ProgramIdShape { num_pages };
        let program = program_id_program(shape);
        let artifacts = build_artifacts(&program, &opts);
        let arenas = program_id_arenas(&elf_digest, pc_start, &decode, &pages);
        let proved = lfm_prove(&program, &artifacts, &arenas, &opts).expect("prove");
        let expected =
            crate::recursion::program_id_from_digest(&elf_digest, pc_start, &decode, &pages);
        assert_eq!(
            digest_bytes(&proved.public_words),
            expected,
            "{num_pages} pages: the machine's fold must match production's"
        );
        assert!(
            verify_against(
                &artifacts.roots,
                &artifacts.program_id,
                artifacts.keccak_rnd_chunks,
                &proved.proof,
                &proved.public_words,
                &opts,
                artifacts.hasher,
                artifacts.chip_set,
            ),
            "{num_pages} pages: the fold must verify"
        );
    }
}

/// ★ (d) Tamper: every folded field must move the id.
///
/// Coherent by construction — nothing asserts, so each forgery proves cleanly
/// and fails on the published id, which is the whole mechanism: the id IS the
/// claim, and a consumer comparing against its own recompute rejects.
#[test]
fn tampered_program_id_inputs_change_the_id() {
    let opts = options();
    let (elf_digest, pc_start, decode, _) = r1g_program_id_inputs();
    let pages = vec![(0x1000u64, [7u8; 32]), (0x2000u64, [9u8; 32])];
    let shape = ProgramIdShape {
        num_pages: pages.len(),
    };
    let program = program_id_program(shape);
    let artifacts = build_artifacts(&program, &opts);
    let honest = lfm_prove(
        &program,
        &artifacts,
        &program_id_arenas(&elf_digest, pc_start, &decode, &pages),
        &opts,
    )
    .expect("honest prove");

    let mut d2 = elf_digest;
    d2[31] ^= 1;
    let mut dec2 = decode;
    dec2[0] ^= 1;
    let mut pages_value = pages.clone();
    pages_value[1].1[31] ^= 1;
    let mut pages_base = pages.clone();
    pages_base[0].0 ^= 1;
    let mut pages_order = pages.clone();
    pages_order.swap(0, 1);

    for (what, arenas) in [
        (
            "elf digest",
            program_id_arenas(&d2, pc_start, &decode, &pages),
        ),
        (
            "entry point",
            program_id_arenas(&elf_digest, pc_start ^ 1, &decode, &pages),
        ),
        (
            "decode root",
            program_id_arenas(&elf_digest, pc_start, &dec2, &pages),
        ),
        (
            "page commitment",
            program_id_arenas(&elf_digest, pc_start, &decode, &pages_value),
        ),
        (
            "page base",
            program_id_arenas(&elf_digest, pc_start, &decode, &pages_base),
        ),
        (
            "page ORDER",
            program_id_arenas(&elf_digest, pc_start, &decode, &pages_order),
        ),
    ] {
        let forged = lfm_prove(&program, &artifacts, &arenas, &opts).expect("prove");
        assert_ne!(
            forged.public_words, honest.public_words,
            "{what}: a change must move the program id"
        );
        assert!(
            !verify_against(
                &artifacts.roots,
                &artifacts.program_id,
                artifacts.keccak_rnd_chunks,
                &forged.proof,
                &honest.public_words,
                &opts,
                artifacts.hasher,
                artifacts.chip_set,
            ),
            "{what}: claiming the honest id must reject"
        );
    }
}

// ============ R1g (i): the REGISTER preprocessed derivation ============

use super::programs::{RegisterDerivationShape, lde_probe_program, register_derivation_program};

/// The inner proof's blowup for the real target (`lfm-RESUME.md`), and the
/// sweep the leg was asked to measure.
const INNER_BLOWUPS: [usize; 3] = [2, 4, 8];

/// Every production `ProofOptions` uses this offset.
const PRODUCTION_COSET_OFFSET: u64 = 3;

fn derivation_shape(blowup: usize) -> RegisterDerivationShape {
    RegisterDerivationShape {
        blowup,
        coset_offset: PRODUCTION_COSET_OFFSET,
    }
}

/// The inner proof's options at a given blowup — the ones whose REGISTER
/// commitment is being derived, which are NOT the LFM proof's own options.
fn inner_options(blowup: usize) -> ProofOptions {
    let opts = GoldilocksCubicProofOptions::with_blowup(blowup as u8).expect("inner options");
    assert_eq!(
        opts.coset_offset, PRODUCTION_COSET_OFFSET,
        "the shape constant must track production's coset offset"
    );
    opts
}

/// Deterministic pseudo-random `u32`s (splitmix64, high word) — a register file
/// whose rows are pairwise distinct and all nonzero, asserted by
/// [`the_fixture_register_boundary_is_mostly_zeros`] rather than assumed.
fn synthetic_register_file(seed: u64) -> Vec<u32> {
    let mut state = seed;
    (0..crate::tables::register::NUM_REGISTER_ADDRESSES)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 32) as u32
        })
        .collect()
}

fn register_arenas(init: &[u32], fini: &[u32]) -> Vec<Vec<LfmWord>> {
    let column = |v: &[u32]| {
        v.iter()
            .map(|&x| super::word::base_word(FE::from(x as u64)))
            .collect()
    };
    vec![column(init), column(fini)]
}

/// ★ Oracle scrutiny, before anything is built on it: the row order this leg
/// restates really is production's.
///
/// `register_word_address_list` is private, so the program assembles the same
/// list from the public per-register helper. This pins the result against the
/// layout the table's own docs state (x0–x31 at 0..63, x254 at 508, x255 at
/// 510–511) — a check with teeth only because it is written from the docs
/// rather than from the assembly under test.
#[test]
fn the_register_offset_column_is_productions_row_order() {
    use crate::tables::register::{
        NUM_REGISTER_ADDRESSES, PC_HI_INDEX, PC_LO_INDEX, X254_INDEX, register_word_addresses,
    };
    let mut expected: Vec<u64> = (0..64).collect();
    expected.extend([508, 510, 511]);
    assert_eq!(expected.len(), NUM_REGISTER_ADDRESSES);

    let mut derived = Vec::new();
    for reg in 0..32u8 {
        derived.extend(register_word_addresses(reg));
    }
    derived.extend(register_word_addresses(254));
    derived.extend(register_word_addresses(255));
    assert_eq!(
        derived, expected,
        "the REGISTER row order moved; the derivation's OFFSET column is built \
         from this list and every derived root depends on it"
    );
    // The positional constants the table exports must agree with it too.
    assert_eq!(derived[X254_INDEX], 508);
    assert_eq!(derived[PC_LO_INDEX], 510);
    assert_eq!(derived[PC_HI_INDEX], 511);
}

/// ★ The emitted LDE against production's own `interpolate_fft` +
/// `evaluate_polynomial_on_lde_domain`, at sizes and offsets production never
/// takes.
///
/// The register leg only ever runs the transform at `n = 128`, offset 3. Per
/// `lfm-target-shape.md`'s degenerate-parameter rule, that means the real
/// differential cannot tell a general emitter from one that is accidentally
/// right there, so the synthetic sizes are the only witness. Execute-only: this
/// is pure `LFM_BALU` arithmetic and the proved test below covers the chips.
#[test]
fn the_emitted_lde_matches_productions_transform() {
    use math::polynomial::Polynomial;
    use stark::prover::evaluate_polynomial_on_lde_domain;

    for (n, blowup, offset) in [
        (2usize, 2usize, 3u64),
        (4, 2, 3),
        (8, 2, 3),
        (8, 4, 3),
        (8, 8, 3),
        (16, 4, 7),
        (32, 2, 1),
        (8, 1, 3),
        (128, 2, 3),
    ] {
        let program = lde_probe_program(n, blowup, offset);
        validate(&program).expect("admission");
        let source = synthetic_register_file(n as u64 * 31 + offset);
        let values: Vec<FE> = (0..n)
            .map(|i| FE::from(source[i % source.len()] as u64 + 1))
            .collect();
        let arenas = vec![values.iter().copied().map(super::word::base_word).collect()];
        let exec = super::executor::execute(&program, &arenas, &super::hash::TestPermutation)
            .unwrap_or_else(|e| panic!("n={n} blowup={blowup}: execution failed: {e:?}"));

        let poly = Polynomial::interpolate_fft::<crate::tables::types::GoldilocksField>(&values)
            .expect("interpolate");
        let expected =
            evaluate_polynomial_on_lde_domain(&poly, blowup, n, &FE::from(offset)).expect("lde");
        let got: Vec<FE> = exec.public_words.iter().map(|(_, w)| w[0]).collect();
        assert_eq!(
            got.len(),
            n * blowup,
            "n={n} blowup={blowup}: the extension must cover the whole domain"
        );
        assert_eq!(
            got, expected,
            "n={n} blowup={blowup} offset={offset}: the emitted LDE must equal production's"
        );
    }
}

/// ★ The whole derivation, against production's own
/// `compute_precomputed_commitment_with_fini`, across the blowup sweep.
///
/// Two register files per blowup and they do different jobs. The REAL fixture's
/// file is what the target runs on; a SYNTHETIC one where all 67 entries are
/// distinct and nonzero is what makes the test able to fail — a real register
/// file is mostly zeros, so an emitter that dropped or duplicated rows could
/// agree with production on it and disagree everywhere else.
#[test]
fn the_register_derivation_matches_production() {
    for blowup in INNER_BLOWUPS {
        let shape = derivation_shape(blowup);
        let opts = inner_options(blowup);
        let program = register_derivation_program(shape);
        validate(&program).expect("admission");

        for (what, init, fini) in register_file_cases() {
            let arenas = register_arenas(&init, &fini);
            let exec = super::executor::execute(&program, &arenas, &super::hash::TestPermutation)
                .unwrap_or_else(|e| panic!("blowup {blowup} / {what}: execution failed: {e:?}"));
            let expected = crate::tables::register::compute_precomputed_commitment_with_fini(
                &opts, &init, &fini,
            );
            assert_eq!(
                digest_bytes(&exec.public_words),
                expected,
                "blowup {blowup} / {what}: the derived root must equal production's"
            );
        }
    }
}

/// The register files the differential runs on: the fixture's real boundary
/// pair, plus synthetics that exercise every row.
fn register_file_cases() -> Vec<(&'static str, Vec<u32>, Vec<u32>)> {
    let (init, fini) = fixture_register_boundary();
    vec![
        ("the fixture's real epoch boundary", init, fini),
        (
            "a synthetic file with every row distinct",
            synthetic_register_file(1),
            synthetic_register_file(2),
        ),
        (
            "init and fini equal (an epoch that changed nothing)",
            synthetic_register_file(3),
            synthetic_register_file(3),
        ),
    ]
}

/// Epoch 0's real `(register_init, reg_fini)` from the proof fixture — the
/// verifier-derived INIT from the entry point and the epoch's bound FINI.
fn fixture_register_boundary() -> (Vec<u32>, Vec<u32>) {
    use std::sync::OnceLock;
    static CELL: OnceLock<(Vec<u32>, Vec<u32>)> = OnceLock::new();
    CELL.get_or_init(|| {
        let blob = proof_fixture::load_or_generate(&fixture_cache());
        let archive = super::proof_fixture::FixtureArchive::open(&blob);
        super::proof_arena::register_boundary(&archive, 0)
    })
    .clone()
}

/// ★ Scrutinise the fixture boundary before trusting the differential that
/// runs on it.
///
/// A register file is mostly zeros and `init` differs from `fini` in only a
/// handful of places, so "the real case passed" is weak evidence on its own.
/// This says exactly how weak, in numbers, which is what justifies the
/// synthetic cases carrying the load in
/// [`the_register_derivation_matches_production`].
#[test]
fn the_fixture_register_boundary_is_mostly_zeros() {
    use crate::tables::register::{NUM_REGISTER_ADDRESSES, PC_LO_INDEX};
    let (init, fini) = fixture_register_boundary();
    assert_eq!(init.len(), NUM_REGISTER_ADDRESSES);
    assert_eq!(fini.len(), NUM_REGISTER_ADDRESSES);

    let nonzero = |v: &[u32]| v.iter().filter(|&&x| x != 0).count();
    let differing = init.iter().zip(&fini).filter(|(a, b)| a != b).count();
    println!(
        "R1g(i) fixture boundary: init {}/{} nonzero, fini {}/{} nonzero, \
         {differing} rows differ (pc {} -> {})",
        nonzero(&init),
        NUM_REGISTER_ADDRESSES,
        nonzero(&fini),
        NUM_REGISTER_ADDRESSES,
        init[PC_LO_INDEX],
        fini[PC_LO_INDEX],
    );
    assert_ne!(
        init, fini,
        "an epoch that changed no register would be a degenerate case"
    );
    assert!(
        nonzero(&fini) > 0,
        "an all-zero fini would make the differential blind to the FINI column"
    );
    // Nonzero is not enough: rows that share a value are still indistinguishable
    // to a differential, so the synthetic file has to be pairwise DISTINCT for
    // "it exercises every row" to mean anything.
    let synthetic = synthetic_register_file(1);
    assert_eq!(
        nonzero(&synthetic),
        NUM_REGISTER_ADDRESSES,
        "the synthetic file must exercise every row, which the real one does not"
    );
    let distinct: std::collections::HashSet<u32> = synthetic.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        NUM_REGISTER_ADDRESSES,
        "the synthetic file's rows must be pairwise distinct, else a dropped or \
         duplicated row could still agree with production"
    );
}

/// ★ The measurement the leg was asked for: permutations against the predicted
/// 255 / 511 / 1023, and the derivation's share of an epoch verify.
///
/// The prediction is `2·leaves − 1` with `leaves = 128·blowup / ROWS_PER_LEAF`,
/// i.e. `128·blowup − 1`. A miss is not something to round off: it would mean
/// the leaf grouping, the domain size or the padding is not what the design
/// says, so this asserts rather than prints.
#[test]
fn register_derivation_cost() {
    // `lfm-target-shape.md`'s *Scale*: keccak permutations per epoch verify.
    let epoch_permutations = |blowup: usize| match blowup {
        2 => Some(1_400_000f64),
        8 => Some(460_000f64),
        _ => None,
    };
    println!(
        "one permutation = {} main cells; one byteswap = {} main cells; \
         KECCAK_RND chunk ceiling {} rows",
        permutation_cells(),
        byteswap_cells(),
        super::chunking::KECCAK_RND_MAX_CHUNK_ROWS,
    );
    println!(
        "blowup  rows   leaves   perms  predicted   instrs   const    balu  bitdec  keccak   \
         main cells   % of epoch hashing"
    );
    for blowup in INNER_BLOWUPS {
        let shape = derivation_shape(blowup);
        let predicted = 128 * blowup - 1;
        let program = register_derivation_program(shape);
        let (main, _aux) = super::airs::lfm_cell_counts(&program);
        let share = epoch_permutations(blowup)
            .map(|total| format!("{:.4}%", 100.0 * shape.permutations() as f64 / total))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{blowup:>6} {:>6} {:>8} {:>7} {:>10} {:>8} {:>7} {:>7} {:>7} {:>7} {:>12} {:>19}",
            shape.lde_rows(),
            shape.leaves(),
            program.groups.keccak.real_rows,
            predicted,
            program.instrs.len(),
            program.groups.const_.real_rows,
            program.groups.balu.real_rows,
            program.groups.bitdec.real_rows,
            program.groups.keccak.real_rows,
            main,
            share,
        );
        assert_eq!(
            shape.permutations(),
            predicted,
            "blowup {blowup}: the shape's own arithmetic must give the predicted count"
        );
        assert_eq!(
            wrap_hash_rows(&program),
            predicted,
            "blowup {blowup}: the EMITTED compression count must be 2·leaves − 1 \
             ({predicted}); a miss means the tree's shape is not what the design \
             says. Counted against the CONFIGURED hash: the claim is about tree \
             shape, and it holds under both — a 48-byte leaf and a 64-byte parent \
             are each one keccak rate block and each one BLAKE3 block"
        );
        // Every leaf is 48 bytes and every parent 64 — one rate block each, so
        // the permutation count is exactly the node count and nothing else.
        assert_eq!(
            super::keccak_host::num_blocks(8 * 3 * stark::commitment::ROWS_PER_LEAF),
            1,
            "a three-column row pair must fit one keccak rate block"
        );
        assert_eq!(
            super::keccak_host::num_blocks(2 * super::edsl::COMMITMENT_BYTES),
            1,
            "a Merkle parent must fit one keccak rate block"
        );

        // Where the arithmetic goes, to the row. The transform is
        // `2 · (n/2·log₂n butterflies + blowup · (n scalings + n/2·log₂n
        // butterflies))` at two rows per butterfly and one per scaling, over
        // the TWO dynamic columns; the swap is 64 rows for each of the leaf's
        // six values. Pinning the split is what makes a later change to either
        // half visible instead of showing up as one moved total.
        let n = shape.num_rows() as u64;
        let butterflies = n / 2 * n.trailing_zeros() as u64;
        let per_column = 2 * butterflies + blowup as u64 * (n + 2 * butterflies);
        let transform = 2 * per_column;
        let swap = shape.leaves() as u64 * 6 * 64;
        assert_eq!(
            program.groups.balu.real_rows as u64,
            transform + swap,
            "blowup {blowup}: LFM_BALU rows must be {transform} of transform plus \
             {swap} of byte swapping"
        );
        assert_eq!(
            program.groups.bitdec.real_rows,
            shape.leaves() * 6,
            "blowup {blowup}: one bit decomposition per leaf value — the leaf \
             gadget is `keccak_leaf_hash` reused, not a second one"
        );
        // Chunking is not a constraint at this scale and the leg should say so
        // rather than leave the next reader to work it out: the whole tree at
        // blowup 8 is 1023 permutations against a ceiling of 2^19 ROWS.
        assert_eq!(
            super::chunking::KeccakChunking::default().chunk_count(shape.permutations()),
            1,
            "blowup {blowup}: the register tree must fit one KECCAK_RND chunk"
        );
        assert_eq!(
            program.groups.select.real_rows, 0,
            "blowup {blowup}: a TREE build knows every child's side at emission \
             time, so it must emit no Select at all; routing it through \
             `keccak_merkle_walk` would put one per parent here"
        );
        println!(
            "  blowup {blowup}: transform {transform} balu rows ({:.1}%), \
             byteswap {swap} ({:.1}%); a Select would cost {} cells against a \
             permutation's {}",
            100.0 * transform as f64 / (transform + swap) as f64,
            100.0 * swap as f64 / (transform + swap) as f64,
            select_cells(),
            permutation_cells(),
        );
    }
}

/// Main-trace cells one `LFM_SELECT` row costs — the unit the tree build avoids
/// by knowing child order at emission time.
fn select_cells() -> u64 {
    use super::chips::select;
    use super::layout;
    (select::cols::NUM_COLUMNS - layout::select::PREP_WIDTH) as u64
}

/// ★ The derivation PROVED, not merely executed — and against the REAL
/// fixture's own options rather than a reconstruction of them.
///
/// The executor mirrors the keccak the chip also does, so an execute-only
/// differential cannot see the `LFM_KECCAK` adapter, the `KECCAK_RND` chunking
/// or the lane plumbing agree with it. Blowup 2 (255 permutations) is the cheap
/// end of the sweep; the shape's arithmetic is what carries 4 and 8, and
/// `register_derivation_cost` asserts it.
///
/// `fixture_options()` is `MIN_PROOF_OPTIONS` — blowup 2, coset offset 3, the
/// only two fields that reach the commitment — so this is not merely "the
/// machine agrees with a production function on some inputs". The root proved
/// here IS the preprocessed REGISTER commitment epoch 0 of the fixture's own
/// continuation was built against.
#[test]
fn the_register_derivation_proves_and_verifies() {
    let opts = options();
    let inner = proof_fixture::fixture_options();
    let shape = RegisterDerivationShape {
        blowup: inner.blowup_factor as usize,
        coset_offset: inner.coset_offset,
    };
    assert_eq!(
        shape,
        derivation_shape(2),
        "the fixture is proved at blowup 2 / offset 3; if that moves, this test          is no longer about the fixture's own commitment"
    );
    let program = register_derivation_program(shape);
    let artifacts = build_artifacts(&program, &opts);
    let (init, fini) = fixture_register_boundary();
    let arenas = register_arenas(&init, &fini);
    let proved = lfm_prove(&program, &artifacts, &arenas, &opts).expect("prove");

    assert_eq!(
        digest_bytes(&proved.public_words),
        crate::tables::register::compute_precomputed_commitment_with_fini(&inner, &init, &fini),
        "the proved root must equal the fixture epoch's own REGISTER commitment"
    );
    assert!(
        verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &proved.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
        ),
        "the derivation must verify"
    );
    println!(
        "R1g(i) proved: blowup {}, {} permutations, {} KECCAK_RND chunks",
        shape.blowup, program.groups.keccak.real_rows, artifacts.keccak_rnd_chunks,
    );
}

/// ★ Tamper: every register word must move the derived root, and claiming the
/// honest root for a tampered file must reject.
///
/// Coherent by construction — the program asserts nothing, so each forgery
/// proves cleanly and fails on the PUBLISHED root. That is the mechanism
/// working: the root is the claim, and in the assembled verifier it is what
/// Phase A absorbs, so a `reg_fini` the prover did not honour produces a root
/// the epoch's own proof was not made against.
#[test]
fn tampering_the_register_files_moves_the_derived_root() {
    let opts = options();
    let shape = derivation_shape(2);
    let program = register_derivation_program(shape);
    let artifacts = build_artifacts(&program, &opts);
    let (init, fini) = fixture_register_boundary();
    let honest = lfm_prove(&program, &artifacts, &register_arenas(&init, &fini), &opts)
        .expect("honest prove");

    use crate::tables::register::{NUM_REGISTER_ADDRESSES, PC_HI_INDEX, X254_INDEX};
    // A zero row, a row the epoch changed, and the two whose values are not
    // plain GPR words — the PC high half and the synthetic commit index, which
    // are the rows an implementation is most likely to mislay.
    let mut cases: Vec<(String, Vec<u32>, Vec<u32>)> = Vec::new();
    for row in [
        0usize,
        5,
        X254_INDEX,
        PC_HI_INDEX,
        NUM_REGISTER_ADDRESSES - 1,
    ] {
        let mut i2 = init.clone();
        i2[row] ^= 1;
        cases.push((format!("init row {row}"), i2, fini.clone()));
        let mut f2 = fini.clone();
        f2[row] ^= 1;
        cases.push((format!("fini row {row}"), init.clone(), f2));
    }

    for (what, i, f) in cases {
        let forged = lfm_prove(&program, &artifacts, &register_arenas(&i, &f), &opts)
            .unwrap_or_else(|e| panic!("{what}: a tampered file must still prove: {e:?}"));
        assert_ne!(
            forged.public_words, honest.public_words,
            "{what}: a changed register must move the derived root"
        );
        assert!(
            !verify_against(
                &artifacts.roots,
                &artifacts.program_id,
                artifacts.keccak_rnd_chunks,
                &forged.proof,
                &honest.public_words,
                &opts,
                artifacts.hasher,
                artifacts.chip_set,
            ),
            "{what}: claiming the honest root must reject"
        );
    }
}

/// ⚠ Documents a LIVE gap — **asserts it SUCCEEDS**, per the guard-test map's
/// convention. If this ever starts failing, something began constraining the
/// arena; re-derive before relaxing it.
///
/// Production's `reg_fini` is a `Vec<u32>`, so every value it can commit is
/// below 2^32 and the type is the whole enforcement. An LFM arena is untyped
/// field elements, and nothing in the derivation narrows them: the program
/// happily extends and commits a column production could not have built.
///
/// ## What is and is not claimed
///
/// This is NOT a hole in the derivation. A root over a non-`u32` column matches
/// no commitment a production epoch proof was made against, so the epoch fails
/// — the same argument that makes the derivation a binding at all. What it IS
/// is a place where the machine's accepted set is WIDER than the RV64
/// verifier's, and the assembled verifier owes one of two things: an arena
/// range check (67 extra `bit_dec`s per column, cheap at this scale), or the
/// argument that no epoch proof can exist over such a column in the first
/// place. That second argument is PLAUSIBLE — REG-C2 puts FINI on the Memory
/// bus as a value word and the memory side decomposes values into bytes — but
/// it is unverified here, and per the standing rule a deferral's safety
/// argument is itself a claim needing evidence. Stated, not assumed.
#[test]
fn the_derivation_extends_a_non_u32_register_value_demonstrating_hazard() {
    let shape = derivation_shape(2);
    let program = register_derivation_program(shape);
    let (init, fini) = fixture_register_boundary();

    let honest = register_arenas(&init, &fini);
    let mut wide = honest.clone();
    // 2^32 — one past every value a `Vec<u32>` can hold.
    wide[1][0] = super::word::base_word(FE::from(1u64 << 32));

    let run = |arenas: &[Vec<LfmWord>]| {
        digest_bytes(
            &super::executor::execute(&program, arenas, &super::hash::TestPermutation)
                .expect("the machine must accept any felt in the arena")
                .public_words,
        )
    };
    assert_ne!(
        run(&wide),
        run(&honest),
        "the out-of-range value must actually reach the commitment, else this \
         test documents nothing"
    );
}

/// The honest-path control for the admission gate as a whole: every kind in
/// `LFM_REGISTRY` still passes, together, in one place.
///
/// Each program has its own `*_is_admissible` test; this one exists because a
/// rejection rule is worth having only if it rejects nothing that ships, and
/// the cheapest way for a new check to be wrong is to be right about the one
/// program its author had in mind. It also fails loudly when a seventh kind is
/// registered without being run past the registrar here.
#[test]
fn every_registered_program_passes_admission() {
    let programs: [(&str, LfmProgram); 6] = [
        ("TrivialV0", trivial_program()),
        ("FriToyV0", fri_toy_program()),
        ("KeccakChainV0", keccak_chain_program()),
        ("KeccakSpongeV0", keccak_sponge_program(KECCAK_SPONGE_LEN)),
        ("TranscriptReplayV0", transcript_replay_program()),
        ("StatementReplayV0", statement_replay_program()),
    ];
    for (name, program) in &programs {
        validate(program)
            .unwrap_or_else(|e| panic!("registered program {name} must pass admission: {e:?}"));
    }
}

/// Check 9 against a *registered* program's committed group — the object the
/// AIR actually reads. The instruction list stays untouched, so checks 1–4 see
/// a pristine program and nothing but check 9 stands between this group and
/// the registry.
#[test]
fn negative_hash_multiplicity_in_a_registered_group_fails_admission() {
    let mut program = trivial_program();
    validate(&program).expect("honest control: the registered program is admissible");

    let negative_one = FE::from(P - 1); // p − 1, i.e. −1 on the bus
    program
        .groups
        .hash
        .set(0, super::layout::hash::MULT1, negative_one);
    assert!(
        matches!(
            validate(&program).unwrap_err(),
            LfmViolation::MultOutOfRange {
                chip: "LFM_HASH",
                row: 0,
                col,
                ..
            } if col == super::layout::hash::MULT1
        ),
        "a negative multiplicity in the committed group must fail admission"
    );
}

// =============================================================================
// ★ R2 — the symmetric conditional hash-family chip group
// =============================================================================
//
// The machine hosts two hash families and a program uses at most one. Carrying
// the other costs a whole AIR per chip for zero rows. `ChipSet` retires the
// unused one — in BOTH directions from one mask, which is the point: post-flip
// the live direction is dropping the KECCAK family, and pre-flip it was
// dropping LFM_BLAKE3.
//
// Unlike the RV64 side's conditional BLAKE3 table, nothing here is
// prover-asserted. The mask is computed from the compiled program at bless
// time, stored in the registry entry and folded into `program_id`, so a
// verifier reads it from the entry it resolved. There is no forgery direction
// to test because there is no prover claim to forge — the tests that matter are
// that the mask is what the programs actually use, that both directions prove
// and verify, and that a wrong mask cannot be substituted.

/// ★ PRIMARY — the live post-flip direction: a program that authenticates with
/// BLAKE3 carries no keccak family at all.
///
/// `StatementReplayV0` is the real one (the continuation-epoch statement bind
/// plus Phase A). Post-flip it hashes with BLAKE3, so LFM_KECCAK, every
/// KECCAK_RND chunk and KECCAK_RC are all absent — four sub-proofs that were
/// pure overhead.
#[test]
fn a_blake3_authenticating_program_carries_no_keccak_family() {
    let opts = options();
    let program = statement_replay_program();
    let artifacts = build_artifacts(&program, &opts);

    assert_eq!(
        (artifacts.chip_set.keccak, artifacts.chip_set.blake3),
        (false, true),
        "post-flip the authenticating programs hash with BLAKE3 and use no keccak"
    );
    assert_eq!(
        artifacts.keccak_rnd_chunks, 0,
        "an absent keccak family carries zero KECCAK_RND instances"
    );

    let airs = super::airs::LfmAirs::new_with_hasher(
        &artifacts.roots,
        &opts,
        artifacts.keccak_rnd_chunks,
        artifacts.hasher,
        artifacts.chip_set,
    );
    let names: Vec<&str> = airs.air_refs().iter().map(|a| a.name()).collect();
    for absent in ["LFM_KECCAK", "KECCAK_RND", "KECCAK_RC"] {
        assert!(
            !names.contains(&absent),
            "{absent} must not be instantiated for a BLAKE3-authenticating program"
        );
    }
    assert!(
        names.contains(&"LFM_BLAKE3"),
        "the family it DOES use must be present"
    );
    assert!(
        names.contains(&"BITWISE"),
        "BITWISE is shared infrastructure — both families send to its buses — \
         and must survive dropping either one"
    );

    // The honest path: it still proves and verifies without them.
    let f = statement_fixture();
    let proved = lfm_prove(&program, &artifacts, &statement_arenas(&f), &opts).expect("prove");
    assert!(
        verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &proved.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
        ),
        "a BLAKE3 program must prove and verify with the keccak family absent"
    );
}

/// The other direction of the SAME mask: a program that is ABOUT keccak keeps
/// the keccak family and drops LFM_BLAKE3.
///
/// This is the control that makes the primary test mean something. It also
/// pins why the mask is per-program rather than keyed on
/// `WrapHash::production()`: the production wrap hash here is BLAKE3, and
/// keying off it would have silently stripped the keccak family from the very
/// programs that exist to exercise keccak.
#[test]
fn a_keccak_program_keeps_the_keccak_family_whatever_the_production_hash() {
    let opts = options();
    let program = keccak_chain_program();
    let artifacts = build_artifacts(&program, &opts);

    assert_eq!(
        (artifacts.chip_set.keccak, artifacts.chip_set.blake3),
        (true, false),
        "a keccak instrument uses keccak and no BLAKE3, regardless of the \
         production commitment hash"
    );
    assert_eq!(artifacts.keccak_rnd_chunks, 1);

    let airs = super::airs::LfmAirs::new_with_hasher(
        &artifacts.roots,
        &opts,
        artifacts.keccak_rnd_chunks,
        artifacts.hasher,
        artifacts.chip_set,
    );
    let names: Vec<&str> = airs.air_refs().iter().map(|a| a.name()).collect();
    assert!(names.contains(&"LFM_KECCAK") && names.contains(&"KECCAK_RC"));
    assert!(
        !names.contains(&"LFM_BLAKE3"),
        "the unused BLAKE3 chip must not be instantiated"
    );
}

/// A program using NEITHER family drops both — the mask is not a two-way
/// switch, it is a per-family presence bit.
#[test]
fn a_program_using_no_hash_family_carries_neither() {
    let opts = options();
    let artifacts = build_artifacts(&trivial_program(), &opts);
    assert_eq!(
        (artifacts.chip_set.keccak, artifacts.chip_set.blake3),
        (false, false),
        "the trivial program hashes nothing"
    );
    let airs = super::airs::LfmAirs::new_with_hasher(
        &artifacts.roots,
        &opts,
        artifacts.keccak_rnd_chunks,
        artifacts.hasher,
        artifacts.chip_set,
    );
    let names: Vec<&str> = airs.air_refs().iter().map(|a| a.name()).collect();
    for absent in ["LFM_KECCAK", "KECCAK_RND", "KECCAK_RC", "LFM_BLAKE3"] {
        assert!(!names.contains(&absent), "{absent} must be absent");
    }
}

/// The registry's masks are what the programs actually compile to.
///
/// The mask is the authority the verifier reads, so a registry entry claiming a
/// family the program does not use — or omitting one it does — would build the
/// wrong AIR set. This recomputes every entry's mask from its program.
#[test]
fn every_registry_mask_is_the_programs_own_usage() {
    let opts = options();
    let cases: [(LfmProgramKind, super::compiler::LfmProgram); 6] = [
        (LfmProgramKind::TrivialV0, trivial_program()),
        (LfmProgramKind::FriToyV0, super::programs::fri_toy_program()),
        (LfmProgramKind::KeccakChainV0, keccak_chain_program()),
        (
            LfmProgramKind::KeccakSpongeV0,
            super::programs::keccak_sponge_program(super::programs::KECCAK_SPONGE_LEN),
        ),
        (
            LfmProgramKind::TranscriptReplayV0,
            super::programs::transcript_replay_program(),
        ),
        (
            LfmProgramKind::StatementReplayV0,
            statement_replay_program(),
        ),
    ];
    for (kind, program) in &cases {
        let entry = super::registry::resolve(*kind, 2).expect("registered");
        let computed = super::airs::ChipSet::for_program(program);
        assert_eq!(
            entry.chip_set, computed,
            "{kind:?}: the registry mask must be the program's own family usage"
        );
        assert_eq!(
            entry.keccak_rnd_chunks,
            computed.keccak_rnd_chunks(
                program
                    .chunking
                    .chunk_count(program.groups.keccak.real_rows)
            ),
            "{kind:?}: chunk count must follow the mask"
        );
        // And the mask is bound: a different one is a different identity.
        let other = super::airs::ChipSet {
            keccak: !computed.keccak,
            blake3: computed.blake3,
        };
        let _ = &opts;
        assert_ne!(
            super::statement::lfm_program_id(
                &entry.roots,
                &entry.log_heights,
                entry.keccak_rnd_chunks,
                entry.hasher,
                other,
            ),
            entry.program_id,
            "{kind:?}: the chip set must be folded into program_id, or a verifier \
             could resolve one mask and build another"
        );
    }
}
