//! Default-format golden proofs under the production RPX pin (REVIEW-FRI F1):
//! the RPX half of `stark::tests::zf_golden_tests` (which covers Keccak and
//! Blake3 and cannot name `RpxStarkHash`, a prover-crate type).
//!
//! Each case proves a small in-repo AIR at `grinding_factor = 0` (so the bytes
//! are reproducible) and pins the SHA-256 of the proof's rkyv bytes plus, so a
//! failure says where the drift is, the digests of its FRI layer roots, terminal
//! coefficients, FRI decommitments and trace/composition openings. Generated at
//! the default format before any S3 prover code existed; regenerate only for a
//! deliberate format change:
//! `cargo test -p lambda-vm-prover --lib tests::zf_rpx_golden_tests::print_goldens -- --ignored --nocapture`.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use sha2::{Digest, Sha256};
use stark::examples::read_only_memory_logup::{
    LogReadOnlyPublicInputs, LogReadOnlyRAP, read_only_logup_trace,
};
use stark::examples::simple_addition::{
    SimpleAdditionAIR, SimpleAdditionPublicInputs, simple_addition_trace,
};
use stark::proof::options::{ProofFormat, ProofOptions};
use stark::proof::stark::StarkProof;
use stark::prover::{GenericProver, IsStarkProver};
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{GenericVerifier, IsStarkVerifier};

use crate::lfm::algebraic_commit::RpxStarkHash;

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type Felt = FieldElement<F>;

pub(crate) fn options(blowup: u8, k: u8, queries: usize, format: ProofFormat) -> ProofOptions {
    ProofOptions {
        blowup_factor: blowup,
        fri_number_of_queries: queries,
        coset_offset: 3,
        grinding_factor: 0,
        fri_final_poly_log_degree: k,
        format,
    }
}

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

macro_rules! fingerprint {
    ($proof:expr) => {{
        let proof = $proof;
        let rk =
            |bytes: Result<rkyv::util::AlignedVec, rkyv::rancor::Error>| hex(&bytes.expect("rkyv"));
        format!(
            "proof {} roots[{}] {} coeffs {} queries {} openings {}",
            rk(rkyv::to_bytes::<rkyv::rancor::Error>(proof)),
            proof.fri_layers_merkle_roots.len(),
            hex(&proof.fri_layers_merkle_roots.concat()),
            rk(rkyv::to_bytes::<rkyv::rancor::Error>(
                &proof.fri_final_poly_coeffs
            )),
            rk(rkyv::to_bytes::<rkyv::rancor::Error>(&proof.query_list)),
            rk(rkyv::to_bytes::<rkyv::rancor::Error>(
                &proof.deep_poly_openings
            )),
        )
    }};
}

pub(crate) fn prove_simple_addition(
    rows: usize,
    o: &ProofOptions,
) -> (
    SimpleAdditionAIR<F>,
    StarkProof<F, F, SimpleAdditionPublicInputs<F>>,
) {
    let air = SimpleAdditionAIR::<F>::new(o);
    let pi = SimpleAdditionPublicInputs {
        a: Felt::from(1u64),
        b: Felt::from(2u64),
    };
    let mut trace = simple_addition_trace::<F>(rows);
    let proof = GenericProver::<F, F, _, RpxStarkHash>::prove(
        &air,
        &mut trace,
        &pi,
        &mut DefaultTranscript::<F>::new(&[]),
    )
    .expect("proving must succeed");
    (air, proof)
}

pub(crate) fn verify_simple_addition(
    air: &SimpleAdditionAIR<F>,
    proof: &StarkProof<F, F, SimpleAdditionPublicInputs<F>>,
) -> bool {
    GenericVerifier::<F, F, _, RpxStarkHash>::verify(
        proof,
        air,
        &mut DefaultTranscript::<F>::new(&[]),
    )
}

pub(crate) fn prove_logup(
    rows: usize,
    o: &ProofOptions,
) -> (
    LogReadOnlyRAP<F, E>,
    StarkProof<F, E, LogReadOnlyPublicInputs<F>>,
) {
    let addr: Vec<Felt> = (0..rows).map(|i| Felt::from((i % 5) as u64 + 1)).collect();
    let val: Vec<Felt> = (0..rows)
        .map(|i| Felt::from(((i % 5) as u64 + 1) * 10))
        .collect();
    let mut trace: TraceTable<F, E> = read_only_logup_trace(addr, val);
    let cols = trace.columns_main();
    let pi = LogReadOnlyPublicInputs {
        a0: cols[0][0],
        v0: cols[1][0],
        a_sorted_0: cols[2][0],
        v_sorted_0: cols[3][0],
        m0: cols[4][0],
    };
    let air = LogReadOnlyRAP::<F, E>::new(o);
    let proof = GenericProver::<F, E, _, RpxStarkHash>::prove(
        &air,
        &mut trace,
        &pi,
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .expect("proving must succeed");
    (air, proof)
}

pub(crate) fn verify_logup(
    air: &LogReadOnlyRAP<F, E>,
    proof: &StarkProof<F, E, LogReadOnlyPublicInputs<F>>,
) -> bool {
    GenericVerifier::<F, E, _, RpxStarkHash>::verify(
        proof,
        air,
        &mut DefaultTranscript::<E>::new(&[]),
    )
}

fn compute_goldens() -> Vec<(String, String)> {
    let d = ProofFormat::DEFAULT;
    let mut out = Vec::new();
    for (rows, blowup) in [(16usize, 2u8), (64, 4)] {
        let o = options(blowup, 2, 5, d);
        let (air, proof) = prove_simple_addition(rows, &o);
        assert!(verify_simple_addition(&air, &proof));
        out.push((
            format!("simple_addition/rpx/rows{rows}/blowup{blowup}"),
            fingerprint!(&proof),
        ));
    }
    for (rows, blowup) in [(32usize, 4u8), (128, 2)] {
        let o = options(blowup, 1, 7, d);
        let (air, proof) = prove_logup(rows, &o);
        assert!(verify_logup(&air, &proof));
        out.push((
            format!("logup/rpx/rows{rows}/blowup{blowup}"),
            fingerprint!(&proof),
        ));
    }
    out
}

const GOLDENS: &[(&str, &str)] = &[
    (
        "simple_addition/rpx/rows16/blowup2",
        "proof 76e4be044a53802e20b5a79c6fddea37893d4c7ba575d5b1b2d6007d6e741191 roots[1] 19764f49df000e57080b4eada26d3d1d3b4d8a7356fe4fa0a779458ffcc0cc94 coeffs c71ca99567bf64cd75e4d2ca5a68533bd196fe44545180370ac90b29cd062b9b queries 1db5f91a7ffd2fa4786315d948666c225859f74b79f97b0eee756531f12c4e29 openings a7254eb12c3eb00518026d8245f76c9c7283090720cf1aacf491af1edbb1a4df",
    ),
    (
        "simple_addition/rpx/rows64/blowup4",
        "proof 54f04bbe46b0330480daf71af2fcafa1fc00dbf698fa538e9168d3c17ae253a9 roots[3] 136c65bbe688080ed90357acda8a3896f8fb8e93e6ab897590684c3d2e5e745d coeffs 51ee895a33735700296d7a776ff1ca28bf7892e7052c28e52a843ebecea5aa2e queries 45c4ec6e74ea93b1a8c98a912b1790936b6856f0884fcf92fcbb6d9be8872b2f openings dc4496a5b2413db451381eddc74765987fb5dd745f6adbcf6ace3814be57717f",
    ),
    (
        "logup/rpx/rows32/blowup4",
        "proof 891ec879640f6a01244829495336a41dfb587214304e4420e2c47f619c3f4d03 roots[3] ec74768e299f79c15f92be8adaa27c38719ee5811fcbf07220163aa5a6d7bacc coeffs afb0be8e7a6b95223d78e5997d681828de2fcefe22704303fc5876b1e3a07fe2 queries cc4f54c61d2c1ad5bbf965c1eb622faec313ed0e8987babea88d9c2ac3cabe87 openings 720412c24f099a3897074effe7f9248cfbd0d370d7ca7cad49d7c794b2ffaf0a",
    ),
    (
        "logup/rpx/rows128/blowup2",
        "proof 16cdff91c22119e7a5bd35be33d0a0e33c09413aba833c1ef4ba48b64bc03b0c roots[5] a928041f346311a1bd49ef81f791370075006cdce88efc45ed5c1608071e5d72 coeffs 709758625cbc3ce3eb8b0f6859198181e95484b5183965163762a3ac4a29852d queries b70a32fb34b697580ddfc50e9a7ac5b29271336301926b0bf62b75242d6c02bf openings 8832140f1efd004ebcd1b70f50a5680abb6d1ede4ca2120d3a826edf6fdef692",
    ),
];

#[test]
fn default_format_rpx_goldens_are_byte_identical() {
    let got = compute_goldens();
    assert_eq!(got.len(), GOLDENS.len(), "one pin per case");
    for ((name, line), (pin_name, pin_line)) in got.iter().zip(GOLDENS) {
        assert_eq!(name, pin_name);
        assert_eq!(
            line, pin_line,
            "{name}: the default-format RPX proof moved (a field whose digest differs is where)"
        );
    }
}

#[test]
#[ignore = "generator for GOLDENS"]
fn print_goldens() {
    for (name, line) in compute_goldens() {
        println!("GOLDEN (\"{name}\", \"{line}\"),");
    }
}

// ---------------------------------------------------------------------------
// S3 (fri = dp) round trips under the production RPX pin: group leaves are
// hashed by the algebraic `Batched` sponge over 3·2^d felts, so the RPX leaf
// path of the group encoding is exercised here (the stark crate's S3 tests
// cover Keccak and Blake3).
// ---------------------------------------------------------------------------

fn dp(schedule: Option<&[u8]>) -> ProofFormat {
    ProofFormat {
        fri_mode: stark::proof::options::FriMode::Dp,
        fri_schedule_override: schedule
            .map(|s| stark::proof::options::FriScheduleOverride::new(s).expect("fits")),
        ..ProofFormat::DEFAULT
    }
}

#[test]
fn rpx_dp_round_trips() {
    // SimpleAddition 2^9 rows, blowup 4, k 1: the chain covers 10 → 3.
    for sched in [None, Some(&[3u8, 1, 3][..]), Some(&[1, 6][..])] {
        let o = options(4, 1, 9, dp(sched));
        let (air, proof) = prove_simple_addition(512, &o);
        assert!(verify_simple_addition(&air, &proof), "{sched:?}");
        if let Some(s) = sched {
            assert_eq!(proof.fri_layers_merkle_roots.len(), s.len());
            let values: usize = s.iter().map(|&d| 1usize << d).sum();
            assert_eq!(proof.query_list[0].layers_evaluations_sym.len(), values);
        }
    }
    // LogReadOnlyRAP (ext3 + aux) 2^7 rows, blowup 4, k 1: 8 → 3.
    for sched in [None, Some(&[2u8, 3][..])] {
        let o = options(4, 1, 7, dp(sched));
        let (air, proof) = prove_logup(128, &o);
        assert!(verify_logup(&air, &proof), "{sched:?}");
        // A tampered group value is rejected.
        let mut bad = proof.clone();
        bad.query_list[0].layers_evaluations_sym[0] += FieldElement::<E>::one();
        assert!(!verify_logup(&air, &bad));
    }
}

/// REVIEW-FRI F1.2 under RPX: the group path at an all-ones schedule commits
/// the same layer roots, terminal polynomial and paths as the legacy pair path
/// (the `Batched`/`Pair` two-element invariant, as a tested fact for the
/// algebraic backend).
#[test]
fn rpx_group_path_at_all_ones_equals_legacy() {
    // LogReadOnlyRAP 2^7 rows, blowup 4, k 1: 5 committed binary layers.
    let legacy = prove_logup(128, &options(4, 1, 7, ProofFormat::DEFAULT)).1;
    let group = prove_logup(128, &options(4, 1, 7, dp(Some(&[1, 1, 1, 1, 1])))).1;
    assert_eq!(legacy.fri_layers_merkle_roots.len(), 5);
    assert_eq!(
        legacy.fri_layers_merkle_roots,
        group.fri_layers_merkle_roots
    );
    assert_eq!(legacy.fri_final_poly_coeffs, group.fri_final_poly_coeffs);
    for (l, g) in legacy.query_list.iter().zip(&group.query_list) {
        for j in 0..5 {
            assert_eq!(
                l.layers_auth_paths[j].merkle_path,
                g.layers_auth_paths[j].merkle_path
            );
            assert!(
                g.layers_evaluations_sym[2 * j..2 * j + 2].contains(&l.layers_evaluations_sym[j])
            );
        }
    }
}
