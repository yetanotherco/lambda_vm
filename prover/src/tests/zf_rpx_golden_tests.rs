//! Golden proofs under the production RPX pin, in TWO formats:
//!
//! - the LEGACY format (every ZF lever off; `ProofFormat::LEGACY`, the stark
//!   crate's default): the RPX half of `stark::tests::zf_golden_tests` (which
//!   covers Keccak and Blake3 and cannot name `RpxStarkHash`, a prover-crate
//!   type). It keeps the legacy bytes pinned after the default flip, so
//!   the rollback arm (every knob off) is still checked against bytes, not
//!   against a round trip;
//! - the PRODUCTION default (`ZfFormat::DEFAULT.proof_format()`):
//!   the bytes every production site stamps. Regenerate only for a
//!   deliberate format change.
//!
//! Each case proves a small in-repo AIR at `grinding_factor = 0` (so the bytes
//! are reproducible) and pins the SHA-256 of the proof's rkyv bytes plus, so a
//! failure says where the drift is, the digests of its FRI layer roots, terminal
//! coefficients, FRI decommitments and trace/composition openings. The legacy
//! pins were generated at the (then default) legacy format before any S3 prover
//! code existed; regenerate either set only for a deliberate format change:
//! `cargo test -p lambda-vm-prover --lib tests::zf_rpx_golden_tests::print_goldens -- --ignored --nocapture`.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use sha2::{Digest, Sha256};
use stark::examples::bus_permutation::{bus_permutation_air, bus_permutation_trace};
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
    compute_goldens_at(ProofFormat::LEGACY)
}

/// The production default's univariate format, the one the goldens below pin.
fn production_format() -> ProofFormat {
    crate::zf_format::ZfFormat::DEFAULT.proof_format()
}

fn compute_goldens_at(d: ProofFormat) -> Vec<(String, String)> {
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

/// PRODUCTION-default pins, generated by `print_goldens` at the default flip.
const PRODUCTION_GOLDENS: &[(&str, &str)] = &[
    (
        "simple_addition/rpx/rows16/blowup2",
        "proof ee8ca7ebe2cd632fd40d3242450377f17f966c85d35ad69dc11d918a61a12fa9 roots[1] 19764f49df000e57080b4eada26d3d1d3b4d8a7356fe4fa0a779458ffcc0cc94 coeffs c71ca99567bf64cd75e4d2ca5a68533bd196fe44545180370ac90b29cd062b9b queries 4fb6a8d93089bd818a6a5a8b0026fe133491446029263b8d89cf3feec24f254a openings f5700d0bf2e5a02d28b973177bd7828d215bbabaa9c4c2a9c5ac59fb8475f293",
    ),
    (
        "simple_addition/rpx/rows64/blowup4",
        "proof 961d5e394cf9913b261fd255b1b2ebd9d9304560a302cd25c6126a62c99728e4 roots[1] e2a8a7ce17b0c97ec91d741f84349494d7344943a84b0feb3a0fb93a43576846 coeffs bb0d9b495382e02c0b4ac8d0d3fca25bccacc363b6433ec1459c3ba4e26f81d0 queries a636ad70b084ad76ec0dcb2c9d904fe582d12e7379f0bb80541b81abf3febdc0 openings 090f06c93b8a9220d6f7d6dbb63302e708a513be939d85d12fd64ceda96da3a7",
    ),
    (
        "logup/rpx/rows32/blowup4",
        "proof 67c9013a35ae703146e41c999cf082433031067576be62f0dd75b8adfac1fcbf roots[1] 26365ee78c3be44e7d96f1e77a2afcc1747e3736153693877bfc4f9dd6a88dff coeffs 7d98df3343a174fe6add0b9188e592bb5ad84b5797385185ff216f887bc5499d queries 57e99bd603476d6df18dedefa00dc727af256bd36ec91b0bc00773a77e6e739a openings dcb750dd32ae77ecc0b1928369db0bc1ee0ed3f1cad5bb0c4f5b7c258a3cf264",
    ),
    (
        "logup/rpx/rows128/blowup2",
        "proof 978ebf4ab14b4f86c378642e53c3c9ab2cdae7732b2fdd09bd07ec90549ba8fb roots[2] 25c71fa19482dd430d9ce9b413bddf5f905f77a11ba5c9c38f32f64b9cb3738e coeffs 7d4fb0cbc585c68b5422bc19757d3d2151a7dde0f5edd29e312deaecb21428df queries 9281910f6451083a9a3ffc0c12ff31d952e3f718cd6a53022fe38c60d7b791b3 openings d0c9d124a01ace817bf58f0ecb4acb9c51b5ffecca32c04402af9de972856f82",
    ),
];

/// LEGACY-format pins (every lever off), generated before any S3 prover code.
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

/// The LEGACY-format RPX goldens: the legacy bytes, unmoved by the
/// default flip.
#[test]
fn legacy_format_rpx_goldens_are_byte_identical() {
    let got = compute_goldens();
    assert_eq!(got.len(), GOLDENS.len(), "one pin per case");
    for ((name, line), (pin_name, pin_line)) in got.iter().zip(GOLDENS) {
        assert_eq!(name, pin_name);
        assert_eq!(
            line, pin_line,
            "{name}: the legacy-format RPX proof moved (a field whose digest differs is where)"
        );
    }
}

/// The PRODUCTION-default RPX goldens (cap auto, `fri=dp`, and
/// whatever `ZfFormat::DEFAULT` stamps). A move here is a production format
/// change.
#[test]
fn production_format_rpx_goldens_are_byte_identical() {
    let f = production_format();
    assert!(
        !f.is_legacy(),
        "the production default is not the legacy format"
    );
    let got = compute_goldens_at(f);
    assert_eq!(got.len(), PRODUCTION_GOLDENS.len(), "one pin per case");
    for ((name, line), (pin_name, pin_line)) in got.iter().zip(PRODUCTION_GOLDENS) {
        assert_eq!(name, pin_name);
        assert_eq!(
            line, pin_line,
            "{name}: the production-default RPX proof moved (a field whose digest differs is where)"
        );
    }
    // The two formats' proofs differ: the production pins are not the legacy
    // ones under another name.
    for ((_, a), (_, b)) in GOLDENS.iter().zip(PRODUCTION_GOLDENS) {
        assert_ne!(a, b);
    }
}

#[test]
#[ignore = "generator for GOLDENS (legacy) and PRODUCTION_GOLDENS"]
fn print_goldens() {
    for (name, line) in compute_goldens() {
        println!("GOLDEN (\"{name}\", \"{line}\"),");
    }
    println!("production format: {:?}", production_format());
    for (name, line) in compute_goldens_at(production_format()) {
        println!("PRODUCTION GOLDEN (\"{name}\", \"{line}\"),");
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
        ..ProofFormat::LEGACY
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

/// Under RPX, the group path at an all-ones schedule commits
/// the same layer roots, terminal polynomial and paths as the legacy pair path
/// (the `Batched`/`Pair` two-element invariant, as a tested fact for the
/// algebraic backend).
#[test]
fn rpx_group_path_at_all_ones_equals_legacy() {
    // LogReadOnlyRAP 2^7 rows, blowup 4, k 1: 5 committed binary layers.
    let legacy = prove_logup(128, &options(4, 1, 7, ProofFormat::LEGACY)).1;
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

/// The production format sites at the PROCESS format (`ZfFormat::global()`):
/// a small ext3 STARK proved and host-verified under RPX with
/// `block_base_options()` (STARK base epochs) and `aggregation_wrap_options()`
/// (every LFM proof). Without a knob it proves at the production default
/// (`ZfFormat::DEFAULT`: cap auto, `fri=dp`, group layers); the knobs select
/// the arms — `LAMBDA_VM_ZF_FRI=pair` (legacy pair layers),
/// `LAMBDA_VM_ZF_ONE_ROW=1|auto` (both sites stamp the one-row mode; a table
/// resolved to one row opens no symmetric rows and commits the FRI input).
/// Every arm proves.
#[test]
fn production_sites_prove_at_the_process_format() {
    use stark::proof::options::{FriMode, OneRowMode};
    let knob = |name: &str| {
        std::env::var(name)
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
    };
    // An unset knob is the production default's value.
    let default = crate::zf_format::ZfFormat::DEFAULT;
    let want = match knob(crate::zf_format::ENV_FRI).as_deref() {
        Some("dp") => FriMode::Dp,
        Some("pair") => FriMode::Pair,
        None => default.fri,
        Some(other) => panic!("unexpected {}={other}", crate::zf_format::ENV_FRI),
    };
    let want_one_row = match knob(crate::zf_format::ENV_ONE_ROW).as_deref() {
        Some("1") => OneRowMode::On,
        Some("auto") => OneRowMode::Auto,
        Some("0") => OneRowMode::Off,
        None => default.one_row,
        Some(other) => panic!("unexpected {}={other}", crate::zf_format::ENV_ONE_ROW),
    };
    assert_eq!(crate::zf_format::ZfFormat::global().fri, want);
    assert_eq!(crate::zf_format::ZfFormat::global().one_row, want_one_row);
    for (site, o) in [
        (
            "block_base_options",
            crate::lfm::proof::block_base_options(),
        ),
        (
            "aggregation_wrap_options",
            crate::lfm::proof::aggregation_wrap_options(),
        ),
    ] {
        assert_eq!(o.format.fri_mode, want, "{site}");
        assert_eq!(o.format.one_row, want_one_row, "{site}");
        // 2^12 rows: LDE 2^14, so both terminals (T = 9, 10) leave committed
        // layers. At that size a `cuda` build commits on the device, whose
        // composition arm needs the AIR's constraint program — so an
        // `AirWithBuses` table, as in production (`LogReadOnlyRAP` panicked in
        // `constraint_program` there).
        let air = bus_permutation_air(&o);
        let mut trace = bus_permutation_trace(1 << 12);
        let proof = GenericProver::<F, E, (), RpxStarkHash>::prove(
            &air,
            &mut trace,
            &(),
            &mut DefaultTranscript::<E>::new(&[]),
        )
        .unwrap_or_else(|e| panic!("{site}: proving must succeed: {e:?}"));
        assert!(
            GenericVerifier::<F, E, (), RpxStarkHash>::verify(
                &proof,
                &air,
                &mut DefaultTranscript::<E>::new(&[]),
            ),
            "{site}: must verify"
        );
        let layers = proof.fri_layers_merkle_roots.len();
        assert!(layers > 0, "{site}: committed layers");
        let values = proof.query_list[0].layers_evaluations_sym.len();
        let one_row = stark::leaf_layout::table_leaf_layout(&air, 1 << 12).is_one_row();
        if want_one_row == OneRowMode::On {
            assert!(one_row, "{site}: one_row = 1 puts every table on one row");
        }
        let sym = &proof.deep_poly_openings[0].main_trace_polys.evaluations_sym;
        assert_eq!(
            sym.is_empty(),
            one_row,
            "{site}: symmetric rows iff row pairs"
        );
        println!(
            "ZF SITE {site}: fri={want} one_row={want_one_row} resolved_one_row={one_row} layers={layers} values={values}"
        );
        if want == FriMode::Dp || one_row {
            assert!(values > layers, "{site}: group encoding");
        } else {
            assert_eq!(values, layers, "{site}: legacy encoding");
        }
    }
}
