//! Default-format golden proofs (REVIEW-FRI F1): the bytes today's prover emits,
//! pinned, so a format lever that claims "the default is byte-identical" is
//! checked against the prover's own output rather than against a round trip
//! (a drifted prover still accepts its own proofs).
//!
//! Proof bytes are reproducible only without grinding (the nonce search is a
//! parallel `find_any`), so every case proves at `grinding_factor = 0`. Each
//! case pins the SHA3-256 of the proof's rkyv bytes (the wire format of
//! record) and, so that a failure says WHERE the drift is, separately the
//! digests of: the FRI layer roots, the terminal coefficients, the per-query FRI
//! decommitments, and the trace/composition openings (plus the layer count).
//! ζ and ι are not in a proof; a ζ drift moves every later layer root and the
//! terminal coefficients, an ι drift moves the decommitment and opening digests.
//!
//! Coverage: both byte hashes this crate owns (Keccak, Blake3; RPX is pinned
//! the same way in the prover crate, `tests::zf_rpx_golden_tests`), blowup 2 and
//! 4, a base-field AIR (`SimpleAddition`, E = F) and an extension-field AIR with
//! an aux trace (`LogReadOnlyRAP`, E = F³), `total_folds` ∈ {0, 1, 2, ≥ 3}, and
//! one multi-table bus proof (CPU/ADD/MUL, `multi_prove`).
//!
//! Generated at the default format BEFORE any S3 prover code existed (commit
//! "H0" of lane I-FRI-H, on `zf/cap-stark` @ 77ea1ab89 + the schedule DP, which
//! changes no prover path). Regenerate only for a deliberate format change:
//! `cargo test -p stark --lib zf_golden_tests::print_goldens -- --ignored --nocapture`.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use sha3::{Digest, Sha3_256};

use crate::config::{Blake3StarkHash, KeccakStarkHash, StarkHash};
use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::examples::read_only_memory_logup::{
    LogReadOnlyPublicInputs, LogReadOnlyRAP, read_only_logup_trace,
};
use crate::examples::simple_addition::{
    SimpleAdditionAIR, SimpleAdditionPublicInputs, simple_addition_trace,
};
use crate::proof::options::{ProofFormat, ProofOptions};
use crate::proof::stark::{MultiProof, StarkProof};
use crate::prover::{GenericProver, IsStarkProver};
use crate::residency_mode::ResidencyMode;
use crate::trace::TraceTable;
use crate::traits::AIR;
use crate::verifier::{GenericVerifier, IsStarkVerifier};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type Felt = FieldElement<F>;

/// Test options at the DEFAULT format with grinding off. `k` is the terminal
/// log-degree, so `total_folds = log2(rows) − k` whenever that is ≥ 0.
pub(crate) fn golden_options(
    blowup: u8,
    k: u8,
    queries: usize,
    format: ProofFormat,
) -> ProofOptions {
    ProofOptions {
        blowup_factor: blowup,
        fri_number_of_queries: queries,
        coset_offset: 3,
        grinding_factor: 0,
        fri_final_poly_log_degree: k,
        format,
    }
}

pub(crate) fn sha3_hex(bytes: &[u8]) -> String {
    let d = Sha3_256::digest(bytes);
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// The pinned digests of one proof, as one line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Fingerprint {
    pub proof: String,
    pub fri_roots: String,
    pub num_fri_roots: usize,
    pub coeffs: String,
    pub queries: String,
    pub openings: String,
}

impl Fingerprint {
    pub(crate) fn line(&self) -> String {
        format!(
            "proof {} roots[{}] {} coeffs {} queries {} openings {}",
            self.proof,
            self.num_fri_roots,
            self.fri_roots,
            self.coeffs,
            self.queries,
            self.openings
        )
    }
}

/// The [`Fingerprint`] of any `StarkProof` (a macro: the rkyv serializer
/// bounds of a generic `StarkProof<F, E, PI>` are not worth spelling out).
macro_rules! fingerprint {
    ($proof:expr) => {{
        let proof = $proof;
        let rk = |bytes: Result<rkyv::util::AlignedVec, rkyv::rancor::Error>| {
            sha3_hex(&bytes.expect("rkyv"))
        };
        Fingerprint {
            proof: rk(rkyv::to_bytes::<rkyv::rancor::Error>(proof)),
            fri_roots: sha3_hex(&proof.fri_layers_merkle_roots.concat()),
            num_fri_roots: proof.fri_layers_merkle_roots.len(),
            coeffs: rk(rkyv::to_bytes::<rkyv::rancor::Error>(
                &proof.fri_final_poly_coeffs,
            )),
            queries: rk(rkyv::to_bytes::<rkyv::rancor::Error>(&proof.query_list)),
            openings: rk(rkyv::to_bytes::<rkyv::rancor::Error>(
                &proof.deep_poly_openings,
            )),
        }
    }};
}
#[allow(unused_imports)] // for the prover-free S3 tests in this crate
pub(crate) use fingerprint;

// ---------------------------------------------------------------------------
// The cases
// ---------------------------------------------------------------------------

/// `SimpleAddition` (E = F) under hash `H`.
pub(crate) fn prove_simple_addition<H: StarkHash>(
    rows: usize,
    options: &ProofOptions,
) -> (
    SimpleAdditionAIR<F>,
    StarkProof<F, F, SimpleAdditionPublicInputs<F>>,
) {
    let air = SimpleAdditionAIR::<F>::new(options);
    let pi = SimpleAdditionPublicInputs {
        a: Felt::from(1u64),
        b: Felt::from(2u64),
    };
    let mut trace = simple_addition_trace::<F>(rows);
    let proof = GenericProver::<F, F, _, H>::prove(
        &air,
        &mut trace,
        &pi,
        &mut DefaultTranscript::<F>::new(&[]),
    )
    .expect("proving must succeed");
    (air, proof)
}

pub(crate) fn verify_simple_addition<H: StarkHash>(
    air: &SimpleAdditionAIR<F>,
    proof: &StarkProof<F, F, SimpleAdditionPublicInputs<F>>,
) -> bool {
    GenericVerifier::<F, F, _, H>::verify(proof, air, &mut DefaultTranscript::<F>::new(&[]))
}

/// A continuous read-only memory over addresses 1..=5, `rows` reads.
fn logup_reads(rows: usize) -> (Vec<Felt>, Vec<Felt>) {
    let addr: Vec<Felt> = (0..rows).map(|i| Felt::from((i % 5) as u64 + 1)).collect();
    let val: Vec<Felt> = (0..rows)
        .map(|i| Felt::from(((i % 5) as u64 + 1) * 10))
        .collect();
    (addr, val)
}

/// `LogReadOnlyRAP` (E = F³, one aux column) under hash `H`.
pub(crate) fn prove_logup<H: StarkHash>(
    rows: usize,
    options: &ProofOptions,
) -> (
    LogReadOnlyRAP<F, E>,
    StarkProof<F, E, LogReadOnlyPublicInputs<F>>,
    LogReadOnlyPublicInputs<F>,
) {
    let (addr, val) = logup_reads(rows);
    let mut trace: TraceTable<F, E> = read_only_logup_trace(addr, val);
    let cols = trace.columns_main();
    let pi = LogReadOnlyPublicInputs {
        a0: cols[0][0],
        v0: cols[1][0],
        a_sorted_0: cols[2][0],
        v_sorted_0: cols[3][0],
        m0: cols[4][0],
    };
    let air = LogReadOnlyRAP::<F, E>::new(options);
    let proof = GenericProver::<F, E, _, H>::prove(
        &air,
        &mut trace,
        &pi,
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .expect("proving must succeed");
    (air, proof, pi)
}

pub(crate) fn verify_logup<H: StarkHash>(
    air: &LogReadOnlyRAP<F, E>,
    proof: &StarkProof<F, E, LogReadOnlyPublicInputs<F>>,
) -> bool {
    GenericVerifier::<F, E, _, H>::verify(proof, air, &mut DefaultTranscript::<E>::new(&[]))
}

/// The CPU/ADD/MUL bus instance (`residency_mode_tests`) under hash `H`.
pub(crate) fn prove_multi<H: StarkHash>(options: &ProofOptions) -> MultiProof<F, E, ()> {
    let (mut cpu_trace, mut add_trace, mut mul_trace) = super::residency_mode_tests::traces();
    let cpu_air = new_cpu_air_with_lookup(options);
    let add_air = new_add_air_with_lookup(options);
    let mul_air = new_mul_air_with_lookup(options);
    let pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];
    GenericProver::<F, E, (), H>::multi_prove(
        pairs,
        &mut DefaultTranscript::<E>::new(&[]),
        #[cfg(feature = "disk-spill")]
        crate::storage_mode::StorageMode::Ram,
        ResidencyMode::default(),
    )
    .expect("proving must succeed")
}

pub(crate) fn verify_multi<H: StarkHash>(
    options: &ProofOptions,
    proof: &MultiProof<F, E, ()>,
) -> bool {
    let cpu_air = new_cpu_air_with_lookup(options);
    let add_air = new_add_air_with_lookup(options);
    let mul_air = new_mul_air_with_lookup(options);
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];
    GenericVerifier::<F, E, (), H>::multi_verify(
        &airs,
        proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    )
}

/// Every golden case: (name, fingerprint line) computed now.
fn compute_goldens() -> Vec<(String, String)> {
    let mut out = Vec::new();
    let d = ProofFormat::DEFAULT;
    // SimpleAddition, k = 2: total_folds = log2(rows) − 2.
    for (hash, rows, blowup) in [
        ("keccak", 4usize, 2u8), // total_folds 0
        ("keccak", 8, 2),        // 1
        ("keccak", 16, 4),       // 2
        ("keccak", 256, 2),      // 6
        ("blake3", 8, 4),        // 1
        ("blake3", 64, 4),       // 4
    ] {
        let o = golden_options(blowup, 2, 5, d);
        let (air, proof) = match hash {
            "keccak" => prove_simple_addition::<KeccakStarkHash>(rows, &o),
            _ => prove_simple_addition::<Blake3StarkHash>(rows, &o),
        };
        let ok = match hash {
            "keccak" => verify_simple_addition::<KeccakStarkHash>(&air, &proof),
            _ => verify_simple_addition::<Blake3StarkHash>(&air, &proof),
        };
        assert!(ok, "golden case must verify");
        out.push((
            format!("simple_addition/{hash}/rows{rows}/blowup{blowup}"),
            fingerprint!(&proof).line(),
        ));
    }
    // LogReadOnlyRAP (ext3 + aux), k = 1.
    for (hash, rows, blowup) in [
        ("blake3", 16usize, 2u8), // total_folds 3
        ("blake3", 128, 4),       // 6
        ("keccak", 32, 4),        // 4
    ] {
        let o = golden_options(blowup, 1, 7, d);
        let (air, proof, _) = match hash {
            "keccak" => prove_logup::<KeccakStarkHash>(rows, &o),
            _ => prove_logup::<Blake3StarkHash>(rows, &o),
        };
        let ok = match hash {
            "keccak" => verify_logup::<KeccakStarkHash>(&air, &proof),
            _ => verify_logup::<Blake3StarkHash>(&air, &proof),
        };
        assert!(ok, "golden case must verify");
        out.push((
            format!("logup/{hash}/rows{rows}/blowup{blowup}"),
            fingerprint!(&proof).line(),
        ));
    }
    // Multi-table bus proof, k = 1 (CPU 8 rows, ADD/MUL 4 rows).
    let o = golden_options(2, 1, 6, d);
    let multi = prove_multi::<Blake3StarkHash>(&o);
    assert!(verify_multi::<Blake3StarkHash>(&o, &multi));
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&multi).expect("rkyv");
    let mut line = format!("multi {}", sha3_hex(&bytes));
    for (i, p) in multi.proofs.iter().enumerate() {
        line.push_str(&format!(" | table{i} {}", fingerprint!(p).line()));
    }
    out.push(("multi/blake3/blowup2".to_string(), line));
    out
}

/// Pinned at the default format (see the module docs).
const GOLDENS: &[(&str, &str)] = &[
    (
        "simple_addition/keccak/rows4/blowup2",
        "proof d72bff5491a61ff5a58c4677dbcf1a2daa17953ea1b976113e21a24b54bec6b7 roots[0] a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a coeffs 283be3dea88b9f9fc3012a6ec6f4dd9452c63f49bf5030c0119426aeca2e2ead queries 52d34b9f6d30aaf0b5bc5a4c5cb5c99909fcc4018af897ee04b9ff4beb69bc0c openings bf85016f5d66788f79a6f55ec69d9829d8f2afad2a792e9432d2e2bc6e85a391",
    ),
    (
        "simple_addition/keccak/rows8/blowup2",
        "proof df6437ee9bbbabb2d22dddfc8ac6888388328454cb6772bc52bd91d33e9c961e roots[0] a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a coeffs 1fa5e702cc4464b2fccd2c2ed05f9544aae1c1f81d6e63ec2907fa1225ee1309 queries 52d34b9f6d30aaf0b5bc5a4c5cb5c99909fcc4018af897ee04b9ff4beb69bc0c openings 02e305d0d828960f22c1c2d46dc01c2cdb04488a7cdaf66c97310061ffb34817",
    ),
    (
        "simple_addition/keccak/rows16/blowup4",
        "proof f513e31eca509f0eb72429e007724d368fa3466007c9bbd3cb0c1a4e62d0231b roots[1] 1b8a5d9014a32fd18488405253b1b382fd94681299ce888d2054b413dc9ffd4e coeffs 6763e71de233097b3c8a2563c8fc958f7dd8258c015d4602ef1be5c5f3d7d1ed queries 3b62f8adbf53c30c0a06b8b8e04c1c655e776ea7193a6eac577e1612a564db09 openings d3e7a59c53cfba8ec96b4014ddeff9c395b6a2b12f0ad4e49139f824de65241c",
    ),
    (
        "simple_addition/keccak/rows256/blowup2",
        "proof 0d21b0b5d2405473c93d17df09db5d3c771f87014e46e6aa26d9f2ee1f20caa8 roots[5] d274efdb442a2c9dec26ace00aadf90e47bb33c46cafc818d35aa2a44025ab82 coeffs 8ac737818bedf94a381b4b10f650dee2d300024d7a0bb9fa5f8200406c293971 queries 247e2e8c14f12cf87975bb5aabf89d7b4cb046e21d889bd736e3c9ab3b3891b3 openings cf8f7514ffd6b2df685f2cbcfaf5e5e37c60868b10c9c9186bd66b26267e2e7c",
    ),
    (
        "simple_addition/blake3/rows8/blowup4",
        "proof d6ebfa4b14b5e9bd6239c7a31e17a146e6995ade0b3101847df515176fe563ba roots[0] a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a coeffs 307d4a0b258158554a0f243871f919fcf4031414b3ecbdd39f287d2d31cba197 queries 52d34b9f6d30aaf0b5bc5a4c5cb5c99909fcc4018af897ee04b9ff4beb69bc0c openings 4632d0d62386964b4e3566b3af2271c6b0800ffe89894d39ddf020eaa5258695",
    ),
    (
        "simple_addition/blake3/rows64/blowup4",
        "proof 8caa0c1311ee2092d1a85c5e2fe6923466ebd8ed89edca123614d4127f1e674b roots[3] 9a7b1bf916dd5bdaee9ce8727a51131c055bcde12f780d48681cb6d51c543691 coeffs 9629f564cf5f72bc3b17b1c88d5864beb200da36ab5c0b2fea2f45905dc8b3f8 queries 0633449ec687a0c183ef547ee538d8620a51f3b3a0d42f570f9c50f5c3e13125 openings 24345624f78a948e8e006559792233857529c9612c603280be9e2779c50f7643",
    ),
    (
        "logup/blake3/rows16/blowup2",
        "proof 51b186c5c4c958d4de0c1359e32d6568780c0631c9b5e7289aa3bd1363997e88 roots[2] d6bad767d9d70cbd82c59d6ee0d13f4415e7ac4046ea2d606ebc305a6bac61ce coeffs ba8e4c0e358be69f2d39a246318d31ccf796086042c469ac060702006e79d62f queries 818f9de07a7ea34f7b1794aa0a62a77fd4c5d700e349c33aba6be5033b93a9d4 openings a7045f424c3096a7554d78e6ff535700eaa8b6af269bcbd6939b56f93bf8391d",
    ),
    (
        "logup/blake3/rows128/blowup4",
        "proof bea6e98f19c49e75401bcc9c21c73dbf3bf9576e5ce80a09a48f9dd5b70fa3ec roots[5] ceec0e7bc2ca1c16410c2222b96ce6d45d3e4573f54eb25378625bba375c4804 coeffs 17bb0241af6871c3359af6a56981dd842c667f6f1ca87227bda570f93c9a566b queries 1d84c98fdb64d057b0af98d2be55dd2c611cfc3cca5fea9abf1b5c1e86aaebc7 openings 4374ff4fab6a7bc892f8769661b6ad5abba2ac4979931c296f4b3d857b547369",
    ),
    (
        "logup/keccak/rows32/blowup4",
        "proof 2caf234c6858994f7a1910bb95b465c9ab7b5fc675f489f3e3fdc818b5d54798 roots[3] 92a3704249af017157fb755e570d4b07f73c151392119f4a8ffd1d96e2511123 coeffs c85284e687629ec7d505bb0723f10324896d4e7ebd05eb6aa3dd57459b5af0f6 queries 67dbdc533f359c0b48f3b14e704e086ac17d785bec9cd53b9891147ce1f925a9 openings ede54d2b9998a60133ac0cf63c88d5b47a487082899becea17711c0e7db55283",
    ),
    (
        "multi/blake3/blowup2",
        "multi eec51cb0e5701209e4709f89fb72b3af2dbc3038f2c80a185a2c74391ebffad5 | table0 proof 1079e69c4d47411814b05c5a7cc960d1c93846a3a873381c0184fa8273300cc3 roots[1] 1c7023dfeb09e6cc2ec141f6ec83e04f95e036ab1ae54a468248adde46a60d79 coeffs cb86ea5b8fe22227a96ad9d6ca4f68cb3bcd07825956d3ab804a98182304bfc8 queries 853284b820c6409aa20eab2a60863f4d1456deed856a8e9baadda3ecc754e7a3 openings e862a605d77e405c43605a4db1c05f49fefba0bda4d722d1dca33440ecf677ba | table1 proof 759ca54156cf6132fe61bd99f21528789bd60a21c190b3ffdc4caea20351f16d roots[0] a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a coeffs bd928b7e79613fa867a51d21ce6af7c18c3bd07aae44c0a83460486f50be9cfb queries c50b6659101c4ff74629092f4030534eec067bfaa650f3048807c4f2bba7ca72 openings 764b653d05d0cc14328bd94b8454420df2611a7d3465375b80235bfdb93f70ad | table2 proof b276a32877699f6e9e105ae05bd3bbfe1416d088187ceb22e710f968b9e5efd9 roots[0] a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a coeffs 1aef1aa9a22d64128d1469e69aa38886572f485071c6d6b046dd4f8bca60beb6 queries c50b6659101c4ff74629092f4030534eec067bfaa650f3048807c4f2bba7ca72 openings 9a7740f93e490a668f8f16dd9d4b00b79cab3e4561a85eecca03c8b5eafc13f6",
    ),
];

#[test]
fn default_format_goldens_are_byte_identical() {
    let got = compute_goldens();
    assert_eq!(got.len(), GOLDENS.len(), "one pin per case");
    for ((name, line), (pin_name, pin_line)) in got.iter().zip(GOLDENS) {
        assert_eq!(name, pin_name);
        assert_eq!(
            line, pin_line,
            "{name}: the default-format proof moved (a field whose digest differs is where)"
        );
    }
}

/// Prints `GOLDENS`. Run only for a deliberate format change.
#[test]
#[ignore = "generator for GOLDENS"]
fn print_goldens() {
    println!("const GOLDENS: &[(&str, &str)] = &[");
    for (name, line) in compute_goldens() {
        println!("    (\n        \"{name}\",\n        \"{line}\",\n    ),");
    }
    println!("];");
}
