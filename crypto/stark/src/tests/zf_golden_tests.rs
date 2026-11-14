//! Default-format golden proofs: the bytes today's prover emits,
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
//! Generated at the default format BEFORE any S3 prover code existed (the
//! schedule DP alone changes no prover path). Regenerate only for a deliberate format change:
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
            $crate::tests::zf_golden_tests::sha3_hex(&bytes.expect("rkyv"))
        };
        $crate::tests::zf_golden_tests::Fingerprint {
            proof: rk(rkyv::to_bytes::<rkyv::rancor::Error>(proof)),
            fri_roots: $crate::tests::zf_golden_tests::sha3_hex(
                &proof.fri_layers_merkle_roots.concat(),
            ),
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
/// An AIR, its proof and its public inputs.
pub(crate) type LogupCase = (
    LogReadOnlyRAP<F, E>,
    StarkProof<F, E, LogReadOnlyPublicInputs<F>>,
    LogReadOnlyPublicInputs<F>,
);

pub(crate) fn prove_logup<H: StarkHash>(rows: usize, options: &ProofOptions) -> LogupCase {
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
        "proof 85b48cca0e29044adfddd2a53b3fc5ee5701a00ae1fadb971e4cb739ba4206df roots[0] a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a coeffs 0efd0a46791ad9dfbc92a2172d9ff442c898a1254f27d1f4ff6752a042cc92e4 queries 52d34b9f6d30aaf0b5bc5a4c5cb5c99909fcc4018af897ee04b9ff4beb69bc0c openings 13e048f2e2dc98c9aeaa112fe2db4e3f1fbf880ea02cc431c652c9c0cc79efd6",
    ),
    (
        "simple_addition/blake3/rows64/blowup4",
        "proof 3046508786b7deb4e94bab1da95b98ae16941c76576ad0872995bff85a2ebf65 roots[3] 789b306926fe325a5ffff55ea041761118afa93b61ed81002dff3b0212d872c4 coeffs 3e9b68ff75e3e978fc26ad4e2d5b5038b0edf03b75e7ee673f8be18a1c9ddf09 queries 4005eeb6e4389e8d4284a49f7b10c37c7314e90ed36a52f834cf1f40d4051ee3 openings c09b74112a3bab1a324e72a5b97778bad12790783a181b02783dc50404a64841",
    ),
    (
        "logup/blake3/rows16/blowup2",
        "proof 80b6ea3dad96d25eb5e5a15ae86c2428f889e7c8482018904777fb4e1cdea526 roots[2] 7afa53dda48561c9714811d640fb8400af722c934c5cda2bf578fcb0a6b80909 coeffs 7323ff8f002587f3197f7cfed56a1a47a9122b1ca4486af7bad467076fef7ccd queries 5c731237a5571ac44f1d4505b15c7605b8c999d459d83cf0dd5e8f7fe03281f9 openings 4e87865ba48369fc1462f06cd62a82223c6ba8d498318f8748266fb88cb7de28",
    ),
    (
        "logup/blake3/rows128/blowup4",
        "proof eff9e180ee04f3ae490f759d2d45327b6b2072d12b66db687d0f4abe6967dcc7 roots[5] 16d3ab6a1e53733112c9226d60ea16267d6a5d3b585b2fa9c9ebe188839b8e51 coeffs f7da2b51582af363c73233a07f4f279deb3be7d24a052cc4feb350e1d5f22197 queries 3c01de90eb63e9a45e2f3f58982b01df2bea4e5af49150a8046b5d597ab43f06 openings 0e17381ff7c01b79b2d70f9e5499811eddc16fbfd5cae1c97227a5c4f5afe74f",
    ),
    (
        "logup/keccak/rows32/blowup4",
        "proof 2caf234c6858994f7a1910bb95b465c9ab7b5fc675f489f3e3fdc818b5d54798 roots[3] 92a3704249af017157fb755e570d4b07f73c151392119f4a8ffd1d96e2511123 coeffs c85284e687629ec7d505bb0723f10324896d4e7ebd05eb6aa3dd57459b5af0f6 queries 67dbdc533f359c0b48f3b14e704e086ac17d785bec9cd53b9891147ce1f925a9 openings ede54d2b9998a60133ac0cf63c88d5b47a487082899becea17711c0e7db55283",
    ),
    (
        "multi/blake3/blowup2",
        "multi d26dbdc86e6b1a705130aa1fd7d0c4b37abb3fb2c3dbeb1c4531022897e63fef | table0 proof 61206972740c227d38817bbfe356671eafd9e30127c4f50e48c494fb9f32bb7a roots[1] d0642e01368afc00fe8c93f63a651456c22fb4ed66e436dd0197b3db96e8badc coeffs ad53ee463eeffd635c729f70ce9f2d6eee99915d8ab41d71ed2dd3e44f7d7f30 queries cd6253544fab288ac6b444674682c783136f9a844f4063e77c457d1a0344dc45 openings 3b3874c263052f7b3ff3a88a46a5fe6d61ed4ff1269bb0c3c31902948622e78f | table1 proof 9ffd29030795ecf21e45dcc2bf8a7a5346b42e7c9a686f71313962f32225afd8 roots[0] a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a coeffs 316b3db1992f4aa50cd501ccd8d77439b52632343967e7990cd80b8527249c14 queries c50b6659101c4ff74629092f4030534eec067bfaa650f3048807c4f2bba7ca72 openings 662020c13182487630717ec7c77193eb9d1b38f5df809af628c43b4f7009ba2c | table2 proof ba5c15e4b05b1937447f949844a967b24d1561ec0a8fc34a492ab0e481b66b2b roots[0] a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a coeffs 4f4c3fa3c5f1051fde857f12e1a886db267212dcb44396fc1ab53cbb270bf344 queries c50b6659101c4ff74629092f4030534eec067bfaa650f3048807c4f2bba7ca72 openings c9a99e4d1a172cf471a4544dfe3e47266f3b90da6222fe2331c92da5913efcc7",
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

/// The sub-coset composition evaluation (two parts at blowup ≥ 4: `H` on
/// `2n` points instead of the whole LDE) must not move a byte: the same case
/// proved with it forced off and forced on gives the same rkyv bytes, both
/// verify, and the "on" prove really took the sub-coset path. The pinned
/// goldens above run under the default (on); this compares the two paths
/// directly, so it also holds if the default changes.
#[test]
fn subcoset_composition_is_byte_identical_to_whole_coset() {
    use crate::prover::{composition_subcoset_tables, set_composition_subcoset};
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_composition_subcoset(None);
        }
    }
    let _restore = Restore;

    for (rows, blowup) in [(128usize, 4u8), (64, 8)] {
        let o = golden_options(blowup, 1, 7, ProofFormat::DEFAULT);

        set_composition_subcoset(Some(false));
        let (air_off, off, _) = prove_logup::<Blake3StarkHash>(rows, &o);

        set_composition_subcoset(Some(true));
        let before = composition_subcoset_tables();
        let (air_on, on, _) = prove_logup::<Blake3StarkHash>(rows, &o);
        assert!(
            composition_subcoset_tables() > before,
            "rows{rows}/blowup{blowup}: the forced-on prove never took the sub-coset path"
        );
        set_composition_subcoset(None);

        assert!(verify_logup::<Blake3StarkHash>(&air_off, &off));
        assert!(verify_logup::<Blake3StarkHash>(&air_on, &on));
        let off_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&off).expect("rkyv");
        let on_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&on).expect("rkyv");
        assert_eq!(
            sha3_hex(&on_bytes),
            sha3_hex(&off_bytes),
            "rows{rows}/blowup{blowup}: sub-coset proof bytes differ from the whole-coset proof"
        );
    }

    // The multi-table bus proof (CPU/ADD/MUL, LogUp, one transcript) at blowup 4.
    let o = golden_options(4, 1, 6, ProofFormat::DEFAULT);
    set_composition_subcoset(Some(false));
    let off = prove_multi::<Blake3StarkHash>(&o);
    set_composition_subcoset(Some(true));
    let before = composition_subcoset_tables();
    let on = prove_multi::<Blake3StarkHash>(&o);
    assert!(
        composition_subcoset_tables() > before,
        "multi/blowup4: the forced-on prove never took the sub-coset path"
    );
    set_composition_subcoset(None);
    assert!(verify_multi::<Blake3StarkHash>(&o, &off));
    assert!(verify_multi::<Blake3StarkHash>(&o, &on));
    let off_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&off).expect("rkyv");
    let on_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&on).expect("rkyv");
    assert_eq!(
        sha3_hex(&on_bytes),
        sha3_hex(&off_bytes),
        "multi/blowup4: sub-coset proof bytes differ from the whole-coset proof"
    );
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

/// The prove-constants warm-up (host side) builds the boundary zerofier
/// inverses a prove would build on first use, and a prove after it has the
/// bytes of one without it.
#[test]
fn a_warmed_prove_has_the_unwarmed_bytes() {
    use crate::prover::{warm_prove_constants, warmed_prove_constants};
    let o = golden_options(4, 1, 7, ProofFormat::DEFAULT);
    let rows = 1usize << 9;
    let (_, cold, pi) = prove_logup::<Blake3StarkHash>(rows, &o);
    let air = LogReadOnlyRAP::<F, E>::new(&o);
    let before = warmed_prove_constants();
    let tables: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = _>,
        usize,
        &_,
    )> = vec![(&air, rows, &pi)];
    warm_prove_constants(&tables);
    assert!(
        warmed_prove_constants() > before,
        "a LogUp table has a boundary constraint"
    );
    let (air_w, warm, _) = prove_logup::<Blake3StarkHash>(rows, &o);
    assert!(verify_logup::<Blake3StarkHash>(&air_w, &warm));
    let b = |p| sha3_hex(&rkyv::to_bytes::<rkyv::rancor::Error>(p).expect("rkyv"));
    assert_eq!(b(&cold), b(&warm));
}
