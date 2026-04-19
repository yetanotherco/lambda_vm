use p3_challenger::{HashChallenger, SerializingChallenger64};
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_goldilocks::Goldilocks;
use p3_keccak::{Keccak256Hash, KeccakF};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
use p3_uni_stark::StarkConfig;

pub type Val = Goldilocks;

/// Cubic extension matching Lambda's `Degree3GoldilocksExtensionField`
/// (irreducible x^3 - 2). Provided by the forked `p3-goldilocks` via
/// `BinomiallyExtendable<3>`.
pub type Challenge = BinomialExtensionField<Val, 3>;

type ByteHash = Keccak256Hash;
type U64Hash = PaddingFreeSponge<KeccakF, 25, 17, 4>;
type FieldHash = SerializingHasher<U64Hash>;
type MyCompress = CompressionFunctionFromHasher<U64Hash, 2, 4>;
pub type ValMmcs = MerkleTreeMmcs<
    [Val; p3_keccak::VECTOR_LEN],
    [u64; p3_keccak::VECTOR_LEN],
    FieldHash,
    MyCompress,
    2,
    4,
>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Dft = Radix2DitParallel<Val>;
pub type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
pub type Challenger = SerializingChallenger64<Val, HashChallenger<u8, ByteHash, 32>>;

pub type P3Config = StarkConfig<Pcs, Challenge, Challenger>;

fn build_mmcs() -> (ValMmcs, ChallengeMmcs, ByteHash) {
    let byte_hash = ByteHash {};
    let u64_hash = U64Hash::new(KeccakF {});
    let field_hash = FieldHash::new(u64_hash);
    let compress = MyCompress::new(u64_hash);
    let val_mmcs = ValMmcs::new(field_hash, compress, 3);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    (val_mmcs, challenge_mmcs, byte_hash)
}

/// Creates a Plonky3 STARK config with parameters matched to Lambda's
/// production config `GoldilocksCubicProofOptions::with_blowup(2)`:
/// blowup=2, 219 FRI queries, grinding=0 (excluded from benchmark).
pub fn matched_params_config() -> P3Config {
    let (val_mmcs, challenge_mmcs, byte_hash) = build_mmcs();
    let dft = Dft::default();
    let challenger = Challenger::from_hasher(vec![], byte_hash);

    // Match Lambda production: blowup=2, queries=219, grinding=0.
    // Grinding excluded from benchmark (identical PoW on both sides).
    let fri_params = FriParameters {
        log_blowup: 1, // blowup = 2
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 219,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 0,
        mmcs: challenge_mmcs,
    };

    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    P3Config::new(pcs, challenger)
}

/// Creates a Plonky3 STARK config with Plonky3's standard benchmark parameters:
/// blowup=2, 100 FRI queries, 16-bit query PoW.
pub fn plonky3_benchmark_config() -> P3Config {
    let (val_mmcs, challenge_mmcs, byte_hash) = build_mmcs();
    let dft = Dft::default();
    let challenger = Challenger::from_hasher(vec![], byte_hash);

    let fri_params = FriParameters::new_benchmark(challenge_mmcs);

    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    P3Config::new(pcs, challenger)
}
