use p3_challenger::{HashChallenger, SerializingChallenger64};
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::CubicTrinomialExtensionField;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_goldilocks::Goldilocks;
use p3_keccak::Keccak256Hash;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_uni_stark::StarkConfig;

pub type Val = Goldilocks;
pub type Challenge = CubicTrinomialExtensionField<Val>;

// Scalar byte-oriented MMCS, deliberately not the Plonky3 production config.
// Leaves are individual field elements, digests are 32 raw bytes, and the
// underlying Keccak path is single-input tiny_keccak. This removes the
// `[Val; VECTOR_LEN]` / `[u64; VECTOR_LEN]` Keccak lanes that the
// vector-friendly upstream config uses (NEON=2, SSE2=2, AVX2=4, AVX-512=8),
// so the Merkle compression cost is one Keccak-f per call on both sides.
type ByteHash = Keccak256Hash;
type FieldHash = SerializingHasher<ByteHash>;
type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
pub type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 2, 32>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Dft = Radix2DitParallel<Val>;
pub type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
pub type Challenger = SerializingChallenger64<Val, HashChallenger<u8, ByteHash, 32>>;

pub type P3Config = StarkConfig<Pcs, Challenge, Challenger>;

/// Packing width of the MMCS leaves (`P` parameter of `MerkleTreeMmcs`).
/// `Val` directly = 1; `[Val; N]` would be `N`. Exposed for the AUDIT line.
pub const VAL_PACKING_WIDTH: usize = 1;

/// Lanes of the underlying Keccak permutation as seen by the MMCS.
/// `Keccak256Hash` is single-input scalar; lane-vectorized `KeccakF` paths
/// would set this to 2/4/8 depending on arch.
pub const HASH_LANES: usize = 1;

fn build_mmcs() -> (ValMmcs, ChallengeMmcs, ByteHash) {
    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(byte_hash);
    let compress = MyCompress::new(byte_hash);
    let val_mmcs = ValMmcs::new(field_hash, compress, 3);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    (val_mmcs, challenge_mmcs, byte_hash)
}

/// Creates a Plonky3 STARK config with parameters matched to Lambda's proof
/// options. `blowup` must be a power of two because Plonky3 stores it as
/// `log_blowup`.
pub fn params_config(blowup: u8, queries: usize, grinding: u8) -> P3Config {
    assert!(
        blowup.is_power_of_two(),
        "blowup must be a power of two for Plonky3"
    );

    let (val_mmcs, challenge_mmcs, byte_hash) = build_mmcs();
    let dft = Dft::default();
    let challenger = Challenger::from_hasher(vec![], byte_hash);

    let fri_params = FriParameters {
        log_blowup: blowup.trailing_zeros() as usize,
        log_final_poly_len: 0,
        // Radix-2 FRI folding (one fold per round) to match Lambda's
        // `fold_evaluations_in_place` (N -> N/2). Plonky3's production
        // config uses arity 8-16 (max_log_arity = 3-4) for fewer rounds.
        max_log_arity: 1,
        num_queries: queries,
        commit_proof_of_work_bits: grinding as usize,
        query_proof_of_work_bits: 0,
        mmcs: challenge_mmcs,
    };

    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    P3Config::new(pcs, challenger)
}

/// Creates a Plonky3 STARK config with parameters matched to Lambda's
/// production config `GoldilocksCubicProofOptions::with_blowup(2)`:
/// blowup=2, 219 FRI queries, grinding=0.
pub fn matched_params_config() -> P3Config {
    params_config(2, 219, 0)
}
