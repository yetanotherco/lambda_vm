use p3_challenger::{HashChallenger, SerializingChallenger64};
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::CubicTrinomialExtensionField;
use p3_field::{Field, PackedValue};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_goldilocks::Goldilocks;
use p3_keccak::Keccak256Hash;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_uni_stark::StarkConfig;

pub type Val = Goldilocks;
pub type Challenge = CubicTrinomialExtensionField<Val>;

type ByteHash = Keccak256Hash;
type FieldHash = SerializingHasher<ByteHash>;
type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
pub type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 2, 32>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Dft = Radix2DitParallel<Val>;
pub type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
pub type Challenger = SerializingChallenger64<Val, HashChallenger<u8, ByteHash, 32>>;
pub type P3Config = StarkConfig<Pcs, Challenge, Challenger>;

fn build_mmcs() -> (ValMmcs, ChallengeMmcs, ByteHash) {
    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(byte_hash);
    let compress = MyCompress::new(byte_hash);
    let val_mmcs = ValMmcs::new(field_hash, compress, 3);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    (val_mmcs, challenge_mmcs, byte_hash)
}

pub fn params_config(blowup: u8, queries: usize, grinding: u8) -> P3Config {
    assert!(blowup.is_power_of_two(), "blowup must be a power of two");

    let (val_mmcs, challenge_mmcs, byte_hash) = build_mmcs();
    let dft = Dft::default();
    let challenger = Challenger::from_hasher(vec![], byte_hash);
    let fri_params = FriParameters {
        log_blowup: blowup.trailing_zeros() as usize,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: queries,
        commit_proof_of_work_bits: grinding as usize,
        query_proof_of_work_bits: 0,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    P3Config::new(pcs, challenger)
}

pub fn val_packing_width() -> usize {
    <Val as Field>::Packing::WIDTH
}

pub fn hash_lanes() -> usize {
    1
}
