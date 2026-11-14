//! ZisK's Poseidon1 host configuration, as the recursion's emitters and tests
//! name it: the Merkle backends and the Fiat–Shamir sponge live in `crypto`
//! (`merkle_tree::backends::p1`, `fiat_shamir::p1_transcript`), where the WHIR
//! crates can reach them; this module re-exports them under the names the
//! in-guest constructions (`super::p1w16_emit`) are checked against, and names
//! the STARK configuration P1 proofs commit under.

pub use crypto::fiat_shamir::p1_transcript::{P1GrindDigest, P1Transcript, P1TranscriptHash};
pub use crypto::merkle_tree::backends::p1::{
    P1BatchBackend, P1LeafHasher, P1PairBackend, leaf, node4,
};
pub use stark::config::P1StarkHashCapped;

/// The 4-ary cap height of every Poseidon1 STARK tree (trace, precomputed,
/// aux, composition, FRI layers): C2, `c_L` for both PRs (lead, 10-11). #1013's
/// box arms measured C2 1.46 s faster in median recursion than C1 with 2.9 GiB
/// less peak VRAM (ULTRA 131), and the node census under the hint-order walk
/// puts C2 4.5 % under C1 (I-P1REC §P2.5).
pub const P1_STARK_CAP: u8 = 2;

/// ★ ZisK's Poseidon1 as a STARK commitment configuration: 4-ary trees of
/// ZisK's tagged leaf hash, capped at 4-ary height [`P1_STARK_CAP`], ground
/// with the width-8 permutation. Prove and verify it with [`P1Transcript`].
pub type P1StarkHash = P1StarkHashCapped<P1_STARK_CAP>;
