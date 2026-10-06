//! ZisK's Poseidon1 host configuration, as the recursion's emitters and tests
//! name it: the Merkle backends and the Fiat–Shamir sponge live in `crypto`
//! (`merkle_tree::backends::p1`, `fiat_shamir::p1_transcript`), where the WHIR
//! crates can reach them; this module re-exports them under the names the
//! in-guest constructions (`super::p1w16_emit`) are checked against.

pub use crypto::fiat_shamir::p1_transcript::{P1GrindDigest, P1Transcript, P1TranscriptHash};
pub use crypto::merkle_tree::backends::p1::{
    P1BatchBackend, P1LeafHasher, P1PairBackend, leaf, node4,
};
