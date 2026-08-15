//! The batched-commitment proving path: one mixed-height MMCS per round and one
//! FRI instance per epoch, instead of one tree and one FRI instance per table.
//!
//! This is an OPT-IN path. The per-table prover and verifier
//! ([`crate::prover::IsStarkProver::multi_prove`],
//! [`crate::verifier::IsStarkVerifier::multi_verify`]) are untouched and produce
//! byte-identical proofs; nothing here is reachable from them.
//!
//! The primitives live one level down — [`crate::fri::mmcs`] (the mixed-height
//! tree) and [`crate::fri::batched`] (height combination, the batched commit
//! phase, and the shared challenge derivation). This module is the wiring: it
//! fixes the transcript sequence, the query-index convention and the per-query
//! fold-with-injection recursion that the prover and the verifier must agree on.

pub mod round4;
