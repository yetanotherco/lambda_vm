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
//!
//! - [`shape`] — which table contributes which matrix to which round, derived
//!   from the AIR set on both sides and never read from a proof.
//! - [`round4`] — the round-4 transcript sequence and the per-query FRI check.
//! - [`proof`] — what a batched epoch proof carries.
//! - [`prover`] — the phase architecture the barriers force.
//! - [`verifier`] — the transcript replay, and ⛔ only the commitment half of a
//!   verification. Read its header before assuming otherwise.

#[cfg(feature = "cuda")]
pub mod gpu;
pub mod proof;
pub mod prover;
pub mod round4;
pub mod shape;
pub mod verifier;
