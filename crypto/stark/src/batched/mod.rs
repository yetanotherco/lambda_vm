//! The batched-commitment path: one mixed-height MMCS per round and one FRI
//! instance per epoch, instead of one tree and one FRI instance per table.
//!
//! Brought over from PR #951 piece by piece. The per-table prover and verifier
//! are untouched and produce byte-identical proofs; nothing here is reachable
//! from them.
//!
//! What is NOT brought over is #951's own driver: it takes every table's trace
//! at once, which is the residency this branch exists to remove. Its phase
//! sequence is the specification the driver here follows — each "per table" step
//! is a pass over the execution, each barrier a point where one ends.

pub mod round4;
