//! Shape derivation for a multi-matrix preprocessed round.
//!
//! [`shape`] records which table contributes which matrix to which round,
//! derived from the AIR set rather than read from a proof. The LFM registry
//! pins a preprocessed round root over several slots' matrices and needs that
//! description to say which widths the round covers
//! (`prover/src/lfm/registry.rs::pinned_prep_widths`).

pub mod shape;
