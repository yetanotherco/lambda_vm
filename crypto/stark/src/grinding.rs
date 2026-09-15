//! The grinding primitive and its device dispatch, re-exported so existing
//! call sites read unchanged. Both live in [`crypto::grinding`], which the
//! multilinear prover can also reach.
//!
//! ⚠ The primitive is generic over its hash and carries **no default** — a
//! proof-of-work hash that defaulted would silently keep grinding on keccak for
//! a configuration that had moved everything else. This crate's univariate
//! prover and verifier are keccak throughout, so they name
//! [`StarkGrindingDigest`] at each call site: one token, but it is a statement
//! rather than an omission, and it is the thing that has to change if this path
//! ever gains a hash parameter of its own.

pub use crypto::grinding::{
    generate_nonce, generate_nonce_maybe_gpu, inner_hash_lanes, is_valid_nonce,
};

/// The hash the univariate prover and verifier grind with: keccak-256, the same
/// hash their Merkle trees and their transcript use.
pub type StarkGrindingDigest = crypto::hash::platform_keccak::PlatformKeccak256;
