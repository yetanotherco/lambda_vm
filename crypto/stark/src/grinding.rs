//! The grinding primitive and its device dispatch, re-exported so existing
//! call sites read unchanged. Both live in [`crypto::grinding`], which the
//! multilinear prover can also reach.

pub use crypto::grinding::{
    generate_nonce, generate_nonce_maybe_gpu, inner_hash_lanes, is_valid_nonce,
};
