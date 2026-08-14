use crate::hash::blake3::Blake3Chain;
use crate::hash::platform_keccak::PlatformKeccak256 as Keccak256;

use super::field_element_vector::{FieldElementPairBackend, FieldElementVectorBackend};

// Vector of field elements backend definitions
pub type BatchKeccak256Backend<F> = FieldElementVectorBackend<F, Keccak256, 32>;

// Fixed-size pair backends (more efficient for FRI layers)
pub type PairKeccak256Backend<F> = FieldElementPairBackend<F, Keccak256, 32>;

/// The BLAKE3 batched-leaf backend, over [`Blake3Chain`] — the single-chunk
/// chain specified in PA-PLAN §1.7.
///
/// It is the *same* generic backend the keccak alias is, with the digest
/// swapped, and that is load-bearing rather than an economy. The leaf byte
/// layout, both streaming routes and the parent framing then have one definition
/// each (`field_element_vector.rs`), so the batched and paired families cannot
/// encode a two-element leaf differently: the invariant `stark::config::StarkHash`
/// requires holds because they are one function, not because two implementations
/// were shown to coincide.
///
/// A parent is `Blake3Chain` over the two concatenated 32-byte nodes — 64 bytes,
/// so one compression with `h = IV`, `t = 0`, `block_len = 64`, `flags =
/// CHUNK_START | CHUNK_END | ROOT`. That is bit-for-bit what the device kernel
/// computes (`math-cuda/kernels/blake3.cu`, `blake3_hash_merkle_parent`), which
/// is what will let a GPU tree and a CPU tree be the same tree once the device
/// leaf kernels land.
pub type BatchBlake3Backend<F> = FieldElementVectorBackend<F, Blake3Chain, 32>;

/// The FRI-layer twin of [`BatchBlake3Backend`] — one leaf per fixed pair, no
/// `Vec` per leaf. See there.
pub type PairBlake3Backend<F> = FieldElementPairBackend<F, Blake3Chain, 32>;
