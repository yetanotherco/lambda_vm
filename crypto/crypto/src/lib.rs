#![allow(clippy::op_ref)]
#![cfg_attr(not(feature = "std"), no_std)]

// `std` pulls in `memmap2` (used by `crypto/stark`'s disk-backed Merkle node
// storage), which doesn't compile on wasm32. Fail loudly here so downstream
// crates that depend on `crypto/crypto` directly with `std` get a clear
// message instead of a transitive memmap2 build error.
#[cfg(all(target_arch = "wasm32", feature = "std"))]
compile_error!(
    "wasm32 targets are not supported with feature \"std\": StorageMode::Disk \
     requires memmap2, which does not compile on wasm32"
);

#[macro_use]
extern crate alloc;

pub mod fiat_shamir;
pub mod hash;
pub mod merkle_tree;

#[cfg(test)]
pub mod tests;
