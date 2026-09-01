#![allow(clippy::op_ref)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(all(target_arch = "wasm32", feature = "disk-spill"))]
compile_error!("the `disk-spill` feature requires memmap2, which does not compile on wasm32");

#[macro_use]
extern crate alloc;

pub mod fiat_shamir;
pub mod hash;
#[cfg(feature = "hash-count")]
pub mod hash_count;
pub mod merkle_tree;
#[cfg(feature = "disk-spill")]
pub mod mmap_util;

#[cfg(test)]
pub mod tests;
