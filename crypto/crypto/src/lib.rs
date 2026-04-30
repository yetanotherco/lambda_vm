#![allow(clippy::op_ref)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(all(target_arch = "wasm32", feature = "disk-spill"))]
compile_error!("the `disk-spill` feature requires memmap2, which does not compile on wasm32");

#[macro_use]
extern crate alloc;

pub mod fiat_shamir;
pub mod hash;
pub mod merkle_tree;

#[cfg(test)]
pub mod tests;
