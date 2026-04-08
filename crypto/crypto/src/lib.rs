#![allow(clippy::op_ref)]
#![cfg_attr(not(feature = "std"), no_std)]
#[macro_use]
extern crate alloc;

pub mod fiat_shamir;
pub mod hash;
pub mod merkle_tree;

/// BufWriter capacity for disk-spill writes (16 MB).
/// Overrides the default 8 KB to reduce write syscall overhead for large spills.
#[cfg(feature = "disk-spill")]
pub const SPILL_BUF_CAPACITY: usize = 16 << 20;

#[cfg(test)]
pub mod tests;
