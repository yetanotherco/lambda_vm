#![allow(clippy::op_ref)]
#![cfg_attr(not(feature = "std"), no_std)]

#[macro_use]
extern crate alloc;

pub mod fiat_shamir;
pub mod hash;
pub mod merkle_tree;

#[cfg(test)]
pub mod tests;
