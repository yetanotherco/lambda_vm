#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod constants;
pub mod elf;
#[cfg(feature = "std")]
pub mod flamegraph;
#[cfg(test)]
pub mod tests;
#[cfg(feature = "std")]
pub mod vm;
