#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod constants;
pub mod elf;
#[cfg(feature = "std")]
pub mod flamegraph;
#[cfg(test)]
pub mod tests;
// `profile` uses std (BTreeMap, io::Write), so gate it like `flamegraph` to
// keep the no_std guest build (riscv64im-lambda-vm-elf) working.
#[cfg(feature = "std")]
pub mod profile;
#[cfg(feature = "std")]
pub mod vm;
