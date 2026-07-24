#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod errors;
pub mod field;
/// Measurement-only ABI structs for the mid-level accelerator stub ecalls (sim/27).
pub mod sim_midlevel;
/// Measurement-only ABI structs for the reduced-opening stub ecalls.
pub mod sim_ro;
pub mod spill_safe;
pub mod traits;
pub mod unsigned_integer;

// These modules don't work in no-std mode
pub mod fft;
#[cfg(feature = "alloc")]
pub mod polynomial;

#[cfg(test)]
pub mod tests;
