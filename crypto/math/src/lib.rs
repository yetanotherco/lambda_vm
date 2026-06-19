#![cfg_attr(not(feature = "std"), no_std)]
// hax pins an old nightly (2025-11-08) whose `core` still gates `cold_path`
// behind a feature flag, even though it stabilized in Rust 1.95. The flag is
// enabled only under the `hax` cfg so the stable production build is untouched
// and the real `cold_path()` calls stay in scope for extraction.
#![cfg_attr(hax, feature(cold_path))]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod errors;
pub mod field;
pub mod spill_safe;
pub mod traits;
pub mod unsigned_integer;

// These modules don't work in no-std mode
pub mod fft;
#[cfg(feature = "alloc")]
pub mod polynomial;

#[cfg(test)]
pub mod tests;
