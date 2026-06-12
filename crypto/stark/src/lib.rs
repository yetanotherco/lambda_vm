// `StorageMode::Disk` uses `memmap2`, which does not build on wasm32.
// Fail at the crate root rather than as a transitive memmap2 error.
#[cfg(all(target_arch = "wasm32", feature = "disk-spill"))]
compile_error!("the `disk-spill` feature requires memmap2, which does not compile on wasm32");

#[cfg(feature = "debug-checks")]
pub mod bus_debug;
pub mod constraints;
pub mod context;
pub mod debug;
pub mod domain;
#[cfg(any(test, feature = "test-utils"))]
pub mod examples;
pub mod frame;
pub mod fri;
#[cfg(feature = "cuda")]
pub mod gpu_lde;
pub mod grinding;
#[cfg(feature = "instruments")]
pub mod instruments;
pub mod lookup;
pub(crate) mod par;
pub mod proof;
pub mod prover;
pub mod r4_denoms;
#[cfg(feature = "disk-spill")]
pub mod storage_mode;
pub mod table;
pub mod trace;
pub mod traits;
pub mod verifier;

#[cfg(test)]
pub mod test_utils;
#[cfg(test)]
pub mod tests;

/// Configurations of the Prover available in compile time
pub mod config;
