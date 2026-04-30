// `StorageMode::Disk` is implemented via `memmap2`, which doesn't compile on
// wasm32. Fail loudly at the top of the crate rather than via a confusing
// transitive memmap2 error deeper in the dep graph.
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
pub mod grinding;
#[cfg(feature = "instruments")]
pub mod instruments;
pub mod lookup;
pub mod proof;
pub mod prover;
#[cfg(feature = "disk-spill")]
pub mod storage_mode;
pub mod table;
pub mod trace;
pub mod traits;
pub mod verifier;

#[cfg(test)]
pub mod tests;

/// Configurations of the Prover available in compile time
pub mod config;
