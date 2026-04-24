// `StorageMode::Disk` is backed by `memmap2` / `tempfile` / `libc`, none of
// which compile for `wasm32-*`. Fail loudly rather than via a confusing
// transitive error deep inside a dependency.
#[cfg(target_arch = "wasm32")]
compile_error!("stark crate does not support wasm32 targets (memmap2 is required)");

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
pub mod storage_mode;
pub mod table;
pub mod trace;
pub mod traits;
pub mod verifier;

#[cfg(test)]
pub mod tests;

/// Configurations of the Prover available in compile time
pub mod config;
