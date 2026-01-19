//! Tests for LogUp bus interactions.

pub mod completeness_tests;
pub mod packing_tests;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub mod property_tests;
pub mod soundness_tests;
