/// Serializes tests that toggle the process-global `LAMBDA_VM_LEGACY_TRACEGEN` env var.
/// MUST be shared by every such test (across files) — a per-file mutex would not serialize
/// them, letting one file's `remove_var` race a concurrent file's legacy build.
#[cfg(test)]
pub static TRACEGEN_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(all(test, feature = "disk-spill"))]
pub mod auto_storage_tests;
#[cfg(test)]
pub mod bitwise_bus_tests;
#[cfg(test)]
pub mod bitwise_tests;
#[cfg(test)]
pub mod branch_bus_tests;
#[cfg(test)]
pub mod branch_constraints_tests;
#[cfg(test)]
pub mod bytewise_tests;
#[cfg(test)]
pub mod commit_tests;
#[cfg(test)]
pub mod compute_commit_bus_offset_tests;
#[cfg(test)]
pub mod constraints_tests;
#[cfg(all(test, feature = "disk-spill"))]
pub mod count_table_lengths_drift_tests;
#[cfg(test)]
pub mod cpu32_tests;
#[cfg(test)]
pub mod cpu_tests;
#[cfg(test)]
pub mod decode_layout_tests;
#[cfg(test)]
pub mod decode_tests;
#[cfg(all(test, feature = "disk-spill"))]
pub mod disk_spill_tests;
#[cfg(test)]
pub mod dvrm_tests;
#[cfg(test)]
pub mod ec_scalar_tests;
#[cfg(test)]
pub mod ecdas_tests;
#[cfg(test)]
pub mod ecsm_tests;
#[cfg(test)]
pub mod eq_tests;
#[cfg(test)]
pub mod keccak_rnd_tests;
#[cfg(test)]
pub mod load_tests;
#[cfg(test)]
pub mod local_to_global_bus_tests;
#[cfg(test)]
pub mod lt_bus_tests;
#[cfg(test)]
pub mod lt_tests;
#[cfg(test)]
pub mod memw_aligned_tests;
#[cfg(test)]
pub mod bitwise_histogram_parity_tests;
#[cfg(test)]
pub mod memw_register_direct_parity_tests;
#[cfg(test)]
pub mod memw_register_tests;
#[cfg(test)]
pub mod memw_tests;
#[cfg(test)]
pub mod mul_tests;
#[cfg(test)]
pub mod page_tests;
#[cfg(test)]
pub mod prove_elfs_tests;
#[cfg(test)]
pub mod recursion_smoke_test;
#[cfg(test)]
pub mod register_tests;
#[cfg(test)]
pub mod shift_tests;
#[cfg(test)]
pub mod statement_tests;
#[cfg(test)]
pub mod static_commitments_tests;
#[cfg(test)]
pub mod store_tests;
#[cfg(test)]
pub mod templates_tests;
#[cfg(test)]
pub mod trace_builder_tests;
#[cfg(test)]
pub mod trace_test_helpers;
