pub mod cpu_table;
pub mod integration_tests;

// 64-bit VM tests
#[cfg(test)]
pub mod bitwise_bus_tests;
#[cfg(test)]
pub mod bitwise_tests;
#[cfg(test)]
pub mod constraints64_tests;
#[cfg(test)]
pub mod cpu_tests;
#[cfg(test)]
pub mod lt_bus_tests;
#[cfg(test)]
pub mod lt_tests;
#[cfg(test)]
pub mod prove_elfs_tests;
#[cfg(test)]
pub mod trace_builder_tests;
