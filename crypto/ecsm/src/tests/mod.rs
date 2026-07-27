//! Test suite and test-only reference arithmetic for the `ecsm` crate.
//!
//! `reference_field` (BigUint `F_p`) and `reference` (affine double-and-add) are
//! the spec-faithful reference implementation used to cross-check the production
//! k256-backed fast path. The `*_tests` modules are the relocated unit tests.
//!
//! This whole tree is gated behind `#[cfg(test)] mod tests;` in `lib.rs`, so the
//! reference code never ships in non-test builds.

pub mod reference;
pub mod reference_field;

mod curve_tests;
mod lib_tests;
mod lincomb2_table_tests;
mod lincomb2_tests;
mod witness_tests;
