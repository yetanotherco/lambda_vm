use math::field::{
    element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};

#[cfg(feature = "debug-checks")]
pub mod bus_debug;
pub mod constraints;
pub mod context;
pub mod debug;
pub mod domain;
pub mod examples;
pub mod frame;
pub mod fri;
pub mod grinding;
pub mod lookup;
pub mod proof;
pub mod prover;
pub mod table;
pub mod trace;
pub mod traits;
pub mod transcript;
pub mod utils;
pub mod verifier;

#[cfg(test)]
pub mod tests;

/// Configurations of the Prover available in compile time
pub mod config;

pub type PrimeField = Stark252PrimeField;
pub type Felt252 = FieldElement<PrimeField>;

/// Prints current jemalloc `stats.allocated` to stderr.
/// No-op when compiled without `jemalloc-stats`.
#[cfg(feature = "jemalloc-stats")]
pub(crate) fn heap_snapshot(label: &str) {
    use tikv_jemalloc_ctl::{epoch, stats};
    epoch::advance().ok();
    if let Ok(allocated) = stats::allocated::read() {
        eprintln!("[HEAP] {}: {} MB", label, allocated / (1024 * 1024));
    }
}

#[cfg(not(feature = "jemalloc-stats"))]
pub(crate) fn heap_snapshot(_label: &str) {}
