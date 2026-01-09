//! Differential fuzzing framework for Metal FFT verification.
//!
//! This module provides tools for comparing Metal GPU FFT results against
//! CPU implementations to ensure correctness across various input sizes
//! and edge cases.

use crate::fft::cpu::ops::fft as cpu_fft_impl;
use crate::fft::cpu::roots_of_unity::get_twiddles;
use crate::field::element::FieldElement;
use crate::field::fields::fft_friendly::u64_goldilocks_native::GoldilocksField;
use crate::field::traits::RootsConfig;
use crate::gpu::metal::{MetalError, MetalFFT};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::time::Instant;

/// Goldilocks prime modulus
const GOLDILOCKS_PRIME: u64 = 0xFFFFFFFF00000001;

/// Result of a differential fuzzing test
#[derive(Debug)]
pub struct FuzzResult {
    /// Size of the input tested
    pub size: usize,
    /// Whether the test passed (Metal == CPU)
    pub passed: bool,
    /// Number of mismatches found
    pub mismatches: usize,
    /// First mismatch details (index, cpu_value, metal_value)
    pub first_mismatch: Option<(usize, u64, u64)>,
    /// CPU execution time in microseconds (if timing enabled)
    #[cfg(feature = "std")]
    pub cpu_time_us: Option<u64>,
    /// Metal execution time in microseconds (if timing enabled)
    #[cfg(feature = "std")]
    pub metal_time_us: Option<u64>,
}

/// Configuration for differential fuzzing
#[derive(Debug, Clone)]
pub struct FuzzConfig {
    /// Minimum FFT size (log2)
    pub min_log_size: u32,
    /// Maximum FFT size (log2)
    pub max_log_size: u32,
    /// Number of random inputs to test per size
    pub iterations_per_size: usize,
    /// Whether to test edge cases (all zeros, all ones, max values)
    pub test_edge_cases: bool,
    /// Whether to enable timing measurements
    pub enable_timing: bool,
}

impl Default for FuzzConfig {
    fn default() -> Self {
        Self {
            min_log_size: 2,    // 4 elements
            max_log_size: 14,   // 16384 elements
            iterations_per_size: 10,
            test_edge_cases: true,
            enable_timing: true,
        }
    }
}

/// Differential fuzzer for comparing Metal and CPU FFT implementations
pub struct DifferentialFuzzer {
    metal_fft: MetalFFT,
    config: FuzzConfig,
    rng_state: u64,
}

impl DifferentialFuzzer {
    /// Create a new differential fuzzer
    pub fn new(config: FuzzConfig) -> Result<Self, MetalError> {
        let metal_fft = MetalFFT::new()?;

        Ok(Self {
            metal_fft,
            config,
            rng_state: 0x123456789ABCDEF0,  // Fixed seed for reproducibility
        })
    }

    /// Create with a specific random seed
    pub fn with_seed(config: FuzzConfig, seed: u64) -> Result<Self, MetalError> {
        let metal_fft = MetalFFT::new()?;

        Ok(Self {
            metal_fft,
            config,
            rng_state: seed,
        })
    }

    /// Simple xorshift64 PRNG for generating test values
    fn next_random(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x % GOLDILOCKS_PRIME
    }

    /// Generate random field elements
    fn generate_random_input(&mut self, size: usize) -> Vec<FieldElement<GoldilocksField>> {
        (0..size)
            .map(|_| FieldElement::from(self.next_random()))
            .collect()
    }

    /// Generate edge case inputs
    fn generate_edge_cases(&self, size: usize) -> Vec<Vec<FieldElement<GoldilocksField>>> {
        let mut cases = Vec::new();

        // All zeros
        cases.push(vec![FieldElement::zero(); size]);

        // All ones
        cases.push(vec![FieldElement::one(); size]);

        // Max values (p-1)
        let max_val = FieldElement::from(GOLDILOCKS_PRIME - 1);
        cases.push(vec![max_val; size]);

        // Alternating zeros and ones
        cases.push(
            (0..size)
                .map(|i| if i % 2 == 0 { FieldElement::zero() } else { FieldElement::one() })
                .collect()
        );

        // Powers of two (modulo p)
        cases.push(
            (0..size)
                .map(|i| {
                    let val = 1u64 << (i % 63);
                    FieldElement::from(val % GOLDILOCKS_PRIME)
                })
                .collect()
        );

        // Sequential values
        cases.push(
            (0..size)
                .map(|i| FieldElement::from(i as u64))
                .collect()
        );

        // Values near field boundaries
        cases.push(
            (0..size)
                .map(|i| {
                    let base = match i % 4 {
                        0 => 0u64,
                        1 => GOLDILOCKS_PRIME - 1,
                        2 => (1u64 << 32) - 1,  // EPSILON
                        _ => 1u64 << 32,
                    };
                    FieldElement::from(base)
                })
                .collect()
        );

        cases
    }

    /// Compute CPU FFT using existing implementation
    fn cpu_fft(&self, input: &[FieldElement<GoldilocksField>]) -> Option<Vec<FieldElement<GoldilocksField>>> {
        let log_n = input.len().trailing_zeros() as u64;
        let twiddles = get_twiddles::<GoldilocksField>(log_n, RootsConfig::BitReverse).ok()?;
        cpu_fft_impl::<GoldilocksField, GoldilocksField>(input, &twiddles).ok()
    }

    /// Compare two FFT results
    fn compare_results(
        cpu: &[FieldElement<GoldilocksField>],
        metal: &[FieldElement<GoldilocksField>],
    ) -> (bool, usize, Option<(usize, u64, u64)>) {
        if cpu.len() != metal.len() {
            return (false, usize::MAX, None);
        }

        let mut mismatches = 0;
        let mut first_mismatch = None;

        for (i, (c, m)) in cpu.iter().zip(metal.iter()).enumerate() {
            if c != m {
                mismatches += 1;
                if first_mismatch.is_none() {
                    // Get raw values for debugging
                    let cpu_val = *c.value();
                    let metal_val = *m.value();
                    first_mismatch = Some((i, cpu_val, metal_val));
                }
            }
        }

        (mismatches == 0, mismatches, first_mismatch)
    }

    /// Run a single fuzz test on given input
    #[cfg(feature = "std")]
    fn run_single_test(&mut self, input: &[FieldElement<GoldilocksField>]) -> FuzzResult {
        let size = input.len();

        // CPU FFT
        let cpu_start = if self.config.enable_timing {
            Some(Instant::now())
        } else {
            None
        };
        let cpu_result = match self.cpu_fft(input) {
            Some(r) => r,
            None => {
                return FuzzResult {
                    size,
                    passed: false,
                    mismatches: size,
                    first_mismatch: None,
                    cpu_time_us: None,
                    metal_time_us: None,
                };
            }
        };
        let cpu_time = cpu_start.map(|s| s.elapsed().as_micros() as u64);

        // Metal FFT
        let metal_start = if self.config.enable_timing {
            Some(Instant::now())
        } else {
            None
        };
        let metal_result = match self.metal_fft.fft(input) {
            Ok(r) => r,
            Err(_) => {
                return FuzzResult {
                    size,
                    passed: false,
                    mismatches: size,
                    first_mismatch: None,
                    cpu_time_us: cpu_time,
                    metal_time_us: None,
                };
            }
        };
        let metal_time = metal_start.map(|s| s.elapsed().as_micros() as u64);

        // Compare
        let (passed, mismatches, first_mismatch) = Self::compare_results(&cpu_result, &metal_result);

        FuzzResult {
            size,
            passed,
            mismatches,
            first_mismatch,
            cpu_time_us: cpu_time,
            metal_time_us: metal_time,
        }
    }

    /// Run a single fuzz test (no_std version)
    #[cfg(not(feature = "std"))]
    fn run_single_test(&mut self, input: &[FieldElement<GoldilocksField>]) -> FuzzResult {
        let size = input.len();

        // CPU FFT
        let cpu_result = match self.cpu_fft(input) {
            Some(r) => r,
            None => {
                return FuzzResult {
                    size,
                    passed: false,
                    mismatches: size,
                    first_mismatch: None,
                };
            }
        };

        // Metal FFT
        let metal_result = match self.metal_fft.fft(input) {
            Ok(r) => r,
            Err(_) => {
                return FuzzResult {
                    size,
                    passed: false,
                    mismatches: size,
                    first_mismatch: None,
                };
            }
        };

        // Compare
        let (passed, mismatches, first_mismatch) = Self::compare_results(&cpu_result, &metal_result);

        FuzzResult {
            size,
            passed,
            mismatches,
            first_mismatch,
        }
    }

    /// Run fuzzing for a specific size
    pub fn fuzz_size(&mut self, log_size: u32) -> Vec<FuzzResult> {
        let size = 1 << log_size;
        let mut results = Vec::new();

        // Test edge cases first
        if self.config.test_edge_cases {
            for edge_case in self.generate_edge_cases(size) {
                results.push(self.run_single_test(&edge_case));
            }
        }

        // Test random inputs
        for _ in 0..self.config.iterations_per_size {
            let input = self.generate_random_input(size);
            results.push(self.run_single_test(&input));
        }

        results
    }

    /// Run the full fuzzing suite
    pub fn run_full_suite(&mut self) -> FuzzReport {
        let mut report = FuzzReport {
            total_tests: 0,
            passed_tests: 0,
            failed_tests: 0,
            results_by_size: Vec::new(),
        };

        for log_size in self.config.min_log_size..=self.config.max_log_size {
            let results = self.fuzz_size(log_size);

            let size = 1usize << log_size;
            let passed = results.iter().filter(|r| r.passed).count();
            let failed = results.len() - passed;

            report.total_tests += results.len();
            report.passed_tests += passed;
            report.failed_tests += failed;

            report.results_by_size.push(SizeReport {
                log_size,
                size,
                total: results.len(),
                passed,
                failed,
                results,
            });
        }

        report
    }
}

/// Report for a specific FFT size
#[derive(Debug)]
pub struct SizeReport {
    /// Log2 of the FFT size
    pub log_size: u32,
    /// Actual FFT size
    pub size: usize,
    /// Total tests run
    pub total: usize,
    /// Tests passed
    pub passed: usize,
    /// Tests failed
    pub failed: usize,
    /// Individual test results
    pub results: Vec<FuzzResult>,
}

/// Full fuzzing report
#[derive(Debug)]
pub struct FuzzReport {
    /// Total number of tests run
    pub total_tests: usize,
    /// Number of tests that passed
    pub passed_tests: usize,
    /// Number of tests that failed
    pub failed_tests: usize,
    /// Results organized by FFT size
    pub results_by_size: Vec<SizeReport>,
}

impl FuzzReport {
    /// Check if all tests passed
    pub fn all_passed(&self) -> bool {
        self.failed_tests == 0
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "Differential Fuzzing Report\n\
             ==========================\n\
             Total tests: {}\n\
             Passed: {} ({:.1}%)\n\
             Failed: {} ({:.1}%)\n",
            self.total_tests,
            self.passed_tests,
            100.0 * self.passed_tests as f64 / self.total_tests as f64,
            self.failed_tests,
            100.0 * self.failed_tests as f64 / self.total_tests as f64,
        )
    }

    /// Get timing summary (if timing was enabled)
    #[cfg(feature = "std")]
    pub fn timing_summary(&self) -> String {
        let mut summary = String::from("\nTiming Summary (microseconds)\n");
        summary.push_str("Size\t\tCPU avg\t\tMetal avg\tSpeedup\n");
        summary.push_str("----\t\t-------\t\t---------\t-------\n");

        for size_report in &self.results_by_size {
            let cpu_times: Vec<u64> = size_report.results
                .iter()
                .filter_map(|r| r.cpu_time_us)
                .collect();
            let metal_times: Vec<u64> = size_report.results
                .iter()
                .filter_map(|r| r.metal_time_us)
                .collect();

            if !cpu_times.is_empty() && !metal_times.is_empty() {
                let cpu_avg = cpu_times.iter().sum::<u64>() / cpu_times.len() as u64;
                let metal_avg = metal_times.iter().sum::<u64>() / metal_times.len() as u64;
                let speedup = if metal_avg > 0 {
                    cpu_avg as f64 / metal_avg as f64
                } else {
                    0.0
                };

                summary.push_str(&format!(
                    "2^{}\t\t{}\t\t{}\t\t{:.2}x\n",
                    size_report.log_size,
                    cpu_avg,
                    metal_avg,
                    speedup,
                ));
            }
        }

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_differential_fuzzer_small() {
        let config = FuzzConfig {
            min_log_size: 2,
            max_log_size: 8,
            iterations_per_size: 5,
            test_edge_cases: true,
            enable_timing: true,
        };

        if let Ok(mut fuzzer) = DifferentialFuzzer::new(config) {
            let report = fuzzer.run_full_suite();

            println!("{}", report.summary());
            #[cfg(feature = "std")]
            println!("{}", report.timing_summary());

            assert!(
                report.all_passed(),
                "Differential fuzzing found {} mismatches",
                report.failed_tests
            );
        }
    }

    #[test]
    fn test_differential_fuzzer_medium() {
        let config = FuzzConfig {
            min_log_size: 8,
            max_log_size: 12,
            iterations_per_size: 3,
            test_edge_cases: true,
            enable_timing: true,
        };

        if let Ok(mut fuzzer) = DifferentialFuzzer::new(config) {
            let report = fuzzer.run_full_suite();

            println!("{}", report.summary());
            #[cfg(feature = "std")]
            println!("{}", report.timing_summary());

            assert!(
                report.all_passed(),
                "Differential fuzzing found {} mismatches",
                report.failed_tests
            );
        }
    }

    #[test]
    fn test_edge_cases_only() {
        let config = FuzzConfig {
            min_log_size: 4,
            max_log_size: 10,
            iterations_per_size: 0,  // Only edge cases
            test_edge_cases: true,
            enable_timing: false,
        };

        if let Ok(mut fuzzer) = DifferentialFuzzer::new(config) {
            let report = fuzzer.run_full_suite();

            println!("Edge cases report:");
            println!("{}", report.summary());

            assert!(
                report.all_passed(),
                "Edge case tests found {} mismatches",
                report.failed_tests
            );
        }
    }

    #[test]
    fn test_reproducibility() {
        // Same seed should produce same results
        let config = FuzzConfig {
            min_log_size: 4,
            max_log_size: 6,
            iterations_per_size: 5,
            test_edge_cases: false,
            enable_timing: false,
        };

        let seed = 42u64;

        if let Ok(mut fuzzer1) = DifferentialFuzzer::with_seed(config.clone(), seed) {
            if let Ok(mut fuzzer2) = DifferentialFuzzer::with_seed(config, seed) {
                // Both fuzzers should generate the same inputs
                let input1 = fuzzer1.generate_random_input(16);
                let input2 = fuzzer2.generate_random_input(16);

                assert_eq!(input1, input2, "Random inputs should be identical with same seed");
            }
        }
    }

    #[test]
    #[ignore] // Run with: cargo test --features metal test_benchmark_large_fft -- --ignored --nocapture
    fn test_benchmark_large_fft() {
        let config = FuzzConfig {
            min_log_size: 18,
            max_log_size: 22,
            iterations_per_size: 3,
            test_edge_cases: false,
            enable_timing: true,
        };

        println!("\nBenchmarking large FFT sizes (2^18 to 2^22)...\n");

        if let Ok(mut fuzzer) = DifferentialFuzzer::new(config) {
            let report = fuzzer.run_full_suite();

            println!("{}", report.summary());
            #[cfg(feature = "std")]
            println!("{}", report.timing_summary());

            assert!(
                report.all_passed(),
                "Large FFT benchmark found {} mismatches",
                report.failed_tests
            );
        } else {
            println!("Metal not available, skipping benchmark");
        }
    }
}
