//! Differential fuzzing module for ARM64 assembly Goldilocks field operations.
//!
//! This module compares the assembly implementations (add_fast, sub_fast)
//! against native Rust implementations to ensure correctness across a wide
//! range of inputs including random values and edge cases.
//!
//! Note: Only add/sub use ASM (faster). mul/square use native Rust (LLVM optimizes better).

use super::u64_goldilocks_native::GOLDILOCKS_PRIME;

#[cfg(all(feature = "asm-arm64", target_arch = "aarch64"))]
use super::u64_goldilocks_asm;

/// EPSILON = 2^32 - 1 = p - 2^64 (i.e., -2^64 mod p)
const EPSILON: u64 = 0xFFFF_FFFF;

/// Configuration for the differential fuzzer.
#[derive(Debug, Clone)]
pub struct FuzzConfig {
    /// Number of random test cases to generate.
    pub num_random_tests: usize,
    /// Whether to test edge cases.
    pub edge_case_tests: bool,
    /// Whether to print detailed output.
    pub verbose: bool,
}

impl Default for FuzzConfig {
    fn default() -> Self {
        Self {
            num_random_tests: 10_000,
            edge_case_tests: true,
            verbose: false,
        }
    }
}

/// Simple xorshift64 PRNG for reproducible random testing.
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    pub fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a random value in [0, GOLDILOCKS_PRIME)
    pub fn next_field_element(&mut self) -> u64 {
        self.next() % GOLDILOCKS_PRIME
    }
}

/// Results of a fuzzing run.
#[derive(Debug, Default)]
pub struct FuzzReport {
    pub mul_tests: usize,
    pub mul_failures: usize,
    pub add_tests: usize,
    pub add_failures: usize,
    pub sub_tests: usize,
    pub sub_failures: usize,
    pub reduce128_tests: usize,
    pub reduce128_failures: usize,
}

impl FuzzReport {
    pub fn total_tests(&self) -> usize {
        self.mul_tests + self.add_tests + self.sub_tests + self.reduce128_tests
    }

    pub fn total_failures(&self) -> usize {
        self.mul_failures + self.add_failures + self.sub_failures + self.reduce128_failures
    }

    pub fn passed(&self) -> bool {
        self.total_failures() == 0
    }
}

/// Differential fuzzer for Goldilocks ASM operations.
pub struct GoldilocksAsmFuzzer {
    config: FuzzConfig,
    prng: Xorshift64,
}

impl GoldilocksAsmFuzzer {
    pub fn new(config: FuzzConfig, seed: u64) -> Self {
        Self {
            config,
            prng: Xorshift64::new(seed),
        }
    }

    pub fn with_default_config(seed: u64) -> Self {
        Self::new(FuzzConfig::default(), seed)
    }

    /// Canonicalize a value to [0, p).
    fn canonicalize(x: u64) -> u64 {
        if x >= GOLDILOCKS_PRIME {
            x - GOLDILOCKS_PRIME
        } else {
            x
        }
    }

    // Reference implementations (native Rust)
    fn native_mul(a: u64, b: u64) -> u64 {
        let product = (a as u128) * (b as u128);
        Self::native_reduce128(product)
    }

    fn native_reduce128(x: u128) -> u64 {
        let x_lo = x as u64;
        let x_hi = (x >> 64) as u64;
        let x_hi_hi = x_hi >> 32;
        let x_hi_lo = x_hi & EPSILON;

        let (t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
        let t0 = if borrow { t0.wrapping_sub(EPSILON) } else { t0 };

        let t1 = x_hi_lo.wrapping_mul(EPSILON);

        let (result, carry) = t0.overflowing_add(t1);
        if carry {
            result.wrapping_add(EPSILON)
        } else {
            result
        }
    }

    fn native_add(a: u64, b: u64) -> u64 {
        let (sum, over) = a.overflowing_add(b);
        let (sum, over2) = sum.overflowing_add((over as u64) * EPSILON);
        if over2 {
            sum.wrapping_add(EPSILON)
        } else {
            sum
        }
    }

    fn native_sub(a: u64, b: u64) -> u64 {
        let (diff, under) = a.overflowing_sub(b);
        let (diff, under2) = diff.overflowing_sub((under as u64) * EPSILON);
        if under2 {
            diff.wrapping_sub(EPSILON)
        } else {
            diff
        }
    }

    /// Generate edge case test pairs (all values reduced to [0, p)).
    ///
    /// Note: The fast add/sub operations only handle single overflow,
    /// which is correct for reduced inputs. These edge cases are designed
    /// to stay within the valid range.
    pub fn edge_cases() -> Vec<(u64, u64)> {
        vec![
            // Zero and identity
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            // Prime boundary (reduced values)
            (GOLDILOCKS_PRIME - 1, 1),
            (1, GOLDILOCKS_PRIME - 1),
            (GOLDILOCKS_PRIME - 1, GOLDILOCKS_PRIME - 1),
            (GOLDILOCKS_PRIME - 1, 2),
            (2, GOLDILOCKS_PRIME - 1),
            // EPSILON cases (all < p)
            (EPSILON, EPSILON),
            (EPSILON, 1),
            (1, EPSILON),
            (EPSILON + 1, EPSILON),
            (EPSILON, EPSILON + 1),
            // Powers of 2 (< p)
            (1u64 << 32, 1u64 << 32),
            (1u64 << 31, 1u64 << 31),
            // Reduced versions of large values
            ((1u64 << 63) % GOLDILOCKS_PRIME, 2),
            (2, (1u64 << 63) % GOLDILOCKS_PRIME),
            // Various reduced values
            (0xDEADBEEF % GOLDILOCKS_PRIME, 0x12345678 % GOLDILOCKS_PRIME),
            // Values that produce specific hi patterns in mul
            ((1u64 << 40), (1u64 << 40)),
            (
                (1u64 << 48) % GOLDILOCKS_PRIME,
                (1u64 << 48) % GOLDILOCKS_PRIME,
            ),
        ]
    }

    /// Run differential tests comparing ASM vs native implementations.
    #[cfg(all(feature = "asm-arm64", target_arch = "aarch64"))]
    pub fn run_differential_tests(&mut self) -> FuzzReport {
        let mut report = FuzzReport::default();

        // Test edge cases
        if self.config.edge_case_tests {
            for (a, b) in Self::edge_cases() {
                self.test_mul(a, b, &mut report);
                self.test_add(a, b, &mut report);
                self.test_sub(a, b, &mut report);
            }
        }

        // Test random cases
        for _ in 0..self.config.num_random_tests {
            let a = self.prng.next_field_element();
            let b = self.prng.next_field_element();

            self.test_mul(a, b, &mut report);
            self.test_add(a, b, &mut report);
            self.test_sub(a, b, &mut report);
        }

        // Test reduce128 with various 128-bit values
        self.test_reduce128_cases(&mut report);

        report
    }

    #[cfg(all(feature = "asm-arm64", target_arch = "aarch64"))]
    fn test_mul(&mut self, a: u64, b: u64, report: &mut FuzzReport) {
        report.mul_tests += 1;
        let asm_result = u64_goldilocks_asm::mul(a, b);
        let native_result = Self::native_mul(a, b);

        if Self::canonicalize(asm_result) != Self::canonicalize(native_result) {
            report.mul_failures += 1;
            if self.config.verbose {
                eprintln!(
                    "MUL MISMATCH: a={}, b={}, asm={}, native={}",
                    a, b, asm_result, native_result
                );
            }
        }
    }

    #[cfg(all(feature = "asm-arm64", target_arch = "aarch64"))]
    fn test_add(&mut self, a: u64, b: u64, report: &mut FuzzReport) {
        report.add_tests += 1;
        let asm_result = u64_goldilocks_asm::add_fast(a, b);
        let native_result = Self::native_add(a, b);

        if Self::canonicalize(asm_result) != Self::canonicalize(native_result) {
            report.add_failures += 1;
            if self.config.verbose {
                eprintln!(
                    "ADD MISMATCH: a={}, b={}, asm={}, native={}",
                    a, b, asm_result, native_result
                );
            }
        }
    }

    #[cfg(all(feature = "asm-arm64", target_arch = "aarch64"))]
    fn test_sub(&mut self, a: u64, b: u64, report: &mut FuzzReport) {
        report.sub_tests += 1;
        let asm_result = u64_goldilocks_asm::sub_fast(a, b);
        let native_result = Self::native_sub(a, b);

        if Self::canonicalize(asm_result) != Self::canonicalize(native_result) {
            report.sub_failures += 1;
            if self.config.verbose {
                eprintln!(
                    "SUB MISMATCH: a={}, b={}, asm={}, native={}",
                    a, b, asm_result, native_result
                );
            }
        }
    }

    #[cfg(all(feature = "asm-arm64", target_arch = "aarch64"))]
    fn test_reduce128_cases(&mut self, report: &mut FuzzReport) {
        // Test specific 128-bit patterns
        let test_cases: Vec<(u64, u64)> = vec![
            (0, 0),
            (35, 0),
            (u64::MAX, 0),
            (0, 1),
            (u64::MAX, u64::MAX),
            (GOLDILOCKS_PRIME, 0),
            (0, EPSILON),
            (EPSILON, EPSILON),
            // Random 128-bit values from multiplications
            (0xDEADBEEF_CAFEBABE, 0x12345678),
            (0xFFFFFFFF_FFFFFFFF, 0xFFFFFFFF),
        ];

        for (lo, hi) in test_cases {
            report.reduce128_tests += 1;
            let x = (lo as u128) | ((hi as u128) << 64);
            let asm_result = u64_goldilocks_asm::reduce128(x);
            let native_result = Self::native_reduce128(x);

            if Self::canonicalize(asm_result) != Self::canonicalize(native_result) {
                report.reduce128_failures += 1;
                if self.config.verbose {
                    eprintln!(
                        "REDUCE128 MISMATCH: lo={}, hi={}, asm={}, native={}",
                        lo, hi, asm_result, native_result
                    );
                }
            }
        }

        // Also test with random 128-bit values from actual multiplications
        for _ in 0..1000 {
            let a = self.prng.next();
            let b = self.prng.next();
            let (lo, hi) = u64_goldilocks_asm::mul_wide(a, b);
            let x = (lo as u128) | ((hi as u128) << 64);

            report.reduce128_tests += 1;
            let asm_result = u64_goldilocks_asm::reduce128(x);
            let native_result = Self::native_reduce128(x);

            if Self::canonicalize(asm_result) != Self::canonicalize(native_result) {
                report.reduce128_failures += 1;
                if self.config.verbose {
                    eprintln!(
                        "REDUCE128 MISMATCH (from mul_wide): a={}, b={}, lo={}, hi={}, asm={}, native={}",
                        a, b, lo, hi, asm_result, native_result
                    );
                }
            }
        }
    }

    /// Placeholder for non-ARM64 platforms
    #[cfg(not(all(feature = "asm-arm64", target_arch = "aarch64")))]
    pub fn run_differential_tests(&mut self) -> FuzzReport {
        FuzzReport::default()
    }
}

#[cfg(test)]
#[cfg(all(feature = "asm-arm64", target_arch = "aarch64"))]
mod tests {
    use super::*;

    #[test]
    fn test_differential_fuzzer_small() {
        let config = FuzzConfig {
            num_random_tests: 1_000,
            edge_case_tests: true,
            verbose: true,
        };
        let mut fuzzer = GoldilocksAsmFuzzer::new(config, 12345);
        let report = fuzzer.run_differential_tests();

        println!("Differential Fuzzer Report:");
        println!(
            "  Multiplication: {}/{} passed",
            report.mul_tests - report.mul_failures,
            report.mul_tests
        );
        println!(
            "  Addition: {}/{} passed",
            report.add_tests - report.add_failures,
            report.add_tests
        );
        println!(
            "  Subtraction: {}/{} passed",
            report.sub_tests - report.sub_failures,
            report.sub_tests
        );
        println!(
            "  Reduce128: {}/{} passed",
            report.reduce128_tests - report.reduce128_failures,
            report.reduce128_tests
        );
        println!(
            "  Total: {}/{} passed",
            report.total_tests() - report.total_failures(),
            report.total_tests()
        );

        assert!(
            report.passed(),
            "Differential fuzzing found {} failures",
            report.total_failures()
        );
    }

    #[test]
    fn test_differential_fuzzer_medium() {
        let config = FuzzConfig {
            num_random_tests: 100_000,
            edge_case_tests: true,
            verbose: false,
        };
        let mut fuzzer = GoldilocksAsmFuzzer::new(config, 67890);
        let report = fuzzer.run_differential_tests();

        assert!(
            report.passed(),
            "Differential fuzzing found {} failures out of {} tests",
            report.total_failures(),
            report.total_tests()
        );
    }

    #[test]
    fn test_reproducibility() {
        // Run the same seed twice and verify identical results
        let config = FuzzConfig {
            num_random_tests: 1_000,
            edge_case_tests: true,
            verbose: false,
        };

        let mut fuzzer1 = GoldilocksAsmFuzzer::new(config.clone(), 42);
        let report1 = fuzzer1.run_differential_tests();

        let mut fuzzer2 = GoldilocksAsmFuzzer::new(config, 42);
        let report2 = fuzzer2.run_differential_tests();

        assert_eq!(report1.mul_tests, report2.mul_tests);
        assert_eq!(report1.mul_failures, report2.mul_failures);
        assert_eq!(report1.add_tests, report2.add_tests);
        assert_eq!(report1.add_failures, report2.add_failures);
    }

    #[test]
    fn test_edge_cases_only() {
        let config = FuzzConfig {
            num_random_tests: 0,
            edge_case_tests: true,
            verbose: true,
        };
        let mut fuzzer = GoldilocksAsmFuzzer::new(config, 0);
        let report = fuzzer.run_differential_tests();

        assert!(
            report.passed(),
            "Edge case testing found {} failures",
            report.total_failures()
        );
    }
}
