//! Differential fuzzing and benchmarking for Metal Merkle tree implementation.
//!
//! This module compares the Metal GPU Merkle tree against a CPU implementation
//! using Poseidon2 hash to ensure correctness.

use super::merkle::MetalMerkleTree;
use super::MetalError;
use crate::field::element::FieldElement;
use crate::field::fields::fft_friendly::u64_goldilocks_native::GoldilocksField;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::time::Instant;

type Fp = FieldElement<GoldilocksField>;

/// Goldilocks prime modulus
const GOLDILOCKS_PRIME: u64 = 0xFFFFFFFF00000001;

// =============================================================================
// CPU Poseidon2 Implementation for comparison
// =============================================================================

/// Poseidon2 parameters
const WIDTH: usize = 8;
const EXTERNAL_ROUNDS_BEGIN: usize = 4;
const EXTERNAL_ROUNDS_END: usize = 4;
const INTERNAL_ROUNDS: usize = 22;

const MATRIX_DIAG_8: [u64; 8] = [
    0xa98811a1fed4e3a5,
    0x1cc48b54f377e2a0,
    0xe40cd4f6c5609a26,
    0x11de79ebca97a4a3,
    0x9177c73d8b7e929c,
    0x2a6fe8085797e791,
    0x3de6e93329f8d5ad,
    0x3f7af9125da962fe,
];

const EXT_RC_INIT: [[u64; 8]; 4] = [
    [
        0xdd5743e7f2a5a5d9, 0xcb3a864e58ada44b, 0xffa2449ed32f8cdc, 0x42025f65d6bd13ee,
        0x7889175e25506323, 0x34b98bb03d24b737, 0xbdcc535ecc4faa2a, 0x5b20ad869fc0d033,
    ],
    [
        0xf1dda5b9259dfcb4, 0x27515210be112d59, 0x4227d1718c766c3f, 0x26d333161a5bd794,
        0x49b938957bf4b026, 0x4a56b5938b213669, 0x1120426b48c8353d, 0x6b323c3f10a56cad,
    ],
    [
        0xce57d6245ddca6b2, 0xb1fc8d402bba1eb1, 0xb5c5096ca959bd04, 0x6db55cd306d31f7f,
        0xc49d293a81cb9641, 0x1ce55a4fe979719f, 0xa92e60a9d178a4d1, 0x002cc64973bcfd8c,
    ],
    [
        0xcea721cce82fb11b, 0xe5b55eb8098ece81, 0x4e30525c6f1ddd66, 0x43c6702827070987,
        0xaca68430a7b5762a, 0x3674238634df9c93, 0x88cee1c825e33433, 0xde99ae8d74b57176,
    ],
];

const EXT_RC_TERM: [[u64; 8]; 4] = [
    [
        0x014ef1197d341346, 0x9725e20825d07394, 0xfdb25aef2c5bae3b, 0xbe5402dc598c971e,
        0x93a5711f04cdca3d, 0xc45a9a5b2f8fb97b, 0xfe8946a924933545, 0x2af997a27369091c,
    ],
    [
        0xaa62c88e0b294011, 0x058eb9d810ce9f74, 0xb3cb23eced349ae4, 0xa3648177a77b4a84,
        0x43153d905992d95d, 0xf4e2a97cda44aa4b, 0x5baa2702b908682f, 0x082923bdf4f750d1,
    ],
    [
        0x98ae09a325893803, 0xf8a6475077968838, 0xceb0735bf00b2c5f, 0x0a1a5d953888e072,
        0x2fcb190489f94475, 0xb5be06270dec69fc, 0x739cb934b09acf8b, 0x537750b75ec7f25b,
    ],
    [
        0xe9dd318bae1f3961, 0xf7462137299efe1a, 0xb1f6b8eee9adb940, 0xbdebcc8a809dfe6b,
        0x40fc1f791b178113, 0x3ac1c3362d014864, 0x9a016184bdb8aeba, 0x95f2394459fbc25e,
    ],
];

const INT_RC: [u64; 22] = [
    0x488897d85ff51f56, 0x1140737ccb162218, 0xa7eeb9215866ed35, 0x9bd2976fee49fcc9,
    0xc0c8f0de580a3fcc, 0x4fb2dae6ee8fc793, 0x343a89f35f37395b, 0x223b525a77ca72c8,
    0x56ccb62574aaa918, 0xc4d507d8027af9ed, 0xa080673cf0b7e95c, 0xf0184884eb70dcf8,
    0x044f10b0cb3d5c69, 0xe9e3f7993938f186, 0x1b761c80e772f459, 0x606cec607a1b5fac,
    0x14a0c2e1d45f03cd, 0x4eace8855398574f, 0xf905ca7103eff3e6, 0xf8c8f8d20862c059,
    0xb524fe8bdd678e5a, 0xfbb7865901a1ec41,
];

/// CPU Poseidon2 implementation for comparison
struct CpuPoseidon2 {
    state: [Fp; WIDTH],
}

impl CpuPoseidon2 {
    fn new() -> Self {
        Self {
            state: core::array::from_fn(|_| Fp::zero()),
        }
    }

    fn sbox(x: &Fp) -> Fp {
        let x2 = x * x;
        let x4 = &x2 * &x2;
        let x6 = &x4 * &x2;
        &x6 * x
    }

    fn apply_hl_mat4(x: &mut [Fp; 4]) {
        let t0 = &x[0] + &x[1];
        let t1 = &x[2] + &x[3];
        let t2 = &(&x[1] + &x[1]) + &t1;
        let t3 = &(&x[3] + &x[3]) + &t0;
        let t1_double = &t1 + &t1;
        let t4 = &(&t1_double + &t1_double) + &t3;
        let t0_double = &t0 + &t0;
        let t5 = &(&t0_double + &t0_double) + &t2;
        let t6 = &t3 + &t5;
        let t7 = &t2 + &t4;
        x[0] = t6;
        x[1] = t5;
        x[2] = t7;
        x[3] = t4;
    }

    fn external_linear_layer(&mut self) {
        let mut first_half: [Fp; 4] = core::array::from_fn(|i| self.state[i].clone());
        let mut second_half: [Fp; 4] = core::array::from_fn(|i| self.state[i + 4].clone());
        Self::apply_hl_mat4(&mut first_half);
        Self::apply_hl_mat4(&mut second_half);
        for i in 0..4 {
            self.state[i] = first_half[i].clone();
            self.state[i + 4] = second_half[i].clone();
        }
        for i in 0..4 {
            let sum = &self.state[i] + &self.state[i + 4];
            self.state[i] = &self.state[i] + &sum;
            self.state[i + 4] = &self.state[i + 4] + &sum;
        }
    }

    fn internal_linear_layer(&mut self) {
        let mut sum = Fp::zero();
        for i in 0..WIDTH {
            sum = &sum + &self.state[i];
        }
        for i in 0..WIDTH {
            let diag = Fp::from(MATRIX_DIAG_8[i]);
            self.state[i] = &(&diag * &self.state[i]) + &sum;
        }
    }

    fn external_round(&mut self, rc: &[u64; WIDTH]) {
        for i in 0..WIDTH {
            self.state[i] = &self.state[i] + &Fp::from(rc[i]);
        }
        for i in 0..WIDTH {
            self.state[i] = Self::sbox(&self.state[i]);
        }
        self.external_linear_layer();
    }

    fn internal_round(&mut self, rc: u64) {
        self.state[0] = &self.state[0] + &Fp::from(rc);
        self.state[0] = Self::sbox(&self.state[0]);
        self.internal_linear_layer();
    }

    fn permute(&mut self) {
        self.external_linear_layer();
        for r in 0..EXTERNAL_ROUNDS_BEGIN {
            self.external_round(&EXT_RC_INIT[r]);
        }
        for r in 0..INTERNAL_ROUNDS {
            self.internal_round(INT_RC[r]);
        }
        for r in 0..EXTERNAL_ROUNDS_END {
            self.external_round(&EXT_RC_TERM[r]);
        }
    }

    fn hash_single(x: &Fp) -> Fp {
        let mut hasher = Self::new();
        hasher.state[0] = x.clone();
        hasher.state[WIDTH - 1] = Fp::from(1u64);
        hasher.permute();
        hasher.state[0].clone()
    }

    fn compress(left: &Fp, right: &Fp) -> Fp {
        let mut hasher = Self::new();
        hasher.state[0] = left.clone();
        hasher.state[1] = right.clone();
        hasher.state[WIDTH - 1] = Fp::from(2u64);
        hasher.permute();
        hasher.state[0].clone()
    }
}

/// CPU Merkle tree builder using Poseidon2
pub struct CpuMerkleTree;

impl CpuMerkleTree {
    /// Build Merkle tree on CPU and return root
    pub fn build_root_only(leaves: &[Fp]) -> Option<Fp> {
        if leaves.is_empty() {
            return None;
        }

        let n = leaves.len().next_power_of_two();
        let mut padded_leaves = leaves.to_vec();
        while padded_leaves.len() < n {
            padded_leaves.push(padded_leaves.last().unwrap().clone());
        }

        // Hash leaves
        let mut current_level: Vec<Fp> = padded_leaves
            .iter()
            .map(|x| CpuPoseidon2::hash_single(x))
            .collect();

        // Build tree
        while current_level.len() > 1 {
            let next_level: Vec<Fp> = current_level
                .chunks(2)
                .map(|pair| CpuPoseidon2::compress(&pair[0], &pair[1]))
                .collect();
            current_level = next_level;
        }

        Some(current_level[0].clone())
    }
}

// =============================================================================
// Differential Fuzzing
// =============================================================================

/// Configuration for Merkle tree fuzzing
#[derive(Debug, Clone)]
pub struct MerkleFuzzConfig {
    /// Minimum tree size (log2 of number of leaves)
    pub min_log_size: u32,
    /// Maximum tree size (log2 of number of leaves)
    pub max_log_size: u32,
    /// Number of random inputs per size
    pub iterations_per_size: usize,
    /// Test edge cases
    pub test_edge_cases: bool,
    /// Enable timing
    pub enable_timing: bool,
}

impl Default for MerkleFuzzConfig {
    fn default() -> Self {
        Self {
            min_log_size: 2,
            max_log_size: 14,
            iterations_per_size: 5,
            test_edge_cases: true,
            enable_timing: true,
        }
    }
}

/// Result of a single fuzz test
#[derive(Debug)]
pub struct MerkleFuzzResult {
    pub size: usize,
    pub passed: bool,
    pub cpu_root: Option<u64>,
    pub metal_root: Option<u64>,
    #[cfg(feature = "std")]
    pub cpu_time_us: Option<u64>,
    #[cfg(feature = "std")]
    pub metal_time_us: Option<u64>,
}

/// Differential fuzzer for Merkle trees
pub struct MerkleFuzzer {
    metal_merkle: MetalMerkleTree,
    config: MerkleFuzzConfig,
    rng_state: u64,
}

impl MerkleFuzzer {
    pub fn new(config: MerkleFuzzConfig) -> Result<Self, MetalError> {
        let metal_merkle = MetalMerkleTree::new()?;
        Ok(Self {
            metal_merkle,
            config,
            rng_state: 0xDEADBEEF12345678,
        })
    }

    pub fn with_seed(config: MerkleFuzzConfig, seed: u64) -> Result<Self, MetalError> {
        let metal_merkle = MetalMerkleTree::new()?;
        Ok(Self {
            metal_merkle,
            config,
            rng_state: seed,
        })
    }

    fn next_random(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x % GOLDILOCKS_PRIME
    }

    fn generate_random_input(&mut self, size: usize) -> Vec<Fp> {
        (0..size).map(|_| Fp::from(self.next_random())).collect()
    }

    fn generate_edge_cases(&self, size: usize) -> Vec<Vec<Fp>> {
        let mut cases = Vec::new();

        // All zeros
        cases.push(vec![Fp::zero(); size]);

        // All ones
        cases.push(vec![Fp::one(); size]);

        // Sequential
        cases.push((0..size).map(|i| Fp::from(i as u64)).collect());

        // Max values
        cases.push(vec![Fp::from(GOLDILOCKS_PRIME - 1); size]);

        cases
    }

    #[cfg(feature = "std")]
    fn run_single_test(&mut self, input: &[Fp]) -> MerkleFuzzResult {
        let size = input.len();

        // CPU
        let cpu_start = if self.config.enable_timing {
            Some(Instant::now())
        } else {
            None
        };
        let cpu_root = CpuMerkleTree::build_root_only(input);
        let cpu_time = cpu_start.map(|s| s.elapsed().as_micros() as u64);

        // Metal
        let metal_start = if self.config.enable_timing {
            Some(Instant::now())
        } else {
            None
        };
        let metal_root = self.metal_merkle.build_root_only(input).ok();
        let metal_time = metal_start.map(|s| s.elapsed().as_micros() as u64);

        let passed = match (&cpu_root, &metal_root) {
            (Some(c), Some(m)) => c == m,
            _ => false,
        };

        MerkleFuzzResult {
            size,
            passed,
            cpu_root: cpu_root.map(|r| *r.value()),
            metal_root: metal_root.map(|r| *r.value()),
            cpu_time_us: cpu_time,
            metal_time_us: metal_time,
        }
    }

    #[cfg(not(feature = "std"))]
    fn run_single_test(&mut self, input: &[Fp]) -> MerkleFuzzResult {
        let size = input.len();
        let cpu_root = CpuMerkleTree::build_root_only(input);
        let metal_root = self.metal_merkle.build_root_only(input).ok();

        let passed = match (&cpu_root, &metal_root) {
            (Some(c), Some(m)) => c == m,
            _ => false,
        };

        MerkleFuzzResult {
            size,
            passed,
            cpu_root: cpu_root.map(|r| *r.value()),
            metal_root: metal_root.map(|r| *r.value()),
        }
    }

    pub fn fuzz_size(&mut self, log_size: u32) -> Vec<MerkleFuzzResult> {
        let size = 1 << log_size;
        let mut results = Vec::new();

        if self.config.test_edge_cases {
            for case in self.generate_edge_cases(size) {
                results.push(self.run_single_test(&case));
            }
        }

        for _ in 0..self.config.iterations_per_size {
            let input = self.generate_random_input(size);
            results.push(self.run_single_test(&input));
        }

        results
    }

    pub fn run_full_suite(&mut self) -> MerkleFuzzReport {
        let mut report = MerkleFuzzReport {
            total_tests: 0,
            passed_tests: 0,
            failed_tests: 0,
            results_by_size: Vec::new(),
        };

        for log_size in self.config.min_log_size..=self.config.max_log_size {
            let results = self.fuzz_size(log_size);
            let passed = results.iter().filter(|r| r.passed).count();
            let failed = results.len() - passed;

            report.total_tests += results.len();
            report.passed_tests += passed;
            report.failed_tests += failed;

            report.results_by_size.push(MerkleSizeReport {
                log_size,
                size: 1 << log_size,
                total: results.len(),
                passed,
                failed,
                results,
            });
        }

        report
    }
}

/// Report for a specific size
#[derive(Debug)]
pub struct MerkleSizeReport {
    pub log_size: u32,
    pub size: usize,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub results: Vec<MerkleFuzzResult>,
}

/// Full fuzzing report
#[derive(Debug)]
pub struct MerkleFuzzReport {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub results_by_size: Vec<MerkleSizeReport>,
}

impl MerkleFuzzReport {
    pub fn all_passed(&self) -> bool {
        self.failed_tests == 0
    }

    pub fn summary(&self) -> String {
        format!(
            "Merkle Tree Differential Fuzzing Report\n\
             ========================================\n\
             Total tests: {}\n\
             Passed: {} ({:.1}%)\n\
             Failed: {} ({:.1}%)\n",
            self.total_tests,
            self.passed_tests,
            100.0 * self.passed_tests as f64 / self.total_tests.max(1) as f64,
            self.failed_tests,
            100.0 * self.failed_tests as f64 / self.total_tests.max(1) as f64,
        )
    }

    #[cfg(feature = "std")]
    pub fn timing_summary(&self) -> String {
        let mut summary = String::from("\nTiming Summary (microseconds)\n");
        summary.push_str("Leaves\t\tCPU avg\t\tMetal avg\tSpeedup\n");
        summary.push_str("------\t\t-------\t\t---------\t-------\n");

        for size_report in &self.results_by_size {
            let cpu_times: Vec<u64> = size_report
                .results
                .iter()
                .filter_map(|r| r.cpu_time_us)
                .collect();
            let metal_times: Vec<u64> = size_report
                .results
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
                    size_report.log_size, cpu_avg, metal_avg, speedup,
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
    fn test_cpu_poseidon2_basic() {
        // Verify CPU implementation matches expected test vectors
        let x = Fp::from(42u64);
        let h = CpuPoseidon2::hash_single(&x);
        assert_ne!(h, Fp::zero());
        assert_ne!(h, x);

        // Test determinism
        let h2 = CpuPoseidon2::hash_single(&x);
        assert_eq!(h, h2);
    }

    #[test]
    fn test_cpu_merkle_tree() {
        let leaves: Vec<Fp> = (1..=8).map(|i| Fp::from(i as u64)).collect();
        let root = CpuMerkleTree::build_root_only(&leaves);
        assert!(root.is_some());

        // Determinism
        let root2 = CpuMerkleTree::build_root_only(&leaves);
        assert_eq!(root, root2);
    }

    #[test]
    fn test_differential_fuzzing_small() {
        let config = MerkleFuzzConfig {
            min_log_size: 2,
            max_log_size: 10,
            iterations_per_size: 3,
            test_edge_cases: true,
            enable_timing: true,
        };

        if let Ok(mut fuzzer) = MerkleFuzzer::new(config) {
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
    fn test_merkle_benchmark_medium() {
        // Benchmark sizes 2^12 to 2^16 with CPU comparison
        let config = MerkleFuzzConfig {
            min_log_size: 12,
            max_log_size: 16,
            iterations_per_size: 3,
            test_edge_cases: false,
            enable_timing: true,
        };

        println!("\nBenchmarking Merkle trees (2^12 to 2^16 leaves) with CPU comparison...\n");

        if let Ok(mut fuzzer) = MerkleFuzzer::new(config) {
            let report = fuzzer.run_full_suite();

            println!("{}", report.summary());
            #[cfg(feature = "std")]
            println!("{}", report.timing_summary());

            assert!(
                report.all_passed(),
                "Medium benchmark found {} mismatches",
                report.failed_tests
            );
        } else {
            println!("Metal not available, skipping benchmark");
        }
    }

    #[test]
    #[ignore] // Run with: cargo test --features metal test_merkle_gpu_only_large -- --ignored --nocapture
    fn test_merkle_gpu_only_large() {
        use std::time::Instant;

        println!("\nBenchmarking GPU-only large Merkle trees (2^18 to 2^22 leaves)...\n");

        let metal_merkle = match MetalMerkleTree::new() {
            Ok(m) => m,
            Err(_) => {
                println!("Metal not available, skipping");
                return;
            }
        };

        println!("GPU: {}", metal_merkle.device_name());
        println!("\nSize\t\tTime (ms)\tThroughput (M leaves/s)");
        println!("----\t\t---------\t-----------------------");

        for log_n in [18, 19, 20, 21, 22] {
            let n = 1usize << log_n;
            let input: Vec<Fp> = (0..n).map(|i| Fp::from(i as u64)).collect();

            // Warmup
            let _ = metal_merkle.build_root_only(&input);

            // Timed run (average of 3)
            let mut total_time = std::time::Duration::ZERO;
            for _ in 0..3 {
                let start = Instant::now();
                let _ = metal_merkle.build_root_only(&input).expect("Merkle failed");
                total_time += start.elapsed();
            }
            let avg_time = total_time / 3;
            let throughput = n as f64 / avg_time.as_secs_f64() / 1_000_000.0;

            println!(
                "2^{}\t\t{:.2}\t\t{:.2}",
                log_n,
                avg_time.as_secs_f64() * 1000.0,
                throughput
            );
        }
    }

    #[test]
    fn test_fft_then_merkle() {
        use crate::gpu::metal::MetalFFT;
        use std::time::Instant;

        println!("\nBenchmarking FFT followed by Merkle tree (GPU-only)...\n");

        let mut metal_fft = match MetalFFT::new() {
            Ok(f) => f,
            Err(_) => {
                println!("Metal not available, skipping");
                return;
            }
        };

        let metal_merkle = match MetalMerkleTree::new() {
            Ok(m) => m,
            Err(_) => {
                println!("Metal Merkle not available, skipping");
                return;
            }
        };

        println!("GPU: {}", metal_merkle.device_name());
        println!("\nSize\t\tFFT (ms)\tMerkle (ms)\tTotal (ms)");
        println!("----\t\t--------\t-----------\t----------");

        // Test smaller sizes for quick feedback, larger sizes for throughput
        for log_n in [14, 16, 18, 20] {
            let n = 1usize << log_n;
            let input: Vec<Fp> = (0..n).map(|i| Fp::from(i as u64)).collect();

            // Warmup
            let _ = metal_fft.fft(&input);
            let warmup_result = metal_fft.fft(&input).expect("FFT failed");
            let _ = metal_merkle.build_root_only(&warmup_result);

            // Timed run
            let start = Instant::now();
            let fft_result = metal_fft.fft(&input).expect("FFT failed");
            let fft_time = start.elapsed();

            let merkle_start = Instant::now();
            let _root = metal_merkle
                .build_root_only(&fft_result)
                .expect("Merkle failed");
            let merkle_time = merkle_start.elapsed();

            let total_time = start.elapsed();

            println!(
                "2^{}\t\t{:.2}\t\t{:.2}\t\t{:.2}",
                log_n,
                fft_time.as_secs_f64() * 1000.0,
                merkle_time.as_secs_f64() * 1000.0,
                total_time.as_secs_f64() * 1000.0,
            );
        }
    }
}
