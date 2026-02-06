// Profiling binary for VM prover to generate flamegraph data.
//
// Run with: `samply record cargo bench --bench profile_vm_prover --features parallel`
// Or with hyperfine: `hyperfine --runs 1 './target/release/deps/profile_vm_prover-*'`
//
// Uses bench_1m.elf which generates ~2^20 CPU rows, matching the bitwise table size.

use lambda_vm_prover::test_utils::rust_elf_bytes;

fn main() {
    // Use fibonacci_iterative.elf: Rust program, fib(1000) with u64
    let elf_name = "fibonacci_iterative";
    let elf_bytes = rust_elf_bytes(elf_name);

    println!("Starting VM prover profiling...");
    println!("Configuration:");
    println!("  - ELF: {} (Rust)", elf_name);
    println!("  - Expected: fib(1000) with u64, ~8k instructions");

    #[cfg(feature = "parallel")]
    println!(
        "  - Parallel: ENABLED (rayon threads: {})",
        rayon::current_num_threads()
    );

    #[cfg(not(feature = "parallel"))]
    println!("  - Parallel: DISABLED");

    println!("\nGenerating proof (this will take a while)...");
    let start = std::time::Instant::now();

    let _proof = lambda_vm_prover::prove(&elf_bytes).expect("Failed to generate proof");

    let elapsed = start.elapsed();
    println!("\nProof generation completed in {:?}", elapsed);
}
