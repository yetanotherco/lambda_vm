// Profiling binary for VM prover to generate flamegraph data.
//
// Run with: `samply record cargo bench --bench profile_vm_prover --features parallel`
// Or with hyperfine: `hyperfine --runs 1 './target/release/deps/profile_vm_prover-*'`
//
// Default ELF: fib_iterative_372k (~372k steps, realistic workload).
// Override:    cargo bench --bench profile_vm_prover --features parallel -- <elf_name>

use lambda_vm_prover::test_utils::asm_elf_bytes;

fn main() {
    let elf_name = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with('-'))
        .unwrap_or_else(|| "fib_iterative_372k".to_string());
    let elf_bytes = asm_elf_bytes(&elf_name);

    println!("Starting VM prover profiling...");
    println!("Configuration:");
    println!("  - ELF: {} (ASM)", elf_name);

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
