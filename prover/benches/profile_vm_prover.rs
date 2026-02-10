// Profiling binary for VM prover to generate flamegraph data.
//
// Run with: `samply record cargo bench --bench profile_vm_prover --features parallel`
// Or with hyperfine: `hyperfine --runs 1 './target/release/deps/profile_vm_prover-*'`
//
// Uses all_instructions_64.elf which exercises all supported RISC-V instructions.

use lambda_vm_prover::test_utils::asm_elf_bytes;

fn main() {
    let elf_name = "all_instructions_64";
    let elf_bytes = asm_elf_bytes(elf_name);

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
