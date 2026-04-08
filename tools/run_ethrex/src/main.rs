use executor::elf::Elf;
use executor::vm::execution::Executor;
use std::time::Instant;

fn main() {
    let elf_path = "../../executor/program_artifacts/rust/ethrex.elf";
    let input_path = "../../executor/tests/ethrex_empty_block.bin";

    println!("Loading ELF from {elf_path}...");
    let elf_data = std::fs::read(elf_path).expect("Failed to read ELF");
    let program = Elf::load(&elf_data).expect("Failed to load ELF");

    println!("Loading input from {input_path}...");
    let private_input = std::fs::read(input_path).expect("Failed to read input");
    println!("Input size: {} bytes", private_input.len());

    println!("Creating executor...");
    let executor = Executor::new(&program, private_input).expect("Failed to create executor");

    println!("Running...");
    let start = Instant::now();
    let result = executor.run().expect("Execution failed");
    let elapsed = start.elapsed();

    println!("Execution succeeded in {:.2?}", elapsed);
    println!("  Instructions executed: {}", result.logs.len());
    println!(
        "  Public output: {} bytes",
        result.return_values.memory_values.len()
    );
}
