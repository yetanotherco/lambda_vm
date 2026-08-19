use executor::{
    elf::Elf,
    vm::execution::{Executor, ExecutorError},
};
use std::fs;

fn main() -> Result<(), ExecutorError> {
    println!("Reading elf");
    let elf_data = std::fs::read("./program_artifacts/rust/ethrex.elf").unwrap();
    let inputs = fs::read("tests/ethrex_simple_tx.bin").unwrap();
    let program = Elf::load(&elf_data).unwrap();
    let executor = Executor::new(&program, inputs, &[])?;
    executor.run()?;
    Ok(())
}
