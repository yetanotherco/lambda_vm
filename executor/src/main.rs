use executor::{
    elf::Elf,
    vm::execution::{ExecutorError, run_program},
};
use std::fs;

fn main() -> Result<(), ExecutorError> {
    println!("Reading elf");
    let elf_data = std::fs::read("./program_artifacts/rust/ethrex.elf").unwrap();
    let inputs = fs::read("tests/ethrex_hoodi.bin").unwrap();
    let program = Elf::load(&elf_data).unwrap();
    run_program(&program, inputs)?;
    Ok(())
}
