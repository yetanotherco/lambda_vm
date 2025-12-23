use executor::{
    elf::Elf,
    vm::execution::{ExecutorError, run_program},
};

fn main() -> Result<(), ExecutorError> {
    println!("Reading elf");
    let elf_data = std::fs::read("./program_artifacts/asm/basic_program.elf").unwrap();
    let program = Elf::load(&elf_data).unwrap();
    println!("Program entry: 0x{:08x}", program.entry_point);
    program.image.iter().for_each(|(addr, word)| {
        println!("0x{addr:08x}: 0x{word:08x}");
    });
    run_program(program.image, program.entry_point, true)?;
    Ok(())
}
