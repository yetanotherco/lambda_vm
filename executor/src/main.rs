use executor::{
    elf::Elf,
    vm::execution::{ExecutorError, run_program},
};
use tracing::{info, trace};

fn main() -> Result<(), ExecutorError> {
    tracing_subscriber::fmt::init();

    info!("Reading elf");
    let elf_data = std::fs::read("./program_artifacts/asm/basic_program.elf").unwrap();
    let program = Elf::load(&elf_data).unwrap();
    info!("Program entry: {:#010x}", program.entry_point);
    program.image.iter().for_each(|(addr, word)| {
        trace!("{:#010x}: {:#010x}", addr, word);
    });
    run_program(program.image, program.entry_point, vec![])?;
    Ok(())
}
