use vm::{elf::Elf, vm::execution::run_program};

fn main() {
    let mut args = std::env::args();
    let elf_filename = args.nth(1).expect("No filename given");
    let elf_data = std::fs::read(elf_filename).expect("Failed to read elf file");
    let program = Elf::load(&elf_data).expect("Failed to load elf program");
    let (logs, _) = run_program(program.image, program.entry_point).expect("Failed to run program");
    // TODO: Prove program
}
