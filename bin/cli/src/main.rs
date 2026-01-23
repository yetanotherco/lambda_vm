use std::path::PathBuf;

use clap::{Parser, ValueHint};
use executor::{elf::Elf, vm::execution::run_program};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(value_parser, value_hint=ValueHint::FilePath)]
    filename: PathBuf,
}

fn main() {
    let args = Args::parse_from(std::env::args());
    let elf_data = std::fs::read(args.filename).expect("Failed to read elf file");
    let program = Elf::load(&elf_data).expect("Failed to load elf program");
    let _result =
        run_program(&program.data, program.entry_point, vec![]).expect("Failed to run program");
    // TODO: Prove program execution using result.logs and result.instructions
}
