use std::path::PathBuf;

use clap::{Parser, ValueHint};
use executor::{elf::Elf, vm::execution::Executor};

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

    let mut executor = Executor::new(&program, vec![]).expect("Failed to create executor");

    while let Some(_logs) = executor.resume().expect("Failed to execute") {
        // Process logs here (e.g., generate trace rows)
    }
}
