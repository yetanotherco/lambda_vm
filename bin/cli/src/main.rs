use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use clap::{Parser, ValueHint};
use executor::{
    elf::{Elf, SymbolTable},
    flamegraph::FlamegraphGenerator,
    vm::execution::run_program,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(value_parser, value_hint=ValueHint::FilePath)]
    filename: PathBuf,

    /// Generate flamegraph folded stacks to file
    #[arg(long, value_hint=ValueHint::FilePath)]
    flamegraph: Option<PathBuf>,
}

fn main() {
    let args = Args::parse_from(std::env::args());
    let elf_data = std::fs::read(&args.filename).expect("Failed to read elf file");
    let program = Elf::load(&elf_data).expect("Failed to load elf program");

    // Save entry_point before moving program.image into run_program
    let entry_point = program.entry_point;

    let execution_result = run_program(program.image, entry_point, vec![])
        .expect("Failed to run program");

    // Generate flamegraph if requested
    if let Some(output_path) = args.flamegraph {
        let symbols = SymbolTable::parse(&elf_data);
        let mut generator = FlamegraphGenerator::new(symbols, entry_point);
        generator
            .process_logs(&execution_result.logs, &execution_result.instructions)
            .expect("Failed to process logs for flamegraph");

        let file = File::create(&output_path).expect("Failed to create flamegraph output file");
        let mut writer = BufWriter::new(file);
        generator
            .write_folded(&mut writer)
            .expect("Failed to write flamegraph output");

        eprintln!(
            "Flamegraph written to {:?} ({} instructions)",
            output_path,
            generator.total_instructions()
        );
    }

    // TODO: Prove program execution using result.logs and result.instructions
}
