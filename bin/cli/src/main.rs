use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use clap::{Parser, ValueHint};
use executor::{
    elf::{Elf, SymbolTable},
    flamegraph::FlamegraphGenerator,
    vm::execution::Executor,
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

    // Save entry_point before moving program.image into Executor
    let entry_point = program.entry_point;

    let mut executor =
        Executor::new(program.image, entry_point, vec![]).expect("Failed to create executor");

    // Set up flamegraph generator if requested
    let mut generator = args.flamegraph.as_ref().map(|_| {
        let symbols = SymbolTable::parse(&elf_data);
        FlamegraphGenerator::new(symbols, entry_point)
    });

    // Execute in chunks, processing logs only if generating flamegraph
    loop {
        let logs = executor.resume().expect("Failed to resume execution");
        match logs {
            Some(logs) => {
                if let Some(ref mut fg) = generator {
                    // Clone logs to release the borrow on executor
                    let logs: Vec<_> = logs.to_vec();
                    fg.process_logs(&logs, &executor.instructions)
                        .expect("Failed to process logs for flamegraph");
                }
            }
            None => break,
        }
    }
    let _return_values = executor.finish().expect("Failed to finish execution");

    // Write flamegraph output if requested
    if let (Some(output_path), Some(generator)) = (args.flamegraph, generator) {
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

    // TODO: Prove program execution
}
