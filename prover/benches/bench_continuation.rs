//! Peak-memory benchmark: monolithic proving vs continuation (streaming-epoch)
//! proving, for large programs.
//!
//! This is a plain one-shot binary (`harness = false`), not a Criterion bench:
//! Criterion measures time over many iterations, whereas the point here is the
//! peak resident set of a SINGLE prove. Wrap it in the OS timer to capture RSS,
//! on Linux:
//!     /usr/bin/time -v <binary> main <elf_path>
//!     /usr/bin/time -v <binary> cont <elf_path> 65536
//!
//! Build + locate the binary:
//!     cargo build --release --bench bench_continuation
//!     ls target/release/deps/bench_continuation-*   # the executable (no .d)
//!
//! Args:
//!     <mode>        "count", "main" (monolithic prove) or "cont" (continuation)
//!     <elf_path>    path to a compiled ELF artifact
//!     [epoch_size]  epoch length in cycles for "cont" (default 65536)
//!
//! Env:
//!     BENCH_PRIVATE_INPUT  optional path to a private-input file (e.g. an
//!                          ethrex ProgramInput .bin). Empty if unset.

use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: bench_continuation <count|main|cont> <elf_path> [epoch_size]");
        std::process::exit(2);
    }
    let mode = args[1].as_str();
    let elf_path = &args[2];
    let elf = std::fs::read(elf_path).expect("failed to read ELF");
    let private_inputs: Vec<u8> = match std::env::var("BENCH_PRIVATE_INPUT") {
        Ok(path) if !path.is_empty() => {
            std::fs::read(&path).expect("failed to read BENCH_PRIVATE_INPUT file")
        }
        _ => Vec::new(),
    };

    let start = Instant::now();
    match mode {
        "count" => {
            // Count cycles by running the executor to completion (no proving).
            // Cycle count is a linear proxy for monolithic proving memory.
            use executor::elf::Elf;
            use executor::vm::execution::Executor;
            let program = Elf::load(&elf).expect("bad ELF");
            let result = Executor::new(&program, private_inputs)
                .expect("executor")
                .run()
                .expect("execution failed");
            println!("cycles = {}", result.logs.len());
        }
        "main" => {
            lambda_vm_prover::prove_with_inputs(&elf, &private_inputs)
                .expect("monolithic prove failed");
            println!("main prove ok ({} bytes ELF)", elf.len());
        }
        "cont" => {
            let epoch_size: usize = args
                .get(3)
                .map(|s| s.parse().expect("bad epoch_size"))
                .unwrap_or(65536);
            let ok = lambda_vm_prover::continuation::prove_and_verify_continuation(
                &elf,
                &private_inputs,
                epoch_size,
            )
            .expect("continuation failed");
            assert!(ok, "continuation did not verify");
            println!("cont prove+verify ok (epoch_size={epoch_size})");
        }
        other => {
            eprintln!("unknown mode {other:?}; use main|cont");
            std::process::exit(2);
        }
    }
    println!("elapsed {:.2}s", start.elapsed().as_secs_f64());
}
