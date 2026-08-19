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
            let result = Executor::new(&program, private_inputs, &[])
                .expect("executor")
                .run()
                .expect("execution failed");
            println!("cycles = {}", result.logs.len());
        }
        "footprint" => {
            // Run to completion, then classify the touched memory by region so we
            // can see how much of the footprint is stack (contiguous, near
            // STACK_TOP) vs the rest (ELF data / heap / private input, low
            // addresses). Tells us whether a stack-specific Vec store would help.
            use executor::elf::Elf;
            use executor::vm::execution::Executor;
            use executor::vm::registers::STACK_TOP;
            let program = Elf::load(&elf).expect("bad ELF");
            let mut ex = Executor::new(&program, private_inputs, &[]).expect("executor");
            while ex.pc() != 0 {
                match ex.resume_with_limit(usize::MAX).expect("execution failed") {
                    Some(_) => {}
                    None => break,
                }
            }
            // Stack lives in the top half of the address space (grows down from
            // STACK_TOP); ELF data / heap / input are in the low addresses.
            const STACK_THRESHOLD: u64 = 1 << 63;
            let (mut stack, mut other) = (0u64, 0u64);
            let (mut min_stack, mut min_other, mut max_other) = (u64::MAX, u64::MAX, 0u64);
            for (addr, _) in ex.memory().iter_bytes() {
                if addr >= STACK_THRESHOLD {
                    stack += 1;
                    min_stack = min_stack.min(addr);
                } else {
                    other += 1;
                    min_other = min_other.min(addr);
                    max_other = max_other.max(addr);
                }
            }
            let total = stack + other;
            let pct = |n: u64| 100.0 * n as f64 / total.max(1) as f64;
            println!("footprint: {total} touched bytes");
            if stack > 0 {
                let span = STACK_TOP - min_stack + 1;
                println!(
                    "  stack: {stack} bytes ({:.1}%), range [{:#x}..={:#x}], span {span} bytes, density {:.1}%",
                    pct(stack),
                    min_stack,
                    STACK_TOP,
                    100.0 * stack as f64 / span as f64,
                );
            }
            if other > 0 {
                println!(
                    "  other (data/heap/input): {other} bytes ({:.1}%), range [{:#x}..={:#x}]",
                    pct(other),
                    min_other,
                    max_other,
                );
            }
        }
        "main" => {
            lambda_vm_prover::prove_with_inputs(&elf, &private_inputs, &[])
                .expect("monolithic prove failed");
            println!("main prove ok ({} bytes ELF)", elf.len());
        }
        "cont" => {
            let epoch_size_log2: u32 = args
                .get(3)
                .map(|s| s.parse().expect("bad epoch_size_log2"))
                .unwrap_or(16);
            // Match the monolithic `main` mode's options (blowup 2) for a fair comparison.
            let opts = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(2)
                .expect("blowup=2 is always valid");
            let output = lambda_vm_prover::continuation::prove_and_verify_continuation(
                &elf,
                &private_inputs,
                epoch_size_log2,
                &opts,
            )
            .expect("continuation failed");
            assert!(output.is_some(), "continuation did not verify");
            println!(
                "cont prove+verify ok (epoch_size_log2={epoch_size_log2}, epoch_size={})",
                1usize << epoch_size_log2
            );
        }
        other => {
            eprintln!("unknown mode {other:?}; use count|footprint|main|cont");
            std::process::exit(2);
        }
    }
    println!("elapsed {:.2}s", start.elapsed().as_secs_f64());
}
