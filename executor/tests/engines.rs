//! The fast engine against the reference executor on real programs: every
//! asm and Rust guest in `program_artifacts` and the ethrex fixtures in this
//! directory, window by window. Every window's logs (or error), the pc and
//! registers after it, and the memory each engine publishes for the walkers
//! must be the reference's, as must the public output and return values.
//!
//! Needs the guest ELFs: `make compile-programs-asm compile-programs-rust`.
use executor::elf::Elf;
use executor::vm::execution::{Executor, ExecutorError};
use executor::vm::fast::FastExecutor;
use executor::vm::flat_memory::FlatLayout;
use executor::vm::logs::Log;

/// The low region's end at 1 MiB and the high region the top 1 MiB: the
/// guests' heap, input and deep stack then live in the paged store and their
/// accesses cross the regions' edges.
const SMALL: FlatLayout = FlatLayout {
    low_end: 1 << 20,
    high_base: 0u64.wrapping_sub(1 << 20),
};

fn record(log: &Log) -> [u64; 5] {
    [
        log.current_pc,
        log.next_pc,
        log.src1_val,
        log.src2_val,
        log.dst_val,
    ]
}

fn error(e: &ExecutorError) -> String {
    format!("{e:?}")
}

/// What one comparison found.
struct Ran {
    cycles: usize,
    /// The reference's error, if the run ended in one.
    error: Option<String>,
}

/// Runs `program` on both engines (the fast one on `layout`) in windows of
/// `window` cycles, at most `max_cycles`, comparing every window; the memory
/// is compared after every `digest_every`-th window and after the last.
fn compare(
    name: &str,
    program: &Elf,
    input: &[u8],
    layout: FlatLayout,
    window: usize,
    max_cycles: usize,
    digest_every: usize,
) -> Ran {
    let reference = Executor::new(program, input.to_vec());
    let fast = FastExecutor::with_layout(program, input.to_vec(), layout);
    let (mut reference, mut fast) = match (reference, fast) {
        (Ok(r), Ok(f)) => (r, f),
        (r, f) => {
            let (r, f) = (r.err().map(|e| error(&e)), f.err().map(|e| error(&e)));
            assert_eq!(r, f, "{name}: setup");
            return Ran {
                cycles: 0,
                error: r,
            };
        }
    };
    assert_eq!(
        reference.memory().snapshot().content_digest(),
        fast.snapshot().content_digest(),
        "{name}: the memory before the first cycle"
    );
    let mut cycles = 0;
    let mut n = 0;
    let end = loop {
        // Every other window handed out, as the block prover takes them.
        let (want, got) = if n % 2 == 0 {
            (
                reference
                    .resume_with_limit(window)
                    .map(|l| l.map(|l| l.iter().map(record).collect::<Vec<_>>())),
                fast.resume_with_limit(window)
                    .map(|l| l.map(|l| l.iter().map(record).collect::<Vec<_>>())),
            )
        } else {
            (
                reference
                    .resume_owned(window)
                    .map(|l| l.map(|l| l.iter().map(record).collect::<Vec<_>>())),
                fast.resume_owned(window)
                    .map(|l| l.map(|l| l.iter().map(record).collect::<Vec<_>>())),
            )
        };
        let (want, got) = (want.map_err(|e| error(&e)), got.map_err(|e| error(&e)));
        match (&want, &got) {
            (Ok(Some(w)), Ok(Some(g))) => {
                assert_eq!(w.len(), g.len(), "{name}: window {n} at cycle {cycles}");
                if let Some(i) = (0..w.len()).find(|&i| w[i] != g[i]) {
                    panic!(
                        "{name}: cycle {}: reference {:x?}, fast {:x?}",
                        cycles + i,
                        w[i],
                        g[i]
                    );
                }
            }
            _ => assert_eq!(want, got, "{name}: window {n} at cycle {cycles}"),
        }
        assert_eq!(reference.pc(), fast.pc(), "{name}: pc after window {n}");
        assert_eq!(
            reference.registers().to_string(),
            fast.registers().to_string(),
            "{name}: registers after window {n}"
        );
        let last = !matches!(want, Ok(Some(_)));
        if last || n % digest_every == 0 {
            assert_eq!(
                reference.memory().snapshot().content_digest(),
                fast.snapshot().content_digest(),
                "{name}: the memory after window {n}"
            );
        }
        match want {
            Ok(Some(logs)) => cycles += logs.len(),
            Ok(None) => break None,
            Err(e) => break Some(e),
        }
        n += 1;
        if cycles >= max_cycles {
            break None;
        }
    };
    if end.is_none() && reference.pc() == 0 {
        let (r, f) = (reference.finish().unwrap(), fast.finish().unwrap());
        assert_eq!(r.memory_values, f.memory_values, "{name}: public output");
        assert_eq!(r.register_values, f.register_values, "{name}: a0, a1");
    }
    Ran { cycles, error: end }
}

fn elfs(dir: &str) -> Vec<(String, Elf)> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{dir}: {e} (build the guests first)"))
        .map(|entry| entry.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "elf"))
        .collect();
    paths.sort();
    paths
        .into_iter()
        .filter_map(|p| {
            let bytes = std::fs::read(&p).unwrap();
            // An ELF the loader refuses is no run for either engine.
            let elf = Elf::load(&bytes).ok()?;
            Some((p.file_name().unwrap().to_string_lossy().into_owned(), elf))
        })
        .collect()
}

/// Every program in `dir` on both layouts.
fn every_program(dir: &str, min_programs: usize) {
    let programs = elfs(dir);
    assert!(
        programs.len() >= min_programs,
        "{dir}: {} programs, expected at least {min_programs}",
        programs.len()
    );
    let (mut cycles, mut errors) = (0, 0);
    for (name, elf) in &programs {
        for layout in [FlatLayout::GUEST, SMALL] {
            let ran = compare(name, elf, &[], layout, 1 << 14, 40_000_000, 16);
            cycles += ran.cycles;
            errors += usize::from(ran.error.is_some());
        }
    }
    eprintln!(
        "ENGINES {dir}: {} programs, {cycles} cycles on both layouts, {errors} runs ended in the same error",
        programs.len()
    );
}

#[test]
fn every_asm_program_runs_alike() {
    every_program("./program_artifacts/asm", 100);
}

#[test]
fn every_rust_program_runs_alike() {
    every_program("./program_artifacts/rust", 30);
}

#[test]
fn the_ethrex_fixtures_run_alike() {
    let elf = std::fs::read("./program_artifacts/rust/ethrex.elf")
        .expect("program_artifacts/rust/ethrex.elf (make compile-programs-rust)");
    let program = Elf::load(&elf).unwrap();
    for fixture in [
        "ethrex_empty_block",
        "ethrex_simple_tx",
        "ethrex_10_transfers",
        "ethrex_bench_4",
    ] {
        let input = std::fs::read(format!("./tests/{fixture}.bin")).unwrap();
        let ran = compare(
            fixture,
            &program,
            &input,
            FlatLayout::GUEST,
            1 << 20,
            usize::MAX,
            4,
        );
        assert_eq!(ran.error, None, "{fixture}");
        eprintln!("ENGINES {fixture}: {} cycles alike", ran.cycles);
    }
}

/// The ethrex guest on a fixture with the small layout: its heap, input and
/// stack in the paged store, its accesses across the regions' edges.
#[test]
fn an_ethrex_fixture_runs_alike_on_the_small_layout() {
    let elf = std::fs::read("./program_artifacts/rust/ethrex.elf")
        .expect("program_artifacts/rust/ethrex.elf (make compile-programs-rust)");
    let program = Elf::load(&elf).unwrap();
    let input = std::fs::read("./tests/ethrex_simple_tx.bin").unwrap();
    let ran = compare(
        "ethrex_simple_tx (small layout)",
        &program,
        &input,
        SMALL,
        1 << 20,
        usize::MAX,
        4,
    );
    assert_eq!(ran.error, None);
}

/// A real block on both engines, record for record. Ignored: it needs a
/// guest ELF and a block's private input, `ENGINES_ELF` and
/// `ENGINES_INPUT`, e.g.
///
/// ```text
/// ENGINES_ELF=ethrex.elf ENGINES_INPUT=ethrex_mainnet_25475471_8140dd6d.bin \
///   cargo test --release -p executor --test engines -- --ignored --nocapture
/// ```
#[test]
#[ignore = "needs ENGINES_ELF and ENGINES_INPUT"]
fn a_block_runs_alike() {
    let elf = std::fs::read(std::env::var("ENGINES_ELF").expect("ENGINES_ELF")).unwrap();
    let input = std::fs::read(std::env::var("ENGINES_INPUT").expect("ENGINES_INPUT")).unwrap();
    let program = Elf::load(&elf).expect("the ELF");
    let ran = compare(
        "block",
        &program,
        &input,
        FlatLayout::GUEST,
        1 << 20,
        usize::MAX,
        8,
    );
    assert_eq!(ran.error, None);
    eprintln!(
        "ENGINES BLOCK: {} cycles, 0 records differ, memory and public output equal",
        ran.cycles
    );
}
