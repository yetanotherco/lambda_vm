//! What the hint mechanism costs on a real ethrex block, in executor cycles.
//!
//! Three runs of the same guest and block:
//!
//! - **silenced**: no hint answered, every one recomputed in software. This is
//!   what the two-pass flow's recording pass executes.
//! - **on demand**: answered during the run — one execution, and the trace that
//!   gets proved.
//! - **explicit arena**: the same arena shipped up front, for comparison.
//!
//! The number that matters for the one-pass change: `silenced` is pure waste
//! under a recording pass, and it is NOT the trace anyone proves.
//!
//! Run explicitly (guest ELF and fixture must exist):
//!   make executor/program_artifacts/rust/ethrex.elf
//!   cargo test -p executor --test hint_cost_ethrex -- --ignored --nocapture

use executor::elf::Elf;
use executor::vm::execution::Executor;
use std::time::Instant;

fn run_case(elf: &Elf, input: &[u8], silenced: bool) -> (usize, std::time::Duration, usize) {
    let t0 = Instant::now();
    let mut ex = Executor::new(elf, input.to_vec(), &[]).expect("executor");
    if silenced {
        ex.silence_hints();
    }
    let result = ex.run().expect("run");
    (result.logs.len(), t0.elapsed(), result.hints.len())
}

#[test]
#[ignore = "measurement — run explicitly"]
fn ethrex_block_hint_cost() {
    let elf_bytes = std::fs::read("program_artifacts/rust/ethrex.elf")
        .expect("ethrex.elf missing — run `make executor/program_artifacts/rust/ethrex.elf`");
    let elf = Elf::load(&elf_bytes).expect("ELF load");

    for fixture in [
        "tests/ethrex_empty_block.bin",
        "tests/ethrex_simple_tx.bin",
        "tests/ethrex_10_transfers.bin",
        // The fixture `/bench` proves. Present only when the real-block cache
        // has been fetched.
        "tests/ethrex_mainnet_25368371.bin",
    ] {
        let Ok(input) = std::fs::read(fixture) else {
            println!("[hint-cost] {fixture}: missing, skipped");
            continue;
        };

        let (silenced_cycles, silenced_time, _) = run_case(&elf, &input, true);
        let (ondemand_cycles, ondemand_time, slots) = run_case(&elf, &input, false);

        let arena: Vec<[u8; 32]> = {
            let ex = Executor::new(&elf, input.clone(), &[]).expect("executor");
            ex.run().expect("run").hints
        };
        let explicit = Executor::new(&elf, input.clone(), &arena)
            .expect("executor")
            .run()
            .expect("run");
        assert_eq!(
            explicit.logs.len(),
            ondemand_cycles,
            "answering on demand must produce the same trace as shipping the arena"
        );

        println!("[hint-cost] {fixture} — {slots} hints");
        println!(
            "[hint-cost]   silenced (recording pass): {silenced_cycles:>10} cycles  {silenced_time:>10.2?}"
        );
        println!(
            "[hint-cost]   on demand (proved trace):  {ondemand_cycles:>10} cycles  {ondemand_time:>10.2?}"
        );
        println!(
            "[hint-cost]   recording pass costs {:.2}x the proved execution",
            silenced_cycles as f64 / ondemand_cycles as f64
        );
    }
}
