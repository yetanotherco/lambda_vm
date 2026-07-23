//! Phase-0 walk-cost measurement for the GPU-memory-walk decision
//! (`reports/tracegen/GPU-MEMORY-WALK-SCOPE.md`). Isolates the CPU memory-model walk
//! cost on a chosen ELF — the work a device memory walk would replace — alongside the
//! register walk for a same-machine baseline. The register walk measured ~8 ms (a net
//! loss on GPU); this decides whether memory is different.
//!
//! Run: `LAMBDA_VM_BENCH_ELF=executor/program_artifacts/rust/memory.elf \
//!   cargo test -p lambda-vm-prover --release --lib walk_cost -- --ignored --nocapture`

use std::env;
use std::fs;

use executor::elf::Elf;
use executor::vm::execution::Executor;

use crate::tables::MaxRowsConfig;
use crate::tables::decode;
use crate::tables::trace_builder;
use crate::tables::trace_builder::Traces;

#[test]
#[ignore = "measurement; set LAMBDA_VM_BENCH_ELF and run --ignored --nocapture"]
fn walk_cost_bench() {
    let path = env::var("LAMBDA_VM_BENCH_ELF")
        .expect("set LAMBDA_VM_BENCH_ELF to an ELF path (e.g. .../rust/memory.elf)");
    let bytes = fs::read(&path).expect("read ELF");
    let elf = Elf::load(&bytes).expect("load ELF");
    let input = env::var("LAMBDA_VM_BENCH_INPUT")
        .ok()
        .map(|p| fs::read(p).expect("read input"))
        .unwrap_or_default();
    let executor = Executor::new(&elf, input.clone()).expect("executor");
    let result = executor.run().expect("run");
    let instructions = decode::instructions_from_elf(&elf).expect("decode instructions");

    let (mem_dur, mem_ops, mem_bytes) =
        trace_builder::time_memory_walk_from_logs(&result.logs, &instructions);
    let (reg_dur, reg_rows) =
        trace_builder::time_register_walk_from_logs(&result.logs, &instructions);

    // Full trace-build for context (walks as a fraction of the whole; GPU-resident
    // fills fire under `cuda`). Reset the timeline so the breakdown is this build only.
    #[cfg(feature = "instruments")]
    stark::instruments::reset_timeline();
    let t = std::time::Instant::now();
    let _traces =
        Traces::from_elf_and_logs_minimal(&elf, &result.logs, &MaxRowsConfig::default(), &input)
            .expect("trace build");
    let full = t.elapsed();

    println!("=== WALK COST [{path}] ===");
    println!("cycles (logs):        {}", result.logs.len());
    println!("MEMORY walk:          {mem_dur:?}   ({mem_ops} load/store ops, {mem_bytes} bytes)");
    println!("REGISTER walk:        {reg_dur:?}   ({reg_rows} rows)");
    println!("FULL CPU trace-build: {full:?}");
    println!(
        "walks as % of build:  mem {:.1}%, reg {:.1}%",
        100.0 * mem_dur.as_secs_f64() / full.as_secs_f64(),
        100.0 * reg_dur.as_secs_f64() / full.as_secs_f64(),
    );

    // Phase breakdown of the trace-build (which phase is the real bottleneck).
    #[cfg(feature = "instruments")]
    {
        let spans = stark::instruments::take_timeline();
        print!("{}", stark::instruments::format_timeline(&spans));
    }
}
