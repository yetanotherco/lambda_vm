//! Trace-build timing: the p5 single-build resident path (flag ON) vs the per-chip host-op device
//! path (flag OFF). The resident-chips flag is read once and cached (`OnceLock`), so ON vs OFF must
//! be SEPARATE process runs — run this twice:
//!
//!   # resident single-build ON
//!   LAMBDA_VM_BENCH_ELF=.../rust/ethrex.elf LAMBDA_VM_BENCH_INPUT=.../ethrex_5tx.bin \
//!     LAMBDA_VM_GPU_RESIDENT_CHIPS=1 \
//!     cargo test -p lambda-vm-prover --release --features cuda,instruments --lib \
//!       gpu_resident_bench -- --ignored --nocapture
//!   # baseline OFF (unset the flag)
//!   LAMBDA_VM_BENCH_ELF=... LAMBDA_VM_BENCH_INPUT=... \
//!     cargo test -p lambda-vm-prover --release --features cuda,instruments --lib \
//!       gpu_resident_bench -- --ignored --nocapture
//!
//! Reports best-of-N `Traces::from_elf_and_logs` wall-clock + the instruments phase tree (p3to5 /
//! p5_generate / per-table gen_* spans) so we can see whether the single upload actually speeds up
//! p5 chip generation or the eager shared-devops upload just adds overhead.

use std::env;
use std::fs;
use std::time::Instant;

use executor::elf::Elf;
use executor::vm::execution::Executor;

use crate::tables::trace_builder::Traces;
use crate::tables::MaxRowsConfig;

#[test]
#[ignore = "measurement; requires GPU + LAMBDA_VM_BENCH_ELF; run --ignored --nocapture"]
fn gpu_resident_bench() {
    let path = env::var("LAMBDA_VM_BENCH_ELF").expect("set LAMBDA_VM_BENCH_ELF (ethrex.elf)");
    let bytes = fs::read(&path).expect("read ELF");
    let elf = Elf::load(&bytes).expect("load ELF");
    let input = env::var("LAMBDA_VM_BENCH_INPUT")
        .ok()
        .map(|p| fs::read(p).expect("read input"))
        .unwrap_or_default();
    let executor = Executor::new(&elf, input.clone()).expect("executor");
    let result = executor.run().expect("run");
    let max_rows = MaxRowsConfig::default();

    let resident = crate::tables::gpu_trace::gpu_resident_chips_enabled();
    println!("=== gpu_resident_bench: resident-chips single-build = {resident} ({} cycles) ===", result.logs.len());

    // Thread the executor-recorded precompile inputs (Option A). Required for the GPU_FULL
    // memory-drop path — without them the ECSM/KECCAK/COMMIT collectors read from the init-only
    // memory_state and fail (e.g. ECSM ScalarIsZero). Byte-identical on the non-drop path.
    let pi = Some(&result.precompile_inputs);

    // Warm-up (GPU context, allocations, code cache) — discarded.
    let _ = Traces::from_elf_and_logs_with_precompiles(&elf, &result.logs, pi, &max_rows, &input)
        .expect("warm");

    let iters = 3usize;
    let mut best = f64::MAX;
    for i in 0..iters {
        #[cfg(feature = "instruments")]
        stark::instruments::reset_timeline();
        let t = Instant::now();
        let traces =
            Traces::from_elf_and_logs_with_precompiles(&elf, &result.logs, pi, &max_rows, &input)
                .expect("build");
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        best = best.min(ms);
        println!("[iter {i}] trace_build = {ms:.1} ms");
        #[cfg(feature = "instruments")]
        if i == iters - 1 {
            println!(
                "--- instruments phase tree ---\n{}",
                stark::instruments::format_timeline(&stark::instruments::take_timeline())
            );
        }
        drop(traces);
    }
    println!("=== BEST trace_build = {best:.1} ms (resident single-build={resident}) ===");
}
