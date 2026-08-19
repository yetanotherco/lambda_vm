//! Asserts predicted `peak_bytes` does not underestimate jemalloc-measured
//! heap during a proof. Lives in its own integration-test binary so that
//! `#[global_allocator]` and `tikv_jemalloc_ctl::stats::allocated` reads are
//! isolated from the rest of the prover test suite.

#![cfg(feature = "disk-spill")]

use lambda_vm_prover::auto_storage::{SAFETY_FRACTION_DEN, SAFETY_FRACTION_NUM, peak_bytes};
use lambda_vm_prover::prove_with_options_and_inputs;
use lambda_vm_prover::tables::MaxRowsConfig;
use lambda_vm_prover::tables::trace_builder::count_table_lengths;
use lambda_vm_prover::test_utils::{asm_elf_bytes, run_asm_elf};
use stark::proof::options::GoldilocksCubicProofOptions;
use stark::prover::storage_estimate_parallelism;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;
use tikv_jemalloc_ctl::{epoch, stats};

#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn allocated_bytes() -> usize {
    epoch::advance().ok();
    stats::allocated::read().unwrap_or(0)
}

#[test]
fn peak_bytes_does_not_underestimate_measured_heap() {
    let (elf, logs, _) = run_asm_elf("fib_iterative_372k");
    let elf_bytes = asm_elf_bytes("fib_iterative_372k");

    let max_rows = MaxRowsConfig::default();
    let lengths = count_table_lengths(&elf, &logs, &max_rows, &[], &[])
        .expect("count_table_lengths succeeds");

    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is valid");
    let predicted =
        peak_bytes(&lengths, opts.blowup_factor, storage_estimate_parallelism()) as usize;

    drop(logs);

    let baseline = allocated_bytes();
    let peak = Arc::new(AtomicUsize::new(baseline));
    let stop = Arc::new(AtomicBool::new(false));

    let sampler = {
        let peak = Arc::clone(&peak);
        let stop = Arc::clone(&stop);
        thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                peak.fetch_max(allocated_bytes(), Ordering::Relaxed);
                thread::sleep(Duration::from_millis(10));
            }
        })
    };

    let _proof = prove_with_options_and_inputs(&elf_bytes, &[], &[], &opts, &max_rows)
        .expect("proof succeeds");

    stop.store(true, Ordering::Relaxed);
    sampler.join().expect("sampler joins");

    let measured = peak.load(Ordering::Relaxed).saturating_sub(baseline);

    eprintln!(
        "peak_bytes calibration: predicted={predicted} bytes, measured_heap={measured} bytes, ratio={:.2}",
        predicted as f64 / measured as f64
    );

    let safety_num = SAFETY_FRACTION_NUM as usize;
    let safety_den = SAFETY_FRACTION_DEN as usize;
    assert!(
        predicted.saturating_mul(safety_den) >= measured.saturating_mul(safety_num),
        "peak_bytes underestimates measured heap: predicted={predicted}, measured={measured}"
    );
}
