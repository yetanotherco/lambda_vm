//! Asserts predicted [`peak_bytes`](crate::auto_storage::peak_bytes) does not
//! underestimate jemalloc-measured heap during a proof.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use stark::proof::options::GoldilocksCubicProofOptions;
use tikv_jemalloc_ctl::{epoch, stats};

use crate::auto_storage;
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::count_table_lengths;
use crate::test_utils::{asm_elf_bytes, run_asm_elf};

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
    let lengths =
        count_table_lengths(&elf, &logs, &max_rows, &[]).expect("count_table_lengths succeeds");

    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is valid");
    let predicted = auto_storage::peak_bytes(
        &lengths,
        opts.blowup_factor,
        stark::prover::table_parallelism(),
    ) as usize;

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

    let _proof = crate::prove_with_options_and_inputs(&elf_bytes, &[], &opts, &max_rows)
        .expect("proof succeeds");

    stop.store(true, Ordering::Relaxed);
    sampler.join().expect("sampler joins");

    let measured = peak.load(Ordering::Relaxed).saturating_sub(baseline);

    eprintln!(
        "peak_bytes calibration: predicted={predicted} bytes, measured_heap={measured} bytes, ratio={:.2}",
        predicted as f64 / measured as f64
    );

    let safety_num = auto_storage::SAFETY_FRACTION_NUM as usize;
    let safety_den = auto_storage::SAFETY_FRACTION_DEN as usize;
    assert!(
        predicted.saturating_mul(safety_den) >= measured.saturating_mul(safety_num),
        "peak_bytes underestimates measured heap below SAFETY_FRACTION ({safety_num}/{safety_den}): \
         predicted={predicted}, measured={measured}"
    );
}
