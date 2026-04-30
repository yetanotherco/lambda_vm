//! Calibration test: predicted [`peak_bytes`] vs measured RSS during a real proof.
//!
//! Runs a small fib_iterative proof, samples the process's RSS while the proof
//! is running, and asserts the prediction is within 2× of the measured peak
//! (after subtracting the pre-proof baseline). RSS includes mmap'd files, the
//! code segment, and allocator slack on top of the heap-only quantity that
//! [`peak_bytes`] models, so the bound is intentionally loose; the test is a
//! regression guard against silent drift, not a tightness measure.
//!
//! [`peak_bytes`]: crate::auto_storage::peak_bytes

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use stark::proof::options::GoldilocksCubicProofOptions;

use crate::auto_storage;
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::count_table_lengths;
use crate::test_utils::{asm_elf_bytes, run_asm_elf};

fn current_rss_bytes() -> Option<usize> {
    let pid = sysinfo::get_current_pid().ok()?;
    let mut sys = sysinfo::System::new();
    sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]));
    sys.process(pid).map(|p| p.memory() as usize)
}

#[test]
fn peak_bytes_within_2x_of_measured_rss() {
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

    // Drop logs etc. before sampling baseline so they don't inflate it.
    drop(logs);

    let baseline = current_rss_bytes().expect("RSS reader works on this platform");
    let peak = Arc::new(AtomicUsize::new(baseline));
    let stop = Arc::new(AtomicBool::new(false));

    let sampler = {
        let peak = Arc::clone(&peak);
        let stop = Arc::clone(&stop);
        thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                if let Some(rss) = current_rss_bytes() {
                    peak.fetch_max(rss, Ordering::Relaxed);
                }
                thread::sleep(Duration::from_millis(50));
            }
        })
    };

    let _proof = crate::prove_with_options_and_inputs(&elf_bytes, &[], &opts, &max_rows)
        .expect("proof succeeds");

    stop.store(true, Ordering::Relaxed);
    sampler.join().expect("sampler joins");

    let measured = peak.load(Ordering::Relaxed).saturating_sub(baseline);

    eprintln!(
        "peak_bytes calibration: predicted={predicted} bytes, measured_above_baseline={measured} bytes"
    );

    assert!(
        predicted.saturating_mul(2) >= measured,
        "peak_bytes underestimates measured RSS by more than 2×: \
         predicted={predicted}, measured={measured}"
    );
    assert!(
        predicted <= measured.saturating_mul(2),
        "peak_bytes overestimates measured RSS by more than 2×: \
         predicted={predicted}, measured={measured}"
    );
}
