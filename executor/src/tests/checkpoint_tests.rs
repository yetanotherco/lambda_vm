//! Tests for executor checkpoints: snapshot the VM state mid-execution,
//! recreate an `Executor` from it, and resume — the concatenated logs must be
//! byte-identical to a straight run. This is the determinism property that
//! makes memory-eviction re-execution (Approach 1) sound.
//!
//! The program is built by hand (no ELF fixture) so the test is hermetic: a run
//! of 100_005 `ADDI x5, x5, 1` instructions followed by `JALR x0, 0(x0)` (jump
//! to address 0, which halts the VM). No syscalls are used, and the >100_000
//! instruction count guarantees the snapshot is taken mid-execution (across a
//! `resume()` 100_000-instruction chunk boundary).

use crate::elf::{Elf, Segment};
use crate::vm::execution::Executor;

const ADDI_X5_X5_1: u32 = 0x0012_8293; // addi x5, x5, 1
const JALR_X0_0_X0: u32 = 0x0000_0067; // jalr x0, 0(x0)  -> pc = 0 -> halt
const N_ADDI: usize = 100_005;
const BASE: u64 = 0x1000;

fn long_program() -> Elf {
    let mut values = vec![ADDI_X5_X5_1; N_ADDI];
    values.push(JALR_X0_0_X0);
    Elf {
        entry_point: BASE,
        data: vec![Segment {
            base_addr: BASE,
            values,
            is_executable: true,
        }],
    }
}

#[test]
fn snapshot_resume_produces_identical_logs() {
    let elf = long_program();

    // Straight run.
    let full = Executor::new(&elf, vec![]).unwrap().run().unwrap().logs;
    assert_eq!(full.len(), N_ADDI + 1, "all instructions should log once");
    assert!(full.len() > 100_000, "must span multiple resume() chunks");

    // Run one chunk (100_000 instructions), snapshot mid-execution, recreate, finish.
    let mut exec = Executor::new(&elf, vec![]).unwrap();
    let mut logs = Vec::new();
    {
        let chunk0 = exec.resume().unwrap().expect("at least one chunk");
        logs.extend_from_slice(chunk0);
    }
    assert!(
        logs.len() < full.len(),
        "snapshot must be taken before the program finishes (got {} of {})",
        logs.len(),
        full.len()
    );

    let snapshot = exec.snapshot();
    let mut resumed = Executor::from_snapshot(&elf, snapshot).expect("recreate from snapshot");
    while let Some(chunk) = resumed.resume().unwrap() {
        logs.extend_from_slice(chunk);
    }

    assert_eq!(logs.len(), full.len(), "log count mismatch after snapshot+resume");
    assert_eq!(logs, full, "snapshot+resume logs must equal the straight run");
}

#[test]
fn from_snapshot_at_start_equals_fresh_run() {
    let elf = long_program();
    let full = Executor::new(&elf, vec![]).unwrap().run().unwrap().logs;

    // Snapshot before running anything, recreate, run: identical to fresh.
    let snapshot = Executor::new(&elf, vec![]).unwrap().snapshot();
    let resumed_logs = Executor::from_snapshot(&elf, snapshot)
        .unwrap()
        .run()
        .unwrap()
        .logs;

    assert_eq!(resumed_logs, full);
}
