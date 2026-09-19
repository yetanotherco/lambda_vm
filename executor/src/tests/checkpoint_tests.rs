//! Executor checkpoints: snapshot the VM mid-execution, rebuild an `Executor`
//! from it, and resume — the concatenated logs must equal a straight run's.
//!
//! That equality is the property everything built on re-execution rests on: a
//! prover may drop what it produced only if it can get exactly that back.
//!
//! The program is built by hand rather than loaded from an ELF fixture, so the
//! test is hermetic: 100_005 `ADDI x5, x5, 1` followed by `JALR x0, 0(x0)`,
//! which jumps to address 0 and halts. No syscalls, and the instruction count
//! is over 100_000 on purpose — the snapshot then lands mid-execution, across a
//! `resume()` chunk boundary, instead of at a point the chunking makes easy.

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

    let full = Executor::new(&elf, vec![]).unwrap().run().unwrap().logs;
    assert_eq!(full.len(), N_ADDI + 1, "every instruction should log once");

    // One chunk, then snapshot: the cut lands mid-execution.
    let mut exec = Executor::new(&elf, vec![]).unwrap();
    let mut logs = Vec::new();
    {
        let chunk0 = exec.resume().unwrap().expect("at least one chunk");
        logs.extend_from_slice(chunk0);
    }
    assert!(
        logs.len() < full.len(),
        "the snapshot must be taken before the program ends (got {} of {})",
        logs.len(),
        full.len()
    );

    let snapshot = exec.snapshot();
    let mut resumed = Executor::from_snapshot(&elf, snapshot).expect("recreate from snapshot");
    while let Some(chunk) = resumed.resume().unwrap() {
        logs.extend_from_slice(chunk);
    }

    assert_eq!(
        logs.len(),
        full.len(),
        "log count differs after snapshot + resume"
    );
    assert_eq!(logs, full, "resumed logs must equal the straight run's");
}
