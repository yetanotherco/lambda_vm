//! The CPU's clock bound.
//!
//! A cycle's timestamp is `4·cycle + 4` over the whole run, and its accesses
//! sit at `+0..=+3`; a CPU padding row continues the `+4` cadence, and the
//! last padding row's PC write (`+1`) is the REGISTER table's final PC token.
//! The CPU holds the timestamp in one column and sends it with a zero high word
//! ("CPU timestamps fit in 32 bits", `cpu.rs`), while MEMW_R, MEMW, REGISTER
//! and the other tables it talks to take a timestamp as two 32-bit words. Below
//! 2^32 the two encodings are one message; at or past it they differ, the
//! buses cannot balance, and the proof cannot verify. A run is refused once a
//! timestamp would reach 2^32 — the walk's window that reaches cycle 2^30 − 1,
//! or the final PC token, which the last chunk's padding rows can put past
//! 2^32 a little below 2^30 cycles — rather than proved and then rejected.
//!
//! `LAMBDA_VM_BLOCK_UNVERIFIABLE_CLOCK=1` proves past the bound anyway, for
//! memory and timing measurements only: the proof does not verify, and the log
//! says so. A release build only: HALT and HINT debug-assert 32-bit timestamps
//! (`halt.rs`, `hint.rs`), so a build with debug assertions refuses instead of
//! proving into them.

use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

use crate::Error;

/// The first timestamp the CPU cannot send: its high word is lost from here.
pub(crate) const CLOCK_LIMIT: u64 = 1 << 32;

/// The knob that proves past [`CLOCK_LIMIT`] anyway (a proof that does not
/// verify).
const UNVERIFIABLE_CLOCK: &str = "LAMBDA_VM_BLOCK_UNVERIFIABLE_CLOCK";

/// Whether the knob's value asks to prove past the bound: `1`, `true` or `yes`.
fn unverifiable_clock_from(value: Option<&str>) -> bool {
    value.is_some_and(|v| matches!(v.trim(), "1" | "true" | "yes"))
}

/// The largest timestamp a window of `cycles` cycles from cycle `first` uses:
/// its last cycle's `4·cycle + 4`, plus the cycle's accesses up to `+3`.
pub(crate) fn window_max_timestamp(first: usize, cycles: usize) -> u64 {
    4 * (first as u64 + cycles as u64) + 3
}

/// The REGISTER table's final PC token: the last CPU padding row's PC write,
/// `halt_timestamp + 4·padding_rows + 1` (`halt_timestamp + 1` without
/// padding) — the largest timestamp the run's trace holds.
pub(crate) fn final_pc_timestamp(halt_timestamp: u64, padding_rows: usize) -> u64 {
    halt_timestamp + 4 * padding_rows as u64 + 1
}

/// Refuses `max_timestamp` at or past [`CLOCK_LIMIT`] (`what` names where it
/// was reached), unless [`UNVERIFIABLE_CLOCK`] is set in a release build: then
/// it warns, once a process, and lets the run go on.
pub(crate) fn check_clock(max_timestamp: u64, what: &str) -> Result<(), Error> {
    check_clock_with(
        max_timestamp,
        what,
        unverifiable_clock_from(std::env::var(UNVERIFIABLE_CLOCK).ok().as_deref()),
    )
}

/// [`check_clock`] with the knob's answer given.
fn check_clock_with(max_timestamp: u64, what: &str, unverifiable: bool) -> Result<(), Error> {
    if max_timestamp < CLOCK_LIMIT {
        return Ok(());
    }
    if unverifiable && cfg!(debug_assertions) {
        return Err(Error::ClockPastLimit(format!(
            "{what} reaches timestamp {max_timestamp}, past the CPU's 32-bit clock (2^32); \
             {UNVERIFIABLE_CLOCK}=1 proves past it only in a release build (HALT and HINT \
             debug-assert 32-bit timestamps)"
        )));
    }
    if unverifiable {
        static WARNED: AtomicBool = AtomicBool::new(false);
        if !WARNED.swap(true, Relaxed) {
            eprintln!(
                "BLOCK CLOCK: UNVERIFIABLE — {what} reaches timestamp {max_timestamp} ≥ 2^32 \
                 (the CPU sends a zero high word); {UNVERIFIABLE_CLOCK}=1 proves anyway, and \
                 THE PROOF WILL NOT VERIFY"
            );
        }
        return Ok(());
    }
    Err(Error::ClockPastLimit(format!(
        "{what} reaches timestamp {max_timestamp}, past the CPU's 32-bit clock (2^32): \
         its buses cannot balance, so the proof would not verify (a run's CPU rows, \
         padding included, number at most 2^30 − 1; {UNVERIFIABLE_CLOCK}=1 proves \
         anyway, for measurements)"
    )))
}

#[cfg(test)]
mod tests {
    use super::{
        CLOCK_LIMIT, check_clock_with, final_pc_timestamp, unverifiable_clock_from,
        window_max_timestamp,
    };
    use crate::Error;

    /// A window ending at cycle 2^30 − 2 (2^30 − 1 cycles) stays under 2^32;
    /// one more cycle puts its timestamp at 2^32.
    #[test]
    fn the_clock_takes_2_30_minus_1_cycles() {
        let last = (1usize << 30) - 1;
        assert_eq!(window_max_timestamp(0, last), CLOCK_LIMIT - 1);
        assert_eq!(window_max_timestamp(last - 5, 5), CLOCK_LIMIT - 1);
        assert!(check_clock_with(window_max_timestamp(0, last), "run", false).is_ok());
        assert_eq!(window_max_timestamp(last, 1), CLOCK_LIMIT + 3);
        assert!(matches!(
            check_clock_with(window_max_timestamp(last, 1), "run", false),
            Err(Error::ClockPastLimit(_))
        ));
        // The final PC token of a run whose padding crosses: refused too.
        assert!(check_clock_with(CLOCK_LIMIT - 1, "pc", false).is_ok());
        assert!(check_clock_with(CLOCK_LIMIT + 1, "pc", false).is_err());
    }

    /// The walk refuses a window whose last cycle's timestamp would reach 2^32,
    /// on both collectors, and takes the window that ends one cycle before.
    #[test]
    fn the_walk_refuses_the_window_that_crosses_2_32() {
        use super::super::{
            DecodeArtifacts, DecodeTable, Elf, collect_cpu_ops_from, collect_cpu_ops_from_table,
        };
        use crate::test_utils::asm_elf_bytes;
        use executor::vm::execution::Executor;

        let program = Elf::load(&asm_elf_bytes("lw_sw_offset_odd")).expect("the ELF loads");
        let logs = Executor::new(&program, Vec::new())
            .expect("the executor starts")
            .run()
            .expect("the program runs")
            .logs;
        let artifacts = DecodeArtifacts::from_elf(&program).expect("the decode artifacts");
        let table = DecodeTable::from_instructions(&artifacts.instructions);
        let n = logs.len();
        // The last cycle at 2^30 − 2: timestamp 2^32 − 4.
        let fits = (1usize << 30) - 1 - n;
        let ops = collect_cpu_ops_from(&logs, &artifacts.instructions, fits).expect("it fits");
        assert_eq!(ops.last().map(|op| op.timestamp), Some(CLOCK_LIMIT - 4));
        let ops = collect_cpu_ops_from_table(&logs, &table, fits).expect("it fits");
        assert_eq!(ops.last().map(|op| op.timestamp), Some(CLOCK_LIMIT - 4));
        // One cycle later: the last cycle's timestamp is 2^32.
        assert!(matches!(
            collect_cpu_ops_from(&logs, &artifacts.instructions, fits + 1),
            Err(Error::ClockPastLimit(_))
        ));
        assert!(matches!(
            collect_cpu_ops_from_table(&logs, &table, fits + 1),
            Err(Error::ClockPastLimit(_))
        ));
    }

    /// A run just under 2^30 cycles whose last chunk's padding rows cross:
    /// every window fits, but the CPU trace's last padded row writes the PC at
    /// 2^32 or past, and that final PC token is refused.
    #[test]
    fn padding_rows_that_cross_2_32_are_refused() {
        use super::super::{DecodeArtifacts, Elf, collect_cpu_ops_from};
        use crate::tables::cpu;
        use crate::test_utils::asm_elf_bytes;
        use executor::vm::execution::Executor;

        let program = Elf::load(&asm_elf_bytes("lw_sw_offset_odd")).expect("the ELF loads");
        let logs = Executor::new(&program, Vec::new())
            .expect("the executor starts")
            .run()
            .expect("the program runs")
            .logs;
        let artifacts = DecodeArtifacts::from_elf(&program).expect("the decode artifacts");
        let n = logs.len();
        let padding = n.next_power_of_two().max(4) - n;
        assert!(padding > 0, "the program's {n} cycles leave padding rows");
        // The halting cycle at 2^30 − 2: the walk takes every window.
        let first = (1usize << 30) - 1 - n;
        let ops = collect_cpu_ops_from(&logs, &artifacts.instructions, first).expect("it fits");
        let halt = ops.last().expect("ops").timestamp;
        assert_eq!(halt, CLOCK_LIMIT - 4);
        // The trace's last padded row writes the PC at the final PC token.
        let trace = cpu::generate_cpu_trace(&ops);
        assert_eq!(trace.num_rows(), n + padding);
        let last_row = *trace
            .get_main(trace.num_rows() - 1, cpu::cols::TIMESTAMP)
            .value();
        let final_pc = final_pc_timestamp(halt, padding);
        assert_eq!(last_row + 1, final_pc);
        assert!(final_pc >= CLOCK_LIMIT);
        assert!(matches!(
            check_clock_with(final_pc, "the CPU's last padding row", false),
            Err(Error::ClockPastLimit(_))
        ));
        // Without padding the same halt fits.
        assert!(check_clock_with(final_pc_timestamp(halt, 0), "pc", false).is_ok());
    }

    /// The knob lets a run past the bound go on in a release build; a build
    /// with debug assertions refuses instead (HALT and HINT assert 32-bit
    /// timestamps).
    #[test]
    fn the_unverifiable_clock_knob_proves_anyway() {
        for ts in [CLOCK_LIMIT, u64::MAX] {
            let run = check_clock_with(ts, "run", true);
            if cfg!(debug_assertions) {
                assert!(matches!(run, Err(Error::ClockPastLimit(_))));
            } else {
                assert!(run.is_ok());
            }
        }
        assert!(unverifiable_clock_from(Some("1")));
        assert!(unverifiable_clock_from(Some(" true ")));
        assert!(unverifiable_clock_from(Some("yes")));
        assert!(!unverifiable_clock_from(Some("0")));
        assert!(!unverifiable_clock_from(Some("")));
        assert!(!unverifiable_clock_from(None));
    }
}
