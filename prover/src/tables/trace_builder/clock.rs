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

use crate::Error;

/// The first timestamp the CPU cannot send: its high word is lost from here.
pub(crate) const CLOCK_LIMIT: u64 = 1 << 32;

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
/// was reached).
pub(crate) fn check_clock(max_timestamp: u64, what: &str) -> Result<(), Error> {
    if max_timestamp < CLOCK_LIMIT {
        return Ok(());
    }
    Err(Error::ClockPastLimit(format!(
        "{what} reaches timestamp {max_timestamp}, past the CPU's 32-bit clock (2^32): \
         its buses cannot balance, so the proof would not verify (a run's CPU rows, \
         padding included, number at most 2^30 − 1)"
    )))
}

#[cfg(test)]
mod tests {
    use super::{CLOCK_LIMIT, check_clock, final_pc_timestamp, window_max_timestamp};
    use crate::Error;

    /// A window ending at cycle 2^30 − 2 (2^30 − 1 cycles) stays under 2^32;
    /// one more cycle puts its timestamp at 2^32.
    #[test]
    fn the_clock_takes_2_30_minus_1_cycles() {
        let last = (1usize << 30) - 1;
        assert_eq!(window_max_timestamp(0, last), CLOCK_LIMIT - 1);
        assert_eq!(window_max_timestamp(last - 5, 5), CLOCK_LIMIT - 1);
        assert!(check_clock(window_max_timestamp(0, last), "run").is_ok());
        assert_eq!(window_max_timestamp(last, 1), CLOCK_LIMIT + 3);
        assert!(matches!(
            check_clock(window_max_timestamp(last, 1), "run"),
            Err(Error::ClockPastLimit(_))
        ));
        // The final PC token of a run whose padding crosses: refused too.
        assert!(check_clock(CLOCK_LIMIT - 1, "pc").is_ok());
        assert!(check_clock(CLOCK_LIMIT + 1, "pc").is_err());
    }

    /// The walk refuses a window whose last cycle's timestamp would reach 2^32,
    /// and takes the window that ends one cycle before.
    #[test]
    fn the_walk_refuses_the_window_that_crosses_2_32() {
        use super::super::{DecodeArtifacts, Elf, collect_cpu_ops_from};
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
        // The last cycle at 2^30 − 2: timestamp 2^32 − 4.
        let fits = (1usize << 30) - 1 - n;
        let ops = collect_cpu_ops_from(&logs, &artifacts.decode, fits).expect("it fits");
        assert_eq!(ops.last().map(|op| op.timestamp), Some(CLOCK_LIMIT - 4));
        // One cycle later: the last cycle's timestamp is 2^32.
        assert!(matches!(
            collect_cpu_ops_from(&logs, &artifacts.decode, fits + 1),
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
        let ops = collect_cpu_ops_from(&logs, &artifacts.decode, first).expect("it fits");
        let halt = ops.last().expect("ops").timestamp;
        assert_eq!(halt, CLOCK_LIMIT - 4);
        // The trace's last padded row writes the PC at the final PC token.
        // TIMESTAMP is the table's own clock `4·(row + 1)`; the AIR's base, four
        // times the rows before the table, puts it on the run's clock.
        let trace = cpu::generate_cpu_trace(&ops, &artifacts.decode);
        assert_eq!(trace.num_rows(), n + padding);
        let base = cpu::TIMESTAMP_STRIDE * first as u64;
        let last_row = base
            + *trace
                .get_main(trace.num_rows() - 1, cpu::cols::TIMESTAMP)
                .value();
        let final_pc = final_pc_timestamp(halt, padding);
        assert_eq!(last_row + 1, final_pc);
        assert!(final_pc >= CLOCK_LIMIT);
        assert!(matches!(
            check_clock(final_pc, "the CPU's last padding row"),
            Err(Error::ClockPastLimit(_))
        ));
        // Without padding the same halt fits.
        assert!(check_clock(final_pc_timestamp(halt, 0), "pc").is_ok());
    }
}
