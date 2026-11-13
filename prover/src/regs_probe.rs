//! A measurement-only probe of the register argument's table shape
//! (`LAMBDA_VM_PROBE_REGS_SHAPE=1`).
//!
//! **A proof made under this knob does not verify, and must never be used as
//! a proof.** The probe exists to time the block's phases with the register
//! accesses moved into the CPU row, before that format is built:
//!
//! - the walk makes no MEMW_R row for a register access a CPU row makes
//!   itself (`rs1`, `rs2`, `rd` and the PC read of AUIPC / JAL), so those
//!   rows are never laid out, committed or argued. Word (`*W`) instructions,
//!   whose register accesses CPU32 makes, the ECALL chips' accesses, HALT's
//!   and every access whose step does not fit the range check keep today's
//!   routing: they stand in for the small table the format keeps beside the
//!   CPU;
//! - the CPU table takes the format's row: [`EXTRA_CPU_COLUMNS`] more
//!   columns (zero here: the previous timestamps and `rd`'s old value) and
//!   the format's 24 interactions (`tables::cpu::probe_interactions`) in place
//!   of today's 20, so its GKR slots and its committed cells are the
//!   format's.
//!
//! Nothing matches the moved accesses on any bus, so the block's buses do not
//! balance and its verifier refuses the proof: the probe measures cost, never
//! correctness. With the knob unset (or `0`) nothing here runs and every
//! proof is today's, byte for byte. Any word other than `0` or `1` is refused
//! ([`check`]), so a misspelt knob cannot time the control as the probe.

/// The knob.
pub const ENV: &str = "LAMBDA_VM_PROBE_REGS_SHAPE";

/// Columns the probe adds to the CPU table: `PREV_TS_1`, `PREV_TS_2`,
/// `PREV_TS_D`, `OLD_RD_0`, `OLD_RD_1` (left zero).
pub const EXTRA_CPU_COLUMNS: usize = 5;

/// The knob's word: unset, empty or `0` is off, `1` is on, anything else is
/// refused.
pub fn parse(word: Option<&str>) -> Result<bool, String> {
    match word.map(str::trim) {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!(
            "{ENV}={other}: 1 (the measurement-only register-shape probe, whose proofs do not \
             verify) or 0 / unset (off)"
        )),
    }
}

/// The knob, read once: `Err` for a word [`parse`] refuses.
fn reading() -> &'static Result<bool, String> {
    static READ: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    READ.get_or_init(|| {
        // The guests never read the environment: the probe is off there.
        #[cfg(target_arch = "riscv64")]
        {
            Ok(false)
        }
        #[cfg(not(target_arch = "riscv64"))]
        {
            parse(std::env::var(ENV).ok().as_deref())
        }
    })
}

/// Whether the probe is on. A refused word reads as off here; the block's
/// prove refuses it first ([`check`]).
pub fn on() -> bool {
    matches!(reading(), Ok(true))
}

/// The knob checked where a block's prove starts: `Err` for a refused word,
/// so a misspelt knob proves nothing.
pub fn check() -> Result<bool, crate::Error> {
    reading()
        .clone()
        .map_err(|e| crate::Error::Prover(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_knob_is_off_unless_its_word_is_one() {
        assert_eq!(parse(None), Ok(false));
        assert_eq!(parse(Some("")), Ok(false));
        assert_eq!(parse(Some("0")), Ok(false));
        assert_eq!(parse(Some(" 0 ")), Ok(false));
        assert_eq!(parse(Some("1")), Ok(true));
        for word in ["2", "on", "yes", "true", "01"] {
            assert!(parse(Some(word)).is_err(), "{word} is refused");
        }
    }

    /// The test process never sets the knob: every other test runs today's
    /// tables.
    #[test]
    fn the_test_process_runs_with_the_probe_off() {
        assert!(!on(), "{ENV} must be unset for the library's tests");
        assert_eq!(check().ok(), Some(false));
    }
}
