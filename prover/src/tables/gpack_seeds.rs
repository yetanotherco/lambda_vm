//! Width seeds for the finish's packed tables (on by default;
//! `LAMBDA_VM_FINISH_SEED_WIDTHS=0` is the rollback).
//!
//! A kind writes its trace straight into the packed form only once its
//! process-wide width hint exists ([`super::gpack::WidthHint`]); until then its
//! first tables — every one of them that starts before the first ends — are
//! built at 8 bytes a cell and packed after. These are the widths each kind
//! learned on the small, median and large blocks (the widest of the three, a
//! census at batch 8's head), seeded before the finish builds anything. A
//! seed too narrow for a block makes that table's fill run again, one too wide
//! narrows the column after: the packed bytes are the words' alone either way.
//! Each seed is `width x count` runs; a kind whose column count differs from its
//! seed's is left unseeded.

use super::gpack::WidthHint;

/// `(kind, columns, its hint, widths)`.
const SEEDS: &[(&str, usize, &WidthHint, &str)] = &[
    ("BRANCH", 14, &super::branch::WIDTHS, "4,1,4x4,1x8"),
    ("BYTEWISE", 26, &super::bytewise::WIDTHS, "1x26"),
    (
        "COPY",
        41,
        &super::copy::WIDTHS,
        "4,1x9,4x2,2x4,4x2,2x5,1x5,2x4,1x9",
    ),
    (
        "CPU",
        42,
        &super::cpu::WIDTHS_INROW,
        "4x2,1x7,4x2,1x10,4,1,4x2,1,4x6,2x4,1,4x5",
    ),
    (
        "CPU32",
        38,
        &super::cpu32::WIDTHS,
        "4,1,4,1x3,2x2,4,1,4,1x3,2x2,1x4,4,1,2x2,1x5,4x2,1x7",
    ),
    (
        "DVRM",
        34,
        &super::dvrm::WIDTHS,
        "2x8,1,2x8,1x2,4x4,2x4,1x7",
    ),
    (
        "ECDAS",
        521,
        &super::ecdas::WIDTHS,
        "4,1x260,8x63,1x34,8x5,2x57,1x35,8x16,2x32,8,2,8x12,1x4",
    ),
    (
        "ECREC",
        1543,
        &super::ecrec::WIDTHS,
        "4,1,4x6,1x166,8x62,1x290,8x62,1x290,8x62,1x98,8x62,1x35,8x15,2x39,8x8,1x194,8,1x7,2x144",
    ),
    ("EQ", 12, &super::eq::WIDTHS, "4x4,1x2,2x4,1x2"),
    (
        "KECCAK",
        907,
        &super::keccak::WIDTHS,
        "4,2,1x488,2,4x4,2x4,1x408",
    ),
    ("LOAD", 18, &super::load::WIDTHS, "4x3,1x15"),
    ("LT", 17, &super::lt::WIDTHS, "4,1x2,4,1x4,2x4,1x5"),
    (
        "MEMW",
        49,
        &super::memw::WIDTHS,
        "1,4x2,1x8,4,1x19,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1x3",
    ),
    (
        "MEMW_A",
        29,
        &super::memw_aligned::WIDTHS,
        "1,2x2,4,1x8,4,1x12,2x2,1x2",
    ),
    (
        "MEMW_R",
        12,
        &super::register_residual::WIDTHS,
        "1,4,1,4x4,2x2,1x3",
    ),
    (
        "MUL",
        26,
        &super::mul::WIDTHS,
        "2x4,1,2x4,1,2x8,1x2,8x4,1x2",
    ),
    (
        "SHIFT",
        29,
        &super::shift::WIDTHS,
        "2x4,1x4,4x2,1x3,2x9,1x5,2,4",
    ),
    ("STORE", 16, &super::store::WIDTHS, "4x3,1x13"),
];

/// `widths`'s runs, `w` or `wxn`, comma-separated; `None` if one does not parse.
fn parse(widths: &str) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    for run in widths.split(',') {
        let (w, n) = match run.split_once('x') {
            Some((w, n)) => (w.parse::<u8>().ok()?, n.parse::<usize>().ok()?),
            None => (run.parse::<u8>().ok()?, 1),
        };
        out.extend(std::iter::repeat_n(w, n));
    }
    Some(out)
}

/// Whether the finish seeds the width hints: on unless
/// `LAMBDA_VM_FINISH_SEED_WIDTHS=0`, read once a process.
pub(crate) fn seed_widths_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        seed_widths_from(
            std::env::var("LAMBDA_VM_FINISH_SEED_WIDTHS")
                .ok()
                .as_deref(),
        )
    })
}

/// [`seed_widths_on`]'s reading of `LAMBDA_VM_FINISH_SEED_WIDTHS`: unset or any
/// word but `0` seeds; `0` is the rollback.
fn seed_widths_from(value: Option<&str>) -> bool {
    value.is_none_or(|v| v.trim() != "0")
}

/// `LAMBDA_VM_FINISH_SEED_CPU=base`: CPU's seed goes to the base CPU table's
/// hint (`cpu::WIDTHS`, 37 columns), where widths of 42 columns are never
/// read; unset or any other word, to the in-row table's
/// (`cpu::WIDTHS_INROW`), which the block's CPU tables are written at. A
/// rollback for measurement.
pub(crate) fn cpu_seed_base() -> bool {
    static BASE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *BASE.get_or_init(|| {
        cpu_seed_base_from(std::env::var("LAMBDA_VM_FINISH_SEED_CPU").ok().as_deref())
    })
}

/// [`cpu_seed_base`]'s reading of `LAMBDA_VM_FINISH_SEED_CPU`.
fn cpu_seed_base_from(value: Option<&str>) -> bool {
    value.is_some_and(|v| v.trim() == "base")
}

/// A kind's seed hint: CPU's in-row one, or under [`cpu_seed_base`] the base
/// one.
fn seed_hint(name: &str, hint: &'static WidthHint) -> &'static WidthHint {
    if name == "CPU" && cpu_seed_base() {
        &super::cpu::WIDTHS
    } else {
        hint
    }
}

/// Seeds every kind's hint that has none yet, once a process; the kinds seeded.
pub(crate) fn seed_finish_widths() -> usize {
    static SEEDED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SEEDED.get_or_init(|| {
        let mut seeded = 0;
        for &(name, cols, hint, widths) in SEEDS {
            let hint = seed_hint(name, hint);
            if let Some(widths) = parse(widths).filter(|w| w.len() == cols)
                && !hint.known(cols)
            {
                hint.learn(&widths);
                seeded += 1;
            }
        }
        seeded
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every seed parses to its kind's column count, and the kinds' column
    /// counts are today's tables' (a seed for a changed layout must be
    /// re-taken, not silently skipped).
    #[test]
    fn every_seed_parses_to_its_kinds_columns() {
        let today = [
            ("BRANCH", super::super::branch::cols::NUM_COLUMNS),
            ("BYTEWISE", super::super::bytewise::cols::NUM_COLUMNS),
            ("COPY", super::super::copy::cols::NUM_COLUMNS),
            ("CPU", super::super::cpu::inrow_cols::NUM_COLUMNS_INROW),
            ("CPU32", super::super::cpu32::cols::NUM_COLUMNS),
            ("DVRM", super::super::dvrm::cols::NUM_COLUMNS),
            ("ECDAS", super::super::ecdas::cols::NUM_COLUMNS),
            ("ECREC", super::super::ecrec::cols::NUM_COLUMNS),
            ("EQ", super::super::eq::cols::NUM_COLUMNS),
            ("KECCAK", super::super::keccak::cols::NUM_COLUMNS),
            ("LOAD", super::super::load::cols::NUM_COLUMNS),
            ("LT", super::super::lt::cols::NUM_COLUMNS),
            ("MEMW", super::super::memw::cols::NUM_COLUMNS),
            ("MEMW_A", super::super::memw_aligned::cols::NUM_COLUMNS),
            ("MEMW_R", super::super::register_residual::cols::NUM_COLUMNS),
            ("MUL", super::super::mul::cols::NUM_COLUMNS),
            ("SHIFT", super::super::shift::cols::NUM_COLUMNS),
            ("STORE", super::super::store::cols::NUM_COLUMNS),
        ];
        assert_eq!(SEEDS.len(), today.len());
        for (&(name, cols, _, widths), &(kind, columns)) in SEEDS.iter().zip(&today) {
            assert_eq!(name, kind);
            assert_eq!(
                cols, columns,
                "{name}: the seed's column count is the table's"
            );
            let w = parse(widths).expect("the seed parses");
            assert_eq!(w.len(), cols, "{name}");
            assert!(
                w.iter().all(|w| [1, 2, 4, 8].contains(w)),
                "{name}: widths are 1, 2, 4 or 8"
            );
        }
    }

    /// The seeds are on unset and for any word but `0`, which rolls them back.
    #[test]
    fn seed_widths_are_on_by_default_and_0_rolls_them_back() {
        assert!(seed_widths_from(None));
        assert!(seed_widths_from(Some("1")));
        assert!(seed_widths_from(Some(" 1 ")));
        assert!(!seed_widths_from(Some("0")));
        assert!(!seed_widths_from(Some(" 0\n")));
    }

    /// CPU's seed is the in-row table's hint, the one the block's CPU tables
    /// are written at: its 42 widths are that table's columns; `base` alone
    /// sends it back to the base table's hint. Once seeded, the in-row hint
    /// holds the seed's widths.
    #[test]
    fn the_cpu_seed_goes_to_the_in_row_hint() {
        let &(_, cols, hint, _) = SEEDS
            .iter()
            .find(|&&(name, ..)| name == "CPU")
            .expect("a CPU seed");
        assert!(std::ptr::eq(hint, &super::super::cpu::WIDTHS_INROW));
        assert_eq!(cols, super::super::cpu::inrow_cols::NUM_COLUMNS_INROW);
        assert!(!cpu_seed_base_from(None));
        assert!(!cpu_seed_base_from(Some("inrow")));
        assert!(cpu_seed_base_from(Some(" base ")));
        seed_finish_widths();
        assert!(super::super::cpu::WIDTHS_INROW.known(cols));
    }
}
