//! A fraction tree's input layer kept on the card without its padding
//! (`LAMBDA_VM_ARGUE_INPUT_REAL`).
//!
//! # What the padding is, and where it sits
//!
//! A table's input layer is a fraction `p/q` for every interaction and row.
//! The interactions are rounded up to a power of two with the fraction `0/1`,
//! and the interaction slot is the high bits of a cell's index, so the padding
//! is the tail of the layer: `[interactions·rows, slots·rows)`. Fewer than
//! half the slots are padding — the count is rounded *up* — so the tail lies in
//! the upper halves `p_hi`, `q_hi` alone. On today's path it is written, folded
//! into the level above and read by the layer's own rounds like any other
//! cell: about a third of the input cells of a block.
//!
//! # What the knob changes
//!
//! On — the default; `0` writes the padding as before — a tree whose input
//! layer has padding writes only the real fractions,
//! folds the level above from them with the padding read as `0/1` (the fold a
//! tree that hands its layer back already runs), and **keeps** them. Its Gruen
//! rounds then read a cell past the stored run as `0/1` and write no folded
//! cell past it — a fold of two padding cells is the padding again — until the
//! rounds have bound the slot bits and the stored run covers the whole half
//! (`math_cuda::gkr::GruenLayer`). The constants are the limbs the padded
//! layer is written with, so every value the card reads, folds and sums is
//! raw-identical to today's: the same round sums, the same halves, the same
//! transcript and proof bytes.
//!
//! Today's layer session (`LAMBDA_VM_ARGUE_GKR_GRUEN=0`) reads every cell of
//! the halves: for it the tree writes the whole layer again, as a tree that
//! handed its layer back does.
//!
//! `check` also writes each kept input layer whole beside it, where the card
//! has room, runs today's Gruen rounds over that one in step, and compares
//! every round's sums and the halves handed to the host, limb for limb. A
//! layer that differs is counted and finished on the whole layer, so the proof
//! is today's either way.

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

/// How a tree keeps its input layer on the card.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputReal {
    /// With its padding, as before.
    Off,
    /// Without it. The default.
    On,
    /// Without it, and compared round by round with the whole layer.
    Check,
}

/// The setting `LAMBDA_VM_ARGUE_INPUT_REAL` names: `0` off, unset, empty or
/// `1` on, `check` on and compared. Anything else is on, said once.
pub fn input_real_setting(value: Option<&str>) -> InputReal {
    match value {
        Some("0") => InputReal::Off,
        None | Some("") | Some("1") => InputReal::On,
        Some("check") => InputReal::Check,
        Some(other) => {
            eprintln!(
                "[gkr] LAMBDA_VM_ARGUE_INPUT_REAL={other} is not 0, 1 or check; keeping each input layer \
                 without its padding (the default)"
            );
            InputReal::On
        }
    }
}

static FORCED: AtomicU8 = AtomicU8::new(0);

/// The setting in effect: a test's override, else the environment's, read
/// once with a banner.
pub fn argue_input_real() -> InputReal {
    match FORCED.load(Ordering::Relaxed) {
        1 => InputReal::Off,
        2 => InputReal::On,
        3 => InputReal::Check,
        _ => {
            static SET: std::sync::OnceLock<InputReal> = std::sync::OnceLock::new();
            *SET.get_or_init(|| {
                let set = input_real_setting(
                    std::env::var("LAMBDA_VM_ARGUE_INPUT_REAL").ok().as_deref(),
                );
                eprintln!(
                    "★ ARGUE INPUT REAL: {}",
                    match set {
                        InputReal::Off => "off (each input layer written with its padding)",
                        InputReal::On => {
                            "on (the default: each input layer kept without its padding; \
                             LAMBDA_VM_ARGUE_INPUT_REAL=0 writes the padding, =check also compares with the \
                             whole layer)"
                        }
                        InputReal::Check => {
                            "check (each input layer kept without its padding and compared round by round \
                             with the whole layer)"
                        }
                    }
                );
                set
            })
        }
    }
}

/// Overrides `LAMBDA_VM_ARGUE_INPUT_REAL` for the whole process — for a test
/// that proves every way in one binary. `None` restores the environment's.
#[doc(hidden)]
pub fn force_argue_input_real(setting: Option<InputReal>) {
    FORCED.store(
        match setting {
            None => 0,
            Some(InputReal::Off) => 1,
            Some(InputReal::On) => 2,
            Some(InputReal::Check) => 3,
        },
        Ordering::Relaxed,
    );
}

/// The setting in effect, for a harness's line.
pub fn input_real_in_effect() -> &'static str {
    match argue_input_real() {
        InputReal::Off => "off",
        InputReal::On => "on",
        InputReal::Check => "check",
    }
}

static PADDED_TREES: AtomicU64 = AtomicU64::new(0);
static KEPT_TREES: AtomicU64 = AtomicU64::new(0);
static CELLS_NOT_STORED: AtomicU64 = AtomicU64::new(0);
static DECLINES: AtomicU64 = AtomicU64::new(0);
static REBUILDS: AtomicU64 = AtomicU64::new(0);
static CHECKED: AtomicU64 = AtomicU64::new(0);
static DIFFERING: AtomicU64 = AtomicU64::new(0);
static CHECK_SKIPPED: AtomicU64 = AtomicU64::new(0);

/// What the knob did this process, for a harness's line.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// Device trees whose input layer has padding, whatever the setting.
    pub padded_trees: u64,
    /// Of those, the ones kept without it.
    pub kept_trees: u64,
    /// The padding cells those trees did not store.
    pub cells_not_stored: u64,
    /// Trees the knob asked for that the card would not promise, or failed to
    /// build: each took today's path instead.
    pub declines: u64,
    /// Kept input layers written whole again for today's layer session.
    pub rebuilds: u64,
    /// Under `check`: kept layers compared with the whole layer to the end.
    pub checked: u64,
    /// Of the layers compared, the ones that differed (finished on the whole
    /// layer).
    pub differing: u64,
    /// Under `check`: kept layers the card had no room to compare.
    pub check_skipped: u64,
}

pub fn counts() -> Counts {
    Counts {
        padded_trees: PADDED_TREES.load(Ordering::Relaxed),
        kept_trees: KEPT_TREES.load(Ordering::Relaxed),
        cells_not_stored: CELLS_NOT_STORED.load(Ordering::Relaxed),
        declines: DECLINES.load(Ordering::Relaxed),
        rebuilds: REBUILDS.load(Ordering::Relaxed),
        checked: CHECKED.load(Ordering::Relaxed),
        differing: DIFFERING.load(Ordering::Relaxed),
        check_skipped: CHECK_SKIPPED.load(Ordering::Relaxed),
    }
}

impl Counts {
    /// The harness's line after its label.
    pub fn line(&self) -> String {
        format!(
            "padded trees {} · kept without padding {} · cells not stored {} · declines {} · rewritten for the \
             old rounds {} · check compared {} · differing {} · skipped {}",
            self.padded_trees,
            self.kept_trees,
            self.cells_not_stored,
            self.declines,
            self.rebuilds,
            self.checked,
            self.differing,
            self.check_skipped
        )
    }
}

/// Zeroes every count, for a harness that reads one prove from a reused
/// process.
pub fn reset_counts() {
    for counter in [
        &PADDED_TREES,
        &KEPT_TREES,
        &CELLS_NOT_STORED,
        &DECLINES,
        &REBUILDS,
        &CHECKED,
        &DIFFERING,
        &CHECK_SKIPPED,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}

// The device driver's (`gpu::input_layer_tree_impl`, `gpu::GruenSession`): a
// build without a device reaches only the knob and the tests.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn note_padded_tree() {
    PADDED_TREES.fetch_add(1, Ordering::Relaxed);
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn note_kept(cells_not_stored: u64) {
    KEPT_TREES.fetch_add(1, Ordering::Relaxed);
    CELLS_NOT_STORED.fetch_add(cells_not_stored, Ordering::Relaxed);
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn note_decline() {
    DECLINES.fetch_add(1, Ordering::Relaxed);
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn note_rebuild() {
    REBUILDS.fetch_add(1, Ordering::Relaxed);
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn note_checked(differed: bool) {
    CHECKED.fetch_add(1, Ordering::Relaxed);
    if differed {
        DIFFERING.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn note_check_skipped() {
    CHECK_SKIPPED.fetch_add(1, Ordering::Relaxed);
}

thread_local! {
    /// ⛔ A FAULT: while armed on a thread, a checked layer's first round's
    /// sums from the kept layer get one added to a limb before they are
    /// compared — what a kernel that read one cell wrong would hand back — so a
    /// test can show the comparison sees it, counts it and finishes on the
    /// whole layer. Per thread: a test's fault must not reach a proof running
    /// beside it.
    static CHECK_FAULT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[doc(hidden)]
pub fn force_check_fault(on: bool) {
    CHECK_FAULT.with(|cell| cell.set(on));
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn check_fault() -> bool {
    CHECK_FAULT.with(std::cell::Cell::get)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_knob_reads_its_variable() {
        assert_eq!(input_real_setting(None), InputReal::On);
        assert_eq!(input_real_setting(Some("")), InputReal::On);
        assert_eq!(input_real_setting(Some("0")), InputReal::Off);
        assert_eq!(input_real_setting(Some("1")), InputReal::On);
        assert_eq!(input_real_setting(Some("check")), InputReal::Check);
        assert_eq!(input_real_setting(Some("on")), InputReal::On);
        assert_eq!(input_real_setting(Some("2")), InputReal::On);
    }

    /// ★ The default keeps each input layer without its padding — the same
    /// proof bytes, which the card tests show — and only `0` writes it.
    #[test]
    fn the_knob_is_on_unless_its_variable_is_zero() {
        assert_eq!(
            argue_input_real(),
            input_real_setting(std::env::var("LAMBDA_VM_ARGUE_INPUT_REAL").ok().as_deref())
        );
        if std::env::var_os("LAMBDA_VM_ARGUE_INPUT_REAL").is_none() {
            assert_eq!(argue_input_real(), InputReal::On);
            assert_eq!(input_real_in_effect(), "on");
        }
    }

    #[test]
    fn the_counts_name_every_field_on_their_line() {
        let counts = Counts {
            padded_trees: 1,
            kept_trees: 2,
            cells_not_stored: 3,
            declines: 4,
            rebuilds: 5,
            checked: 6,
            differing: 7,
            check_skipped: 8,
        };
        assert_eq!(
            counts.line(),
            "padded trees 1 · kept without padding 2 · cells not stored 3 · declines 4 · rewritten for the old \
             rounds 5 · check compared 6 · differing 7 · skipped 8"
        );
    }
}
