//! The argue's round trips made cheap (`LAMBDA_VM_ARGUE_ROUND_SLOT`).
//!
//! On — the default; `0` keeps today's round trips — a GKR layer's rounds on
//! the card take their sums from a slot of pinned host memory the round
//! kernel's last block writes (`math_cuda::round_slot`): no second launch to
//! reduce the block partials, no pageable read-back, a host spin on a sequence
//! word instead of a stream wait. And a bin's trees are built without waiting
//! for each one's output fraction: the outputs are read when the ladder
//! absorbs them. A table's fused zerocheck keeps today's round trips: through
//! a slot its launches ran slower on whole blocks.
//!
//! The sums are the limbs today's second launch writes, reduced by
//! `ext3_from_raw` as before, so the transcript and the proof bytes do not
//! move. `check` also runs today's second launch and read-back every round,
//! compares the two limb for limb, counts a difference and uses today's.

use std::sync::atomic::{AtomicU8, Ordering};

/// How a round's sums come back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundSlotSetting {
    /// Today's second launch and read-back.
    Off,
    /// Through a slot. The default.
    On,
    /// Through a slot, and today's beside it, compared.
    Check,
}

/// The setting `LAMBDA_VM_ARGUE_ROUND_SLOT` names: `0` off, unset, empty or
/// `1` on, `check` on and compared. Anything else is on, said once — `wait`,
/// the next round queued behind a card-side wait, is not built.
pub fn round_slot_setting(value: Option<&str>) -> RoundSlotSetting {
    match value {
        Some("0") => RoundSlotSetting::Off,
        None | Some("") | Some("1") => RoundSlotSetting::On,
        Some("check") => RoundSlotSetting::Check,
        Some(other) => {
            eprintln!(
                "[argue] LAMBDA_VM_ARGUE_ROUND_SLOT={other} is not 0, 1 or check; reading the GKR rounds' \
                 sums from a slot (the default)"
            );
            RoundSlotSetting::On
        }
    }
}

static FORCED: AtomicU8 = AtomicU8::new(0);

/// The setting in effect: a test's override, else the environment's, read
/// once with a banner.
pub fn argue_round_slot() -> RoundSlotSetting {
    match FORCED.load(Ordering::Relaxed) {
        1 => RoundSlotSetting::Off,
        2 => RoundSlotSetting::On,
        3 => RoundSlotSetting::Check,
        _ => {
            static SET: std::sync::OnceLock<RoundSlotSetting> = std::sync::OnceLock::new();
            *SET.get_or_init(|| {
                let set = round_slot_setting(
                    std::env::var("LAMBDA_VM_ARGUE_ROUND_SLOT").ok().as_deref(),
                );
                eprintln!(
                    "★ ARGUE ROUND SLOT: {}",
                    match set {
                        RoundSlotSetting::Off => {
                            "off (LAMBDA_VM_ARGUE_ROUND_SLOT=0: today's second launch and read-back every round)"
                        }
                        RoundSlotSetting::On => {
                            "on (the default; the GKR rounds' sums from a slot the round kernel's last block \
                             writes, a bin's trees built without a wait each; =0 restores today's, =check \
                             compares them)"
                        }
                        RoundSlotSetting::Check => {
                            "check (the rounds' sums from a slot and today's read-back, compared every round)"
                        }
                    }
                );
                set
            })
        }
    }
}

/// Overrides `LAMBDA_VM_ARGUE_ROUND_SLOT` for the whole process — for a test
/// that proves every way in one binary. `None` restores the environment's.
#[doc(hidden)]
pub fn force_argue_round_slot(setting: Option<RoundSlotSetting>) {
    FORCED.store(
        match setting {
            None => 0,
            Some(RoundSlotSetting::Off) => 1,
            Some(RoundSlotSetting::On) => 2,
            Some(RoundSlotSetting::Check) => 3,
        },
        Ordering::Relaxed,
    );
}

/// The setting in effect, for a harness's line.
pub fn round_slot_in_effect() -> &'static str {
    match argue_round_slot() {
        RoundSlotSetting::Off => "off",
        RoundSlotSetting::On => "on",
        RoundSlotSetting::Check => "check",
    }
}

/// The setting as the card's sessions take it.
#[cfg(feature = "cuda")]
pub(crate) fn device_setting() -> math_cuda::round_slot::RoundSlot {
    match argue_round_slot() {
        RoundSlotSetting::Off => math_cuda::round_slot::RoundSlot::Off,
        RoundSlotSetting::On => math_cuda::round_slot::RoundSlot::On,
        RoundSlotSetting::Check => math_cuda::round_slot::RoundSlot::Check,
    }
}

/// What the slots did this process, for a harness's line after its label.
pub fn counts_line() -> String {
    #[cfg(feature = "cuda")]
    let c = math_cuda::round_slot::counts();
    #[cfg(feature = "cuda")]
    let v = [
        c.slot_rounds,
        c.fall_backs,
        c.declines,
        c.second_launches,
        c.checked,
        c.mismatches,
    ];
    #[cfg(not(feature = "cuda"))]
    let v = [0u64; 6];
    format_counts(v)
}

fn format_counts(v: [u64; 6]) -> String {
    format!(
        "rounds through a slot {} · stream-wait fall-backs {} · declines {} · second launches {} · check \
         compared {} · mismatches {}",
        v[0], v[1], v[2], v[3], v[4], v[5]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_knob_reads_its_variable() {
        assert_eq!(round_slot_setting(None), RoundSlotSetting::On);
        assert_eq!(round_slot_setting(Some("")), RoundSlotSetting::On);
        assert_eq!(round_slot_setting(Some("0")), RoundSlotSetting::Off);
        assert_eq!(round_slot_setting(Some("1")), RoundSlotSetting::On);
        assert_eq!(round_slot_setting(Some("check")), RoundSlotSetting::Check);
        assert_eq!(
            round_slot_setting(Some("wait")),
            RoundSlotSetting::On,
            "the queued round is not built"
        );
        assert_eq!(round_slot_setting(Some("on")), RoundSlotSetting::On);
    }

    /// ★ The default reads the GKR rounds' sums from a slot — the same proof
    /// bytes, which the card tests show — and only `0` keeps today's round
    /// trips.
    #[test]
    fn the_knob_is_on_unless_its_variable_is_zero() {
        assert_eq!(
            argue_round_slot(),
            round_slot_setting(std::env::var("LAMBDA_VM_ARGUE_ROUND_SLOT").ok().as_deref())
        );
        if std::env::var_os("LAMBDA_VM_ARGUE_ROUND_SLOT").is_none() {
            assert_eq!(argue_round_slot(), RoundSlotSetting::On);
            assert_eq!(round_slot_in_effect(), "on");
        }
    }

    #[test]
    fn the_counts_name_every_field_on_their_line() {
        assert_eq!(
            format_counts([1, 2, 3, 4, 5, 6]),
            "rounds through a slot 1 · stream-wait fall-backs 2 · declines 3 · second launches 4 · check \
             compared 5 · mismatches 6"
        );
    }
}
