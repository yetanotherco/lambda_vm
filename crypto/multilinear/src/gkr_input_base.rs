//! Keep eligible GKR input numerators in the base field until the first bind.

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

static FORCED: AtomicU8 = AtomicU8::new(0);
static STEPS: AtomicU64 = AtomicU64::new(0);

fn setting(value: Option<&str>) -> bool {
    value == Some("1")
}

/// `LAMBDA_VM_GKR_INPUT_BASE_NUM=1` requests eight-byte input numerators.
pub fn enabled() -> bool {
    match FORCED.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| {
                setting(
                    std::env::var("LAMBDA_VM_GKR_INPUT_BASE_NUM")
                        .ok()
                        .as_deref(),
                )
            })
        }
    }
}

/// Overrides the setting for integration tests.
#[doc(hidden)]
pub fn force(value: Option<bool>) {
    FORCED.store(
        value.map_or(0, |on| if on { 1 } else { 2 }),
        Ordering::Relaxed,
    );
}

/// Number of two-layer sessions that opened with base numerators.
pub fn steps() -> u64 {
    STEPS.load(Ordering::Relaxed)
}

pub(crate) fn reset_steps() {
    STEPS.store(0, Ordering::Relaxed);
}

#[cfg(feature = "cuda")]
pub(crate) fn note_step() {
    STEPS.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::setting;

    #[test]
    fn only_one_enables_base_numerators() {
        assert!(setting(Some("1")));
        for value in [None, Some(""), Some("0"), Some("on"), Some("true")] {
            assert!(!setting(value));
        }
    }
}
