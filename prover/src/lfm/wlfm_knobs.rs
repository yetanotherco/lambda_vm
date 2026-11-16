//! The W-LFM prover's knobs read from the environment. They choose how the
//! prove holds its tables, never what the proof is; [`super::wlfm`] holds the
//! format and reads no environment.

/// `LAMBDA_VM_WLFM_NARROW_CHIPS=1`: the LFM chips' traces packed narrow
/// ([`multilinear::narrow::NarrowColumns`]) before the prove takes them, so
/// their columns go up at the bytes their words need and the card widens
/// them, as the lean tables' do. Off by default; a prover's choice, never the
/// proof's: the card widens the same words. Read once a process.
pub const NARROW_CHIPS_ENV: &str = "LAMBDA_VM_WLFM_NARROW_CHIPS";

/// Whether the prove packs the LFM chips narrow: [`NARROW_CHIPS_ENV`], or a
/// test's override.
pub fn narrow_chips() -> bool {
    #[cfg(test)]
    if let Some(on) = super::wlfm::test_format::narrow_chips() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var(NARROW_CHIPS_ENV).is_ok_and(|v| v.trim() == "1"))
}
