//! Optional per-stage host↔device byte counters for profiling.
//!
//! Enabled by setting the `GPU_STAGE_BYTES` env var to any non-empty value
//! other than `0`. When disabled, every `add_*` call is a single relaxed
//! atomic load plus a branch, so the counters can be left compiled into the
//! `cuda` build without measurable cost.
//!
//! The counters attribute the round-2..4 host↔device round-trips the GPU
//! profiling found to dominate transfer, so composition-resident vs
//! FRI-resident can be prioritized with byte counts rather than inference:
//!   * `comp_dh_d2h`     — D2H of the composition evaluations `d_h` per table
//!   * `comp_h01_h2d`    — H2D of decomposed H₀/H₁ as input to the LDE extend
//!   * `comp_h01_lde_d2h`— D2H of the extended H₀/H₁ LDE result back to host
//!   * `comp_merkle_h2d` — H2D re-upload of those parts for the Merkle commit
//!   * `comp_deep_h2d`   — H2D re-upload of composition parts for DEEP fallback
//!   * `deep_out_d2h`    — D2H of the completed DEEP codeword before FRI
//!   * `fri_initial_h2d` — H2D of that codeword when GPU FRI starts
//!   * `fri_layer_d2h`   — D2H of every FRI layer's evaluations
//!   * `query_gather`    — D2H of Merkle paths during query openings

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

static COMP_DH_D2H: AtomicU64 = AtomicU64::new(0);
static COMP_H01_H2D: AtomicU64 = AtomicU64::new(0);
static COMP_H01_LDE_D2H: AtomicU64 = AtomicU64::new(0);
static COMP_MERKLE_H2D: AtomicU64 = AtomicU64::new(0);
static COMP_DEEP_H2D: AtomicU64 = AtomicU64::new(0);
static DEEP_OUT_D2H: AtomicU64 = AtomicU64::new(0);
static FRI_INITIAL_H2D: AtomicU64 = AtomicU64::new(0);
static FRI_LAYER_D2H: AtomicU64 = AtomicU64::new(0);
static QUERY_GATHER: AtomicU64 = AtomicU64::new(0);

/// Cached once per process; the env var is read a single time.
pub fn enabled() -> bool {
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| {
        std::env::var("GPU_STAGE_BYTES")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
    })
}

#[inline]
fn add(counter: &AtomicU64, bytes: usize) {
    if enabled() {
        counter.fetch_add(bytes as u64, Ordering::Relaxed);
    }
}

pub fn add_comp_dh_d2h(bytes: usize) {
    add(&COMP_DH_D2H, bytes);
}
pub fn add_comp_h01_h2d(bytes: usize) {
    add(&COMP_H01_H2D, bytes);
}
pub fn add_comp_h01_lde_d2h(bytes: usize) {
    add(&COMP_H01_LDE_D2H, bytes);
}
pub fn add_comp_merkle_h2d(bytes: usize) {
    add(&COMP_MERKLE_H2D, bytes);
}
pub fn add_comp_deep_h2d(bytes: usize) {
    add(&COMP_DEEP_H2D, bytes);
}
pub fn add_deep_out_d2h(bytes: usize) {
    add(&DEEP_OUT_D2H, bytes);
}
pub fn add_fri_initial_h2d(bytes: usize) {
    add(&FRI_INITIAL_H2D, bytes);
}
pub fn add_fri_layer_d2h(bytes: usize) {
    add(&FRI_LAYER_D2H, bytes);
}
pub fn add_query_gather(bytes: usize) {
    add(&QUERY_GATHER, bytes);
}

/// Zero every counter (call before a measured prove).
pub fn reset() {
    for c in [
        &COMP_DH_D2H,
        &COMP_H01_H2D,
        &COMP_H01_LDE_D2H,
        &COMP_MERKLE_H2D,
        &COMP_DEEP_H2D,
        &DEEP_OUT_D2H,
        &FRI_INITIAL_H2D,
        &FRI_LAYER_D2H,
        &QUERY_GATHER,
    ] {
        c.store(0, Ordering::Relaxed);
    }
}

/// Formatted summary, or `None` if counting is disabled. Composition rows are
/// grouped so the composition-vs-FRI split is read directly.
pub fn report() -> Option<String> {
    if !enabled() {
        return None;
    }
    let mb = |c: &AtomicU64| c.load(Ordering::Relaxed) as f64 / 1e6;
    let comp = mb(&COMP_DH_D2H)
        + mb(&COMP_H01_H2D)
        + mb(&COMP_H01_LDE_D2H)
        + mb(&COMP_MERKLE_H2D)
        + mb(&COMP_DEEP_H2D);
    let deep_fri_bridge = mb(&DEEP_OUT_D2H) + mb(&FRI_INITIAL_H2D);
    let fri = mb(&FRI_LAYER_D2H);
    let q = mb(&QUERY_GATHER);
    let mut s = String::from("GPU stage bytes (host<->device, MB):\n");
    s.push_str(&format!(
        "  composition d_h D2H       {:>10.1}\n",
        mb(&COMP_DH_D2H)
    ));
    s.push_str(&format!(
        "  composition H0/H1 in  H2D {:>10.1}\n",
        mb(&COMP_H01_H2D)
    ));
    s.push_str(&format!(
        "  composition H0/H1 LDE D2H {:>10.1}\n",
        mb(&COMP_H01_LDE_D2H)
    ));
    s.push_str(&format!(
        "  composition Merkle    H2D {:>10.1}\n",
        mb(&COMP_MERKLE_H2D)
    ));
    s.push_str(&format!(
        "  composition DEEP      H2D {:>10.1}\n",
        mb(&COMP_DEEP_H2D)
    ));
    s.push_str(&format!("  composition SUBTOTAL      {:>10.1}\n", comp));
    s.push_str(&format!(
        "  DEEP output           D2H {:>10.1}\n",
        mb(&DEEP_OUT_D2H)
    ));
    s.push_str(&format!(
        "  FRI initial codeword  H2D {:>10.1}\n",
        mb(&FRI_INITIAL_H2D)
    ));
    s.push_str(&format!(
        "  DEEP->FRI bridge TOTAL    {:>10.1}\n",
        deep_fri_bridge
    ));
    s.push_str(&format!("  FRI layer evals D2H       {:>10.1}\n", fri));
    s.push_str(&format!("  query gather D2H          {:>10.1}\n", q));
    s.push_str(&format!(
        "  TOTAL (counted)           {:>10.1}\n",
        comp + deep_fri_bridge + fri + q
    ));
    Some(s)
}
