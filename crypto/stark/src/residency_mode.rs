/// Whether Round 1 keeps every table's main LDE resident until its fused task
/// runs, or drops it after the commit and recomputes it inside the task.
///
/// Fiat-Shamir requires the main *roots* to be absorbed before the shared LogUp
/// challenges are sampled; it says nothing about the LDE buffers, so keeping
/// them is a performance choice. `Retain` makes it; `RecomputeLde` trades one
/// extra forward NTT per table for turning an `O(N)` retention into an
/// `O(table_parallelism)` transient. The Merkle tree is kept either way, so a
/// recompute never re-hashes and the root that entered the transcript stays the
/// root openings are checked against.
///
/// `RecomputeLdeDevice` is the same trade on the device, for a prove whose
/// tables do not fit the card together (a whole block in one proof): the
/// device commit's LDE, tree and trace snapshot are all freed once the root is
/// taken, and the table's fused task commits the trace again on the device.
/// That second commit re-hashes, so the prover checks its root against the one
/// Round 1 absorbed and refuses the proof when they differ.
///
/// The choice is invisible to the proof: same roots, same transcript order,
/// same proof bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResidencyMode {
    /// Keep every main LDE from its Round-1 commit until its table's fused
    /// task consumes it.
    #[default]
    Retain,
    /// Drop each main LDE once its root is absorbed and recompute it from the
    /// still-resident trace at the top of the table's fused task.
    ///
    /// Also releases each table's aux columns from the caller-owned
    /// `TraceTable` when that table's proof is complete — a documented part of
    /// this mode's contract, since it mutates caller-visible state. Callers
    /// that read a trace's aux columns after `multi_prove` returns must use
    /// `Retain`.
    RecomputeLde,
    /// Commit every table on the device in Round 1 and keep only its root: the
    /// device LDE, the device tree and the trace snapshot are freed before the
    /// next table is admitted, and no host LDE is downloaded. At the top of
    /// the table's fused task the trace is committed on the device again (LDE
    /// and tree), and the prover refuses the proof with
    /// `ProvingError::RecomputedCommitmentMismatch` unless the new root equals
    /// the absorbed one. From there the task runs exactly as under `Retain`.
    ///
    /// A table the device declines in Round 1 (below the LDE floor) is
    /// committed on the host with its full tree kept, and its LDE is recomputed
    /// on the host as under `RecomputeLde`. Releases the aux columns like
    /// `RecomputeLde`. Without the `cuda` feature it is `RecomputeLde`.
    RecomputeLdeDevice,
}

impl ResidencyMode {
    /// True when main LDEs are dropped after Round 1 and recomputed on demand.
    pub fn recomputes_main_lde(self) -> bool {
        matches!(self, Self::RecomputeLde | Self::RecomputeLdeDevice)
    }

    /// True when the Round-1 commit must run on the host, because the LDE it
    /// drops is recomputed there (`RecomputeLde`).
    pub fn recomputes_on_host(self) -> bool {
        matches!(self, Self::RecomputeLde)
    }

    /// True when a device-committed table is committed again on the device in
    /// its fused task (`RecomputeLdeDevice`).
    pub fn recommits_on_device(self) -> bool {
        matches!(self, Self::RecomputeLdeDevice)
    }
}

/// Tables committed a second time on the device under
/// [`ResidencyMode::RecomputeLdeDevice`], process-wide. A test that means to
/// exercise the device recommit reads it before and after, so a run where
/// every table fell to the host arm cannot pass as one that recommitted.
pub static DEVICE_RECOMMITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Test-only fault injection for the recommit's root check.
#[cfg(any(test, feature = "test-utils"))]
pub mod test_hooks {
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// `0` is off. `i + 1` adds one to the first main-trace cell of table `i`
    /// right before its device recommit, so the recommitted root differs from
    /// the absorbed one. Cleared by the recommit that fires it.
    pub static PERTURB_BEFORE_RECOMMIT: AtomicUsize = AtomicUsize::new(0);

    /// Arm [`PERTURB_BEFORE_RECOMMIT`] for table `idx`.
    pub fn perturb_before_recommit(idx: usize) {
        PERTURB_BEFORE_RECOMMIT.store(idx + 1, Ordering::SeqCst);
    }

    /// Whether table `idx` is the armed one; disarms the hook when it is.
    pub fn take_perturbation(idx: usize) -> bool {
        PERTURB_BEFORE_RECOMMIT
            .compare_exchange(idx + 1, 0, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }
}
