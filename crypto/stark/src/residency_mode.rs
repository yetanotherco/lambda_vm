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
}

impl ResidencyMode {
    /// True when main LDEs are dropped after Round 1 and recomputed on demand.
    pub fn recomputes_main_lde(self) -> bool {
        matches!(self, Self::RecomputeLde)
    }
}
