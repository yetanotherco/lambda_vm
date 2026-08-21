//! What a batched epoch proof carries.
//!
//! These types live here and NOT in [`crate::proof`] on purpose: the per-table
//! `StarkProof` / `MultiProof` rkyv layouts are the production wire format, and
//! the batched path is opt-in. Keeping its types in this module makes the
//! default path byte-identical by construction rather than by test —
//! `git diff <base>..HEAD -- crypto/stark/src/proof/` stays empty.
//!
//! # What is NOT here, and why
//!
//! No per-table Merkle roots, no per-table FRI layer roots, no per-table query
//! list. One epoch commits four mixed-height MMCS roots (preprocessed, main,
//! aux, composition parts) and runs ONE FRI instance, so a query costs one
//! authentication path per round instead of one per table per round. That is the
//! proof-size win; everything per-table that survives is data the verifier
//! cannot derive — OOD evaluations, bus sums, public inputs.

use math::field::element::FieldElement;
use math::field::traits::IsField;

use crate::config::Commitment;
use crate::fri::fri_decommit::FriDecommitment;
use crate::fri::mmcs::MixedOpening;
use crate::lookup::BusPublicInputs;
use crate::proof::stark::PolynomialOpenings;
use crate::table::Table;

/// The per-table data a batched epoch proof still has to carry.
#[derive(Debug, Clone)]
pub struct BatchedTableData<E: IsField, PI> {
    /// This table's interpolation-domain size. The verifier derives the table's
    /// height — and therefore every index reduction — from this, so it is bound
    /// into the round-4 shape histogram before any challenge depends on it.
    pub trace_length: usize,
    /// tⱼ(z·gᵏ): the current-row block (all columns).
    pub trace_ood_evaluations: Table<E>,
    /// tⱼ(z·gᵏ): the pruned next-row block (masked columns only).
    pub trace_ood_next_evaluations: Table<E>,
    /// Hᵢ(z^N).
    pub composition_poly_parts_ood_evaluation: Vec<FieldElement<E>>,
    /// LogUp bus sums, when the table has a RAP.
    pub bus_public_inputs: Option<BusPublicInputs<E>>,
    /// Public inputs for the boundary constraints.
    pub public_inputs: PI,
    /// A table excluded from the batched FRI class keeps a terminal-only
    /// instance of its own: its DEEP codeword IS its terminal codeword, sent as
    /// the `2^(h - blowup_log)` coefficients of the polynomial it evaluates.
    /// `None` for a table in the batched class.
    ///
    /// See [`crate::fri::batched::FriInstancePlan`]: the partition is derived
    /// from the shape by both sides and is never sent, so this field's presence
    /// is checked against the derived plan rather than trusted.
    pub standalone_final_poly_coeffs: Option<Vec<FieldElement<E>>>,
}

/// One query's openings: one authentication path per batched round, plus the
/// FRI layer decommitment.
#[derive(Debug, Clone)]
pub struct BatchedQueryOpening<F: IsField, E: IsField> {
    /// Preprocessed openings, ONE PER PREPROCESSED TABLE in AIR order — each a
    /// standard row-pair opening against that table's own precomputed tree.
    ///
    /// ★ Deliberately NOT a round of the mixed MMCS (this is #768's
    /// arrangement, kept for the same reason): the per-table precomputed trees
    /// are exactly the ones `air.precomputed_commitment()` pins, so the
    /// verifier absorbs and compares roots it already owns — the per-table
    /// path's critical soundness check, verbatim — and a recursive verifier
    /// binds each root with the provenance machinery that already exists
    /// (interned constant / derived in-machine / ELF-attested). A fused
    /// mixed-height prep root has no in-machine binding story: its provenance
    /// classes are mixed into one digest, which is the M-8 blocker this layout
    /// dissolves. Empty when the epoch has no preprocessed table.
    pub prep: Vec<PolynomialOpenings<F>>,
    /// Main round — always present; every table contributes a matrix.
    pub main: MixedOpening<F>,
    /// Auxiliary round. `None` when no table has a RAP.
    pub aux: Option<MixedOpening<E>>,
    /// Composition-parts round — always present.
    pub parts: MixedOpening<E>,
    /// The batched FRI instance's per-layer openings for this query.
    pub fri: FriDecommitment<E>,
}

/// One epoch, one proof.
#[derive(Debug, Clone)]
pub struct BatchedMultiProof<F: IsField, E: IsField, PI> {
    pub tables: Vec<BatchedTableData<E, PI>>,
    /// ★ There is deliberately NO `prep_root` here. Preprocessed matrices are
    /// committed per table and their roots are `air.precomputed_commitment()`
    /// — absorbed by both sides FROM THE AIR SET, never from the proof,
    /// exactly as the per-table path's Phase A does. The proof carries only
    /// the per-query openings ([`BatchedQueryOpening::prep`]).
    pub main_root: Commitment,
    pub aux_root: Option<Commitment>,
    pub parts_root: Commitment,
    /// The batched FRI instance's committed layer roots.
    pub fri_layer_roots: Vec<Commitment>,
    /// The batched FRI instance's terminal polynomial.
    pub fri_final_poly_coeffs: Vec<FieldElement<E>>,
    pub nonce: Option<u64>,
    pub queries: Vec<BatchedQueryOpening<F, E>>,
}

/// What the batched prove cost in residency and in recomputation.
///
/// Returned rather than logged because it is the number the campaign's
/// projection is missing (MMCS-PLAN §1.1 prices the commitment work but not the
/// LDE rebuilds the phase barriers force). A test can assert on it, which is
/// what keeps "the batched builder does not hold every table's LDE" falsifiable
/// at the PROVER level instead of only at the primitive's.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BatchedProveStats {
    /// Highest number of bytes of MAIN and AUX LDE simultaneously alive across
    /// the whole prove, counted as the buffers are created and dropped.
    ///
    /// ★ This is the number the acceptance test asserts on, and the reason it is
    /// reported separately from the parts. Under `ResidencyMode::RecomputeLde`
    /// it must be flat in the table count — bounded by the widest single table,
    /// not by the epoch. If the streaming builder were bypassed, or if any phase
    /// quietly retained what it read, this would grow with `N` instead, which is
    /// precisely the failure MMCS-PLAN §3.3 warns gives the win back inside the
    /// same commit.
    pub peak_trace_lde_bytes: usize,
    /// Bytes of composition parts held at the peak. These are `O(N)` BY DESIGN —
    /// recomputing them is a second constraint evaluation — so they are counted
    /// apart from the trace LDEs rather than allowed to mask their behaviour.
    pub retained_parts_bytes: usize,
    /// The two above at the moment either was highest. Reported for budgeting;
    /// the falsifiable claim lives in `peak_trace_lde_bytes`.
    pub peak_lde_bytes: usize,
    /// How many times a main LDE was expanded from a trace. One per table is the
    /// floor (the commit itself); every phase barrier that follows costs another
    /// forward NTT per table under `RecomputeLde`.
    pub main_lde_expansions: usize,
    /// Same for the auxiliary LDE.
    pub aux_lde_expansions: usize,
    /// How many times a table's composition parts were computed. Recomputing
    /// these means re-running constraint evaluation, so the batched prover
    /// retains them instead; this counter exists to make that visible if it ever
    /// stops being true.
    pub parts_computations: usize,
    /// Wall clock per phase, indices 0..6 = phases 1..6 (main commit, aux
    /// commit, composition parts, OOD, DEEP+FRI, openings). A latency
    /// breakdown of the whole prove: the six entries plus the pre-phase
    /// setup sum to the call's wall time. Returned in the stats — not logged —
    /// for the same reason the residency numbers are: the A/B harness prints
    /// the struct, so every box run carries its own phase profile.
    pub phase_wall: [core::time::Duration; 6],
    /// Wall clock spent inside LDE expansions (main and aux, all phases) —
    /// the recompute traffic itself, separated from what the phases do with
    /// the buffers. Under `RecomputeLde` this is the price of the residency
    /// mode; under `Retain` it is the floor (one main + one aux per table).
    pub lde_expansion_wall: core::time::Duration,
}

/// Running account of live LDE bytes, so [`BatchedProveStats`] reports what the
/// prover actually held rather than what its comments claim.
#[derive(Debug, Default)]
pub(crate) struct ResidencyLedger {
    live: usize,
    peak: usize,
}

impl ResidencyLedger {
    pub(crate) fn alloc(&mut self, bytes: usize) {
        self.live += bytes;
        self.peak = self.peak.max(self.live);
    }

    pub(crate) fn free(&mut self, bytes: usize) {
        self.live = self.live.saturating_sub(bytes);
    }

    pub(crate) fn peak(&self) -> usize {
        self.peak
    }
}

/// Bytes a row-major LDE buffer of `len` field elements occupies.
pub(crate) fn lde_bytes<E: IsField>(len: usize) -> usize {
    len * core::mem::size_of::<FieldElement<E>>()
}
