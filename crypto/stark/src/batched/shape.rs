//! The epoch's committed shape — which table contributes a matrix to which
//! batched round, at what height and width.
//!
//! Every number here is derived from the AIR set and the per-table trace
//! lengths, never read out of a proof. That is what lets the verifier rebuild
//! the shape it must pass to [`crate::fri::mmcs::MixedMmcs::verify_batch`] and
//! to [`crate::fri::batched::absorb_shape_histogram`] instead of trusting the
//! prover's word for it (`fri/mmcs.rs`, "Width binding").
//!
//! # Why one type and not four lists
//!
//! Four rounds are batched (preprocessed, main, aux, composition parts) and each
//! has a DIFFERENT participation list: only preprocessed tables contribute a
//! preprocessed matrix, only tables with a RAP contribute an aux matrix. The
//! index a matrix has inside its round is therefore NOT its table index, and the
//! two are easy to confuse — a confusion that shows up as an opening
//! authenticated at the wrong leaf rather than as a compile error. [`RoundShape`]
//! keeps the mapping in one place so both sides read it from the same code.

use crate::traits::AIR;

/// Which tables contribute a matrix to one batched round, and with what shape.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct RoundShape {
    /// Contributing table indices, ascending. Position `i` in this vector is
    /// matrix `i` of the round — the order the MMCS concatenates leaves in, and
    /// the order openings are presented in.
    pub tables: Vec<usize>,
    /// `(log_height, width)` per contributing matrix, in `tables` order.
    pub dims: Vec<(usize, usize)>,
}

impl RoundShape {
    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    pub fn heights(&self) -> Vec<usize> {
        self.dims.iter().map(|(h, _)| *h).collect()
    }

    pub fn widths(&self) -> Vec<usize> {
        self.dims.iter().map(|(_, w)| *w).collect()
    }

    /// The round's own tallest matrix — the index space
    /// [`crate::fri::mmcs::MixedMmcs::verify_batch`] accepts. `None` for an
    /// empty round.
    pub fn h_max(&self) -> Option<usize> {
        self.dims.iter().map(|(h, _)| *h).max()
    }
}

/// The shape of every batched round in one epoch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpochShape {
    /// `log2` of each table's LDE length, in table order. This is the FRI's
    /// shape: query indices live in the tallest of these domains.
    pub heights: Vec<usize>,
    /// Preprocessed columns. Empty when no table is preprocessed.
    pub prep: RoundShape,
    /// Main trace columns — every table. For a preprocessed table this is the
    /// MULTIPLICITY columns only, matching the per-table path's split
    /// (`commit_main_trace`: `[0, num_precomputed)` is the preprocessed matrix,
    /// `[num_precomputed, total)` the committed main one).
    pub main: RoundShape,
    /// Auxiliary (RAP) columns. Empty when no table has a RAP.
    pub aux: RoundShape,
    /// Composition-polynomial parts — every table.
    pub parts: RoundShape,
}

/// Why an epoch cannot be proved (or verified) with one batched instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    /// No tables at all.
    Empty,
    /// A table's LDE length is not a power of two, is 1, or overflows a shift.
    /// Heights come from proof-supplied trace lengths on the verifier's side, so
    /// this is a rejection, never a panic.
    BadHeight { table: usize, lde_size: usize },
    /// A table declares zero committed main columns, so it has no matrix to
    /// contribute and no leaf to open.
    NoMainColumns { table: usize },
    /// The batched path commits ONE FRI instance for the whole epoch, so every
    /// table must agree on the parameters that instance is defined by. The
    /// per-table path has no such requirement, which is exactly why this is
    /// checked rather than assumed.
    MixedProofOptions { table: usize, field: &'static str },
}

impl core::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ShapeError::Empty => write!(f, "an epoch needs at least one table"),
            ShapeError::BadHeight { table, lde_size } => write!(
                f,
                "table {table}: LDE length {lde_size} is not a power of two greater than 1"
            ),
            ShapeError::NoMainColumns { table } => {
                write!(f, "table {table} commits no main columns")
            }
            ShapeError::MixedProofOptions { table, field } => write!(
                f,
                "table {table} disagrees with table 0 on `{field}`; one batched FRI \
                 instance needs one set of parameters"
            ),
        }
    }
}

/// The epoch-wide FRI parameters, once every table has been checked to agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpochFriParams {
    pub blowup_log: u32,
    pub coset_offset: u64,
    pub grinding_factor: u8,
    pub num_queries: usize,
    pub final_poly_log_degree: u32,
}

impl EpochShape {
    /// Derive the shape from the AIR set and each table's interpolation-domain
    /// size (`trace_length`).
    ///
    /// The prover passes the trace lengths it is about to prove; the verifier
    /// passes the ones the proof declares. Both then hold the same `EpochShape`
    /// without either having read it from the other.
    pub fn derive<F, E, PI>(
        airs: &[&dyn AIR<Field = F, FieldExtension = E, PublicInputs = PI>],
        trace_lengths: &[usize],
    ) -> Result<(Self, EpochFriParams), ShapeError>
    where
        F: math::field::traits::IsFFTField
            + math::field::traits::IsSubFieldOf<E>
            + Send
            + Sync
            + 'static,
        E: math::field::traits::IsField + Send + Sync + 'static,
    {
        if airs.is_empty() || airs.len() != trace_lengths.len() {
            return Err(ShapeError::Empty);
        }

        let first = airs[0].options();
        let params = EpochFriParams {
            blowup_log: (first.blowup_factor as usize).trailing_zeros(),
            coset_offset: first.coset_offset,
            grinding_factor: first.grinding_factor,
            num_queries: first.fri_number_of_queries,
            final_poly_log_degree: first.fri_final_poly_log_degree as u32,
        };

        let mut heights = Vec::with_capacity(airs.len());
        let mut prep = RoundShape::default();
        let mut main = RoundShape::default();
        let mut aux = RoundShape::default();
        let mut parts = RoundShape::default();

        for (table, (air, &trace_length)) in airs.iter().zip(trace_lengths).enumerate() {
            let options = air.options();
            for (field, same) in [
                ("blowup_factor", options.blowup_factor == first.blowup_factor),
                ("coset_offset", options.coset_offset == first.coset_offset),
                (
                    "grinding_factor",
                    options.grinding_factor == first.grinding_factor,
                ),
                (
                    "fri_number_of_queries",
                    options.fri_number_of_queries == first.fri_number_of_queries,
                ),
                (
                    "fri_final_poly_log_degree",
                    options.fri_final_poly_log_degree == first.fri_final_poly_log_degree,
                ),
            ] {
                if !same {
                    return Err(ShapeError::MixedProofOptions { table, field });
                }
            }

            let lde_size = trace_length
                .checked_mul(options.blowup_factor as usize)
                .ok_or(ShapeError::BadHeight {
                    table,
                    lde_size: usize::MAX,
                })?;
            if !lde_size.is_power_of_two() || lde_size < 2 || lde_size.trailing_zeros() >= u32::BITS
            {
                return Err(ShapeError::BadHeight { table, lde_size });
            }
            let h = lde_size.trailing_zeros() as usize;
            heights.push(h);

            let (total_main_cols, aux_cols) = air.trace_layout();
            let num_precomputed = if air.is_preprocessed() {
                air.num_precomputed_columns()
            } else {
                0
            };
            let committed_main = total_main_cols
                .checked_sub(num_precomputed)
                .ok_or(ShapeError::NoMainColumns { table })?;
            if committed_main == 0 {
                return Err(ShapeError::NoMainColumns { table });
            }

            if num_precomputed > 0 {
                prep.tables.push(table);
                prep.dims.push((h, num_precomputed));
            }
            main.tables.push(table);
            main.dims.push((h, committed_main));
            if aux_cols > 0 && air.has_aux_trace() {
                aux.tables.push(table);
                aux.dims.push((h, aux_cols));
            }
            let num_parts = air.composition_poly_degree_bound(trace_length) / trace_length;
            parts.tables.push(table);
            parts.dims.push((h, num_parts.max(1)));
        }

        Ok((
            Self {
                heights,
                prep,
                main,
                aux,
                parts,
            },
            params,
        ))
    }

    /// The epoch's tallest LDE — the domain query indices are drawn in.
    pub fn h_max(&self) -> usize {
        self.heights.iter().copied().max().unwrap_or(0)
    }

    /// The widths the round-4 shape histogram binds: one per table, in table
    /// order, summing every matrix that table contributes across all four rounds.
    ///
    /// Summing rather than listing per round is deliberate. The histogram's job
    /// is to make two epochs with different shapes produce different challenges,
    /// and `absorb_shape_histogram` takes one `(height, width)` pair per entry.
    /// A table's total committed width moves whenever ANY of its four matrices
    /// changes width, so the sum separates exactly the epochs the four separate
    /// lists would — while staying one entry per table, which is what keeps the
    /// prover's and the verifier's histograms the same length without either
    /// having to agree on a round ordering.
    pub fn total_widths(&self) -> Vec<usize> {
        let mut widths = vec![0usize; self.heights.len()];
        for round in [&self.prep, &self.main, &self.aux, &self.parts] {
            for (&table, (_, w)) in round.tables.iter().zip(round.dims.iter()) {
                widths[table] += *w;
            }
        }
        widths
    }
}
