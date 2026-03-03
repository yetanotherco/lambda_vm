//! COMMIT (ECALL) table for writing bytes to stdout.
//!
//! This table handles the `write` syscall (ECALL #3): writing bytes from a memory
//! buffer to stdout. It uses a **recursive design** — each row commits one byte,
//! and rows are linked via a self-referencing "CommitNextByte" bus.
//!
//! Only the first row of each commit sequence receives from the CPU's ECALL bus;
//! subsequent rows receive from the previous commit row via the CommitNextByte bus.
//!
//! ## Columns
//! - `timestamp`: DWordWL (2 cols) — timestamp of the ECALL
//! - `address`: DWordWL (2 cols) — current buffer address
//! - `count`: DWordWL (2 cols) — remaining byte count
//! - `first`: Bit — first row in a commit sequence
//! - `end`: Bit — last row (count was 0)
//! - `mu`: Bit — multiplicity (1 for real rows, 0 for padding)
//! - `value`: Byte — the byte being committed
//! - `index`: DWordWL (2 cols) — global commit index
//! - `address_incr`: DWordWL (2 cols) — address + 1
//! - `count_decr`: DWordHL (4 cols) — count - 1 as 4 halfwords (or all 0xFFFF when count=0)
//!
//! ## Bus Interactions
//! - **Receiver**: EcallCommit bus — receives `[timestamp_lo, timestamp_hi]` from CPU (mult = first)
//! - **Sender**: CommitNextByte bus — sends to next row (mult = mu - end)
//! - **Receiver**: CommitNextByte bus — receives from prev row (mult = mu - first)
//! - **Sender**: IsHalfword bus — range checks for count_decr halfwords (×4, mult = mu)
//! - **Sender**: IsByte bus — range check for value (mult = mu)
//!
//! ## Constraints
//! - `range_first`: first * (1 - first) = 0
//! - `range_end`: end * (1 - end) = 0
//! - `range_mu`: mu * (1 - mu) = 0
//! - `first_or_end_implies_mu`: (first + end - first*end) * (1 - mu) = 0
//! - `end_detection`: end * ((65535 - count_decr_0) + (65535 - count_decr_1)
//!   + (65535 - count_decr_2) + (65535 - count_decr_3)) = 0
//!
//! ## Memory Interactions (deferred to Phase 2)
//! Register reads (x10, x11, x12, x254), memory byte reads, address_incr ADD bus,
//! and commit output verification are all deferred.

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

// =========================================================================
// Column indices for COMMIT table
// =========================================================================

/// Column definitions for the COMMIT table.
pub mod cols {
    // Timestamp (DWordWL: 2 cols)
    /// timestamp[0]: low 32 bits
    pub const TIMESTAMP_0: usize = 0;
    /// timestamp[1]: high 32 bits
    pub const TIMESTAMP_1: usize = 1;

    // Buffer address (DWordWL: 2 cols)
    /// address[0]: low 32 bits
    pub const ADDRESS_0: usize = 2;
    /// address[1]: high 32 bits
    pub const ADDRESS_1: usize = 3;

    // Remaining byte count (DWordWL: 2 cols)
    /// count[0]: low 32 bits
    pub const COUNT_0: usize = 4;
    /// count[1]: high 32 bits
    pub const COUNT_1: usize = 5;

    // Control bits
    /// first: 1 if this is the first row of a commit sequence
    pub const FIRST: usize = 6;
    /// end: 1 if this is the last row (count was 0)
    pub const END: usize = 7;
    /// mu: multiplicity bit (1 for real rows, 0 for padding)
    pub const MU: usize = 8;

    // Byte value being committed
    /// value: the byte [0, 256) being committed at this row
    pub const VALUE: usize = 9;

    // Global commit index (DWordWL: 2 cols)
    /// index[0]: low 32 bits of global commit index
    pub const INDEX_0: usize = 10;
    /// index[1]: high 32 bits of global commit index
    pub const INDEX_1: usize = 11;

    // address + 1 result (DWordWL: 2 cols)
    /// address_incr[0]: low 32 bits of (address + 1)
    pub const ADDRESS_INCR_0: usize = 12;
    /// address_incr[1]: high 32 bits of (address + 1)
    pub const ADDRESS_INCR_1: usize = 13;

    // count - 1 result (DWordHL: 4 halfword cols)
    // When count > 0: count_decr = count - 1, decomposed into 4 halfwords
    // When count = 0: count_decr = 0xFFFF_FFFF_FFFF_FFFF (all halfwords = 0xFFFF)
    /// count_decr[0]: halfword 0 (bits 0-15)
    pub const COUNT_DECR_0: usize = 14;
    /// count_decr[1]: halfword 1 (bits 16-31)
    pub const COUNT_DECR_1: usize = 15;
    /// count_decr[2]: halfword 2 (bits 32-47)
    pub const COUNT_DECR_2: usize = 16;
    /// count_decr[3]: halfword 3 (bits 48-63)
    pub const COUNT_DECR_3: usize = 17;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 18;
}

// =========================================================================
// Operation type
// =========================================================================

/// A single row in the COMMIT table.
///
/// Each row represents one byte being committed from a buffer. Rows are linked
/// via the CommitNextByte bus to form a chain for each commit ECALL.
#[derive(Debug, Clone)]
pub struct CommitOperation {
    /// Timestamp of the originating ECALL
    pub timestamp: u64,
    /// Current buffer address for this byte
    pub address: u64,
    /// Remaining byte count (including this byte, 0 on end row)
    pub count: u64,
    /// Whether this is the first row of a commit sequence
    pub first: bool,
    /// Whether this is the end row (count was 0, no byte committed)
    pub end: bool,
    /// The byte value being committed (0 on end row)
    pub value: u8,
    /// Global commit index (accumulated across all ECALLs)
    pub index: u64,
}

// =========================================================================
// Trace generation
// =========================================================================

/// Generates the COMMIT trace table from a list of operations.
///
/// Each operation becomes one row. The table is padded to the next power of 2 (min 4).
/// Padding rows have all zeros (first=0, end=0, mu=0).
pub fn generate_commit_trace(
    ops: &[CommitOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let n = ops.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (row_idx, op) in ops.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;

        // Timestamp (DWordWL)
        data[base + cols::TIMESTAMP_0] = FE::from(op.timestamp & 0xFFFF_FFFF);
        data[base + cols::TIMESTAMP_1] = FE::from(op.timestamp >> 32);

        // Address (DWordWL)
        data[base + cols::ADDRESS_0] = FE::from(op.address & 0xFFFF_FFFF);
        data[base + cols::ADDRESS_1] = FE::from(op.address >> 32);

        // Count (DWordWL)
        data[base + cols::COUNT_0] = FE::from(op.count & 0xFFFF_FFFF);
        data[base + cols::COUNT_1] = FE::from(op.count >> 32);

        // Control bits
        data[base + cols::FIRST] = FE::from(op.first as u64);
        data[base + cols::END] = FE::from(op.end as u64);
        // mu = 1 for all real rows (first, middle, and end rows)
        data[base + cols::MU] = FE::one();

        // Value
        data[base + cols::VALUE] = FE::from(op.value as u64);

        // Index (DWordWL)
        data[base + cols::INDEX_0] = FE::from(op.index & 0xFFFF_FFFF);
        data[base + cols::INDEX_1] = FE::from(op.index >> 32);

        // address_incr = address + 1 (wrapping)
        let address_incr = op.address.wrapping_add(1);
        data[base + cols::ADDRESS_INCR_0] = FE::from(address_incr & 0xFFFF_FFFF);
        data[base + cols::ADDRESS_INCR_1] = FE::from(address_incr >> 32);

        // count_decr: if count == 0, use 0xFFFF_FFFF_FFFF_FFFF; else count - 1
        let count_decr = if op.count == 0 {
            u64::MAX
        } else {
            op.count - 1
        };
        data[base + cols::COUNT_DECR_0] = FE::from(count_decr & 0xFFFF);
        data[base + cols::COUNT_DECR_1] = FE::from((count_decr >> 16) & 0xFFFF);
        data[base + cols::COUNT_DECR_2] = FE::from((count_decr >> 32) & 0xFFFF);
        data[base + cols::COUNT_DECR_3] = FE::from((count_decr >> 48) & 0xFFFF);
    }

    // Padding rows are already zero (first=0, end=0, mu=0)

    TraceTable::new_main(data, cols::NUM_COLUMNS, num_rows)
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Creates all bus interactions for the COMMIT table.
///
/// The COMMIT table:
/// - **Receives** EcallCommit from CPU with `[timestamp_lo, timestamp_hi]` (mult = first)
/// - **Sends** to CommitNextByte with `[timestamp, address_incr, count_decr]` (mult = mu - end)
/// - **Receives** from CommitNextByte with `[timestamp, address, count]` (mult = mu - first)
/// - **Sends** to IsHalfword for count_decr range checks (×4, mult = mu)
/// - **Sends** to IsByte for value range check (mult = mu)
///
/// Memory interactions (register reads, byte reads, address_incr ADD bus, commit output)
/// are deferred to Phase 2.
pub fn bus_interactions() -> Vec<BusInteraction> {
    vec![
        // 1. Receive ECALL from CPU (mult = first)
        // Only the first row of each commit sequence receives from the CPU.
        BusInteraction::receiver(
            BusId::EcallCommit,
            Multiplicity::Column(cols::FIRST),
            vec![
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
            ],
        ),
        // 2. Send to CommitNextByte (mult = mu - end)
        // Non-end rows send their successor's expected values.
        // Sends: [timestamp, address_incr, count_decr]
        BusInteraction::sender(
            BusId::CommitNextByte,
            Multiplicity::Linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::MU,
                },
                LinearTerm::Column {
                    coefficient: -1,
                    column: cols::END,
                },
            ]),
            vec![
                // timestamp (DWordWL: 2 Direct elements)
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                // address_incr (DWordWL: 2 Direct elements)
                BusValue::Packed {
                    start_column: cols::ADDRESS_INCR_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::ADDRESS_INCR_1,
                    packing: Packing::Direct,
                },
                // count_decr (DWordHL: 4 halfwords → 2 bus elements)
                BusValue::Packed {
                    start_column: cols::COUNT_DECR_0,
                    packing: Packing::DWordHL,
                },
            ],
        ),
        // 3. Receive from CommitNextByte (mult = mu - first)
        // Non-first rows receive their values from the previous row's send.
        // Receives: [timestamp, address, count] — must match sender's format
        BusInteraction::receiver(
            BusId::CommitNextByte,
            Multiplicity::Linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::MU,
                },
                LinearTerm::Column {
                    coefficient: -1,
                    column: cols::FIRST,
                },
            ]),
            vec![
                // timestamp (DWordWL: 2 Direct elements)
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                // address (DWordWL: 2 Direct elements)
                BusValue::Packed {
                    start_column: cols::ADDRESS_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::ADDRESS_1,
                    packing: Packing::Direct,
                },
                // count (DWordWL: 2 Direct → 2 bus elements)
                // DWordWL produces same 2 bus elements as DWordHL when values match
                BusValue::Packed {
                    start_column: cols::COUNT_0,
                    packing: Packing::DWordWL,
                },
            ],
        ),
        // 4. Range checks: IsHalfword for count_decr components (×4, mult = mu)
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: cols::COUNT_DECR_0,
                packing: Packing::Direct,
            }],
        ),
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: cols::COUNT_DECR_1,
                packing: Packing::Direct,
            }],
        ),
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: cols::COUNT_DECR_2,
                packing: Packing::Direct,
            }],
        ),
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: cols::COUNT_DECR_3,
                packing: Packing::Direct,
            }],
        ),
        // 5. IsByte for value (mult = mu)
        BusInteraction::sender(
            BusId::IsByte,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: cols::VALUE,
                packing: Packing::Direct,
            }],
        ),
    ]
}

// =========================================================================
// Constraints
// =========================================================================

/// Creates all constraints for the COMMIT table.
///
/// Returns the constraint objects and the next available constraint index.
///
/// Constraints:
/// 0. `range_first`: first * (1 - first) = 0
/// 1. `range_end`: end * (1 - end) = 0
/// 2. `range_mu`: mu * (1 - mu) = 0
/// 3. `first_or_end_implies_mu`: (first + end - first*end) * (1 - mu) = 0
/// 4. `end_detection`: end * ((65535 - count_decr_0) + (65535 - count_decr_1)
///   + (65535 - count_decr_2) + (65535 - count_decr_3)) = 0
pub fn create_constraints(constraint_idx_start: usize) -> (Vec<CommitConstraint>, usize) {
    let constraints = vec![
        CommitConstraint {
            kind: CommitConstraintKind::RangeFirst,
            constraint_idx: constraint_idx_start,
        },
        CommitConstraint {
            kind: CommitConstraintKind::RangeEnd,
            constraint_idx: constraint_idx_start + 1,
        },
        CommitConstraint {
            kind: CommitConstraintKind::RangeMu,
            constraint_idx: constraint_idx_start + 2,
        },
        CommitConstraint {
            kind: CommitConstraintKind::FirstOrEndImpliesMu,
            constraint_idx: constraint_idx_start + 3,
        },
        CommitConstraint {
            kind: CommitConstraintKind::EndDetection,
            constraint_idx: constraint_idx_start + 4,
        },
    ];
    let next_idx = constraint_idx_start + constraints.len();
    (constraints, next_idx)
}

/// The kind of COMMIT constraint.
#[derive(Debug, Clone, Copy)]
enum CommitConstraintKind {
    /// first * (1 - first) = 0
    RangeFirst,
    /// end * (1 - end) = 0
    RangeEnd,
    /// mu * (1 - mu) = 0
    RangeMu,
    /// (first + end - first*end) * (1 - mu) = 0
    FirstOrEndImpliesMu,
    /// end * ((65535 - count_decr_0) + (65535 - count_decr_1) + (65535 - count_decr_2) + (65535 - count_decr_3)) = 0
    EndDetection,
}

/// A constraint for the COMMIT table.
pub struct CommitConstraint {
    kind: CommitConstraintKind,
    constraint_idx: usize,
}

impl CommitConstraint {
    fn compute<F, E>(
        &self,
        step: &stark::table::TableView<F, E>,
    ) -> math::field::element::FieldElement<F>
    where
        F: math::field::traits::IsSubFieldOf<E>,
        E: math::field::traits::IsField,
    {
        let one = math::field::element::FieldElement::<F>::one();

        match self.kind {
            CommitConstraintKind::RangeFirst => {
                let first = step.get_main_evaluation_element(0, cols::FIRST).clone();
                // first * (1 - first)
                &first * (&one - &first)
            }
            CommitConstraintKind::RangeEnd => {
                let end = step.get_main_evaluation_element(0, cols::END).clone();
                // end * (1 - end)
                &end * (&one - &end)
            }
            CommitConstraintKind::RangeMu => {
                let mu = step.get_main_evaluation_element(0, cols::MU).clone();
                // mu * (1 - mu)
                &mu * (&one - &mu)
            }
            CommitConstraintKind::FirstOrEndImpliesMu => {
                let first = step.get_main_evaluation_element(0, cols::FIRST).clone();
                let end = step.get_main_evaluation_element(0, cols::END).clone();
                let mu = step.get_main_evaluation_element(0, cols::MU).clone();
                // (first + end - first*end) * (1 - mu)
                let first_or_end = &first + &end - &first * &end;
                first_or_end * (one - mu)
            }
            CommitConstraintKind::EndDetection => {
                let end = step.get_main_evaluation_element(0, cols::END).clone();
                let c0 = step
                    .get_main_evaluation_element(0, cols::COUNT_DECR_0)
                    .clone();
                let c1 = step
                    .get_main_evaluation_element(0, cols::COUNT_DECR_1)
                    .clone();
                let c2 = step
                    .get_main_evaluation_element(0, cols::COUNT_DECR_2)
                    .clone();
                let c3 = step
                    .get_main_evaluation_element(0, cols::COUNT_DECR_3)
                    .clone();
                let max_half = math::field::element::FieldElement::<F>::from(65535u64);
                // end * ((65535 - c0) + (65535 - c1) + (65535 - c2) + (65535 - c3))
                let sum =
                    (&max_half - &c0) + (&max_half - &c1) + (&max_half - &c2) + (max_half - c3);
                end * sum
            }
        }
    }
}

use math::field::element::FieldElement;
use stark::constraints::transition::TransitionConstraint;
use stark::traits::TransitionEvaluationContext;

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for CommitConstraint {
    fn degree(&self) -> usize {
        match self.kind {
            CommitConstraintKind::RangeFirst
            | CommitConstraintKind::RangeEnd
            | CommitConstraintKind::RangeMu
            | CommitConstraintKind::EndDetection => 2,
            CommitConstraintKind::FirstOrEndImpliesMu => 3,
        }
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        0
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<GoldilocksField, GoldilocksExtension>,
        transition_evaluations: &mut [FieldElement<GoldilocksExtension>],
    ) {
        match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values: _,
                rap_challenges: _,
                ..
            } => {
                let constraint_value = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = constraint_value.to_extension();
            }

            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _,
                rap_challenges: _,
                ..
            } => {
                let constraint_value = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = constraint_value;
            }
        }
    }
}
