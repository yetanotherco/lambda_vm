//! CPU_BITWISE chip constraints (Phase 2 step 8 — option α).
//!
//! The CPU_BITWISE chip handles AND/OR/XOR rows (and their `*W` variants).
//! It reuses the CPU column layout, which carries both byte cells
//! (`ARG1[0..7]`, etc.) and u32 limbs (`ARG1_LO/HI`, etc.). On these rows two
//! representations need to agree:
//!
//! - The bitwise lookup (`BusId::Bitwise`) sources individual bytes from
//!   `ARG1[i]`, `ARG2[i]`, `RES[i]`, so the byte cells determine the AND/OR/
//!   XOR result that gets attested via the lookup table.
//! - The register-write path (`RVD` -> M5 sender) reads the u32 limbs
//!   (`RES_LO/HI`), so the u32 limbs determine which value lands in `rd`.
//!
//! Without a packing constraint a malicious prover could populate bytes and
//! u32 limbs inconsistently — bitwise lookup passes, but a different value
//! gets written to the destination register. We close the gap with six linear
//! packing constraints, one per u32 limb:
//!
//! ```text
//! ARG1_LO = ARG1[0] + 2^8*ARG1[1] + 2^16*ARG1[2] + 2^24*ARG1[3]
//! ARG1_HI = ARG1[4] + 2^8*ARG1[5] + 2^16*ARG1[6] + 2^24*ARG1[7]
//! ARG2_LO = ARG2[0] + 2^8*ARG2[1] + 2^16*ARG2[2] + 2^24*ARG2[3]
//! ARG2_HI = ARG2[4] + 2^8*ARG2[5] + 2^16*ARG2[6] + 2^24*ARG2[7]
//! RES_LO  = RES[0]  + 2^8*RES[1]  + 2^16*RES[2]  + 2^24*RES[3]
//! RES_HI  = RES[4]  + 2^8*RES[5]  + 2^16*RES[6]  + 2^24*RES[7]
//! ```
//!
//! All six constraints are degree 1 and unconditional on the CPU_BITWISE
//! table (every non-padding row is a bitwise op, padding rows are all zeros
//! so each equation collapses to `0 - 0 = 0`).

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use stark::constraints::transition::{TransitionConstraint, TransitionConstraintEvaluator};
use stark::table::TableView;

use crate::tables::cpu::cols;
use crate::tables::types::{GoldilocksExtension, GoldilocksField};

/// Packing constraint: `limb_col = byte_cols[0] + 2^8*byte_cols[1] +
/// 2^16*byte_cols[2] + 2^24*byte_cols[3]`.
pub struct BitwisePackConstraint {
    /// u32 limb column being packed into.
    limb_col: usize,
    /// Index of the first of four consecutive byte columns.
    byte_start: usize,
    constraint_idx: usize,
}

impl BitwisePackConstraint {
    pub fn new(limb_col: usize, byte_start: usize, constraint_idx: usize) -> Self {
        Self {
            limb_col,
            byte_start,
            constraint_idx,
        }
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let limb = step.get_main_evaluation_element(0, self.limb_col).clone();
        let b0 = step.get_main_evaluation_element(0, self.byte_start).clone();
        let b1 = step
            .get_main_evaluation_element(0, self.byte_start + 1)
            .clone();
        let b2 = step
            .get_main_evaluation_element(0, self.byte_start + 2)
            .clone();
        let b3 = step
            .get_main_evaluation_element(0, self.byte_start + 3)
            .clone();

        let coef_1: FieldElement<F> = FieldElement::from(256u64);
        let coef_2: FieldElement<F> = FieldElement::from(65536u64);
        let coef_3: FieldElement<F> = FieldElement::from(16777216u64);

        // limb - (b0 + 256*b1 + 65536*b2 + 16777216*b3) = 0
        limb - (b0 + coef_1 * b1 + coef_2 * b2 + coef_3 * b3)
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for BitwisePackConstraint {
    fn degree(&self) -> usize {
        1
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn evaluate<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        self.compute(step)
    }
}

/// Total number of CPU_BITWISE constraints.
pub const NUM_CPU_BITWISE_CONSTRAINTS: usize = 6;

/// Builds the six byte→u32 packing constraints for the CPU_BITWISE chip.
pub fn create_all_cpu_bitwise_constraints()
-> Vec<Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>> {
    let entries: [(usize, usize); 6] = [
        (cols::ARG1_LO, cols::ARG1[0]),
        (cols::ARG1_HI, cols::ARG1[4]),
        (cols::ARG2_LO, cols::ARG2[0]),
        (cols::ARG2_HI, cols::ARG2[4]),
        (cols::RES_LO, cols::RES[0]),
        (cols::RES_HI, cols::RES[4]),
    ];

    entries
        .into_iter()
        .enumerate()
        .map(|(i, (limb_col, byte_start))| {
            BitwisePackConstraint::new(limb_col, byte_start, i).boxed()
        })
        .collect()
}
