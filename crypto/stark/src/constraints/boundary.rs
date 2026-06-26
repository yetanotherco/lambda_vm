use alloc::vec::Vec;

use math::field::{element::FieldElement, traits::IsField};

/// Represents a boundary constraint that must hold in an execution trace:
///   * col: The column of the trace where the constraint must hold
///   * step: The step (or row) of the trace where the constraint must hold
///   * value: The value the constraint must have in that column and step
#[derive(Debug)]
pub struct BoundaryConstraint<F: IsField> {
    pub col: usize,
    pub step: usize,
    pub value: FieldElement<F>,
    pub is_aux: bool,
}

impl<F: IsField> BoundaryConstraint<F> {
    pub fn new_main(col: usize, step: usize, value: FieldElement<F>) -> Self {
        Self {
            col,
            step,
            value,
            is_aux: false,
        }
    }

    pub fn new_aux(col: usize, step: usize, value: FieldElement<F>) -> Self {
        Self {
            col,
            step,
            value,
            is_aux: true,
        }
    }

    /// Boundary constraint for a trace with a single main column.
    pub fn new_simple_main(step: usize, value: FieldElement<F>) -> Self {
        Self {
            col: 0,
            step,
            value,
            is_aux: false,
        }
    }
}

/// All the boundary constraints that must hold for an execution trace.
#[derive(Default, Debug)]
pub struct BoundaryConstraints<F: IsField> {
    pub constraints: Vec<BoundaryConstraint<F>>,
}

impl<F: IsField> BoundaryConstraints<F> {
    /// Instantiate from a vector of `BoundaryConstraint` elements.
    pub fn from_constraints(constraints: Vec<BoundaryConstraint<F>>) -> Self {
        Self { constraints }
    }
}
