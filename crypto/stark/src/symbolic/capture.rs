//! Capture front-end: run a constraint's generic `evaluate` over the symbolic
//! fields and snapshot the recorded arena into a [`ConstraintProgram`].

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as GoldilocksExtension;
use math::field::goldilocks::GoldilocksField;

use crate::constraints::transition::TransitionConstraint;
use crate::table::TableView;

use super::ir::{ConstraintProgram, Dim, Op};
use super::sym_field::{SymExt, SymField, leaf_base, record_leaf, with_arena};

/// Capture a single algebraic transition constraint into a flat IR program.
///
/// Builds a symbolic `TableView<SymField, SymExt>` whose main cells are
/// `Var { main: true, offset: 0, row: 0, col }` leaves (1 step, 1 row,
/// `num_main_cols` columns; aux is empty for the minimal algebraic set), runs
/// `c.evaluate::<SymField, SymExt>(step)`, and records the returned id as the
/// single root.
///
/// `num_main_cols` must be at least one greater than any column index the
/// constraint reads.
pub fn capture_constraint<T>(c: &T, num_main_cols: usize) -> ConstraintProgram
where
    T: TransitionConstraint<GoldilocksField, GoldilocksExtension>,
{
    let (nodes, dims, root) = with_arena(|| {
        // Build the symbolic frame's single step: one row of `num_main_cols`
        // main cells, each a recorded leaf. Aux is empty.
        let row: Vec<FieldElement<SymField>> = (0..num_main_cols)
            .map(|col| {
                let id = record_leaf(
                    Op::Var {
                        main: true,
                        offset: 0,
                        row: 0,
                        col: col as u16,
                    },
                    Dim::D1,
                );
                leaf_base(id)
            })
            .collect();

        let step: TableView<SymField, SymExt> = TableView::new(vec![row], vec![Vec::new()]);

        // Run the real constraint body under the symbolic fields.
        let result = c.evaluate::<SymField, SymExt>(&step);
        *result.value()
    });

    ConstraintProgram {
        nodes,
        dims,
        roots: vec![root.id],
    }
}
