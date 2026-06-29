//! Differential tests for the symbolic-field constraint capture spike.
//!
//! For each algebraic transition constraint, capture it into a flat IR via the
//! symbolic recording fields, then assert that interpreting the IR reproduces
//! the constraint's real `evaluate::<GoldilocksField, GoldilocksExtension>`
//! bit-for-bit over many random main rows.

use crate::constraints::cpu::ProductZeroConstraint;
use crate::constraints::templates::{AddConstraint, AddOperand, IsBitConstraint};
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};

use math::field::element::FieldElement;
use stark::constraints::transition::TransitionConstraint;
use stark::symbolic::{capture_constraint, eval_program_base};
use stark::table::TableView;

/// Number of random trials per constraint.
const TRIALS: usize = 1000;

/// Column count for the symbolic frame; larger than any column index read by
/// the constraints under test (CPU columns go up to 37).
const NUM_COLS: usize = 64;

/// A tiny deterministic SplitMix64 PRNG so the test needs no `rand` dependency
/// and is fully reproducible.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Run the differential check: capture `c`, then for `TRIALS` random rows
/// compare the real `evaluate` against the IR interpreter, bit-for-bit.
fn assert_ir_matches_evaluate<T>(c: &T, label: &str)
where
    T: TransitionConstraint<GoldilocksField, GoldilocksExtension>,
{
    let prog = capture_constraint(c, NUM_COLS);
    eprintln!("[{label}] captured {} IR nodes", prog.len());

    let mut rng = SplitMix64::new(0xDEAD_BEEF_CAFE_F00D ^ (label.len() as u64));

    for trial in 0..TRIALS {
        // Build a random main row.
        let row: Vec<FE> = (0..NUM_COLS).map(|_| FE::from(rng.next_u64())).collect();

        // Real evaluate: wrap the row in a base/ext TableView (1 row, no aux).
        let real_step: TableView<GoldilocksField, GoldilocksExtension> =
            TableView::new(vec![row.clone()], vec![Vec::new()]);
        let real: FieldElement<GoldilocksField> =
            c.evaluate::<GoldilocksField, GoldilocksExtension>(&real_step);

        // IR interpreter over the same row.
        let got = eval_program_base(&prog, &row);

        assert_eq!(
            real, got,
            "[{label}] mismatch at trial {trial}: real={real:?} got={got:?}"
        );
    }
}

#[test]
fn test_ir_matches_is_bit_unconditional() {
    // X * (1 - X), X at column 7.
    let c = IsBitConstraint::unconditional(7, 0);
    assert_ir_matches_evaluate(&c, "is_bit_unconditional");
}

#[test]
fn test_ir_matches_is_bit_conditional() {
    // cond * X * (1 - X), cond at column 3, X at column 5.
    let c = IsBitConstraint::new(3, 5, 0);
    assert_ir_matches_evaluate(&c, "is_bit_conditional");
}

#[test]
fn test_ir_matches_add_constraint_carries() {
    // 64-bit ADD with embedded carries, DWordWL operands.
    // cond at col 0; lhs=[1,2], rhs=[3,4], sum=[5,6].
    let (carry0, carry1) = AddConstraint::new_pair(
        vec![0],
        AddOperand::dword(1),
        AddOperand::dword(3),
        AddOperand::dword(5),
        0,
    );
    assert_ir_matches_evaluate(&carry0, "add_carry_0");
    assert_ir_matches_evaluate(&carry1, "add_carry_1");
}

#[test]
fn test_ir_matches_product_zero() {
    // col_a * col_b, columns 12 and 17.
    let c = ProductZeroConstraint::new(12, 17, 0);
    assert_ir_matches_evaluate(&c, "product_zero");
}
