use math::field::{element::FieldElement, goldilocks::GoldilocksField, traits::IsFFTField};
use math::polynomial::Polynomial;

use crate::constraints::boundary::{BoundaryConstraint, BoundaryConstraints};

type PrimeField = GoldilocksField;

#[test]
fn zerofier_is_the_correct_one() {
    let one = FieldElement::<PrimeField>::one();

    // Fibonacci constraints:
    //   * a0 = 1
    //   * a1 = 1
    //   * a7 = 32
    let a0 = BoundaryConstraint::new_simple_main(0, one);
    let a1 = BoundaryConstraint::new_simple_main(1, one);
    let result = BoundaryConstraint::new_simple_main(7, FieldElement::<PrimeField>::from(32));

    let constraints = BoundaryConstraints::from_constraints(vec![a0, a1, result]);

    let primitive_root = PrimeField::get_primitive_root_of_unity(3).unwrap();

    // P_0(x) = (x - 1)
    let a0_zerofier = Polynomial::new(&[-one, one]);
    // P_1(x) = (x - w^1)
    let a1_zerofier = Polynomial::new(&[-primitive_root.pow(1u32), one]);
    // P_res(x) = (x - w^7)
    let res_zerofier = Polynomial::new(&[-primitive_root.pow(7u32), one]);

    let expected_zerofier = a0_zerofier * a1_zerofier * res_zerofier;

    let zerofier = constraints.compute_zerofier(&primitive_root, 0);

    assert_eq!(expected_zerofier, zerofier);
}
