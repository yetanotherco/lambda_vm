use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsField;
use math::polynomial::Polynomial;

type FE = FieldElement<GoldilocksField>;

/// FRI polynomial folding: computes P_even(x) + beta * P_odd(x)
/// where P(x) = P_even(x^2) + x * P_odd(x^2)
fn fold_polynomial<F>(
    poly: &Polynomial<FieldElement<F>>,
    beta: &FieldElement<F>,
) -> Polynomial<FieldElement<F>>
where
    F: IsField,
{
    let coefficients = poly.coefficients();
    if coefficients.is_empty() {
        return Polynomial::new(&[]);
    }

    let mut result = Vec::with_capacity(coefficients.len().div_ceil(2));

    for chunk in coefficients.chunks(2) {
        let folded = if chunk.len() == 2 {
            &chunk[0] + &(&chunk[1] * beta)
        } else {
            chunk[0].clone()
        };
        result.push(folded);
    }

    Polynomial::new(&result)
}

#[test]
fn test_fold_power_of_2() {
    let p0 = Polynomial::new(&[
        FE::new(3),
        FE::new(1),
        FE::new(2),
        FE::new(7),
        FE::new(3),
        FE::new(5),
        FE::new(4),
        FE::new(2),
    ]);
    let beta = FE::new(4);
    let p1 = fold_polynomial(&p0, &beta);
    assert_eq!(
        p1,
        Polynomial::new(&[FE::new(7), FE::new(30), FE::new(23), FE::new(12)])
    );

    let gamma = FE::new(3);
    let p2 = fold_polynomial(&p1, &gamma);
    assert_eq!(p2, Polynomial::new(&[FE::new(97), FE::new(59)]));

    let delta = FE::new(2);
    let p3 = fold_polynomial(&p2, &delta);
    assert_eq!(p3, Polynomial::new(&[FE::new(215)]));
    assert_eq!(p3.degree(), 0);
}

#[test]
fn test_fold_size_2() {
    let p2 = Polynomial::new(&[FE::new(10), FE::new(20)]);
    let beta = FE::new(3);
    let result = fold_polynomial(&p2, &beta);
    assert_eq!(result, Polynomial::new(&[FE::new(70)]));
}

