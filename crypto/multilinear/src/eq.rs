//! The equality kernel `eq(r, x) = ∏_i (r_i·x_i + (1 - r_i)(1 - x_i))` and the
//! shift kernel, `shift_k(x, y) = 1` iff `index(y) = index(x) + k mod 2^n`, of
//! which the cyclic rotation `rot` is the `k = 1` case.

use math::field::{element::FieldElement, traits::IsField};

use crate::{Error, mle::Mle};

/// Builds the table of `eq(r, x)` for every `x` in `{0,1}^n`, in `O(2^n)`.
///
/// Doubles the table one variable at a time. Each step prepends its variable as
/// the new most significant bit, so the variables are consumed **back to
/// front**: that leaves variable 0 in the high bit, which is the indexing
/// convention [`Mle`](crate::mle::Mle) folds on.
pub fn eq_evals<F: IsField>(r: &[FieldElement<F>]) -> Vec<FieldElement<F>> {
    let mut table = vec![FieldElement::<F>::one()];
    for r_i in r.iter().rev() {
        let mut next = Vec::with_capacity(table.len() * 2);
        let one_minus = FieldElement::<F>::one() - r_i;
        for v in &table {
            next.push(v * &one_minus);
        }
        for v in &table {
            next.push(v * r_i);
        }
        table = next;
    }
    table
}

/// The multilinear extension of `eq(r, ·)`.
pub fn eq_mle<F: IsField>(r: &[FieldElement<F>]) -> Result<Mle<F>, Error> {
    Mle::new(eq_evals(r))
}

/// Evaluates `eq(r, x)` directly, without materializing the table.
pub fn eq_eval<F: IsField>(
    r: &[FieldElement<F>],
    x: &[FieldElement<F>],
) -> Result<FieldElement<F>, Error> {
    if r.len() != x.len() {
        return Err(Error::VariableCountMismatch {
            expected: r.len(),
            got: x.len(),
        });
    }
    let one = FieldElement::<F>::one();
    Ok(r.iter().zip(x).fold(one.clone(), |acc, (r_i, x_i)| {
        acc * (r_i * x_i + (&one - r_i) * (&one - x_i))
    }))
}

/// `rot` and `eq` together; the recursion needs both.
///
/// `next_f(z) = Σ_y rot(z, y)·f(y)`, which is how a rotated table gets bound to
/// the committed column.
pub fn eq_and_rot_eval<F: IsField>(
    x: &[FieldElement<F>],
    y: &[FieldElement<F>],
) -> Result<(FieldElement<F>, FieldElement<F>), Error> {
    if x.len() != y.len() {
        return Err(Error::VariableCountMismatch {
            expected: x.len(),
            got: y.len(),
        });
    }
    let one = FieldElement::<F>::one();
    let mut eq = one.clone();
    let mut rot = one.clone();

    // Recursion: rot(x, y) = y₀(1−x₀)·eq(rest) + (1−y₀)x₀·rot(rest), unrolled
    // with the least significant variable outermost. `rot` on the tail carries
    // "the low bits wrapped", which is what makes the next bit up increment.
    //
    // Our variable 0 is the most significant bit, so the fold runs front to
    // back — the opposite of the usual least-significant-first presentation.
    for (x_i, y_i) in x.iter().zip(y) {
        rot = y_i * (&one - x_i) * &eq + (&one - y_i) * x_i * &rot;
        eq *= x_i * y_i + (&one - x_i) * (&one - y_i);
    }
    Ok((eq, rot))
}

/// Just the rotation kernel. See [`eq_and_rot_eval`].
pub fn rot_eval<F: IsField>(
    x: &[FieldElement<F>],
    y: &[FieldElement<F>],
) -> Result<FieldElement<F>, Error> {
    Ok(eq_and_rot_eval(x, y)?.1)
}

/// `shift_k(x, y) = 1` iff `index(y) = index(x) + k mod 2^n`, extended
/// multilinearly. `k = 0` is [`eq_eval`], `k = 1` is [`rot_eval`].
///
/// Adds the constant `k` bit by bit from the least significant end, which is
/// variable `n − 1`. `state[c]` is the weight of the bits handled so far having
/// produced carry `c`; the shift wraps, so both carries are accepted at the end.
pub fn shift_eval<F: IsField>(
    x: &[FieldElement<F>],
    y: &[FieldElement<F>],
    k: usize,
) -> Result<FieldElement<F>, Error> {
    if x.len() != y.len() {
        return Err(Error::VariableCountMismatch {
            expected: x.len(),
            got: y.len(),
        });
    }
    let one = FieldElement::<F>::one();
    let zero = FieldElement::<F>::zero();
    let mut state = [one.clone(), zero.clone()];

    for (t, (x_j, y_j)) in x.iter().zip(y).rev().enumerate() {
        let eq_j = x_j * y_j + (&one - x_j) * (&one - y_j);
        // y_j = 1 − x_j: x_j = 1 carries out, x_j = 0 does not.
        let carry = x_j * (&one - y_j);
        let no_carry = (&one - x_j) * y_j;

        let mut next = [zero.clone(), zero.clone()];
        for (c, weight) in state.iter().enumerate() {
            match ((k >> t) & 1) + c {
                0 => next[0] += weight * &eq_j,
                1 => {
                    next[0] += weight * &no_carry;
                    next[1] += weight * &carry;
                }
                _ => next[1] += weight * &eq_j,
            }
        }
        state = next;
    }

    Ok(&state[0] + &state[1])
}

/// The table of `shift_k(x, ·)` over the cube, in `O(2^n)`.
///
/// Same carry recursion as [`shift_eval`], but each step doubles the table
/// instead of multiplying `y_j`'s weight out — prepending `y_j` as the new most
/// significant bit, which is the indexing [`Mle`] folds on.
pub fn shift_evals<F: IsField>(x: &[FieldElement<F>], k: usize) -> Vec<FieldElement<F>> {
    let one = FieldElement::<F>::one();
    let zero = FieldElement::<F>::zero();
    let mut state = [vec![one.clone()], vec![zero.clone()]];

    for (t, x_j) in x.iter().rev().enumerate() {
        let one_minus = &one - x_j;
        let len = state[0].len();
        let mut next = [vec![zero.clone(); 2 * len], vec![zero.clone(); 2 * len]];

        for (c, table) in state.iter().enumerate() {
            match ((k >> t) & 1) + c {
                // y_j = x_j, carry unchanged.
                0 => {
                    for (i, v) in table.iter().enumerate() {
                        next[0][i] += &one_minus * v;
                        next[0][len + i] += x_j * v;
                    }
                }
                // y_j = 1 − x_j: the `y_j = 1` half needs x_j = 0 and keeps the
                // carry clear, the `y_j = 0` half needs x_j = 1 and raises it.
                1 => {
                    for (i, v) in table.iter().enumerate() {
                        next[1][i] += x_j * v;
                        next[0][len + i] += &one_minus * v;
                    }
                }
                // y_j = x_j, carry out.
                _ => {
                    for (i, v) in table.iter().enumerate() {
                        next[1][i] += &one_minus * v;
                        next[1][len + i] += x_j * v;
                    }
                }
            }
        }
        state = next;
    }

    let [no_carry, carried] = state;
    no_carry
        .into_iter()
        .zip(carried)
        .map(|(a, b)| a + b)
        .collect()
}

/// The multilinear extension of `shift_k(x, ·)`.
pub fn shift_mle<F: IsField>(x: &[FieldElement<F>], k: usize) -> Result<Mle<F>, Error> {
    Mle::new(shift_evals(x, k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::goldilocks::GoldilocksField as F;

    type FE = FieldElement<F>;

    fn point(vals: &[u64]) -> Vec<FE> {
        vals.iter().map(|v| FE::from(*v)).collect()
    }

    /// The hypercube corner for `index`, variable 0 most significant.
    fn corner(index: usize, num_vars: usize) -> Vec<FE> {
        (0..num_vars)
            .map(|i| FE::from(((index >> (num_vars - 1 - i)) & 1) as u64))
            .collect()
    }

    #[test]
    fn is_the_indicator_on_the_hypercube() {
        let n = 3;
        for r_idx in 0..(1u64 << n) {
            let r = point(&[(r_idx >> 2) & 1, (r_idx >> 1) & 1, r_idx & 1]);
            let table = eq_evals(&r);
            assert_eq!(table.len(), 1 << n);
            for (x_idx, value) in table.iter().enumerate() {
                let expected = if x_idx as u64 == r_idx {
                    FE::one()
                } else {
                    FE::zero()
                };
                assert_eq!(value, &expected, "eq(r={r_idx}, x={x_idx})");
            }
        }
    }

    #[test]
    fn table_and_direct_evaluation_agree_on_corners() {
        let r = point(&[5, 9, 2]);
        for (x_idx, entry) in eq_evals(&r).into_iter().enumerate() {
            let x = point(&[
                ((x_idx >> 2) & 1) as u64,
                ((x_idx >> 1) & 1) as u64,
                (x_idx & 1) as u64,
            ]);
            assert_eq!(entry, eq_eval(&r, &x).unwrap());
        }
    }

    #[test]
    fn table_is_the_multilinear_extension() {
        // Evaluating the eq table as an MLE off the cube must match the
        // product formula.
        let r = point(&[3, 11]);
        let mle = eq_mle(&r).unwrap();
        let x = point(&[7, 13]);
        assert_eq!(mle.evaluate(&x).unwrap(), eq_eval(&r, &x).unwrap());
    }

    #[test]
    fn sums_to_one_over_the_cube() {
        // Σ_x eq(r, x) = 1 for any r, since eq interpolates a single corner.
        let r = point(&[4, 6, 8]);
        let total = eq_evals(&r).into_iter().fold(FE::zero(), |acc, v| acc + v);
        assert_eq!(total, FE::one());
    }

    #[test]
    fn is_symmetric_in_its_arguments() {
        let a = point(&[2, 3]);
        let b = point(&[5, 7]);
        assert_eq!(eq_eval(&a, &b).unwrap(), eq_eval(&b, &a).unwrap());
    }

    #[test]
    fn zero_variables_gives_the_empty_product() {
        assert_eq!(eq_evals::<F>(&[]), vec![FE::one()]);
        assert_eq!(eq_eval::<F>(&[], &[]).unwrap(), FE::one());
    }

    #[test]
    fn rejects_mismatched_arity() {
        assert_eq!(
            eq_eval(&point(&[1, 2]), &point(&[1])).unwrap_err(),
            Error::VariableCountMismatch {
                expected: 2,
                got: 1
            }
        );
    }

    #[test]
    fn rot_is_the_successor_indicator_on_the_hypercube() {
        for num_vars in 1..=4usize {
            let size = 1usize << num_vars;
            for xi in 0..size {
                for yi in 0..size {
                    let x = corner(xi, num_vars);
                    let y = corner(yi, num_vars);
                    let expected = if yi == (xi + 1) % size {
                        FE::one()
                    } else {
                        FE::zero()
                    };
                    assert_eq!(
                        rot_eval(&x, &y).unwrap(),
                        expected,
                        "n={num_vars}, x={xi}, y={yi}"
                    );
                }
            }
        }
    }

    #[test]
    fn rot_wraps_the_last_index_to_the_first() {
        let n = 3;
        let last = corner(7, n);
        let first = corner(0, n);
        assert_eq!(rot_eval(&last, &first).unwrap(), FE::one());
    }

    #[test]
    fn rot_reproduces_a_shifted_table() {
        // The identity the commitment scheme will lean on:
        // next_f(z) = Σ_y rot(z, y)·f(y), for z on the cube.
        let n = 3;
        let size = 1usize << n;
        let f: Vec<FE> = (0..size as u64).map(|i| FE::from(i * 7 + 5)).collect();

        for zi in 0..size {
            let z = corner(zi, n);
            let summed = (0..size).fold(FE::zero(), |acc, yi| {
                acc + rot_eval(&z, &corner(yi, n)).unwrap() * f[yi]
            });
            assert_eq!(summed, f[(zi + 1) % size], "z={zi}");
        }
    }

    #[test]
    fn rot_matches_its_table_off_the_cube() {
        // The closed form must be the multilinear extension of the table, so
        // the verifier can evaluate it at a random point.
        let n = 3;
        let size = 1usize << n;
        let z = point(&[5, 9, 2]);

        // Brute-force the MLE of rot(·, y) for a fixed y by extending the table.
        for yi in 0..size {
            let table: Vec<FE> = (0..size)
                .map(|xi| rot_eval(&corner(xi, n), &corner(yi, n)).unwrap())
                .collect();
            let mle = Mle::new(table).unwrap();
            assert_eq!(
                mle.evaluate(&z).unwrap(),
                rot_eval(&z, &corner(yi, n)).unwrap(),
                "y={yi}"
            );
        }
    }

    #[test]
    fn eq_and_rot_agree_with_the_standalone_helpers() {
        let x = point(&[3, 11, 4]);
        let y = point(&[7, 13, 2]);
        let (eq, rot) = eq_and_rot_eval(&x, &y).unwrap();
        assert_eq!(eq, eq_eval(&x, &y).unwrap());
        assert_eq!(rot, rot_eval(&x, &y).unwrap());
    }

    #[test]
    fn rot_rejects_mismatched_arity() {
        assert!(rot_eval(&point(&[1, 2]), &point(&[1])).is_err());
    }

    #[test]
    fn shift_is_the_offset_indicator_on_the_hypercube() {
        for num_vars in 1..=4usize {
            let size = 1usize << num_vars;
            // Past `size` too, so wrapping is exercised as its own case.
            for k in 0..(size + 3) {
                for xi in 0..size {
                    for yi in 0..size {
                        let expected = if yi == (xi + k) % size {
                            FE::one()
                        } else {
                            FE::zero()
                        };
                        assert_eq!(
                            shift_eval(&corner(xi, num_vars), &corner(yi, num_vars), k).unwrap(),
                            expected,
                            "n={num_vars}, k={k}, x={xi}, y={yi}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn shift_zero_is_eq_and_shift_one_is_rot() {
        // Off the cube, where agreeing on corners would not be enough.
        let x = point(&[3, 11, 4]);
        let y = point(&[7, 13, 2]);
        assert_eq!(shift_eval(&x, &y, 0).unwrap(), eq_eval(&x, &y).unwrap());
        assert_eq!(shift_eval(&x, &y, 1).unwrap(), rot_eval(&x, &y).unwrap());
    }

    #[test]
    fn shift_table_matches_the_closed_form() {
        let x = point(&[5, 9, 2]);
        for k in 0..10usize {
            for (yi, entry) in shift_evals(&x, k).into_iter().enumerate() {
                assert_eq!(entry, shift_eval(&x, &corner(yi, 3), k).unwrap(), "k={k}");
            }
        }
    }

    #[test]
    fn shift_table_is_the_multilinear_extension() {
        // The verifier evaluates the kernel at a random point while the prover
        // folds the table, so the two must be the same polynomial.
        let x = point(&[3, 11, 4]);
        let z = point(&[7, 13, 2]);
        for k in 0..8usize {
            assert_eq!(
                shift_mle(&x, k).unwrap().evaluate(&z).unwrap(),
                shift_eval(&x, &z, k).unwrap(),
                "k={k}"
            );
        }
    }

    #[test]
    fn shift_reproduces_a_shifted_column() {
        // The identity the reduction leans on:
        // f_shift_k(z) = Σ_y shift_k(z, y)·f(y).
        let n = 3;
        let size = 1usize << n;
        let f: Vec<FE> = (0..size as u64).map(|i| FE::from(i * 7 + 5)).collect();
        let z = point(&[5, 9, 2]);

        for k in 0..6usize {
            let shifted = Mle::new((0..size).map(|i| f[(i + k) % size]).collect()).unwrap();
            let summed = shift_evals(&z, k)
                .into_iter()
                .zip(&f)
                .fold(FE::zero(), |acc, (w, v)| acc + w * v);
            assert_eq!(summed, shifted.evaluate(&z).unwrap(), "k={k}");
        }
    }

    #[test]
    fn shift_sums_to_one_over_the_cube() {
        // The kernel picks out one corner, so its extension sums to one at any
        // point — a cheap check that the carry recursion loses no weight.
        let x = point(&[4, 6, 8]);
        for k in 0..10usize {
            let total = shift_evals(&x, k)
                .into_iter()
                .fold(FE::zero(), |acc, v| acc + v);
            assert_eq!(total, FE::one(), "k={k}");
        }
    }

    #[test]
    fn shift_rejects_mismatched_arity() {
        assert!(shift_eval(&point(&[1, 2]), &point(&[1]), 1).is_err());
    }
}
