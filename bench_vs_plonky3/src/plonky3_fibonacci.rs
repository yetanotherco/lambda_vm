use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;

/// Multi-sequence Fibonacci AIR for Plonky3.
///
/// Each sequence uses 2 columns (left, right) in a 2-row window, where each
/// Plonky3 row stores two consecutive Lambda rows:
///   local.left  = x_{2i}
///   local.right = x_{2i+1}
///   next.left   = x_{2i+2} = local.left + local.right
///   next.right  = x_{2i+3} = local.right + next.left
///
/// This packs two consecutive Lambda trace rows into one Plonky3 row. It is the
/// closest encoding of Lambda's `row + 2` Fibonacci transition available in
/// Plonky3's current/next-row AIR window while keeping the same committed cell
/// count.
///
/// Boundary constraints at the first row pin each sequence's initial (a, b)
/// values against public inputs, matching Lambda's `FibonacciMultiColumnAIR`.
///
/// Public values layout: `[a_0, b_0, a_1, b_1, ..., a_{N-1}, b_{N-1}]`
/// where `N = num_sequences`.
///
/// For `num_sequences` sequences, the AIR has `2 * num_sequences` columns
/// and `2 * num_sequences` public values.
pub struct P3FibonacciAir {
    pub num_sequences: usize,
}

impl<F: PrimeCharacteristicRing> BaseAir<F> for P3FibonacciAir {
    fn width(&self) -> usize {
        2 * self.num_sequences
    }

    fn num_public_values(&self) -> usize {
        2 * self.num_sequences
    }
}

/// One sequence's (local_left, local_right, next_left, next_right, a, b)
/// snapshot extracted from an `AirBuilder`. Factored out to keep the
/// `Air::eval` signature readable (clippy::type_complexity).
type FibPairRow<AB> = (
    <AB as AirBuilder>::Var,
    <AB as AirBuilder>::Var,
    <AB as AirBuilder>::Var,
    <AB as AirBuilder>::Var,
    <AB as AirBuilder>::PublicVar,
    <AB as AirBuilder>::PublicVar,
);

impl<AB: AirBuilder> Air<AB> for P3FibonacciAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let next = main.next_slice();

        // Collect (left, right, next_left, next_right, a, b) per sequence so that
        // `pis`'s borrow on `builder` can end before we mutate `builder`.
        let rows: Vec<FibPairRow<AB>> = {
            let pis = builder.public_values();
            (0..self.num_sequences)
                .map(|seq| {
                    (
                        local[2 * seq],
                        local[2 * seq + 1],
                        next[2 * seq],
                        next[2 * seq + 1],
                        pis[2 * seq],
                        pis[2 * seq + 1],
                    )
                })
                .collect()
        };
        drop(main);

        for (left, right, next_left, next_right, a, b) in rows {
            // Boundary: first row pins (left, right) = (a, b)
            let mut when_first_row = builder.when_first_row();
            when_first_row.assert_eq(left, a);
            when_first_row.assert_eq(right, b);

            let mut when_transition = builder.when_transition();
            // Advance two Lambda rows per Plonky3 row.
            when_transition.assert_eq(next_left, left + right);
            when_transition.assert_eq(next_right, right + next_left);
        }
    }
}

/// Generates a Fibonacci trace for Plonky3.
///
/// For `num_sequences` sequences and `num_rows` rows (must be power of 2),
/// produces a `RowMajorMatrix` with `2 * num_sequences` columns. When
/// comparing against Lambda's one-column-per-sequence trace, pass
/// `lambda_trace_length / 2` as `num_rows`.
///
/// Each sequence `s` starts with initial values matching Lambda's
/// `create_initial_values()`: `left = s + 1`, `right = s + 2`.
pub fn generate_fibonacci_trace(
    num_sequences: usize,
    num_rows: usize,
) -> RowMajorMatrix<Goldilocks> {
    assert!(num_rows.is_power_of_two(), "num_rows must be a power of 2");
    let width = 2 * num_sequences;
    let mut values = vec![Goldilocks::ZERO; width * num_rows];

    for seq in 0..num_sequences {
        let mut left = Goldilocks::from_u64((seq + 1) as u64);
        let mut right = Goldilocks::from_u64((seq + 2) as u64);

        for row in 0..num_rows {
            values[row * width + 2 * seq] = left;
            values[row * width + 2 * seq + 1] = right;
            let next_left = left + right;
            let next_right = right + next_left;
            left = next_left;
            right = next_right;
        }
    }

    RowMajorMatrix::new(values, width)
}

/// Builds public values matching `generate_fibonacci_trace`'s initial values:
/// `[a_0, b_0, a_1, b_1, ...] = [1, 2, 2, 3, 3, 4, ...]`
pub fn public_values(num_sequences: usize) -> Vec<Goldilocks> {
    let mut pis = Vec::with_capacity(2 * num_sequences);
    for seq in 0..num_sequences {
        pis.push(Goldilocks::from_u64((seq + 1) as u64));
        pis.push(Goldilocks::from_u64((seq + 2) as u64));
    }
    pis
}
