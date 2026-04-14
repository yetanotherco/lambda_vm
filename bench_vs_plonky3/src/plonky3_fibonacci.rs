use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;

/// Multi-sequence Fibonacci AIR for Plonky3.
///
/// Each sequence uses 2 columns (left, right) in a 2-row window:
///   next.left  = local.right
///   next.right = local.left + local.right
///
/// For `num_sequences` sequences, the AIR has `2 * num_sequences` columns.
///
/// Equivalence with Lambda's Fibonacci AIR:
///   Lambda: `num_sequences` columns, `L` rows, constraint col[i+2] = col[i+1] + col[i]
///   Plonky3: `2 * num_sequences` columns, `L/2` rows, same total cells
pub struct P3FibonacciAir {
    pub num_sequences: usize,
}

impl<F: PrimeCharacteristicRing> BaseAir<F> for P3FibonacciAir {
    fn width(&self) -> usize {
        2 * self.num_sequences
    }
}

impl<AB: AirBuilder> Air<AB> for P3FibonacciAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let next = main.next_slice();

        for seq in 0..self.num_sequences {
            let left = local[2 * seq];
            let right = local[2 * seq + 1];
            let next_left = next[2 * seq];
            let next_right = next[2 * seq + 1];

            // Shift: next row's left = current row's right
            builder.when_transition().assert_eq(next_left, right);
            // Fibonacci: next row's right = current left + current right
            builder.when_transition().assert_eq(next_right, left + right);
        }
    }
}

/// Generates a Fibonacci trace for Plonky3.
///
/// For `num_sequences` sequences and `num_rows` rows (must be power of 2),
/// produces a `RowMajorMatrix` with `2 * num_sequences` columns.
///
/// Each sequence `s` starts with initial values:
///   left = s + 1, right = s + 2
/// matching Lambda's `create_initial_values()`.
pub fn generate_fibonacci_trace(num_sequences: usize, num_rows: usize) -> RowMajorMatrix<Goldilocks> {
    assert!(num_rows.is_power_of_two(), "num_rows must be a power of 2");
    let width = 2 * num_sequences;
    let mut values = vec![Goldilocks::ZERO; width * num_rows];

    for seq in 0..num_sequences {
        // Initial values matching Lambda: (seq+1, seq+2)
        let mut left = Goldilocks::from_u64((seq + 1) as u64);
        let mut right = Goldilocks::from_u64((seq + 2) as u64);

        for row in 0..num_rows {
            values[row * width + 2 * seq] = left;
            values[row * width + 2 * seq + 1] = right;
            let new_right = left + right;
            left = right;
            right = new_right;
        }
    }

    RowMajorMatrix::new(values, width)
}
