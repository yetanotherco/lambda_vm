use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

use crate::plonky3_config::Val;

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

    fn num_constraints(&self) -> Option<usize> {
        Some(4 * self.num_sequences)
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(2)
    }
}

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
            let mut when_first_row = builder.when_first_row();
            when_first_row.assert_eq(left, a);
            when_first_row.assert_eq(right, b);

            let mut when_transition = builder.when_transition();
            when_transition.assert_eq(next_left, left + right);
            when_transition.assert_eq(next_right, right + next_left);
        }
    }
}

pub fn generate_fibonacci_trace(num_sequences: usize, rows: usize) -> RowMajorMatrix<Val> {
    assert!(rows.is_power_of_two(), "rows must be a power of two");
    let width = 2 * num_sequences;
    let mut values = vec![Val::ZERO; width * rows];

    for seq in 0..num_sequences {
        let mut left = Val::from_u64((seq + 1) as u64);
        let mut right = Val::from_u64((seq + 2) as u64);

        for row in 0..rows {
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

pub fn public_values(num_sequences: usize) -> Vec<Val> {
    let mut pis = Vec::with_capacity(2 * num_sequences);
    for seq in 0..num_sequences {
        pis.push(Val::from_u64((seq + 1) as u64));
        pis.push(Val::from_u64((seq + 2) as u64));
    }
    pis
}
