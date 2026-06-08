use crate::fft::bit_reversing::in_place_bit_reverse_permute;
use crate::fft::bowers_fft::{LayerTwiddles, bowers_fft_opt_fused, bowers_ifft_opt};
use crate::fft::bowers_fft_batch::{
    bowers_fft_batch_row_major, bowers_ifft_batch_row_major, in_place_bit_reverse_permute_row_major,
};
use crate::field::element::FieldElement;
use crate::field::goldilocks::GoldilocksField;
use alloc::vec;
use alloc::vec::Vec;

type F = GoldilocksField;
type FE = FieldElement<F>;

fn col_major_to_row_major(cols: &[Vec<FE>]) -> (Vec<FE>, usize) {
    let m = cols.len();
    if m == 0 {
        return (Vec::new(), 0);
    }
    let n = cols[0].len();
    let mut row_major = vec![FE::zero(); n * m];
    for r in 0..n {
        for c in 0..m {
            row_major[r * m + c] = cols[c][r];
        }
    }
    (row_major, m)
}

fn row_major_to_col_major(buf: &[FE], m: usize) -> Vec<Vec<FE>> {
    if m == 0 {
        return Vec::new();
    }
    let n = buf.len() / m;
    let mut cols: Vec<Vec<FE>> = (0..m).map(|_| Vec::with_capacity(n)).collect();
    for r in 0..n {
        for c in 0..m {
            cols[c].push(buf[r * m + c]);
        }
    }
    cols
}

fn sample_columns(n: usize, m: usize, seed: u64) -> Vec<Vec<FE>> {
    (0..m)
        .map(|c| {
            (0..n)
                .map(|r| FE::from(seed.wrapping_add((c as u64) * 1_000_003 + r as u64)))
                .collect()
        })
        .collect()
}

#[test]
fn bit_reverse_row_major_matches_single_column_per_column() {
    for log_n in 1..6 {
        let n = 1usize << log_n;
        for m in [1usize, 2, 3, 4, 5, 8] {
            let cols = sample_columns(n, m, 42);
            let mut expected_cols = cols.clone();
            for c in expected_cols.iter_mut() {
                in_place_bit_reverse_permute(c);
            }
            let (mut row_major, _) = col_major_to_row_major(&cols);
            in_place_bit_reverse_permute_row_major(&mut row_major, m);
            let actual_cols = row_major_to_col_major(&row_major, m);
            assert_eq!(actual_cols, expected_cols, "log_n={log_n} m={m}");
        }
    }
}

#[test]
fn batched_fft_matches_single_column_fft() {
    for log_n in 1..7 {
        let n = 1usize << log_n;
        let tw = LayerTwiddles::<F>::new(log_n as u64).unwrap();
        for m in [1usize, 2, 3, 5, 8] {
            let cols = sample_columns(n, m, 7);
            let mut expected = cols.clone();
            for col in expected.iter_mut() {
                bowers_fft_opt_fused::<F, F>(col, &tw).unwrap();
            }

            let (mut row_major, _) = col_major_to_row_major(&cols);
            bowers_fft_batch_row_major::<F, F>(&mut row_major, m, &tw).unwrap();
            let actual = row_major_to_col_major(&row_major, m);
            assert_eq!(actual, expected, "log_n={log_n} m={m}");
        }
    }
}

#[test]
fn batched_ifft_matches_single_column_ifft() {
    for log_n in 1..7 {
        let n = 1usize << log_n;
        let tw = LayerTwiddles::<F>::new_inverse(log_n as u64).unwrap();
        for m in [1usize, 2, 3, 5, 8] {
            let cols = sample_columns(n, m, 11);
            let mut expected = cols.clone();
            for col in expected.iter_mut() {
                bowers_ifft_opt::<F, F>(col, &tw).unwrap();
            }

            let (mut row_major, _) = col_major_to_row_major(&cols);
            bowers_ifft_batch_row_major::<F, F>(&mut row_major, m, &tw).unwrap();
            let actual = row_major_to_col_major(&row_major, m);
            assert_eq!(actual, expected, "log_n={log_n} m={m}");
        }
    }
}

#[test]
fn fft_then_ifft_round_trip_batched() {
    for log_n in 1..7 {
        let n = 1usize << log_n;
        let n_inv = FE::from(n as u64).inv().unwrap();
        let fwd = LayerTwiddles::<F>::new(log_n as u64).unwrap();
        let inv = LayerTwiddles::<F>::new_inverse(log_n as u64).unwrap();

        for m in [1usize, 2, 4] {
            let cols = sample_columns(n, m, 13);
            let original = cols.clone();
            let (mut buf, _) = col_major_to_row_major(&cols);

            // Forward: natural -> bit-reversed
            bowers_fft_batch_row_major::<F, F>(&mut buf, m, &fwd).unwrap();
            in_place_bit_reverse_permute_row_major(&mut buf, m);

            // Inverse: bit-reversed -> natural
            in_place_bit_reverse_permute_row_major(&mut buf, m);
            bowers_ifft_batch_row_major::<F, F>(&mut buf, m, &inv).unwrap();
            for x in buf.iter_mut() {
                *x = &*x * &n_inv;
            }

            let recovered = row_major_to_col_major(&buf, m);
            assert_eq!(recovered, original, "log_n={log_n} m={m}");
        }
    }
}
