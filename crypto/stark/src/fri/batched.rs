use math::field::element::FieldElement;
use math::field::traits::IsField;

/// Combine DEEP polynomial codewords by their FRI height for batched FRI.
///
/// Each element of `inputs` is a pair `(codeword, height)` where `height` is
/// the log₂ of the codeword length (i.e. `codeword.len() == 2^height`).
/// The global index `i` into `inputs` is used to derive the mixing power
/// `alpha^i` (index 0 → alpha^0 = 1, index 1 → alpha^1, …).
///
/// Returns a `Vec` of length `max_height + 1`.  Index `h` contains
/// `Some(combined)` where `combined[j] = Σ_{i : height_i == h} alpha^i * codeword_i[j]`,
/// or `None` when no input has height `h`.
pub fn combine_by_height<E>(
    inputs: &[(Vec<FieldElement<E>>, usize)],
    alpha: &FieldElement<E>,
) -> Vec<Option<Vec<FieldElement<E>>>>
where
    E: IsField,
    FieldElement<E>: Clone,
{
    if inputs.is_empty() {
        return vec![];
    }

    let max_height = inputs
        .iter()
        .map(|(_, h)| *h)
        .max()
        .expect("inputs is non-empty so max height exists");

    let mut out: Vec<Option<Vec<FieldElement<E>>>> = vec![None; max_height + 1];

    // Precompute alpha^0, alpha^1, …, alpha^(n-1) via repeated multiplication.
    let mut alpha_pows: Vec<FieldElement<E>> = Vec::with_capacity(inputs.len());
    let mut cur = FieldElement::one();
    for _ in 0..inputs.len() {
        alpha_pows.push(cur.clone());
        cur = &cur * alpha;
    }

    for (i, (codeword, height)) in inputs.iter().enumerate() {
        let h = *height;
        let expected_len = 1usize << h;
        assert_eq!(
            codeword.len(),
            expected_len,
            "codeword at index {i} has length {} but height {h} expects {expected_len}",
            codeword.len()
        );

        let a_i = &alpha_pows[i];

        match &mut out[h] {
            None => {
                let combined: Vec<FieldElement<E>> = codeword.iter().map(|x| a_i * x).collect();
                out[h] = Some(combined);
            }
            Some(acc) => {
                for (j, x) in codeword.iter().enumerate() {
                    acc[j] = &acc[j] + &(a_i * x);
                }
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::element::FieldElement;
    use math::field::goldilocks::GoldilocksField;

    type FE = FieldElement<GoldilocksField>;

    #[test]
    fn combine_by_height_two_height3_one_height2() {
        // Three codewords: indices 0, 1 have height 3 (length 8);
        //                  index 2 has height 2 (length 4).
        let cw0: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let cw1: Vec<FE> = (10u64..=17).map(FE::from).collect();
        let cw2: Vec<FE> = (100u64..=103).map(FE::from).collect();

        let alpha = FE::from(7u64);

        let inputs: Vec<(Vec<FE>, usize)> =
            vec![(cw0.clone(), 3), (cw1.clone(), 3), (cw2.clone(), 2)];

        let out = combine_by_height(&inputs, &alpha);

        // Output vec length = max_height + 1 = 4 (indices 0..=3 only).
        assert_eq!(out.len(), 4, "output length should be max_height+1 = 4");

        // Heights 0 and 1 have no inputs.
        assert!(out[0].is_none(), "height 0 should be None");
        assert!(out[1].is_none(), "height 1 should be None");

        // Height 3: combined[j] = alpha^0 * cw0[j] + alpha^1 * cw1[j]
        let alpha0 = FE::one();
        let alpha1 = alpha.clone();
        let expected3: Vec<FE> = cw0
            .iter()
            .zip(cw1.iter())
            .map(|(a, b)| &(&alpha0 * a) + &(&alpha1 * b))
            .collect();

        let got3 = out[3].as_ref().expect("height 3 should be Some");
        assert_eq!(
            got3.len(),
            8,
            "height-3 combined codeword should have length 8"
        );
        assert_eq!(got3, &expected3, "height-3 combined values mismatch");

        // Height 2: combined[j] = alpha^2 * cw2[j]
        let alpha2 = &alpha * &alpha;
        let expected2: Vec<FE> = cw2.iter().map(|x| &alpha2 * x).collect();

        let got2 = out[2].as_ref().expect("height 2 should be Some");
        assert_eq!(
            got2.len(),
            4,
            "height-2 combined codeword should have length 4"
        );
        assert_eq!(got2, &expected2, "height-2 combined values mismatch");
    }
}
