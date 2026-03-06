use math::field::{element::FieldElement, traits::IsField};

/// A rational number `numerator / denominator` in a field.
///
/// Used in the GKR protocol to represent LogUp contributions before they are
/// reduced to a single field element via batch inversion. Fraction addition
/// uses cross-multiplication to avoid per-addition inversions.
pub struct Fraction<E: IsField> {
    pub numerator: FieldElement<E>,
    pub denominator: FieldElement<E>,
}

impl<E: IsField> Fraction<E> {
    /// Create a new fraction `numerator / denominator`.
    pub fn new(numerator: FieldElement<E>, denominator: FieldElement<E>) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    /// Add two fractions via cross-multiplication:
    ///   a/b + c/d = (a*d + c*b) / (b*d)
    ///
    /// No reduction or normalization is performed.
    pub fn add(&self, other: &Fraction<E>) -> Fraction<E> {
        let numerator =
            &(&self.numerator * &other.denominator) + &(&other.numerator * &self.denominator);
        let denominator = &self.denominator * &other.denominator;
        Fraction {
            numerator,
            denominator,
        }
    }
}

/// One layer of the summation tree used in the GKR protocol.
///
/// Each layer stores parallel arrays of numerators and denominators representing
/// fractions at that level. The leaf layer has N fractions; each subsequent layer
/// halves the count by pairwise fraction addition, until the root layer has 1.
pub struct SummationLayer<E: IsField> {
    pub numerators: Vec<FieldElement<E>>,
    pub denominators: Vec<FieldElement<E>>,
}

/// Build a summation tree from leaf fractions.
///
/// Takes N leaf fractions (as parallel numerator/denominator vectors) and
/// returns layers from leaves (index 0, size N) to root (last index, size 1).
/// Each layer is built by pairwise fraction addition:
///   parent_n = left_n * right_d + right_n * left_d
///   parent_d = left_d * right_d
///
/// # Panics
/// Panics if `leaf_numerators` and `leaf_denominators` have different lengths,
/// or if the length is not a power of 2.
pub fn build_summation_tree<E: IsField>(
    leaf_numerators: Vec<FieldElement<E>>,
    leaf_denominators: Vec<FieldElement<E>>,
) -> Vec<SummationLayer<E>> {
    let n = leaf_numerators.len();
    assert_eq!(
        n,
        leaf_denominators.len(),
        "numerators and denominators must have the same length"
    );
    assert!(n.is_power_of_two(), "number of leaves must be a power of 2");

    // Number of layers: log2(n) + 1 (leaves + intermediate + root)
    let num_layers = n.trailing_zeros() as usize + 1;
    let mut layers = Vec::with_capacity(num_layers);

    // Layer 0: the leaves themselves
    layers.push(SummationLayer {
        numerators: leaf_numerators,
        denominators: leaf_denominators,
    });

    // Build each subsequent layer by pairwise fraction addition
    for layer_idx in 1..num_layers {
        let prev = &layers[layer_idx - 1];
        let prev_len = prev.numerators.len();
        let new_len = prev_len / 2;

        let mut numerators = Vec::with_capacity(new_len);
        let mut denominators = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let left_n = &prev.numerators[2 * i];
            let left_d = &prev.denominators[2 * i];
            let right_n = &prev.numerators[2 * i + 1];
            let right_d = &prev.denominators[2 * i + 1];

            // Cross-multiply: (left_n * right_d + right_n * left_d) / (left_d * right_d)
            let parent_n = &(left_n * right_d) + &(right_n * left_d);
            let parent_d = left_d * right_d;

            numerators.push(parent_n);
            denominators.push(parent_d);
        }

        layers.push(SummationLayer {
            numerators,
            denominators,
        });
    }

    layers
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;

    type FE = FieldElement<GoldilocksField>;

    #[test]
    fn test_fraction_add() {
        // (3/5) + (7/11) = (3*11 + 7*5) / (5*11) = (33 + 35) / 55 = 68/55
        let a = Fraction::new(FE::from(3u64), FE::from(5u64));
        let b = Fraction::new(FE::from(7u64), FE::from(11u64));
        let result = a.add(&b);

        assert_eq!(result.numerator, FE::from(68u64));
        assert_eq!(result.denominator, FE::from(55u64));
    }

    #[test]
    fn test_build_summation_tree_4_leaves() {
        // 4 leaves: 1/2, 3/4, 5/6, 7/8
        // Layer 0 (4 fractions): [1/2, 3/4, 5/6, 7/8]
        // Layer 1 (2 fractions):
        //   pair 0: 1/2 + 3/4 = (1*4 + 3*2)/(2*4) = 10/8
        //   pair 1: 5/6 + 7/8 = (5*8 + 7*6)/(6*8) = 82/48
        // Layer 2 (1 fraction, root):
        //   10/8 + 82/48 = (10*48 + 82*8)/(8*48) = (480 + 656)/384 = 1136/384
        let nums = vec![
            FE::from(1u64),
            FE::from(3u64),
            FE::from(5u64),
            FE::from(7u64),
        ];
        let dens = vec![
            FE::from(2u64),
            FE::from(4u64),
            FE::from(6u64),
            FE::from(8u64),
        ];

        let tree = build_summation_tree(nums, dens);

        // Should have 3 layers: 4 -> 2 -> 1
        assert_eq!(tree.len(), 3);
        assert_eq!(tree[0].numerators.len(), 4);
        assert_eq!(tree[1].numerators.len(), 2);
        assert_eq!(tree[2].numerators.len(), 1);

        // Layer 1 checks
        assert_eq!(tree[1].numerators[0], FE::from(10u64)); // 1*4 + 3*2
        assert_eq!(tree[1].denominators[0], FE::from(8u64)); // 2*4
        assert_eq!(tree[1].numerators[1], FE::from(82u64)); // 5*8 + 7*6
        assert_eq!(tree[1].denominators[1], FE::from(48u64)); // 6*8

        // Root checks
        assert_eq!(tree[2].numerators[0], FE::from(1136u64)); // 10*48 + 82*8
        assert_eq!(tree[2].denominators[0], FE::from(384u64)); // 8*48

        // Verify the root fraction equals the actual sum:
        // 1/2 + 3/4 + 5/6 + 7/8 = 12/24 + 18/24 + 20/24 + 21/24 = 71/24
        // Check: 1136/384 = 71/24 (both sides: 1136*24 = 27264, 71*384 = 27264)
        let root_n = &tree[2].numerators[0];
        let root_d = &tree[2].denominators[0];
        assert_eq!(
            root_n * &FE::from(24u64),
            &FE::from(71u64) * root_d
        );
    }

    #[test]
    fn test_build_summation_tree_8_leaves() {
        // 8 leaves: i/(i+1) for i in 1..=8, i.e., 1/2, 2/3, 3/4, 4/5, 5/6, 6/7, 7/8, 8/9
        let nums: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let dens: Vec<FE> = (2..=9).map(|i| FE::from(i as u64)).collect();

        let tree = build_summation_tree(nums.clone(), dens.clone());

        // Should have 4 layers: 8 -> 4 -> 2 -> 1
        assert_eq!(tree.len(), 4);
        assert_eq!(tree[0].numerators.len(), 8);
        assert_eq!(tree[1].numerators.len(), 4);
        assert_eq!(tree[2].numerators.len(), 2);
        assert_eq!(tree[3].numerators.len(), 1);

        // Compute expected root by sequential fraction addition
        let mut acc = Fraction::new(nums[0].clone(), dens[0].clone());
        for i in 1..8 {
            acc = acc.add(&Fraction::new(nums[i].clone(), dens[i].clone()));
        }

        // The tree root and the sequential sum should represent the same rational number:
        // tree_n / tree_d == acc_n / acc_d  <=>  tree_n * acc_d == acc_n * tree_d
        let root_n = &tree[3].numerators[0];
        let root_d = &tree[3].denominators[0];
        assert_eq!(
            root_n * &acc.denominator,
            &acc.numerator * root_d,
            "Tree root must equal sequential sum as a fraction"
        );
    }

    #[test]
    fn test_build_summation_tree_single_leaf() {
        // Edge case: 1 leaf (2^0 = 1)
        let nums = vec![FE::from(42u64)];
        let dens = vec![FE::from(7u64)];

        let tree = build_summation_tree(nums, dens);

        // Should have 1 layer (just the leaf)
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].numerators.len(), 1);
        assert_eq!(tree[0].numerators[0], FE::from(42u64));
        assert_eq!(tree[0].denominators[0], FE::from(7u64));
    }

    #[test]
    #[should_panic(expected = "number of leaves must be a power of 2")]
    fn test_build_summation_tree_non_power_of_2_panics() {
        let nums = vec![FE::from(1u64), FE::from(2u64), FE::from(3u64)];
        let dens = vec![FE::from(1u64), FE::from(1u64), FE::from(1u64)];
        let _ = build_summation_tree(nums, dens);
    }

    #[test]
    #[should_panic(expected = "numerators and denominators must have the same length")]
    fn test_build_summation_tree_mismatched_lengths_panics() {
        let nums = vec![FE::from(1u64), FE::from(2u64)];
        let dens = vec![FE::from(1u64)];
        let _ = build_summation_tree(nums, dens);
    }
}
