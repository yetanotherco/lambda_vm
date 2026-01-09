#[cfg(test)]
mod tests {
    use crate::cyclic_group::IsGroup;
    use crate::msm::{naive, pippenger};
    use crate::{
        elliptic_curve::{
            short_weierstrass::curves::bls12_381::curve::BLS12381Curve, traits::IsEllipticCurve,
        },
        unsigned_integer::element::UnsignedInteger,
    };
    use alloc::vec::Vec;
    use proptest::{collection, prelude::*, prop_assert_eq, prop_compose, proptest};

    const _CASES: u32 = 20;
    const _MAX_WSIZE: usize = 8;
    const _MAX_LEN: usize = 30;

    prop_compose! {
        fn unsigned_integer()(limbs: [u64; 6]) -> UnsignedInteger<6> {
            UnsignedInteger::from_limbs(limbs)
        }
    }

    prop_compose! {
        fn unsigned_integer_vec()(vec in collection::vec(unsigned_integer(), 0.._MAX_LEN)) -> Vec<UnsignedInteger<6>> {
            vec
        }
    }

    prop_compose! {
        fn point()(power: u128) -> <BLS12381Curve as IsEllipticCurve>::PointRepresentation {
            BLS12381Curve::generator().operate_with_self(power)
        }
    }

    prop_compose! {
        fn points_vec()(vec in collection::vec(point(), 0.._MAX_LEN)) -> Vec<<BLS12381Curve as IsEllipticCurve>::PointRepresentation> {
            vec
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: _CASES, .. ProptestConfig::default()
          })]
        // Property-based test that ensures `pippenger::msm` gives same result as `naive::msm`.
        #[test]
        fn test_pippenger_matches_naive_msm(window_size in 1.._MAX_WSIZE, cs in unsigned_integer_vec(), points in points_vec()) {
            let min_len = cs.len().min(points.len());
            let cs = cs[..min_len].to_vec();
            let points = points[..min_len].to_vec();

            let pippenger = pippenger::msm_with(&cs, &points, window_size);
            let naive = naive::msm(&cs, &points).unwrap();

            prop_assert_eq!(naive, pippenger);
        }

        // Property-based test that ensures `pippenger::msm_with` gives same result as `pippenger::parallel_msm_with`.
        #[test]
        #[cfg(feature = "parallel")]
        fn test_parallel_pippenger_matches_sequential(window_size in 1.._MAX_WSIZE, cs in unsigned_integer_vec(), points in points_vec()) {
            let min_len = cs.len().min(points.len());
            let cs = cs[..min_len].to_vec();
            let points = points[..min_len].to_vec();

            let sequential = pippenger::msm_with(&cs, &points, window_size);
            let parallel = pippenger::parallel_msm_with(&cs, &points, window_size);

            prop_assert_eq!(parallel, sequential);
        }
    }
}
