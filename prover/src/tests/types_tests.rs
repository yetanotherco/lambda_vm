use crate::tables::types::{FE, zeroed_fe_vec};

/// Guards the `zeroed_fe_vec` invariant: a calloc'd all-zero buffer
/// reinterpreted as `Vec<FE>` must equal an element-wise `FE::zero()` fill.
#[test]
fn zeroed_fe_vec_matches_fe_zero() {
    for len in [0usize, 1, 7, 64, 1024] {
        assert_eq!(zeroed_fe_vec(len), vec![FE::zero(); len], "len={len}");
    }
}
