//! The exported S3 vectors (FRI.md §10 (c), (d)) under the production RPX pin,
//! written next to the Keccak/Blake3 ones in
//! `crypto/stark/tests/vectors/zf_fri/` (the stark crate cannot name
//! `RpxStarkHash`). Regenerated in memory and required byte-equal to the
//! checked-in files; regenerate after a deliberate format change:
//! `cargo test -p lambda-vm-prover --lib tests::zf_rpx_vectors::write_vectors -- --ignored`.

use stark::fri::vectors::{VectorFile, check_or_write, leaf_digests_json, proof_vectors};

use crate::lfm::algebraic_commit::RpxStarkHash;

fn all() -> Vec<VectorFile> {
    let mut v = vec![leaf_digests_json::<RpxStarkHash>("rpx")];
    v.extend(proof_vectors::<RpxStarkHash>("rpx"));
    v
}

#[test]
fn rpx_vectors_are_current() {
    let files = all();
    assert_eq!(files.len(), 1 + 5 * 2);
    let bad = check_or_write(&files, false);
    assert!(
        bad.is_empty(),
        "stale or missing vector files {bad:?}; regenerate with \
         `cargo test -p lambda-vm-prover --lib tests::zf_rpx_vectors::write_vectors -- --ignored`"
    );
}

#[test]
#[ignore = "writes crypto/stark/tests/vectors/zf_fri"]
fn write_vectors() {
    assert!(check_or_write(&all(), true).is_empty());
}
