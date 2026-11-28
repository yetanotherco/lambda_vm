//! The exported S3 and S2 vectors ((a)–(e) in the README) under Keccak and Blake3 are
//! current: regenerated in memory and byte-equal to the checked-in files in
//! `crypto/stark/tests/vectors/zf_fri/` (the RPX files: the prover crate's
//! `tests::zf_rpx_vectors`). Regenerate after a deliberate format change:
//! `cargo test -p stark --lib zf_fri_vectors::write_vectors -- --ignored`.

use crate::config::{Blake3StarkHash, KeccakStarkHash};
use crate::fri::vectors::{
    VectorFile, check_or_write, group_fold_json, leaf_digests_json, one_row_leaf_digests_json,
    one_row_proof_vectors, proof_vectors, schedules_json,
};

fn all() -> Vec<VectorFile> {
    let mut v = vec![
        schedules_json(),
        group_fold_json(),
        leaf_digests_json::<KeccakStarkHash>("keccak"),
        leaf_digests_json::<Blake3StarkHash>("blake3"),
    ];
    v.extend(proof_vectors::<KeccakStarkHash>("keccak"));
    v.extend(proof_vectors::<Blake3StarkHash>("blake3"));
    // (e) S2.
    v.push(one_row_leaf_digests_json::<KeccakStarkHash>("keccak"));
    v.push(one_row_leaf_digests_json::<Blake3StarkHash>("blake3"));
    v.extend(one_row_proof_vectors::<KeccakStarkHash>("keccak"));
    v.extend(one_row_proof_vectors::<Blake3StarkHash>("blake3"));
    v
}

#[test]
fn vectors_are_current() {
    let files = all();
    assert_eq!(files.len(), 4 + 2 * 5 * 2 + 2 + 2 * 2 * 2);
    let bad = check_or_write(&files, false);
    assert!(
        bad.is_empty(),
        "stale or missing vector files {bad:?}; regenerate with \
         `cargo test -p stark --lib zf_fri_vectors::write_vectors -- --ignored`"
    );
}

#[test]
#[ignore = "writes crypto/stark/tests/vectors/zf_fri"]
fn write_vectors() {
    assert!(check_or_write(&all(), true).is_empty());
}
