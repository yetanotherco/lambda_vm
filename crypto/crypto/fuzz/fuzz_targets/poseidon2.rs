#![no_main]

use arbitrary::Arbitrary;
use crypto::hash::poseidon2::{Fp, Poseidon2};
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    a: u64,
    b: u64,
    c: u64,
}

fuzz_target!(|input: FuzzInput| {
    let a = Fp::from(input.a);
    let b = Fp::from(input.b);
    let c = Fp::from(input.c);

    // 1. Domain separation: hash(a,b) != hash_many([a,b])
    assert_ne!(
        Poseidon2::hash(&a, &b),
        Poseidon2::hash_many(&[a, b]),
        "Domain separation violated"
    );

    // 2. Non-commutativity (when a != b)
    if a != b {
        assert_ne!(
            Poseidon2::compress(&a, &b),
            Poseidon2::compress(&b, &a),
            "Compress should be non-commutative"
        );
    }

    // 3. Determinism
    assert_eq!(
        Poseidon2::hash(&a, &b),
        Poseidon2::hash(&a, &b),
        "Hash should be deterministic"
    );

    // 4. hash_vec delegation for length 1
    assert_eq!(
        Poseidon2::hash_vec(&[a]),
        Poseidon2::hash_single(&a),
        "hash_vec([x]) should equal hash_single(x)"
    );

    // 5. hash_vec delegation for length 2+
    assert_eq!(
        Poseidon2::hash_vec(&[a, b, c]),
        Poseidon2::hash_many(&[a, b, c]),
        "hash_vec should equal hash_many for len >= 2"
    );

    // 6. Domain separation: hash_single vs hash_many
    assert_ne!(
        Poseidon2::hash_single(&a),
        Poseidon2::hash_many(&[a]),
        "hash_single should differ from hash_many"
    );
});
