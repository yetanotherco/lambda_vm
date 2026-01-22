#![no_main]

use arbitrary::Arbitrary;
use crypto::hash::poseidon2::{Fp, Poseidon2};
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    // Use Vec to test arbitrary lengths:
    // - Empty (edge case)
    // - < RATE (single permutation)
    // - > RATE (multi-round absorption)
    inputs: Vec<u64>,
    // Dedicated inputs for specific property tests
    a: u64,
    b: u64,
    c: u64,
}

fuzz_target!(|fuzz_data: FuzzInput| {
    let elements: Vec<Fp> = fuzz_data.inputs.iter().map(|&x| Fp::from(x)).collect();
    let a = Fp::from(fuzz_data.a);
    let b = Fp::from(fuzz_data.b);
    let c = Fp::from(fuzz_data.c);

    // 1. Variable Length & Consistency check
    // hash_vec should be consistent with hash_single (len=1) and hash_many (len!=1)
    if !elements.is_empty() {
        let vec_hash = Poseidon2::hash_vec(&elements);

        if elements.len() == 1 {
            assert_eq!(
                vec_hash,
                Poseidon2::hash_single(&elements[0]),
                "hash_vec(len=1) != hash_single"
            );
        } else {
            assert_eq!(
                vec_hash,
                Poseidon2::hash_many(&elements),
                "hash_vec(len!=1) != hash_many"
            );
        }
    } else {
        // Handle empty case:
        // In debug, hash_vec panics (assertion). In release, it returns 0.
        // We verify hash_many allows empty (padding logic) and returns non-zero.
        let many_hash = Poseidon2::hash_many(&elements);
        assert_ne!(many_hash, Fp::zero(), "hash_many([]) should be non-zero due to padding");
    }

    // 2. Domain separation: hash(a,b) != hash_many([a,b])
    // hash uses domain tag 2, hash_many uses 10* padding
    assert_ne!(
        Poseidon2::hash(&a, &b),
        Poseidon2::hash_many(&[a, b]),
        "Domain separation violated: hash(a,b) == hash_many([a,b])"
    );

    // 3. Non-commutativity (when a != b)
    if a != b {
        assert_ne!(
            Poseidon2::compress(&a, &b),
            Poseidon2::compress(&b, &a),
            "Compress should be non-commutative"
        );
    }

    // 4. Determinism
    assert_eq!(
        Poseidon2::hash(&a, &b),
        Poseidon2::hash(&a, &b),
        "Hash should be deterministic"
    );

    // 5. Domain separation: hash_single vs hash_many
    assert_ne!(
        Poseidon2::hash_single(&a),
        Poseidon2::hash_many(&[a]),
        "hash_single should differ from hash_many"
    );

    // 6. Length extension resistance
    assert_ne!(
        Poseidon2::hash_many(&[a, b]),
        Poseidon2::hash_many(&[a, b, c]),
        "Different length inputs should produce different hashes"
    );

    // 7. Prefix resistance
    assert_ne!(
        Poseidon2::hash_many(&[a, b, c]),
        Poseidon2::hash_many(&[b, c]),
        "Prefix removal should change hash"
    );

    // 8. Non-zero outputs
    assert_ne!(Poseidon2::hash(&a, &b), Fp::zero(), "Hash should not be zero");
    assert_ne!(
        Poseidon2::hash_single(&a),
        Fp::zero(),
        "Hash single should not be zero"
    );

    // 9. Collision resistance
    if a != b {
        assert_ne!(
            Poseidon2::hash_single(&a),
            Poseidon2::hash_single(&b),
            "hash_single should be collision-resistant"
        );
    }
});
