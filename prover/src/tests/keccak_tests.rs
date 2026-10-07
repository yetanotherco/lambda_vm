//! Tests for the KECCAK (core) table.

// Soundness regression: μ must be a bit. Every KECCAK constraint and bus send is
// weighted by μ, so μ = 2 or −1 was indistinguishable from μ = 1 at the
// constraint level while on the bus a μ = −1 row inverts a range check (carrier).
// KECCAK does not deduplicate (one row per hash, μ ∈ {0,1}), so `MuIsBit`
// (μ·(1−μ)=0) is the right fix and forbids μ = −1.
//
// DEFENSIVE: this is a constraint-presence regression (μ ∉ {0,1} is rejected), not
// an end-to-end forgery. The carrier was confirmed structurally, not demonstrated.

use crate::tables::keccak::{KeccakConstraints, KeccakOperation, cols, generate_keccak_trace};
use crate::tables::types::FE;
use crate::test_utils::{busless_air, validate_busless};

#[test]
fn mu_is_bit_constrained() {
    let op = KeccakOperation {
        timestamp: 7,
        state_addr: 0x1000,
        input: [0x0102030405060708u64; 25],
        output: [0u64; 25],
    };
    let honest = generate_keccak_trace(std::slice::from_ref(&op));
    let air = busless_air(cols::NUM_COLUMNS, KeccakConstraints);

    assert!(
        validate_busless(&air, &honest),
        "honest trace must validate"
    );
    assert_eq!(*honest.main_table.get(0, cols::MU), FE::one());

    for bad in [FE::from(2u64), -FE::one()] {
        let mut t = honest.clone();
        t.main_table.set(0, cols::MU, bad);
        assert!(
            !validate_busless(&air, &t),
            "μ ∉ {{0,1}} must be rejected by MuIsBit"
        );
    }
}
