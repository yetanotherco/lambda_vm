//! Tests for the KECCAK_RND table.

use crate::tables::keccak_rnd::*;
use crate::tables::types::*;

use executor::vm::instruction::execution::{KECCAK_RHO, keccak_f1600};

/// pi is a spec virtual variable. Verify the inlined expression
/// (rot_left[sx,sy,l_byte] + rot_right[sx,sy,r_byte]) matches the byte of
/// rho(theta) for a non-trivial state. Uses mu=0 padding rows as a trivial
/// sanity check (all zeros), then a non-zero-input round as the real test.
#[test]
fn test_pi_virtual_matches_rotate() {
    // Use a non-zero input so theta_lanes are non-trivial.
    let input = [0x0102030405060708u64; 25];
    let mut output = input;
    keccak_f1600(&mut output);
    let op = KeccakRoundOperation {
        timestamp: 42,
        input,
        output,
    };
    let trace = generate_keccak_rnd_trace(&[op]);

    // Recompute theta for round 0 in u64 to compare against virtual pi.
    let mut c = [0u64; 5];
    for x in 0..5 {
        c[x] = input[x] ^ input[x + 5] ^ input[x + 10] ^ input[x + 15] ^ input[x + 20];
    }
    let mut d = [0u64; 5];
    for x in 0..5 {
        d[x] = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
    }
    let mut theta_lanes = [0u64; 25];
    for x in 0..5 {
        for y in 0..5 {
            theta_lanes[x + 5 * y] = input[x + 5 * y] ^ d[x];
        }
    }

    for x in 0..5 {
        for y in 0..5 {
            let sx = (x + 3 * y) % 5;
            let sy = x;
            let rotated = theta_lanes[sx + 5 * sy].rotate_left(KECCAK_RHO[sx][sy]);
            for z in 0..8 {
                let (l_col, r_col) = cols::pi_src_cols(x, y, z);
                let virtual_pi = *trace.get_main(0, l_col) + *trace.get_main(0, r_col);
                let expected = FE::from((rotated >> (z * 8)) & 0xFF);
                assert_eq!(
                    virtual_pi, expected,
                    "virtual pi mismatch at ({x},{y},{z}): sx={sx}, sy={sy}"
                );
            }
        }
    }
}

// Soundness regression: μ must be a bit. The shift identities are `μ · identity`,
// and every bus send except the ARE_BYTES byte checks (multiplicity 1) is
// weighted by μ, so μ = 2, 7 or −1 used to be indistinguishable from μ = 1 at the
// constraint level while scaling those sends (and with −1 turning them into
// receives). The chip does not deduplicate, so `MuIsBit` (μ·(1−μ)=0) is the right
// fix (like SHIFT/LOAD).
//
// DEFENSIVE: this is a constraint-presence regression (μ ∉ {0,1} is rejected), not
// an end-to-end forgery.
mod mu_bit_regression {
    use crate::tables::keccak_rnd::{
        KeccakRndConstraints, KeccakRoundOperation, cols, generate_keccak_rnd_trace,
    };
    use crate::tables::types::FE;
    use crate::test_utils::{busless_air, validate_busless};
    use executor::vm::instruction::execution::keccak_f1600;

    #[test]
    fn mu_is_bit_constrained() {
        let input = [0x0102030405060708u64; 25];
        let mut output = input;
        keccak_f1600(&mut output);
        let honest = generate_keccak_rnd_trace(&[KeccakRoundOperation {
            timestamp: 42,
            input,
            output,
        }]);
        let air = busless_air(cols::NUM_COLUMNS, KeccakRndConstraints);

        assert!(
            validate_busless(&air, &honest),
            "honest trace must validate"
        );
        assert_eq!(*honest.main_table.get(0, cols::MU), FE::one());

        // μ = 2, 7, −1 are now all rejected by MuIsBit.
        for bad in [FE::from(2u64), FE::from(7u64), -FE::one()] {
            let mut t = honest.clone();
            t.main_table.set(0, cols::MU, bad);
            assert!(
                !validate_busless(&air, &t),
                "μ ∉ {{0,1}} must be rejected by MuIsBit"
            );
        }
    }
}

// Why the byte checks must not depend on μ: the θ shift identity reads
// `cxz_left` as a HALFWORD (`byte_lo + 256·byte_hi`), while ARE_BYTES
// range-checks the two bytes. So the bytes are NOT pinned by the constraints — a
// non-canonical split keeps the halfword (and the identity) unchanged while
// sending an out-of-range byte to ARE_BYTES. The bus catches it: the byte checks
// are sent with multiplicity 1 (spec #1017), so no μ can invert them and the
// out-of-range tuple stays unmatched. This test documents the constraint-level
// gap.
mod are_bytes_carrier_poc {
    use crate::tables::keccak_rnd::{
        KeccakRndConstraints, KeccakRoundOperation, cols, generate_keccak_rnd_trace,
    };
    use crate::tables::types::FE;
    use crate::test_utils::{busless_air, validate_busless};
    use executor::vm::instruction::execution::keccak_f1600;

    #[test]
    fn non_canonical_cxz_byte_passes_constraints() {
        let input = [0x0102030405060708u64; 25];
        let mut output = input;
        keccak_f1600(&mut output);
        let honest = generate_keccak_rnd_trace(&[KeccakRoundOperation {
            timestamp: 42,
            input,
            output,
        }]);
        let air = busless_air(cols::NUM_COLUMNS, KeccakRndConstraints);
        assert!(
            validate_busless(&air, &honest),
            "honest trace must validate"
        );

        // Row 0 is a real round row (μ = 1). Take the first θ halfword of x = 0.
        let lo_col = cols::cxz_left(0, 0);
        let hi_col = cols::cxz_left(0, 1);
        let lo = honest.main_table.get(0, lo_col).to_raw();
        let hi = honest.main_table.get(0, hi_col).to_raw();
        let value = lo + 256 * hi; // the halfword the identity reads

        // Non-canonical split: byte_lo = p−1 (= −1), byte_hi = (value + 1)/256.
        // Then byte_lo + 256·byte_hi = value (mod p), so the identity is unchanged,
        // but byte_lo is out of range for ARE_BYTES.
        let inv_256 = FE::from(256u64).inv().expect("256 invertible");
        let new_hi = (FE::from(value) + FE::one()) * inv_256;
        let mut forged = honest.clone();
        forged.main_table.set(0, lo_col, -FE::one());
        forged.main_table.set(0, hi_col, new_hi);

        // Sanity: the halfword is preserved.
        let lo2 = *forged.main_table.get(0, lo_col);
        let hi2 = *forged.main_table.get(0, hi_col);
        assert_eq!(lo2 + FE::from(256u64) * hi2, FE::from(value));
        // And the forged low byte is out of the byte range [0, 256).
        assert!((-FE::one()).to_raw() >= 256);

        // The AIR accepts the non-canonical bytes: the constraints read only the
        // halfword, so they never range-constrain the bytes.
        assert!(
            validate_busless(&air, &forged),
            "out-of-range cxz_left byte must still satisfy KECCAK_RND's constraints"
        );
    }
}

// The θ carry bit `Cxz_right` is IS_BIT-checked on every row, not only when
// μ = 1 (spec #1017 dropped the `cond = μ`). A padding row (μ = 0) holding
// `Cxz_right = 2` must be rejected.
mod cxz_right_bit_regression {
    use crate::tables::keccak_rnd::{
        KeccakRndConstraints, KeccakRoundOperation, cols, generate_keccak_rnd_trace,
    };
    use crate::tables::types::FE;
    use crate::test_utils::{busless_air, validate_busless};
    use executor::vm::instruction::execution::keccak_f1600;

    #[test]
    fn cxz_right_is_bit_on_padding_rows() {
        let input = [0x0102030405060708u64; 25];
        let mut output = input;
        keccak_f1600(&mut output);
        let honest = generate_keccak_rnd_trace(&[KeccakRoundOperation {
            timestamp: 42,
            input,
            output,
        }]);
        let air = busless_air(cols::NUM_COLUMNS, KeccakRndConstraints);
        assert!(
            validate_busless(&air, &honest),
            "honest trace must validate"
        );

        // 24 round rows padded to 32: the last row is padding (μ = 0).
        let pad = honest.main_table.height - 1;
        assert_eq!(*honest.main_table.get(pad, cols::MU), FE::zero());

        for bad in [FE::from(2u64), -FE::one()] {
            let mut t = honest.clone();
            t.main_table.set(pad, cols::cxz_right_bit(0, 0), bad);
            assert!(
                !validate_busless(&air, &t),
                "Cxz_right ∉ {{0,1}} must be rejected even on a μ = 0 row"
            );
        }
    }
}

// THE PRECONDITION LOCAL LOOKUPS REST ON, PINNED. The static check of local
// lookups holds RECEIVERS to verifier-fixed columns; it says nothing of a
// sender. A local sender whose multiplicity can be −1 is a receiver on
// committed values, and cancels a false tuple with every check on. So every
// local sender must have multiplicity `One` or a column the AIR holds to
// {0, 1}. KECCAK_RND, the only local sender today, sends `One` (the byte
// checks) or `Column(MU)` with MU bit-constrained (`MuIsBit`). This pins both,
// for every form that sends local lookups: a new multiplicity form, a new
// column, or the bit constraint dropped turns it red.
mod local_sender_multiplicity {
    use crate::tables::keccak_rnd::{
        KeccakRndConstraints, KeccakRndForm, KeccakRoundOperation, bus_interactions_for, cols,
        generate_keccak_rnd_trace,
    };
    use crate::tables::local_bytes::LOCAL_BUSES;
    use crate::tables::types::FE;
    use crate::test_utils::{busless_air, validate_busless};
    use executor::vm::instruction::execution::keccak_f1600;
    use stark::lookup::Multiplicity;

    /// Every form, by an exhaustive match: a new form is a compile error here
    /// until it says whether it sends local lookups.
    fn forms() -> Vec<KeccakRndForm> {
        let all = [
            KeccakRndForm::Full,
            KeccakRndForm::Lean,
            KeccakRndForm::LeanLocal,
        ];
        for form in all {
            match form {
                KeccakRndForm::Full | KeccakRndForm::Lean | KeccakRndForm::LeanLocal => {}
            }
        }
        all.to_vec()
    }

    #[test]
    fn keccak_rnds_local_senders_send_one_or_a_bit_constrained_column() {
        let mut columns: Vec<usize> = Vec::new();
        let mut local_forms = 0;
        for form in forms() {
            let local: Vec<_> = bus_interactions_for(form)
                .into_iter()
                .filter(|i| LOCAL_BUSES.contains(&i.bus_id))
                .collect();
            assert_eq!(
                local.is_empty(),
                !form.byte_lookups_local(),
                "{form:?}: local interactions exactly under a local form"
            );
            if local.is_empty() {
                continue;
            }
            local_forms += 1;
            let (mut ones, mut mus) = (0usize, 0usize);
            for interaction in &local {
                assert!(interaction.is_sender, "{form:?}: KECCAK_RND only sends");
                match interaction.multiplicity {
                    Multiplicity::One => ones += 1,
                    Multiplicity::Column(c) => {
                        mus += 1;
                        if !columns.contains(&c) {
                            columns.push(c);
                        }
                    }
                    ref other => panic!(
                        "{form:?}: a local sender's multiplicity is {other:?}: only One or a \
                         bit-constrained column keeps a send from being a receive"
                    ),
                }
            }
            assert_eq!((ones, mus), (200, 808), "{form:?}: the 1,008 byte lookups");
        }
        assert_eq!(local_forms, 1, "one form sends local lookups today");
        assert_eq!(columns, vec![cols::MU], "the one multiplicity column is MU");

        // MU is held to {0, 1} by KECCAK_RND's own constraints: a value outside
        // has no valid trace (μ = −1 would turn every μ-weighted send into a
        // receive).
        let input = [0x0102030405060708u64; 25];
        let mut output = input;
        keccak_f1600(&mut output);
        let honest = generate_keccak_rnd_trace(&[KeccakRoundOperation {
            timestamp: 7,
            input,
            output,
        }]);
        let air = busless_air(cols::NUM_COLUMNS, KeccakRndConstraints);
        assert!(
            validate_busless(&air, &honest),
            "the honest trace validates"
        );
        for bad in [-FE::one(), FE::from(2u64)] {
            let mut t = honest.clone();
            t.main_table.set(0, cols::MU, bad);
            assert!(
                !validate_busless(&air, &t),
                "MU = {bad:?} must have no valid trace"
            );
        }
    }
}
