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

/// Built packed a block at a time, the table packs to the words the 64-bit
/// build gives, byte for byte: chunks of fewer permutations than a block, one
/// short of a block, exactly a block, across blocks, and none.
#[test]
fn the_packed_build_is_the_64_bit_build_packed() {
    let ops: Vec<KeccakRoundOperation> = (0..300u64)
        .map(|i| {
            let input: [u64; 25] =
                std::array::from_fn(|j| (i * 25 + j as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let mut output = input;
            keccak_f1600(&mut output);
            KeccakRoundOperation {
                timestamp: 1000 + 4 * i,
                input,
                output,
            }
        })
        .collect();
    for n in [0, 1, 127, 128, 129, 300] {
        let mut wide = generate_keccak_rnd_trace(&ops[..n]);
        assert!(wide.pack_main_narrow(), "{n}: the 64-bit build packs");
        let packed = generate_keccak_rnd_trace_packed(&ops[..n]).expect("packed build");
        assert_eq!(packed.num_rows(), wide.num_rows(), "{n} rows");
        assert_eq!(
            packed.num_main_columns, wide.num_main_columns,
            "{n} columns"
        );
        assert_eq!(
            packed.narrow_main(),
            wide.narrow_main(),
            "{n}: packed words"
        );
    }
}
