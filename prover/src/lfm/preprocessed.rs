//! The preprocessed-column checks the WHIR verifier owes, in the machine.
//!
//! On the multilinear path nothing but `check_preprocessed` says a preprocessed
//! table is the one the program implies: the commitment binds a prover to the
//! columns it committed, not to the right ones. The host discharges that by
//! rebuilding each column and evaluating its multilinear extension at the
//! reduced point — `2^n` per column, and a real epoch carries 16,777,888 such
//! fold steps. A verifier that is itself proven cannot pay it: BITWISE alone is
//! eleven columns of 2^20, about 23 M straight-line rows, four times the whole
//! of today's recursion wrap.
//!
//! It does not have to, because BITWISE's columns are **multilinear in the row
//! index**. [`crate::tables::bitwise::preprocessed_mle_at`] is the closed form
//! and carries the argument; this module is the same expression emitted, so the
//! two can be read side by side and a divergence is visible rather than
//! inferred.
//!
//! # What this leg is NOT
//!
//! It is not a substitute for binding the ELF-derived columns. DECODE's five
//! are a function of the program, not of the index, and no closed form exists
//! for them — they need a commitment whose root is pinned in the program text.
//! This leg covers the columns whose values follow from where they sit.

use crate::tables::bitwise::{NUM_PRECOMPUTED_COLS, NUM_VARS};
use crate::tables::types::FEE;

use super::builder::{Ext, LfmBuilder};

/// INSTRUCTIONS this leg emits, by construction rather than by measurement.
///
/// Every term names the shape it comes from, so a change to the emitter that
/// does not change this number is as visible as one that does. Pinned against
/// the emitter by `preprocessed_tests::the_bitwise_leg_emits_its_closed_form`.
///
/// ⚠ Instructions, not ALU rows — the two differ by the interned constant, and
/// the first draft of this function counted rows and came in one short. A
/// `Const` is an `LFM_CONST` row like any other and the emitter pays for it.
///
/// Identity operations are not emitted and so are not counted: `2^0` is that
/// same interned one, a running sum starts at its first term rather than at
/// zero, and the final `suffix` advance feeds nothing.
pub const fn bitwise_preprocessed_rows() -> usize {
    let interned_one = 1; // `1`, interned once: 2^0, every (1 − b), and each accumulator's identity
    let pow2 = 15; // 2^1..2^15 by doubling; 2^0 is that constant
    let linear = 7 + 7 + 3; // X, Y, Z: one MulAdd per bit past the first
    let logic = 4 + 7 * 7; // AND/OR/XOR; bit 0 is cheaper because its weight is one
    let is_zero = 20 + 19; // eq(0, ·): a subtract per bit, a product per bit past the first
    let prefix = 15; // the halfword's running linear form
    let z_not = 4; // the four (1 − z_bit) factors, hoisted out of the sixteen indicators
    let indicators = 16 * 3; // eq(j, Z): four factors, three products
    let sll = 1 + 15 * 3; // j = 0 needs no weight and no accumulate
    let sllc_and_suffix = 3 + 13 * 4 + 2; // j = 0 is empty, j = 15 advances nothing
    interned_one
        + pow2
        + linear
        + logic
        + is_zero
        + prefix
        + z_not
        + indicators
        + sll
        + sllc_and_suffix
}

/// ★ The eleven BITWISE preprocessed columns' multilinear extensions at one
/// point, emitted — `O(NUM_VARS)` rows where the fold is `O(2^NUM_VARS)`.
///
/// `point` is in the order [`multilinear::mle::Mle::evaluate_in`] takes it, so
/// its FIRST coordinate binds the HIGH half of the table and coordinate `i`
/// carries index bit `NUM_VARS - 1 - i`. The two run in opposite directions;
/// the reversal is written once, here and in the host closed form, and nowhere
/// else.
///
/// Returns the columns in [`crate::tables::bitwise::generate_bitwise_row`]'s
/// order: X, Y, Z, AND, OR, XOR, MSB8, MSB16, ZERO, SLL, SLLC.
pub fn emit_bitwise_preprocessed(b: &mut LfmBuilder, point: &[Ext]) -> [Ext; NUM_PRECOMPUTED_COLS] {
    assert_eq!(
        point.len(),
        NUM_VARS,
        "the point must have one coordinate per BITWISE row-index bit"
    );
    // The variable carried by index bit k.
    let v: Vec<Ext> = (0..NUM_VARS).map(|k| point[NUM_VARS - 1 - k]).collect();
    let one = b.ext_const(&FEE::one());

    // Powers of two by doubling. 2^0 is the constant itself, so a weight of one
    // multiplies nothing.
    let mut pow2 = Vec::with_capacity(16);
    pow2.push(one);
    for i in 1..16 {
        let prev = pow2[i - 1];
        pow2.push(b.eadd(prev, prev));
    }

    // X, Y, Z: linear forms over disjoint bit ranges, accumulated from the
    // lowest bit so the first term needs no weight.
    let mut linear = [one; 3];
    for (slot, (lo, len)) in [(0usize, 8usize), (8, 8), (16, 4)].into_iter().enumerate() {
        let mut acc = v[lo];
        for i in 1..len {
            acc = b.emul_add(pow2[i], v[lo + i], acc);
        }
        linear[slot] = acc;
    }
    let [x, y, z] = linear;

    // AND, OR, XOR: one product of two DISTINCT variables per bit, so each is
    // degree one in each of them.
    let mut and = one;
    let mut or = one;
    let mut xor = one;
    for i in 0..8 {
        let (a, c) = (v[i], v[8 + i]);
        let ab = b.emul(a, c);
        let sum = b.eadd(a, c);
        let or_bit = b.esub(sum, ab);
        let xor_bit = b.esub(or_bit, ab);
        if i == 0 {
            // Weight one: the accumulators start at the bit itself.
            and = ab;
            or = or_bit;
            xor = xor_bit;
        } else {
            and = b.emul_add(pow2[i], ab, and);
            or = b.emul_add(pow2[i], or_bit, or);
            xor = b.emul_add(pow2[i], xor_bit, xor);
        }
    }

    // Bit 7 of X, and bit 15 of X + 256·Y, which is bit 7 of Y.
    let msb8 = v[7];
    let msb16 = v[15];

    // ZERO is eq(0, ·): the product over every bit of (1 − b).
    let mut is_zero = one;
    for (k, vk) in v.iter().enumerate().take(NUM_VARS) {
        let term = b.esub(one, *vk);
        is_zero = if k == 0 { term } else { b.emul(is_zero, term) };
    }

    // Halfword bit i is index bit i for i < 16, so one prefix sweep serves both
    // shifts: prefix[m] = Σ_{i ≤ m} 2^i · bit(i).
    let mut prefix = Vec::with_capacity(16);
    let mut running = v[0];
    prefix.push(running);
    for i in 1..16 {
        running = b.emul_add(pow2[i], v[i], running);
        prefix.push(running);
    }

    // The four complements, hoisted: sixteen indicators share them.
    let z_not: Vec<Ext> = (0..4).map(|k| b.esub(one, v[16 + k])).collect();

    let mut sll = one;
    let mut sllc = one;
    // `suffix` holds halfword >> (16 − j) on entry to iteration j, advanced by
    // suffix_{j+1} = 2·suffix_j + bit(15 − j). It is zero at j = 0, which is
    // why SLLC starts one iteration later than SLL.
    let mut suffix = one;
    for j in 0..16usize {
        // eq(j, the four Z bits): four factors, and the first needs no product.
        let mut indicator = one;
        for k in 0..4 {
            let factor = if (j >> k) & 1 == 1 {
                v[16 + k]
            } else {
                z_not[k]
            };
            indicator = if k == 0 {
                factor
            } else {
                b.emul(indicator, factor)
            };
        }
        // (halfword << j) mod 2^16 keeps bits 0 ..= 15 − j at weight 2^(i + j).
        let shifted = if j == 0 {
            prefix[15]
        } else {
            b.emul(prefix[15 - j], pow2[j])
        };
        let term = b.emul(indicator, shifted);
        sll = if j == 0 { term } else { b.eadd(sll, term) };

        if j > 0 {
            let carried = b.emul(indicator, suffix);
            sllc = if j == 1 {
                carried
            } else {
                b.eadd(sllc, carried)
            };
        }
        // The last advance would feed nothing.
        if j < 15 {
            suffix = if j == 0 {
                v[15]
            } else {
                let doubled = b.eadd(suffix, suffix);
                b.eadd(doubled, v[15 - j])
            };
        }
    }

    [x, y, z, and, or, xor, msb8, msb16, is_zero, sll, sllc]
}
