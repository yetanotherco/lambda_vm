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
use crate::tables::types::{FE, FEE, GoldilocksExtension};

use super::builder::{Ext, LfmBuilder};
use super::word::{LfmWord, ext_word};

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

// =============================================================================
// The preprocessed columns NO closed form covers
// =============================================================================

/// ⛔ The largest `num_vars` [`emit_const_mle_at`] will serve — A CAP, NOT A
/// TUNING KNOB.
///
/// That emitter is the host's own fold with the leaves interned, so it is
/// `O(2^n)`: a table whose unsettled preprocessed columns sat at twenty
/// variables would emit a million rows a column and nothing in the program would
/// say so. The two tables that DO sit at twenty variables are exactly the two
/// that never arrive here — BITWISE has [`emit_bitwise_preprocessed`], and
/// DECODE's five are settled out of band by the prepared opening, which is what
/// `settled_out_of_band` counts. What is left in a real continuation epoch is
/// KECCAK_RC at five variables and REGISTER at seven, so this sits above both
/// and far below twenty.
///
/// A new preprocessed table above the cap is an EMIT-TIME REFUSAL. That refusal
/// is the whole point: the alternative is a program that silently becomes
/// unprovable, which is a failure nobody reads until a prove does not finish.
pub const MAX_CONST_MLE_VARS: usize = 12;

/// INSTRUCTIONS the shared `eq(point, ·)` table costs over `num_vars`
/// variables.
///
/// Level 0 is ONE `Sub`: its two entries are `1 − p_0` and the wire `p_0`
/// itself, because the parent is the interned one and multiplying by it emits
/// nothing. Every later level turns each of its `2^i` entries into two — one
/// `Mul` by `p_i` for the set bit and one `Sub` taking the complement from the
/// parent — so level `i` costs `2^{i + 1}` and the tail sums to `2^n⁺¹ − 4`.
///
/// The interned `1` is not counted here: it is an `LFM_CONST` shared with every
/// other leg of the program, and this form counts operation rows.
pub const fn eq_table_rows(num_vars: usize) -> usize {
    if num_vars == 0 {
        // `eq` over no variables is the interned one, and nothing is emitted.
        return 0;
    }
    1 + (1usize << (num_vars + 1)) - 4
}

/// INSTRUCTIONS one constant column's fold against a built `eq` table costs.
///
/// One row per NONZERO entry — a `Mul` for the first and a `MulAdd` for each
/// one after, which are the same row — and a zero entry emits nothing at all,
/// which is why a sparse column is cheap. An ALL-zero column emits no operation
/// and is the interned zero.
///
/// ⚠ A coefficient of ONE is NOT special-cased, and the form says so rather
/// than the emitter hiding it: skipping the multiply for a unit coefficient
/// would save a row per such entry and cost a second shape the form has to
/// know about. The saving is a handful of rows on the two tables this serves.
pub fn const_column_rows(column: &[FE]) -> usize {
    column.iter().filter(|value| **value != FE::zero()).count()
}

/// INSTRUCTIONS [`emit_const_mle_at`] emits for one table's columns at one
/// point: the shared `eq` table, then each column's fold.
///
/// The `eq` table is built ONCE per call and every column of the table folds
/// against it, which is the structure — a table's preprocessed columns are all
/// claimed at that table's own reduced point.
pub fn const_mle_rows(columns: &[&[FE]], num_vars: usize) -> usize {
    eq_table_rows(num_vars)
        + columns
            .iter()
            .map(|column| const_column_rows(column))
            .sum::<usize>()
}

/// The `LFM_CONST` words [`emit_const_mle_at`] interns, deduplicated and BY
/// VALUE, so a caller can union them into the one pool its program has.
///
/// Three kinds: `FEE::one()`, which level 0 subtracts from and which every
/// other leg shares; each DISTINCT nonzero coefficient, embedded into the
/// extension the accumulator lives in; and `FEE::zero()` when some column is
/// entirely zero, because that column IS the interned zero.
pub fn const_mle_constants(columns: &[&[FE]]) -> Vec<LfmWord> {
    let mut words: Vec<LfmWord> = vec![ext_word(&FEE::one())];
    for column in columns {
        if column.iter().all(|value| *value == FE::zero()) {
            let zero = ext_word(&FEE::zero());
            if !words.contains(&zero) {
                words.push(zero);
            }
        }
        for value in column.iter() {
            if *value == FE::zero() {
                continue;
            }
            let word = ext_word(&value.to_extension::<GoldilocksExtension>());
            if !words.contains(&word) {
                words.push(word);
            }
        }
    }
    words
}

/// ★ `Mle::evaluate_in` over EMIT-TIME CONSTANT columns at a WIRE point —
/// `check_preprocessed`'s remaining obligation, emitted.
///
/// `check_preprocessed` (`multilinear_table.rs:959`) evaluates every
/// preprocessed column past `settled_out_of_band` at the reduced point and
/// compares it against the value that table's own argument settled on. The
/// columns are a property of the program the epoch is OF, so they are program
/// text here and only the point is a wire — which is what makes the fold a
/// straight line of interned coefficients rather than a pass over hinted data.
///
/// `point` is in [`multilinear::mle::Mle::evaluate_in`]'s order: `point[0]`
/// binds the HIGH index bit, so the `eq` table is doubled with `point[0]` most
/// significant and entry `j` is the row whose index is `j`. Getting this
/// backwards is silent on a symmetric column and wrong on every other, which is
/// why it is a named mutation rather than a comment.
///
/// Returns one value per column, in the order given.
///
/// # Panics
///
/// Above [`MAX_CONST_MLE_VARS`], and on a column whose length is not `2^n`.
/// Both are emit-time shape refusals, the `epoch_verify.rs:171-179` idiom: a
/// table of another shape has no way to be supplied to a program that emits a
/// fixed straight line.
pub fn emit_const_mle_at(b: &mut LfmBuilder, columns: &[&[FE]], point: &[Ext]) -> Vec<Ext> {
    let num_vars = point.len();
    assert!(
        num_vars <= MAX_CONST_MLE_VARS,
        "a preprocessed column at {num_vars} variables costs 2^{num_vars} rows a column; \
         it needs a closed form or a prepared opening, not this route"
    );
    let height = 1usize << num_vars;
    for column in columns {
        assert_eq!(
            column.len(),
            height,
            "a preprocessed column must have one entry per row of its table"
        );
    }

    let one = b.ext_const(&FEE::one());
    // `eq[j] = Π_i (p_i if bit i of j else 1 − p_i)`, doubled with `point[0]`
    // most significant. Level 0's parent is the interned one, so its set half is
    // the wire itself and only its clear half is emitted.
    let mut eq: Vec<Ext> = vec![one];
    for (level, &p) in point.iter().enumerate() {
        let mut next: Vec<Ext> = Vec::with_capacity(eq.len() * 2);
        for &parent in &eq {
            let set = if level == 0 { p } else { b.emul(parent, p) };
            let clear = if level == 0 {
                b.esub(one, p)
            } else {
                b.esub(parent, set)
            };
            next.push(clear);
            next.push(set);
        }
        eq = next;
    }
    debug_assert_eq!(eq.len(), height);

    columns
        .iter()
        .map(|column| {
            let mut acc: Option<Ext> = None;
            for (row, value) in column.iter().enumerate() {
                if *value == FE::zero() {
                    continue;
                }
                let coefficient = b.ext_const(&value.to_extension::<GoldilocksExtension>());
                acc = Some(match acc {
                    None => b.emul(coefficient, eq[row]),
                    Some(running) => b.emul_add(coefficient, eq[row], running),
                });
            }
            acc.unwrap_or_else(|| b.ext_const(&FEE::zero()))
        })
        .collect()
}

// =============================================================================
// The OFFSET ramp — the cross-epoch proof's page tables
// =============================================================================

/// ★ THE IDENTITY RAMP'S MULTILINEAR EXTENSION, IN CLOSED FORM.
///
/// `page::offset_column()` is `(0..DEFAULT_PAGE_SIZE).map(FE::from)` — the row
/// index itself — and it is the preprocessed column every GLOBAL_MEMORY table
/// carries (`continuation.rs:241-251`: OFFSET alone on a private-input page,
/// OFFSET and INIT on any other). Its extension is the one multilinear
/// polynomial with a one-line answer:
///
/// ```text
///   MLE(r) = Σ_i i · eq(r, i) = Σ_k 2^k · r_k
/// ```
///
/// because `Σ_i bit_k(i) · eq(r, i)` marginalises to `r_k`. So a `2^18` fold per
/// page becomes `num_vars − 1` rows, and the cross-epoch proof's thirty-five
/// page tables cost about six hundred rows between them instead of nine million.
///
/// ⚠ THE BIT ORDER IS THE WHOLE CORRECTNESS OF THIS, and it is taken from
/// [`emit_bitwise_preprocessed`] rather than assumed: index bit `k` is carried
/// by `point[num_vars − 1 − k]`, so `point[0]` is the HIGH bit. Horner from
/// `point[0]` therefore accumulates the right weights, and the gate against
/// `Mle::evaluate_in` over the real column is what says so — a reversed
/// convention is a different number at every point but the symmetric ones.
pub fn emit_offset_ramp(b: &mut LfmBuilder, point: &[Ext]) -> Ext {
    assert!(
        !point.is_empty(),
        "a ramp over 2^0 rows is the constant zero, and no page is that shape"
    );
    let two = b.ext_const(&FEE::from(2u64));
    // `point[0]` carries 2^(n-1); each step doubles what is held and adds the
    // next coordinate, so the last one lands at 2^0.
    let mut acc = point[0];
    for coordinate in &point[1..] {
        acc = b.emul_add(two, acc, *coordinate);
    }
    acc
}

/// INSTRUCTIONS [`emit_offset_ramp`] emits, CONST-FREE.
///
/// One `MulAdd` per coordinate after the first; the leading coordinate is a
/// wire the caller already holds, so it costs nothing.
pub const fn offset_ramp_rows(num_vars: usize) -> usize {
    num_vars.saturating_sub(1)
}

/// The ONE constant [`emit_offset_ramp`] interns.
///
/// By value, like [`const_mle_constants`], because an epoch's or a global's
/// program has ONE pool and thirty-five ramps share this word between them —
/// a form that charged a constant per page would over-count by thirty-four.
pub fn offset_ramp_constants() -> Vec<LfmWord> {
    vec![ext_word(&FEE::from(2u64))]
}

/// The host's own closed form, for the differential.
///
/// ⚠ A THIRD DERIVATION, deliberately: the test compares the EMITTED value
/// against `Mle::evaluate_in` over the real column (the fold being replaced) and
/// against this (the arithmetic being claimed). Comparing the emitter against
/// only this one would be two halves that share an author.
pub fn offset_ramp_at(point: &[FEE]) -> FEE {
    assert!(!point.is_empty(), "a ramp needs at least one variable");
    let two = FEE::from(2u64);
    let mut acc = point[0];
    for coordinate in &point[1..] {
        acc = &acc * &two + coordinate;
    }
    acc
}
