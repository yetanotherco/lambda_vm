//! Polynomial primitives the WHIR verifier's legs share.
//!
//! `eq` is the one everything reaches for: the GKR layer relation multiplies by
//! it once per layer, the stacked-evaluation weight is a product of two of
//! them per column, and every sumcheck closes against one. So it is worth
//! emitting once, in the cheapest shape, with its cost written down.

use crate::tables::types::{FE, FEE};

use super::builder::{Ext, LfmBuilder};
use super::word::{LfmWord, ext_word};

/// INSTRUCTIONS [`emit_eq_eval`] emits over `n` variables, as the FIRST leg in
/// its program.
///
/// Four rows per variable and one product to fold each into the running
/// accumulator, except the first, which *is* the accumulator. Plus the interned
/// `1`, counted the way [`super::preprocessed::bitwise_preprocessed_rows`]
/// counts it — an `LFM_CONST` row, paid once per program rather than once per
/// leg (`LfmBuilder::ext_const` interns by canonical word), so a second leg in
/// the same program costs [`eq_eval_rows_again`] and this number is an upper
/// bound there.
///
/// ⚠ The three terms sum to exactly `5n`, so a measurement of ONE leg cannot
/// tell them apart: it forces `per_variable + fold_per_variable = 5` and
/// `interned_one = fold_per_variable`, and leaves the split free. What pins the
/// split is a second leg in the same program, which pays `5n − interned_one`.
/// That is why the cost has two pins rather than one.
///
/// Pinned against the emitter by
/// `whir_poly_tests::the_eq_leg_emits_its_closed_form` (this form) and
/// `whir_poly_tests::the_interned_one_is_paid_once_per_program` (the split).
pub const fn eq_eval_rows(n: usize) -> usize {
    if n == 0 {
        // `eq` over no variables is the empty product. Nothing is emitted but
        // the constant it returns.
        return 1;
    }
    INTERNED_ONE + eq_eval_rows_again(n)
}

/// INSTRUCTIONS a FURTHER [`emit_eq_eval`] emits in a program that already has
/// one: the same rows less the constant, which is already interned.
///
/// This is the form the assembled verifier's census adds up — it emits `eq`
/// once per GKR layer, twice per stacked column and once per sumcheck close,
/// and only the first of those pays for the `1`.
pub const fn eq_eval_rows_again(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let per_variable = 4 * n; // r·x, 1−r, 1−x, and the MulAdd that joins them
    let fold = n - 1; // one product per variable past the first
    per_variable + fold
}

/// The `LFM_CONST` row holding `1`, paid once per program.
const INTERNED_ONE: usize = 1;

/// ★ `eq(r, x) = Π_i [ r_i·x_i + (1−r_i)(1−x_i) ]`, emitted.
///
/// The factor is written as `(1−r)(1−x) + r·x` rather than as the sum of two
/// products, so the join is one `MulAdd` and a variable costs four rows instead
/// of five. `MulAdd` is the same one row as `Mul` on `LFM_XALU`, which is what
/// makes the shape free to choose.
///
/// Panics if the two points differ in length — the host returns an error there,
/// but a straight-line program's lengths are emit-time constants, so a mismatch
/// is a bug in the emitter rather than a condition to carry at runtime.
pub fn emit_eq_eval(b: &mut LfmBuilder, r: &[Ext], x: &[Ext]) -> Ext {
    assert_eq!(
        r.len(),
        x.len(),
        "eq's two points must have the same number of variables"
    );
    let one = b.ext_const(&FEE::one());
    let mut acc = one;
    for (i, (&ri, &xi)) in r.iter().zip(x).enumerate() {
        let rx = b.emul(ri, xi);
        let not_r = b.esub(one, ri);
        let not_x = b.esub(one, xi);
        let term = b.emul_add(not_r, not_x, rx);
        acc = if i == 0 { term } else { b.emul(acc, term) };
    }
    acc
}

/// INSTRUCTIONS one [`emit_sumcheck_round`] emits at degree `d`, once the
/// program has interned its constants.
///
/// One subtract to recover `g(0)`; `d(d+1)/2` subtracts for the forward
/// difference triangle (`d` at the first level down to one at the last, and
/// every entry of it is used); `d − 1` `MulAdd`s for the Newton steps `u_1 ..
/// u_{d−1}` (`u_0` is the challenge itself and costs nothing); and `d` more for
/// the nest that folds them. So **7 rows at degree 2, 12 at degree 3, and 42 at
/// degree 7** — the degrees the verifier reaches are 3 in the GKR ladder
/// (`gkr.rs:522`), 2 in `claim_reduce` (`claim_reduce.rs:291`) and in the WHIR
/// chain (`whir_chain.rs:991`), and the worst rule's degree in the main batched
/// sumcheck (`batch.rs:431`, `degree_of`), which is table-dependent.
///
/// Pinned as a SLOPE — one more round in the same program — by
/// `whir_poly_tests::a_sumcheck_round_costs_its_closed_form`, so the constants
/// below cancel out of it and are pinned separately.
pub const fn sumcheck_round_rows(degree: usize) -> usize {
    let d = clamp_degree(degree);
    let recover_g0 = 1;
    let triangle = d * (d + 1) / 2;
    let newton_steps = d - 1; // u_0 is the challenge; only u_1 .. u_{d−1} cost
    let nest = d; // one MulAdd per level of the nest
    recover_g0 + triangle + newton_steps + nest
}

/// `LFM_CONST` rows a degree-`d` round interns, paid once per program however
/// many rounds share the degree: the pair `(1/(j+1), −j/(j+1))` for each Newton
/// step `u_1 .. u_{d−1}`.
///
/// ⛔⛔ **NEVER SUM THIS ACROSS ROUNDS OR LEGS**, and that is not a style note —
/// it is the reason a program-level pool could not be formed from it for as long
/// as this function was the only one here. Counts ADD where values MERGE: the
/// builder interns on the canonical word, so a degree-13 round and a degree-3
/// round in one program SHARE the steps `u_1, u_2`, and adding their counts
/// charges those four words twice. There is no scalar that composes.
///
/// ⇒ The program-level answer is [`sumcheck_round_constants`] at the program's
/// MAXIMUM degree — see that function for why the max alone is the whole pool.
/// This form remains correct for exactly one thing: what ONE round of this
/// degree would intern in a program containing nothing else.
pub const fn sumcheck_round_consts(degree: usize) -> usize {
    2 * (clamp_degree(degree) - 1)
}

/// ★★ THE NEWTON PAIRS A DEGREE-`d` ROUND INTERNS, BY VALUE — and, at a
/// program's maximum degree, THE WHOLE PROGRAM'S NEWTON POOL.
///
/// ★ **THE NESTING LAW, which is what makes one call enough.** Step `j` interns
/// `(1/(j+1), −j/(j+1))` and a degree-`d` round runs steps `u_1 .. u_{d−1}`, so
/// the set for `d` is a SUBSET of the set for any larger degree. The union over
/// every round of every leg is therefore exactly the set for the largest degree
/// among them, and no summation is involved anywhere:
///
/// ```text
///     pool(program) = sumcheck_round_constants(max over rounds of d)
/// ```
///
/// ⚠ **`≤ 2(d − 1)`, NOT `=`, AND THIS RETURNS A SET.** These are field
/// elements: nothing forbids `1/(a+1) == −b/(b+1)` for some pair in Goldilocks,
/// and a collision would make the true pool smaller than the count form says.
/// So the words are deduplicated here and a caller compares SETS. The identity
/// against [`sumcheck_round_consts`] is worth asserting precisely because it is
/// a claim that can fail rather than a restatement.
///
/// ⚠ `u_0` is the challenge itself and interns nothing, which is why the range
/// starts at 1 — a form that started at 0 would name `1/1` and `−0/1`, the
/// already-interned `one` and `zero`, and over-count every program by two.
pub fn sumcheck_round_constants(degree: usize) -> Vec<LfmWord> {
    let d = clamp_degree(degree);
    let mut words: Vec<LfmWord> = Vec::new();
    for j in 1..d {
        for value in newton_step_constants(j) {
            let word = ext_word(&value);
            if !words.contains(&word) {
                words.push(word);
            }
        }
    }
    words
}

/// ★ The largest degree whose Newton set a pool has FULLY interned — the same
/// quantity as a program's maximum sumcheck degree, read off the PROGRAM
/// instead of off its shapes.
///
/// ⛔ TWO SOURCES FOR ONE NUMBER, WHICH IS THE POINT. A caller derives the
/// maximum degree from the shapes (`GKR_SUMCHECK_DEGREE`, `REDUCE_DEGREE`, the
/// chain's, and each table's `sumcheck_degree()`); this derives it from the
/// words the compiled program actually holds. They must agree, and a
/// disagreement is a real finding: either a leg runs at a degree no shape
/// predicts, or a form names a degree no leg reaches.
///
/// ⚠ `1` when nothing is interned, because degree 1 runs no Newton step — the
/// same clamp the emitter and the forms use, so "no steps" and "one step's
/// worth of nothing" are the same answer here as everywhere else.
pub fn interned_newton_degree(pool: &[LfmWord]) -> usize {
    let mut degree = 1usize;
    loop {
        let next = degree + 1;
        if sumcheck_round_constants(next)
            .iter()
            .all(|word| pool.contains(word))
        {
            degree = next;
        } else {
            return degree;
        }
    }
}

/// The two constants Newton step `j` interns: `(1/(j+1), −j/(j+1))`.
///
/// ★ **ONE DERIVATION, TWO CALLERS.** [`emit_newton_step`] interns exactly these
/// and [`sumcheck_round_constants`] names exactly these, so the pool a program
/// PAYS and the pool a form PREDICTS are the same expression rather than two
/// that have to be kept in step. The gap this closes existed because the only
/// form here returned a count, and a count cannot be compared against a value.
fn newton_step_constants(j: usize) -> [FEE; 2] {
    let inv = FE::from((j + 1) as u64)
        .inv()
        .expect("j + 1 is a small nonzero Goldilocks element");
    [
        FEE::new([inv, FE::zero(), FE::zero()]),
        FEE::new([
            FE::zero() - FE::from(j as u64) * inv,
            FE::zero(),
            FE::zero(),
        ]),
    ]
}

/// `verify_rounds` clamps the degree to at least one (`sumcheck.rs:366`), so a
/// leg emitted for degree 0 must cost what degree 1 costs rather than underflow
/// the forms above.
const fn clamp_degree(degree: usize) -> usize {
    if degree == 0 { 1 } else { degree }
}

/// ★ One sumcheck round's claim recursion, as `sumcheck::verify_rounds`
/// computes it (`crypto/multilinear/src/sumcheck.rs:357-397`).
///
/// The round polynomial travels as its evaluations at `1 .. d`; `g(0)` is not
/// sent, because `g(0) + g(1)` is the claim carried in. So the nodes are
/// `0 .. d` with values `[claim − e_0, e_0, …, e_{d−1}]`, and the new claim is
/// that polynomial at the round's challenge.
///
/// ★ **Why Newton rather than Lagrange.** The nodes are `0 .. d`, equally
/// spaced and known at emit time, so
///
/// ```text
///     g(r) = y_0 + u_0·(Δ¹ + u_1·(Δ² + u_2·(Δ³ + …))),   u_j = (r − j)/(j + 1)
/// ```
///
/// with `Δ^k` the forward differences of the values. `u_0 = r`, and every other
/// `u_j` is `r·(1/(j+1)) + (−j/(j+1))` — one `MulAdd` against two interned
/// constants, not a subtract and a scale. The host's `interpolate`
/// (`sumcheck.rs:71-92`) rebuilds Lagrange denominators and batch-inverts them
/// every round; that is a host inefficiency this does not copy. Newton and
/// Lagrange are the same polynomial through the same `d + 1` points, so the two
/// agree identically — and unlike a barycentric form this never divides by
/// `r − node`, so there is no special case when a challenge lands on one.
///
/// ⚠ **The challenge is an INPUT here: this leg neither absorbs nor draws.**
/// What the transcript replay owes it, stated so the two cannot drift: absorb
/// `e_0 … e_{d−1}` in that order, as field elements, then draw exactly one
/// challenge. Until the replay supplies it, a caller's challenge is hinted, and
/// a hinted challenge is a forgery by the arena rule (`builder.rs:617-620`:
/// never derive challenges from arenas) — test-only, never a verifier.
///
/// The host's length check (`evaluations.len() == degree`) has no counterpart
/// here and needs none: the emitter reads exactly `d` words at fixed offsets,
/// so a proof carrying a different count has no way to be supplied. That is the
/// `epoch_verify.rs:171-179` idiom — the absence of a second value, not an
/// assert somebody could forget.
pub fn emit_sumcheck_round(
    b: &mut LfmBuilder,
    claim: Ext,
    evaluations: &[Ext],
    challenge: Ext,
) -> Ext {
    let d = evaluations.len();
    assert!(
        d >= 1,
        "a sumcheck round sends at least one evaluation; the host clamps the degree to 1"
    );

    // `g(0)` is recovered, not sent.
    let mut level = Vec::with_capacity(d + 1);
    level.push(b.esub(claim, evaluations[0]));
    level.extend_from_slice(evaluations);

    // The difference triangle, keeping `Δ^k y_0` — the head of every level.
    let mut deltas = Vec::with_capacity(d + 1);
    deltas.push(level[0]);
    while level.len() > 1 {
        let next: Vec<Ext> = level
            .windows(2)
            .map(|pair| b.esub(pair[1], pair[0]))
            .collect();
        deltas.push(next[0]);
        level = next;
    }

    // The nest, from the innermost level out.
    let mut acc = deltas[d];
    for k in (0..d).rev() {
        let u = if k == 0 {
            challenge
        } else {
            emit_newton_step(b, challenge, k)
        };
        acc = b.emul_add(u, acc, deltas[k]);
    }
    acc
}

/// A group of rounds against a running claim, which is what every caller of
/// `sumcheck::verify_rounds` actually asks for. Returns the claim the group
/// leaves; the point is the challenges it was handed, so there is nothing to
/// return for it.
pub fn emit_sumcheck_rounds(
    b: &mut LfmBuilder,
    claim: Ext,
    rounds: &[Vec<Ext>],
    challenges: &[Ext],
) -> Ext {
    assert_eq!(
        rounds.len(),
        challenges.len(),
        "a sumcheck group draws exactly one challenge per round"
    );
    let mut current = claim;
    for (evaluations, &r) in rounds.iter().zip(challenges) {
        current = emit_sumcheck_round(b, current, evaluations, r);
    }
    current
}

/// `u_j = (r − j)/(j + 1)`, one `MulAdd` against two interned constants.
fn emit_newton_step(b: &mut LfmBuilder, r: Ext, j: usize) -> Ext {
    let [scale, shift] = newton_step_constants(j);
    let scale = b.ext_const(&scale);
    let shift = b.ext_const(&shift);
    b.emul_add(r, scale, shift)
}

/// The three kernels a shift step can read.
///
/// `eq_j` accepts the bit unchanged, `carry` accepts `x_j = 1, y_j = 0` (the
/// only way a `+1` carries out of this bit), and `no_carry` accepts
/// `x_j = 0, y_j = 1` (the only way it does not). Named rather than indexed
/// because which of them a step reads is what its cost is made of.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ShiftKernel {
    Eq,
    Carry,
    NoCarry,
}

/// One contribution a step makes: which carry state weights it, which kernel it
/// multiplies, and which carry state it lands on.
type ShiftStep = (usize, ShiftKernel, usize);

/// The contributions a step makes, given the constant's bit and which carry
/// states can be nonzero.
///
/// This IS the host's `match ((k >> t) & 1) + c` (`eq.rs:163-172`), read as
/// structure rather than as arithmetic: a zero bit with no carry in keeps the
/// bit equal and the carry out zero; a one bit with no carry in (or a zero bit
/// with a carry in) either lands and carries or lands and does not; a one bit
/// with a carry in must equal and carry.
fn shift_contributions(bit: usize, live: [bool; 2]) -> Vec<ShiftStep> {
    let mut out = Vec::new();
    for (c, &alive) in live.iter().enumerate() {
        if !alive {
            continue;
        }
        match bit + c {
            0 => out.push((c, ShiftKernel::Eq, 0)),
            1 => {
                out.push((c, ShiftKernel::NoCarry, 0));
                out.push((c, ShiftKernel::Carry, 1));
            }
            _ => out.push((c, ShiftKernel::Eq, 1)),
        }
    }
    out
}

/// INSTRUCTIONS [`emit_shift_eval`] emits over `n` variables at shift `k`,
/// once the program has interned its `1`.
///
/// Counted from the recursion's structure rather than from the emitter. At each
/// variable, from the LEAST significant end:
///
/// - the kernels that step's contributions read cost `1` for `1 − x_j` if any
///   of them is `eq` or `no_carry`, `1` for `1 − y_j` if any is `eq` or
///   `carry`, then `2` for `eq` (one `Mul` and one `MulAdd`), `1` for `carry`
///   and `1` for `no_carry`;
/// - each contribution costs one row — a `Mul` for the first one landing on a
///   state and a `MulAdd` for every later one — except a contribution weighted
///   by a state that is still the literal `1`, whose first landing is the
///   kernel wire itself and costs nothing;
/// - the two states are added at the end, which the shift's wrap is the reason
///   for, unless only one of them can be nonzero.
///
/// ★ **At `k = 0` this must be [`eq_eval_rows_again`], and that is a check and
/// not a coincidence.** With every bit zero the second carry state is never
/// reached, every step reads `eq` alone, and the recursion IS `eq`'s: four rows
/// a variable and one fold per variable past the first, `5n − 1`. The two forms
/// are derived from different code and pinned against each other.
///
/// Bits of `k` at or above `n` are never read (`eq.rs:155`), so this is a
/// function of `k mod 2^n` — pinned in the tests rather than assumed.
pub fn shift_eval_rows(n: usize, k: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut live = [true, false];
    let mut unit = [true, false];
    let mut rows = 0;

    for t in 0..n {
        let steps = shift_contributions((k >> t) & 1, live);
        let reads = |kernel: ShiftKernel| steps.iter().any(|&(_, used, _)| used == kernel);
        let uses_eq = reads(ShiftKernel::Eq);
        let uses_carry = reads(ShiftKernel::Carry);
        let uses_no_carry = reads(ShiftKernel::NoCarry);

        rows += usize::from(uses_eq || uses_no_carry); // 1 − x_j
        rows += usize::from(uses_eq || uses_carry); // 1 − y_j
        rows += 2 * usize::from(uses_eq);
        rows += usize::from(uses_carry);
        rows += usize::from(uses_no_carry);

        let mut next_live = [false; 2];
        for &(source, _, target) in &steps {
            let first = !next_live[target];
            rows += usize::from(!(first && unit[source]));
            next_live[target] = true;
        }
        live = next_live;
        // A state that has been through a step is a product of kernels, never
        // the literal one again.
        unit = [false, false];
    }

    rows + usize::from(live[0] && live[1])
}

/// ★ `shift_k(x, y)`, emitted — `multilinear::eq::shift_eval`
/// (`eq.rs:141-176`), the kernel `claim_reduce` settles a shifted column with.
///
/// `shift_k(x, y) = 1` exactly when `index(y) = index(x) + k mod 2^n`, extended
/// multilinearly. The host adds the constant `k` bit by bit from the least
/// significant end, carrying; `state[c]` is the weight of the bits handled so
/// far having produced carry `c`, and because the shift WRAPS both carries are
/// accepted at the end.
///
/// `k` is emit-time — it is a factor's frame offset, which is program structure
/// — so the emitter knows each bit and specialises: a step whose carry state
/// cannot be nonzero emits nothing for it, and the first step's weight is the
/// literal `1` and costs no multiply. That specialisation is why `k = 0`
/// collapses to exactly [`emit_eq_eval`]'s shape.
pub fn emit_shift_eval(b: &mut LfmBuilder, x: &[Ext], y: &[Ext], k: usize) -> Ext {
    assert_eq!(
        x.len(),
        y.len(),
        "a shift's two points must have the same number of variables"
    );
    let one = b.ext_const(&FEE::one());
    let n = x.len();
    if n == 0 {
        // The empty product: every index agrees with itself, at any shift.
        return one;
    }

    // `None` is the zero weight; `Some(None)` is the literal one, which costs no
    // multiply; `Some(Some(w))` is a wire.
    let mut state: [Option<Option<Ext>>; 2] = [Some(None), None];

    for t in 0..n {
        // `x.iter().zip(y).rev()`: `t` counts from the LAST variable, which is
        // the least significant bit of an index.
        let j = n - 1 - t;
        let (xj, yj) = (x[j], y[j]);
        let live = [state[0].is_some(), state[1].is_some()];
        let steps = shift_contributions((k >> t) & 1, live);
        let reads = |kernel: ShiftKernel| steps.iter().any(|&(_, used, _)| used == kernel);
        let uses_eq = reads(ShiftKernel::Eq);
        let uses_carry = reads(ShiftKernel::Carry);
        let uses_no_carry = reads(ShiftKernel::NoCarry);

        let not_x = (uses_eq || uses_no_carry).then(|| b.esub(one, xj));
        let not_y = (uses_eq || uses_carry).then(|| b.esub(one, yj));
        let eq = uses_eq.then(|| {
            let xy = b.emul(xj, yj);
            b.emul_add(
                not_x.expect("eq reads 1 − x"),
                not_y.expect("eq reads 1 − y"),
                xy,
            )
        });
        let carry = uses_carry.then(|| b.emul(xj, not_y.expect("carry reads 1 − y")));
        let no_carry = uses_no_carry.then(|| b.emul(not_x.expect("no_carry reads 1 − x"), yj));
        let kernel = |which: ShiftKernel| match which {
            ShiftKernel::Eq => eq.expect("the step reads eq"),
            ShiftKernel::Carry => carry.expect("the step reads carry"),
            ShiftKernel::NoCarry => no_carry.expect("the step reads no_carry"),
        };

        let mut next: [Option<Ext>; 2] = [None, None];
        for &(source, which, target) in &steps {
            let value = kernel(which);
            let weight = state[source].expect("a dead state makes no contribution");
            next[target] = Some(match (next[target], weight) {
                // The first landing, weighted by the literal one: the kernel
                // itself, with nothing to emit.
                (None, None) => value,
                (None, Some(w)) => b.emul(w, value),
                (Some(acc), None) => b.eadd(acc, value),
                (Some(acc), Some(w)) => b.emul_add(w, value, acc),
            });
        }
        state = [next[0].map(Some), next[1].map(Some)];
    }

    match (state[0], state[1]) {
        (Some(a), Some(c)) => {
            let a = a.expect("a state past the first step is a wire");
            let c = c.expect("a state past the first step is a wire");
            b.eadd(a, c)
        }
        (Some(only), None) | (None, Some(only)) => {
            only.expect("a state past the first step is a wire")
        }
        (None, None) => unreachable!("some carry state survives every step"),
    }
}

/// INSTRUCTIONS [`emit_challenge_powers`] emits, once the program has interned
/// its `1`: one multiply per power past the first.
///
/// ⚠ One FEWER than the host runs. `multilinear::challenge_powers`
/// (`lib.rs:60-69`) multiplies once per element and throws the last product
/// away, because it accumulates before returning the current power. The values
/// are identical; only the dead row is missing.
pub const fn challenge_powers_rows(count: usize) -> usize {
    count.saturating_sub(1)
}

/// ★ `[1, γ, γ², …]` — `multilinear::challenge_powers`, emitted.
///
/// The batching weights of every batched statement in the verifier: the three
/// rules of the main sumcheck, the factor values of `claim_reduce`, and the
/// stacked evaluation's columns. It starts at ONE and not at γ, which is what
/// makes the first term of every batch free.
pub fn emit_challenge_powers(b: &mut LfmBuilder, gamma: Ext, count: usize) -> Vec<Ext> {
    let mut powers = Vec::with_capacity(count);
    let mut acc = b.ext_const(&FEE::one());
    for i in 0..count {
        if i > 0 {
            acc = b.emul(acc, gamma);
        }
        powers.push(acc);
    }
    powers
}
