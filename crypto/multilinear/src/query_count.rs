//! ★ The query count, derived in INTEGERS.
//!
//! `ChainConfig::with_security` computed `num_queries` with `f64` `sqrt`,
//! `log2` and `ceil`. That is fine on a host and a problem everywhere else this
//! protocol is going: a field-native verifier has no floating point, and a
//! prover and verifier that disagree by one query do not fail gracefully — the
//! transcript diverges and every later challenge is different.
//!
//! So the derivation lives here, in fixed-point integer arithmetic, and the
//! `f64` version survives only as the reference the tests enumerate against
//! over the whole parameter grid.
//!
//! # The formula, unchanged
//!
//! ```text
//! rate           = 2^-log_blowup
//! proximity      = 1 - sqrt(rate) - 1/300          (the Johnson bound)
//! bits_per_query = -log2(1 - proximity)
//!                = -log2(sqrt(rate) + 1/300)
//! target         = security_bits + log2(rounds)    (a union bound over rounds)
//! num_queries    = max(ceil(max(target - grind, 0) / bits_per_query), 1)
//! ```
//!
//! ⚠ **This is not a soundness analysis and this module does not make it one.**
//! It is the conservative mirror of the parameters the univariate prover ships,
//! as `with_security`'s own doc says. Moving it to integers changes who can
//! evaluate it, not what it claims.
//!
//! # Precision, and why the answers are the same
//!
//! Everything is Q62 — 62 fractional bits in a `u128` — against `f64`'s 53 bits
//! of mantissa, so this form is strictly more precise than the one it replaces.
//! Where they could still differ is a grid point whose true ratio sits within a
//! rounding step of an integer, and `ceil` then goes either way. That is not
//! argued here, it is ENUMERATED: the grid test walks every point the protocol
//! can reach and fails on any disagreement.
//!
//! Q62 rather than Q64 for one concrete reason: the log2 loop squares its
//! running value, and a Q64 mantissa in `[1, 2)` squares to 130 bits, which a
//! `u128` does not hold. At Q62 the square fits with two bits to spare.

/// Fractional bits in the fixed-point representation.
const FRAC: u32 = 62;
/// The value `1.0`.
const ONE: u128 = 1 << FRAC;

/// Integer square root of a `u128`, by bit-by-bit restoring subtraction.
///
/// Exact: returns `floor(sqrt(n))`. Written out rather than reached for through
/// a float, which is the thing this module exists to avoid.
fn isqrt(n: u128) -> u128 {
    if n == 0 {
        return 0;
    }
    // The largest power of four not exceeding `n`.
    let mut bit: u128 = 1u128 << ((127 - n.leading_zeros()) & !1u32);
    let mut rem = n;
    let mut root: u128 = 0;
    while bit != 0 {
        if rem >= root + bit {
            rem -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    root
}

/// `log2(x / 2^FRAC)` in Q62, for `x > 0`.
///
/// The integer part comes from the leading bit; the fraction from the classic
/// squaring loop — square the mantissa, and a result at or above two both emits
/// a one bit and halves the value.
fn log2_fixed(x: u128) -> i128 {
    debug_assert!(x > 0, "log2 of zero is not a number this protocol uses");

    // Normalise the mantissa into `[1, 2)`, i.e. `[2^FRAC, 2^(FRAC+1))`.
    let bits = 128 - x.leading_zeros(); // position of the leading one, 1-based
    let exponent = bits as i128 - 1 - FRAC as i128;
    let mut mantissa = if exponent >= 0 {
        x >> (exponent as u32)
    } else {
        x << ((-exponent) as u32)
    };
    debug_assert!((ONE..ONE << 1).contains(&mantissa));

    let mut fraction: u128 = 0;
    let mut weight = ONE >> 1;
    for _ in 0..FRAC {
        // `mantissa` is in `[1, 2)`, so the square is in `[1, 4)` and fits.
        mantissa = (mantissa * mantissa) >> FRAC;
        if mantissa >= ONE << 1 {
            mantissa >>= 1;
            fraction |= weight;
        }
        weight >>= 1;
    }

    (exponent << FRAC) + fraction as i128
}

/// `sqrt(2^-log_blowup)` in Q62.
///
/// `sqrt(2^-b) * 2^62 = sqrt(2^(124 - b))`, so one integer square root does it
/// — exactly when `124 - b` is even, floored otherwise.
fn sqrt_rate(log_blowup: usize) -> u128 {
    assert!(
        log_blowup < 124,
        "a rate of 2^-{log_blowup} is not a code this protocol can use"
    );
    isqrt(1u128 << (124 - log_blowup as u32))
}

/// ★ Queries needed for `security_bits` under the Johnson bound, given the
/// round count and the proof of work spent on the query challenge.
///
/// The integer twin of what `ChainConfig::with_security` used to compute in
/// `f64`, and the one the configuration now uses.
pub fn num_queries(log_blowup: usize, rounds: usize, security_bits: u8, grind_query: u8) -> usize {
    // 1 - proximity = sqrt(rate) + 1/300.
    let one_over_300 = ONE / 300;
    let w = sqrt_rate(log_blowup) + one_over_300;

    // A rate whose `w` reached 1 would buy nothing per query. `log_blowup >= 1`
    // keeps `w <= 0.708`.
    assert!(w < ONE, "each query must buy a positive number of bits");
    let bits_per_query = -log2_fixed(w);
    debug_assert!(bits_per_query > 0);

    let rounds = rounds.max(1);
    let target = ((security_bits as i128) << FRAC) + log2_fixed((rounds as u128) << FRAC);
    let left = (target - ((grind_query as i128) << FRAC)).max(0);

    // Both sides are Q62, so the ratio is a plain integer one. Written out
    // rather than `div_ceil`, which is unstable for `i128`; both operands are
    // known non-negative here (`left` is clamped, `bits_per_query` asserted
    // positive), so the rounding has no sign case to get wrong.
    let queries = (left + bits_per_query - 1) / bits_per_query;
    (queries as usize).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `f64` derivation this replaces, verbatim from `with_security` as it
    /// stood — the reference, kept only so the integer form can be checked
    /// against it.
    fn f64_reference(
        log_blowup: usize,
        rounds: usize,
        security_bits: u8,
        grind_query: u8,
    ) -> usize {
        let rounds = rounds.max(1);
        let rate = 1.0 / (1u64 << log_blowup) as f64;
        let proximity = 1.0 - rate.sqrt() - 1.0 / 300.0;
        let bits_per_query = -(1.0 - proximity).log2();

        let target = security_bits as f64 + (rounds as f64).log2();
        let left = (target - grind_query as f64).max(0.0);
        (left / bits_per_query).ceil().max(1.0) as usize
    }

    /// ★★ The whole realistic grid, enumerated. Roughly 4.3 million points.
    ///
    /// Not a spot check and not an argument: the one way the two forms can
    /// differ is a ratio landing within a rounding step of an integer, and the
    /// only honest way to know whether that happens anywhere the protocol can
    /// reach is to look at every point it can reach.
    #[test]
    fn the_integer_derivation_agrees_with_the_f64_reference() {
        let mut checked = 0u64;
        for log_blowup in 1..=4usize {
            for rounds in 1..=64usize {
                for security_bits in 0..=255u8 {
                    for grind_query in 0..=64u8 {
                        let got = num_queries(log_blowup, rounds, security_bits, grind_query);
                        let want = f64_reference(log_blowup, rounds, security_bits, grind_query);
                        assert_eq!(
                            got, want,
                            "blowup 2^{log_blowup}, {rounds} rounds, {security_bits} bits, \
                             grind {grind_query}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, 4 * 64 * 256 * 65, "the grid must be fully walked");
    }

    /// ✓ The grid above is worth walking only if the answers vary across it.
    #[test]
    fn the_grid_is_not_one_answer_everywhere() {
        let mut seen = std::collections::BTreeSet::new();
        for log_blowup in 1..=4usize {
            for security_bits in [0u8, 64, 128, 255] {
                for grind_query in [0u8, 20, 64] {
                    seen.insert(num_queries(log_blowup, 6, security_bits, grind_query));
                }
            }
        }
        assert!(
            seen.len() > 10,
            "only {} distinct counts across the sample: the grid is degenerate",
            seen.len()
        );
    }

    /// ★ The shipped posture, pinned to its literal: blowup 4, 128 bits, 20
    /// bits of query grinding, one round — the same 110 the univariate
    /// prover's own accounting gives.
    #[test]
    fn the_shipped_posture_is_110_at_one_round() {
        assert_eq!(num_queries(2, 1, 128, 20), 110);
    }

    /// ★ And 112 / 113 at the round counts a real proof reaches.
    #[test]
    fn the_shipped_posture_is_112_then_113_as_the_rounds_grow() {
        for rounds in 4..=7 {
            assert_eq!(num_queries(2, rounds, 128, 20), 112, "{rounds} rounds");
        }
        for rounds in 8..=15 {
            assert_eq!(num_queries(2, rounds, 128, 20), 113, "{rounds} rounds");
        }
    }

    /// More grinding buys fewer queries, and a wider blowup buys more per
    /// query — the two monotonicities the formula is supposed to have, checked
    /// rather than assumed.
    #[test]
    fn the_count_moves_the_way_the_parameters_say_it_should() {
        let at = |b, g| num_queries(b, 6, 128, g);
        assert!(
            at(2, 30) < at(2, 20),
            "grinding must reduce the query count"
        );
        assert!(
            at(3, 20) < at(2, 20),
            "a wider blowup must buy more per query"
        );
        assert!(at(4, 20) < at(3, 20));
        for g in 0..64u8 {
            assert!(
                at(2, g + 1) <= at(2, g),
                "the count must not rise with grinding at {g}"
            );
        }
    }

    /// The floor: a configuration that has already ground past its target still
    /// checks one position.
    #[test]
    fn at_least_one_query_is_always_drawn() {
        assert_eq!(num_queries(2, 1, 0, 64), 1);
        assert_eq!(num_queries(2, 1, 10, 64), 1);
    }

    /// `isqrt` against the definition: `r^2 <= n < (r+1)^2`.
    #[test]
    fn the_integer_square_root_is_the_floor_of_the_real_one() {
        for n in [0u128, 1, 2, 3, 4, 5, 99, 100, 101, 1 << 40, (1 << 62) + 7] {
            let r = isqrt(n);
            assert!(r * r <= n, "isqrt({n}) = {r} is too large");
            assert!((r + 1).checked_mul(r + 1).is_none_or(|s| s > n));
        }
        // The shapes `sqrt_rate` actually asks for.
        for b in 1..=4u32 {
            let n = 1u128 << (124 - b);
            let r = isqrt(n);
            assert!(r * r <= n && (r + 1) * (r + 1) > n, "blowup 2^{b}");
        }
    }

    /// `log2_fixed` against `f64::log2` — a different algorithm for the same
    /// number, to within the precision the fixed point carries.
    #[test]
    fn the_fixed_point_log2_agrees_with_the_floating_one() {
        for v in [
            1.0f64,
            1.5,
            2.0,
            3.0,
            7.0,
            64.0,
            0.5,
            0.25,
            0.1,
            0.708,
            1.0 / 300.0,
        ] {
            let x = (v * (ONE as f64)) as u128;
            let got = log2_fixed(x) as f64 / ONE as f64;
            let want = (x as f64 / ONE as f64).log2();
            assert!((got - want).abs() < 1e-15, "log2({v}): {got} vs {want}");
        }
    }

    /// ✓ Exact on the powers of two, where the answer is an integer and any
    /// drift in the squaring loop would show as a fraction.
    #[test]
    fn the_fixed_point_log2_is_exact_on_powers_of_two() {
        for k in -30i32..=30 {
            let x = if k >= 0 { ONE << k } else { ONE >> (-k) };
            assert_eq!(
                log2_fixed(x),
                (k as i128) << FRAC,
                "log2(2^{k}) must be exactly {k}"
            );
        }
    }
}
