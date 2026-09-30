//! The row order of the six deduplicated tables: LT, BRANCH, BYTEWISE, EQ, MUL
//! and DVRM.
//!
//! Each of them merges equal operations into one row, with the multiplicities
//! summed, through a `std::collections::HashMap`. Laid out in the map's
//! iteration order, the rows followed `RandomState`'s seed, which is new for
//! every map: two builds of one run's traces differed (FAST job 247: `add.elf`'s
//! LT[0]), and so did their proofs. No RV64 proof was reproducible, and a byte
//! gate could compare program ids only, never whole proofs.
//!
//! [`unique_rows`] sorts the unique rows by the operation's key instead: the
//! derived `Ord`, the struct's fields in declaration order. The heights, the
//! multiplicities and the multiset of rows are unchanged. None of the six AIRs
//! reads a row other than its own (every constraint is on offset 0; the LogUp
//! running sum is rebuilt from whatever order the main trace has), so every
//! order is a valid trace and the verifier is unchanged. Only which permutation
//! the prover commits to becomes a function of the run.
//!
//! `LAMBDA_VM_CANONICAL_ROWS=0` restores the map's iteration order, for an A/B
//! against the order before the switch.

use std::collections::HashMap;
use std::sync::OnceLock;

/// `LAMBDA_VM_CANONICAL_ROWS`: unset, empty or `1` (the default) sorts the six
/// tables' unique rows by key; `0` lays them out in `HashMap` iteration order,
/// as before the switch. Anything else stops the run.
pub const CANONICAL_ROWS_ENV: &str = "LAMBDA_VM_CANONICAL_ROWS";

/// [`CANONICAL_ROWS_ENV`] for a raw value.
pub fn canonical_rows_setting(raw: Option<&str>) -> bool {
    match raw.map(str::trim) {
        None | Some("") | Some("1") => true,
        Some("0") => false,
        Some(other) => panic!("{CANONICAL_ROWS_ENV} must be 0 or 1, got {other:?}"),
    }
}

/// Whether the six tables lay their unique rows out sorted by key: read once
/// per process from [`CANONICAL_ROWS_ENV`], with a banner on stderr. There is
/// no in-process override: a test that needs the other order runs in a
/// process of its own under `LAMBDA_VM_CANONICAL_ROWS=0`, so no test can flip
/// the order under another test's traces.
pub fn canonical_rows() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    *ENV.get_or_init(|| {
        let on = canonical_rows_setting(std::env::var(CANONICAL_ROWS_ENV).ok().as_deref());
        let line = if on {
            "[prover] dedup table rows (LT, BRANCH, BYTEWISE, EQ, MUL, DVRM): sorted by key \
             (the default)\n"
                .to_string()
        } else {
            format!(
                "[prover] dedup table rows (LT, BRANCH, BYTEWISE, EQ, MUL, DVRM): HashMap \
                 iteration order ({CANONICAL_ROWS_ENV}=0)\n"
            )
        };
        use std::io::Write;
        let _ = std::io::stderr().write_all(line.as_bytes());
        on
    })
}

/// The unique rows of a table deduplicated through `map`, in the order the
/// table lays them out: ascending by key under [`canonical_rows`] (the
/// default), the map's iteration order otherwise.
pub(crate) fn unique_rows<K: Ord, M>(map: HashMap<K, M>) -> Vec<(K, M)> {
    unique_rows_in(map, canonical_rows())
}

fn unique_rows_in<K: Ord, M>(map: HashMap<K, M>, canonical: bool) -> Vec<(K, M)> {
    let mut rows: Vec<(K, M)> = map.into_iter().collect();
    if canonical {
        // A map's keys are distinct, so an unstable sort is still one order.
        rows.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_is_canonical_unless_exactly_zero() {
        assert!(canonical_rows_setting(None));
        assert!(canonical_rows_setting(Some("")));
        assert!(canonical_rows_setting(Some(" 1 ")));
        assert!(!canonical_rows_setting(Some("0")));
        assert!(!canonical_rows_setting(Some("0\n")));
    }

    #[test]
    #[should_panic(expected = "must be 0 or 1")]
    fn the_switch_refuses_anything_else() {
        canonical_rows_setting(Some("yes"));
    }

    fn map_of(keys: impl Iterator<Item = u64>) -> HashMap<(u64, bool), u64> {
        let mut map = HashMap::new();
        for k in keys {
            *map.entry((k.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 7, k % 3 == 0))
                .or_insert(0) += 1;
        }
        map
    }

    /// The canonical rows are the map's entries, each once, ascending by key.
    #[test]
    fn canonical_rows_are_the_entries_sorted_by_key() {
        let map = map_of((0..600).map(|i| i % 257));
        let mut expected: Vec<_> = map.iter().map(|(k, m)| (*k, *m)).collect();
        expected.sort();
        let rows = unique_rows_in(map, true);
        assert_eq!(rows, expected);
        assert!(rows.windows(2).all(|w| w[0].0 < w[1].0));
        assert_eq!(rows.iter().map(|(_, m)| m).sum::<u64>(), 600);
    }

    /// The control for the tests that call this order reproducible: two maps
    /// with the same entries, inserted in the same order, iterate in different
    /// orders (each `HashMap::new()` draws new `RandomState` keys), and the
    /// canonical order erases the difference.
    #[test]
    fn the_hash_map_order_differs_between_maps_and_the_canonical_order_does_not() {
        let a = map_of(0..256);
        let b = map_of(0..256);
        assert_eq!(a, b);
        let (ha, hb) = (
            unique_rows_in(a.clone(), false),
            unique_rows_in(b.clone(), false),
        );
        assert_ne!(
            ha, hb,
            "two maps of 256 entries iterated alike: the control cannot tell orders apart"
        );
        assert_eq!(unique_rows_in(a, true), unique_rows_in(b, true));
    }
}
