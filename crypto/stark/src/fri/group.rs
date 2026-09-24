//! Group-leaf FRI layers (S3): a committed layer of fold exponent `d` groups
//! `2^d` consecutive bit-reversed evaluations per leaf (FRI.md §1).
//!
//! # Why a group is a coset, and how it folds
//!
//! A layer of length `n = 2^b` on the coset `o_b·⟨ω_n⟩` stores, at position
//! `p`, the value at `o_b·ω_n^{br_b(p)}`. Bit reversal over `b` bits moves the
//! low `d` bits of `p = g·2^d + t` to the top, so the group of leaf `g` holds
//! the full fiber `x_g·⟨ω_{2^d}⟩` of `x ↦ x^{2^d}`, `x_g = o_b·ω_n^{br_{b−d}(g)}`,
//! in bit-reversed order (`point(g·2^d + t) = x_g·ω_{2^d}^{br_d(t)}`), and
//! `x_g^{2^d}` is the point at position `g` of the layer folded `d` times.
//!
//! The prover folds a committed layer with ONE challenge `ζ` as `d` successive
//! binary folds with `ζ, ζ², …, ζ^{2^{d−1}}` (the unchanged binary fold, so the
//! arity-`2^d` fold `2^d·Σ ζ^i f_i` of Haböck 2022/1216 eq. (3)). The verifier
//! runs the same `d` levels on the group alone ([`group_fold`]): at level `ℓ`
//! the pair `(2j, 2j+1)` sits at `(X, −X)`,
//! `X = x_g^{2^ℓ}·ω_{2^{d−ℓ}}^{br_{d−ℓ−1}(j)}`, and
//! `u'_j = (u_{2j} + u_{2j+1}) + ζ^{2^ℓ}·X⁻¹·(u_{2j} − u_{2j+1})`, exactly the
//! prover's `fold_evaluations_in_place` restricted to one fiber.
//!
//! # What a query checks per layer (the load-bearing checks)
//!
//! 1. the group is the leaf: hashed in full (`H::Batched` over the `2^d`
//!    values) and authenticated against the layer root at `leaf = p >> d`,
//!    with the exact path length;
//! 2. the slot check `group[p & (2^d − 1)] == v` — the round-consistency check
//!    tying this layer to the value the previous fold produced;
//! 3. the group fold with `ζ_j` gives the value at `p >> d` of the next layer.
//!
//! Dropping 1 or 2 is a soundness break; `fri_group_tests` has a named test
//! that turns red for each (M1, M2).

use crypto::merkle_tree::traits::IsStreamingLeafBackend;
use math::fft::bit_reversing::reverse_index;
use math::field::element::FieldElement;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use math::traits::AsBytes;

use crate::config::Commitment;
use crate::fri::terminal::FriFoldLayout;
use crate::merkle_caps::TreeCheck;

/// Verifier mutations for the load-bearing tests (M1, M2). Test builds only;
/// production has no switch. Thread-local: the host verifier is sequential,
/// so a test that sets one affects only its own verification.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GroupMutation {
    None,
    /// M1: skip `group[slot] == v`.
    SkipSlotCheck,
    /// M2: skip the group's Merkle authentication.
    SkipLeafAuth,
}

#[cfg(test)]
thread_local! {
    pub(crate) static GROUP_MUTATION: core::cell::Cell<GroupMutation> =
        const { core::cell::Cell::new(GroupMutation::None) };
}

#[inline]
fn mutated(_m: u8) -> bool {
    #[cfg(test)]
    {
        let m = match _m {
            1 => GroupMutation::SkipSlotCheck,
            _ => GroupMutation::SkipLeafAuth,
        };
        GROUP_MUTATION.with(|c| c.get() == m)
    }
    #[cfg(not(test))]
    {
        false
    }
}

/// `ω_{2^d}^t` for `t < 2^d`, `ω_{2^d}` the field's primitive `2^d`-th root —
/// the same root the LDE domain's `ω_N^{N/2^d}` is (both are powers of the
/// field's two-adic generator; `fri_group_tests::group_leaf_is_a_coset` checks
/// it against the prover's own domain).
pub(crate) fn roots_of_unity_table<F: IsFFTField>(d: u32) -> Option<Vec<FieldElement<F>>> {
    let w = F::get_primitive_root_of_unity(u64::from(d)).ok()?;
    let n = 1usize << d;
    let mut out = Vec::with_capacity(n);
    let mut acc = FieldElement::<F>::one();
    for _ in 0..n {
        out.push(acc.clone());
        acc = &acc * &w;
    }
    Some(out)
}

/// The group fold (see the module docs): `d = log2(group.len())` binary folds
/// of the `2^d` values of one leaf with `ζ, ζ², …`, given `x_g⁻¹` (the inverse
/// of the leaf's coset base) and `roots = roots_of_unity_table(d)`. Returns
/// the value at the leaf's position in the layer folded `d` times.
pub(crate) fn group_fold<F, E>(
    group: &[FieldElement<E>],
    zeta: &FieldElement<E>,
    x_g_inv: &FieldElement<F>,
    roots: &[FieldElement<F>],
) -> FieldElement<E>
where
    F: IsField + IsSubFieldOf<E>,
    E: IsField,
{
    let n = group.len();
    debug_assert!(n.is_power_of_two() && roots.len() == n);
    let d = n.trailing_zeros();
    let mut vals = group.to_vec();
    let mut xinv = x_g_inv.clone();
    let mut z = zeta.clone();
    for level in 0..d {
        let half = vals.len() / 2;
        // Pair j of this level: X⁻¹ = x_g^{−2^ℓ} · ω_{2^d}^{−2^ℓ·br_{d−ℓ−1}(j)}.
        for j in 0..half {
            let br = if half > 1 {
                reverse_index(j, half as u64)
            } else {
                0
            };
            // 2^ℓ·br < 2^{d−1} < n: already reduced.
            let e = br << level;
            let c = &roots[(n - e) % n];
            let x_inv_j = &xinv * c;
            let lo = &vals[2 * j];
            let hi = &vals[2 * j + 1];
            let sum = lo + hi;
            let diff = lo - hi;
            vals[j] = &sum + &(&x_inv_j * &(&z * &diff));
        }
        vals.truncate(half);
        xinv = xinv.square();
        z = z.square();
    }
    vals.swap_remove(0)
}

/// The FRI checks of one query under a group-encoded layout (every format but
/// the legacy one): per committed layer `j`, the group is authenticated at
/// `leaf = p >> d_j` by `checks[j]` — the layer tree's check, built once per
/// tree at the layout's depth with its Merkle cap (`TreeCheck`; exact path
/// length `depth − c`, query 0 the cap's owner) — with path `paths(j)`, the
/// slot check `group[p & (2^{d_j} − 1)] == v` holds, and `v` becomes the group
/// fold with layer `j`'s challenge; finally `terminal[p] == v`.
///
/// Layer `j`'s challenge is `zetas[j + 1]` for row pairs (`zetas[0]` drove the
/// uncommitted fold 0) and `zetas[j]` under one row (layer 0 is the committed
/// DEEP codeword, so no fold precedes it) — [`FriFoldLayout::num_zetas`].
///
/// * `v` / `y_inv`: the query's value at committed layer 0 and the inverse of
///   its point there (row pairs: fold 0 already applied by the caller; one
///   row: the DEEP value at `x_r` and `x_r⁻¹` — the layer-0 slot check is then
///   the input-slot check `group₀[slot] == DEEP(x_r)`);
/// * `query`: the query's position in proof order (query 0 is every capped
///   layer's owner opening);
/// * `iota`: the query's position in committed layer 0;
/// * `values`: the flat per-query group values (the proof's
///   `layers_evaluations_sym` under this encoding), length already checked by
///   the caller to be `layout.opened_values_per_query()`;
/// * `roots_tables[d]`: `roots_of_unity_table(d)` for every `d` in the schedule.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_query_groups<'p, F, E, B>(
    layout: &FriFoldLayout,
    checks: &[TreeCheck<'_>],
    query: usize,
    paths: impl Fn(usize) -> &'p [Commitment],
    values: &[FieldElement<E>],
    zetas: &[FieldElement<E>],
    iota: usize,
    mut v: FieldElement<E>,
    mut y_inv: FieldElement<F>,
    terminal_codeword: &[FieldElement<E>],
    roots_tables: &[Vec<FieldElement<F>>],
) -> bool
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
    FieldElement<E>: AsBytes + Sync + Send,
    B: IsStreamingLeafBackend<E, Node = Commitment>,
{
    if checks.len() != layout.num_committed
        || values.len() != layout.opened_values_per_query()
        || zetas.len() != layout.num_zetas()
    {
        return false;
    }
    let zeta_offset = usize::from(!layout.one_row);
    let mut index = iota;
    let mut offset = 0usize;
    let mut ok = true;
    for (j, &d) in layout.schedule.iter().enumerate() {
        let d = u32::from(d);
        let n = 1usize << d;
        let group = &values[offset..offset + n];
        offset += n;
        let leaf = index >> d;
        let slot = index & (n - 1);

        // (2) the slot check.
        if group[slot] != v && !mutated(1) {
            ok = false;
        }
        // (1) the group is the leaf, authenticated with the exact depth.
        let leaf_hash = B::hash_data_from_slices(group, &[]);
        if !checks[j].verify::<B>(query, paths(j), leaf, leaf_hash) && !mutated(2) {
            ok = false;
        }
        // (3) fold: x_g⁻¹ = y⁻¹ · ω_{2^d}^{br_d(slot)}.
        let Some(table) = roots_tables.get(d as usize) else {
            return false;
        };
        if table.len() != n {
            return false;
        }
        let br_slot = if n > 1 {
            reverse_index(slot, n as u64)
        } else {
            0
        };
        let x_g_inv = &y_inv * &table[br_slot];
        v = group_fold::<F, E>(group, &zetas[j + zeta_offset], &x_g_inv, table);
        for _ in 0..d {
            y_inv = y_inv.square();
        }
        index = leaf;
    }
    let terminal_ok = terminal_codeword.get(index).is_some_and(|t| &v == t);
    ok & terminal_ok
}
