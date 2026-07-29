//! Shared `(domain, addr)` sorted-keys uniqueness argument for the field-storage
//! tables — `FEXT_PAGE`, `FEXT_LOCAL_TO_GLOBAL`, and `GLOBAL_FIELD_MEMORY`.
//!
//! Each is a sparse table over the same key space — memory domain `∈ {3,4,5}` and a
//! 64-bit address — that emits one per-cell token on a shared bus. Two rows for the
//! same cell would emit two tokens (e.g. two zero-init tokens), letting a prover reset
//! a cell mid-run. This argument makes the keys strictly ascending so each active cell
//! appears exactly once: rows sorted by `(domain, addr)`, active rows contiguous at the
//! top, domain constrained to `{3,4,5}` and stepping by 1 or 2 on a change, and the
//! address strictly increasing within a domain via an ALU `LT` lookup (with the addr
//! limbs range-checked to `[0, 2^32)` so the lookup is sound).
//!
//! The three tables differ only in their column indices, captured by
//! [`SortedKeysLayout`]; the constraints, bus interactions, trace-fill and provider
//! collectors are identical and live here, so the soundness argument has one home.

use stark::constraints::builder::{ConstraintBuilder, RowDomain};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::table::Table;

use crate::constraints::templates::emit_is_bit;

use super::bitwise::{BitwiseOperation, BitwiseOperationType};
use super::lt::LtOperation;
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, alu_op};

/// The columns the sorted-keys uniqueness argument reads and writes. All three
/// field-storage tables provide their own indices for these logical columns.
pub struct SortedKeysLayout {
    /// Memory domain (3, 4, or 5).
    pub domain: usize,
    /// Cell address (`DWordWL`: `addr_0` = low word, `addr_1` = high word).
    pub addr_0: usize,
    pub addr_1: usize,
    /// Real-row selector / multiplicity.
    pub mu: usize,
    /// Half-word decomposition of the two addr limbs (`IsHalfword`-checked).
    pub addr0_hw_lo: usize,
    pub addr0_hw_hi: usize,
    pub addr1_hw_lo: usize,
    pub addr1_hw_hi: usize,
    /// The next row's addr limbs, copied in for the current-row-only addr-LT lookup.
    pub next_addr_0: usize,
    pub next_addr_1: usize,
    /// 1 iff this row and the next share a domain.
    pub same_dom: usize,
    /// `μ_next · same_dom`: gates the addr strict-increase LT.
    pub sel_same: usize,
}

impl SortedKeysLayout {
    /// Emit the 12 uniqueness constraints (indices 0..=11): `IS_BIT(μ)` (0), domain
    /// `∈ {3,4,5}` (1), `IS_BIT(same_dom)` (2), addr-limb recompose (3, 4), then the
    /// strict-ascending transition checks exempting the last row — `μ` non-increasing
    /// (5), `sel_same` definition (6), same-domain ⇒ equal domain (7), domain steps by
    /// 1 or 2 on a change (8), next-addr copies (9, 10) — and finally `IS_BIT(sel_same)`
    /// (11), which unlike (6) applies to *every* row.
    ///
    /// Constraint (11) is what makes `sel_same` safe as the addr-LT sender's
    /// multiplicity. On interior rows (6) already pins `sel_same = μ_next·same_dom`, a
    /// product of two bits, so it is `{0,1}` there by construction — but (6) is
    /// `except_last`-gated, leaving the last row's `sel_same` a free field element.
    /// Since it is the LT sender's `Multiplicity::Column`, a free last-row value of `−1`
    /// would let a prover cancel a forced `+1` LT claim of the same tuple (the sum-based
    /// LogUp balance permits negative multiplicities), erasing the strict-increase check
    /// and re-opening duplicate-cell / token-cycle forgeries. The ungated `IS_BIT`
    /// forbids any non-`{0,1}` last-row multiplicity (the honest fill leaves it 0), which
    /// closes that cancellation for all three field-storage tables at once.
    pub fn emit_constraints<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        &self,
        b: &mut B,
    ) {
        emit_is_bit(b, 0, self.mu, None);

        let d = b.main(0, self.domain);
        let three = b.const_base(3);
        let four = b.const_base(4);
        let five = b.const_base(5);
        b.emit_base(1, (d.clone() - three) * (d.clone() - four) * (d - five));

        emit_is_bit(b, 2, self.same_dom, None);

        let two16 = b.const_base(1 << 16);
        let a0 = b.main(0, self.addr_0);
        let a0_lo = b.main(0, self.addr0_hw_lo);
        let a0_hi = b.main(0, self.addr0_hw_hi);
        b.emit_base(3, a0 - (a0_lo + two16.clone() * a0_hi));
        let a1 = b.main(0, self.addr_1);
        let a1_lo = b.main(0, self.addr1_hw_lo);
        let a1_hi = b.main(0, self.addr1_hw_hi);
        b.emit_base(4, a1 - (a1_lo + two16.clone() * a1_hi));

        let tr = RowDomain::except_last(1);
        let one = b.one();
        let two = b.const_base(2);

        let mu_cur = b.main(0, self.mu);
        let mu_next = b.main(1, self.mu);
        let same = b.main(0, self.same_dom);
        let sel = b.main(0, self.sel_same);
        let d_cur = b.main(0, self.domain);
        let d_next = b.main(1, self.domain);

        b.emit_base_rows(5, tr, mu_next.clone() * (one.clone() - mu_cur));
        b.emit_base_rows(6, tr, sel.clone() - mu_next.clone() * same);
        b.emit_base_rows(7, tr, sel.clone() * (d_next.clone() - d_cur.clone()));

        let sel_diff = mu_next - sel;
        let delta = d_next - d_cur;
        b.emit_base_rows(8, tr, sel_diff * (delta.clone() - one) * (delta - two));

        let na0 = b.main(0, self.next_addr_0);
        let addr0_next = b.main(1, self.addr_0);
        b.emit_base_rows(9, tr, na0 - addr0_next);
        let na1 = b.main(0, self.next_addr_1);
        let addr1_next = b.main(1, self.addr_1);
        b.emit_base_rows(10, tr, na1 - addr1_next);

        emit_is_bit(b, 11, self.sel_same, None);
    }

    /// Number of constraints [`emit_constraints`](Self::emit_constraints) emits.
    pub const NUM_CONSTRAINTS: usize = 12;

    /// The uniqueness bus interactions: the `addr[i] < addr[i+1]` ALU LT on same-domain
    /// active transitions (multiplicity `sel_same`), plus the four `IsHalfword` checks
    /// pinning the addr limbs to `[0, 2^32)` (multiplicity `mu`).
    pub fn bus_interactions(&self) -> Vec<BusInteraction> {
        let is_halfword = |col: usize| {
            BusInteraction::sender(
                BusId::IsHalfword,
                Multiplicity::Column(self.mu),
                vec![direct(col)],
            )
        };
        vec![
            BusInteraction::sender(
                BusId::Alu,
                Multiplicity::Column(self.sel_same),
                vec![
                    BusValue::Packed {
                        start_column: self.addr_0,
                        packing: Packing::DWordWL,
                    },
                    BusValue::Packed {
                        start_column: self.next_addr_0,
                        packing: Packing::DWordWL,
                    },
                    BusValue::constant(alu_op::LT as u64),
                    BusValue::constant(1),
                    BusValue::constant(0),
                ],
            ),
            is_halfword(self.addr0_hw_lo),
            is_halfword(self.addr0_hw_hi),
            is_halfword(self.addr1_hw_lo),
            is_halfword(self.addr1_hw_hi),
        ]
    }

    /// Fill the uniqueness helper columns, given the caller has already set `domain`,
    /// `addr_0`/`addr_1` and `mu` on the `num_active` real rows (sorted ascending by
    /// `(domain, addr)`): the addr half-words on each real row, a valid domain (3) on
    /// the padding rows so the ungated domain constraint holds, and the cross-row
    /// `next_addr`/`same_dom`/`sel_same` helpers (the last row's transition is exempt).
    pub fn fill_trace(
        &self,
        table: &mut Table<GoldilocksField>,
        num_active: usize,
        num_rows: usize,
    ) {
        for row in 0..num_active {
            let lo = table.get(row, self.addr_0).to_raw();
            let hi = table.get(row, self.addr_1).to_raw();
            table.set_fe(row, self.addr0_hw_lo, FE::from(lo & 0xFFFF));
            table.set_fe(row, self.addr0_hw_hi, FE::from(lo >> 16));
            table.set_fe(row, self.addr1_hw_lo, FE::from(hi & 0xFFFF));
            table.set_fe(row, self.addr1_hw_hi, FE::from(hi >> 16));
        }

        for row in num_active..num_rows {
            table.set_fe(row, self.domain, FE::from(3u64));
        }

        for row in 0..num_rows - 1 {
            let next_addr_0 = *table.get(row + 1, self.addr_0);
            let next_addr_1 = *table.get(row + 1, self.addr_1);
            let cur_dom = *table.get(row, self.domain);
            let next_dom = *table.get(row + 1, self.domain);
            let next_active = *table.get(row + 1, self.mu) == FE::one();
            let same = cur_dom == next_dom;

            table.set_fe(row, self.next_addr_0, next_addr_0);
            table.set_fe(row, self.next_addr_1, next_addr_1);
            table.set_fe(
                row,
                self.same_dom,
                if same { FE::one() } else { FE::zero() },
            );
            table.set_fe(
                row,
                self.sel_same,
                if same && next_active {
                    FE::one()
                } else {
                    FE::zero()
                },
            );
        }
    }
}

fn direct(col: usize) -> BusValue {
    BusValue::Packed {
        start_column: col,
        packing: Packing::Direct,
    }
}

/// The `addr[i] < addr[i+1]` ALU LT ops the uniqueness argument needs — one per
/// same-domain consecutive pair in the sorted `(domain, addr)` cell set — which the
/// providing LT table must receive. Data-level (no column indices), so every
/// field-storage table shares it.
pub fn collect_lt(cells: impl IntoIterator<Item = (u64, u64)>) -> Vec<LtOperation> {
    let mut cells: Vec<(u64, u64)> = cells.into_iter().collect();
    cells.sort_by_key(|&(domain, addr)| (domain, addr));
    let mut ops = Vec::new();
    for pair in cells.windows(2) {
        if pair[0].0 == pair[1].0 {
            ops.push(LtOperation::new(pair[0].1, pair[1].1, false));
        }
    }
    ops
}

/// The four `IsHalfword` provider rows per touched cell (both addr limbs split into
/// half-words), matching the addr-limb range checks in [`SortedKeysLayout::bus_interactions`].
pub fn collect_bitwise(addrs: impl IntoIterator<Item = u64>) -> Vec<BitwiseOperation> {
    let mut ops = Vec::new();
    for addr in addrs {
        for word in [addr & 0xFFFF_FFFF, addr >> 32] {
            for hv in [word & 0xFFFF, (word >> 16) & 0xFFFF] {
                ops.push(BitwiseOperation::halfword(
                    BitwiseOperationType::IsHalf,
                    (hv & 0xFF) as u8,
                    ((hv >> 8) & 0xFF) as u8,
                ));
            }
        }
    }
    ops
}
