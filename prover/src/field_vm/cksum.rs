//! The `FIELD_VM_CKSUM` chip: option B's binding of the cells that cross
//! between the halves, when the two halves are two proofs.
//!
//! One preprocessed row per crossing cell, in the order both halves agree on.
//! Every real row reads its cell from MEM (`FIELD_VM_MEM`, like any Field VM
//! read) and folds it into a Horner accumulator over Fp³,
//!
//! ```text
//!     acc_0 = x_0,   acc_{i+1} = acc_i·α + x_{i+1},   s = acc_{n−1},
//! ```
//!
//! padding rows keeping it, and the last row sends `(CKSUM_ADDR, s)` on
//! `FIELD_VM_PUBLIC`, so the verifier closes `s` as a public cell.
//!
//! `α` sits in three columns, one value down the table, and the last row
//! sends `(CKSUM_ALPHA_ADDR, α)` too: the verifier closes it at the `α` it
//! draws outside both proofs, from the guest's hash of the crossing cells and
//! the roots committed before this table ([`binding_alpha`]), so the AIR
//! itself does not depend on the proof. The guest
//! computes the same Horner sum over its copies, and the outer verifier checks
//! the two agree: two different vectors agree at a random `α` with
//! probability at most `n / p³`.

use stark::config::Commitment;
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet, RowDomain};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::proof::options::ProofOptions;
use stark::trace::TraceTable;

use crate::tables::types::{BusId, FE, FEE, GoldilocksExtension, GoldilocksField, zeroed_fe_vec};

type F = GoldilocksField;
type E = GoldilocksExtension;

/// The public-cell address `s` is published under: above every memory address.
pub const CKSUM_ADDR: u64 = 1 << 40;
/// The public-cell address `α` is published under.
pub const CKSUM_ALPHA_ADDR: u64 = CKSUM_ADDR + 1;

pub mod cols {
    /// The cell's Field VM memory address.
    pub const FADDR: usize = 0;
    pub const IS_REAL: usize = 1;
    pub const IS_FIRST: usize = 2;
    pub const IS_LAST: usize = 3;
    pub const NUM_PRECOMPUTED_COLS: usize = 4;
    pub const V0: usize = NUM_PRECOMPUTED_COLS;
    pub const ACC0: usize = V0 + 3;
    pub const AL0: usize = ACC0 + 3;
    pub const NUM_COLUMNS: usize = AL0 + 3;
}

fn height(n: usize, min_rows: usize) -> usize {
    n.next_power_of_two().max(min_rows)
}

fn fill_prep(faddr: &[u64], min_rows: usize) -> TraceTable<F, E> {
    assert!(!faddr.is_empty(), "a binding covers at least one cell");
    let h = height(faddr.len(), min_rows);
    let mut trace =
        TraceTable::new_main(zeroed_fe_vec(h * cols::NUM_COLUMNS), cols::NUM_COLUMNS, 1);
    let t = &mut trace.main_table;
    for (i, &a) in faddr.iter().enumerate() {
        t.set(i, cols::FADDR, FE::from(a));
        t.set(i, cols::IS_REAL, FE::one());
    }
    t.set(0, cols::IS_FIRST, FE::one());
    t.set(h - 1, cols::IS_LAST, FE::one());
    trace
}

/// The verifier-side columns for the multilinear path, which cannot rely on
/// the legacy preprocessed root alone to bind these addresses.
pub fn preprocessed_columns(faddr: &[u64], min_rows: usize) -> Vec<Vec<FE>> {
    let trace = fill_prep(faddr, min_rows);
    (0..cols::NUM_PRECOMPUTED_COLS)
        .map(|c| {
            (0..trace.num_rows())
                .map(|r| *trace.main_table.get(r, c))
                .collect()
        })
        .collect()
}

/// `acc·α` in Fp³ = Fp[t]/(t³ − 2), coordinates low first.
fn mul_alpha<X>(a: [X; 3], al: [X; 3], two: X) -> [X; 3]
where
    X: Clone + core::ops::Add<Output = X> + core::ops::Mul<Output = X>,
{
    let p = |i: usize, j: usize| a[i].clone() * al[j].clone();
    [
        p(0, 0) + two.clone() * (p(1, 2) + p(2, 1)),
        p(0, 1) + p(1, 0) + two * p(2, 2),
        p(0, 2) + p(1, 1) + p(2, 0),
    ]
}

/// The trace for the cells at `faddr` holding `values`, and `s`.
pub fn generate_trace(
    faddr: &[u64],
    values: &[FEE],
    alpha: FEE,
    min_rows: usize,
) -> (TraceTable<F, E>, FEE) {
    assert_eq!(faddr.len(), values.len(), "one value per crossing cell");
    let mut trace = fill_prep(faddr, min_rows);
    let s = horner(values, alpha);
    let h = trace.num_rows();
    let t = &mut trace.main_table;
    let mut acc = FEE::zero();
    for r in 0..h {
        if let Some(v) = values.get(r) {
            acc = if r == 0 { *v } else { acc * alpha + v };
            for (k, c) in v.value().iter().enumerate() {
                t.set(r, cols::V0 + k, *c);
            }
        }
        for (k, c) in acc.value().iter().enumerate() {
            t.set(r, cols::ACC0 + k, *c);
        }
        for (k, c) in alpha.value().iter().enumerate() {
            t.set(r, cols::AL0 + k, *c);
        }
    }
    (trace, s)
}

/// `Σ_i x_i·α^{n−1−i}`.
pub fn horner(values: &[FEE], alpha: FEE) -> FEE {
    values.iter().fold(FEE::zero(), |acc, v| acc * alpha + v)
}

/// The Merkle root of the preprocessed columns: the verifier rebuilds it from
/// the crossing cells' addresses.
pub fn commitment(faddr: &[u64], options: &ProofOptions, min_rows: usize) -> Commitment {
    let trace = fill_prep(faddr, min_rows);
    let columns: Vec<Vec<FE>> = (0..cols::NUM_PRECOMPUTED_COLS)
        .map(|c| {
            (0..trace.num_rows())
                .map(|r| *trace.main_table.get(r, c))
                .collect()
        })
        .collect();
    super::prove::commit_prep(&columns, options)
}

pub fn bus_interactions() -> Vec<BusInteraction> {
    let direct = |c| BusValue::Packed {
        start_column: c,
        packing: Packing::Direct,
    };
    vec![
        BusInteraction::sender(
            BusId::FieldVmMem,
            Multiplicity::Column(cols::IS_REAL),
            vec![
                direct(cols::FADDR),
                BusValue::constant(0),
                BusValue::constant(0),
                direct(cols::V0),
                direct(cols::V0 + 1),
                direct(cols::V0 + 2),
            ],
        ),
        BusInteraction::sender(
            BusId::FieldVmPublic,
            Multiplicity::Column(cols::IS_LAST),
            vec![
                BusValue::constant(CKSUM_ADDR),
                direct(cols::ACC0),
                direct(cols::ACC0 + 1),
                direct(cols::ACC0 + 2),
            ],
        ),
        BusInteraction::sender(
            BusId::FieldVmPublic,
            Multiplicity::Column(cols::IS_LAST),
            vec![
                BusValue::constant(CKSUM_ALPHA_ADDR),
                direct(cols::AL0),
                direct(cols::AL0 + 1),
                direct(cols::AL0 + 2),
            ],
        ),
    ]
}

/// The Horner steps under the `α` of the `AL` columns, which stay one value.
#[derive(Clone, Copy)]
pub struct CksumConstraints;

impl ConstraintSet<F, E> for CksumConstraints {
    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let col = |b: &B, row: usize, c: usize| -> [B::Expr; 3] {
            core::array::from_fn(|k| b.main(row, c + k))
        };
        let acc = col(b, 0, cols::ACC0);
        let v = col(b, 0, cols::V0);
        let first = b.main(0, cols::IS_FIRST);
        // The first row starts the sum at its cell.
        for k in 0..3 {
            b.emit_base(k, first.clone() * (acc[k].clone() - v[k].clone()));
        }
        // acc' = real'·(acc·α + v') + (1 − real')·acc.
        let al = col(b, 0, cols::AL0);
        let next_al = col(b, 1, cols::AL0);
        let two = b.const_base(2);
        let step = mul_alpha(acc.clone(), al.clone(), two);
        let next_acc = col(b, 1, cols::ACC0);
        let next_v = col(b, 1, cols::V0);
        let real = b.main(1, cols::IS_REAL);
        let idle = b.one() - real.clone();
        for k in 0..3 {
            let want = real.clone() * (step[k].clone() + next_v[k].clone())
                + idle.clone() * acc[k].clone();
            b.emit_base_rows(3 + k, RowDomain::except_last(1), next_acc[k].clone() - want);
        }
        for k in 0..3 {
            b.emit_base_rows(
                6 + k,
                RowDomain::except_last(1),
                next_al[k].clone() - al[k].clone(),
            );
        }
    }
}

/// The guest's hash of the crossing cells, region by region as its memory
/// holds them: width-16 Poseidon1 as an overwrite sponge, the IN and BACK
/// cell counts in lanes 12 and 13, then IN's cells and BACK's, four cells
/// (twelve felts) a permutation, each region's last chunk zero-padded; the
/// digest is lanes 0..4.
pub fn crossing_digest(in_cells: &[[u64; 3]], back_cells: &[[u64; 3]]) -> [u64; 4] {
    use crypto::hash::poseidon1_w16::{Fp, permute};
    let mut state = [Fp::zero(); 16];
    state[12] = Fp::from(in_cells.len() as u64);
    state[13] = Fp::from(back_cells.len() as u64);
    for region in [in_cells, back_cells] {
        for chunk in region.chunks(4) {
            for lane in &mut state[..12] {
                *lane = Fp::zero();
            }
            for (i, cell) in chunk.iter().enumerate() {
                for (k, x) in cell.iter().enumerate() {
                    state[3 * i + k] = Fp::from(*x);
                }
            }
            state = permute(state);
        }
    }
    core::array::from_fn(|i| state[i].canonical())
}

/// `α` for a binding: Poseidon1 over the guest's digest and MEM's main root
/// (four little-endian words, reduced), under a domain tag; lanes 0..3.
pub fn binding_alpha(c_bin: &[u64; 4], c_fld: &Commitment) -> FEE {
    use crypto::hash::poseidon1_w16::{Fp, permute};
    let mut state = [Fp::zero(); 16];
    for (i, w) in c_bin.iter().enumerate() {
        state[i] = Fp::from(*w);
    }
    for (i, chunk) in c_fld.chunks_exact(8).enumerate() {
        state[4 + i] = Fp::from(u64::from_le_bytes(chunk.try_into().expect("eight bytes")));
    }
    state[12] = Fp::from(u32::from_le_bytes(*b"CKSM") as u64);
    let out = permute(state);
    FEE::new([out[0], out[1], out[2]])
}

/// The domain tag [`field_roots_digest`] starts its sponge with.
pub const ROOTS_TAG: u64 = u32::from_le_bytes(*b"CKRT") as u64;

/// Digest the ordered roots of the non-CKSUM WHIR groups into the 32-byte
/// commitment input used by [`binding_alpha`]: width-16 Poseidon1 as an
/// overwrite sponge, the tag and the root count in lanes 12 and 13, three
/// roots (each four big-endian words, reduced: the LFM's digest of a
/// commitment) a permutation, the last chunk zero-padded; the digest is lanes
/// 0..4, little-endian. The recursion recomputes it with the LFM's `Hash16`.
pub fn field_roots_digest(roots: &[Commitment]) -> Commitment {
    use crypto::hash::poseidon1_w16::{Fp, permute};
    let mut state = [Fp::zero(); 16];
    state[12] = Fp::from(ROOTS_TAG);
    state[13] = Fp::from(roots.len() as u64);
    for chunk in roots.chunks(3) {
        for lane in &mut state[..12] {
            *lane = Fp::zero();
        }
        for (i, root) in chunk.iter().enumerate() {
            for (k, word) in root.chunks_exact(8).enumerate() {
                state[4 * i + k] =
                    Fp::from(u64::from_be_bytes(word.try_into().expect("eight bytes")));
            }
        }
        state = permute(state);
    }
    let mut out = [0u8; 32];
    for (lane, bytes) in state.iter().zip(out.chunks_exact_mut(8)) {
        bytes.copy_from_slice(&lane.canonical().to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::element::FieldElement;
    use stark::constraints::builder::ProverEvalFolder;
    use stark::frame::Frame;
    use stark::table::TableView;
    use stark::traits::TransitionEvaluationContext;

    const NUM_CONSTRAINTS: usize = 9;

    fn failing(trace: &TraceTable<F, E>, row: usize) -> Vec<usize> {
        let h = trace.num_rows();
        let get = |r: usize| -> Vec<FE> {
            (0..cols::NUM_COLUMNS)
                .map(|c| *trace.main_table.get(r % h, c))
                .collect()
        };
        let frame = Frame::<F, E>::new(vec![
            TableView::new(vec![get(row)], vec![vec![]]),
            TableView::new(vec![get(row + 1)], vec![vec![]]),
        ]);
        let no_e: Vec<FieldElement<E>> = vec![];
        let offset_e = FieldElement::<E>::zero();
        let ctx =
            TransitionEvaluationContext::new_prover(frame.as_row_frame(), &no_e, &no_e, &offset_e);
        let mut base = vec![FE::zero(); NUM_CONSTRAINTS];
        let mut ext = vec![FieldElement::<E>::zero(); NUM_CONSTRAINTS];
        let mut folder = ProverEvalFolder::new(&ctx, &mut base, &mut ext);
        CksumConstraints.eval(&mut folder);
        let last = row + 1 == h;
        base.iter()
            .enumerate()
            .filter(|(i, v)| **v != FE::zero() && !(last && *i >= 3))
            .map(|(i, _)| i)
            .collect()
    }

    fn sample(n: usize) -> (Vec<u64>, Vec<FEE>, FEE) {
        let values: Vec<FEE> = (0..n as u64)
            .map(|i| {
                FEE::new([
                    FE::from(3 * i + 1),
                    FE::from(i * i + 7),
                    FE::from(5 * i + 2),
                ])
            })
            .collect();
        let alpha = FEE::new([FE::from(11u64), FE::from(13u64), FE::from(17u64)]);
        ((0..n as u64).map(|i| 100 + 3 * i).collect(), values, alpha)
    }

    #[test]
    fn honest_rows_and_padding_satisfy_every_constraint() {
        let (faddr, values, alpha) = sample(5);
        let (trace, s) = generate_trace(&faddr, &values, alpha, 8);
        assert_eq!(s, horner(&values, alpha));
        let last = trace.num_rows() - 1;
        let acc: FEE = FEE::new(core::array::from_fn(|k| {
            *trace.main_table.get(last, cols::ACC0 + k)
        }));
        assert_eq!(acc, s, "the last row holds s");
        for row in 0..trace.num_rows() {
            assert_eq!(failing(&trace, row), Vec::<usize>::new(), "row {row}");
        }
    }

    /// A changed cell value with the honest accumulator fails the step that
    /// reads it; a changed accumulator fails too.
    #[test]
    fn a_changed_value_or_accumulator_fails() {
        let (faddr, values, alpha) = sample(5);
        let (mut trace, _) = generate_trace(&faddr, &values, alpha, 8);
        let v = *trace.main_table.get(2, cols::V0);
        trace.main_table.set(2, cols::V0, v + FE::one());
        assert!(!failing(&trace, 1).is_empty());
        let (mut trace, _) = generate_trace(&faddr, &values, alpha, 8);
        let a = *trace.main_table.get(3, cols::ACC0 + 1);
        trace.main_table.set(3, cols::ACC0 + 1, a + FE::one());
        assert!(!failing(&trace, 2).is_empty());
    }

    /// `α` changing down the table fails the step into the changed row.
    #[test]
    fn a_changed_alpha_fails() {
        let (faddr, values, alpha) = sample(5);
        let (mut trace, _) = generate_trace(&faddr, &values, alpha, 8);
        let a = *trace.main_table.get(6, cols::AL0 + 2);
        trace.main_table.set(6, cols::AL0 + 2, a + FE::one());
        assert!(failing(&trace, 5).contains(&8));
    }

    #[test]
    fn the_constraint_count_matches_what_eval_emits() {
        assert_eq!(CksumConstraints.meta().len(), NUM_CONSTRAINTS);
    }

    #[test]
    fn the_digest_depends_on_every_cell_and_the_count() {
        let cells: Vec<[u64; 3]> = (0..9u64).map(|i| [i, i + 1, i + 2]).collect();
        let d = crossing_digest(&cells[..5], &cells[5..]);
        let mut other = cells.clone();
        other[8][2] += 1;
        assert_ne!(d, crossing_digest(&other[..5], &other[5..]));
        assert_ne!(d, crossing_digest(&cells[..4], &cells[4..]));
        assert_ne!(d, crossing_digest(&cells[..5], &cells[5..8]));
    }
}
