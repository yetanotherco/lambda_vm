//! P1W16 core chip — one width-16 Poseidon1 permutation per `P1w16Permute`
//! ecall.
//!
//! Reads and writes the 16-felt state in place at `x10`, like KECCAK
//! (`keccak.rs`): an `Ecall` receiver, a `Memw` read of x10, and one combined
//! read+write `Memw` per lane with `old` = input bytes and `value` = output
//! bytes.
//!
//! Each lane's felt is the linear combination `lo + 2^32·hi` of its eight
//! range-checked bytes, so the 32 felts cost no columns. Two things make it
//! the felt the executor saw:
//!
//! - **canonicity**: with `d = hi − (2^32 − 1)` and one inverse column,
//!   `d·(1 − d·inv) = 0` and `(1 − d·inv)·lo = 0` force `lo = 0` exactly when
//!   `hi` is all ones, so the bytes encode a value below p;
//! - **rounds**: the LFM chip's compact layout (`lfm::p1w16_chip`, `x³ = a³`
//!   and `x⁷ = (x³)²·a` per S-box, the state between rounds carried as affine
//!   forms) with `μ` scaling the round constants, so an all-zero padding row
//!   satisfies it.
//!
//! ## Column layout
//!
//! | Group     | Size | Description                            |
//! |-----------|------|----------------------------------------|
//! | timestamp |    2 | DWordWL                                |
//! | addr      |    8 | State address as DWordBL               |
//! | input     |  128 | Input lanes as bytes [16][8]           |
//! | output    |  128 | Output lanes as bytes [16][8]          |
//! | state_ptr |   64 | Per-lane DWordHL addresses [16][4]     |
//! | canon_inv |   32 | Canonicity inverses, input then output |
//! | rounds    |  300 | `x³`, `x⁷` per S-box (8 full, 22 partial) |
//! | mu        |    1 | Multiplicity flag                      |

use crypto::hash::poseidon1_w16::{NUM_ROUNDS, STATE_FELTS, constants::ROUND_CONSTANTS};
use executor::vm::instruction::execution::P1W16_SYSCALL_NUMBER;
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, alu_op};
use crate::constraints::templates::{AddOperand, INV_SHIFT_32};
use crate::lfm::p1w16_chip::{Affine, Mds, mds_affine, sboxed_lanes};

/// Felts in the permuted state.
pub const LANES: usize = STATE_FELTS;

pub mod cols {
    use super::{LANES, NUM_ROUNDS, sboxed_lanes};

    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;
    pub const ADDR: usize = 2;
    pub const INPUT: usize = ADDR + 8;
    pub const OUTPUT: usize = INPUT + 8 * LANES;
    pub const STATE_PTR: usize = OUTPUT + 8 * LANES;
    /// One inverse per felt: input lanes `0..16`, output lanes `16..32`.
    pub const CANON_INV: usize = STATE_PTR + 4 * LANES;
    pub const ROUNDS: usize = CANON_INV + 2 * LANES;
    pub const MU: usize = block(NUM_ROUNDS);
    pub const NUM_COLUMNS: usize = MU + 1;

    /// First column of round `r`'s block: `x³` then `x⁷` per S-boxed lane.
    pub const fn block(r: usize) -> usize {
        let mut off = ROUNDS;
        let mut i = 0;
        while i < r {
            off += 2 * sboxed_lanes(i);
            i += 1;
        }
        off
    }

    pub const fn x3(r: usize, lane: usize) -> usize {
        block(r) + lane
    }

    pub const fn x7(r: usize, lane: usize) -> usize {
        block(r) + sboxed_lanes(r) + lane
    }

    #[inline]
    pub const fn addr(byte: usize) -> usize {
        ADDR + byte
    }

    #[inline]
    pub const fn input(lane: usize, byte: usize) -> usize {
        INPUT + lane * 8 + byte
    }

    #[inline]
    pub const fn output(lane: usize, byte: usize) -> usize {
        OUTPUT + lane * 8 + byte
    }

    #[inline]
    pub const fn state_ptr(lane: usize, hw: usize) -> usize {
        STATE_PTR + lane * 4 + hw
    }

    /// Canonicity inverse of felt `f` (`f < 16` input lane, else output lane).
    #[inline]
    pub const fn canon_inv(f: usize) -> usize {
        CANON_INV + f
    }

    /// First byte of felt `f`.
    #[inline]
    pub const fn felt(f: usize) -> usize {
        if f < LANES {
            input(f, 0)
        } else {
            output(f - LANES, 0)
        }
    }
}

#[derive(Debug, Clone)]
pub struct P1w16Operation {
    pub timestamp: u64,
    pub state_addr: u64,
    pub input: [u64; LANES],
    pub output: [u64; LANES],
}

pub fn generate_p1w16_trace(
    ops: &[P1w16Operation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let n = ops.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row, op) in ops.iter().enumerate() {
        table.set_dword_wl(row, cols::TIMESTAMP_0, op.timestamp);
        table.set_dword_bl(row, cols::addr(0), op.state_addr);
        for lane in 0..LANES {
            table.set_dword_bl(row, cols::input(lane, 0), op.input[lane]);
            table.set_dword_bl(row, cols::output(lane, 0), op.output[lane]);
            let ptr = op
                .state_addr
                .checked_add(lane as u64 * 8)
                .expect("p1w16 state address range must be validated by the executor");
            table.set_dword_hl(row, cols::state_ptr(lane, 0), ptr);
        }
        table.set_fe(row, cols::MU, FE::one());
        fill_rounds(table, row, &op.input, &op.output);
    }

    // Padding rows are all zero except the canonicity inverses: `hi = 0` there,
    // and `(1 − d·inv)·lo = 0` holds for any inverse, but `d·(1 − d·inv) = 0`
    // needs the real one.
    let pad_inv = canon_inverse(0);
    for row in n..num_rows {
        for f in 0..2 * LANES {
            table.set_fe(row, cols::canon_inv(f), pad_inv);
        }
    }

    trace
}

/// `1/(hi − (2^32 − 1))`, or zero when `hi` is all ones.
fn canon_inverse(hi: u32) -> FE {
    let d = FE::from(hi as u64) - FE::from(u32::MAX as u64);
    d.inv().unwrap_or(FE::zero())
}

/// The canonicity inverses and the S-box witness for one row. `output` only
/// feeds the inverses: the last constraints tie it to the rounds.
fn fill_rounds<T: VmTable>(table: &mut T, row: usize, input: &[u64; LANES], output: &[u64; LANES]) {
    for (f, w) in input.iter().chain(output).enumerate() {
        table.set_fe(row, cols::canon_inv(f), canon_inverse((w >> 32) as u32));
    }
    let mut s: [FE; LANES] = input.map(FE::from);
    for (r, rc) in ROUND_CONSTANTS.iter().enumerate() {
        let a: [FE; LANES] = core::array::from_fn(|i| s[i] + FE::from(rc[i]));
        let mut f = a;
        for lane in 0..sboxed_lanes(r) {
            let x3 = a[lane] * a[lane] * a[lane];
            let x7 = x3 * x3 * a[lane];
            table.set_fe(row, cols::x3(r, lane), x3);
            table.set_fe(row, cols::x7(r, lane), x7);
            f[lane] = x7;
        }
        s = crypto::hash::poseidon1_w16::mds(&f);
    }
}

fn packed(column: usize) -> BusValue {
    BusValue::Packed {
        start_column: column,
        packing: Packing::Direct,
    }
}

/// `Σ col[start + i]·step^i` over `n` columns.
fn linear(start: usize, n: usize, step: i64) -> BusValue {
    BusValue::linear(
        (0..n)
            .map(|i| LinearTerm::Column {
                coefficient: step.pow(i as u32),
                column: start + i,
            })
            .collect(),
    )
}

pub fn bus_interactions() -> Vec<BusInteraction> {
    let mu = || Multiplicity::Column(cols::MU);
    let mut interactions = Vec::new();

    // 1. Ecall receiver: (ts_lo, ts_hi, syscall_lo32, syscall_hi32).
    interactions.push(BusInteraction::receiver(
        BusId::Ecall,
        mu(),
        vec![
            packed(cols::TIMESTAMP_0),
            packed(cols::TIMESTAMP_1),
            BusValue::constant(P1W16_SYSCALL_NUMBER & 0xFFFF_FFFF),
            BusValue::constant(P1W16_SYSCALL_NUMBER >> 32),
        ],
    ));

    // 2. Memw read of x10 (base address 20), old = value = addr.
    {
        let addr_lo = linear(cols::addr(0), 4, 256);
        let addr_hi = linear(cols::addr(4), 4, 256);
        let mut values = Vec::with_capacity(24);
        values.push(addr_lo.clone());
        values.push(addr_hi.clone());
        values.extend((2..8).map(|_| BusValue::constant(0)));
        values.push(BusValue::constant(1));
        values.push(BusValue::constant(20));
        values.push(BusValue::constant(0));
        values.push(addr_lo);
        values.push(addr_hi);
        values.extend((2..8).map(|_| BusValue::constant(0)));
        values.push(packed(cols::TIMESTAMP_0));
        values.push(packed(cols::TIMESTAMP_1));
        values.extend([1, 0, 0].map(BusValue::constant));
        interactions.push(BusInteraction::sender(BusId::Memw, mu(), values));
    }

    // 3. IsHalfword on every state_ptr halfword.
    for lane in 0..LANES {
        for hw in 0..4 {
            interactions.push(BusInteraction::sender(
                BusId::IsHalfword,
                mu(),
                vec![packed(cols::state_ptr(lane, hw))],
            ));
        }
    }

    // 4. Alignment: addr[0] & 7 = 0.
    interactions.push(BusInteraction::sender(
        BusId::ByteAlu,
        mu(),
        vec![
            BusValue::constant(alu_op::AND as u64),
            packed(cols::addr(0)),
            BusValue::constant(7),
            BusValue::constant(0),
        ],
    ));

    // 5. AreBytes on addr, input and output bytes, in adjacent pairs.
    let byte_ranges = [
        (cols::ADDR, 8),
        (cols::INPUT, 8 * LANES),
        (cols::OUTPUT, 8 * LANES),
    ];
    for (start, len) in byte_ranges {
        for i in (0..len).step_by(2) {
            interactions.push(BusInteraction::sender(
                BusId::AreBytes,
                mu(),
                vec![packed(start + i), packed(start + i + 1)],
            ));
        }
    }

    // 6. One combined read+write Memw per lane: old = input, value = output.
    for lane in 0..LANES {
        let mut values = Vec::with_capacity(24);
        values.extend((0..8).map(|b| packed(cols::input(lane, b))));
        values.push(BusValue::constant(0));
        values.push(linear(cols::state_ptr(lane, 0), 2, 65536));
        values.push(linear(cols::state_ptr(lane, 2), 2, 65536));
        values.extend((0..8).map(|b| packed(cols::output(lane, b))));
        values.push(packed(cols::TIMESTAMP_0));
        values.push(packed(cols::TIMESTAMP_1));
        values.extend([0, 0, 1].map(BusValue::constant));
        interactions.push(BusInteraction::sender(BusId::Memw, mu(), values));
    }

    interactions
}

/// Index of the top-lane no-overflow constraint; the ADD pairs take `0..`.
const NO_OVERFLOW: usize = 2 * LANES;
/// First canonicity constraint; felt `f` owns `CANONICITY + 2f` and `+ 1`.
pub const CANONICITY: usize = NO_OVERFLOW + 1;
/// First round constraint.
pub const ROUNDS: usize = CANONICITY + 4 * LANES;
/// Two per S-box, then one per output lane.
pub const NUM_CONSTRAINTS: usize = ROUNDS + 2 * (8 * LANES + 22) + LANES;

#[derive(Clone, Copy)]
pub struct P1w16Constraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for P1w16Constraints {
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        use crate::constraints::templates::emit_add_pair;

        // state_ptr[lane] = addr + 8·lane, gated on μ.
        for lane in 0..LANES {
            emit_add_pair(
                b,
                lane * 2,
                &[cols::MU],
                &AddOperand::from_dword_bl(cols::ADDR),
                &AddOperand::constant((lane * 8) as i64),
                &AddOperand::from_dword_hl(cols::state_ptr(lane, 0)),
            );
        }

        // μ · carry_1 = 0: addr + 120 does not wrap.
        let top = LANES - 1;
        let word = |b: &mut B, start: usize, n: usize, step: u64| {
            (0..n)
                .map(|i| b.main(0, start + i) * b.const_base(step.pow(i as u32)))
                .reduce(|x, y| x + y)
                .expect("non-empty")
        };
        let addr_lo = word(b, cols::addr(0), 4, 256);
        let addr_hi = word(b, cols::addr(4), 4, 256);
        let ptr_lo = word(b, cols::state_ptr(top, 0), 2, 65536);
        let ptr_hi = word(b, cols::state_ptr(top, 2), 2, 65536);
        let inv_2_32 = b.const_base(INV_SHIFT_32);
        let offset = b.const_base((top * 8) as u64);
        let carry_0 = (addr_lo + offset - ptr_lo) * inv_2_32.clone();
        let carry_1 = (addr_hi + carry_0 - ptr_hi) * inv_2_32;
        let mu = b.main(0, cols::MU);
        b.emit_base(NO_OVERFLOW, mu.clone() * carry_1);

        // Canonicity of the 32 felts.
        for f in 0..2 * LANES {
            let start = cols::felt(f);
            let lo = word(b, start, 4, 256);
            let hi = word(b, start + 4, 4, 256);
            let all_ones = b.const_base(u32::MAX as u64);
            let d = hi - all_ones;
            let not_inv = b.one() - d.clone() * b.main(0, cols::canon_inv(f));
            b.emit_base(CANONICITY + 2 * f, d * not_inv.clone());
            b.emit_base(CANONICITY + 2 * f + 1, not_inv * lo);
        }

        // The rounds, over affine forms in the columns and μ: a felt is its
        // eight bytes at 256^i.
        let width = cols::NUM_COLUMNS;
        let felt = |start: usize| {
            let mut a = Affine::zero(width);
            for i in 0..8 {
                a.col[start + i] = FE::from(1u64 << (8 * i));
            }
            a
        };
        let mut state: Vec<Affine> = (0..LANES).map(|l| felt(cols::input(l, 0))).collect();
        let mut idx = ROUNDS;
        for (r, rc_row) in ROUND_CONSTANTS.iter().enumerate() {
            let a: Vec<Affine> = state
                .iter()
                .zip(rc_row)
                .map(|(s, rc)| {
                    let mut x = s.clone();
                    x.m += FE::from(*rc);
                    x
                })
                .collect();
            let mut f = a.clone();
            for lane in 0..sboxed_lanes(r) {
                let a_e = a[lane].expr(b, &mu);
                let x3 = b.main(0, cols::x3(r, lane));
                let x7 = b.main(0, cols::x7(r, lane));
                b.emit_base(idx, x3.clone() - a_e.clone() * a_e.clone() * a_e.clone());
                b.emit_base(idx + 1, x7 - x3.clone() * x3 * a_e);
                f[lane] = Affine::column(width, cols::x7(r, lane));
                idx += 2;
            }
            state = mds_affine(Mds::Circulant, &f);
        }
        for (lane, s) in state.iter().enumerate() {
            let out = felt(cols::output(lane, 0)).expr(b, &mu);
            b.emit_base(idx, out - s.expr(b, &mu));
            idx += 1;
        }
        debug_assert_eq!(idx, NUM_CONSTRAINTS);
        let _ = NUM_ROUNDS;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::element::FieldElement;
    use stark::constraints::builder::ProverEvalFolder;
    use stark::frame::Frame;
    use stark::table::TableView;
    use stark::traits::TransitionEvaluationContext;

    const P: u64 = math::field::goldilocks::GOLDILOCKS_PRIME;

    fn eval_row(trace: &TraceTable<GoldilocksField, GoldilocksExtension>, row: usize) -> Vec<FE> {
        let main: Vec<FE> = (0..cols::NUM_COLUMNS)
            .map(|c| *trace.main_table.get(row, c))
            .collect();
        let frame = Frame::<GoldilocksField, GoldilocksExtension>::new(vec![TableView::new(
            vec![main],
            vec![vec![]],
        )]);
        let no_e: Vec<FieldElement<GoldilocksExtension>> = vec![];
        let offset_e = FieldElement::<GoldilocksExtension>::zero();
        let ctx =
            TransitionEvaluationContext::new_prover(frame.as_row_frame(), &no_e, &no_e, &offset_e);
        let mut base = vec![FE::zero(); NUM_CONSTRAINTS];
        let mut ext = vec![FieldElement::<GoldilocksExtension>::zero(); NUM_CONSTRAINTS];
        let mut folder = ProverEvalFolder::new(&ctx, &mut base, &mut ext);
        P1w16Constraints.eval(&mut folder);
        base
    }

    /// Indices of the constraints that fail on `row`.
    fn failing(trace: &TraceTable<GoldilocksField, GoldilocksExtension>, row: usize) -> Vec<usize> {
        eval_row(trace, row)
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != FE::zero())
            .map(|(i, _)| i)
            .collect()
    }

    /// Canonical states: edge values and a splitmix64 stream reduced mod p.
    fn states() -> Vec<[u64; LANES]> {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            (z ^ (z >> 31)) % P
        };
        let mut v = vec![
            [0; LANES],
            [P - 1; LANES],
            core::array::from_fn(|i| [0, 1, P - 1, 0xFFFF_FFFF, 0xFFFF_FFFE_FFFF_FFFF][i % 5]),
        ];
        v.extend((0..4).map(|_| core::array::from_fn(|_| next())));
        v
    }

    fn op(input: [u64; LANES], output: [u64; LANES]) -> P1w16Operation {
        P1w16Operation {
            timestamp: 0x1234_5678_9ABC,
            state_addr: 0x0000_0001_FFFF_FFC0,
            input,
            output,
        }
    }

    fn honest(input: [u64; LANES]) -> P1w16Operation {
        op(
            input,
            executor::vm::instruction::execution::p1w16_permute(input),
        )
    }

    #[test]
    fn the_constraint_count_matches_what_eval_emits() {
        assert_eq!(P1w16Constraints.meta().len(), NUM_CONSTRAINTS);
        assert_eq!(cols::NUM_COLUMNS, 663);
    }

    /// The executor's permutation and the LFM's `Hash16` permutation agree.
    #[test]
    fn the_executor_and_lfm_permutations_agree() {
        for input in states() {
            let exec = executor::vm::instruction::execution::p1w16_permute(input);
            let felts = input.map(FE::from);
            assert_eq!(
                exec.map(FE::from),
                crypto::hash::poseidon1_w16::permute(felts)
            );
        }
    }

    #[test]
    fn honest_rows_and_padding_satisfy_every_constraint() {
        let ops: Vec<P1w16Operation> = states().into_iter().map(honest).collect();
        let trace = generate_p1w16_trace(&ops);
        assert!(trace.num_rows() > ops.len(), "the trace has padding rows");
        for row in 0..trace.num_rows() {
            assert_eq!(failing(&trace, row), Vec::<usize>::new(), "row {row}");
        }
    }

    /// Any output other than the permutation fails, and only in the rounds.
    #[test]
    fn a_wrong_output_fails_the_rounds() {
        let input = states()[3];
        let mut out = executor::vm::instruction::execution::p1w16_permute(input);
        out[7] = (out[7] + 1) % P;
        let trace = generate_p1w16_trace(&[op(input, out)]);
        let bad = failing(&trace, 0);
        assert!(!bad.is_empty());
        assert!(bad.iter().all(|&i| i >= ROUNDS), "{bad:?}");
    }

    /// An input lane stored as `7 + p`: the rounds read it as 7, the witness
    /// is the honest one for 7, and only that felt's canonicity can object.
    #[test]
    fn a_non_canonical_input_fails_only_its_canonicity() {
        let mut input = states()[2];
        input[4] = 7;
        let mut o = honest(input);
        o.input[4] += P;
        let trace = generate_p1w16_trace(&[o]);
        assert_eq!(failing(&trace, 0), vec![CANONICITY + 2 * 4 + 1]);
    }

    /// An output lane written as `3 + p`: its canonicity pair fires.
    #[test]
    fn a_non_canonical_output_fails_its_canonicity() {
        let mut o = honest(states()[3]);
        o.output[2] = 3 + P;
        let trace = generate_p1w16_trace(&[o]);
        let bad = failing(&trace, 0);
        let canon = CANONICITY + 2 * (LANES + 2) + 1;
        assert!(bad.contains(&canon), "{bad:?}");
        assert!(bad.iter().all(|&i| i == canon || i >= ROUNDS), "{bad:?}");
    }
}
