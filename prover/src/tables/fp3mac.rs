//! FP3MAC core chip — one cubic-extension multiply-add per `Fp3Mac` ecall.
//!
//! `x10` points to `[acc0, acc1, acc2, al0, al1, al2]`, `x11` to `[x0, x1, x2]`;
//! the row proves `acc ← acc·al + x` in `Fp3 = Fp[t]/(t³ − 2)` (the prover's
//! `Degree3GoldilocksExtensionField`) and writes `acc` back in place. Like
//! KECCAK (`keccak.rs`) and P1W16 (`p1w16.rs`): an `Ecall` receiver, `Memw`
//! reads of x10 and x11, and one combined read+write `Memw` per word — `acc`
//! with `old` = input and `value` = output, `al` and `x` with both the input.
//!
//! Each word's felt is `lo + 2^32·hi` over its eight range-checked bytes, held
//! canonical by one inverse column (`d·(1 − d·inv) = 0`, `(1 − d·inv)·lo = 0`,
//! `d = hi − (2^32 − 1)`), as in `p1w16.rs`. The product is three degree-2
//! constraints with no constants, so an all-zero padding row satisfies them.
//!
//! ## Column layout
//!
//! | Group     | Size | Description                                    |
//! |-----------|------|------------------------------------------------|
//! | timestamp |    2 | DWordWL                                        |
//! | addr      |    8 | State address (x10) as DWordBL                 |
//! | caddr     |    8 | Cell address (x11) as DWordBL                  |
//! | input     |   72 | `acc`, `al`, `x` words as bytes [9][8]         |
//! | output    |   24 | `acc` out as bytes [3][8]                      |
//! | ptr       |   36 | Per-word DWordHL addresses [9][4]              |
//! | canon_inv |   12 | Canonicity inverses, input words then output   |
//! | mu        |    1 | Multiplicity flag                              |

use executor::vm::instruction::execution::FP3MAC_SYSCALL_NUMBER;
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, alu_op};
use crate::constraints::templates::{AddOperand, INV_SHIFT_32};

/// Words the row reads: `acc` (3), `al` (3), `x` (3).
pub const WORDS: usize = 9;
/// Words the state holds at x10 (`acc`, `al`); the rest are the cell at x11.
pub const STATE_WORDS: usize = 6;

pub mod cols {
    use super::WORDS;

    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;
    pub const ADDR: usize = 2;
    pub const CADDR: usize = ADDR + 8;
    pub const INPUT: usize = CADDR + 8;
    pub const OUTPUT: usize = INPUT + 8 * WORDS;
    pub const PTR: usize = OUTPUT + 8 * 3;
    /// One inverse per felt: input words `0..9`, output words `9..12`.
    pub const CANON_INV: usize = PTR + 4 * WORDS;
    pub const MU: usize = CANON_INV + WORDS + 3;
    pub const NUM_COLUMNS: usize = MU + 1;

    #[inline]
    pub const fn addr(byte: usize) -> usize {
        ADDR + byte
    }

    #[inline]
    pub const fn caddr(byte: usize) -> usize {
        CADDR + byte
    }

    #[inline]
    pub const fn input(word: usize, byte: usize) -> usize {
        INPUT + word * 8 + byte
    }

    #[inline]
    pub const fn output(word: usize, byte: usize) -> usize {
        OUTPUT + word * 8 + byte
    }

    #[inline]
    pub const fn ptr(word: usize, hw: usize) -> usize {
        PTR + word * 4 + hw
    }

    #[inline]
    pub const fn canon_inv(f: usize) -> usize {
        CANON_INV + f
    }

    /// First byte of felt `f` (`f < 9` input word, else output word).
    #[inline]
    pub const fn felt(f: usize) -> usize {
        if f < WORDS {
            input(f, 0)
        } else {
            output(f - WORDS, 0)
        }
    }
}

#[derive(Debug, Clone)]
pub struct Fp3MacOperation {
    pub timestamp: u64,
    pub state_addr: u64,
    pub cell_addr: u64,
    /// `acc`, `al`, `x`.
    pub input: [u64; WORDS],
    /// `acc·al + x`.
    pub output: [u64; 3],
}

impl Fp3MacOperation {
    /// The address of read word `w`.
    pub fn word_addr(&self, w: usize) -> u64 {
        let (base, i) = if w < STATE_WORDS {
            (self.state_addr, w)
        } else {
            (self.cell_addr, w - STATE_WORDS)
        };
        base.checked_add(i as u64 * 8)
            .expect("fp3mac operand ranges must be validated by the executor")
    }

    /// The value word `w` holds after the ecall.
    pub fn word_out(&self, w: usize) -> u64 {
        if w < 3 { self.output[w] } else { self.input[w] }
    }
}

pub fn generate_fp3mac_trace(
    ops: &[Fp3MacOperation],
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
        table.set_dword_bl(row, cols::caddr(0), op.cell_addr);
        for w in 0..WORDS {
            table.set_dword_bl(row, cols::input(w, 0), op.input[w]);
            table.set_dword_hl(row, cols::ptr(w, 0), op.word_addr(w));
        }
        for (k, &v) in op.output.iter().enumerate() {
            table.set_dword_bl(row, cols::output(k, 0), v);
        }
        for (f, w) in op.input.iter().chain(&op.output).enumerate() {
            table.set_fe(row, cols::canon_inv(f), canon_inverse((w >> 32) as u32));
        }
        table.set_fe(row, cols::MU, FE::one());
    }

    // Padding rows are all zero except the canonicity inverses (`hi = 0`).
    let pad_inv = canon_inverse(0);
    for row in n..num_rows {
        for f in 0..WORDS + 3 {
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

/// A `Memw` read of register `reg` holding the 8-byte value at `start`.
fn register_read(reg: u64, start: usize) -> Vec<BusValue> {
    let lo = linear(start, 4, 256);
    let hi = linear(start + 4, 4, 256);
    let mut values = Vec::with_capacity(24);
    values.push(lo.clone());
    values.push(hi.clone());
    values.extend((2..8).map(|_| BusValue::constant(0)));
    values.push(BusValue::constant(1));
    values.push(BusValue::constant(2 * reg));
    values.push(BusValue::constant(0));
    values.push(lo);
    values.push(hi);
    values.extend((2..8).map(|_| BusValue::constant(0)));
    values.push(packed(cols::TIMESTAMP_0));
    values.push(packed(cols::TIMESTAMP_1));
    values.extend([1, 0, 0].map(BusValue::constant));
    values
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
            BusValue::constant(FP3MAC_SYSCALL_NUMBER & 0xFFFF_FFFF),
            BusValue::constant(FP3MAC_SYSCALL_NUMBER >> 32),
        ],
    ));

    // 2. Memw reads of x10 and x11.
    for (reg, start) in [(10, cols::ADDR), (11, cols::CADDR)] {
        interactions.push(BusInteraction::sender(
            BusId::Memw,
            mu(),
            register_read(reg, start),
        ));
    }

    // 3. IsHalfword on every word pointer halfword.
    for w in 0..WORDS {
        for hw in 0..4 {
            interactions.push(BusInteraction::sender(
                BusId::IsHalfword,
                mu(),
                vec![packed(cols::ptr(w, hw))],
            ));
        }
    }

    // 4. Alignment: addr[0] & 7 = 0 and caddr[0] & 7 = 0.
    for start in [cols::ADDR, cols::CADDR] {
        interactions.push(BusInteraction::sender(
            BusId::ByteAlu,
            mu(),
            vec![
                BusValue::constant(alu_op::AND as u64),
                packed(start),
                BusValue::constant(7),
                BusValue::constant(0),
            ],
        ));
    }

    // 5. AreBytes on the addresses, inputs and outputs, in adjacent pairs.
    let byte_ranges = [
        (cols::ADDR, 16),
        (cols::INPUT, 8 * WORDS),
        (cols::OUTPUT, 8 * 3),
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

    // 6. One combined read+write Memw per word: `acc` old = in, value = out;
    //    `al` and `x` old = value = in.
    for w in 0..WORDS {
        let out = |b: usize| {
            if w < 3 {
                cols::output(w, b)
            } else {
                cols::input(w, b)
            }
        };
        let mut values = Vec::with_capacity(24);
        values.extend((0..8).map(|b| packed(cols::input(w, b))));
        values.push(BusValue::constant(0));
        values.push(linear(cols::ptr(w, 0), 2, 65536));
        values.push(linear(cols::ptr(w, 2), 2, 65536));
        values.extend((0..8).map(|b| packed(out(b))));
        values.push(packed(cols::TIMESTAMP_0));
        values.push(packed(cols::TIMESTAMP_1));
        values.extend([0, 0, 1].map(BusValue::constant));
        interactions.push(BusInteraction::sender(BusId::Memw, mu(), values));
    }

    interactions
}

/// The ADD pairs take `0..2·WORDS`; then the two no-overflow constraints.
const NO_OVERFLOW: usize = 2 * WORDS;
/// First canonicity constraint; felt `f` owns `CANONICITY + 2f` and `+ 1`.
pub const CANONICITY: usize = NO_OVERFLOW + 2;
/// The three product constraints.
pub const PRODUCT: usize = CANONICITY + 2 * (WORDS + 3);
pub const NUM_CONSTRAINTS: usize = PRODUCT + 3;

#[derive(Clone, Copy)]
pub struct Fp3MacConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for Fp3MacConstraints {
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        use crate::constraints::templates::emit_add_pair;

        // ptr[w] = base + 8·i, gated on μ: the state words off x10, the cell's
        // off x11.
        for w in 0..WORDS {
            let (base, i) = if w < STATE_WORDS {
                (cols::ADDR, w)
            } else {
                (cols::CADDR, w - STATE_WORDS)
            };
            emit_add_pair(
                b,
                w * 2,
                &[cols::MU],
                &AddOperand::from_dword_bl(base),
                &AddOperand::constant((i * 8) as i64),
                &AddOperand::from_dword_hl(cols::ptr(w, 0)),
            );
        }

        let word = |b: &mut B, start: usize, n: usize, step: u64| {
            (0..n)
                .map(|i| b.main(0, start + i) * b.const_base(step.pow(i as u32)))
                .reduce(|x, y| x + y)
                .expect("non-empty")
        };
        // μ · carry_1 = 0 on each range's top word: no wrap.
        let mu = b.main(0, cols::MU);
        for (k, (base, top, offset)) in [
            (cols::ADDR, STATE_WORDS - 1, 8 * (STATE_WORDS - 1)),
            (cols::CADDR, WORDS - 1, 8 * (WORDS - STATE_WORDS - 1)),
        ]
        .into_iter()
        .enumerate()
        {
            let addr_lo = word(b, base, 4, 256);
            let addr_hi = word(b, base + 4, 4, 256);
            let ptr_lo = word(b, cols::ptr(top, 0), 2, 65536);
            let ptr_hi = word(b, cols::ptr(top, 2), 2, 65536);
            let inv_2_32 = b.const_base(INV_SHIFT_32);
            let off = b.const_base(offset as u64);
            let carry_0 = (addr_lo + off - ptr_lo) * inv_2_32.clone();
            let carry_1 = (addr_hi + carry_0 - ptr_hi) * inv_2_32;
            b.emit_base(NO_OVERFLOW + k, mu.clone() * carry_1);
        }

        // Canonicity of the twelve felts.
        let mut felts = Vec::with_capacity(WORDS + 3);
        for f in 0..WORDS + 3 {
            let start = cols::felt(f);
            let lo = word(b, start, 4, 256);
            let hi = word(b, start + 4, 4, 256);
            let all_ones = b.const_base(u32::MAX as u64);
            let d = hi.clone() - all_ones;
            let not_inv = b.one() - d.clone() * b.main(0, cols::canon_inv(f));
            b.emit_base(CANONICITY + 2 * f, d * not_inv.clone());
            b.emit_base(CANONICITY + 2 * f + 1, not_inv * lo.clone());
            let two_32 = b.const_base(1 << 32);
            felts.push(lo + hi * two_32);
        }

        // out = acc·al + x with t³ = 2.
        let (a, c, x, out) = (&felts[0..3], &felts[3..6], &felts[6..9], &felts[9..12]);
        let two = b.const_base(2);
        let p = |i: usize, j: usize| a[i].clone() * c[j].clone();
        let prod = [
            p(0, 0) + (p(1, 2) + p(2, 1)) * two.clone(),
            p(0, 1) + p(1, 0) + p(2, 2) * two.clone(),
            p(0, 2) + p(1, 1) + p(2, 0),
        ];
        for (k, pk) in prod.into_iter().enumerate() {
            b.emit_base(PRODUCT + k, out[k].clone() - (pk + x[k].clone()));
        }
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
        Fp3MacConstraints.eval(&mut folder);
        base
    }

    fn failing(trace: &TraceTable<GoldilocksField, GoldilocksExtension>, row: usize) -> Vec<usize> {
        eval_row(trace, row)
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != FE::zero())
            .map(|(i, _)| i)
            .collect()
    }

    /// Canonical words: edge values and a splitmix64 stream reduced mod p.
    fn words(n: usize) -> Vec<[u64; WORDS]> {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            (z ^ (z >> 31)) % P
        };
        let mut v = vec![
            [0; WORDS],
            [P - 1; WORDS],
            core::array::from_fn(|i| [0, 1, P - 1, 0xFFFF_FFFF, 0xFFFF_FFFE_FFFF_FFFF][i % 5]),
        ];
        v.extend((0..n).map(|_| core::array::from_fn(|_| next())));
        v
    }

    fn honest(input: [u64; WORDS]) -> Fp3MacOperation {
        let state: [u64; 6] = core::array::from_fn(|i| input[i]);
        let x = [input[6], input[7], input[8]];
        Fp3MacOperation {
            timestamp: 0x1234_5678_9ABC,
            state_addr: 0x0000_0001_FFFF_FFC0,
            cell_addr: 0x0000_0002_0000_0040,
            input,
            output: executor::vm::instruction::execution::fp3mac(state, x),
        }
    }

    #[test]
    fn the_constraint_count_matches_what_eval_emits() {
        assert_eq!(Fp3MacConstraints.meta().len(), NUM_CONSTRAINTS);
        assert_eq!(cols::NUM_COLUMNS, 163);
    }

    /// The executor's helper is the prover's `FEE` arithmetic.
    #[test]
    fn the_executor_agrees_with_the_extension_field() {
        use crate::tables::types::FEE;
        for w in words(16) {
            let ext = |i: usize| FEE::new([FE::from(w[i]), FE::from(w[i + 1]), FE::from(w[i + 2])]);
            let want = ext(0) * ext(3) + ext(6);
            let got = executor::vm::instruction::execution::fp3mac(
                core::array::from_fn(|i| w[i]),
                [w[6], w[7], w[8]],
            );
            assert_eq!(got.map(FE::from), *want.value());
        }
    }

    #[test]
    fn honest_rows_and_padding_satisfy_every_constraint() {
        let ops: Vec<Fp3MacOperation> = words(4).into_iter().map(honest).collect();
        let trace = generate_fp3mac_trace(&ops);
        assert!(trace.num_rows() > ops.len(), "the trace has padding rows");
        for row in 0..trace.num_rows() {
            assert_eq!(failing(&trace, row), Vec::<usize>::new(), "row {row}");
        }
    }

    /// Any other output fails, and only the product.
    #[test]
    fn a_wrong_output_fails_the_product() {
        let mut o = honest(words(4)[4]);
        o.output[1] = (o.output[1] + 1) % P;
        let trace = generate_fp3mac_trace(&[o]);
        assert_eq!(failing(&trace, 0), vec![PRODUCT + 1]);
    }

    /// An input word stored as `7 + p`: the product reads it as 7, so only its
    /// canonicity objects.
    #[test]
    fn a_non_canonical_input_fails_only_its_canonicity() {
        let mut input = words(3)[3];
        input[4] = 7;
        let mut o = honest(input);
        o.input[4] += P;
        let trace = generate_fp3mac_trace(&[o]);
        assert_eq!(failing(&trace, 0), vec![CANONICITY + 2 * 4 + 1]);
    }

    /// An output word written as `v + p` for a `v < 2^32 − 1`: only its
    /// canonicity objects.
    #[test]
    fn a_non_canonical_output_fails_only_its_canonicity() {
        let mut input = [0u64; WORDS];
        input[6] = 3;
        let mut o = honest(input);
        assert_eq!(o.output, [3, 0, 0]);
        o.output[0] += P;
        let trace = generate_fp3mac_trace(&[o]);
        assert_eq!(failing(&trace, 0), vec![CANONICITY + 2 * WORDS + 1]);
    }
}
