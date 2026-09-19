//! SHA256MSGSCHED: the schedule words 16..63, each the sum of four earlier words
//! with two of them passed through σ0 and σ1 first.
//!
//! `w[i-15]` and `w[i-2]` are held as 32 bits rather than as single values, so
//! σ0 and σ1 become expressions instead of ROTXOR requests: a rotation is an
//! index permutation of the bits and costs nothing, and a three-way XOR of bits
//! is a degree-3 polynomial with no columns and no lookups. That is how OpenVM,
//! ZisK and RISC0 arithmetize SHA-256; none of them has a rotate chip. See
//! SHA256_OPT_ZKVM_SURVEY.md.
//!
//! The trade: 74 columns against 14, but 96 of the 224 rows the ROTXOR chip used
//! to spend per compression block disappear, and this chip drops from 10 bus
//! interactions to 8. With the round chip doing the same for Σ0/Σ1, ROTXOR has
//! no callers left and is gone.
//!
//! The output is a WordHL range-checked by `IS_HALF` lookup. Adding four
//! range-checked words into a range-checked word only needs the carry bounded,
//! and the carry of four 32-bit values does not reach a byte.
use super::{
    sha256_common::*,
    types::{BusId, GoldilocksExtension as E, GoldilocksField as F},
};
use stark::{
    constraints::builder::{ConstraintBuilder, ConstraintSet},
    lookup::BusInteraction,
    trace::TraceTable,
};

pub const TS: usize = 0;
pub const INDEX: usize = 2;
pub const AMOUNT: usize = 3;
pub const OUT: usize = 4;
pub const BACK7: usize = 6;
pub const BACK16: usize = 7;
/// `w[i-15]` and `w[i-2]`, bit by bit, least-significant first.
pub const B15: usize = 8;
pub const B2: usize = 40;
/// The spec keeps this virtual. Making it so needs `carry_value` (stark's
/// `ColumnUnsigned` term) and that path does not balance the bus yet — see
/// PR990_SPEC_LIKE_ROTXOR.md. One column, and the constraint is equivalent.
pub const CARRY: usize = 72;
pub const MU: usize = 73;
pub const WIDTH: usize = 74;

/// How many times `w[i]` is read: once by the round chip, plus once for every
/// later schedule row that looks back at it.
pub fn amount(i: usize) -> u64 {
    1 + [2, 7, 15, 16]
        .into_iter()
        .filter(|d| i + d >= 16 && i + d < 64)
        .count() as u64
}

pub fn generate(ops: &[super::sha256::Operation]) -> TraceTable<F, E> {
    let mut rows = TraceRows::new(ops.len() * 48, WIDTH);
    for op in ops {
        let w = executor::sha256::schedule(&op.message);
        for i in 16..64 {
            rows.push(|r| {
                r[TS] = op.timestamp & 0xffffffff;
                r[TS + 1] = op.timestamp >> 32;
                r[INDEX] = i as u64;
                r[AMOUNT] = amount(i);
                r[BACK7] = w[i - 7] as u64;
                r[BACK16] = w[i - 16] as u64;
                put_bits(r, B15, w[i - 15] as u64, 32);
                put_bits(r, B2, w[i - 2] as u64, 32);
                r[OUT] = (w[i] & 0xffff) as u64;
                r[OUT + 1] = (w[i] >> 16) as u64;
                let sum = r[BACK16]
                    + executor::sha256::sigma(w[i - 15], 0) as u64
                    + r[BACK7]
                    + executor::sha256::sigma(w[i - 2], 1) as u64;
                r[CARRY] = sum >> 32;
                r[MU] = 1;
            });
        }
    }
    rows.finish()
}

pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut v = Vec::with_capacity(8);
    for (d, back) in [
        (2, bits(B2, 32)),
        (7, col(BACK7)),
        (15, bits(B15, 32)),
        (16, col(BACK16)),
    ] {
        v.push(send(
            BusId::ShaM,
            MU,
            vec![col(TS), col(TS + 1), lin(vec![(INDEX, 1)], -d), back],
        ));
    }
    // `out`'s two halves by lookup; the index offset and the carry are the
    // spec's two IS_BYTE checks, which one paired send covers.
    for i in 0..2 {
        v.push(send(BusId::IsHalfword, MU, vec![col(OUT + i)]));
    }
    v.push(send(
        BusId::AreBytes,
        MU,
        vec![lin(vec![(INDEX, 1)], -16), col(CARRY)],
    ));
    let mut out = recv(
        BusId::ShaM,
        MU,
        vec![
            col(TS),
            col(TS + 1),
            col(INDEX),
            lin(vec![(OUT, 1), (OUT + 1, 65536)], 0),
        ],
    );
    out.multiplicity = stark::lookup::Multiplicity::Column(AMOUNT);
    v.push(out);
    v
}

/// The BITWISE lookups this chip performs: two `IS_HALF` for the output's
/// halves and one paired `ARE_BYTES` for the index offset and the carry.
pub fn bitwise_ops(ops: &[super::sha256::Operation]) -> Vec<super::bitwise::BitwiseOperation> {
    use super::bitwise::{BitwiseOperation as Op, BitwiseOperationType as Ty};
    let mut v = Vec::with_capacity(ops.len() * 48 * 3);
    for op in ops {
        let w = executor::sha256::schedule(&op.message);
        for i in 16..64 {
            for half in [w[i] & 0xffff, w[i] >> 16] {
                v.push(Op::halfword(Ty::IsHalf, half as u8, (half >> 8) as u8));
            }
            let sum = w[i - 16] as u64
                + executor::sha256::sigma(w[i - 15], 0) as u64
                + w[i - 7] as u64
                + executor::sha256::sigma(w[i - 2], 1) as u64;
            v.push(Op::halfword(
                Ty::AreBytes,
                (i - 16) as u8,
                (sum >> 32) as u8,
            ));
        }
    }
    v
}

#[derive(Clone, Copy)]
pub struct Constraints;
impl ConstraintSet<F, E> for Constraints {
    /// The three-way XOR inside σ/Σ is degree 3. LogUp already raises the
    /// composition bound to 3, so declaring it here costs nothing and keeps the
    /// declaration honest.
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let mut id = 0;
        check_bits(b, &mut id, MU, 1);
        check_bits(b, &mut id, B15, 32);
        check_bits(b, &mut id, B2, 32);
        // Four range-checked words into a range-checked word: only the carry
        // needs bounding, which the ARE_BYTES send above does.
        let out = b.main(0, OUT) + b.const_base(65536) * b.main(0, OUT + 1);
        let s0 = sigma(b, B15, 0);
        let s1 = sigma(b, B2, 1);
        b.emit_base(
            id,
            b.main(0, BACK16) + s0 + b.main(0, BACK7) + s1
                - out
                - b.const_base(1 << 32) * b.main(0, CARRY),
        );
        id += 1;
        // A row that carries nothing may not claim reads of it.
        b.emit_base(id, (b.const_base(1) - b.main(0, MU)) * b.main(0, AMOUNT));
    }
}
