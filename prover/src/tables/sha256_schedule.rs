//! SHA256MSGSCHED, following `spec/src/sha256msgsched.toml`: the schedule words
//! 16..63, each one the sum of four earlier words with two of them passed
//! through ROTXOR first.
//!
//! The output is a WordHL range-checked by `IS_HALF` lookup rather than a
//! 32-column bit decomposition, which is what takes the chip from 46 columns to
//! 14. The spec's argument for why that is enough: adding four range-checked
//! words into a range-checked word only needs the carry bounded, and the carry
//! of four 32-bit values does not reach a byte.
use super::{
    sha256_common::*,
    sha256_rotxor as rot,
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
pub const BACK2: usize = 6;
pub const BACK7: usize = 7;
pub const BACK15: usize = 8;
pub const BACK16: usize = 9;
pub const S0: usize = 10;
pub const S1: usize = 11;
/// The spec keeps this virtual. Making it so needs `carry_value` (stark's
/// `ColumnUnsigned` term) and that path does not balance the bus yet — see
/// PR990_SPEC_LIKE_ROTXOR.md. One column, and the constraint is equivalent.
pub const CARRY: usize = 12;
pub const MU: usize = 13;
pub const WIDTH: usize = 14;

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
                r[BACK2] = w[i - 2] as u64;
                r[BACK7] = w[i - 7] as u64;
                r[BACK15] = w[i - 15] as u64;
                r[BACK16] = w[i - 16] as u64;
                r[S0] = executor::sha256::sigma(w[i - 15], 0) as u64;
                r[S1] = executor::sha256::sigma(w[i - 2], 1) as u64;
                r[OUT] = (w[i] & 0xffff) as u64;
                r[OUT + 1] = (w[i] >> 16) as u64;
                let sum = r[BACK16] + r[S0] + r[BACK7] + r[S1];
                r[CARRY] = sum >> 32;
                r[MU] = 1;
            });
        }
    }
    rows.finish()
}

pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut v = Vec::with_capacity(10);
    for (d, back) in [(2, BACK2), (7, BACK7), (15, BACK15), (16, BACK16)] {
        v.push(send(
            BusId::ShaM,
            MU,
            vec![col(TS), col(TS + 1), lin(vec![(INDEX, 1)], -d), col(back)],
        ));
    }
    v.push(rot::request(MU, col(BACK15), 0, col(S0)));
    v.push(rot::request(MU, col(BACK2), 1, col(S1)));
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
    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let mut id = 0;
        check_bits(b, &mut id, MU, 1);
        // Four range-checked words into a range-checked word: only the carry
        // needs bounding, which the ARE_BYTES send above does.
        let out = b.main(0, OUT) + b.const_base(65536) * b.main(0, OUT + 1);
        b.emit_base(
            id,
            b.main(0, BACK16) + b.main(0, S0) + b.main(0, BACK7) + b.main(0, S1)
                - out
                - b.const_base(1 << 32) * b.main(0, CARRY),
        );
        id += 1;
        // A row that carries nothing may not claim reads of it.
        b.emit_base(id, (b.const_base(1) - b.main(0, MU)) * b.main(0, AMOUNT));
    }
}
