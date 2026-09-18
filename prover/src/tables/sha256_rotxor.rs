//! ROTXOR, following `spec/src/rotxor.toml` faithfully: the rotations come from
//! `HWSL` half-word shifts and the XORs from `BYTE_ALU`, rather than the bit
//! gates the first implementation used.
//!
//! Computes, for a 32-bit `a` and nibble-sized `r0`, `r1`, `r2`:
//!
//! ```text
//!   (a >>> (16 + r0)) ^ (a >>> (16 + r0 - r1)) ^ (a >>> r2)   if last_rot
//!   (a >>> (16 + r0)) ^ (a >>> (16 + r0 - r1)) ^ (a >>  r2)   otherwise
//! ```
//!
//! Every column is range-checked by the lookups themselves — a half-word reaches
//! `HWSL` only as `X + 256*Y` over BITWISE's byte columns, a byte reaches
//! `BYTE_ALU` only as one of its operands — so the chip declares no range checks
//! of its own. That is the trade this file exists to measure: 39 columns and 15
//! bus interactions here, against 197 columns and 2 interactions in the bit
//! version.
use super::{
    sha256_common::*,
    types::{BusId, GoldilocksExtension as E, GoldilocksField as F, alu_op},
};
use stark::{
    constraints::builder::{ConstraintBuilder, ConstraintSet},
    lookup::{BusInteraction, BusValue},
    trace::TraceTable,
};

// `a` and the six shift results are WordHL (two 16-bit half-words); `out` and
// the four XOR operands are WordBL (four bytes).
pub const A: usize = 0;
pub const R0: usize = 2;
pub const R1: usize = 3;
pub const R2: usize = 4;
pub const LAST_ROT: usize = 5;
pub const OUT: usize = 6;
pub const A0_LEFT: usize = 10;
pub const A0_RIGHT: usize = 12;
pub const A1_LEFT: usize = 14;
pub const A1_RIGHT: usize = 16;
pub const A2_LEFT: usize = 18;
pub const A2_RIGHT: usize = 20;
pub const A0: usize = 22;
pub const A1: usize = 26;
pub const A2: usize = 30;
pub const A01: usize = 34;
pub const MU: usize = 38;
pub const WIDTH: usize = 39;

/// `(r0, r1, r2, last_rot)` for σ0, σ1, Σ0, Σ1 — the spec's encoding, where the
/// rotations are `16+r0` and `16+r0−r1`. Unchanged from the bit version: this is
/// the bus signature, and callers send these as constants.
pub const PARAMS: [[u64; 4]; 4] = [[2, 11, 3, 0], [3, 2, 10, 0], [6, 9, 2, 1], [9, 14, 6, 1]];

/// Half-word `i` of the WordBL at `c`, the spec's `cast(_, "WordHL")`.
fn hl(c: usize, i: usize) -> BusValue {
    lin(vec![(c + 2 * i, 1), (c + 2 * i + 1, 256)], 0)
}

/// The 32-bit value of the WordBL at `c`, the spec's `cast(_, "Word")`.
fn word_bl(c: usize) -> BusValue {
    lin((0..4).map(|j| (c + j, 1i64 << (8 * j))).collect(), 0)
}

fn put_bytes(row: &mut [u64], c: usize, x: u32) {
    for j in 0..4 {
        row[c + j] = ((x >> (8 * j)) & 0xff) as u64;
    }
}

/// One row's worth of intermediates. Shared by trace generation and by the
/// BITWISE lookup census so the two cannot drift: every value this chip commits
/// to is also a value it looks up, and a mismatch between the two is a bus
/// imbalance rather than a wrong answer, which is much harder to read.
struct Parts {
    a: [u64; 2],
    a0_left: [u64; 2],
    a0_right: [u64; 2],
    a1_left: [u64; 2],
    a1_right: [u64; 2],
    a2_left: [u64; 2],
    a2_right: [u64; 2],
    a0: u32,
    a1: u32,
    a2: u32,
    a01: u32,
    out: u32,
}

fn parts(x: u32, kind: usize) -> Parts {
    let [r0, r1, r2, last_rot] = PARAMS[kind];
    let a = [(x & 0xffff) as u64, (x >> 16) as u64];

    // HWSL[a[i], 16 - r0] -> (a << (16-r0), a >> r0), per half-word.
    let a0_left = std::array::from_fn(|i| (a[i] << (16 - r0)) & 0xffff);
    let a0_right = std::array::from_fn(|i| a[i] >> r0);
    let a0 = x.rotate_right(16 + r0 as u32);

    // HWSL[a0[i], r1] -> (a0 << r1, a0 >> (16-r1)).
    let a0_hl = [(a0 & 0xffff) as u64, (a0 >> 16) as u64];
    let a1_left = std::array::from_fn(|i| (a0_hl[i] << r1) & 0xffff);
    let a1_right = std::array::from_fn(|i| a0_hl[i] >> (16 - r1));
    let a1 = a0.rotate_left(r1 as u32);

    // HWSL[a[i], 16 - r2]; `last_rot` decides whether the bits that fall out of
    // the low half-word wrap into the high one.
    let a2_left = std::array::from_fn(|i| (a[i] << (16 - r2)) & 0xffff);
    let a2_right = std::array::from_fn(|i| a[i] >> r2);
    let a2 = if last_rot == 1 {
        x.rotate_right(r2 as u32)
    } else {
        x >> r2
    };

    let a01 = a0 ^ a1;
    Parts {
        a,
        a0_left,
        a0_right,
        a1_left,
        a1_right,
        a2_left,
        a2_right,
        a0,
        a1,
        a2,
        a01,
        out: a01 ^ a2,
    }
}

pub fn generate(ops: &[(u32, usize)]) -> TraceTable<F, E> {
    let mut rows = TraceRows::new(ops.len(), WIDTH);
    for &(x, kind) in ops {
        let [r0, r1, r2, last_rot] = PARAMS[kind];
        let p = parts(x, kind);
        rows.push(|r| {
            r[A] = p.a[0];
            r[A + 1] = p.a[1];
            r[R0] = r0;
            r[R1] = r1;
            r[R2] = r2;
            r[LAST_ROT] = last_rot;
            for (c, half) in [
                (A0_LEFT, p.a0_left),
                (A0_RIGHT, p.a0_right),
                (A1_LEFT, p.a1_left),
                (A1_RIGHT, p.a1_right),
                (A2_LEFT, p.a2_left),
                (A2_RIGHT, p.a2_right),
            ] {
                r[c..c + 2].copy_from_slice(&half);
            }
            put_bytes(r, A0, p.a0);
            put_bytes(r, A1, p.a1);
            put_bytes(r, A2, p.a2);
            put_bytes(r, A01, p.a01);
            put_bytes(r, OUT, p.out);
            r[MU] = 1;
        });
    }
    rows.finish()
}

/// The BITWISE lookups this chip performs: six `HWSL` and eight `BYTE_ALU[XOR]`
/// per row. The bit version performed none, so nothing fed them before.
pub fn bitwise_ops(ops: &[(u32, usize)]) -> Vec<super::bitwise::BitwiseOperation> {
    use super::bitwise::{BitwiseOperation as Op, BitwiseOperationType as Ty};
    let mut v = Vec::with_capacity(ops.len() * 14);
    for &(x, kind) in ops {
        let [r0, r1, r2, _] = PARAMS[kind];
        let p = parts(x, kind);
        let a0_hl = [(p.a0 & 0xffff) as u64, (p.a0 >> 16) as u64];
        for (src, z) in [(p.a, 16 - r0), (a0_hl, r1), (p.a, 16 - r2)] {
            for h in src {
                v.push(Op::new(Ty::Hwsl, h as u8, (h >> 8) as u8, z as u8));
            }
        }
        for (l, r) in [(p.a0, p.a1), (p.a01, p.a2)] {
            for j in 0..4 {
                v.push(Op::byte_op(
                    Ty::ByteAluXor,
                    (l >> (8 * j)) as u8,
                    (r >> (8 * j)) as u8,
                ));
            }
        }
    }
    v
}

pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut v = Vec::with_capacity(15);
    // `16 - r0` and `16 - r2` are the shift amounts; reaching HWSL as a Z in
    // [1,15] is what pins r0 and r2 to the range the spec assumes.
    for (shift, left, right) in [
        (lin(vec![(R0, -1)], 16), A0_LEFT, A0_RIGHT),
        (lin(vec![(R2, -1)], 16), A2_LEFT, A2_RIGHT),
    ] {
        for i in 0..2 {
            v.push(send(
                BusId::Hwsl,
                MU,
                vec![col(A + i), shift.clone(), col(left + i), col(right + i)],
            ));
        }
    }
    for i in 0..2 {
        v.push(send(
            BusId::Hwsl,
            MU,
            vec![hl(A0, i), col(R1), col(A1_LEFT + i), col(A1_RIGHT + i)],
        ));
    }
    for (x, y, out) in [(A0, A1, A01), (A01, A2, OUT)] {
        for j in 0..4 {
            v.push(send(
                BusId::ByteAlu,
                MU,
                vec![
                    BusValue::constant(alu_op::XOR as u64),
                    col(x + j),
                    col(y + j),
                    col(out + j),
                ],
            ));
        }
    }
    v.push(recv(
        BusId::ShaRot,
        MU,
        vec![
            lin(vec![(A, 1), (A + 1, 65536)], 0),
            col(R0),
            col(R1),
            col(R2),
            col(LAST_ROT),
            word_bl(OUT),
        ],
    ));
    v
}

/// Unchanged from the bit version: the bus signature is the spec's, so a caller
/// cannot tell which implementation is behind it.
pub fn request(mu: usize, input: BusValue, kind: usize, out: BusValue) -> BusInteraction {
    let mut v = vec![input];
    v.extend(PARAMS[kind].map(BusValue::constant));
    v.push(out);
    send(BusId::ShaRot, mu, v)
}

#[derive(Clone, Copy)]
pub struct Constraints;
impl ConstraintSet<F, E> for Constraints {
    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let mut id = 0;
        check_bits(b, &mut id, LAST_ROT, 1);
        check_bits(b, &mut id, MU, 1);

        // A right rotation reassembles crosswise: half-word i takes what stayed
        // in it and what fell out of the other one.
        for (word, left, right) in [(A0, A0_LEFT, A0_RIGHT), (A1, A1_LEFT, A1_RIGHT)] {
            for i in 0..2 {
                let half =
                    b.main(0, word + 2 * i) + b.const_base(256) * b.main(0, word + 2 * i + 1);
                b.emit_base(id, half - b.main(0, left + i) - b.main(0, right + 1 - i));
                id += 1;
            }
        }
        // a2 may be a plain shift instead: `last_rot` gates the wrap.
        let half0 = b.main(0, A2) + b.const_base(256) * b.main(0, A2 + 1);
        b.emit_base(id, half0 - b.main(0, A2_LEFT + 1) - b.main(0, A2_RIGHT));
        id += 1;
        let half1 = b.main(0, A2 + 2) + b.const_base(256) * b.main(0, A2 + 3);
        b.emit_base(
            id,
            half1 - b.main(0, LAST_ROT) * b.main(0, A2_LEFT) - b.main(0, A2_RIGHT + 1),
        );
    }
}
