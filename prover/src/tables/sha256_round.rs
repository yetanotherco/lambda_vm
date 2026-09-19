//! SHA256ROUND, following `spec/src/sha256round.toml`: Ch and Maj come from
//! `BYTE_ALU` lookups over byte-split state words, not from per-bit gates.
//!
//! The state splits asymmetrically on purpose, as the spec has it: `b`, `c`,
//! `f`, `g` are WordBL because they feed byte lookups, while `d` and `h` stay
//! single field elements because they only ever appear in an addition. `ch`,
//! `maj`, `temp1` and `temp2` are the spec's *virtual* variables — they are
//! expressions over those columns, never stored.
//!
//! `a` and `e` are held as 32 bits instead, so Σ0 and Σ1 are expressions rather
//! than ROTXOR requests: a rotation is an index permutation of the bits, and a
//! three-way XOR of bits is degree 3 with no columns and no lookups. Their
//! bytes are still available to `BYTE_ALU` as linear combinations of eight bits,
//! so no byte column is needed for them either. With the schedule chip doing the
//! same for σ0/σ1, the ROTXOR chip has no callers left and has been removed —
//! which is how OpenVM, ZisK and RISC0 arithmetize SHA-256. See
//! SHA256_OPT_ZKVM_SURVEY.md.
use super::{
    sha256_common::*,
    types::{BusId, GoldilocksExtension as E, GoldilocksField as F, alu_op},
};
use stark::{
    constraints::builder::{ConstraintBuilder, ConstraintSet},
    lookup::{BusInteraction, BusValue},
    trace::TraceTable,
};

pub const TS: usize = 0;
pub const INDEX: usize = 2;
/// `a` and `e`, bit by bit, least-significant first.
pub const A: usize = 3;
pub const B: usize = 35;
pub const C: usize = 39;
pub const D: usize = 43;
pub const E_: usize = 44;
pub const FF: usize = 76;
pub const G: usize = 80;
pub const H: usize = 84;
pub const OUT_A: usize = 85;
pub const OUT_E: usize = 87;
pub const A_AND_B: usize = 89;
pub const A_XOR_B: usize = 93;
pub const C_AND_A_XOR_B: usize = 97;
pub const E_AND_F: usize = 101;
pub const NOT_E_AND_G: usize = 105;
pub const K: usize = 109;
pub const W: usize = 110;
/// The spec keeps the two carries virtual. They are columns here because
/// recovering them as expressions needs a multiply by 2^-32, and a bus value's
/// coefficients are i64 — which that constant does not fit. Two columns out of
/// sixty, and the constraint is the same either way.
pub const CARRY_A: usize = 111;
pub const CARRY_E: usize = 112;
pub const MU: usize = 113;
pub const WIDTH: usize = 114;

/// The 32-bit value of the WordBL at `c`.
fn word_bl(c: usize) -> BusValue {
    lin((0..4).map(|j| (c + j, 1i64 << (8 * j))).collect(), 0)
}

/// The 32-bit value of the WordHL at `c`.
fn word_hl(c: usize) -> BusValue {
    lin(vec![(c, 1), (c + 1, 65536)], 0)
}

fn put_bytes(row: &mut [u64], c: usize, x: u32) {
    for j in 0..4 {
        row[c + j] = ((x >> (8 * j)) & 0xff) as u64;
    }
}

pub fn generate(ops: &[super::sha256::Operation]) -> TraceTable<F, E> {
    let mut rows = TraceRows::new(ops.len() * 64, WIDTH);
    for op in ops {
        let w = executor::sha256::schedule(&op.message);
        let mut s = op.state_words();
        for (i, &word) in w.iter().enumerate() {
            let out = executor::sha256::round(s, word, executor::sha256::K[i]);
            rows.push(|r| {
                let [a, b, c, d, e, f, g, h] = s;
                r[TS] = op.timestamp & 0xffffffff;
                r[TS + 1] = op.timestamp >> 32;
                r[INDEX] = i as u64;
                put_bits(r, A, a as u64, 32);
                put_bytes(r, B, b);
                put_bytes(r, C, c);
                r[D] = d as u64;
                put_bits(r, E_, e as u64, 32);
                put_bytes(r, FF, f);
                put_bytes(r, G, g);
                r[H] = h as u64;

                put_bytes(r, A_AND_B, a & b);
                put_bytes(r, A_XOR_B, a ^ b);
                put_bytes(r, C_AND_A_XOR_B, c & (a ^ b));
                put_bytes(r, E_AND_F, e & f);
                put_bytes(r, NOT_E_AND_G, !e & g);

                r[K] = executor::sha256::K[i] as u64;
                r[W] = word as u64;

                // out_a and out_e are WordHL, so the halves go in separately.
                r[OUT_A] = (out[0] & 0xffff) as u64;
                r[OUT_A + 1] = (out[0] >> 16) as u64;
                r[OUT_E] = (out[4] & 0xffff) as u64;
                r[OUT_E + 1] = (out[4] >> 16) as u64;

                let ch = (e & f) as u64 + (!e & g) as u64;
                let maj = (a & b) as u64 + (c & (a ^ b)) as u64;
                let temp1 =
                    h as u64 + executor::sha256::sigma(e, 3) as u64 + ch + r[K] + r[W];
                let temp2 = executor::sha256::sigma(a, 2) as u64 + maj;
                r[CARRY_A] = (temp1 + temp2) >> 32;
                r[CARRY_E] = (d as u64 + temp1) >> 32;
                r[MU] = 1;
            });
            s = out;
        }
    }
    rows.finish()
}

pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut v = Vec::with_capacity(30);
    // Ch and Maj, byte by byte. `255 - e[j]` is the byte complement the spec
    // uses so `not_e_and_g` needs no extra opsel.
    for (op, x, x_is_bits, complement_x, y, out) in [
        (alu_op::AND, A, true, false, B, A_AND_B),
        (alu_op::XOR, A, true, false, B, A_XOR_B),
        (alu_op::AND, C, false, false, A_XOR_B, C_AND_A_XOR_B),
        (alu_op::AND, E_, true, false, FF, E_AND_F),
        (alu_op::AND, E_, true, true, G, NOT_E_AND_G),
    ] {
        for j in 0..4 {
            let left = if x_is_bits {
                byte_bits(x, j, complement_x)
            } else if complement_x {
                lin(vec![(x + j, -1)], 255)
            } else {
                col(x + j)
            };
            v.push(send(
                BusId::ByteAlu,
                MU,
                vec![
                    BusValue::constant(op as u64),
                    left,
                    col(y + j),
                    col(out + j),
                ],
            ));
        }
    }
    v.push(send(BusId::ShaK, MU, vec![col(INDEX), col(K)]));
    v.push(send(
        BusId::ShaM,
        MU,
        vec![col(TS), col(TS + 1), col(INDEX), col(W)],
    ));
    // The two outputs are range-checked as half-words by lookup; the carries as
    // bytes, which one paired ARE_BYTES send covers.
    for c in [OUT_A, OUT_E] {
        for i in 0..2 {
            v.push(send(BusId::IsHalfword, MU, vec![col(c + i)]));
        }
    }
    v.push(send(BusId::AreBytes, MU, vec![col(CARRY_A), col(CARRY_E)]));

    let mut input = vec![col(TS), col(TS + 1), col(INDEX)];
    input.extend([
        bits(A, 32),
        word_bl(B),
        word_bl(C),
        col(D),
        bits(E_, 32),
        word_bl(FF),
        word_bl(G),
        col(H),
    ]);
    let mut output = vec![col(TS), col(TS + 1), lin(vec![(INDEX, 1)], 1)];
    output.extend([
        word_hl(OUT_A),
        bits(A, 32),
        word_bl(B),
        word_bl(C),
        word_hl(OUT_E),
        bits(E_, 32),
        word_bl(FF),
        word_bl(G),
    ]);
    v.push(recv(BusId::ShaRound, MU, input));
    v.push(send(BusId::ShaRound, MU, output));
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
        let word = |b: &B, c: usize| {
            (0..4).fold(b.const_base(0), |acc, j| {
                acc + b.const_base(1u64 << (8 * j)) * b.main(0, c + j)
            })
        };
        let half = |b: &B, c: usize| b.main(0, c) + b.const_base(65536) * b.main(0, c + 1);

        let mut id = 0;
        check_bits(b, &mut id, MU, 1);
        check_bits(b, &mut id, A, 32);
        check_bits(b, &mut id, E_, 32);

        // temp1 = h + S1 + ch + k + w, temp2 = S0 + maj — the spec's virtuals,
        // with ch and maj summed straight out of the BYTE_ALU results.
        let ch = word(b, E_AND_F) + word(b, NOT_E_AND_G);
        let maj = word(b, A_AND_B) + word(b, C_AND_A_XOR_B);
        let temp1 = b.main(0, H) + sigma(b, E_, 3) + ch + b.main(0, K) + b.main(0, W);
        let temp2 = sigma(b, A, 2) + maj;

        b.emit_base(
            id,
            temp1.clone() + temp2 - half(b, OUT_A) - b.const_base(1 << 32) * b.main(0, CARRY_A),
        );
        id += 1;
        b.emit_base(
            id,
            b.main(0, D) + temp1 - half(b, OUT_E) - b.const_base(1 << 32) * b.main(0, CARRY_E),
        );
    }
}

/// The BITWISE lookups this chip performs: twenty `BYTE_ALU`, four `IS_HALF`
/// for the two outputs' half-words, and one paired `ARE_BYTES` for the carries.
pub fn bitwise_ops(ops: &[super::sha256::Operation]) -> Vec<super::bitwise::BitwiseOperation> {
    use super::bitwise::{BitwiseOperation as Op, BitwiseOperationType as Ty};
    let mut v = Vec::with_capacity(ops.len() * 64 * 25);
    for op in ops {
        let w = executor::sha256::schedule(&op.message);
        let mut s = op.state_words();
        for (i, &word) in w.iter().enumerate() {
            let [a, b, c, d, e, f, g, h] = s;
            for (ty, x, y) in [
                (Ty::ByteAluAnd, a, b),
                (Ty::ByteAluXor, a, b),
                (Ty::ByteAluAnd, c, a ^ b),
                (Ty::ByteAluAnd, e, f),
                (Ty::ByteAluAnd, !e, g),
            ] {
                for j in 0..4 {
                    v.push(Op::byte_op(ty, (x >> (8 * j)) as u8, (y >> (8 * j)) as u8));
                }
            }
            let out = executor::sha256::round(s, word, executor::sha256::K[i]);
            for half in [out[0] & 0xffff, out[0] >> 16, out[4] & 0xffff, out[4] >> 16] {
                v.push(Op::halfword(Ty::IsHalf, half as u8, (half >> 8) as u8));
            }
            let ch = (e & f) as u64 + (!e & g) as u64;
            let maj = (a & b) as u64 + (c & (a ^ b)) as u64;
            let temp1 = h as u64
                + executor::sha256::sigma(e, 3) as u64
                + ch
                + executor::sha256::K[i] as u64
                + word as u64;
            let temp2 = executor::sha256::sigma(a, 2) as u64 + maj;
            v.push(Op::halfword(
                Ty::AreBytes,
                ((temp1 + temp2) >> 32) as u8,
                ((d as u64 + temp1) >> 32) as u8,
            ));
            s = out;
        }
    }
    v
}
