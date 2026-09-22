//! SHA256ROUND: the spec's round chain, with bit-decomposed state to constrain
//! Ch/Maj directly. This substitutes local bit gates for BYTE_ALU lookups.
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
pub const STATE: usize = 3;
pub const OUT: usize = 259;
pub const S0: usize = 323;
pub const S1: usize = 324;
pub const W: usize = 325;
pub const K: usize = 326;
pub const CARRY: usize = 327;
pub const MU: usize = 333;
pub const WIDTH: usize = 334;
pub fn generate(ops: &[super::sha256::Operation]) -> TraceTable<F, E> {
    let mut rows = TraceRows::new(ops.len() * 64, WIDTH);
    for op in ops {
        let w = executor::sha256::schedule(&op.message);
        let mut s = op.state_words();
        for (i, &word) in w.iter().enumerate() {
            let out = executor::sha256::round(s, word, executor::sha256::K[i]);
            rows.push(|r| {
                r[0] = op.timestamp & 0xffffffff;
                r[1] = op.timestamp >> 32;
                r[2] = i as u64;
                for (j, &state_word) in s.iter().enumerate() {
                    put_bits(r, STATE + j * 32, state_word as u64, 32);
                }
                put_bits(r, OUT, out[0] as u64, 32);
                put_bits(r, OUT + 32, out[4] as u64, 32);
                r[S0] = executor::sha256::sigma(s[0], 2) as u64;
                r[S1] = executor::sha256::sigma(s[4], 3) as u64;
                r[W] = word as u64;
                r[K] = executor::sha256::K[i] as u64;
                let t1 =
                    s[7] as u64 + r[S1] + ((s[4] & s[5]) ^ (!s[4] & s[6])) as u64 + r[W] + r[K];
                let t2 = r[S0] + ((s[0] & s[1]) ^ (s[0] & s[2]) ^ (s[1] & s[2])) as u64;
                put_bits(r, CARRY, (t1 + t2) >> 32, 3);
                put_bits(r, CARRY + 3, (s[3] as u64 + t1) >> 32, 3);
                r[MU] = 1;
            });
            s = out;
        }
    }
    rows.finish()
}
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut input = vec![col(0), col(1), col(2)];
    input.extend((0..8).map(|j| bits(STATE + 32 * j, 32)));
    let mut output = vec![col(0), col(1), lin(vec![(2, 1)], 1)];
    output.extend(
        [
            OUT,
            STATE,
            STATE + 32,
            STATE + 64,
            OUT + 32,
            STATE + 128,
            STATE + 160,
            STATE + 192,
        ]
        .map(|c| bits(c, 32)),
    );
    vec![
        recv(BusId::ShaRound, MU, input),
        send(BusId::ShaRound, MU, output),
        send(BusId::ShaM, MU, vec![col(0), col(1), col(2), col(W)]),
        send(BusId::ShaK, MU, vec![col(2), col(K)]),
        rot::request(MU, bits(STATE, 32), 2, col(S0)),
        rot::request(MU, bits(STATE + 128, 32), 3, col(S1)),
    ]
}
#[derive(Clone, Copy)]
pub struct Constraints;
impl ConstraintSet<F, E> for Constraints {
    fn max_degree(&self) -> usize {
        3
    }
    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let mut id = 0;
        check_bits(b, &mut id, STATE, 320);
        check_bits(b, &mut id, CARRY, 6);
        check_bits(b, &mut id, MU, 1);
        let mut ch = b.const_base(0);
        let mut maj = b.const_base(0);
        for i in 0..32 {
            let a = b.main(0, STATE + i);
            let bb = b.main(0, STATE + 32 + i);
            let c = b.main(0, STATE + 64 + i);
            let e = b.main(0, STATE + 128 + i);
            let f = b.main(0, STATE + 160 + i);
            let g = b.main(0, STATE + 192 + i);
            let m =
                a.clone() * bb.clone() + c * (a.clone() + bb.clone() - b.const_base(2) * a * bb);
            maj = maj + b.const_base(1u64 << i) * m;
            ch = ch + b.const_base(1u64 << i) * (e.clone() * f + (b.const_base(1) - e) * g);
        }
        let t1 = word(b, STATE + 224, 32) + b.main(0, S1) + ch + b.main(0, W) + b.main(0, K);
        let t2 = b.main(0, S0) + maj;
        b.emit_base(
            id,
            t1.clone() + t2 - word(b, OUT, 32) - b.const_base(1 << 32) * word(b, CARRY, 3),
        );
        id += 1;
        b.emit_base(
            id,
            word(b, STATE + 96, 32) + t1
                - word(b, OUT + 32, 32)
                - b.const_base(1 << 32) * word(b, CARRY + 3, 3),
        );
    }
}
