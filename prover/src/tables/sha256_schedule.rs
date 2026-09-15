//! SHA256_M lookback schedule, words 16..63. Dependency multiplicities are
//! derived from the fixed schedule DAG; every word is bound to this invocation.
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
pub const WIDTH: usize = 46;
pub const MU: usize = 45;
// timestamp 0..2, index 2, back2/back7/back15/back16 3..7, s0/s1 7..9,
// out bits 9..41, carry bits 41..43, amount 43, index-16 44, mu 45.
pub fn amount(i: usize) -> u64 {
    1 + [2, 7, 15, 16]
        .into_iter()
        .filter(|d| i + d >= 16 && i + d < 64)
        .count() as u64
}
pub fn generate(ops: &[super::sha256::Operation]) -> TraceTable<F, E> {
    let mut rows = vec![];
    for op in ops {
        let w = executor::sha256::schedule(&op.message);
        for i in 16..64 {
            let mut r = vec![0; WIDTH];
            r[0] = op.timestamp & 0xffffffff;
            r[1] = op.timestamp >> 32;
            r[2] = i as u64;
            for (j, d) in [2, 7, 15, 16].into_iter().enumerate() {
                r[3 + j] = w[i - d] as u64;
            }
            r[7] = executor::sha256::sigma(w[i - 15], 0) as u64;
            r[8] = executor::sha256::sigma(w[i - 2], 1) as u64;
            put_bits(&mut r, 9, w[i] as u64, 32);
            let sum = r[6] + r[7] + r[4] + r[8];
            put_bits(&mut r, 41, sum >> 32, 2);
            r[43] = amount(i);
            r[44] = (i - 16) as u64;
            r[MU] = 1;
            rows.push(r);
        }
    }
    trace(rows, WIDTH)
}
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut v = vec![];
    for (j, d) in [2, 7, 15, 16].into_iter().enumerate() {
        v.push(send(
            BusId::ShaM,
            MU,
            vec![col(0), col(1), lin(vec![(2, 1)], -d), col(3 + j)],
        ));
    }
    v.push(rot::request(MU, col(5), 0, col(7)));
    v.push(rot::request(MU, col(3), 1, col(8)));
    v.push(recv(
        BusId::ShaM,
        43,
        vec![col(0), col(1), col(2), bits(9, 32)],
    ));
    v.push(send(
        BusId::AreBytes,
        MU,
        vec![col(44), stark::lookup::BusValue::constant(0)],
    ));
    v
}
#[derive(Clone, Copy)]
pub struct Constraints;
impl ConstraintSet<F, E> for Constraints {
    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let mut id = 0;
        check_bits(b, &mut id, 9, 34);
        check_bits(b, &mut id, MU, 1);
        b.emit_base(
            id,
            b.main(0, 6) + b.main(0, 7) + b.main(0, 4) + b.main(0, 8)
                - word(b, 9, 32)
                - b.const_base(1 << 32) * word(b, 41, 2),
        );
        id += 1;
        b.emit_base(
            id,
            b.main(0, MU) * (b.main(0, 2) - b.const_base(16) - b.main(0, 44)),
        );
        id += 1;
        b.emit_base(id, (b.const_base(1) - b.main(0, MU)) * b.main(0, 43));
    }
}
