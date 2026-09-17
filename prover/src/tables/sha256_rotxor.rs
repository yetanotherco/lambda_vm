//! ROTXOR specialized to the four parameter tuples used by SHA256.
//! Bit constraints replace the spec's HWSL/byte lookups, preserving its bus.
use super::{
    sha256_common::*,
    types::{BusId, GoldilocksExtension as E, GoldilocksField as F},
};
use stark::{
    constraints::builder::{ConstraintBuilder, ConstraintSet},
    lookup::{BusInteraction, BusValue},
    trace::TraceTable,
};
pub const WIDTH: usize = 197;
pub const MU: usize = 68;
pub const PARAMS: [[u64; 4]; 4] = [[2, 11, 3, 0], [3, 2, 10, 0], [6, 9, 2, 1], [9, 14, 6, 1]];
pub fn generate(ops: &[(u32, usize)]) -> TraceTable<F, E> {
    let mut rows = TraceRows::new(ops.len(), WIDTH);
    for &(x, k) in ops {
        rows.push(|r| {
            put_bits(r, 0, x as u64, 32);
            put_bits(r, 32, executor::sha256::sigma(x, k) as u64, 32);
            r[64 + k] = 1;
            r[MU] = 1;
            for (j, (a, b)) in [(7, 18), (17, 19), (2, 13), (6, 11)]
                .into_iter()
                .enumerate()
            {
                put_bits(
                    r,
                    69 + 32 * j,
                    (x.rotate_right(a) ^ x.rotate_right(b)) as u64,
                    32,
                );
            }
        });
    }
    rows.finish()
}
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut v = vec![bits(0, 32)];
    for (j, _) in PARAMS[0].iter().enumerate() {
        v.push(lin(
            (0..4).map(|k| (64 + k, PARAMS[k][j] as i64)).collect(),
            0,
        ));
    }
    v.push(bits(32, 32));
    vec![recv(BusId::ShaRot, MU, v)]
}
pub fn request(mu: usize, input: BusValue, kind: usize, out: BusValue) -> BusInteraction {
    let mut v = vec![input];
    v.extend(PARAMS[kind].map(BusValue::constant));
    v.push(out);
    send(BusId::ShaRot, mu, v)
}
#[derive(Clone, Copy)]
pub struct Constraints;
impl ConstraintSet<F, E> for Constraints {
    fn max_degree(&self) -> usize {
        3
    }
    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let mut id = 0;
        check_bits(b, &mut id, 0, WIDTH);
        let sum = (0..4).fold(b.const_base(0), |s, k| s + b.main(0, 64 + k));
        b.emit_base(id, sum - b.main(0, MU));
        id += 1;
        for i in 0..32 {
            let mut expected = b.const_base(0);
            for (k, (r0, r1, r2, rot)) in [
                (7, 18, 3, false),
                (17, 19, 10, false),
                (2, 13, 22, true),
                (6, 11, 25, true),
            ]
            .into_iter()
            .enumerate()
            {
                let x = b.main(0, (i + r0) % 32);
                let y = b.main(0, (i + r1) % 32);
                let z = if rot || i + r2 < 32 {
                    b.main(0, (i + r2) % 32)
                } else {
                    b.const_base(0)
                };
                let xy = b.main(0, 69 + 32 * k + i);
                b.emit_base(
                    id,
                    xy.clone() - (x.clone() + y.clone() - b.const_base(2) * x * y),
                );
                id += 1;
                let xor = xy.clone() + z.clone() - b.const_base(2) * xy * z;
                expected = expected + b.main(0, 64 + k) * xor;
            }
            b.emit_base(id, b.main(0, 32 + i) - expected);
            id += 1;
        }
    }
}
