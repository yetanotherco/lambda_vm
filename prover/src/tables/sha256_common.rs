//! Shared encodings for the SHA-256 chips. Words use least-significant bit first
//! internally; the core chip binds these words to big-endian memory bytes.
use super::types::{BusId, GoldilocksExtension as E, GoldilocksField as F, VmTable};
use stark::constraints::builder::ConstraintBuilder;
use stark::{
    lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing},
    trace::TraceTable,
};
pub fn col(c: usize) -> BusValue {
    BusValue::Packed {
        start_column: c,
        packing: Packing::Direct,
    }
}
pub fn lin(v: Vec<(usize, i64)>, k: i64) -> BusValue {
    BusValue::linear(
        v.into_iter()
            .map(|(column, coefficient)| LinearTerm::Column {
                column,
                coefficient,
            })
            .chain(std::iter::once(LinearTerm::Constant(k)))
            .collect(),
    )
}
pub fn bits(c: usize, n: usize) -> BusValue {
    lin((0..n).map(|i| (c + i, 1i64 << i)).collect(), 0)
}
pub fn be(c: usize) -> BusValue {
    lin((0..4).map(|i| (c + i, 1i64 << (8 * (3 - i)))).collect(), 0)
}
pub fn half(c: usize) -> BusValue {
    lin(vec![(c, 1), (c + 1, 65536)], 0)
}
pub fn send(bus: BusId, mu: usize, v: Vec<BusValue>) -> BusInteraction {
    BusInteraction::sender(bus, Multiplicity::Column(mu), v)
}
pub fn recv(bus: BusId, mu: usize, v: Vec<BusValue>) -> BusInteraction {
    BusInteraction::receiver(bus, Multiplicity::Column(mu), v)
}
pub fn trace(rows: Vec<Vec<u64>>, width: usize) -> TraceTable<F, E> {
    let n = rows.len().next_power_of_two().max(4);
    let mut t = TraceTable::new_main(super::types::zeroed_fe_vec(n * width), width, 1);
    for (r, row) in rows.iter().enumerate() {
        for (c, x) in row.iter().enumerate() {
            t.main_table.set_u64(r, c, *x);
        }
    }
    t
}
pub fn put_bits(row: &mut [u64], c: usize, x: u64, n: usize) {
    for i in 0..n {
        row[c + i] = (x >> i) & 1;
    }
}
pub fn word<B: ConstraintBuilder<F, E>>(b: &B, c: usize, n: usize) -> B::Expr {
    let mut x = b.const_base(0);
    for i in 0..n {
        x = x + b.main(0, c + i) * b.const_base(1u64 << i);
    }
    x
}
pub fn check_bits<B: ConstraintBuilder<F, E>>(b: &mut B, idx: &mut usize, c: usize, n: usize) {
    for i in 0..n {
        let x = b.main(0, c + i);
        b.emit_base(*idx, x.clone() * (x - b.const_base(1)));
        *idx += 1;
    }
}
