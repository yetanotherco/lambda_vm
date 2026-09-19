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
/// Row-at-a-time writer over a table allocated once at its padded size.
///
/// The chips here are wide (SHA256ROUND is 114 columns) and produce a hundred
/// rows per compression call, so collecting them into a `Vec<Vec<u64>>` first
/// would allocate once per row and keep a second copy of the whole table alive
/// while this one is filled.
pub struct TraceRows {
    table: TraceTable<F, E>,
    scratch: Vec<u64>,
    row: usize,
    expected: usize,
}

impl TraceRows {
    pub fn new(num_rows: usize, width: usize) -> Self {
        let n = num_rows.next_power_of_two().max(4);
        Self {
            table: TraceTable::new_main(super::types::zeroed_fe_vec(n * width), width, 1),
            scratch: vec![0; width],
            row: 0,
            expected: num_rows,
        }
    }

    pub fn push(&mut self, fill: impl FnOnce(&mut [u64])) {
        self.scratch.fill(0);
        fill(&mut self.scratch);
        for (c, x) in self.scratch.iter().enumerate() {
            self.table.main_table.set_u64(self.row, c, *x);
        }
        self.row += 1;
    }

    /// The row count passed to [`TraceRows::new`] sizes the table, so a caller
    /// that pushes a different number of rows would silently leave real rows
    /// zeroed (or run past the end).
    pub fn finish(self) -> TraceTable<F, E> {
        debug_assert_eq!(
            self.row, self.expected,
            "pushed {} rows into a table sized for {}",
            self.row, self.expected
        );
        self.table
    }
}
/// `(rotate, rotate, rotate-or-shift, is_shift)` for σ0, σ1, Σ0, Σ1.
const SIGMA: [[usize; 4]; 4] = [[7, 18, 3, 1], [17, 19, 10, 1], [2, 13, 22, 0], [6, 11, 25, 0]];

/// σ_kind or Σ_kind of the 32 bits at `c`, as an expression.
///
/// Bit `i` of a right rotation by `n` is bit `(i+n) mod 32` of the input, and of
/// a right shift it is bit `i+n` or zero — both are free re-indexings. The
/// three-way XOR is `x+y+z − 2(xy+xz+yz) + 4xyz`; where a shift contributes a
/// zero bit it collapses to `x+y−2xy` and to degree 2. This is what lets the
/// round and schedule chips drop their ROTXOR requests — see
/// SHA256_OPT_ZKVM_SURVEY.md for how OpenVM, ZisK and RISC0 do the same.
pub fn sigma<B: ConstraintBuilder<F, E>>(b: &B, c: usize, kind: usize) -> B::Expr {
    let [ra, rb, rc, is_shift] = SIGMA[kind];
    let mut acc = b.const_base(0);
    for i in 0..32 {
        let x = b.main(0, c + (i + ra) % 32);
        let y = b.main(0, c + (i + rb) % 32);
        let bit = if is_shift == 1 && i + rc >= 32 {
            x.clone() + y.clone() - b.const_base(2) * (x * y)
        } else {
            let z = b.main(0, c + if is_shift == 1 { i + rc } else { (i + rc) % 32 });
            x.clone() + y.clone() + z.clone()
                - b.const_base(2)
                    * (x.clone() * y.clone() + x.clone() * z.clone() + y.clone() * z.clone())
                + b.const_base(4) * (x * y * z)
        };
        acc = acc + b.const_base(1u64 << i) * bit;
    }
    acc
}

/// Byte `j` of the 32 bit columns at `c`, or its complement — the form a
/// `BYTE_ALU` operand takes when the word is held as bits.
pub fn byte_bits(c: usize, j: usize, complement: bool) -> BusValue {
    let s = if complement { -1i64 } else { 1 };
    lin(
        (0..8).map(|i| (c + 8 * j + i, s << i)).collect(),
        if complement { 255 } else { 0 },
    )
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
