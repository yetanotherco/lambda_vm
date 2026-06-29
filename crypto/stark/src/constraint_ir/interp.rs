//! CPU interpreter for a captured [`ConstraintProgram`].
//!
//! A single forward pass over the topologically ordered nodes evaluates each
//! node into a [`Value`] (base `D1` or extension `D3`), reusing the real
//! `FieldElement` arithmetic so per-op results are bit-identical to the boxed
//! constraint path. Mixed-dimension ops auto-embed the `D1` operand into `D3`,
//! mirroring the field tower's `F: IsSubFieldOf<E>` arithmetic.

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as GoldilocksExtension;
use math::field::goldilocks::GoldilocksField;

use super::ir::{ConstraintProgram, Dim, Op};

type Fp = FieldElement<GoldilocksField>;
type Fp3 = FieldElement<GoldilocksExtension>;

/// A node's computed value: base field (`D1`) or degree-3 extension (`D3`).
#[derive(Clone, Copy, Debug)]
enum Value {
    D1(Fp),
    D3(Fp3),
}

impl Value {
    /// Promote to the extension field, embedding a base value if needed.
    fn to_ext(self) -> Fp3 {
        match self {
            Value::D1(x) => x.to_extension::<GoldilocksExtension>(),
            Value::D3(x) => x,
        }
    }

    fn as_base(self) -> Fp {
        match self {
            Value::D1(x) => x,
            Value::D3(_) => {
                panic!("expected a base (D1) value but found an extension (D3) value")
            }
        }
    }
}

/// Evaluate the program's single root over a base-field main row.
///
/// `main_row[col]` resolves `Var { main: true, col, .. }` leaves. The minimal
/// algebraic constraint set only reads main columns at offset 0, row 0 and
/// returns a base-field (`D1`) value, so this returns a `FieldElement<F>`.
pub fn eval_program_base(prog: &ConstraintProgram, main_row: &[Fp]) -> Fp {
    let mut values: Vec<Value> = Vec::with_capacity(prog.nodes.len());

    for (i, op) in prog.nodes.iter().enumerate() {
        let v = match *op {
            Op::Const1(c) => Value::D1(Fp::from(c)),
            Op::Const3([c0, c1, c2]) => {
                Value::D3(Fp3::from_raw([Fp::from(c0), Fp::from(c1), Fp::from(c2)]))
            }
            Op::Var { main, row, col, .. } => {
                assert!(main, "aux leaves are not part of the minimal algebraic set");
                assert_eq!(row, 0, "minimal set reads row 0 only");
                Value::D1(main_row[col as usize])
            }
            Op::Add(a, b) => binop(&values, a, b, prog.dims[i], |x, y| x + y, |x, y| x + y),
            Op::Sub(a, b) => binop(&values, a, b, prog.dims[i], |x, y| x - y, |x, y| x - y),
            Op::Mul(a, b) => binop(&values, a, b, prog.dims[i], |x, y| x * y, |x, y| x * y),
            Op::Neg(a) => match (values[a as usize], prog.dims[i]) {
                (Value::D1(x), Dim::D1) => Value::D1(-x),
                (val, Dim::D3) => Value::D3(-val.to_ext()),
                (Value::D3(x), Dim::D1) => Value::D3(-x), // dim mismatch, keep ext
            },
            Op::Embed(a) => Value::D3(values[a as usize].to_ext()),
        };
        values.push(v);
    }

    let root = prog.roots[0];
    values[root as usize].as_base()
}

/// Apply a binary op, auto-embedding to the extension field when the result
/// dimension is `D3` (or either operand is already `D3`).
#[inline]
fn binop(
    values: &[Value],
    a: u32,
    b: u32,
    result_dim: Dim,
    base_op: impl Fn(Fp, Fp) -> Fp,
    ext_op: impl Fn(Fp3, Fp3) -> Fp3,
) -> Value {
    let va = values[a as usize];
    let vb = values[b as usize];
    match (va, vb, result_dim) {
        (Value::D1(x), Value::D1(y), Dim::D1) => Value::D1(base_op(x, y)),
        _ => Value::D3(ext_op(va.to_ext(), vb.to_ext())),
    }
}
