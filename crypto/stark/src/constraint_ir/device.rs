//! Device (GPU) lowering of a Goldilocks [`ConstraintProgram`] to a flat,
//! `#[repr(C)]` blob a CUDA interpreter kernel can walk directly — plus a CPU
//! reference walker over that same blob.
//!
//! This is the *one* concrete lowering point: the rest of the constraint IR is
//! field-generic (`<F, E>`), but genericity does not cross to CUDA, so here we
//! commit to the Goldilocks base field and its degree-3 extension. Any other
//! field tower never reaches this module — it stays on the generic
//! [`interp`](super::interp) path.
//!
//! Two things live here:
//!
//! - [`DeviceProgram::lower`] — serialize a `ConstraintProgram<Goldilocks,
//!   GoldilocksExt3>` into flat, device-uploadable arrays: a `#[repr(C)]`
//!   [`DeviceNode`] list (with *normalized* dim tags), `u64` / `[u64; 3]`
//!   constant tables, `roots`, `num_base`, and the packed per-thread scratch
//!   layout `val_offsets` (base node = 1 `u64` slot, ext node = 3) the kernel's
//!   dim-split walk addresses. Field constants become raw limbs via `FieldElement::to_raw`,
//!   which is byte-identical to how [`crate::gpu_lde`] already hands Goldilocks
//!   elements to the device (a `#[repr(transparent)]` `u64` / `[u64; 3]`).
//!
//! - [`eval_device_program`] — a CPU forward pass over the *flat* node array
//!   (not the [`Op`] enum), decoding leaves from the raw limb tables and
//!   reproducing the exact [`interp::run`](super::interp) semantics
//!   (per-node [`Dim`] drives base-vs-extension arithmetic, mixed operands
//!   auto-embed). It is the model of the GPU kernel's per-thread walk, so a
//!   bit-for-bit match against [`eval_program`](super::interp::eval_program)
//!   pins the on-device layout and control flow *before* any CUDA exists. The
//!   kernel is then a transliteration of this walk with the `FieldElement`
//!   arithmetic swapped for `goldilocks.cuh` / `ext3.cuh`.

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as GoldilocksExtension;
use math::field::goldilocks::GoldilocksField;

use super::ir::{ConstraintProgram, Dim, Op};

type FpE = FieldElement<GoldilocksField>;
type Ext3E = FieldElement<GoldilocksExtension>;

// -------------------------------------------------------------------------
// Wire tags — MUST match the CUDA kernel's `switch (op)` and dim checks.
// -------------------------------------------------------------------------

/// `a` = index into `base_consts`.
pub const OP_CONST_BASE: u32 = 0;
/// `a` = index into `ext_consts`.
pub const OP_CONST_EXT: u32 = 1;
/// Trace-cell read; `a`/`b` pack the [`Op::Var`] fields (see [`pack_var`]).
pub const OP_VAR: u32 = 2;
/// `a` = index into the per-proof `rap_challenges` uniform buffer.
pub const OP_RAP_CHALLENGE: u32 = 3;
/// `a` = index into the per-proof `logup_alpha_powers` uniform buffer.
pub const OP_ALPHA_POW: u32 = 4;
/// The per-proof LogUp table offset uniform; no operands.
pub const OP_TABLE_OFFSET: u32 = 5;
/// `a`, `b` = node ids.
pub const OP_ADD: u32 = 6;
/// `a`, `b` = node ids.
pub const OP_SUB: u32 = 7;
/// `a`, `b` = node ids.
pub const OP_MUL: u32 = 8;
/// `a` = node id.
pub const OP_NEG: u32 = 9;
/// `a` = node id (base → extension embed).
pub const OP_EMBED: u32 = 10;

/// Node result is a base-field value.
pub const DIM_BASE: u32 = 0;
/// Node result is an extension-field value.
pub const DIM_EXT: u32 = 1;

/// One flattened IR instruction: 16 bytes, `#[repr(C)]` for a 1:1 device
/// upload. `op` is an `OP_*` tag; the meaning of `a`/`b` depends on `op` (node
/// ids for arithmetic, table indices for constants/uniforms, packed [`Op::Var`]
/// fields for [`OP_VAR`]); `dim` is [`DIM_BASE`] or [`DIM_EXT`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceNode {
    pub op: u32,
    pub a: u32,
    pub b: u32,
    pub dim: u32,
}

/// Pack an [`Op::Var`]'s fields into the `(a, b)` operand words:
/// `a` holds `col`; `b` holds `main` (bit 16), `offset` (bits 8..16), `row`
/// (bits 0..8). The CUDA kernel unpacks with the same layout.
#[inline]
pub fn pack_var(main: bool, offset: u8, row: u8, col: u16) -> (u32, u32) {
    let a = col as u32;
    let b = ((main as u32) << 16) | ((offset as u32) << 8) | (row as u32);
    (a, b)
}

/// Inverse of [`pack_var`]: `(main, offset, row, col)`.
#[inline]
pub fn unpack_var(a: u32, b: u32) -> (bool, u8, u8, u16) {
    let col = (a & 0xFFFF) as u16;
    let main = ((b >> 16) & 1) != 0;
    let offset = ((b >> 8) & 0xFF) as u8;
    let row = (b & 0xFF) as u8;
    (main, offset, row, col)
}

/// A [`ConstraintProgram`] lowered to flat, device-uploadable arrays. Constants
/// are canonical raw limbs (`u64` base / `[u64; 3]` extension), matching the
/// `#[repr(transparent)]` layout the GPU trace buffers already use.
#[derive(Clone, Debug)]
pub struct DeviceProgram {
    /// Topologically ordered instruction list (id `i` references only `< i`).
    pub nodes: Vec<DeviceNode>,
    /// Base-field constant table, indexed by [`OP_CONST_BASE`].
    pub base_consts: Vec<u64>,
    /// Extension-field constant table, indexed by [`OP_CONST_EXT`].
    pub ext_consts: Vec<[u64; 3]>,
    /// Per-constraint root node ids, indexed by `constraint_idx`.
    pub roots: Vec<u32>,
    /// Number of leading ([`DIM_BASE`]-rooted) constraints written to
    /// `base_evals`; the rest go to `ext_evals`.
    pub num_base: u32,
    /// Per-node scratch slot offsets (prefix sums, `nodes.len() + 1` entries):
    /// a [`DIM_BASE`] node owns 1 `u64` slot, a [`DIM_EXT`] node owns 3, so
    /// node `i`'s value lives at slots `val_offsets[i]..val_offsets[i+1]` of
    /// the kernel's per-thread scratch and an operand's width is the offset
    /// delta. Consistent with the (normalized) `dim` tags by construction.
    pub val_offsets: Vec<u32>,
}

impl DeviceProgram {
    /// Lower a concrete-Goldilocks [`ConstraintProgram`] to its flat device
    /// form. Pure serialization — no field arithmetic, no device access.
    ///
    /// Dim tags are *normalized*: a node's wire `dim` is [`DIM_BASE`] iff its
    /// value is statically base — an inherently base leaf (base constant,
    /// main-trace read) or a `Base`-tagged arithmetic node whose operands are
    /// all base. The builder's typing join already guarantees this shape, so
    /// normalization is an identity in practice; re-deriving it here turns
    /// `DIM_BASE` into a wire-format guarantee the kernel's split base
    /// arithmetic can rely on. A hypothetical mismatched `Base` tag degrades to
    /// [`DIM_EXT`], which evaluates identically — the same fallback
    /// [`eval_device_program`]'s dynamic rule applies.
    pub fn lower(prog: &ConstraintProgram<GoldilocksField, GoldilocksExtension>) -> Self {
        let mut is_base: Vec<bool> = Vec::with_capacity(prog.nodes.len());
        let nodes = prog
            .nodes
            .iter()
            .zip(prog.dims.iter())
            .map(|(op, dim)| {
                let tagged_base = matches!(dim, Dim::Base);
                let (op, a, b, base) = match *op {
                    Op::ConstBase(idx) => (OP_CONST_BASE, idx, 0, true),
                    Op::ConstExt(idx) => (OP_CONST_EXT, idx, 0, false),
                    Op::Var {
                        main,
                        offset,
                        row,
                        col,
                    } => {
                        let (a, b) = pack_var(main, offset, row, col);
                        (OP_VAR, a, b, main)
                    }
                    Op::RapChallenge { idx } => (OP_RAP_CHALLENGE, idx as u32, 0, false),
                    Op::AlphaPow { idx } => (OP_ALPHA_POW, idx as u32, 0, false),
                    Op::TableOffset => (OP_TABLE_OFFSET, 0, 0, false),
                    Op::Add(a, b) => (
                        OP_ADD,
                        a,
                        b,
                        tagged_base && is_base[a as usize] && is_base[b as usize],
                    ),
                    Op::Sub(a, b) => (
                        OP_SUB,
                        a,
                        b,
                        tagged_base && is_base[a as usize] && is_base[b as usize],
                    ),
                    Op::Mul(a, b) => (
                        OP_MUL,
                        a,
                        b,
                        tagged_base && is_base[a as usize] && is_base[b as usize],
                    ),
                    Op::Neg(a) => (OP_NEG, a, 0, tagged_base && is_base[a as usize]),
                    Op::Embed(a) => (OP_EMBED, a, 0, false),
                };
                is_base.push(base);
                DeviceNode {
                    op,
                    a,
                    b,
                    dim: if base { DIM_BASE } else { DIM_EXT },
                }
            })
            .collect();

        // Packed scratch layout: base nodes take 1 u64 slot, ext nodes 3.
        let mut val_offsets = Vec::with_capacity(is_base.len() + 1);
        let mut off = 0u32;
        for &base in &is_base {
            val_offsets.push(off);
            off += if base { 1 } else { 3 };
        }
        val_offsets.push(off);

        let base_consts = prog.base_consts.iter().map(|c| *c.value()).collect();
        let ext_consts = prog.ext_consts.iter().map(encode_ext).collect();

        DeviceProgram {
            nodes,
            base_consts,
            ext_consts,
            roots: prog.roots.clone(),
            num_base: prog.num_base as u32,
            val_offsets,
        }
    }
}

/// A node's computed value during the walk: base or extension field element.
#[derive(Clone)]
enum Value {
    Base(FpE),
    Ext(Ext3E),
}

impl Value {
    /// Promote to the extension field, embedding a base value if needed.
    fn to_ext(&self) -> Ext3E {
        match self {
            Value::Base(x) => (*x).to_extension::<GoldilocksExtension>(),
            Value::Ext(x) => *x,
        }
    }

    fn as_base(&self) -> &FpE {
        match self {
            Value::Base(x) => x,
            Value::Ext(_) => panic!("expected a base value but found an extension value"),
        }
    }
}

/// `[u64; 3]` → extension element.
#[inline]
fn decode_ext(limbs: [u64; 3]) -> Ext3E {
    Ext3E::from_raw([
        FpE::from_raw(limbs[0]),
        FpE::from_raw(limbs[1]),
        FpE::from_raw(limbs[2]),
    ])
}

/// Extension element → `[u64; 3]`.
#[inline]
fn encode_ext(x: &Ext3E) -> [u64; 3] {
    let limbs = x.value();
    [*limbs[0].value(), *limbs[1].value(), *limbs[2].value()]
}

/// Apply a binary op, auto-embedding to the extension when the result dimension
/// is [`DIM_EXT`] (or either operand is already an extension value) — the exact
/// rule of [`interp::binop`](super::interp).
#[inline]
fn binop(
    values: &[Value],
    a: u32,
    b: u32,
    dim: u32,
    base_op: impl Fn(FpE, FpE) -> FpE,
    ext_op: impl Fn(Ext3E, Ext3E) -> Ext3E,
) -> Value {
    let va = &values[a as usize];
    let vb = &values[b as usize];
    if dim == DIM_BASE
        && let (Value::Base(x), Value::Base(y)) = (va, vb)
    {
        return Value::Base(base_op(*x, *y));
    }
    Value::Ext(ext_op(va.to_ext(), vb.to_ext()))
}

/// Full prover-shaped forward pass over the *flat* device blob, in raw limbs —
/// the CPU model of the GPU kernel. Mirrors
/// [`eval_program`](super::interp::eval_program): base-rooted constraints
/// (`c < num_base`) land in `base_evals`, the rest in `ext_evals`, with the
/// same auto-embed semantics.
///
/// `main[offset][col]` / `aux[offset][col]` are the frame's trace cells;
/// `rap_challenges` / `alpha_powers` / `table_offset` are the per-proof
/// uniforms. All values are the same raw `u64` / `[u64; 3]` limbs the device
/// buffers carry.
#[allow(clippy::too_many_arguments)]
pub fn eval_device_program(
    dev: &DeviceProgram,
    main: &[Vec<u64>],
    aux: &[Vec<[u64; 3]>],
    rap_challenges: &[[u64; 3]],
    alpha_powers: &[[u64; 3]],
    table_offset: [u64; 3],
    base_evals: &mut [u64],
    ext_evals: &mut [[u64; 3]],
) {
    let mut values: Vec<Value> = Vec::with_capacity(dev.nodes.len());

    for node in &dev.nodes {
        let v = match node.op {
            OP_CONST_BASE => Value::Base(FpE::from_raw(dev.base_consts[node.a as usize])),
            OP_CONST_EXT => Value::Ext(decode_ext(dev.ext_consts[node.a as usize])),
            OP_VAR => {
                let (is_main, offset, _row, col) = unpack_var(node.a, node.b);
                if is_main {
                    Value::Base(FpE::from_raw(main[offset as usize][col as usize]))
                } else {
                    Value::Ext(decode_ext(aux[offset as usize][col as usize]))
                }
            }
            OP_RAP_CHALLENGE => Value::Ext(decode_ext(rap_challenges[node.a as usize])),
            OP_ALPHA_POW => Value::Ext(decode_ext(alpha_powers[node.a as usize])),
            OP_TABLE_OFFSET => Value::Ext(decode_ext(table_offset)),
            OP_ADD => binop(
                &values,
                node.a,
                node.b,
                node.dim,
                |x, y| x + y,
                |x, y| x + y,
            ),
            OP_SUB => binop(
                &values,
                node.a,
                node.b,
                node.dim,
                |x, y| x - y,
                |x, y| x - y,
            ),
            OP_MUL => binop(
                &values,
                node.a,
                node.b,
                node.dim,
                |x, y| x * y,
                |x, y| x * y,
            ),
            OP_NEG => {
                let val = &values[node.a as usize];
                if node.dim == DIM_BASE {
                    match val {
                        Value::Base(x) => Value::Base(-x),
                        // Dim/value mismatch: keep it in the extension, as interp does.
                        Value::Ext(x) => Value::Ext(-*x),
                    }
                } else {
                    Value::Ext(-val.to_ext())
                }
            }
            OP_EMBED => Value::Ext(values[node.a as usize].to_ext()),
            other => panic!("unknown device op tag {other}"),
        };
        values.push(v);
    }

    for (c, &root) in dev.roots.iter().enumerate() {
        let v = &values[root as usize];
        if (c as u32) < dev.num_base {
            base_evals[c] = *v.as_base().value();
        } else {
            ext_evals[c] = encode_ext(&v.to_ext());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint_ir::builder::IrBuilder;
    use crate::constraint_ir::interp::eval_program;
    use crate::frame::Frame;
    use crate::table::TableView;
    use crate::traits::TransitionEvaluationContext;

    type Gl = GoldilocksField;
    type Ext = GoldilocksExtension;

    fn fp(v: u64) -> FpE {
        FpE::from(v)
    }
    fn ext3(a: u64, b: u64, c: u64) -> Ext3E {
        Ext3E::from_raw([fp(a), fp(b), fp(c)])
    }

    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn ext(&mut self) -> Ext3E {
            ext3(self.next_u64(), self.next_u64(), self.next_u64())
        }
    }

    #[test]
    fn pack_var_roundtrips() {
        for &(m, off, row, col) in &[
            (true, 0u8, 0u8, 0u16),
            (false, 1, 0, 7),
            (true, 3, 0, 65535),
            (false, 255, 255, 12345),
        ] {
            let (a, b) = pack_var(m, off, row, col);
            assert_eq!(unpack_var(a, b), (m, off, row, col));
        }
    }

    /// A program that exercises every `Op` variant and both dims, with a
    /// base-rooted constraint and extension (LogUp-shaped) roots, next-row
    /// reads, and mixed base×ext arithmetic. Roots: 0 (base) is a pure base
    /// expression; 1 and 2 (ext) touch challenges, alpha powers, table offset,
    /// aux, embed and negation.
    fn all_ops_program() -> ConstraintProgram<Gl, Ext> {
        let mut b = IrBuilder::<Gl, Ext>::new();

        // Root 0 (base): (m0 + m1) * 2 - m0_next , all base, incl. next-row.
        let m0 = b.main(0, 0);
        let m1 = b.main(0, 1);
        let m0n = b.main(1, 0);
        let two = b.const_base(2);
        let sum = b.add(m0, m1);
        let scaled = b.mul(sum, two);
        let base_root = b.sub(scaled, m0n);
        b.emit(0, base_root);

        // Root 1 (ext): m0 * challenge(0) + alpha_pow(1) * aux(0,0) - table_offset
        let ch = b.challenge(0);
        let ap = b.alpha_power(1);
        let au = b.aux(0, 0);
        let off = b.table_offset();
        let t1 = b.mul(m0, ch); // base × ext → ext (auto-embed)
        let t2 = b.mul(ap, au); // ext × ext
        let s = b.add(t1, t2);
        let ext_root = b.sub(s, off);
        b.emit(1, ext_root);

        // Root 2 (ext): embed(m1) + (-aux(0,1)) + const_ext
        let em = b.embed(m1);
        let au1 = b.aux(0, 1);
        let nau1 = b.neg(au1); // ext negation
        let ce = b.const_ext(ext3(9, 8, 7));
        let s2 = b.add(em, nau1);
        let ext_root2 = b.add(s2, ce);
        b.emit(2, ext_root2);

        b.finish(1) // 1 base root, 2 ext roots
    }

    #[test]
    fn lower_val_offsets_and_dims_are_consistent() {
        let prog = all_ops_program();
        let dev = DeviceProgram::lower(&prog);

        // Prefix sums: one u64 slot per DIM_BASE node, three per DIM_EXT node.
        assert_eq!(dev.val_offsets.len(), dev.nodes.len() + 1);
        let mut off = 0u32;
        for (i, n) in dev.nodes.iter().enumerate() {
            assert_eq!(dev.val_offsets[i], off, "offset of node {i}");
            off += if n.dim == DIM_BASE { 1 } else { 3 };
        }
        assert_eq!(*dev.val_offsets.last().unwrap(), off, "total slots");

        // Builder programs are already well-tagged: normalization is an
        // identity, so wire dims match the builder dims.
        for (n, dim) in dev.nodes.iter().zip(prog.dims.iter()) {
            let expected = match dim {
                Dim::Base => DIM_BASE,
                Dim::Ext => DIM_EXT,
            };
            assert_eq!(n.dim, expected);
        }

        // Root widths: root 0 is base (1 slot), roots 1 and 2 are ext (3).
        let width = |id: u32| dev.val_offsets[id as usize + 1] - dev.val_offsets[id as usize];
        assert_eq!(width(dev.roots[0]), 1);
        assert_eq!(width(dev.roots[1]), 3);
        assert_eq!(width(dev.roots[2]), 3);
    }

    #[test]
    fn lower_normalizes_mismatched_base_dim() {
        // Hand-crafted mismatch (unreachable via IrBuilder's typing join): an
        // Add tagged `Base` whose second operand is an aux (Ext) read. `lower`
        // must degrade the wire dim to DIM_EXT (3-slot scratch) so the kernel's
        // base arithmetic never sees an ext operand, and the walk must still
        // match the generic interpreter, whose dynamic rule applies the same
        // ext fallback.
        let prog = ConstraintProgram::<Gl, Ext> {
            nodes: vec![
                Op::Var {
                    main: true,
                    offset: 0,
                    row: 0,
                    col: 0,
                },
                Op::Var {
                    main: false,
                    offset: 0,
                    row: 0,
                    col: 0,
                },
                Op::Add(0, 1),
            ],
            dims: vec![Dim::Base, Dim::Ext, Dim::Base],
            base_consts: vec![],
            ext_consts: vec![],
            roots: vec![2],
            num_base: 0,
        };
        let dev = DeviceProgram::lower(&prog);
        assert_eq!(dev.nodes[0].dim, DIM_BASE);
        assert_eq!(dev.nodes[1].dim, DIM_EXT);
        assert_eq!(dev.nodes[2].dim, DIM_EXT, "mismatched Base tag degrades");
        assert_eq!(dev.val_offsets, vec![0, 1, 4, 7]);

        let mut rng = SplitMix64(0xFEED_FACE_0BAD_F00D);
        for _ in 0..100 {
            let m = fp(rng.next_u64());
            let a = rng.ext();

            let steps = vec![TableView::<Gl, Ext>::new(vec![vec![m]], vec![vec![a]])];
            let frame = Frame::<Gl, Ext>::new(steps);
            let table_offset = Ext3E::zero();
            let ctx = TransitionEvaluationContext::new_prover(
                frame.as_row_frame(),
                &[],
                &[],
                &table_offset,
            );
            let mut ext_ref = vec![Ext3E::zero(); 1];
            eval_program(&prog, &ctx, &mut [], &mut ext_ref);

            let mut ext_dev = vec![[0u64; 3]; 1];
            eval_device_program(
                &dev,
                &[vec![*m.value()]],
                &[vec![encode_ext(&a)]],
                &[],
                &[],
                [0, 0, 0],
                &mut [],
                &mut ext_dev,
            );
            assert_eq!(ext_dev[0], encode_ext(&ext_ref[0]));
        }
    }

    #[test]
    fn device_walk_matches_interp_all_ops() {
        let prog = all_ops_program();
        let dev = DeviceProgram::lower(&prog);

        let mut rng = SplitMix64(0x0123_4567_89AB_CDEF);
        for _ in 0..1000 {
            // Two frame steps (offset 0 and 1), 2 main cols + 2 aux cols each.
            let main_vals: Vec<Vec<FpE>> = (0..2)
                .map(|_| vec![fp(rng.next_u64()), fp(rng.next_u64())])
                .collect();
            let aux_vals: Vec<Vec<Ext3E>> = (0..2).map(|_| vec![rng.ext(), rng.ext()]).collect();
            let rap = vec![rng.ext(), rng.ext()];
            let alpha = vec![rng.ext(), rng.ext()];
            let offset = rng.ext();

            // Reference: the generic interpreter over the ConstraintProgram.
            let steps: Vec<TableView<Gl, Ext>> = main_vals
                .iter()
                .zip(aux_vals.iter())
                .map(|(m, a)| TableView::<Gl, Ext>::new(vec![m.clone()], vec![a.clone()]))
                .collect();
            let frame = Frame::<Gl, Ext>::new(steps);
            let ctx = TransitionEvaluationContext::new_prover(
                frame.as_row_frame(),
                &rap,
                &alpha,
                &offset,
            );
            let mut base_ref = vec![FpE::zero(); 1];
            let mut ext_ref = vec![Ext3E::zero(); 3];
            eval_program(&prog, &ctx, &mut base_ref, &mut ext_ref);

            // Device walk over the flat blob, in raw limbs.
            let main_raw: Vec<Vec<u64>> = main_vals
                .iter()
                .map(|r| r.iter().map(|x| *x.value()).collect())
                .collect();
            let aux_raw: Vec<Vec<[u64; 3]>> = aux_vals
                .iter()
                .map(|r| r.iter().map(encode_ext).collect())
                .collect();
            let rap_raw: Vec<[u64; 3]> = rap.iter().map(encode_ext).collect();
            let alpha_raw: Vec<[u64; 3]> = alpha.iter().map(encode_ext).collect();
            let mut base_dev = vec![0u64; 1];
            let mut ext_dev = vec![[0u64; 3]; 3];
            eval_device_program(
                &dev,
                &main_raw,
                &aux_raw,
                &rap_raw,
                &alpha_raw,
                encode_ext(&offset),
                &mut base_dev,
                &mut ext_dev,
            );

            // Base root 0.
            assert_eq!(base_dev[0], *base_ref[0].value());
            // Ext roots 1 and 2 (absolute indices; slot 0 unused for ext).
            assert_eq!(ext_dev[1], encode_ext(&ext_ref[1]));
            assert_eq!(ext_dev[2], encode_ext(&ext_ref[2]));
        }
    }
}
