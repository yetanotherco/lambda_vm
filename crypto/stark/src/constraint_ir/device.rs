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
//! ## Slot-based value layout (dim-split)
//!
//! The kernel keeps per-thread value scratch in global memory, so its size and
//! traffic are the dominant cost of the constraint walk. The lowering therefore
//! does three things beyond serializing ops:
//!
//! - **Dim split**: a node's value lives in a *base* (`u64`) or *ext*
//!   (`[u64; 3]`) slot class according to its [`Dim`] tag, and arithmetic on
//!   base nodes is base-field arithmetic. Because embedding is a ring
//!   homomorphism and the device field ops are bit-identical to the CPU's,
//!   this is bit-for-bit equal to the all-ext evaluation it replaces, at a
//!   third of the scratch traffic and ~1/9 of the multiply cost for base
//!   nodes.
//! - **Liveness slot reuse**: slots are assigned by a linear scan that frees
//!   an operand's slot at its last use, so the scratch working set is the
//!   program's max-live-set, not its node count (root nodes are pinned: both
//!   kernels read them after the walk).
//! - **Uniform propagation**: row-invariant leaves (constants, RAP
//!   challenges, LogUp alpha powers, the table offset) never materialize as
//!   nodes or slots; operands reference the tiny uniform tables directly.
//!   They only stay as nodes in the degenerate case where one is itself a
//!   constraint root.
//!
//! Two things live here:
//!
//! - [`DeviceProgram::lower`] — the lowering itself. Field constants become
//!   raw limbs via `FieldElement::to_raw`, byte-identical to how
//!   [`crate::gpu_lde`] hands Goldilocks elements to the device.
//!
//! - [`eval_device_program`] — a CPU forward pass over the *flat* node array
//!   (not the [`Op`] enum), decoding operands exactly as the kernel does and
//!   reproducing the [`interp::run`](super::interp) semantics. It is the model
//!   of the GPU kernel's per-thread walk, so a bit-for-bit match against
//!   [`eval_program`](super::interp::eval_program) pins the on-device layout
//!   and control flow *before* any CUDA runs. The kernel is a transliteration
//!   of this walk with the `FieldElement` arithmetic swapped for
//!   `goldilocks.cuh` / `ext3.cuh`.

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as GoldilocksExtension;
use math::field::goldilocks::GoldilocksField;

use super::ir::{ConstraintProgram, Dim, Op};

type FpE = FieldElement<GoldilocksField>;
type Ext3E = FieldElement<GoldilocksExtension>;

// -------------------------------------------------------------------------
// Wire tags — MUST match the CUDA kernel's `switch (op)` and operand decode.
// -------------------------------------------------------------------------

/// `a` = index into `base_consts` (only when a uniform leaf is itself a root).
pub const OP_CONST_BASE: u32 = 0;
/// `a` = index into `ext_consts` (only when a uniform leaf is itself a root).
pub const OP_CONST_EXT: u32 = 1;
/// Trace-cell read; `a`/`b` pack the [`Op::Var`] fields (see [`pack_var`]).
pub const OP_VAR: u32 = 2;
/// `a` = index into the per-proof `rap_challenges` uniform buffer (root-only).
pub const OP_RAP_CHALLENGE: u32 = 3;
/// `a` = index into the per-proof `logup_alpha_powers` uniform buffer
/// (root-only).
pub const OP_ALPHA_POW: u32 = 4;
/// The per-proof LogUp table offset uniform; no operands (root-only).
pub const OP_TABLE_OFFSET: u32 = 5;
/// `a`, `b` = encoded operands (see `OPK_*`).
pub const OP_ADD: u32 = 6;
/// `a`, `b` = encoded operands.
pub const OP_SUB: u32 = 7;
/// `a`, `b` = encoded operands.
pub const OP_MUL: u32 = 8;
/// `a` = encoded operand.
pub const OP_NEG: u32 = 9;
/// `a` = encoded operand (base → extension embed).
pub const OP_EMBED: u32 = 10;

// -- operand encoding: `kind << OPK_SHIFT | payload` ----------------------

/// Bit position of the 3-bit operand kind.
pub const OPK_SHIFT: u32 = 29;
/// Mask of the 29-bit operand payload (slot or table index).
pub const OPK_PAYLOAD_MASK: u32 = (1 << OPK_SHIFT) - 1;
/// Payload = base (`u64`) scratch-slot index.
pub const OPK_BASE_SLOT: u32 = 0;
/// Payload = ext (`[u64; 3]`) scratch-slot index.
pub const OPK_EXT_SLOT: u32 = 1;
/// Payload = `base_consts` index.
pub const OPK_BASE_CONST: u32 = 2;
/// Payload = `ext_consts` index.
pub const OPK_EXT_CONST: u32 = 3;
/// Payload = per-proof `rap_challenges` index.
pub const OPK_RAP: u32 = 4;
/// Payload = per-proof `logup_alpha_powers` index.
pub const OPK_ALPHA: u32 = 5;
/// The per-proof table offset (payload unused).
pub const OPK_OFFSET: u32 = 6;

/// In a node's `res` word and in `roots` entries: bit 31 set = ext slot,
/// clear = base slot; low bits = the slot index.
pub const RES_EXT_BIT: u32 = 1 << 31;

/// One flattened IR instruction: 16 bytes, `#[repr(C)]` for a 1:1 device
/// upload. `op` is an `OP_*` tag; `a`/`b` are encoded operands (`OPK_*` kinds
/// for arithmetic, packed [`Op::Var`] fields for [`OP_VAR`], raw table indices
/// for root-pinned uniform leaves); `res` is the result slot with [`RES_EXT_BIT`]
/// selecting the slot class.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct DeviceNode {
    pub op: u32,
    pub a: u32,
    pub b: u32,
    pub res: u32,
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

/// A [`ConstraintProgram`] lowered to flat, device-uploadable arrays with
/// dim-split, liveness-reused value slots. Constants are canonical raw limbs
/// (`u64` base / `[u64; 3]` extension), matching the `#[repr(transparent)]`
/// layout the GPU trace buffers already use.
#[derive(Clone, Debug)]
pub struct DeviceProgram {
    /// Topologically ordered instruction list (operands reference slots
    /// already written or uniform tables). Uniform leaves and dead nodes are
    /// not materialized.
    pub nodes: Vec<DeviceNode>,
    /// Base-field constant table, indexed by [`OPK_BASE_CONST`] operands (and
    /// [`OP_CONST_BASE`] root nodes).
    pub base_consts: Vec<u64>,
    /// Extension-field constant table, indexed by [`OPK_EXT_CONST`] operands
    /// (and [`OP_CONST_EXT`] root nodes).
    pub ext_consts: Vec<[u64; 3]>,
    /// Per-constraint root slots (`slot | RES_EXT_BIT`), indexed by
    /// `constraint_idx`. Root slots are pinned — never reused — so both
    /// kernels can read them after the walk.
    pub roots: Vec<u32>,
    /// Number of leading base-rooted constraints written to `base_evals`; the
    /// rest go to `ext_evals`.
    pub num_base: u32,
    /// Size of the base (`u64`) slot class, per thread.
    pub num_base_slots: u32,
    /// Size of the ext (`[u64; 3]`) slot class, per thread.
    pub num_ext_slots: u32,
}

/// Whether an op is a row-invariant leaf (uniform per proof).
fn is_uniform_leaf(op: &Op) -> bool {
    matches!(
        op,
        Op::ConstBase(_)
            | Op::ConstExt(_)
            | Op::RapChallenge { .. }
            | Op::AlphaPow { .. }
            | Op::TableOffset
    )
}

/// The (up to two) operand node ids of an op.
fn operands(op: &Op) -> [Option<u32>; 2] {
    match *op {
        Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) => [Some(a), Some(b)],
        Op::Neg(a) | Op::Embed(a) => [Some(a), None],
        _ => [None, None],
    }
}

impl DeviceProgram {
    /// Lower a concrete-Goldilocks [`ConstraintProgram`] to its flat device
    /// form: dim-split slot assignment with liveness reuse, uniform-leaf
    /// propagation into operands, and root pinning. Pure serialization plus
    /// the slot scan — no field arithmetic, no device access.
    pub fn lower(prog: &ConstraintProgram<GoldilocksField, GoldilocksExtension>) -> Self {
        let n = prog.nodes.len();
        assert!(
            n <= OPK_PAYLOAD_MASK as usize,
            "program of {n} nodes exceeds the 29-bit slot space"
        );

        // Liveness: last consumer (by node id) of every node, plus root pins.
        let mut used = vec![false; n];
        let mut last_use = vec![0u32; n];
        for (i, op) in prog.nodes.iter().enumerate() {
            for operand in operands(op).into_iter().flatten() {
                used[operand as usize] = true;
                last_use[operand as usize] = i as u32;
            }
        }
        let mut is_root = vec![false; n];
        for &r in &prog.roots {
            is_root[r as usize] = true;
            used[r as usize] = true;
        }

        // A node materializes (gets a slot) unless it is a propagated uniform
        // leaf or dead. Uniform leaves stay only when they are roots (the
        // post-walk emit reads slots).
        let emitted: Vec<bool> = (0..n)
            .map(|i| used[i] && (!is_uniform_leaf(&prog.nodes[i]) || is_root[i]))
            .collect();

        let enc_uniform = |op: &Op| -> u32 {
            match *op {
                Op::ConstBase(idx) => {
                    debug_assert!(idx <= OPK_PAYLOAD_MASK);
                    (OPK_BASE_CONST << OPK_SHIFT) | idx
                }
                Op::ConstExt(idx) => {
                    debug_assert!(idx <= OPK_PAYLOAD_MASK);
                    (OPK_EXT_CONST << OPK_SHIFT) | idx
                }
                Op::RapChallenge { idx } => (OPK_RAP << OPK_SHIFT) | idx as u32,
                Op::AlphaPow { idx } => (OPK_ALPHA << OPK_SHIFT) | idx as u32,
                Op::TableOffset => OPK_OFFSET << OPK_SHIFT,
                _ => unreachable!("not a uniform leaf"),
            }
        };

        // Linear-scan slot assignment with per-class free lists.
        const UNASSIGNED: u32 = u32::MAX;
        let mut slot_of = vec![UNASSIGNED; n];
        let mut free_base: Vec<u32> = Vec::new();
        let mut free_ext: Vec<u32> = Vec::new();
        let mut num_base_slots = 0u32;
        let mut num_ext_slots = 0u32;
        let mut nodes = Vec::with_capacity(n);

        for i in 0..n {
            if !emitted[i] {
                continue;
            }
            let op = &prog.nodes[i];
            let dim = prog.dims[i];

            // Encode operands while their slots are still assigned.
            let enc_operand = |j: u32| -> u32 {
                let j = j as usize;
                if !emitted[j] {
                    return enc_uniform(&prog.nodes[j]);
                }
                let slot = slot_of[j];
                debug_assert_ne!(slot, UNASSIGNED, "operand before definition");
                match prog.dims[j] {
                    Dim::Base => (OPK_BASE_SLOT << OPK_SHIFT) | slot,
                    Dim::Ext => (OPK_EXT_SLOT << OPK_SHIFT) | slot,
                }
            };

            let (tag, a, b) = match *op {
                Op::ConstBase(idx) => (OP_CONST_BASE, idx, 0),
                Op::ConstExt(idx) => (OP_CONST_EXT, idx, 0),
                Op::Var {
                    main,
                    offset,
                    row,
                    col,
                } => {
                    let (a, b) = pack_var(main, offset, row, col);
                    (OP_VAR, a, b)
                }
                Op::RapChallenge { idx } => (OP_RAP_CHALLENGE, idx as u32, 0),
                Op::AlphaPow { idx } => (OP_ALPHA_POW, idx as u32, 0),
                Op::TableOffset => (OP_TABLE_OFFSET, 0, 0),
                Op::Add(x, y) => (OP_ADD, enc_operand(x), enc_operand(y)),
                Op::Sub(x, y) => (OP_SUB, enc_operand(x), enc_operand(y)),
                Op::Mul(x, y) => (OP_MUL, enc_operand(x), enc_operand(y)),
                Op::Neg(x) => (OP_NEG, enc_operand(x), 0),
                Op::Embed(x) => (OP_EMBED, enc_operand(x), 0),
            };

            // Free operand slots at their last use (roots stay pinned). The
            // `slot_of` reset guards the a == b double-free.
            for operand in operands(op).into_iter().flatten() {
                let j = operand as usize;
                if emitted[j] && !is_root[j] && last_use[j] == i as u32 && slot_of[j] != UNASSIGNED
                {
                    match prog.dims[j] {
                        Dim::Base => free_base.push(slot_of[j]),
                        Dim::Ext => free_ext.push(slot_of[j]),
                    }
                    slot_of[j] = UNASSIGNED;
                }
            }

            // Allocate the result slot (a freed operand slot may be reused —
            // the kernel reads operands before writing the result).
            let slot = match dim {
                Dim::Base => free_base.pop().unwrap_or_else(|| {
                    num_base_slots += 1;
                    num_base_slots - 1
                }),
                Dim::Ext => free_ext.pop().unwrap_or_else(|| {
                    num_ext_slots += 1;
                    num_ext_slots - 1
                }),
            };
            slot_of[i] = slot;

            let res = match dim {
                Dim::Base => slot,
                Dim::Ext => slot | RES_EXT_BIT,
            };
            nodes.push(DeviceNode { op: tag, a, b, res });
        }

        let roots = prog
            .roots
            .iter()
            .map(|&r| {
                let slot = slot_of[r as usize];
                debug_assert_ne!(slot, UNASSIGNED, "root without a slot");
                match prog.dims[r as usize] {
                    Dim::Base => slot,
                    Dim::Ext => slot | RES_EXT_BIT,
                }
            })
            .collect();

        let base_consts = prog.base_consts.iter().map(|c| *c.value()).collect();
        let ext_consts = prog.ext_consts.iter().map(encode_ext).collect();

        DeviceProgram {
            nodes,
            base_consts,
            ext_consts,
            roots,
            num_base: prog.num_base as u32,
            num_base_slots,
            num_ext_slots,
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

/// Full prover-shaped forward pass over the *flat* device blob, in raw limbs —
/// the CPU model of the GPU kernel: dim-split slot files, encoded-operand
/// loads, mixed ops evaluated as full ext ops on embedded operands (the GPU's
/// mixed-op shortcuts are bit-identical to that by construction). Mirrors
/// [`eval_program`](super::interp::eval_program): base-rooted constraints
/// (`c < num_base`) land in `base_evals`, the rest in `ext_evals`.
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
    let mut base_slots = vec![FpE::zero(); dev.num_base_slots as usize];
    let mut ext_slots = vec![Ext3E::zero(); dev.num_ext_slots as usize];

    let load_base = |enc: u32, base_slots: &[FpE]| -> FpE {
        let payload = (enc & OPK_PAYLOAD_MASK) as usize;
        match enc >> OPK_SHIFT {
            OPK_BASE_SLOT => base_slots[payload],
            OPK_BASE_CONST => FpE::from_raw(dev.base_consts[payload]),
            other => panic!("base operand with non-base kind {other}"),
        }
    };
    let load_ext = |enc: u32, base_slots: &[FpE], ext_slots: &[Ext3E]| -> Ext3E {
        let payload = (enc & OPK_PAYLOAD_MASK) as usize;
        match enc >> OPK_SHIFT {
            OPK_BASE_SLOT => base_slots[payload].to_extension::<GoldilocksExtension>(),
            OPK_EXT_SLOT => ext_slots[payload],
            OPK_BASE_CONST => {
                FpE::from_raw(dev.base_consts[payload]).to_extension::<GoldilocksExtension>()
            }
            OPK_EXT_CONST => decode_ext(dev.ext_consts[payload]),
            OPK_RAP => decode_ext(rap_challenges[payload]),
            OPK_ALPHA => decode_ext(alpha_powers[payload]),
            OPK_OFFSET => decode_ext(table_offset),
            other => panic!("unknown operand kind {other}"),
        }
    };

    for node in &dev.nodes {
        let res_slot = (node.res & !RES_EXT_BIT) as usize;
        let res_ext = node.res & RES_EXT_BIT != 0;
        match node.op {
            OP_CONST_BASE => base_slots[res_slot] = FpE::from_raw(dev.base_consts[node.a as usize]),
            OP_CONST_EXT => ext_slots[res_slot] = decode_ext(dev.ext_consts[node.a as usize]),
            OP_VAR => {
                let (is_main, offset, _row, col) = unpack_var(node.a, node.b);
                if is_main {
                    base_slots[res_slot] = FpE::from_raw(main[offset as usize][col as usize]);
                } else {
                    ext_slots[res_slot] = decode_ext(aux[offset as usize][col as usize]);
                }
            }
            OP_RAP_CHALLENGE => ext_slots[res_slot] = decode_ext(rap_challenges[node.a as usize]),
            OP_ALPHA_POW => ext_slots[res_slot] = decode_ext(alpha_powers[node.a as usize]),
            OP_TABLE_OFFSET => ext_slots[res_slot] = decode_ext(table_offset),
            OP_ADD => {
                if res_ext {
                    ext_slots[res_slot] = load_ext(node.a, &base_slots, &ext_slots)
                        + load_ext(node.b, &base_slots, &ext_slots);
                } else {
                    base_slots[res_slot] =
                        load_base(node.a, &base_slots) + load_base(node.b, &base_slots);
                }
            }
            OP_SUB => {
                if res_ext {
                    ext_slots[res_slot] = load_ext(node.a, &base_slots, &ext_slots)
                        - load_ext(node.b, &base_slots, &ext_slots);
                } else {
                    base_slots[res_slot] =
                        load_base(node.a, &base_slots) - load_base(node.b, &base_slots);
                }
            }
            OP_MUL => {
                if res_ext {
                    ext_slots[res_slot] = load_ext(node.a, &base_slots, &ext_slots)
                        * load_ext(node.b, &base_slots, &ext_slots);
                } else {
                    base_slots[res_slot] =
                        load_base(node.a, &base_slots) * load_base(node.b, &base_slots);
                }
            }
            OP_NEG => {
                if res_ext {
                    ext_slots[res_slot] = -load_ext(node.a, &base_slots, &ext_slots);
                } else {
                    base_slots[res_slot] = -load_base(node.a, &base_slots);
                }
            }
            OP_EMBED => {
                ext_slots[res_slot] = load_ext(node.a, &base_slots, &ext_slots);
            }
            other => panic!("unknown device op tag {other}"),
        }
    }

    for (c, &root) in dev.roots.iter().enumerate() {
        let slot = (root & !RES_EXT_BIT) as usize;
        let is_ext = root & RES_EXT_BIT != 0;
        if (c as u32) < dev.num_base {
            assert!(!is_ext, "base-rooted constraint with an ext root slot");
            base_evals[c] = *base_slots[slot].value();
        } else if is_ext {
            ext_evals[c] = encode_ext(&ext_slots[slot]);
        } else {
            ext_evals[c] = encode_ext(&base_slots[slot].to_extension::<GoldilocksExtension>());
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

    /// Lowering invariants of the slot allocator: uniform leaves are
    /// propagated (no nodes), the slot classes are bounded by the max-live-set
    /// (strictly fewer slots than nodes for a program with dead-after-use
    /// intermediates), and slot indices stay in range.
    #[test]
    fn lowering_reuses_slots_and_propagates_uniforms() {
        let prog = all_ops_program();
        let dev = DeviceProgram::lower(&prog);

        // No uniform leaf is materialized (none is a root here).
        for n in &dev.nodes {
            assert!(
                !matches!(
                    n.op,
                    OP_CONST_BASE
                        | OP_CONST_EXT
                        | OP_RAP_CHALLENGE
                        | OP_ALPHA_POW
                        | OP_TABLE_OFFSET
                ),
                "uniform leaf materialized as a node"
            );
        }
        // Slot classes are within bounds and smaller than the node count.
        let total_slots = (dev.num_base_slots + dev.num_ext_slots) as usize;
        assert!(total_slots < prog.nodes.len());
        for n in &dev.nodes {
            let slot = n.res & !RES_EXT_BIT;
            if n.res & RES_EXT_BIT != 0 {
                assert!(slot < dev.num_ext_slots);
            } else {
                assert!(slot < dev.num_base_slots);
            }
        }
        for &r in &dev.roots {
            let slot = r & !RES_EXT_BIT;
            if r & RES_EXT_BIT != 0 {
                assert!(slot < dev.num_ext_slots);
            } else {
                assert!(slot < dev.num_base_slots);
            }
        }
    }

    /// A uniform leaf that is itself a root must still materialize (the
    /// post-walk emit reads a slot).
    #[test]
    fn uniform_root_is_materialized() {
        let mut b = IrBuilder::<Gl, Ext>::new();
        let c = b.const_base(7);
        b.emit(0, c);
        let prog = b.finish(1);
        let dev = DeviceProgram::lower(&prog);

        assert!(dev.nodes.iter().any(|n| n.op == OP_CONST_BASE));
        let mut base_evals = vec![0u64; 1];
        let mut ext_evals: Vec<[u64; 3]> = vec![];
        eval_device_program(
            &dev,
            &[],
            &[],
            &[],
            &[],
            [0, 0, 0],
            &mut base_evals,
            &mut ext_evals,
        );
        assert_eq!(base_evals[0], 7);
    }

    /// Randomized differential: a synthetic DAG with heavy slot churn (long
    /// chains whose intermediates die immediately) evaluates identically
    /// through the interpreter and the slot-reusing device walk.
    #[test]
    fn slot_reuse_differential_random_chains() {
        let mut b = IrBuilder::<Gl, Ext>::new();
        let m0 = b.main(0, 0);
        let m1 = b.main(0, 1);
        let ch = b.challenge(0);

        // Base chain: alternating add/mul over rotating leaves.
        let mut acc = m0;
        for k in 0..50u64 {
            let c = b.const_base(k + 2);
            let t = if k % 2 == 0 {
                b.add(acc, c)
            } else {
                b.mul(acc, m1)
            };
            acc = t;
        }
        b.emit(0, acc);

        // Ext chain crossing dims each step.
        let mut eacc = b.mul(m0, ch);
        for k in 0..50u64 {
            let c = b.const_base(k + 100);
            let t = b.mul(eacc, c); // ext × base
            let u = b.sub(t, ch);
            eacc = u;
        }
        b.emit(1, eacc);
        let prog = b.finish(1);
        let dev = DeviceProgram::lower(&prog);

        // Slot reuse must keep the live-set small despite 100+ nodes.
        assert!(dev.num_base_slots <= 8, "base slots {}", dev.num_base_slots);
        assert!(dev.num_ext_slots <= 8, "ext slots {}", dev.num_ext_slots);

        let mut rng = SplitMix64(0xDEAD_BEEF_0BAD_F00D);
        for _ in 0..500 {
            let main_vals: Vec<Vec<FpE>> = (0..2)
                .map(|_| vec![fp(rng.next_u64()), fp(rng.next_u64())])
                .collect();
            let aux_vals: Vec<Vec<Ext3E>> = (0..2).map(|_| vec![rng.ext()]).collect();
            let rap = vec![rng.ext()];
            let alpha = vec![rng.ext()];
            let offset = rng.ext();

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
            let mut ext_ref = vec![Ext3E::zero(); 2];
            eval_program(&prog, &ctx, &mut base_ref, &mut ext_ref);

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
            let mut ext_dev = vec![[0u64; 3]; 2];
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

            assert_eq!(base_dev[0], *base_ref[0].value());
            assert_eq!(ext_dev[1], encode_ext(&ext_ref[1]));
        }
    }
}
