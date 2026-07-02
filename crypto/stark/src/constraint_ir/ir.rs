//! Flat intermediate representation (IR) for captured transition constraints.
//!
//! A [`ConstraintProgram`] is a topologically ordered list of [`Op`] nodes plus
//! a per-constraint root id. It is produced by the builder capture front-end
//! (see [`crate::constraint_ir::builder`]) and consumed by the CPU interpreter
//! (see [`crate::constraint_ir::interp`]).
//!
//! The IR is generic over a field tower `<F, E>` (default: the Goldilocks base
//! field and its degree-3 extension). Each node carries a [`Dim`] tag
//! distinguishing base-field values ([`Dim::Base`]) from extension-field values
//! ([`Dim::Ext`]). Field constants live in side tables (`base_consts` /
//! `ext_consts`) referenced by index, so [`Op`] stays a plain `Copy + Eq + Hash`
//! payload of `u32`s with no bounds on `F`/`E` — this keeps the builder's
//! `(Op, Dim)` common-subexpression map cheap and correct regardless of the
//! field (`FieldElement` values would otherwise poison that key type, since
//! non-canonical representations compare equal under `PartialEq` but hash
//! differently).

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as GoldilocksExtension;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsField;

/// Field-arithmetic dimension of a node's value: base field ([`Dim::Base`]) or
/// its extension ([`Dim::Ext`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Dim {
    /// Base field.
    #[default]
    Base,
    /// Extension field.
    Ext,
}

/// One IR instruction. Operand fields are `u32` ids into the program's `nodes`
/// arena; a node with id `i` only references nodes with id `< i`. Constant ops
/// carry a `u32` index into the program's `base_consts` / `ext_consts` tables
/// rather than the field value itself, so `Op` is `Copy + Eq + Hash` for any
/// field tower.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Op {
    /// A base-field literal: `base_consts[idx]`.
    ConstBase(u32),
    /// An extension-field literal: `ext_consts[idx]`.
    ConstExt(u32),
    /// A leaf read of a trace cell. `main` selects the main trace (base field)
    /// vs the aux trace (extension field); `offset`/`row` select the frame
    /// step/row, `col` the column.
    Var {
        /// `true` for a main-trace column read, `false` for an aux read.
        main: bool,
        /// Frame step index (0-based).
        offset: u8,
        /// Row within the step.
        row: u8,
        /// Column index.
        col: u16,
    },
    /// A periodic column read: `periodic_values[idx]` at the current row
    /// ([`Dim::Base`]).
    Periodic { idx: u16 },
    /// A LogUp RAP challenge: `rap_challenges[idx]` ([`Dim::Ext`], uniform per
    /// proof).
    RapChallenge { idx: u16 },
    /// A precomputed LogUp alpha power: `logup_alpha_powers[idx]` ([`Dim::Ext`],
    /// uniform per proof).
    AlphaPow { idx: u16 },
    /// The LogUp table offset `L/N` ([`Dim::Ext`], uniform per proof).
    TableOffset,
    /// `nodes[a] + nodes[b]`.
    Add(u32, u32),
    /// `nodes[a] - nodes[b]`.
    Sub(u32, u32),
    /// `nodes[a] * nodes[b]`.
    Mul(u32, u32),
    /// `-nodes[a]`.
    Neg(u32),
    /// Embed a base value into the extension (`<F as IsSubFieldOf<E>>::embed`).
    Embed(u32),
}

/// A captured program for one transition constraint (or a set of them).
///
/// `nodes` is topologically ordered (id `i` references only `< i`). `dims[i]`
/// is the result dimension of `nodes[i]`. `roots[c]` is the node id of
/// constraint `c`'s value. `base_consts` / `ext_consts` hold the field literals
/// referenced by `Op::ConstBase` / `Op::ConstExt`.
#[derive(Clone, Debug)]
pub struct ConstraintProgram<F: IsField = GoldilocksField, E: IsField = GoldilocksExtension> {
    /// Topologically ordered instruction list.
    pub nodes: Vec<Op>,
    /// Per-node result dimension, parallel to `nodes`.
    pub dims: Vec<Dim>,
    /// Base-field constant table, indexed by `Op::ConstBase`.
    pub base_consts: Vec<FieldElement<F>>,
    /// Extension-field constant table, indexed by `Op::ConstExt`.
    pub ext_consts: Vec<FieldElement<E>>,
    /// Per-constraint root node ids, indexed by `constraint_idx`.
    pub roots: Vec<u32>,
    /// Number of constraints (a prefix of `roots`) that are base-field
    /// ([`Dim::Base`]) rooted, matching `AIR::num_base_transition_constraints()`.
    /// The prover interpreter writes these into `base_evals`; the rest (LogUp,
    /// always [`Dim::Ext`]) go into `ext_evals[num_base..]`.
    pub num_base: usize,
    /// `false` if any constraint in this program was captured via the
    /// default capture body (i.e. it has no real capture impl — see
    /// [`crate::constraint_ir::builder::IrBuilder::mark_unsupported`]).
    /// Callers must not interpret an incomplete program.
    pub complete: bool,
}

impl<F: IsField, E: IsField> ConstraintProgram<F, E> {
    /// Number of nodes in the program (an effectiveness measure for hash-consing).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the program has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}
