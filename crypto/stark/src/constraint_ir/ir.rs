//! Flat intermediate representation (IR) for captured transition constraints.
//!
//! A [`ConstraintProgram`] is a topologically ordered list of [`Op`] nodes plus
//! a per-constraint root id. It is produced by the builder capture front-end
//! (see [`crate::constraint_ir::builder`]) and consumed by the CPU interpreter
//! (see [`crate::constraint_ir::interp`]).
//!
//! The IR is single-field over Goldilocks, with a [`Dim`] tag distinguishing
//! base (`D1`, one `u64`) from the degree-3 extension (`D3`, three `u64`).

/// Field-arithmetic dimension of a node's value: base Goldilocks (`D1`) or its
/// degree-3 extension (`D3`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Dim {
    /// Base field (one Goldilocks `u64`).
    #[default]
    D1,
    /// Degree-3 extension (`[u64; 3]`).
    D3,
}

/// One IR instruction. Operand fields are `u32` ids into the program's `nodes`
/// arena; a node with id `i` only references nodes with id `< i`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Op {
    /// A base-field literal (already reduced mod the Goldilocks prime).
    Const1(u64),
    /// An extension-field literal `[c0, c1, c2]` (each component reduced).
    Const3([u64; 3]),
    /// A leaf read of a main-trace cell. `main` is always `true` for the
    /// minimal algebraic set captured by the spike; aux reads would set it
    /// `false`. `offset`/`row` select the frame step/row, `col` the column.
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
    /// A periodic column read: `periodic_values[idx]` at the current row (`D1`).
    Periodic { idx: u16 },
    /// A LogUp RAP challenge: `rap_challenges[idx]` (`D3`, uniform per proof).
    RapChallenge { idx: u16 },
    /// A precomputed LogUp alpha power: `logup_alpha_powers[idx]` (`D3`, uniform
    /// per proof).
    AlphaPow { idx: u16 },
    /// The LogUp table offset `L/N` (`D3`, uniform per proof).
    TableOffset,
    /// `nodes[a] + nodes[b]`.
    Add(u32, u32),
    /// `nodes[a] - nodes[b]`.
    Sub(u32, u32),
    /// `nodes[a] * nodes[b]`.
    Mul(u32, u32),
    /// `-nodes[a]`.
    Neg(u32),
    /// Embed a `D1` value into `D3` (`<F as IsSubFieldOf<E>>::embed`).
    Embed(u32),
}

/// A captured program for one transition constraint (or a set of them).
///
/// `nodes` is topologically ordered (id `i` references only `< i`). `dims[i]`
/// is the result dimension of `nodes[i]`. `roots[c]` is the node id of
/// constraint `c`'s value.
#[derive(Clone, Debug)]
pub struct ConstraintProgram {
    /// Topologically ordered instruction list.
    pub nodes: Vec<Op>,
    /// Per-node result dimension, parallel to `nodes`.
    pub dims: Vec<Dim>,
    /// Per-constraint root node ids, indexed by `constraint_idx`.
    pub roots: Vec<u32>,
    /// Number of constraints (a prefix of `roots`) that are base-field (`D1`)
    /// rooted, matching `AIR::num_base_transition_constraints()`. The prover
    /// interpreter writes these into `base_evals`; the rest (LogUp, always
    /// `D3`) go into `ext_evals[num_base..]`.
    pub num_base: usize,
    /// `false` if any constraint in this program was captured via the
    /// default `TransitionConstraintEvaluator::capture` (i.e. it has no real
    /// `Capture` impl — see [`crate::constraint_ir::builder::IrBuilder::mark_unsupported`]).
    /// Callers (the prover/verifier bridge) must not interpret an incomplete
    /// program — fall back to the boxed `TransitionConstraintEvaluator` path
    /// instead. Every production lambda_vm AIR captures cleanly
    /// (`complete: true`); this only trips for the `examples/`/test-only
    /// AIRs that predate the IR migration.
    pub complete: bool,
}

impl ConstraintProgram {
    /// Number of nodes in the program (an effectiveness measure for hash-consing).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the program has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}
