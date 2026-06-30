//! Explicit-builder capture front-end (Plan B).
//!
//! Where the symbolic-field front-end (Plan A) records IR by running a
//! constraint's generic `evaluate` over recording field types, this front-end
//! builds the same [`ConstraintProgram`] through an explicit [`IrBuilder`]:
//! each constraint implements [`Capture`](super::Capture) and translates its
//! `evaluate` body into builder calls (`main`, `add`, `sub`, `mul`, ...).
//!
//! No fake field, no thread-local arena. The builder hash-conses every node on
//! `(Op, Dim)` and only emits leaves for columns the constraint actually reads,
//! so captured programs are minimal.

use std::collections::HashMap;

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;

use super::ir::{ConstraintProgram, Dim, Op};

/// A handle to a node in an [`IrBuilder`]: its arena id and result dimension.
///
/// `Copy` so constraint bodies read like ordinary field arithmetic.
#[derive(Clone, Copy, Debug)]
pub struct Expr {
    id: u32,
    dim: Dim,
}

impl Expr {
    /// The node's result dimension.
    pub fn dim(self) -> Dim {
        self.dim
    }
}

/// Builds a [`ConstraintProgram`] from explicit node-construction calls.
///
/// Nodes are appended in topological order (id `i` references only `< i`) and
/// hash-consed on `(Op, Dim)`, so structurally identical subexpressions share a
/// single id. Base-field constants are additionally deduplicated by value via
/// `const_cache`. Node id `0` is reserved for `Op::Const1(0)`, matching the
/// interpreter's convention and Plan A's arena.
pub struct IrBuilder {
    nodes: Vec<Op>,
    dims: Vec<Dim>,
    cse: HashMap<(Op, Dim), u32>,
    const_cache: HashMap<u64, u32>,
    roots: Vec<u32>,
}

impl Default for IrBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl IrBuilder {
    /// Create a builder with the reserved `Op::Const1(0)` node at id 0.
    pub fn new() -> Self {
        let mut b = IrBuilder {
            nodes: Vec::new(),
            dims: Vec::new(),
            cse: HashMap::new(),
            const_cache: HashMap::new(),
            roots: Vec::new(),
        };
        // Reserve id 0 = Const1(0). `const_base(0)` will hash-cons to this.
        let zero = b.push(Op::Const1(0), Dim::D1);
        debug_assert_eq!(zero.id, 0);
        b.const_cache.insert(0, 0);
        b
    }

    /// Append (or reuse) a node with the given op and result dimension.
    fn push(&mut self, op: Op, dim: Dim) -> Expr {
        if let Some(&id) = self.cse.get(&(op, dim)) {
            return Expr { id, dim };
        }
        let id = self.nodes.len() as u32;
        self.nodes.push(op);
        self.dims.push(dim);
        self.cse.insert((op, dim), id);
        Expr { id, dim }
    }

    // ---------------------------------------------------------------------
    // Leaves
    // ---------------------------------------------------------------------

    /// A main-trace column read at the given frame `offset`, row 0.
    pub fn main(&mut self, offset: u8, col: usize) -> Expr {
        self.push(
            Op::Var {
                main: true,
                offset,
                row: 0,
                col: col as u16,
            },
            Dim::D1,
        )
    }

    /// An aux-trace column read at the given frame `offset`, row 0 (`D3`).
    pub fn aux(&mut self, offset: u8, col: usize) -> Expr {
        self.push(
            Op::Var {
                main: false,
                offset,
                row: 0,
                col: col as u16,
            },
            Dim::D3,
        )
    }

    /// A periodic column read at the current row (`D1`).
    pub fn periodic(&mut self, idx: usize) -> Expr {
        self.push(Op::Periodic { idx: idx as u16 }, Dim::D1)
    }

    /// A LogUp RAP challenge, uniform per proof (`D3`).
    pub fn challenge(&mut self, idx: usize) -> Expr {
        self.push(Op::RapChallenge { idx: idx as u16 }, Dim::D3)
    }

    /// A precomputed LogUp alpha power, uniform per proof (`D3`).
    pub fn alpha_power(&mut self, idx: usize) -> Expr {
        self.push(Op::AlphaPow { idx: idx as u16 }, Dim::D3)
    }

    /// The LogUp table offset `L/N`, uniform per proof (`D3`).
    pub fn table_offset(&mut self) -> Expr {
        self.push(Op::TableOffset, Dim::D3)
    }

    // ---------------------------------------------------------------------
    // Constants
    // ---------------------------------------------------------------------

    /// A base-field constant from a `u64`, reduced and deduplicated by value.
    pub fn const_base(&mut self, v: u64) -> Expr {
        let canon = *FieldElement::<GoldilocksField>::from(v).value();
        if let Some(&id) = self.const_cache.get(&canon) {
            return Expr { id, dim: Dim::D1 };
        }
        let e = self.push(Op::Const1(canon), Dim::D1);
        self.const_cache.insert(canon, e.id);
        e
    }

    /// A base-field constant from an `i64`; negatives map to `p - |v|`.
    pub fn const_signed(&mut self, v: i64) -> Expr {
        let canon = *FieldElement::<GoldilocksField>::from(v).value();
        if let Some(&id) = self.const_cache.get(&canon) {
            return Expr { id, dim: Dim::D1 };
        }
        let e = self.push(Op::Const1(canon), Dim::D1);
        self.const_cache.insert(canon, e.id);
        e
    }

    /// An extension-field constant `[c0, c1, c2]`, each component reduced.
    /// Dedup is via the general `(Op, Dim)` hash-cons (`push`); no separate
    /// cache is needed since `Const3` is `Eq + Hash`.
    pub fn const_ext(&mut self, v: [u64; 3]) -> Expr {
        let c0 = *FieldElement::<GoldilocksField>::from(v[0]).value();
        let c1 = *FieldElement::<GoldilocksField>::from(v[1]).value();
        let c2 = *FieldElement::<GoldilocksField>::from(v[2]).value();
        self.push(Op::Const3([c0, c1, c2]), Dim::D3)
    }

    /// The base-field constant `1`.
    pub fn one(&mut self) -> Expr {
        self.const_base(1)
    }

    // ---------------------------------------------------------------------
    // Arithmetic
    // ---------------------------------------------------------------------

    /// `a + b`. Result is `D1` only if both operands are `D1`.
    pub fn add(&mut self, a: Expr, b: Expr) -> Expr {
        let dim = Self::join(a.dim, b.dim);
        self.push(Op::Add(a.id, b.id), dim)
    }

    /// `a - b`. Result is `D1` only if both operands are `D1`.
    pub fn sub(&mut self, a: Expr, b: Expr) -> Expr {
        let dim = Self::join(a.dim, b.dim);
        self.push(Op::Sub(a.id, b.id), dim)
    }

    /// `a * b`. Result is `D1` only if both operands are `D1`.
    pub fn mul(&mut self, a: Expr, b: Expr) -> Expr {
        let dim = Self::join(a.dim, b.dim);
        self.push(Op::Mul(a.id, b.id), dim)
    }

    /// `-a`. Preserves the operand's dimension.
    pub fn neg(&mut self, a: Expr) -> Expr {
        self.push(Op::Neg(a.id), a.dim)
    }

    /// Typing join: `(D1, D1) -> D1`; any `D3` operand -> `D3`.
    fn join(a: Dim, b: Dim) -> Dim {
        match (a, b) {
            (Dim::D1, Dim::D1) => Dim::D1,
            _ => Dim::D3,
        }
    }

    // ---------------------------------------------------------------------
    // Emit / finish
    // ---------------------------------------------------------------------

    /// Record `e` as the root for constraint `constraint_idx`.
    ///
    /// `roots` is indexed by `constraint_idx` (grown/filled with sentinel `0`
    /// as needed), so constraints can be captured in any order and a full
    /// per-table program (one `emit` per `TransitionConstraintEvaluator` in
    /// `transition_constraints()`) ends up with `roots[c]` = constraint `c`'s
    /// value, matching `AIR::num_transition_constraints()` indexing.
    pub fn emit(&mut self, constraint_idx: usize, e: Expr) {
        if self.roots.len() <= constraint_idx {
            self.roots.resize(constraint_idx + 1, 0);
        }
        self.roots[constraint_idx] = e.id;
    }

    /// Consume the builder and produce the captured program.
    ///
    /// `num_base` is the number of leading (by `constraint_idx`) constraints
    /// that are base-field (`D1`) rooted, matching
    /// `AIR::num_base_transition_constraints()`.
    pub fn finish(self, num_base: usize) -> ConstraintProgram {
        ConstraintProgram {
            nodes: self.nodes,
            dims: self.dims,
            roots: self.roots,
            num_base,
        }
    }
}
