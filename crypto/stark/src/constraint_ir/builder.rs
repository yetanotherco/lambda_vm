//! Explicit-builder capture front-end.
//!
//! Every transition constraint is captured into a flat [`ConstraintProgram`]
//! through an explicit [`IrBuilder`]: each constraint translates its algebra
//! into builder calls (`main`, `add`, `sub`, `mul`, ...). No fake field, no
//! thread-local arena.
//!
//! The builder hash-conses every node on `(Op, Dim)` and only emits leaves for
//! columns the constraint actually reads, so captured programs are minimal.
//! Field constants live in the [`ConstraintProgram`]'s `base_consts` /
//! `ext_consts` side tables; the builder deduplicates them by value via a linear
//! scan (`FieldElement`'s canonicalizing `PartialEq`) — the tables are tiny and
//! capture runs once at setup, so no hash map is needed there (and none would be
//! sound: `FieldElement`'s derived `Hash` and manual `Eq` disagree on
//! non-canonical representations).

use std::collections::HashMap;

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as GoldilocksExtension;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsField;

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
/// single id. Field constants are deduplicated by value in the `base_consts` /
/// `ext_consts` tables (linear scan). Node id `0` is reserved for the base-field
/// zero (`Op::ConstBase(0)`, `base_consts[0] = 0`), matching the interpreter's
/// convention.
pub struct IrBuilder<F: IsField = GoldilocksField, E: IsField = GoldilocksExtension> {
    nodes: Vec<Op>,
    dims: Vec<Dim>,
    cse: HashMap<(Op, Dim), u32>,
    base_consts: Vec<FieldElement<F>>,
    ext_consts: Vec<FieldElement<E>>,
    roots: Vec<u32>,
    /// Set by [`Self::mark_unsupported`] when a constraint couldn't be
    /// captured. Propagated to [`ConstraintProgram::complete`] so callers know
    /// not to interpret an incomplete program.
    complete: bool,
}

impl<F: IsField, E: IsField> Default for IrBuilder<F, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: IsField, E: IsField> IrBuilder<F, E> {
    /// Create a builder with the reserved base-field zero node at id 0.
    pub fn new() -> Self {
        let mut b = IrBuilder {
            nodes: Vec::new(),
            dims: Vec::new(),
            cse: HashMap::new(),
            base_consts: Vec::new(),
            ext_consts: Vec::new(),
            roots: Vec::new(),
            complete: true,
        };
        // Reserve id 0 = ConstBase(0) = base-field zero. `const_base(0)` will
        // dedup to this.
        let zero = b.const_base(0);
        debug_assert_eq!(zero.id, 0);
        b
    }

    /// Record that the constraint currently being captured has no capture
    /// implementation. Does not panic and does not emit a root for it — the
    /// resulting program is marked incomplete (see
    /// [`ConstraintProgram::complete`]) so callers know not to interpret it.
    pub fn mark_unsupported(&mut self) {
        self.complete = false;
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
            Dim::Base,
        )
    }

    /// An aux-trace column read at the given frame `offset`, row 0
    /// ([`Dim::Ext`]).
    pub fn aux(&mut self, offset: u8, col: usize) -> Expr {
        self.push(
            Op::Var {
                main: false,
                offset,
                row: 0,
                col: col as u16,
            },
            Dim::Ext,
        )
    }

    /// A periodic column read at the current row ([`Dim::Base`]).
    pub fn periodic(&mut self, idx: usize) -> Expr {
        self.push(Op::Periodic { idx: idx as u16 }, Dim::Base)
    }

    /// A LogUp RAP challenge, uniform per proof ([`Dim::Ext`]).
    pub fn challenge(&mut self, idx: usize) -> Expr {
        self.push(Op::RapChallenge { idx: idx as u16 }, Dim::Ext)
    }

    /// A precomputed LogUp alpha power, uniform per proof ([`Dim::Ext`]).
    pub fn alpha_power(&mut self, idx: usize) -> Expr {
        self.push(Op::AlphaPow { idx: idx as u16 }, Dim::Ext)
    }

    /// The LogUp table offset `L/N`, uniform per proof ([`Dim::Ext`]).
    pub fn table_offset(&mut self) -> Expr {
        self.push(Op::TableOffset, Dim::Ext)
    }

    // ---------------------------------------------------------------------
    // Constants
    // ---------------------------------------------------------------------

    /// Intern a base-field constant into `base_consts`, deduplicating by value.
    fn intern_base(&mut self, fe: FieldElement<F>) -> Expr {
        let idx = match self.base_consts.iter().position(|c| c == &fe) {
            Some(idx) => idx,
            None => {
                let idx = self.base_consts.len();
                self.base_consts.push(fe);
                idx
            }
        };
        self.push(Op::ConstBase(idx as u32), Dim::Base)
    }

    /// Intern an extension-field constant into `ext_consts`, deduplicating by
    /// value.
    fn intern_ext(&mut self, fe: FieldElement<E>) -> Expr {
        let idx = match self.ext_consts.iter().position(|c| c == &fe) {
            Some(idx) => idx,
            None => {
                let idx = self.ext_consts.len();
                self.ext_consts.push(fe);
                idx
            }
        };
        self.push(Op::ConstExt(idx as u32), Dim::Ext)
    }

    /// A base-field constant from a `u64`, reduced and deduplicated by value.
    pub fn const_base(&mut self, v: u64) -> Expr {
        self.intern_base(FieldElement::<F>::from(v))
    }

    /// A base-field constant from an `i64`; negatives map to `p - |v|`.
    pub fn const_signed(&mut self, v: i64) -> Expr {
        self.intern_base(FieldElement::<F>::from(v))
    }

    /// An extension-field constant, deduplicated by value.
    pub fn const_ext(&mut self, v: FieldElement<E>) -> Expr {
        self.intern_ext(v)
    }

    /// The base-field constant `1`.
    pub fn one(&mut self) -> Expr {
        self.const_base(1)
    }

    // ---------------------------------------------------------------------
    // Arithmetic
    // ---------------------------------------------------------------------

    /// `a + b`. Result is [`Dim::Base`] only if both operands are base.
    pub fn add(&mut self, a: Expr, b: Expr) -> Expr {
        let dim = Self::join(a.dim, b.dim);
        self.push(Op::Add(a.id, b.id), dim)
    }

    /// `a - b`. Result is [`Dim::Base`] only if both operands are base.
    pub fn sub(&mut self, a: Expr, b: Expr) -> Expr {
        let dim = Self::join(a.dim, b.dim);
        self.push(Op::Sub(a.id, b.id), dim)
    }

    /// `a * b`. Result is [`Dim::Base`] only if both operands are base.
    pub fn mul(&mut self, a: Expr, b: Expr) -> Expr {
        let dim = Self::join(a.dim, b.dim);
        self.push(Op::Mul(a.id, b.id), dim)
    }

    /// `-a`. Preserves the operand's dimension.
    pub fn neg(&mut self, a: Expr) -> Expr {
        self.push(Op::Neg(a.id), a.dim)
    }

    /// Explicitly embed a base value into the extension ([`Dim::Ext`]).
    pub fn embed(&mut self, a: Expr) -> Expr {
        self.push(Op::Embed(a.id), Dim::Ext)
    }

    /// Typing join: `(Base, Base) -> Base`; any `Ext` operand -> `Ext`.
    fn join(a: Dim, b: Dim) -> Dim {
        match (a, b) {
            (Dim::Base, Dim::Base) => Dim::Base,
            _ => Dim::Ext,
        }
    }

    // ---------------------------------------------------------------------
    // Emit / finish
    // ---------------------------------------------------------------------

    /// Record `e` as the root for constraint `constraint_idx`.
    ///
    /// `roots` is indexed by `constraint_idx` (grown/filled with sentinel `0`
    /// as needed), so constraints can be captured in any order and a full
    /// per-table program ends up with `roots[c]` = constraint `c`'s value.
    pub fn emit(&mut self, constraint_idx: usize, e: Expr) {
        if self.roots.len() <= constraint_idx {
            self.roots.resize(constraint_idx + 1, 0);
        }
        self.roots[constraint_idx] = e.id;
    }

    /// Consume the builder and produce the captured program.
    ///
    /// `num_base` is the number of leading (by `constraint_idx`) constraints
    /// that are base-field ([`Dim::Base`]) rooted, matching
    /// `AIR::num_base_transition_constraints()`.
    pub fn finish(self, num_base: usize) -> ConstraintProgram<F, E> {
        ConstraintProgram {
            nodes: self.nodes,
            dims: self.dims,
            base_consts: self.base_consts,
            ext_consts: self.ext_consts,
            roots: self.roots,
            num_base,
            complete: self.complete,
        }
    }
}
