//! Symbolic recording fields `SymField` (base) and `SymExt` (extension).
//!
//! These are zero-sized marker types that implement [`IsField`] (and, for
//! `SymField`, [`IsSubFieldOf<SymExt>`]) exactly like `GoldilocksField` and its
//! degree-3 extension do — except that every arithmetic operation *records* a
//! node into a thread-local arena instead of computing a value. Running a
//! constraint's generic `evaluate::<SymField, SymExt>(..)` therefore captures
//! the constraint's algebraic structure into a flat IR (see
//! [`crate::symbolic::ir`]).
//!
//! ## Why this compiles without `IsFFTField`/`IsPrimeField`
//!
//! The capture front-end never instantiates an `AIR<Field = SymField>`. It only
//! calls a constraint's `evaluate::<SymField, SymExt>`, whose bound is just
//! `FF: IsSubFieldOf<EE>, EE: IsField`. So `SymField` needs `IsField` +
//! `IsSubFieldOf<SymExt>` only — the FFT/prime-field methods are unreachable.
//!
//! Methods that cannot be symbolic (`inv`, `div`, the real `ByteConversion`)
//! are never reached by any algebraic constraint body and are left as
//! `unimplemented!()` so that a future constraint hitting them fails loudly
//! rather than producing silently wrong IR. `eq` returns a conservative `false`
//! so the runtime "skip zero term" optimizations are never taken at capture
//! time (the IR must be valid for all rows).

use std::cell::RefCell;
use std::collections::HashMap;

use math::field::element::FieldElement;
use math::field::errors::FieldError;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsField, IsSubFieldOf};
use math::traits::ByteConversion;

use super::ir::{Dim, Op};

/// A recorded value: a node id into the arena plus the dimension it denotes.
///
/// This is the `BaseType` for both `SymField` (always `D1`) and `SymExt`
/// (always `D3`). It is `Copy` POD, so the heavy `.clone()` use inside
/// constraint bodies is free and correct.
///
/// `Default` yields node id 0 (`Dim::D1`); the arena reserves node 0 as
/// `Const1(0)`, so a stray `SymId::default()` is the base-field zero.
#[derive(Clone, Copy, Debug, Default)]
pub struct SymId {
    /// Node id into the arena's `nodes` vector.
    pub id: u32,
    /// Dimension of the value this id denotes.
    pub dim: Dim,
}

/// `ByteConversion` is a `BaseType` bound on `IsField`, but the capture path
/// never serializes a `SymId`. These stubs satisfy the bound and panic loudly
/// if ever reached.
impl ByteConversion for SymId {
    const BYTE_LEN: usize = 0;

    fn to_bytes_be(&self) -> Vec<u8> {
        unimplemented!("ByteConversion::to_bytes_be is not symbolic")
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        unimplemented!("ByteConversion::to_bytes_le is not symbolic")
    }

    fn from_bytes_be(_bytes: &[u8]) -> Result<Self, math::errors::ByteConversionError> {
        unimplemented!("ByteConversion::from_bytes_be is not symbolic")
    }

    fn from_bytes_le(_bytes: &[u8]) -> Result<Self, math::errors::ByteConversionError> {
        unimplemented!("ByteConversion::from_bytes_le is not symbolic")
    }
}

// =============================================================================
// Thread-local recording arena
// =============================================================================

/// The recording arena: a topologically ordered node list with a hash-consing
/// (common-subexpression-elimination) map keyed on the full `Op` (which encodes
/// operand ids). The CSE key includes the opcode and operand ids; nodes of
/// different `Dim` never collide because the `Op` variant differs (`Const1` vs
/// `Const3`) or the operand-id subgraphs differ — and we additionally store the
/// dim alongside, never merging across dims.
struct Arena {
    nodes: Vec<Op>,
    dims: Vec<Dim>,
    /// CSE map: `(Op, Dim) -> id`. The dim is part of the key so a `D1` and a
    /// `D3` node that happen to share an `Op` shape (e.g. `Add(a, b)`) are kept
    /// distinct.
    cse: HashMap<(Op, Dim), u32>,
}

impl Arena {
    fn new() -> Self {
        let mut arena = Arena {
            nodes: Vec::new(),
            dims: Vec::new(),
            cse: HashMap::new(),
        };
        // Reserve node id 0 = Const1(0) so SymId::default() is the zero element.
        let zero = arena.intern(Op::Const1(0), Dim::D1);
        debug_assert_eq!(zero, 0);
        arena
    }

    /// Intern a node, returning its id; identical `(Op, Dim)` pairs are deduped.
    fn intern(&mut self, op: Op, dim: Dim) -> u32 {
        if let Some(&id) = self.cse.get(&(op, dim)) {
            return id;
        }
        let id = self.nodes.len() as u32;
        self.nodes.push(op);
        self.dims.push(dim);
        self.cse.insert((op, dim), id);
        id
    }
}

thread_local! {
    static ARENA: RefCell<Option<Arena>> = const { RefCell::new(None) };
}

/// Run `f` with a fresh recording arena installed, then return the captured
/// `(nodes, dims)`. Panics if called re-entrantly (capture is single-shot).
pub fn with_arena<R>(f: impl FnOnce() -> R) -> (Vec<Op>, Vec<Dim>, R) {
    ARENA.with(|cell| {
        assert!(
            cell.borrow().is_none(),
            "with_arena called re-entrantly; capture must be single-shot"
        );
        *cell.borrow_mut() = Some(Arena::new());
    });
    let result = f();
    ARENA.with(|cell| {
        let arena = cell
            .borrow_mut()
            .take()
            .expect("arena disappeared during capture");
        (arena.nodes, arena.dims, result)
    })
}

/// Record an op and return its `SymId` tagged with `dim`.
fn record(op: Op, dim: Dim) -> SymId {
    let id = ARENA.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let arena = borrow
            .as_mut()
            .expect("recording outside a with_arena scope");
        arena.intern(op, dim)
    });
    SymId { id, dim }
}

/// Record a leaf node (a column read) and return its `SymId`.
pub fn record_leaf(op: Op, dim: Dim) -> SymId {
    record(op, dim)
}

// =============================================================================
// SymField — base-field recorder (records D1 nodes)
// =============================================================================

/// Base-field recording marker. Every `IsField` op records a `Dim::D1` node.
#[derive(Debug, Clone)]
pub struct SymField;

impl IsField for SymField {
    type BaseType = SymId;

    fn add(a: &SymId, b: &SymId) -> SymId {
        record(Op::Add(a.id, b.id), Dim::D1)
    }

    fn sub(a: &SymId, b: &SymId) -> SymId {
        record(Op::Sub(a.id, b.id), Dim::D1)
    }

    fn mul(a: &SymId, b: &SymId) -> SymId {
        record(Op::Mul(a.id, b.id), Dim::D1)
    }

    fn neg(a: &SymId) -> SymId {
        record(Op::Neg(a.id), Dim::D1)
    }

    fn inv(_a: &SymId) -> Result<SymId, FieldError> {
        unimplemented!("SymField::inv: no algebraic constraint inverts")
    }

    fn div(_a: &SymId, _b: &SymId) -> Result<SymId, FieldError> {
        unimplemented!("SymField::div: no algebraic constraint divides")
    }

    fn eq(_a: &SymId, _b: &SymId) -> bool {
        // Conservative: never claim equality so runtime zero-skip optimizations
        // are not taken during capture. The captured IR must hold for all rows.
        false
    }

    fn zero() -> SymId {
        // Node id 0 is the reserved Const1(0).
        record(Op::Const1(0), Dim::D1)
    }

    fn one() -> SymId {
        record(Op::Const1(1), Dim::D1)
    }

    fn from_u64(x: u64) -> SymId {
        // Fold the real Goldilocks reduction so the stored literal is canonical.
        let reduced = GoldilocksField::from_u64(x);
        record(Op::Const1(reduced), Dim::D1)
    }

    fn from_base_type(x: SymId) -> SymId {
        x
    }
}

// =============================================================================
// SymExt — extension-field recorder (records D3 nodes)
// =============================================================================

/// Extension-field recording marker. Every `IsField` op records a `Dim::D3`
/// node.
#[derive(Debug, Clone)]
pub struct SymExt;

impl IsField for SymExt {
    type BaseType = SymId;

    fn add(a: &SymId, b: &SymId) -> SymId {
        record(Op::Add(a.id, b.id), Dim::D3)
    }

    fn sub(a: &SymId, b: &SymId) -> SymId {
        record(Op::Sub(a.id, b.id), Dim::D3)
    }

    fn mul(a: &SymId, b: &SymId) -> SymId {
        record(Op::Mul(a.id, b.id), Dim::D3)
    }

    fn neg(a: &SymId) -> SymId {
        record(Op::Neg(a.id), Dim::D3)
    }

    fn inv(_a: &SymId) -> Result<SymId, FieldError> {
        unimplemented!("SymExt::inv: no algebraic constraint inverts")
    }

    fn div(_a: &SymId, _b: &SymId) -> Result<SymId, FieldError> {
        unimplemented!("SymExt::div: no algebraic constraint divides")
    }

    fn eq(_a: &SymId, _b: &SymId) -> bool {
        false
    }

    fn zero() -> SymId {
        record(Op::Const3([0, 0, 0]), Dim::D3)
    }

    fn one() -> SymId {
        record(Op::Const3([1, 0, 0]), Dim::D3)
    }

    fn from_u64(x: u64) -> SymId {
        let reduced = GoldilocksField::from_u64(x);
        record(Op::Const3([reduced, 0, 0]), Dim::D3)
    }

    fn from_base_type(x: SymId) -> SymId {
        x
    }
}

// =============================================================================
// IsSubFieldOf<SymExt> for SymField — mixed base x ext arithmetic
// =============================================================================

impl IsSubFieldOf<SymExt> for SymField {
    fn mul(a: &SymId, b: &SymId) -> SymId {
        // base x ext -> ext. The interpreter sees a D1xD3 mul and does the
        // 3-mul base x ext path, matching the real IsSubFieldOf::mul.
        record(Op::Mul(a.id, b.id), Dim::D3)
    }

    fn add(a: &SymId, b: &SymId) -> SymId {
        record(Op::Add(a.id, b.id), Dim::D3)
    }

    fn sub(a: &SymId, b: &SymId) -> SymId {
        record(Op::Sub(a.id, b.id), Dim::D3)
    }

    fn div(_a: &SymId, _b: &SymId) -> Result<SymId, FieldError> {
        unimplemented!("SymField as IsSubFieldOf<SymExt>::div: not reached")
    }

    fn embed(a: SymId) -> SymId {
        record(Op::Embed(a.id), Dim::D3)
    }

    fn to_subfield_vec(_b: SymId) -> Vec<SymId> {
        unimplemented!("SymField as IsSubFieldOf<SymExt>::to_subfield_vec: not reached")
    }
}

/// Construct a `FieldElement<SymField>` wrapping a raw `SymId` leaf.
pub fn leaf_base(id: SymId) -> FieldElement<SymField> {
    FieldElement::from_raw(id)
}

/// Construct a `FieldElement<SymExt>` wrapping a raw `SymId` leaf.
#[allow(dead_code)]
pub fn leaf_ext(id: SymId) -> FieldElement<SymExt> {
    FieldElement::from_raw(id)
}
