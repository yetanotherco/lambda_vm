//! The register-budgeted lowering of a Goldilocks [`ConstraintProgram`] for the
//! bounded-slot composition interpreter (`kernels/constraint_si.cu`).
//!
//! [`DeviceProgram::lower`](super::device::DeviceProgram::lower) emits the nodes
//! in build order with every root pinned to the end of the walk, so the
//! interpreter's per-thread slot file holds thousands of words (KECCAK_RND:
//! 4,350) and lives in global memory. This lowering emits the same nodes in a
//! different order, under a fixed per-row budget of `B` words that a kernel can
//! keep in shared memory:
//!
//! - **Roots in index order, each root's cone on demand.** Root `c` is computed
//!   right before it is accumulated (`OP_ACC`), operands with the larger
//!   register need first (Sethi–Ullman), so a cone holds few values at once.
//! - **Leaves are operands, not slots.** A trace cell is read from the LDE at
//!   each use (`SIK_MAIN` / `SIK_AUX`), a constant or a per-proof uniform from
//!   its table (`SIK_BCONST` / `SIK_EUNI`). Only interior values take slots.
//! - **Shared values are kept while the budget allows and recomputed after.**
//!   A value stays in its slot until its last use in the current root, then
//!   until the next root that needs it, unless the budget is full: then the
//!   value whose next use is the furthest away is evicted (Belady), and it is
//!   computed again from its operands when a later root needs it.
//!
//! ⛔ BIT-IDENTICAL BY CONSTRUCTION. Goldilocks values on the device are
//! non-canonical `u64`s, so a value's bits depend on the exact operations that
//! produced it. Every step here is the call `eval_program_row` makes for the
//! node's op, operand dims and result dim (the mixed base/ext shortcuts
//! included), with the operands in IR order; a recomputed value is the same
//! function of the same leaves, so it has the same bits; and the transition
//! sum adds root `c` after roots `0..c` with the interpreter's `mul` /
//! `mul_base` choice. [`validate`] replays a lowered program symbolically and
//! checks each of those facts; [`eval_budgeted_row`] is the host model of the
//! kernel's walk.

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as GoldilocksExtension;
use math::field::goldilocks::GoldilocksField;

use super::ir::{ConstraintProgram, Dim, Op};

type FpE = FieldElement<GoldilocksField>;
type Ext3E = FieldElement<GoldilocksExtension>;
type Program = ConstraintProgram<GoldilocksField, GoldilocksExtension>;

// -------------------------------------------------------------------------
// Wire format — MUST match `kernels/constraint_si.cu`.
// -------------------------------------------------------------------------

/// Base result: `add(a, b)`, both operands base.
pub const SI_BADD: u32 = 0;
/// Base result: `sub(a, b)`.
pub const SI_BSUB: u32 = 1;
/// Base result: `mul(a, b)`.
pub const SI_BMUL: u32 = 2;
/// Base result: `neg(a)`.
pub const SI_BNEG: u32 = 3;
/// Ext result, neither operand base: `ext3::add(a, b)`.
pub const SI_EADD: u32 = 4;
/// Ext result, neither operand base: `ext3::sub(a, b)`.
pub const SI_ESUB: u32 = 5;
/// Ext result, neither operand base: `ext3::mul(a, b)`.
pub const SI_EMUL: u32 = 6;
/// Ext result, `a` base: `{add(x, y.a), y.b, y.c}` with `y` = `b` as ext3.
pub const SI_EADD_BX: u32 = 7;
/// Ext result, `a` base: `{sub(x, y.a), sub(0, y.b), sub(0, y.c)}`.
pub const SI_ESUB_BX: u32 = 8;
/// Ext result, `a` base: `mul_base(y, x)`.
pub const SI_EMUL_BX: u32 = 9;
/// Ext result, `a` not base, `b` base: `{add(x.a, y), x.b, x.c}`.
pub const SI_EADD_XB: u32 = 10;
/// Ext result, `a` not base, `b` base: `{sub(x.a, y), x.b, x.c}`.
pub const SI_ESUB_XB: u32 = 11;
/// Ext result, `a` not base, `b` base: `mul_base(x, y)`.
pub const SI_EMUL_XB: u32 = 12;
/// Ext result: `ext3::neg(a as ext3)`.
pub const SI_ENEG: u32 = 13;
/// Ext result: `a as ext3` (a base operand embeds as `{x, 0, 0}`).
pub const SI_EMBED: u32 = 14;
/// Transition sum, base root: `sum = ext3::add(sum, mul_base(beta[b], a))`.
pub const SI_ACC_B: u32 = 15;
/// Transition sum, ext root: `sum = ext3::add(sum, ext3::mul(beta[b], a))`.
pub const SI_ACC_E: u32 = 16;
/// The number of opcodes (the kernel's switch covers `0..SI_NUM_OPS`).
pub const SI_NUM_OPS: u32 = 17;

/// Bit position of the 3-bit operand kind.
pub const SIK_SHIFT: u32 = 29;
/// Mask of the 29-bit operand payload.
pub const SIK_PAYLOAD_MASK: u32 = (1 << SIK_SHIFT) - 1;
/// Payload = the base slot's word.
pub const SIK_BSLOT: u32 = 0;
/// Payload = the ext slot's first word (components at `w`, `w + 1`, `w + 2`).
pub const SIK_ESLOT: u32 = 1;
/// Payload = a main (base) column: `col | offset << SI_COL_OFFSET_SHIFT`.
pub const SIK_MAIN: u32 = 2;
/// Payload = an aux (ext3) column: `col | offset << SI_COL_OFFSET_SHIFT`.
pub const SIK_AUX: u32 = 3;
/// Payload = a `base_consts` index.
pub const SIK_BCONST: u32 = 4;
/// Payload = an index into the ext uniform table: the program's `ext_consts`,
/// then `num_rap` RAP challenges, then `num_alpha` LogUp alpha powers, then
/// the table offset ([`BudgetedProgram::ext_uniform_table`]).
pub const SIK_EUNI: u32 = 5;
/// Where a column operand keeps its frame offset (0 or 1).
pub const SI_COL_OFFSET_SHIFT: u32 = 20;
/// The largest column index a column operand can name.
pub const SI_MAX_COL: u32 = (1 << SI_COL_OFFSET_SHIFT) - 1;

/// One step of a budgeted program: 16 bytes, `#[repr(C)]`, uploaded as is
/// (one `uint4` per step on the device). `a`/`b` are encoded operands (`SIK_*`)
/// except for `OP_ACC*`, whose `b` is the root's index into `beta`; `dst` is
/// the result's first word (unused by `OP_ACC*`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SiStep {
    pub op: u32,
    pub a: u32,
    pub b: u32,
    pub dst: u32,
}

/// What a lowering cost, for the census and the choice of budget.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BudgetStats {
    /// Steps that compute a node (every step but the accumulations).
    pub compute_steps: usize,
    /// Distinct nodes computed (a recomputation counts once here).
    pub distinct_nodes: usize,
    /// Accumulation steps (one per root).
    pub acc_steps: usize,
    /// Column operands read (main and aux, every use).
    pub col_reads: usize,
    /// Base words and ext slots the lowering was allowed.
    pub base_capacity: u32,
    pub ext_capacity: u32,
}

/// A [`ConstraintProgram`] lowered for the bounded-slot interpreter.
#[derive(Clone, Debug)]
pub struct BudgetedProgram {
    /// The steps, in run order.
    pub steps: Vec<SiStep>,
    /// For each step, the IR node it computes (`u32::MAX` for an accumulation).
    /// Not uploaded: [`validate`] reads it.
    pub step_nodes: Vec<u32>,
    /// Base constants, raw limbs (`SIK_BCONST`).
    pub base_consts: Vec<u64>,
    /// The program's ext constants, raw limbs: the head of the ext uniform
    /// table.
    pub ext_consts: Vec<[u64; 3]>,
    /// RAP challenges the ext uniform table carries (the program reads
    /// `0..num_rap`).
    pub num_rap: u32,
    /// LogUp alpha powers the ext uniform table carries.
    pub num_alpha: u32,
    /// Words per row: base words `0..base_words`, then the ext slots.
    pub num_words: u32,
    /// The base words; ext slot `e` occupies words `base_words + 3e ..+3`.
    pub base_words: u32,
    /// The number of roots (= the length of `beta`).
    pub num_roots: u32,
    /// Lowering statistics.
    pub stats: BudgetStats,
}

impl BudgetedProgram {
    /// The ext uniform table for one proof: the program's ext constants, the
    /// first `num_rap` RAP challenges, the first `num_alpha` alpha powers and
    /// the table offset. `None` when the proof carries fewer challenges or
    /// powers than the program reads.
    pub fn ext_uniform_table(
        &self,
        rap: &[[u64; 3]],
        alpha: &[[u64; 3]],
        offset: [u64; 3],
    ) -> Option<Vec<[u64; 3]>> {
        let (nr, na) = (self.num_rap as usize, self.num_alpha as usize);
        if rap.len() < nr || alpha.len() < na {
            return None;
        }
        let mut t = Vec::with_capacity(self.ext_consts.len() + nr + na + 1);
        t.extend_from_slice(&self.ext_consts);
        t.extend_from_slice(&rap[..nr]);
        t.extend_from_slice(&alpha[..na]);
        t.push(offset);
        Some(t)
    }

    /// The steps as `u32` words, four per step (the device upload).
    pub fn packed_steps(&self) -> Vec<u32> {
        self.steps
            .iter()
            .flat_map(|s| [s.op, s.a, s.b, s.dst])
            .collect()
    }
}

/// Why a program has no budgeted lowering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BudgetError {
    /// A leaf the wire format cannot name (a frame offset above 1, a column
    /// above [`SI_MAX_COL`], a table index above the payload).
    Unencodable { node: u32 },
    /// A node whose dims the interpreter's op semantics do not cover.
    BadDims { node: u32 },
    /// No split of the budget into base words and ext slots fits the
    /// program's pinned values.
    OverBudget,
}

const NONE: u32 = u32::MAX;

fn operands(op: &Op) -> [Option<u32>; 2] {
    match *op {
        Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) => [Some(a), Some(b)],
        Op::Neg(a) | Op::Embed(a) => [Some(a), None],
        _ => [None, None],
    }
}

fn enc(kind: u32, payload: u32) -> u32 {
    debug_assert!(payload <= SIK_PAYLOAD_MASK);
    (kind << SIK_SHIFT) | payload
}

/// Whether an encoded operand is base-valued (a base slot, a main column or a
/// base constant): the `opk_is_base` of `eval_program_row`.
pub fn sik_is_base(e: u32) -> bool {
    matches!(e >> SIK_SHIFT, SIK_BSLOT | SIK_MAIN | SIK_BCONST)
}

/// The program-wide facts every split shares.
struct Shape<'p> {
    prog: &'p Program,
    /// Leaf operand encoding (`NONE` for an interior node).
    leaf: Vec<u32>,
    /// Ordered operands to evaluate (interior operands only, the larger need
    /// first).
    order: Vec<[u32; 2]>,
    /// The unbounded demand schedule (every interior node computed once, at
    /// its first need, and held): each node's position in it.
    first_pos: Vec<u32>,
    /// Each accumulation's position in that schedule.
    acc_pos: Vec<u32>,
    /// For each interior node, the positions in that schedule that read it,
    /// ascending: the next-use oracle of the budgeted run.
    uses: Vec<Vec<u32>>,
    /// A node's recomputation cost when nothing below it is held: the steps
    /// of its interior cone as a tree, capped.
    cost: Vec<u32>,
    num_rap: u32,
    num_alpha: u32,
}

fn words(d: Dim) -> u32 {
    match d {
        Dim::Base => 1,
        Dim::Ext => 3,
    }
}

impl<'p> Shape<'p> {
    fn new(prog: &'p Program) -> Result<Self, BudgetError> {
        let n = prog.nodes.len();
        let ne = prog.ext_consts.len() as u32;
        let (mut num_rap, mut num_alpha) = (0u32, 0u32);
        for op in &prog.nodes {
            match *op {
                Op::RapChallenge { idx } => num_rap = num_rap.max(idx as u32 + 1),
                Op::AlphaPow { idx } => num_alpha = num_alpha.max(idx as u32 + 1),
                _ => {}
            }
        }
        let mut leaf = vec![NONE; n];
        for (i, op) in prog.nodes.iter().enumerate() {
            let bad = BudgetError::Unencodable { node: i as u32 };
            let e = match *op {
                Op::ConstBase(idx) => {
                    if idx > SIK_PAYLOAD_MASK || prog.dims[i] != Dim::Base {
                        return Err(bad);
                    }
                    enc(SIK_BCONST, idx)
                }
                Op::ConstExt(idx) => uniform(idx, &bad)?,
                Op::RapChallenge { idx } => uniform(ne + idx as u32, &bad)?,
                Op::AlphaPow { idx } => uniform(ne + num_rap + idx as u32, &bad)?,
                Op::TableOffset => uniform(ne + num_rap + num_alpha, &bad)?,
                Op::Var {
                    main,
                    offset,
                    row,
                    col,
                } => {
                    if offset > 1 || row != 0 || col as u32 > SI_MAX_COL {
                        return Err(bad);
                    }
                    let payload = col as u32 | (offset as u32) << SI_COL_OFFSET_SHIFT;
                    let (kind, dim) = if main {
                        (SIK_MAIN, Dim::Base)
                    } else {
                        (SIK_AUX, Dim::Ext)
                    };
                    if prog.dims[i] != dim {
                        return Err(BudgetError::BadDims { node: i as u32 });
                    }
                    enc(kind, payload)
                }
                _ => NONE,
            };
            leaf[i] = e;
        }
        // Uniform ext leaves must be ext-dimensioned.
        for (i, op) in prog.nodes.iter().enumerate() {
            let ext_leaf = matches!(
                op,
                Op::ConstExt(_) | Op::RapChallenge { .. } | Op::AlphaPow { .. } | Op::TableOffset
            );
            if ext_leaf && prog.dims[i] != Dim::Ext {
                return Err(BudgetError::BadDims { node: i as u32 });
            }
            // A base result needs base operands; neg keeps its operand's dim.
            let ops = operands(op);
            match *op {
                Op::Add(..) | Op::Sub(..) | Op::Mul(..) | Op::Neg(..)
                    if prog.dims[i] == Dim::Base =>
                {
                    if ops
                        .into_iter()
                        .flatten()
                        .any(|o| prog.dims[o as usize] != Dim::Base)
                    {
                        return Err(BudgetError::BadDims { node: i as u32 });
                    }
                }
                Op::Embed(..) if prog.dims[i] != Dim::Ext => {
                    return Err(BudgetError::BadDims { node: i as u32 });
                }
                _ => {}
            }
        }

        // Register need (Sethi–Ullman on the tree view of the DAG): the words
        // a node's evaluation holds at its peak, leaves free.
        let mut need = vec![0u32; n];
        let mut order = vec![[NONE; 2]; n];
        for i in 0..n {
            if leaf[i] != NONE {
                continue;
            }
            let w = words(prog.dims[i]);
            let inner: Vec<u32> = operands(&prog.nodes[i])
                .into_iter()
                .flatten()
                .filter(|&o| leaf[o as usize] == NONE)
                .collect();
            let (nd, ord) = match inner.as_slice() {
                [] => (w, [NONE; 2]),
                [x] => (need[*x as usize].max(w), [*x, NONE]),
                [x, y] if x == y => (need[*x as usize].max(w), [*x, NONE]),
                [x, y] => {
                    let (nx, ny) = (need[*x as usize], need[*y as usize]);
                    let (wx, wy) = (words(prog.dims[*x as usize]), words(prog.dims[*y as usize]));
                    // Holding the first result while the second is computed.
                    let xy = nx.max(wx + ny).max(wx + wy);
                    let yx = ny.max(wy + nx).max(wx + wy);
                    if xy <= yx {
                        (xy.max(w), [*x, *y])
                    } else {
                        (yx.max(w), [*y, *x])
                    }
                }
                _ => unreachable!("at most two operands"),
            };
            need[i] = nd;
            order[i] = ord;
        }

        // The unbounded demand schedule: roots in index order, each root's
        // cone depth first with the larger need first, every interior node
        // computed once at its first need. The budgeted run computes nodes for
        // the first time in exactly this order (a recomputation only re-runs
        // nodes computed before), so these positions are its timeline.
        let mut first_pos = vec![NONE; n];
        let mut acc_pos = Vec::with_capacity(prog.roots.len());
        let mut uses: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut pos = 0u32;
        let mut stack: Vec<(u32, u8)> = Vec::new();
        for &r in &prog.roots {
            stack.push((r, 0));
            while let Some(&(i, phase)) = stack.last() {
                let iu = i as usize;
                if phase == 0 && (leaf[iu] != NONE || first_pos[iu] != NONE) {
                    stack.pop();
                    continue;
                }
                let ord = order[iu];
                match phase {
                    0 | 1 => {
                        stack.last_mut().unwrap().1 = phase + 1;
                        if ord[phase as usize] != NONE {
                            stack.push((ord[phase as usize], 0));
                        }
                    }
                    _ => {
                        first_pos[iu] = pos;
                        for o in ord.into_iter().filter(|&o| o != NONE) {
                            uses[o as usize].push(pos);
                        }
                        pos += 1;
                        stack.pop();
                    }
                }
            }
            acc_pos.push(pos);
            if leaf[r as usize] == NONE {
                uses[r as usize].push(pos);
            }
            pos += 1;
        }
        const COST_CAP: u32 = 1 << 16;
        let mut cost = vec![0u32; n];
        for i in 0..n {
            if leaf[i] == NONE {
                cost[i] = order[i]
                    .into_iter()
                    .filter(|&o| o != NONE)
                    .fold(1u32, |c, o| c.saturating_add(cost[o as usize]))
                    .min(COST_CAP);
            }
        }
        Ok(Shape {
            prog,
            leaf,
            order,
            first_pos,
            acc_pos,
            uses,
            cost,
            num_rap,
            num_alpha,
        })
    }
}

fn uniform(idx: u32, bad: &BudgetError) -> Result<u32, BudgetError> {
    if idx > SIK_PAYLOAD_MASK {
        return Err(bad.clone());
    }
    Ok(enc(SIK_EUNI, idx))
}

/// A slot reference before the final word numbering: base word or ext slot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Loc {
    Base(u32),
    Ext(u32),
}

/// An operand before the final word numbering.
#[derive(Clone, Copy, Debug)]
enum Arg {
    Leaf(u32),
    Slot(Loc),
}

#[derive(Clone, Copy, Debug)]
struct RawStep {
    op: u32,
    a: Option<Arg>,
    b: Option<Arg>,
    beta: u32,
    dst: Option<Loc>,
    node: u32,
}

/// How a full budget picks the value to evict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// The value whose next use is furthest away (Belady).
    Furthest,
    /// The furthest next use per unit of recomputation cost.
    FurthestPerCost,
}

/// One allocation run under a fixed split: `nb` base words, `ne` ext slots.
///
/// The run walks the demand schedule ([`Shape::first_pos`]); `now` is the
/// schedule position of the last first computation or accumulation. A held
/// value's next use is its first schedule use after `now`: when a slot is
/// needed and none is free, the unpinned value with the furthest next use is
/// evicted (none = dead, evicted first), and a later need computes it again
/// from its operands.
struct Run<'s, 'p> {
    s: &'s Shape<'p>,
    loc: Vec<Option<Loc>>,
    holder_b: Vec<u32>,
    holder_e: Vec<u32>,
    free_b: std::collections::BTreeSet<u32>,
    free_e: std::collections::BTreeSet<u32>,
    pins: Vec<u32>,
    computed: Vec<bool>,
    cursor: Vec<u32>,
    now: u32,
    policy: Policy,
    steps: Vec<RawStep>,
    step_limit: usize,
}

impl<'s, 'p> Run<'s, 'p> {
    fn new(s: &'s Shape<'p>, nb: u32, ne: u32, policy: Policy, step_limit: usize) -> Self {
        let n = s.prog.nodes.len();
        Run {
            s,
            loc: vec![None; n],
            holder_b: vec![NONE; nb as usize],
            holder_e: vec![NONE; ne as usize],
            free_b: (0..nb).collect(),
            free_e: (0..ne).collect(),
            pins: vec![0; n],
            computed: vec![false; n],
            cursor: vec![0; n],
            now: 0,
            policy,
            steps: Vec::new(),
            step_limit,
        }
    }

    /// The first schedule use of `i` after `now` (`None`: no use left).
    fn next_use(&mut self, i: usize) -> Option<u32> {
        let list = &self.s.uses[i];
        let mut k = self.cursor[i] as usize;
        while k < list.len() && list[k] <= self.now {
            k += 1;
        }
        self.cursor[i] = k as u32;
        list.get(k).copied()
    }

    fn release(&mut self, i: usize) {
        match self.loc[i].take() {
            Some(Loc::Base(w)) => {
                self.holder_b[w as usize] = NONE;
                self.free_b.insert(w);
            }
            Some(Loc::Ext(e)) => {
                self.holder_e[e as usize] = NONE;
                self.free_e.insert(e);
            }
            None => {}
        }
    }

    fn dead(&mut self, i: usize) -> bool {
        self.pins[i] == 0 && self.next_use(i).is_none()
    }

    /// A free slot of the node's class, evicting the value needed furthest in
    /// the future when none is free. `None` when every held value is pinned.
    fn alloc(&mut self, dim: Dim) -> Option<Loc> {
        let free = match dim {
            Dim::Base => self.free_b.pop_first().map(Loc::Base),
            Dim::Ext => self.free_e.pop_first().map(Loc::Ext),
        };
        if free.is_some() {
            return free;
        }
        let holders = match dim {
            Dim::Base => self.holder_b.clone(),
            Dim::Ext => self.holder_e.clone(),
        };
        let mut best: Option<(u64, usize)> = None;
        for &h in &holders {
            if h == NONE || self.pins[h as usize] > 0 {
                continue;
            }
            let i = h as usize;
            let key = match (self.next_use(i), self.policy) {
                (None, _) => u64::MAX,
                (Some(u), Policy::Furthest) => u as u64,
                (Some(u), Policy::FurthestPerCost) => {
                    ((u - self.now) as u64).saturating_mul(1 << 20) / self.s.cost[i] as u64
                }
            };
            if best.is_none_or(|(k, _)| key > k) {
                best = Some((key, i));
            }
        }
        let (_, victim) = best?;
        let l = self.loc[victim];
        self.release(victim);
        match l {
            Some(Loc::Base(w)) => {
                self.free_b.remove(&w);
                Some(Loc::Base(w))
            }
            Some(Loc::Ext(e)) => {
                self.free_e.remove(&e);
                Some(Loc::Ext(e))
            }
            None => unreachable!("a holder has a slot"),
        }
    }

    fn arg(&self, i: u32) -> Arg {
        let l = self.s.leaf[i as usize];
        if l != NONE {
            Arg::Leaf(l)
        } else {
            Arg::Slot(self.loc[i as usize].expect("an ensured operand holds a slot"))
        }
    }

    /// Make node `top`'s value available (a leaf, or a held slot), computing
    /// whatever is missing below it.
    fn ensure(&mut self, top: u32) -> Result<(), BudgetError> {
        // (node, phase): 0 entering, 1 first operand ready, 2 both ready.
        let mut stack: Vec<(u32, u8)> = vec![(top, 0)];
        while let Some(&(i, phase)) = stack.last() {
            let iu = i as usize;
            let ord = self.s.order[iu];
            match phase {
                0 => {
                    if self.s.leaf[iu] != NONE || self.loc[iu].is_some() {
                        stack.pop();
                        continue;
                    }
                    if self.steps.len() > self.step_limit {
                        return Err(BudgetError::OverBudget);
                    }
                    stack.last_mut().unwrap().1 = 1;
                    if ord[0] != NONE {
                        stack.push((ord[0], 0));
                    }
                }
                1 => {
                    if ord[0] != NONE {
                        self.pins[ord[0] as usize] += 1;
                    }
                    stack.last_mut().unwrap().1 = 2;
                    if ord[1] != NONE {
                        stack.push((ord[1], 0));
                    }
                }
                _ => {
                    if ord[1] != NONE {
                        self.pins[ord[1] as usize] += 1;
                    }
                    self.compute(i)?;
                    stack.pop();
                }
            }
        }
        Ok(())
    }

    /// Emit node `i`'s step: its operands are ready (and pinned by `ensure`).
    fn compute(&mut self, i: u32) -> Result<(), BudgetError> {
        let iu = i as usize;
        let op = self.s.prog.nodes[iu];
        let dim = self.s.prog.dims[iu];
        let [oa, ob] = operands(&op);
        let a = oa.map(|o| self.arg(o));
        let b = ob.map(|o| self.arg(o));
        let tag = opcode(self.s.prog, i).ok_or(BudgetError::BadDims { node: i })?;
        // A first computation moves the timeline to its schedule position; a
        // recomputation happens at the current one.
        if !std::mem::replace(&mut self.computed[iu], true) {
            debug_assert!(self.s.first_pos[iu] >= self.now || self.now == 0);
            self.now = self.s.first_pos[iu];
        }
        // Release the pins `ensure` took, and every operand with no use left
        // (its slot may then hold the result: the kernel reads the operands
        // before it writes).
        for o in self.s.order[iu].into_iter().filter(|&o| o != NONE) {
            self.pins[o as usize] -= 1;
        }
        for o in self.s.order[iu].into_iter().filter(|&o| o != NONE) {
            if self.loc[o as usize].is_some() && self.dead(o as usize) {
                self.release(o as usize);
            }
        }
        let dst = self.alloc(dim).ok_or(BudgetError::OverBudget)?;
        match dst {
            Loc::Base(w) => self.holder_b[w as usize] = i,
            Loc::Ext(e) => self.holder_e[e as usize] = i,
        }
        self.loc[iu] = Some(dst);
        self.steps.push(RawStep {
            op: tag,
            a,
            b,
            beta: 0,
            dst: Some(dst),
            node: i,
        });
        Ok(())
    }

    fn run(mut self) -> Result<Vec<RawStep>, BudgetError> {
        let prog = self.s.prog;
        for c in 0..prog.roots.len() as u32 {
            let r = prog.roots[c as usize];
            self.ensure(r)?;
            let a = self.arg(r);
            let tag = match prog.dims[r as usize] {
                Dim::Base => SI_ACC_B,
                Dim::Ext => SI_ACC_E,
            };
            self.steps.push(RawStep {
                op: tag,
                a: Some(a),
                b: None,
                beta: c,
                dst: None,
                node: NONE,
            });
            self.now = self.s.acc_pos[c as usize];
            if self.s.leaf[r as usize] == NONE && self.dead(r as usize) {
                self.release(r as usize);
            }
        }
        Ok(self.steps)
    }
}

/// The opcode the interpreter's op semantics select for node `i`: its op,
/// its result dim, and which operand (if any) is base-valued.
fn opcode(prog: &Program, i: u32) -> Option<u32> {
    let op = prog.nodes[i as usize];
    let [oa, ob] = operands(&op);
    let base_dim = |o: Option<u32>| o.is_some_and(|o| prog.dims[o as usize] == Dim::Base);
    Some(match (op, prog.dims[i as usize]) {
        (Op::Add(..), Dim::Base) => SI_BADD,
        (Op::Sub(..), Dim::Base) => SI_BSUB,
        (Op::Mul(..), Dim::Base) => SI_BMUL,
        (Op::Neg(..), Dim::Base) => SI_BNEG,
        (Op::Neg(..), Dim::Ext) => SI_ENEG,
        (Op::Embed(..), _) => SI_EMBED,
        (Op::Add(..) | Op::Sub(..) | Op::Mul(..), Dim::Ext) => {
            let k = match op {
                Op::Add(..) => 0,
                Op::Sub(..) => 1,
                _ => 2,
            };
            if base_dim(oa) {
                SI_EADD_BX + k
            } else if base_dim(ob) {
                SI_EADD_XB + k
            } else {
                SI_EADD + k
            }
        }
        _ => return None,
    })
}

/// Number the words (base words first, compacted to those used, then the ext
/// slots), encode every operand and build the program.
fn finish(s: &Shape<'_>, raw: Vec<RawStep>, nb: u32, ne: u32) -> BudgetedProgram {
    let prog = s.prog;
    let mut base_used = 0u32;
    let mut ext_used = 0u32;
    for st in &raw {
        for l in [st.a, st.b]
            .into_iter()
            .flatten()
            .filter_map(|a| match a {
                Arg::Slot(l) => Some(l),
                Arg::Leaf(_) => None,
            })
            .chain(st.dst)
        {
            match l {
                Loc::Base(w) => base_used = base_used.max(w + 1),
                Loc::Ext(e) => ext_used = ext_used.max(e + 1),
            }
        }
    }
    let word = |l: Loc| match l {
        Loc::Base(w) => w,
        Loc::Ext(e) => base_used + 3 * e,
    };
    let encode = |a: Arg| match a {
        Arg::Leaf(e) => e,
        Arg::Slot(l @ Loc::Base(_)) => enc(SIK_BSLOT, word(l)),
        Arg::Slot(l @ Loc::Ext(_)) => enc(SIK_ESLOT, word(l)),
    };
    let mut stats = BudgetStats {
        base_capacity: nb,
        ext_capacity: ne,
        ..Default::default()
    };
    let mut computed = vec![false; prog.nodes.len()];
    let mut steps = Vec::with_capacity(raw.len());
    let mut step_nodes = Vec::with_capacity(raw.len());
    for st in &raw {
        let a = st.a.map(encode).unwrap_or(0);
        let b = match st.op {
            SI_ACC_B | SI_ACC_E => st.beta,
            _ => st.b.map(encode).unwrap_or(0),
        };
        if st.node == NONE {
            stats.acc_steps += 1;
        } else {
            stats.compute_steps += 1;
            if !std::mem::replace(&mut computed[st.node as usize], true) {
                stats.distinct_nodes += 1;
            }
        }
        for e in [st.a, st.b].into_iter().flatten() {
            if let Arg::Leaf(e) = e
                && matches!(e >> SIK_SHIFT, SIK_MAIN | SIK_AUX)
            {
                stats.col_reads += 1;
            }
        }
        steps.push(SiStep {
            op: st.op,
            a,
            b,
            dst: st.dst.map(word).unwrap_or(0),
        });
        step_nodes.push(st.node);
    }
    BudgetedProgram {
        steps,
        step_nodes,
        base_consts: prog.base_consts.iter().map(|c| *c.value()).collect(),
        ext_consts: prog.ext_consts.iter().map(encode_ext).collect(),
        num_rap: s.num_rap,
        num_alpha: s.num_alpha,
        num_words: base_used + 3 * ext_used,
        base_words: base_used,
        num_roots: prog.roots.len() as u32,
        stats,
    }
}

/// Lower `prog` under one split of the per-row budget: `nb` base words and
/// `ne` ext slots (`nb + 3·ne` words), evicting by `policy`.
pub fn lower_with_split(
    prog: &Program,
    nb: u32,
    ne: u32,
    policy: Policy,
) -> Result<BudgetedProgram, BudgetError> {
    let s = Shape::new(prog)?;
    lower_split(&s, nb, ne, policy)
}

fn lower_split(
    s: &Shape<'_>,
    nb: u32,
    ne: u32,
    policy: Policy,
) -> Result<BudgetedProgram, BudgetError> {
    // A run that recomputes this much has thrashed: refuse the split.
    let limit = 8 * s.prog.nodes.len() + 64;
    let raw = Run::new(s, nb, ne, policy, limit).run()?;
    Ok(finish(s, raw, nb, ne))
}

/// Lower `prog` within `budget_words` words a row: every split of the budget
/// into base words and ext slots, under both eviction policies, is tried, and
/// the lowering with the fewest steps is kept (ties: fewer words).
pub fn lower_budgeted(prog: &Program, budget_words: u32) -> Result<BudgetedProgram, BudgetError> {
    let s = Shape::new(prog)?;
    let mut best: Option<BudgetedProgram> = None;
    let max_ne = budget_words / 3;
    // Every split up to 64 of them, a grid of 64 above.
    let stride = max_ne.div_ceil(64).max(1);
    let mut ne = 1;
    while ne <= max_ne {
        let nb = budget_words - 3 * ne;
        for policy in [Policy::Furthest, Policy::FurthestPerCost] {
            if nb >= 1
                && let Ok(p) = lower_split(&s, nb, ne, policy)
            {
                let better = best
                    .as_ref()
                    .is_none_or(|b| (p.steps.len(), p.num_words) < (b.steps.len(), b.num_words));
                if better {
                    best = Some(p);
                }
            }
        }
        ne += stride;
    }
    best.ok_or(BudgetError::OverBudget)
}

fn encode_ext(x: &Ext3E) -> [u64; 3] {
    let limbs = x.value();
    [*limbs[0].value(), *limbs[1].value(), *limbs[2].value()]
}

fn decode_ext(l: [u64; 3]) -> Ext3E {
    Ext3E::from_raw([
        FpE::from_raw(l[0]),
        FpE::from_raw(l[1]),
        FpE::from_raw(l[2]),
    ])
}

/// Replay a lowered program symbolically against the IR it came from: every
/// step computes its node's op with its node's operands in IR order (a slot
/// operand must hold that operand's value at that point, a leaf operand must
/// be that leaf), with the opcode the interpreter's dims select; the result
/// lands in a slot of the node's class; and the accumulations add roots
/// `0..n` in order, each from its root's value with the root dim's opcode.
pub fn validate(prog: &Program, bp: &BudgetedProgram) -> Result<(), String> {
    let s = Shape::new(prog).map_err(|e| format!("{e:?}"))?;
    if bp.steps.len() != bp.step_nodes.len() {
        return Err("step/node length mismatch".into());
    }
    let words = bp.num_words as usize;
    // holds[w] = (node, component) for every word.
    let mut holds: Vec<Option<(u32, u8)>> = vec![None; words];
    let mut next_acc = 0u32;
    let check_operand = |holds: &[Option<(u32, u8)>], e: u32, node: u32| -> Result<(), String> {
        let want_leaf = s.leaf[node as usize];
        let kind = e >> SIK_SHIFT;
        let p = (e & SIK_PAYLOAD_MASK) as usize;
        if want_leaf != NONE {
            return if e == want_leaf {
                Ok(())
            } else {
                Err(format!(
                    "leaf operand {e:#x} is not node {node}'s {want_leaf:#x}"
                ))
            };
        }
        match (kind, prog.dims[node as usize]) {
            (SIK_BSLOT, Dim::Base) => {
                if p >= bp.base_words as usize || holds[p] != Some((node, 0)) {
                    return Err(format!(
                        "base word {p} does not hold node {node}: {:?}",
                        holds.get(p)
                    ));
                }
            }
            (SIK_ESLOT, Dim::Ext) => {
                if p < bp.base_words as usize || p + 3 > words {
                    return Err(format!("ext word {p} out of the ext region"));
                }
                for k in 0..3 {
                    if holds[p + k] != Some((node, k as u8)) {
                        return Err(format!("ext word {} does not hold node {node}.{k}", p + k));
                    }
                }
            }
            _ => {
                return Err(format!(
                    "operand kind {kind} for node {node} of the wrong class"
                ));
            }
        }
        Ok(())
    };
    for (k, (st, &node)) in bp.steps.iter().zip(&bp.step_nodes).enumerate() {
        let err = |m: String| format!("step {k} ({st:?}, node {node}): {m}");
        if matches!(st.op, SI_ACC_B | SI_ACC_E) {
            if node != NONE || st.b != next_acc {
                return Err(err(format!(
                    "accumulation {} out of order (want {next_acc})",
                    st.b
                )));
            }
            let r = prog.roots[next_acc as usize];
            let want = match prog.dims[r as usize] {
                Dim::Base => SI_ACC_B,
                Dim::Ext => SI_ACC_E,
            };
            if st.op != want {
                return Err(err(
                    "accumulation opcode disagrees with the root's dim".into()
                ));
            }
            check_operand(&holds, st.a, r).map_err(err)?;
            next_acc += 1;
            continue;
        }
        if node == NONE || node as usize >= prog.nodes.len() || s.leaf[node as usize] != NONE {
            return Err(err("not an interior node".into()));
        }
        let op = prog.nodes[node as usize];
        let dim = prog.dims[node as usize];
        let [oa, ob] = operands(&op);
        let want = opcode(prog, node).ok_or_else(|| err("no opcode for this node".into()))?;
        if st.op != want {
            return Err(err(format!("opcode {} (want {want})", st.op)));
        }
        if let Some(o) = oa {
            check_operand(&holds, st.a, o).map_err(err)?;
        }
        if let Some(o) = ob {
            check_operand(&holds, st.b, o).map_err(err)?;
        }
        let d = st.dst as usize;
        match dim {
            Dim::Base => {
                if d >= bp.base_words as usize {
                    return Err(err("base result outside the base words".into()));
                }
                holds[d] = Some((node, 0));
            }
            Dim::Ext => {
                if d < bp.base_words as usize
                    || d + 3 > words
                    || !(d - bp.base_words as usize).is_multiple_of(3)
                {
                    return Err(err("ext result outside an ext slot".into()));
                }
                for c in 0..3 {
                    holds[d + c] = Some((node, c as u8));
                }
            }
        }
    }
    if next_acc != prog.roots.len() as u32 {
        return Err(format!(
            "{next_acc} of {} roots accumulated",
            prog.roots.len()
        ));
    }
    Ok(())
}

/// The host model of the kernel's walk for one row: the transition sum
/// `Σ_c beta_c · C_c` (before the zerofier and the boundary terms), in raw
/// limbs. `main[offset][col]` / `aux[offset][col]` are the frame's cells,
/// `uniforms` the ext uniform table ([`BudgetedProgram::ext_uniform_table`]),
/// `beta` one coefficient per root. Mixed ops are evaluated as full ext ops on
/// embedded operands, as [`eval_device_program`](super::device::eval_device_program)
/// does (the kernel's shortcuts are bit-identical to that).
pub fn eval_budgeted_row(
    bp: &BudgetedProgram,
    main: &[Vec<u64>],
    aux: &[Vec<[u64; 3]>],
    uniforms: &[[u64; 3]],
    beta: &[[u64; 3]],
) -> [u64; 3] {
    let mut w = vec![0u64; bp.num_words as usize];
    let base = |w: &[u64], e: u32| -> FpE {
        let p = (e & SIK_PAYLOAD_MASK) as usize;
        match e >> SIK_SHIFT {
            SIK_BSLOT => FpE::from_raw(w[p]),
            SIK_MAIN => {
                let (col, off) = (p & SI_MAX_COL as usize, p >> SI_COL_OFFSET_SHIFT);
                FpE::from_raw(main[off][col])
            }
            SIK_BCONST => FpE::from_raw(bp.base_consts[p]),
            k => panic!("base operand of kind {k}"),
        }
    };
    let ext = |w: &[u64], e: u32| -> Ext3E {
        let p = (e & SIK_PAYLOAD_MASK) as usize;
        match e >> SIK_SHIFT {
            SIK_BSLOT | SIK_MAIN | SIK_BCONST => base(w, e).to_extension::<GoldilocksExtension>(),
            SIK_ESLOT => decode_ext([w[p], w[p + 1], w[p + 2]]),
            SIK_AUX => {
                let (col, off) = (p & SI_MAX_COL as usize, p >> SI_COL_OFFSET_SHIFT);
                decode_ext(aux[off][col])
            }
            SIK_EUNI => decode_ext(uniforms[p]),
            k => panic!("operand of kind {k}"),
        }
    };
    let put_b = |w: &mut [u64], d: u32, v: FpE| w[d as usize] = *v.value();
    let put_e = |w: &mut [u64], d: u32, v: Ext3E| {
        let l = encode_ext(&v);
        w[d as usize..d as usize + 3].copy_from_slice(&l);
    };
    let mut sum = Ext3E::zero();
    for st in &bp.steps {
        match st.op {
            SI_BADD => {
                let v = base(&w, st.a) + base(&w, st.b);
                put_b(&mut w, st.dst, v)
            }
            SI_BSUB => {
                let v = base(&w, st.a) - base(&w, st.b);
                put_b(&mut w, st.dst, v)
            }
            SI_BMUL => {
                let v = base(&w, st.a) * base(&w, st.b);
                put_b(&mut w, st.dst, v)
            }
            SI_BNEG => {
                let v = -base(&w, st.a);
                put_b(&mut w, st.dst, v)
            }
            SI_EADD | SI_EADD_BX | SI_EADD_XB => {
                let v = ext(&w, st.a) + ext(&w, st.b);
                put_e(&mut w, st.dst, v)
            }
            SI_ESUB | SI_ESUB_BX | SI_ESUB_XB => {
                let v = ext(&w, st.a) - ext(&w, st.b);
                put_e(&mut w, st.dst, v)
            }
            SI_EMUL | SI_EMUL_BX | SI_EMUL_XB => {
                let v = ext(&w, st.a) * ext(&w, st.b);
                put_e(&mut w, st.dst, v)
            }
            SI_ENEG => {
                let v = -ext(&w, st.a);
                put_e(&mut w, st.dst, v)
            }
            SI_EMBED => {
                let v = ext(&w, st.a);
                put_e(&mut w, st.dst, v)
            }
            SI_ACC_B => {
                let c = decode_ext(beta[st.b as usize]);
                sum += c * base(&w, st.a).to_extension::<GoldilocksExtension>();
            }
            SI_ACC_E => {
                let c = decode_ext(beta[st.b as usize]);
                sum += c * ext(&w, st.a);
            }
            other => panic!("unknown budgeted opcode {other}"),
        }
    }
    encode_ext(&sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint_ir::builder::IrBuilder;
    use crate::constraint_ir::device::{DeviceProgram, eval_device_program};

    type Gl = GoldilocksField;
    type Ext = GoldilocksExtension;

    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn e3(&mut self) -> [u64; 3] {
            [self.next_u64(), self.next_u64(), self.next_u64()]
        }
    }

    /// Every op, both dims, shared subexpressions across roots, a uniform
    /// root, a column root, a next-row read, mixed ops on either side.
    fn program() -> Program {
        let mut b = IrBuilder::<Gl, Ext>::new();
        let m0 = b.main(0, 0);
        let m1 = b.main(0, 1);
        let m0n = b.main(1, 0);
        let two = b.const_base(2);
        let s = b.add(m0, m1);
        let sh = b.mul(s, s); // shared by roots 0 and 3
        let r0 = b.sub(sh, m0n);
        b.emit(0, r0);
        let nm = b.neg(sh);
        let r1 = b.mul(nm, two);
        b.emit(1, r1);
        b.emit(2, m1); // a column root
        let ch = b.challenge(0);
        let ap = b.alpha_power(1);
        let a0 = b.aux(0, 0);
        let a1n = b.aux(1, 1);
        let off = b.table_offset();
        let t1 = b.mul(sh, ch); // base × ext
        let t2 = b.mul(ap, a0);
        let t3 = b.add(t1, t2);
        let t4 = b.sub(t3, off);
        let t5 = b.sub(a1n, m0); // ext − base
        let t6 = b.mul(t4, t5);
        b.emit(3, t6);
        let em = b.embed(m1);
        let ne = b.neg(a0);
        let t7 = b.add(em, ne);
        let t8 = b.sub(m0, a1n); // base − ext
        let t9 = b.add(t7, t8);
        let t10 = b.mul(t9, sh); // ext × base
        let t11 = b.add(t10, m1); // ext + base
        b.emit(4, t11);
        b.emit(5, ch); // a uniform root
        b.finish(3)
    }

    /// The reference transition sum: the device walk's evals, accumulated
    /// in root order as the composition kernel does.
    fn reference_sum(
        prog: &Program,
        main: &[Vec<u64>],
        aux: &[Vec<[u64; 3]>],
        rap: &[[u64; 3]],
        alpha: &[[u64; 3]],
        offset: [u64; 3],
        beta: &[[u64; 3]],
    ) -> [u64; 3] {
        let dev = DeviceProgram::lower(prog);
        let n = prog.roots.len();
        let mut be = vec![0u64; n];
        let mut ee = vec![[0u64; 3]; n];
        eval_device_program(&dev, main, aux, rap, alpha, offset, &mut be, &mut ee);
        let mut sum = Ext3E::zero();
        for (c, &r) in prog.roots.iter().enumerate() {
            let v = match prog.dims[r as usize] {
                Dim::Base => FpE::from_raw(be[c]).to_extension::<Ext>(),
                Dim::Ext => decode_ext(ee[c]),
            };
            sum += decode_ext(beta[c]) * v;
        }
        encode_ext(&sum)
    }

    fn frame(rng: &mut SplitMix64, mc: usize, ac: usize) -> (Vec<Vec<u64>>, Vec<Vec<[u64; 3]>>) {
        let main = (0..2)
            .map(|_| (0..mc).map(|_| rng.next_u64()).collect())
            .collect();
        let aux = (0..2)
            .map(|_| (0..ac).map(|_| rng.e3()).collect())
            .collect();
        (main, aux)
    }

    fn check_parity(prog: &Program, bp: &BudgetedProgram, seed: u64, rows: usize) {
        validate(prog, bp).expect("the lowering validates");
        let mut rng = SplitMix64(seed);
        for _ in 0..rows {
            let (main, aux) = frame(&mut rng, 4, 4);
            let rap: Vec<[u64; 3]> = (0..2).map(|_| rng.e3()).collect();
            let alpha: Vec<[u64; 3]> = (0..2).map(|_| rng.e3()).collect();
            let offset = rng.e3();
            let beta: Vec<[u64; 3]> = (0..prog.roots.len()).map(|_| rng.e3()).collect();
            let uni = bp.ext_uniform_table(&rap, &alpha, offset).unwrap();
            let got = eval_budgeted_row(bp, &main, &aux, &uni, &beta);
            let want = reference_sum(prog, &main, &aux, &rap, &alpha, offset, &beta);
            assert_eq!(got, want);
        }
    }

    #[test]
    fn every_split_matches_the_device_walk() {
        let prog = program();
        let mut lowered = 0;
        for policy in [Policy::Furthest, Policy::FurthestPerCost] {
            for nb in 1..8 {
                for ne in 1..6 {
                    if let Ok(bp) = lower_with_split(&prog, nb, ne, policy) {
                        check_parity(&prog, &bp, 0x5EED ^ (nb as u64) << 8 ^ ne as u64, 50);
                        lowered += 1;
                    }
                }
            }
        }
        assert!(lowered > 40, "{lowered} splits lowered");
    }

    #[test]
    fn the_tightest_split_recomputes_and_still_matches() {
        let prog = program();
        let roomy = lower_with_split(&prog, 32, 32, Policy::Furthest).unwrap();
        let tight = (1..8)
            .flat_map(|nb| (1..6).map(move |ne| (nb, ne)))
            .filter_map(|(nb, ne)| lower_with_split(&prog, nb, ne, Policy::Furthest).ok())
            .min_by_key(|p| p.num_words)
            .unwrap();
        assert!(tight.num_words < roomy.num_words);
        assert!(tight.stats.compute_steps >= roomy.stats.compute_steps);
        assert_eq!(roomy.stats.compute_steps, roomy.stats.distinct_nodes);
        check_parity(&prog, &tight, 7, 100);
    }

    #[test]
    fn lower_budgeted_keeps_within_the_budget() {
        let prog = program();
        for budget in [6u32, 8, 12, 16, 32, 64] {
            if let Ok(bp) = lower_budgeted(&prog, budget) {
                assert!(bp.num_words <= budget, "{} > {budget}", bp.num_words);
                check_parity(&prog, &bp, budget as u64, 20);
            }
        }
    }

    #[test]
    fn mutations_are_refused_and_change_the_sum() {
        let prog = program();
        let bp = lower_with_split(&prog, 3, 2, Policy::Furthest).unwrap();
        validate(&prog, &bp).unwrap();
        let mut rng = SplitMix64(99);
        let (main, aux) = frame(&mut rng, 4, 4);
        let rap: Vec<[u64; 3]> = (0..2).map(|_| rng.e3()).collect();
        let alpha: Vec<[u64; 3]> = (0..2).map(|_| rng.e3()).collect();
        let offset = rng.e3();
        let beta: Vec<[u64; 3]> = (0..prog.roots.len()).map(|_| rng.e3()).collect();
        let uni = bp.ext_uniform_table(&rap, &alpha, offset).unwrap();
        let good = eval_budgeted_row(&bp, &main, &aux, &uni, &beta);

        // (1) an accumulation's coefficient index.
        let mut m = bp.clone();
        let k = m.steps.iter().position(|s| s.op == SI_ACC_E).unwrap();
        m.steps[k].b = (m.steps[k].b + 1) % prog.roots.len() as u32;
        assert!(validate(&prog, &m).is_err());
        assert_ne!(eval_budgeted_row(&m, &main, &aux, &uni, &beta), good);

        // (2) a column operand's frame offset.
        let mut m = bp.clone();
        let k = m
            .steps
            .iter()
            .position(|s| s.a >> SIK_SHIFT == SIK_MAIN)
            .unwrap();
        m.steps[k].a ^= 1 << SI_COL_OFFSET_SHIFT;
        assert!(validate(&prog, &m).is_err());
        assert_ne!(eval_budgeted_row(&m, &main, &aux, &uni, &beta), good);

        // (3) a slot read after its value was overwritten: point a slot
        // operand at another word of its class that holds a different value.
        let mut caught = false;
        for k in 0..bp.steps.len() {
            let s = bp.steps[k];
            if s.a >> SIK_SHIFT != SIK_BSLOT || bp.base_words < 2 {
                continue;
            }
            let p = s.a & SIK_PAYLOAD_MASK;
            let mut m = bp.clone();
            m.steps[k].a = enc(SIK_BSLOT, (p + 1) % bp.base_words);
            if validate(&prog, &m).is_err()
                && eval_budgeted_row(&m, &main, &aux, &uni, &beta) != good
            {
                caught = true;
                break;
            }
        }
        assert!(caught, "no stale-slot mutation was refused");
    }
}
