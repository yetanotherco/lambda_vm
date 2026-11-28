//! A table's constraint part lowered in chunks of its roots, for the fused
//! zerocheck's later rounds where a launch is one walk a thread
//! (`LAMBDA_VM_ZC_CHUNKED`, [`crate::gpu_fused`]).
//!
//! Late in the rounds the cube is narrower than the card, so a launch runs one
//! thread a cube index and its time is one walk of the whole program: about
//! 0.9 ms for KECCAK_RND's 2,181 steps, about 7 ms for ECDAS's 21 k, however
//! few indices are left. Splitting the roots into chunks turns that walk into
//! many short ones side by side.
//!
//! A chunk is a run of consecutive roots whose distinct factor reads stay
//! within a cap. A root wider than the cap is split into additive pieces,
//! each retaining the same beta. A chunk owns every step its pieces reach,
//! so adding the chunks restores `C = Σ β_i·root_i`.
//!
//! One u64 a step: op (8 bits) | a (20) << 8 | b (20) << 28 | res (16) << 48.
//! A constant operand is an immediate (`MULK` / `ADDK` / `SUBK` / `KSUB`, an
//! index into `kconst`): the constraint part is base-field, so `x·k` costs
//! three base products where an extension constant costs nine. `ACC` adds
//! `β[b]·slot[a]` to the chunk's sum. The op codes MUST stay in sync with
//! `crypto/math-cuda/kernels/sumcheck.cu` (`zc_round_chunked`).

use std::collections::HashMap;

use math::field::{element::FieldElement, traits::IsField};

use crate::{fused::Constraints, program::Op};

/// The chunked code's op codes.
pub mod op {
    pub const CONST: u32 = 0;
    pub const VAR: u32 = 1;
    pub const ADD: u32 = 2;
    pub const SUB: u32 = 3;
    pub const MUL: u32 = 4;
    pub const NEG: u32 = 5;
    pub const ACC: u32 = 6;
    pub const MULK: u32 = 7;
    pub const ADDK: u32 = 8;
    pub const SUBK: u32 = 9;
    pub const KSUB: u32 = 10;
}

const FIELD: u64 = (1 << 20) - 1;
const MAX_RES: usize = (1 << 16) - 1;

/// Maximum distinct factor columns read by any production chunk.
pub const CHUNKED_LEAF_CAP: usize = 16;

/// A constraint part in chunks: the code, each chunk's `[start, end)` in it
/// (`bounds`, one more than the chunks), the base constants the immediates
/// index, and the widest chunk's live set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkedFused {
    pub code: Vec<u64>,
    pub bounds: Vec<u32>,
    pub kconst: Vec<u64>,
    pub max_slots: usize,
}

impl ChunkedFused {
    pub fn chunks(&self) -> usize {
        self.bounds.len().saturating_sub(1)
    }

    #[cfg(feature = "cuda")]
    #[doc(hidden)]
    pub fn device_program(&self) -> math_cuda::argue_fused::ChunkedProgram<'_> {
        math_cuda::argue_fused::ChunkedProgram {
            code: &self.code,
            bounds: &self.bounds,
            kconst: &self.kconst,
            max_slots: self.max_slots,
        }
    }
}

/// One step's word, or `None` when an operand does not fit its field.
pub fn encode(op: u32, a: usize, b: usize, res: usize) -> Option<u64> {
    if a as u64 > FIELD || b as u64 > FIELD || res > MAX_RES {
        return None;
    }
    Some(u64::from(op) | ((a as u64) << 8) | ((b as u64) << 28) | ((res as u64) << 48))
}

/// A step's fields: `(op, a, b, res)`.
pub fn decode(word: u64) -> (u32, usize, usize, usize) {
    (
        (word & 0xFF) as u32,
        ((word >> 8) & FIELD) as usize,
        ((word >> 28) & FIELD) as usize,
        (word >> 48) as usize,
    )
}

/// The operands a step reads, as step indices.
fn operands<B: IsField>(step: &Op<B>) -> [Option<u32>; 2] {
    match *step {
        Op::Fixed(_) | Op::Var(_) => [None, None],
        Op::Neg(a) => [Some(a), None],
        Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) => [Some(a), Some(b)],
    }
}

/// Every step `starts` reach, marked in `mark` with `stamp` (a fresh stamp a
/// call, so the array is never cleared), pushed onto `out`.
fn reach<B: IsField>(
    steps: &[Op<B>],
    starts: impl IntoIterator<Item = u32>,
    mark: &mut [u32],
    stamp: u32,
    out: &mut Vec<u32>,
) {
    let mut stack: Vec<u32> = starts.into_iter().collect();
    while let Some(s) = stack.pop() {
        if mark[s as usize] == stamp {
            continue;
        }
        mark[s as usize] = stamp;
        out.push(s);
        stack.extend(operands(&steps[s as usize]).into_iter().flatten());
    }
}

/// Column sets are kept only up to the cap. A wider child's reads are also
/// too wide for any parent that references it.
fn union_reads(reads: &[Option<Vec<u32>>], a: u32, b: u32, cap: usize) -> Option<Vec<u32>> {
    merge_reads(
        reads.get(a as usize)?.as_ref()?,
        reads.get(b as usize)?.as_ref()?,
        cap,
    )
}

fn merge_reads(left: &[u32], right: &[u32], cap: usize) -> Option<Vec<u32>> {
    let mut columns = left.to_vec();
    columns.extend_from_slice(right);
    columns.sort_unstable();
    columns.dedup();
    (columns.len() <= cap).then_some(columns)
}

struct Piece {
    step: u32,
    reads: Vec<u32>,
}

/// Group neighboring terms before distributing a product. This keeps a
/// product of wide sums from expanding to one product for every pair of terms.
fn add_piece<B: IsField>(
    steps: &mut Vec<Op<B>>,
    pieces: &mut Vec<Piece>,
    step: u32,
    reads: &[u32],
    negate: bool,
    cap: usize,
) -> Option<()> {
    if let Some(last) = pieces.last_mut()
        && let Some(reads) = merge_reads(&last.reads, reads, cap)
    {
        let sum = if negate {
            Op::Sub(last.step, step)
        } else {
            Op::Add(last.step, step)
        };
        last.step = push_step(steps, sum)?;
        last.reads = reads;
    } else {
        let step = if negate {
            push_step(steps, Op::Neg(step))?
        } else {
            step
        };
        pieces.push(Piece {
            step,
            reads: reads.to_vec(),
        });
    }
    Some(())
}

fn push_step<B: IsField>(steps: &mut Vec<Op<B>>, op: Op<B>) -> Option<u32> {
    let index = u32::try_from(steps.len()).ok()?;
    steps.push(op);
    Some(index)
}

/// Expand a wide expression into a sum of pieces within its column budget.
/// Addition and subtraction contribute their terms directly. Distribute a
/// product only when its operands together exceed the budget.
fn split_step<B: IsField>(
    root: u32,
    cap: usize,
    reads: &[Option<Vec<u32>>],
    steps: &mut Vec<Op<B>>,
    products: &mut HashMap<(u32, usize), Vec<Piece>>,
) -> Option<Vec<Piece>> {
    let fit = |step: u32, cap: usize| {
        reads[step as usize]
            .as_ref()
            .filter(|cols| cols.len() <= cap)
    };
    let mut pending = vec![(root, false)];
    let mut pieces = Vec::new();
    while let Some((step, negate)) = pending.pop() {
        if let Some(reads) = fit(step, cap) {
            add_piece(steps, &mut pieces, step, reads, negate, cap)?;
            continue;
        }
        match steps[step as usize] {
            Op::Add(a, b) => pending.extend([(b, negate), (a, negate)]),
            Op::Sub(a, b) => pending.extend([(b, !negate), (a, negate)]),
            Op::Neg(a) => pending.push((a, !negate)),
            Op::Mul(a, b) => {
                let key = (step, cap);
                if !products.contains_key(&key) {
                    let left_cap = if let Some(reads) = fit(a, cap / 2) {
                        reads.len()
                    } else if let Some(reads) = fit(b, cap / 2) {
                        cap - reads.len()
                    } else {
                        cap / 2
                    };
                    let left = split_step(a, left_cap, reads, steps, products)?;
                    let right = split_step(b, cap - left_cap, reads, steps, products)?;
                    let mut terms = Vec::new();
                    for a in &left {
                        for b in &right {
                            terms.push(Piece {
                                step: push_step(steps, Op::Mul(a.step, b.step))?,
                                reads: merge_reads(&a.reads, &b.reads, cap)?,
                            });
                        }
                    }
                    products.insert(key, terms);
                }
                for term in &products[&key] {
                    add_piece(steps, &mut pieces, term.step, &term.reads, negate, cap)?;
                }
            }
            Op::Fixed(_) | Op::Var(_) => return None,
        }
    }
    Some(pieces)
}

/// Lowers `constraints` in chunks under `leaf_cap` distinct factor reads (0 =
/// one chunk). `None` if the arithmetic cannot fit the cap or the encoding.
pub fn lower_chunked<B: IsField + 'static>(
    constraints: &Constraints<B>,
    leaf_cap: usize,
) -> Option<ChunkedFused> {
    let original_steps = constraints.steps();
    if constraints.roots().is_empty() {
        return None;
    }
    let mut steps = original_steps.to_vec();
    let mut roots = Vec::new();
    let mut reads = Vec::with_capacity(steps.len());
    for step in &steps {
        let columns = match *step {
            Op::Fixed(_) => Some(Vec::new()),
            Op::Var(factor) => Some(vec![factor]),
            Op::Neg(a) => reads.get(a as usize).cloned()?,
            Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) => union_reads(&reads, a, b, leaf_cap),
        };
        reads.push(columns);
    }
    for (&root, &selector) in constraints.roots().iter().zip(constraints.selectors()) {
        let root = if let Some(selector) = selector {
            let factor = u32::try_from(selector).ok()?;
            let term = push_step(&mut steps, Op::Var(factor))?;
            reads.push(Some(vec![factor]));
            let product = push_step(&mut steps, Op::Mul(root, term))?;
            reads.push(union_reads(&reads, root, term, leaf_cap));
            product
        } else {
            root
        };
        roots.push(root);
    }
    let mut root_pieces = Vec::new();
    let mut products = HashMap::new();
    for (beta, root) in roots.into_iter().enumerate() {
        if leaf_cap == 0 || reads.get(root as usize)?.is_some() {
            root_pieces.push((root, beta));
            continue;
        }
        root_pieces.extend(
            split_step(root, leaf_cap, &reads, &mut steps, &mut products)?
                .into_iter()
                .map(|piece| (piece.step, beta)),
        );
    }
    let roots = root_pieces;
    let n = steps.len();
    // Constants, once each, in step order.
    let mut kidx = vec![u32::MAX; n];
    let mut kconst = Vec::new();
    for (s, step) in steps.iter().enumerate() {
        if let Op::Fixed(c) = step {
            kidx[s] = kconst.len() as u32;
            kconst.push(crate::gpu_fused::base_raw(c)?);
        }
    }
    let is_const = |s: u32| matches!(steps[s as usize], Op::Fixed(_));

    // Roots into chunks, in order, by the factors each reaches.
    let mut mark = vec![0u32; n];
    let mut stamp = 0u32;
    let mut chunks: Vec<Vec<usize>> = Vec::new();
    {
        let mut current: Vec<usize> = Vec::new();
        let mut in_current = std::collections::BTreeSet::new();
        let mut reached = Vec::new();
        for (i, &(root, _)) in roots.iter().enumerate() {
            stamp += 1;
            reached.clear();
            reach(&steps, [root], &mut mark, stamp, &mut reached);
            let mut mine: Vec<u32> = reached
                .iter()
                .filter_map(|&s| match steps[s as usize] {
                    Op::Var(f) => Some(f),
                    _ => None,
                })
                .collect();
            mine.sort_unstable();
            mine.dedup();
            if leaf_cap > 0 && mine.len() > leaf_cap {
                return None;
            }
            let new = mine.iter().filter(|f| !in_current.contains(*f)).count();
            if leaf_cap > 0 && !current.is_empty() && in_current.len() + new > leaf_cap {
                chunks.push(std::mem::take(&mut current));
                in_current.clear();
            }
            in_current.extend(mine);
            current.push(i);
        }
        if !current.is_empty() {
            chunks.push(current);
        }
    }

    let mut code = Vec::new();
    let mut bounds = vec![0u32];
    let mut max_slots = 0usize;
    let mut reached = Vec::new();
    for chunk in &chunks {
        stamp += 1;
        reached.clear();
        reach(
            &steps,
            chunk.iter().map(|&i| roots[i].0),
            &mut mark,
            stamp,
            &mut reached,
        );
        reached.sort_unstable();
        let chunk_stamp = stamp;
        let in_chunk = |s: u32, mark: &[u32]| mark[s as usize] == chunk_stamp;
        // A constant needs a slot only when an op cannot take it as an
        // immediate: both operands constant, a negation, or a root.
        let mut needs = vec![false; n];
        for &s in &reached {
            match steps[s as usize] {
                Op::Fixed(_) => {}
                Op::Var(_) => needs[s as usize] = true,
                Op::Neg(a) => {
                    needs[s as usize] = true;
                    if is_const(a) {
                        needs[a as usize] = true;
                    }
                }
                Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) => {
                    needs[s as usize] = true;
                    if is_const(a) && is_const(b) {
                        needs[a as usize] = true;
                        needs[b as usize] = true;
                    }
                }
            }
        }
        for &i in chunk {
            needs[roots[i].0 as usize] = true;
        }
        // Items in step order, each root's ACCs right after it; last reads by
        // item position.
        enum Item {
            Step(u32),
            Acc(usize),
        }
        let mut accs_of: std::collections::BTreeMap<u32, Vec<usize>> = Default::default();
        for &i in chunk {
            accs_of.entry(roots[i].0).or_default().push(i);
        }
        let mut items = Vec::with_capacity(reached.len() + chunk.len());
        for &s in &reached {
            debug_assert!(in_chunk(s, &mark));
            if needs[s as usize] {
                items.push(Item::Step(s));
            }
            if let Some(accs) = accs_of.get(&s) {
                items.extend(accs.iter().map(|&i| Item::Acc(i)));
            }
        }
        let mut last = vec![usize::MAX; n];
        for (at, item) in items.iter().enumerate() {
            match *item {
                Item::Acc(i) => last[roots[i].0 as usize] = at,
                Item::Step(s) => {
                    for x in operands(&steps[s as usize]).into_iter().flatten() {
                        if needs[x as usize] {
                            last[x as usize] = at;
                        }
                    }
                }
            }
        }
        let mut frees: Vec<Vec<u32>> = vec![Vec::new(); items.len()];
        for &s in &reached {
            let l = last[s as usize];
            if needs[s as usize] && l != usize::MAX {
                frees[l].push(s);
            }
        }
        let mut slot = vec![usize::MAX; n];
        let mut free: Vec<usize> = Vec::new();
        let mut used = 0usize;
        for (at, item) in items.iter().enumerate() {
            match *item {
                Item::Acc(i) => {
                    let (root, beta) = roots[i];
                    code.push(encode(op::ACC, slot[root as usize], beta, 0)?);
                    for &x in &frees[at] {
                        free.push(slot[x as usize]);
                    }
                }
                Item::Step(s) => {
                    let (tag, a, b) = match steps[s as usize] {
                        Op::Fixed(_) => (op::CONST, kidx[s as usize] as usize, 0),
                        Op::Var(f) => (op::VAR, f as usize, 0),
                        Op::Neg(a) => (op::NEG, slot[a as usize], 0),
                        Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) => {
                            let (plain, kright, kleft) = match steps[s as usize] {
                                Op::Add(..) => (op::ADD, op::ADDK, op::ADDK),
                                Op::Sub(..) => (op::SUB, op::SUBK, op::KSUB),
                                _ => (op::MUL, op::MULK, op::MULK),
                            };
                            match (is_const(a), is_const(b)) {
                                (true, true) | (false, false) => {
                                    (plain, slot[a as usize], slot[b as usize])
                                }
                                (false, true) => {
                                    (kright, slot[a as usize], kidx[b as usize] as usize)
                                }
                                (true, false) => {
                                    (kleft, slot[b as usize], kidx[a as usize] as usize)
                                }
                            }
                        }
                    };
                    // The operands' slots free before the result takes one: the
                    // kernel reads a step's operands before it writes it.
                    for &x in &frees[at] {
                        if x != s {
                            free.push(slot[x as usize]);
                        }
                    }
                    let res = free.pop().unwrap_or_else(|| {
                        used += 1;
                        used - 1
                    });
                    slot[s as usize] = res;
                    code.push(encode(tag, a, b, res)?);
                }
            }
        }
        if code.len() > u32::MAX as usize {
            return None;
        }
        bounds.push(code.len() as u32);
        max_slots = max_slots.max(used.max(1));
    }
    if max_slots > MAX_RES {
        return None;
    }
    Some(ChunkedFused {
        code,
        bounds,
        kconst,
        max_slots,
    })
}

/// The chunks' sum at one point, walked as the kernel walks it: every chunk's
/// `Σ β_i·root_i` over `values` (factor slot → value), added up. A reference
/// for tests: it must equal the constraint part's `Σ β_i·root_i` there.
pub fn eval_chunked<E: IsField>(
    chunked: &ChunkedFused,
    values: &[FieldElement<E>],
    betas: &[FieldElement<E>],
) -> Option<FieldElement<E>> {
    let mut total = FieldElement::<E>::zero();
    for c in 0..chunked.chunks() {
        let mut slots = vec![FieldElement::<E>::zero(); chunked.max_slots];
        let mut sum = FieldElement::<E>::zero();
        let (start, end) = (chunked.bounds[c] as usize, chunked.bounds[c + 1] as usize);
        for &word in &chunked.code[start..end] {
            let (tag, a, b, res) = decode(word);
            let k = |i: usize| chunked.kconst.get(i).map(|&v| FieldElement::<E>::from(v));
            let value = match tag {
                op::ACC => {
                    sum += betas.get(b)? * slots.get(a)?;
                    continue;
                }
                op::CONST => k(a)?,
                op::VAR => values.get(a)?.clone(),
                op::ADD => slots.get(a)? + slots.get(b)?,
                op::SUB => slots.get(a)? - slots.get(b)?,
                op::MUL => slots.get(a)? * slots.get(b)?,
                op::NEG => -slots.get(a)?,
                op::MULK => slots.get(a)? * k(b)?,
                op::ADDK => slots.get(a)? + k(b)?,
                op::SUBK => slots.get(a)? - k(b)?,
                op::KSUB => k(b)? - slots.get(a)?,
                _ => return None,
            };
            *slots.get_mut(res)? = value;
        }
        total += sum;
    }
    Some(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Gl;

    type FB = FieldElement<Gl>;
    type FE = FieldElement<Ext3>;

    fn next(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    fn ext(seed: &mut u64) -> FE {
        FE::new([next(seed).into(), next(seed).into(), next(seed).into()])
    }

    /// A random DAG over `factors` factors with `roots` roots that share
    /// subexpressions and read constants on either side.
    fn random_constraints(seed: u64, factors: u32, steps: usize, roots: usize) -> Constraints<Gl> {
        let mut s = seed | 1;
        let mut ops: Vec<Op<Gl>> = Vec::new();
        for f in 0..factors {
            ops.push(Op::Var(f));
        }
        ops.push(Op::Fixed(FB::from(7u64)));
        ops.push(Op::Fixed(FB::from(0xFFFF_FFFF_0000_0000u64)));
        while ops.len() < steps {
            let len = ops.len() as u64;
            let a = (next(&mut s) % len) as u32;
            let b = (next(&mut s) % len) as u32;
            ops.push(match next(&mut s) % 4 {
                0 => Op::Add(a, b),
                1 => Op::Sub(a, b),
                2 => Op::Mul(a, b),
                _ => Op::Neg(a),
            });
        }
        let root_steps: Vec<u32> = (0..roots)
            .map(|_| (steps as u64 - 1 - next(&mut s) % (steps as u64 / 2)) as u32)
            .collect();
        let selectors = vec![None; root_steps.len()];
        Constraints::from_extension::<Ext3>(
            &ops.iter()
                .map(|op| match op {
                    Op::Fixed(c) => Op::Fixed((*c).to_extension::<Ext3>()),
                    Op::Var(i) => Op::Var(*i),
                    Op::Add(a, b) => Op::Add(*a, *b),
                    Op::Sub(a, b) => Op::Sub(*a, *b),
                    Op::Mul(a, b) => Op::Mul(*a, *b),
                    Op::Neg(a) => Op::Neg(*a),
                })
                .collect::<Vec<_>>(),
            &root_steps,
            &selectors,
            3,
        )
        .expect("a base-field program")
    }

    /// `Σ β_i·root_i` straight from the steps.
    fn direct(constraints: &Constraints<Gl>, values: &[FE], betas: &[FE]) -> FE {
        let mut out: Vec<FE> = Vec::with_capacity(constraints.steps().len());
        for step in constraints.steps() {
            let v = match *step {
                Op::Fixed(ref c) => (*c).to_extension::<Ext3>(),
                Op::Var(i) => values[i as usize],
                Op::Add(a, b) => &out[a as usize] + &out[b as usize],
                Op::Sub(a, b) => &out[a as usize] - &out[b as usize],
                Op::Mul(a, b) => &out[a as usize] * &out[b as usize],
                Op::Neg(a) => -&out[a as usize],
            };
            out.push(v);
        }
        constraints
            .roots()
            .iter()
            .zip(betas)
            .fold(FE::zero(), |acc, (&r, beta)| acc + beta * &out[r as usize])
    }

    /// At every supported cap, the chunks add up to the constraint part.
    #[test]
    fn the_chunks_add_up_to_the_constraint_part() {
        for (seed, factors, steps, roots) in [
            (1u64, 12u32, 80usize, 9usize),
            (2, 40, 600, 60),
            (3, 5, 30, 1),
        ] {
            let constraints = random_constraints(seed, factors, steps, roots);
            let mut s = seed * 31 + 5;
            let values: Vec<FE> = (0..factors).map(|_| ext(&mut s)).collect();
            let betas: Vec<FE> = (0..roots).map(|_| ext(&mut s)).collect();
            let want = direct(&constraints, &values, &betas);
            for cap in [0usize, 64, 16] {
                let chunked = lower_chunked(&constraints, cap).expect("lowers");
                assert_eq!(
                    eval_chunked(&chunked, &values, &betas),
                    Some(want),
                    "seed {seed} cap {cap}: the chunks' sum"
                );
                if cap == 0 {
                    assert_eq!(chunked.chunks(), 1, "cap 0 is one chunk");
                }
            }
        }
    }

    /// The reference evaluator is not vacuous: one changed op changes the sum.
    #[test]
    fn a_changed_op_changes_the_sum() {
        let constraints = random_constraints(9, 20, 300, 25);
        let mut s = 77u64;
        let values: Vec<FE> = (0..20).map(|_| ext(&mut s)).collect();
        let betas: Vec<FE> = (0..25).map(|_| ext(&mut s)).collect();
        let want = direct(&constraints, &values, &betas);
        let mut chunked = lower_chunked(&constraints, 16).expect("lowers");
        let at = chunked
            .code
            .iter()
            .position(|&w| matches!(decode(w).0, op::MUL | op::SUB))
            .expect("a MUL or SUB");
        let (_, a, b, res) = decode(chunked.code[at]);
        chunked.code[at] = encode(op::ADD, a, b, res).unwrap();
        assert_ne!(eval_chunked(&chunked, &values, &betas), Some(want));
    }

    /// A selector counts toward the read cap. Encoding refuses what overflows.
    #[test]
    fn what_it_cannot_lower_it_declines() {
        let selected = Constraints::<Gl>::from_extension::<Ext3>(
            &[Op::Var(0), Op::Var(1), Op::Add(0, 1)],
            &[2],
            &[Some(2)],
            3,
        )
        .unwrap();
        let chunks = lower_chunked(&selected, 2).expect("distribute the selector");
        assert_eq!(chunks.chunks(), 2);
        let values = [FE::from(2u64), FE::from(3u64), FE::from(4u64)];
        assert_eq!(
            eval_chunked(&chunks, &values, &[FE::from(5u64)]),
            Some(FE::from(100u64))
        );
        assert!(lower_chunked(&selected, 1).is_none());
        assert!(encode(op::ADD, 1 << 20, 0, 0).is_none());
        assert!(encode(op::ADD, 0, 0, 1 << 16).is_none());
        let w = encode(op::KSUB, 123_456, 654_321, 4_321).unwrap();
        assert_eq!(decode(w), (op::KSUB, 123_456, 654_321, 4_321));
    }
}
