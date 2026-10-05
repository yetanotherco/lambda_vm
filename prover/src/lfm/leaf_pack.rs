//! The block tree's PADDED partition (partition v3, `NOEPOCH_PARTITION_MODEL=3`;
//! I-PADLEAF §4): leaves packed by their padded LFM chip heights.
//!
//! # Why
//!
//! v2 ([`super::block_leaf::partition_by_rule`]) fills leaves by closed-form
//! permutations — `LFM_HASH`'s rows — alone. Every other chip of a leaf is
//! padded to a power of two too, and its rows depend on WHICH instances the
//! leaf holds. v2's least-loaded rule in AIR order hands each leaf a run of the
//! same kinds, so the chips' loads are correlated rather than mixed: at the
//! median block 36 of 54 leaves carry `LFM_LANES` at 2^19 while 18 others
//! carry `LFM_BALU` near 2^18 (ULTRA mc12). Mixing them keeps leaves under
//! their steps at the same leaf count.
//!
//! # The model
//!
//! A leaf's real rows per chip are the front (the statement, Phase A over
//! every root, the publishes), the carrier's COMMIT-bus target on leaf 0, and
//! the sum of its instances' forks. Each term is MEASURED with the production
//! leaf emitter over a probe list ([`super::block_leaf::probe_leaf`]), once per
//! distinct instance shape: the front is `probe(i) + probe(j) − probe(i, j)`,
//! an instance its probe less the front. `LFM_CONST` is pooled by value, so a
//! fork adds almost none of its constants; it is modelled as the largest
//! probe's pool. Nothing here estimates a row.
//!
//! # The packer
//!
//! Start from v2's partition at v2's leaf count, keep the rule's seeds where
//! the rule puts them, and improve by local search: move one instance to
//! another leaf, or swap two, whenever that lowers the modelled padded cells,
//! or keeps them and lowers a concave potential over each chip's excess above
//! its lower step (`Σ w · √(excess · half)`). The potential is what lets the
//! search cross plateaus of the step function: it prefers concentrating a
//! chip's excess in few leaves, so the others can drop under their step. No
//! leaf's `LFM_HASH` rows pass [`LEAF_PERMS_CAP`]. Integer arithmetic and a
//! fixed visiting order, so prover and verifier derive the same lists.
//!
//! # What it cannot change
//!
//! The partition stays a pure function of the ELF, the options, the shape and
//! the model's version: the verifier derives it and never reads it from a
//! proof. v3 returns v2's partition unless its own models strictly fewer cells,
//! and every list is validated by [`BlockPartition::new`].

use std::collections::BTreeMap;

use super::airs::{HASH_SLOT, LFM_CHIP_NAMES, NUM_LFM_CHIPS};
use super::block_leaf::{BlockPartition, probe_leaf, seed_leaves};
use super::block_plan::{BlockTreePlan, LEAF_PERMS_CAP, PlannedInstance, partition_for};
use super::chunking::HashChunking;
use super::compiler::LfmProgram;
use super::layout::padded_rows;

/// Improvement passes over every leaf pair before the search stops anyway.
const MAX_ROUNDS: usize = 64;

/// Each chip's real rows and cells a row in `program`, in census order, a
/// split `LFM_HASH` summed into one entry.
pub(crate) fn chip_rows(program: &LfmProgram) -> Vec<(&'static str, u64, u64)> {
    let mut out: Vec<(&'static str, u64, u64)> = Vec::new();
    for c in super::airs::lfm_chip_census_with_hasher(program, crate::hash_pin::BLOCK_HASHER) {
        let width = (c.main_cols + 3 * c.aux_cols) as u64;
        match out.iter_mut().find(|(name, _, _)| *name == c.name) {
            Some(entry) => entry.1 += c.real_rows,
            None => out.push((c.name, c.real_rows, width)),
        }
    }
    out
}

/// What makes two instances' forks the same program text up to constants: the
/// AIR kind, whether a preprocessed root is absorbed, the challenge shape
/// (its fork index aside) and the verification shape and lowering.
fn shape_key(inst: &PlannedInstance) -> String {
    let kind = inst.name.split(['[', ':']).next().unwrap_or(&inst.name);
    let mut challenge = inst.challenge.clone();
    challenge.index = 0;
    format!(
        "{kind}|{}|{challenge:?}|{:?}|{:?}",
        inst.precomputed_root.is_some(),
        inst.verify,
        inst.analysis.report()
    )
}

/// Rows per chip class, in a model's chip order; the classes a leaf lacks stay
/// zero (and have no width).
type Rows = [u64; NUM_LFM_CHIPS];

/// A block's leaves as rows per LFM chip: the measured front, carrier term and
/// per-shape forks, and each chip's cells a row.
pub struct ChipModel {
    names: Vec<&'static str>,
    widths: Rows,
    /// `LFM_HASH`'s position in `names`.
    hash: usize,
    /// Chips some fork adds rows to, `LFM_HASH` aside: the potential's terms.
    stepped: Vec<usize>,
    front: Rows,
    carrier: Rows,
    /// Each instance's shape, an index into `forks`.
    shape_of: Vec<usize>,
    forks: Vec<Rows>,
}

impl ChipModel {
    /// Measure the model of `plan`'s instances: one probe per distinct
    /// instance shape, three for the front and one for the carrier term.
    pub fn probe(plan: &BlockTreePlan) -> Result<Self, String> {
        let n = plan.num_instances();
        if n < 3 {
            return Err(format!("{n} instances: the front's probe pair needs three"));
        }
        let mut reps: Vec<usize> = Vec::new();
        let mut shape_of: Vec<usize> = Vec::with_capacity(n);
        let mut seen: BTreeMap<String, usize> = BTreeMap::new();
        for (i, inst) in plan.instances().iter().enumerate() {
            let next = reps.len();
            let r = *seen.entry(shape_key(inst)).or_insert(next);
            if r == next {
                reps.push(i);
            }
            shape_of.push(r);
        }
        // The probe pair: the two cheapest instances, ties by index.
        let costs = plan.costs();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| (costs[i], i));
        let (i, j) = (order[0], order[1]);

        let mut jobs: Vec<(Vec<usize>, bool)> = reps.iter().map(|&r| (vec![r], false)).collect();
        jobs.push((vec![i], false));
        jobs.push((vec![j], false));
        jobs.push((vec![i, j], false));
        jobs.push((vec![i], true));
        let run = |(list, carries): &(Vec<usize>, bool)| -> Result<_, String> {
            let rest: Vec<usize> = (0..n).filter(|x| !list.contains(x)).collect();
            let partition = BlockPartition::new(vec![list.clone(), rest], n)?;
            Ok(chip_rows(&probe_leaf(plan, &partition, *carries)))
        };
        #[cfg(feature = "parallel")]
        let probes: Vec<Vec<(&'static str, u64, u64)>> = {
            use rayon::prelude::*;
            jobs.par_iter().map(run).collect::<Result<_, String>>()?
        };
        #[cfg(not(feature = "parallel"))]
        let probes: Vec<Vec<(&'static str, u64, u64)>> =
            jobs.iter().map(run).collect::<Result<_, String>>()?;

        let names: Vec<&'static str> = probes[0].iter().map(|c| c.0).collect();
        if names.len() > NUM_LFM_CHIPS {
            return Err(format!("{} chip classes in a leaf", names.len()));
        }
        let mut widths: Rows = [0; NUM_LFM_CHIPS];
        for (w, c) in widths.iter_mut().zip(&probes[0]) {
            *w = c.2;
        }
        if let Some(k) = probes.iter().position(|p| {
            p.iter().map(|c| c.0).ne(names.iter().copied())
                || p.iter()
                    .map(|c| c.2)
                    .ne(widths[..names.len()].iter().copied())
        }) {
            return Err(format!("probe {k} has another chip set than probe 0"));
        }
        let position = |name: &str| names.iter().position(|&n| n == name);
        let hash = position(LFM_CHIP_NAMES[HASH_SLOT]).ok_or("a leaf has no LFM_HASH")?;
        let pooled = position(LFM_CHIP_NAMES[0]);
        let rows = |p: &[(&'static str, u64, u64)]| -> Rows {
            let mut r: Rows = [0; NUM_LFM_CHIPS];
            for (slot, c) in r.iter_mut().zip(p) {
                *slot = c.1;
            }
            r
        };
        let m = reps.len();
        let (pi, pj, pij, carried) = (
            rows(&probes[m]),
            rows(&probes[m + 1]),
            rows(&probes[m + 2]),
            rows(&probes[m + 3]),
        );
        let mut front: Rows = [0; NUM_LFM_CHIPS];
        let mut carrier: Rows = [0; NUM_LFM_CHIPS];
        for c in 0..names.len() {
            front[c] = (pi[c] + pj[c]).saturating_sub(pij[c]);
            carrier[c] = carried[c].saturating_sub(pi[c]);
        }
        let mut forks: Vec<Rows> = probes[..m]
            .iter()
            .map(|p| {
                let r = rows(p);
                let mut fork: Rows = [0; NUM_LFM_CHIPS];
                for c in 0..names.len() {
                    fork[c] = r[c].saturating_sub(front[c]);
                }
                fork
            })
            .collect();
        if let Some(c) = pooled {
            front[c] = probes.iter().map(|p| p[c].1).max().unwrap_or(0);
            for fork in &mut forks {
                fork[c] = 0;
            }
        }
        let stepped: Vec<usize> = (0..names.len())
            .filter(|&c| c != hash && forks.iter().any(|f| f[c] > 0))
            .collect();
        Ok(Self {
            names,
            widths,
            hash,
            stepped,
            front,
            carrier,
            shape_of,
            forks,
        })
    }

    pub fn names(&self) -> &[&'static str] {
        &self.names
    }

    pub fn widths(&self) -> &[u64] {
        &self.widths[..self.names.len()]
    }

    pub fn front(&self) -> &[u64] {
        &self.front[..self.names.len()]
    }

    pub fn carrier(&self) -> &[u64] {
        &self.carrier[..self.names.len()]
    }

    /// Instance `i`'s fork, rows per chip.
    pub fn instance(&self, i: usize) -> &[u64] {
        &self.forks[self.shape_of[i]][..self.names.len()]
    }

    /// Distinct instance shapes the model probed.
    pub fn num_shapes(&self) -> usize {
        self.forks.len()
    }

    fn rows_of(&self, list: &[usize], carries: bool) -> Rows {
        let mut rows = self.front;
        if carries {
            add(&mut rows, &self.carrier);
        }
        for &i in list {
            add(&mut rows, &self.forks[self.shape_of[i]]);
        }
        rows
    }

    /// A leaf over `list` as rows per chip: the front, the carrier term when
    /// it `carries`, and its instances' forks.
    pub fn leaf_rows(&self, list: &[usize], carries: bool) -> Vec<u64> {
        self.rows_of(list, carries)[..self.names.len()].to_vec()
    }

    /// `LFM_HASH`'s real rows in `rows` (as [`Self::leaf_rows`]).
    pub fn hash_rows(&self, rows: &[u64]) -> u64 {
        rows[self.hash]
    }

    /// The cells a leaf of `rows` (as [`Self::leaf_rows`]) commits: each chip
    /// at its padded height, a split `LFM_HASH` ([`HashChunking::for_rows`],
    /// the pipeline's default rule) as its two chunks.
    pub fn padded_cells(&self, rows: &[u64]) -> u64 {
        (0..self.names.len())
            .map(|c| self.padded_chip(c, rows[c]) * self.widths[c])
            .sum()
    }

    fn padded_chip(&self, c: usize, rows: u64) -> u64 {
        let rows = rows as usize;
        let padded = if c == self.hash {
            let split = HashChunking::for_rows(rows);
            (0..split.chunk_count(rows))
                .map(|k| padded_rows(split.chunk_range(rows, k).len()))
                .sum()
        } else {
            padded_rows(rows)
        };
        padded as u64
    }

    /// The search's tie-breaker: `Σ w · √(excess · half)` over the stepped
    /// chips, `half` the power of two below a chip's padded height and
    /// `excess` its rows above `half`. Concave in the excess, so moving rows
    /// from a leaf barely over its step to one far over lowers it.
    fn potential(&self, rows: &Rows) -> u64 {
        self.stepped
            .iter()
            .map(|&c| {
                let half = self.padded_chip(c, rows[c]) / 2;
                match rows[c].checked_sub(half) {
                    Some(excess) if half >= 4 && excess > 0 => {
                        self.widths[c] * (excess as u128 * half as u128).isqrt() as u64
                    }
                    _ => 0,
                }
            })
            .sum()
    }

    /// `(cells, potential)` of `rows`.
    fn score(&self, rows: &Rows) -> (u64, u64) {
        (self.padded_cells(rows), self.potential(rows))
    }
}

fn add(acc: &mut Rows, term: &Rows) {
    for (a, t) in acc.iter_mut().zip(term) {
        *a += t;
    }
}

fn sub(acc: &mut Rows, term: &Rows) {
    for (a, t) in acc.iter_mut().zip(term) {
        *a -= t;
    }
}

/// One leaf during the search: its list (in visiting order), its modelled rows
/// and their score.
struct Leaf {
    list: Vec<usize>,
    rows: Rows,
    cells: u64,
    potential: u64,
}

impl Leaf {
    fn new(model: &ChipModel, list: Vec<usize>, carries: bool) -> Self {
        let rows = model.rows_of(&list, carries);
        let (cells, potential) = model.score(&rows);
        Self {
            list,
            rows,
            cells,
            potential,
        }
    }

    fn set(&mut self, rows: Rows, (cells, potential): (u64, u64)) {
        (self.rows, self.cells, self.potential) = (rows, cells, potential);
    }
}

/// Whether two leaves rescored as `sa`, `sb` beat their present scores: fewer
/// cells, or as many and a lower potential.
fn improves(a: &Leaf, b: &Leaf, sa: (u64, u64), sb: (u64, u64)) -> bool {
    (sa.0 + sb.0, sa.1 + sb.1) < (a.cells + b.cells, a.potential + b.potential)
}

/// A swap found: `y`'s position in its leaf, and both leaves' new rows and
/// scores.
type Swap = (usize, Rows, Rows, (u64, u64), (u64, u64));

/// Two distinct leaves, mutably.
fn pair(leaves: &mut [Leaf], a: usize, b: usize) -> (&mut Leaf, &mut Leaf) {
    debug_assert_ne!(a, b);
    if a < b {
        let (lo, hi) = leaves.split_at_mut(b);
        (&mut lo[a], &mut hi[0])
    } else {
        let (lo, hi) = leaves.split_at_mut(a);
        (&mut hi[0], &mut lo[b])
    }
}

/// The local search from `start` (the module docs): moves, then swaps, over
/// every ordered leaf pair, until a pass improves nothing. `seeded` instances
/// never move; leaf `carrier` carries the COMMIT-bus target.
pub(crate) fn pack(
    model: &ChipModel,
    start: &[Vec<usize>],
    seeded: &[bool],
    carrier: usize,
) -> Vec<Vec<usize>> {
    let k = start.len();
    let mut leaves: Vec<Leaf> = start
        .iter()
        .enumerate()
        .map(|(l, list)| Leaf::new(model, list.clone(), l == carrier))
        .collect();
    let cap = LEAF_PERMS_CAP as u64;
    let hash = model.hash;
    for _ in 0..MAX_ROUNDS {
        let mut improved = false;
        for a in 0..k {
            for b in 0..k {
                if a == b {
                    continue;
                }
                let (la, lb) = pair(&mut leaves, a, b);
                // Moves a → b. A shape that does not improve is not retried
                // until something changes.
                let mut tried: Vec<usize> = Vec::new();
                let mut at = 0;
                while at < la.list.len() {
                    let x = la.list[at];
                    let shape = model.shape_of[x];
                    if seeded[x] || tried.contains(&shape) {
                        at += 1;
                        continue;
                    }
                    let v = &model.forks[shape];
                    let (mut ra, mut rb) = (la.rows, lb.rows);
                    sub(&mut ra, v);
                    add(&mut rb, v);
                    if rb[hash] <= cap {
                        let (sa, sb) = (model.score(&ra), model.score(&rb));
                        if improves(la, lb, sa, sb) {
                            la.list.remove(at);
                            lb.list.push(x);
                            la.set(ra, sa);
                            lb.set(rb, sb);
                            tried.clear();
                            improved = true;
                            continue;
                        }
                    }
                    tried.push(shape);
                    at += 1;
                }
                if b < a {
                    continue;
                }
                // Swaps x ∈ a ↔ y ∈ b: for each x, the best improving y.
                let mut at = 0;
                while at < la.list.len() {
                    let x = la.list[at];
                    if seeded[x] {
                        at += 1;
                        continue;
                    }
                    let sx = model.shape_of[x];
                    let vx = &model.forks[sx];
                    let mut tried: Vec<usize> = vec![sx];
                    let mut best: Option<Swap> = None;
                    for (bt, &y) in lb.list.iter().enumerate() {
                        let sy = model.shape_of[y];
                        if seeded[y] || tried.contains(&sy) {
                            continue;
                        }
                        tried.push(sy);
                        let vy = &model.forks[sy];
                        let (mut ra, mut rb) = (la.rows, lb.rows);
                        sub(&mut ra, vx);
                        add(&mut ra, vy);
                        sub(&mut rb, vy);
                        add(&mut rb, vx);
                        if ra[hash] > cap || rb[hash] > cap {
                            continue;
                        }
                        let (sa, sb) = (model.score(&ra), model.score(&rb));
                        let better = match &best {
                            None => improves(la, lb, sa, sb),
                            Some((_, _, _, ba, bb)) => {
                                (sa.0 + sb.0, sa.1 + sb.1) < (ba.0 + bb.0, ba.1 + bb.1)
                            }
                        };
                        if better {
                            best = Some((bt, ra, rb, sa, sb));
                        }
                    }
                    match best {
                        Some((bt, ra, rb, sa, sb)) => {
                            let y = lb.list.remove(bt);
                            la.list.remove(at);
                            la.list.push(y);
                            lb.list.push(x);
                            la.set(ra, sa);
                            lb.set(rb, sb);
                            improved = true;
                        }
                        None => at += 1,
                    }
                }
            }
        }
        if !improved {
            break;
        }
    }
    leaves.into_iter().map(|leaf| leaf.list).collect()
}

/// Modelled padded cells of a partition's leaves.
pub fn partition_cells(model: &ChipModel, lists: &[Vec<usize>], carrier: usize) -> u64 {
    lists
        .iter()
        .enumerate()
        .map(|(l, list)| model.padded_cells(&model.leaf_rows(list, l == carrier)))
        .sum()
}

/// The search's second start: the rule's seeds, then every other instance —
/// shapes with the most `LFM_HASH` rows first, a shape's instances together —
/// on the leaf with the fewest `LFM_HASH` rows so far (ties: the lower leaf).
/// Each shape lands spread over the leaves, so every chip starts balanced.
pub(crate) fn spread_start(
    model: &ChipModel,
    seeds: &[Vec<usize>],
    seeded: &[bool],
    carrier: usize,
) -> Vec<Vec<usize>> {
    let mut lists: Vec<Vec<usize>> = seeds.to_vec();
    let mut hash: Vec<u64> = lists
        .iter()
        .enumerate()
        .map(|(l, list)| model.rows_of(list, l == carrier)[model.hash])
        .collect();
    let mut rest: Vec<usize> = (0..seeded.len()).filter(|&i| !seeded[i]).collect();
    rest.sort_by_key(|&i| {
        let shape = model.shape_of[i];
        (std::cmp::Reverse(model.forks[shape][model.hash]), shape, i)
    });
    for i in rest {
        let l = (0..lists.len())
            .min_by_key(|&l| (hash[l], l))
            .expect("at least one leaf");
        lists[l].push(i);
        hash[l] += model.forks[model.shape_of[i]][model.hash];
    }
    lists
}

/// The padded partition of `plan`'s instances (partition v3): [`pack`] from
/// their v2 partition and from [`spread_start`], at v2's leaf count. The
/// packed lists that model the fewest cells, provided no leaf's `LFM_HASH`
/// rows pass [`LEAF_PERMS_CAP`] (or v2's own heaviest leaf, if heavier); v2's
/// partition itself unless one models strictly fewer cells than it.
pub fn padded_partition(plan: &BlockTreePlan) -> Result<BlockPartition, String> {
    let names: Vec<&str> = plan.instances().iter().map(|i| i.name.as_str()).collect();
    let v2 = &partition_for(&names, &plan.costs())?;
    let k = v2.num_leaves();
    if k < 2 || plan.num_instances() < 3 {
        return Ok(v2.clone());
    }
    let model = ChipModel::probe(plan)?;
    let (seeds, seeded) = seed_leaves(&names, k);
    let carrier = plan.carrier();
    let heaviest = |lists: &[Vec<usize>]| {
        lists
            .iter()
            .enumerate()
            .map(|(l, list)| model.rows_of(list, l == carrier)[model.hash])
            .max()
            .unwrap_or(0)
    };
    let bound = heaviest(v2.leaves()).max(LEAF_PERMS_CAP as u64);
    let from_v2 = || pack(&model, v2.leaves(), &seeded, carrier);
    let from_spread = || {
        pack(
            &model,
            &spread_start(&model, &seeds, &seeded, carrier),
            &seeded,
            carrier,
        )
    };
    #[cfg(feature = "parallel")]
    let (a, b) = rayon::join(from_v2, from_spread);
    #[cfg(not(feature = "parallel"))]
    let (a, b) = (from_v2(), from_spread());
    let mut best = (partition_cells(&model, v2.leaves(), carrier), None);
    for lists in [a, b] {
        let cells = partition_cells(&model, &lists, carrier);
        if cells < best.0 && heaviest(&lists) <= bound {
            best = (cells, Some(lists));
        }
    }
    match best.1 {
        Some(lists) => BlockPartition::new(lists, plan.num_instances()),
        None => Ok(v2.clone()),
    }
}
