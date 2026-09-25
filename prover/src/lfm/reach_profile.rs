//! ★ How much of an LFM program could run at once — the dependency structure,
//! measured rather than argued.
//!
//! # Why this exists
//!
//! Lane E measured the executor at **83% one software RPX permutation per
//! `Instr::Hash`** (644,250 of a wrap's 5,388,182 instructions, at 2,356 ns a
//! permutation on Zen 5). That makes "run the permutations in parallel" the
//! largest remaining lever on the host — and the case for it so far rests on
//! *reading the emitters*: the per-table legs are transcript-forked, the
//! per-query blocks are contiguous, so the hash DAG "should be" ~10⁴ wide and
//! ~60 deep.
//!
//! ⛔ **That is an argument, not a number, and the campaign's own miss list is
//! mostly arguments that had no row for something the code does.** This module
//! replaces it with a measurement taken from the program itself, before anyone
//! writes a parallel executor.
//!
//! # What it computes, and why the answer is exact rather than heuristic
//!
//! Addresses are dense, assigned in emission order, written once, and every
//! operand address is strictly below its destination (`instr.rs`), so the
//! program IS a topologically sorted DAG and one forward pass suffices.
//!
//! For each instruction the pass carries `hash_depth` = the length of the
//! longest chain of *dependent `Instr::Hash` instructions* ending there. Group
//! the hash instructions by that number and the groups have a property no
//! partition heuristic can claim:
//!
//! > **Two hash instructions at the same level cannot depend on each other.**
//! > If `j` depended on `i` by any path, `hash_depth[j] ≥ hash_depth[i] + 1`,
//! > because `i` is itself a hash. So the levels ARE the available parallelism,
//! > with no segmentation, no guessing, and no soundness argument to make.
//!
//! From the level sizes, the ideal wall at `W` parallel workers is
//! `Σ_level ceil(size / W)` permutation-steps — the list-scheduling bound for
//! unit-cost tasks. [`ReachProfile::describe`] prints that ladder, which is the
//! reading that prices the lever at the worker counts a box can actually run.
//!
//! ⚠ **What the bound is NOT.** It counts permutation STEPS and charges nothing
//! for the memory traffic, the record appends, or the scheduling itself; it
//! assumes a level's work can be handed out freely. It is therefore a FLOOR on
//! a parallel executor's hash phase and an upper bound on the speedup — the
//! honest use is "this is the most the lever could ever be worth", and a
//! measured arm still has to earn it.

use std::collections::BTreeMap;

use super::compiler::LfmProgram;
use super::instr::Instr;

/// The dependency structure of one program, and the const-pool hazard's size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReachProfile {
    pub instrs: usize,
    pub num_addrs: u64,
    /// `Instr::Hash` count — the permutations the executor runs.
    pub hash_rows: usize,
    /// Longest chain of DEPENDENT hash instructions. The serial floor.
    pub hash_depth: usize,
    /// Longest chain of dependent instructions of any kind.
    pub instr_depth: usize,
    /// `hash_levels[k]` = hash instructions at hash-depth `k + 1`. Every entry
    /// is a set of mutually independent permutations.
    pub hash_levels: Vec<u32>,
    /// `Instr::Const` count — the globally interned pool, and the one thing
    /// that makes a contiguous partition unsafe (`builder.rs` interns across
    /// the whole program, so a constant first emitted inside query 5 is read by
    /// query 7).
    pub consts: usize,
    /// Reads whose writer is an `Instr::Const`: the cross-segment edges a
    /// const-hoisting pre-pass removes.
    pub const_reads: u64,
    /// Reads whose address no instruction writes. The compiler's pass-1
    /// tripwire says this is zero; counting it here checks the same invariant
    /// from the other side, on a real program, for free.
    pub unwritten_reads: u64,
}

/// Whether the level structure was computed over a program that has any hashes
/// at all — a program with none is legal and its ladder is empty.
impl ReachProfile {
    /// Permutation STEPS at `workers` parallel workers: `Σ ceil(level / W)`.
    ///
    /// At `workers = 1` this is exactly [`Self::hash_rows`] (what the executor
    /// does today); as `workers → ∞` it falls to [`Self::hash_depth`].
    pub fn steps_at(&self, workers: usize) -> u64 {
        assert!(workers >= 1, "a schedule has at least one worker");
        self.hash_levels
            .iter()
            .map(|&n| u64::from(n).div_ceil(workers as u64))
            .sum()
    }

    /// The widest level — the most permutations that are ever simultaneously
    /// available.
    pub fn widest_level(&self) -> u32 {
        self.hash_levels.iter().copied().max().unwrap_or(0)
    }

    /// Mean available width, `hash_rows / hash_depth`.
    pub fn mean_width(&self) -> f64 {
        if self.hash_depth == 0 {
            return 0.0;
        }
        self.hash_rows as f64 / self.hash_depth as f64
    }

    /// The report. `ns_per_perm` is the measured host cost of one permutation
    /// (2,356 on Zen 5, 3,000 on Zen 4) so the ladder reads in seconds rather
    /// than in steps.
    pub fn describe(&self, label: &str, ns_per_perm: f64) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        let secs = |steps: u64| steps as f64 * ns_per_perm / 1e9;
        let _ = writeln!(
            s,
            "   ★ REACH {label}: {} instrs · {} addrs · {} hash rows ({:.1}%)",
            self.instrs,
            self.num_addrs,
            self.hash_rows,
            100.0 * self.hash_rows as f64 / self.instrs.max(1) as f64,
        );
        let _ = writeln!(
            s,
            "     depth: hash {} · any-instr {} · mean width {:.0} · widest level {}",
            self.hash_depth,
            self.instr_depth,
            self.mean_width(),
            self.widest_level(),
        );
        let _ = write!(s, "     ideal hash wall:");
        for w in [1usize, 2, 4, 8, 16, 30, 64] {
            let _ = write!(s, "  W={w} {:.2}s", secs(self.steps_at(w)));
        }
        let _ = writeln!(
            s,
            "\n     floor (W=∞) {:.3}s = {} steps · speedup ceiling {:.1}×",
            secs(self.hash_depth as u64),
            self.hash_depth,
            if self.hash_depth == 0 {
                0.0
            } else {
                self.hash_rows as f64 / self.hash_depth as f64
            },
        );
        let _ = writeln!(
            s,
            "     const pool: {} consts, {} reads of them (the cross-block edges a \
             hoist removes) · unwritten reads {}",
            self.consts, self.const_reads, self.unwritten_reads,
        );
        // A coarse shape of the ladder: where the work sits, not every level.
        let mut buckets: BTreeMap<u32, (u32, u64)> = BTreeMap::new();
        for &n in &self.hash_levels {
            let bucket = if n == 0 { 0 } else { n.next_power_of_two() };
            let e = buckets.entry(bucket).or_default();
            e.0 += 1;
            e.1 += u64::from(n);
        }
        let _ = write!(s, "     level widths (≤2^k → levels, hashes):");
        for (b, (levels, hashes)) in buckets {
            let _ = write!(s, "  {b}→{levels}/{hashes}");
        }
        s.push('\n');
        s
    }
}

/// One forward pass over the program. `O(instructions + addresses)` time and
/// `4 · addresses + 8 · instructions` bytes of scratch.
///
/// ⚠ Uses [`Instr::reads`] and [`Instr::writes`] rather than a second
/// transcription of the operand conventions. They allocate a small `Vec` per
/// call, which costs this analysis a few seconds on a 7M-instruction program —
/// deliberately paid, because a private copy of "which operands are live under
/// which selector" is exactly the kind of duplicate that drifts and then
/// reports a dependency structure the machine does not have.
pub fn profile(program: &LfmProgram) -> ReachProfile {
    const NO_WRITER: u32 = u32::MAX;
    let n = program.instrs.len();
    assert!(
        n < NO_WRITER as usize,
        "the profile indexes instructions in a u32; this program has {n}"
    );

    let mut writer = vec![NO_WRITER; program.num_addrs as usize];
    let mut hash_depth = vec![0u32; n];
    let mut instr_depth = vec![0u32; n];
    let mut levels: Vec<u32> = Vec::new();
    let (mut consts, mut hash_rows) = (0usize, 0usize);
    let (mut const_reads, mut unwritten_reads) = (0u64, 0u64);

    for (i, instr) in program.instrs.iter().enumerate() {
        let (mut hd, mut id) = (0u32, 0u32);
        for r in instr.reads() {
            let w = writer[r.0 as usize];
            if w == NO_WRITER {
                unwritten_reads += 1;
                continue;
            }
            if matches!(program.instrs[w as usize], Instr::Const { .. }) {
                const_reads += 1;
            }
            hd = hd.max(hash_depth[w as usize]);
            id = id.max(instr_depth[w as usize]);
        }
        let is_hash = matches!(instr, Instr::Hash { .. });
        if is_hash {
            hash_rows += 1;
            hd += 1;
            let k = hd as usize - 1;
            if levels.len() <= k {
                levels.resize(k + 1, 0);
            }
            levels[k] += 1;
        }
        if matches!(instr, Instr::Const { .. }) {
            consts += 1;
        }
        hash_depth[i] = hd;
        instr_depth[i] = id + 1;
        for a in instr.writes() {
            writer[a.0 as usize] = i as u32;
        }
    }

    ReachProfile {
        instrs: n,
        num_addrs: program.num_addrs,
        hash_rows,
        hash_depth: levels.len(),
        instr_depth: instr_depth.iter().copied().max().unwrap_or(0) as usize,
        hash_levels: levels,
        consts,
        const_reads,
        unwritten_reads,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lfm::builder::LfmBuilder;
    use crate::lfm::compiler::compile;
    use crate::tables::types::FE;

    fn word(v: u64) -> crate::lfm::word::LfmWord {
        core::array::from_fn(|i| FE::from(v + i as u64))
    }

    /// ★★ The gate: a program whose parallel structure is known by
    /// construction, and the profile must report exactly it.
    ///
    /// Four independent chains of three compressions. The right answer is three
    /// levels of four, a speedup ceiling of 4×, and a W=4 schedule that takes
    /// three steps where W=1 takes twelve.
    #[test]
    fn the_profile_reports_a_known_chain_structure() {
        let mut b = LfmBuilder::new();
        let seed = b.digest_const(word(1));
        for chain in 0..4u64 {
            let mut acc = b.digest_const(word(100 + chain));
            for _ in 0..3 {
                acc = b.compress(acc, seed);
            }
        }
        let p = profile(&compile(b.finish()));

        assert_eq!(p.hash_rows, 12, "twelve compressions");
        assert_eq!(p.hash_depth, 3, "three deep");
        assert_eq!(
            p.hash_levels,
            vec![4, 4, 4],
            "four independent at each depth"
        );
        assert_eq!(p.widest_level(), 4);
        assert_eq!(p.steps_at(1), 12, "one worker does every permutation");
        assert_eq!(p.steps_at(2), 6);
        assert_eq!(p.steps_at(4), 3, "four workers reach the critical path");
        assert_eq!(p.steps_at(30), 3, "and cannot beat it");
        assert_eq!(p.unwritten_reads, 0, "every read has a writer");
    }

    /// The counter-case, so the test above is not passing for a shape-blind
    /// reason: ONE chain of twelve is depth twelve and width one, and no worker
    /// count helps.
    #[test]
    fn a_single_chain_has_no_parallelism_to_find() {
        let mut b = LfmBuilder::new();
        let seed = b.digest_const(word(1));
        let mut acc = b.digest_const(word(7));
        for _ in 0..12 {
            acc = b.compress(acc, seed);
        }
        let p = profile(&compile(b.finish()));

        assert_eq!(p.hash_rows, 12);
        assert_eq!(p.hash_depth, 12, "a chain is as deep as it is long");
        assert_eq!(p.hash_levels, vec![1; 12]);
        assert_eq!(p.steps_at(30), 12, "thirty workers buy nothing");
    }

    /// ★ The const-pool hazard, resolved as a property rather than a hope.
    ///
    /// `builder.rs` interns constants across the WHOLE program, so a contiguous
    /// partition has cross-block read edges into `Instr::Const` cells. Hoisting
    /// every `Const` to a pre-pass removes those edges — and is only sound
    /// because a `Const` reads nothing, which is what this asserts. If a future
    /// `Const` variant ever gained an operand, a parallel executor built on the
    /// hoist would race, and this test is what would say so first.
    #[test]
    fn a_const_depends_on_nothing_so_hoisting_it_is_sound() {
        let mut b = LfmBuilder::new();
        let x = b.felt_const(FE::from(7));
        let y = b.felt_const(FE::from(5));
        let s = b.add(x, y);
        let _ = b.mul(s, y);
        let d = b.digest_const(word(3));
        let _ = b.compress(d, d);
        let program = compile(b.finish());

        let mut consts = 0;
        for i in &program.instrs {
            if matches!(i, Instr::Const { .. }) {
                consts += 1;
                assert!(
                    i.reads().is_empty(),
                    "an Instr::Const must read nothing, or the const hoist that \
                     makes a contiguous partition safe is unsound: {i:?}"
                );
            }
        }
        assert!(consts > 0, "the program must contain constants to check");
        assert_eq!(profile(&program).consts, consts, "the profile counts them");
    }

    /// The describe() line renders, carries the ladder, and does not panic on a
    /// program with no hashes at all.
    #[test]
    fn a_program_without_hashes_profiles_to_an_empty_ladder() {
        let mut b = LfmBuilder::new();
        let x = b.felt_const(FE::from(7));
        let y = b.felt_const(FE::from(5));
        let _ = b.add(x, y);
        let p = profile(&compile(b.finish()));

        assert_eq!(p.hash_rows, 0);
        assert_eq!(p.hash_depth, 0);
        assert_eq!(p.steps_at(4), 0);
        assert_eq!(p.mean_width(), 0.0);
        assert!(p.describe("empty", 2356.0).contains("0 hash rows"));
    }
}
