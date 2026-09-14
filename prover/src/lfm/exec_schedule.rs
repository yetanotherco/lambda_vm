//! The execution schedule: which instructions of an LFM program may run at
//! once, computed from the program itself rather than guessed.
//!
//! # Why the levels are exact rather than heuristic
//!
//! Addresses are dense, assigned in emission order, written once, and every
//! operand address is strictly below its destination (`instr.rs`), so the
//! program IS a topologically sorted DAG and one forward pass suffices.
//!
//! Carry, per instruction, `depth` = the length of the longest chain of
//! *dependent [`Instr::Hash`] instructions* ending there. Group by it and the
//! groups have a property no partition heuristic can claim:
//!
//! > **Two hash instructions at the same depth cannot depend on each other.**
//! > If `j` depended on `i` by any path, `depth[j] ≥ depth[i] + 1`, because `i`
//! > is itself a hash. So the levels ARE the available parallelism, with no
//! > segmentation, no guessing, and no soundness argument to make.
//!
//! ★ The same number also orders the work that is NOT a hash, and the argument
//! is one line: if a non-hash `X` has depth `d` and a hash `H` reads `X`, then
//! `H` is a hash, so `depth(H) = d + 1`. **Therefore no hash at depth `d` reads
//! any non-hash at depth `d`** — running a level's hashes first and its non-hash
//! instructions after is safe. Non-hash work at depth `d` may read the hashes at
//! `d` and may read other non-hash work at `d`, and program order (a topological
//! order) covers both. So the whole program is covered by:
//!
//! ```text
//! for d in 0..=D:  { all hashes at depth d, in any order }  then
//!                  { all non-hash instructions at depth d, in program order }
//! ```
//!
//! `Instr::Const` reads nothing, so every constant is depth 0 and the globally
//! interned pool (`builder.rs`) runs first, before anything reads it.
//!
//! # What this module is not
//!
//! It is not [`super::reach_profile`], which measures the same structure for
//! reporting. That one also carries `instr_depth`, a writer array and a
//! histogram, and it allocates a `Vec` per instruction through `Instr::reads`.
//! This one is on the proving path and keeps one array per address and one per
//! instruction, both dropped before the machine is built.

use super::compiler::LfmProgram;
use super::instr::{Addr, Instr};

/// How a program's instructions are grouped for execution.
///
/// `order`-free by construction: the two index vectors below hold instruction
/// indices (and, for hashes, record row numbers) already grouped by level and
/// ascending inside a level, so the executor never sorts and never searches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelSchedule {
    /// Record row numbers of the hash instructions, grouped by level. Ascending
    /// within a level, which is program order, which is the order the serial
    /// executor appends them in.
    hash_rows: Vec<u32>,
    /// `hash_starts[d] .. hash_starts[d + 1]` is level `d`'s slice of
    /// [`Self::hash_rows`].
    hash_starts: Vec<u32>,
    /// Program index of the hash instruction that owns each record row, indexed
    /// BY ROW. `hash_instr[r]` is the `r`-th hash in program order.
    hash_instr: Vec<u32>,
    /// Program indices of the non-hash instructions, grouped by level and
    /// ascending (= program order) within one.
    other: Vec<u32>,
    /// `other_starts[d] .. other_starts[d + 1]` is level `d`'s slice of
    /// [`Self::other`].
    other_starts: Vec<u32>,
    /// Each instruction's row in ITS OWN chip's record vector, in PROGRAM order.
    ///
    /// ⛔ The level order is not program order, so an executor that appended as
    /// it went would hand the trace fill ten permuted vectors. Every arm writes
    /// its slot instead, and this is the slot.
    record_row: Vec<u32>,
}

impl LevelSchedule {
    /// Levels in the schedule. Equal to the longest dependent hash chain plus
    /// one for the depth-0 level that holds the constants.
    pub fn levels(&self) -> usize {
        self.hash_starts.len() - 1
    }

    /// Record rows of the hashes at level `d` — mutually independent by the
    /// property in this module's header.
    pub fn hashes_at(&self, d: usize) -> &[u32] {
        let (a, b) = (
            self.hash_starts[d] as usize,
            self.hash_starts[d + 1] as usize,
        );
        &self.hash_rows[a..b]
    }

    /// Program indices of the non-hash instructions at level `d`, in program
    /// order. They may depend on each other and on the hashes at `d`.
    pub fn others_at(&self, d: usize) -> &[u32] {
        let (a, b) = (
            self.other_starts[d] as usize,
            self.other_starts[d + 1] as usize,
        );
        &self.other[a..b]
    }

    /// The program index of the hash instruction that fills record row `row`.
    #[inline]
    pub fn instr_of_row(&self, row: u32) -> usize {
        self.hash_instr[row as usize] as usize
    }

    /// Instruction `i`'s row in its own chip's record vector.
    #[inline]
    pub fn record_row(&self, i: usize) -> u32 {
        self.record_row[i]
    }

    /// Bytes this schedule holds while the program executes — the figure the
    /// design note quotes, computed rather than estimated.
    pub fn resident_bytes(&self) -> usize {
        size_of::<u32>()
            * (self.hash_rows.len()
                + self.hash_starts.len()
                + self.hash_instr.len()
                + self.other.len()
                + self.other_starts.len()
                + self.record_row.len())
    }
}

/// The forward pass, with a level-merge factor.
///
/// ⛔ **`merge` is `1` on every production path and the executor passes nothing
/// else.** Any other value is a DELIBERATELY WRONG SCHEDULE and exists for the
/// gate that proves the right one is load-bearing: `merge = 2` puts depths `2k`
/// and `2k+1` in one level, so a hash whose input another hash of the merged
/// pair produces is launched before that input exists.
///
/// The executor must then fail, on every run and at the same address, rather
/// than sometimes. It does, and the reason is the borrow checker rather than a
/// check: during a level the workers hold `&WriteOnceMemory` and nothing holds
/// `&mut`, so no write can land while the level runs and the premature read is
/// always [`super::executor::LfmExecError::ReadBeforeWrite`] — never a race, and
/// never timing-dependent. A gate that passed with the level boundary removed
/// would not be a gate; this is what shows it does not.
pub fn build_with_merge(program: &LfmProgram, merge: u32) -> LevelSchedule {
    assert!(merge >= 1, "a level merge factor is at least 1");
    let n = program.instrs.len();
    assert!(
        u32::try_from(n).is_ok(),
        "the schedule indexes instructions in a u32; this program has {n}"
    );

    // ⚠ Depth per ADDRESS, not per writing instruction. The reach profile keeps
    // a writer index and a per-instruction depth and reads one through the
    // other; carrying the depth directly is the same answer in half the memory,
    // and it makes an unwritten read fall out correctly rather than needing a
    // sentinel: it reads 0, its hash lands in level 0, and the executor's own
    // `read_word` raises `ReadBeforeWrite` there. Nothing here has to detect it.
    let mut addr_depth = vec![0u32; program.num_addrs as usize];
    let mut depth = vec![0u32; n];
    let (mut hash_counts, mut other_counts) = (vec![0u32; 1], vec![0u32; 1]);
    let mut scratch: Vec<Addr> = Vec::with_capacity(32);

    for (i, instr) in program.instrs.iter().enumerate() {
        scratch.clear();
        instr.reads_into(&mut scratch);
        let mut d = 0u32;
        for r in &scratch {
            d = d.max(addr_depth[r.0 as usize]);
        }
        let is_hash = matches!(instr, Instr::Hash { .. });
        if is_hash {
            d += 1;
        }
        scratch.clear();
        instr.writes_into(&mut scratch);
        for w in &scratch {
            addr_depth[w.0 as usize] = d;
        }

        // Rounded UP, so `merge = 2` pairs depths 1+2, 3+4, … and leaves depth 0
        // alone. That matters for the mutation gate: collapsing depth 0 into
        // depth 1 would break the schedule at the constants, and the resulting
        // failure would say nothing about whether a hash can be launched before
        // the HASH it depends on. With the offset, the first thing the merged
        // schedule gets wrong is exactly that.
        let level = (d.div_ceil(merge)) as usize;
        depth[i] = level as u32;
        let counts = if is_hash {
            &mut hash_counts
        } else {
            &mut other_counts
        };
        if counts.len() <= level {
            counts.resize(level + 1, 0);
        }
        counts[level] += 1;
    }
    drop(addr_depth);

    let num_levels = hash_counts.len().max(other_counts.len());
    hash_counts.resize(num_levels, 0);
    other_counts.resize(num_levels, 0);

    // Prefix sums, then a stable scatter: `counts` becomes the cursor array and
    // ends up one level shifted, which is exactly the starts array with a
    // leading zero prepended.
    let starts = |counts: &[u32]| -> Vec<u32> {
        let mut s = Vec::with_capacity(counts.len() + 1);
        let mut acc = 0u32;
        s.push(0);
        for &c in counts {
            acc += c;
            s.push(acc);
        }
        s
    };
    let hash_starts = starts(&hash_counts);
    let other_starts = starts(&other_counts);

    let hash_total = *hash_starts.last().expect("starts is never empty") as usize;
    let other_total = *other_starts.last().expect("starts is never empty") as usize;
    let mut hash_rows = vec![0u32; hash_total];
    let mut hash_instr = vec![0u32; hash_total];
    let mut other = vec![0u32; other_total];
    let mut hash_cursor = hash_starts.clone();
    let mut other_cursor = other_starts.clone();

    // Program order, so every bucket ends up ascending: the hash rows of a level
    // come out in the order the serial executor would have pushed them, and the
    // non-hash instructions of a level come out in program order, which is the
    // topological order they need.
    let mut row = 0u32;
    let mut record_row = vec![0u32; n];
    let mut chip_rows = ChipRows::default();
    for (i, instr) in program.instrs.iter().enumerate() {
        let level = depth[i] as usize;
        record_row[i] = chip_rows.take(instr);
        if matches!(instr, Instr::Hash { .. }) {
            hash_instr[row as usize] = i as u32;
            let at = &mut hash_cursor[level];
            hash_rows[*at as usize] = row;
            *at += 1;
            row += 1;
        } else {
            let at = &mut other_cursor[level];
            other[*at as usize] = i as u32;
            *at += 1;
        }
    }
    chip_rows.assert_matches_census(program);

    LevelSchedule {
        hash_rows,
        hash_starts,
        hash_instr,
        other,
        other_starts,
        record_row,
    }
}

/// One row counter per chip, advanced in program order.
///
/// ★ **This is where the length check that `resize` would have made vacuous went
/// instead, and it is a stronger one.** The executor used to `push` and assert
/// at the end that each vector's length equalled its column group's row count —
/// a real check that every instruction produced exactly one record. Slot-writing
/// sets those lengths up front, so that assertion would pass whatever happened.
/// Counting the rows the schedule hands out and comparing THOSE to the census
/// says the same thing, earlier, and about the schedule rather than about the
/// vector.
#[derive(Default)]
struct ChipRows {
    const_: u32,
    balu: u32,
    xalu: u32,
    select: u32,
    bitdec: u32,
    hash: u32,
    keccak: u32,
    blake3: u32,
    lanes: u32,
    hint: u32,
    public: u32,
}

impl ChipRows {
    /// The next row for `instr`'s chip, post-incrementing that chip's counter.
    /// Every arm of the executor pushes exactly one record, so every
    /// instruction takes exactly one row.
    #[inline]
    fn take(&mut self, instr: &Instr) -> u32 {
        let c = match instr {
            Instr::Const { .. } => &mut self.const_,
            Instr::BaseAlu { .. } => &mut self.balu,
            Instr::ExtAlu { .. } => &mut self.xalu,
            Instr::Select { .. } => &mut self.select,
            Instr::BitDec { .. } => &mut self.bitdec,
            Instr::Hash { .. } => &mut self.hash,
            Instr::KeccakF(_) => &mut self.keccak,
            Instr::Blake3(_) => &mut self.blake3,
            // Pack and Unpack are one chip: both open an `LFM_LANES` row.
            Instr::Pack { .. } | Instr::Unpack { .. } => &mut self.lanes,
            Instr::Hint { .. } => &mut self.hint,
            Instr::Public { .. } => &mut self.public,
        };
        let row = *c;
        *c += 1;
        row
    }

    fn assert_matches_census(&self, program: &LfmProgram) {
        let g = &program.groups;
        let mine = [
            self.const_,
            self.balu,
            self.xalu,
            self.select,
            self.bitdec,
            self.hash,
            self.keccak,
            self.blake3,
            self.lanes,
            self.hint,
            self.public,
        ];
        let census = [
            g.const_.real_rows,
            g.balu.real_rows,
            g.xalu.real_rows,
            g.select.real_rows,
            g.bitdec.real_rows,
            g.hash.real_rows,
            g.keccak.real_rows,
            g.blake3.real_rows,
            g.lanes.real_rows,
            g.hint.real_rows,
            g.public.real_rows,
        ];
        assert_eq!(
            mine.map(|v| v as usize),
            census,
            "the schedule handed out a different number of record rows than the \
             compiler emitted column-group rows (order: const, balu, xalu, \
             select, bitdec, hash, keccak, blake3, lanes, hint, public)"
        );
    }
}

/// The schedule a program executes under.
pub fn build(program: &LfmProgram) -> LevelSchedule {
    build_with_merge(program, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lfm::builder::LfmBuilder;
    use crate::lfm::compiler::compile;
    use crate::lfm::word::LfmWord;
    use crate::tables::types::FE;

    fn word(v: u64) -> LfmWord {
        core::array::from_fn(|i| FE::from(v + i as u64))
    }

    /// ★ The gate: a program whose parallel structure is known by construction,
    /// and the schedule must report exactly it.
    ///
    /// Four independent chains of three compressions over a shared seed. The
    /// right answer is a depth-0 level holding the two constants and no hashes,
    /// then three levels of four hashes each.
    fn four_chains_of_three() -> crate::lfm::compiler::LfmProgram {
        let mut b = LfmBuilder::new();
        let seed = b.digest_const(word(1));
        for chain in 0..4u64 {
            let mut acc = b.digest_const(word(100 + chain));
            for _ in 0..3 {
                acc = b.compress(acc, seed);
            }
        }
        compile(b.finish())
    }

    #[test]
    fn the_schedule_reports_a_known_chain_structure() {
        let s = build(&four_chains_of_three());
        assert_eq!(s.levels(), 4, "one const level and three hash levels");
        assert!(
            s.hashes_at(0).is_empty(),
            "a hash is at least depth 1, so level 0 holds no hashes"
        );
        for d in 1..4 {
            assert_eq!(
                s.hashes_at(d).len(),
                4,
                "four independent compressions at depth {d}"
            );
        }
        // Row numbers partition 0..12 exactly once: the record vector is filled
        // by slot, so a repeated or missing row is a silently wrong witness.
        let mut seen: Vec<u32> = (0..4).flat_map(|d| s.hashes_at(d).to_vec()).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..12).collect::<Vec<_>>(), "every row exactly once");
        // Ascending within a level, which is what lets the apply step walk a
        // level's rows in the order the serial executor pushed them.
        for d in 0..4 {
            let rows = s.hashes_at(d);
            assert!(rows.windows(2).all(|w| w[0] < w[1]), "level {d} ascends");
        }
    }

    /// The counter-case, so the test above is not passing for a shape-blind
    /// reason: ONE chain of twelve is twelve levels of one, and no schedule
    /// finds parallelism that is not there.
    #[test]
    fn a_single_chain_has_no_parallelism_to_find() {
        let mut b = LfmBuilder::new();
        let seed = b.digest_const(word(1));
        let mut acc = b.digest_const(word(7));
        for _ in 0..12 {
            acc = b.compress(acc, seed);
        }
        let s = build(&compile(b.finish()));
        assert_eq!(s.levels(), 13, "twelve chained hashes plus the const level");
        for d in 1..13 {
            assert_eq!(s.hashes_at(d).len(), 1, "a chain is one wide at every step");
        }
    }

    /// ★ Every instruction lands in exactly one level, hash or not. A schedule
    /// that dropped one would execute a program with a hole in it, and the
    /// witness would be wrong in a way only the prover would notice.
    #[test]
    fn every_instruction_is_scheduled_exactly_once() {
        let program = crate::lfm::programs::fri_toy_program();
        let s = build(&program);
        let mut seen = vec![0u32; program.instrs.len()];
        for d in 0..s.levels() {
            for &row in s.hashes_at(d) {
                seen[s.instr_of_row(row)] += 1;
            }
            for &i in s.others_at(d) {
                seen[i as usize] += 1;
            }
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "{} of {} instructions were scheduled a number of times other than once",
            seen.iter().filter(|&&c| c != 1).count(),
            seen.len(),
        );
    }

    /// ★★ The ordering property the whole design rests on, checked on a real
    /// program rather than argued: **no instruction at level `d` reads an
    /// address written at a level above `d`, and no hash at level `d` reads one
    /// written by a non-hash at level `d`.** The second half is what makes
    /// "hashes first, then the rest" safe inside a level.
    #[test]
    fn a_level_never_reads_above_itself() {
        let program = crate::lfm::programs::fri_toy_program();
        let s = build(&program);

        // level[i], and whether i is a hash — rebuilt from the schedule, not
        // from a second depth pass, so this checks the schedule and not a copy
        // of the code that made it.
        let mut level = vec![u32::MAX; program.instrs.len()];
        let mut is_hash = vec![false; program.instrs.len()];
        for d in 0..s.levels() {
            for &row in s.hashes_at(d) {
                let i = s.instr_of_row(row);
                level[i] = d as u32;
                is_hash[i] = true;
            }
            for &i in s.others_at(d) {
                level[i as usize] = d as u32;
            }
        }

        let mut writer = vec![u32::MAX; program.num_addrs as usize];
        for (i, instr) in program.instrs.iter().enumerate() {
            for w in instr.writes() {
                writer[w.0 as usize] = i as u32;
            }
        }

        let (mut checked, mut hash_checked) = (0u64, 0u64);
        for (i, instr) in program.instrs.iter().enumerate() {
            for r in instr.reads() {
                let w = writer[r.0 as usize];
                if w == u32::MAX {
                    continue; // unwritten; the executor raises ReadBeforeWrite
                }
                let (mine, theirs) = (level[i], level[w as usize]);
                assert!(
                    theirs <= mine,
                    "instruction {i} at level {mine} reads address {} written at \
                     level {theirs} — the schedule would run it too early",
                    r.0
                );
                checked += 1;
                if is_hash[i] {
                    // ★ The half that makes "hashes first, then the rest" safe:
                    // a hash reads NOTHING at its own level. Not "nothing from a
                    // non-hash at its own level" — nothing at all, because a
                    // hash reading another hash would be one level deeper by the
                    // definition of the depth.
                    assert!(
                        theirs < mine,
                        "hash {i} at level {mine} reads address {} written at \
                         the SAME level by {w} — running a level's hashes before \
                         its other work would launch this one too early",
                        r.0
                    );
                    hash_checked += 1;
                }
            }
        }
        assert!(
            checked > 500 && hash_checked > 50,
            "the program must exercise both halves of the property: {checked} \
             read edges of which {hash_checked} are a hash's"
        );
    }

    /// The mutation the gate in `exec_identity_tests` rides on, checked here at
    /// the structural level: merging adjacent depths really does put a hash and
    /// one of its inputs in one level, so there is something for the executor to
    /// fail on.
    #[test]
    fn merging_levels_puts_a_hash_in_the_same_level_as_its_input() {
        let program = four_chains_of_three();
        let merged = build_with_merge(&program, 2);
        assert!(
            merged.levels() < build(&program).levels(),
            "the merge must actually collapse levels"
        );
        // Depths 1 and 2 — two HASH depths, one feeding the other — now share
        // level 1, while depth 0 (the constants) keeps its own level.
        assert_eq!(
            merged.hashes_at(0).len(),
            0,
            "depth 0 must stay a hash-free level, or the mutation would break at \
             the constants and prove nothing about hash ordering"
        );
        assert_eq!(
            merged.hashes_at(1).len(),
            8,
            "depths 1 and 2 hold four hashes each and must now share a level"
        );
    }
}
