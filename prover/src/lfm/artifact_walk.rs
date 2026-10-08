//! The artifact build's walk: every commit a program's artifacts make, in the
//! order and the windows the build has always made them, run as one pass (the
//! whole build, `registry::build_artifacts_with_hasher`) or as two (the halves
//! around a card hold, `registry::build_artifacts_with_device_section`).

use stark::config::Commitment;
use stark::leaf_layout::LeafLayout;
use stark::proof::options::ProofOptions;

use crate::tables::{bitwise, keccak_rc};

use super::airs::{BLAKE3_SLOT, ChipSet, HASH_SLOT, NUM_LFM_CHIPS, blake3_chunk_rows};
use super::commit::{
    SharedAdmission, commit_group_device_or_host_in, commit_group_host_with, commit_reaches_device,
};
use super::compiler::{ColumnGroup, LfmProgram};
use super::hash::HasherKind;
use super::registry::{
    BLAKE3_CHUNK_LABEL, HASH_CHUNK_LABEL, LfmArtifacts, LfmOneRowRoots, PREP_GROUP_LABEL,
    PROGRAM_GROUP_SLOTS, groups_in_flight, map_maybe_parallel, program_groups,
};
use super::statement::lfm_program_id_chunked;
use super::trace::range_group;

/// Which of a build's commits one walk makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Pass {
    /// Every commit, device or host as each group routes — the build as it
    /// has always run.
    All,
    /// Only the commits that stay on the host, through the host-only path.
    Host,
    /// Only the commits that reach the device.
    Device,
}

impl Pass {
    /// Whether this pass makes the commit of a `rows × width` group.
    fn takes(self, rows: usize, width: usize, options: &ProofOptions, layout: LeafLayout) -> bool {
        match self {
            Pass::All => true,
            Pass::Host => !commit_reaches_device(rows, width, options, layout),
            Pass::Device => commit_reaches_device(rows, width, options, layout),
        }
    }

    /// The device bytes this pass's commit of a `rows × width` group takes:
    /// the fused commit's device set when the commit reaches the card, none
    /// when it stays on the host or is left to the other pass.
    fn device_bytes(
        self,
        rows: usize,
        width: usize,
        options: &ProofOptions,
        layout: LeafLayout,
    ) -> u64 {
        if self == Pass::Host
            || !self.takes(rows, width, options, layout)
            || !commit_reaches_device(rows, width, options, layout)
        {
            return 0;
        }
        stark::device_set::commit_device_set_rpl(
            rows,
            width,
            options.blowup_factor as usize,
            true,
            layout.rows_per_leaf(),
        )
        .total()
    }

    /// Commit `group` if this pass takes it.
    fn commit(
        self,
        label: &'static str,
        group: &ColumnGroup,
        options: &ProofOptions,
        layout: LeafLayout,
        shared: SharedAdmission,
    ) -> Option<Commitment> {
        self.takes(group.padded_rows, group.width, options, layout)
            .then(|| self.commit_taken(label, group, options, layout, shared))
    }

    /// Commit `LFM_BLAKE3` chunk `chunk` if this pass takes it, deciding on the
    /// chunk's shape BEFORE materializing it, so a pass that leaves the chunk
    /// to the other never builds it.
    fn commit_blake3_chunk(
        self,
        program: &LfmProgram,
        chunk: usize,
        rows: usize,
        options: &ProofOptions,
        layout: LeafLayout,
        shared: SharedAdmission,
    ) -> Option<Commitment> {
        self.takes(rows, program.groups.blake3.width, options, layout)
            .then(|| {
                let group = program.blake3_chunk_group(chunk);
                self.commit_taken(BLAKE3_CHUNK_LABEL, &group, options, layout, shared)
            })
    }

    fn commit_taken(
        self,
        label: &'static str,
        group: &ColumnGroup,
        options: &ProofOptions,
        layout: LeafLayout,
        shared: SharedAdmission,
    ) -> Commitment {
        match self {
            Pass::Host => commit_group_host_with(group, options, layout),
            Pass::All | Pass::Device => {
                commit_group_device_or_host_in(label, group, options, layout, shared)
            }
        }
    }
}

/// One window of a walk's commits through the build's fork
/// ([`map_maybe_parallel`]), the window's device bytes taken from the shared
/// VRAM gate FIRST, on this thread (`admit`: `stark::prover::shared_vram_admit`
/// in a build), and held until every commit of the window is made.
///
/// ⛔ A commit in the fork runs on a rayon worker, which must never wait on the
/// gate. Each commit used to ask for its own bytes there, and a worker waiting
/// in a prove's `join` could run one on top of the prove's frame and park on
/// bytes that prove keeps. The window is the unit the walk already bounds
/// residency by, so the bytes held are those the window's commits could hold
/// at once.
fn commit_window<T: Sync, R: Send, P>(
    items: &[T],
    device_bytes: impl Fn(&T) -> u64,
    admit: impl FnOnce(u64) -> Option<P>,
    commit: impl Fn(&T, SharedAdmission) -> R + Sync + Send,
) -> Vec<R> {
    let bytes = items
        .iter()
        .map(device_bytes)
        .fold(0u64, u64::saturating_add);
    // No commit of the window reaches the card: no gate to ask (and a build
    // without a card never touches it).
    let held = if bytes > 0 { admit(bytes) } else { None };
    let shared = SharedAdmission::Window {
        held: held.is_some(),
    };
    let roots = map_maybe_parallel(items, |item| commit(item, shared));
    drop(held);
    roots
}

/// One leaf layout's commits: slots 0..=10, the `LFM_BLAKE3` chunks, the
/// `LFM_HASH` tail. `None` where the walk's pass left a commit to the other.
struct LayoutRoots {
    slots: Vec<Option<Commitment>>,
    blake3: Vec<Option<Commitment>>,
    tail: Vec<Option<Commitment>>,
}

impl LayoutRoots {
    fn iter(&self) -> impl Iterator<Item = &Option<Commitment>> {
        self.slots.iter().chain(&self.blake3).chain(&self.tail)
    }

    /// Slot by slot, the root from whichever side made it.
    ///
    /// # Panics
    ///
    /// On a slot made on both sides or on neither. Both halves route on one
    /// predicate, so either is a routing bug, and assembling around it would
    /// put a root nobody made — or one made twice under different rules — into
    /// the program's identity.
    fn merge(self, other: Option<LayoutRoots>) -> LayoutRoots {
        let pick = |a: Vec<Option<Commitment>>, b: Option<Vec<Option<Commitment>>>| {
            let b = b.unwrap_or_else(|| vec![None; a.len()]);
            assert_eq!(a.len(), b.len(), "the two halves walked different plans");
            a.into_iter()
                .zip(b)
                .map(|pair| match pair {
                    (Some(root), None) | (None, Some(root)) => Some(root),
                    (Some(_), Some(_)) => panic!("an artifact commit was made on both sides"),
                    (None, None) => panic!("an artifact commit was made on neither side"),
                })
                .collect()
        };
        let (slots, blake3, tail) = match other {
            Some(o) => (Some(o.slots), Some(o.blake3), Some(o.tail)),
            None => (None, None, None),
        };
        LayoutRoots {
            slots: pick(self.slots, slots),
            blake3: pick(self.blake3, blake3),
            tail: pick(self.tail, tail),
        }
    }

    /// The three lists, every commit made.
    fn complete(self) -> (Vec<Commitment>, Vec<Commitment>, Vec<Commitment>) {
        let all = |v: Vec<Option<Commitment>>| -> Vec<Commitment> {
            v.into_iter()
                .map(|r| r.expect("every artifact commit is made before assembly"))
                .collect()
        };
        (all(self.slots), all(self.blake3), all(self.tail))
    }
}

/// One walk's commits under both leaf layouts.
pub(super) struct Walked {
    row_pair: LayoutRoots,
    /// Present when the options' format builds one-row roots.
    one_row: Option<LayoutRoots>,
}

impl Walked {
    fn iter(&self) -> impl Iterator<Item = &Option<Commitment>> {
        self.row_pair
            .iter()
            .chain(self.one_row.iter().flat_map(LayoutRoots::iter))
    }

    /// Commits this walk made.
    pub(super) fn made(&self) -> usize {
        self.iter().filter(|r| r.is_some()).count()
    }

    /// Commits this walk left to the other half.
    pub(super) fn left(&self) -> usize {
        self.iter().filter(|r| r.is_none()).count()
    }

    pub(super) fn merge(self, other: Option<Walked>) -> Walked {
        let (row_pair, one_row) = match other {
            Some(o) => (Some(o.row_pair), o.one_row),
            None => (None, None),
        };
        Walked {
            row_pair: self.row_pair.merge(row_pair),
            one_row: self.one_row.map(|mine| mine.merge(one_row)),
        }
    }
}

/// What an artifact build commits, materialized once and walked by one or two
/// passes.
pub(super) struct BuildPlan<'p> {
    program: &'p LfmProgram,
    range: ColumnGroup,
    /// A split `LFM_HASH`'s chunks; empty for an unsplit program.
    hash_chunks: Vec<ColumnGroup>,
    /// The padded height of each `LFM_BLAKE3` chunk, in chunk order.
    blake3_rows: Vec<usize>,
    one_row: bool,
}

impl<'p> BuildPlan<'p> {
    pub(super) fn new(program: &'p LfmProgram, options: &ProofOptions) -> Self {
        // A split `LFM_HASH` commits chunk 0 at slot 5 and chunk 1 as a
        // tail below. An unsplit program materializes nothing here — slot 5 is the
        // compiled group itself, exactly as before the split existed.
        let hash_chunks: Vec<ColumnGroup> = if program.hash_chunk_count() > 1 {
            (0..program.hash_chunk_count())
                .map(|c| program.hash_chunk_group(c))
                .collect()
        } else {
            Vec::new()
        };
        Self {
            program,
            range: range_group(),
            hash_chunks,
            // The chunk heights are arithmetic (`blake3_chunk_rows`), so they are
            // known without materializing a single chunk group.
            blake3_rows: blake3_chunk_rows(program),
            one_row: options.format.one_row != stark::proof::options::OneRowMode::Off,
        }
    }

    /// Slots 0..=10 in slot order: the program's own nine groups, then
    /// `LFM_RANGE`. Slot 11 (`LFM_BLAKE3`) is not here because it is the CHUNKED
    /// one: it contributes one committed matrix per chunk, built and absorbed
    /// after this list, which is where slot order puts it anyway.
    fn groups(&self) -> [&ColumnGroup; 11] {
        let program_slots = program_groups(self.program);
        std::array::from_fn(|i| {
            if i == HASH_SLOT && !self.hash_chunks.is_empty() {
                &self.hash_chunks[0]
            } else if i < PROGRAM_GROUP_SLOTS {
                program_slots[i]
            } else {
                &self.range
            }
        })
    }

    /// The hash tail: `LFM_HASH` chunks 1.. (chunk 0 is slot 5).
    fn tail(&self) -> Vec<&ColumnGroup> {
        self.hash_chunks.iter().skip(1).collect()
    }

    pub(super) fn walk(&self, options: &ProofOptions, pass: Pass) -> Walked {
        use stark::leaf_layout::LeafLayout::{Row, RowPair};
        let groups = self.groups();
        let tail = self.tail();
        let chunks: Vec<usize> = (0..self.blake3_rows.len()).collect();

        // ★ N GROUPS IN FLIGHT, NOT ONE AND NOT ELEVEN. Each pass is an independent
        // expand-and-commit of its own matrix, so the loop parallelizes on its face;
        // what stopped it was the residency trade the old comment named — "peak
        // residency is one group's LDE" — taken when host memory was the binding
        // constraint on this path. It is not any more: lane P measured a production
        // L1 node's host peak at 14.4 GiB of 57.53.
        //
        // ⚠ So the trade is RE-PRICED, not discarded. Running all eleven at once
        // would multiply the peak by the group count for a program whose group sizes
        // we do not control; a fixed window multiplies it by at most
        // `groups_in_flight()` and says so.
        //
        // ⛔ AND THE WINDOW IS THE SECOND-ORDER HALF. `lde_columns` → `dispatch_fft`
        // already sends any buffer of 2^14 elements or more to the parallel Bowers
        // FFT, so the work inside this loop was never strictly serial — but its
        // per-layer block threshold leaves the early layers sequential, and lane P
        // measures the result at 445% CPU (4.5 of 30.7 cores) on a production node
        // against `emit_merkle`'s 2,652%. The COLUMN-level `par_iter` is what closes
        // that gap; this window covers only the short groups, where each column's
        // FFT stays sequential.
        //
        // ⚠ My own laptop A/B read both as a wash (serial 1.547 s, windowed 1.447 s
        // on a 27 MiB program). That fixture concentrates its felts in ONE NARROW
        // group, which is close to the worst case for a per-column spread — it is a
        // statement about the fixture, not about the change. Lane P's production
        // split is the number to plan against.
        //
        // ⓘ ON A CUDA BUILD THE WINDOW ALSO BOUNDS THE CARD. Each group goes to
        // the card, where the residency that matters is `stark::device_set`'s:
        // under the shared VRAM gate the window's sets are admitted together, on
        // this thread before the fork ([`commit_window`]); under the derivation's
        // gate each dispatch is. The window still bounds the host path exactly as
        // before, which is the path a machine with no card takes.
        let in_flight = groups_in_flight();
        let admit = stark::prover::shared_vram_admit;
        let group_bytes = |layout: LeafLayout| {
            move |g: &&ColumnGroup| pass.device_bytes(g.padded_rows, g.width, options, layout)
        };
        let chunk_bytes = |layout: LeafLayout| {
            move |c: &usize| {
                pass.device_bytes(
                    self.blake3_rows[*c],
                    self.program.groups.blake3.width,
                    options,
                    layout,
                )
            }
        };
        let mut slots = vec![None; groups.len()];
        for (base, window) in groups.chunks(in_flight).enumerate() {
            let commits = commit_window(window, group_bytes(RowPair), admit, |g, shared| {
                pass.commit(PREP_GROUP_LABEL, g, options, RowPair, shared)
            });
            for (k, root) in commits.into_iter().enumerate() {
                slots[base * in_flight + k] = root;
            }
        }
        // Then `LFM_BLAKE3`, one window of chunks at a time: materialize each
        // chunk's group, expand it, commit it, drop both. Peak residency is the
        // window's chunks — which is still the point of chunking this chip, at a
        // bound that names itself.
        let mut blake3 = Vec::with_capacity(chunks.len());
        for window in chunks.chunks(in_flight) {
            blake3.extend(commit_window(
                window,
                chunk_bytes(RowPair),
                admit,
                |c, shared| {
                    let rows = self.blake3_rows[*c];
                    pass.commit_blake3_chunk(self.program, *c, rows, options, RowPair, shared)
                },
            ));
        }
        // The hash tail, committed like the BLAKE3 chunks.
        let mut tail_roots = Vec::with_capacity(tail.len());
        for window in tail.chunks(in_flight) {
            tail_roots.extend(commit_window(
                window,
                group_bytes(RowPair),
                admit,
                |g, shared| pass.commit(HASH_CHUNK_LABEL, g, options, RowPair, shared),
            ));
        }
        let row_pair = LayoutRoots {
            slots,
            blake3,
            tail: tail_roots,
        };
        // The one-row (S2) roots of the same groups: each list is dispatched
        // whole, as it always has been, so each list is one window.
        let one_row = self.one_row.then(|| LayoutRoots {
            slots: commit_window(&groups, group_bytes(Row), admit, |g, shared| {
                pass.commit(PREP_GROUP_LABEL, g, options, Row, shared)
            }),
            blake3: commit_window(&chunks, chunk_bytes(Row), admit, |c, shared| {
                let rows = self.blake3_rows[*c];
                pass.commit_blake3_chunk(self.program, *c, rows, options, Row, shared)
            }),
            tail: commit_window(&tail, group_bytes(Row), admit, |g, shared| {
                pass.commit(HASH_CHUNK_LABEL, g, options, Row, shared)
            }),
        });
        Walked { row_pair, one_row }
    }

    /// The artifacts from a walk that made every commit.
    pub(super) fn assemble(
        &self,
        options: &ProofOptions,
        hasher: HasherKind,
        walked: Walked,
    ) -> LfmArtifacts {
        let groups = self.groups();
        let tail = self.tail();
        let program = self.program;
        let mut roots = [[0u8; 32]; NUM_LFM_CHIPS];
        let mut log_heights = [0u8; NUM_LFM_CHIPS];

        // Heights from the compiled groups: the walk committed them and the
        // digest binds them, so both read one derivation.
        for (i, g) in groups.iter().enumerate() {
            log_heights[i] = g.padded_rows.trailing_zeros() as u8;
        }
        // Slot 11's own entry is chunk 0's — the array stays the shape a
        // single-table program has.
        let blake3_chunk_log_heights: Vec<u8> = self
            .blake3_rows
            .iter()
            .map(|rows| rows.trailing_zeros() as u8)
            .collect();
        log_heights[BLAKE3_SLOT] = blake3_chunk_log_heights[0];

        let (slot_roots, blake3_chunk_roots, tail_roots) = walked.row_pair.complete();
        roots[..slot_roots.len()].copy_from_slice(&slot_roots);
        roots[BLAKE3_SLOT] = blake3_chunk_roots[0];
        // Chunk 0 of the hash is slot 5's root; the tail follows it.
        let mut hash_chunk_roots: Vec<Commitment> = vec![roots[HASH_SLOT]];
        let mut hash_chunk_log_heights: Vec<u8> = vec![log_heights[HASH_SLOT]];
        hash_chunk_roots.extend(tail_roots);
        hash_chunk_log_heights.extend(tail.iter().map(|g| g.padded_rows.trailing_zeros() as u8));
        // Slot 12 (KECCAK_RND) keeps the all-zero sentinel installed above.
        roots[13] = keccak_rc::preprocessed_commitment(options);
        log_heights[13] = keccak_rc::NUM_ROWS.trailing_zeros() as u8;
        roots[14] = bitwise::preprocessed_commitment(options);
        log_heights[14] = bitwise::NUM_ROWS.trailing_zeros() as u8;

        // The families this program uses, and the chunk count that follows from
        // them: zero KECCAK_RND instances when the keccak family is absent, since
        // the chunking policy's floor of one exists only to keep an unused chip
        // present.
        let chip_set = ChipSet::for_program_with_hasher(program, hasher);
        let keccak_rnd_chunks = chip_set.keccak_rnd_chunks(
            program
                .chunking
                .chunk_count(program.groups.keccak.real_rows),
        );
        // ★ `blake3_chunk_roots` is NOT mask-gated the way `keccak_rnd_chunks` is,
        // and the asymmetry is real: `KECCAK_RND` commits nothing, so an absent
        // family can drop to zero instances for free, while `LFM_BLAKE3`'s
        // instruction group is COMMITTED whether the family is used or not — slot 11
        // has always carried a root and a height for a program that never
        // compresses. So the chunk lists describe what was committed (never empty)
        // and the mask decides what a proof carries, exactly where it always did:
        // `ChipSet::num_airs` and `LfmAirs::air_refs`.

        let program_id = lfm_program_id_chunked(
            &roots,
            &log_heights,
            keccak_rnd_chunks,
            hasher,
            chip_set,
            &blake3_chunk_roots,
            &blake3_chunk_log_heights,
            &hash_chunk_roots,
            &hash_chunk_log_heights,
        );
        let one_row_roots = walked
            .one_row
            .map(|one_row| Self::one_row_roots(options, one_row));
        LfmArtifacts {
            roots,
            log_heights,
            keccak_rnd_chunks,
            blake3_chunk_roots,
            blake3_chunk_log_heights,
            hash_chunk_roots,
            hash_chunk_log_heights,
            hasher,
            chip_set,
            program_id,
            one_row_roots,
        }
    }

    /// The one-row roots of every committed group, plus the static tables'
    /// one-row twins.
    fn one_row_roots(options: &ProofOptions, walked: LayoutRoots) -> LfmOneRowRoots {
        use stark::leaf_layout::LeafLayout::Row;
        let (slot_roots, blake3_chunk_roots, tail_roots) = walked.complete();
        let mut roots: [Option<Commitment>; NUM_LFM_CHIPS] = [None; NUM_LFM_CHIPS];
        for (slot, root) in slot_roots.into_iter().enumerate() {
            roots[slot] = Some(root);
        }
        roots[BLAKE3_SLOT] = blake3_chunk_roots.first().copied();
        roots[13] = keccak_rc::preprocessed_commitment_for(options, Row);
        roots[14] = bitwise::preprocessed_commitment_for(options, Row);
        // Chunk 0 is slot 5's one-row root; the hash tail follows.
        let mut hash_chunk_roots: Vec<Commitment> = roots[HASH_SLOT].into_iter().collect();
        hash_chunk_roots.extend(tail_roots);
        LfmOneRowRoots {
            roots,
            blake3_chunk_roots,
            hash_chunk_roots,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roots(slots: &[Option<u8>]) -> LayoutRoots {
        LayoutRoots {
            slots: slots.iter().map(|s| s.map(|b| [b; 32])).collect(),
            blake3: vec![],
            tail: vec![],
        }
    }

    /// Each slot from the side that made it.
    #[test]
    fn two_disjoint_halves_merge_slot_by_slot() {
        let merged = roots(&[Some(1), None, Some(3)]).merge(Some(roots(&[None, Some(2), None])));
        assert_eq!(
            merged.slots,
            vec![Some([1; 32]), Some([2; 32]), Some([3; 32])]
        );
        let alone = roots(&[Some(1), Some(2)]).merge(None);
        assert_eq!(alone.slots, vec![Some([1; 32]), Some([2; 32])]);
    }

    /// ⛔ A slot committed by both halves is a routing bug, refused.
    #[test]
    #[should_panic(expected = "made on both sides")]
    fn a_slot_made_on_both_sides_is_refused() {
        let _ = roots(&[Some(1)]).merge(Some(roots(&[Some(1)])));
    }

    /// ⛔ And so is a slot neither made — including when the device half never
    /// ran, which is the case a missing `enter_device` call would produce.
    #[test]
    #[should_panic(expected = "made on neither side")]
    fn a_slot_made_on_neither_side_is_refused() {
        let _ = roots(&[Some(1), None]).merge(None);
    }

    /// ★ A window's device bytes are asked for ONCE, on the thread that forks
    /// it and before the fork, and held until its last commit is made; every
    /// commit in the fork is told they are held and asks for nothing itself.
    #[test]
    fn a_window_asks_for_its_bytes_once_on_the_forking_thread() {
        use std::sync::atomic::{AtomicBool, Ordering};
        struct Held<'a>(&'a AtomicBool);
        impl Drop for Held<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::SeqCst);
            }
        }
        let here = std::thread::current().id();
        let asked = std::sync::Mutex::new(Vec::new());
        let held = AtomicBool::new(false);
        let made = commit_window(
            &[3u64, 0, 5, 2],
            |&bytes| bytes,
            |bytes| {
                asked
                    .lock()
                    .unwrap()
                    .push((bytes, std::thread::current().id()));
                held.store(true, Ordering::SeqCst);
                Some(Held(&held))
            },
            |&bytes, shared| {
                assert!(
                    held.load(Ordering::SeqCst),
                    "a commit runs with the window's bytes held"
                );
                (bytes, shared)
            },
        );
        assert_eq!(*asked.lock().unwrap(), vec![(10, here)]);
        assert!(!held.load(Ordering::SeqCst), "given back after the window");
        assert_eq!(
            made,
            [3, 0, 5, 2].map(|b| (b, SharedAdmission::Window { held: true })),
            "every commit, in order, told its bytes are held"
        );
    }

    /// A window none of whose commits reaches the card asks for nothing, and
    /// a gate that is off (`None`) leaves each commit to the derivation's gate.
    #[test]
    fn a_window_with_no_device_bytes_or_no_gate_holds_nothing() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let asked = AtomicUsize::new(0);
        let ask = |_: u64| -> Option<()> {
            asked.fetch_add(1, Ordering::SeqCst);
            None
        };
        let host = commit_window(&[1u8, 2], |_| 0, ask, |_, shared| shared);
        assert_eq!(asked.load(Ordering::SeqCst), 0, "no device bytes: no ask");
        let off = commit_window(&[1u8, 2], |_| 7, ask, |_, shared| shared);
        assert_eq!(asked.load(Ordering::SeqCst), 1, "one ask for the window");
        for shared in host.into_iter().chain(off) {
            assert_eq!(shared, SharedAdmission::Window { held: false });
        }
    }
}
