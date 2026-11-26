//! Phase A's commits on two cards: beside card 0's commit lane (the loop of
//! [`BlockCommitted::commit_groups_with`] on the calling thread), a second
//! lane commits groups on the second card, on a thread of its own in the
//! second card's device role.
//!
//! - Each group keeps the index it had when it was handed out (its place in
//!   the block), so its commit-time point, its roots and its answers are the
//!   same on either lane.
//! - The lanes' blocks are put back together by group index
//!   ([`BlockCommitted::merge_lanes`]); a missing or doubled index is an
//!   error.
//! - Phase A takes no main transcript, and phase B absorbs the roots block
//!   once, in group order: the proof is the same for any assignment of groups
//!   to lanes.
//! - A second lane that fails hands every group it holds back, the second
//!   card is marked failed, and card 0 commits them: the prove goes on on one
//!   card.

use std::sync::Mutex;

use super::*;

/// How phase A's two commit lanes share a block's groups.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum LanePolicy {
    /// Each lane takes the first group left whenever it is free for one.
    #[default]
    NextFree,
    /// Group `g` on lane `cards[g]` (lane 0 for a missing entry); once the
    /// second lane is gone, lane 0 takes every group left.
    Fixed(Vec<u8>),
}

/// Phase A's commit lanes ([`BlockCommitted::commit_on_lanes`]): one, as
/// today, or one a card.
#[derive(Clone, Debug, Default)]
pub struct PhaseALanes {
    /// The two-card knob whose split must be on, with its second card ready,
    /// for a second lane; `None`: one lane.
    pub knob: Option<TwoCardKnob>,
    pub policy: LanePolicy,
    /// A test's: two lanes with no second card (the second lane's counts kept
    /// as the second card's).
    #[doc(hidden)]
    pub on_host: bool,
    /// A test's: the second lane fails before its `k`-th commit.
    #[doc(hidden)]
    pub fault: Option<usize>,
}

/// The groups not yet taken, each at its index, shared by the lanes.
pub(super) struct LaneFeed<'a, F, E>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    state: Mutex<FeedState<'a, F, E>>,
    policy: LanePolicy,
    fault: Option<usize>,
}

struct FeedState<'a, F, E>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    /// By group index: the group, until a lane takes it.
    left: Vec<Option<Vec<CommittedTable<'a, F, E>>>>,
    /// The second lane failed and handed its groups back: lane 0 takes any.
    second_gone: bool,
    /// Lane 0 failed: no lane takes another group.
    stopped: bool,
}

impl<'a, F, E> LaneFeed<'a, F, E>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    fn new(
        groups: Vec<Vec<CommittedTable<'a, F, E>>>,
        policy: LanePolicy,
        fault: Option<usize>,
    ) -> Self {
        Self {
            state: Mutex::new(FeedState {
                left: groups.into_iter().map(Some).collect(),
                second_gone: false,
                stopped: false,
            }),
            policy,
            fault,
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, FeedState<'a, F, E>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether lane `lane` may take group `g` under the policy.
    fn wants(&self, g: usize, lane: u8) -> bool {
        match &self.policy {
            LanePolicy::NextFree => true,
            LanePolicy::Fixed(cards) => cards.get(g).copied().unwrap_or(0).min(1) == lane,
        }
    }

    /// The next group lane `lane` commits, with its index; `None` when it has
    /// none left (or the feed stopped, or the second lane is gone).
    fn take(&self, lane: u8) -> Option<(usize, Vec<CommittedTable<'a, F, E>>)> {
        let mut state = self.state();
        if state.stopped || (lane == 1 && state.second_gone) {
            return None;
        }
        let any = lane == 0 && state.second_gone;
        let g = (0..state.left.len())
            .find(|&g| state.left[g].is_some() && (any || self.wants(g, lane)))?;
        state.left[g].take().map(|group| (g, group))
    }

    /// The iterator lane `lane`'s loop reads.
    pub(super) fn lane(
        &self,
        lane: u8,
    ) -> impl Iterator<Item = (usize, Vec<CommittedTable<'a, F, E>>)> + '_ {
        std::iter::from_fn(move || self.take(lane))
    }

    /// The second lane failed: its groups go back at their indexes, and from
    /// now on lane 0 takes every group left.
    pub(super) fn hand_back(&self, groups: Vec<(usize, Vec<CommittedTable<'a, F, E>>)>) {
        let mut state = self.state();
        for (g, group) in groups {
            if let Some(slot) = state.left.get_mut(g) {
                *slot = Some(group);
            }
        }
        state.second_gone = true;
    }

    fn stop(&self) {
        self.state().stopped = true;
    }

    fn has_left(&self) -> bool {
        self.state().left.iter().any(Option::is_some)
    }

    /// A test's fault: the second lane fails before its `commits`-th commit.
    pub(super) fn fault_at(&self, commits: usize) -> bool {
        self.fault == Some(commits)
    }
}

impl PhaseALanes {
    /// Why phase A over `groups` groups runs one lane, or `None` for two: not
    /// asked (no knob), a build without `parallel`, the knob off, its second
    /// card not ready, or one group. A test's lanes on the host ask for two
    /// whatever the knob and the card say.
    pub(crate) fn one_lane_because(&self, groups: usize) -> Option<&'static str> {
        if !self.on_host {
            let Some(knob) = self.knob else {
                return Some("not asked");
            };
            if !cfg!(feature = "parallel") {
                return Some("a build without parallel");
            }
            if !knob.on() {
                return Some(match knob {
                    TwoCardKnob::Split => "the base's split off",
                    TwoCardKnob::Levels => "the levels' split off",
                });
            }
            if !multilinear::gpu::two_card_split_ready() {
                return Some("the second card not ready");
            }
        }
        if groups < 2 {
            return Some("one group");
        }
        None
    }
}

/// One group's share of a lane's block, on its way to its index.
struct LaneGroup<'a, F, E>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    tables: Vec<CommittedTable<'a, F, E>>,
    retired: RetiredStack<F>,
    roots: Vec<Commitment>,
    stamp: GroupStamps,
    ood: OodClaims<E>,
}

impl<'a, F, E> BlockCommitted<'a, F, E>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    /// [`Self::commit`] on `lanes`: one lane on the calling thread (today's
    /// commit, every call and line as it was) unless the lanes' knob is on
    /// and its second card ready (or a test's lanes on the host), with two
    /// groups or more. Then a second lane commits groups on the second card
    /// beside card 0's, each group under the index it was handed out with,
    /// and the lanes' blocks are merged by index: the same block, the same
    /// proof. A failed second lane's groups are committed on card 0; lane 0's
    /// error is the prove's. Prints one line when a knob asked
    /// (`BLOCK TWO-CARD LEVEL PHASE A: on …` or `off (<why>)`).
    #[allow(clippy::too_many_arguments)]
    pub fn commit_on_lanes<H, T>(
        tables: Vec<CommittedTable<'a, F, E>>,
        sizes: &[usize],
        config: &ChainConfig,
        drop_levels: multilinear::whir_commit::TreeDrop,
        narrow: Narrowing,
        upload_ahead: bool,
        lanes: &PhaseALanes,
    ) -> Result<Self, MlError>
    where
        H: WhirHash,
        T: crypto::fiat_shamir::is_transcript::IsTranscript<E> + Default + Send + 'static,
    {
        let label = match lanes.knob {
            Some(TwoCardKnob::Split) => "BLOCK TWO-CARD PHASE A",
            _ => "BLOCK TWO-CARD LEVEL PHASE A",
        };
        if let Some(why) = lanes.one_lane_because(sizes.len()) {
            if lanes.knob.is_some() || lanes.on_host {
                eprintln!("{label}: off ({why})");
            }
            return Self::commit::<H, T>(tables, sizes, config, drop_levels, narrow, upload_ahead);
        }
        if sizes.iter().sum::<usize>() != tables.len() {
            return Err(MlError::QueryCountMismatch {
                expected: tables.len(),
                got: sizes.iter().sum(),
            });
        }
        let n = sizes.len();
        let mut tables = tables.into_iter();
        let groups: Vec<Vec<CommittedTable<'a, F, E>>> = sizes
            .iter()
            .map(|&size| tables.by_ref().take(size).collect())
            .collect();
        let feed = LaneFeed::new(groups, lanes.policy.clone(), lanes.fault);
        let runs_from = multilinear::gpu::joint_runs();
        let host_from = multilinear::gpu::host_fallbacks();
        let declines_from = multilinear::gpu::cross_card_declines();
        let started = Instant::now();
        let probe = CsProbe::default();
        let run = |lane: u8| {
            Self::commit_groups_with::<H, T>(
                feed.lane(lane),
                config,
                drop_levels,
                narrow,
                upload_ahead,
                None,
                None,
                &probe,
                None,
                None,
                Some((&feed, lane)),
            )
        };
        let (first, second) = std::thread::scope(|scope| {
            // The second lane: a plain thread in the second card's role (not
            // the second card's pool), handing its role on to the threads it
            // starts.
            let second = scope.spawn(|| match lanes.on_host {
                true => multilinear::gpu::counted_as_second(|| run(1)),
                false => multilinear::gpu::in_second_role(|| run(1)),
            });
            let first = run(0);
            if first.is_err() {
                feed.stop();
            }
            let second = second
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
            (first, second)
        });
        let mut parts = vec![first?];
        match second {
            Ok(part) => parts.push(part),
            Err(e) => {
                if !lanes.on_host {
                    multilinear::gpu::mark_second_failed("a phase-A commit lane");
                }
                eprintln!(
                    "{label}: the second lane failed ({e:?}); its groups committed on card 0"
                );
            }
        }
        // What the second lane handed back once lane 0 had ended: committed
        // here, in the main role.
        if feed.has_left() {
            parts.push(run(0)?);
        }
        let block = Self::merge_lanes(parts, n, config, upload_ahead)?;
        print_leaves_streamed(runs_from, host_from);
        print_lanes(
            label,
            &lanes.policy,
            &block.stamps,
            started.elapsed().as_secs_f64(),
            multilinear::gpu::cross_card_declines().saturating_sub(declines_from),
            lanes.on_host,
        );
        Ok(block)
    }

    /// The lanes' blocks, each with the indexes of its groups in the order it
    /// committed them, put together in group order: every per-group value
    /// (tables, retired stack, roots, stamps, commit-time claims) at its
    /// index among `n`. An index out of range, filled twice or left empty is
    /// an error. The rest is lane 0's block (the first part), with the lists
    /// one lane leaves for a prove with no spill and no local lookups.
    pub(crate) fn merge_lanes(
        parts: Vec<(Self, Vec<usize>)>,
        n: usize,
        config: &ChainConfig,
        upload_ahead: bool,
    ) -> Result<Self, MlError> {
        let mut slots: Vec<Option<LaneGroup<'a, F, E>>> = (0..n).map(|_| None).collect();
        let mut base: Option<Self> = None;
        for (mut part, indexes) in parts {
            let count = indexes.len();
            if [
                part.sizes.len(),
                part.groups.len(),
                part.stamps.len(),
                part.ood.len(),
            ]
            .iter()
            .any(|&len| len != count)
            {
                return Err(MlError::QueryCountMismatch {
                    expected: count,
                    got: part.sizes.len(),
                });
            }
            let mut tables = std::mem::take(&mut part.tables).into_iter();
            let mut roots = std::mem::take(&mut part.roots).into_iter();
            let groups = std::mem::take(&mut part.groups);
            let stamps = std::mem::take(&mut part.stamps);
            let ood = std::mem::take(&mut part.ood);
            let sizes = std::mem::take(&mut part.sizes);
            for ((((g, size), retired), stamp), ood) in indexes
                .into_iter()
                .zip(sizes)
                .zip(groups)
                .zip(stamps)
                .zip(ood)
            {
                let trees =
                    multilinear::stacked_eval::round0_trees(retired.layout().num_polys(), config)
                        .len();
                let group_tables: Vec<_> = tables.by_ref().take(size).collect();
                let group_roots: Vec<_> = roots.by_ref().take(trees).collect();
                if group_tables.len() != size || group_roots.len() != trees {
                    return Err(MlError::QueryCountMismatch {
                        expected: size,
                        got: group_tables.len(),
                    });
                }
                match slots.get_mut(g) {
                    Some(slot @ None) => {
                        *slot = Some(LaneGroup {
                            tables: group_tables,
                            retired,
                            roots: group_roots,
                            stamp,
                            ood,
                        });
                    }
                    _ => {
                        return Err(MlError::QueryCountMismatch {
                            expected: n,
                            got: g,
                        });
                    }
                }
            }
            if tables.next().is_some() || roots.next().is_some() {
                return Err(MlError::QueryCountMismatch {
                    expected: count,
                    got: count + 1,
                });
            }
            if base.is_none() {
                base = Some(part);
            }
        }
        let filled = slots.iter().filter(|slot| slot.is_some()).count();
        let (Some(mut block), true) = (base, filled == n) else {
            return Err(MlError::QueryCountMismatch {
                expected: n,
                got: filled,
            });
        };
        for slot in slots.into_iter().flatten() {
            block.sizes.push(slot.tables.len());
            block.tables.extend(slot.tables);
            block.groups.push(slot.retired);
            block.roots.extend(slot.roots);
            block.stamps.push(slot.stamp);
            block.ood.push(slot.ood);
        }
        // As one lane leaves them: a group's local argue begun in phase A,
        // one a group when the next group is taken beside a commit (none
        // begun here); and each table's slot in a spill (none).
        block.local_started = match upload_ahead {
            true => (0..n).map(|_| None).collect(),
            false => Vec::new(),
        };
        block.spilled = vec![None; block.tables.len()];
        Ok(block)
    }
}

/// The lanes' line: by card, the groups committed there, their commits (with
/// the retires and the commit-time answers), uploads (and the part paid on
/// the lane), the lane's idle seconds in the span, and its ledger's peak.
fn print_lanes(
    label: &str,
    policy: &LanePolicy,
    stamps: &[GroupStamps],
    span: f64,
    declines: u64,
    on_host: bool,
) {
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let card = |c: u8| {
        let mine: Vec<&GroupStamps> = stamps.iter().filter(|s| s.commit_card == c).collect();
        // `+ 0.0`: an empty sum is −0.0.
        let commit = mine
            .iter()
            .map(|s| s.commit + s.retire + s.ood)
            .sum::<f64>()
            + 0.0;
        let uploads = mine.iter().map(|s| s.upload_a).sum::<f64>() + 0.0;
        let paid = mine.iter().map(|s| s.upload_paid).sum::<f64>() + 0.0;
        let peak = mine.iter().map(|s| s.commit_reserved).max().unwrap_or(0);
        let budget = match (c, on_host) {
            (_, true) => 0,
            (0, false) => multilinear::gpu::reserve_budget(),
            (_, false) => multilinear::gpu::in_second_role(multilinear::gpu::reserve_budget),
        };
        format!(
            "card {c}: {} groups, commit {commit:.3} s, uploads {uploads:.3} s ({paid:.3} on the \
             path), idle {:.3} s, ledger peak {:.2} of {:.2} GiB",
            mine.len(),
            (span - commit - paid).max(0.0),
            gib(peak),
            gib(budget)
        )
    };
    let policy = match policy {
        LanePolicy::NextFree => "next free",
        LanePolicy::Fixed(_) => "fixed",
    };
    eprintln!(
        "{label}: on ({policy}{}) · {} · {} · span {span:.3} s · cross-card declines {declines}",
        if on_host { ", on the host" } else { "" },
        card(0),
        card(1)
    );
}
