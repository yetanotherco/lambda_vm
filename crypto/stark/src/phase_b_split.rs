//! Phase B's groups over two cards (`LAMBDA_VM_TWO_CARD_SPLIT`): which card
//! proves which group, and the order each card takes its own groups in.
//!
//! Each group proves on its own fork of the post-roots transcript and its
//! share of the proof is put back by group index, so any partition of the
//! groups over the cards, each card taking its groups in any order, proves the
//! same bytes ([`crate::multilinear_block::block_prove_in_order`]). What is
//! here only decides the schedule: a pure function of the layout, never of a
//! clock or an earlier run.

/// A group's phase-B price for the assignment: its cells, plus about a
/// million cells per table for each table's fixed argue and opening cost
/// (a fit of the large block's per-group argue: ≈ 0.94 ms per M cells, ≈ 1.9
/// ms per table).
pub fn split_price(cells: u64, tables: usize) -> u64 {
    cells.saturating_add((tables as u64) << 20)
}

/// Each group's card, 0 or 1: the groups `pinned` to card 0 first (counted in
/// its load), then the rest by longest-processing-time first — by price,
/// descending, ties to the lower group — each to the card with the smaller
/// load, ties to card 0. Deterministic in `prices` and `pinned` alone.
pub fn assign_cards(prices: &[u64], pinned: &[bool]) -> Vec<u8> {
    let n = prices.len();
    let mut card = vec![0u8; n];
    let mut load = [0u128; 2];
    for (g, &price) in prices.iter().enumerate() {
        if pinned.get(g).copied().unwrap_or(false) {
            load[0] += u128::from(price);
        }
    }
    let mut free: Vec<usize> = (0..n)
        .filter(|&g| !pinned.get(g).copied().unwrap_or(false))
        .collect();
    free.sort_by(|&a, &b| prices[b].cmp(&prices[a]).then(a.cmp(&b)));
    for g in free {
        let c = usize::from(load[1] < load[0]);
        card[g] = c as u8;
        load[c] += u128::from(prices[g]);
    }
    card
}

/// Each card's summed price under `card`.
pub fn card_loads(prices: &[u64], card: &[u8]) -> [u128; 2] {
    let mut load = [0u128; 2];
    for (g, &c) in card.iter().enumerate() {
        load[usize::from(c.min(1))] += u128::from(prices[g]);
    }
    load
}

/// The groups still to come in phase B, shared by the cards' drivers: each
/// driver takes, at its turn, the first group of the order that is its own, so
/// each card follows phase B's order. A driver claims its next group (whose
/// columns go up beside this argue) at the same turn; a claimed group is out
/// of the order and nothing moves it.
///
/// With one card every group is card 0's and the sequence of turns, hook
/// views and next groups is the order's own, as phase B without a split.
#[derive(Clone, Debug)]
pub struct TurnQueue {
    /// Groups whose turn started, in that order.
    started: Vec<usize>,
    /// Each card's claimed next group.
    claimed: [Option<usize>; 2],
    /// Groups neither started nor claimed, in phase B's order.
    rest: Vec<usize>,
    /// Each group's card.
    card: Vec<u8>,
}

/// A card's turn: its group; how many turns started before it, over both
/// cards (`turn`: each of `0..n` once); and the order as the re-ranking hook
/// reads it (`view[place] == group`; what follows `place` is the hook's to
/// re-rank). With one card `place == turn`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Turn {
    pub group: usize,
    pub turn: usize,
    pub place: usize,
    pub view: Vec<usize>,
}

impl TurnQueue {
    /// Phase B's `order` over cards `card` (one per group).
    pub fn new(order: &[usize], card: Vec<u8>) -> Self {
        Self {
            started: Vec::with_capacity(order.len()),
            claimed: [None, None],
            rest: order.to_vec(),
            card,
        }
    }

    fn is_mine(&self, lane: usize, g: usize) -> bool {
        usize::from(self.card.get(g).copied().unwrap_or(0).min(1)) == lane
    }

    /// Card `lane`'s next turn, `None` when it has no group left: its claimed
    /// group, else the first of the order that is its own. The view is the
    /// groups started, the other card's claimed group, this group, then the
    /// rest: the hook may re-rank only the rest.
    pub fn begin(&mut self, lane: usize) -> Option<Turn> {
        let g = match self.claimed[lane].take() {
            Some(g) => g,
            None => {
                let at = self.rest.iter().position(|&g| self.is_mine(lane, g))?;
                self.rest.remove(at)
            }
        };
        let turn = self.started.len();
        let mut view = self.started.clone();
        view.extend(self.claimed[1 - lane]);
        let place = view.len();
        view.push(g);
        view.extend_from_slice(&self.rest);
        self.started.push(g);
        Some(Turn {
            group: g,
            turn,
            place,
            view,
        })
    }

    /// The groups the hook may re-rank now: the rest.
    pub fn rest(&self) -> &[usize] {
        &self.rest
    }

    /// The hook's re-ranking of the rest; refused unless it is a permutation
    /// of it.
    pub fn rerank(&mut self, rest: Vec<usize>) -> Result<(), usize> {
        let mut was = self.rest.clone();
        let mut now = rest.clone();
        was.sort_unstable();
        now.sort_unstable();
        if was != now {
            return Err(self.rest.len());
        }
        self.rest = rest;
        Ok(())
    }

    /// Card `lane` claims its next group: the first of the rest that is its
    /// own, taken out of the rest.
    pub fn claim_next(&mut self, lane: usize) -> Option<usize> {
        let at = self.rest.iter().position(|&g| self.is_mine(lane, g))?;
        let g = self.rest.remove(at);
        self.claimed[lane] = Some(g);
        Some(g)
    }

    /// Groups whose turn started, in that order.
    pub fn started(&self) -> &[usize] {
        &self.started
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A test's re-ranking hook: `(place, view)` to the rest re-ranked, or
    /// `None`.
    type Hook<'h> = dyn Fn(usize, &[usize]) -> Option<Vec<usize>> + 'h;

    /// One card's turns over `order`, the hook re-ranking per `hook(place,
    /// view)`: (group, place, view, next) at each turn.
    fn one_card_turns(
        order: &[usize],
        hook: &Hook<'_>,
    ) -> Vec<(usize, usize, Vec<usize>, Option<usize>)> {
        let mut q = TurnQueue::new(order, vec![0; order.len()]);
        let mut out = Vec::new();
        while let Some(turn) = q.begin(0) {
            if let Some(rest) = hook(turn.place, &turn.view) {
                q.rerank(rest).expect("a permutation");
            }
            let next = q.claim_next(0);
            assert_eq!(turn.turn, turn.place, "one card: the place is the turn");
            out.push((turn.group, turn.place, turn.view, next));
        }
        out
    }

    /// Today's loop over `order` with the same hook: (g, p, order as the hook
    /// read it, next).
    fn todays_turns(
        order: &[usize],
        hook: &Hook<'_>,
    ) -> Vec<(usize, usize, Vec<usize>, Option<usize>)> {
        let mut order = order.to_vec();
        let mut out = Vec::new();
        for p in 0..order.len() {
            let g = order[p];
            let view = order.clone();
            if let Some(rest) = hook(p, &order) {
                order[p + 1..].copy_from_slice(&rest);
            }
            out.push((g, p, view, order.get(p + 1).copied()));
        }
        out
    }

    #[test]
    fn one_card_takes_todays_turns_views_and_next_groups() {
        let order = [3usize, 0, 4, 1, 2, 5];
        let none = |_: usize, _: &[usize]| None;
        assert_eq!(one_card_turns(&order, &none), todays_turns(&order, &none));
        // A hook that re-ranks once, at the second turn: the rest reversed.
        let once = |p: usize, view: &[usize]| {
            (p == 1).then(|| view[p + 1..].iter().rev().copied().collect())
        };
        assert_eq!(one_card_turns(&order, &once), todays_turns(&order, &once));
        // A hook that re-ranks at every turn.
        let every = |p: usize, view: &[usize]| {
            let mut rest = view[p + 1..].to_vec();
            let by = 1.min(rest.len());
            rest.rotate_left(by);
            Some(rest)
        };
        assert_eq!(one_card_turns(&order, &every), todays_turns(&order, &every));
    }

    /// Two cards, turns interleaved as `script` says: every group is started
    /// once, by its own card, each card in the order's sequence of its groups.
    #[test]
    fn two_cards_partition_the_order_and_keep_its_sequence() {
        let order = [5usize, 2, 7, 0, 1, 6, 3, 4];
        let card = vec![0, 1, 0, 1, 1, 0, 0, 1];
        for script in [
            vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
            vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        ] {
            let mut q = TurnQueue::new(&order, card.clone());
            let mut taken: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
            let mut turns = Vec::new();
            for &lane in &script {
                if let Some(turn) = q.begin(lane) {
                    turns.push(turn.turn);
                    assert_eq!(turn.view[turn.place], turn.group);
                    assert_eq!(usize::from(card[turn.group]), lane);
                    taken[lane].push(turn.group);
                    q.claim_next(lane);
                }
            }
            for (lane, taken) in taken.iter_mut().enumerate() {
                while let Some(turn) = q.begin(lane) {
                    turns.push(turn.turn);
                    taken.push(turn.group);
                    q.claim_next(lane);
                }
                let own: Vec<usize> = order
                    .iter()
                    .copied()
                    .filter(|&g| usize::from(card[g]) == lane)
                    .collect();
                assert_eq!(*taken, own, "card {lane} under {script:?}");
            }
            let mut all = q.started().to_vec();
            all.sort_unstable();
            assert_eq!(all, (0..order.len()).collect::<Vec<_>>());
            assert_eq!(
                turns,
                (0..order.len()).collect::<Vec<_>>(),
                "turns count up"
            );
        }
    }

    /// A claimed group never moves: the hook re-ranks only the rest, and the
    /// other card's claimed group sits before this turn's place in the view.
    #[test]
    fn a_claimed_group_is_out_of_the_hooks_reach() {
        let order = [0usize, 1, 2, 3, 4, 5];
        let card = vec![0, 1, 0, 1, 0, 1];
        let mut q = TurnQueue::new(&order, card);
        let t0 = q.begin(0).expect("card 0's first");
        assert_eq!((t0.group, t0.place), (0, 0));
        assert_eq!(q.claim_next(0), Some(2));
        let t1 = q.begin(1).expect("card 1's first");
        // started [0], card 0's claimed 2, this group 1, then the rest.
        assert_eq!(t1.view, vec![0, 2, 1, 3, 4, 5]);
        assert_eq!(t1.place, 2);
        assert_eq!(q.rest(), &[3, 4, 5]);
        // The hook's whole re-ranking puts 5 first; 2 stays card 0's next.
        q.rerank(vec![5, 4, 3]).expect("a permutation");
        assert_eq!(q.claim_next(1), Some(5));
        assert_eq!(q.begin(0).map(|t| t.group), Some(2));
        // A re-ranking that is not a permutation of the rest is refused.
        assert_eq!(q.rerank(vec![4, 4]), Err(2));
        assert_eq!(q.rerank(vec![3, 4, 2]), Err(2));
    }

    #[test]
    fn the_assignment_is_deterministic_and_ties_go_to_card_0() {
        let prices = [10u64, 10, 10, 10];
        let a = assign_cards(&prices, &[false; 4]);
        assert_eq!(a, vec![0, 1, 0, 1], "equal prices alternate, card 0 first");
        assert_eq!(assign_cards(&prices, &[false; 4]), a);
        assert_eq!(assign_cards(&[], &[]), Vec::<u8>::new());
        assert_eq!(assign_cards(&[7], &[false]), vec![0]);
    }

    /// Longest-processing-time first: the larger card's load is within one
    /// group's price of half the total, on a spread like the large block's.
    #[test]
    fn the_assignment_balances_within_one_groups_price() {
        // A layout-like spread: a few big groups, many small ones.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for trial in 0..50 {
            let n = 5 + trial % 45;
            let prices: Vec<u64> = (0..n)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let big = (state >> 60) == 0;
                    (state >> 33) % if big { 2_000_000_000 } else { 400_000_000 } + 1
                })
                .collect();
            let card = assign_cards(&prices, &vec![false; n]);
            let load = card_loads(&prices, &card);
            let total: u128 = load.iter().sum();
            let max = *prices.iter().max().expect("groups") as u128;
            assert!(
                load[0].max(load[1]) * 2 <= total + 2 * max,
                "trial {trial}: loads {load:?}, total {total}, largest group {max}"
            );
            assert!(card.contains(&1) || n == 1, "card 1 got nothing");
        }
    }

    /// A pinned group is card 0's whatever its price, and counts in card 0's
    /// load: the rest balance against it.
    #[test]
    fn pinned_groups_stay_on_card_0_and_count_in_its_load() {
        let prices = [100u64, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10];
        let mut pinned = vec![false; prices.len()];
        pinned[0] = true;
        let card = assign_cards(&prices, &pinned);
        assert_eq!(card[0], 0);
        assert!(
            card[1..].iter().all(|&c| c == 1),
            "the 100 pinned on card 0 outweighs the other ten: {card:?}"
        );
        let all_pinned = assign_cards(&prices, &vec![true; prices.len()]);
        assert!(all_pinned.iter().all(|&c| c == 0));
    }

    #[test]
    fn the_price_adds_a_million_cells_a_table() {
        assert_eq!(split_price(0, 0), 0);
        assert_eq!(split_price(5, 2), 5 + 2 * (1 << 20));
        assert_eq!(split_price(u64::MAX, 3), u64::MAX);
    }
}
