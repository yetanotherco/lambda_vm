//! The plan of a stacked opening's lean rounds in TILE form.
//!
//! A stacked opening's weight is, inside each column, `scale · eq(z, row)`.
//! Cut the stack into tiles of `2^u` positions, `u` no more than the smallest
//! column's variables: columns sit at multiples of their own size, so a tile
//! lies inside one column or one gap, and there
//!
//! ```text
//! w(T, j) = ω(T) · eq(λ, j),   ω(T) = scale · eq(z_hi, T − the column's first tile)
//! ```
//!
//! with `λ` the point's last `u` coordinates and `z_hi` the rest. Columns whose
//! `λ` are equal are one CLASS and share one table. One pass over the message
//! then gives every tile's value at each class it can meet, and every lean
//! round runs on tables `2^u` times smaller than the stack — the same round
//! polynomials (`whir_tile_*` in `math-cuda/kernels/whir_fold.cu` writes the
//! identity out; [`tests`] checks it against the host prover's own rounds).
//!
//! This module is the host's part: which `u`, which classes, the tables, and
//! whether the form fits. It reads no device and is built without one.

use std::collections::BTreeMap;

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;

/// The widest tile: `2^10` positions. A column of `2^20` or `2^21` rows splits
/// its point's table there today (`num_vars / 2`), so for those the two halves
/// are the ones the share form already builds.
pub const TILE_BITS_CAP: usize = 10;

/// Slot tables the kernels have room for (`WHIR_TILE_MAX_SLOTS`).
pub const MAX_SLOTS: usize = 8;

/// One column as the tile form reads it: its point's table split at the plan's
/// tile, the high half at `hi_at` and its class's low half at `lo_at`, in
/// extension elements into [`TilePlan::eq`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileShare {
    pub stack_offset: usize,
    pub num_vars: usize,
    pub hi_at: usize,
    pub lo_at: usize,
    pub scale: [u64; 3],
}

/// A stacked opening's lean rounds, planned in tile form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TilePlan {
    /// A tile is `2^tile_bits` stack positions.
    pub tile_bits: usize,
    /// Classes in the stack: distinct tails of `tile_bits` coordinates.
    pub classes: usize,
    /// Slot tables to hold: the most classes a column of tiles can meet,
    /// `min(classes, 2^lean_rounds)`.
    pub slots: usize,
    /// The columns, by stack offset.
    pub shares: Vec<TileShare>,
    /// Every `eq` half, three limbs an element.
    pub eq: Vec<u64>,
}

/// Device bytes the tile form holds through its rounds: a `u16` a tile, each
/// column of tiles' slots, the two slot tables at three limbs a tile and
/// slot, a challenge. `math_cuda::whir_open::TileSpec::device_bytes`, which a
/// `cuda` build's test holds this to.
pub const fn device_bytes(
    n_stack: usize,
    tile_bits: usize,
    lean_rounds: usize,
    slots: usize,
) -> u64 {
    write_bytes(n_stack, tile_bits, lean_rounds, slots)
        + ((24 * slots as u64) << (n_stack - tile_bits))
}

/// What it keeps beside the bound tables while it writes them: all of the
/// above but the message's slot tables (`TileSpec::write_bytes`).
pub const fn write_bytes(
    n_stack: usize,
    tile_bits: usize,
    lean_rounds: usize,
    slots: usize,
) -> u64 {
    let tiles = 1u64 << (n_stack - tile_bits);
    let columns = tiles >> lean_rounds;
    let slots = slots as u64;
    2 * tiles + 8 * slots * columns + 24 * slots * tiles + 24
}

/// Whether the form fits a stack of `2^n_stack`: the rounds bind tile
/// variables only, the kernels have the slots, and the opening never holds
/// more than the share form does while it writes its tables — the position
/// map (a `u16` a position, which the tile form does not allocate) beside
/// the bound tables. Through the rounds all the form holds is no larger than
/// those tables; through the write what it keeps is no larger than the map.
pub const fn fits(n_stack: usize, tile_bits: usize, lean_rounds: usize, slots: usize) -> bool {
    tile_bits <= n_stack
        && lean_rounds >= 1
        && lean_rounds <= n_stack - tile_bits
        && slots >= 1
        && slots <= MAX_SLOTS
        && device_bytes(n_stack, tile_bits, lean_rounds, slots) <= 48u64 << (n_stack - lean_rounds)
        && write_bytes(n_stack, tile_bits, lean_rounds, slots) <= 2u64 << n_stack
}

/// The tile form of the opening whose weight is `shares` — `(stack offset,
/// point as limbs, scale)` per column — over `2^n_stack`, for `lean_rounds`
/// rounds. `None` when the stack does not take it: a column off its own
/// alignment, or so small (or the classes so many) that the form would hold
/// more than the tables written after it.
pub fn plan(
    shares: &[(usize, Vec<u64>, [u64; 3])],
    n_stack: usize,
    lean_rounds: usize,
) -> Option<TilePlan> {
    if shares.is_empty() || lean_rounds == 0 {
        return None;
    }
    let tile_bits = shares
        .iter()
        .map(|(_, point, _)| point.len() / 3)
        .min()?
        .min(TILE_BITS_CAP);
    let mut order: Vec<usize> = (0..shares.len()).collect();
    order.sort_by_key(|i| shares[*i].0);

    let mut eq: Vec<u64> = Vec::new();
    let mut table = |coordinates: &[u64]| -> usize {
        let point: Vec<FieldElement<Ext3>> = coordinates
            .chunks_exact(3)
            .map(crate::gpu::ext3_from_raw::<Ext3>)
            .collect();
        let at = eq.len() / 3;
        for value in crate::eq::eq_evals(&point) {
            let [a, b, c] = value.value();
            eq.extend_from_slice(&[*a.value(), *b.value(), *c.value()]);
        }
        at
    };
    // One table per distinct head and per distinct tail, limb for limb: a
    // table's columns share both, and a class is a tail.
    let mut heads: BTreeMap<&[u64], usize> = BTreeMap::new();
    let mut tails: BTreeMap<&[u64], usize> = BTreeMap::new();
    let mut planned = Vec::with_capacity(shares.len());
    for &i in &order {
        let (offset, point, scale) = &shares[i];
        if !point.len().is_multiple_of(3) {
            return None;
        }
        let num_vars = point.len() / 3;
        if num_vars > n_stack || offset & ((1usize << num_vars) - 1) != 0 {
            return None;
        }
        let (head, tail) = point.split_at((num_vars - tile_bits) * 3);
        let hi_at = match heads.get(head) {
            Some(at) => *at,
            None => {
                let at = table(head);
                heads.insert(head, at);
                at
            }
        };
        let lo_at = match tails.get(tail) {
            Some(at) => *at,
            None => {
                let at = table(tail);
                tails.insert(tail, at);
                at
            }
        };
        planned.push(TileShare {
            stack_offset: *offset,
            num_vars,
            hi_at,
            lo_at,
            scale: *scale,
        });
    }
    let classes = tails.len();
    let slots = classes.min(1usize.checked_shl(lean_rounds as u32)?);
    if !fits(n_stack, tile_bits, lean_rounds, slots) {
        return None;
    }
    Some(TilePlan {
        tile_bits,
        classes,
        slots,
        shares: planned,
        eq,
    })
}

#[cfg(test)]
mod tests {
    //! The tile form against the host prover's own rounds: the weight written
    //! out by [`weight_table`](crate::stacked_eval::weight_table), the message
    //! lifted, [`sumcheck::prove_rounds`](crate::sumcheck::prove_rounds) over
    //! their product — what a stacked opening does without a device — and the
    //! same rounds over the plan's slot tables, on one transcript. A second
    //! source for `whir_host_kat.cpp`'s check of the kernels, in another
    //! language and with THIS planner, which the kernels' test does not use.

    use super::*;
    use crate::mle::Mle;
    use crate::poly::Composed;
    use crate::stacked_eval::{WeightShare, weight_table};
    use crate::sumcheck;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::goldilocks::GoldilocksField as Gl;

    type FE = FieldElement<Gl>;
    type EE = FieldElement<Ext3>;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn ext(&mut self) -> EE {
            EE::new([
                FE::from(self.next()),
                FE::from(self.next()),
                FE::from(self.next()),
            ])
        }
    }

    fn limbs(value: &EE) -> [u64; 3] {
        crate::gpu::ext3_raw(value).expect("an ext3 element")
    }

    fn from_limbs(eq: &[u64], at: usize) -> EE {
        crate::gpu::ext3_from_raw::<Ext3>(&eq[at * 3..at * 3 + 3])
    }

    /// A stack of `2^n`: columns of at least `2^min_vars` rows at multiples of
    /// their own size, with gaps, their points drawn from `tails` tails of
    /// `tail_vars` coordinates (so that many classes at most) and a few heads
    /// a height; `tall` puts a column of `2^(n−1)` first.
    struct Stack {
        n: usize,
        columns: Vec<(usize, Vec<EE>, EE)>,
        message: Vec<FE>,
    }

    fn stack(
        n: usize,
        min_vars: usize,
        tail_vars: usize,
        tails: usize,
        tall: bool,
        seed: u64,
    ) -> Stack {
        let mut rng = Rng(seed | 1);
        let tails: Vec<Vec<EE>> = (0..tails)
            .map(|_| (0..tail_vars).map(|_| rng.ext()).collect())
            .collect();
        let mut heads: Vec<Vec<Vec<EE>>> = vec![Vec::new(); n + 1];
        let mut point = |m: usize, rng: &mut Rng| -> Vec<EE> {
            let pick = (rng.next() % 3) as usize;
            while heads[m].len() <= pick {
                let head = (0..m - tail_vars).map(|_| rng.ext()).collect();
                heads[m].push(head);
            }
            let mut point = heads[m][pick].clone();
            point.extend(
                tails[(rng.next() % tails.len() as u64) as usize]
                    .iter()
                    .cloned(),
            );
            point
        };
        let len = 1usize << n;
        let (mut cursor, mut stop) = (0usize, len);
        let mut columns = Vec::new();
        if tall {
            let at = if rng.next() & 1 == 1 { len / 2 } else { 0 };
            columns.push((at, point(n - 1, &mut rng), rng.ext()));
            if at == 0 {
                cursor = len / 2;
            } else {
                stop = len / 2;
            }
        }
        loop {
            let top = (n - 2).max(min_vars);
            let m = min_vars + (rng.next() as usize) % (top - min_vars + 1);
            let size = 1usize << m;
            let mut at = cursor.div_ceil(size) * size;
            if rng.next().is_multiple_of(5) {
                at += size;
            }
            if at + size > stop {
                break;
            }
            columns.push((at, point(m, &mut rng), rng.ext()));
            cursor = at + size;
        }
        let mut message = vec![FE::zero(); len];
        for (at, point, _) in &columns {
            for cell in &mut message[*at..*at + (1usize << point.len())] {
                *cell = FE::from(rng.next());
            }
        }
        Stack {
            n,
            columns,
            message,
        }
    }

    /// The host prover's rounds over `(weight, message)`: the evaluations each
    /// round sends, the challenges, and the two tables as the rounds leave
    /// them.
    fn host_rounds(
        weight: Mle<Ext3>,
        message: Mle<Ext3>,
        rounds: usize,
    ) -> (Vec<Vec<EE>>, Vec<EE>, Mle<Ext3>, Mle<Ext3>) {
        let mut transcript = DefaultTranscript::<Ext3>::new(b"lean-tiles");
        let mut poly = Composed::new(vec![weight, message], |v: &[EE]| &v[0] * &v[1], 2).unwrap();
        let (proofs, challenges) =
            sumcheck::prove_rounds(&mut poly, rounds, &mut transcript).unwrap();
        let mut parts = poly.into_polys();
        let message = parts.pop().unwrap();
        let weight = parts.pop().unwrap();
        (
            proofs.into_iter().map(|p| p.evaluations).collect(),
            challenges,
            weight,
            message,
        )
    }

    /// One stack: the plan's slot tables built here, their rounds on the same
    /// transcript, and the bound weight rebuilt from the bound slot tables.
    /// Returns the plan's `(classes, slots)`, or `None` when the stack does
    /// not take the form.
    fn check(s: &Stack, rounds: usize) -> Option<(usize, usize)> {
        let raw: Vec<(usize, Vec<u64>, [u64; 3])> = s
            .columns
            .iter()
            .map(|(at, point, scale)| (*at, point.iter().flat_map(limbs).collect(), limbs(scale)))
            .collect();
        let plan = plan(&raw, s.n, rounds)?;
        let u = plan.tile_bits;
        let tiles = 1usize << (s.n - u);
        let columns = tiles >> rounds;

        // Today: the weight as a table, the message lifted, the host's rounds.
        let shares: Vec<WeightShare<'_, Ext3>> = s
            .columns
            .iter()
            .map(|(at, point, scale)| WeightShare {
                offset: *at,
                point,
                scale: *scale,
            })
            .collect();
        let weight = weight_table(&shares, s.n).unwrap();
        let message =
            Mle::new(s.message.iter().map(|v| v.to_extension::<Ext3>()).collect()).unwrap();
        let (want, want_challenges, want_weight, _) = host_rounds(weight, message, rounds);

        // The tile form. Which share holds a tile, then each column of tiles'
        // slots in the order its tiles meet their classes.
        let share_of = |tile: usize| -> Option<&TileShare> {
            plan.shares.iter().find(|share| {
                let first = share.stack_offset >> u;
                tile >= first && tile < first + (1usize << (share.num_vars - u))
            })
        };
        let mut slot_lo: Vec<Vec<usize>> = vec![Vec::new(); columns];
        let mut omega = vec![vec![EE::zero(); tiles]; plan.slots];
        let mut phi = vec![vec![EE::zero(); tiles]; plan.slots];
        for (column, slots) in slot_lo.iter_mut().enumerate() {
            for q in 0..1usize << rounds {
                let tile = q * columns + column;
                let Some(share) = share_of(tile) else {
                    continue;
                };
                let slot = match slots.iter().position(|lo| *lo == share.lo_at) {
                    Some(slot) => slot,
                    None => {
                        slots.push(share.lo_at);
                        slots.len() - 1
                    }
                };
                assert!(
                    slot < plan.slots,
                    "a column of tiles met more classes than slots"
                );
                omega[slot][tile] = crate::gpu::ext3_from_raw::<Ext3>(&share.scale)
                    * from_limbs(&plan.eq, share.hi_at + tile - (share.stack_offset >> u));
            }
        }
        for tile in 0..tiles {
            for (slot, lo_at) in slot_lo[tile % columns].iter().enumerate() {
                phi[slot][tile] = (0..1usize << u)
                    .map(|j| &s.message[(tile << u) + j] * &from_limbs(&plan.eq, lo_at + j))
                    .fold(EE::zero(), |acc, term| acc + term);
            }
        }
        let slots = plan.slots;
        let mut tables: Vec<Mle<Ext3>> = Vec::with_capacity(2 * slots);
        for table in omega.into_iter().chain(phi) {
            tables.push(Mle::new(table).unwrap());
        }
        let mut transcript = DefaultTranscript::<Ext3>::new(b"lean-tiles");
        let mut poly = Composed::new(
            tables,
            move |v: &[EE]| (0..slots).fold(EE::zero(), |acc, k| acc + &v[k] * &v[slots + k]),
            2,
        )
        .unwrap();
        let (proofs, challenges) =
            sumcheck::prove_rounds(&mut poly, rounds, &mut transcript).unwrap();
        let got: Vec<Vec<EE>> = proofs.into_iter().map(|p| p.evaluations).collect();
        assert_eq!(got, want, "the tiles' round polynomials are the opening's");
        assert_eq!(challenges, want_challenges);

        // The weight the rounds leave, from the bound `Ω` and the classes'
        // tables.
        let bound = poly.into_polys();
        for (y, want) in want_weight.evals().iter().enumerate() {
            let (tile, j) = (y >> u, y & ((1usize << u) - 1));
            let got = slot_lo[tile % columns].iter().enumerate().fold(
                EE::zero(),
                |acc, (slot, lo_at)| {
                    acc + &bound[slot].evals()[tile] * from_limbs(&plan.eq, lo_at + j)
                },
            );
            assert_eq!(&got, want, "the bound weight at {y}");
        }
        Some((plan.classes, plan.slots))
    }

    /// ★ The tile form's rounds are the host prover's, on stacks of one to
    /// eight classes, with gaps and a column taller than the blocks the
    /// rounds bind, for one to four rounds.
    #[test]
    fn the_tile_form_sends_the_host_provers_rounds() {
        let mut by_slots = [0usize; MAX_SLOTS + 1];
        let mut planned = 0;
        for trial in 0..96u64 {
            let n = 11 + (trial % 3) as usize; // 11..13
            let tail_vars = 4 + (trial / 3 % 3) as usize; // the smallest column: 4..6
            let rounds = 1 + (trial / 9 % 4) as usize; // 1..4
            let tails = 1 + (trial % 9) as usize;
            let s = stack(
                n,
                tail_vars,
                tail_vars,
                tails,
                trial % 5 == 0,
                0x9E37_79B9 + trial,
            );
            if let Some((_, slots)) = check(&s, rounds) {
                by_slots[slots] += 1;
                planned += 1;
            }
        }
        println!("{planned} stacks planned; by slots {by_slots:?}");
        assert!(by_slots[1] > 0 && by_slots[2] > 0 && by_slots[4] > 0 && by_slots[8] > 0);
    }

    /// A column shorter than the cap sets the tile, and columns of one height
    /// in a table share both halves; the classes are the distinct tails.
    #[test]
    fn the_plan_cuts_at_the_smallest_column_and_names_a_class_by_its_tail() {
        let mut rng = Rng(0xC0FFEE);
        let table: Vec<u64> = (0..12).flat_map(|_| limbs(&rng.ext())).collect();
        let tail = table[(12 - 7) * 3..].to_vec();
        // Another height whose last seven coordinates are the table's.
        let mut short: Vec<u64> = (0..2).flat_map(|_| limbs(&rng.ext())).collect();
        short.extend_from_slice(&tail);
        let other: Vec<u64> = (0..7).flat_map(|_| limbs(&rng.ext())).collect();
        let scale = limbs(&rng.ext());
        let shares = vec![
            (0usize, table.clone(), scale),
            (1 << 12, table.clone(), scale),
            (1 << 13, short, scale),
            ((1 << 13) + (1 << 9), other, scale),
        ];
        let plan = plan(&shares, 16, 3).expect("the stack takes the form");
        assert_eq!(plan.tile_bits, 7, "the smallest column has seven variables");
        assert_eq!((plan.classes, plan.slots), (2, 2));
        assert_eq!(plan.shares[0].hi_at, plan.shares[1].hi_at);
        assert_eq!(plan.shares[0].lo_at, plan.shares[1].lo_at);
        assert_eq!(
            plan.shares[2].lo_at, plan.shares[0].lo_at,
            "one tail, one class"
        );
        assert_ne!(plan.shares[2].hi_at, plan.shares[0].hi_at);
        assert_ne!(plan.shares[3].lo_at, plan.shares[0].lo_at);
        // 2^5 + 2^7 (the table), 2^2 (the short head), 2^0 + 2^7 (the other).
        assert_eq!(plan.eq.len() / 3, 32 + 128 + 4 + 1 + 128);

        // Tall columns are cut at the cap.
        let tall: Vec<u64> = (0..15).flat_map(|_| limbs(&rng.ext())).collect();
        let plan = super::plan(&[(0, tall, scale)], 16, 3).unwrap();
        assert_eq!((plan.tile_bits, plan.slots), (TILE_BITS_CAP, 1));
        assert_eq!(plan.eq.len() / 3, 32 + 1024);
    }

    /// What does not take the form: a column off its own alignment, no column,
    /// no round, and a stack whose form would hold more than the tables
    /// written after it.
    #[test]
    fn the_plan_declines_what_the_form_cannot_hold() {
        let mut rng = Rng(7);
        let scale = limbs(&rng.ext());
        let mut point =
            |vars: usize| -> Vec<u64> { (0..vars).flat_map(|_| limbs(&rng.ext())).collect() };
        assert!(plan(&[], 16, 3).is_none());
        assert!(plan(&[(0, point(8), scale)], 16, 0).is_none());
        assert!(
            plan(&[(3 << 6, point(8), scale)], 16, 3).is_none(),
            "off its alignment"
        );
        // A column of two variables beside a tall one of its class: tiles of
        // four positions, and the form at three rounds would be over twice the
        // tables after it.
        let tall = point(12);
        let small = tall[10 * 3..].to_vec();
        let stack = [(0, tall.clone(), scale), (1 << 12, small, scale)];
        assert!(plan(&stack, 16, 3).is_none());
        // A column of three variables, one round: through the round the form
        // is under the tables after it, but what the write keeps beside them
        // would be more than the share form's position map.
        let three = tall[9 * 3..].to_vec();
        let stack = [(0, tall.clone(), scale), (1 << 12, three, scale)];
        assert!(plan(&stack, 16, 1).is_none());
        assert!(device_bytes(16, 3, 1, 1) <= 48 << 15 && write_bytes(16, 3, 1, 1) > 2 << 16);
        // Four variables fit, at one round and at three.
        let four = tall[8 * 3..].to_vec();
        let stack = [(0, tall, scale), (1 << 12, four, scale)];
        for rounds in [1, 3] {
            assert_eq!(
                plan(&stack, 16, rounds).map(|p| (p.tile_bits, p.slots)),
                Some((4, 1))
            );
        }
        // Nine classes at four rounds are more slots than the kernels have.
        let nine: Vec<_> = (0..9).map(|c| (c << 12, point(12), scale)).collect();
        assert!(plan(&nine, 16, 4).is_none());
        // At three rounds a column of tiles meets eight of them at most.
        assert_eq!(
            plan(&nine, 16, 3).map(|p| (p.classes, p.slots)),
            Some((9, 8))
        );
    }

    /// The two stacks `tests/whir_lean_device.rs` proves on the card: the first,
    /// with a column of eight rows and nine points, is declined; the second, no
    /// column under `2^9` rows and four tables, is planned with four slots.
    #[test]
    fn the_device_tests_two_stacks_are_one_declined_and_one_planned() {
        let stack = |heights: &[usize], tables: &[usize]| -> Vec<(usize, Vec<u64>, [u64; 3])> {
            let layout = crate::stacking::StackedLayout::build(heights, 16).unwrap();
            layout
                .placements()
                .iter()
                .zip(heights.iter().zip(tables))
                .map(|(place, (&h, &table))| {
                    let point = (0..h)
                        .flat_map(|j| limbs(&EE::from(101 + 13 * table as u64 + j as u64)))
                        .collect();
                    (place.offset, point, [1, 0, 0])
                })
                .collect()
        };
        let first = stack(
            &[13, 13, 13, 12, 12, 10, 8, 6, 3],
            &[0, 1, 2, 3, 4, 5, 6, 7, 8],
        );
        assert!(plan(&first, 16, 3).is_none());
        let second = stack(
            &[13, 13, 13, 12, 12, 12, 11, 9, 9],
            &[0, 0, 0, 1, 1, 1, 2, 3, 3],
        );
        let planned = plan(&second, 16, 3).expect("the second stack takes the form");
        assert_eq!(
            (planned.tile_bits, planned.classes, planned.slots),
            (9, 4, 4)
        );
        assert!(plan(&second, 16, 6).is_some(), "and at six rounds");
    }

    /// ★ The opening's working set does not grow: wherever the form fits, all
    /// it holds through the rounds is no larger than the tables written once
    /// they are spent, and what it keeps through the write is no larger than
    /// the share form's position map, which it never allocates. At the
    /// production stack and three rounds: 6.4 MiB a slot against 768 MiB, and
    /// 3.3 MiB a slot against 256 MiB.
    #[test]
    fn the_form_holds_no_more_than_the_tables_written_after_it() {
        for n in [16usize, 22, 25, 27, 28] {
            for lean_rounds in 1..=6usize {
                for tile_bits in 0..=TILE_BITS_CAP.min(n) {
                    for slots in 1..=MAX_SLOTS {
                        if fits(n, tile_bits, lean_rounds, slots) {
                            let after = 48u64 << (n - lean_rounds);
                            assert!(device_bytes(n, tile_bits, lean_rounds, slots) <= after);
                            // Beside those tables, no more than the share
                            // form's position map.
                            assert!(write_bytes(n, tile_bits, lean_rounds, slots) <= 2u64 << n);
                        }
                    }
                }
            }
        }
        let mib = |bytes: u64| bytes as f64 / (1u64 << 20) as f64;
        let after = 48u64 << (27 - 3);
        assert_eq!(after, 768 << 20);
        for (slots, want) in [(1usize, 6.4), (2, 12.5), (8, 49.3)] {
            let held = device_bytes(27, 10, 3, slots);
            println!(
                "2^27, tiles of 2^10, three rounds, {slots} slot(s): {:.1} MiB",
                mib(held)
            );
            assert!((mib(held) - want).abs() < 0.1, "{} MiB", mib(held));
            assert!(fits(27, 10, 3, slots));
            // Through the write: `Ω`, the tile map and the slots, against the
            // 256 MiB map the share form holds there.
            let kept = write_bytes(27, 10, 3, slots);
            assert!(kept * 8 < 2 << 27, "{} MiB", mib(kept));
        }
        // The smallest tile that fits at the production stack and three
        // rounds: 2^4 positions for one slot, 2^7 for eight.
        let smallest = |slots| (0..=TILE_BITS_CAP).find(|u| fits(27, *u, 3, slots));
        assert_eq!(
            (smallest(1), smallest(2), smallest(8)),
            (Some(4), Some(5), Some(7))
        );
    }

    /// The device's own rule is this one.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_device_fits_what_the_plan_fits() {
        assert_eq!(MAX_SLOTS, math_cuda::whir_open::TILE_MAX_SLOTS);
        for n in [12usize, 20, 27] {
            for lean_rounds in 0..=7usize {
                for tile_bits in 0..=n {
                    for slots in 0..=MAX_SLOTS + 1 {
                        let spec = math_cuda::whir_open::TileSpec {
                            tile_bits,
                            lean_rounds,
                            slots,
                            wide: false,
                            check: false,
                        };
                        assert_eq!(
                            spec.fits(n),
                            fits(n, tile_bits, lean_rounds, slots),
                            "2^{n}, tiles 2^{tile_bits}, {lean_rounds} rounds, {slots} slots"
                        );
                        if spec.fits(n) {
                            assert_eq!(
                                spec.device_bytes(n),
                                device_bytes(n, tile_bits, lean_rounds, slots)
                            );
                        }
                    }
                }
            }
        }
    }
}
