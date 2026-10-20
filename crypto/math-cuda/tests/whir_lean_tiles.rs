//! The lean rounds in TILE form — one pass over the message, then every round
//! on tables a tile's size smaller — against the share form they replace and
//! the materialised session both stand in for, on the device.
//!
//! Runs on a GPU box via `make test-math-cuda`. What is compared is every
//! round's two evaluations three ways (the session over the written-out
//! factors, `whir_lean_round` over the shares, the tile form), then the bound
//! tables and the message at a point, as VALUES (canonical limbs): the tile
//! form computes the same field elements by another sequence of operations.
//! The plan of each stack is `multilinear::lean_tiles::plan`, the production
//! planner; `tests/host_kat/whir_host_kat.cpp` checks the same kernels on the
//! host against a planner of its own.
//!
//! One test measures the card's high water, so every test here takes
//! [`DEVICE`] as `tests/whir_lean.rs` does.
//!
//! `the_tile_form_at_the_production_stack` is the measurement (ignored: a
//! `2^27` stack, about a minute): the share form against the tile form at the
//! shapes a block's chains have, values compared, milliseconds printed.

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
use math::field::goldilocks::GoldilocksField as Gl;
use math_cuda::whir_open::{LeanMessage, LeanRound0, LeanShare, OpeningSession, TileSpec};

type FE = FieldElement<Gl>;
type FE3 = FieldElement<Ext3>;

static DEVICE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn device() -> std::sync::MutexGuard<'static, ()> {
    DEVICE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

const P: u64 = 0xFFFF_FFFF_0000_0001;

fn canonical(limbs: &[u64]) -> Vec<u64> {
    limbs
        .iter()
        .map(|&v| if v >= P { v - P } else { v })
        .collect()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % P
    }

    fn ext(&mut self) -> FE3 {
        FE3::new([
            FE::from(self.next()),
            FE::from(self.next()),
            FE::from(self.next()),
        ])
    }

    fn point(&mut self, vars: usize) -> Vec<u64> {
        (0..vars).flat_map(|_| limbs(&self.ext())).collect()
    }
}

fn limbs(value: &FE3) -> [u64; 3] {
    let [a, b, c] = value.value();
    [*a.value(), *b.value(), *c.value()]
}

/// A stacked polynomial of `2^n`: its columns with their stack offsets, and
/// its weight's shares as `(offset, point, scale)`.
struct Stack {
    n: usize,
    columns: Vec<(Vec<u64>, usize)>,
    raw_shares: Vec<(usize, Vec<u64>, [u64; 3])>,
}

/// Columns `(offset, variables, point)` with random data and scales.
fn stack_of(n: usize, placed: Vec<(usize, usize, Vec<u64>)>, seed: u64) -> Stack {
    let mut rng = Rng(seed | 1);
    let mut columns = Vec::with_capacity(placed.len());
    let mut raw_shares = Vec::with_capacity(placed.len());
    for (offset, vars, point) in placed {
        assert_eq!(point.len(), vars * 3);
        columns.push(((0..1usize << vars).map(|_| rng.next()).collect(), offset));
        raw_shares.push((offset, point, limbs(&rng.ext())));
    }
    Stack {
        n,
        columns,
        raw_shares,
    }
}

/// Columns of at least `2^min_vars` rows at multiples of their own size, with
/// gaps, their points ending in one of `tails` tails of `min_vars`
/// coordinates and starting with one of a few heads a height; `tall` puts a
/// column of `2^(n−1)` first.
fn mixed_stack(n: usize, min_vars: usize, tails: usize, tall: bool, seed: u64) -> Stack {
    let mut rng = Rng(seed | 1);
    let tails: Vec<Vec<u64>> = (0..tails).map(|_| rng.point(min_vars)).collect();
    let mut heads: Vec<Vec<Vec<u64>>> = vec![Vec::new(); n + 1];
    let mut point = |m: usize, rng: &mut Rng| -> Vec<u64> {
        let pick = (rng.next() % 3) as usize;
        while heads[m].len() <= pick {
            let head = rng.point(m - min_vars);
            heads[m].push(head);
        }
        let mut point = heads[m][pick].clone();
        point.extend_from_slice(&tails[(rng.next() % tails.len() as u64) as usize]);
        point
    };
    let len = 1usize << n;
    let (mut cursor, mut stop) = (0usize, len);
    let mut placed = Vec::new();
    if tall {
        let at = if rng.next() & 1 == 1 { len / 2 } else { 0 };
        placed.push((at, n - 1, point(n - 1, &mut rng)));
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
        placed.push((at, m, point(m, &mut rng)));
        cursor = at + size;
    }
    placed.sort_by_key(|p| p.0);
    stack_of(n, placed, seed ^ 0x5EED)
}

/// The shares as the share form reads them: each point's table split at half
/// its variables, a pair of halves a share.
fn share_form(s: &Stack) -> (Vec<LeanShare>, Vec<u64>) {
    let mut shares = Vec::with_capacity(s.raw_shares.len());
    let mut eq: Vec<u64> = Vec::new();
    for (offset, point, scale) in &s.raw_shares {
        let vars = point.len() / 3;
        let lo_bits = vars / 2;
        let coordinates: Vec<FE3> = point
            .chunks_exact(3)
            .map(multilinear::gpu::ext3_from_raw::<Ext3>)
            .collect();
        let (hi, lo) = coordinates.split_at(vars - lo_bits);
        let hi_at = eq.len() / 3;
        for value in multilinear::eq::eq_evals(hi) {
            eq.extend_from_slice(&limbs(&value));
        }
        let lo_at = eq.len() / 3;
        for value in multilinear::eq::eq_evals(lo) {
            eq.extend_from_slice(&limbs(&value));
        }
        shares.push(LeanShare {
            stack_offset: *offset,
            num_vars: vars,
            lo_bits,
            hi_at,
            lo_at,
            scale: *scale,
        });
    }
    (shares, eq)
}

/// The shares as the tile form reads them, from the production planner.
fn tile_form(
    s: &Stack,
    rounds: usize,
    wide: bool,
    check: bool,
) -> (Vec<LeanShare>, Vec<u64>, TileSpec) {
    let plan = multilinear::lean_tiles::plan(&s.raw_shares, s.n, rounds)
        .expect("the stack takes the tile form");
    let shares = plan
        .shares
        .iter()
        .map(|share| LeanShare {
            stack_offset: share.stack_offset,
            num_vars: share.num_vars,
            lo_bits: plan.tile_bits,
            hi_at: share.hi_at,
            lo_at: share.lo_at,
            scale: share.scale,
        })
        .collect();
    let spec = TileSpec {
        tile_bits: plan.tile_bits,
        lean_rounds: rounds,
        slots: plan.slots,
        wide,
        check,
    };
    (shares, plan.eq, spec)
}

/// The opening's rule, `w·f`, lowered as the device session runs it.
fn product() -> multilinear::gpu::Lowered {
    let mut builder = multilinear::program::Builder::<Ext3>::new();
    let weight = builder.var(0);
    let message = builder.var(1);
    let root = builder.mul(weight, message);
    let program = builder.finish(root).expect("program");
    multilinear::gpu::lower(&program).expect("lowers")
}

/// Nodes 1 and 2, as `run_rounds` passes them.
fn nodes() -> Vec<u64> {
    vec![1, 0, 0, 2, 0, 0]
}

/// `rounds` rounds three ways at the same challenges — the materialised
/// session, the share form, the tile form — then the bound tables and the
/// message at a point. Returns the tile form's slots.
fn tiles_match(s: &Stack, resident: bool, rounds: usize, wide: bool, check: bool) -> usize {
    let what = format!(
        "2^{} {} {rounds} rounds{}{}",
        s.n,
        if resident { "resident" } else { "parts" },
        if wide { " wide" } else { "" },
        if check { " check" } else { "" }
    );
    let parts: Vec<(&[u64], usize)> = s
        .columns
        .iter()
        .map(|(column, offset)| (column.as_slice(), *offset))
        .collect();
    let store = math_cuda::columns::DeviceColumns::upload(
        &s.columns
            .iter()
            .map(|(c, _)| c.as_slice())
            .collect::<Vec<_>>(),
    )
    .expect("the card took the columns (needs a GPU)");
    let store_parts: Vec<(usize, usize)> = s
        .columns
        .iter()
        .enumerate()
        .map(|(i, (_, offset))| (i, *offset))
        .collect();
    let len = 1usize << s.n;
    let message = || {
        if resident {
            LeanMessage::Resident {
                store: &store,
                parts: &store_parts,
            }
        } else {
            LeanMessage::Parts(&parts)
        }
    };

    let mut session = if resident {
        OpeningSession::from_shares_and_resident(&s.raw_shares, len, &store, &store_parts)
    } else {
        OpeningSession::from_shares_and_parts(&s.raw_shares, len, &parts)
    }
    .expect("materialised session");
    let (old_shares, old_eq) = share_form(s);
    let mut old = LeanRound0::new(&old_shares, &old_eq, s.n, message())
        .expect("device")
        .expect("the share form takes this stack");
    let (new_shares, new_eq, spec) = tile_form(s, rounds, wide, check);
    let mut new = LeanRound0::with_tiles(&new_shares, &new_eq, s.n, message(), Some(spec))
        .expect("device")
        .expect("the tile form takes this stack");
    assert!(
        new.is_tiled() && !old.is_tiled(),
        "{what}: each opening is the form it was asked for"
    );
    let (checked_before, mismatched_before) = math_cuda::whir_open::tile_check_counts();

    let lowered = product();
    let mut reference = session
        .sumcheck(
            &lowered.nodes,
            &lowered.consts,
            lowered.num_slots,
            lowered.root_slot,
        )
        .expect("session rounds");
    let mut rng = Rng(0xABCD);
    for round in 1..=rounds {
        let want = reference.round(&nodes()).expect("materialised round");
        let share = old.round(&nodes()).expect("share-form round");
        let tile = new.round(&nodes()).expect("tile-form round");
        assert_eq!(
            canonical(&share),
            canonical(&want),
            "{what}: round {round}: the share form"
        );
        assert_eq!(
            canonical(&tile),
            canonical(&want),
            "{what}: round {round}: the tile form"
        );
        let alpha = limbs(&rng.ext());
        reference.fold(&alpha).expect("fold");
        old.bind(&alpha);
        new.bind(&alpha);
    }
    drop(reference);
    session.bound(rounds);
    let (checked, mismatched) = math_cuda::whir_open::tile_check_counts();
    if check {
        assert_eq!(
            (checked - checked_before, mismatched - mismatched_before),
            (rounds as u64, 0),
            "{what}: every round was compared with the share form, and none differed"
        );
    }

    let (want_w, want_f) = session.tables_to_host().expect("read back");
    let old_bound = old.materialize().expect("materialise");
    let new_bound = new.materialize().expect("materialise");
    for (name, bound) in [("share", &old_bound), ("tile", &new_bound)] {
        let (got_w, got_f) = bound.tables_to_host().expect("read back");
        assert_eq!(
            canonical(&got_w),
            canonical(&want_w),
            "{what}: the {name} form's bound weight"
        );
        assert_eq!(
            canonical(&got_f),
            canonical(&want_f),
            "{what}: the {name} form's bound message"
        );
    }
    let point: Vec<u64> = rng.point(s.n - rounds);
    assert_eq!(
        canonical(&new_bound.evaluate_message(&point).expect("evaluate")),
        canonical(&session.evaluate_message(&point).expect("evaluate")),
        "{what}: the message at a point"
    );
    spec.slots
}

/// ★ The tile form's rounds and bound tables are the share form's and the
/// materialised session's: stacks of one, two and many classes, with gaps and
/// a column taller than the blocks the rounds bind, three rounds and six, the
/// message read from host parts and from resident columns, the pass over the
/// message a thread a tile and a block a tile.
#[test]
fn the_tile_rounds_match_the_share_form_and_the_materialised_session() {
    let _device = device();
    let mut seen = std::collections::BTreeSet::new();
    for (case, (n, min_vars, tails, tall)) in [
        (16usize, 8usize, 1usize, false),
        (16, 8, 2, true),
        (17, 9, 8, false),
        (17, 7, 3, true),
        (18, 10, 2, false),
        (18, 11, 5, true),
    ]
    .into_iter()
    .enumerate()
    {
        let s = mixed_stack(n, min_vars, tails, tall, 0x71 + case as u64);
        for rounds in [math_cuda::whir_open::LEAN_ROUNDS_DEFAULT, 6] {
            if multilinear::lean_tiles::plan(&s.raw_shares, s.n, rounds).is_none() {
                continue;
            }
            for (resident, wide) in [(false, false), (true, false), (true, true), (false, true)] {
                seen.insert(tiles_match(&s, resident, rounds, wide, false));
            }
        }
    }
    println!("slots covered: {seen:?}");
    assert!(seen.contains(&1) && seen.contains(&2) && seen.iter().any(|slots| *slots >= 4));
}

/// The checking form compares every round with the share form on the card,
/// counts it, and finds none different.
#[test]
fn the_checking_form_compares_every_round_and_counts_it() {
    let _device = device();
    let s = mixed_stack(17, 8, 3, true, 0x99);
    tiles_match(&s, true, 3, false, true);
    tiles_match(&s, false, 3, false, true);
}

/// Clears the check's fault on the way out, whatever happens.
struct FaultOff;

impl Drop for FaultOff {
    fn drop(&mut self) {
        math_cuda::whir_open::force_tile_check_fault(false);
    }
}

/// ★ The comparison is load-bearing: with every tile-form value moved by one
/// before it is compared, every round counts as different — and the opening
/// still sends the share form's values, today's.
#[test]
fn the_checking_form_counts_a_round_that_differs_and_sends_todays_values() {
    let _device = device();
    let s = mixed_stack(17, 8, 3, true, 0x99);
    let parts: Vec<(&[u64], usize)> = s
        .columns
        .iter()
        .map(|(column, offset)| (column.as_slice(), *offset))
        .collect();
    let (old_shares, old_eq) = share_form(&s);
    let mut old = LeanRound0::new(&old_shares, &old_eq, s.n, LeanMessage::Parts(&parts))
        .expect("device")
        .expect("the share form takes this stack");
    let (new_shares, new_eq, spec) = tile_form(&s, 3, false, true);
    let mut new = LeanRound0::with_tiles(
        &new_shares,
        &new_eq,
        s.n,
        LeanMessage::Parts(&parts),
        Some(spec),
    )
    .expect("device")
    .expect("the tile form takes this stack");
    let (checked_before, mismatched_before) = math_cuda::whir_open::tile_check_counts();
    let fault = FaultOff;
    math_cuda::whir_open::force_tile_check_fault(true);
    let mut rng = Rng(0xABCD);
    for round in 1..=3 {
        let want = old.round(&nodes()).expect("share-form round");
        let sent = new.round(&nodes()).expect("checking round");
        assert_eq!(
            canonical(&sent),
            canonical(&want),
            "round {round}: a checking opening sends the share form's values"
        );
        let alpha = limbs(&rng.ext());
        old.bind(&alpha);
        new.bind(&alpha);
    }
    drop(fault);
    let (checked, mismatched) = math_cuda::whir_open::tile_check_counts();
    assert_eq!(
        (checked - checked_before, mismatched - mismatched_before),
        (3, 3),
        "every round compared, every one counted as different"
    );
}

/// The opening the tile form declines is the share form's: a stack with a
/// column shorter than the plan can cut is not planned, and `with_tiles`
/// refuses shares that are not split at the tile.
#[test]
fn a_stack_the_tile_form_does_not_take_opens_in_share_form() {
    let _device = device();
    let mut rng = Rng(0x33);
    // A column of four rows beside columns of 2^12: tiles of four positions
    // would hold more than the tables written after three rounds.
    let placed = vec![
        (0usize, 12usize, rng.point(12)),
        (1 << 12, 12, rng.point(12)),
        (1 << 13, 2, rng.point(2)),
    ];
    let s = stack_of(16, placed, 0x44);
    assert!(multilinear::lean_tiles::plan(&s.raw_shares, s.n, 3).is_none());
    let parts: Vec<(&[u64], usize)> = s
        .columns
        .iter()
        .map(|(column, offset)| (column.as_slice(), *offset))
        .collect();
    let (shares, eq) = share_form(&s);
    let spec = TileSpec {
        tile_bits: 6,
        lean_rounds: 3,
        slots: 2,
        wide: false,
        check: false,
    };
    let tiled = LeanRound0::with_tiles(&shares, &eq, s.n, LeanMessage::Parts(&parts), Some(spec))
        .expect("declining touches no device");
    assert!(
        tiled.is_none(),
        "shares split at half their variables are not the tile form's"
    );
    let lean = LeanRound0::new(&shares, &eq, s.n, LeanMessage::Parts(&parts))
        .expect("device")
        .expect("the share form takes this stack");
    assert!(!lean.is_tiled());
}

/// One opening's three rounds and its bound tables, with the pool's high
/// water over them. `tiles` is `None` for the share form.
fn high_water(
    s: &Stack,
    store: &math_cuda::columns::DeviceColumns,
    tiles: Option<bool>,
) -> (u64, u64) {
    let store_parts: Vec<(usize, usize)> = s
        .columns
        .iter()
        .enumerate()
        .map(|(i, (_, offset))| (i, *offset))
        .collect();
    let message = LeanMessage::Resident {
        store,
        parts: &store_parts,
    };
    let be = math_cuda::device::backend().expect("a device");
    math_cuda::device::drain_and_trim().expect("drain");
    math_cuda::device::reset_pool_high_water().expect("reset");
    let (held_before, _) = math_cuda::device::pool_used_bytes().expect("pool");
    let mut lean = match tiles {
        None => {
            let (shares, eq) = share_form(s);
            LeanRound0::new(&shares, &eq, s.n, message)
        }
        Some(wide) => {
            let (shares, eq, spec) = tile_form(s, 3, wide, false);
            LeanRound0::with_tiles(&shares, &eq, s.n, message, Some(spec))
        }
    }
    .expect("device")
    .expect("the opening takes this stack");
    let mut rng = Rng(0xABCD);
    for _ in 0..3 {
        lean.round(&nodes()).expect("round");
        lean.bind(&limbs(&rng.ext()));
    }
    let accounted = lean.device_bytes();
    let bound = lean.materialize().expect("materialise");
    be.ctx.synchronize().expect("sync");
    let (_, peak) = math_cuda::device::pool_used_bytes().expect("pool");
    drop(bound);
    (peak.saturating_sub(held_before), accounted)
}

/// ★ The tile form adds nothing to an opening's working set. It is freed
/// before the bound tables are allocated, and all it holds is smaller than
/// they are ([`TileSpec::fits`]), so the pool's high water over an opening's
/// rounds and its `materialize` is the share form's.
#[test]
fn the_tile_form_adds_nothing_to_an_openings_high_water() {
    let _device = device();
    // 16 columns of 2^18 at one point: 2^22, the shape of a block's chain at
    // a thirty-second of its size.
    let mut rng = Rng(0x22);
    let point = rng.point(18);
    let s = stack_of(
        22,
        (0..16).map(|c| (c << 18, 18, point.clone())).collect(),
        0x23,
    );
    let store = math_cuda::columns::DeviceColumns::upload(
        &s.columns
            .iter()
            .map(|(c, _)| c.as_slice())
            .collect::<Vec<_>>(),
    )
    .expect("the card took the columns (needs a GPU)");
    let (share_peak, share_accounted) = high_water(&s, &store, None);
    let (tile_peak, tile_accounted) = high_water(&s, &store, Some(false));
    let (wide_peak, _) = high_water(&s, &store, Some(true));
    let tables = 48u64 << (22 - 3);
    println!(
        "opening high water at 2^22, three rounds: share form {share_peak} B (accounted {share_accounted} B \
         before its tables), tile form {tile_peak} B (accounted {tile_accounted} B), wide {wide_peak} B; \
         the bound tables {tables} B"
    );
    assert!(
        tile_accounted > share_accounted,
        "the tile form holds its tables during the rounds"
    );
    assert!(tile_accounted - share_accounted < tables);
    // A page of slack: the pool rounds what it hands out.
    assert!(
        tile_peak <= share_peak + (64 << 10) && wide_peak <= share_peak + (64 << 10),
        "the tile form raised the opening's high water: {tile_peak} / {wide_peak} B against {share_peak} B"
    );
}

// ─────────────────────────────── the measurement ─────────────────────────────

/// Which opening a measured chain runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Form {
    Share,
    Tile,
    Wide,
}

impl Form {
    fn name(self) -> &'static str {
        match self {
            Form::Share => "share",
            Form::Tile => "tile",
            Form::Wide => "wide",
        }
    }
}

/// One opening's steps, in milliseconds: the construction (the position map;
/// for the tile form also its plan and its pass over the message), each of
/// the three rounds, and the bound tables' write.
#[derive(Clone, Copy, Default)]
struct Steps {
    setup: f64,
    rounds: [f64; 3],
    write: f64,
}

impl Steps {
    /// What the rounds cost a chain: everything but the write.
    fn rounds_ms(&self) -> f64 {
        self.setup + self.rounds.iter().sum::<f64>()
    }
}

/// What an opening leaves to compare: its rounds' evaluations, and the bound
/// weight and message, all as canonical limbs.
type Values = (Vec<Vec<u64>>, Vec<u64>, Vec<u64>);

/// The three challenges every measured opening binds.
fn bench_challenges() -> Vec<[u64; 3]> {
    let mut rng = Rng(0xBE7C);
    (0..3).map(|_| limbs(&rng.ext())).collect()
}

/// One opening from prepared shares: its steps, each waited for, its rounds'
/// values, and the session its tables are in.
fn timed_opening(
    shares: &[LeanShare],
    eq: &[u64],
    spec: Option<TileSpec>,
    n: usize,
    store: &math_cuda::columns::DeviceColumns,
    store_parts: &[(usize, usize)],
) -> (Steps, Vec<Vec<u64>>, OpeningSession) {
    let be = math_cuda::device::backend().expect("a device");
    let ms = |from: std::time::Instant| from.elapsed().as_secs_f64() * 1e3;
    let mut steps = Steps::default();
    let from = std::time::Instant::now();
    let mut lean = LeanRound0::with_tiles(
        shares,
        eq,
        n,
        LeanMessage::Resident {
            store,
            parts: store_parts,
        },
        spec,
    )
    .expect("device")
    .expect("the opening takes this stack");
    lean.synchronize().expect("sync");
    steps.setup = ms(from);
    let mut values = Vec::with_capacity(3);
    for (round, alpha) in bench_challenges().iter().enumerate() {
        let from = std::time::Instant::now();
        values.push(lean.round(&nodes()).expect("round"));
        steps.rounds[round] = ms(from);
        lean.bind(alpha);
    }
    let from = std::time::Instant::now();
    let bound = lean.materialize().expect("materialise");
    be.ctx.synchronize().expect("sync");
    steps.write = ms(from);
    (steps, values, bound)
}

fn median_min_max(mut values: Vec<f64>) -> String {
    values.sort_by(|a, b| a.partial_cmp(b).expect("a time"));
    format!(
        "{:.3}/{:.3}/{:.3}",
        values[values.len() / 2],
        values[0],
        values[values.len() - 1]
    )
}

/// The shapes a block's chains have, at the production stack `2^27`:
/// `S1` 64 columns of `2^21` rows at one point (a chain of full tables);
/// `S2` 2,048 columns of `2^16` at one point (a keccak-round chain);
/// `S3` sixteen columns of `2^20` at one point and 56 of `2^21` at another, so
/// every column of tiles meets two classes (a chain holding a comparison
/// table); `S4` eight points over eight blocks of eight columns of `2^21`, so
/// every column of tiles meets eight.
fn production_shape(name: &str) -> Stack {
    let mut rng = Rng(0x5A9E ^ name.len() as u64 ^ (name.as_bytes()[1] as u64) << 8);
    let placed: Vec<(usize, usize, Vec<u64>)> = match name {
        "S1" => {
            let point = rng.point(21);
            (0..64).map(|c| (c << 21, 21, point.clone())).collect()
        }
        "S2" => {
            let point = rng.point(16);
            (0..2048).map(|c| (c << 16, 16, point.clone())).collect()
        }
        "S3" => {
            let (short, tall) = (rng.point(20), rng.point(21));
            (0..16)
                .map(|c| (c << 20, 20, short.clone()))
                .chain((8..64).map(|c| (c << 21, 21, tall.clone())))
                .collect()
        }
        "S4" => {
            let points: Vec<Vec<u64>> = (0..8).map(|_| rng.point(21)).collect();
            (0..64)
                .map(|c| (c << 21, 21, points[c / 8].clone()))
                .collect()
        }
        other => panic!("no shape {other}"),
    };
    stack_of(27, placed, 0x27)
}

/// ★ THE MEASUREMENT: one chain's lean rounds at the production stack, the
/// share form against the tile form (a thread a tile, and a block a tile), at
/// the four shapes. Three warm-ups and twenty timed openings a form, every
/// step waited for. Values first: the three rounds' evaluations and both
/// bound tables equal across the forms, as canonical limbs. Then, per shape
/// and form, `TILE BENCH` lines (median/min/max milliseconds), the pool's
/// high water over an opening (`TILE MEM`), and the host's planning time
/// (`TILE PLAN`). Whoever runs it reads its verdict from these lines;
/// nothing here asserts a time.
#[test]
#[ignore = "the measurement: a 2^27 stack on the card, about a minute"]
fn the_tile_form_at_the_production_stack() {
    let _device = device();
    const WARM: usize = 3;
    const TIMED: usize = 20;
    for name in ["S1", "S2", "S3", "S4"] {
        let s = production_shape(name);
        let store = math_cuda::columns::DeviceColumns::upload(
            &s.columns
                .iter()
                .map(|(c, _)| c.as_slice())
                .collect::<Vec<_>>(),
        )
        .expect("the card took the columns (needs a GPU)");
        let store_parts: Vec<(usize, usize)> = s
            .columns
            .iter()
            .enumerate()
            .map(|(i, (_, offset))| (i, *offset))
            .collect();

        // The host's part, timed apart: it is in neither form's card time.
        let plan_ms = |f: &dyn Fn()| {
            let times: Vec<f64> = (0..5)
                .map(|_| {
                    let from = std::time::Instant::now();
                    f();
                    from.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            median_min_max(times)
        };
        println!(
            "TILE PLAN shape={name} share_plan_ms={} tile_plan_ms={}",
            plan_ms(&|| drop(share_form(&s))),
            plan_ms(&|| drop(tile_form(&s, 3, false, false)))
        );
        let (share_shares, share_eq) = share_form(&s);
        let (tile_shares, tile_eq, spec) = tile_form(&s, 3, false, false);
        let wide = TileSpec { wide: true, ..spec };

        let mut reference: Option<Values> = None;
        let (mut rounds_equal, mut tables_equal) = (true, true);
        for form in [Form::Share, Form::Tile, Form::Wide] {
            let (shares, eq, spec) = match form {
                Form::Share => (&share_shares, &share_eq, None),
                Form::Tile => (&tile_shares, &tile_eq, Some(spec)),
                Form::Wide => (&tile_shares, &tile_eq, Some(wide)),
            };
            let mut all: Vec<Steps> = Vec::with_capacity(TIMED);
            for rep in 0..WARM + TIMED {
                let (steps, values, bound) =
                    timed_opening(shares, eq, spec, s.n, &store, &store_parts);
                if rep >= WARM {
                    all.push(steps);
                }
                if rep == 0 {
                    let values: Vec<Vec<u64>> = values.iter().map(|v| canonical(v)).collect();
                    let (w, f) = bound.tables_to_host().expect("read back");
                    let (w, f) = (canonical(&w), canonical(&f));
                    match &reference {
                        None => reference = Some((values, w, f)),
                        Some((want, want_w, want_f)) => {
                            rounds_equal &= &values == want;
                            tables_equal &= &w == want_w && &f == want_f;
                        }
                    }
                }
                drop(bound);
            }
            let column =
                |pick: &dyn Fn(&Steps) -> f64| median_min_max(all.iter().map(pick).collect());
            println!(
                "TILE BENCH shape={name} form={} reps={TIMED} setup_ms={} round1_ms={} round2_ms={} \
                 round3_ms={} write_ms={} rounds_ms={} chain_ms={}",
                form.name(),
                column(&|t| t.setup),
                column(&|t| t.rounds[0]),
                column(&|t| t.rounds[1]),
                column(&|t| t.rounds[2]),
                column(&|t| t.write),
                column(&|t| t.rounds_ms()),
                column(&|t| t.rounds_ms() + t.write),
            );
        }
        println!(
            "TILE VALUES shape={name} rounds_equal={rounds_equal} tables_equal={tables_equal} \
             tile_bits={} slots={}",
            spec.tile_bits, spec.slots
        );
        let (share_peak, _) = high_water(&s, &store, None);
        let (tile_peak, _) = high_water(&s, &store, Some(false));
        let (wide_peak, _) = high_water(&s, &store, Some(true));
        println!(
            "TILE MEM shape={name} share_peak={share_peak} tile_peak={tile_peak} wide_peak={wide_peak}"
        );
        assert!(
            rounds_equal && tables_equal,
            "{name}: the forms' values differ"
        );
    }
}
