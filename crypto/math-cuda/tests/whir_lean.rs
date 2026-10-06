//! The lean first rounds of a WHIR opening against the materialised session
//! they replace, on the device.
//!
//! Runs on a GPU box via `make test-math-cuda`. The reference is today's path
//! itself — [`OpeningSession`] built from the same shares and parts, rounds
//! through its [`SumcheckSession`] with the product program lowered by
//! `multilinear::gpu::lower` — so what is compared is every round's two
//! evaluations, the bound tables once the rounds are spent, and the message's
//! value at an out-of-domain-like point, as VALUES (canonical limbs): the lean
//! path computes the same field elements by a different sequence of
//! operations. `tests/host_kat/whir_host_kat.cpp` pins the same arithmetic on
//! the host.
//!
//! The two working-set tests (the lean opening's, and the fused fold's) are
//! here too, and every test in this binary takes [`DEVICE`]: they read the
//! driver's free memory before and after, and a neighbour allocating in the
//! same process at the same time would be counted as theirs. The parity tests
//! elsewhere run in parallel as usual; this binary does not.

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
use math::field::goldilocks::GoldilocksField as Gl;
use math_cuda::whir_open::{LeanMessage, LeanRound0, LeanShare, OpeningSession};

type FE = FieldElement<Gl>;
type FE3 = FieldElement<Ext3>;

/// Held by every test here, so a driver measurement never overlaps another
/// test's allocations.
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
}

fn limbs(value: &FE3) -> [u64; 3] {
    let [a, b, c] = value.value();
    [*a.value(), *b.value(), *c.value()]
}

/// A stacked polynomial of `2^n`: a table of columns sharing a point, a tall
/// column, short ones scattered in the middle, and gaps — its parts, its weight
/// shares in the materialised session's form, and the same shares in lean form
/// with their points' `eq` halves.
struct Stack {
    n: usize,
    columns: Vec<(Vec<u64>, usize)>,
    raw_shares: Vec<(usize, Vec<u64>, [u64; 3])>,
    lean: Vec<LeanShare>,
    eq: Vec<u64>,
}

fn stack(n: usize, seed: u64) -> Stack {
    let mut rng = Rng(seed);
    let mut placed: Vec<(usize, usize, Vec<FE3>)> = Vec::new(); // (offset, vars, point)
    let table_vars = n - 4;
    let table_point: Vec<FE3> = (0..table_vars).map(|_| rng.ext()).collect();
    for c in 0..3 {
        placed.push((c << table_vars, table_vars, table_point.clone()));
    }
    let tall_vars = n - 1;
    placed.push((
        1 << tall_vars,
        tall_vars,
        (0..tall_vars).map(|_| rng.ext()).collect(),
    ));
    for c in 0..12usize {
        let vars = 2 + c % 4;
        let at = (1 << (n - 2)) + (c << 6);
        placed.push((at, vars, (0..vars).map(|_| rng.ext()).collect()));
    }
    placed.sort_by_key(|p| p.0);

    let mut columns = Vec::new();
    let mut raw_shares = Vec::new();
    let mut lean = Vec::new();
    let mut eq: Vec<u64> = Vec::new();
    for (offset, vars, point) in &placed {
        let column: Vec<u64> = (0..(1usize << vars)).map(|_| rng.next()).collect();
        columns.push((column, *offset));
        let scale = rng.ext();
        let raw_point: Vec<u64> = point.iter().flat_map(limbs).collect();
        raw_shares.push((*offset, raw_point, limbs(&scale)));
        let lo_bits = vars / 2;
        let (hi, lo) = point.split_at(vars - lo_bits);
        let hi_at = eq.len() / 3;
        for value in multilinear::eq::eq_evals(hi) {
            eq.extend_from_slice(&limbs(&value));
        }
        let lo_at = eq.len() / 3;
        for value in multilinear::eq::eq_evals(lo) {
            eq.extend_from_slice(&limbs(&value));
        }
        lean.push(LeanShare {
            stack_offset: *offset,
            num_vars: *vars,
            lo_bits,
            hi_at,
            lo_at,
            scale: limbs(&scale),
        });
    }
    Stack {
        n,
        columns,
        raw_shares,
        lean,
        eq,
    }
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

/// Six rounds both ways at the same challenges, then the bound tables and the
/// message at a point.
fn lean_matches_materialised(s: &Stack, resident: bool) {
    lean_matches_materialised_with(s, resident, false);
}

/// The same, with `overlay`: a weight term `scale·eq(p, ·)` over the whole
/// stack, gaps included (a commit-time out-of-domain claim, I-WOOD) — added to
/// the materialised session by `add_scaled_eq` before its first round, and
/// read by the lean rounds as the row past the columns'.
fn lean_matches_materialised_with(s: &Stack, resident: bool, overlay: bool) {
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
    );
    let store_parts: Vec<(usize, usize)> = s
        .columns
        .iter()
        .enumerate()
        .map(|(i, (_, offset))| (i, *offset))
        .collect();
    let len = 1usize << s.n;

    let mut session = if resident {
        OpeningSession::from_shares_and_resident(
            &s.raw_shares,
            len,
            store.as_ref().expect("the card took the columns"),
            &store_parts,
        )
    } else {
        OpeningSession::from_shares_and_parts(&s.raw_shares, len, &parts)
    }
    .unwrap_or_else(|e| panic!("materialised session (needs a GPU): {e:?}"));
    let mut eq = s.eq.clone();
    let lean_overlay = overlay.then(|| {
        let mut rng = Rng(0xD00D);
        let point: Vec<FE3> = (0..s.n).map(|_| rng.ext()).collect();
        let scale = rng.ext();
        let raw: Vec<u64> = point.iter().flat_map(limbs).collect();
        session
            .add_scaled_eq(&raw, &limbs(&scale))
            .expect("the materialised overlay");
        let lo_bits = s.n / 2;
        let (hi, lo) = point.split_at(s.n - lo_bits);
        let hi_at = eq.len() / 3;
        for value in multilinear::eq::eq_evals(hi) {
            eq.extend_from_slice(&limbs(&value));
        }
        let lo_at = eq.len() / 3;
        for value in multilinear::eq::eq_evals(lo) {
            eq.extend_from_slice(&limbs(&value));
        }
        LeanShare {
            stack_offset: 0,
            num_vars: s.n,
            lo_bits,
            hi_at,
            lo_at,
            scale: limbs(&scale),
        }
    });
    let message = if resident {
        LeanMessage::Resident {
            store: store.as_ref().expect("the card took the columns"),
            parts: &store_parts,
        }
    } else {
        LeanMessage::Parts(&parts)
    };
    let mut lean = LeanRound0::new_with_overlay(&s.lean, lean_overlay, &eq, s.n, message)
        .expect("device")
        .expect("the lean opening takes this stack");

    let lowered = product();
    let mut rounds = session
        .sumcheck(
            &lowered.nodes,
            &lowered.consts,
            lowered.num_slots,
            lowered.root_slot,
        )
        .expect("session rounds");
    let mut rng = Rng(0xABCD);
    for round in 1..=6 {
        let want = rounds.round(&nodes()).expect("materialised round");
        let got = lean.round(&nodes()).expect("lean round");
        assert_eq!(
            canonical(&got),
            canonical(&want),
            "round {round} ({}{}): the evaluations differ",
            if resident { "resident" } else { "parts" },
            if overlay { ", overlay" } else { "" }
        );
        let alpha = limbs(&rng.ext());
        rounds.fold(&alpha).expect("fold");
        lean.bind(&alpha);
    }
    drop(rounds);
    session.bound(6);
    let (want_w, want_f) = session.tables_to_host().expect("read back");

    let bound = lean.materialize().expect("materialise");
    let (got_w, got_f) = bound.tables_to_host().expect("read back");
    assert_eq!(
        canonical(&got_w),
        canonical(&want_w),
        "the bound weight differs"
    );
    assert_eq!(
        canonical(&got_f),
        canonical(&want_f),
        "the bound message differs"
    );

    let point: Vec<u64> = (0..s.n - 6).flat_map(|_| limbs(&rng.ext())).collect();
    assert_eq!(
        canonical(&bound.evaluate_message(&point).expect("evaluate")),
        canonical(&session.evaluate_message(&point).expect("evaluate")),
        "the message at a point differs"
    );
}

#[test]
fn the_lean_rounds_match_the_materialised_session_from_host_parts() {
    let _device = device();
    lean_matches_materialised(&stack(14, 0x51), false);
}

#[test]
fn the_lean_rounds_match_the_materialised_session_from_resident_columns() {
    let _device = device();
    lean_matches_materialised(&stack(15, 0x77), true);
}

#[test]
fn the_lean_rounds_with_an_overlay_match_the_materialised_session_from_host_parts() {
    let _device = device();
    lean_matches_materialised_with(&stack(14, 0x52), false, true);
}

#[test]
fn the_lean_rounds_with_an_overlay_match_the_materialised_session_from_resident_columns() {
    let _device = device();
    lean_matches_materialised_with(&stack(15, 0x78), true, true);
}

/// ★ The accounting the lean path exists for. At 2^22 the materialised
/// session holds the weight and the lifted message, `2·2^22` extension values
/// (192 MiB), and built them through a base staging buffer (32 MiB more); the
/// lean opening over resident columns holds a `u16` per position (8 MiB), the
/// shares, the `eq` halves and the round scratch. Measured through the driver
/// with the pool drained, and by the structures' own accounting.
#[test]
fn the_lean_opening_holds_a_fraction_of_the_materialised_one() {
    let _device = device();
    let s = stack(22, 0x99);
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
    let be = math_cuda::device::backend().expect("a device");

    math_cuda::device::drain_and_trim().expect("drain");
    let before = be.free_vram_bytes().expect("free");
    let session =
        OpeningSession::from_shares_and_resident(&s.raw_shares, len, &store, &store_parts)
            .expect("session");
    be.ctx.synchronize().expect("sync");
    let materialised = before.saturating_sub(be.free_vram_bytes().expect("free"));
    let materialised_accounted = session.device_bytes();
    drop(session);

    math_cuda::device::drain_and_trim().expect("drain");
    let before = be.free_vram_bytes().expect("free");
    let lean = LeanRound0::new(
        &s.lean,
        &s.eq,
        s.n,
        LeanMessage::Resident {
            store: &store,
            parts: &store_parts,
        },
    )
    .expect("device")
    .expect("the lean opening takes this stack");
    be.ctx.synchronize().expect("sync");
    let lean_taken = before.saturating_sub(be.free_vram_bytes().expect("free"));
    let lean_accounted = lean.device_bytes();

    let codeword = (1u64 << (s.n + 2)) * 8;
    println!(
        "opening working set at 2^{}: materialised {materialised} B measured / \
         {materialised_accounted} B accounted ({:.2}× the base codeword); lean \
         {lean_taken} B measured / {lean_accounted} B accounted ({:.3}×)",
        s.n,
        materialised_accounted as f64 / codeword as f64,
        lean_accounted as f64 / codeword as f64,
    );
    assert_eq!(materialised_accounted, (len as u64) * 48);
    assert!(lean_accounted * 8 < materialised_accounted);
    assert!(
        lean_taken * 4 < materialised,
        "lean took {lean_taken} B from the card against {materialised} B"
    );
}

/// Two-inverse and the inverse generator of each of `levels` levels from a
/// domain with generator `g`: what `multilinear::gpu` passes a fold.
fn fold_scalars(g: FE, levels: usize) -> (u64, Vec<u64>) {
    let two_inv = *FE::from(2u64).inv().expect("2 is invertible").value();
    let mut g_inv = g.inv().expect("a generator is invertible");
    let mut g_invs = Vec::with_capacity(levels);
    for _ in 0..levels {
        g_invs.push(*g_inv.value());
        g_inv = g_inv.square();
    }
    (two_inv, g_invs)
}

fn raw_challenges(from: u64, count: usize) -> Vec<u64> {
    (from..from + count as u64)
        .flat_map(|seed| [seed * 31 + 7, seed * 17 + 5, seed + 3])
        .collect()
}

/// The fused fold's working set is its output: folding a committed codeword
/// six levels takes `2^(n+2−6)` extension values from the card, where the
/// level-by-level fold's first two levels alone take `2^(n+1) + 2^n` of them
/// while the codeword is resident. Measured through the driver, pool drained.
#[test]
fn the_fused_fold_takes_only_its_output_from_the_card() {
    use multilinear::whir::Domain;
    let _device = device();
    let (num_vars, log_blowup) = (20usize, 2usize);
    let evals: Vec<u64> = (0..(1u64 << num_vars)).map(|i| i * 7 + 3).collect();
    let (committed, _root) = math_cuda::whir::commit_codeword(
        &evals,
        log_blowup,
        1,
        false,
        math_cuda::DeviceHash::Keccak256,
    )
    .unwrap_or_else(|e| panic!("device commit (needs a GPU): {e:?}"));
    let domain = Domain::<Gl>::new(num_vars + log_blowup).expect("domain");
    let (two_inv, g_invs) = fold_scalars(*domain.generator(), 6);
    let alphas = raw_challenges(1, 6);
    let be = math_cuda::device::backend().expect("a device");

    let taken = |fused: bool| -> u64 {
        math_cuda::device::drain_and_trim().expect("drain");
        let before = be.free_vram_bytes().expect("free");
        let folded = if fused {
            math_cuda::whir::fold_resident_fused(&committed, two_inv, &g_invs, &alphas)
        } else {
            math_cuda::whir::fold_resident(&committed, two_inv, &g_invs, &alphas)
        }
        .expect("fold");
        be.ctx.synchronize().expect("sync");
        let after = be.free_vram_bytes().expect("free");
        drop(folded);
        before.saturating_sub(after)
    };
    let stepped = taken(false);
    let fused = taken(true);
    let codeword = (1u64 << (num_vars + log_blowup)) * 8;
    println!(
        "fold working set at 2^{}: level by level {stepped} B ({:.2}× the codeword), fused {fused} B ({:.3}×)",
        num_vars + log_blowup,
        stepped as f64 / codeword as f64,
        fused as f64 / codeword as f64,
    );
    assert!(
        fused * 8 < stepped,
        "the fused fold took {fused} B against the level-by-level {stepped} B"
    );
}
