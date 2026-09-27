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

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
use math::field::goldilocks::GoldilocksField as Gl;
use math_cuda::whir_open::{LeanMessage, LeanRound0, LeanShare, OpeningSession};

type FE = FieldElement<Gl>;
type FE3 = FieldElement<Ext3>;

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
    let message = if resident {
        LeanMessage::Resident {
            store: store.as_ref().expect("the card took the columns"),
            parts: &store_parts,
        }
    } else {
        LeanMessage::Parts(&parts)
    };
    let mut lean = LeanRound0::new(&s.lean, &s.eq, s.n, message)
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
            "round {round} ({}): the evaluations differ",
            if resident { "resident" } else { "parts" }
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
    lean_matches_materialised(&stack(14, 0x51), false);
}

#[test]
fn the_lean_rounds_match_the_materialised_session_from_resident_columns() {
    lean_matches_materialised(&stack(15, 0x77), true);
}

/// ★ The accounting the lean path exists for. At 2^22 the materialised
/// session holds the weight and the lifted message, `2·2^22` extension values
/// (192 MiB), and built them through a base staging buffer (32 MiB more); the
/// lean opening over resident columns holds a `u16` per position (8 MiB), the
/// shares, the `eq` halves and the round scratch. Measured through the driver
/// with the pool drained, and by the structures' own accounting.
#[test]
fn the_lean_opening_holds_a_fraction_of_the_materialised_one() {
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
