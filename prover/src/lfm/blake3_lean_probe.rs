//! Prove + verify the lean, narrow BLAKE3 chip ([`super::blake3_lean`])
//! standalone, and time it on the card (I-B3CHIP: D-B3-3 S3 and S4).
//!
//! The `blake3_probe` pattern: the chip's real interactions and constraints,
//! its fixed table, and a synthetic `LfmMem` mirror standing in for the
//! machine's memory. Every digest the honest probes prove is checked against
//! the `blake3` crate and against the base's own BLAKE3 Merkle backend
//! (`BatchBlake3StdBackend`), so "BLAKE3" here means the hash the base commits
//! with, not a reimplementation agreeing with itself.
//!
//! Proved under the LFM's own options ([`crate::lfm::proof::aggregation_wrap_options`]:
//! blowup 4, 112 queries, the process format) with `multi_prove_bound`, the
//! LFM's prove entry, under either commitment hash ([`Arm`]).

use crypto::fiat_shamir::is_transcript::{IsStarkTranscript, IsTranscript};
use stark::config::StarkHash;
use stark::constraints::builder::EmptyConstraints;
use stark::leaf_layout::LeafLayout;
use stark::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, BusValue, Multiplicity,
    NullBoundaryConstraintBuilder, Packing,
};
use stark::proof::options::ProofOptions;
use stark::proof::stark::MultiProof;
use stark::proof::view::MultiProofView;
use stark::prover::IsStarkProver;
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::IsStarkVerifier;

use crate::tables::types::{
    BusId, FE, FEE, GoldilocksExtension, GoldilocksField, VmTable, zeroed_fe_vec,
};

use super::blake3_lean::{
    self as lean, Drops, Fault, FaultKind, HeadConstraints, LeanOp, Lin, ROUNDS, RoundConstraints,
    TableHist, head, round, table,
};
use super::commit::{commit_lde_columns_under, lde_columns};

type F = GoldilocksField;
type E = GoldilocksExtension;
type DynAir<'a> = &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>;
type RoundAir = AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), RoundConstraints>;
type HeadAir = AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), HeadConstraints>;
type PlainAir = AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints>;
type Proof = (MultiProof<F, E, ()>, Vec<Vec<FEE>>);

const PROBE_TAG: &[u8] = b"LFM_B3R_PROBE_V1";

/// BLAKE3 flags.
const CHUNK_START: u32 = 1;
const CHUNK_END: u32 = 2;
const ROOT: u32 = 8;

// =========================================================================
// The two commitment hashes
// =========================================================================

/// A commitment configuration and its transcript.
pub(crate) trait Arm {
    type S: StarkHash;
    type T: IsStarkTranscript<E, F> + Clone + Send;
    const NAME: &'static str;
    fn transcript() -> Self::T;
}

/// The STARK pipeline's BLAKE3 (`Blake3StarkHash`, the build's round count).
pub(crate) struct B3Arm;
impl Arm for B3Arm {
    type S = stark::config::Blake3StarkHash;
    type T = stark::config::DefaultStarkTranscript<E>;
    const NAME: &'static str = "b3";
    fn transcript() -> Self::T {
        let mut t = stark::config::DefaultStarkTranscript::<E>::new(&[]);
        t.append_bytes(PROBE_TAG);
        t
    }
}

/// Today's LFM: ZisK's Poseidon1, 4-ary trees capped at C2.
pub(crate) struct P1Arm;
impl Arm for P1Arm {
    type S = super::p1_commit::P1StarkHash;
    type T = super::p1_commit::P1Transcript;
    const NAME: &'static str = "p1";
    fn transcript() -> Self::T {
        let mut t = super::p1_commit::P1Transcript::with_seed(&[]);
        t.append_bytes(PROBE_TAG);
        t
    }
}

// =========================================================================
// The ops
// =========================================================================

/// A small deterministic generator (xorshift64*), so every probe is fixed.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn felt(&mut self) -> u64 {
        loop {
            let v = self.next();
            if v < 0xFFFF_FFFF_0000_0001 {
                return v;
            }
        }
    }
}

/// Address allocator for the synthetic memory.
struct Addrs(u64);
impl Addrs {
    fn take(&mut self) -> u64 {
        self.0 += 1;
        self.0
    }
}

fn digest_bytes(d: &[u32; 8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, w) in d.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

/// A Merkle parent: one compression of the two children's digests, IV, the
/// one-block flags — `blake3::hash(left ‖ right)`.
fn parent_op(left: [u8; 32], right: [u8; 32], a: &mut Addrs) -> LeanOp {
    let mut block = [0u8; 64];
    block[..32].copy_from_slice(&left);
    block[32..].copy_from_slice(&right);
    LeanOp {
        h: [0; 8],
        iv: true,
        block,
        block_len: 64,
        flags: CHUNK_START | CHUNK_END | ROOT,
        felt_mode: false,
        h_addr: [0; 2],
        m_addr: core::array::from_fn(|_| a.take()),
        out_addr: core::array::from_fn(|_| a.take()),
        out_cnt: [1; 2],
    }
}

/// A leaf over `felts` (≤ 128, one chunk): ⌈8·n/64⌉ chained compressions of
/// the felts' big-endian bytes — `blake3::hash` of those bytes, which is what
/// the base's `BatchBlake3StdBackend` hashes a leaf to.
fn leaf_ops(felts: &[u64], a: &mut Addrs) -> Vec<LeanOp> {
    let bytes: Vec<u8> = felts.iter().flat_map(|f| f.to_be_bytes()).collect();
    let blocks = bytes.len().div_ceil(64).max(1);
    assert!(bytes.len() <= 1024, "one chunk");
    let mut ops: Vec<LeanOp> = Vec::with_capacity(blocks);
    for j in 0..blocks {
        let mut block = [0u8; 64];
        let part = &bytes[64 * j..(64 * j + 64).min(bytes.len())];
        block[..part.len()].copy_from_slice(part);
        let mut flags = 0;
        if j == 0 {
            flags |= CHUNK_START;
        }
        if j == blocks - 1 {
            flags |= CHUNK_END | ROOT;
        }
        let prev = ops.last();
        let op = LeanOp {
            h: prev.map_or([0; 8], LeanOp::digest),
            iv: j == 0,
            block,
            block_len: part.len() as u32,
            flags,
            felt_mode: true,
            h_addr: prev.map_or([0; 2], |p| p.out_addr),
            m_addr: [a.take(), a.take(), 0, 0],
            out_addr: core::array::from_fn(|_| a.take()),
            out_cnt: [1; 2],
        };
        ops.push(op);
    }
    ops
}

/// A leaf-shaped mix: per group, one 64-felt base leaf (8 chained
/// compressions in leaf mode) and 16 Merkle parents, D-B3-3 §3.1's ratio for a
/// base query round (an 8-compression leaf and a 14–17-compression path).
/// Truncated to `n` compressions; a tail shorter than a leaf is parents.
fn mixed_ops(n: usize, seed: u64) -> Vec<LeanOp> {
    let mut rng = Rng(seed | 1);
    let mut a = Addrs(1_000);
    let mut ops = Vec::with_capacity(n);
    while ops.len() < n {
        if n - ops.len() >= 8 {
            let felts: Vec<u64> = (0..64).map(|_| rng.felt()).collect();
            ops.extend(leaf_ops(&felts, &mut a));
        }
        for _ in 0..16 {
            if ops.len() == n {
                break;
            }
            let mut d = || -> [u8; 32] { core::array::from_fn(|_| rng.next() as u8) };
            let (l, r) = (d(), d());
            ops.push(parent_op(l, r, &mut a));
        }
    }
    ops.truncate(n);
    ops
}

/// The S3 probe's ops: a 64-felt leaf, a 3-felt leaf (a partial block), a
/// parent of two real digests, then parents of random digests, `n` in all.
fn probe_ops(n: usize) -> Vec<LeanOp> {
    let mut rng = Rng(0xB3C0_FFEE);
    let mut a = Addrs(1_000);
    let felts: Vec<u64> = (0..64).map(|_| rng.felt()).collect();
    let mut ops = leaf_ops(&felts, &mut a);
    let short: Vec<u64> = (0..3).map(|_| rng.felt()).collect();
    ops.extend(leaf_ops(&short, &mut a));
    let (l, r) = (
        digest_bytes(&ops[7].digest()),
        digest_bytes(&ops[8].digest()),
    );
    ops.push(parent_op(l, r, &mut a));
    while ops.len() < n {
        let mut d = || -> [u8; 32] { core::array::from_fn(|_| rng.next() as u8) };
        let (l, r) = (d(), d());
        ops.push(parent_op(l, r, &mut a));
    }
    assert!(ops.len() == n, "probe_ops needs n ≥ 10");
    ops
}

// =========================================================================
// The traces
// =========================================================================

/// What a build is asked to do besides the honest compression.
#[derive(Clone, Copy, Debug)]
struct Spec {
    /// A fault in one op's witness.
    fault: Option<(usize, Fault)>,
    /// One op proved over six rounds instead of seven.
    six_round_op: Option<usize>,
    drops: Drops,
    max_chunk_log: u32,
}

impl Default for Spec {
    fn default() -> Self {
        Spec {
            fault: None,
            six_round_op: None,
            drops: Drops::default(),
            max_chunk_log: lean::MAX_CHUNK_LOG,
        }
    }
}

struct Traces {
    rounds: Vec<TraceTable<F, E>>,
    head: TraceTable<F, E>,
    table: TraceTable<F, E>,
    mirror: TraceTable<F, E>,
    /// The digest each op's head wrote.
    digests: Vec<[u32; 8]>,
    hist: TableHist,
}

/// The mirror: `[ADDR, V0..V3, SEND_MULT, RECV_MULT]`, one row a memory word
/// the chip touches (`blake3_probe`'s, verbatim in shape).
mod mirror {
    pub const ADDR: usize = 0;
    pub const V0: usize = 1;
    pub const SEND_MULT: usize = 5;
    pub const RECV_MULT: usize = 6;
    pub const NUM_COLUMNS: usize = 7;
}

fn mirror_token() -> Vec<BusValue> {
    (0..5)
        .map(|c| BusValue::Packed {
            start_column: c,
            packing: Packing::Direct,
        })
        .collect()
}

fn mirror_interactions() -> Vec<BusInteraction> {
    vec![
        BusInteraction::sender(
            BusId::LfmMem,
            Multiplicity::Column(mirror::SEND_MULT),
            mirror_token(),
        ),
        BusInteraction::receiver(
            BusId::LfmMem,
            Multiplicity::Column(mirror::RECV_MULT),
            mirror_token(),
        ),
    ]
}

fn mirror_trace(ops: &[LeanOp], digests: &[[u32; 8]]) -> TraceTable<F, E> {
    let mut rows: Vec<(u64, [u64; 4], u64, u64)> = Vec::with_capacity(8 * ops.len());
    for (op, d) in ops.iter().zip(digests) {
        if !op.iv {
            for w in 0..2 {
                rows.push((
                    op.h_addr[w],
                    core::array::from_fn(|l| u64::from(op.h[4 * w + l])),
                    1,
                    0,
                ));
            }
        }
        if op.felt_mode {
            let felts = op.felts();
            for w in 0..2 {
                rows.push((
                    op.m_addr[w],
                    core::array::from_fn(|l| felts[4 * w + l]),
                    1,
                    0,
                ));
            }
        } else {
            let m = op.message();
            for w in 0..4 {
                rows.push((
                    op.m_addr[w],
                    core::array::from_fn(|l| u64::from(m[4 * w + l])),
                    1,
                    0,
                ));
            }
        }
        for w in 0..2 {
            rows.push((
                op.out_addr[w],
                core::array::from_fn(|l| u64::from(d[4 * w + l])),
                0,
                op.out_cnt[w],
            ));
        }
    }
    let n = rows.len().next_power_of_two().max(4);
    let mut data = zeroed_fe_vec(n * mirror::NUM_COLUMNS);
    for (r, (addr, v, send, recv)) in rows.into_iter().enumerate() {
        let row = &mut data[r * mirror::NUM_COLUMNS..(r + 1) * mirror::NUM_COLUMNS];
        row[mirror::ADDR] = FE::from(addr);
        for l in 0..4 {
            row[mirror::V0 + l] = FE::from(v[l]);
        }
        row[mirror::SEND_MULT] = FE::from(send);
        row[mirror::RECV_MULT] = FE::from(recv);
    }
    TraceTable::new_main(data, mirror::NUM_COLUMNS, 1)
}

fn digest_of(fs: &[u32; 16]) -> [u32; 8] {
    core::array::from_fn(|i| fs[i] ^ fs[i + 8])
}

/// Fill every round row and head row. Uniform seven-round builds with no fault
/// go parallel (S4's 2^18 compressions); the rest are the probes' few dozen.
fn fill(ops: &[LeanOp], spec: &Spec) -> (Vec<FE>, usize, Vec<FE>, Vec<[u32; 8]>, TableHist) {
    let n = ops.len();
    let rounds_of = |k: usize| {
        if spec.six_round_op == Some(k) {
            ROUNDS - 1
        } else {
            ROUNDS
        }
    };
    let total: usize = (0..n).map(rounds_of).sum();
    let head_rows = n.next_power_of_two().max(4);
    let mut round_data = zeroed_fe_vec(total * round::NUM_COLUMNS);
    let mut head_data = zeroed_fe_vec(head_rows * head::NUM_COLUMNS);
    let mut digests = vec![[0u32; 8]; n];

    #[cfg(feature = "parallel")]
    if spec.fault.is_none() && spec.six_round_op.is_none() {
        use rayon::prelude::*;
        let hist = round_data
            .par_chunks_mut(ROUNDS * round::NUM_COLUMNS)
            .zip(head_data.par_chunks_mut(head::NUM_COLUMNS))
            .zip(digests.par_iter_mut())
            .zip(ops.par_iter())
            .enumerate()
            .fold(
                TableHist::default,
                |mut hist, (k, (((rows, hrow), dig), op))| {
                    let fs = lean::fill_rounds(rows, k as u64 + 1, op, None, ROUNDS, &mut hist);
                    lean::fill_head(hrow, k as u64 + 1, op, &fs, &mut hist);
                    *dig = digest_of(&fs);
                    hist
                },
            )
            .reduce(TableHist::default, |a, b| a.merge(&b));
        return (round_data, total, head_data, digests, hist);
    }

    let mut hist = TableHist::default();
    let mut off = 0usize;
    for (k, op) in ops.iter().enumerate() {
        let r = rounds_of(k);
        let rows = &mut round_data[off * round::NUM_COLUMNS..(off + r) * round::NUM_COLUMNS];
        let fault = spec.fault.filter(|&(at, _)| at == k).map(|(_, f)| f);
        let fs = lean::fill_rounds(rows, k as u64 + 1, op, fault, r, &mut hist);
        let hrow = &mut head_data[k * head::NUM_COLUMNS..(k + 1) * head::NUM_COLUMNS];
        lean::fill_head(hrow, k as u64 + 1, op, &fs, &mut hist);
        digests[k] = digest_of(&fs);
        off += r;
    }
    (round_data, total, head_data, digests, hist)
}

fn build_traces(ops: &[LeanOp], spec: &Spec) -> Traces {
    let (mut round_data, total, head_data, digests, hist) = fill(ops, spec);
    let sizes = lean::chunk_rows(total, spec.max_chunk_log, lean::MIN_CHUNK_LOG);
    // Split from the back so each chunk's rows move once.
    let mut rounds = Vec::with_capacity(sizes.len());
    let mut starts = Vec::with_capacity(sizes.len());
    let mut s = 0usize;
    for &size in &sizes {
        starts.push(s);
        s += size;
    }
    for (&start, &size) in starts.iter().zip(&sizes).rev() {
        let real = total.saturating_sub(start).min(size);
        let mut data = round_data.split_off(start * round::NUM_COLUMNS);
        data.resize(size * round::NUM_COLUMNS, FE::zero());
        debug_assert!(real <= size);
        rounds.push(TraceTable::new_main(data, round::NUM_COLUMNS, 1));
    }
    rounds.reverse();
    let mirror = mirror_trace(ops, &digests);
    Traces {
        rounds,
        head: TraceTable::new_main(head_data, head::NUM_COLUMNS, 1),
        table: lean::table_trace(&hist),
        mirror,
        digests,
        hist,
    }
}

// =========================================================================
// The AIRs
// =========================================================================

fn prefix_columns(t: &TraceTable<F, E>, width: usize) -> Vec<Vec<FE>> {
    (0..width)
        .map(|c| (0..t.num_rows()).map(|r| *t.main_table.get(r, c)).collect())
        .collect()
}

fn root_under<S: StarkHash>(columns: &[Vec<FE>], opts: &ProofOptions) -> stark::config::Commitment {
    commit_lde_columns_under::<S>(&lde_columns(columns, opts), LeafLayout::RowPair)
}

struct Airs {
    rounds: Vec<RoundAir>,
    head: HeadAir,
    table: PlainAir,
    mirror: PlainAir,
}

impl Airs {
    fn refs(&self) -> Vec<DynAir<'_>> {
        let mut v: Vec<DynAir<'_>> = self.rounds.iter().map(|a| a as DynAir<'_>).collect();
        v.push(&self.head);
        v.push(&self.table);
        v.push(&self.mirror);
        v
    }
}

/// The preprocessed roots of every AIR, under `S`.
struct Roots {
    rounds: Vec<stark::config::Commitment>,
    head: stark::config::Commitment,
    table: stark::config::Commitment,
}

fn roots_under<S: StarkHash>(t: &Traces, opts: &ProofOptions) -> Roots {
    Roots {
        rounds: t
            .rounds
            .iter()
            .map(|r| root_under::<S>(&prefix_columns(r, round::PREP_WIDTH), opts))
            .collect(),
        head: root_under::<S>(&prefix_columns(&t.head, head::PREP_WIDTH), opts),
        table: root_under::<S>(&lean::table_prep_columns(), opts),
    }
}

fn airs(roots: &Roots, opts: &ProofOptions, drops: Drops) -> Airs {
    let rounds = roots
        .rounds
        .iter()
        .map(|&root| {
            AirWithBuses::new(
                round::NUM_COLUMNS,
                AuxiliaryTraceBuildData {
                    interactions: lean::round_interactions(),
                },
                opts,
                1,
                RoundConstraints { drops },
            )
            .with_name("B3R_ROUND")
            .with_preprocessed(root, round::PREP_WIDTH)
        })
        .collect();
    Airs {
        rounds,
        head: AirWithBuses::new(
            head::NUM_COLUMNS,
            AuxiliaryTraceBuildData {
                interactions: lean::head_interactions(),
            },
            opts,
            1,
            HeadConstraints,
        )
        .with_name("B3R_HEAD")
        .with_preprocessed(roots.head, head::PREP_WIDTH),
        table: AirWithBuses::new(
            table::NUM_COLUMNS,
            AuxiliaryTraceBuildData {
                interactions: lean::table_interactions(),
            },
            opts,
            1,
            EmptyConstraints,
        )
        .with_name("B3R_XOR")
        .with_preprocessed(roots.table, table::PREP_WIDTH),
        mirror: AirWithBuses::new(
            mirror::NUM_COLUMNS,
            AuxiliaryTraceBuildData {
                interactions: mirror_interactions(),
            },
            opts,
            1,
            EmptyConstraints,
        )
        .with_name("B3R_MEM_MIRROR"),
    }
}

fn options() -> ProofOptions {
    super::proof::aggregation_wrap_options()
}

fn prove<A: Arm>(airs: &Airs, t: &mut Traces) -> Result<Proof, stark::prover::ProvingError> {
    let mut pairs: Vec<(DynAir<'_>, &mut TraceTable<F, E>, &())> = Vec::new();
    for (air, trace) in airs.rounds.iter().zip(t.rounds.iter_mut()) {
        pairs.push((air, trace, &()));
    }
    pairs.push((&airs.head, &mut t.head, &()));
    pairs.push((&airs.table, &mut t.table, &()));
    pairs.push((&airs.mirror, &mut t.mirror, &()));
    let mut transcript = A::transcript();
    stark::prover::GenericProver::<F, E, (), A::S>::multi_prove_bound(
        pairs,
        &mut transcript,
        stark::residency_mode::ResidencyMode::Retain,
    )
}

fn verify<A: Arm>(airs: &Airs, proof: &Proof) -> bool {
    let refs = airs.refs();
    let mut transcript = A::transcript();
    stark::verifier::GenericVerifier::<F, E, (), A::S>::multi_verify_views_bound(
        &refs,
        MultiProofView::Owned(&proof.0),
        &proof.1,
        &mut transcript,
        &FEE::zero(),
    )
}

/// Build, optionally corrupt, prove and verify. `Err` = the prover refused.
fn round_trip_with<A: Arm>(
    ops: &[LeanOp],
    spec: Spec,
    mutate: impl FnOnce(&mut Traces),
) -> Result<bool, String> {
    let opts = options();
    let mut t = build_traces(ops, &spec);
    mutate(&mut t);
    let roots = roots_under::<A::S>(&t, &opts);
    let airs = airs(&roots, &opts, spec.drops);
    match prove::<A>(&airs, &mut t) {
        Ok(proof) => Ok(verify::<A>(&airs, &proof)),
        Err(e) => Err(format!("{e:?}")),
    }
}

fn round_trip(
    ops: &[LeanOp],
    spec: Spec,
    mutate: impl FnOnce(&mut Traces),
) -> Result<bool, String> {
    round_trip_with::<B3Arm>(ops, spec, mutate)
}

fn assert_not_accepted(what: &str, r: Result<bool, String>) {
    match r {
        Ok(true) => panic!("{what} must not produce an accepted proof, but the proof verified"),
        Ok(false) => println!("B3R NEGATIVE {what}: verifier rejected"),
        Err(e) => println!(
            "B3R NEGATIVE {what}: prover refused ({})",
            e.chars().take(120).collect::<String>()
        ),
    }
}

// =========================================================================
// The census
// =========================================================================

/// Base-field-equivalent cells of an AIR's row: main + 3·aux, with
/// `aux = ⌈interactions / 2⌉` (`airs.rs`' census law, the one `5,545` and
/// `328` were measured under), preprocessed columns excluded.
fn row_cells(main: usize, interactions: usize) -> usize {
    main + 3 * interactions.div_ceil(2)
}

/// ★ S3's census: the built layout against the pre-registered closed form
/// (I-B3CHIP §1.1), and the pieces of the per-compression cell bill.
#[test]
fn the_lean_chip_census_matches_its_closed_form() {
    let round_i = lean::round_interactions().len();
    let head_i = lean::head_interactions().len();
    let table_i = lean::table_interactions().len();

    // The round row: 40 carried state cells, 16 message words, 8 G's of 40.
    assert_eq!(round::MAIN_COLUMNS, 40 + 16 + 8 * 40);
    assert_eq!(round::MAIN_COLUMNS, 376);
    assert_eq!(
        round_i,
        8 * 16 + 2,
        "128 XOR/split lookups + the two chain tokens"
    );
    assert_eq!(round_i.div_ceil(2), 65);
    assert_eq!(row_cells(round::MAIN_COLUMNS, round_i), 571);
    // The head: block 64, h limbs 8 + bytes 16, final state 64, digest 32,
    // canonicity 16.
    assert_eq!(head::MAIN_COLUMNS, 64 + 8 + 16 + 64 + 32 + 16);
    assert_eq!(head::MAIN_COLUMNS, 200);
    assert_eq!(head_i, 2 + 32 + 8 + 32 + 10);
    assert_eq!(head_i.div_ceil(2), 42);
    assert_eq!(row_cells(head::MAIN_COLUMNS, head_i), 326);

    let main = ROUNDS * round::MAIN_COLUMNS + head::MAIN_COLUMNS;
    let aux = ROUNDS * round_i.div_ceil(2) + head_i.div_ceil(2);
    let cells = main + 3 * aux;
    assert_eq!((main, aux, cells), (2_832, 497, 4_323));

    // The same counts as the AIRs report them: the framework's own aux width.
    let opts = options();
    let dummy = Roots {
        rounds: vec![[0u8; 32]],
        head: [0u8; 32],
        table: [0u8; 32],
    };
    let a = airs(&dummy, &opts, Drops::default());
    let (rm, ra) = a.rounds[0].trace_layout();
    let (hm, ha) = a.head.trace_layout();
    assert_eq!((rm, ra), (round::NUM_COLUMNS, 65));
    assert_eq!((hm, ha), (head::NUM_COLUMNS, 42));

    // The pieces (I-B3CHIP §1.1), each as cells a compression.
    let core_g = ROUNDS * 8 * (40 + 3 * 8); // 16 lookups = 8 aux a G
    let carried = ROUNDS * (40 + 16);
    let tokens = ROUNDS * 3; // two interactions a row = one aux column
    assert_eq!(core_g + carried + tokens, ROUNDS * 571);
    let aux_share = 3.0 * aux as f64 / cells as f64;

    let (t_m, t_a) = a.table.trace_layout();
    println!(
        "\nB3R CENSUS: cells a compression {cells} = main {main} + 3·aux {aux} \
         (LogUp aux share {:.1} %); rows a compression {ROUNDS} round + 1 head\n\
         B3R CENSUS ROUND: main {} · interactions {round_i} · aux {} · {} cells a row · \
         preprocessed {} · opened felts a row {}\n\
         B3R CENSUS HEAD: main {} · interactions {head_i} · aux {} · {} cells a row · \
         preprocessed {} · opened felts a row {}\n\
         B3R CENSUS TABLE: 2^16 rows · {} columns ({} preprocessed) · {table_i} interactions · \
         aux {t_a} · {} cells in all\n\
         B3R CENSUS PIECES: G core {core_g} · carried state + message {carried} · round tokens \
         {tokens} · head {}\n\
         B3R CENSUS AGAINST: today's LFM_HASH BLAKE3 socket 5,545 (×{:.3}) · D-B3-3 lean \
         3,800 (×{:.3}) · its band 3,300–4,300",
        100.0 * aux_share,
        round::MAIN_COLUMNS,
        round_i.div_ceil(2),
        row_cells(round::MAIN_COLUMNS, round_i),
        round::PREP_WIDTH,
        round::NUM_COLUMNS + 3 * ra,
        head::MAIN_COLUMNS,
        head_i.div_ceil(2),
        row_cells(head::MAIN_COLUMNS, head_i),
        head::PREP_WIDTH,
        head::NUM_COLUMNS + 3 * ha,
        t_m,
        table::PREP_WIDTH,
        table::ROWS * (t_m - table::PREP_WIDTH + 3 * t_a),
        row_cells(head::MAIN_COLUMNS, head_i),
        cells as f64 / 5_545.0,
        cells as f64 / 3_800.0,
    );
    // What the same chip would cost with three interactions an aux column
    // (degree-4 LogUp batching, a STARK change), for the re-price.
    let deg4 = ROUNDS * (round::MAIN_COLUMNS + 3 * round_i.div_ceil(3))
        + head::MAIN_COLUMNS
        + 3 * head_i.div_ceil(3);
    println!("B3R CENSUS IF DEGREE-4 LOGUP (3 a column): {deg4} cells a compression");
}

/// Every constraint is at most cubic and some reach 3; the round row has 64,
/// the head 48; every index is emitted once.
#[test]
fn the_lean_constraints_are_cubic_and_dense() {
    use stark::constraints::builder::{
        CaptureBuilder, ConstraintSet, RootKind, num_base_from_meta,
    };
    fn check<C: ConstraintSet<F, E>>(set: &C, count: usize, what: &str) {
        let meta = set.meta();
        assert_eq!(meta.len(), count, "{what}: constraint count");
        for (i, m) in meta.iter().enumerate() {
            assert_eq!(m.constraint_idx, i, "{what}: dense and ordered");
            assert_eq!(m.kind, RootKind::Base, "{what}: base constraints");
        }
        let mut cb = CaptureBuilder::<F, E>::new();
        set.eval(&mut cb);
        let (_prog, degrees) = cb.finish(num_base_from_meta(&meta));
        assert!(degrees.iter().all(|&(_, d)| d <= 3), "{what}: degree ≤ 3");
        assert_eq!(
            degrees.iter().map(|&(_, d)| d).max(),
            Some(3),
            "{what}: some cubic"
        );
    }
    check(
        &RoundConstraints::default(),
        lean::ROUND_CONSTRAINTS,
        "round",
    );
    check(&HeadConstraints, lean::HEAD_CONSTRAINTS, "head");
}

/// `7·N` rows chunk with no padding: `2^(k+2) + 2^(k+1) + 2^k`.
#[test]
fn seven_n_rows_chunk_exactly() {
    assert_eq!(
        lean::chunk_rows(7 << 17, 19, 3),
        vec![1 << 19, 1 << 18, 1 << 17]
    );
    assert_eq!(
        lean::chunk_rows(7 << 18, 19, 3),
        vec![1 << 19, 1 << 19, 1 << 19, 1 << 18]
    );
    assert_eq!(lean::chunk_rows(7 * 64, 19, 3), vec![256, 128, 64]);
    assert_eq!(lean::chunk_rows(21, 19, 3), vec![16, 8]);
}

// =========================================================================
// The digests are BLAKE3's
// =========================================================================

/// The chip's witness digests — what its head writes to memory — are the
/// `blake3` crate's and the base backend's: a 64-felt leaf (eight chained
/// compressions), a 3-felt leaf (one partial block) and a parent.
#[test]
fn the_lean_digests_are_blake3() {
    use crypto::merkle_tree::backends::types::BatchBlake3StdBackend;
    use crypto::merkle_tree::traits::IsMerkleTreeBackend;

    let ops = probe_ops(12);
    let t = build_traces(&ops, &Spec::default());
    for (op, d) in ops.iter().zip(&t.digests) {
        assert_eq!(
            *d,
            op.digest(),
            "the head's digest is the op's 7-round compression"
        );
    }
    // The 64-felt leaf: the chain's last digest.
    let felts: Vec<FE> = ops[..8]
        .iter()
        .flat_map(|op| op.felts())
        .map(FE::from)
        .collect();
    let bytes: Vec<u8> = ops[..8].iter().flat_map(|op| op.block).collect();
    let leaf = digest_bytes(&t.digests[7]);
    assert_eq!(
        leaf,
        *blake3::hash(&bytes).as_bytes(),
        "64-felt leaf = blake3::hash"
    );
    assert_eq!(
        leaf,
        BatchBlake3StdBackend::<F>::hash_data(&felts),
        "64-felt leaf = the base's BLAKE3 leaf"
    );
    // The partial block.
    let short = digest_bytes(&t.digests[8]);
    assert_eq!(short, *blake3::hash(&ops[8].block[..24]).as_bytes());
    let short_felts: Vec<FE> = ops[8].felts()[..3].iter().map(|&f| FE::from(f)).collect();
    assert_eq!(short, BatchBlake3StdBackend::<F>::hash_data(&short_felts));
    // The parent of the two leaves.
    let parent = digest_bytes(&t.digests[9]);
    assert_eq!(parent, *blake3::hash(&ops[9].block).as_bytes());
    assert_eq!(
        parent,
        BatchBlake3StdBackend::<F>::hash_new_parent(&leaf, &short),
        "parent = the base's BLAKE3 parent"
    );
    // The fixed table saw every lookup the chip sends.
    let per_compression = ROUNDS * 128 + 32 + 32 + 8;
    assert_eq!(t.hist.total() as usize, ops.len() * per_compression);
}

// =========================================================================
// Prove + verify
// =========================================================================

#[test]
fn the_lean_chip_proves_and_verifies_under_blake3_commitments() {
    assert_eq!(
        round_trip(&probe_ops(16), Spec::default(), |_| {}),
        Ok(true)
    );
}

#[test]
fn the_lean_chip_proves_and_verifies_under_p1_commitments() {
    assert_eq!(
        round_trip_with::<P1Arm>(&probe_ops(16), Spec::default(), |_| {}),
        Ok(true)
    );
}

/// 64 compressions, the brief's ceiling, in three exact chunks (256 + 128 + 64).
#[test]
fn the_lean_chip_proves_64_compressions_in_three_chunks() {
    let ops = probe_ops(64);
    let t = build_traces(&ops, &Spec::default());
    assert_eq!(
        t.rounds
            .iter()
            .map(TraceTable::num_rows)
            .collect::<Vec<_>>(),
        vec![256, 128, 64]
    );
    assert_eq!(round_trip(&ops, Spec::default(), |_| {}), Ok(true));
}

/// Small chunks: a chain that crosses four tables still balances.
#[test]
fn the_round_chain_crosses_chunks() {
    let spec = Spec {
        max_chunk_log: 4,
        ..Spec::default()
    };
    let ops = probe_ops(10);
    assert_eq!(build_traces(&ops, &spec).rounds.len(), 5);
    assert_eq!(round_trip(&ops, spec, |_| {}), Ok(true));
}

/// A padding row is entirely zero and is what μ = 0 stands behind.
#[test]
fn padding_rows_are_all_zero() {
    let t = build_traces(&probe_ops(10), &Spec::default());
    let last = t.rounds.last().expect("a chunk");
    let row = last.main_table.get_row(last.num_rows() - 1);
    assert!(row.iter().all(|c| *c == FE::zero()));
}

// =========================================================================
// Negatives
// =========================================================================

fn at(round: usize, g: usize, kind: FaultKind) -> Spec {
    Spec {
        fault: Some((9, Fault { round, g, kind })),
        ..Spec::default()
    }
}

/// CONTROL: if this fails every negative below reports on a broken harness.
#[test]
fn negative_control_the_untampered_proof_verifies() {
    assert_eq!(
        round_trip(&probe_ops(12), Spec::default(), |_| {}),
        Ok(true)
    );
}

/// A wrong carry: `a' = a + b + m + 1`, everything downstream consistent. Only
/// the implicit add3 carry sees it (S = k·2³² − 1).
#[test]
fn a_wrong_add3_carry_rejects() {
    assert_not_accepted(
        "wrong add3 carry",
        round_trip(&probe_ops(12), at(3, 5, FaultKind::Carry3), |_| {}),
    );
}

/// The same for an add2 (`c' = c + d' + 1`).
#[test]
fn a_wrong_add2_carry_rejects() {
    assert_not_accepted(
        "wrong add2 carry",
        round_trip(&probe_ops(12), at(0, 2, FaultKind::Carry2), |_| {}),
    );
}

/// A top bit flipped in place: a rebuilt byte leaves [0, 256) and the next
/// lookup finds no row.
#[test]
fn a_flipped_top_bit_rejects() {
    assert_not_accepted(
        "flipped top bit",
        round_trip(&probe_ops(12), Spec::default(), |t| {
            let col = round::g_base(2) + round::T + 1;
            let v = *t.rounds[0].main_table.get(5, col);
            t.rounds[0].main_table.set_fe(5, col, FE::one() - v);
        }),
    );
}

/// A top bit set to 2: booleanity.
#[test]
fn a_non_boolean_top_bit_rejects() {
    assert_not_accepted(
        "non-boolean top bit",
        round_trip(&probe_ops(12), Spec::default(), |t| {
            let col = round::g_base(6) + round::T + 3;
            t.rounds[0].main_table.set_fe(3, col, FE::from(2u64));
        }),
    );
}

/// A wrong rotation at 12 (⋙13), consistent downstream.
#[test]
fn a_wrong_rotation_at_12_rejects() {
    assert_not_accepted(
        "rotation 13 for 12",
        round_trip(&probe_ops(12), at(1, 0, FaultKind::Rot13), |_| {}),
    );
}

/// A wrong rotation at 7 (⋙6), consistent downstream.
#[test]
fn a_wrong_rotation_at_7_rejects() {
    assert_not_accepted(
        "rotation 6 for 7",
        round_trip(&probe_ops(12), at(6, 7, FaultKind::Rot6), |_| {}),
    );
}

/// A split nibble swapped (lo ↔ hi of one `x2` byte), at the first byte whose
/// two nibbles differ — a swap of equal nibbles changes nothing.
#[test]
fn a_swapped_split_nibble_rejects() {
    assert_not_accepted(
        "swapped split nibble",
        round_trip(&probe_ops(12), Spec::default(), |t| {
            let tb = &mut t.rounds[0].main_table;
            let (row, col) = (0..8)
                .flat_map(|r| (0..8).flat_map(move |g| (0..4).map(move |k| (r, g, k))))
                .map(|(r, g, k)| (r, round::g_base(g) + round::P2 + 2 * k))
                .find(|&(r, c)| tb.get(r, c) != tb.get(r, c + 1))
                .expect("some byte has two different nibbles");
            let (lo, hi) = (*tb.get(row, col), *tb.get(row, col + 1));
            tb.set_fe(row, col, hi);
            tb.set_fe(row, col + 1, lo);
        }),
    );
}

/// A digest that is not BLAKE3's (1): one compression of six rounds, its
/// digest consistent in memory. The chain must reach round 7.
#[test]
fn a_six_round_digest_rejects() {
    let spec = Spec {
        six_round_op: Some(9),
        ..Spec::default()
    };
    let ops = probe_ops(12);
    assert_ne!(build_traces(&ops, &spec).digests[9], ops[9].digest());
    assert_not_accepted("six-round digest", round_trip(&ops, spec, |_| {}));
}

/// A digest that is not BLAKE3's (2): memory claims a different digest word.
#[test]
fn a_wrong_digest_in_memory_rejects() {
    assert_not_accepted(
        "wrong digest in memory",
        round_trip(&probe_ops(12), Spec::default(), |t| {
            // The mirror's rows that receive a digest carry RECV_MULT ≠ 0.
            let rows = t.mirror.num_rows();
            let r = (0..rows)
                .find(|&r| *t.mirror.main_table.get(r, mirror::RECV_MULT) != FE::zero())
                .expect("a digest row");
            let v = *t.mirror.main_table.get(r, mirror::V0 + 2);
            t.mirror.main_table.set_fe(r, mirror::V0 + 2, v + FE::one());
        }),
    );
}

/// A digest that is not BLAKE3's (3): one digest byte changed in the head.
#[test]
fn a_tampered_digest_byte_rejects() {
    assert_not_accepted(
        "tampered digest byte",
        round_trip(&probe_ops(12), Spec::default(), |t| {
            let col = head::OB + 13;
            let v = *t.head.main_table.get(4, col);
            t.head.main_table.set_fe(4, col, v + FE::one());
        }),
    );
}

/// An IV row whose `h` is not the IV.
#[test]
fn a_wrong_iv_rejects() {
    assert_not_accepted(
        "wrong IV",
        round_trip(&probe_ops(12), Spec::default(), |t| {
            let v = *t.head.main_table.get(9, head::HL + 3);
            t.head.main_table.set_fe(9, head::HL + 3, v + FE::one());
        }),
    );
}

/// A leaf felt hashed through its non-canonical encoding `v + p` (the felt
/// memory holds is `v`): canonicity.
#[test]
fn a_non_canonical_leaf_felt_rejects() {
    let mut ops = probe_ops(12);
    // Op 8 is the 3-felt leaf: re-encode its first felt as v + p.
    let p: u64 = 0xFFFF_FFFF_0000_0001;
    let v: u64 = 5;
    ops[8].block[..8].copy_from_slice(&(v + p).to_be_bytes());
    assert_not_accepted(
        "non-canonical felt",
        round_trip(&ops, Spec::default(), |_| {}),
    );
}

// =========================================================================
// Mutations: each check is load-bearing
// =========================================================================

/// With the implicit-carry identity dropped, the wrong-carry witness is
/// ACCEPTED: nothing else in the chip sees it.
#[test]
fn mutation_the_add3_carry_identity_is_load_bearing() {
    let mut spec = at(3, 5, FaultKind::Carry3);
    spec.drops.add3_carry = true;
    assert_eq!(
        round_trip(&probe_ops(12), spec, |_| {}),
        Ok(true),
        "with the identity dropped the forged carry must verify — else the \
         negative above proves nothing about the identity"
    );
}

/// The forged `b''` (low bit flipped, top bits solved over the field) is
/// rejected with booleanity and ACCEPTED without it.
#[test]
fn mutation_top_bit_booleanity_is_load_bearing() {
    let spec = at(4, 3, FaultKind::ForgedB2);
    assert_not_accepted(
        "forged b'' (booleanity on)",
        round_trip(&probe_ops(12), spec, |_| {}),
    );
    let mut dropped = spec;
    dropped.drops.top_bits = true;
    assert_eq!(round_trip(&probe_ops(12), dropped, |_| {}), Ok(true));
}

/// The solver reproduces the true bits for the honest `b''`.
#[test]
fn the_top_bit_solver_is_exact_on_honest_input() {
    for z in [0u32, 1, 0x8000_0000, 0xDEAD_BEEF, 0x7F7F_7F7F, u32::MAX] {
        let bits = lean::solve_top_bits(z, z.rotate_right(7));
        for (k, t) in bits.iter().enumerate() {
            assert_eq!(
                *t,
                FE::from(u64::from((z >> (8 * k + 7)) & 1)),
                "z {z:#x} bit {k}"
            );
        }
    }
}

// =========================================================================
// A parent's child check, from widths × queries
// =========================================================================

/// The tables of an S4-shaped program: round chunks, head, table, mirror, as
/// `(name, air, trace rows)`. Roots are placeholders: shapes do not read them.
fn s4_shape_tables(log_n: u32, max_chunk_log: u32) -> (Airs, Vec<usize>) {
    let n = 1usize << log_n;
    let chunks = lean::chunk_rows(ROUNDS * n, max_chunk_log, lean::MIN_CHUNK_LOG);
    let roots = Roots {
        rounds: vec![[0u8; 32]; chunks.len()],
        head: [0u8; 32],
        table: [0u8; 32],
    };
    let a = airs(&roots, &options(), Drops::default());
    // The S4 mix (`mixed_ops`): per 24 compressions, 8 leaf blocks (7 chained)
    // and 16 parents — about 7.1 memory words a compression.
    let mirror_rows = ((n as f64 * (8.0 * 4.0 + 7.0 * 2.0 + 16.0 * 6.0) / 24.0).ceil() as usize)
        .next_power_of_two();
    let mut rows = chunks;
    rows.extend([n, table::ROWS, mirror_rows]);
    (a, rows)
}

/// BLAKE3 compressions to verify one table's sub-proof (every query's leaves
/// and paths, its FRI, its caps), binary trees at the LFM format.
fn child_check_compressions(air: DynAir<'_>, rows: usize) -> usize {
    use super::epoch_verify::{TableVerifyShape, table_permutations_for};
    let (shape, _) = TableVerifyShape::derive_with(air, rows, stark::config::TreeGeometry::BINARY)
        .expect("a shape at the LFM format");
    table_permutations_for(&shape, super::edsl::WrapHash::Blake3)
}

/// ★ What a parent pays to check a child's BLAKE3 tables, from the widths and
/// the query count alone (the closed form the emitted legs are pinned against,
/// `epoch_verify::table_permutations_for`), at S4's two sizes and two chunk
/// caps. The non-hash tables of a real leaf are added in the re-price.
#[test]
fn the_child_check_of_the_lean_tables() {
    let opts = options();
    println!(
        "\nB3R CHILD CHECK: format blowup {} · queries {} · binary BLAKE3 trees",
        opts.blowup_factor, opts.fri_number_of_queries
    );
    for log_n in [17u32, 18] {
        for cap in [19u32, 21] {
            let (a, rows) = s4_shape_tables(log_n, cap);
            let refs = a.refs();
            let mut total = 0usize;
            let mut parts = Vec::new();
            for (air, &r) in refs.iter().zip(&rows) {
                let c = child_check_compressions(*air, r);
                total += c;
                parts.push(format!("{}@2^{}={c}", air_name(*air), r.trailing_zeros()));
            }
            println!(
                "B3R CHILD CHECK 2^{log_n} compressions, chunk cap 2^{cap}: {total} compressions \
                 ({:.3} G cells at 4,323) · {}",
                total as f64 * 4_323.0 / 1e9,
                parts.join(" · ")
            );
        }
    }
    // The same leg for one wide row a compression at today's socket width
    // (3,460 main + 13 preprocessed + 695 aux), 2^18 rows: the width effect.
    let wide = synthetic("WIDE_5545", 3_460 + 13, 13, 1_390, &opts);
    let w = child_check_compressions(&wide, 1 << 18);
    println!(
        "B3R CHILD CHECK WIDE (today's socket shape, one row a compression, 2^18 rows): {w} \
         compressions ({:.3} G cells at 5,545)",
        w as f64 * 5_545.0 / 1e9
    );
}

/// A table of the given shape with no constraints and `interactions` sends,
/// for the closed form only.
fn synthetic(
    name: &str,
    main: usize,
    prep: usize,
    interactions: usize,
    opts: &ProofOptions,
) -> PlainAir {
    let air = AirWithBuses::<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints>::new(
        main,
        AuxiliaryTraceBuildData {
            interactions: (0..interactions)
                .map(|i| {
                    BusInteraction::sender(
                        0xB3_FFu64,
                        Multiplicity::Column(0),
                        vec![Lin::col(prep + i % (main - prep)).bus()],
                    )
                })
                .collect(),
        },
        opts,
        1,
        EmptyConstraints,
    )
    .with_name(name);
    if prep > 0 {
        air.with_preprocessed([0u8; 32], prep)
    } else {
        air
    }
}

/// What today's Poseidon1 child check spends on its hash table, and how much
/// dearer a narrow non-hash table's check is under BLAKE3's binary trees than
/// under Poseidon1's 4-ary ones (`r` for the non-hash part of a child).
#[test]
fn the_child_check_hash_ratio_of_narrow_tables() {
    use super::epoch_verify::{TableVerifyShape, table_permutations_for};
    let opts = options();
    let p1_geom = stark::config::TreeGeometry::of::<super::p1_commit::P1StarkHash>();
    let under = |air: &PlainAir, rows: usize, geom, hash| {
        let (shape, _) = TableVerifyShape::derive_with(air, rows, geom).expect("a shape");
        table_permutations_for(&shape, hash)
    };
    // The P1 W16 socket: 329 columns (13 preprocessed), 8 interactions.
    let socket = synthetic("P1_SOCKET", 329, 13, 8, &opts);
    let s_p1 = under(&socket, 1 << 19, p1_geom, super::edsl::WrapHash::Poseidon1);
    println!("\nB3R CHILD CHECK P1 SOCKET (329 cols, 2^19 rows, P1 4-ary C2): {s_p1} permutations");
    // D-B3-3's non-hash leaf tables, at the median leaf's heights (aux: 2 ext columns each, I).
    let mut p1_sum = 0usize;
    let mut b3_sum = 0usize;
    for (name, main, log) in [
        ("XALU", 23, 22),
        ("SELECT", 25, 21),
        ("LANES", 16, 21),
        ("BALU", 14, 21),
        ("HINT", 6, 21),
    ] {
        let air = synthetic(name, main, 0, 4, &opts);
        let p = under(&air, 1 << log, p1_geom, super::edsl::WrapHash::Poseidon1);
        let b = under(
            &air,
            1 << log,
            stark::config::TreeGeometry::BINARY,
            super::edsl::WrapHash::Blake3,
        );
        p1_sum += p;
        b3_sum += b;
        println!(
            "B3R CHILD CHECK NON-HASH {name} ({main} cols, 2^{log}): P1 {p} · B3 {b} · r {:.3}",
            b as f64 / p as f64
        );
    }
    println!(
        "B3R CHILD CHECK NON-HASH r (sum): {:.3} (P1 {p1_sum}, B3 {b3_sum})",
        b3_sum as f64 / p1_sum as f64
    );
}

fn air_name(air: DynAir<'_>) -> String {
    air.name().to_string()
}

// =========================================================================
// S4: the card
// =========================================================================

/// Cells a proof covers, three ways: `(main + 3·aux` with the preprocessed
/// columns, the same without them, main only`)` — all at padded rows.
fn proved_cells(airs: &Airs, t: &Traces) -> (u64, u64, u64) {
    let refs = airs.refs();
    let rows: Vec<usize> = t
        .rounds
        .iter()
        .map(TraceTable::num_rows)
        .chain([t.head.num_rows(), t.table.num_rows(), t.mirror.num_rows()])
        .collect();
    let mut out = (0u64, 0u64, 0u64);
    for (air, &r) in refs.iter().zip(&rows) {
        let (main, aux) = air.trace_layout();
        let prep = if air.is_preprocessed() {
            air.num_precomputed_columns()
        } else {
            0
        };
        let r = r as u64;
        out.0 += r * (main + 3 * aux) as u64;
        out.1 += r * (main - prep + 3 * aux) as u64;
        out.2 += r * main as u64;
    }
    out
}

fn threads() -> usize {
    #[cfg(feature = "parallel")]
    return rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    1
}

fn env_list(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// The card's pool high-water over `f`, in bytes: the larger of the pool in
/// use before and its high-water after (the driver resets the mark to zero).
#[cfg(feature = "cuda")]
fn pool_peak<R>(f: impl FnOnce() -> R) -> (R, Option<u64>) {
    let _ = math_cuda::device::drain_and_trim();
    let before = math_cuda::device::pool_used_bytes()
        .ok()
        .map(|(now, _)| now);
    let _ = math_cuda::device::reset_pool_high_water();
    let r = f();
    let _ = math_cuda::device::drain_and_trim();
    let after = math_cuda::device::pool_used_bytes()
        .ok()
        .map(|(_, high)| high);
    (r, after.map(|h| h.max(before.unwrap_or(0))))
}

#[cfg(not(feature = "cuda"))]
fn pool_peak<R>(f: impl FnOnce() -> R) -> (R, Option<u64>) {
    (f(), None)
}

fn s4_run<A: Arm>(roots: &Roots, traces: &Traces, opts: &ProofOptions, label: &str, log_n: u32) {
    let airs = airs(roots, opts, Drops::default());
    let mut t = traces.clone_traces();
    let (cells, cells_noprep, main_only) = proved_cells(&airs, &t);
    let start = std::time::Instant::now();
    let (proof, peak) = pool_peak(|| prove::<A>(&airs, &mut t));
    let secs = start.elapsed().as_secs_f64();
    let proof = proof.expect("the S4 prove");
    let v = std::time::Instant::now();
    let ok = verify::<A>(&airs, &proof);
    let vsecs = v.elapsed().as_secs_f64();
    let peak_s = peak.map_or("-".to_string(), |b| {
        format!(
            "{:.2} GiB · {:.2} B/cell",
            b as f64 / (1u64 << 30) as f64,
            b as f64 / cells as f64
        )
    });
    println!(
        "B3R S4 RESULT: arm {} · run {label} · log_n {log_n} · compressions {} · cells {cells} \
         (no-prep {cells_noprep}, main-only {main_only}) · prove {secs:.3} s · c {:.3} ns/cell \
         (no-prep {:.3}, main-only {:.3}) · pool peak {peak_s} · verify {vsecs:.2} s · {}",
        A::NAME,
        1u64 << log_n,
        secs * 1e9 / cells as f64,
        secs * 1e9 / cells_noprep as f64,
        secs * 1e9 / main_only as f64,
        if ok { "verified" } else { "REJECTED" }
    );
    assert!(ok, "an S4 proof must verify");
}

impl Traces {
    fn clone_traces(&self) -> Traces {
        Traces {
            rounds: self.rounds.clone(),
            head: self.head.clone(),
            table: self.table.clone(),
            mirror: self.mirror.clone(),
            digests: Vec::new(),
            hist: TableHist::default(),
        }
    }
}

/// ★ S4 (box, `--features cuda`): prove 2^17 and 2^18 lean compressions under
/// the LFM's options on the card, alternating BLAKE3 and Poseidon1
/// commitments, every proof verified on the host.
///
/// Knobs (env): `B3R_S4_LOGN` (default `17,18`), `B3R_S4_ARMS` (default
/// `b3,p1,p1,b3,b3,p1`, after one untimed warm-up of each), `B3R_S4_CHUNK`
/// (the round chunk cap, default 19). Lines: `B3R S4 CONFIG`, `B3R S4 SHAPE`,
/// `B3R S4 RESULT` (one a run), `B3R S4 DONE`.
#[test]
#[ignore]
fn s4_the_lean_chip_on_the_card() {
    let opts = options();
    let logs: Vec<u32> = env_list("B3R_S4_LOGN", "17,18")
        .iter()
        .map(|s| s.parse().expect("B3R_S4_LOGN: integers"))
        .collect();
    let arms = env_list("B3R_S4_ARMS", "b3,p1,p1,b3,b3,p1");
    let chunk: u32 = std::env::var("B3R_S4_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(lean::MAX_CHUNK_LOG);
    println!(
        "B3R S4 CONFIG: blowup {} · queries {} · final poly 2^{} · format {:?} · cuda {} · \
         rayon threads {} · round chunk cap 2^{chunk} · arms {}",
        opts.blowup_factor,
        opts.fri_number_of_queries,
        opts.fri_final_poly_log_degree,
        opts.format,
        cfg!(feature = "cuda"),
        threads(),
        arms.join(",")
    );
    for log_n in logs {
        let n = 1usize << log_n;
        let spec = Spec {
            max_chunk_log: chunk,
            ..Spec::default()
        };
        let b = std::time::Instant::now();
        let ops = mixed_ops(n, 0x5EED_0000 + u64::from(log_n));
        let traces = build_traces(&ops, &spec);
        let build_s = b.elapsed().as_secs_f64();
        let leaf = ops.iter().filter(|o| o.felt_mode).count();
        for (op, d) in ops.iter().zip(&traces.digests).step_by(997) {
            assert_eq!(*d, op.digest(), "S4 digests are the 7-round compression");
        }
        let r = std::time::Instant::now();
        let roots_b3 = roots_under::<<B3Arm as Arm>::S>(&traces, &opts);
        let roots_p1 = roots_under::<<P1Arm as Arm>::S>(&traces, &opts);
        let roots_s = r.elapsed().as_secs_f64();
        let shape_airs = airs(&roots_b3, &opts, Drops::default());
        let (cells, cells_noprep, main_only) = proved_cells(&shape_airs, &traces);
        println!(
            "B3R S4 SHAPE: log_n {log_n} · compressions {n} ({leaf} leaf blocks) · round chunks {:?} · \
             head 2^{} · table 2^16 · mirror 2^{} · cells {cells} (no-prep {cells_noprep}, \
             main-only {main_only}) · {:.1} cells a compression · trace build {build_s:.2} s · \
             roots (both hashes) {roots_s:.2} s",
            traces
                .rounds
                .iter()
                .map(|t| t.num_rows().trailing_zeros())
                .collect::<Vec<_>>(),
            traces.head.num_rows().trailing_zeros(),
            traces.mirror.num_rows().trailing_zeros(),
            cells_noprep as f64 / n as f64,
        );
        let mut seq: Vec<(String, String)> = vec![
            ("b3".to_string(), "warm".to_string()),
            ("p1".to_string(), "warm".to_string()),
        ];
        seq.extend(
            arms.iter()
                .enumerate()
                .map(|(i, a)| (a.clone(), format!("{}", i + 1))),
        );
        for (arm, label) in &seq {
            match arm.as_str() {
                "b3" => s4_run::<B3Arm>(&roots_b3, &traces, &opts, label, log_n),
                "p1" => s4_run::<P1Arm>(&roots_p1, &traces, &opts, label, log_n),
                other => panic!("B3R_S4_ARMS: unknown arm {other}"),
            }
        }
    }
    println!("B3R S4 DONE");
}
