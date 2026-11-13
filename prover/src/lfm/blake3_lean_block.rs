//! The lean BLAKE3 tables ([`super::super::blake3_lean`]) proved by the base's
//! own block prover ([`stark::multilinear_block`]), and timed on the card
//! beside the STARK prove of the same tables.
//!
//! Test-only, a child of the lean probe so it reads the probe's traces, AIRs
//! and timers as they are. The question it answers: what does a group of lean
//! BLAKE3 rows cost the block prover (commit, batched argue, opening) under
//! either commitment hash, against the per-table STARK the probe measured.
//!
//! # What is proved
//!
//! The probe's tables in its order (round chunks, head, byte table, memory
//! mirror), the head cut into chunks of at most 2^[`HEAD_CHUNK_LOG`] rows so no
//! table outgrows a group. Each is a table of the block, cut into groups by
//! [`multilinear_block::block_groups`] at the base's
//! [`crate::block_whir::BLOCK_GROUP_POLYS`], under the base's pinned format
//! ([`BlockFormat::pinned`]: blowup 2, the batched argue, one chain a run of a
//! group's polynomials). The commitment hash is a run's [`Commits`]: the base's
//! Poseidon1 with its one round-0 tree a run, Poseidon1 with a tree a
//! polynomial, or standard 7-round BLAKE3 ([`multilinear::whir_hash::Blake3Whir`])
//! with a tree a polynomial (the one-tree commit is Poseidon1's alone). Phase A
//! is [`BlockCommitted::commit`] with the production block options' tree drop,
//! narrowing and upload-ahead; phase B is
//! [`multilinear_block::block_prove_in_order`] in group order, under the
//! phase-B overlap a run names.
//!
//! # What binds the preprocessed columns
//!
//! The verifier evaluates them ([`TableLayout::statement_with_preprocessed`]):
//! no prepared stack is committed or opened. The verifier's copies are taken
//! from the probe's own build, as the STARK probe takes its preprocessed roots
//! from its traces; a recursion that ships this would derive them from the
//! program. So the prover's time here carries no prepared opening: a timing
//! probe, not a format.
//!
//! # The statement
//!
//! A tag, the commitment arm and every table's shape, absorbed before any root,
//! on both sides. The memory mirror closes `LfmMem` on its own, so the tables
//! owe a zero balance.

use multilinear::mle::Mle;
use multilinear::whir_chain::{ChainConfig, Round0Trees};
use multilinear::whir_hash::WhirHash;
use stark::multilinear_block::{self, BlockCommitted, BlockProved, GroupStamps, PhaseBOverlap};
use stark::multilinear_table::batched::VerifierChecks;
use stark::multilinear_table::{CommittedTable, TableLayout, TableStatement};

use crate::block_whir::{BLOCK_GROUP_POLYS, BlockFormat, BlockOptions};
use crate::multilinear_prove::{layout_of, stacks};

use super::*;

type P1H = crate::hash_pin::BlockWhirHash;
type B3H = multilinear::whir_hash::Blake3Whir;

const STATEMENT_TAG: &[u8] = b"LFM_B3R_BLOCK_PROBE_V2";

/// The head's chunk cap: 2^19 rows of 217 columns, ≈ 114 M cells, so three
/// chunks fill a group.
const HEAD_CHUNK_LOG: u32 = 19;

/// A run's commitments: a verifier-side constant both sides name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Commits {
    /// The base's: Poseidon1, one round-0 tree a run of a group's polynomials.
    P1,
    /// Poseidon1 with a round-0 tree a polynomial: the BLAKE3 arm's tree shape
    /// under the base's hash.
    P1Each,
    /// Standard 7-round BLAKE3, a round-0 tree a polynomial.
    B3,
}

impl Commits {
    fn parse(name: &str) -> Self {
        match name {
            "p1" => Commits::P1,
            "p1each" => Commits::P1Each,
            "b3" => Commits::B3,
            other => panic!("LVP_COMMITS: unknown arm {other} (p1, p1each, b3)"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Commits::P1 => "p1",
            Commits::P1Each => "p1each",
            Commits::B3 => "b3",
        }
    }

    fn config(self, shapes: &[(usize, usize)]) -> ChainConfig {
        let mut config = BlockFormat::pinned().chain_config(shapes);
        if self != Commits::P1 {
            config.format.round0 = Round0Trees::Each;
        }
        config
    }
}

// =========================================================================
// The tables, column by column
// =========================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Round,
    Head,
    Xor,
    Mirror,
}

/// Every table of a block in table order (round chunks, head chunks, the byte
/// table, the mirror), each as its columns.
struct Chunked {
    kinds: Vec<Kind>,
    columns: Vec<Vec<Vec<FE>>>,
    digests: Vec<[u32; 8]>,
}

impl Chunked {
    /// The probe's traces as they are (one head table), columns copied out.
    fn from_traces(t: &Traces) -> Self {
        let mut kinds = vec![Kind::Round; t.rounds.len()];
        kinds.extend([Kind::Head, Kind::Xor, Kind::Mirror]);
        let columns = t
            .rounds
            .iter()
            .chain([&t.head, &t.table, &t.mirror])
            .map(TraceTable::columns_main)
            .collect();
        Chunked {
            kinds,
            columns,
            digests: t.digests.clone(),
        }
    }

    fn heads(&self) -> usize {
        self.kinds.iter().filter(|&&k| k == Kind::Head).count()
    }

    fn rounds(&self) -> usize {
        self.kinds.iter().filter(|&&k| k == Kind::Round).count()
    }
}

/// Row-major rows into columns.
fn transpose(data: &[FE], width: usize) -> Vec<Vec<FE>> {
    let rows = data.len() / width;
    let column = |c: usize| (0..rows).map(|r| data[r * width + c]).collect::<Vec<FE>>();
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..width).into_par_iter().map(column).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..width).map(column).collect()
    }
}

/// One op's seven round rows and its head row, honest and seven rounds.
fn fill_op(
    rows: &mut [FE],
    head_row: &mut [FE],
    k: usize,
    op: &LeanOp,
    hist: &mut TableHist,
) -> [u32; 8] {
    let fs = lean::fill_rounds(rows, k as u64 + 1, op, None, ROUNDS, hist);
    lean::fill_head(head_row, k as u64 + 1, op, &fs, hist);
    digest_of(&fs)
}

/// The probe's honest tables for `ops`, built one round chunk at a time so the
/// host never holds a chunk's rows twice: each chunk is filled row by row,
/// turned into columns and its rows let go. Round chunks of at most
/// 2^`chunk_log` rows (`build_traces`' cut), head chunks of at most
/// 2^`head_chunk_log`.
fn build_chunked(ops: &[LeanOp], chunk_log: u32, head_chunk_log: u32) -> Chunked {
    use std::collections::BTreeMap;
    const W: usize = round::NUM_COLUMNS;
    const HW: usize = head::NUM_COLUMNS;
    let n = ops.len();
    let total = ROUNDS * n;
    let sizes = lean::chunk_rows(total, chunk_log, lean::MIN_CHUNK_LOG);
    let mut starts = Vec::with_capacity(sizes.len());
    let mut at = 0usize;
    for &size in &sizes {
        starts.push(at);
        at += size;
    }
    let head_rows = n.next_power_of_two().max(4);
    let mut head_data = zeroed_fe_vec(head_rows * HW);
    let mut digests = vec![[0u32; 8]; n];
    let mut hist = TableHist::default();
    // The ops whose seven rows straddle a chunk boundary: filled once, apart.
    let mut straddle: BTreeMap<usize, Vec<FE>> = BTreeMap::new();
    for &s in starts.iter().skip(1) {
        if s % ROUNDS != 0 && s < total {
            let k = s / ROUNDS;
            let mut rows = zeroed_fe_vec(ROUNDS * W);
            digests[k] = fill_op(
                &mut rows,
                &mut head_data[k * HW..(k + 1) * HW],
                k,
                &ops[k],
                &mut hist,
            );
            straddle.insert(k, rows);
        }
    }
    let mut kinds = Vec::new();
    let mut columns = Vec::new();
    for (&s, &size) in starts.iter().zip(&sizes) {
        let e = s + size;
        let mut data = zeroed_fe_vec(size * W);
        // The ops wholly inside [s, e).
        let k0 = s.div_ceil(ROUNDS).min(n);
        let k1 = (e.min(total) / ROUNDS).max(k0);
        if k1 > k0 {
            let rows = &mut data[(ROUNDS * k0 - s) * W..(ROUNDS * k1 - s) * W];
            let heads = &mut head_data[k0 * HW..k1 * HW];
            let digs = &mut digests[k0..k1];
            let part = &ops[k0..k1];
            #[cfg(feature = "parallel")]
            let h = {
                use rayon::prelude::*;
                rows.par_chunks_mut(ROUNDS * W)
                    .zip(heads.par_chunks_mut(HW))
                    .zip(digs.par_iter_mut())
                    .zip(part.par_iter())
                    .enumerate()
                    .fold(TableHist::default, |mut h, (i, (((r, hr), d), op))| {
                        *d = fill_op(r, hr, k0 + i, op, &mut h);
                        h
                    })
                    .reduce(TableHist::default, |a, b| a.merge(&b))
            };
            #[cfg(not(feature = "parallel"))]
            let h = {
                let mut h = TableHist::default();
                for (i, (((r, hr), d), op)) in rows
                    .chunks_mut(ROUNDS * W)
                    .zip(heads.chunks_mut(HW))
                    .zip(digs.iter_mut())
                    .zip(part.iter())
                    .enumerate()
                {
                    *d = fill_op(r, hr, k0 + i, op, &mut h);
                }
                h
            };
            hist = hist.merge(&h);
        }
        // The straddling ops' rows that fall in this chunk.
        for (&k, rows) in straddle.range(s.saturating_sub(ROUNDS) / ROUNDS..=e / ROUNDS) {
            for r in ROUNDS * k..ROUNDS * k + ROUNDS {
                if (s..e).contains(&r) {
                    let src = (r - ROUNDS * k) * W;
                    data[(r - s) * W..(r - s + 1) * W].copy_from_slice(&rows[src..src + W]);
                }
            }
        }
        kinds.push(Kind::Round);
        columns.push(transpose(&data, W));
    }
    let head_chunk = head_rows.min(1usize << head_chunk_log);
    let mut head_columns = transpose(&head_data, HW);
    drop(head_data);
    let chunks = head_rows / head_chunk;
    let mut heads: Vec<Vec<Vec<FE>>> = (0..chunks).map(|_| Vec::with_capacity(HW)).collect();
    for col in head_columns.iter_mut() {
        for (c, head) in heads.iter_mut().enumerate() {
            head.push(col[c * head_chunk..(c + 1) * head_chunk].to_vec());
        }
        *col = Vec::new();
    }
    for head in heads {
        kinds.push(Kind::Head);
        columns.push(head);
    }
    kinds.push(Kind::Xor);
    columns.push(lean::table_trace(&hist).columns_main());
    kinds.push(Kind::Mirror);
    columns.push(mirror_trace(ops, &digests).columns_main());
    Chunked {
        kinds,
        columns,
        digests,
    }
}

// =========================================================================
// The AIRs, the frame
// =========================================================================

/// The block's AIRs in table order: [`Chunked`]'s kinds.
struct BlockAirs {
    rounds: Vec<RoundAir>,
    heads: Vec<HeadAir>,
    table: PlainAir,
    mirror: PlainAir,
}

impl BlockAirs {
    /// The probe's AIRs with placeholder preprocessed roots (the block path
    /// reads none), one head AIR a head chunk.
    fn new(rounds: usize, heads: usize, drops: Drops) -> Self {
        let opts = options();
        let roots = Roots {
            rounds: vec![[0u8; 32]; rounds],
            head: [0u8; 32],
            table: [0u8; 32],
        };
        let a = airs(&roots, &opts, drops);
        let heads = (0..heads)
            .map(|_| {
                AirWithBuses::new(
                    head::NUM_COLUMNS,
                    AuxiliaryTraceBuildData {
                        interactions: lean::head_interactions(),
                    },
                    &opts,
                    1,
                    HeadConstraints,
                )
                .with_name("B3R_HEAD")
                .with_preprocessed([0u8; 32], head::PREP_WIDTH)
            })
            .collect();
        BlockAirs {
            rounds: a.rounds,
            heads,
            table: a.table,
            mirror: a.mirror,
        }
    }

    fn of(c: &Chunked, drops: Drops) -> Self {
        Self::new(c.rounds(), c.heads(), drops)
    }

    fn refs(&self) -> Vec<DynAir<'_>> {
        let mut v: Vec<DynAir<'_>> = self.rounds.iter().map(|a| a as DynAir<'_>).collect();
        v.extend(self.heads.iter().map(|a| a as DynAir<'_>));
        v.push(&self.table);
        v.push(&self.mirror);
        v
    }
}

/// What both sides derive from the tables' shapes: never read from a proof.
struct Frame {
    commits: Commits,
    /// `(main width, height in variables)` a table, in table order.
    shapes: Vec<(usize, usize)>,
    /// Tables a group.
    sizes: Vec<usize>,
    config: ChainConfig,
    /// The verifier's copy of each table's preprocessed columns.
    prep: Vec<Vec<Mle<F>>>,
}

fn frame(airs: &BlockAirs, c: &Chunked, commits: Commits) -> Result<Frame, String> {
    let refs = airs.refs();
    if refs.len() != c.columns.len() {
        return Err(format!(
            "{} AIRs for {} tables",
            refs.len(),
            c.columns.len()
        ));
    }
    let mut shapes = Vec::with_capacity(refs.len());
    let mut prep = Vec::with_capacity(refs.len());
    for (air, cols) in refs.iter().zip(&c.columns) {
        let width = air.trace_layout().0;
        let rows = cols.first().map_or(0, Vec::len);
        if cols.len() != width || !rows.is_power_of_two() || cols.iter().any(|c| c.len() != rows) {
            return Err(format!(
                "{}: {} columns of {rows} rows for a {width}-column AIR",
                air.name(),
                cols.len()
            ));
        }
        shapes.push((width, rows.trailing_zeros() as usize));
        let count = if air.is_preprocessed() {
            air.num_precomputed_columns()
        } else {
            0
        };
        prep.push(
            cols[..count]
                .iter()
                .map(|c| Mle::new(c.clone()).map_err(|e| format!("{}: {e:?}", air.name())))
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    let config = commits.config(&shapes);
    let sizes = multilinear_block::block_groups(&shapes, config.format.stack, BLOCK_GROUP_POLYS)
        .map_err(|e| format!("{e:?}"))?;
    Ok(Frame {
        commits,
        shapes,
        sizes,
        config,
        prep,
    })
}

fn absorb_statement<T: IsTranscript<E>>(transcript: &mut T, frame: &Frame) {
    transcript.append_bytes(STATEMENT_TAG);
    transcript.append_bytes(frame.commits.name().as_bytes());
    transcript.append_bytes(&(frame.shapes.len() as u64).to_le_bytes());
    for &(width, num_vars) in &frame.shapes {
        transcript.append_bytes(&(width as u64).to_le_bytes());
        transcript.append_bytes(&(num_vars as u64).to_le_bytes());
    }
}

/// The block's tables, their columns moved in.
fn committed_tables<'a>(
    airs: &'a BlockAirs,
    c: Chunked,
    frame: &Frame,
) -> Result<Vec<CommittedTable<'a, F, E>>, String> {
    airs.refs()
        .into_iter()
        .zip(c.columns)
        .zip(&frame.shapes)
        .map(|((air, mut columns), &(width, num_vars))| {
            let layout = layout_of(air, width, num_vars).map_err(|e| format!("{e:?}"))?;
            CommittedTable::from_layout(layout, |col| core::mem::take(&mut columns[col as usize]))
                .map_err(|e| format!("{}: {e:?}", air.name()))
        })
        .collect()
}

// =========================================================================
// Prove, verify
// =========================================================================

/// Seconds of a block prove, by phase.
struct BlockSecs {
    commit: f64,
    phase_b: f64,
}

/// Phase A then phase B under `Hs`, the tables moved in.
fn prove_block_under<Hs: WhirHash>(
    tables: Vec<CommittedTable<'_, F, E>>,
    frame: &Frame,
    overlap: PhaseBOverlap,
) -> Result<(BlockProved<F, E>, BlockSecs), String>
where
    <Hs as WhirHash>::Sponge: IsTranscript<E>
        + crypto::fiat_shamir::transcript_hash::HasTranscriptHash<Hash = <Hs as WhirHash>::Transcript>
        + Clone
        + Send
        + Default,
{
    let options = BlockOptions::production();
    let mut transcript = <Hs as WhirHash>::sponge();
    absorb_statement(&mut transcript, frame);
    let t = std::time::Instant::now();
    let block = BlockCommitted::commit::<Hs, <Hs as WhirHash>::Sponge>(
        tables,
        &frame.sizes,
        &frame.config,
        options.drop_levels,
        options.narrow,
        options.upload_ahead,
    )
    .map_err(|e| format!("commit: {e:?}"))?;
    let commit = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let order: Vec<usize> = (0..frame.sizes.len()).collect();
    let proved = multilinear_block::block_prove_in_order::<F, E, <Hs as WhirHash>::Sponge, Hs>(
        block.with_phase_b_overlap(overlap),
        &frame.config,
        &mut transcript,
        &[],
        &[],
        &|g| g,
        &multilinear_block::ArgueDeviation::default(),
        &|_| {},
        &order,
        None,
    )
    .map_err(|e| format!("prove: {e:?}"))?;
    let phase_b = t.elapsed().as_secs_f64();
    Ok((proved, BlockSecs { commit, phase_b }))
}

fn prove_block(
    tables: Vec<CommittedTable<'_, F, E>>,
    frame: &Frame,
    overlap: PhaseBOverlap,
) -> Result<(BlockProved<F, E>, BlockSecs), String> {
    match frame.commits {
        Commits::P1 | Commits::P1Each => prove_block_under::<P1H>(tables, frame, overlap),
        Commits::B3 => prove_block_under::<B3H>(tables, frame, overlap),
    }
}

/// The host's verdict on a block proof under `Hs`, against the frame alone.
fn verify_block_under<Hs: WhirHash>(
    airs: &BlockAirs,
    frame: &Frame,
    proved: &BlockProved<F, E>,
) -> Result<(), String>
where
    <Hs as WhirHash>::Sponge: IsTranscript<E>
        + crypto::fiat_shamir::transcript_hash::HasTranscriptHash<Hash = <Hs as WhirHash>::Transcript>
        + Clone
        + Send
        + Default,
{
    let (proof, argues, prepared, _, commit_ood) = proved;
    let refs = airs.refs();
    let layouts: Vec<TableLayout<'_, F, E>> = refs
        .iter()
        .zip(&frame.shapes)
        .map(|(air, &(width, num_vars))| layout_of(*air, width, num_vars))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("{e:?}"))?;
    let statements: Vec<TableStatement<'_, F, E>> = layouts
        .iter()
        .zip(&frame.prep)
        .map(|(layout, prep)| layout.statement_with_preprocessed(prep))
        .collect();
    let (stacked, domains) =
        stacks(&frame.shapes, &frame.sizes, &frame.config).map_err(|e| format!("{e:?}"))?;
    let mut transcript = <Hs as WhirHash>::sponge();
    absorb_statement(&mut transcript, frame);
    multilinear_block::block_verify_with::<F, E, <Hs as WhirHash>::Sponge, Hs>(
        proof,
        commit_ood,
        argues,
        prepared,
        &[],
        &statements,
        &stacked,
        &domains,
        &frame.sizes,
        &FEE::zero(),
        &frame.config,
        &mut transcript,
        false,
        VerifierChecks::ALL,
    )
    .map_err(|e| format!("{e:?}"))
}

fn verify_block(airs: &BlockAirs, frame: &Frame, proved: &BlockProved<F, E>) -> Result<(), String> {
    match frame.commits {
        Commits::P1 | Commits::P1Each => verify_block_under::<P1H>(airs, frame, proved),
        Commits::B3 => verify_block_under::<B3H>(airs, frame, proved),
    }
}

/// Build, optionally corrupt, prove under `prover` and verify under the
/// frame's commits (`verifier` when given). `Err` = the prover refused.
fn block_round_trip_with(
    ops: &[LeanOp],
    spec: Spec,
    prover: Commits,
    verifier: Option<Commits>,
    mutate: impl FnOnce(&mut Traces),
    corrupt: impl FnOnce(&mut Frame, &mut BlockProved<F, E>),
) -> Result<bool, String> {
    let mut t = build_traces(ops, &spec);
    mutate(&mut t);
    let c = Chunked::from_traces(&t);
    let airs = BlockAirs::of(&c, spec.drops);
    let mut f = frame(&airs, &c, prover)?;
    let tables = committed_tables(&airs, c, &f)?;
    let (mut proved, _) = prove_block(tables, &f, PhaseBOverlap::Off)?;
    corrupt(&mut f, &mut proved);
    if let Some(v) = verifier {
        f.commits = v;
        f.config = v.config(&f.shapes);
    }
    match verify_block(&airs, &f, &proved) {
        Ok(()) => Ok(true),
        Err(e) => {
            println!("LVP REJECTED: {}", e.chars().take(160).collect::<String>());
            Ok(false)
        }
    }
}

fn block_round_trip(
    ops: &[LeanOp],
    spec: Spec,
    mutate: impl FnOnce(&mut Traces),
    corrupt: impl FnOnce(&mut Frame, &mut BlockProved<F, E>),
) -> Result<bool, String> {
    block_round_trip_with(ops, spec, Commits::P1, None, mutate, corrupt)
}

// =========================================================================
// Laptop: the block path accepts the lean tables, and refuses forgeries
// =========================================================================

#[test]
fn the_lean_tables_prove_and_verify_through_the_block_prover() {
    let ops = probe_ops(64);
    let c = Chunked::from_traces(&build_traces(&ops, &Spec::default()));
    let airs = BlockAirs::of(&c, Drops::default());
    let f = frame(&airs, &c, Commits::P1).expect("a frame");
    assert_eq!(
        f.config.log_blowup, 1,
        "the pinned block format is blowup 2"
    );
    assert_eq!(
        f.sizes,
        vec![f.shapes.len()],
        "64 compressions are one group"
    );
    assert_eq!(
        block_round_trip(&ops, Spec::default(), |_| {}, |_, _| {}),
        Ok(true)
    );
}

/// The two other commitment arms prove and verify too.
#[test]
fn the_lean_tables_prove_and_verify_under_blake3_and_tree_a_poly_commits() {
    for commits in [Commits::B3, Commits::P1Each] {
        assert_eq!(
            block_round_trip_with(
                &probe_ops(16),
                Spec::default(),
                commits,
                None,
                |_| {},
                |_, _| {}
            ),
            Ok(true),
            "{commits:?}"
        );
    }
}

/// A proof under one commitment arm is not accepted under the other (the
/// statement names the arm, and the transcripts differ from the first root).
#[test]
fn a_proof_under_one_hash_rejects_under_the_other() {
    for (prover, verifier) in [
        (Commits::B3, Commits::P1Each),
        (Commits::P1Each, Commits::B3),
    ] {
        assert_not_accepted(
            &format!("block: proved {prover:?}, verified {verifier:?}"),
            block_round_trip_with(
                &probe_ops(12),
                Spec::default(),
                prover,
                Some(verifier),
                |_| {},
                |_, _| {},
            ),
        );
    }
}

/// The chunked build is the probe's build, table for table, with the head cut
/// into chunks.
#[test]
fn the_chunked_build_is_the_probes_build() {
    let ops = mixed_ops(100, 7);
    let spec = Spec {
        max_chunk_log: 5,
        ..Spec::default()
    };
    let t = build_traces(&ops, &spec);
    let whole = Chunked::from_traces(&t);
    // Head whole: the same tables exactly.
    let same = build_chunked(&ops, 5, 20);
    assert_eq!(same.kinds, whole.kinds);
    assert_eq!(same.columns, whole.columns);
    assert_eq!(same.digests, whole.digests);
    // Head in chunks of 2^4 rows: the same head, cut.
    let cut = build_chunked(&ops, 5, 4);
    assert_eq!(cut.heads(), 128 / 16);
    let heads: Vec<&Vec<Vec<FE>>> = cut
        .kinds
        .iter()
        .zip(&cut.columns)
        .filter(|(k, _)| **k == Kind::Head)
        .map(|(_, c)| c)
        .collect();
    let head = &whole.columns[whole.rounds()];
    for col in 0..head::NUM_COLUMNS {
        let joined: Vec<FE> = heads.iter().flat_map(|h| h[col].iter().copied()).collect();
        assert_eq!(joined, head[col], "head column {col}");
    }
    assert_eq!(
        &cut.columns[..cut.rounds()],
        &whole.columns[..whole.rounds()]
    );
    let n = cut.columns.len();
    assert_eq!(
        &cut.columns[n - 2..],
        &whole.columns[whole.columns.len() - 2..]
    );
}

/// The head in chunks proves and verifies.
#[test]
fn the_chunked_head_proves_and_verifies() {
    let c = build_chunked(&probe_ops(40), 6, 3);
    assert!(c.heads() > 1);
    let airs = BlockAirs::of(&c, Drops::default());
    let f = frame(&airs, &c, Commits::B3).expect("a frame");
    let tables = committed_tables(&airs, c, &f).expect("tables");
    let (proved, _) = prove_block(tables, &f, PhaseBOverlap::Off).expect("prove");
    assert_eq!(verify_block(&airs, &f, &proved), Ok(()));
}

/// The box's shapes group as the job registers them: 2^17 compressions one
/// group, 2^19 four, 2^21 sixteen (the mirror's height from the S4 mix's
/// estimate; head chunks of 2^19 rows).
#[test]
fn the_card_shapes_group_as_registered() {
    for (log_n, groups) in [(17u32, 1usize), (19, 4), (21, 16)] {
        let (a, rows) = s4_shape_tables(log_n, lean::MAX_CHUNK_LOG);
        let mut shapes: Vec<(usize, usize)> = a
            .refs()
            .iter()
            .zip(&rows)
            .map(|(air, &r)| (air.trace_layout().0, r.trailing_zeros() as usize))
            .collect();
        // The head (after the round chunks) in chunks of 2^HEAD_CHUNK_LOG rows.
        let h = shapes.len() - 3;
        let (width, vars) = shapes[h];
        let cut = vars.min(HEAD_CHUNK_LOG as usize);
        shapes.splice(
            h..h + 1,
            std::iter::repeat_n((width, cut), 1 << (vars - cut)),
        );
        let config = BlockFormat::pinned().chain_config(&shapes);
        let sizes =
            multilinear_block::block_groups(&shapes, config.format.stack, BLOCK_GROUP_POLYS)
                .expect("groups");
        assert_eq!(
            sizes.len(),
            groups,
            "2^{log_n}: groups {sizes:?} over {shapes:?}"
        );
    }
}

/// A wrong carry (consistent downstream): only the round constraint sees it,
/// so the block's constraint sumcheck must.
#[test]
fn a_wrong_carry_rejects_through_the_block_prover() {
    assert_not_accepted(
        "block: wrong add3 carry",
        block_round_trip(
            &probe_ops(12),
            at(3, 5, FaultKind::Carry3),
            |_| {},
            |_, _| {},
        ),
    );
}

/// A digest word in memory that the head did not write: the bus.
#[test]
fn a_wrong_digest_in_memory_rejects_through_the_block_prover() {
    assert_not_accepted(
        "block: wrong digest in memory",
        block_round_trip(
            &probe_ops(12),
            Spec::default(),
            |t| {
                let rows = t.mirror.num_rows();
                let r = (0..rows)
                    .find(|&r| *t.mirror.main_table.get(r, mirror::RECV_MULT) != FE::zero())
                    .expect("a digest row");
                let v = *t.mirror.main_table.get(r, mirror::V0 + 2);
                t.mirror.main_table.set_fe(r, mirror::V0 + 2, v + FE::one());
            },
            |_, _| {},
        ),
    );
}

/// The verifier's byte table differs from the committed one in one value: the
/// evaluated preprocessed columns are what binds the table.
#[test]
fn a_wrong_preprocessed_column_rejects_through_the_block_prover() {
    assert_not_accepted(
        "block: verifier's byte table differs",
        block_round_trip(
            &probe_ops(12),
            Spec::default(),
            |_| {},
            |f, _| {
                // The byte table sits after the round chunks and the head.
                let xor = f.prep.len() - 2;
                let mut col = f.prep[xor][table::X].evals().to_vec();
                col[7] += FE::one();
                f.prep[xor][table::X] = Mle::new(col).expect("same height");
            },
        ),
    );
}

/// One commit-time answer changed in the proof.
#[test]
fn a_tampered_commit_answer_rejects_through_the_block_prover() {
    assert_not_accepted(
        "block: tampered commit-time answer",
        block_round_trip(
            &probe_ops(12),
            Spec::default(),
            |_| {},
            |_, proved| {
                proved.4[0] = &proved.4[0] + &FEE::one();
            },
        ),
    );
}

// =========================================================================
// The card
// =========================================================================

fn overlap_of(name: &str) -> PhaseBOverlap {
    match name {
        "pinned" => PhaseBOverlap::Pinned,
        "on" => PhaseBOverlap::On,
        "off" => PhaseBOverlap::Off,
        other => panic!("LVP_OVERLAPS: unknown overlap {other} (pinned, on, off)"),
    }
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// A group's card promise at its peak: the largest of the ledger's promises
/// through its commit, its argue, its openings and its transition.
fn group_reserved(s: &GroupStamps) -> u64 {
    s.commit_reserved
        .max(s.argue_reserved)
        .max(s.open_reserved)
        .max(s.transition_reserved)
}

fn group_line(label: &str, g: usize, s: &GroupStamps, kinds: &str) -> String {
    format!(
        "LVP GROUP: {label} · group {g} · kinds {kinds} · tables {} · polys {} · cells {:.1} M · \
         upload {:.3} paid {:.3} · commit {:.3} · argue {:.3} · encode {:.3} · open {:.3} · \
         open paid {:.3} · wait {:.3} · retire {:.3} · ood {:.3} · overlapped {} · reserved {:.2} \
         GiB · {:.2} B a cell",
        s.tables,
        s.polys,
        s.cells as f64 / 1e6,
        s.upload_a,
        s.upload_paid,
        s.commit,
        s.argue,
        s.encode,
        s.open,
        s.open_paid,
        s.wait_a,
        s.retire,
        s.ood,
        s.overlapped,
        gib(group_reserved(s)),
        group_reserved(s) as f64 / s.cells.max(1) as f64,
    )
}

/// Each group's table kinds, as `R` round, `H` head, `X` byte table, `M`
/// mirror, in table order.
fn group_kinds(kinds: &[Kind], sizes: &[usize]) -> Vec<String> {
    let mut at = 0usize;
    sizes
        .iter()
        .map(|&size| {
            let s: String = kinds[at..at + size]
                .iter()
                .map(|k| match k {
                    Kind::Round => 'R',
                    Kind::Head => 'H',
                    Kind::Xor => 'X',
                    Kind::Mirror => 'M',
                })
                .collect();
            at += size;
            s
        })
        .collect()
}

/// One timed block prove: the tables built and moved in first (untimed), then
/// phase A and phase B under the pool's high-water, then the host verify.
fn block_run(ops: &[LeanOp], commits: Commits, overlap: &str, label: &str, log_n: u32) -> f64 {
    let b = std::time::Instant::now();
    let c = build_chunked(ops, lean::MAX_CHUNK_LOG, HEAD_CHUNK_LOG);
    let kinds = c.kinds.clone();
    let airs = BlockAirs::of(&c, Drops::default());
    let f = frame(&airs, &c, commits).expect("the frame");
    let tables = committed_tables(&airs, c, &f).expect("the block's tables");
    let build_s = b.elapsed().as_secs_f64();
    let main: u64 = f.shapes.iter().map(|&(w, v)| (w as u64) << v).sum();
    let group_kinds = group_kinds(&kinds, &f.sizes);
    // The STARK arm's clock: from before the pool's trim to after it.
    eprintln!(
        "LVP RUN BEGIN: arm block · commits {} · overlap {overlap} · run {label} · log_n {log_n} · \
         tables built in {build_s:.2} s (untimed)",
        commits.name()
    );
    let start = std::time::Instant::now();
    let (r, peak) = pool_peak(|| prove_block(tables, &f, overlap_of(overlap)));
    let secs = start.elapsed().as_secs_f64();
    let (proved, phases) = r.expect("the block prove");
    let v = std::time::Instant::now();
    let verdict = verify_block(&airs, &f, &proved);
    let vsecs = v.elapsed().as_secs_f64();
    let stamps = &proved.3;
    let run_label = format!(
        "run {label} · commits {} · overlap {overlap}",
        commits.name()
    );
    for (g, s) in stamps.iter().enumerate() {
        eprintln!("{}", group_line(&run_label, g, s, &group_kinds[g]));
    }
    // What the groups' own stamps account for, and what is left: the run's
    // seam as the stamps see it.
    let terms: f64 = stamps
        .iter()
        .map(|s| s.upload_paid + s.commit + s.argue + s.open_paid)
        .sum();
    let largest = stamps.iter().map(|s| s.cells).max().unwrap_or(1).max(1);
    let peak_s = peak.map_or("-".to_string(), |b| {
        format!(
            "{:.2} GiB · {:.2} B a main cell · {:.2} B a cell of the largest group",
            gib(b),
            b as f64 / main as f64,
            b as f64 / largest as f64
        )
    });
    eprintln!(
        "LVP RESULT: arm block · commits {} · overlap {overlap} · run {label} · log_n {log_n} · \
         compressions {} · groups {} · main cells {main} · prove {secs:.3} s (commit {:.3} · phase \
         B {:.3}) · group terms {terms:.3} s · rest {:.3} s · µs a compression {:.3} · pool peak \
         {peak_s} · verify {vsecs:.2} s · {}",
        commits.name(),
        1u64 << log_n,
        f.sizes.len(),
        phases.commit,
        phases.phase_b,
        secs - terms,
        secs * 1e6 / (1u64 << log_n) as f64,
        match &verdict {
            Ok(()) => "verified".to_string(),
            Err(e) => format!("REJECTED {e}"),
        }
    );
    assert!(verdict.is_ok(), "a block proof must verify");
    secs
}

/// Every digest the head wrote is the op's 7-round compression, and every
/// parent and every leaf chain's last digest is the `blake3` crate's hash.
fn digests_are_blake3(ops: &[LeanOp], digests: &[[u32; 8]]) -> (usize, usize) {
    assert_eq!(ops.len(), digests.len());
    let mut parents = 0usize;
    let mut leaves = 0usize;
    let mut chain: Vec<u8> = Vec::new();
    for (op, d) in ops.iter().zip(digests) {
        assert_eq!(*d, op.digest(), "a head digest is the 7-round compression");
        if !op.felt_mode {
            assert_eq!(
                digest_bytes(d),
                *blake3::hash(&op.block).as_bytes(),
                "a parent's digest is blake3::hash(left ‖ right)"
            );
            parents += 1;
            continue;
        }
        if op.flags & CHUNK_START != 0 {
            chain.clear();
        }
        chain.extend_from_slice(&op.block[..op.block_len as usize]);
        if op.flags & ROOT != 0 {
            assert_eq!(
                digest_bytes(d),
                *blake3::hash(&chain).as_bytes(),
                "a leaf chain's last digest is blake3::hash of its bytes"
            );
            leaves += 1;
        }
    }
    (parents, leaves)
}

/// ★ The card (box, `--features cuda`, `LAMBDA_VM_BASE_SPLIT=1`): the lean
/// tables through the block prover at each size and commitment arm, the STARK
/// prove of the same tables in the same binary as the control at 2^17 and
/// below.
///
/// Knobs (env): `LVP_LOGN` (default `17,19`), `LVP_RUNS` (default 3, after one
/// untimed warm-up of every arm), `LVP_COMMITS` (the block arms' commitments,
/// default `p1`; `p1`, `p1each`, `b3`), `LVP_OVERLAPS` (the block arms'
/// phase-B overlaps, default `pinned,on`), `LVP_STARK` (the STARK control's
/// arms at 2^17 and below, default `p1,b3`; empty for none). Timed run `r`
/// takes the arms in their listed order when `r` is odd and reversed when it is
/// even. Every block run builds its tables afresh (untimed), so the host holds
/// one block's columns at a time. Lines: `LVP CONFIG`, `LVP SHAPE`,
/// `LVP DIGESTS`, `LVP RUN BEGIN`, `LVP GROUP` (one a group a block run),
/// `LVP RESULT` (one a block run), `B3R S4 RESULT` (one a STARK run),
/// `LVP DONE`; under `LAMBDA_VM_BASE_SPLIT`, the batched argue's own
/// `BATCHED ARGUE …` lines, one set a group in group order.
#[test]
#[ignore]
fn lvp_the_lean_tables_through_the_block_prover_on_the_card() {
    let opts = options();
    let logs: Vec<u32> = env_list("LVP_LOGN", "17,19")
        .iter()
        .map(|s| s.parse().expect("LVP_LOGN: integers"))
        .collect();
    let runs: usize = std::env::var("LVP_RUNS")
        .ok()
        .map(|s| s.parse().expect("LVP_RUNS: an integer"))
        .unwrap_or(3);
    let commits: Vec<Commits> = env_list("LVP_COMMITS", "p1")
        .iter()
        .map(|s| Commits::parse(s))
        .collect();
    let overlaps = env_list("LVP_OVERLAPS", "pinned,on");
    for o in &overlaps {
        overlap_of(o);
    }
    let stark_arms = env_list("LVP_STARK", "p1,b3");
    let arms: Vec<(Commits, String)> = commits
        .iter()
        .flat_map(|&c| overlaps.iter().map(move |o| (c, o.clone())))
        .collect();
    let block_config = BlockFormat::pinned().chain_config(&[(1, 20)]);
    eprintln!(
        "LVP CONFIG: block blowup 2^{} · queries {} · stack 2^{} · argue {:?} · group {:?} · \
         round0 {:?} (p1; p1each and b3 Each) · leaves {:?} · group polys {BLOCK_GROUP_POLYS} · \
         head chunk 2^{HEAD_CHUNK_LOG} · split lines {} · STARK blowup {} queries {} · cuda {} · \
         rayon threads {} · runs {runs} · commits {} · overlaps {} · stark arms {}",
        block_config.log_blowup,
        block_config.num_queries,
        block_config.format.stack.get(),
        block_config.format.argue,
        block_config.format.group,
        block_config.format.round0,
        block_config.format.leaves,
        multilinear::whir_split::enabled(),
        opts.blowup_factor,
        opts.fri_number_of_queries,
        cfg!(feature = "cuda"),
        threads(),
        commits
            .iter()
            .map(|c| c.name())
            .collect::<Vec<_>>()
            .join(","),
        overlaps.join(","),
        stark_arms.join(","),
    );
    for log_n in logs {
        let n = 1usize << log_n;
        let ops = mixed_ops(n, 0x5EED_0000 + u64::from(log_n));
        {
            let b = std::time::Instant::now();
            let c = build_chunked(&ops, lean::MAX_CHUNK_LOG, HEAD_CHUNK_LOG);
            let build_s = b.elapsed().as_secs_f64();
            let (parents, leaves) = digests_are_blake3(&ops, &c.digests);
            eprintln!(
                "LVP DIGESTS: log_n {log_n} · {n} head digests = the 7-round compression · \
                 {parents} parents and {leaves} leaf chains = blake3::hash"
            );
            let airs = BlockAirs::of(&c, Drops::default());
            let f = frame(&airs, &c, Commits::P1).expect("the frame");
            let main: u64 = f.shapes.iter().map(|&(w, v)| (w as u64) << v).sum();
            let kinds = group_kinds(&c.kinds, &f.sizes);
            eprintln!(
                "LVP SHAPE: log_n {log_n} · compressions {n} · tables {} · groups {} [{}] · main \
                 cells {main} · {:.1} main cells a compression · queries {} · tables built in \
                 {build_s:.2} s",
                f.shapes.len(),
                f.sizes.len(),
                kinds.join(" "),
                main as f64 / n as f64,
                f.config.num_queries,
            );
        }
        let stark = log_n <= 17 && !stark_arms.is_empty();
        let traces = stark.then(|| build_traces(&ops, &Spec::default()));
        let stark_roots = traces.as_ref().map(|t| {
            (
                roots_under::<<P1Arm as Arm>::S>(t, &opts),
                roots_under::<<B3Arm as Arm>::S>(t, &opts),
            )
        });
        let stark_run = |arm: &str, label: &str| {
            let (p1, b3) = stark_roots.as_ref().expect("STARK roots");
            let t = traces.as_ref().expect("STARK traces");
            match arm {
                "p1" => s4_run::<P1Arm>(p1, t, &opts, label, log_n),
                "b3" => s4_run::<B3Arm>(b3, t, &opts, label, log_n),
                other => panic!("LVP_STARK: unknown arm {other}"),
            }
        };
        // Warm-ups: every arm once, untimed.
        if stark {
            for arm in &stark_arms {
                stark_run(arm, "warm");
            }
        }
        for (c, o) in &arms {
            block_run(&ops, *c, o, "warm", log_n);
        }
        for run in 1..=runs {
            let label = format!("{run}");
            if stark {
                for arm in &stark_arms {
                    stark_run(arm, &label);
                }
            }
            let mut order: Vec<&(Commits, String)> = arms.iter().collect();
            if run % 2 == 0 {
                order.reverse();
            }
            for (c, o) in order {
                block_run(&ops, *c, o, &label, log_n);
            }
        }
    }
    eprintln!("LVP DONE");
}
