//! Local lookups argued per group ([`BlockCommitted::with_local_lookups`]) on
//! a small block: the multi-table example's CPU, ADD and MUL in a group of
//! their own (no local interaction), and groups of SEND tables whose byte
//! tuples are looked up in a fixed RECV table of the same group on two local
//! buses, under the group's own `(z_g, α_g)`.
//!
//! Round trips under each packing and bin cap; a group with no local
//! interaction argues today's bytes; the local transcript spelled out by hand
//! and moved by every input; and the forgeries the format must refuse, each
//! with the mutant (a check off, or a binding dropped on both sides) that
//! accepts it.

// The test's columns are column-major (`columns[c][row]`): rows are indexed.
#![allow(clippy::needless_range_loop)]

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::{
    element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as Fp,
};
use multilinear::{
    Error as MlError,
    mle::Mle,
    stacked_eval,
    stacking::StackedLayout,
    whir::Domain,
    whir_chain::{ArgueFormat, ChainConfig, GkrPack, GroupFold},
    whir_commit::{Commitment, TreeDrop},
    whir_hash::{KeccakWhir, WhirHash},
};

use crate::constraints::builder::EmptyConstraints;
use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, BusValue, LinearTerm, Multiplicity,
    NullBoundaryConstraintBuilder,
};
use crate::multilinear_air::Uniforms;
use crate::multilinear_block::{
    self, ArgueDeviation, BlockCommitted, BlockLocal, BlockLocalCheck, BlockProved,
    LOCAL_LOGUP_LABEL, Narrowing,
};
use crate::multilinear_table::{
    CommittedTable, TableLayout, TableStatement,
    batched::{LocalBinding, ProverFaults, VerifierChecks, Where},
    global_layouts,
};
use crate::proof::options::ProofOptions;
use crate::traits::AIR;

use super::phase_b_overlap_tests::config_for;

type FE = FieldElement<Fp>;
type EE = FieldElement<Ext>;
type Air = AirWithBuses<Fp, Ext, NullBoundaryConstraintBuilder, (), EmptyConstraints>;
type H = KeccakWhir;
type T = <H as WhirHash>::Sponge;

/// `(x, y, x ⊕ y)` over two-bit `x, y`.
const LOCAL_XOR: u64 = 46;
/// `(x, y)` over two-bit `x, y`.
const LOCAL_PAIR: u64 = 47;
/// A global bus each SEND row balances by itself.
const GLOBAL: u64 = 5;
const BUSES: [u64; 2] = [LOCAL_XOR, LOCAL_PAIR];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Cpu,
    Add,
    Mul,
    Send,
    Recv,
}

/// CPU ADD MUL | SEND RECV | SEND SEND RECV.
const KINDS: [Kind; 8] = [
    Kind::Cpu,
    Kind::Add,
    Kind::Mul,
    Kind::Send,
    Kind::Recv,
    Kind::Send,
    Kind::Send,
    Kind::Recv,
];
const SIZES: [usize; 3] = [3, 2, 3];
/// The RECV's fixed columns: `X, Y, X ⊕ Y`.
const FIXED: usize = 3;

fn felts(values: &[u64]) -> Vec<FE> {
    values.iter().map(|v| FE::from(*v)).collect()
}

fn column(values: impl IntoIterator<Item = u64>) -> Vec<FE> {
    values.into_iter().map(FE::from).collect()
}

/// The RECV's interactions: receives `LOCAL_XOR[X, Y, S]·MA` and
/// `LOCAL_PAIR[X, Y]·MB`.
fn recv_interactions() -> Vec<BusInteraction> {
    vec![
        BusInteraction::receiver(
            LOCAL_XOR,
            Multiplicity::Column(3),
            vec![
                BusValue::column(0),
                BusValue::column(1),
                BusValue::column(2),
            ],
        ),
        BusInteraction::receiver(
            LOCAL_PAIR,
            Multiplicity::Column(4),
            vec![BusValue::column(0), BusValue::column(1)],
        ),
    ]
}

/// The SEND's interactions: `LOCAL_XOR[a, b, c]`, `LOCAL_PAIR[b, a]`, and the
/// global pair over `g`; with `constants`, three more `LOCAL_XOR` sends whose
/// multiplicities are those constants (a statement constant a local lookup
/// reads).
fn send_interactions(constants: Option<[i64; 3]>) -> Vec<BusInteraction> {
    let mut interactions = vec![
        BusInteraction::sender(
            LOCAL_XOR,
            Multiplicity::One,
            vec![
                BusValue::column(0),
                BusValue::column(1),
                BusValue::column(2),
            ],
        ),
        BusInteraction::sender(
            LOCAL_PAIR,
            Multiplicity::One,
            vec![BusValue::column(1), BusValue::column(0)],
        ),
        BusInteraction::sender(GLOBAL, Multiplicity::One, vec![BusValue::column(3)]),
        BusInteraction::receiver(GLOBAL, Multiplicity::One, vec![BusValue::column(3)]),
    ];
    if let Some(constants) = constants {
        for (k, tuple) in constants.iter().zip([[0, 1, 2], [1, 0, 2], [0, 0, 2]]) {
            interactions.push(BusInteraction::sender(
                LOCAL_XOR,
                Multiplicity::Linear(vec![LinearTerm::Constant(*k)]),
                tuple.iter().map(|&c| BusValue::column(c)).collect(),
            ));
        }
    }
    interactions
}

fn air(columns: usize, interactions: Vec<BusInteraction>) -> Air {
    AirWithBuses::new(
        columns,
        AuxiliaryTraceBuildData { interactions },
        &ProofOptions::default_test_options(),
        1,
        EmptyConstraints,
    )
}

/// One AIR a kind; group 1's SEND (table 3) has its own, which carries the
/// constant sends when there are constants.
struct Airs {
    cpu: Air,
    add: Air,
    mul: Air,
    send: Air,
    first_send: Air,
    recv: Air,
}

/// The table whose SEND reads the constants.
const FIRST_SEND: usize = 3;

impl Airs {
    fn new(constants: Option<[i64; 3]>) -> Self {
        let options = ProofOptions::default_test_options();
        Self {
            cpu: new_cpu_air_with_lookup(&options),
            add: new_add_air_with_lookup(&options),
            mul: new_mul_air_with_lookup(&options),
            send: air(4, send_interactions(None)),
            first_send: air(4, send_interactions(constants)),
            recv: air(5, recv_interactions()),
        }
    }

    /// Table `t`'s AIR.
    fn of(&self, t: usize) -> &Air {
        match KINDS[t] {
            Kind::Cpu => &self.cpu,
            Kind::Add => &self.add,
            Kind::Mul => &self.mul,
            Kind::Send if t == FIRST_SEND => &self.first_send,
            Kind::Send => &self.send,
            Kind::Recv => &self.recv,
        }
    }
}

/// A SEND's eight rows: `a, b` two-bit, `c = a ⊕ b`, `g` anything.
fn send_columns(seed: u64) -> Vec<Vec<FE>> {
    let a: Vec<u64> = (0..8).map(|i| (i + seed) % 4).collect();
    let b: Vec<u64> = (0..8).map(|i| (3 * i + seed + 1) % 4).collect();
    let c: Vec<u64> = a.iter().zip(&b).map(|(a, b)| a ^ b).collect();
    vec![
        column(a),
        column(b),
        column(c),
        column((0..8).map(|i| 100 * seed + i)),
    ]
}

/// The RECV's fixed columns, sixteen rows: row `r` is `(r mod 4, r div 4)`.
fn fixed_columns() -> Vec<Vec<FE>> {
    vec![
        column((0..16).map(|r| r % 4)),
        column((0..16).map(|r| r / 4)),
        column((0..16).map(|r| (r % 4) ^ (r / 4))),
    ]
}

/// The RECV over the given fixed columns, its multiplicities counting the
/// SENDs' tuples exactly as sent (a false tuple matches no row).
fn recv_columns(fixed: &[Vec<FE>], sends: &[&Vec<Vec<FE>>]) -> Vec<Vec<FE>> {
    let mut ma = vec![FE::zero(); 16];
    let mut mb = vec![FE::zero(); 16];
    for send in sends {
        for i in 0..8 {
            let (a, b, c) = (send[0][i], send[1][i], send[2][i]);
            for r in 0..16 {
                if (fixed[0][r], fixed[1][r], fixed[2][r]) == (a, b, c) {
                    ma[r] += FE::one();
                }
                if (fixed[0][r], fixed[1][r]) == (b, a) {
                    mb[r] += FE::one();
                }
            }
        }
    }
    let mut columns = fixed.to_vec();
    columns.push(ma);
    columns.push(mb);
    columns
}

/// The example's CPU (8 rows), ADD and MUL (4 rows each).
fn example_columns() -> [Vec<Vec<FE>>; 3] {
    [
        vec![
            felts(&[1, 0, 1, 0, 1, 1, 0, 0]),
            felts(&[0, 1, 0, 1, 0, 0, 1, 1]),
            felts(&[1, 2, 3, 4, 5, 6, 7, 8]),
            felts(&[10, 20, 30, 40, 50, 60, 70, 80]),
            felts(&[11, 40, 33, 160, 55, 66, 490, 640]),
        ],
        vec![
            felts(&[1, 3, 5, 6]),
            felts(&[10, 30, 50, 60]),
            felts(&[11, 33, 55, 66]),
            felts(&[1, 1, 1, 1]),
        ],
        vec![
            felts(&[2, 4, 7, 8]),
            felts(&[20, 40, 70, 80]),
            felts(&[40, 160, 490, 640]),
            felts(&[1, 1, 1, 1]),
        ],
    ]
}

/// A block: every table's columns, and whether the verifier holds the RECVs'
/// first three columns as fixed (preprocessed) — the digests follow it.
#[derive(Clone)]
struct Block {
    columns: Vec<Vec<Vec<FE>>>,
    fixed: bool,
}

impl Block {
    /// The honest block.
    fn honest() -> Self {
        Self::with(|_| {})
    }

    /// The honest SENDs, changed by `edit` (by table), then the RECVs'
    /// multiplicities counted over its group's SENDs as they stand.
    fn with(edit: impl Fn(&mut [Vec<Vec<FE>>])) -> Self {
        let example = example_columns();
        let mut columns: Vec<Vec<Vec<FE>>> = KINDS
            .iter()
            .enumerate()
            .map(|(t, kind)| match kind {
                Kind::Cpu => example[0].clone(),
                Kind::Add => example[1].clone(),
                Kind::Mul => example[2].clone(),
                Kind::Send => send_columns(t as u64),
                Kind::Recv => Vec::new(),
            })
            .collect();
        edit(&mut columns);
        recount(&mut columns, &fixed_columns());
        Self {
            columns,
            fixed: true,
        }
    }
}

/// Every RECV's multiplicities, counted over its group's SENDs, its fixed
/// columns `fixed`.
fn recount(columns: &mut [Vec<Vec<FE>>], fixed: &[Vec<FE>]) {
    let mut start = 0;
    for &size in &SIZES {
        let group = start..start + size;
        for t in group.clone() {
            if KINDS[t] == Kind::Recv {
                let sends: Vec<Vec<Vec<FE>>> = group
                    .clone()
                    .filter(|&s| KINDS[s] == Kind::Send)
                    .map(|s| columns[s].clone())
                    .collect();
                let refs: Vec<&Vec<Vec<FE>>> = sends.iter().collect();
                columns[t] = recv_columns(fixed, &refs);
            }
        }
        start += size;
    }
}

fn num_vars(columns: &[Vec<FE>]) -> usize {
    columns[0].len().trailing_zeros() as usize
}

/// The block's config: the batched argue at `bin_log_cells`, packed by
/// `pack`.
fn config(bin_log_cells: u8, pack: GkrPack) -> ChainConfig {
    config_for(
        ArgueFormat::Batched {
            bin_log_cells,
            pack,
        },
        GroupFold::PerPoly,
    )
}

fn production() -> ChainConfig {
    config_for(ArgueFormat::BATCHED, GroupFold::PerPoly)
}

/// Every table's digest, as the verifier holds the tables (`fixed`), the
/// constants in or (a mutant) out.
fn digests(airs: &Airs, block: &Block, constants: bool) -> Vec<[u8; 32]> {
    KINDS
        .iter()
        .zip(&block.columns)
        .enumerate()
        .map(|(t, (&kind, columns))| {
            let fixed = if kind == Kind::Recv && block.fixed {
                FIXED
            } else {
                0
            };
            multilinear_block::table_digest_with::<Ext, T>(
                num_vars(columns),
                columns.len(),
                fixed,
                airs.of(t).bus_interactions(),
                constants,
            )
        })
        .collect()
}

fn local(airs: &Airs, block: &Block) -> BlockLocal {
    BlockLocal {
        buses: BUSES.to_vec(),
        digests: digests(airs, block, true),
    }
}

/// The block proved under `config`, with local lookups `local` (or today's
/// format), group `deviation.group` arguing as `deviation` says.
fn prove(
    airs: &Airs,
    block: &Block,
    config: &ChainConfig,
    local: Option<BlockLocal>,
    deviation: ArgueDeviation,
) -> Result<BlockProved<Fp, Ext>, MlError> {
    let tables: Vec<CommittedTable<'_, Fp, Ext>> = block
        .columns
        .iter()
        .enumerate()
        .map(|(t, columns)| {
            let air = airs.of(t);
            CommittedTable::new(
                air.constraint_program(),
                air.constraints_meta(),
                air.bus_interactions(),
                columns.len(),
                num_vars(columns),
                Uniforms::default(),
                |col| columns[col as usize].clone(),
            )
            .expect("a table")
        })
        .collect();
    let mut committed = BlockCommitted::commit::<H, T>(
        tables,
        &SIZES,
        config,
        TreeDrop::uniform(1),
        Narrowing::Wide,
        false,
    )?;
    if let Some(local) = local {
        committed = committed.with_local_lookups(local);
    }
    let order: Vec<usize> = (0..SIZES.len()).collect();
    let mut transcript = H::sponge();
    multilinear_block::block_prove_in_order::<Fp, Ext, _, H>(
        committed,
        config,
        &mut transcript,
        &[],
        &[],
        &|g| g,
        &deviation,
        &|_| {},
        &order,
        None,
    )
}

/// The honest prover on the host, with this group's faults.
fn faulted(group: usize, faults: ProverFaults) -> ArgueDeviation {
    ArgueDeviation {
        group: Some(group),
        faults,
        at: Where::Host,
    }
}

fn honest() -> ArgueDeviation {
    ArgueDeviation {
        at: Where::Host,
        ..ArgueDeviation::default()
    }
}

/// The groups' stacks, as both sides rebuild them.
fn stacks(block: &Block, config: &ChainConfig) -> Vec<StackedLayout> {
    let shapes: Vec<(usize, usize)> = block
        .columns
        .iter()
        .map(|columns| (columns.len(), num_vars(columns)))
        .collect();
    global_layouts(&shapes, &SIZES, config.format.stack).expect("the stacks")
}

/// The verifier's reading of `proved` against `airs` and its own fixed
/// columns, with `checks` and local lookups over `digests` (or today's
/// format, `None`).
fn verify(
    airs: &Airs,
    block: &Block,
    config: &ChainConfig,
    proved: &BlockProved<Fp, Ext>,
    checks: VerifierChecks,
    digests: Option<&[[u8; 32]]>,
) -> Result<(), MlError> {
    let fixed: Vec<Mle<Fp>> = fixed_columns()
        .into_iter()
        .map(|values| Mle::new(values).expect("a column"))
        .collect();
    let layouts: Vec<TableLayout<'_, Fp, Ext>> = block
        .columns
        .iter()
        .enumerate()
        .map(|(t, columns)| {
            let air = airs.of(t);
            TableLayout::new(
                air.constraint_program(),
                air.constraints_meta(),
                air.bus_interactions(),
                columns.len(),
                num_vars(columns),
                Uniforms::default(),
            )
            .expect("a layout")
        })
        .collect();
    let statements: Vec<TableStatement<'_, Fp, Ext>> = layouts
        .iter()
        .zip(&KINDS)
        .map(|(layout, &kind)| {
            if kind == Kind::Recv && block.fixed {
                layout.statement_with_preprocessed(&fixed)
            } else {
                layout.statement()
            }
        })
        .collect();
    let stacks = stacks(block, config);
    let domains: Vec<Domain<Fp>> = stacks
        .iter()
        .map(|layout| Domain::new(layout.n_stack() + config.log_blowup).expect("a domain"))
        .collect();
    let mut transcript = H::sponge();
    multilinear_block::block_verify_with::<Fp, Ext, _, H>(
        &proved.0,
        &proved.4,
        &proved.1,
        &proved.2,
        &[],
        &statements,
        &stacks,
        &domains,
        &SIZES,
        &EE::zero(),
        config,
        &mut transcript,
        false,
        checks,
        digests.map(|digests| BlockLocalCheck {
            buses: &BUSES,
            digests,
            argues: &proved.5,
        }),
    )
}

/// [`verify`] with every check, under the honest digests.
fn verify_local(
    airs: &Airs,
    block: &Block,
    config: &ChainConfig,
    proved: &BlockProved<Fp, Ext>,
) -> Result<(), MlError> {
    verify(
        airs,
        block,
        config,
        proved,
        VerifierChecks::ALL,
        Some(&digests(airs, block, true)),
    )
}

fn bytes(proved: &BlockProved<Fp, Ext>) -> Vec<u8> {
    bincode::serialize(&(&proved.0, &proved.1, &proved.2, &proved.4, &proved.5)).expect("serialize")
}

/// Group 1's SEND (table 3) with `c` one more on every row of its first
/// row's `(a, b)`: a false tuple on `LOCAL_XOR`.
fn flip_first_pair(columns: &mut [Vec<Vec<FE>>]) {
    let send = &mut columns[FIRST_SEND];
    let (a, b) = (send[0][0], send[1][0]);
    for i in 0..8 {
        if (send[0][i], send[1][i]) == (a, b) {
            send[2][i] += FE::one();
        }
    }
}

/// [`flip_first_pair`] with the RECV counting honestly: the false tuple
/// matches no row.
fn false_tuple() -> Block {
    Block::with(flip_first_pair)
}

/// The false tuple received: group 1's RECV holds `S + 1` in the row of the
/// flipped `(a, b)` — fixed columns of the verifier's or not, `fixed` —
/// and counts over that; group 2's RECV stays true.
fn false_tuple_received(fixed: bool) -> Block {
    let honest = Block::honest();
    let (a, b) = (
        honest.columns[FIRST_SEND][0][0],
        honest.columns[FIRST_SEND][1][0],
    );
    let row = (0..16)
        .find(|&r| fixed_columns()[0][r] == a && fixed_columns()[1][r] == b)
        .unwrap();
    let mut block = false_tuple();
    let mut wrong = fixed_columns();
    wrong[2][row] += FE::one();
    let mut received = block.columns.clone();
    recount(&mut received, &wrong);
    block.columns[4] = received[4].clone();
    block.fixed = fixed;
    block
}

// ── round trips ──────────────────────────────────────────────────────────

/// The block proves and verifies under local lookups, under every packing
/// and a bin cap that puts every tree in one bin or each in its own; the
/// groups with local interactions carry a local argue, the other an empty
/// one.
#[test]
fn a_block_with_local_lookups_proves_and_verifies() {
    let airs = Airs::new(None);
    let block = Block::honest();
    for config in [
        production(),
        config(26, GkrPack::Off),
        config(2, GkrPack::Off),
        config(2, GkrPack::PRODUCTION),
    ] {
        let proved = prove(&airs, &block, &config, Some(local(&airs, &block)), honest())
            .unwrap_or_else(|e| panic!("{:?}: {e:?}", config.format.argue));
        assert_eq!(proved.5.len(), SIZES.len());
        assert!(proved.5[0].is_empty(), "group 0 has no local interaction");
        assert!(!proved.5[1].is_empty() && !proved.5[2].is_empty());
        verify_local(&airs, &block, &config, &proved)
            .unwrap_or_else(|e| panic!("{:?}: {e:?}", config.format.argue));
    }
}

/// The card's prover (on a laptop, its host fallbacks) proves the host
/// reference's bytes under local lookups.
#[test]
fn the_device_prover_proves_the_host_bytes_under_local_lookups() {
    let airs = Airs::new(None);
    let block = Block::honest();
    let config = production();
    let host = prove(&airs, &block, &config, Some(local(&airs, &block)), honest()).unwrap();
    let device = prove(
        &airs,
        &block,
        &config,
        Some(local(&airs, &block)),
        ArgueDeviation::default(),
    )
    .unwrap();
    assert_eq!(bytes(&host), bytes(&device));
}

/// Off is today: a block proved without local lookups carries no local
/// argue; and under local lookups a group with no local interaction argues
/// exactly as today (its fork, its weights, its bytes).
#[test]
fn a_group_with_no_local_interaction_argues_todays_bytes() {
    let airs = Airs::new(None);
    let block = Block::honest();
    let config = production();
    let off = prove(&airs, &block, &config, None, honest()).unwrap();
    assert!(off.5.is_empty(), "today's format carries no local argue");
    let on = prove(&airs, &block, &config, Some(local(&airs, &block)), honest()).unwrap();
    let group = |proved: &BlockProved<Fp, Ext>, g: usize| {
        bincode::serialize(&(&proved.1[g], &proved.0.columns[g])).expect("serialize")
    };
    assert_eq!(group(&off, 0), group(&on, 0), "group 0 is today's");
    assert_ne!(group(&off, 1), group(&on, 1), "group 1 argues locally");
    assert_eq!(off.0.roots, on.0.roots, "phase A is untouched");
    assert_eq!(off.4, on.4, "phase A is untouched");
}

// ── the local transcript ─────────────────────────────────────────────────

fn sample_layout() -> StackedLayout {
    StackedLayout::build(&[4, 4, 4], 5).unwrap()
}

/// The local transcript spelled out by hand: the commit-time prefix, every root,
/// `z0`, every answer, the label, every digest, THEN `z_g` and `α_g`; `z0`
/// is the commit-time point itself.
#[test]
fn the_local_transcript_binds_prefix_roots_point_answers_label_digests_in_order() {
    let config = production();
    let layout = sample_layout();
    let roots: Vec<Commitment> = vec![[7u8; 32], [9u8; 32]];
    let answers = vec![EE::from(3u64), EE::from(5u64)];
    let digests = [[1u8; 32], [2u8; 32], [3u8; 32]];
    let (z, alpha, own) = multilinear_block::local_challenges::<Ext, T>(
        2,
        &layout,
        &roots,
        &answers,
        &digests,
        &config,
        LocalBinding::ALL,
    )
    .unwrap();

    let mut by_hand = T::default();
    by_hand.append_bytes(&multilinear_block::commit_ood_prefix(2, &layout, &config));
    for root in &roots {
        by_hand.append_bytes(root);
    }
    let z0: EE = by_hand.sample_field_element();
    assert_eq!(
        z0,
        multilinear_block::commit_ood_point::<Ext, T>(2, &layout, &roots, &config).unwrap(),
        "the commit-time point is the local transcript's first draw"
    );
    for answer in &answers {
        by_hand.append_field_element(answer);
    }
    by_hand.append_bytes(LOCAL_LOGUP_LABEL);
    for digest in &digests {
        by_hand.append_bytes(digest);
    }
    let z_hand: EE = by_hand.sample_field_element();
    let alpha_hand: EE = by_hand.sample_field_element();
    assert_eq!((z, alpha), (z_hand, alpha_hand));
    assert_eq!(own.state(), by_hand.state());
}

/// `(z_g, α_g)` move with the group's index, every root, every answer and
/// every digest byte; an answer count other than the stack's is refused.
#[test]
fn the_local_challenges_move_with_every_root_answer_and_digest() {
    let config = production();
    let layout = sample_layout();
    let roots: Vec<Commitment> = vec![[7u8; 32], [9u8; 32]];
    let answers = vec![EE::from(3u64), EE::from(5u64)];
    let digests = vec![[1u8; 32], [2u8; 32]];
    let at = |g: usize, roots: &[Commitment], answers: &[EE], digests: &[[u8; 32]]| {
        let (z, alpha, _) = multilinear_block::local_challenges::<Ext, T>(
            g,
            &layout,
            roots,
            answers,
            digests,
            &config,
            LocalBinding::ALL,
        )
        .unwrap();
        (z, alpha)
    };
    let base = at(0, &roots, &answers, &digests);
    assert_ne!(base, at(1, &roots, &answers, &digests), "the group");
    for k in 0..roots.len() {
        let mut moved = roots.clone();
        moved[k][31] ^= 1;
        assert_ne!(base, at(0, &moved, &answers, &digests), "root {k}");
    }
    for k in 0..answers.len() {
        let mut moved = answers.clone();
        moved[k] += EE::one();
        assert_ne!(base, at(0, &roots, &moved, &digests), "answer {k}");
    }
    for k in 0..digests.len() {
        for byte in [0, 31] {
            let mut moved = digests.clone();
            moved[k][byte] ^= 1;
            assert_ne!(base, at(0, &roots, &answers, &moved), "digest {k}");
        }
    }
    assert_ne!(
        base,
        at(0, &roots, &answers, &digests[..1]),
        "a digest fewer"
    );
    assert!(
        multilinear_block::local_challenges::<Ext, T>(
            0,
            &layout,
            &roots,
            &answers[..1],
            &digests,
            &config,
            LocalBinding::ALL,
        )
        .is_err(),
        "an answer per polynomial"
    );
}

/// A table's digest moves with its height, width, fixed count and every part
/// of every interaction — its bus, side, multiplicity, values and each
/// constant; the mutant without constants does not see a constant.
#[test]
fn a_table_digest_moves_with_everything_a_local_lookup_reads() {
    let base = send_interactions(Some([1, 2, 3]));
    let digest = |n: usize, w: usize, p: usize, i: &[BusInteraction]| {
        multilinear_block::table_digest::<Ext, T>(n, w, p, i)
    };
    let d = digest(3, 4, 0, &base);
    assert_ne!(d, digest(4, 4, 0, &base), "height");
    assert_ne!(d, digest(3, 5, 0, &base), "width");
    assert_ne!(d, digest(3, 4, 1, &base), "fixed count");
    assert_ne!(d, digest(3, 4, 0, &base[..base.len() - 1]), "count");
    let mut moved = base.clone();
    moved[0].bus_id = LOCAL_PAIR;
    assert_ne!(d, digest(3, 4, 0, &moved), "bus");
    let mut moved = base.clone();
    moved[0].is_sender = false;
    assert_ne!(d, digest(3, 4, 0, &moved), "side");
    let mut moved = base.clone();
    moved[0].multiplicity = Multiplicity::Column(3);
    assert_ne!(d, digest(3, 4, 0, &moved), "multiplicity");
    let mut moved = base.clone();
    moved[1].values.swap(0, 1);
    assert_ne!(d, digest(3, 4, 0, &moved), "values");
    let constants = send_interactions(Some([1, 2, 4]));
    assert_ne!(d, digest(3, 4, 0, &constants), "a constant");
    let mutant =
        |i: &[BusInteraction]| multilinear_block::table_digest_with::<Ext, T>(3, 4, 0, i, false);
    assert_eq!(mutant(&base), mutant(&constants), "the mutant ignores it");
}

/// The batch weights under local lookups are `λ^{5t}` (today's `λ^{3t}`).
#[test]
fn local_lookups_weight_table_t_by_lambda_to_the_five_t() {
    let lambda = EE::from(7u64);
    let weights = crate::multilinear_table::batched::table_weights(&lambda, 4, 5);
    for (t, w) in weights.iter().enumerate() {
        assert_eq!(*w, lambda.pow(5 * t as u64), "table {t}");
    }
    let today = crate::multilinear_table::batched::table_weights(&lambda, 4, 3);
    for (t, w) in today.iter().enumerate() {
        assert_eq!(*w, lambda.pow(3 * t as u64), "table {t}");
    }
}

// ── forgeries ────────────────────────────────────────────────────────────

/// A false tuple with honest counts is refused by the group's local
/// balance; with that check off (the mutant) it verifies: the local balance
/// is what refuses it.
#[test]
fn a_false_local_lookup_is_refused_by_the_local_balance() {
    let airs = Airs::new(None);
    let block = false_tuple();
    let config = production();
    let proved = prove(&airs, &block, &config, Some(local(&airs, &block)), honest()).unwrap();
    assert_eq!(
        verify_local(&airs, &block, &config, &proved),
        Err(MlError::LocalImbalance { group: 1 })
    );
    let inert = VerifierChecks {
        local_balance: false,
        ..VerifierChecks::ALL
    };
    verify(
        &airs,
        &block,
        &config,
        &proved,
        inert,
        Some(&digests(&airs, &block, true)),
    )
    .expect("the mutant without the local balance accepts the false tuple");
}

/// A receiver of a local bus that reads committed columns (the verifier
/// fixes none of the RECV's) is refused by the static check before anything
/// is read; with the check off (the mutant), a false tuple balanced through
/// it verifies — so the check is soundness-bearing.
#[test]
fn a_dynamic_local_receiver_is_refused_by_the_static_check() {
    let airs = Airs::new(None);
    let block = false_tuple_received(false);
    let config = production();
    let proved = prove(&airs, &block, &config, Some(local(&airs, &block)), honest()).unwrap();
    assert_eq!(
        verify_local(&airs, &block, &config, &proved),
        Err(MlError::LocalReceiverNotFixed {
            table: 4,
            interaction: 0
        })
    );
    let inert = VerifierChecks {
        local_receivers: false,
        ..VerifierChecks::ALL
    };
    verify(
        &airs,
        &block,
        &config,
        &proved,
        inert,
        Some(&digests(&airs, &block, true)),
    )
    .expect("the mutant without the static check accepts the false tuple");
}

/// A RECV whose fixed column is wrong where a false tuple is received:
/// the local balance holds, `check_preprocessed` refuses it, and with that
/// check off (the mutant) it verifies.
#[test]
fn a_wrong_fixed_column_is_refused_by_the_preprocessed_check() {
    let airs = Airs::new(None);
    let block = false_tuple_received(true);
    let config = production();
    let proved = prove(&airs, &block, &config, Some(local(&airs, &block)), honest()).unwrap();
    assert!(verify_local(&airs, &block, &config, &proved).is_err());
    let inert = VerifierChecks {
        preprocessed: false,
        ..VerifierChecks::ALL
    };
    verify(
        &airs,
        &block,
        &config,
        &proved,
        inert,
        Some(&digests(&airs, &block, true)),
    )
    .expect("the mutant without the preprocessed check accepts the wrong column");
}

/// The coordinates of `x` over the base field.
fn coords(x: &EE) -> [FE; 3] {
    *x.value()
}

/// `δ ∈ F³` with `Σ_j δ_j·a_j = target`, by elimination over the base field.
fn solve(a: [EE; 3], target: &EE) -> [FE; 3] {
    let mut m: Vec<[FE; 4]> = (0..3)
        .map(|c| {
            [
                coords(&a[0])[c],
                coords(&a[1])[c],
                coords(&a[2])[c],
                coords(target)[c],
            ]
        })
        .collect();
    for col in 0..3 {
        let pivot = (col..3)
            .find(|&r| m[r][col] != FE::zero())
            .expect("independent");
        m.swap(col, pivot);
        let inv = m[col][col].inv().unwrap();
        for k in 0..4 {
            m[col][k] *= inv;
        }
        for r in 0..3 {
            if r != col {
                let f = m[r][col];
                for k in 0..4 {
                    let v = m[col][k];
                    m[r][k] = m[r][k] - f * v;
                }
            }
        }
    }
    [m[0][3], m[1][3], m[2][3]]
}

/// The fingerprint `bus + Σ_j α^j·v_j`, as the lowering builds it.
fn fingerprint(bus: u64, values: &[FE], alpha: &EE) -> EE {
    let mut f = EE::from(bus);
    let mut power = *alpha;
    for v in values {
        f += &power * v.to_extension::<Ext>();
        power *= alpha;
    }
    f
}

/// Group 1's local sum under `(z, α)` over `columns` (its SEND, table 3, and
/// RECV, table 4; the SEND's constant sends at `constants`).
fn group_one_sum(columns: &[Vec<Vec<FE>>], z: &EE, alpha: &EE, constants: Option<[FE; 3]>) -> EE {
    let (send, recv) = (&columns[3], &columns[4]);
    let frac = |m: FE, bus: u64, values: &[FE]| {
        m.to_extension::<Ext>() * (z - fingerprint(bus, values, alpha)).inv().unwrap()
    };
    let mut sum = EE::zero();
    for i in 0..8 {
        let (a, b, c) = (send[0][i], send[1][i], send[2][i]);
        sum += frac(FE::one(), LOCAL_XOR, &[a, b, c]);
        sum += frac(FE::one(), LOCAL_PAIR, &[b, a]);
        if let Some(k) = constants {
            sum += frac(k[0], LOCAL_XOR, &[a, b, c]);
            sum += frac(k[1], LOCAL_XOR, &[b, a, c]);
            sum += frac(k[2], LOCAL_XOR, &[a, a, c]);
        }
    }
    for r in 0..16 {
        let (x, y, s) = (recv[0][r], recv[1][r], recv[2][r]);
        sum = sum - frac(recv[3][r], LOCAL_XOR, &[x, y, s]);
        sum = sum - frac(recv[4][r], LOCAL_PAIR, &[x, y]);
    }
    sum
}

/// Draw `(z_g, α_g)` before the group's roots and answers (the mutant
/// binding, on both sides) and a prover fixes its multiplicities after the
/// draw: three RECV counts moved so a false tuple balances. The mutant
/// verifier accepts it; the real one, whose challenges come after the roots
/// and answers, refuses it — and refuses the same columns proved honestly.
#[test]
fn local_challenges_drawn_before_the_roots_let_a_prover_fix_its_counts() {
    let airs = Airs::new(None);
    let config = production();
    let mut block = false_tuple();
    let blind = LocalBinding {
        roots: false,
        answers: false,
        digests: true,
    };
    // The mutant's challenges need nothing committed.
    let layout = &stacks(&block, &config)[1];
    let zeros = vec![EE::zero(); layout.num_polys()];
    let digests = digests(&airs, &block, true);
    let (z, alpha, _) = multilinear_block::local_challenges::<Ext, T>(
        1,
        layout,
        &[],
        &zeros,
        &digests[3..5],
        &config,
        blind,
    )
    .unwrap();
    // Balance the false tuple with three RECV counts.
    let off = group_one_sum(&block.columns, &z, &alpha, None);
    assert_ne!(off, EE::zero());
    let rows = [0, 5, 10];
    let recv = &block.columns[4];
    let a = rows.map(|r| {
        -(z - fingerprint(LOCAL_XOR, &[recv[0][r], recv[1][r], recv[2][r]], &alpha))
            .inv()
            .unwrap()
    });
    let delta = solve(a, &-off);
    for (r, d) in rows.iter().zip(delta) {
        block.columns[4][3][*r] += d;
    }
    assert_eq!(
        group_one_sum(&block.columns, &z, &alpha, None),
        EE::zero(),
        "forged"
    );

    let mutant = faulted(
        1,
        ProverFaults {
            local_binding: Some(blind),
            ..ProverFaults::default()
        },
    );
    let forged = prove(&airs, &block, &config, Some(local(&airs, &block)), mutant).unwrap();
    let checks = VerifierChecks {
        local_binding: blind,
        ..VerifierChecks::ALL
    };
    verify(&airs, &block, &config, &forged, checks, Some(&digests))
        .expect("the mutant binding accepts the forgery");
    assert!(verify_local(&airs, &block, &config, &forged).is_err());
    let honest_proof = prove(&airs, &block, &config, Some(local(&airs, &block)), honest()).unwrap();
    assert_eq!(
        verify_local(&airs, &block, &config, &honest_proof),
        Err(MlError::LocalImbalance { group: 1 })
    );
}

/// The `i64` an AIR's constant takes for the field element `x`.
fn as_constant(x: FE) -> i64 {
    const P: u64 = 0xffff_ffff_0000_0001;
    let v = x.canonical_u64();
    let signed = if v < 1 << 63 {
        v as i64
    } else {
        -((P - v) as i64)
    };
    assert_eq!(FE::from(signed), x, "the constant round-trips");
    signed
}

/// A statement constant a local lookup reads, left out of the digests (the
/// mutant, on both sides): the prover sees `(z_g, α_g)`, then picks the three
/// constants that balance a false tuple. The mutant accepts; the real
/// digests bind the constants, so the same proof — and the same block
/// proved under them — is refused.
#[test]
fn a_constant_chosen_after_the_local_challenges_is_refused_by_the_digests() {
    let config = production();
    let block = false_tuple();
    let zero = Airs::new(Some([0, 0, 0]));
    // The digests without constants do not move with them.
    let blind = digests(&zero, &block, false);
    let first = prove(
        &zero,
        &block,
        &config,
        Some(BlockLocal {
            buses: BUSES.to_vec(),
            digests: blind.clone(),
        }),
        honest(),
    )
    .unwrap();
    let stacks = stacks(&block, &config);
    let layout = &stacks[1];
    let trees = |g: usize| stacked_eval::round0_trees(stacks[g].num_polys(), &config).len();
    let roots = &first.0.roots[trees(0)..trees(0) + trees(1)];
    let polys = stacks[0].num_polys();
    let answers = &first.4[polys..polys + layout.num_polys()];
    let (z, alpha, _) = multilinear_block::local_challenges::<Ext, T>(
        1,
        layout,
        roots,
        answers,
        &blind[3..5],
        &config,
        LocalBinding::ALL,
    )
    .unwrap();
    let off = group_one_sum(&block.columns, &z, &alpha, None);
    let send = &block.columns[3];
    let sums = [[0usize, 1, 2], [1, 0, 2], [0, 0, 2]].map(|tuple| {
        (0..8).fold(EE::zero(), |acc, i| {
            let values: Vec<FE> = tuple.iter().map(|&c| send[c][i]).collect();
            acc + (z - fingerprint(LOCAL_XOR, &values, &alpha)).inv().unwrap()
        })
    });
    let k = solve(sums, &-off);
    assert_eq!(
        group_one_sum(&block.columns, &z, &alpha, Some(k)),
        EE::zero(),
        "forged"
    );
    let chosen = Airs::new(Some(k.map(as_constant)));
    assert_eq!(
        digests(&chosen, &block, false),
        blind,
        "the mutant is blind"
    );
    let forged = prove(
        &chosen,
        &block,
        &config,
        Some(BlockLocal {
            buses: BUSES.to_vec(),
            digests: blind.clone(),
        }),
        honest(),
    )
    .unwrap();
    verify(
        &chosen,
        &block,
        &config,
        &forged,
        VerifierChecks::ALL,
        Some(&blind),
    )
    .expect("the mutant digests accept the chosen constants");
    assert!(verify_local(&chosen, &block, &config, &forged).is_err());
    let bound = prove(
        &chosen,
        &block,
        &config,
        Some(local(&chosen, &block)),
        honest(),
    )
    .unwrap();
    assert_eq!(
        verify_local(&chosen, &block, &config, &bound),
        Err(MlError::LocalImbalance { group: 1 })
    );
}

/// A forged local output that brings the group's local sum to zero, its
/// ladder over the true tree: the ladder's first residual refuses it.
#[test]
fn a_forged_local_output_is_refused_by_its_ladder() {
    let airs = Airs::new(None);
    let block = false_tuple();
    let config = production();
    let forged = prove(
        &airs,
        &block,
        &config,
        Some(local(&airs, &block)),
        faulted(
            1,
            ProverFaults {
                local_balance_with: Some(0),
                ..ProverFaults::default()
            },
        ),
    )
    .unwrap();
    let sum = forged.5[1]
        .bus_outputs
        .iter()
        .fold(EE::zero(), |acc, (p, q)| acc + p * q.inv().unwrap());
    assert_eq!(sum, EE::zero(), "the outputs balance");
    assert!(matches!(
        verify_local(&airs, &block, &config, &forged),
        Err(MlError::LayerRelationMismatch { .. })
    ));
}

/// The C phase's local pairs: a local claim absorbed off by one, two tables'
/// local claims swapped, and a table weighted `λ^{5t+1}` are each refused.
#[test]
fn the_local_pairs_in_the_constraint_phase_are_bound() {
    let airs = Airs::new(None);
    let block = Block::honest();
    let config = production();
    for faults in [
        ProverFaults {
            local_tamper_claim: Some(0),
            ..ProverFaults::default()
        },
        ProverFaults {
            local_swap_claims: Some((0, 1)),
            ..ProverFaults::default()
        },
        ProverFaults {
            wrong_weight: Some(1),
            ..ProverFaults::default()
        },
    ] {
        let proved = prove(
            &airs,
            &block,
            &config,
            Some(local(&airs, &block)),
            faulted(2, faults),
        )
        .unwrap();
        assert!(
            verify_local(&airs, &block, &config, &proved).is_err(),
            "{faults:?}"
        );
    }
}

/// Format confusion both ways: a local proof read by today's verifier, and
/// today's proof read by a local verifier, are refused.
#[test]
fn local_and_todays_proofs_are_refused_by_the_other_format() {
    let airs = Airs::new(None);
    let block = Block::honest();
    let config = production();
    let local_proof = prove(&airs, &block, &config, Some(local(&airs, &block)), honest()).unwrap();
    assert!(
        verify(
            &airs,
            &block,
            &config,
            &local_proof,
            VerifierChecks::ALL,
            None
        )
        .is_err()
    );
    let today = prove(&airs, &block, &config, None, honest()).unwrap();
    verify(&airs, &block, &config, &today, VerifierChecks::ALL, None).expect("today verifies");
    assert_eq!(
        verify_local(&airs, &block, &config, &today),
        Err(MlError::ArgueFormatMismatch),
        "no local argue at all"
    );
    let mut padded = today.clone();
    padded.5 = vec![Default::default(); SIZES.len()];
    assert!(verify_local(&airs, &block, &config, &padded).is_err());
}

/// The local argue's shape is the verifier's: a missing or extra output, a
/// missing or extra ladder, a local argue for a group with none, are refused
/// by name.
#[test]
fn the_local_argues_shape_is_the_verifiers() {
    let airs = Airs::new(None);
    let block = Block::honest();
    let config = production();
    let proved = prove(&airs, &block, &config, Some(local(&airs, &block)), honest()).unwrap();
    verify_local(&airs, &block, &config, &proved).unwrap();
    let tampered = |edit: &dyn Fn(&mut BlockProved<Fp, Ext>)| {
        let mut p = proved.clone();
        edit(&mut p);
        verify_local(&airs, &block, &config, &p)
    };
    assert_eq!(
        tampered(&|p| {
            p.5[1].bus_outputs.pop();
        }),
        Err(MlError::ArgueShapeMismatch {
            part: "local trees"
        })
    );
    assert_eq!(
        tampered(&|p| {
            let extra = p.5[1].bus_outputs[0];
            p.5[1].bus_outputs.push(extra);
        }),
        Err(MlError::ArgueShapeMismatch {
            part: "local trees"
        })
    );
    assert_eq!(
        tampered(&|p| {
            p.5[2].gkr.pop();
        }),
        Err(MlError::ArgueShapeMismatch { part: "local bins" })
    );
    assert_eq!(
        tampered(&|p| {
            let extra = p.5[2].gkr[0].clone();
            p.5[2].gkr.push(extra);
        }),
        Err(MlError::ArgueShapeMismatch { part: "local bins" })
    );
    assert_eq!(
        tampered(&|p| p.5[0] = p.5[1].clone()),
        Err(MlError::ArgueShapeMismatch {
            part: "local argue"
        })
    );
    assert_eq!(
        tampered(&|p| {
            p.5.pop();
        }),
        Err(MlError::ArgueFormatMismatch)
    );
}

/// The local argues of two groups traded: each is read on its own group's
/// transcript, so both are refused.
#[test]
fn a_local_argue_moved_to_another_group_is_refused() {
    let airs = Airs::new(None);
    let block = Block::honest();
    let config = production();
    let mut proved = prove(&airs, &block, &config, Some(local(&airs, &block)), honest()).unwrap();
    proved.5.swap(1, 2);
    assert!(verify_local(&airs, &block, &config, &proved).is_err());
}

/// The local state goes into each fork on both sides: a verifier whose forks skip the local state
/// refuses an honest proof.
#[test]
fn each_groups_fork_absorbs_its_local_state_first() {
    let airs = Airs::new(None);
    let block = Block::honest();
    let config = production();
    let proved = prove(&airs, &block, &config, Some(local(&airs, &block)), honest()).unwrap();
    let skipped = VerifierChecks {
        local_state: false,
        ..VerifierChecks::ALL
    };
    assert!(
        verify(
            &airs,
            &block,
            &config,
            &proved,
            skipped,
            Some(&digests(&airs, &block, true)),
        )
        .is_err()
    );
}

/// A group's local receiver counting another group's sends: group 1's RECV
/// counts nothing, group 2's counts group 1's sends too. Both groups'
/// local balances refuse it (group 1 first), though the block as a whole
/// balances.
#[test]
fn a_local_lookup_received_in_another_group_is_refused() {
    let airs = Airs::new(None);
    let mut block = Block::honest();
    let fixed = fixed_columns();
    let sends: Vec<&Vec<Vec<FE>>> = vec![&block.columns[3], &block.columns[5], &block.columns[6]];
    let all = recv_columns(&fixed, &sends);
    let none = recv_columns(&fixed, &[]);
    block.columns[4] = none;
    block.columns[7] = all;
    let config = production();
    let proved = prove(&airs, &block, &config, Some(local(&airs, &block)), honest()).unwrap();
    assert_eq!(
        verify_local(&airs, &block, &config, &proved),
        Err(MlError::LocalImbalance { group: 1 })
    );
    let group_one_off = VerifierChecks {
        local_balance: false,
        ..VerifierChecks::ALL
    };
    verify(
        &airs,
        &block,
        &config,
        &proved,
        group_one_off,
        Some(&digests(&airs, &block, true)),
    )
    .expect("the sum over the block balances: only each group's own balance refuses it");
}
