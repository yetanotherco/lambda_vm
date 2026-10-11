//! D-ONECHAIN B1: one WHIR chain per run of a group's stacked polynomials
//! ([`GroupFold::Indexed`]) on the host, through the stacked opening the
//! block's groups use. The gates are REV-B2-JUDGE §4's B1-1 to B1-7, each named
//! where it is tested; the mutations it lists are run by hand against these.

use crypto::fiat_shamir::{default_transcript::DefaultTranscript, is_transcript::IsTranscript};
use crypto::merkle_tree::cap::CapPolicy;
use math::{
    field::{
        element::FieldElement,
        extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
        goldilocks::GoldilocksField as F,
        traits::{IsField, IsSubFieldOf},
    },
    traits::AsBytes,
};

use crate::{
    Error,
    mle::Mle,
    stacked_eval::{
        Claimed, Refusal, StackedCommitment, StackedProof, index_chains, prove, verify,
        verify_traced,
    },
    stacking::{StackedLayout, borrow},
    sumcheck,
    whir::Domain,
    whir_chain::{
        self, ChainConfig, ChainFormat, FirstFold, GrindBits, GroupFold, GroupedShape,
        MAX_INDEX_POLYS, MAX_INDEX_VARS, RoundOpenings, WhirFolds, check_index_domain, eq_index,
        grouped_faults::{self, Fault},
        index_vars,
    },
    whir_commit::Commitment,
    whir_hash::{KeccakWhir, P1Whir, WhirHash},
    whir_round::{GroupedRoundProof, RoundProof},
};

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

/// A chain config: round 0 folds `first` (`First`), or `log_folding` (uniform).
fn config(
    first: Option<usize>,
    log_folding: usize,
    num_queries: usize,
    group: GroupFold,
) -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding,
        num_queries,
        grind: GrindBits::default(),
        format: ChainFormat {
            folds: first.map_or(WhirFolds::Uniform, |k| {
                WhirFolds::First(FirstFold::new(k).unwrap())
            }),
            group,
            ..ChainFormat::DEFAULT
        },
    }
}

fn indexed(first: Option<usize>, log_folding: usize, num_queries: usize) -> ChainConfig {
    config(first, log_folding, num_queries, GroupFold::Indexed)
}

fn column(num_vars: usize, seed: u64) -> Mle<F> {
    Mle::new(
        (0..(1u64 << num_vars))
            .map(|i| FE::from(i.wrapping_add(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11))
            .collect(),
    )
    .unwrap()
}

/// A stack committed and opened, each column at a point of its own.
struct Opened<E: IsField> {
    layout: StackedLayout,
    roots: Vec<Commitment>,
    domain: Domain<F>,
    points: Vec<Vec<FieldElement<E>>>,
    values: Vec<FieldElement<E>>,
    proof: StackedProof<F, E>,
}

fn open<E, H, T>(
    heights: &[usize],
    n_stack: usize,
    cfg: &ChainConfig,
    seed: u64,
    transcript: fn() -> T,
) -> Result<Opened<E>, Error>
where
    F: IsSubFieldOf<E>,
    E: IsField + Send + Sync + 'static,
    FieldElement<E>: AsBytes + Send + Sync,
    T: IsTranscript<E>,
    H: WhirHash,
{
    let columns: Vec<Mle<F>> = heights
        .iter()
        .enumerate()
        .map(|(i, &h)| column(h, seed + 31 * i as u64))
        .collect();
    let points: Vec<Vec<FieldElement<E>>> = heights
        .iter()
        .enumerate()
        .map(|(i, &h)| {
            (0..h)
                .map(|j| FieldElement::<E>::from(7 + seed + 11 * i as u64 + 3 * j as u64))
                .collect()
        })
        .collect();
    let values = columns
        .iter()
        .zip(&points)
        .map(|(c, p)| c.evaluate_in(p))
        .collect::<Result<Vec<_>, _>>()?;
    let layout = StackedLayout::build(heights, n_stack)?;
    let stacked = StackedCommitment::<F, H>::commit(layout.clone(), &borrow(&columns), None, cfg)?;
    let roots = stacked.roots();
    let proof = prove::<F, E, T, H, _>(
        &stacked,
        &borrow(&columns),
        None,
        &Claimed::PerColumn(&points),
        &values,
        cfg,
        &mut transcript(),
    )?;
    Ok(Opened {
        layout,
        roots,
        domain: stacked.domain().clone(),
        points,
        values,
        proof,
    })
}

fn check<E, H, T>(
    o: &Opened<E>,
    proof: &StackedProof<F, E>,
    cfg: &ChainConfig,
    transcript: fn() -> T,
) -> Result<(), Error>
where
    F: IsSubFieldOf<E>,
    E: IsField + Send + Sync + 'static,
    FieldElement<E>: AsBytes + Send + Sync,
    T: IsTranscript<E>,
    H: WhirHash,
{
    verify::<F, E, T, H>(
        proof,
        &o.layout,
        &o.roots,
        &Claimed::PerColumn(&o.points),
        &o.values,
        &o.domain,
        cfg,
        &mut transcript(),
    )
}

/// The tower production runs: base-field stacks, extension challenges.
fn tower() -> <KeccakWhir as WhirHash>::Sponge {
    let mut t = KeccakWhir::sponge();
    t.append_bytes(b"onechain");
    t
}

fn tower_p1() -> <P1Whir as WhirHash>::Sponge {
    let mut t = P1Whir::sponge();
    t.append_bytes(b"onechain");
    t
}

/// The base field alone, where a value-level forgery can keep a combination.
fn base() -> DefaultTranscript<F> {
    DefaultTranscript::<F>::new(b"onechain-base")
}

fn open_tower(heights: &[usize], n_stack: usize, cfg: &ChainConfig) -> Opened<Ext> {
    open::<Ext, KeccakWhir, _>(heights, n_stack, cfg, 3, tower).unwrap()
}

fn check_tower(
    o: &Opened<Ext>,
    proof: &StackedProof<F, Ext>,
    cfg: &ChainConfig,
) -> Result<(), Error> {
    check::<Ext, KeccakWhir, _>(o, proof, cfg, tower)
}

/// [`check`] with the refusal's cause: the chain, and the check inside it that
/// refused ([`verify_traced`]) — what a negative asserts, so it cannot pass on
/// another error.
fn trace<E, H, T>(
    o: &Opened<E>,
    proof: &StackedProof<F, E>,
    cfg: &ChainConfig,
    transcript: fn() -> T,
) -> Result<(), Refusal>
where
    F: IsSubFieldOf<E>,
    E: IsField + Send + Sync + 'static,
    FieldElement<E>: AsBytes + Send + Sync,
    T: IsTranscript<E>,
    H: WhirHash,
{
    verify_traced::<F, E, T, H>(
        proof,
        &o.layout,
        &o.roots,
        &Claimed::PerColumn(&o.points),
        &o.values,
        &o.domain,
        cfg,
        &mut transcript(),
    )
}

fn trace_tower(
    o: &Opened<Ext>,
    proof: &StackedProof<F, Ext>,
    cfg: &ChainConfig,
) -> Result<(), Refusal> {
    trace::<Ext, KeccakWhir, _>(o, proof, cfg, tower)
}

/// Chain 0 (over polynomials 0..), refused by `cause`.
fn in_chain_0(cause: Error) -> Result<(), Refusal> {
    Err(Refusal::Chain {
        chain: 0,
        poly: 0,
        cause,
    })
}

fn grouped(proof: &StackedProof<F, Ext>, chain: usize) -> &GroupedRoundProof<F, Ext> {
    match &proof.polys[chain].rounds[0].openings {
        RoundOpenings::BaseGrouped(g) => g,
        _ => panic!("chain {chain} is not grouped"),
    }
}

fn grouped_mut(proof: &mut StackedProof<F, Ext>, chain: usize) -> &mut GroupedRoundProof<F, Ext> {
    match &mut proof.polys[chain].rounds[0].openings {
        RoundOpenings::BaseGrouped(g) => g,
        _ => panic!("chain {chain} is not grouped"),
    }
}

/// The named shape a refusal carries.
fn shape(result: Result<(), Error>) -> GroupedShape {
    match result {
        Err(Error::GroupedShape(s)) => s,
        other => panic!("expected a named shape refusal, got {other:?}"),
    }
}

/// Three 2^8 polynomials, one chain over all three (`m = 2`, slot 3 absent),
/// round 0 folding 6 data variables: 8 messages, blocks of 64.
const N3: usize = 8;

fn p3() -> (Opened<Ext>, ChainConfig) {
    let cfg = indexed(Some(6), 4, 3);
    let o = open_tower(&[N3; 3], N3, &cfg);
    assert_eq!(o.layout.num_polys(), 3);
    assert_eq!(o.proof.polys.len(), 1);
    (o, cfg)
}

// ───────────────────────────── B1-1: the split ─────────────────────────────

/// REV-B2-JUDGE B1-1: consecutive runs of at most four, the remainder last; a
/// 38-polynomial singleton (CPU stated at 2^27) is nine runs of 4 and one of
/// 2, and nothing past `MAX_INDEX_VARS` index variables is ever cut.
#[test]
fn index_chains_cut_consecutive_runs_of_at_most_four() {
    let widths = |n: usize| -> Vec<usize> {
        index_chains(n, GroupFold::Indexed)
            .iter()
            .map(|r| r.len())
            .collect()
    };
    assert_eq!(widths(38), [vec![4; 9], vec![2]].concat());
    assert_eq!(index_chains(5, GroupFold::Indexed), vec![0..4, 4..5]);
    assert_eq!(index_chains(9, GroupFold::Indexed), vec![0..4, 4..8, 8..9]);
    assert_eq!(index_chains(3, GroupFold::Indexed), vec![0..3]);
    assert_eq!(MAX_INDEX_POLYS, 4);
    for n in 0..=64 {
        let chains = index_chains(n, GroupFold::Indexed);
        let mut next = 0;
        for (c, run) in chains.iter().enumerate() {
            assert_eq!(run.start, next, "n={n}: consecutive");
            assert!((1..=MAX_INDEX_POLYS).contains(&run.len()), "n={n}");
            assert!(index_vars(run.len()) <= MAX_INDEX_VARS, "n={n}");
            if c + 1 < chains.len() {
                assert_eq!(run.len(), MAX_INDEX_POLYS, "n={n}: only the last is short");
            }
            next = run.end;
        }
        assert_eq!(next, n, "n={n}: every polynomial in one chain");
        assert_eq!(
            index_chains(n, GroupFold::PerPoly),
            (0..n).map(|i| i..i + 1).collect::<Vec<_>>(),
            "n={n}: per polynomial is today's"
        );
    }
}

/// REV-B2-JUDGE B1-1 (b): a singleton of 5 and of 9 stacked polynomials splits
/// into [4, 1] and [4, 4, 1] and verifies; the runs of 4 are grouped chains and
/// the run of 1 is today's.
#[test]
fn a_singleton_of_five_and_of_nine_split_and_verify() {
    let cfg = indexed(None, 2, 3);
    for (polys, want) in [(5usize, vec![4, 1]), (9, vec![4, 4, 1])] {
        let o = open_tower(&vec![6; polys], 6, &cfg);
        assert_eq!(o.layout.num_polys(), polys);
        assert_eq!(o.proof.polys.len(), want.len());
        for (chain, &width) in want.iter().enumerate() {
            let round0 = &o.proof.polys[chain].rounds[0];
            assert_eq!(round0.sumcheck.len(), 2 + index_vars(width));
            match (&round0.openings, width) {
                (RoundOpenings::BaseGrouped(g), 4) => {
                    assert!(g.current.iter().all(|q| q.len() == 4));
                }
                (RoundOpenings::Base(_), 1) => {}
                _ => panic!("P={polys}: chain {chain} of width {width}"),
            }
        }
        check_tower(&o, &o.proof, &cfg).unwrap_or_else(|e| panic!("P={polys}: {e:?}"));
    }
}

/// REV-B2-JUDGE B1-1 (c) and B1-2's chain count: a proof whose chains are not
/// the derived split is refused by name before anything is absorbed — one
/// chain over five, one chain fewer or one more — and a chain asked to open
/// five polynomials is refused before anything is read.
#[test]
fn a_chain_count_other_than_the_split_is_refused() {
    let cfg = indexed(None, 2, 3);
    let o5 = open_tower(&[6; 5], 6, &cfg);
    let mut one = o5.proof.clone();
    one.polys.pop();
    assert_eq!(
        shape(check_tower(&o5, &one, &cfg)),
        GroupedShape::ChainCount {
            expected: 2,
            got: 1
        }
    );
    let mut three = o5.proof.clone();
    three.polys.push(o5.proof.polys[1].clone());
    assert_eq!(
        shape(check_tower(&o5, &three, &cfg)),
        GroupedShape::ChainCount {
            expected: 2,
            got: 3
        }
    );
    let o9 = open_tower(&[6; 9], 6, &cfg);
    let mut short = o9.proof.clone();
    short.polys.remove(1);
    assert_eq!(
        shape(check_tower(&o9, &short, &cfg)),
        GroupedShape::ChainCount {
            expected: 3,
            got: 2
        }
    );
    // The chain verifier itself takes no run wider than four, nor an empty one.
    for roots in [&o5.roots[..], &o5.roots[..0]] {
        let refused = whir_chain::verify_grouped::<F, Ext, _, _, KeccakWhir>(
            &o5.proof.polys[0],
            roots,
            |_: &[EE]| Ok(EE::one()),
            EE::zero(),
            6,
            &o5.domain,
            &cfg,
            &mut tower(),
        );
        assert_eq!(
            shape(refused),
            GroupedShape::RunWidth { polys: roots.len() }
        );
    }
}

/// REV-B2-JUDGE B1-1: the runtime refusal when the virtual chain's domain does
/// not exist, `n + m + log_blowup` past the two-adicity (REV-B2-A C5).
#[test]
fn an_index_past_the_two_adicity_is_refused() {
    assert!(check_index_domain(27, 2, 2, 32).is_ok());
    assert!(check_index_domain(28, 2, 2, 32).is_ok());
    for (n, m, lb) in [(29, 2, 2), (27, 2, 4), (27, 6, 2), (32, 1, 0)] {
        assert_eq!(
            shape(check_index_domain(n, m, lb, 32)),
            GroupedShape::IndexDomain {
                num_vars: n,
                index_vars: m,
                log_blowup: lb,
                two_adicity: 32
            }
        );
    }
}

/// REV-B2-JUDGE B1-1: that refusal is the stacked verifier's, on its own
/// layout and config, before anything is read: a three-polynomial opening
/// checked at a blowup where `n + m + log_blowup` passes the two-adicity is
/// refused by name. A per-polynomial verifier binds no index there.
#[test]
fn the_verifier_refuses_an_index_past_the_two_adicity_before_reading() {
    let (o, cfg) = p3();
    let mut wide = cfg;
    wide.log_blowup = 32 - N3 - 2 + 1;
    assert_eq!(
        shape(check_tower(&o, &o.proof, &wide)),
        GroupedShape::IndexDomain {
            num_vars: N3,
            index_vars: 2,
            log_blowup: wide.log_blowup,
            two_adicity: 32
        }
    );
    // A per-polynomial verifier binds no index there: what refuses the one
    // grouped chain is its count against three polynomials.
    wide.format.group = GroupFold::PerPoly;
    assert_eq!(
        check_tower(&o, &o.proof, &wide),
        Err(Error::QueryCountMismatch {
            expected: 3,
            got: 1
        })
    );
}

// ─────────────────────── B1-2: exact shapes, by name ───────────────────────

/// REV-B2-JUDGE B1-2: one root per polynomial under `Indexed`, as today — a
/// root slice one short or one long is refused before anything is absorbed,
/// whatever the chain count.
#[test]
fn a_root_slice_other_than_one_per_polynomial_is_refused() {
    let (o, cfg) = p3();
    let mut long = o.roots.clone();
    long.push(o.roots[0]);
    for roots in [&o.roots[..2], &long[..]] {
        let refused = verify::<F, Ext, _, KeccakWhir>(
            &o.proof,
            &o.layout,
            roots,
            &Claimed::PerColumn(&o.points),
            &o.values,
            &o.domain,
            &cfg,
            &mut tower(),
        );
        assert_eq!(
            refused,
            Err(Error::QueryCountMismatch {
                expected: 3,
                got: roots.len()
            })
        );
    }
}

/// REV-B2-JUDGE B1-2: per query exactly `P'` blocks — one fewer, or one in the
/// absent slot, is refused by name.
#[test]
fn a_query_with_other_than_one_block_per_tree_is_refused() {
    let (o, cfg) = p3();
    let mut fewer = o.proof.clone();
    grouped_mut(&mut fewer, 0).current[1].pop();
    assert_eq!(
        shape(check_tower(&o, &fewer, &cfg)),
        GroupedShape::BlocksPerQuery {
            query: 1,
            expected: 3,
            got: 2
        }
    );
    let mut absent = o.proof.clone();
    let extra = grouped(&o.proof, 0).current[1][0].clone();
    grouped_mut(&mut absent, 0).current[1].push(extra);
    assert_eq!(
        shape(check_tower(&o, &absent, &cfg)),
        GroupedShape::BlocksPerQuery {
            query: 1,
            expected: 3,
            got: 4
        }
    );
}

/// REV-B2-JUDGE B1-2: every block exactly `2^k0` values — not the virtual
/// `2^(k0+m)`, not `2^(k0−1)`, not one more.
#[test]
fn a_block_of_another_width_is_refused() {
    let (o, cfg) = p3();
    for width in [1usize << 8, 1 << 5, (1 << 6) + 1] {
        let mut proof = o.proof.clone();
        grouped_mut(&mut proof, 0).current[2][1]
            .values
            .resize(width, FE::one());
        assert_eq!(
            shape(check_tower(&o, &proof, &cfg)),
            GroupedShape::BlockWidth {
                query: 2,
                tree: 1,
                expected: 64,
                got: width
            },
            "width {width}"
        );
    }
}

/// REV-B2-JUDGE B1-2: exactly one successor block a query, and `Q` queries.
#[test]
fn a_missing_successor_block_or_query_is_refused() {
    let (o, cfg) = p3();
    let mut no_successor = o.proof.clone();
    grouped_mut(&mut no_successor, 0).next.pop();
    assert_eq!(
        shape(check_tower(&o, &no_successor, &cfg)),
        GroupedShape::SuccessorBlocks {
            expected: 3,
            got: 2
        }
    );
    let mut fewer = o.proof.clone();
    let g = grouped_mut(&mut fewer, 0);
    g.current.pop();
    g.next.pop();
    assert_eq!(
        shape(check_tower(&o, &fewer, &cfg)),
        GroupedShape::Queries {
            expected: 3,
            got: 2
        }
    );
}

/// REV-B2-JUDGE B1-2: round 0 of a grouped chain carries exactly `k0 + m`
/// messages — not `k0` (6 instead of 8), not `k0 + m + 1`.
#[test]
fn round_zero_with_other_than_k0_plus_m_messages_is_refused() {
    let (o, cfg) = p3();
    assert_eq!(o.proof.polys[0].rounds[0].sumcheck.len(), 8);
    let mut six = o.proof.clone();
    six.polys[0].rounds[0].sumcheck.truncate(6);
    assert_eq!(
        shape(check_tower(&o, &six, &cfg)),
        GroupedShape::Round0Messages {
            expected: 8,
            got: 6
        }
    );
    let mut nine = o.proof.clone();
    let extra = nine.polys[0].rounds[0].sumcheck[0].clone();
    nine.polys[0].rounds[0].sumcheck.push(extra);
    assert_eq!(
        shape(check_tower(&o, &nine, &cfg)),
        GroupedShape::Round0Messages {
            expected: 8,
            got: 9
        }
    );
}

/// The per-polynomial variant built from a grouped round: tree 0's blocks.
fn as_base(g: &GroupedRoundProof<F, Ext>) -> RoundOpenings<F, Ext> {
    RoundOpenings::Base(RoundProof {
        current: g.current.iter().map(|q| q[0].clone()).collect(),
        next: g.next.clone(),
    })
}

/// A per-polynomial round 0 dressed as a grouped one of one tree.
fn as_grouped(openings: &RoundOpenings<F, Ext>) -> RoundOpenings<F, Ext> {
    match openings {
        RoundOpenings::Base(p) => RoundOpenings::BaseGrouped(GroupedRoundProof {
            current: p.current.iter().map(|o| vec![o.clone()]).collect(),
            next: p.next.clone(),
        }),
        _ => panic!("not a base round"),
    }
}

/// REV-B2-JUDGE B1-2: `BaseGrouped` exactly at round 0 of a grouped chain
/// (final or not). `Base` is refused there; `BaseGrouped` is refused at a
/// later round, under `PerPoly`, and on a chain over one polynomial.
#[test]
fn the_grouped_variant_is_refused_wherever_it_does_not_belong() {
    let (o, cfg) = p3();
    // `Base` at a grouped round 0.
    let mut base0 = o.proof.clone();
    base0.polys[0].rounds[0].openings = as_base(grouped(&o.proof, 0));
    assert_eq!(
        shape(check_tower(&o, &base0, &cfg)),
        GroupedShape::Variant { round: 0 }
    );
    // `BaseGrouped` at round 1.
    let mut late = o.proof.clone();
    late.polys[0].rounds[1].openings = RoundOpenings::BaseGrouped(grouped(&o.proof, 0).clone());
    assert_eq!(
        shape(check_tower(&o, &late, &cfg)),
        GroupedShape::Variant { round: 1 }
    );
    // `BaseGrouped` under `PerPoly`, at today's round 0.
    let per_poly = config(Some(6), 4, 3, GroupFold::PerPoly);
    let today = open_tower(&[N3; 3], N3, &per_poly);
    let mut dressed = today.proof.clone();
    dressed.polys[0].rounds[0].openings = as_grouped(&today.proof.polys[0].rounds[0].openings);
    assert_eq!(
        shape(check_tower(&today, &dressed, &per_poly)),
        GroupedShape::Variant { round: 0 }
    );
    // `BaseGrouped` on the singleton chain of an indexed split (P' = 1).
    let cfg5 = indexed(None, 2, 3);
    let o5 = open_tower(&[6; 5], 6, &cfg5);
    let mut single = o5.proof.clone();
    single.polys[1].rounds[0].openings = as_grouped(&o5.proof.polys[1].rounds[0].openings);
    assert_eq!(
        shape(check_tower(&o5, &single, &cfg5)),
        GroupedShape::Variant { round: 0 }
    );
    // `Base` at a grouped round 0 that is also the last round (R = 1).
    let one = indexed(Some(6), 4, 3);
    let r1 = open_tower(&[3; 3], 3, &one);
    assert_eq!(r1.proof.polys[0].rounds.len(), 1);
    let mut base_final = r1.proof.clone();
    base_final.polys[0].rounds[0].openings = as_base(grouped(&r1.proof, 0));
    assert_eq!(
        shape(check_tower(&r1, &base_final, &one)),
        GroupedShape::Variant { round: 0 }
    );
}

// ─────────────────── B1-3: one convention for the index ────────────────────

/// Columns of mixed heights at their own points, so a column's selector — its
/// prefix — runs past `k0` and straddles the index the chain inserts there.
const MIXED: [usize; 9] = [8, 3, 5, 8, 2, 8, 6, 4, 8];

/// REV-B2-JUDGE B1-3: mixed heights round-trip, with some column's selector
/// longer than `k0`, in a chain over three or four polynomials.
#[test]
fn mixed_heights_straddle_the_index_and_verify() {
    let cfg = indexed(None, 2, 3);
    let o = open_tower(&MIXED, 8, &cfg);
    let k0 = cfg.schedule(8)[0];
    let runs = index_chains(o.layout.num_polys(), GroupFold::Indexed);
    assert!(
        runs.iter().any(|r| r.len() >= 3),
        "a chain over three or more: {runs:?}"
    );
    assert!(
        o.layout
            .placements()
            .iter()
            .any(|p| p.prefix_bits().len() > k0),
        "some selector straddles the index"
    );
    check_tower(&o, &o.proof, &cfg).unwrap();
}

/// REV-B2-JUDGE B1-3: each reading of the index is load-bearing. With `ρ`
/// swapped in the combine only, `ρ` swapped in the weight only, the index
/// factor dropped from the weight, or the absent slot given weight 1, the
/// honest proof is refused; with none of them it verifies.
#[test]
fn each_reading_of_the_index_is_load_bearing() {
    let cfg = indexed(None, 2, 3);
    let o = open_tower(&[3, 5, 8, 2, 4], 8, &cfg);
    assert_eq!(
        o.layout.num_polys(),
        3,
        "one chain over three, slot 3 absent"
    );
    grouped_faults::with(Fault::None, || check_tower(&o, &o.proof, &cfg)).unwrap();
    // The combine's reading refuses at round 0's first fold; the weight's at
    // the final check.
    for (fault, cause) in [
        (
            Fault::SwapRhoInCombine,
            Error::FoldInconsistent { query: 0 },
        ),
        (Fault::SwapRhoInWeight, Error::EvaluationMismatch),
        (Fault::DropIndexFactor, Error::EvaluationMismatch),
        (Fault::AbsentSlotWeightOne, Error::EvaluationMismatch),
    ] {
        let refused = grouped_faults::with(fault, || trace_tower(&o, &o.proof, &cfg));
        assert_eq!(refused, in_chain_0(cause), "{fault:?}");
    }
    check_tower(&o, &o.proof, &cfg).unwrap();
}

/// `eq_index` is `eq(ρ, bits(p))` with `ρ_0` against the high bit, and sums to
/// one over the slots.
#[test]
fn eq_index_reads_rho_against_the_high_bit() {
    let rho = [EE::from(5), EE::from(9)];
    let one = EE::one();
    assert_eq!(eq_index(&rho, 0), (&one - &rho[0]) * (&one - &rho[1]));
    assert_eq!(eq_index(&rho, 1), (&one - &rho[0]) * &rho[1]);
    assert_eq!(eq_index(&rho, 2), &rho[0] * (&one - &rho[1]));
    assert_eq!(eq_index(&rho, 3), &rho[0] * &rho[1]);
    let total = (0..4).fold(EE::zero(), |acc, p| acc + eq_index(&rho, p));
    assert_eq!(total, one);
    assert_eq!(eq_index::<Ext>(&[], 0), one);
}

// ──────────────────────── B1-4: the grouped schedule ───────────────────────

/// REV-B2-JUDGE B1-4: `(k0, m, tail = schedule(n)[1..])` at the production
/// config — the round count, the folds and so the trees and their caps are
/// the per-polynomial chain's, `n + m` messages in all, Q 114 — and never
/// `schedule(n + m)`.
#[test]
fn the_grouped_schedule_widens_round_zero_only() {
    let production = |group: GroupFold| {
        let mut c = ChainConfig::with_security_folds(
            2,
            4,
            WhirFolds::First(FirstFold::new(6).unwrap()),
            27,
            128,
            GrindBits::query_only(18),
        );
        c.format.group = group;
        c
    };
    let (per_poly, grouped) = (
        production(GroupFold::PerPoly),
        production(GroupFold::Indexed),
    );
    assert_eq!((per_poly.num_queries, grouped.num_queries), (114, 114));
    for n in 1..=27usize {
        let folds = grouped.schedule(n);
        assert_eq!(folds, per_poly.schedule(n), "n={n}: the folds");
        for m in 0..=MAX_INDEX_VARS {
            let messages = grouped.index_schedule(n, m);
            assert_eq!(messages.len(), per_poly.rounds(n), "n={n} m={m}: R");
            assert_eq!(messages.iter().sum::<usize>(), n + m, "n={n} m={m}");
            assert_eq!(messages[0], folds[0] + m, "n={n} m={m}");
            assert_eq!(messages[1..], folds[1..], "n={n} m={m}: the tail");
        }
        for arity in [2, 4] {
            assert_eq!(
                grouped.tree_caps_at(n, arity),
                per_poly.tree_caps_at(n, arity),
                "n={n}: tree caps"
            );
        }
    }
    assert_eq!(grouped.index_schedule(27, 2), vec![8, 4, 4, 4, 4, 4, 1]);
    assert_ne!(grouped.index_schedule(27, 2), grouped.schedule(29));
    assert_eq!(grouped.schedule(29), vec![6, 4, 4, 4, 4, 4, 3]);
}

/// REV-B2-JUDGE B1-4 (J2): a grouped chain whose round 0 is its last round
/// (`n ≤ k0`) round-trips for `P' = 2, 3, 4`; its round 0 opens no successor.
#[test]
fn grouped_chains_of_one_round_round_trip() {
    let cfg = indexed(Some(6), 4, 3);
    for polys in 2..=4usize {
        let o = open_tower(&vec![3; polys], 3, &cfg);
        assert_eq!(o.proof.polys.len(), 1);
        let chain = &o.proof.polys[0];
        assert_eq!(chain.rounds.len(), 1, "P'={polys}: R = 1");
        assert_eq!(chain.rounds[0].sumcheck.len(), 3 + index_vars(polys));
        let g = grouped(&o.proof, 0);
        assert!(g.next.is_empty());
        assert!(g.current.iter().all(|q| q.len() == polys));
        check_tower(&o, &o.proof, &cfg).unwrap_or_else(|e| panic!("P'={polys}: {e:?}"));
    }
}

// ───────────────────────── B1-5: the format word ───────────────────────────

/// REV-B2-JUDGE B1-5: a grouped proof is refused by a per-polynomial verifier
/// and the reverse.
#[test]
fn an_indexed_proof_and_a_per_poly_proof_refuse_each_others_verifier() {
    let grouped_cfg = indexed(Some(6), 4, 3);
    let per_poly = config(Some(6), 4, 3, GroupFold::PerPoly);
    let g = open_tower(&[N3; 3], N3, &grouped_cfg);
    let p = open_tower(&[N3; 3], N3, &per_poly);
    assert_eq!(g.roots, p.roots, "the commitments are the same");
    check_tower(&g, &g.proof, &grouped_cfg).unwrap();
    check_tower(&p, &p.proof, &per_poly).unwrap();
    assert_eq!(
        check_tower(&g, &g.proof, &per_poly),
        Err(Error::QueryCountMismatch {
            expected: 3,
            got: 1
        }),
        "one chain where a per-polynomial verifier counts three"
    );
    assert_eq!(
        shape(check_tower(&p, &p.proof, &grouped_cfg)),
        GroupedShape::ChainCount {
            expected: 1,
            got: 3
        }
    );
}

// ──────────────────────── B1-6: default bytes ─────────────────────────────

/// REV-B2-JUDGE B1-6: the appended variant does not grow the archived enum,
/// so a proof without it keeps its rkyv layout.
#[test]
fn the_grouped_variant_does_not_grow_the_archived_enum() {
    #[derive(rkyv::Archive)]
    #[allow(dead_code)]
    enum Before {
        Base(RoundProof<F, Ext>),
        Extension(RoundProof<Ext, Ext>),
    }
    assert_eq!(
        size_of::<rkyv::Archived<RoundOpenings<F, Ext>>>(),
        size_of::<rkyv::Archived<Before>>()
    );
    assert_eq!(
        align_of::<rkyv::Archived<RoundOpenings<F, Ext>>>(),
        align_of::<rkyv::Archived<Before>>()
    );
}

/// REV-B2-JUDGE B1-7: one polynomial under `Indexed` is today's chain, byte for
/// byte, and so is a P = 5 split's singleton chain against a lone polynomial.
#[test]
fn one_polynomial_under_indexed_is_todays_chain() {
    let per_poly = config(Some(6), 4, 3, GroupFold::PerPoly);
    let grouped_cfg = indexed(Some(6), 4, 3);
    let today = open_tower(&[7], 7, &per_poly);
    let one = open_tower(&[7], 7, &grouped_cfg);
    assert_eq!(format!("{:?}", one.proof), format!("{:?}", today.proof));
    check_tower(&one, &one.proof, &grouped_cfg).unwrap();
}

// ──────────────────── B1-7: the note's vectors and negatives ──────────────

/// REV-B2-JUDGE B1-7: round trips at stacks of 6 to 12 variables and 1 to 5
/// polynomials, over pseudo-random layouts (tall columns, and short ones packed
/// among them).
#[test]
fn random_layouts_round_trip() {
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = |bound: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed % bound
    };
    for n in 6..=12usize {
        let cfg = if n >= 7 {
            indexed(Some(6), 4, 3)
        } else {
            indexed(None, 2, 3)
        };
        for polys in 1..=5usize {
            let mut shorts = Vec::new();
            let mut cells = 0usize;
            for _ in 0..1 + next(4) {
                let h = 1 + next(n as u64 - 1) as usize;
                if cells + (1 << h) <= 1 << n {
                    cells += 1 << h;
                    shorts.push(h);
                }
            }
            // Tallest first: powers of two in falling order pack with no
            // alignment gap, so the shorts fill one polynomial.
            shorts.sort_unstable_by(|a, b| b.cmp(a));
            let at = next(polys as u64) as usize;
            let mut heights = vec![n; polys - 1];
            heights.splice(at..at, shorts);
            let o =
                open::<Ext, KeccakWhir, _>(&heights, n, &cfg, n as u64 + 17 * polys as u64, tower)
                    .unwrap();
            assert_eq!(o.layout.num_polys(), polys, "n={n} heights={heights:?}");
            check_tower(&o, &o.proof, &cfg)
                .unwrap_or_else(|e| panic!("n={n} P={polys} {heights:?}: {e:?}"));
        }
    }
}

/// REV-B2-JUDGE B1-7: a tampered value in any slot `p` is refused by tree
/// `p`'s own opening at that query (and reported as the chain's).
#[test]
fn a_tampered_value_in_any_slot_is_refused() {
    let (o, cfg) = p3();
    for p in 0..3 {
        let mut proof = o.proof.clone();
        grouped_mut(&mut proof, 0).current[1][p].values[5] += FE::one();
        assert_eq!(
            trace_tower(&o, &proof, &cfg),
            in_chain_0(Error::GroupedOpeningRejected { query: 1, tree: p }),
            "slot {p}"
        );
        assert_eq!(
            check_tower(&o, &proof, &cfg),
            Err(Error::ColumnOpeningRejected { column: 0 })
        );
    }
}

/// REV-B2-JUDGE B1-7: blocks swapped between trees, and one tree's block from
/// another position, are refused: each block opens against its own tree at the
/// query's one position.
#[test]
fn blocks_from_another_tree_or_position_are_refused() {
    let (o, cfg) = p3();
    let mut swapped = o.proof.clone();
    grouped_mut(&mut swapped, 0).current[1].swap(0, 1);
    assert_eq!(
        trace_tower(&o, &swapped, &cfg),
        in_chain_0(Error::GroupedOpeningRejected { query: 1, tree: 0 }),
        "tree 0 reads tree 1's block"
    );

    let g = grouped(&o.proof, 0);
    let other = (0..g.current.len())
        .find(|&j| g.current[j][2].values != g.current[1][2].values)
        .expect("two queries at different positions");
    let mut moved = o.proof.clone();
    grouped_mut(&mut moved, 0).current[1][2] = g.current[other][2].clone();
    assert_eq!(
        trace_tower(&o, &moved, &cfg),
        in_chain_0(Error::GroupedOpeningRejected { query: 1, tree: 2 }),
        "tree 2 at query {other}'s position"
    );
}

/// Each tree's path is load-bearing on its own, not just through the fold: in
/// the base field a prover CAN change blocks 1 and 2 so that `Σ_p eq(ρ, p)·
/// block_p` — and so the fold — is unchanged; only tree 1's and tree 2's own
/// openings refuse it. (Over the tower the base blocks cannot absorb an
/// extension ratio, which is why the in-guest lift must stay pinned.)
#[test]
fn a_combination_preserving_forgery_is_refused_by_the_paths() {
    let cfg = indexed(Some(6), 4, 3);
    let o = open::<F, KeccakWhir, _>(&[N3; 3], N3, &cfg, 5, base).unwrap();
    // `ρ`, replayed: the values, γ, then round 0's messages (no grind bits).
    let mut t = base();
    for v in &o.values {
        t.append_field_element(v);
    }
    let _gamma: FE = t.sample_field_element();
    let claim =
        sumcheck::verify_rounds(&o.proof.polys[0].rounds[0].sumcheck, FE::zero(), 2, &mut t)
            .unwrap();
    let k0 = cfg.schedule(N3)[0];
    let rho = &claim.point[k0..];
    let (e1, e2) = (eq_index(rho, 1), eq_index(rho, 2));
    let ratio = &e1 * e2.inv().unwrap();

    let mut forged = o.proof.clone();
    let RoundOpenings::BaseGrouped(g) = &mut forged.polys[0].rounds[0].openings else {
        panic!("grouped");
    };
    let honest = crate::whir_round::combine_blocks(&g.current[1], &[eq_index(rho, 0), e1, e2]);
    for i in 0..g.current[1][1].values.len() {
        let delta = FE::from(1 + i as u64);
        g.current[1][1].values[i] += delta;
        let kept = &g.current[1][2].values[i] - &delta * &ratio;
        g.current[1][2].values[i] = kept;
    }
    let kept = crate::whir_round::combine_blocks(&g.current[1], &[eq_index(rho, 0), e1, e2]);
    assert_eq!(kept, honest, "the forgery keeps the combination");
    assert_eq!(
        trace::<F, KeccakWhir, _>(&o, &forged, &cfg, base),
        in_chain_0(Error::GroupedOpeningRejected { query: 1, tree: 1 }),
        "tree 1's own opening, not the fold"
    );
    check::<F, KeccakWhir, _>(&o, &o.proof, &cfg, base).unwrap();
}

/// The grouped opening under Poseidon1's 4-ary trees and their caps, as the
/// block's base runs it: each round-0 tree carries its own cap; the proof is
/// refused under another cap.
#[test]
fn a_grouped_opening_round_trips_under_p1_caps() {
    for policy in [CapPolicy::Off, CapPolicy::Fixed(1)] {
        let mut cfg = indexed(Some(6), 4, 3);
        cfg.format.arity4_cap = policy;
        for polys in [3usize, 5] {
            let o = open::<Ext, P1Whir, _>(&vec![N3; polys], N3, &cfg, 9, tower_p1).unwrap();
            check::<Ext, P1Whir, _>(&o, &o.proof, &cfg, tower_p1)
                .unwrap_or_else(|e| panic!("{policy} P={polys}: {e:?}"));
            let mut other = cfg;
            other.format.arity4_cap = if policy == CapPolicy::Off {
                CapPolicy::Fixed(1)
            } else {
                CapPolicy::Off
            };
            // Uncapped trees read under a cap have no cap to check; capped ones
            // read uncapped walk the owner's cap words to no root.
            let cause = if policy == CapPolicy::Off {
                Error::CapRejected
            } else {
                Error::GroupedOpeningRejected { query: 0, tree: 0 }
            };
            assert_eq!(
                trace::<Ext, P1Whir, _>(&o, &o.proof, &other, tower_p1),
                in_chain_0(cause),
                "{policy} P={polys}: read under {}",
                other.format.arity4_cap
            );
        }
    }
    // And a binary cap under KeccakWhir.
    let mut cfg = indexed(Some(6), 4, 3);
    cfg.format.cap = CapPolicy::Fixed(2);
    let o = open_tower(&[N3; 4], N3, &cfg);
    check_tower(&o, &o.proof, &cfg).unwrap();
}

// ─────────────────── from the code review (REV-OC-B) ───────────────────

/// The code review's M-a, host: each round-0 tree is bound to its OWN cap and
/// root. Under a binary cap, trees 0 and 1 trading the cap their owner paths
/// carry (query 0) is refused at tree 0's cap, `CapRejected`; under
/// Poseidon1's 4-ary cap, trading the whole owner paths likewise.
#[test]
fn round_zero_trees_cannot_trade_caps() {
    let mut cfg = indexed(Some(6), 4, 3);
    cfg.format.cap = CapPolicy::Fixed(2);
    let o = open_tower(&[N3; 3], N3, &cfg);
    check_tower(&o, &o.proof, &cfg).unwrap();
    let cap_nodes = 1usize << 2;
    let mut traded = o.proof.clone();
    let g = grouped_mut(&mut traded, 0);
    let len0 = g.current[0][0].proof.merkle_path.len();
    let len1 = g.current[0][1].proof.merkle_path.len();
    assert_eq!(len0, len1);
    assert!(len0 >= cap_nodes, "the owner path carries the cap");
    let cap0: Vec<_> = g.current[0][0].proof.merkle_path[len0 - cap_nodes..].to_vec();
    let cap1: Vec<_> = g.current[0][1].proof.merkle_path[len1 - cap_nodes..].to_vec();
    assert_ne!(cap0, cap1, "two trees, two caps");
    g.current[0][0].proof.merkle_path[len0 - cap_nodes..].clone_from_slice(&cap1);
    g.current[0][1].proof.merkle_path[len1 - cap_nodes..].clone_from_slice(&cap0);
    assert_eq!(
        trace_tower(&o, &traded, &cfg),
        in_chain_0(Error::CapRejected),
        "tree 0's cap is tree 1's"
    );

    let mut cfg = indexed(Some(6), 4, 3);
    cfg.format.arity4_cap = CapPolicy::Fixed(1);
    let o = open::<Ext, P1Whir, _>(&[N3; 3], N3, &cfg, 9, tower_p1).unwrap();
    check::<Ext, P1Whir, _>(&o, &o.proof, &cfg, tower_p1).unwrap();
    let mut traded = o.proof.clone();
    let g = grouped_mut(&mut traded, 0);
    let path0 = g.current[0][0].proof.merkle_path.clone();
    g.current[0][0].proof.merkle_path = g.current[0][1].proof.merkle_path.clone();
    g.current[0][1].proof.merkle_path = path0;
    assert_eq!(
        trace::<Ext, P1Whir, _>(&o, &traded, &cfg, tower_p1),
        in_chain_0(Error::CapRejected),
        "P1: tree 0's owner path is tree 1's"
    );
}

/// The code review's M-b, host: how many successor blocks a grouped round 0
/// carries is the verifier's schedule, never the proof's — a final round 0
/// (R = 1) presenting successor blocks, and a non-final one presenting none,
/// are each refused by name before anything is opened.
#[test]
fn successor_blocks_follow_the_schedule_not_the_proof() {
    let cfg = indexed(Some(6), 4, 3);
    let (o, _) = p3();
    let r1 = open_tower(&[3; 3], 3, &cfg);
    assert_eq!(r1.proof.polys[0].rounds.len(), 1, "R = 1");
    let mut extra = r1.proof.clone();
    grouped_mut(&mut extra, 0).next = grouped(&o.proof, 0).next.clone();
    assert_eq!(
        shape(check_tower(&r1, &extra, &cfg)),
        GroupedShape::SuccessorBlocks {
            expected: 0,
            got: 3
        }
    );
    let mut none = o.proof.clone();
    grouped_mut(&mut none, 0).next.clear();
    assert_eq!(
        shape(check_tower(&o, &none, &cfg)),
        GroupedShape::SuccessorBlocks {
            expected: 3,
            got: 0
        }
    );
}

// ─────────── each binding of a grouped round 0, on its own ───────────

/// [`open`] over the caller's columns, each column `i` claimed at the point
/// [`open`] gives it under `seed` — so two stacks that differ in one column
/// are claimed at the same points.
fn open_columns<E, H, T>(
    columns: &[Mle<F>],
    n_stack: usize,
    cfg: &ChainConfig,
    seed: u64,
    transcript: fn() -> T,
) -> Opened<E>
where
    F: IsSubFieldOf<E>,
    E: IsField + Send + Sync + 'static,
    FieldElement<E>: AsBytes + Send + Sync,
    T: IsTranscript<E>,
    H: WhirHash,
{
    let heights: Vec<usize> = columns.iter().map(Mle::num_vars).collect();
    let points: Vec<Vec<FieldElement<E>>> = heights
        .iter()
        .enumerate()
        .map(|(i, &h)| {
            (0..h)
                .map(|j| FieldElement::<E>::from(7 + seed + 11 * i as u64 + 3 * j as u64))
                .collect()
        })
        .collect();
    let values = columns
        .iter()
        .zip(&points)
        .map(|(c, p)| c.evaluate_in(p))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let layout = StackedLayout::build(&heights, n_stack).unwrap();
    let stacked =
        StackedCommitment::<F, H>::commit(layout.clone(), &borrow(columns), None, cfg).unwrap();
    let roots = stacked.roots();
    let proof = prove::<F, E, T, H, _>(
        &stacked,
        &borrow(columns),
        None,
        &Claimed::PerColumn(&points),
        &values,
        cfg,
        &mut transcript(),
    )
    .unwrap();
    Opened {
        layout,
        roots,
        domain: stacked.domain().clone(),
        points,
        values,
        proof,
    }
}

/// EACH round-0 tree of a grouped chain is bound to the root at its own
/// position, its cap included — tree 0 as much as a later one. The forgery,
/// for every slot `p` in turn: slot `p` commits another polynomial, and the
/// proof is that stack's honest one (its own trees, caps and paths,
/// self-consistent), claimed against the stated roots, which differ from its
/// own at slot `p` alone. The claim is false — slot `p`'s values are not the
/// evaluations of the polynomial its stated root commits — and every check
/// but tree `p`'s binding to that root passes; trading two trees' caps
/// ([`round_zero_trees_cannot_trade_caps`]) breaks two bindings at once and
/// cannot show either alone. A capped tree refuses at its cap (`CapRejected`,
/// before any query is opened), an uncapped one at its first opening.
#[test]
fn a_tree_committing_another_polynomial_is_refused_at_its_own_root() {
    fn case<H: WhirHash, T: IsTranscript<Ext>>(
        cfg: &ChainConfig,
        capped: bool,
        transcript: fn() -> T,
    ) {
        let honest: Vec<Mle<F>> = (0..3).map(|i| column(N3, 3 + 31 * i)).collect();
        for slot in 0..3 {
            let mut other = honest.clone();
            other[slot] = column(N3, 1_000 + slot as u64);
            let a = open_columns::<Ext, H, T>(&honest, N3, cfg, 3, transcript);
            let mut b = open_columns::<Ext, H, T>(&other, N3, cfg, 3, transcript);
            assert_eq!(a.points, b.points);
            assert_ne!(a.values, b.values);
            for p in 0..3 {
                assert_eq!(a.roots[p] == b.roots[p], p != slot, "slot {slot}: root {p}");
            }
            assert_eq!(trace::<Ext, H, T>(&b, &b.proof, cfg, transcript), Ok(()));
            b.roots = a.roots.clone();
            let cause = if capped {
                Error::CapRejected
            } else {
                Error::GroupedOpeningRejected {
                    query: 0,
                    tree: slot,
                }
            };
            assert_eq!(
                trace::<Ext, H, T>(&b, &b.proof, cfg, transcript),
                in_chain_0(cause),
                "slot {slot}, capped {capped}"
            );
        }
    }
    for cap in [CapPolicy::Fixed(2), CapPolicy::Off] {
        let mut cfg = indexed(Some(6), 4, 3);
        cfg.format.cap = cap;
        case::<KeccakWhir, _>(&cfg, cap != CapPolicy::Off, tower);
    }
    for cap in [CapPolicy::Fixed(1), CapPolicy::Off] {
        let mut cfg = indexed(Some(6), 4, 3);
        cfg.format.arity4_cap = cap;
        case::<P1Whir, _>(&cfg, cap != CapPolicy::Off, tower_p1);
    }
}

/// The grouped round 0's SUCCESSOR is authenticated by its own cap against the
/// root the round absorbed (`whir_round::verify_grouped`'s copy of the
/// per-polynomial round's check): a successor cap word that does not hash to
/// that root is refused at the cap, before any query's opening is read.
#[test]
fn a_grouped_successor_cap_off_its_root_is_refused_at_the_cap() {
    let mut cfg = indexed(Some(6), 4, 3);
    cfg.format.cap = CapPolicy::Fixed(1);
    let caps = cfg.tree_caps_at(N3, 2);
    assert!(caps[1] >= 1, "a capped successor: {caps:?}");
    let o = open_tower(&[N3; 3], N3, &cfg);
    check_tower(&o, &o.proof, &cfg).unwrap();
    let mut forged = o.proof.clone();
    let owner = &mut grouped_mut(&mut forged, 0).next[0].proof.merkle_path;
    let last = owner.len() - 1;
    owner[last][0] ^= 1;
    assert_eq!(
        trace_tower(&o, &forged, &cfg),
        in_chain_0(Error::CapRejected)
    );
}

/// A grouped round 0 that is the chain's last round (R = 1) compares each
/// query's combined fold with the final value — its own check, ahead of the
/// final weight check. At a blowup of one the round-0 trees have one leaf, so
/// a moved final value moves no query: the openings stay valid and the fold is
/// what refuses (the final check alone would say `EvaluationMismatch`).
#[test]
fn a_one_round_grouped_chain_compares_its_folds_with_the_final_value() {
    let mut cfg = indexed(Some(6), 4, 3);
    cfg.log_blowup = 0;
    let r1 = open_tower(&[3; 3], 3, &cfg);
    assert_eq!(r1.proof.polys[0].rounds.len(), 1);
    check_tower(&r1, &r1.proof, &cfg).unwrap();
    let mut forged = r1.proof.clone();
    forged.polys[0].final_value += EE::one();
    assert_eq!(
        trace_tower(&r1, &forged, &cfg),
        in_chain_0(Error::FoldInconsistent { query: 0 })
    );
}
