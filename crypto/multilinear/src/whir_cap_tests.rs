//! W1 — the Merkle cap on WHIR chains (design/CAP.md §5), end to end on the
//! host: every tree's paths stop `c` levels below its root, and the tree's cap
//! rides on its first opening in proof order (the owner path).
//!
//! Round-level fixtures that need the query positions (the unreached cap node
//! and the internal-node leaf of REVIEW-CAP M1) live in `whir_round::tests`,
//! where the query draw is reachable.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::{
    element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as F,
};

use crate::{
    Error,
    mle::Mle,
    whir::Domain,
    whir_chain::{
        CapPolicy, ChainConfig, ChainFormat, ChainProof, ChainRound, GrindBits, RoundOpenings,
        commit, prove, verify,
    },
    whir_commit::Commitment,
    whir_hash::{KeccakWhir, RpxWhir, WhirHash},
};

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

fn config(log_folding: usize, num_queries: usize, cap: CapPolicy) -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding,
        num_queries,
        grind: GrindBits::default(),
        format: ChainFormat {
            cap,
            ..ChainFormat::DEFAULT
        },
    }
}

struct Chain {
    proof: ChainProof<F, Ext>,
    root: Commitment,
    z: Vec<EE>,
    y: EE,
    domain: Domain<F>,
}

/// A base-field polynomial proved over the cubic tower, as production does:
/// round 0's blocks are base, every later one extension.
fn prove_chain<H: WhirHash>(num_vars: usize, cfg: &ChainConfig, seed: u64) -> Chain {
    let f = Mle::new(
        (0..(1u64 << num_vars))
            .map(|i| FE::from((i.wrapping_add(seed)).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11))
            .collect(),
    )
    .unwrap();
    let z: Vec<EE> = (0..num_vars)
        .map(|i| EE::from(101 + 3 * i as u64 + seed))
        .collect();
    let y = f.evaluate_in(&z).unwrap();
    let (commitment, domain) = commit::<F, H>(&f, cfg, true).unwrap();
    let proof = prove::<F, Ext, _, H>(
        &f,
        &z,
        &commitment,
        &domain,
        cfg,
        &mut DefaultTranscript::<Ext>::new(b"whir-cap"),
    )
    .unwrap();
    Chain {
        proof,
        root: commitment.root(),
        z,
        y,
        domain,
    }
}

fn check<H: WhirHash>(
    c: &Chain,
    proof: &ChainProof<F, Ext>,
    cfg: &ChainConfig,
) -> Result<(), Error> {
    verify::<F, Ext, _, H>(
        proof,
        &c.root,
        &c.z,
        c.y,
        &c.domain,
        cfg,
        &mut DefaultTranscript::<Ext>::new(b"whir-cap"),
    )
}

/// Path lengths of round `r`'s current and successor openings.
fn path_lens(round: &ChainRound<F, Ext>) -> (Vec<usize>, Vec<usize>) {
    match &round.openings {
        RoundOpenings::Base(p) => (
            p.current
                .iter()
                .map(|o| o.proof.merkle_path.len())
                .collect(),
            p.next.iter().map(|o| o.proof.merkle_path.len()).collect(),
        ),
        RoundOpenings::Extension(p) => (
            p.current
                .iter()
                .map(|o| o.proof.merkle_path.len())
                .collect(),
            p.next.iter().map(|o| o.proof.merkle_path.len()).collect(),
        ),
    }
}

fn current_path(round: &mut ChainRound<F, Ext>, i: usize) -> &mut Vec<Commitment> {
    match &mut round.openings {
        RoundOpenings::Base(p) => &mut p.current[i].proof.merkle_path,
        RoundOpenings::Extension(p) => &mut p.current[i].proof.merkle_path,
    }
}

fn next_path(round: &mut ChainRound<F, Ext>, i: usize) -> &mut Vec<Commitment> {
    match &mut round.openings {
        RoundOpenings::Base(p) => &mut p.next[i].proof.merkle_path,
        RoundOpenings::Extension(p) => &mut p.next[i].proof.merkle_path,
    }
}

/// Tree depths of a chain: tree `t` has `D_t − k_t` levels.
fn tree_depths(cfg: &ChainConfig, num_vars: usize) -> Vec<usize> {
    let mut d = num_vars + cfg.log_blowup;
    cfg.schedule(num_vars)
        .iter()
        .map(|k| {
            d -= k;
            d
        })
        .collect()
}

const SHAPES: [(usize, usize); 4] = [(6, 2), (5, 2), (3, 4), (9, 3)];
const POLICIES: [CapPolicy; 4] = [
    CapPolicy::Fixed(1),
    CapPolicy::Fixed(2),
    CapPolicy::Fixed(3),
    CapPolicy::Auto,
];

#[test]
fn tree_caps_follow_the_policy_and_are_zero_by_default() {
    // The production chain: blowup 2, fold 4, 128 bits, uniform 20-bit grinds.
    let mut cfg = ChainConfig::with_security(2, 4, 25, 128, GrindBits::uniform(20));
    assert_eq!(cfg.num_queries, 112);
    assert_eq!(tree_depths(&cfg, 25), vec![23, 19, 15, 11, 7, 3, 2]);
    assert_eq!(cfg.tree_caps(25), vec![0; 7], "the default caps nothing");
    cfg.format.cap = CapPolicy::Fixed(0);
    assert_eq!(cfg.tree_caps(25), vec![0; 7]);
    cfg.format.cap = CapPolicy::Auto;
    // 112 and 224 openings: 3, clamped to the depth of the last tree.
    assert_eq!(cfg.tree_caps(25), vec![3, 3, 3, 3, 3, 3, 2]);
    cfg.format.cap = CapPolicy::Fixed(5);
    assert_eq!(cfg.tree_caps(25), vec![5, 5, 5, 5, 5, 3, 2]);

    // Q = 3: tree 0 is opened 3 times (auto 0), every later tree 6 (auto 2).
    let small = config(2, 3, CapPolicy::Auto);
    assert_eq!(tree_depths(&small, 6), vec![6, 4, 2]);
    assert_eq!(small.tree_caps(6), vec![0, 2, 2]);
    // One round: the only tree is opened Q times.
    let one = config(4, 25, CapPolicy::Auto);
    assert_eq!(one.schedule(3), vec![3]);
    assert_eq!(
        one.tree_caps(3),
        vec![2],
        "25 openings -> 3, clamped to depth 2"
    );
}

/// Round trips at every policy, one- and multi-round schedules, a remainder
/// last round, base and extension rounds, both hashes; every path carries
/// exactly `depth − c` siblings and the owner exactly `2^c` more.
#[test]
fn capped_chains_round_trip_with_the_owner_path_lengths() {
    fn run<H: WhirHash>() {
        for (num_vars, k) in SHAPES {
            for q in [3usize, 25] {
                for policy in POLICIES {
                    let cfg = config(k, q, policy);
                    let caps = cfg.tree_caps(num_vars);
                    let depths = tree_depths(&cfg, num_vars);
                    let c = prove_chain::<H>(num_vars, &cfg, 5);
                    let tag = format!("{} S={num_vars} k={k} Q={q} {policy}", H::NAME);
                    for (r, round) in c.proof.rounds.iter().enumerate() {
                        let (cur, nxt) = path_lens(round);
                        let owner = |t: usize, i: usize, owned: bool| {
                            depths[t] - caps[t]
                                + if owned && i == 0 && caps[t] > 0 {
                                    1 << caps[t]
                                } else {
                                    0
                                }
                        };
                        for (i, len) in cur.iter().enumerate() {
                            assert_eq!(*len, owner(r, i, r == 0), "{tag}: round {r} current {i}");
                        }
                        for (i, len) in nxt.iter().enumerate() {
                            assert_eq!(*len, owner(r + 1, i, true), "{tag}: round {r} next {i}");
                        }
                    }
                    check::<H>(&c, &c.proof, &cfg).unwrap_or_else(|e| panic!("{tag}: {e:?}"));
                }
            }
        }
    }
    run::<KeccakWhir>();
    run::<RpxWhir>();
}

/// The default moves no byte: `Off` and `Fixed(0)` give the same archived
/// proof, and every path is the full depth.
#[test]
fn the_default_format_is_byte_identical_to_a_zero_cap() {
    for (num_vars, k) in SHAPES {
        let off = prove_chain::<KeccakWhir>(num_vars, &config(k, 3, CapPolicy::Off), 1);
        let zero = prove_chain::<KeccakWhir>(num_vars, &config(k, 3, CapPolicy::Fixed(0)), 1);
        let a = rkyv::to_bytes::<rkyv::rancor::Error>(&off.proof).unwrap();
        let b = rkyv::to_bytes::<rkyv::rancor::Error>(&zero.proof).unwrap();
        assert_eq!(a.as_slice(), b.as_slice(), "S={num_vars} k={k}");
        let cfg = config(k, 3, CapPolicy::Off);
        let depths = tree_depths(&cfg, num_vars);
        for (r, round) in off.proof.rounds.iter().enumerate() {
            let (cur, nxt) = path_lens(round);
            assert!(cur.iter().all(|l| *l == depths[r]));
            assert!(nxt.iter().all(|l| *l == depths[r + 1]));
        }
    }
}

/// REVIEW-CAP S2: the cap changes no transcript value. The same witness under
/// `Off`, `Fixed(3)` and `Auto` (no grinding, so the nonces are fixed) gives
/// the same sumchecks, roots, out-of-domain values, nonces and final value;
/// only the paths differ.
#[test]
fn the_cap_moves_no_transcript_value() {
    for (num_vars, k) in SHAPES {
        let base = prove_chain::<RpxWhir>(num_vars, &config(k, 25, CapPolicy::Off), 3);
        for policy in [CapPolicy::Fixed(3), CapPolicy::Auto] {
            let capped = prove_chain::<RpxWhir>(num_vars, &config(k, 25, policy), 3);
            assert_eq!(capped.root, base.root);
            assert_eq!(capped.proof.final_value, base.proof.final_value);
            for (a, b) in capped.proof.rounds.iter().zip(&base.proof.rounds) {
                assert_eq!(a.next_root, b.next_root);
                assert_eq!(a.ood_value, b.ood_value);
                assert_eq!(a.nonces, b.nonces);
                let ev = |r: &ChainRound<F, Ext>| -> Vec<EE> {
                    r.sumcheck
                        .iter()
                        .flat_map(|s| s.evaluations.clone())
                        .collect()
                };
                assert_eq!(ev(a), ev(b));
            }
        }
    }
}

fn expect_err(c: &Chain, forged: &ChainProof<F, Ext>, cfg: &ChainConfig, what: &str) -> Error {
    match check::<KeccakWhir>(c, forged, cfg) {
        Ok(()) => panic!("{what}: the forgery verified"),
        Err(e) => e,
    }
}

/// The tamper arm. The chain is `[2, 2, 2]` over 6 variables at `Fixed(2)`:
/// trees of depth 6, 4, 2, every one capped at 2.
#[test]
fn a_tampered_cap_or_owner_path_is_rejected() {
    let cfg = config(2, 3, CapPolicy::Fixed(2));
    assert_eq!(cfg.tree_caps(6), vec![2, 2, 2]);
    let c = prove_chain::<KeccakWhir>(6, &cfg, 9);
    check::<KeccakWhir>(&c, &c.proof, &cfg).unwrap();
    let depths = [6usize, 4, 2];

    // Tree 0's cap node: rides at the end of round 0's first current path.
    for j in 0..4 {
        let mut forged = c.proof.clone();
        current_path(&mut forged.rounds[0], 0)[depths[0] - 2 + j][5] ^= 1;
        assert!(matches!(
            expect_err(&c, &forged, &cfg, "tree-0 cap"),
            Error::CapRejected
        ));
    }
    // Tree t's cap node, t = 1, 2: rides on round t − 1's first successor path.
    for (t, depth) in depths.iter().enumerate().skip(1) {
        let mut forged = c.proof.clone();
        next_path(&mut forged.rounds[t - 1], 0)[depth - 2 + 1][0] ^= 1;
        assert!(matches!(
            expect_err(&c, &forged, &cfg, "tree-t cap"),
            Error::CapRejected
        ));
    }
    // Round t's first CURRENT opening carrying a cap as well: its path must
    // be exactly depth − c, so a second cap for the same tree is refused.
    for (t, depth) in depths.iter().enumerate().skip(1) {
        let mut forged = c.proof.clone();
        let cap = next_path(&mut forged.rounds[t - 1], 0)[depth - 2..].to_vec();
        current_path(&mut forged.rounds[t], 0).extend(cap);
        assert!(matches!(
            expect_err(&c, &forged, &cfg, "a second cap on round t's current[0]"),
            Error::OpeningRejected { query: 0 }
        ));
    }
    // The owner path one short (a cap node dropped) and one long.
    let mut forged = c.proof.clone();
    current_path(&mut forged.rounds[0], 0).pop();
    assert!(matches!(
        expect_err(&c, &forged, &cfg, "owner short"),
        Error::CapRejected
    ));
    let mut forged = c.proof.clone();
    let extra = current_path(&mut forged.rounds[0], 0)[0];
    current_path(&mut forged.rounds[0], 0).push(extra);
    assert!(matches!(
        expect_err(&c, &forged, &cfg, "owner long"),
        Error::CapRejected
    ));
    // A non-owner path one node long, and one short.
    let mut forged = c.proof.clone();
    let extra = current_path(&mut forged.rounds[0], 1)[0];
    current_path(&mut forged.rounds[0], 1).push(extra);
    assert!(matches!(
        expect_err(&c, &forged, &cfg, "non-owner long"),
        Error::OpeningRejected { query: 1 }
    ));
    let mut forged = c.proof.clone();
    // Tree 1 (depth 4, two siblings below its cap), as round 0's successor.
    assert!(next_path(&mut forged.rounds[0], 2).pop().is_some());
    assert!(matches!(
        expect_err(&c, &forged, &cfg, "non-owner short"),
        Error::OpeningRejected { query: 2 }
    ));
    // The cap moved from the first opening to the second.
    let mut forged = c.proof.clone();
    let cap: Vec<Commitment> = current_path(&mut forged.rounds[0], 0)
        .drain(depths[0] - 2..)
        .collect();
    current_path(&mut forged.rounds[0], 1).extend(cap);
    expect_err(&c, &forged, &cfg, "cap moved to query 1");
    // A sibling below the cap.
    let mut forged = c.proof.clone();
    current_path(&mut forged.rounds[1], 2)[0][0] ^= 1;
    assert!(matches!(
        expect_err(&c, &forged, &cfg, "sibling"),
        Error::OpeningRejected { query: 2 }
    ));

    // A proof made under one policy is refused under another: the cap height
    // is the verifier's constant, never read from the proof.
    expect_err(
        &c,
        &c.proof,
        &config(2, 3, CapPolicy::Fixed(1)),
        "c=2 read as c=1",
    );
    expect_err(
        &c,
        &c.proof,
        &config(2, 3, CapPolicy::Fixed(3)),
        "c=2 read as c=3",
    );
    expect_err(
        &c,
        &c.proof,
        &config(2, 3, CapPolicy::Off),
        "c=2 read as off",
    );
    let off = prove_chain::<KeccakWhir>(6, &config(2, 3, CapPolicy::Off), 9);
    expect_err(&off, &off.proof, &cfg, "off read as c=2");
}

/// A one-round chain has only the final round: tree 0's owner is its first
/// current opening there, and a flipped cap node is still refused.
#[test]
fn a_one_round_chain_carries_its_cap_on_the_final_openings() {
    let cfg = config(4, 25, CapPolicy::Auto);
    assert_eq!(cfg.tree_caps(3), vec![2]);
    let c = prove_chain::<RpxWhir>(3, &cfg, 2);
    assert_eq!(c.proof.rounds.len(), 1);
    check::<RpxWhir>(&c, &c.proof, &cfg).unwrap();
    let mut forged = c.proof.clone();
    let path = current_path(&mut forged.rounds[0], 0);
    // depth 2, cap 2: no siblings, four cap nodes.
    assert_eq!(path.len(), 4);
    path[3][7] ^= 1;
    assert!(matches!(
        check::<RpxWhir>(&c, &forged, &cfg),
        Err(Error::CapRejected)
    ));
}
