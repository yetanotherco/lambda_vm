//! D-WHIR-P1 S1: ZisK's Poseidon1 ([`P1Whir`]) on the WHIR chain, on the host.
//! Its trees are 4-ary: a path carries three siblings per 4-ary level, a cap
//! counts 4-ary levels (`ChainFormat::arity4_cap`), and an odd-depth tree's top
//! group is two real nodes and two padding digests.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::hash::poseidon1_stark::{Merkle4, linear_hash};
use crypto::hash::rpx::digest_to_commitment;
use crypto::merkle_tree::cap::{CapPolicy, cap_len};
use math::field::{
    element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as F,
};

use crate::{
    Error,
    mle::Mle,
    whir::Domain,
    whir_chain::{
        ChainConfig, ChainFormat, ChainProof, ChainRound, GrindBits, RoundOpenings, commit, prove,
        verify,
    },
    whir_commit::{
        Codeword, CodewordCommitment, Commitment, TreeDrop, tree_cap_height, verify_opening,
    },
    whir_hash::{KeccakWhir, P1Whir, RpxWhir, WhirHash},
};

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

fn config(log_folding: usize, num_queries: usize, arity4_cap: CapPolicy) -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding,
        num_queries,
        grind: GrindBits::default(),
        format: ChainFormat {
            arity4_cap,
            ..ChainFormat::DEFAULT
        },
    }
}

/// The configuration's own sponge with a test tag absorbed: the transcript a
/// block proof under `H` starts from.
fn sponge<H: WhirHash>() -> H::Sponge {
    let mut t = H::sponge();
    t.append_bytes(b"whir-p1");
    t
}

struct Chain {
    proof: ChainProof<F, Ext>,
    root: Commitment,
    z: Vec<EE>,
    y: EE,
    domain: Domain<F>,
}

fn poly(num_vars: usize, seed: u64) -> Mle<F> {
    Mle::new(
        (0..(1u64 << num_vars))
            .map(|i| FE::from((i.wrapping_add(seed)).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11))
            .collect(),
    )
    .unwrap()
}

/// A base-field polynomial proved over the cubic tower, as production does.
fn prove_chain<H: WhirHash>(num_vars: usize, cfg: &ChainConfig, seed: u64) -> Chain {
    let f = poly(num_vars, seed);
    let z: Vec<EE> = (0..num_vars)
        .map(|i| EE::from(101 + 3 * i as u64 + seed))
        .collect();
    let y = f.evaluate_in(&z).unwrap();
    let (commitment, domain) = commit::<F, H>(&f, cfg, true).unwrap();
    let proof =
        prove::<F, Ext, _, H>(&f, &z, &commitment, &domain, cfg, &mut sponge::<H>()).unwrap();
    Chain {
        proof,
        root: commitment.root(),
        z,
        y,
        domain,
    }
}

/// `proof` against `c`'s statement under `V`'s hash and sponge.
fn check<V: WhirHash>(
    c: &Chain,
    proof: &ChainProof<F, Ext>,
    cfg: &ChainConfig,
) -> Result<(), Error> {
    verify::<F, Ext, _, V>(
        proof,
        &c.root,
        &c.z,
        c.y,
        &c.domain,
        cfg,
        &mut sponge::<V>(),
    )
}

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

/// Binary depth (`log2` leaves) of each of a chain's trees.
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
    CapPolicy::Off,
    CapPolicy::Fixed(1),
    CapPolicy::Fixed(2),
    CapPolicy::Fixed(3),
];

/// The production chain's trees at arity 4: depths 23 … 2 are 12 … 1 levels,
/// and `Fixed(2)` caps each at two levels (one on the one-level last tree).
/// At arity 2 the arity-4 policy is never read.
#[test]
fn tree_caps_at_arity_4_count_four_ary_levels() {
    let mut cfg = ChainConfig::with_security(2, 4, 25, 128, GrindBits::uniform(20));
    assert_eq!(tree_depths(&cfg, 25), vec![23, 19, 15, 11, 7, 3, 2]);
    cfg.format.arity4_cap = CapPolicy::Fixed(2);
    assert_eq!(cfg.tree_caps_at(25, 4), vec![2, 2, 2, 2, 2, 2, 1]);
    cfg.format.arity4_cap = CapPolicy::Fixed(9);
    assert_eq!(
        cfg.tree_caps_at(25, 4),
        vec![8, 8, 8, 6, 4, 2, 1],
        "clamped to the levels and to 2^MAX_CAP_HEIGHT nodes"
    );
    cfg.format.arity4_cap = CapPolicy::Off;
    assert_eq!(cfg.tree_caps_at(25, 4), vec![0; 7]);
    for binary in [CapPolicy::Off, CapPolicy::Auto, CapPolicy::Fixed(5)] {
        for four in [CapPolicy::Off, CapPolicy::Fixed(2)] {
            cfg.format.cap = binary;
            cfg.format.arity4_cap = four;
            assert_eq!(
                cfg.tree_caps_at(25, 2),
                cfg.tree_caps(25),
                "{binary} / {four}"
            );
            let off = ChainFormat {
                arity4_cap: CapPolicy::Off,
                ..cfg.format
            };
            assert_eq!(
                cfg.tree_caps(25),
                ChainConfig { format: off, ..cfg }.tree_caps(25),
                "the arity-4 cap moves no binary height"
            );
        }
    }
    assert_eq!(tree_cap_height(CapPolicy::Fixed(3), 10, 5, 2), 3);
    assert_eq!(tree_cap_height(CapPolicy::Fixed(3), 10, 5, 4), 3);
    assert_eq!(tree_cap_height(CapPolicy::Fixed(3), 10, 3, 4), 2);
    assert_eq!(tree_cap_height(CapPolicy::Fixed(3), 0, 9, 4), 0);
}

/// Round trips at every arity-4 cap, one- and multi-round schedules, odd and
/// even depths, base and extension rounds; every path carries exactly
/// `3(h − c)` siblings and the owner its `cap_len` cap nodes more.
#[test]
fn p1_chains_round_trip_with_the_owner_path_lengths() {
    for (num_vars, k) in SHAPES {
        for q in [3usize, 25] {
            for policy in POLICIES {
                let cfg = config(k, q, policy);
                let caps = cfg.tree_caps_at(num_vars, 4);
                let depths = tree_depths(&cfg, num_vars);
                let c = prove_chain::<P1Whir>(num_vars, &cfg, 5);
                let tag = format!("S={num_vars} k={k} Q={q} {policy}");
                let len = |t: usize, i: usize, owned: bool| {
                    let walked = 3 * (depths[t].div_ceil(2) - caps[t]);
                    walked
                        + if owned && i == 0 && caps[t] > 0 {
                            cap_len(depths[t], caps[t], 4).unwrap()
                        } else {
                            0
                        }
                };
                for (r, round) in c.proof.rounds.iter().enumerate() {
                    let (cur, nxt) = path_lens(round);
                    for (i, l) in cur.iter().enumerate() {
                        assert_eq!(*l, len(r, i, r == 0), "{tag}: round {r} current {i}");
                    }
                    for (i, l) in nxt.iter().enumerate() {
                        assert_eq!(*l, len(r + 1, i, true), "{tag}: round {r} next {i}");
                    }
                }
                check::<P1Whir>(&c, &c.proof, &cfg).unwrap_or_else(|e| panic!("{tag}: {e:?}"));
            }
        }
    }
}

/// The arity-4 cap is inert under a binary hash: an RPX proof under
/// `arity4_cap = Fixed(2)` is the default format's, byte for byte.
#[test]
fn an_arity4_cap_moves_no_rpx_byte() {
    for (num_vars, k) in SHAPES {
        let a = prove_chain::<RpxWhir>(num_vars, &config(k, 3, CapPolicy::Off), 1);
        let b = prove_chain::<RpxWhir>(num_vars, &config(k, 3, CapPolicy::Fixed(2)), 1);
        assert_eq!(
            rkyv::to_bytes::<rkyv::rancor::Error>(&a.proof)
                .unwrap()
                .as_slice(),
            rkyv::to_bytes::<rkyv::rancor::Error>(&b.proof)
                .unwrap()
                .as_slice(),
            "S={num_vars} k={k}"
        );
    }
}

/// The hashes do not cross-verify: a P1 proof is refused by an RPX or a
/// keccak verifier, and an RPX proof by a P1 verifier, whatever the cap.
#[test]
fn the_bases_do_not_cross_verify() {
    for policy in [CapPolicy::Off, CapPolicy::Fixed(2)] {
        let cfg = config(2, 3, policy);
        let p1 = prove_chain::<P1Whir>(6, &cfg, 4);
        check::<P1Whir>(&p1, &p1.proof, &cfg).unwrap();
        assert!(check::<RpxWhir>(&p1, &p1.proof, &cfg).is_err());
        assert!(check::<KeccakWhir>(&p1, &p1.proof, &cfg).is_err());
        let rpx = prove_chain::<RpxWhir>(6, &cfg, 4);
        check::<RpxWhir>(&rpx, &rpx.proof, &cfg).unwrap();
        assert!(check::<P1Whir>(&rpx, &rpx.proof, &cfg).is_err());
    }
}

fn expect_err(c: &Chain, forged: &ChainProof<F, Ext>, cfg: &ChainConfig, what: &str) -> Error {
    match check::<P1Whir>(c, forged, cfg) {
        Ok(()) => panic!("{what}: the forgery verified"),
        Err(e) => e,
    }
}

/// The tamper arm, `[2, 2, 2]` over 6 variables at `Fixed(1)`: trees of
/// binary depth 6, 4, 2 (3, 2, 1 levels), each capped at one 4-ary level (a
/// cap of four nodes).
#[test]
fn a_tampered_p1_cap_or_path_is_rejected() {
    let cfg = config(2, 3, CapPolicy::Fixed(1));
    assert_eq!(cfg.tree_caps_at(6, 4), vec![1, 1, 1]);
    let c = prove_chain::<P1Whir>(6, &cfg, 9);
    check::<P1Whir>(&c, &c.proof, &cfg).unwrap();
    // Walked siblings below the cap, per tree: 3 × (levels − 1).
    let walked = [6usize, 3, 0];

    // Every node of tree 0's cap (the end of round 0's first current path).
    for j in 0..4 {
        let mut forged = c.proof.clone();
        current_path(&mut forged.rounds[0], 0)[walked[0] + j][5] ^= 1;
        assert!(matches!(
            expect_err(&c, &forged, &cfg, "tree-0 cap"),
            Error::CapRejected
        ));
    }
    // Every node of tree t's cap, t = 1, 2 (round t − 1's first successor path).
    for (t, w) in walked.iter().enumerate().skip(1) {
        for j in 0..4 {
            let mut forged = c.proof.clone();
            next_path(&mut forged.rounds[t - 1], 0)[w + j][0] ^= 1;
            assert!(matches!(
                expect_err(&c, &forged, &cfg, "tree-t cap"),
                Error::CapRejected
            ));
        }
    }
    // Every sibling of a non-owner path below the cap.
    for s in 0..walked[0] {
        let mut forged = c.proof.clone();
        current_path(&mut forged.rounds[0], 1)[s][3] ^= 1;
        assert!(matches!(
            expect_err(&c, &forged, &cfg, "sibling"),
            Error::OpeningRejected { query: 1 }
        ));
    }
    // The owner path one short and one long.
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
    // A non-owner path one long, and one short.
    let mut forged = c.proof.clone();
    let extra = current_path(&mut forged.rounds[0], 1)[0];
    current_path(&mut forged.rounds[0], 1).push(extra);
    assert!(matches!(
        expect_err(&c, &forged, &cfg, "non-owner long"),
        Error::OpeningRejected { query: 1 }
    ));
    let mut forged = c.proof.clone();
    assert!(next_path(&mut forged.rounds[0], 2).pop().is_some());
    assert!(matches!(
        expect_err(&c, &forged, &cfg, "non-owner short"),
        Error::OpeningRejected { query: 2 }
    ));
    // The cap moved from the first opening to the second.
    let mut forged = c.proof.clone();
    let cap: Vec<Commitment> = current_path(&mut forged.rounds[0], 0)
        .drain(walked[0]..)
        .collect();
    current_path(&mut forged.rounds[0], 1).extend(cap);
    expect_err(&c, &forged, &cfg, "cap moved to query 1");

    // The height is the verifier's constant, never read from the proof.
    for other in [CapPolicy::Off, CapPolicy::Fixed(2)] {
        expect_err(&c, &c.proof, &config(2, 3, other), "another arity-4 cap");
    }
    let off = prove_chain::<P1Whir>(6, &config(2, 3, CapPolicy::Off), 9);
    expect_err(&off, &off.proof, &cfg, "off read as c=1");
}

fn codeword(log_len: usize, seed: u64) -> Vec<FE> {
    (0..1u64 << log_len)
        .map(|i| FE::from(i.wrapping_mul(0x9E37_79B9).wrapping_add(seed) >> 3))
        .collect()
}

/// Every binary depth 1–9 at cap 0: every opening verifies with exactly
/// `3⌈d/2⌉` siblings, and none at an index past the tree. At an odd depth the
/// walk reaches the top group's two padding siblings: they are the zero digest,
/// and anything else there is refused. The root binds them already (they are
/// hashed into it); the host's own zero check (REV-P1-JUDGE F4) is defence in
/// depth, which a mutation dropping it leaves this test green.
#[test]
fn every_depth_opens_and_an_odd_tops_padding_is_bound() {
    for depth in 1..=9usize {
        for k in [0usize, 2] {
            let cw = codeword(depth + k, depth as u64);
            let commitment = CodewordCommitment::<F, P1Whir>::new(&cw, k).unwrap();
            assert_eq!(commitment.depth(), depth);
            assert_eq!(commitment.levels(), depth.div_ceil(2));
            let root = commitment.root();
            for j in 0..commitment.num_leaves() {
                let opening = commitment.open(j).unwrap();
                assert_eq!(opening.proof.merkle_path.len(), 3 * depth.div_ceil(2));
                assert!(
                    verify_opening::<F, P1Whir>(&root, depth, j, &opening),
                    "d={depth} k={k} leaf {j}"
                );
                assert!(
                    !verify_opening::<F, P1Whir>(
                        &root,
                        depth,
                        j + commitment.num_leaves(),
                        &opening
                    ),
                    "d={depth} k={k}: an index past the tree"
                );
                if depth % 2 == 1 {
                    let n = opening.proof.merkle_path.len();
                    for at in [n - 2, n - 1] {
                        assert_eq!(opening.proof.merkle_path[at], [0u8; 32], "padding");
                        let mut forged = opening.clone();
                        forged.proof.merkle_path[at][31] ^= 1;
                        assert!(
                            !verify_opening::<F, P1Whir>(&root, depth, j, &forged),
                            "d={depth} k={k} leaf {j}: a moved padding sibling {at}"
                        );
                    }
                }
            }
        }
    }
}

/// A P1 root is ZisK's tree: the tagged `linear_hash` of each fold block's
/// felts in coset order (an ext3 value's three coefficients in order), folded
/// by `Merkle4` with zero padding — at odd and even depths, base and ext3.
#[test]
fn a_p1_root_is_zisks_merkle4_over_the_blocks() {
    for (log_len, k) in [(5usize, 2usize), (6, 2), (7, 4), (9, 3), (10, 4)] {
        let cw = codeword(log_len, 3);
        let commitment = CodewordCommitment::<F, P1Whir>::new(&cw, k).unwrap();
        let n = commitment.num_leaves();
        let leaves: Vec<_> = (0..n)
            .map(|j| {
                let felts: Vec<FE> = (0..1usize << k).map(|t| cw[j + t * n]).collect();
                linear_hash(&felts)
            })
            .collect();
        let want = digest_to_commitment(&Merkle4::new(&leaves).unwrap().root());
        assert_eq!(commitment.root(), want, "base log_len={log_len} k={k}");

        let ext: Vec<EE> = cw
            .iter()
            .enumerate()
            .map(|(i, v)| EE::new([*v, FE::from(i as u64), v.square()]))
            .collect();
        let commitment = CodewordCommitment::<Ext, P1Whir>::new(&ext, k).unwrap();
        let leaves: Vec<_> = (0..n)
            .map(|j| {
                let felts: Vec<FE> = (0..1usize << k)
                    .flat_map(|t| ext[j + t * n].value().to_vec())
                    .collect();
                linear_hash(&felts)
            })
            .collect();
        let want = digest_to_commitment(&Merkle4::new(&leaves).unwrap().root());
        assert_eq!(commitment.root(), want, "ext3 log_len={log_len} k={k}");
    }
}

/// The drop at arity 4 counts 4-ary levels, half the binary request (the same
/// leaves a block), never past the frontier where a kept node covers `4^d`
/// whole leaves, and never into the tallest cap.
#[test]
fn the_arity4_drop_halves_the_request_and_stops_at_whole_groups() {
    let drop = TreeDrop { device: 8, host: 4 };
    assert_eq!(drop.levels_at(true, 23, 2, 4), 4);
    assert_eq!(drop.levels_at(false, 23, 2, 4), 2);
    assert_eq!(drop.levels_at(true, 23, 3, 2), 8, "binary as before");
    assert_eq!(
        drop.levels_at(true, 7, 0, 4),
        3,
        "odd: the two-node top stays"
    );
    assert_eq!(drop.levels_at(true, 8, 0, 4), 4, "even: down to the root");
    assert_eq!(drop.levels_at(true, 7, 2, 4), 2, "never into the cap");
    assert_eq!(drop.levels_at(true, 3, 0, 4), 1);
    assert_eq!(drop.levels_at(true, 1, 0, 4), 0);
}

/// A retired and revived P1 commitment serves the whole tree's paths byte for
/// byte, capped or not, at odd and even depths and every drop; the owner's cap
/// is read from the kept levels; and a wrong codeword is refused when a path
/// is read.
#[test]
fn a_retired_p1_commitment_serves_the_full_tree_paths() {
    for (log_len, k) in [(9usize, 2usize), (10, 2), (12, 1)] {
        let cw = codeword(log_len, 17);
        let full = CodewordCommitment::<F, P1Whir>::new(&cw, k).unwrap();
        let n = full.num_leaves();
        let indices: Vec<usize> = (0..12usize).map(|q| (q * 2_654_435_761) % n).collect();
        for policy in [CapPolicy::Off, CapPolicy::Fixed(1), CapPolicy::Fixed(2)] {
            let c = tree_cap_height(policy, indices.len(), full.depth(), 4);
            let want = full.open_many_capped(&indices, c, true).unwrap();
            for d in [0usize, 2, 3, 4, 6, 8] {
                let tag = format!("log_len={log_len} k={k} {policy} drop {d}");
                let retired = CodewordCommitment::<F, P1Whir>::new(&cw, k)
                    .unwrap()
                    .retire(TreeDrop::uniform(d), policy)
                    .unwrap();
                assert_eq!(retired.root(), full.root(), "{tag}");
                let revived = retired
                    .clone()
                    .revive::<F, P1Whir>(Codeword::Host(cw.clone()))
                    .unwrap();
                let got = revived.open_many_capped(&indices, c, true).unwrap();
                assert_eq!(format!("{got:?}"), format!("{want:?}"), "{tag}");
                let mut bad = cw.clone();
                for j in &indices {
                    bad[*j] += FE::one();
                }
                let wrong = retired.revive::<F, P1Whir>(Codeword::Host(bad)).unwrap();
                assert!(
                    matches!(
                        wrong.open_many_capped(&indices, c, true),
                        Err(Error::RecomputedCodewordMismatch { .. })
                    ),
                    "{tag}: a wrong codeword"
                );
            }
        }
    }
}

/// A stacked opening under P1, as the block's groups use it: committed,
/// proved and verified over the extension; a stack retired and revived opens
/// to the same bytes at every drop and verifies against the committed roots;
/// and the same proof under another arity-4 cap or another hash is refused.
#[test]
fn a_p1_stacked_opening_round_trips_and_survives_a_retire() {
    use crate::stacked_eval::{Claimed, StackedCommitment, prove, verify};
    use crate::stacking::{StackedLayout, borrow};

    let num_vars = 6;
    let columns: Vec<Mle<F>> = (0..5u64).map(|i| poly(num_vars, 7 + 13 * i)).collect();
    let layout = StackedLayout::build(&[num_vars; 5], 8).unwrap();
    assert!(layout.num_polys() >= 2);
    let at: Vec<EE> = (0..num_vars)
        .map(|i| EE::new([FE::from(3 + i as u64), FE::from(5), FE::from(i as u64)]))
        .collect();
    let claimed: Vec<EE> = columns
        .iter()
        .map(|c| c.evaluate_in(&at).unwrap())
        .collect();
    for policy in [CapPolicy::Off, CapPolicy::Fixed(1), CapPolicy::Fixed(2)] {
        let cfg = config(2, 3, policy);
        let kept =
            StackedCommitment::<F, P1Whir>::commit(layout.clone(), &borrow(&columns), None, &cfg)
                .unwrap();
        let roots = kept.roots();
        let proof = prove(
            &kept,
            &borrow(&columns),
            None,
            &Claimed::Shared(&at),
            &claimed,
            &cfg,
            &mut sponge::<P1Whir>(),
        )
        .unwrap();
        let verified = |p: &crate::stacked_eval::StackedProof<F, Ext>, cfg: &ChainConfig| {
            verify::<F, _, _, P1Whir>(
                p,
                &layout,
                &roots,
                &Claimed::Shared(&at),
                &claimed,
                kept.domain(),
                cfg,
                &mut sponge::<P1Whir>(),
            )
        };
        verified(&proof, &cfg).unwrap_or_else(|e| panic!("{policy}: {e:?}"));
        let other = if policy == CapPolicy::Fixed(1) {
            CapPolicy::Fixed(2)
        } else {
            CapPolicy::Fixed(1)
        };
        assert!(
            verified(&proof, &config(2, 3, other)).is_err(),
            "{policy} read as {other}"
        );
        assert!(
            verify::<F, _, _, RpxWhir>(
                &proof,
                &layout,
                &roots,
                &Claimed::Shared(&at),
                &claimed,
                kept.domain(),
                &cfg,
                &mut sponge::<RpxWhir>(),
            )
            .is_err(),
            "{policy}: an RPX verifier"
        );
        let want = format!("{proof:?}");
        for drop in [0usize, 1, 3, 8, 64] {
            let retired = StackedCommitment::<F, P1Whir>::commit(
                layout.clone(),
                &borrow(&columns),
                None,
                &cfg,
            )
            .unwrap()
            .retire(TreeDrop::uniform(drop), &cfg)
            .unwrap();
            assert_eq!(retired.roots(), roots);
            let revived = retired
                .revive::<P1Whir, _>(&borrow(&columns), None, &cfg)
                .unwrap();
            let again = prove(
                &revived,
                &borrow(&columns),
                None,
                &Claimed::Shared(&at),
                &claimed,
                &cfg,
                &mut sponge::<P1Whir>(),
            )
            .unwrap();
            assert_eq!(format!("{again:?}"), want, "{policy} drop {drop}");
        }
    }
}

/// ★ A WHIR P1 tree's internal node never opens as a leaf: the node one 4-ary
/// level above block 0, with the path from it upward (three siblings short),
/// lands on the cap at index 0 — only the capped check's exact length refuses
/// it. At odd and even depths, uncapped and capped.
#[test]
fn an_internal_node_never_opens_as_a_leaf() {
    use crypto::merkle_tree::backends::p1::P1BatchBackend;
    use crypto::merkle_tree::cap::{
        split_owner_path_arity, verify_merkle_path_to_cap_from_leaf_hash,
    };
    use crypto::merkle_tree::traits::IsMerkleTreeBackend;
    type B = P1BatchBackend<F>;

    for (log_len, k) in [(9usize, 2usize), (10, 2), (12, 3)] {
        let cw = codeword(log_len, 29);
        let commitment = CodewordCommitment::<F, P1Whir>::new(&cw, k).unwrap();
        let d = commitment.depth();
        let n = commitment.num_leaves();
        for c in [0usize, 1, 2] {
            let tag = format!("d={d} c={c}");
            let owner = commitment
                .open_many_capped(&[0], c, true)
                .unwrap()
                .remove(0);
            let (siblings, cap) =
                split_owner_path_arity(&owner.proof.merkle_path, d, c, 4).expect("an owner path");
            // At c = 0 the owner path carries no cap: the cap is the root.
            let root = [commitment.root()];
            let cap = if c == 0 { &root[..] } else { cap };
            let leaf = B::hash_data(&owner.values);
            assert!(
                verify_merkle_path_to_cap_from_leaf_hash::<B>(siblings, cap, d, 0, leaf),
                "{tag}: the honest opening"
            );
            let block0: Vec<FE> = (0..1usize << k).map(|t| cw[t * n]).collect();
            assert_eq!(owner.values, block0);
            let parent = B::hash_four(&[leaf, siblings[0], siblings[1], siblings[2]]);
            assert!(
                !verify_merkle_path_to_cap_from_leaf_hash::<B>(&siblings[3..], cap, d, 0, parent),
                "{tag}: an internal node opened as leaf 0"
            );
        }
    }
}

/// REV-P1W-A: a P1 chain verifies under its own arity-4 cap only. Proved at
/// `Fixed(2)` (the WHIR block's C2) over trees of odd and even depth, it is
/// refused under every other cap policy that yields another geometry and under
/// the binary hashes; `Off` and `Fixed(0)` are one geometry (the tag renders
/// both `C0`) and cross-verify, as they must.
#[test]
fn rp1wa_a_p1_chain_verifies_under_its_own_cap_only() {
    for (num_vars, log_folding) in [(7usize, 2usize), (8, 3), (9, 2)] {
        let c2 = config(log_folding, 5, CapPolicy::Fixed(2));
        let chain = prove_chain::<P1Whir>(num_vars, &c2, 17);
        check::<P1Whir>(&chain, &chain.proof, &c2).unwrap();
        for other in [
            CapPolicy::Off,
            CapPolicy::Fixed(0),
            CapPolicy::Fixed(1),
            CapPolicy::Fixed(3),
            CapPolicy::Auto,
        ] {
            let cfg = config(log_folding, 5, other);
            if cfg.tree_caps_at(num_vars, 4) == c2.tree_caps_at(num_vars, 4) {
                continue;
            }
            assert!(
                check::<P1Whir>(&chain, &chain.proof, &cfg).is_err(),
                "n {num_vars} k {log_folding}: a C2 proof verified under {other:?}"
            );
        }
        assert!(check::<RpxWhir>(&chain, &chain.proof, &c2).is_err());
        assert!(check::<KeccakWhir>(&chain, &chain.proof, &c2).is_err());
        let off = config(log_folding, 5, CapPolicy::Off);
        let uncapped = prove_chain::<P1Whir>(num_vars, &off, 18);
        check::<P1Whir>(&uncapped, &uncapped.proof, &config(log_folding, 5, CapPolicy::Fixed(0)))
            .unwrap();
    }
}
