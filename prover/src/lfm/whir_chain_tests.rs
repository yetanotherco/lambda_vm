//! Gates for the assembled WHIR chain.
//!
//! The proof under test is a real one: `whir_chain::prove` over a real
//! polynomial and a real commitment, under the RPX transcript the machine
//! replays. Small — a 2^6 polynomial, three queries — because the chain's
//! shapes are what this checks and a 2^25 stack is a box's work, not a
//! laptop's.
//!
//! # ★ Why executing the honest proof IS the stream comparison
//!
//! The machine never compares its challenges to the host's. It USES them, and
//! every one of them feeds a refusal: a wrong query index opens the wrong leaf
//! and fails at the root, a wrong folding challenge folds to something the
//! successor block does not hold, a wrong sumcheck challenge leaves a claim the
//! tail's division does not close. So a proof that the host accepts executing
//! here is already the statement that every drawn value agrees.
//!
//! What the recording transcript IS used for is a second check, on the host and
//! not on the machine: the SHAPE of the draw stream. `verify_weighted` must draw
//! one extension challenge per sumcheck round plus `z0` and `gamma` on every
//! round but the last, and `Q` bounded draws per round — `num_vars + 2(R − 1)`
//! and `Q·R`. Those counts are what the emitter's order is built on, so if the
//! host ever reorders, this says so rather than leaving the machine to fail at
//! a root and look like a hashing bug.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::fiat_shamir::transcript_hash::RpxTranscriptHash;
use multilinear::mle::Mle;
use multilinear::whir::Domain;
use multilinear::whir_chain::{
    ChainConfig, ChainProof, ChainRound, GrindBits, RoundOpenings, commit, prove, verify,
};
use multilinear::whir_hash::RpxWhir;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::algebraic_commit::commitment_to_digest;
use super::builder::{Ext, Felt, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::edsl::WrapDigest;
use super::executor::execute;
use super::validator::validate;
use super::whir_chain::{
    ChainRoundWires, ChainShape, QueryOpening, RoundNonces, chain_opening_perms,
    emit_verify_weighted,
};
use super::whir_open::BlockValues;
use super::whir_poly::emit_eq_eval;
use super::whir_transcript::WhirTranscript;
use super::word::{LfmWord, ext_word};

type F = GoldilocksField;
type E = GoldilocksExtension;
type HostTranscript = DefaultTranscript<E, RpxTranscriptHash>;

/// A transcript that records what it hands out, delegating everything.
///
/// The chain's challenges are drawn interleaved with its sumchecks and never
/// returned, and re-deriving them would mirror the host rather than read it.
struct Recording {
    inner: HostTranscript,
    sampled: Vec<FEE>,
    drawn_u64: Vec<u64>,
}

impl Recording {
    fn new() -> Self {
        Self {
            inner: HostTranscript::new(&[]),
            sampled: Vec::new(),
            drawn_u64: Vec::new(),
        }
    }
}

impl IsTranscript<E> for Recording {
    fn append_field_element(&mut self, element: &FEE) {
        self.inner.append_field_element(element);
    }
    fn append_bytes(&mut self, new_bytes: &[u8]) {
        self.inner.append_bytes(new_bytes);
    }
    fn state(&self) -> [u8; 32] {
        self.inner.state()
    }
    fn sample_field_element(&mut self) -> FEE {
        let drawn = self.inner.sample_field_element();
        self.sampled.push(drawn);
        drawn
    }
    fn sample_u64(&mut self, upper_bound: u64) -> u64 {
        let drawn = self.inner.sample_u64(upper_bound);
        self.drawn_u64.push(drawn);
        drawn
    }
}

/// A current block's wires, in the field its round holds them in.
enum CurrentBlock {
    Base(Vec<Felt>),
    Ext(Vec<Ext>),
}

impl CurrentBlock {
    fn as_block(&self) -> BlockValues<'_> {
        match self {
            CurrentBlock::Base(v) => BlockValues::Base(v),
            CurrentBlock::Ext(v) => BlockValues::Ext(v),
        }
    }
}

fn pseudo_mle(num_vars: usize, seed: u64) -> Mle<F> {
    let vals = (0..1usize << num_vars)
        .map(|i| {
            let mixed = (i as u64)
                .wrapping_mul(0x2545_F491_4F6C_DD1D)
                .wrapping_add(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            FE::from(mixed >> 13)
        })
        .collect();
    Mle::new(vals).expect("a cube")
}

fn point(num_vars: usize, seed: u64) -> Vec<FEE> {
    (0..num_vars)
        .map(|i| {
            FEE::new([
                FE::from(101 + seed + i as u64),
                FE::from(7 * i as u64 + 3),
                FE::from(i as u64 + 1),
            ])
        })
        .collect()
}

fn config(num_queries: usize, grind: u8) -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 4,
        num_queries,
        grind: GrindBits::uniform(grind),
    }
}

/// Everything one run needs on both sides.
struct Fixture {
    proof: ChainProof<F, E>,
    z: Vec<FEE>,
    y: FEE,
    root: LfmWord,
    /// The same root as the host holds it — kept rather than rebuilt from the
    /// lanes, because a second conversion is a second thing to get wrong.
    root_bytes: [u8; 32],
    domain: Domain<F>,
    shape: ChainShape,
    /// Kept so a failing run can print which draw diverged; the counts it is
    /// checked for are asserted in `fixture` above.
    #[allow(dead_code)]
    recorded: Recording,
}

/// Proves and verifies one chain on the host, keeping the verifier's draws.
///
/// ⚠ Under `RpxTranscriptHash`, not the default keccak one — the machine's
/// replay reproduces that hash and no other, so a fixture on the default
/// transcript would be a fixture of a different protocol.
fn fixture(num_vars: usize, num_queries: usize, grind: u8) -> Fixture {
    let cfg = config(num_queries, grind);
    let f = pseudo_mle(num_vars, 11);
    let z = point(num_vars, 0);
    // `evaluate_in`, not `evaluate`: the claimed point is in the cubic
    // extension, which is where every WHIR challenge lives.
    let y = f.evaluate_in::<E>(&z).expect("f takes its own point");

    let (commitment, domain) =
        commit::<F, RpxWhir>(&f, &cfg, true).expect("the polynomial commits");
    let mut proving = HostTranscript::new(&[]);
    let proof = prove::<F, E, _, RpxWhir>(&f, &z, &commitment, &domain, &cfg, &mut proving)
        .expect("the chain proves");

    let mut recorded = Recording::new();
    verify::<F, E, _, RpxWhir>(
        &proof,
        &commitment.root(),
        &z,
        y,
        &domain,
        &cfg,
        &mut recorded,
    )
    .expect("the control proof must verify");

    // ★ The host's draw stream has the shape the emitter's order assumes: one
    // challenge per sumcheck round, `z0` and `gamma` on every round but the
    // last, and `Q` bounded draws a round. Derived from the schedule, not read
    // off the recorder.
    let shape = ChainShape::new(&cfg, num_vars);
    let rounds = shape.rounds();
    assert_eq!(
        recorded.sampled.len(),
        num_vars + 2 * (rounds - 1),
        "extension draws: one a sumcheck round ({num_vars} in all), plus z0 and gamma on \
         each of the {} rounds with a successor",
        rounds - 1
    );
    assert_eq!(
        recorded.drawn_u64.len(),
        num_queries * rounds,
        "bounded draws: {num_queries} query positions in each of {rounds} rounds"
    );

    Fixture {
        proof,
        z,
        y,
        root: commitment_to_digest(&commitment.root()),
        root_bytes: commitment.root(),
        domain,
        shape,
        recorded,
    }
}

/// Where every wire of one round lives in the arena. Built once and used by
/// both the program and the arena filler, so the two cannot drift.
struct Layout {
    shape: ChainShape,
    /// The arena index each round's block of wires starts at.
    round_at: Vec<u32>,
    total: u32,
}

impl Layout {
    fn new(shape: &ChainShape) -> Self {
        let mut round_at = Vec::with_capacity(shape.rounds());
        // z, y, the root, and the final value.
        let mut at = (shape.num_vars + 3) as u32;
        for r in 0..shape.rounds() {
            round_at.push(at);
            at += Self::round_words(shape, r);
        }
        Self {
            shape: shape.clone(),
            round_at,
            total: at,
        }
    }

    fn round_words(shape: &ChainShape, r: usize) -> u32 {
        let k = shape.schedule[r];
        // The sumcheck's two evaluations a round, three nonces, and per query
        // the current block plus its path.
        let mut n = (2 * k + 3) as u32;
        let depth = shape.current_depth(r);
        let block = 1usize << k;
        n += (shape.num_queries * (block + depth)) as u32;
        if let Some(next_depth) = shape.next_depth(r) {
            // The successor root, its out-of-domain value, and per query its
            // block and path.
            n += 2;
            let next_block = 1usize << shape.schedule[r + 1];
            n += (shape.num_queries * (next_block + next_depth)) as u32;
        }
        n
    }
}

/// Builds the program for one shape, publishing every extension challenge the
/// machine draws in the order it draws them.
fn chain_program(shape: &ChainShape) -> LfmProgram {
    let layout = Layout::new(shape);
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(layout.total);
    let mut transcript = WhirTranscript::new();

    let z: Vec<Ext> = (0..shape.num_vars)
        .map(|i| b.hint_word(arena, i as u32).as_ext())
        .collect();
    let y = b.hint_word(arena, shape.num_vars as u32).as_ext();
    let root = b.hint_word(arena, shape.num_vars as u32 + 1);
    let root_lanes = b.unpack(root);
    let final_value = b.hint_word(arena, shape.num_vars as u32 + 2).as_ext();

    // Owned storage, because the wires borrow from it.
    let mut sumchecks: Vec<Vec<Vec<Ext>>> = Vec::new();
    let mut currents: Vec<Vec<(CurrentBlock, Vec<WrapDigest>)>> = Vec::new();
    let mut nexts: Vec<Vec<(Vec<Ext>, Vec<WrapDigest>)>> = Vec::new();
    let mut roots: Vec<Option<super::builder::Cell>> = Vec::new();
    let mut oods: Vec<Option<Ext>> = Vec::new();
    let mut nonces: Vec<RoundNonces> = Vec::new();

    for r in 0..shape.rounds() {
        let mut at = layout.round_at[r];
        let next_word = |b: &mut LfmBuilder, at: &mut u32| {
            let cell = b.hint_word(arena, *at);
            *at += 1;
            cell
        };
        let k = shape.schedule[r];
        let sumcheck: Vec<Vec<Ext>> = (0..k)
            .map(|_| {
                (0..2)
                    .map(|_| next_word(&mut b, &mut at).as_ext())
                    .collect()
            })
            .collect();
        // The nonces are FELTS: `append_bytes(&nonce.to_be_bytes())` is one
        // big-endian felt, which is how the grind absorbs them.
        let folding = b.hint_felt(arena, at);
        let ood_nonce = b.hint_felt(arena, at + 1);
        let query = b.hint_felt(arena, at + 2);
        at += 3;

        let depth = shape.current_depth(r);
        let block = 1usize << k;
        // ★ ROUND 0's current codeword is BASE on the host
        // (`whir_chain.rs:983`), so its block hashes ONE felt a value and not
        // three. Hinting it as extension wires would hash forty-eight felts
        // where the committer hashed sixteen and the root would never match —
        // which is exactly how this test first failed.
        let current: Vec<(CurrentBlock, Vec<WrapDigest>)> = (0..shape.num_queries)
            .map(|_| {
                let values = if r == 0 {
                    let felts: Vec<Felt> = (0..block)
                        .map(|_| {
                            let f = b.hint_felt(arena, at);
                            at += 1;
                            f
                        })
                        .collect();
                    CurrentBlock::Base(felts)
                } else {
                    CurrentBlock::Ext(
                        (0..block)
                            .map(|_| next_word(&mut b, &mut at).as_ext())
                            .collect(),
                    )
                };
                let path: Vec<WrapDigest> = (0..depth)
                    .map(|_| WrapDigest::from_cell(next_word(&mut b, &mut at)))
                    .collect();
                (values, path)
            })
            .collect();

        let (next_root, ood_value, next) = match shape.next_depth(r) {
            Some(next_depth) => {
                let nr = next_word(&mut b, &mut at);
                let ov = next_word(&mut b, &mut at).as_ext();
                let next_block = 1usize << shape.schedule[r + 1];
                let next: Vec<(Vec<Ext>, Vec<WrapDigest>)> = (0..shape.num_queries)
                    .map(|_| {
                        let values: Vec<Ext> = (0..next_block)
                            .map(|_| next_word(&mut b, &mut at).as_ext())
                            .collect();
                        let path: Vec<WrapDigest> = (0..next_depth)
                            .map(|_| WrapDigest::from_cell(next_word(&mut b, &mut at)))
                            .collect();
                        (values, path)
                    })
                    .collect();
                (Some(nr), Some(ov), next)
            }
            None => (None, None, Vec::new()),
        };

        sumchecks.push(sumcheck);
        currents.push(current);
        nexts.push(next);
        roots.push(next_root);
        oods.push(ood_value);
        nonces.push(RoundNonces {
            folding,
            ood: ood_nonce,
            query,
        });
    }

    let current_openings: Vec<Vec<QueryOpening<'_>>> = currents
        .iter()
        .map(|round| {
            round
                .iter()
                .map(|(values, path)| QueryOpening {
                    values: values.as_block(),
                    siblings: path,
                })
                .collect()
        })
        .collect();
    let next_openings: Vec<Vec<QueryOpening<'_>>> = nexts
        .iter()
        .map(|round| {
            round
                .iter()
                .map(|(values, path)| QueryOpening {
                    values: BlockValues::Ext(values),
                    siblings: path,
                })
                .collect()
        })
        .collect();

    let wires: Vec<ChainRoundWires<'_>> = (0..shape.rounds())
        .map(|r| ChainRoundWires {
            sumcheck: &sumchecks[r],
            next_root: roots[r],
            ood_value: oods[r],
            nonces: nonces[r],
            current: &current_openings[r],
            next: &next_openings[r],
        })
        .collect();

    emit_verify_weighted(
        &mut b,
        &mut transcript,
        &wires,
        &root_lanes,
        final_value,
        y,
        shape,
        &layout.shape_domain(),
        |b, alphas| emit_eq_eval(b, &z, alphas),
    );

    let program = compile(b.finish());
    validate(&program).expect("the chain leg must be admissible");
    program
}

impl Layout {
    fn shape_domain(&self) -> Domain<F> {
        Domain::<F>::new(self.shape.domain_log[0]).expect("the chain's first domain")
    }
}

/// The arena in the order [`chain_program`] hints it.
fn chain_arena(fixture: &Fixture, proof: &ChainProof<F, E>) -> Vec<LfmWord> {
    let shape = &fixture.shape;
    let mut words: Vec<LfmWord> = fixture.z.iter().map(ext_word).collect();
    words.push(ext_word(&fixture.y));
    words.push(fixture.root);
    words.push(ext_word(&proof.final_value));

    for (r, round) in proof.rounds.iter().enumerate() {
        for sc in &round.sumcheck {
            for e in &sc.evaluations {
                words.push(ext_word(e));
            }
        }
        for nonce in [round.nonces.folding, round.nonces.ood, round.nonces.query] {
            words.push([FE::from(nonce), FE::zero(), FE::zero(), FE::zero()]);
        }
        push_openings(&mut words, round, true);
        if shape.next_depth(r).is_some() {
            words.push(commitment_to_digest(
                round.next_root.as_ref().expect("a successor root"),
            ));
            words.push(ext_word(
                round.ood_value.as_ref().expect("an out-of-domain value"),
            ));
            push_openings(&mut words, round, false);
        }
    }
    words
}

/// One round's query openings, current or successor, block then path.
fn push_openings(words: &mut Vec<LfmWord>, round: &ChainRound<F, E>, current: bool) {
    match &round.openings {
        RoundOpenings::Base(p) => {
            if current {
                for opening in &p.current {
                    // A base value arrives as `(v, 0, 0, 0)`.
                    for v in &opening.values {
                        words.push([*v, FE::zero(), FE::zero(), FE::zero()]);
                    }
                    for node in &opening.proof.merkle_path {
                        words.push(commitment_to_digest(node));
                    }
                }
            } else {
                for opening in &p.next {
                    for v in &opening.values {
                        words.push(ext_word(v));
                    }
                    for node in &opening.proof.merkle_path {
                        words.push(commitment_to_digest(node));
                    }
                }
            }
        }
        RoundOpenings::Extension(p) => {
            let side = if current { &p.current } else { &p.next };
            for opening in side {
                for v in &opening.values {
                    words.push(ext_word(v));
                }
                for node in &opening.proof.merkle_path {
                    words.push(commitment_to_digest(node));
                }
            }
        }
    }
}

/// ★ The assembled chain executes on a proof the host accepts.
///
/// Every challenge the machine draws feeds a refusal, so this is the stream
/// comparison — see the module doc.
#[test]
fn the_chain_executes_on_a_proof_the_host_accepts() {
    for (num_vars, num_queries) in [(6usize, 3usize), (6, 5), (5, 3)] {
        let f = fixture(num_vars, num_queries, 0);
        let program = chain_program(&f.shape);
        let arena = chain_arena(&f, &f.proof);
        println!(
            "chain S={num_vars} Q={num_queries}: {} rounds, {} instructions, opening \
             permutations predicted {}",
            f.shape.rounds(),
            program.instrs.len(),
            chain_opening_perms(&f.shape)
        );
        execute(&program, &[arena], &crate::hash_pin::BLOCK_HASHER).unwrap_or_else(|e| {
            panic!("S={num_vars} Q={num_queries}: the machine refused an accepted proof: {e:?}")
        });
    }
}

/// ★ The tamper arm: three halves, five sites.
///
/// ⚠ Run at a REAL grind width, and the reason is a catch this arm made on
/// itself. At zero grind bits `check_grind` returns before reading the nonce
/// (`whir_chain.rs:131-133`), so flipping one is not a forgery at all — the
/// host accepts it, and the arm's middle half said so: "the host must reject
/// the forgery, or the machine's refusal is a refusal of something valid". A
/// version without that half would have recorded the machine refusing a proof
/// the host ACCEPTS as a soundness success, when it is a completeness bug.
#[test]
fn a_tampered_chain_cannot_execute() {
    let grind = 8u8;
    let f = fixture(6, 3, grind);
    let program = chain_program(&f.shape);

    assert!(
        execute(
            &program,
            &[chain_arena(&f, &f.proof)],
            &crate::hash_pin::BLOCK_HASHER
        )
        .is_ok(),
        "the untouched proof must execute, or the arm below proves nothing"
    );

    let host_rejects = |proof: &ChainProof<F, E>, root_bytes: &[u8; 32]| -> bool {
        let mut t = Recording::new();
        verify::<F, E, _, RpxWhir>(
            proof,
            root_bytes,
            &f.z,
            f.y,
            &f.domain,
            &config(3, grind),
            &mut t,
        )
        .is_err()
    };
    let honest_root_bytes = f.root_bytes;

    let mut sites: Vec<(&str, ChainProof<F, E>)> = Vec::new();

    // A final value: the tail's division no longer closes.
    let mut forged = f.proof.clone();
    forged.final_value += FEE::one();
    sites.push(("the final value", forged));

    // A sumcheck evaluation: the claim the group leaves moves.
    let mut forged = f.proof.clone();
    forged.rounds[0].sumcheck[0].evaluations[0] += FEE::one();
    sites.push(("a sumcheck evaluation", forged));

    // An out-of-domain value: `gamma` moves, and with it every later draw.
    let mut forged = f.proof.clone();
    if let Some(v) = forged.rounds[0].ood_value.as_mut() {
        *v += FEE::one();
    }
    sites.push(("an out-of-domain value", forged));

    // A grind nonce: the range check fails.
    let mut forged = f.proof.clone();
    forged.rounds[0].nonces.query ^= 1;
    sites.push(("a grind nonce", forged));

    // A Merkle sibling: the walk arrives somewhere else.
    let mut forged = f.proof.clone();
    match &mut forged.rounds[0].openings {
        RoundOpenings::Base(p) => p.current[0].proof.merkle_path[0][0] ^= 1,
        RoundOpenings::Extension(p) => p.current[0].proof.merkle_path[0][0] ^= 1,
    }
    sites.push(("a Merkle sibling", forged));

    for (name, forged) in &sites {
        assert!(
            host_rejects(forged, &honest_root_bytes),
            "{name}: the host must reject the forgery, or the machine's refusal is \
             a refusal of something valid"
        );
        assert!(
            execute(
                &program,
                &[chain_arena(&f, forged)],
                &crate::hash_pin::BLOCK_HASHER
            )
            .is_err(),
            "{name}: the machine must refuse the forgery"
        );
    }
    println!("chain tamper arm: {} sites, all refused", sites.len());
}

/// ★ The grind is spent where the host spends it: `3R − 1` times, and a wrong
/// nonce at any of them has no execution.
///
/// Run at a real grind width rather than zero, so the nonce search actually
/// happens and the machine's range check is exercised. Small bits, because the
/// prover's search is `2^bits` hashes.
#[test]
fn the_chain_spends_its_grinds_where_the_host_does() {
    let bits = 8u8;
    let f = fixture(6, 3, bits);
    let program = chain_program(&f.shape);
    let rounds = f.shape.rounds();

    assert!(
        execute(
            &program,
            &[chain_arena(&f, &f.proof)],
            &crate::hash_pin::BLOCK_HASHER
        )
        .is_ok(),
        "the ground proof must execute"
    );

    let mut refused = 0;
    for r in 0..rounds {
        let last = r + 1 == rounds;
        let mut spots: Vec<&str> = vec!["folding", "query"];
        if !last {
            spots.push("ood");
        }
        for spot in spots {
            let mut forged = f.proof.clone();
            match spot {
                "folding" => forged.rounds[r].nonces.folding ^= 1,
                "query" => forged.rounds[r].nonces.query ^= 1,
                _ => forged.rounds[r].nonces.ood ^= 1,
            }
            assert!(
                execute(
                    &program,
                    &[chain_arena(&f, &forged)],
                    &crate::hash_pin::BLOCK_HASHER
                )
                .is_err(),
                "round {r}, the {spot} nonce: a wrong nonce must have no execution"
            );
            refused += 1;
        }
    }
    assert_eq!(
        refused,
        3 * rounds - 1,
        "a chain spends 3R − 1 nonces; every one of them must be checked"
    );
    println!("chain grinds: {refused} nonces, all checked, at {bits} bits");
}

/// ★ The production shape, checked by arithmetic rather than by a run.
///
/// The epoch's group-0 chains are `n_stack = 25` at `k = 4`, blowup 4, `Q =
/// 112` (sh1, 2026-09-18). That shape is far past what a laptop proves, but the
/// permutation form is a function of the tree depths and block widths alone, so
/// it can be evaluated without emitting anything — and it is the one number the
/// campaign's sizing carries, so it is worth pinning against a derivation done
/// a second way.
///
/// The second derivation, by hand off `verify_weighted`'s round structure:
/// schedule `[4,4,4,4,4,4,1]`, domains 27/23/19/15/11/7/3; current depths
/// 23+19+15+11+7+3+2 = 80 and successor depths 19+15+11+7+3+2 = 57, so 137
/// parents; current leaves 2 (round 0 is 16 BASE felts) + 5x6 (48 extension
/// felts) + 1 (the 6-felt tail block) = 33 and successor leaves 5x6 + 1 = 31,
/// so 64 leaf blocks. 201 a query, 22,512 a chain.
#[test]
fn the_production_shape_reproduces_the_campaigns_permutation_count() {
    let shape = ChainShape::new(&config(112, 20), 25);
    assert_eq!(shape.schedule, vec![4, 4, 4, 4, 4, 4, 1]);
    assert_eq!(shape.domain_log, vec![27, 23, 19, 15, 11, 7, 3]);

    let parents: usize = (0..shape.rounds())
        .map(|r| shape.current_depth(r) + shape.next_depth(r).unwrap_or(0))
        .sum();
    assert_eq!(parents, 137, "Merkle parents a query");

    let perms = chain_opening_perms(&shape);
    println!("production chain S=25 k=4 Q=112: {perms} permutations, {parents} parents a query");
    assert_eq!(
        perms, 22_512,
        "the opening form at the production shape must reproduce the hand derivation"
    );
    assert_eq!(perms % shape.num_queries, 0);
    assert_eq!(perms / shape.num_queries, 201, "permutations a query");

    // The tail round's block is the one whose felt count is not a multiple of
    // eight: two extension values are six felts, and reading the capacity off
    // the VALUE count instead would make it two.
    assert_eq!(shape.current_felts(6), 6);
    assert_eq!(
        shape.current_felts(0),
        16,
        "round 0 is base: one felt a value"
    );
    assert_eq!(shape.current_felts(1), 48, "every later round is three");
}

/// A one-refusal program: hints its inputs, runs the refusal, publishes a
/// witness so the program has an output.
fn refusal_program(which: &str) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(3);
    let one = b.ext_const(&FEE::one());
    let a = b.hint_word(arena, 0).as_ext();
    let c = b.hint_word(arena, 1).as_ext();
    let d = b.hint_word(arena, 2).as_ext();
    match which {
        "ood" => super::whir_chain::emit_require_out_of_domain(&mut b, a, one),
        _ => super::whir_chain::emit_final_check(&mut b, a, c, d, one),
    }
    b.public(a.as_cell());
    let program = compile(b.finish());
    validate(&program).expect("a refusal leg must be admissible");
    program
}

fn runs(program: &LfmProgram, words: [FEE; 3]) -> bool {
    let arena: Vec<LfmWord> = words.iter().map(ext_word).collect();
    execute(program, &[arena], &crate::hash_pin::BLOCK_HASHER).is_ok()
}

/// ★ The two refusals a chain-level gate CANNOT exercise, driven directly.
///
/// Mutations CC and CD deleted each of these from the assembled chain and all
/// four chain tests stayed green — measured, not feared. The reason is the same
/// for both: the input that would trip them is one a Fiat-Shamir transcript does
/// not produce. An in-domain `z0` has negligible probability, and a weight and
/// claim that are BOTH zero is a point no honest or tampered fixture reaches.
///
/// So they are called with the inputs the protocol never will, each beside a
/// control that must execute — otherwise the refusal would be a refusal of
/// everything, which is the sibling failure.
#[test]
fn the_refusals_a_real_proof_cannot_reach() {
    let ood = refusal_program("ood");
    let zero = FEE::zero();
    let two = FEE::one() + FEE::one();
    // `z0` raised to the domain's size: one means in-domain.
    assert!(
        runs(&ood, [two, zero, zero]),
        "an out-of-domain point must execute, or the refusal below refuses everything"
    );
    assert!(
        !runs(&ood, [FEE::one(), zero, zero]),
        "an IN-domain point must have no execution: this is `OodPointInDomain`, and a \
         transcript never draws one, so nothing else in this suite can see it"
    );

    let tail = refusal_program("tail");
    // (claim, weight, final_value): the honest relation is final = claim/weight.
    assert!(
        runs(&tail, [two + two, two, two]),
        "4 / 2 == 2 must execute, or the refusal below refuses everything"
    );
    assert!(
        !runs(&tail, [two + two, two, FEE::one()]),
        "a final value that is not claim/weight must have no execution"
    );
    assert!(
        !runs(&tail, [zero, zero, FEE::one()]),
        "★ the DOUBLE ZERO: the host returns DegenerateEvaluationPoint here, and the \
         cross-multiplied form `final·weight == claim` would accept any final value. \
         This is the whole reason the tail inverts."
    );
    assert!(
        !runs(&tail, [two, zero, FEE::one()]),
        "a zero weight with a nonzero claim must have no execution either"
    );
}
