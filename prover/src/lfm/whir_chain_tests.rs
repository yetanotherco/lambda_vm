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

use core::cell::RefCell;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::fiat_shamir::transcript_hash::RpxTranscriptHash;
use math::traits::AsBytes;
use multilinear::mle::Mle;
use multilinear::whir::Domain;
use multilinear::whir_chain::{
    ChainConfig, ChainProof, GrindBits, RoundOpenings, commit, prove, verify,
};
use multilinear::whir_hash::RpxWhir;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::algebraic_commit::commitment_to_digest;
use super::builder::{Ext, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::validator::validate;
use super::whir_chain::{
    ChainShape, RoundStorage, chain_grind_perms, chain_hash_schedule, chain_opening_perms,
    chain_perms, chain_rows, chain_schedule_perms, chain_schedule_rows, chain_shape_rows,
    emit_verify_weighted, push_round_words, round_words,
};
use super::whir_poly::{emit_eq_eval, eq_eval_rows_again};
use super::whir_transcript::{SpongeEntry, SpongeHash, WhirTranscript};
use super::word::{LfmWord, ext_word};

type F = GoldilocksField;
type E = GoldilocksExtension;
type HostTranscript = DefaultTranscript<E, RpxTranscriptHash>;

/// Bytes one squeeze hands out (`default_transcript.rs:19`).
const SQUEEZE_BYTES: usize = 32;

/// Bytes one candidate consumes (`:130-132`).
const CANDIDATE_BYTES: usize = 8;

/// ★ The host's duplex, reconstructed from the calls the VERIFIER makes.
///
/// This is the second derivation the chain's schedule half is gated against,
/// and it reads nothing of the emitter. Every rule is the host's own:
///
/// - an absorb appends its bytes and invalidates the output buffer
///   (`default_transcript.rs:205-210`);
/// - a candidate is eight bytes, and refills with ONE `sample()` when fewer
///   than eight remain (`:125-134`);
/// - `sample()` hashes everything absorbed since the last one and RE-ABSORBS
///   its 32-byte digest, so the buffer afterwards is four felts and not none
///   (`:104-110`);
/// - `state()` hashes that same buffer and leaves it alone (`:236-239`).
///
/// ★ **The reconstruction is not trusted, it is CHECKED.** `shadow` is a second
/// `DefaultTranscript` fed exactly the same bytes and squeezed at exactly the
/// points this reconstruction says the host squeezes; every value the real
/// transcript hands back is compared against the one the shadow's bytes give.
/// A squeeze in the wrong place leaves different bytes in the buffer, so the
/// very next draw disagrees — which is what makes this a check and not a
/// restatement of the same belief twice.
pub(super) struct HostDuplex {
    shadow: HostTranscript,
    out: [u8; SQUEEZE_BYTES],
    out_pos: usize,
    /// Bytes absorbed since the last squeeze.
    buffered: usize,
    pub(super) hashes: Vec<SpongeHash>,
}

impl HostDuplex {
    fn new() -> Self {
        Self {
            shadow: HostTranscript::new(&[]),
            out: [0u8; SQUEEZE_BYTES],
            out_pos: SQUEEZE_BYTES,
            buffered: 0,
            hashes: Vec::new(),
        }
    }

    /// Felts in the buffer. A hash reads whole 8-byte groups, so a buffer that
    /// is not a multiple of eight would mean a value straddling two felts —
    /// the alignment the emitter refuses by construction, asserted here rather
    /// than rounded away.
    fn felts(&self) -> usize {
        assert_eq!(
            self.buffered % CANDIDATE_BYTES,
            0,
            "the chain absorbs whole felts: {} bytes buffered",
            self.buffered
        );
        self.buffered / CANDIDATE_BYTES
    }

    fn absorb(&mut self, bytes: &[u8]) {
        self.shadow.append_bytes(bytes);
        self.buffered += bytes.len();
        self.out_pos = SQUEEZE_BYTES;
    }

    /// `append_field_element` absorbs exactly the bytes the element streams
    /// (`:193-206`), so the shadow is given those same bytes and their count is
    /// MEASURED rather than assumed to be three felts.
    fn absorb_element(&mut self, element: &FEE) {
        let mut bytes = Vec::new();
        element.stream_bytes(&mut |chunk| bytes.extend_from_slice(chunk));
        self.absorb(&bytes);
    }

    fn state(&mut self) {
        let felts = self.felts();
        self.hashes.push(SpongeHash::State(felts));
    }

    fn candidate(&mut self) -> u64 {
        if self.out_pos + CANDIDATE_BYTES > SQUEEZE_BYTES {
            let felts = self.felts();
            self.hashes.push(SpongeHash::Squeeze(felts));
            self.out = self.shadow.sample();
            self.buffered = SQUEEZE_BYTES;
            self.out_pos = 0;
        }
        let mut bytes = [0u8; CANDIDATE_BYTES];
        bytes.copy_from_slice(&self.out[self.out_pos..self.out_pos + CANDIDATE_BYTES]);
        self.out_pos += CANDIDATE_BYTES;
        u64::from_be_bytes(bytes)
    }
}

/// A transcript that records what it hands out, delegating everything.
///
/// The chain's challenges are drawn interleaved with its sumchecks and never
/// returned, and re-deriving them would mirror the host rather than read it.
///
/// It also carries [`HostDuplex`], which reconstructs the host's hash schedule
/// from these same calls. `state()` takes `&self`, so the reconstruction lives
/// behind a `RefCell`.
pub(super) struct Recording {
    inner: HostTranscript,
    pub(super) sampled: Vec<FEE>,
    pub(super) drawn_u64: Vec<u64>,
    pub(super) duplex: RefCell<HostDuplex>,
}

impl Recording {
    pub(super) fn new() -> Self {
        Self {
            inner: HostTranscript::new(&[]),
            sampled: Vec::new(),
            drawn_u64: Vec::new(),
            duplex: RefCell::new(HostDuplex::new()),
        }
    }
}

/// ★ The recorder's hash, NAMED — because `multi_verify` will not take a
/// transcript that does not name one.
///
/// This is not a convenience: the bound exists so that a keccak transcript
/// cannot be passed under an RPX `H`, which is the half-configured arm that ran
/// undetected for a day. It is TRUE here rather than asserted: `Recording`'s
/// inner transcript is `DefaultTranscript<E, RpxTranscriptHash>` (the alias at
/// the top of this file), so a recorder handed to an RPX verify runs the same
/// sponge the verify does. Changing that alias must change this line with it.
impl crypto::fiat_shamir::transcript_hash::HasTranscriptHash for Recording {
    type Hash = RpxTranscriptHash;
}

impl IsTranscript<E> for Recording {
    fn append_field_element(&mut self, element: &FEE) {
        self.duplex.borrow_mut().absorb_element(element);
        self.inner.append_field_element(element);
    }
    fn append_bytes(&mut self, new_bytes: &[u8]) {
        self.duplex.borrow_mut().absorb(new_bytes);
        self.inner.append_bytes(new_bytes);
    }
    fn state(&self) -> [u8; 32] {
        self.duplex.borrow_mut().state();
        self.inner.state()
    }
    fn sample_field_element(&mut self) -> FEE {
        // Three candidates, one a coordinate and in coordinate order
        // (`extensions_goldilocks.rs:574-581`); each coordinate IS its
        // candidate, because `candidate_in_range` accepted it.
        let mine: [FE; 3] = {
            let mut duplex = self.duplex.borrow_mut();
            core::array::from_fn(|_| FE::from(duplex.candidate()))
        };
        let drawn = self.inner.sample_field_element();
        assert_eq!(
            *drawn.value(),
            mine,
            "the reconstructed duplex must reproduce the host's challenge — a squeeze in the \
             wrong place is what this catches"
        );
        self.sampled.push(drawn);
        drawn
    }
    fn sample_u64(&mut self, upper_bound: u64) -> u64 {
        let mine = self.duplex.borrow_mut().candidate();
        let drawn = self.inner.sample_u64(upper_bound);
        assert_eq!(
            mine % upper_bound,
            drawn,
            "the reconstructed duplex must reproduce the host's query position"
        );
        self.drawn_u64.push(drawn);
        drawn
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
        format: multilinear::whir_chain::ChainFormat::DEFAULT,
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
            at += round_words(shape, r);
        }
        Self {
            shape: shape.clone(),
            round_at,
            total: at,
        }
    }
}

/// Builds the program for one shape, publishing every extension challenge the
/// machine draws in the order it draws them.
pub(super) fn chain_program(shape: &ChainShape) -> LfmProgram {
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
    let storage = RoundStorage::hint(&mut b, arena, layout.round_at[0], shape);
    let (current_openings, next_openings) = storage.openings();
    let wires = storage.wires(&current_openings, &next_openings);

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

    push_round_words(&mut words, shape, proof);
    words
}

/// ★ The assembled chain executes on a proof the host accepts.
///
/// Every challenge the machine draws feeds a refusal, so this is the stream
/// comparison — see the module doc.
#[test]
fn the_chain_executes_on_a_proof_the_host_accepts() {
    // ★ `S = 9` is the three-round shape: the only one here carrying a round
    // that is neither the first nor the last, and therefore the only one where
    // a successor block is opened and then re-opened as a current one.
    for (num_vars, num_queries) in [(6usize, 3usize), (6, 5), (5, 3), (9, 3)] {
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

/// The shapes the cost forms are gated at: `(num_vars, num_queries, grind)`.
///
/// Grind 0 is not a smaller version of grind 8. With no query grind the query
/// phase is entered with a candidate still in hand, which is the `avail > 0`
/// branch of the schedule — a branch the production shape never takes, and one
/// a suite run only at a real grind width would never reach. `S = 9` is the
/// only shape here with three rounds, so it is the only one carrying a round
/// that is neither the first nor the last.
const COST_SHAPES: [(usize, usize, u8); 5] =
    [(6, 3, 0), (6, 5, 0), (5, 3, 0), (6, 3, 8), (9, 3, 8)];

pub(super) fn count_rows(program: &LfmProgram, want: fn(&super::instr::Instr) -> bool) -> usize {
    program.instrs.iter().filter(|instr| want(instr)).count()
}

pub(super) fn const_rows(program: &LfmProgram) -> usize {
    count_rows(program, |i| matches!(i, super::instr::Instr::Const { .. }))
}

/// `LFM_HASH` invocations: one per sponge permutation, whether it is a leaf's
/// duplex block or a Merkle parent's compress.
pub(super) fn perm_rows(program: &LfmProgram) -> usize {
    count_rows(program, |i| matches!(i, super::instr::Instr::Hash { .. }))
}

pub(super) fn hint_rows(program: &LfmProgram) -> usize {
    count_rows(program, |i| matches!(i, super::instr::Instr::Hint { .. }))
}

/// Rows the chain PROGRAM carries that are not the chain's own cost.
///
/// Derived from the test's own [`Layout`] and not from the program's
/// histogram, so a form row that turned into a hint could not hide inside the
/// subtraction: one `Hint` per arena word the layout declares, the caller's
/// single `Unpack` of the root (shared by every query against it, which is why
/// the opening form does not charge it), and the weight closure —
/// `emit_eq_eval(z, alphas)` is the caller's `W`, and `chain_fixed_rows`
/// deliberately counts only the out-of-domain `eq`s. The closure's interned `1`
/// is a `Const` row and is subtracted with the other constants, so the
/// second-leg form is the right one here.
fn chain_plumbing(shape: &ChainShape) -> usize {
    Layout::new(shape).total as usize + 1 + eq_eval_rows_again(shape.num_vars)
}

/// ★ GATE ONE for the schedule: the emitter's hash schedule is the HOST's.
///
/// `chain_hash_schedule` is derived from the round structure — the absorbs, the
/// draws, the `state()` reads, the candidates a squeeze hands out. The host's
/// is reconstructed by [`HostDuplex`] from the calls a real `whir_chain::verify`
/// makes on a real proof, and checked against the values that verify received.
/// Two derivations that share no code, compared event for event and felt count
/// for felt count.
///
/// This is the half the previous instance could not write, and the reason its
/// row form was not quotable: a sponge's cost is what its BUFFER holds at each
/// hash, which is a running quantity and not a shape.
#[test]
fn the_schedule_is_the_host_transcripts() {
    for (num_vars, num_queries, grind) in COST_SHAPES {
        let f = fixture(num_vars, num_queries, grind);
        let host = f.recorded.duplex.borrow().hashes.clone();
        let mine = chain_hash_schedule(&f.shape, SpongeEntry::fresh());

        let squeezes = mine
            .iter()
            .filter(|h| matches!(h, SpongeHash::Squeeze(_)))
            .count();
        let states = mine.len() - squeezes;
        println!(
            "schedule S={num_vars} Q={num_queries} grind={grind}: {squeezes} squeezes, \
             {states} state reads, {} rows, {} permutations",
            chain_schedule_rows(&f.shape, SpongeEntry::fresh()),
            chain_schedule_perms(&f.shape, SpongeEntry::fresh()),
        );

        assert_eq!(
            mine.len(),
            host.len(),
            "S={num_vars} Q={num_queries} grind={grind}: the host performs {} transcript \
             hashes and the form says {}",
            host.len(),
            mine.len()
        );
        for (i, (derived, observed)) in mine.iter().zip(&host).enumerate() {
            assert_eq!(
                derived,
                observed,
                "S={num_vars} Q={num_queries} grind={grind}: hash {i} of {} — the form says \
                 {derived:?} and the host's own transcript did {observed:?}",
                host.len()
            );
        }

        // A state read happens exactly where a grind is spent, and nowhere
        // else: `3R − 1` at a real width, none at zero bits.
        let expected_states = if grind == 0 {
            0
        } else {
            3 * f.shape.rounds() - 1
        };
        assert_eq!(
            states, expected_states,
            "state reads are one per grind check spent"
        );
    }
}

/// ★ GATE TWO: the chain's rows and permutations against the program it emits.
///
/// Pinned separately and on purpose (the opening leg's reason, one level up): a
/// form that moved work between the sponge and the arithmetic at constant total
/// fails one of the two rather than neither. The permutation form has THREE
/// terms — the query phase's openings, two per grind check, and the schedule's
/// own — and only the first of them was pinned before this test.
#[test]
fn the_chain_emits_its_closed_form() {
    for (num_vars, num_queries, grind) in COST_SHAPES {
        let f = fixture(num_vars, num_queries, grind);
        let program = chain_program(&f.shape);
        let entry = SpongeEntry::fresh();

        let hints = hint_rows(&program);
        assert_eq!(
            hints,
            Layout::new(&f.shape).total as usize,
            "every arena word is hinted exactly once — the plumbing subtraction below is only \
             honest while this holds"
        );

        let consts = const_rows(&program);
        let measured = program.instrs.len() - consts - chain_plumbing(&f.shape);
        let predicted = chain_rows(&f.shape, entry);
        let perms = perm_rows(&program);
        let predicted_perms = chain_perms(&f.shape, entry);

        println!(
            "chain S={num_vars} Q={num_queries} grind={grind}: {measured} rows \
             ({} shape + {} schedule predicted {predicted}); {perms} permutations \
             ({} openings + {} grind + {} schedule predicted {predicted_perms}); \
             {consts} constants, {hints} hints, {} instructions",
            chain_shape_rows(&f.shape),
            chain_schedule_rows(&f.shape, entry),
            chain_opening_perms(&f.shape),
            chain_grind_perms(&f.shape),
            chain_schedule_perms(&f.shape, entry),
            program.instrs.len(),
        );

        assert_eq!(
            measured, predicted,
            "S={num_vars} Q={num_queries} grind={grind}: rows"
        );
        assert_eq!(
            perms, predicted_perms,
            "S={num_vars} Q={num_queries} grind={grind}: permutations"
        );
    }
}

/// ★ What a chain is ENTERED with reaches its first SQUEEZE, and stops there.
///
/// The property the census rests on: a chain inside an assembled verifier finds
/// the sponge holding whatever the statement and the tables before it left, and
/// that could in principle change every later buffer length. It does not — but
/// the boundary is not where this test first claimed it was.
///
/// ⚠ It was written as "the first HASH carries the entering buffer and nothing
/// after it does", and it failed: a `state()` hashes the buffer WITHOUT
/// clearing it (`default_transcript.rs:236-239`), so a grind's state read is a
/// hash that passes the entering felts straight through to the squeeze behind
/// it. The boundary is the first SQUEEZE, which is the first event that resets
/// the buffer to the digest. Recorded rather than quietly corrected, because
/// the wrong version is the one a reader would assume.
///
/// The entering `out_pos` is never read at all: the first draw of the first
/// round always follows an absorb, and an absorb invalidates the output buffer.
#[test]
fn what_a_chain_is_entered_with_reaches_its_first_squeeze_and_stops() {
    let shape = ChainShape::new(&config(4, 8), 6);
    let fresh = chain_hash_schedule(&shape, SpongeEntry::fresh());
    let first_squeeze = fresh
        .iter()
        .position(|h| matches!(h, SpongeHash::Squeeze(_)))
        .expect("a chain squeezes");

    for (buffered, out_pos) in [(0, 0), (3, 2), (7, 4), (12, 1)] {
        let entry = SpongeEntry {
            buffered_felts: buffered,
            out_pos,
        };
        let entered = chain_hash_schedule(&shape, entry);
        assert_eq!(entered.len(), fresh.len(), "the event count cannot move");
        for i in 0..=first_squeeze {
            assert_eq!(
                entered[i].felts(),
                fresh[i].felts() + buffered,
                "hash {i} is at or before the first squeeze, so it carries the entering buffer"
            );
            assert_eq!(
                core::mem::discriminant(&entered[i]),
                core::mem::discriminant(&fresh[i]),
                "hash {i} is the same KIND either way"
            );
        }
        assert_eq!(
            &entered[first_squeeze + 1..],
            &fresh[first_squeeze + 1..],
            "every hash after the first squeeze is identical — that squeeze reset the buffer \
             to the digest"
        );
    }

    // The entering `out_pos` changes nothing: at one buffer length, every
    // position gives the same schedule.
    for out_pos in 0..=4 {
        assert_eq!(
            chain_hash_schedule(
                &shape,
                SpongeEntry {
                    buffered_felts: 5,
                    out_pos
                }
            ),
            chain_hash_schedule(
                &shape,
                SpongeEntry {
                    buffered_felts: 5,
                    out_pos: 0
                }
            ),
            "a chain never draws before it absorbs, so what is in hand on entry is dropped"
        );
    }
}

/// ★ The production chain's cost, whole: the number the epoch census adds up.
///
/// The forms are gated at [`COST_SHAPES`]; this evaluates them at the shape the
/// campaign runs and pins the answer, so a change to any term has to be
/// restated here before it can be quoted. Every number was derived by hand in
/// the design note before this test was written, which is what makes the first
/// run of it a measurement rather than a transcription.
///
/// ⚠ The sizing note's per-chain figures are NOT these. Its 22,830 permutations
/// predate both the grind term and the schedule term; the 318 it implies for the
/// transcript is 42 above the 276 the schedule actually costs, and was never
/// re-derived. Quote these.
#[test]
fn the_production_chain_costs_what_the_census_quotes() {
    // `multilinear_prove.rs:93`: blowup 2, fold 4, 128 bits, uniform 20-bit
    // grinds — the config the block is proven under.
    let shape = ChainShape::new(&config(112, 20), 25);
    let entry = SpongeEntry::fresh();
    let schedule = chain_hash_schedule(&shape, entry);
    let squeezes = schedule
        .iter()
        .filter(|h| matches!(h, SpongeHash::Squeeze(_)))
        .count();

    assert_eq!(squeezes, 233, "squeezes a chain");
    assert_eq!(schedule.len() - squeezes, 20, "state reads = 3R − 1 grinds");
    assert_eq!(chain_schedule_rows(&shape, entry), 836, "schedule rows");
    assert_eq!(chain_schedule_perms(&shape, entry), 276, "schedule perms");
    assert_eq!(chain_grind_perms(&shape), 40, "two permutations a grind");
    assert_eq!(chain_shape_rows(&shape), 184_673, "shape rows");

    println!(
        "production chain S=25 k=4 Q=112 grind=20: {} rows, {} permutations",
        chain_rows(&shape, entry),
        chain_perms(&shape, entry)
    );
    assert_eq!(chain_rows(&shape, entry), 185_509, "rows a chain");
    assert_eq!(chain_perms(&shape, entry), 22_828, "permutations a chain");
}

/// ★ The chain's F1 AT THE PRODUCTION SHAPE, and the constants the campaign's
/// figure does not carry.
///
/// `the_chain_emits_its_closed_form` runs at five to nine variables; the number
/// the campaign quotes — 185,509 rows a chain — was derived by EVALUATING the
/// form at `S = 25, k = 4, Q = 112`, not by emitting the program. This emits it.
/// A program costs nothing to build but its own construction: `chain_program`
/// takes the SHAPE alone, so no proof, no commitment and no ELF are involved.
///
/// ⚠ It also measures what `chain_rows` does NOT carry. The F1 above subtracts
/// `const_rows` before comparing, so `chain_rows` is a form over rows that are
/// neither `LFM_CONST` nor plumbing — and V1e's per-table form INCLUDES its
/// constants. Adding a chain figure to a table figure without saying which
/// convention the sum is in is the mistake this number exists to prevent.
///
/// `#[ignore]`d because it builds a program of a few hundred thousand
/// instructions, which is a second or two and a few hundred megabytes — fine on
/// a laptop, and not something every `cargo test` should pay.
#[test]
#[ignore = "builds a production-shape chain program; run it when the census needs the number"]
fn the_production_chain_emits_its_closed_form() {
    let shape = ChainShape::new(&config(112, 20), 25);
    assert_eq!(
        shape.schedule,
        vec![4, 4, 4, 4, 4, 4, 1],
        "the production schedule"
    );

    let program = chain_program(&shape);
    let entry = SpongeEntry::fresh();
    let consts = const_rows(&program);
    let hints = hint_rows(&program);
    let plumbing = chain_plumbing(&shape);
    let measured = program.instrs.len() - consts - plumbing;
    let predicted = chain_rows(&shape, entry);
    let perms = perm_rows(&program);
    let predicted_perms = chain_perms(&shape, entry);

    println!(
        "PRODUCTION chain S=25 k=4 Q=112 grind=20: {measured} rows against {predicted} predicted \
         ({} shape + {} schedule); {perms} permutations against {predicted_perms} \
         ({} openings + {} grind + {} schedule); {consts} CONSTANTS, {hints} hints, \
         {} instructions whole",
        chain_shape_rows(&shape),
        chain_schedule_rows(&shape, entry),
        chain_opening_perms(&shape),
        chain_grind_perms(&shape),
        chain_schedule_perms(&shape, entry),
        program.instrs.len(),
    );
    println!(
        "  ⇒ the const-free figure the campaign quotes is {measured}; the same chain's whole \
         instruction count LESS its hinted arena is {}",
        measured + consts
    );

    assert_eq!(
        hints,
        Layout::new(&shape).total as usize,
        "every arena word hinted once"
    );
    assert_eq!(measured, predicted, "the production shape's rows");
    assert_eq!(
        perms, predicted_perms,
        "the production shape's permutations"
    );
}

/// ⛔ THE GENESIS THRESHOLD'S BUDGET, ASSERTED WHERE IT CAN BE COMPUTED.
///
/// `crate::continuation::PREPARED_LEG_ROWS` is PART 2's term: the rows the
/// stacked chain costs ONCE, which the candidate set's total savings must
/// exceed before any page is carried. It is a CONSTANT there rather than a
/// call, because that module sits below `crate::lfm` — a routing rule the
/// prover, the verifier and the emitter must all agree on cannot depend on the
/// emitter's row accounting. This is the assertion that pays for the constant,
/// in the one place the form exists.
///
/// ⚠ IT IS NOT WHAT A PAGE IS CHARGED. What one more carried page adds to the
/// leg is `continuation::marginal_stacked_rows`, a form evaluated at the run's
/// own bracket whose pin lives in `whir_stacked_tests`. The two terms are separate because the
/// costs are: one chain, however many pages ride it.
///
/// ★ WHAT THE CONSTANT IS: 175,066 rows, the chain over a stacked family
/// polynomial at 24 variables. The block's stack is THREE columns of 2^18 — 20
/// variables — so it is charged at a figure larger than it can cost, which is
/// the conservative direction: a page must be worth more than the stack could
/// possibly cost before it joins.
///
/// ⚠ CONFIGURATION IS PART OF THE NUMBER. Chain rows move with blowup, folding
/// and the query count, so this asserts at the shape the campaign's production
/// figures are quoted at — the same `config(112, 20)` the chain's own F1 above
/// uses. A posture change reddens this rather than silently retuning a routing
/// rule nobody is looking at.
///
/// It asserts a BAND and not an equality: the constant's job is to be larger
/// than the 20-variable stack and no larger than the 24-variable one it was
/// read from. An equality would redden on any change to the schedule, which is
/// a different finding from "the routing rule has drifted".
#[test]
fn the_genesis_threshold_budget_is_in_band_at_the_production_shape() {
    let at_20 = chain_shape_rows(&ChainShape::new(&config(112, 20), 20));
    let at_24 = chain_shape_rows(&ChainShape::new(&config(112, 20), 24));
    let budget = crate::continuation::PREPARED_LEG_ROWS;
    println!(
        "GENESIS BUDGET: {budget} rows against a chain of {at_20} at 20 variables and \
         {at_24} at 24, Q=112 grind=20 blowup=2 fold=4"
    );
    assert!(
        at_20 < at_24,
        "a stack of fewer variables must cost fewer chain rows, or the budget's \
         conservatism argument does not hold ({at_20} at 20, {at_24} at 24)"
    );
    assert!(
        budget >= at_20,
        "the threshold charges {budget} rows for a stack that costs {at_20} at the \
         block's 20 variables: pages would be left sparse that the opening could carry"
    );
    assert!(
        budget <= at_24,
        "the threshold charges {budget} rows for a stack that costs at most {at_24}: \
         the budget has drifted above the cost it stands for"
    );
}

/// ★ AND THE ROUTING DECISION DOES NOT SIT NEAR THAT BAND.
///
/// The block's three dense pages cost 2,100,474, 4,127,238 and 4,020,858 rows
/// by the sparse form, and its 27 zero pages cost 18 each. Whatever the chain
/// figure moves to within any plausible posture, the same three pages are
/// selected — which is the argument that this is a routing rule rather than a
/// tuning knob, made against the cost form rather than against the constant.
#[test]
fn the_blocks_genesis_routing_is_insensitive_to_the_chain_figure() {
    let cheapest_dense = 2_100_474usize;
    let dearest_sparse = 18usize;
    for stack_vars in 18..=25 {
        let rows = chain_shape_rows(&ChainShape::new(&config(112, 20), stack_vars));
        assert!(
            dearest_sparse <= rows && rows < cheapest_dense,
            "at {stack_vars} stacked variables the chain costs {rows} rows, which falls \
             outside ({dearest_sparse}, {cheapest_dense}) — the block's selection would \
             change with the posture and the pre-registered three pages are no longer \
             a property of the run"
        );
    }
}

/// ⛔⛔ THE THRESHOLD AND THE SPARSE CAP MUST NOT DISAGREE, OR THE PROGRAM
/// CANNOT BE BUILT AT ALL.
///
/// Two independent rules decide what happens to a genesis page. The two-part
/// routing rule in `crate::continuation` decides whether the PREPARED OPENING
/// carries it; [`super::preprocessed::MAX_SPARSE_INIT_ENTRIES`] decides whether
/// the sparse leg is willing to EMIT it, and refuses above the cap because a
/// column that dense has no cheap closed form.
///
/// If the threshold ever left a page sparse that the cap then refused, the page
/// would have no route at all: too dense to emit, not dense enough to stack,
/// and the emit-time refusal would fire on a program nobody could fix by
/// changing either constant alone. The two rules must therefore overlap, with
/// the threshold strictly the tighter one.
///
/// ⛔⛔ AND THE BOUND IS RE-DERIVED FROM THE LIVE RULE, NOT CARRIED OVER. Under
/// the retired single-page rule the densest page left sparse carried 9,724
/// entries; under the two-part rule it carries 9,730, because a page is now
/// charged its own marginal on top of the chain. ⚠ BOTH ARE `<= 60,000`, so a
/// test left at the old number stays GREEN while measuring a rule that no
/// longer exists — which is the only reason this is worth saying twice.
/// [`crate::continuation::densest_sparse_entries`] is where the quantity now
/// lives, so it moves when the rule does.
///
/// ★ THIS IS THE STATE THE BLOCK WAS ACTUALLY IN. V1j's block bundle arm
/// refused at that cap — "these preprocessed columns carry 116692 nonzero
/// entries ... the cap is 60000 entries" for page `0x0` — because the prepared
/// route did not exist yet and every genesis page went to the sparse leg. Once
/// the threshold routes the dense pages to the opening, no page reaching the
/// sparse leg can be within six times the cap, and ⛔ THE CAP SHOULD NEVER FIRE
/// AGAIN. A refusal from it after this lands is not a page that needs a bigger
/// cap; it is these two constants having drifted apart.
#[test]
fn every_page_the_threshold_leaves_sparse_is_one_the_sparse_leg_will_emit() {
    use crate::continuation::{
        BLOCK_STACK_VARS, PAGE_NUM_VARS, PREPARED_LEG_ROWS, chain_is_paid, densest_sparse_entries,
        fixed_stack_vars, marginal_stacked_rows, page_savings, sparse_leg_rows,
    };
    let num_vars = PAGE_NUM_VARS;
    let cap = super::preprocessed::MAX_SPARSE_INIT_ENTRIES;
    // ⚠ AT THE BLOCK'S BRACKET, WHICH IS NOW PART OF THE QUESTION. The bound
    // grows with the run's genesis page count — one entry per `num_vars` rows
    // of marginal — so a bound quoted without its bracket is a bound for one
    // run. The sweep below covers the rest.
    let marginal = marginal_stacked_rows(num_vars, BLOCK_STACK_VARS);
    let densest_sparse = densest_sparse_entries(num_vars, BLOCK_STACK_VARS);

    // ⛔ BOTH CONTRASTS WITH THE RETIRED RULE, because they are different
    // quantities and both have been quoted. Its BREAK-EVEN (the least S it
    // carried) was 9,725 against this rule's 9,732 at the block's bracket; its
    // DENSEST SPARSE page carried 9,724 against this rule's 9,731. `<= 60,000`
    // is true of all four, which is exactly why a test left at either old
    // number stays green while measuring a rule that no longer exists.
    let retired_break_even = (PREPARED_LEG_ROWS - num_vars) / num_vars + 1;
    println!(
        "ROUTE OVERLAP: at the block's bracket {BLOCK_STACK_VARS} (marginal {marginal}) \
         the densest page the rule can leave sparse carries {densest_sparse} entries \
         and the least it carries is {}, against a sparse-leg cap of {cap}. A LONE \
         page's bracket gives {}; the retired single-page rule's pair was {} and \
         {retired_break_even}.",
        densest_sparse + 1,
        densest_sparse_entries(num_vars, fixed_stack_vars(num_vars, 1)),
        retired_break_even - 1,
    );
    assert_eq!(retired_break_even, 9_725);
    assert_eq!(
        densest_sparse, 9_731,
        "the bound the cap must clear is the two-part rule's AT THE BLOCK'S BRACKET; a \
         lone page's bracket gives 9,730 and the retired single-page rule's pair was \
         9,724 / 9,725 — all four clear the cap, which is exactly why this must be \
         re-derived rather than re-read"
    );

    // ⚠ EXACT, BOTH WAYS — a bound asserted only from above could be any number
    // larger than the truth. The page at the bound must route sparse and the one
    // above it must not, or `densest_sparse_entries` is inverting the wrong form.
    assert!(!chain_is_paid(page_savings(
        num_vars,
        BLOCK_STACK_VARS,
        densest_sparse
    )));
    assert!(chain_is_paid(page_savings(
        num_vars,
        BLOCK_STACK_VARS,
        densest_sparse + 1
    )));

    assert!(
        densest_sparse <= cap,
        "a page carrying {densest_sparse} entries routes SPARSE and is then REFUSED by \
         the {cap}-entry cap: it has no route at all, and neither constant can be fixed \
         without the other"
    );

    // ⛔ AND THE MARGIN MUST NOT CLOSE WHEN THE LITERAL MOVES. The marginal is
    // a form of the run's own bracket, and the bound rises one entry per
    // `num_vars` rows as a run stands taller. The overlap is asserted over every
    // marginal up to four orders past the block's, which is where the
    // arithmetic says it would finally close:
    // `(PREPARED_LEG_ROWS + m - num_vars) / num_vars <= cap` fails above
    // `m = 904,952`. ⚠ A run would need more genesis pages than there are
    // addresses to reach that, so the sweep is deliberately far past anything
    // reachable.
    let mut m = marginal;
    while m <= 900_000 {
        let bound = (PREPARED_LEG_ROWS + m - num_vars) / num_vars;
        assert!(
            bound <= cap,
            "at marginal {m} the rule can leave a page of {bound} entries sparse, and \
             the sparse leg refuses above {cap}"
        );
        // The sparse leg it would then emit must be a leg, not a claim.
        assert!(sparse_leg_rows(num_vars, bound) > 0);
        m = (m * 2).max(m + 1);
    }
    // ⚠ AND THE CHECK CAN FAIL: one marginal past that point must breach the
    // cap, or the sweep above proves nothing about where the margin closes.
    assert!((PREPARED_LEG_ROWS + 1_000_000 - num_vars) / num_vars > cap);
}

/// ⛔⛔ A LOWER BOUND ON PART 1's MARGINAL, READ OFF THE EMITTER — AND
/// DELIBERATELY NOT A PIN.
///
/// `crate::continuation::marginal_stacked_rows` is a FORM whose last term is a
/// BOUND, and its pin — the form read against the differenced
/// `stacked_verify_cost` — lives in `whir_stacked_tests` and belongs to the lane
/// that owns that cost form.
///
/// ⛔ THIS TEST MUST NOT ASSERT THAT EQUALITY, and the reason is the whole
/// point of a pin. [`super::whir_stacked::weight_at_rows`] is a PARTIAL view:
/// it carries the `eq` and the prefix indicators, while `stacked_verify_cost`
/// pays three more terms per column outside it — and a fourth, an absorb into a
/// THREADED sponge whose row cost depends on where the previous columns left
/// the buffer. A second equality here, against the partial form, would
/// CONTRADICT the real pin the day it lands, and two pins on one constant is
/// precisely the duplication a pin exists to prevent.
///
/// ★ WHAT IT DOES ASSERT, and it is worth having: the two terms
/// `weight_at_rows` DOES account for, differenced across one more page inside
/// ONE power-of-two bracket so the stack height does not move — `eq` 90 plus
/// two indicators of six — and that the literal is at least that plus one. An
/// inequality cannot conflict with a measurement.
///
/// ⚠ THE `+ 1` IS THE SHARED `Sub`, AND DIFFERENCING IS WHAT SHOWS IT IS NOT A
/// MARGINAL. `weight_at_rows` emits one per prefix POSITION any column reads as
/// a zero bit, which is per-POLYNOMIAL: across 29 and 30 pages at one height
/// the term contributes ZERO, which is why `billed` is 102 and not 103. The
/// literal charges it anyway, so `>= billed + 1` is exactly the statement that
/// the literal contains every term this form can see.
#[test]
fn the_marginal_is_at_least_what_the_weight_term_bills() {
    use super::whir_stacked::weight_at_rows;
    use crate::continuation::{
        BLOCK_STACK_VARS, MAX_SPONGE_MARGINAL, PAGE_NUM_VARS, PAGE_PREPROCESSED_COLUMNS,
        STACKED_ROWS_PER_COLUMN, marginal_stacked_rows,
    };
    let marginal = marginal_stacked_rows(PAGE_NUM_VARS, BLOCK_STACK_VARS);

    // A page's two preprocessed columns settle at ONE point — its table's — so
    // the groups are the pages.
    let layout_for = |pages: usize| {
        let columns = pages * PAGE_PREPROCESSED_COLUMNS;
        let layout = stark::multilinear_table::global_layout(&[(columns, PAGE_NUM_VARS)])
            .expect("a stack of whole pages");
        let group_of: Vec<usize> = (0..columns)
            .map(|column| column / PAGE_PREPROCESSED_COLUMNS)
            .collect();
        (layout, group_of)
    };

    // 29 and 30 pages both stand at the block's bracket, and in ONE polynomial
    // — or the difference would span two chains.
    let (at_29, groups_29) = layout_for(29);
    let (at_30, groups_30) = layout_for(30);
    assert_eq!(at_29.n_stack(), BLOCK_STACK_VARS);
    assert_eq!(at_30.n_stack(), BLOCK_STACK_VARS);
    assert_eq!(at_30.num_polys(), 1, "or the difference spans two chains");

    let rows_29 = weight_at_rows(&at_29, 0, &groups_29);
    let rows_30 = weight_at_rows(&at_30, 0, &groups_30);
    let billed = rows_30 - rows_29;
    println!(
        "MARGINAL: the weight closure alone bills {billed} rows for the thirtieth page \
         ({rows_29} -> {rows_30} at {} variables); the form charges {marginal}",
        at_30.n_stack()
    );

    // The two terms this form accounts for, to the row.
    assert_eq!(
        billed,
        5 * PAGE_NUM_VARS + PAGE_PREPROCESSED_COLUMNS * (BLOCK_STACK_VARS - PAGE_NUM_VARS)
    );
    assert_eq!(billed, 102);

    // ⛔ THE GAP, DECOMPOSED — a statement about which terms the form CONTAINS,
    // never about what the stack bills. Everything `weight_at_rows` cannot see
    // is the per-column rows and the sponge bound, and the shared `Sub` is in
    // neither number: it is per-POLYNOMIAL and differences to zero.
    assert_eq!(
        marginal - billed,
        PAGE_PREPROCESSED_COLUMNS * STACKED_ROWS_PER_COLUMN + MAX_SPONGE_MARGINAL,
        "the form charges {marginal} where the weight closure bills {billed}; the gap \
         must be exactly the per-column rows plus the sponge bound"
    );
    assert!(
        marginal > billed,
        "the form charges {marginal} rows for a page the weight closure alone bills \
         {billed} for: it would be below a term it must contain"
    );
}

/// ⚠ A LIMIT OF PART 2's SINGLE CHAIN TERM, STATED WHERE IT CAN BE MEASURED.
///
/// [`crate::continuation::PREPARED_LEG_ROWS`] prices ONE chain, and one chain
/// is what the stack costs while its columns fit one polynomial.
/// `global_layout` caps a stack at [`stark::multilinear_table::MAX_STACK_VARS`]
/// and spills the rest into another polynomial, each of which runs its own
/// chain — so past that width part 2 is weighing a set against a fraction of
/// what carrying it costs.
///
/// ⛔ THIS IS A PRICING LIMIT AND NOT A SOUNDNESS ONE. All three parties still
/// reach the same set from the same data; the rule would simply buy pages whose
/// keep it had under-quoted. It is recorded rather than guarded because the
/// block sits 21x under it and no run this prover has seen comes near — and if
/// one ever does, the fix is a chain term that counts polynomials, not a bigger
/// constant.
#[test]
fn the_single_chain_term_prices_a_stack_of_at_most_sixty_four_pages() {
    use crate::continuation::{PAGE_NUM_VARS, PAGE_PREPROCESSED_COLUMNS};
    let per_page = PAGE_PREPROCESSED_COLUMNS;
    let max_stack_vars = stark::multilinear_table::MAX_STACK_VARS;

    let polys_at = |pages: usize| {
        stark::multilinear_table::global_layout(&[(pages * per_page, PAGE_NUM_VARS)])
            .expect("a stack of whole pages")
            .num_polys()
    };
    // `2 * pages * 2^18` cells must fit `2^25`.
    let widest = 1 << (max_stack_vars - PAGE_NUM_VARS - 1);
    assert_eq!(widest, 64);
    assert_eq!(polys_at(widest), 1, "the last width that is one chain");
    assert!(
        polys_at(widest + 1) > 1,
        "one page past it must spill, or this boundary is not where it is claimed"
    );
    // The block carries three.
    assert_eq!(polys_at(3), 1);
}
