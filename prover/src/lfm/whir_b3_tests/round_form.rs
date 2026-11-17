//! ★ The in-guest verifier of the WHIR round form under 7-round BLAKE3,
//! against the host's `whir_chain::verify` on the same proofs.
//!
//! Under `RoundForm::Whir` a round that commits a successor opens its current
//! tree alone, at the round's own count of queries; each query's fold value
//! `v_j` becomes the claim `f_{r+1}(pow(y_j)) = v_j`, `y_j` the folded domain's
//! point at the drawn position, weighted by `γ^{j+1}` for a `γ` drawn after
//! every position; the successor lives on round `r`'s domain squared
//! `min(shrink, k_r)` times; and the final weight sums every out-of-domain and
//! shift claim's `eq` at the challenges after it.
//!
//! What is pinned here:
//! - every proof the host accepts executes, under both lowerings, and its
//!   compressions are the closed form's (`chain_perms`);
//! - the machine's draws (every extension challenge and every query
//!   position) are the host verifier's, draw for draw, under both forms, and
//!   the weight the tail closes on is the host's formula evaluated on the
//!   host's own draws (each out-of-domain and each shift claim);
//! - every tamper the host refuses has no execution: values, nonces, paths,
//!   caps, the root, a round short of its count, a successor block, a proof
//!   under another shrink or the other form.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use multilinear::eq::eq_eval;
use multilinear::whir::Domain;
use multilinear::whir_chain::{
    CapPolicy, ChainConfig, ChainFormat, ChainProof, FinalPoly, FirstFold, GrindBits, LeafOrder,
    RoundForm, RoundOpenings, WhirFolds, WhirRounds, commit, ood_point, prove, verify,
};
use multilinear::whir_hash::Blake3Whir;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::super::builder::Ext;
use super::super::compiler::{LfmProgram, compile};
use super::super::edsl::{WrapDigest, WrapHash};
use super::super::executor::execute;
use super::super::validator::validate;
use super::super::whir_chain::{
    ChainShape, RoundStorage, byte_digest_words, chain_draws, chain_perms, emit_verify_weighted,
    push_round_words, round_words, try_push_round_words,
};
use super::super::whir_open::RootLanes;
use super::super::whir_poly::emit_eq_eval;
use super::super::whir_transcript::{SpongeEntry, WhirTranscript};
use super::super::word::{LfmWord, ext_word, word_as_ext};
use super::{HostTranscript, builder, hash_rows, under_both};

type F = GoldilocksField;
type E = GoldilocksExtension;

/// One small chain's parameters: the security target the counts are sized
/// for is small, so the chains open tens of positions, not hundreds.
#[derive(Clone, Copy, Debug)]
pub(super) struct Case {
    pub(super) n: usize,
    pub(super) lb: usize,
    pub(super) first: Option<usize>,
    pub(super) k: usize,
    /// `None`: today's form; `Some(s)`: the round form at shrink `s`.
    pub(super) shrink: Option<usize>,
    pub(super) bits: u8,
    pub(super) grind: u8,
    pub(super) cap: CapPolicy,
    pub(super) leaves: LeafOrder,
    pub(super) final_vars: Option<u8>,
    /// Under the round form: each tree's cap held to what its own openings
    /// pay for (`WhirRounds::caps_by_queries`), the policy's the ceiling.
    pub(super) caps_by_q: bool,
}

impl Case {
    /// The same chain under today's form.
    fn fri(self) -> Self {
        Self {
            shrink: None,
            caps_by_q: false,
            ..self
        }
    }
}

/// The config the host proves and verifies `c` under: the counts sized by
/// `with_security_folds`, then the round form's target (which
/// `check_round_form` holds to round 0's count).
pub(super) fn config(c: &Case) -> ChainConfig {
    let folds = c.first.map_or(WhirFolds::Uniform, |k| {
        WhirFolds::First(FirstFold::new(k).expect("a tested first fold"))
    });
    let mut cfg = ChainConfig::with_security_folds(
        c.lb,
        c.k,
        folds,
        c.n,
        c.bits,
        GrindBits::uniform(c.grind),
    );
    cfg.format.cap = c.cap;
    cfg.format.leaves = c.leaves;
    if let Some(v) = c.final_vars {
        cfg.format.final_poly = FinalPoly::vars(v).expect("a tested final width");
    }
    match c.shrink {
        Some(s) => {
            let cfg = cfg
                .with_whir_rounds(s, c.bits, c.n)
                .expect("a tested shrink");
            if c.caps_by_q {
                cfg.with_caps_by_queries().expect("a round-form config")
            } else {
                cfg
            }
        }
        None => cfg,
    }
}

/// The round-form chains every gate runs: a shrink below the fold (the rate
/// rising), equal to it (the rate fixed), the schedule the base is to run
/// (rate 1/2, first fold 4, four a round, shrink 3) at a small size, a first fold of 6,
/// natural and coset leaves, uncapped, a fixed and the automatic cap, grind
/// off and on, and a final polynomial.
pub(super) fn cases() -> Vec<Case> {
    let c = |n, lb, first, k, shrink, grind, cap, leaves, final_vars| Case {
        n,
        lb,
        first,
        k,
        shrink: Some(shrink),
        bits: 20,
        grind,
        cap,
        leaves,
        final_vars,
        caps_by_q: false,
    };
    // Caps by Q under a cap the later trees' counts do not pay for, so the
    // rule moves a cap: natural and coset, a final polynomial, the automatic
    // policy.
    let q = |n, grind, cap, leaves, final_vars| Case {
        caps_by_q: true,
        ..c(n, 1, Some(4), 4, 3, grind, cap, leaves, final_vars)
    };
    vec![
        q(12, 4, CapPolicy::Fixed(6), LeafOrder::Coset, None),
        q(14, 0, CapPolicy::Fixed(8), LeafOrder::Natural, None),
        q(14, 4, CapPolicy::Fixed(8), LeafOrder::Coset, Some(2)),
        q(14, 0, CapPolicy::Auto, LeafOrder::Coset, None),
        // A rate rising 3 bits a round at a tiny target: the last round's
        // one query pays for no cap (height 0, the tree checked against its
        // root) where the policy's cap is 4.
        Case {
            shrink: Some(1),
            bits: 2,
            ..q(14, 0, CapPolicy::Fixed(4), LeafOrder::Natural, None)
        },
        c(
            9,
            2,
            None,
            2,
            1,
            0,
            CapPolicy::Off,
            LeafOrder::Natural,
            None,
        ),
        c(9, 2, None, 2, 2, 0, CapPolicy::Off, LeafOrder::Coset, None),
        c(
            10,
            1,
            Some(4),
            3,
            2,
            4,
            CapPolicy::Fixed(2),
            LeafOrder::Coset,
            None,
        ),
        c(
            12,
            1,
            Some(4),
            4,
            3,
            4,
            CapPolicy::Auto,
            LeafOrder::Coset,
            None,
        ),
        c(
            11,
            2,
            Some(6),
            4,
            3,
            0,
            CapPolicy::Fixed(2),
            LeafOrder::Natural,
            None,
        ),
        c(
            12,
            1,
            Some(4),
            4,
            3,
            4,
            CapPolicy::Fixed(2),
            LeafOrder::Coset,
            Some(2),
        ),
    ]
}

fn point(n: usize) -> Vec<FEE> {
    (0..n)
        .map(|i| {
            FEE::new([
                FE::from(201 + i as u64),
                FE::from(5 * i as u64 + 9),
                FE::from(i as u64 + 4),
            ])
        })
        .collect()
}

pub(super) struct Fixture {
    pub(super) cfg: ChainConfig,
    pub(super) proof: ChainProof<F, E>,
    z: Vec<FEE>,
    y: FEE,
    root: [u8; 32],
    domain: Domain<F>,
    pub(super) shape: ChainShape,
}

/// One chain proved over `Blake3Whir` under `c`, accepted by the host.
pub(super) fn fixture(c: &Case) -> Fixture {
    let cfg = config(c);
    let f = super::chain::pseudo_mle(c.n, 23);
    let z = point(c.n);
    let y = f.evaluate_in::<E>(&z).expect("f takes its own point");
    let (commitment, domain) =
        commit::<F, Blake3Whir>(&f, &cfg, true).expect("the polynomial commits");
    let proof = prove::<F, E, _, Blake3Whir>(
        &f,
        &z,
        &commitment,
        &domain,
        &cfg,
        &mut HostTranscript::new(&[]),
    )
    .expect("the chain proves");
    let fx = Fixture {
        shape: ChainShape::for_hash(&cfg, c.n, WrapHash::Blake3),
        cfg,
        proof,
        z,
        y,
        root: commitment.root(),
        domain,
    };
    assert!(
        host_accepts(&fx, &fx.proof),
        "{c:?}: the host accepts its own proof"
    );
    assert_eq!(fx.shape.whir(), c.shrink.is_some());
    fx
}

pub(super) fn host_accepts(f: &Fixture, proof: &ChainProof<F, E>) -> bool {
    let mut t = HostTranscript::new(&[]);
    verify::<F, E, _, Blake3Whir>(proof, &f.root, &f.z, f.y, &f.domain, &f.cfg, &mut t).is_ok()
}

/// What a program published, in order: each draw's kind.
#[derive(Clone, Debug)]
enum Published {
    Ext,
    Index(usize),
}

/// `z`, `y`, the root (two words), the final value, then the rounds; with
/// `publish`, every draw the chain makes and its closing claim and weight
/// are published after it.
fn emit(shape: &ChainShape, publish: bool) -> (LfmProgram, Vec<Published>, chain_draws::Draws) {
    let mut b = builder();
    let head = (shape.num_vars + 4) as u32;
    let total = head
        + (0..shape.rounds())
            .map(|r| round_words(shape, r))
            .sum::<u32>();
    let arena = b.declare_arena(total);
    let mut transcript = WhirTranscript::for_builder(&mut b);
    let z: Vec<Ext> = (0..shape.num_vars)
        .map(|i| b.hint_word(arena, i as u32).as_ext())
        .collect();
    let n = shape.num_vars as u32;
    let y = b.hint_word(arena, n).as_ext();
    let r0 = b.hint_word(arena, n + 1);
    let r1 = b.hint_word(arena, n + 2);
    let root = RootLanes::unpack(&mut b, WrapDigest::from_pair(r0, r1));
    let final_value = b.hint_word(arena, n + 3).as_ext();
    let storage = RoundStorage::hint(&mut b, arena, head, shape);
    let (current, next) = storage.openings();
    let wires = storage.wires(&current, &next);
    let domain = Domain::<F>::new(shape.domain_log[0]).expect("the first domain");
    let ((), draws) = chain_draws::record(|| {
        emit_verify_weighted(
            &mut b,
            &mut transcript,
            &wires,
            root,
            final_value,
            y,
            shape,
            &domain,
            |b, alphas| emit_eq_eval(b, &z, alphas),
        )
    });
    let mut published = Vec::new();
    if publish {
        for draw in &draws.draws {
            match draw {
                chain_draws::Draw::Ext(e) => {
                    b.public(e.as_cell());
                    published.push(Published::Ext);
                }
                chain_draws::Draw::Index(bits) => {
                    for bit in bits {
                        b.public(bit.as_cell());
                    }
                    published.push(Published::Index(bits.len()));
                }
            }
        }
        let [(claim, weight)] = draws.closing.as_slice() else {
            panic!("one chain closes once")
        };
        b.public(claim.as_cell());
        b.public(weight.as_cell());
    }
    let program = compile(b.finish());
    validate(&program).expect("the round-form chain leg is admissible");
    (program, published, draws)
}

pub(super) fn program(shape: &ChainShape) -> LfmProgram {
    emit(shape, false).0
}

pub(super) fn arena(f: &Fixture, proof: &ChainProof<F, E>) -> Vec<LfmWord> {
    arena_as(&f.shape, f, proof)
}

fn arena_as(shape: &ChainShape, f: &Fixture, proof: &ChainProof<F, E>) -> Vec<LfmWord> {
    let mut words: Vec<LfmWord> = f.z.iter().map(ext_word).collect();
    words.push(ext_word(&f.y));
    words.extend(byte_digest_words(&f.root));
    words.push(ext_word(&proof.final_value));
    push_round_words(&mut words, shape, proof);
    words
}

pub(super) fn runs(program: &LfmProgram, words: Vec<LfmWord>) -> bool {
    execute(program, &[words], &crate::hash_pin::BLOCK_HASHER).is_ok()
}

/// ★ THE ROUND-FORM GATE. Every round-form chain the host accepts executes,
/// under both lowerings; it hashes with BLAKE3 alone, and its compressions are
/// the closed form's — fewer than the same chain's under today's form.
#[test]
fn a_round_form_blake3_chain_executes_and_costs_its_closed_form() {
    under_both(|lowering| {
        for c in cases() {
            let f = fixture(&c);
            let leg = program(&f.shape);
            execute(&leg, &[arena(&f, &f.proof)], &crate::hash_pin::BLOCK_HASHER)
                .unwrap_or_else(|e| panic!("{lowering} {c:?}: refused an accepted proof: {e:?}"));
            let (emitted, other) = hash_rows(&leg);
            assert_eq!(other, 0, "{c:?}: BLAKE3 alone");
            assert_eq!(
                emitted,
                chain_perms(&f.shape, SpongeEntry::fresh_for(WrapHash::Blake3)),
                "{lowering} {c:?}: the closed form"
            );
            if c.caps_by_q {
                let policy = ChainShape::for_hash(
                    &config(&Case {
                        caps_by_q: false,
                        ..c
                    }),
                    c.n,
                    WrapHash::Blake3,
                );
                assert!(
                    f.shape
                        .caps
                        .iter()
                        .zip(&policy.caps)
                        .all(|(by_q, cap)| by_q <= cap),
                    "{c:?}: caps by Q never raises a cap"
                );
                println!(
                    "CAPS BY Q {c:?}: caps {:?} against the policy's {:?}",
                    f.shape.caps, policy.caps
                );
                if c.shrink == Some(1) {
                    let last = (0..f.shape.rounds())
                        .rfind(|&r| f.shape.queried(r))
                        .expect("a queried round");
                    assert_eq!(
                        (f.shape.queries(last), f.shape.caps[last]),
                        (1, 0),
                        "{c:?}: one query pays for no cap"
                    );
                    assert!(policy.caps[last] > 0, "{c:?}: the policy caps it");
                }
            }
            let today = ChainShape::for_hash(&config(&c.fri()), c.n, WrapHash::Blake3);
            let today_perms = chain_perms(&today, SpongeEntry::fresh_for(WrapHash::Blake3));
            println!(
                "ROUND FORM {lowering} {c:?}: schedule {:?} · domains {:?} · queries {:?} · caps {:?} \
                 · {emitted} compressions (today's form {today_perms}, Q {})",
                f.shape.schedule,
                f.shape.domain_log,
                f.shape.round_queries,
                f.shape.caps,
                today.num_queries,
            );
        }
    });
}

/// One draw of the host verifier's transcript.
#[derive(Clone, Debug, PartialEq)]
enum HostDraw {
    Ext(FEE),
    Index(u64),
}

/// The host's transcript, recording every value it hands the verifier.
struct Recording {
    inner: HostTranscript,
    draws: Vec<HostDraw>,
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
        self.draws.push(HostDraw::Ext(drawn));
        drawn
    }
    fn sample_u64(&mut self, upper_bound: u64) -> u64 {
        let drawn = self.inner.sample_u64(upper_bound);
        self.draws.push(HostDraw::Index(drawn));
        drawn
    }
}

/// The host verifier's draws on the fixture's own proof.
fn host_draws(f: &Fixture) -> Vec<HostDraw> {
    let mut t = Recording {
        inner: HostTranscript::new(&[]),
        draws: Vec::new(),
    };
    verify::<F, E, _, Blake3Whir>(&f.proof, &f.root, &f.z, f.y, &f.domain, &f.cfg, &mut t)
        .expect("the host accepts its own proof");
    t.draws
}

/// ★ The final weight by the host's formula on the host's own draws, a second
/// derivation that reads nothing of the emitter: `eq(z, α)`, plus each
/// out-of-domain claim's `γ·eq(ood_point(z0), α_{bound..})`, plus — under the
/// round form — each query's `γ^{j+1}·eq(pow(ω^{q_j}), α_{bound..})`, `ω` the
/// folded domain's generator. The draws are walked by the round structure,
/// and every one of them must be spent.
fn host_weight(f: &Fixture, draws: &[HostDraw]) -> FEE {
    let shape = &f.shape;
    let mut alphas: Vec<FEE> = Vec::new();
    let mut claims: Vec<(FEE, Vec<FEE>, usize)> = Vec::new();
    let mut domain = f.domain.clone();
    let mut at = 0usize;
    let take_ext = |at: &mut usize| match &draws[*at] {
        HostDraw::Ext(v) => {
            *at += 1;
            *v
        }
        other => panic!("draw {at}: an extension draw expected, got {other:?}"),
    };
    for r in 0..shape.rounds() {
        let k = shape.schedule[r];
        let point: Vec<FEE> = (0..shape.messages(r)).map(|_| take_ext(&mut at)).collect();
        let bound = alphas.len() + point.len();
        let n_next = shape.chain_vars() - bound;
        let mut folded = domain.clone();
        for _ in 0..k {
            folded = folded.squared().expect("a folded domain");
        }
        if shape.queried(r) {
            if shape.next_depth(r).is_some() {
                let z0 = take_ext(&mut at);
                let gamma = take_ext(&mut at);
                claims.push((gamma, ood_point(&z0, n_next), bound));
            }
            let qs: Vec<u64> = (0..shape.queries(r))
                .map(|_| match &draws[at] {
                    HostDraw::Index(q) => {
                        at += 1;
                        *q
                    }
                    other => panic!("draw {at}: a position expected, got {other:?}"),
                })
                .collect();
            if shape.whir() && shape.next_depth(r).is_some() {
                let gamma = take_ext(&mut at);
                let mut power = gamma;
                for &q in &qs {
                    let y: FEE = folded.generator().pow(q).to_extension();
                    claims.push((power, ood_point(&y, n_next), bound));
                    power *= gamma;
                }
            }
        }
        alphas.extend(point);
        domain = if shape.whir() {
            let mut d = domain.clone();
            for _ in 0..shape.shrink(r) {
                d = d.squared().expect("a successor domain");
            }
            d
        } else {
            folded
        };
    }
    assert_eq!(at, draws.len(), "every host draw is spent by the walk");
    let mut weight = eq_eval(&f.z, &alphas).expect("the claim's eq");
    for (scale, p, bound) in &claims {
        weight += *scale * eq_eval(p, &alphas[*bound..]).expect("a claim's eq");
    }
    weight
}

/// ★ THE STREAM KAT. Under both forms, the machine's draws — every extension
/// challenge (sumcheck, `z0`, the out-of-domain `γ`, the round form's `γ`) and
/// every query position — are the host verifier's on the same proof, draw for
/// draw; the weight the machine's tail closes on is the host's formula on the
/// host's draws; and the claim is that weight times the final value.
#[test]
fn a_round_form_blake3_chain_draws_the_hosts_stream_and_closes_on_its_weight() {
    under_both(|lowering| {
        for c in cases().into_iter().flat_map(|c| [c, c.fri()]) {
            let f = fixture(&c);
            let (leg, published, _) = emit(&f.shape, true);
            let exec = execute(&leg, &[arena(&f, &f.proof)], &crate::hash_pin::BLOCK_HASHER)
                .unwrap_or_else(|e| panic!("{lowering} {c:?}: refused an accepted proof: {e:?}"));
            let words: Vec<LfmWord> = exec.public_words.iter().map(|(_, w)| *w).collect();
            let mut at = 0usize;
            let machine: Vec<HostDraw> = published
                .iter()
                .map(|kind| match kind {
                    Published::Ext => {
                        at += 1;
                        HostDraw::Ext(word_as_ext(&words[at - 1]).expect("an extension draw"))
                    }
                    Published::Index(nbits) => {
                        let mut q = 0u64;
                        for bit in 0..*nbits {
                            if words[at + bit][0] == FE::one() {
                                q |= 1 << bit;
                            }
                        }
                        at += nbits;
                        HostDraw::Index(q)
                    }
                })
                .collect();
            let claim = word_as_ext(&words[at]).expect("the closing claim");
            let weight = word_as_ext(&words[at + 1]).expect("the closing weight");
            assert_eq!(at + 2, words.len(), "nothing else is published");
            let host = host_draws(&f);
            assert_eq!(
                machine.len(),
                host.len(),
                "{lowering} {c:?}: the machine draws {} values, the host {}",
                machine.len(),
                host.len()
            );
            for (i, (m, h)) in machine.iter().zip(&host).enumerate() {
                assert_eq!(m, h, "{lowering} {c:?}: draw {i}");
            }
            assert_eq!(
                weight,
                host_weight(&f, &host),
                "{lowering} {c:?}: the tail's weight"
            );
            assert_eq!(
                claim,
                weight * f.proof.final_value,
                "{lowering} {c:?}: the claim closes on the final value"
            );
            let extensions = host
                .iter()
                .filter(|d| matches!(d, HostDraw::Ext(_)))
                .count();
            println!(
                "STREAM {lowering} {c:?}: {} draws ({extensions} extension), weight {weight:?}",
                host.len()
            );
        }
    });
}

/// Flips the last byte of `path`'s first node; false when the path is empty.
fn flip_first(path: &mut [[u8; 32]]) -> bool {
    match path.first_mut() {
        Some(node) => {
            node[31] ^= 0x80;
            true
        }
        None => false,
    }
}

/// The current openings of a per-polynomial chain's round `r`: its base
/// values (round 0) or extension values, by `(query, value)`.
fn current_mut(
    proof: &mut ChainProof<F, E>,
    r: usize,
) -> &mut Vec<multilinear::whir_commit::CosetOpening<E>> {
    match &mut proof.rounds[r].openings {
        RoundOpenings::Extension(p) => &mut p.current,
        _ => panic!("round {r} is extension-valued"),
    }
}

/// ★ THE ROUND-FORM TAMPER ARM. Every forgery the host refuses has no
/// execution: the final value, a sumcheck evaluation at the first and the last
/// round, an out-of-domain value at every round that has one, a folding, an
/// out-of-domain and a query nonce, a round-0 base value, a later round's
/// opened value (the fold value its claim carries) and Merkle sibling at every
/// queried round, a later round's cap node (on its own first opening), the
/// root; a round one opening short of its count and a successor block are
/// refused by the arena before the leaf runs.
#[test]
fn a_tampered_round_form_blake3_chain_cannot_execute() {
    under_both(|lowering| {
        for c in cases().into_iter().filter(|c| c.grind > 0 || c.caps_by_q) {
            let f = fixture(&c);
            let leg = program(&f.shape);
            assert!(
                runs(&leg, arena(&f, &f.proof)),
                "{c:?}: the untouched proof executes, or the arm proves nothing"
            );
            let rounds = f.proof.rounds.len();
            let queried: Vec<usize> = (0..rounds).filter(|&r| f.shape.queried(r)).collect();
            let mut sites: Vec<(String, ChainProof<F, E>)> = Vec::new();
            let mut forged = f.proof.clone();
            forged.final_value += FEE::one();
            sites.push(("the final value".into(), forged));
            for r in [0, rounds - 1] {
                let mut forged = f.proof.clone();
                forged.rounds[r].sumcheck[0].evaluations[1] += FEE::one();
                sites.push((format!("round {r}'s sumcheck evaluation"), forged));
            }
            for r in 0..rounds {
                if f.proof.rounds[r].ood_value.is_some() {
                    let mut forged = f.proof.clone();
                    if let Some(v) = forged.rounds[r].ood_value.as_mut() {
                        *v += FEE::one();
                    }
                    sites.push((format!("round {r}'s out-of-domain value"), forged));
                }
            }
            let grinds = if c.grind > 0 {
                &[0usize, 1, 2][..]
            } else {
                &[]
            };
            for (slot, name) in [(0usize, "folding"), (1, "out-of-domain"), (2, "query")]
                .into_iter()
                .filter(|(slot, _)| grinds.contains(slot))
            {
                let r = if slot == 1 { 0 } else { 1 };
                let mut forged = f.proof.clone();
                let nonces = &mut forged.rounds[r].nonces;
                match slot {
                    0 => nonces.folding ^= 1,
                    1 => nonces.ood ^= 1,
                    _ => nonces.query ^= 1,
                }
                sites.push((format!("round {r}'s {name} nonce"), forged));
            }
            let mut forged = f.proof.clone();
            match &mut forged.rounds[0].openings {
                RoundOpenings::Base(p) => p.current[1].values[3] += FE::one(),
                _ => unreachable!("round 0 opens base blocks"),
            }
            sites.push(("a round-0 base value".into(), forged));
            for &r in queried.iter().filter(|&&r| r >= 1) {
                // The second opening, or the only one of a round of one query.
                let i = 1.min(current_mut(&mut f.proof.clone(), r).len() - 1);
                let mut forged = f.proof.clone();
                current_mut(&mut forged, r)[i].values[0] += FEE::one();
                sites.push((format!("round {r}'s opened value"), forged));
                let mut forged = f.proof.clone();
                if flip_first(&mut current_mut(&mut forged, r)[i].proof.merkle_path) {
                    let height = f.shape.caps[r];
                    sites.push((
                        format!("round {r}'s Merkle sibling (cap height {height})"),
                        forged,
                    ));
                }
                if f.shape.caps[r] > 0 {
                    let mut forged = f.proof.clone();
                    let path = &mut current_mut(&mut forged, r)[0].proof.merkle_path;
                    let last = path.len() - 1;
                    path[last][31] ^= 1;
                    sites.push((format!("round {r}'s cap node"), forged));
                }
            }
            for (name, forged) in &sites {
                assert!(
                    !host_accepts(&f, forged),
                    "{c:?}: {name}: the host must refuse it"
                );
                assert!(
                    !runs(&leg, arena(&f, forged)),
                    "{lowering} {c:?}: {name}: the machine must refuse it"
                );
            }
            // Refused by the arena's shape before the leaf runs.
            let mut layout: Vec<(String, ChainProof<F, E>)> = Vec::new();
            let mut short = f.proof.clone();
            current_mut(&mut short, 1).pop();
            layout.push(("round 1 one opening short".into(), short));
            let mut padded = f.proof.clone();
            let extra = current_mut(&mut padded, 1)[0].clone();
            current_mut(&mut padded, 1).push(extra);
            layout.push(("round 1 one opening over".into(), padded));
            let mut successor = f.proof.clone();
            let block = current_mut(&mut successor, 1)[0].clone();
            match &mut successor.rounds[0].openings {
                RoundOpenings::Base(p) => p.next.push(block),
                _ => unreachable!("round 0 opens base blocks"),
            }
            layout.push(("a successor block at round 0".into(), successor));
            for (name, forged) in &layout {
                assert!(!host_accepts(&f, forged), "{c:?}: {name}: host");
                let mut words = Vec::new();
                assert!(
                    try_push_round_words(&mut words, &f.shape, forged).is_err(),
                    "{c:?}: {name}: the arena refuses it"
                );
                assert!(
                    !runs(&leg, arena(&f, forged)),
                    "{lowering} {c:?}: {name}: no execution"
                );
            }
            // A root off in its second word: every round-0 opening fails.
            let mut words = arena(&f, &f.proof);
            words[f.shape.num_vars + 2][3] += FE::one();
            assert!(!runs(&leg, words), "{c:?}: a root off");
            println!(
                "ROUND FORM TAMPER {lowering} {c:?}: {} sites and {} shapes refused",
                sites.len(),
                layout.len()
            );
        }
    });
}

/// ★ Domains, counts and caps are verifier constants. A program under one
/// shrink refuses the proof made under another (and so does the host); a
/// program under caps by Q and one under the policy's caps refuse each other's
/// proofs (the cap-rule mismatch); and today's program and the round form's
/// refuse each other's proofs: the arena a proof would need under the other's
/// shape is not the one the program reads.
#[test]
fn a_round_form_program_refuses_another_shrink_and_the_other_form() {
    under_both(|lowering| {
        let base = Case {
            n: 12,
            lb: 1,
            first: Some(4),
            k: 4,
            shrink: Some(3),
            bits: 20,
            grind: 0,
            cap: CapPolicy::Fixed(2),
            leaves: LeafOrder::Coset,
            final_vars: None,
            caps_by_q: false,
        };
        let three = fixture(&Case {
            cap: CapPolicy::Fixed(6),
            ..base
        });
        let by_q = fixture(&Case {
            cap: CapPolicy::Fixed(6),
            caps_by_q: true,
            ..base
        });
        assert_ne!(
            three.shape.caps, by_q.shape.caps,
            "caps by Q moves a cap at this shape"
        );
        let two = fixture(&Case {
            shrink: Some(2),
            ..base
        });
        let today = fixture(&base.fri());
        let programs: Vec<(&str, &Fixture, LfmProgram)> = [
            ("shrink 3", &three),
            ("shrink 3 caps by Q", &by_q),
            ("shrink 2", &two),
            ("today", &today),
        ]
        .into_iter()
        .map(|(name, f)| (name, f, program(&f.shape)))
        .collect();
        for (name, f, leg) in &programs {
            for (other_name, other, _) in &programs {
                let accepted = runs(leg, arena_as(&f.shape, f, &other.proof));
                assert_eq!(
                    accepted,
                    name == other_name,
                    "{lowering}: the {name} program on a {other_name} proof"
                );
                assert_eq!(
                    host_accepts(f, &other.proof),
                    name == other_name,
                    "the host under {name} on a {other_name} proof"
                );
            }
        }
    });
}

/// A group opening's chain over a run of `shape.trees ≥ 2` polynomials, for
/// the census: `z` (the run's variables), `y`, one root a round-0 tree (two
/// words each), the final value, then the rounds.
fn grouped_program(shape: &ChainShape) -> LfmProgram {
    use super::super::whir_chain::emit_verify_grouped;
    let mut b = builder();
    let trees = shape.trees_at(0);
    let vars = shape.chain_vars();
    let head = (vars + 2 + 2 * trees) as u32;
    let arena = b.declare_arena(head + RoundStorage::words(shape));
    let mut transcript = WhirTranscript::for_builder(&mut b);
    let z: Vec<Ext> = (0..vars)
        .map(|i| b.hint_word(arena, i as u32).as_ext())
        .collect();
    let y = b.hint_word(arena, vars as u32).as_ext();
    let roots: Vec<RootLanes> = (0..trees)
        .map(|t| {
            let at = (vars + 1 + 2 * t) as u32;
            let (r0, r1) = (b.hint_word(arena, at), b.hint_word(arena, at + 1));
            RootLanes::unpack(&mut b, WrapDigest::from_pair(r0, r1))
        })
        .collect();
    let final_value = b.hint_word(arena, head - 1).as_ext();
    let storage = RoundStorage::hint(&mut b, arena, head, shape);
    let (current, next) = storage.openings();
    let wires = storage.wires(&current, &next);
    let domain = Domain::<F>::new(shape.domain_log[0]).expect("the first domain");
    emit_verify_grouped(
        &mut b,
        &mut transcript,
        &wires,
        &roots,
        final_value,
        y,
        shape,
        &domain,
        |b, alphas, _| emit_eq_eval(b, &z, alphas),
    );
    compile(b.finish())
}

/// ★ THE ROUND FORM'S CENSUS: one BLAKE3 base group's opening chain (a run of
/// `B3_BASE_GROUP_POLYS` 2^27 polynomials, the base's block format — its cap,
/// its final polynomial, one tree a run, coset leaves — under the leaf's lean
/// query check), by closed form, for today's format, the first fold 4 alone
/// (today's form), and the round form at the base's intended schedule (blowup 2,
/// first fold 4, four a round, shrink 3; 128 bits), at first fold 6, and with
/// caps by Q (each tree's cap held to what its own openings pay for: the
/// base's own round-form config, checked equal here). Then today's and the two
/// round forms' programs are emitted (lean lowering,
/// no proof: the shape alone fixes the program), their compressions held to
/// the closed form, and their cells priced: compressions at the lean tables'
/// 2,832 cells each plus the field chips' real rows times their value columns
/// (and with their LogUp columns). Prints; asserts the forms agree with the
/// emission and that the round form spends fewer compressions.
#[test]
#[ignore = "census: emits three production-shape group chain programs"]
fn the_round_form_census_of_a_blake3_base_group() {
    use super::super::whir_chain::{
        ChipRows, LeafQuery, chain_cap_perms, chain_grind_perms, chain_opening_perms,
    };
    use crate::block_whir::{B3_BASE_GROUP_POLYS, B3Rounds, BlockFormat};
    use crate::hash_pin::{BlockFamily, ProofFamily};
    use crate::zf_format::ZfFormat;
    use stark::proof::options::BaseFormat;
    let zf = ZfFormat::DEFAULT.with_base(BaseFormat::B3_WHIR);
    let format = BlockFormat {
        zf,
        ..BlockFormat::production()
    };
    let n = zf.whir_stack.get();
    let polys = B3_BASE_GROUP_POLYS;
    let today = format.chain_config(&[(polys, n)]);
    // The base's own configs at first fold 4: today's form, and the round
    // form the base runs (shrink 3, caps by Q, its counts for the 128-bit
    // target and the union over the tallest stack).
    let mut zf4 = zf;
    zf4.whir_folds = WhirFolds::First(FirstFold::new(4).expect("first fold 4"));
    let at = |zf: ZfFormat, rounds: B3Rounds| {
        BlockFormat {
            zf,
            b3_rounds: rounds,
            ..BlockFormat::production()
        }
        .chain_config(&[(polys, n)])
    };
    let fri4 = at(zf4, B3Rounds::Fri);
    let base_w3 = at(zf4, B3Rounds::WHIR3);
    // The base's round form with the policy's caps: its own shrink, target
    // and union bound (never re-derived here), caps by Q off.
    let RoundForm::Whir(w) = base_w3.format.rounds else {
        panic!("the base's round form")
    };
    assert!(w.caps_by_queries(), "the base's round form takes caps by Q");
    let policy_caps = ChainConfig {
        format: ChainFormat {
            rounds: RoundForm::Whir(
                WhirRounds::new(w.shrink(), w.security_bits(), w.union_rounds())
                    .expect("the base's parameters"),
            ),
            ..base_w3.format
        },
        ..base_w3
    };
    assert_eq!(policy_caps.with_caps_by_queries(), Some(base_w3));
    assert_eq!(
        multilinear::whir_chain::check_round_form(&policy_caps),
        Ok(()),
        "the same counts"
    );
    let arms = [
        ("today's format", today),
        ("first fold 4, today's form", fri4),
        ("round form k0 4 s 3, policy caps", policy_caps),
        ("round form k0 6 s 3, caps by Q", at(zf, B3Rounds::WHIR3)),
        ("round form k0 4 s 3, caps by Q (the base's)", base_w3),
    ];
    let shape_of = |cfg: &ChainConfig| {
        ChainShape {
            query: LeafQuery::Lean,
            ..ChainShape::for_hash(cfg, n, WrapHash::Blake3)
        }
        .run(polys)
    };
    let entry = SpongeEntry::fresh_for(WrapHash::Blake3);
    let mut totals = Vec::new();
    for (name, cfg) in &arms {
        let shape = shape_of(cfg);
        let total = chain_perms(&shape, entry);
        let (openings, grinds) = (chain_opening_perms(&shape), chain_grind_perms(&shape));
        println!(
            "CENSUS {name}: blowup 2^{} · schedule {:?} · domains {:?} · queries {:?} (Q0 {}) · caps {:?} \
             · final {:?} · {total} compressions (openings {openings} incl. caps {}, grinds {grinds}, \
             transcript {})",
            cfg.log_blowup,
            shape.schedule,
            shape.domain_log,
            (0..shape.rounds())
                .filter(|&r| shape.queried(r))
                .map(|r| shape.queries(r))
                .collect::<Vec<_>>(),
            shape.num_queries,
            shape.caps,
            shape.final_at,
            chain_cap_perms(&shape),
            total - openings - grinds,
        );
        totals.push(total);
    }
    println!(
        "CENSUS round form k0 4 s 3 against today's format: {} -> {} compressions ({:+}, {:+.1} %)",
        totals[0],
        totals[2],
        totals[2] as i64 - totals[0] as i64,
        100.0 * (totals[2] as f64 / totals[0] as f64 - 1.0)
    );
    println!(
        "CENSUS caps by Q: {} -> {} compressions ({:+}) against the round form without them; \
         {:+.1} % against today's format",
        totals[2],
        totals[4],
        totals[4] as i64 - totals[2] as i64,
        100.0 * (totals[4] as f64 / totals[0] as f64 - 1.0)
    );
    assert!(totals[2] < totals[0] && totals[2] < totals[1]);
    assert!(totals[4] < totals[0]);
    // Pinned: today's format, the round form under the policy's caps (8 on
    // every tree) and under caps by Q (the base's own, 8 7 7 6 6).
    assert_eq!(
        (totals[0], totals[2], totals[4]),
        (28_768, 11_705, 11_461),
        "the census's closed forms"
    );

    super::LEAN.with(|c| c.set(true));
    for (name, cfg) in [&arms[0], &arms[2], &arms[4]] {
        let shape = shape_of(cfg);
        let leg = grouped_program(&shape);
        let (compressions, other) = hash_rows(&leg);
        assert_eq!(other, 0);
        assert_eq!(
            compressions,
            chain_perms(&shape, entry),
            "{name}: the closed form"
        );
        let chips = super::super::airs::lfm_chip_census_with_hasher(
            &leg,
            leg.hasher(<BlockFamily as ProofFamily>::SOCKET),
        );
        let muxes = super::lean_mux_cells(&leg);
        let field: u64 = chips
            .iter()
            .map(|c| c.real_rows * c.main_cols as u64)
            .sum::<u64>()
            + muxes.real_main;
        let field_aux: u64 = chips
            .iter()
            .map(|c| c.real_rows * (c.main_cols + 3 * c.aux_cols) as u64)
            .sum::<u64>()
            + muxes.real_main
            + 3 * muxes.real_aux;
        let hash = compressions as u64 * 2_832;
        println!(
            "CENSUS CELLS {name}: {} instructions · {compressions} compressions ({hash} lean cells) · \
             field main {field} (with LogUp {field_aux}) · total {} (with LogUp {}) · rows {:?}",
            leg.instrs.len(),
            hash + field,
            hash + field_aux,
            ChipRows::of(&leg)
        );
    }
    super::LEAN.with(|c| c.set(false));
}

/// ★ A round-form config the host refuses is refused here: its stamped
/// target gives another round-0 count than the one set (a count set by hand
/// after the form was stamped), and `check_round_form` refuses it on both host
/// sides, so no program is emitted for it.
#[test]
#[should_panic(expected = "no chain is emitted for a config the host refuses")]
fn a_round_form_config_the_host_refuses_emits_no_chain() {
    let c = cases()[3];
    let mut cfg = config(&c);
    assert!(multilinear::whir_chain::check_round_form(&cfg).is_ok());
    cfg.num_queries += 1;
    assert!(multilinear::whir_chain::check_round_form(&cfg).is_err());
    let _ = ChainShape::for_hash(&cfg, c.n, WrapHash::Blake3);
}

/// ★ Each round's out-of-domain point is refused inside its SUCCESSOR's
/// domain, the host's: the domains the emission checks are
/// `ChainConfig::round_domain_logs`' successors', round by round — bigger
/// than the folded ones under a shrink below the fold. And the check itself:
/// a point of the successor's domain has no execution, one just outside it
/// executes, and the same point is not refused by a check against the folded
/// domain (so the domain the check takes is load-bearing).
#[test]
fn the_out_of_domain_check_takes_the_successors_domain() {
    use super::super::whir_chain::emit_out_of_domain_point;
    let mut moved = false;
    for c in cases() {
        let cfg = config(&c);
        let shape = ChainShape::for_hash(&cfg, c.n, WrapHash::Blake3);
        let (_, _, draws) = emit(&shape, false);
        let domains = cfg.round_domain_logs(c.n);
        let want: Vec<usize> = (0..shape.rounds())
            .filter(|&r| shape.next_depth(r).is_some())
            .map(|r| domains[r + 1])
            .collect();
        assert_eq!(draws.ood_domains, want, "{c:?}: the successors' domains");
        let folded: Vec<usize> = (0..shape.rounds())
            .filter(|&r| shape.next_depth(r).is_some())
            .map(|r| shape.current_depth(r))
            .collect();
        moved |= want != folded;
    }
    assert!(moved, "some case's successor domain is not its fold's");

    // The check alone: `z0` of order `2^d` lies in a domain of `2^d` points.
    let refused = |z0_log: usize, domain_log: usize| {
        let mut b = builder();
        let arena = b.declare_arena(1);
        let z0 = b.hint_word(arena, 0).as_ext();
        let one = b.ext_const(&FEE::one());
        let domain = Domain::<F>::new(domain_log).expect("a domain");
        let point = emit_out_of_domain_point(&mut b, z0, &domain, 2, one);
        b.public(point[1].as_cell());
        let program = compile(b.finish());
        let z0 = *Domain::<F>::new(z0_log).expect("a domain").generator();
        !runs(&program, vec![[z0, FE::zero(), FE::zero(), FE::zero()]])
    };
    assert!(refused(10, 10), "a point of the successor's domain");
    assert!(!refused(11, 10), "a point just outside it");
    assert!(
        !refused(10, 7),
        "the folded domain's check would let the successor's point through"
    );
}

/// ★ The lean all-leaf tail check stays off under the round form: its claims
/// sit at the sampled positions, and the tail check reads a successor on the
/// folded domain. A lean round-form chain whose later trees lie wholly in
/// their caps with enough queries a leaf (today's form would check them once
/// a leaf) has no tail round, and its honest proof executes.
#[test]
fn a_lean_round_form_chain_has_no_tail_round_and_executes() {
    use super::super::whir_chain::LeafQuery;
    under_both(|lowering| {
        let c = Case {
            n: 9,
            lb: 1,
            first: Some(6),
            k: 1,
            shrink: Some(6),
            bits: 50,
            grind: 0,
            cap: CapPolicy::Fixed(4),
            leaves: LeafOrder::Coset,
            final_vars: None,
            caps_by_q: false,
        };
        let f = fixture(&c);
        let lean = ChainShape {
            query: LeafQuery::Lean,
            ..f.shape.clone()
        };
        let as_today = ChainShape {
            form: multilinear::whir_chain::RoundForm::Fri,
            round_queries: Vec::new(),
            ..lean.clone()
        };
        let shaped: Vec<usize> = (0..lean.rounds())
            .filter(|&r| as_today.tail_round(r))
            .collect();
        assert!(
            !shaped.is_empty(),
            "{c:?}: rounds {:?} caps {:?} Q {}: a tail-shaped round",
            lean.domain_log,
            lean.caps,
            lean.num_queries
        );
        assert!(
            (0..lean.rounds()).all(|r| !lean.tail_round(r)),
            "no tail round under the round form"
        );
        let leg = program(&lean);
        let words = arena_as(&lean, &f, &f.proof);
        execute(&leg, &[words], &crate::hash_pin::BLOCK_HASHER)
            .unwrap_or_else(|e| panic!("{lowering}: the lean round-form chain refused: {e:?}"));
        assert_eq!(
            hash_rows(&leg).0,
            chain_perms(&lean, SpongeEntry::fresh_for(WrapHash::Blake3))
        );
        println!("LEAN ROUND FORM {lowering}: tail-shaped rounds {shaped:?} checked per query");
    });
}
