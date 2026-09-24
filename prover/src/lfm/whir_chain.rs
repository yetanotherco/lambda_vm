//! `whir_chain::verify_weighted` as a machine leg — the WHIR chain, assembled.
//!
//! This is the whole of `crypto/multilinear/src/whir_chain.rs:938-1105` and it
//! is where the wrap's cost lives: about ninety per cent of a chain's
//! permutations are the query phase's openings, and a chain is run once per
//! stacked polynomial.
//!
//! Every brick is already gated on its own against the host function it
//! replaces — the grind against `is_valid_nonce`
//! ([`super::whir_transcript::emit_grind_check`]), the degree-2 sumcheck
//! against `verify_rounds` ([`super::whir_poly`]), `eq` against `eq_eval`, the
//! opening against `verify_opening` ([`super::whir_open`]), the fold against
//! `fold_coset` ([`super::whir_fold`]), and the transcript against the K1–K6
//! vectors. What this module adds is the ORDER, and the order is exactly what
//! its own gate checks: the host's challenge stream, observed through a
//! recording transcript, against the stream the machine derives from the
//! sponge.
//!
//! # The round structure, and the three places a nonce is spent
//!
//! Per round: `check_grind(folding)`; `k_r` sumcheck rounds at degree two, each
//! absorbing its two evaluations and drawing one challenge; then the domain is
//! squared `k_r` times, which is emit-time and free. A round that has a
//! successor absorbs its root, draws `z0`, refuses an in-domain `z0`, absorbs
//! `y0`, spends the OOD nonce, draws `gamma`, folds `gamma·y0` into the claim,
//! spends the QUERY nonce, and opens `Q` queries against both trees. The last
//! round absorbs `final_value` instead, spends only the query nonce, and opens
//! `Q` queries against one tree — `verify_final` has no successor
//! (`whir_chain.rs:1125-1141`). That is why a chain has `3R − 1` grinds.
//!
//! # ★ The out-of-domain squarings are SHARED, and exactly
//!
//! `require_out_of_domain` squares `z0` once per bit of the successor domain
//! (`:93-96`) and `ood_point` keeps `z0^(2^j)` for `j < num_vars − bound`
//! (`:72-81`). Those are the same chain of squarings, and the second is a
//! PREFIX of the first: `bound_r` variables are bound after round `r`, the
//! successor domain is `2^(num_vars + 2 − bound_r)`, so the domain check needs
//! exactly two more squarings than the point does — always, at every round and
//! every stack height. One chain of `D_{r+1}` multiplies serves both. Computing
//! them apart would cost `2·D_{r+1} − 2`, which at `S = 25` is 156 multiplies a
//! chain against 78.
//!
//! # ★ The successor slot is a MUX, and it is not free
//!
//! `leaf_and_slot` splits the query into a leaf index and a slot
//! (`whir_commit.rs:119-121`); both bounds are powers of two, so the split is a
//! partition of the `BitDec`'s bits and costs nothing. But `nxt.values[slot]`
//! (`whir_round.rs:160`) then selects one of `2^k` extension values by the high
//! bits, which is `2^k − 1` `Select` rows a query a round. At `k = 4`, `Q =
//! 112` and six successor rounds that is 8,400 rows a chain. The slot is always
//! in range — `slot < 2^{k_{r+1}}` follows from `q < 2^{D_{r+1}}` — so there is
//! no out-of-range branch to emit, which is the host's `QueryOutOfRange` made
//! unreachable rather than checked.
//!
//! # ⚠ The tail divides, and must
//!
//! The host computes `required = claim · weight⁻¹` and rejects a zero weight
//! with `DegenerateEvaluationPoint` (`:1095-1102`). Asserting
//! `final_value · weight == claim` instead would be one row cheaper and
//! strictly WEAKER: it accepts every `final_value` when `weight` and `claim`
//! are both zero, where the host rejects. So the tail inverts —
//! `div(one, weight)` has no satisfying assignment at zero, which IS the
//! refusal — and then compares.
//!
//! # What is emit-time, and therefore has no runtime check
//!
//! The host's shape rejections (`proof.rounds.len() != schedule.len()`,
//! `round.sumcheck.len() != k`, the base-versus-extension opening variant at
//! `:983`, `openings.current.len() != num_queries`) are all properties of the
//! PROGRAM here: the emitter reads a fixed number of wires at fixed offsets, so
//! a proof of another shape has no way to be supplied. That is the
//! `epoch_verify.rs:171-179` idiom — the absence of a second value rather than
//! an assert somebody could forget.

use multilinear::whir::Domain;
use multilinear::whir_chain::{ChainConfig, ChainProof, ChainRound, RoundOpenings};

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::algebraic_commit::commitment_to_digest;
use super::builder::{Bit, Cell, Ext, Felt, LfmBuilder};
use super::edsl::WrapDigest;
use super::whir_fold::{emit_fold_coset, fold_coset_rows};
use super::whir_open::{
    BlockValues, CapCells, TreeAuth, cap_check_perms, cap_check_rows, verify_opening_perms_capped,
    verify_opening_rows_capped,
};
use super::whir_poly::{
    emit_eq_eval, emit_sumcheck_round, eq_eval_rows_again, sumcheck_round_rows,
};
use super::whir_transcript::{
    COORDINATES_PER_EXT, SpongeEntry, SpongeHash, SpongeSchedule, WhirTranscript, emit_grind_check,
};
use super::word::{LfmWord, ext_word};

/// The degree the weight raises the plain `f` term to (`whir_chain.rs:991`).
pub(crate) const SUMCHECK_DEGREE: usize = 2;

/// One query's opening against one tree: the block, and its authentication
/// path.
pub struct QueryOpening<'a> {
    pub values: BlockValues<'a>,
    pub siblings: &'a [WrapDigest],
}

/// The three nonces a round spends. The out-of-domain one is unused on the last
/// round and is not read there.
#[derive(Clone, Copy)]
pub struct RoundNonces {
    pub folding: Felt,
    pub ood: Felt,
    pub query: Felt,
}

/// One round of the chain, as wires.
pub struct ChainRoundWires<'a> {
    /// `k_r` sumcheck rounds, each carrying `SUMCHECK_DEGREE` evaluations.
    pub sumcheck: &'a [Vec<Ext>],
    /// The successor's root as a word. `None` on the last round, where the
    /// folded message is a constant and `final_value` is sent instead.
    pub next_root: Option<Cell>,
    /// The successor's value at the out-of-domain point. `None` on the last
    /// round.
    pub ood_value: Option<Ext>,
    pub nonces: RoundNonces,
    /// Per query: the block of the current codeword that folds onto the query.
    pub current: &'a [QueryOpening<'a>],
    /// Per query: the successor block holding the folded value. Empty on the
    /// last round.
    pub next: &'a [QueryOpening<'a>],
    /// The current tree's Merkle cap, when this round OWNS it: round 0 with
    /// `caps[0] > 0`. Empty otherwise (W1; a later round's current tree was
    /// authenticated as the round before's successor).
    pub current_cap: &'a [WrapDigest],
    /// The successor tree's Merkle cap, when it has one (`caps[r + 1] > 0`).
    pub next_cap: &'a [WrapDigest],
}

/// The shape of one chain: everything the closed forms below are a function of.
///
/// Built from the same `ChainConfig::schedule` the host runs, so a change to
/// the schedule moves both sides together rather than only one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainShape {
    /// Variables folded in each round.
    pub schedule: Vec<usize>,
    /// `log2` of the domain each round starts at: `D_0 = num_vars +
    /// log_blowup`, then `D_{r+1} = D_r − k_r`.
    pub domain_log: Vec<usize>,
    pub num_vars: usize,
    pub num_queries: usize,
    pub grind: (usize, usize, usize),
    /// Each tree's Merkle cap height (W1): tree `r` is round `r`'s current
    /// tree. From the same `ChainConfig::tree_caps` the host prover and
    /// verifier use; all zero at the default format.
    pub caps: Vec<usize>,
}

impl ChainShape {
    pub fn new(config: &ChainConfig, num_vars: usize) -> Self {
        let schedule = config.schedule(num_vars);
        let mut domain_log = Vec::with_capacity(schedule.len());
        let mut d = num_vars + config.log_blowup;
        for &k in &schedule {
            domain_log.push(d);
            d -= k;
        }
        let caps = config.tree_caps(num_vars);
        debug_assert_eq!(caps.len(), schedule.len(), "one cap height per tree");
        Self {
            schedule,
            domain_log,
            num_vars,
            num_queries: config.num_queries,
            caps,
            grind: (
                config.grind.folding as usize,
                config.grind.ood as usize,
                config.grind.query as usize,
            ),
        }
    }

    pub fn rounds(&self) -> usize {
        self.schedule.len()
    }

    /// The current tree's depth at round `r`: its leaves are the round's domain
    /// folded by that round's `k`.
    pub fn current_depth(&self, r: usize) -> usize {
        self.domain_log[r] - self.schedule[r]
    }

    /// The successor tree's depth at round `r`, which is the NEXT round's
    /// current depth. Only rounds before the last have one.
    pub fn next_depth(&self, r: usize) -> Option<usize> {
        (r + 1 < self.rounds()).then(|| self.current_depth(r + 1))
    }

    /// Round `r`'s current tree's cap height. ⚠ [`current_depth`](Self::current_depth)
    /// stays the index-bit count; the sibling count is
    /// [`current_path`](Self::current_path).
    pub fn current_cap(&self, r: usize) -> usize {
        self.caps[r]
    }

    /// Siblings on a path to round `r`'s current tree's cap.
    pub fn current_path(&self, r: usize) -> usize {
        self.current_depth(r) - self.caps[r]
    }

    /// The successor tree's cap height at round `r`.
    pub fn next_cap(&self, r: usize) -> Option<usize> {
        (r + 1 < self.rounds()).then(|| self.caps[r + 1])
    }

    /// Felts in round `r`'s current block: one per value in round 0, where the
    /// codeword is still base-field, and three after.
    pub fn current_felts(&self, r: usize) -> usize {
        let values = 1usize << self.schedule[r];
        if r == 0 { values } else { 3 * values }
    }

    /// Variables bound after round `r`.
    pub fn bound(&self, r: usize) -> usize {
        self.schedule[..=r].iter().sum()
    }
}

/// PERMUTATIONS one chain costs in its query phase and its openings — the term
/// the wrap's cost is dominated by, and a function of the tree depths and block
/// widths alone.
///
/// The transcript's own permutations are NOT in this: they depend on the
/// sponge's buffer at each squeeze, which is a schedule and not a shape, and
/// mixing the two would make a form that no longer says where its cost is.
/// They are [`chain_schedule_perms`], and the grinds' two-apiece are
/// [`chain_grind_perms`]; [`chain_perms`] is the sum of the three.
pub fn chain_opening_perms(shape: &ChainShape) -> usize {
    let mut per_query = 0;
    for r in 0..shape.rounds() {
        per_query += verify_opening_perms_capped(
            shape.current_felts(r),
            shape.current_depth(r),
            shape.caps[r],
        );
        if let Some(depth) = shape.next_depth(r) {
            per_query +=
                verify_opening_perms_capped(3 << shape.schedule[r + 1], depth, shape.caps[r + 1]);
        }
    }
    shape.num_queries * per_query + chain_cap_perms(shape)
}

/// PERMUTATIONS the chain's cap checks cost: each capped tree's cap hashed up
/// to its root once, `2^c − 1` parents. Zero at the default. Part of
/// [`chain_opening_perms`], stated apart so the per-tree term is visible.
pub fn chain_cap_perms(shape: &ChainShape) -> usize {
    shape.caps.iter().map(|&c| cap_check_perms(c)).sum()
}

/// INSTRUCTIONS one chain's query phase costs: per round, per query, the two
/// openings, the fold, the successor slot's mux and the assert that closes it.
pub fn chain_query_rows(shape: &ChainShape) -> usize {
    let mut per_round = 0;
    for r in 0..shape.rounds() {
        let felts = shape.current_felts(r);
        let unpacks = if r == 0 {
            0
        } else {
            1usize << shape.schedule[r]
        };
        let depth = shape.current_depth(r);
        // The index draw, the current opening, and the fold.
        let mut q = 1
            + verify_opening_rows_capped(felts, unpacks, depth, shape.caps[r])
            + fold_coset_rows(1usize << shape.schedule[r], depth);
        match shape.next_depth(r) {
            Some(next_depth) => {
                let next_block = 1usize << shape.schedule[r + 1];
                // The successor opening, the slot mux, and `folded == claimed`.
                q += verify_opening_rows_capped(
                    3 * next_block,
                    next_block,
                    next_depth,
                    shape.caps[r + 1],
                ) + (next_block - 1)
                    + 2;
            }
            // `folded == final_value`.
            None => q += 2,
        }
        per_round += shape.num_queries * q;
    }
    per_round
}

/// INSTRUCTIONS one chain costs OUTSIDE its query phase: the grinds' fixed
/// parts, the sumchecks, the out-of-domain chain, the claim updates, and the
/// tail.
///
/// ⚠ The grinds' `state_rows` term is NOT here. A `state()` hashes whatever the
/// sponge is holding at that moment, which is a property of the schedule rather
/// than of the shape; it is in [`chain_hash_schedule`], as a `State` event,
/// beside the transcript's own squeezes.
pub fn chain_fixed_rows(shape: &ChainShape) -> usize {
    let (folding, ood, query) = shape.grind;
    let mut rows = 0;
    for r in 0..shape.rounds() {
        let k = shape.schedule[r];
        rows += grind_fixed(folding);
        // Per sumcheck round: two absorbed evaluations (one `Unpack` each), one
        // drawn challenge (one `Pack`), and the round itself.
        rows += k * (SUMCHECK_DEGREE + 1 + sumcheck_round_rows(SUMCHECK_DEGREE));
        match shape.next_depth(r) {
            Some(_) => {
                let next_domain_log = shape.domain_log[r] - k;
                // The successor root: one `Unpack` into four absorbed felts.
                rows += 1;
                // `z0`: one `Pack`.
                rows += 1;
                // The squaring chain, shared between the domain check and the
                // out-of-domain point, plus the `!= 1` refusal.
                rows += next_domain_log + 2;
                // `y0` absorbed, the ood grind, `gamma` drawn, the claim update.
                rows += 1 + grind_fixed(ood) + 1 + 1;
                rows += grind_fixed(query);
            }
            None => {
                // `final_value` absorbed, then the query grind.
                rows += 1 + grind_fixed(query);
            }
        }
    }
    // The tail: one `eq` and one `MulAdd` per out-of-domain claim, then the
    // inversion, the multiply and the assert.
    for r in 0..shape.rounds() - 1 {
        rows += eq_eval_rows_again(shape.num_vars - shape.bound(r)) + 1;
    }
    rows += 1 + 1 + 2;
    // W1: each capped tree's cap check, once (its hinted words are arena
    // words, counted with the arena like every other hint).
    rows += shape.caps.iter().map(|&c| cap_check_rows(c)).sum::<usize>();
    rows
}

/// A grind's rows beyond the sponge state it reads — see
/// [`super::whir_transcript::grind_check_rows`], whose `state_rows` term
/// belongs to the schedule.
const fn grind_fixed(bits: usize) -> usize {
    super::whir_transcript::grind_check_rows(bits)
}

/// PERMUTATIONS the grinds cost: two a check — one for the inner hash of
/// `PREFIX ‖ seed ‖ factor` and one for the outer hash of `inner ‖ nonce`
/// (`epoch.rs:614-640`). A check at zero bits costs none, because it is not
/// emitted at all.
///
/// The count of checks is the round structure's `3R − 1`, written as the three
/// terms it comes from so a config that grinds in only one place still reads
/// correctly.
pub fn chain_grind_perms(shape: &ChainShape) -> usize {
    let (folding, ood, query) = shape.grind;
    let rounds = shape.rounds();
    let checks = rounds * usize::from(folding > 0)
        + (rounds - 1) * usize::from(ood > 0)
        + rounds * usize::from(query > 0);
    PERMS_PER_GRIND * checks
}

/// Permutations one grind check spends: its two hashes.
const PERMS_PER_GRIND: usize = 2;

/// ★ The chain's SCHEDULE: every hash its transcript performs, in order, and
/// how many felts each one hashes.
///
/// This is the half of the chain's cost that the shape alone does not give,
/// because a sponge's cost is what its BUFFER holds at each hash and the buffer
/// is a running quantity. It is derived here from the round structure — the
/// absorbs, the draws, the `state()` reads and the candidates a squeeze hands
/// out — and NOT read off [`WhirTranscript`], which is a different state
/// machine written for a different purpose. Its gate is the host's own call
/// stream (`whir_chain_tests::the_schedule_is_the_host_transcripts`).
///
/// Every absorb by the shape it comes from: `SUMCHECK_DEGREE ·
/// COORDINATES_PER_EXT` felts a sumcheck round, `DIGEST_FELTS` for a successor
/// root, `COORDINATES_PER_EXT` for `y0` or `final_value`, `NONCE_FELTS` a
/// spent grind. Every draw: one extension challenge a sumcheck round, `z0` and
/// `gamma` on a round with a successor, and `num_queries` bounded draws in the
/// query phase — which come FIRST and together, because `sample_queries` draws
/// them all before any opening is checked (`whir_round.rs:68-76`).
pub fn chain_hash_schedule(shape: &ChainShape, entry: SpongeEntry) -> Vec<SpongeHash> {
    let mut sponge = SpongeSchedule::new(entry);
    chain_sponge(shape, &mut sponge);
    sponge.hashes().to_vec()
}

/// ★ The same round structure, driven into a schedule the CALLER owns — so the
/// sponge a chain leaves behind is reachable, and the next leg of the same
/// program continues from it.
///
/// ⚠ **This is the one place the chain's round structure is written**, and
/// [`chain_hash_schedule`] is a thin wrapper over it rather than a second copy.
/// A caller that needs the successor entry — `stacked_eval`'s wrapper threading
/// one sponge through every polynomial's chain — could instead replay the
/// structure beside this one, and that replay would be a second thing to keep
/// in step: the schedule's gate
/// (`whir_chain_tests::the_schedule_is_the_host_transcripts`) compares THIS
/// walk against a real verifier's calls, and would say nothing about a copy.
/// Sharing the walk puts the caller inside that gate.
pub fn chain_sponge(shape: &ChainShape, sponge: &mut SpongeSchedule) {
    let (folding, ood, query) = shape.grind;

    for r in 0..shape.rounds() {
        sponge.grind(folding);
        for _ in 0..shape.schedule[r] {
            sponge.absorb(SUMCHECK_DEGREE * COORDINATES_PER_EXT);
            sponge.draw_ext();
        }
        match shape.next_depth(r) {
            Some(_) => {
                sponge.absorb(super::whir_transcript::DIGEST_FELTS);
                sponge.draw_ext();
                sponge.absorb(COORDINATES_PER_EXT);
                sponge.grind(ood);
                sponge.draw_ext();
                sponge.grind(query);
            }
            None => {
                sponge.absorb(COORDINATES_PER_EXT);
                sponge.grind(query);
            }
        }
        for _ in 0..shape.num_queries {
            sponge.candidate();
        }
    }
}

/// INSTRUCTIONS the schedule costs: the `Pack`s that build each hash's words
/// and the `Unpack` a squeeze spends reading its digest.
pub fn chain_schedule_rows(shape: &ChainShape, entry: SpongeEntry) -> usize {
    chain_hash_schedule(shape, entry)
        .iter()
        .map(|hash| hash.rows())
        .sum()
}

/// PERMUTATIONS the schedule costs: one per rate-8 block of every hash.
pub fn chain_schedule_perms(shape: &ChainShape, entry: SpongeEntry) -> usize {
    chain_hash_schedule(shape, entry)
        .iter()
        .map(|hash| hash.perms())
        .sum()
}

/// ★ INSTRUCTIONS one chain costs, whole: the shape half and the schedule half.
///
/// The two are pinned apart and not only as this sum, because they move
/// independently — a form that shifted work from the sponge to the arithmetic
/// at constant total would fail one of them rather than neither.
pub fn chain_rows(shape: &ChainShape, entry: SpongeEntry) -> usize {
    chain_shape_rows(shape) + chain_schedule_rows(shape, entry)
}

/// ★ PERMUTATIONS one chain costs, whole, in the THREE terms it has: the query
/// phase's openings, the grinds' two hashes each, and the transcript's own.
///
/// ⚠ Only the first was pinned before this; the sizing note's per-chain
/// permutation figure predates the other two and is not reproduced by them.
pub fn chain_perms(shape: &ChainShape, entry: SpongeEntry) -> usize {
    chain_opening_perms(shape) + chain_grind_perms(shape) + chain_schedule_perms(shape, entry)
}

/// ★★ THE FOLD CONSTANTS ONE WHOLE CHAIN INTERNS, by value — every round's,
/// unioned, over that round's own domain.
///
/// ⛔ A UNION AND NOT A MAXIMUM. Round `r` folds over the base domain squared
/// `Σ_{j<r} schedule[j]` times, so each round's generator differs and the same
/// exponent set yields different field elements. Nothing nests here the way the
/// sumcheck round's Newton pairs do, and a form that took a max of anything
/// would be quietly wrong.
///
/// The domain advance mirrors the emitter's own (`:484-489`): `k` squarings per
/// round, where `k` is that round's schedule entry.
pub fn chain_fold_constants(
    shape: &ChainShape,
    domain: &Domain<GoldilocksField>,
) -> Vec<super::word::LfmWord> {
    let mut words: Vec<super::word::LfmWord> = Vec::new();
    let mut current = domain.clone();
    for r in 0..shape.rounds() {
        for word in super::whir_fold::fold_coset_constants(
            &current,
            shape.schedule[r],
            shape.current_depth(r),
        ) {
            if !words.contains(&word) {
                words.push(word);
            }
        }
        for _ in 0..shape.schedule[r] {
            current = current
                .squared()
                .expect("the schedule never folds past the domain");
        }
    }
    words
}

/// ★ `whir_chain::verify_weighted`, emitted.
///
/// `weight_at` is the caller's closure, mirroring the host's `W` parameter: it
/// is handed the concatenated round challenges and returns the weight at that
/// point. `whir_chain::verify` passes `eq_eval(z, ·)`, which is
/// [`emit_eq_eval`]; a stacked chain passes
/// [`super::whir_stacked::emit_weight_at`].
///
/// Every refusal is a division with no satisfying assignment rather than a
/// branch, so a proof the host rejects has no execution here.
#[allow(clippy::too_many_arguments)]
pub fn emit_verify_weighted(
    b: &mut LfmBuilder,
    transcript: &mut WhirTranscript,
    rounds: &[ChainRoundWires<'_>],
    root_lanes: &[Felt; 4],
    final_value: Ext,
    y: Ext,
    shape: &ChainShape,
    domain: &Domain<GoldilocksField>,
    weight_at: impl FnOnce(&mut LfmBuilder, &[Ext]) -> Ext,
) {
    assert_eq!(
        rounds.len(),
        shape.rounds(),
        "one set of wires per scheduled round"
    );
    let one = b.ext_const(&FEE::one());
    let (grind_folding, grind_ood, grind_query) = shape.grind;

    let mut claim = y;
    let mut alphas: Vec<Ext> = Vec::with_capacity(shape.num_vars);
    // Tree 0: its cap (when it has one) is authenticated against the root
    // here, once; every later tree where its root is absorbed.
    let mut current_tree = tree_auth(b, shape.caps[0], rounds[0].current_cap, root_lanes);
    let mut current_domain = domain.clone();
    // Each out-of-domain claim: its batching weight, its point, and how many
    // variables were bound when it entered.
    let mut ood: Vec<(Ext, Vec<Ext>, usize)> = Vec::new();

    for (r, round) in rounds.iter().enumerate() {
        let k = shape.schedule[r];
        assert_eq!(round.sumcheck.len(), k, "round {r} folds {k} variables");

        emit_grind_check(b, transcript, grind_folding as u8, round.nonces.folding);

        // The sumcheck, interleaved: a round's challenge is drawn only after
        // its evaluations are absorbed, which is the order `verify_rounds`
        // takes (`sumcheck.rs:386-390`) and the seam the round primitive was
        // left with.
        let mut point = Vec::with_capacity(k);
        for evaluations in round.sumcheck {
            assert_eq!(evaluations.len(), SUMCHECK_DEGREE);
            for e in evaluations {
                transcript.absorb_ext(b, *e);
            }
            let challenge = transcript.sample_ext(b);
            claim = emit_sumcheck_round(b, claim, evaluations, challenge);
            point.push(challenge);
        }

        let mut next_domain = current_domain.clone();
        for _ in 0..k {
            next_domain = next_domain
                .squared()
                .expect("the schedule never folds past the domain");
        }
        let bound = alphas.len() + k;

        let next_tree = match (round.next_root, round.ood_value, shape.next_depth(r)) {
            (Some(next_root), Some(y0), Some(_)) => {
                let lanes = b.unpack(next_root);
                transcript.absorb_felts(b, &lanes);

                let z0 = transcript.sample_ext(b);

                // ONE squaring chain for two consumers. `powers[j] = z0^(2^j)`:
                // the out-of-domain point takes the first `num_vars − bound` of
                // them and the domain check takes the last.
                let depth = next_domain.log_size();
                let mut powers = Vec::with_capacity(depth + 1);
                powers.push(z0);
                for j in 0..depth {
                    let squared = b.emul(powers[j], powers[j]);
                    powers.push(squared);
                }
                emit_require_out_of_domain(b, powers[depth], one);
                let ood_point: Vec<Ext> = powers[..shape.num_vars - bound].to_vec();

                transcript.absorb_ext(b, y0);
                emit_grind_check(b, transcript, grind_ood as u8, round.nonces.ood);
                let gamma = transcript.sample_ext(b);
                claim = b.emul_add(gamma, y0, claim);
                ood.push((gamma, ood_point, bound));

                emit_grind_check(b, transcript, grind_query as u8, round.nonces.query);
                Some(tree_auth(b, shape.caps[r + 1], round.next_cap, &lanes))
            }
            (None, None, None) => {
                assert!(round.next_cap.is_empty(), "the last round has no successor");
                transcript.absorb_ext(b, final_value);
                emit_grind_check(b, transcript, grind_query as u8, round.nonces.query);
                None
            }
            _ => panic!(
                "round {r}: a successor root and an out-of-domain value exist together, and \
                 exactly on the rounds the schedule gives a successor"
            ),
        };

        emit_query_phase(
            b,
            transcript,
            round,
            shape,
            r,
            &current_domain,
            &point,
            &current_tree,
            next_tree.as_ref(),
            final_value,
        );

        if let Some(tree) = next_tree {
            current_tree = tree;
        }
        alphas.extend(point);
        current_domain = next_domain;
    }

    // The accumulated weight: the caller's own, plus each out-of-domain claim's
    // batched `eq` over the challenges that came after it.
    let mut weight = weight_at(b, &alphas);
    for (gamma, point, bound) in &ood {
        let eq = emit_eq_eval(b, point, &alphas[*bound..]);
        weight = b.emul_add(*gamma, eq, weight);
    }

    emit_final_check(b, claim, weight, final_value, one);
}

/// How a tree's openings are checked: against its root lanes when its cap
/// height is 0 (today's emission, unchanged), else against its cap, hinted as
/// `cap` and authenticated against the root lanes HERE — the one place a
/// tree's [`CapCells`] are made.
fn tree_auth(
    b: &mut LfmBuilder,
    cap_height: usize,
    cap: &[WrapDigest],
    root_lanes: &[Felt; 4],
) -> TreeAuth {
    if cap_height == 0 {
        assert!(cap.is_empty(), "an uncapped tree carries no cap wires");
        TreeAuth::Root(*root_lanes)
    } else {
        assert_eq!(cap.len(), 1usize << cap_height, "a cap is 2^c digests");
        TreeAuth::Cap(CapCells::authenticate(
            b,
            cap,
            std::slice::from_ref(root_lanes),
        ))
    }
}

/// ★ `require_out_of_domain` (`whir_chain.rs:88-101`), emitted as a REFUSAL.
///
/// The host rejects `z0` whose `2^log_size`-th power is one, because such a
/// point lies inside the domain and constrains nothing beyond the in-domain
/// queries. Here `raised − 1` is inverted: `1/0` has no satisfying assignment,
/// so an in-domain `z0` has no execution.
///
/// ⚠ **Its own function because no honest fixture reaches it.** A transcript
/// draws an in-domain `z0` with negligible probability, so a chain-level
/// mutation that deletes this check passes every gate — measured, not assumed:
/// mutation CC removed it and all four chain tests stayed green. A refusal that
/// nothing can exercise is a check that cannot fail, and the cure is to make it
/// callable so a test can hand it the input the protocol never will.
pub fn emit_require_out_of_domain(b: &mut LfmBuilder, raised: Ext, one: Ext) {
    let gap = b.esub(raised, one);
    let _ = b.ediv(one, gap);
}

/// ★ The chain's last check: `final_value == claim · weight⁻¹`.
///
/// Inverting rather than cross-multiplying, and the difference is soundness
/// rather than taste. `final_value · weight == claim` is one row cheaper and
/// accepts EVERY `final_value` when `weight` and `claim` are both zero, where
/// the host returns `DegenerateEvaluationPoint` (`whir_chain.rs:1095-1102`).
///
/// ⚠ **Its own function for the same reason as above.** Weight and claim are
/// both zero only at a point a transcript does not produce, so mutation CD
/// replaced this with the cross-multiplied form and all four chain tests stayed
/// green. The gap is closed by calling this directly with the degenerate input.
pub fn emit_final_check(b: &mut LfmBuilder, claim: Ext, weight: Ext, final_value: Ext, one: Ext) {
    let inv = b.ediv(one, weight);
    let required = b.emul(claim, inv);
    b.assert_eq_ext(final_value, required);
}

/// `whir_round::verify` and `verify_final`: draw every query position, then
/// check every opening.
///
/// The draws come first and together, because that is the order
/// `sample_queries` takes (`whir_round.rs:68-76`) and the transcript is
/// sequential — interleaving them with the openings would change every
/// challenge after the first.
#[allow(clippy::too_many_arguments)]
fn emit_query_phase(
    b: &mut LfmBuilder,
    transcript: &mut WhirTranscript,
    round: &ChainRoundWires<'_>,
    shape: &ChainShape,
    r: usize,
    current_domain: &Domain<GoldilocksField>,
    alphas: &[Ext],
    current_tree: &TreeAuth,
    next_tree: Option<&TreeAuth>,
    final_value: Ext,
) {
    let depth = shape.current_depth(r);
    debug_assert_eq!(current_tree.cap_height(), shape.caps[r]);
    assert_eq!(round.current.len(), shape.num_queries);
    let queries: Vec<Vec<_>> = (0..shape.num_queries)
        .map(|_| transcript.sample_u64_pow2(b, depth))
        .collect();

    match (next_tree, shape.next_depth(r)) {
        (Some(next_tree), Some(next_depth)) => {
            let next_block = 1usize << shape.schedule[r + 1];
            assert_eq!(round.next.len(), shape.num_queries);
            for (q, bits) in queries.iter().enumerate() {
                let current = &round.current[q];
                let next = &round.next[q];
                current_tree.verify_opening(b, current.values, bits, current.siblings);
                // `leaf_and_slot`: the low `next_depth` bits index the successor
                // leaf and the high ones choose the slot inside it. Both bounds
                // are powers of two, so this is a partition of the bits.
                let (leaf_bits, slot_bits) = bits.split_at(next_depth);
                next_tree.verify_opening(b, next.values, leaf_bits, next.siblings);

                let folded =
                    emit_fold_coset(b, &block_ext(current.values), current_domain, bits, alphas);
                let claimed = emit_slot_mux(b, &block_ext(next.values), slot_bits);
                b.assert_eq_ext(folded, claimed);
                debug_assert_eq!(1usize << slot_bits.len(), next_block);
            }
        }
        _ => {
            for (q, bits) in queries.iter().enumerate() {
                let current = &round.current[q];
                current_tree.verify_opening(b, current.values, bits, current.siblings);
                let folded =
                    emit_fold_coset(b, &block_ext(current.values), current_domain, bits, alphas);
                b.assert_eq_ext(folded, final_value);
            }
        }
    }
}

/// A block as extension wires. A base block's values lift for free — a base
/// cell is `(v, 0, 0, 0)` — and the lift is sound because the leaf hash pinned
/// those lanes; [`super::whir_open`]'s module doc carries that obligation.
fn block_ext(values: BlockValues<'_>) -> Vec<Ext> {
    match values {
        BlockValues::Base(felts) => felts.iter().map(|f| f.as_ext()).collect(),
        BlockValues::Ext(cells) => cells.to_vec(),
    }
}

/// `values[slot]` where `slot` is given as bits, LOW first: a balanced mux,
/// `values.len() − 1` `Select` rows.
///
/// `Select` returns both arms for one row (`builder.rs:298-314`), so a level
/// that halves `2m` values into `m` costs `m` rows and the whole mux costs
/// `values.len() − 1`.
///
/// ⚠ The pairing is `(2t, 2t + 1)` and not `(t, t + half)`, because the bits
/// arrive low first. `t` against `t + half` differs in the HIGH bit, so pairing
/// that way while consuming the low bit first reads the index backwards — it
/// agrees at every palindromic slot and nowhere else, which is the shape of a
/// bug a small fixture hides.
fn emit_slot_mux(b: &mut LfmBuilder, values: &[Ext], bits: &[Bit]) -> Ext {
    assert_eq!(
        values.len(),
        1usize << bits.len(),
        "a slot mux covers the whole successor block"
    );
    let mut level: Vec<Ext> = values.to_vec();
    for bit in bits {
        level = level
            .chunks_exact(2)
            .map(|pair| {
                let (chosen, _) = b.select(*bit, pair[0].as_cell(), pair[1].as_cell());
                chosen.as_ext()
            })
            .collect();
    }
    level[0]
}

/// The half of a chain's cost that does not depend on the sponge's schedule.
///
/// The other half is [`chain_schedule_rows`] and the sum is [`chain_rows`].
pub fn chain_shape_rows(shape: &ChainShape) -> usize {
    chain_fixed_rows(shape) + chain_query_rows(shape)
}

// =============================================================================
// One chain's wires in an arena — MOVED OUT OF `whir_chain_tests` (V1g)
// =============================================================================
//
// These four were `pub(super)` inside the chain's own test module, which is
// `#[cfg(test)]`, so nothing in a production module could reach them. The
// assembled epoch verify (`whir_epoch`) is a production module that holds EIGHT
// chains plus the DECODE group's, and it has to hint their wires out of one
// arena; a second copy of this walk beside the one the chain suite gates would
// be exactly the drift the chain suite exists to prevent. So the walk moves
// here and the suite keeps gating it.
//
// Nothing about them changed but their visibility and `round_words` losing an
// `impl` block: the chain suite is the control that says so.

/// A current block's wires, in the field its round holds them in.
pub enum CurrentBlock {
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

/// ★ One chain's round wires, hinted and OWNED — because `ChainRoundWires`
/// borrows them.
///
/// Extracted from [`chain_program`] so a caller that emits SEVERAL chains in one
/// program — `stacked_eval`'s wrapper, one chain per stacked polynomial — builds
/// them from this walk rather than from a second copy of it. The chain suite is
/// the control that the extraction moved nothing.
pub struct RoundStorage {
    shape: ChainShape,
    sumchecks: Vec<Vec<Vec<Ext>>>,
    currents: Vec<Vec<(CurrentBlock, Vec<WrapDigest>)>>,
    nexts: Vec<Vec<(Vec<Ext>, Vec<WrapDigest>)>>,
    roots: Vec<Option<super::builder::Cell>>,
    oods: Vec<Option<Ext>>,
    nonces: Vec<RoundNonces>,
    /// Per round: the current tree's cap when the round owns it (round 0),
    /// and the successor's cap. Empty where the tree is uncapped.
    current_caps: Vec<Vec<WrapDigest>>,
    next_caps: Vec<Vec<WrapDigest>>,
}

impl RoundStorage {
    /// Words one chain's rounds occupy, which is what a caller placing several
    /// chains in one arena advances by.
    pub fn words(shape: &ChainShape) -> u32 {
        (0..shape.rounds()).map(|r| round_words(shape, r)).sum()
    }

    /// Hints every round's wires out of `arena`, starting at `base`, in the
    /// order [`push_round_words`] writes them.
    pub fn hint(
        b: &mut LfmBuilder,
        arena: super::instr::ArenaId,
        base: u32,
        shape: &ChainShape,
    ) -> Self {
        let mut sumchecks: Vec<Vec<Vec<Ext>>> = Vec::new();
        let mut currents: Vec<Vec<(CurrentBlock, Vec<WrapDigest>)>> = Vec::new();
        let mut nexts: Vec<Vec<(Vec<Ext>, Vec<WrapDigest>)>> = Vec::new();
        let mut roots: Vec<Option<super::builder::Cell>> = Vec::new();
        let mut oods: Vec<Option<Ext>> = Vec::new();
        let mut nonces: Vec<RoundNonces> = Vec::new();
        let mut current_caps: Vec<Vec<WrapDigest>> = Vec::new();
        let mut next_caps: Vec<Vec<WrapDigest>> = Vec::new();

        let mut at = base;
        for r in 0..shape.rounds() {
            let next_word = |b: &mut LfmBuilder, at: &mut u32| {
                let cell = b.hint_word(arena, *at);
                *at += 1;
                cell
            };
            let k = shape.schedule[r];
            let sumcheck: Vec<Vec<Ext>> = (0..k)
                .map(|_| (0..2).map(|_| next_word(b, &mut at).as_ext()).collect())
                .collect();
            // The nonces are FELTS: `append_bytes(&nonce.to_be_bytes())` is one
            // big-endian felt, which is how the grind absorbs them.
            let folding = b.hint_felt(arena, at);
            let ood_nonce = b.hint_felt(arena, at + 1);
            let query = b.hint_felt(arena, at + 2);
            at += 3;

            // W1: tree 0's cap, right after round 0's nonces (the words the
            // owner path carried after its siblings).
            let current_cap: Vec<WrapDigest> = if r == 0 {
                (0..cap_words(shape.caps[0]))
                    .map(|_| WrapDigest::from_cell(next_word(b, &mut at)))
                    .collect()
            } else {
                Vec::new()
            };
            let depth = shape.current_path(r);
            let block = 1usize << k;
            // ★ ROUND 0's current codeword is BASE on the host
            // (`whir_chain.rs:983`), so its block hashes ONE felt a value and
            // not three. Hinting it as extension wires would hash forty-eight
            // felts where the committer hashed sixteen and the root would never
            // match — which is exactly how this test first failed.
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
                            (0..block).map(|_| next_word(b, &mut at).as_ext()).collect(),
                        )
                    };
                    let path: Vec<WrapDigest> = (0..depth)
                        .map(|_| WrapDigest::from_cell(next_word(b, &mut at)))
                        .collect();
                    (values, path)
                })
                .collect();

            let (next_root, ood_value, next, next_cap) = match shape.next_depth(r) {
                Some(next_depth) => {
                    let nr = next_word(b, &mut at);
                    let ov = next_word(b, &mut at).as_ext();
                    // W1: the successor's cap, after its root and ood value.
                    let next_cap: Vec<WrapDigest> = (0..cap_words(shape.caps[r + 1]))
                        .map(|_| WrapDigest::from_cell(next_word(b, &mut at)))
                        .collect();
                    let next_depth = next_depth - shape.caps[r + 1];
                    let next_block = 1usize << shape.schedule[r + 1];
                    let next: Vec<(Vec<Ext>, Vec<WrapDigest>)> = (0..shape.num_queries)
                        .map(|_| {
                            let values: Vec<Ext> = (0..next_block)
                                .map(|_| next_word(b, &mut at).as_ext())
                                .collect();
                            let path: Vec<WrapDigest> = (0..next_depth)
                                .map(|_| WrapDigest::from_cell(next_word(b, &mut at)))
                                .collect();
                            (values, path)
                        })
                        .collect();
                    (Some(nr), Some(ov), next, next_cap)
                }
                None => (None, None, Vec::new(), Vec::new()),
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
            current_caps.push(current_cap);
            next_caps.push(next_cap);
        }
        assert_eq!(
            at - base,
            Self::words(shape),
            "the round walk and the word count are one derivation"
        );

        Self {
            shape: shape.clone(),
            sumchecks,
            currents,
            nexts,
            roots,
            oods,
            nonces,
            current_caps,
            next_caps,
        }
    }

    /// The query openings, current and successor, borrowing this storage.
    #[allow(clippy::type_complexity)]
    pub fn openings(&self) -> (Vec<Vec<QueryOpening<'_>>>, Vec<Vec<QueryOpening<'_>>>) {
        let current: Vec<Vec<QueryOpening<'_>>> = self
            .currents
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
        let next: Vec<Vec<QueryOpening<'_>>> = self
            .nexts
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
        (current, next)
    }

    /// The wires `emit_verify_weighted` takes, borrowing this storage and the
    /// openings built from it.
    pub fn wires<'a>(
        &'a self,
        current: &'a [Vec<QueryOpening<'a>>],
        next: &'a [Vec<QueryOpening<'a>>],
    ) -> Vec<ChainRoundWires<'a>> {
        (0..self.shape.rounds())
            .map(|r| ChainRoundWires {
                sumcheck: &self.sumchecks[r],
                next_root: self.roots[r],
                ood_value: self.oods[r],
                nonces: self.nonces[r],
                current: &current[r],
                next: &next[r],
                current_cap: &self.current_caps[r],
                next_cap: &self.next_caps[r],
            })
            .collect()
    }
}

pub fn round_words(shape: &ChainShape, r: usize) -> u32 {
    let k = shape.schedule[r];
    // The sumcheck's two evaluations a round, three nonces, and per query
    // the current block plus its path (to the cap, when the tree has one).
    let mut n = (2 * k + 3) as u32;
    let depth = shape.current_path(r);
    let block = 1usize << k;
    n += (shape.num_queries * (block + depth)) as u32;
    if r == 0 {
        // W1: tree 0's cap words.
        n += cap_words(shape.caps[0]) as u32;
    }
    if let Some(next_depth) = shape.next_depth(r) {
        // The successor root, its out-of-domain value, its cap, and per query
        // its block and path.
        n += 2 + cap_words(shape.caps[r + 1]) as u32;
        let next_block = 1usize << shape.schedule[r + 1];
        n += (shape.num_queries * (next_block + next_depth - shape.caps[r + 1])) as u32;
    }
    n
}

/// Words a tree's cap occupies in the arena: `2^c` one-word digests, none at
/// `c = 0` (an uncapped tree is checked against its root).
fn cap_words(cap: usize) -> usize {
    if cap == 0 { 0 } else { 1usize << cap }
}

/// One chain's round wires, in the order [`RoundStorage::hint`] reads them.
///
/// Split out of [`chain_arena`] for the same reason [`RoundStorage`] was split
/// out of [`chain_program`]: a program holding several chains fills one arena
/// with several of these, and a second copy of the order would be a second thing
/// to keep in step with the walk that reads it.
pub fn push_round_words(
    words: &mut Vec<LfmWord>,
    shape: &ChainShape,
    proof: &ChainProof<GoldilocksField, GoldilocksExtension>,
) {
    for (r, round) in proof.rounds.iter().enumerate() {
        for sc in &round.sumcheck {
            for e in &sc.evaluations {
                words.push(ext_word(e));
            }
        }
        for nonce in [round.nonces.folding, round.nonces.ood, round.nonces.query] {
            words.push([FE::from(nonce), FE::zero(), FE::zero(), FE::zero()]);
        }
        // W1: round 0 owns tree 0, whose cap rides at the end of its first
        // current path; it goes to the arena here and the path goes without it.
        let current_cap = if r == 0 { cap_words(shape.caps[0]) } else { 0 };
        push_openings(words, round, true, current_cap);
        if shape.next_depth(r).is_some() {
            words.push(commitment_to_digest(
                round.next_root.as_ref().expect("a successor root"),
            ));
            words.push(ext_word(
                round.ood_value.as_ref().expect("an out-of-domain value"),
            ));
            push_openings(words, round, false, cap_words(shape.caps[r + 1]));
        }
    }
}

/// One round's query openings, current or successor, block then path.
///
/// `owner_cap` is the number of cap words the side's FIRST opening carries at
/// the end of its path (the owner-path encoding, W1): they are written first,
/// ahead of every block, and the path is written without them. Zero at the
/// default. A malformed path is not repaired here: its words are written as
/// they are, and the arena's length (a verifier constant) refuses it.
fn push_openings(
    words: &mut Vec<LfmWord>,
    round: &ChainRound<GoldilocksField, GoldilocksExtension>,
    current: bool,
    owner_cap: usize,
) {
    fn side<V>(
        words: &mut Vec<LfmWord>,
        openings: &[multilinear::whir_commit::CosetOpening<V>],
        owner_cap: usize,
        value_word: impl Fn(&math::field::element::FieldElement<V>) -> LfmWord,
    ) where
        V: math::field::traits::IsField,
    {
        let owner_split = openings
            .first()
            .map_or(0, |o| o.proof.merkle_path.len().saturating_sub(owner_cap));
        if let Some(owner) = openings.first() {
            for node in &owner.proof.merkle_path[owner_split..] {
                words.push(commitment_to_digest(node));
            }
        }
        for (i, opening) in openings.iter().enumerate() {
            for v in &opening.values {
                words.push(value_word(v));
            }
            let path = if i == 0 {
                &opening.proof.merkle_path[..owner_split]
            } else {
                &opening.proof.merkle_path[..]
            };
            for node in path {
                words.push(commitment_to_digest(node));
            }
        }
    }
    match &round.openings {
        RoundOpenings::Base(p) => {
            if current {
                // A base value arrives as `(v, 0, 0, 0)`.
                side(words, &p.current, owner_cap, |v| {
                    [*v, FE::zero(), FE::zero(), FE::zero()]
                });
            } else {
                side(words, &p.next, owner_cap, ext_word);
            }
        }
        RoundOpenings::Extension(p) => {
            let openings = if current { &p.current } else { &p.next };
            side(words, openings, owner_cap, ext_word);
        }
    }
}
