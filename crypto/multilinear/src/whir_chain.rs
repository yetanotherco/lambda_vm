//! WHIR with its rounds chained: fold `k` variables at a time, committing the
//! successor codeword each round, so a query opens a block of `2^k` instead of
//! the whole message.
//!
//! The **committed** codeword is base-field, because a trace is. The extension
//! only enters at the first fold, so the largest structure in the proof — the
//! codeword and the Merkle leaves over it, `blowup` times the trace — costs a
//! third of what it would over the extension. That is why the first round's
//! openings have a different value type from every later one, and the type says
//! so rather than leaving it to a reader.
//!
//! [`whir_eval`](crate::whir_eval) is the one-round case, and there a block
//! *is* the message — which is what dominates its proof size and its verifier's
//! work. Here the sumcheck runs in groups of `k`: after each group the codeword
//! folds by `2^k`, the successor is committed **before** the queries are drawn,
//! and each query checks that folding a current block really lands on the
//! successor's value there. The last group has no successor — the message is a
//! constant by then, and that constant is sent.
//!
//! Each round also draws an **out-of-domain** point `z0` and takes the
//! successor's value there. Nothing about the domain constrains that value, so
//! answering it is what forces the committed word to be near a *single*
//! codeword rather than merely near the code — the in-domain queries alone
//! leave a prover room to be near several. The claim is folded into the weight
//! the next group carries, batched with a challenge, which is why the whole
//! thing runs on the weighted form:
//!
//! ```text
//! w_{r+1} = w_r(alpha_r, ·) + gamma_r·eq(ood_r, ·)
//! C_{r+1} = C_r'            + gamma_r·y0_r
//! ```
//!
//! Three challenges per round are worth redrawing for a cheating prover — the
//! folding randomness, the out-of-domain batching challenge and the query
//! positions — and [`GrindBits`] can put a [proof of work](crypto::grinding)
//! before each: retrying one costs `2^bits` hashes. Only the query grind raises
//! the proven minimum as placed. It sits right before the positions, which is
//! what lets a query count buy more soundness than its own bits. The folding
//! grind sits before the round's FIRST sumcheck message, so the first folding
//! challenge is redrawn by varying that message without grinding again. The
//! out-of-domain grind comes after the out-of-domain point, so the only
//! challenge it guards is the batching one, which has far more bits than the
//! target without any grind. So [`GrindBits::query_only`] loses no proven
//! bits, and with [`NonceLayout::Spent`] the proof carries one nonce a round.
//!
//! The claim is the weighted one, `Σ_x w(x)·f(x) = y`, so a stacked multi-point
//! claim chains just as an evaluation does.

use crypto::fiat_shamir::is_transcript::IsTranscript;
pub use crypto::merkle_tree::cap::CapPolicy;
use math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsPrimeField, IsSubFieldOf},
    },
    traits::AsBytes,
};

use crate::{
    Error,
    eq::{eq_eval, eq_mle},
    mle::Mle,
    narrow::HostColumn,
    poly::Composed,
    sumcheck::{self, RoundProof as SumcheckRoundProof},
    whir::{Domain, encode, fold_codeword_k, lift_coefficients},
    whir_commit::{Codeword, CodewordCommitment, Commitment, fold_coset, verify_opening_capped},
    whir_hash::{DeviceHashKey, GrindingDigest, WhirHash},
    whir_round::{
        self, GroupedRoundProof, RoundCaps, RoundCommitments, RoundConfig, RoundProof, TreeCheck,
    },
};

pub use crate::whir_commit::LeafOrder;

/// Refuses a format whose round-0 leaf order `H`'s trees do not take: the
/// coset order is defined for Poseidon1 trees alone — the production hash,
/// base and recursion alike — so under any other hash it is not a state a
/// prover or a verifier can reach. Called before a round-0 tree is built
/// ([`commit_stacked`]), before a chain is proved, and before a proof is read.
pub fn check_leaf_order<H: WhirHash>(format: &ChainFormat) -> Result<(), Error> {
    match format.leaves {
        LeafOrder::Natural => Ok(()),
        LeafOrder::Coset if H::DEVICE == DeviceHashKey::Poseidon1 => Ok(()),
        order => Err(Error::LeafOrderUnsupported {
            order,
            hash: H::NAME,
        }),
    }
}

/// `w(x)·f(x)`, the shape every group's sumcheck runs over.
fn product<E: IsField>(values: &[FieldElement<E>]) -> FieldElement<E> {
    &values[0] * &values[1]
}

/// The multilinear point where the univariate lift is `z0`.
///
/// [`lift_coefficients`] puts variable `j` on the monomial `X^(2^j)`, so
/// `F(z0) = f(z0, z0^2, z0^4, ..)`.
pub fn ood_point<E: IsField>(z0: &FieldElement<E>, num_vars: usize) -> Vec<FieldElement<E>> {
    let mut power = z0.clone();
    (0..num_vars)
        .map(|_| {
            let current = power.clone();
            power = power.square();
            current
        })
        .collect()
}

/// Rejects a point that landed inside `domain`, where it would constrain
/// nothing beyond the in-domain queries.
///
/// Sampling from the transcript makes this negligible; it is checked rather
/// than assumed.
fn require_out_of_domain<F, E>(z0: &FieldElement<E>, domain: &Domain<F>) -> Result<(), Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E>,
    E: IsField + Send + Sync + 'static,
{
    let mut power = z0.clone();
    for _ in 0..domain.log_size() {
        power = power.square();
    }
    if power == FieldElement::<E>::one() {
        return Err(Error::OodPointInDomain);
    }
    Ok(())
}

/// Proof of work over the transcript state, absorbed before the next challenge.
///
/// Retrying that challenge then costs `2^bits` hashes. Zero bits is a no-op, so
/// a caller that has not chosen its parameters yet pays nothing.
fn grind<E, T, H>(transcript: &mut T, bits: u8) -> Result<u64, Error>
where
    E: IsField + Send + Sync + 'static,
    T: IsTranscript<E>,
    H: WhirHash,
{
    if bits == 0 {
        return Ok(0);
    }
    let nonce =
        crypto::grinding::generate_nonce_maybe_gpu::<GrindingDigest<H>>(&transcript.state(), bits)
            .ok_or(Error::GrindingFailed { bits })?;
    transcript.append_bytes(&nonce.to_be_bytes());
    Ok(nonce)
}

/// The verifier's half: the nonce must pass against the same state, and it is
/// absorbed the same way.
fn check_grind<E, T, H>(transcript: &mut T, bits: u8, nonce: u64) -> Result<(), Error>
where
    E: IsField + Send + Sync + 'static,
    T: IsTranscript<E>,
    H: WhirHash,
{
    if bits == 0 {
        return Ok(());
    }
    if !crypto::grinding::is_valid_nonce::<GrindingDigest<H>>(&transcript.state(), nonce, bits) {
        return Err(Error::GrindingRejected { bits });
    }
    transcript.append_bytes(&nonce.to_be_bytes());
    Ok(())
}

/// A nonce slot the round does not spend: a grind of zero bits, or the last
/// round's out-of-domain slot. [`NonceLayout::Three`] carries it and never
/// reads it; [`NonceLayout::Spent`] has no such nonce, so the field standing in
/// for it must be 0. No transcript is touched either way.
fn check_unspent(layout: NonceLayout, nonce: u64) -> Result<(), Error> {
    match layout {
        NonceLayout::Spent if nonce != 0 => Err(Error::UnspentNonce),
        _ => Ok(()),
    }
}

/// One nonce slot, as the layout has it: [`check_grind`] where the grind spends
/// bits, [`check_unspent`] where it does not.
fn check_nonce<E, T, H>(
    transcript: &mut T,
    bits: u8,
    nonce: u64,
    layout: NonceLayout,
) -> Result<(), Error>
where
    E: IsField + Send + Sync + 'static,
    T: IsTranscript<E>,
    H: WhirHash,
{
    if bits == 0 {
        return check_unspent(layout, nonce);
    }
    check_grind::<E, T, H>(transcript, bits, nonce)
}

/// `w + gamma·eq`, the weight the next group carries.
fn batch_weight<E: IsField + 'static>(
    w: &Mle<E>,
    eq: &Mle<E>,
    gamma: &FieldElement<E>,
) -> Result<Mle<E>, Error> {
    Mle::new(
        w.evals()
            .iter()
            .zip(eq.evals())
            .map(|(a, b)| a + gamma * b)
            .collect(),
    )
}

/// Proof-of-work bits, in the three places a prover would want to redraw.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GrindBits {
    /// Before a group's folding challenges.
    pub folding: u8,
    /// Before the out-of-domain batching challenge.
    pub ood: u8,
    /// Before the query positions.
    pub query: u8,
}

impl GrindBits {
    /// The same count in all three places.
    pub const fn uniform(bits: u8) -> Self {
        Self {
            folding: bits,
            ood: bits,
            query: bits,
        }
    }

    /// Bits before the query positions only: the one grind that raises the
    /// proven minimum as placed (module header). The query count reads `query`
    /// alone, so it equals [`uniform`](Self::uniform)'s at the same bits.
    pub const fn query_only(bits: u8) -> Self {
        Self {
            folding: 0,
            ood: 0,
            query: bits,
        }
    }
}

/// Blowup, fold factor, query count and proof of work.
///
/// `format` is the proof FORMAT ([`ChainFormat`]: the W1 cap and W2 fold
/// levers); its default is today's format. Like the rest of the config it is
/// a verifier-side constant, never read from a proof. The fold schedule is
/// absorbed into the statement through [`ChainConfig::fold_word`], whose value
/// at the default is `log_folding` itself (today's bytes); the rest of the
/// format is not absorbed (`push_config` binds it as `_`): absorbing it would
/// move every transcript at the default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainConfig {
    /// `log2` of the code's inverse rate.
    pub log_blowup: usize,
    /// Variables folded per round. The last round takes whatever is left.
    pub log_folding: usize,
    /// Positions checked per round.
    pub num_queries: usize,
    /// Proof of work before each redrawable challenge.
    pub grind: GrindBits,
    /// The chain's proof format. [`ChainFormat::DEFAULT`] = today.
    pub format: ChainFormat,
}

/// The proof-format levers of a WHIR chain. Grouped so a literal
/// `ChainConfig` names the format in one line (`format: ChainFormat::DEFAULT`)
/// and a lever added later touches this struct only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ChainFormat {
    /// Merkle cap policy for the chain's commitment trees (W1). `Off` = today.
    pub cap: CapPolicy,
    /// Per-round fold schedule (W2). `Uniform` = today (`log_folding` every
    /// round, the remainder last).
    pub folds: WhirFolds,
    /// How wide a stacked polynomial may get (S2). [`StackVars::LEGACY`] =
    /// today (25).
    pub stack: StackVars,
    /// Which nonces a round carries (P2). [`NonceLayout::Three`] = today.
    pub nonces: NonceLayout,
    /// How a proof's tables argue their constraints and buses (D-BATCH).
    /// [`ArgueFormat::PerTable`] = today.
    pub argue: ArgueFormat,
    /// The Merkle cap policy of a hash whose trees are 4-ary (Poseidon1,
    /// D-WHIR-P1 D2), its heights in 4-ary levels: [`cap`](Self::cap) prices
    /// binary caps only. Read at arity 4 alone, so it cannot move a binary
    /// hash's bytes; `Off` = today.
    pub arity4_cap: CapPolicy,
    /// How a stack's polynomials are cut into chains (D-ONECHAIN, W-14).
    /// [`GroupFold::PerPoly`] = today.
    pub group: GroupFold,
    /// Where the round-0 trees put their leaves. [`LeafOrder::Natural`] =
    /// today.
    pub leaves: LeafOrder,
}

impl ChainFormat {
    /// Today's format: every lever off.
    pub const DEFAULT: Self = Self {
        cap: CapPolicy::Off,
        folds: WhirFolds::Uniform,
        stack: StackVars::LEGACY,
        nonces: NonceLayout::Three,
        argue: ArgueFormat::PerTable,
        arity4_cap: CapPolicy::Off,
        group: GroupFold::PerPoly,
        leaves: LeafOrder::Natural,
    };

    /// True when this is today's format (`Fixed(0)` counts as `Off`).
    pub fn is_default(&self) -> bool {
        self.cap.is_off()
            && self.folds == WhirFolds::Uniform
            && self.stack == StackVars::LEGACY
            && self.nonces == NonceLayout::Three
            && self.argue == ArgueFormat::PerTable
            && self.arity4_cap.is_off()
            && self.group == GroupFold::PerPoly
            && self.leaves == LeafOrder::Natural
    }

    /// The cap policy of trees with `arity` children per node:
    /// [`cap`](Self::cap) at arity 2, [`arity4_cap`](Self::arity4_cap) at 4.
    pub fn cap_policy_at(&self, arity: usize) -> CapPolicy {
        if arity == 4 {
            self.arity4_cap
        } else {
            self.cap
        }
    }
}

/// How a proof's tables argue their constraints and buses (D-BATCH,
/// `thoughts/zf/gap2/fix2/D-BATCH.md` §2).
///
/// [`PerTable`](Self::PerTable) is today's: each table runs its own LogUp-GKR,
/// its own constraint sumcheck and its own claim reduction.
/// [`Batched`](Self::Batched) runs one lockstep GKR per bin of tables, one
/// front-loaded constraint sumcheck over every table, and no claim reduction
/// for a table without shifted reads (`stark::multilinear_table::batched`).
///
/// ★ A FORMAT VALUE, NOT A HINT. The two produce different proof types, and a
/// verifier refuses a proof argued under the other variant: it reads the
/// variant from its own config, never from a proof. The bin cap is part of
/// the format because the bins are part of the transcript.
///
/// A verifier-side constant like the rest of [`ChainFormat`], and not absorbed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ArgueFormat {
    /// One argument per table. Today's format.
    #[default]
    PerTable,
    /// One argument per proof, the GKR trees in bins of at most
    /// `2^bin_log_cells` input cells each.
    Batched {
        /// `log2` of a bin's cap on `Σ 2^{k_T}`, `k_T` a tree's input
        /// variables. A tree at or over the cap sits alone.
        bin_log_cells: u8,
    },
}

impl ArgueFormat {
    /// The production cap: `2^26` input cells a bin. D-BATCH §4.4 planned
    /// `2^27` with `2^26` its fallback (§8.2); on the block's 15 epochs bins of
    /// `2^27` overran the card (12 tree builds refused, FAST 347) and bins of
    /// `2^26` refused none.
    pub const BATCHED: Self = Self::Batched { bin_log_cells: 26 };
}

/// Which proof-of-work nonces a round's encoding carries (P2's format half).
///
/// A round has three nonce slots, folding, out-of-domain and query, and spends
/// the ones whose grind has bits. At zero bits a grind searches, reads and
/// absorbs nothing, and the last round's out-of-domain slot has no grind at all
/// (there is no successor to batch).
///
/// ★ WHY IT IS A FORMAT FIELD. Under [`Three`](Self::Three) an unspent slot is
/// carried and never read, so any value in it verifies: an unbound field.
/// [`GrindBits::query_only`] leaves two of those a round. [`Spent`](Self::Spent)
/// takes them out of the format. The in-guest encoding
/// (`prover::lfm::whir_chain`'s arena) has no word for an unspent nonce, so
/// there is nothing to set. The host [`RoundNonces`] keeps its three fields, so
/// both layouts share one proof type and `Three` keeps its bytes, and the
/// verifier refuses a nonzero value in a field the layout does not carry
/// ([`Error::UnspentNonce`]).
///
/// A verifier-side constant like the rest of [`ChainFormat`], and not absorbed:
/// the grind bits, which decide which slots are spent, are (the statement's
/// grind trailer).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum NonceLayout {
    /// Three nonces a round, whatever the bits. Today's format.
    #[default]
    Three,
    /// Only the nonces the round spends: one a round under
    /// [`GrindBits::query_only`].
    Spent,
}

/// How wide one stacked polynomial may get, in variables: the stack cap (S2).
///
/// The knob between proof size and peak memory, a resource tradeoff rather
/// than an optimization. Every extra variable halves the number of stacked
/// polynomials — and so the number of chains, which is most of a WHIR proof
/// and of the work of verifying one in-guest — and doubles the biggest single
/// allocation: a polynomial of `n` variables is encoded over
/// `2^(n + log_blowup)` field elements, 1 GiB in the base field at 25 and
/// blowup 4, 4 GiB at 27.
///
/// ★ A FORMAT VALUE, NOT A HINT. A group's layout is a function of its shapes
/// and this cap, so the prover, the host verifier and the in-guest emitters
/// all build it from the config they are given, and a proof stacked under one
/// cap does not verify under another. Nothing reads it from a proof.
///
/// Constructed only through [`StackVars::new`], so a cap of 0 or wider than
/// [`StackVars::WIDEST`] is not a value a config can hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StackVars(u8);

impl StackVars {
    /// Today's cap: one 1 GiB base codeword.
    pub const LEGACY: Self = Self(25);

    /// The widest cap a format may ask for. 27 is the widest measured to fit
    /// a 32 GiB card (block 25368371, three runs: the argument's reserved peak
    /// 24,218 MiB of a 25,688 MiB budget, the device peak at most 30,736 MiB);
    /// at 28 the argument's reservations land 2–2.6 GiB over the budget in the
    /// heaviest epochs. The device commit needs the NTT's grid split past 26.
    pub const WIDEST: usize = 27;

    /// `None` outside `1..=WIDEST`.
    pub const fn new(n: usize) -> Option<Self> {
        if n >= 1 && n <= Self::WIDEST {
            Some(Self(n as u8))
        } else {
            None
        }
    }

    pub const fn get(self) -> usize {
        self.0 as usize
    }
}

impl Default for StackVars {
    fn default() -> Self {
        Self::LEGACY
    }
}

/// See [`WHIR_CAP_IMPLEMENTED`]. S2's stack is in: the layouts the prover, the
/// host verifier and the in-guest emitters build, the device commit of a
/// 2^29-point codeword (the NTT and Möbius grid split) and the argument's
/// ledger (the rooms given back across it).
pub const WHIR_STACK_IMPLEMENTED: bool = true;

/// Which WHIR format levers THIS build implements. A lever that is only
/// parsed must not be selectable (see `stark::proof::options::
/// MERKLE_CAP_IMPLEMENTED`). Each flag is flipped in the commit that
/// makes the lever real.
///
/// W1 (the Merkle cap) is real: host prover and verifier ([`ChainConfig::
/// tree_caps`], the owner-path encoding), the device (`paths_and_cap`), and
/// the in-guest verifier and its cost model (`prover::lfm::whir_chain`).
pub const WHIR_CAP_IMPLEMENTED: bool = true;

/// The widest fold any round of a chain may take.
///
/// The stack is tested up to it and no further: the GPU commit/fold parity
/// (`math-cuda` `whir_commit`/`whir_fold`, k = 6) and the in-guest fold
/// emitter (`lfm::whir_fold_tests`, k = 5 and 6). `k0 = 7` loses on in-guest
/// instructions, so nothing above 6 is opened.
pub const MAX_FOLD: usize = 6;

/// The per-round fold schedule of a chain (W2).
///
/// ★ Why a FIRST fold and not a list. A config serves chains of every height
/// (`chain_config` takes the tallest stack, and each chain folds its own
/// `num_vars`), so a per-round list would have to say what a shorter chain
/// does with it. The lever is the first fold alone —
/// tree 0 is the only base-field tree, opened `Q` times rather than `2Q`, and
/// every variable it takes shortens every later tree — so the schedule is
/// "`k0`, then today's uniform walk", a function of `(k0, log_folding,
/// num_vars)` at every height. There is no DP.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum WhirFolds {
    /// `log_folding` variables every round, the remainder last. Today's format.
    #[default]
    Uniform,
    /// The first round folds `k0` variables (all of them when the chain has
    /// fewer); every later round is today's walk: `log_folding`, the remainder
    /// last. `first5` / `first6`.
    First(FirstFold),
}

/// See [`WHIR_CAP_IMPLEMENTED`]. W2 is in: the host chain, the statement word,
/// `agrees_with`, the production config, the GPU parity at k = 6 and the
/// in-guest gates at k = 5 and 6.
pub const WHIR_FOLDS_IMPLEMENTED: bool = true;

/// A first-round fold, `1 ..= MAX_FOLD`. Constructed only through
/// [`FirstFold::new`], so a fold of 0 or wider than the tested stack is not a
/// value a config can hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FirstFold(u8);

impl FirstFold {
    /// `None` outside `1..=MAX_FOLD`.
    pub const fn new(k: usize) -> Option<Self> {
        if k >= 1 && k <= MAX_FOLD {
            Some(Self(k as u8))
        } else {
            None
        }
    }

    pub const fn get(self) -> usize {
        self.0 as usize
    }
}

/// The top bit of a non-uniform [`ChainConfig::fold_word`]. A uniform word is
/// `log_folding`, far below it, so no non-default word equals a default one.
pub const FOLD_WORD_TAG: u64 = 1 << 63;

/// The bit of [`ChainConfig::fold_word`] that says [`GroupFold::Indexed`],
/// always beside [`FOLD_WORD_TAG`]. No other word sets it: a `First` word's
/// fields sit at bits 52+, 48 and the low nibble.
pub const FOLD_WORD_INDEXED: u64 = 1 << 47;

/// The bit of [`ChainConfig::fold_word`] that says [`LeafOrder::Coset`],
/// always beside [`FOLD_WORD_TAG`]. No other word sets it: a `First` word's
/// fields sit at bits 52+, 48 and the low nibble, and [`FOLD_WORD_INDEXED`]
/// is bit 47.
pub const FOLD_WORD_COSET: u64 = 1 << 46;

/// How a stack's polynomials are opened (D-ONECHAIN, the ledger's W-14).
///
/// [`PerPoly`](Self::PerPoly) is today's: one chain per stacked polynomial.
/// [`Indexed`](Self::Indexed) cuts them into consecutive runs of at most
/// [`MAX_INDEX_POLYS`] (`stacked_eval::index_chains`) and opens a run of
/// `P' ≥ 2` with ONE chain at `n + m` variables, `m = ⌈log2 P'⌉`
/// ([`index_vars`]):
/// - round 0 runs `k0 + m` sumcheck rounds, the `m` index challenges `ρ` LAST;
/// - it opens the run's `P'` round-0 trees at ONE position set, combines each
///   query's blocks as `Σ_p eq(ρ, p)·block_p` ([`eq_index`]) and folds that as
///   today's `2^k0` block;
/// - rounds ≥ 1 are today's chain on the one successor.
///
/// A run of one polynomial is today's chain exactly. The chain is today's at
/// `n + m` variables restricted to round-0 words whose absent slots are zero
/// (D-ONECHAIN §3.3, Lemma B), which needs `m ≤ 2` and `n + m + log_blowup`
/// within the field's two-adicity ([`check_index_domain`]).
///
/// ★ A FORMAT VALUE, NOT A HINT. A verifier-side constant like the rest of
/// [`ChainFormat`], bound into the statement through [`ChainConfig::fold_word`]
/// ([`FOLD_WORD_INDEXED`]), and never read from a proof.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GroupFold {
    /// One chain per stacked polynomial. Today's format.
    #[default]
    PerPoly,
    /// One chain per run of at most [`MAX_INDEX_POLYS`] polynomials.
    Indexed,
}

/// The most index variables a grouped chain binds. `m = 3` drops the
/// conjectured round-0 fold term below 128 bits (127.393 at a 2^27 stack,
/// REV-B2-JUDGE §5), and past it Lemma B's virtual domain stops existing.
pub const MAX_INDEX_VARS: usize = 2;

/// The most polynomials one grouped chain opens: `2^MAX_INDEX_VARS`.
pub const MAX_INDEX_POLYS: usize = 1 << MAX_INDEX_VARS;

/// `m = ⌈log2 P'⌉`: the index variables a chain over `polys` polynomials
/// binds. 0 for one polynomial, which is today's chain.
pub const fn index_vars(polys: usize) -> usize {
    polys.next_power_of_two().trailing_zeros() as usize
}

/// `eq(ρ, p)`, with `ρ_0` against the MOST significant of `p`'s `ρ.len()`
/// bits: index variable `y_0` is the high bit of the slot, as the index rounds
/// bind it (the run's tables concatenated in slot order, variable 0 the
/// highest).
///
/// ★ THE ONE CONVENTION. The prover's factors after the index rounds, its
/// round-0 successor, the verifier's combine and the verifier's weight all
/// read it, so they cannot disagree (REV-B2-JUDGE B1-3).
pub fn eq_index<E: IsField>(rho: &[FieldElement<E>], p: usize) -> FieldElement<E> {
    let m = rho.len();
    rho.iter()
        .enumerate()
        .fold(FieldElement::<E>::one(), |acc, (i, r)| {
            if (p >> (m - 1 - i)) & 1 == 1 {
                acc * r
            } else {
                acc * (FieldElement::<E>::one() - r)
            }
        })
}

/// Refuses an index width whose virtual chain has no domain: Lemma B's
/// `H_{n+m+lb}` exists only when `n + m + log_blowup ≤` the field's two-adicity
/// (REV-B2-A C5). A verifier constant check, run before anything is read.
pub fn check_index_domain(
    num_vars: usize,
    index_vars: usize,
    log_blowup: usize,
    two_adicity: usize,
) -> Result<(), Error> {
    if num_vars + index_vars + log_blowup > two_adicity {
        return Err(Error::GroupedShape(GroupedShape::IndexDomain {
            num_vars,
            index_vars,
            log_blowup,
            two_adicity,
        }));
    }
    Ok(())
}

/// Test only: verifier faults a grouped-chain test switches on, one at a time,
/// to show that each use of the index convention is load-bearing
/// (REV-B2-JUDGE B1-3): with any of them an honest proof must be refused.
/// Thread-local, so a test that sets one moves no other test's verifier.
#[cfg(test)]
pub(crate) mod grouped_faults {
    /// Which piece of the verifier reads the index wrongly.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub(crate) enum Fault {
        #[default]
        None,
        /// The round-0 combine reads `ρ` reversed (`ρ_0 ↔ ρ_1` at `m = 2`).
        SwapRhoInCombine,
        /// The final weight reads `ρ` reversed.
        SwapRhoInWeight,
        /// The final weight drops `eq(ρ, p)`: `Σ_p w_p` at the data point.
        DropIndexFactor,
        /// The final weight gives each absent slot `p ≥ P'` a weight of 1.
        AbsentSlotWeightOne,
    }

    std::thread_local! {
        static FAULT: std::cell::Cell<Fault> = const { std::cell::Cell::new(Fault::None) };
    }

    pub(crate) fn get() -> Fault {
        FAULT.with(std::cell::Cell::get)
    }

    /// Runs `f` under `fault`, restoring the previous one after.
    pub(crate) fn with<R>(fault: Fault, f: impl FnOnce() -> R) -> R {
        struct Restore(Fault);
        impl Drop for Restore {
            fn drop(&mut self) {
                FAULT.with(|c| c.set(self.0));
            }
        }
        let _restore = Restore(FAULT.with(|c| c.replace(fault)));
        f()
    }
}

/// The shape a grouped opening must have, refused by name before any block is
/// indexed or zipped (REV-B2-JUDGE B1-2). A lenient count would let a prover
/// choose, after the positions are drawn, whether a slot reads its committed
/// block or nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupedShape {
    /// A stacked proof's chain count is not the derived split's.
    ChainCount { expected: usize, got: usize },
    /// A chain asked to open no polynomial, or more than [`MAX_INDEX_POLYS`].
    RunWidth { polys: usize },
    /// The virtual chain's domain does not exist ([`check_index_domain`]).
    IndexDomain {
        num_vars: usize,
        index_vars: usize,
        log_blowup: usize,
        two_adicity: usize,
    },
    /// Round 0 of a grouped chain carries other than `k0 + m` messages.
    Round0Messages { expected: usize, got: usize },
    /// A round's openings are the grouped variant where they must not be, or
    /// not it where they must be: round 0 of a chain over `P' ≥ 2` and nowhere
    /// else.
    Variant { round: usize },
    /// Other than `Q` queries.
    Queries { expected: usize, got: usize },
    /// A query carries other than one block per round-0 tree.
    BlocksPerQuery {
        query: usize,
        expected: usize,
        got: usize,
    },
    /// A block of other than `2^k0` values.
    BlockWidth {
        query: usize,
        tree: usize,
        expected: usize,
        got: usize,
    },
    /// Other than one successor block a query (none on a final round).
    SuccessorBlocks { expected: usize, got: usize },
}

impl ChainConfig {
    /// Parameters for a security target, in the **same regime the univariate
    /// prover uses**: the Johnson bound, `proximity = 1 − √rate − 1/300`, so
    /// each query buys `−log2(1 − proximity)` bits. Grinding on the query phase
    /// is subtracted from what the queries have to buy, as it is there.
    ///
    /// Every round is spot-checked independently, so their errors add: the
    /// target carries `log2(rounds)` of margin on top (a union bound).
    ///
    /// **What this is not.** WHIR's own analysis is per-round and gets to use
    /// the out-of-domain point, which this ignores entirely. So it is the
    /// conservative mirror of the parameters the repo already ships — read it as
    /// "no weaker than the FRI ones next door" — and **not** a soundness proof
    /// about this protocol. Doing that analysis is what would let the query
    /// count come down.
    pub fn with_security(
        log_blowup: usize,
        log_folding: usize,
        num_vars: usize,
        security_bits: u8,
        grind: GrindBits,
    ) -> Self {
        Self::with_security_folds(
            log_blowup,
            log_folding,
            WhirFolds::Uniform,
            num_vars,
            security_bits,
            grind,
        )
    }

    /// [`with_security`](Self::with_security) under a fold schedule.
    ///
    /// ★ THE ONE Q RULE. The query count is one number for every chain the
    /// config serves, so the union bound is charged the WORST round count any
    /// chain of at most `num_vars` variables has under `folds`. For `Uniform`
    /// that is `ceil(num_vars / log_folding)` — today's count, exactly — and a
    /// first fold `k0 >= log_folding` never has more rounds than that at any
    /// height, so it never raises Q (`first5`/`first6` keep 112 at 25).
    ///
    /// The rate is `2^-log_blowup` in every round whatever the schedule (the
    /// domain loses `k_r` bits as the message loses `k_r` variables), so the
    /// per-query bits do not move; only `rounds` does. The disclaimer on
    /// [`with_security`](Self::with_security) applies unchanged.
    pub fn with_security_folds(
        log_blowup: usize,
        log_folding: usize,
        folds: WhirFolds,
        num_vars: usize,
        security_bits: u8,
        grind: GrindBits,
    ) -> Self {
        let mut config = Self {
            log_blowup,
            log_folding,
            num_queries: 0,
            grind,
            format: ChainFormat {
                folds,
                ..ChainFormat::DEFAULT
            },
        };
        let rounds = (1..=num_vars)
            .map(|m| config.rounds(m))
            .max()
            .unwrap_or(0)
            .max(1);
        // ★ Integers, not `f64`. The arithmetic and its provenance are in
        // [`crate::query_count`]; what matters here is that the count a
        // verifier has to reproduce no longer needs floating point to
        // reproduce it, and that the answers did not move — the shipped
        // posture's 110 / 112 / 113 are pinned in both places.
        config.num_queries =
            crate::query_count::num_queries(log_blowup, rounds, security_bits, grind.query);
        config
    }

    /// Rounds a chain of `num_vars` variables runs: `schedule(num_vars).len()`.
    pub fn rounds(&self, num_vars: usize) -> usize {
        self.schedule(num_vars).len()
    }

    /// The statement's fold word: what the three host absorbs and the LFM's
    /// `push_config` write where they wrote `log_folding`.
    ///
    /// - `Uniform` → `log_folding`: `4u64` at the default, today's bytes.
    /// - `First(k0)` → `FOLD_WORD_TAG | log_folding << 52 | 1 << 48 | k0`:
    ///   a prefix encoding with a one-entry prefix (tail
    ///   `log_folding`, length 1, the fold in the low nibble).
    ///
    /// - [`GroupFold::Indexed`] → either word above with [`FOLD_WORD_TAG`] and
    ///   [`FOLD_WORD_INDEXED`] set, so a grouped statement shares no word with
    ///   a per-polynomial one under any fold schedule.
    /// - [`LeafOrder::Coset`] → any word above with [`FOLD_WORD_TAG`] and
    ///   [`FOLD_WORD_COSET`] set: the round-0 trees' roots and paths follow the
    ///   order, so a statement under one order shares no word with the other.
    ///
    /// Every chain's schedule is a function of this word and its own
    /// `num_vars`, which the statement already binds, so binding the word
    /// binds every schedule — including the heights where two policies give
    /// the same schedule (`first6` and `uniform4` at `num_vars <= 4`), where
    /// only the word tells the proofs apart.
    pub fn fold_word(&self) -> u64 {
        let word = match self.format.folds {
            WhirFolds::Uniform => self.log_folding as u64,
            WhirFolds::First(k0) => {
                FOLD_WORD_TAG
                    | ((self.log_folding as u64 & 0x7ff) << 52)
                    | (1 << 48)
                    | k0.get() as u64
            }
        };
        let word = match self.format.group {
            GroupFold::PerPoly => word,
            GroupFold::Indexed => word | FOLD_WORD_TAG | FOLD_WORD_INDEXED,
        };
        match self.format.leaves {
            LeafOrder::Natural => word,
            LeafOrder::Coset => word | FOLD_WORD_TAG | FOLD_WORD_COSET,
        }
    }

    /// The sumcheck messages each round of a chain over `num_vars`-variable
    /// polynomials carries when it binds `index_vars` index variables:
    /// [`schedule`](Self::schedule) with round 0 widened by them.
    ///
    /// ★ (k0, m, tail), NEVER `schedule(n + m)` (REV-B2-JUDGE J1). The blocks
    /// round 0 opens are still `2^k0` wide — the index is folded by combining
    /// the trees' blocks, not by a wider block — so the folds, the trees, their
    /// caps, the round count and with it Q are the per-polynomial chain's.
    /// `schedule(n + m)` would fold index variables in round 1, another
    /// protocol; `First(k0 + m)` would open blocks no tree has.
    pub fn index_schedule(&self, num_vars: usize, index_vars: usize) -> Vec<usize> {
        let mut messages = self.schedule(num_vars);
        if let Some(first) = messages.first_mut() {
            *first += index_vars;
        }
        messages
    }

    /// The Merkle cap height of each of the chain's `R` commitment trees, tree
    /// `t` being the one round `t` opens as its current codeword (W1).
    ///
    /// Tree `t` has depth `D_t − k_t` (its leaves are round `t`'s domain
    /// folded by that round's `k`) and is opened `Q` times when `t = 0` (round
    /// 0's current openings) and `2Q` times after (round `t − 1`'s successor
    /// openings and round `t`'s current ones, the last tree included). The
    /// height is [`CapPolicy::height`] of those two public numbers, so the
    /// prover, the host verifier and the in-guest emitter derive the same
    /// heights from the config alone. All zero at the default.
    pub fn tree_caps(&self, num_vars: usize) -> Vec<usize> {
        self.tree_caps_at(num_vars, 2)
    }

    /// [`tree_caps`](Self::tree_caps) for trees of `arity` children per node:
    /// the policy at that arity ([`ChainFormat::cap_policy_at`]) over that
    /// arity's levels ([`crate::whir_commit::tree_cap_height`]). At arity 2 it
    /// is `tree_caps`; the chain's prover and verifier call it with their
    /// hash's arity.
    pub fn tree_caps_at(&self, num_vars: usize, arity: usize) -> Vec<usize> {
        let policy = self.format.cap_policy_at(arity);
        let mut domain_log = num_vars + self.log_blowup;
        self.schedule(num_vars)
            .iter()
            .enumerate()
            .map(|(t, &k)| {
                domain_log -= k;
                let openings = if t == 0 {
                    self.num_queries
                } else {
                    2 * self.num_queries
                };
                crate::whir_commit::tree_cap_height(policy, openings, domain_log, arity)
            })
            .collect()
    }

    /// Variables folded in each round: `log_folding` until the remainder.
    ///
    /// Under [`WhirFolds::First`] the first round takes `k0` (or everything,
    /// when there is less), and the rest is this same walk.
    pub fn schedule(&self, num_vars: usize) -> Vec<usize> {
        let step = self.log_folding.max(1);
        let mut left = num_vars;
        let mut out = Vec::new();
        if let WhirFolds::First(k0) = self.format.folds {
            let take = k0.get().min(left);
            if take > 0 {
                out.push(take);
                left -= take;
            }
        }
        while left > 0 {
            let take = step.min(left);
            out.push(take);
            left -= take;
        }
        out
    }
}

/// A round's proof-of-work nonces.
///
/// Zero where the config asks for no bits, and the out-of-domain one is unused
/// on the last round. Under [`NonceLayout::Spent`] those unspent fields are not
/// part of the format and must stay zero; see there.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct RoundNonces {
    pub folding: u64,
    pub ood: u64,
    pub query: u64,
}

/// The blocks one round opens.
///
/// Only the first round's current codeword is base-field; every later one has
/// already been folded with an extension challenge. The successor is always in
/// the extension.
///
/// ⚠ APPEND ONLY. A variant's serde index is its position, and rkyv lays the
/// enum out at its widest variant: [`BaseGrouped`](Self::BaseGrouped) is last
/// and, like the others, two `Vec`s, so neither encoding of a proof that does
/// not use it moves (REV-B2-JUDGE B1-6).
#[derive(
    Clone,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub enum RoundOpenings<F: IsField, E: IsField> {
    Base(RoundProof<F, E>),
    Extension(RoundProof<E, E>),
    /// Round 0 of a [`GroupFold::Indexed`] chain over `P' ≥ 2` polynomials,
    /// and nowhere else: per query one base block per round-0 tree, then the
    /// successor block ([`GroupedRoundProof`]).
    BaseGrouped(GroupedRoundProof<F, E>),
}

/// One round of the chain.
#[derive(
    Clone,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct ChainRound<F: IsField, E: IsField> {
    /// The sumcheck rounds this group consumed.
    pub sumcheck: Vec<SumcheckRoundProof<E>>,
    /// The successor's root. `None` on the last round, where the folded message
    /// is a constant and is sent instead.
    pub next_root: Option<Commitment>,
    /// The successor's value at the out-of-domain point. `None` on the last
    /// round: there is no successor left to be near a codeword.
    pub ood_value: Option<FieldElement<E>>,
    pub nonces: RoundNonces,
    pub openings: RoundOpenings<F, E>,
}

/// A chained proof that `Σ_x w(x)·f(x) = y`.
#[derive(
    Clone,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct ChainProof<F: IsField, E: IsField> {
    pub rounds: Vec<ChainRound<F, E>>,
    /// The constant the codeword folds to — the prover's claim for `f(α)`.
    pub final_value: FieldElement<E>,
}

impl<F: IsField, E: IsField> ChainProof<F, E> {
    /// Codeword elements the openings carry. What chaining is for.
    pub fn opened_elements(&self) -> usize {
        self.rounds
            .iter()
            .map(|r| match &r.openings {
                RoundOpenings::Base(p) => block_size(&p.current) + block_size::<E>(&p.next),
                RoundOpenings::Extension(p) => {
                    block_size::<E>(&p.current) + block_size::<E>(&p.next)
                }
                RoundOpenings::BaseGrouped(p) => {
                    p.current.iter().map(|q| block_size(q)).sum::<usize>()
                        + block_size::<E>(&p.next)
                }
            })
            .sum()
    }
}

fn block_size<V: IsField>(openings: &[crate::whir_commit::CosetOpening<V>]) -> usize {
    openings.iter().map(|c| c.values.len()).sum()
}

/// Commits `f`, blocked for the first round's fold.
///
/// The codeword stays in the base field: nothing extension-valued has touched
/// the trace yet, and this is the biggest allocation in the proof.
/// Commits one polynomial's codeword.
///
/// `transient` says whether this commitment has to promise the device room its
/// own commit and opening take — true when it stands alone, false when a
/// caller committing several has promised that turn once for all of them.
/// A polynomial to commit and open, which may not exist yet.
///
/// A stacked polynomial is its columns written at their offsets and zeros in
/// between. Assembling it here is a pass over every byte the device is about to
/// read anyway, so the parts are handed over as they are and whoever needs a
/// whole one builds it — which on the device path is nobody.
pub struct Stacked<'a, F: IsField> {
    /// `(column, offset in elements)`; a column's values are read on the host
    /// only by a host path ([`HostColumn`]).
    pub parts: Vec<(&'a dyn HostColumn<F>, usize)>,
    /// The same parts as `(index into the epoch's columns, offset)`, when the
    /// card already holds them — then the scatter is a copy at device
    /// bandwidth instead of the trace crossing the bus again.
    pub resident: Option<(&'a crate::gpu::ResidentColumns, Vec<(usize, usize)>)>,
    pub num_vars: usize,
}

impl<F: IsField + 'static> Stacked<'_, F> {
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// The polynomial itself, assembled here. Only the host paths ask — a
    /// device writes the parts where they go and never sees a whole one.
    pub fn assemble(&self) -> Result<Mle<F>, Error> {
        let mut buffer = vec![FieldElement::<F>::zero(); 1usize << self.num_vars];
        for (column, offset) in &self.parts {
            let column = column.host();
            buffer[*offset..*offset + column.len()].clone_from_slice(column.evals());
        }
        Mle::new(buffer)
    }
}

/// A whole polynomial is a stacked one of a single part at offset zero.
pub fn commit<F, H>(
    f: &Mle<F>,
    config: &ChainConfig,
    transient: bool,
) -> Result<(CodewordCommitment<F, H>, Domain<F>), Error>
where
    F: IsFFTField + IsPrimeField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
{
    commit_stacked(
        &Stacked {
            parts: vec![(f as &dyn HostColumn<F>, 0)],
            resident: None,
            num_vars: f.num_vars(),
        },
        config,
        transient,
    )
}

pub fn commit_stacked<F, H>(
    f: &Stacked<'_, F>,
    config: &ChainConfig,
    transient: bool,
) -> Result<(CodewordCommitment<F, H>, Domain<F>), Error>
where
    F: IsFFTField + IsPrimeField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
{
    check_leaf_order::<H>(&config.format)?;
    let num_vars = f.num_vars();
    let schedule = config.schedule(num_vars);
    let first = schedule.first().copied().unwrap_or(0);
    let domain = Domain::<F>::new(num_vars + config.log_blowup)?;
    // On a device the codeword stays there: the chain folds it and opens a
    // handful of its values, and it is the biggest array the proof holds.
    // The card places its leaves in natural order only, so a round-0 tree in
    // another order is built on the host.
    let attempt = match &f.resident {
        _ if config.format.leaves != LeafOrder::Natural => None,
        Some((store, parts)) => crate::gpu::commit_resident(
            store,
            parts,
            num_vars,
            config.log_blowup,
            first,
            transient,
            H::DEVICE,
        ),
        None => crate::gpu::commit_parts(
            &f.parts,
            num_vars,
            config.log_blowup,
            first,
            transient,
            H::DEVICE,
        ),
    };
    let commitment = match attempt {
        Some((codeword, nodes)) => CodewordCommitment::from_device(codeword, nodes, first)?,
        // ⚠ COUNTED, because this arm is otherwise silent. The device declining
        // is not a slower path to the same place: the codeword is assembled,
        // lifted and encoded here, and the commitment then holds that codeword
        // AND its node array on the host until the proof ends. An epoch that
        // fills the card partway through lands here for every commit after,
        // and the only visible symptoms are host memory and a utilisation
        // figure — neither of which names the cause.
        None => {
            crate::gpu::note_host_fallback();
            let codeword = encode::<F, F>(&lift_coefficients(&f.assemble()?), &domain)?;
            match config.format.leaves {
                LeafOrder::Natural => CodewordCommitment::from_codeword(codeword, first)?,
                leaves => CodewordCommitment::from_codeword_on_host_in(codeword, first, leaves)?,
            }
        }
    };
    Ok((commitment, domain))
}

/// The codeword [`commit_stacked`] commits, with no tree: on the card when the
/// card would have committed it, on the host otherwise. What a retired
/// commitment is revived over (`stacked_eval::RetiredStack::revive`); the
/// values are the committed ones because the transform is the same one.
pub fn encode_stacked<F>(
    f: &Stacked<'_, F>,
    config: &ChainConfig,
    transient: bool,
) -> Result<Codeword<F>, Error>
where
    F: IsFFTField + IsPrimeField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
{
    let num_vars = f.num_vars();
    let attempt = match &f.resident {
        Some((store, parts)) => {
            crate::gpu::encode_resident(store, parts, num_vars, config.log_blowup, transient)
        }
        None => crate::gpu::encode_parts(&f.parts, num_vars, config.log_blowup, transient),
    };
    match attempt {
        Some(codeword) => {
            // The transform is queued on the codeword's stream; the opening
            // needs it done, and a stopwatch around this call should see it.
            if !codeword.synchronize() {
                return Err(Error::DeviceFailed {
                    stage: "re-encoding a retired codeword",
                });
            }
            Ok(Codeword::Device(codeword))
        }
        None => {
            crate::gpu::note_host_fallback();
            let domain = Domain::<F>::new(num_vars + config.log_blowup)?;
            Ok(Codeword::Host(encode::<F, F>(
                &lift_coefficients(&f.assemble()?),
                &domain,
            )?))
        }
    }
}

/// The product of a chain's two tables, which is what its rounds evaluate —
/// and what sizes a device session's slot file.
pub(crate) fn opening_program<E: IsField + 'static>() -> Result<crate::program::Program<E>, Error> {
    let mut builder = crate::program::Builder::<E>::new();
    let weight = builder.var(0);
    let message = builder.var(1);
    let root = builder.mul(weight, message);
    builder.finish(root)
}

/// The two tables a chain's rounds fold: the weight it carries and the message
/// it is opening.
///
/// A group of rounds folds them, the group after that reads what is left, and
/// they are each the width of a stacked polynomial — so on a device they stay
/// there and only the round's few field elements cross the bus. The flow
/// around them is the same either way: this is the only thing that differs.
enum Factors<F: IsField + IsSubFieldOf<E>, E: IsField> {
    Host {
        weight: Mle<E>,
        message: Mle<E>,
        field: core::marker::PhantomData<F>,
    },
    Device(crate::gpu::OpeningFactors),
    /// Round 0 of a grouped chain, before it runs: each polynomial's
    /// `(weight, message)` tables, on the host. Its first [`rounds`]
    /// (Self::rounds) call runs them in lockstep, binds the index, and leaves
    /// one [`Host`](Self::Host) pair behind ([`lockstep_rounds`]).
    Lockstep {
        parts: Vec<(Mle<E>, Mle<E>)>,
        index_vars: usize,
    },
    /// The same on a card: each polynomial's opening there. Its first
    /// [`rounds`](Self::rounds) call runs them in lockstep, binds the index,
    /// and leaves one [`Device`](Self::Device) opening behind
    /// ([`crate::gpu::GroupedFactors::rounds`]) — the host's steps, so the
    /// host's messages and tables.
    DeviceLockstep {
        parts: crate::gpu::GroupedFactors,
        index_vars: usize,
        num_vars: usize,
    },
}

impl<F, E> Factors<F, E>
where
    F: IsField + IsSubFieldOf<E> + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<E>: Send + Sync,
{
    fn program() -> Result<crate::program::Program<E>, Error> {
        opening_program::<E>()
    }

    fn new(f: &Mle<F>, weight: Mle<E>) -> Result<Self, Error> {
        if let Some(device) = crate::gpu::open_on_device(f, &weight, &Self::program()?) {
            return Ok(Self::Device(device));
        }
        // The sumcheck's factors have to share a field, so the message is
        // lifted here. The codeword is not, which is where the size is.
        let message = Mle::new(
            f.evals()
                .iter()
                .map(|v| v.clone().to_extension::<E>())
                .collect(),
        )?;
        Ok(Self::Host {
            weight,
            message,
            field: core::marker::PhantomData,
        })
    }

    /// The same from a weight given as shares: on a device they are written
    /// straight into its buffer, and the host builds the table only if none
    /// takes them.
    ///
    /// `on_device` says whether the codeword being opened is on one. When it
    /// is, the host building the tables is a fallback and is counted
    /// ([`crate::gpu::open_host_fallbacks`]); when it is not, it is the path.
    fn from_shares(
        f: &Stacked<'_, F>,
        shares: &[crate::stacked_eval::WeightShare<'_, E>],
        n_stack: usize,
        on_device: bool,
    ) -> Result<Self, Error> {
        if let Some(device) = crate::gpu::open_shared(f, shares, n_stack, &Self::program()?) {
            return Ok(Self::Device(device));
        }
        if on_device {
            crate::gpu::note_open_host_fallback(n_stack);
        }
        // Only here does a stacked polynomial have to exist on the host.
        Self::new(
            &f.assemble()?,
            crate::stacked_eval::weight_table(shares, n_stack)?,
        )
    }

    fn num_vars(&self) -> usize {
        match self {
            Self::Host { message, .. } => message.num_vars(),
            Self::Device(device) => device.num_vars(),
            Self::Lockstep { parts, .. } => parts.first().map_or(0, |(_, f)| f.num_vars()),
            Self::DeviceLockstep { num_vars, .. } => *num_vars,
        }
    }

    /// One group of `rounds` rounds, leaving the factors folded on them.
    fn rounds<T: IsTranscript<E>>(
        &mut self,
        rounds: usize,
        transcript: &mut T,
    ) -> Result<sumcheck::RoundGroup<E>, Error> {
        match self {
            Self::Host {
                weight, message, ..
            } => {
                let polys = vec![
                    core::mem::replace(weight, Mle::new(vec![FieldElement::<E>::zero()])?),
                    core::mem::replace(message, Mle::new(vec![FieldElement::<E>::zero()])?),
                ];
                let mut poly = Composed::new(polys, product::<E>, 2)?;
                let out = sumcheck::prove_rounds(&mut poly, rounds, transcript)?;
                let mut parts = poly.into_polys();
                *message = parts.pop().expect("two factors");
                *weight = parts.pop().expect("two factors");
                Ok(out)
            }
            Self::Device(device) => {
                let (proofs, challenges, _) = device.rounds(rounds, 2, |evaluations| {
                    for value in evaluations {
                        transcript.append_field_element(value);
                    }
                    transcript.sample_field_element()
                })?;
                Ok((proofs, challenges))
            }
            Self::Lockstep { parts, index_vars } => {
                let index_vars = *index_vars;
                let (proofs, challenges, weight, message) =
                    lockstep_rounds(core::mem::take(parts), rounds, index_vars, transcript)?;
                *self = Self::Host {
                    weight,
                    message,
                    field: core::marker::PhantomData,
                };
                Ok((proofs, challenges))
            }
            Self::DeviceLockstep { .. } => {
                let Self::DeviceLockstep {
                    parts, index_vars, ..
                } = core::mem::replace(
                    self,
                    Self::Lockstep {
                        parts: Vec::new(),
                        index_vars: 0,
                    },
                )
                else {
                    return Err(Error::NoVariablesLeft);
                };
                let (proofs, challenges, combined) =
                    parts.rounds(rounds, index_vars, |evaluations| {
                        for value in evaluations {
                            transcript.append_field_element(value);
                        }
                        transcript.sample_field_element()
                    })?;
                *self = Self::Device(combined);
                Ok((proofs, challenges))
            }
        }
    }

    fn evaluate_message(&mut self, point: &[FieldElement<E>]) -> Result<FieldElement<E>, Error> {
        match self {
            Self::Host { message, .. } => message.evaluate(point),
            Self::Device(device) => device.evaluate_message(point),
            // Round 0's rounds turn lockstep factors into one pair before any
            // out-of-domain point exists.
            Self::Lockstep { .. } | Self::DeviceLockstep { .. } => Err(Error::NoVariablesLeft),
        }
    }

    /// The successor a `levels`-level fold of `codeword` would make, encoded
    /// from the message these rounds bound — on a device, when the factors and
    /// the codeword are both there, unless `LAMBDA_VM_WHIR_ENCODE_SUCCESSOR=0`.
    /// `None` is the fold's turn.
    fn encode_successor<C: IsField>(
        &mut self,
        codeword: &Codeword<C>,
        levels: usize,
    ) -> Option<Codeword<E>> {
        match (self, codeword.device()) {
            (Self::Device(device), Some(held)) => {
                device.encode_successor(held, levels).map(Codeword::Device)
            }
            _ => None,
        }
    }

    /// `weight += gamma·eq(point, ·)`, the weight the next group carries.
    fn add_scaled_eq(
        &mut self,
        point: &[FieldElement<E>],
        gamma: &FieldElement<E>,
    ) -> Result<(), Error> {
        match self {
            Self::Host { weight, .. } => {
                *weight = batch_weight(weight, &eq_mle(point)?, gamma)?;
                Ok(())
            }
            Self::Device(device) => device.add_scaled_eq(point, gamma),
            Self::Lockstep { .. } | Self::DeviceLockstep { .. } => Err(Error::NoVariablesLeft),
        }
    }
}

/// Round 0 of a grouped chain's sumcheck, on the host (D-ONECHAIN §2.1, §2.5).
///
/// - `rounds` data rounds over the run's `P'` pairs IN LOCKSTEP: each round's
///   message is the sum of theirs and one challenge binds them all, since
///   `Σ_y W·F` over the slots is `Σ_p w_p·f_p`.
/// - Then the `m = index_vars` index rounds over the bound tables, laid out
///   slot by slot (`y` the high variables, absent slots zero): `ρ`.
///
/// Returns the `rounds + m` messages and challenges, and the pair the chain
/// carries on: `Σ_p eq(ρ, p)·(w_p, f_p)` at the data challenges, built with
/// [`eq_index`], the convention the verifier's combine and weight read. The
/// index rounds' own folded tables are that same pair when the conventions
/// agree, and are dropped; a disagreement would show as an honest proof that
/// does not verify.
#[allow(clippy::type_complexity)]
fn lockstep_rounds<E, T>(
    parts: Vec<(Mle<E>, Mle<E>)>,
    rounds: usize,
    index_vars: usize,
    transcript: &mut T,
) -> Result<
    (
        Vec<SumcheckRoundProof<E>>,
        Vec<FieldElement<E>>,
        Mle<E>,
        Mle<E>,
    ),
    Error,
>
where
    E: IsField + Send + Sync + 'static,
    FieldElement<E>: Send + Sync,
    T: IsTranscript<E>,
{
    use crate::poly::SumcheckPolynomial;
    let mut polys = parts
        .into_iter()
        .map(|(weight, message)| Composed::new(vec![weight, message], product::<E>, 2))
        .collect::<Result<Vec<_>, Error>>()?;
    let mut proofs = Vec::with_capacity(rounds + index_vars);
    let mut challenges = Vec::with_capacity(rounds + index_vars);
    for _ in 0..rounds {
        let mut sent = vec![FieldElement::<E>::zero(); 2];
        for poly in &polys {
            for (total, value) in sent
                .iter_mut()
                .zip(sumcheck::round_evaluations(poly, 2, false))
            {
                *total += value;
            }
        }
        for value in &sent {
            transcript.append_field_element(value);
        }
        let r: FieldElement<E> = transcript.sample_field_element();
        for poly in &mut polys {
            poly.fix_first_variable(&r)?;
        }
        proofs.push(SumcheckRoundProof { evaluations: sent });
        challenges.push(r);
    }

    let tables: Vec<(Mle<E>, Mle<E>)> = polys
        .into_iter()
        .map(|poly| {
            let mut factors = poly.into_polys();
            match (factors.pop(), factors.pop()) {
                (Some(message), Some(weight)) => Ok((weight, message)),
                _ => Err(Error::EmptyPolynomial),
            }
        })
        .collect::<Result<_, Error>>()?;
    let width = tables.first().map_or(1, |(weight, _)| weight.len());
    let slots = 1usize << index_vars;
    let mut weight = Vec::with_capacity(width * slots);
    let mut message = Vec::with_capacity(width * slots);
    for (w, f) in &tables {
        weight.extend_from_slice(w.evals());
        message.extend_from_slice(f.evals());
    }
    weight.resize(width * slots, FieldElement::<E>::zero());
    message.resize(width * slots, FieldElement::<E>::zero());
    let mut index = Composed::new(vec![Mle::new(weight)?, Mle::new(message)?], product::<E>, 2)?;
    let (index_proofs, rho) = sumcheck::prove_rounds(&mut index, index_vars, transcript)?;
    drop(index);

    let eqs: Vec<FieldElement<E>> = (0..tables.len()).map(|p| eq_index(&rho, p)).collect();
    let combine = |pick: fn(&(Mle<E>, Mle<E>)) -> &Mle<E>| -> Result<Mle<E>, Error> {
        let mut out = vec![FieldElement::<E>::zero(); width];
        for (table, eq) in tables.iter().zip(&eqs) {
            for (acc, value) in out.iter_mut().zip(pick(table).evals()) {
                *acc += eq * value;
            }
        }
        Mle::new(out)
    };
    let weight = combine(|t| &t.0)?;
    let message = combine(|t| &t.1)?;
    proofs.extend(index_proofs);
    challenges.extend(rho);
    Ok((proofs, challenges, weight, message))
}

/// What a round folds: the codeword and the commitment that answers for it.
///
/// Only the first round's is base-field. Folding it with an extension challenge
/// is what lifts it, so every later round is `Extension`.
enum Current<'a, F: IsField + 'static, E: IsField + 'static, H: WhirHash>
where
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    Base(&'a CodewordCommitment<F, H>),
    /// Round 0 of a grouped chain: the run's round-0 commitments, in slot
    /// order.
    Grouped(&'a [&'a CodewordCommitment<F, H>]),
    Extension(CodewordCommitment<E, H>),
}

/// Proves `f(z) = y`.
pub fn prove<F, E, T, H>(
    f: &Mle<F>,
    z: &[FieldElement<E>],
    commitment: &CodewordCommitment<F, H>,
    domain: &Domain<F>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<ChainProof<F, E>, Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    H: WhirHash,
{
    prove_weighted::<F, E, T, H>(f, eq_mle(z)?, commitment, domain, config, transcript)
}

/// The same for a weight given as the shares of a stacked polynomial's
/// columns, which a device writes into its own buffer and the host
/// materializes only if none does.
#[allow(clippy::too_many_arguments)]
pub fn prove_shared<F, E, T, H>(
    f: &Stacked<'_, F>,
    shares: &[crate::stacked_eval::WeightShare<'_, E>],
    n_stack: usize,
    commitment: &CodewordCommitment<F, H>,
    domain: &Domain<F>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<ChainProof<F, E>, Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    H: WhirHash,
{
    let factors =
        Factors::<F, E>::from_shares(f, shares, n_stack, commitment.codeword().device().is_some())?;
    prove_with_factors::<F, E, T, H>(
        f.num_vars(),
        factors,
        Current::Base(commitment),
        domain,
        config,
        transcript,
    )
}

/// One chain over a run of `P'` stacked polynomials ([`GroupFold::Indexed`],
/// D-ONECHAIN §2.1): `polys[p]` committed as `commitments[p]`, its weight
/// given as `shares[p]`. The claim is the sum of the polynomials' claims.
///
/// Round 0 runs its data rounds over the `P'` pairs in lockstep and then the
/// `m` index rounds ([`lockstep_rounds`]), folds each round-0 codeword and
/// combines the folds with `eq(ρ, p)` into the one successor, and opens the
/// `P'` trees at one position set ([`whir_round::prove_grouped`]). Every later
/// round is today's.
///
/// `2 ≤ P' ≤ MAX_INDEX_POLYS`; a run of one is [`prove_shared`]. Round 0 runs
/// on a card when it takes every polynomial's opening
/// ([`crate::gpu::open_grouped`]), on the host otherwise — counted as a
/// fallback when the codewords are on a card
/// ([`crate::gpu::note_grouped_host_fallback`]). The successor is encoded from
/// the combined bound message when the factors and codewords are on a card,
/// as a per-polynomial chain's is, and folded and combined otherwise
/// ([`fold_grouped`]).
#[allow(clippy::too_many_arguments)]
pub fn prove_grouped_shared<F, E, T, H>(
    polys: &[Stacked<'_, F>],
    shares: &[Vec<crate::stacked_eval::WeightShare<'_, E>>],
    n_stack: usize,
    commitments: &[&CodewordCommitment<F, H>],
    domain: &Domain<F>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<ChainProof<F, E>, Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    H: WhirHash,
{
    let width = commitments.len();
    if !(2..=MAX_INDEX_POLYS).contains(&width) || polys.len() != width || shares.len() != width {
        return Err(Error::GroupedShape(GroupedShape::RunWidth { polys: width }));
    }
    check_index_domain(
        n_stack,
        index_vars(width),
        config.log_blowup,
        F::TWO_ADICITY as usize,
    )?;
    let factors = match crate::gpu::open_grouped(polys, shares, n_stack, &opening_program::<E>()?) {
        Some(parts) => Factors::DeviceLockstep {
            parts,
            index_vars: index_vars(width),
            num_vars: n_stack,
        },
        None => {
            if commitments.iter().any(|c| c.codeword().device().is_some()) {
                crate::gpu::note_grouped_host_fallback(width, n_stack);
            }
            // Only here do the run's stacked polynomials and weights exist on
            // the host.
            let parts = polys
                .iter()
                .zip(shares)
                .map(|(f, shares)| -> Result<(Mle<E>, Mle<E>), Error> {
                    let message = Mle::new(
                        f.assemble()?
                            .evals()
                            .iter()
                            .map(|v| v.clone().to_extension::<E>())
                            .collect(),
                    )?;
                    Ok((crate::stacked_eval::weight_table(shares, n_stack)?, message))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            Factors::Lockstep {
                parts,
                index_vars: index_vars(width),
            }
        }
    };
    prove_with_factors::<F, E, T, H>(
        n_stack,
        factors,
        Current::Grouped(commitments),
        domain,
        config,
        transcript,
    )
}

/// Proves `Σ_x w(x)·f(x) = y` for a weight the verifier can evaluate itself.
pub fn prove_weighted<F, E, T, H>(
    f: &Mle<F>,
    weight: Mle<E>,
    commitment: &CodewordCommitment<F, H>,
    domain: &Domain<F>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<ChainProof<F, E>, Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    H: WhirHash,
{
    let factors = Factors::<F, E>::new(f, weight)?;
    prove_with_factors::<F, E, T, H>(
        f.num_vars(),
        factors,
        Current::Base(commitment),
        domain,
        config,
        transcript,
    )
}

/// The chain itself, over factors that are wherever they are. `current` is
/// round 0's commitment, or a grouped chain's commitments
/// ([`Current::Grouped`]); `num_vars` is a polynomial's, without the index.
fn prove_with_factors<F, E, T, H>(
    num_vars: usize,
    mut factors: Factors<F, E>,
    mut current: Current<'_, F, E, H>,
    domain: &Domain<F>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<ChainProof<F, E>, Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    H: WhirHash,
{
    check_leaf_order::<H>(&config.format)?;
    let round0: &[&CodewordCommitment<F, H>] = match &current {
        Current::Base(held) => std::slice::from_ref(held),
        Current::Grouped(held) => held,
        Current::Extension(_) => &[],
    };
    if let Some(held) = round0
        .iter()
        .find(|held| held.leaves() != config.format.leaves)
    {
        return Err(Error::LeafOrderMismatch {
            expected: config.format.leaves,
            got: held.leaves(),
        });
    }
    // One chain, counted — arm F's identity is per chain, and a group can open
    // several (`stacked_eval::prove` runs one per commitment).
    crate::whir_split::bump(&crate::whir_split::CHAIN_COUNT);
    let schedule = config.schedule(num_vars);
    // One cap height per tree, in the hash's tree levels; all zero at the
    // default format.
    let caps = config.tree_caps_at(num_vars, crate::whir_hash::arity::<F, H>());
    // The codeword comes out of the commitment rather than being encoded
    // again: it is the same array, and the NTT is not cheap.
    let mut current_domain = domain.clone();

    let mut rounds = Vec::with_capacity(schedule.len());
    let mut final_value = FieldElement::<E>::zero();

    for (r, &k) in schedule.iter().enumerate() {
        // The loop's OWN wall, so the six are measured against the thing that
        // contains them rather than against `open_groups` two layers up. What
        // sits between this and `open_groups` — `Factors::from_shares`, the
        // weights, the stacked polys, the domain clone, the proof assembled
        // after the last round — is reported as `setup_tail`, named rather
        // than left as a gap for a tolerance to swallow.
        let __wc_round = crate::whir_split::mark();
        // One per iteration, so arm F can derive the `open_many` calls this
        // loop should have made (`2R − 1` per chain) from the run's own count.
        crate::whir_split::bump(&crate::whir_split::ROUND_COUNT);
        // ── the round's six slots, under `LAMBDA_VM_BASE_SPLIT=1` ──
        // They partition the round, so `open_groups - Σ(six)` is loop overhead
        // and nothing else. All three grinds share one slot: they are the same
        // 20-bit search and a lever on one is a lever on all three.
        let __wc_g = crate::whir_split::mark();
        let mut nonces = RoundNonces {
            folding: grind::<E, T, H>(transcript, config.grind.folding)?,
            ..RoundNonces::default()
        };
        crate::whir_split::add(&crate::whir_split::GRIND, __wc_g);

        let __wc_sc = crate::whir_split::mark();
        let (sumcheck_rounds, challenges) = factors.rounds(k, transcript)?;
        crate::whir_split::add(&crate::whir_split::SUMCHECK, __wc_sc);
        // A grouped round 0 also bound its index: the challenges past the
        // first `k` are `ρ`, and only the first `k` fold a block.
        let (alphas, rho) = challenges.split_at(k.min(challenges.len()));

        // The fold lands in the extension whichever field it started in, so
        // the field is the only thing the two cases differ in — and a codeword
        // the device holds is folded where it is. The first one, the widest,
        // is encoded from the message the rounds just bound instead: the same
        // codeword (`LAMBDA_VM_WHIR_ENCODE_SUCCESSOR=0` folds it).
        let __wc_fold = crate::whir_split::mark();
        let (folded, folded_domain) = match &current {
            Current::Base(held) => match factors.encode_successor(held.codeword(), k) {
                Some(encoded) => {
                    let mut folded_domain = current_domain.clone();
                    for _ in 0..k {
                        folded_domain = folded_domain.squared()?;
                    }
                    (encoded, folded_domain)
                }
                None => fold_held::<F, F, E>(held.codeword(), &current_domain, alphas)?,
            },
            Current::Grouped(held) => match held
                .first()
                .and_then(|first| factors.encode_successor(first.codeword(), k))
            {
                // The run's combined bound message encoded: the successor
                // `Σ_p eq(ρ, p)·Fold_α(C_p)` is, the fold and the encoding
                // being linear and equal polynomial by polynomial.
                Some(encoded) => {
                    let mut folded_domain = current_domain.clone();
                    for _ in 0..k {
                        folded_domain = folded_domain.squared()?;
                    }
                    (encoded, folded_domain)
                }
                None => fold_grouped::<F, E, H>(held, &current_domain, alphas, rho)?,
            },
            Current::Extension(held) => {
                fold_held::<F, E, E>(held.codeword(), &current_domain, alphas)?
            }
        };
        crate::whir_split::add(&crate::whir_split::FOLD, __wc_fold);

        // The successor is committed before the queries are drawn, so it cannot
        // be chosen to match them.
        let __wc_cf = crate::whir_split::mark();
        let next = match schedule.get(r + 1) {
            Some(&next_k) => {
                let next = commit_folded::<E, H>(folded, next_k)?;
                transcript.append_bytes(&next.root());
                Some(next)
            }
            None => {
                final_value = first_value::<E>(&folded)?;
                transcript.append_field_element(&final_value);
                None
            }
        };
        crate::whir_split::add(&crate::whir_split::COMMIT_FOLDED, __wc_cf);
        let next_root = next.as_ref().map(|c| c.root());

        // Out of domain: a value the queries cannot vouch for, so answering it
        // pins the successor to one codeword.
        // ⛔ TWO OOD WINDOWS, NOT ONE SPANNING WINDOW. The out-of-domain block
        // contains a grind, and the grind belongs to GRIND — so a single
        // window opened before it and closed after would count that time
        // TWICE, once in each slot, and the six would sum to more than the
        // `open_groups` wall that contains them. Arm E read exactly that on the
        // first two real runs: a NEGATIVE remainder of -3.6s and then -4.0s.
        // Slots that partition must not nest; these two abut the grind instead.
        let ood_value = if next.is_some() {
            let __wc_ood_a = crate::whir_split::mark();
            let z0: FieldElement<E> = transcript.sample_field_element();
            require_out_of_domain::<F, E>(&z0, &folded_domain)?;
            let point = ood_point(&z0, factors.num_vars());
            let y0 = factors.evaluate_message(&point)?;
            transcript.append_field_element(&y0);
            crate::whir_split::add(&crate::whir_split::OOD, __wc_ood_a);

            let __wc_g2 = crate::whir_split::mark();
            nonces.ood = grind::<E, T, H>(transcript, config.grind.ood)?;
            crate::whir_split::add(&crate::whir_split::GRIND, __wc_g2);

            let __wc_ood_b = crate::whir_split::mark();
            let gamma: FieldElement<E> = transcript.sample_field_element();
            factors.add_scaled_eq(&point, &gamma)?;
            crate::whir_split::add(&crate::whir_split::OOD, __wc_ood_b);
            Some(y0)
        } else {
            None
        };

        let __wc_g3 = crate::whir_split::mark();
        nonces.query = grind::<E, T, H>(transcript, config.grind.query)?;
        crate::whir_split::add(&crate::whir_split::GRIND, __wc_g3);
        let round_config = RoundConfig {
            num_queries: config.num_queries,
            log_folding: k,
        };
        let __wc_q = crate::whir_split::mark();
        // Tree `r` is opened under `caps[r]`, and carries its cap on its first
        // opening in proof order: round 0's first current opening for tree 0,
        // round `r − 1`'s first successor opening for every later tree.
        let round_caps = RoundCaps {
            current: caps[r],
            current_owner: r == 0,
            next: caps.get(r + 1).copied().unwrap_or(0),
        };
        let openings =
            match (&current, &next) {
                (Current::Base(held), Some(next)) => RoundOpenings::Base(whir_round::prove(
                    *held,
                    next,
                    &round_config,
                    round_caps,
                    transcript,
                )?),
                (Current::Base(held), None) => RoundOpenings::Base(final_openings::<F, E, T, H>(
                    held,
                    &round_config,
                    round_caps,
                    transcript,
                )?),
                (Current::Extension(held), Some(next)) => RoundOpenings::Extension(
                    whir_round::prove(held, next, &round_config, round_caps, transcript)?,
                ),
                (Current::Extension(held), None) => {
                    RoundOpenings::Extension(final_openings::<E, E, T, H>(
                        held,
                        &round_config,
                        round_caps,
                        transcript,
                    )?)
                }
                (Current::Grouped(held), next) => {
                    RoundOpenings::BaseGrouped(whir_round::prove_grouped(
                        held,
                        next.as_ref(),
                        &round_config,
                        round_caps,
                        transcript,
                    )?)
                }
            };
        crate::whir_split::add(&crate::whir_split::QUERIES, __wc_q);

        rounds.push(ChainRound {
            sumcheck: sumcheck_rounds,
            next_root,
            ood_value,
            nonces,
            openings,
        });
        crate::whir_split::add(&crate::whir_split::ROUND, __wc_round);
        if let Some(next) = next {
            current = Current::Extension(next);
        }
        current_domain = folded_domain;
    }

    Ok(ChainProof {
        rounds,
        final_value,
    })
}

/// Folds a codeword wherever it is, leaving the result where it was.
fn fold_held<F, C, N>(
    codeword: &Codeword<C>,
    domain: &Domain<F>,
    alphas: &[FieldElement<N>],
) -> Result<(Codeword<N>, Domain<F>), Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<C> + IsSubFieldOf<N> + 'static,
    C: IsField + IsSubFieldOf<N> + 'static,
    N: IsField + 'static,
    FieldElement<C>: AsBytes + Sync + Send,
{
    if let Some(folded) = codeword
        .device()
        .and_then(|device| device.fold(domain.generator(), alphas))
    {
        let mut folded_domain = domain.clone();
        for _ in alphas {
            folded_domain = folded_domain.squared()?;
        }
        return Ok((Codeword::Device(folded), folded_domain));
    }
    // A codeword the device holds has no copy here: if the device turned the
    // fold down, this round cannot happen anywhere.
    let values = codeword.host().ok_or(Error::DeviceFailed {
        stage: "codeword fold",
    })?;
    let (folded, folded_domain) = fold_codeword_k::<F, C, N>(values, domain, alphas)?;
    Ok((Codeword::Host(folded), folded_domain))
}

/// Round 0's successor in a grouped chain: each round-0 codeword folded by the
/// data challenges where it is, the folds combined with `eq(ρ, p)`
/// ([`eq_index`]) — on the card when every fold landed there
/// ([`crate::gpu::combine_codewords`]), on the host otherwise, a card fold
/// brought home first. Block `j` of it is the fold of `Σ_p eq(ρ, p)·block_p(j)`,
/// which is what the verifier checks (D-ONECHAIN §3.3, Lemma B item 5).
fn fold_grouped<F, E, H>(
    held: &[&CodewordCommitment<F, H>],
    domain: &Domain<F>,
    alphas: &[FieldElement<E>],
    rho: &[FieldElement<E>],
) -> Result<(Codeword<E>, Domain<F>), Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    H: WhirHash,
{
    let eqs: Vec<FieldElement<E>> = (0..held.len()).map(|p| eq_index(rho, p)).collect();
    let mut folds = Vec::with_capacity(held.len());
    let mut folded_domain = domain.clone();
    for commitment in held {
        let (folded, at) = fold_held::<F, F, E>(commitment.codeword(), domain, alphas)?;
        folds.push(folded);
        folded_domain = at;
    }
    if let Some(on_card) = folds
        .iter()
        .map(Codeword::device)
        .collect::<Option<Vec<_>>>()
        && let Some(combined) = crate::gpu::combine_codewords(&on_card, &eqs)
    {
        return Ok((Codeword::Device(combined), folded_domain));
    }
    let mut combined: Vec<FieldElement<E>> = Vec::new();
    for (folded, eq) in folds.iter().zip(&eqs) {
        let brought;
        let values: &[FieldElement<E>] = match folded {
            Codeword::Host(values) => values,
            Codeword::Device(device) => {
                brought = device.values::<E>().ok_or(Error::DeviceFailed {
                    stage: "a grouped round-0 fold brought home",
                })?;
                &brought
            }
        };
        if combined.is_empty() {
            combined = values.iter().map(|v| eq * v).collect();
        } else {
            for (acc, v) in combined.iter_mut().zip(values) {
                *acc += eq * v;
            }
        }
    }
    Ok((Codeword::Host(combined), folded_domain))
}

/// Commits a folded codeword where it is.
fn commit_folded<N, H>(
    codeword: Codeword<N>,
    log_folding: usize,
) -> Result<CodewordCommitment<N, H>, Error>
where
    N: IsField + 'static,
    H: WhirHash,
    FieldElement<N>: AsBytes + Sync + Send,
{
    match codeword {
        Codeword::Host(values) => CodewordCommitment::from_codeword(values, log_folding),
        Codeword::Device(device) => {
            let nodes = device
                .commit(log_folding, H::DEVICE)
                .ok_or(Error::DeviceFailed { stage: "fold tree" })?;
            CodewordCommitment::from_device(device, nodes, log_folding)
        }
    }
}

/// The value the last fold leaves behind.
fn first_value<N>(codeword: &Codeword<N>) -> Result<FieldElement<N>, Error>
where
    N: IsField + 'static,
{
    match codeword {
        Codeword::Host(values) => values.first().cloned().ok_or(Error::EmptyPolynomial),
        Codeword::Device(device) => device.first().ok_or(Error::DeviceFailed {
            stage: "final value",
        }),
    }
}

/// The last round's openings: only the current codeword's blocks, since what
/// they fold to is the constant the prover sends.
///
/// Mirrors [`whir_round`]'s query draw, so both sides sample the same
/// positions.
fn final_openings<C, N, T, H>(
    current: &CodewordCommitment<C, H>,
    config: &RoundConfig,
    caps: RoundCaps,
    transcript: &mut T,
) -> Result<RoundProof<C, N>, Error>
where
    C: IsField + 'static,
    N: IsField,
    FieldElement<C>: AsBytes + Sync + Send,
    T: IsTranscript<N>,
    H: WhirHash,
{
    let __wq_sample = crate::whir_split::mark();
    let queries: Vec<usize> = (0..config.num_queries)
        .map(|_| transcript.sample_u64(current.num_leaves() as u64) as usize)
        .collect();
    crate::whir_split::add(&crate::whir_split::QUERY_SAMPLE, __wq_sample);
    // ⛔ ONE `open_many` here, not two: the final round has no successor to
    // open. That is the `− 1` in arm F's `2R − 1`.
    Ok(RoundProof {
        current: current.open_many_capped(&queries, caps.current, caps.current_owner)?,
        next: Vec::new(),
    })
}

/// Verifies `f(z) = y`.
pub fn verify<F, E, T, H>(
    proof: &ChainProof<F, E>,
    root: &Commitment,
    z: &[FieldElement<E>],
    y: FieldElement<E>,
    domain: &Domain<F>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<(), Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    H: WhirHash,
{
    verify_weighted::<F, E, T, _, H>(
        proof,
        root,
        |alphas: &[FieldElement<E>]| eq_eval(z, alphas),
        y,
        z.len(),
        domain,
        config,
        transcript,
    )
}

/// Verifies `Σ_x w(x)·f(x) = y`.
///
/// `weight_at` is the weight's closed form, evaluated at the concatenation of
/// every round's challenges.
#[allow(clippy::too_many_arguments)]
pub fn verify_weighted<'a, F, E, T, W, H>(
    proof: &'a ChainProof<F, E>,
    root: &'a Commitment,
    weight_at: W,
    y: FieldElement<E>,
    num_vars: usize,
    domain: &Domain<F>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<(), Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    W: FnOnce(&[FieldElement<E>]) -> Result<FieldElement<E>, Error>,
    H: WhirHash,
{
    verify_chain::<F, E, T, W, H>(
        proof,
        core::slice::from_ref(root),
        weight_at,
        y,
        num_vars,
        domain,
        config,
        transcript,
    )
}

/// Verifies ONE chain over a run of `roots.len() = P'` stacked polynomials
/// ([`GroupFold::Indexed`], D-ONECHAIN §2.6): `Σ_x Σ_p eq(y, p)·w_p(x)·f_p(x)
/// = y_claim`, round 0 over the `P'` trees at one position set.
///
/// `num_vars` is a polynomial's, without the index. `weight_at` is the
/// weight's closed form at the chain's challenges in binding order, `n + m` of
/// them: round 0's `k0` data challenges, its `m` index challenges `ρ`, then
/// every later round's. A run of one is [`verify_weighted`], byte for byte.
#[allow(clippy::too_many_arguments)]
pub fn verify_grouped<'a, F, E, T, W, H>(
    proof: &'a ChainProof<F, E>,
    roots: &'a [Commitment],
    weight_at: W,
    y: FieldElement<E>,
    num_vars: usize,
    domain: &Domain<F>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<(), Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    W: FnOnce(&[FieldElement<E>]) -> Result<FieldElement<E>, Error>,
    H: WhirHash,
{
    verify_chain::<F, E, T, W, H>(
        proof, roots, weight_at, y, num_vars, domain, config, transcript,
    )
}

/// Where a round's openings may be which variant (REV-B2-JUDGE B1-2):
/// [`RoundOpenings::BaseGrouped`] exactly at round 0 of a chain over `P' ≥ 2`
/// (final or not), [`RoundOpenings::Base`] exactly at round 0 of a chain over
/// one polynomial, [`RoundOpenings::Extension`] after round 0. A one-polynomial
/// chain's refusals are today's.
fn check_variant<F: IsField, E: IsField>(
    r: usize,
    grouped: bool,
    openings: &RoundOpenings<F, E>,
) -> Result<(), Error> {
    match (openings, r == 0, grouped) {
        (RoundOpenings::BaseGrouped(_), true, true) => Ok(()),
        (RoundOpenings::BaseGrouped(_), _, _) | (RoundOpenings::Base(_), true, true) => {
            Err(Error::GroupedShape(GroupedShape::Variant { round: r }))
        }
        // Only the first round's current codeword is base-field. Said here
        // rather than left to a Merkle check failing on the byte width.
        (RoundOpenings::Base(_), first, _) | (RoundOpenings::Extension(_), first, _)
            if first != matches!(openings, RoundOpenings::Base(_)) =>
        {
            Err(Error::RoundCountMismatch {
                expected: 0,
                got: r,
            })
        }
        _ => Ok(()),
    }
}

/// [`verify_weighted`] and [`verify_grouped`]: one chain over `roots.len()`
/// polynomials. At one polynomial every check and every transcript call is
/// [`verify_weighted`]'s as it was.
#[allow(clippy::too_many_arguments)]
fn verify_chain<'a, F, E, T, W, H>(
    proof: &'a ChainProof<F, E>,
    roots: &'a [Commitment],
    weight_at: W,
    y: FieldElement<E>,
    num_vars: usize,
    domain: &Domain<F>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<(), Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    W: FnOnce(&[FieldElement<E>]) -> Result<FieldElement<E>, Error>,
    H: WhirHash,
{
    check_leaf_order::<H>(&config.format)?;
    // The run's width, and with it the index, are the caller's derivation
    // (`stacked_eval::index_chains`) — never the proof's.
    let width = roots.len();
    let Some(root) = roots.first().filter(|_| width <= MAX_INDEX_POLYS) else {
        return Err(Error::GroupedShape(GroupedShape::RunWidth { polys: width }));
    };
    let grouped = width >= 2;
    let index = index_vars(width);
    if grouped {
        check_index_domain(num_vars, index, config.log_blowup, F::TWO_ADICITY as usize)?;
    }
    // The folds — and so the trees, their caps and the round count — are the
    // per-polynomial chain's; round 0 alone carries `m` more messages.
    let schedule = config.schedule(num_vars);
    let messages = config.index_schedule(num_vars, index);
    if proof.rounds.len() != schedule.len() {
        return Err(Error::RoundCountMismatch {
            expected: schedule.len(),
            got: proof.rounds.len(),
        });
    }

    let mut claim = y;
    let mut alphas: Vec<FieldElement<E>> = Vec::with_capacity(num_vars + index);
    // Tree 0 is authenticated by the cap round 0's first current opening
    // carries; every later tree by the check the round that committed it
    // returned. A verifier constant per tree, derived from the config and the
    // hash alone. A grouped round 0 has `P'` trees 0, each today's tree 0.
    let caps = config.tree_caps_at(num_vars, crate::whir_hash::arity::<F, H>());
    // Which nonce slots the proof carries: a verifier constant too.
    let layout = config.format.nonces;
    // Round-0 trees put their leaves in the chain's order; every later tree is
    // a successor's, natural.
    let mut current: TreeCheck<'a> = TreeCheck::Owner {
        root,
        cap_height: caps[0],
        leaves: config.format.leaves,
    };
    let round0_trees: Vec<TreeCheck<'a>> = if grouped {
        roots
            .iter()
            .map(|root| TreeCheck::Owner {
                root,
                cap_height: caps[0],
                leaves: config.format.leaves,
            })
            .collect()
    } else {
        Vec::new()
    };
    let mut current_domain = domain.clone();
    // Each round's out-of-domain claim, and how many variables were bound when
    // it entered the weight — the challenges after that are where its `eq`
    // lands.
    let mut ood: Vec<(FieldElement<E>, Vec<FieldElement<E>>, usize)> = Vec::new();

    for (r, (round, (&k, &sent))) in proof
        .rounds
        .iter()
        .zip(schedule.iter().zip(&messages))
        .enumerate()
    {
        if round.sumcheck.len() != sent {
            if grouped && r == 0 {
                return Err(Error::GroupedShape(GroupedShape::Round0Messages {
                    expected: sent,
                    got: round.sumcheck.len(),
                }));
            }
            return Err(Error::RoundCountMismatch {
                expected: k,
                got: round.sumcheck.len(),
            });
        }
        check_variant(r, grouped, &round.openings)?;
        check_nonce::<E, T, H>(
            transcript,
            config.grind.folding,
            round.nonces.folding,
            layout,
        )?;
        // The weight raises the degree of the plain `f` term to two.
        let group = sumcheck::verify_rounds(&round.sumcheck, claim, 2, transcript)?;
        claim = group.expected_evaluation;
        // Round 0 of a grouped chain bound its index last: `ρ` folds no block.
        let (data, rho) = group.point.split_at(k);

        let mut next_domain = current_domain.clone();
        for _ in 0..k {
            next_domain = next_domain.squared()?;
        }
        let bound = alphas.len() + group.point.len();

        let round_config = RoundConfig {
            num_queries: config.num_queries,
            log_folding: k,
        };
        match (
            &round.next_root,
            round.ood_value.as_ref(),
            schedule.get(r + 1),
        ) {
            (Some(next_root), Some(y0), Some(&next_k)) => {
                transcript.append_bytes(next_root);

                let z0: FieldElement<E> = transcript.sample_field_element();
                require_out_of_domain::<F, E>(&z0, &next_domain)?;
                let point = ood_point(&z0, num_vars + index - bound);
                transcript.append_field_element(y0);

                check_nonce::<E, T, H>(transcript, config.grind.ood, round.nonces.ood, layout)?;
                let gamma: FieldElement<E> = transcript.sample_field_element();
                claim += &gamma * y0;
                ood.push((gamma, point, bound));

                check_nonce::<E, T, H>(transcript, config.grind.query, round.nonces.query, layout)?;
                let commitments = RoundCommitments {
                    current,
                    next_root,
                    next_num_leaves: next_domain.size() >> next_k,
                    next_cap_height: caps[r + 1],
                };
                let next_check = match &round.openings {
                    RoundOpenings::Base(openings) => whir_round::verify::<F, F, E, T, H>(
                        openings,
                        commitments,
                        &current_domain,
                        data,
                        &round_config,
                        transcript,
                    )?,
                    RoundOpenings::Extension(openings) => whir_round::verify::<F, E, E, T, H>(
                        openings,
                        commitments,
                        &current_domain,
                        data,
                        &round_config,
                        transcript,
                    )?,
                    RoundOpenings::BaseGrouped(openings) => {
                        whir_round::verify_grouped::<F, E, T, H>(
                            openings,
                            &round0_trees,
                            whir_round::GroupedEnd::Successor {
                                root: next_root,
                                num_leaves: next_domain.size() >> next_k,
                                cap_height: caps[r + 1],
                            },
                            &current_domain,
                            data,
                            rho,
                            &round_config,
                            transcript,
                        )?
                        .ok_or(Error::GroupedShape(
                            GroupedShape::SuccessorBlocks {
                                expected: config.num_queries,
                                got: 0,
                            },
                        ))?
                    }
                };
                current = TreeCheck::Checked(next_check);
            }
            (None, None, None) => {
                // No successor, so no out-of-domain grind: the slot is never
                // spent.
                check_unspent(layout, round.nonces.ood)?;
                transcript.append_field_element(&proof.final_value);
                check_nonce::<E, T, H>(transcript, config.grind.query, round.nonces.query, layout)?;
                match &round.openings {
                    RoundOpenings::Base(openings) => verify_final::<F, F, E, T, H>(
                        openings,
                        current,
                        &current_domain,
                        data,
                        &round_config,
                        &proof.final_value,
                        transcript,
                    )?,
                    RoundOpenings::Extension(openings) => verify_final::<F, E, E, T, H>(
                        openings,
                        current,
                        &current_domain,
                        data,
                        &round_config,
                        &proof.final_value,
                        transcript,
                    )?,
                    RoundOpenings::BaseGrouped(openings) => {
                        whir_round::verify_grouped::<F, E, T, H>(
                            openings,
                            &round0_trees,
                            whir_round::GroupedEnd::Final(&proof.final_value),
                            &current_domain,
                            data,
                            rho,
                            &round_config,
                            transcript,
                        )?;
                    }
                }
            }
            // A successor or an out-of-domain value where the schedule ends, or
            // one missing where it does not.
            _ => {
                return Err(Error::RoundCountMismatch {
                    expected: schedule.len(),
                    got: r,
                });
            }
        }

        alphas.extend(group.point);
        current_domain = next_domain;
    }

    // The accumulated weight at the full challenge point: the caller's own,
    // plus each round's batched `eq` over the challenges that came after it.
    let mut weight = weight_at(&alphas)?;
    for (gamma, point, bound) in &ood {
        weight += gamma * eq_eval(point, &alphas[*bound..])?;
    }

    // The wire: the sumcheck's residual is w(α)·f(α), and the fully folded
    // codeword is f(α). Both must name the same value.
    let required = weight
        .inv()
        .ok()
        .map(|inv| claim * inv)
        .ok_or(Error::DegenerateEvaluationPoint)?;
    if proof.final_value != required {
        return Err(Error::EvaluationMismatch);
    }

    Ok(())
}

/// The last round: every queried block must fold to the constant that was sent.
fn verify_final<'a, F, C, N, T, H>(
    openings: &'a RoundProof<C, N>,
    current: TreeCheck<'a>,
    current_domain: &Domain<F>,
    alphas: &[FieldElement<N>],
    config: &RoundConfig,
    final_value: &FieldElement<N>,
    transcript: &mut T,
) -> Result<(), Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<C> + IsSubFieldOf<N>,
    C: IsField + IsSubFieldOf<N> + 'static,
    N: IsField + 'static,
    FieldElement<C>: AsBytes + Sync + Send,
    T: IsTranscript<N>,
    H: WhirHash,
{
    if openings.current.len() != config.num_queries || !openings.next.is_empty() {
        return Err(Error::QueryCountMismatch {
            expected: config.num_queries,
            got: openings.current.len(),
        });
    }
    let num_leaves = current_domain.size() >> config.log_folding;
    let depth = num_leaves.trailing_zeros() as usize;
    // The tree's check, built once from its first opening, after the count
    // guard above, so indexing it never panics. With no openings there is nothing to check.
    let Some(first) = openings.current.first() else {
        return Ok(());
    };
    let (check, first_siblings) = current.open::<C, H>(depth, first)?;

    for (i, opening) in openings.current.iter().enumerate() {
        let q = transcript.sample_u64(num_leaves as u64) as usize;
        let siblings = if i == 0 {
            first_siblings
        } else {
            opening.proof.merkle_path.as_slice()
        };
        let at = current.position(q, depth);
        if !verify_opening_capped::<C, H>(&check, at, opening, siblings) {
            return Err(Error::OpeningRejected { query: i });
        }
        if fold_coset::<F, C, N>(&opening.values, current_domain, q, alphas)? != *final_value {
            return Err(Error::FoldInconsistent { query: i });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::goldilocks::GoldilocksField as F;

    use crate::{eq::eq_evals, whir_eval, whir_hash::KeccakWhir};

    type FE = FieldElement<F>;

    fn transcript() -> DefaultTranscript<F> {
        DefaultTranscript::<F>::new(b"whir-chain-test")
    }

    /// The current codeword's blocks, whichever variant holds them. In these
    /// tests `F` and `E` coincide, so both arms read the same.
    fn current_blocks(openings: &RoundOpenings<F, F>) -> &[crate::whir_commit::CosetOpening<F>] {
        match openings {
            RoundOpenings::Base(p) => &p.current,
            RoundOpenings::Extension(p) => &p.current,
            RoundOpenings::BaseGrouped(_) => panic!("a per-polynomial chain"),
        }
    }

    fn current_blocks_mut(
        openings: &mut RoundOpenings<F, F>,
    ) -> &mut [crate::whir_commit::CosetOpening<F>] {
        match openings {
            RoundOpenings::Base(p) => &mut p.current,
            RoundOpenings::Extension(p) => &mut p.current,
            RoundOpenings::BaseGrouped(_) => panic!("a per-polynomial chain"),
        }
    }

    fn config(log_folding: usize) -> ChainConfig {
        ChainConfig {
            log_blowup: 2,
            log_folding,
            num_queries: 3,
            grind: GrindBits::default(),
            format: ChainFormat::DEFAULT,
        }
    }

    /// Deterministic pseudo-random values; the seed is mixed before the shift so
    /// nearby seeds cannot collapse to the same polynomial.
    fn pseudo_mle(num_vars: usize, seed: u64) -> Mle<F> {
        let vals: Vec<FE> = (0..(1u64 << num_vars))
            .map(|i| {
                let mixed = i
                    .wrapping_add(seed)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                FE::from(mixed >> 13)
            })
            .collect();
        Mle::new(vals).unwrap()
    }

    fn point(num_vars: usize) -> Vec<FE> {
        (0..num_vars).map(|i| FE::from(101 + i as u64)).collect()
    }

    fn run(num_vars: usize, log_folding: usize) -> Result<ChainProof<F, F>, Error> {
        let cfg = config(log_folding);
        let f = pseudo_mle(num_vars, 11);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true)?;
        let proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())?;
        verify::<F, F, _, KeccakWhir>(
            &proof,
            &commitment.root(),
            &z,
            y,
            &domain,
            &cfg,
            &mut transcript(),
        )?;
        Ok(proof)
    }

    #[test]
    fn an_honest_evaluation_verifies() {
        for num_vars in 1..=6usize {
            for log_folding in 1..=3usize {
                run(num_vars, log_folding)
                    .unwrap_or_else(|e| panic!("num_vars={num_vars}, k={log_folding}: {e:?}"));
            }
        }
    }

    #[test]
    fn the_query_count_follows_the_johnson_bound() {
        // Blowup 4, 128 bits, 20 bits of query grinding, one round: the same
        // 110 the univariate prover's own accounting gives.
        let cfg = ChainConfig::with_security(
            2,
            4,
            4,
            128,
            GrindBits {
                query: 20,
                ..GrindBits::default()
            },
        );
        assert_eq!(cfg.num_queries, 110);

        // A wider blowup buys more per query.
        let wider = ChainConfig::with_security(
            3,
            4,
            4,
            128,
            GrindBits {
                query: 20,
                ..GrindBits::default()
            },
        );
        assert!(wider.num_queries < cfg.num_queries);

        // More rounds cost a union bound, and grinding takes queries off.
        let deeper = ChainConfig::with_security(2, 4, 64, 128, GrindBits::default());
        let ground = ChainConfig::with_security(2, 4, 64, 128, GrindBits::uniform(20));
        assert!(deeper.num_queries > cfg.num_queries);
        assert!(ground.num_queries < deeper.num_queries);
    }

    /// And the parameters it picks actually run.
    #[test]
    fn a_proof_at_the_chosen_parameters_verifies() {
        let num_vars = 5;
        // Small bits so the grind is instant; the query count is the real one.
        let cfg = ChainConfig::with_security(2, 2, num_vars, 32, GrindBits::uniform(8));
        assert!(cfg.num_queries > 10);

        let (proof, root, domain, y) = prove_ground(num_vars, &cfg);
        verify_ground(&proof, &root, &domain, y, &cfg, num_vars).unwrap();
    }

    #[test]
    fn the_schedule_takes_the_remainder_last() {
        assert_eq!(config(3).schedule(10), vec![3, 3, 3, 1]);
        assert_eq!(config(3).schedule(9), vec![3, 3, 3]);
        assert_eq!(config(4).schedule(3), vec![3]);
        assert_eq!(config(1).schedule(3), vec![1, 1, 1]);
    }

    // ---------------------------------------------------------------
    // W2: the first-fold schedule.
    // ---------------------------------------------------------------

    fn first(k0: usize, log_folding: usize) -> ChainConfig {
        ChainConfig {
            format: ChainFormat {
                folds: WhirFolds::First(FirstFold::new(k0).unwrap()),
                ..ChainFormat::DEFAULT
            },
            ..config(log_folding)
        }
    }

    /// Today's `schedule` body, verbatim, as the reference the default must
    /// reproduce.
    fn uniform_reference(log_folding: usize, num_vars: usize) -> Vec<usize> {
        let step = log_folding.max(1);
        let mut left = num_vars;
        let mut out = Vec::new();
        while left > 0 {
            let take = step.min(left);
            out.push(take);
            left -= take;
        }
        out
    }

    #[test]
    fn the_schedule_is_uniform_by_default() {
        for k in 1..=MAX_FOLD {
            for n in 0..=40 {
                assert_eq!(
                    config(k).schedule(n),
                    uniform_reference(k, n),
                    "k={k} n={n}"
                );
                assert_eq!(config(k).rounds(n), uniform_reference(k, n).len());
            }
        }
        assert_eq!(ChainFormat::DEFAULT.folds, WhirFolds::Uniform);
        assert_eq!(WhirFolds::default(), WhirFolds::Uniform);
    }

    #[test]
    fn with_security_folds_uniform_is_with_security() {
        for k in 1..=MAX_FOLD {
            for n in 0..=40 {
                for grind in [GrindBits::default(), GrindBits::uniform(20)] {
                    assert_eq!(
                        ChainConfig::with_security_folds(2, k, WhirFolds::Uniform, n, 128, grind),
                        ChainConfig::with_security(2, k, n, 128, grind),
                        "k={k} n={n}"
                    );
                }
            }
        }
        // And today's production numbers.
        let today = ChainConfig::with_security(2, 4, 25, 128, GrindBits::uniform(20));
        assert_eq!((today.rounds(25), today.num_queries), (7, 112));
    }

    #[test]
    fn the_default_fold_word_is_log_folding() {
        for k in 1..=MAX_FOLD {
            assert_eq!(config(k).fold_word(), k as u64);
        }
        assert_eq!(config(4).fold_word().to_le_bytes(), 4u64.to_le_bytes());
    }

    /// The first-fold schedules, by hand, and the clamp at small heights.
    #[test]
    fn the_first_fold_schedules() {
        let (f5, f6) = (first(5, 4), first(6, 4));
        assert_eq!(f6.schedule(25), vec![6, 4, 4, 4, 4, 3]);
        assert_eq!(f6.schedule(24), vec![6, 4, 4, 4, 4, 2]);
        assert_eq!(f6.schedule(23), vec![6, 4, 4, 4, 4, 1]);
        assert_eq!(f5.schedule(25), vec![5, 4, 4, 4, 4, 4]);
        assert_eq!(f5.schedule(24), vec![5, 4, 4, 4, 4, 3]);
        assert_eq!(f5.schedule(23), vec![5, 4, 4, 4, 4, 2]);
        // The first round takes everything when there is less than k0.
        assert_eq!(f6.schedule(0), Vec::<usize>::new());
        assert_eq!(f6.schedule(3), vec![3]);
        assert_eq!(f6.schedule(6), vec![6]);
        assert_eq!(f6.schedule(7), vec![6, 1]);
        assert_eq!(f5.schedule(9), vec![5, 4]);
        assert_eq!(f5.schedule(11), vec![5, 4, 2]);
        assert_eq!(f6.schedule(9), vec![6, 3]);
        // Every schedule covers exactly `n`, starts at min(k0, n), then walks
        // today's uniform body over the rest; no fold exceeds MAX_FOLD.
        for k0 in 1..=MAX_FOLD {
            for k in 1..=4 {
                let c = first(k0, k);
                for n in 0..=40 {
                    let s = c.schedule(n);
                    assert_eq!(s.iter().sum::<usize>(), n);
                    assert!(s.iter().all(|&x| (1..=MAX_FOLD).contains(&x)));
                    if n > 0 {
                        assert_eq!(s[0], k0.min(n));
                        assert_eq!(s[1..], uniform_reference(k, n - s[0])[..]);
                    }
                }
            }
        }
    }

    #[test]
    fn a_first_fold_outside_the_tested_stack_is_unconstructible() {
        assert!(FirstFold::new(0).is_none());
        assert!(FirstFold::new(MAX_FOLD + 1).is_none());
        assert!(FirstFold::new(64).is_none());
        for k in 1..=MAX_FOLD {
            assert_eq!(FirstFold::new(k).unwrap().get(), k);
        }
    }

    /// S2: a stack cap of 0 or past the widest one measured to fit is not a
    /// value a config can hold, and today's 25 is the default everywhere one is
    /// spelled.
    #[test]
    fn a_stack_cap_outside_the_measured_range_is_unconstructible() {
        assert!(StackVars::new(0).is_none());
        assert!(StackVars::new(StackVars::WIDEST + 1).is_none());
        assert!(StackVars::new(64).is_none());
        for n in 1..=StackVars::WIDEST {
            assert_eq!(StackVars::new(n).unwrap().get(), n);
        }
        assert_eq!(StackVars::LEGACY.get(), 25);
        assert_eq!(StackVars::default(), StackVars::LEGACY);
        assert_eq!(ChainFormat::DEFAULT.stack, StackVars::LEGACY);
        assert_eq!(ChainFormat::default(), ChainFormat::DEFAULT);
        assert!(ChainFormat::DEFAULT.is_default());
        let wide = ChainFormat {
            stack: StackVars::new(27).unwrap(),
            ..ChainFormat::DEFAULT
        };
        assert!(!wide.is_default(), "a wider stack is not today's format");
    }

    #[test]
    fn every_fold_word_is_distinct_and_the_non_default_ones_are_tagged() {
        // {Uniform, First(k0)} × {PerPoly, Indexed} (REV-B2-JUDGE B1-5): every
        // word distinct, and every non-default one tagged.
        let indexed = |mut c: ChainConfig| {
            c.format.group = GroupFold::Indexed;
            c
        };
        let mut seen = std::collections::HashSet::new();
        for k in 1..=MAX_FOLD {
            let uniform = config(k).fold_word();
            assert_eq!(uniform & FOLD_WORD_TAG, 0);
            assert!(seen.insert(uniform));
            let w = indexed(config(k)).fold_word();
            assert_ne!(w & FOLD_WORD_TAG, 0, "k={k} uniform, indexed");
            assert_ne!(w & FOLD_WORD_INDEXED, 0, "k={k} uniform, indexed");
            assert!(seen.insert(w), "k={k} uniform, indexed: {w:#x} repeats");
            for k0 in 1..=MAX_FOLD {
                let w = first(k0, k).fold_word();
                assert_ne!(w & FOLD_WORD_TAG, 0, "k={k} k0={k0}");
                assert_eq!(w & FOLD_WORD_INDEXED, 0, "k={k} k0={k0}");
                assert!(seen.insert(w), "k={k} k0={k0}: {w:#x} repeats");
                let w = indexed(first(k0, k)).fold_word();
                assert_ne!(w & FOLD_WORD_TAG, 0, "k={k} k0={k0} indexed");
                assert_ne!(w & FOLD_WORD_INDEXED, 0, "k={k} k0={k0} indexed");
                assert!(seen.insert(w), "k={k} k0={k0} indexed: {w:#x} repeats");
            }
        }
        // The production words, spelled out.
        assert_eq!(first(6, 4).fold_word(), 0x8041_0000_0000_0006);
        assert_eq!(first(5, 4).fold_word(), 0x8041_0000_0000_0005);
        assert_eq!(indexed(first(6, 4)).fold_word(), 0x8041_8000_0000_0006);
        assert_eq!(indexed(config(4)).fold_word(), 0x8000_8000_0000_0004);
    }

    /// No accepted first fold raises the round count at any height, so the
    /// union bound is never charged more and Q never rises; at the production
    /// tallest (25) it is today's 112.
    #[test]
    fn a_first_fold_never_raises_the_query_count() {
        let g = GrindBits::uniform(20);
        for k0 in [5usize, 6] {
            let folds = WhirFolds::First(FirstFold::new(k0).unwrap());
            for tallest in 1..=32 {
                let today = ChainConfig::with_security(2, 4, tallest, 128, g);
                let arm = ChainConfig::with_security_folds(2, 4, folds, tallest, 128, g);
                assert_eq!(arm.format.folds, folds);
                for m in 1..=tallest {
                    assert!(arm.rounds(m) <= today.rounds(m), "k0={k0} m={m}");
                }
                assert!(
                    arm.num_queries <= today.num_queries,
                    "k0={k0} tallest={tallest}"
                );
            }
            let at25 = ChainConfig::with_security_folds(2, 4, folds, 25, 128, g);
            assert_eq!((at25.rounds(25), at25.num_queries), (6, 112), "k0={k0}");
        }
        // The rule charges the WORST height, not the tallest: a first fold
        // narrower than the uniform one has more rounds at some shorter chain,
        // and Q follows that one.
        let narrow = ChainConfig::with_security_folds(
            2,
            4,
            WhirFolds::First(FirstFold::new(1).unwrap()),
            8,
            128,
            GrindBits::default(),
        );
        let worst = (1..=8).map(|m| narrow.rounds(m)).max().unwrap();
        assert_eq!(worst, 3);
        assert_eq!(
            narrow.num_queries,
            crate::query_count::num_queries(2, worst, 128, 0)
        );
    }

    fn run_with(cfg: &ChainConfig, num_vars: usize) -> Result<ChainProof<F, F>, Error> {
        let f = pseudo_mle(num_vars, 13);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();
        let (commitment, domain) = commit::<F, KeccakWhir>(&f, cfg, true)?;
        let proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, cfg, &mut transcript())?;
        verify::<F, F, _, KeccakWhir>(
            &proof,
            &commitment.root(),
            &z,
            y,
            &domain,
            cfg,
            &mut transcript(),
        )?;
        Ok(proof)
    }

    #[test]
    fn a_first_fold_proof_verifies_and_opens_its_wide_block() {
        for (k0, n) in [(5, 9), (5, 11), (6, 9), (6, 11), (6, 6), (6, 3)] {
            let cfg = first(k0, 4);
            let proof = run_with(&cfg, n).unwrap_or_else(|e| panic!("k0={k0} n={n}: {e:?}"));
            assert_eq!(proof.rounds.len(), cfg.rounds(n));
            for (round, &k) in proof.rounds.iter().zip(&cfg.schedule(n)) {
                assert_eq!(round.sumcheck.len(), k);
                for opening in current_blocks(&round.openings) {
                    assert_eq!(opening.values.len(), 1 << k);
                }
            }
        }
    }

    #[test]
    fn a_tampered_wide_base_block_is_rejected() {
        let cfg = first(6, 4);
        let num_vars = 11;
        let f = pseudo_mle(num_vars, 83);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();
        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let honest =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())
                .unwrap();
        assert_eq!(
            current_blocks(&honest.rounds[0].openings)[0].values.len(),
            64
        );
        // The last value of the 64-wide block, so the check must read all of it.
        let mut proof = honest.clone();
        current_blocks_mut(&mut proof.rounds[0].openings)[0].values[63] += FE::one();
        let err = verify::<F, F, _, KeccakWhir>(
            &proof,
            &commitment.root(),
            &z,
            y,
            &domain,
            &cfg,
            &mut transcript(),
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                Error::OpeningRejected { .. } | Error::FoldInconsistent { .. }
            ),
            "{err:?}"
        );
    }

    /// A proof made under one schedule is refused under another: the round
    /// count (or the first round's sumcheck length) disagrees.
    #[test]
    fn a_first_fold_proof_is_refused_under_the_uniform_schedule() {
        let num_vars = 11;
        let (f6, uniform) = (first(6, 4), config(4));
        let f = pseudo_mle(num_vars, 13);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();
        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &f6, true).unwrap();
        let proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &f6, &mut transcript())
                .unwrap();
        let err = verify::<F, F, _, KeccakWhir>(
            &proof,
            &commitment.root(),
            &z,
            y,
            &domain,
            &uniform,
            &mut transcript(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::RoundCountMismatch { .. }), "{err:?}");
        // And the other way round.
        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &uniform, true).unwrap();
        let proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &uniform, &mut transcript())
                .unwrap();
        let err = verify::<F, F, _, KeccakWhir>(
            &proof,
            &commitment.root(),
            &z,
            y,
            &domain,
            &f6,
            &mut transcript(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::RoundCountMismatch { .. }), "{err:?}");
    }

    /// The point of chaining: a query opens a block of `2^k`, not the message.
    #[test]
    fn a_block_is_the_fold_size_not_the_message() {
        let num_vars = 6;
        let log_folding = 2;
        let proof = run(num_vars, log_folding).unwrap();

        assert_eq!(proof.rounds.len(), 3);
        for round in &proof.rounds {
            for opening in current_blocks(&round.openings) {
                assert_eq!(opening.values.len(), 1 << log_folding);
            }
        }
    }

    /// And what that buys, against the one-round argument on the same
    /// polynomial.
    #[test]
    fn chaining_opens_far_less_than_one_round_does() {
        let num_vars = 8;
        let cfg = config(2);
        let f = pseudo_mle(num_vars, 13);
        let z = point(num_vars);

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let chained =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())
                .unwrap();

        let one_round_cfg = whir_eval::EvalConfig {
            log_blowup: cfg.log_blowup,
            num_queries: cfg.num_queries,
        };
        let (one_commitment, one_domain) =
            whir_eval::commit::<F, F, KeccakWhir>(&f, &one_round_cfg).unwrap();
        let one_round = whir_eval::prove::<F, F, _, KeccakWhir>(
            &f,
            &z,
            &one_commitment,
            &one_domain,
            &one_round_cfg,
            &mut transcript(),
        )
        .unwrap();

        let one_round_elements: usize = one_round.openings.iter().map(|o| o.values.len()).sum();
        // One round sends `num_queries` blocks of the whole message; chaining
        // sends small blocks, two per query per round.
        assert_eq!(one_round_elements, 3 * (1 << num_vars));
        assert!(
            chained.opened_elements() * 4 < one_round_elements,
            "chained {} vs one round {one_round_elements}",
            chained.opened_elements()
        );
    }

    #[test]
    fn a_false_evaluation_is_rejected() {
        let cfg = config(2);
        let num_vars = 5;
        let f = pseudo_mle(num_vars, 17);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())
                .unwrap();

        assert!(
            verify::<F, F, _, KeccakWhir>(
                &proof,
                &commitment.root(),
                &z,
                y + FE::one(),
                &domain,
                &cfg,
                &mut transcript(),
            )
            .is_err()
        );
    }

    #[test]
    fn a_forged_final_value_is_rejected() {
        let cfg = config(2);
        let num_vars = 5;
        let f = pseudo_mle(num_vars, 19);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let mut proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())
                .unwrap();
        proof.final_value += FE::one();

        assert!(
            verify::<F, F, _, KeccakWhir>(
                &proof,
                &commitment.root(),
                &z,
                y,
                &domain,
                &cfg,
                &mut transcript()
            )
            .is_err()
        );
    }

    #[test]
    fn a_tampered_opening_in_a_middle_round_is_rejected() {
        let cfg = config(2);
        let num_vars = 6;
        let f = pseudo_mle(num_vars, 23);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let mut proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())
                .unwrap();
        assert!(proof.rounds.len() >= 3);
        current_blocks_mut(&mut proof.rounds[1].openings)[0].values[0] += FE::one();

        let err = verify::<F, F, _, KeccakWhir>(
            &proof,
            &commitment.root(),
            &z,
            y,
            &domain,
            &cfg,
            &mut transcript(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::OpeningRejected { .. }));
    }

    /// The first round's blocks are hashed as base-field bytes, so tampering
    /// with one has to be caught by that hash and not by a later fold.
    #[test]
    fn a_tampered_opening_in_the_base_round_is_rejected() {
        let cfg = config(2);
        let num_vars = 6;
        let f = pseudo_mle(num_vars, 79);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let mut proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())
                .unwrap();
        assert!(matches!(proof.rounds[0].openings, RoundOpenings::Base(_)));
        current_blocks_mut(&mut proof.rounds[0].openings)[0].values[0] += FE::one();

        let err = verify::<F, F, _, KeccakWhir>(
            &proof,
            &commitment.root(),
            &z,
            y,
            &domain,
            &cfg,
            &mut transcript(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::OpeningRejected { .. }));
    }

    /// A prover who commits one polynomial and argues about another: the fold
    /// check between consecutive codewords is what catches it.
    #[test]
    fn a_proof_about_a_different_polynomial_is_rejected() {
        let cfg = config(2);
        let num_vars = 5;
        let f = pseudo_mle(num_vars, 29);
        let g = pseudo_mle(num_vars, 31);
        let z = point(num_vars);

        let (f_commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let proof =
            prove::<F, F, _, KeccakWhir>(&g, &z, &f_commitment, &domain, &cfg, &mut transcript())
                .unwrap();

        let err = verify::<F, F, _, KeccakWhir>(
            &proof,
            &f_commitment.root(),
            &z,
            g.evaluate(&z).unwrap(),
            &domain,
            &cfg,
            &mut transcript(),
        )
        .unwrap_err();
        // The codeword now comes out of the commitment, so a prover cannot be
        // inconsistent between the two: the lie has nowhere to go but the wire
        // between the sumcheck and the folded value.
        assert_eq!(err, Error::EvaluationMismatch);
    }

    #[test]
    fn a_round_missing_its_successor_is_rejected() {
        let cfg = config(2);
        let num_vars = 6;
        let f = pseudo_mle(num_vars, 37);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let mut proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())
                .unwrap();
        proof.rounds[0].next_root = None;

        assert!(
            verify::<F, F, _, KeccakWhir>(
                &proof,
                &commitment.root(),
                &z,
                y,
                &domain,
                &cfg,
                &mut transcript()
            )
            .is_err()
        );
    }

    #[test]
    fn a_proof_replayed_under_another_transcript_is_rejected() {
        let cfg = config(2);
        let num_vars = 5;
        let f = pseudo_mle(num_vars, 41);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())
                .unwrap();

        let mut other = DefaultTranscript::<F>::new(b"a-different-statement");
        assert!(
            verify::<F, F, _, KeccakWhir>(
                &proof,
                &commitment.root(),
                &z,
                y,
                &domain,
                &cfg,
                &mut other
            )
            .is_err()
        );
    }

    /// A weight that is not an `eq`: the shape a stacked multi-point claim
    /// takes, chained the same way.
    #[test]
    fn a_two_point_weighted_claim_chains() {
        let cfg = config(2);
        let num_vars = 5;
        let f = pseudo_mle(num_vars, 43);
        let a = point(num_vars);
        let b: Vec<FE> = a.iter().map(|x| x + FE::from(7)).collect();
        let gamma = FE::from(5);

        // w = eq(a, ·) + gamma·eq(b, ·)
        let table: Vec<FE> = eq_evals(&a)
            .into_iter()
            .zip(eq_evals(&b))
            .map(|(p, q)| p + gamma * q)
            .collect();
        let weight = Mle::new(table).unwrap();
        let y = f.evaluate(&a).unwrap() + gamma * f.evaluate(&b).unwrap();

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let proof = prove_weighted::<F, F, _, KeccakWhir>(
            &f,
            weight,
            &commitment,
            &domain,
            &cfg,
            &mut transcript(),
        )
        .unwrap();

        verify_weighted::<F, F, _, _, KeccakWhir>(
            &proof,
            &commitment.root(),
            |at: &[FE]| Ok(eq_eval(&a, at)? + gamma * eq_eval(&b, at)?),
            y,
            num_vars,
            &domain,
            &cfg,
            &mut transcript(),
        )
        .unwrap();
    }

    /// The committed codeword is base-field, so only the first round's blocks
    /// are — that is where the memory goes, and the type says which round is
    /// which rather than leaving it implicit.
    #[test]
    fn only_the_first_round_opens_base_field_blocks() {
        let proof = run(6, 2).unwrap();
        assert!(proof.rounds.len() >= 3);
        assert!(matches!(proof.rounds[0].openings, RoundOpenings::Base(_)));
        for round in &proof.rounds[1..] {
            assert!(matches!(round.openings, RoundOpenings::Extension(_)));
        }
    }

    /// And the verifier says so: a first round presenting extension blocks is
    /// refused outright, rather than left to fail on a byte-width mismatch
    /// inside a hash.
    #[test]
    fn a_first_round_claiming_extension_blocks_is_rejected() {
        let cfg = config(2);
        let num_vars = 6;
        let f = pseudo_mle(num_vars, 73);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let mut proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())
                .unwrap();

        // `F` and `E` coincide here, so the same blocks fit the other variant.
        if let RoundOpenings::Base(openings) = proof.rounds[0].openings.clone() {
            proof.rounds[0].openings = RoundOpenings::Extension(openings);
        }

        assert!(
            verify::<F, F, _, KeccakWhir>(
                &proof,
                &commitment.root(),
                &z,
                y,
                &domain,
                &cfg,
                &mut transcript()
            )
            .is_err()
        );
    }

    // ---------------------------------------------------------------
    // Out of domain.
    // ---------------------------------------------------------------

    /// The convention the out-of-domain claim rests on: the multilinear point
    /// `(z, z^2, z^4, ..)` is where the univariate lift is `z`. Pinned against
    /// the codeword itself, at every domain point.
    #[test]
    fn the_out_of_domain_point_is_where_the_lift_is_sampled() {
        use crate::whir::{encode, lift_coefficients};

        let num_vars = 4;
        let f = pseudo_mle(num_vars, 53);
        let domain = Domain::<F>::new(num_vars + 2).unwrap();
        let codeword = encode::<F, F>(&lift_coefficients(&f), &domain).unwrap();

        let mut x = FE::one();
        for value in &codeword {
            assert_eq!(*value, f.evaluate(&ood_point(&x, num_vars)).unwrap());
            x *= domain.generator();
        }
    }

    /// And the property the whole round rests on: the folded codeword encodes
    /// the message the sumcheck folded, in the same variable order.
    #[test]
    fn the_folded_codeword_encodes_the_folded_message() {
        use crate::whir::{encode, lift_coefficients};

        let num_vars = 4;
        let f = pseudo_mle(num_vars, 59);
        let domain = Domain::<F>::new(num_vars + 2).unwrap();
        let codeword = encode::<F, F>(&lift_coefficients(&f), &domain).unwrap();

        let alphas = [FE::from(3), FE::from(5)];
        let (folded, folded_domain) =
            crate::whir::fold_codeword_k::<F, F, F>(&codeword, &domain, &alphas).unwrap();

        let mut message = f;
        for alpha in &alphas {
            message.fix_first_variable_in_place(alpha).unwrap();
        }
        let re_encoded = encode::<F, F>(&lift_coefficients(&message), &folded_domain).unwrap();
        assert_eq!(folded, re_encoded);
    }

    /// ★ The same at the production shape, through the prover's own steps: a
    /// stacked base-field commitment at blowup 4, a first fold of six into the
    /// cubic extension and folds of four after it, the message bound by the
    /// chain's sumcheck on challenges from its transcript, the weight batched
    /// with the out-of-domain term between rounds. Every round's successor —
    /// the fold of the codeword it opens — is the bound message encoded on the
    /// folded domain, byte for byte as the tree hashes it. So a successor can
    /// be encoded from the `2^(n−k)` message instead of folded from the
    /// `2^(n+blowup)` codeword (D-SPEED lever 5a). The domain is the subgroup
    /// itself: there is no coset offset to match.
    ///
    /// Two near misses must differ, so the equality is not one any encoding
    /// would show: the message encoded without the lift's bit reversal, and the
    /// first round's variables bound from the other end.
    #[test]
    fn every_successor_is_the_bound_message_encoded_at_the_production_shape() {
        use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as E;
        type EE = FieldElement<E>;
        let bytes =
            |values: &[EE]| -> Vec<u8> { values.iter().flat_map(|v| v.as_bytes()).collect() };
        let lift = |m: &Mle<F>| -> Mle<E> {
            Mle::new(m.evals().iter().map(|v| (*v).to_extension::<E>()).collect()).unwrap()
        };

        let num_vars = 15;
        let cfg = first(6, 4);
        let schedule = cfg.schedule(num_vars);
        assert_eq!(schedule, [6, 4, 4, 1], "the production schedule's shape");

        // Columns of mixed heights where the production layout puts them.
        let heights = [13usize, 13, 12, 11, 10, 8, 6, 3];
        let columns: Vec<Mle<F>> = heights
            .iter()
            .enumerate()
            .map(|(i, &h)| pseudo_mle(h, 71 + 13 * i as u64))
            .collect();
        let layout = crate::stacking::StackedLayout::build(&heights, num_vars).unwrap();
        assert_eq!(layout.num_polys(), 1);
        let stacked = Stacked {
            parts: layout
                .placements()
                .iter()
                .zip(&columns)
                .map(|(place, column)| (column as &dyn HostColumn<F>, place.offset))
                .collect(),
            resident: None,
            num_vars,
        };
        let (commitment, domain) = commit_stacked::<F, KeccakWhir>(&stacked, &cfg, true).unwrap();
        let base = commitment
            .codeword()
            .host()
            .expect("no device: the host committed")
            .to_vec();
        let message = stacked.assemble().unwrap();

        let weight = Mle::new(
            (0..1u64 << num_vars)
                .map(|i| EE::from(i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 7))
                .collect(),
        )
        .unwrap();
        let mut factors = Factors::<F, E>::new(&message, weight).unwrap();
        let mut transcript = DefaultTranscript::<E>::new(b"whir-chain-5a");

        let mut current: Vec<EE> = Vec::new();
        let mut current_domain = domain;
        for (r, &k) in schedule.iter().enumerate() {
            let (_, alphas) = factors.rounds(k, &mut transcript).unwrap();
            let (folded, folded_domain) = if r == 0 {
                crate::whir::fold_codeword_k_on_host::<F, F, E>(&base, &current_domain, &alphas)
            } else {
                crate::whir::fold_codeword_k_on_host::<F, E, E>(&current, &current_domain, &alphas)
            }
            .unwrap();
            let Factors::Host { message: bound, .. } = &factors else {
                panic!("no device: the host bound the message");
            };
            let left: usize = num_vars - schedule[..=r].iter().sum::<usize>();
            assert_eq!(bound.num_vars(), left);
            assert_eq!(folded.len(), 1 << (left + cfg.log_blowup));
            let encoded = encode::<F, E>(&lift_coefficients(bound), &folded_domain).unwrap();
            assert_eq!(
                bytes(&encoded),
                bytes(&folded),
                "round {r} (fold {k}): the encoded message is not the folded codeword"
            );

            if r == 0 {
                let unreversed =
                    encode::<F, E>(&crate::whir::monomial_coefficients(bound), &folded_domain)
                        .unwrap();
                assert_ne!(
                    bytes(&unreversed),
                    bytes(&folded),
                    "the lift's order matters"
                );
                let mut other_end = lift(&message);
                for alpha in &alphas {
                    other_end.fix_last_variable_in_place(alpha).unwrap();
                }
                let wrong = encode::<F, E>(&lift_coefficients(&other_end), &folded_domain).unwrap();
                assert_ne!(bytes(&wrong), bytes(&folded), "the binding order matters");
            }

            if schedule.get(r + 1).is_some() {
                let z0: EE = transcript.sample_field_element();
                let point = ood_point(&z0, factors.num_vars());
                let y0 = factors.evaluate_message(&point).unwrap();
                transcript.append_field_element(&y0);
                let gamma: EE = transcript.sample_field_element();
                factors.add_scaled_eq(&point, &gamma).unwrap();
            }
            current = folded;
            current_domain = folded_domain;
        }
    }

    /// The grouped round 0's successor — each round-0 codeword folded by the
    /// data challenges, the folds combined with `eq(ρ, p)` ([`fold_grouped`])
    /// — is the encoding of the message the lockstep rounds leave bound
    /// ([`lockstep_rounds`]), for runs of 2, 3 and 4, and so is every later
    /// round's successor. So a card that holds the run's combined factors may
    /// encode the successor from them instead of folding `P'` codewords. A
    /// near miss must differ: `ρ` read from the other end, where two index
    /// variables make the order matter.
    #[test]
    fn a_grouped_successor_is_the_runs_bound_message_encoded() {
        use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as E;
        type EE = FieldElement<E>;
        let bytes =
            |values: &[EE]| -> Vec<u8> { values.iter().flat_map(|v| v.as_bytes()).collect() };
        let num_vars = 11;
        let cfg = first(6, 4);
        let schedule = cfg.schedule(num_vars);
        assert_eq!(schedule, [6, 4, 1]);
        for width in 2..=MAX_INDEX_POLYS {
            let polys: Vec<Mle<F>> = (0..width)
                .map(|p| pseudo_mle(num_vars, 71 + 13 * p as u64))
                .collect();
            let stacks: Vec<Stacked<'_, F>> = polys
                .iter()
                .map(|f| Stacked {
                    parts: vec![(f as &dyn HostColumn<F>, 0)],
                    resident: None,
                    num_vars,
                })
                .collect();
            let committed: Vec<(CodewordCommitment<F, KeccakWhir>, Domain<F>)> = stacks
                .iter()
                .map(|s| commit_stacked::<F, KeccakWhir>(s, &cfg, true).unwrap())
                .collect();
            let domain = committed[0].1.clone();
            let held: Vec<&CodewordCommitment<F, KeccakWhir>> =
                committed.iter().map(|(c, _)| c).collect();
            let parts: Vec<(Mle<E>, Mle<E>)> = polys
                .iter()
                .enumerate()
                .map(|(p, f)| {
                    let weight = Mle::new(
                        (0..1u64 << num_vars)
                            .map(|i| {
                                EE::from(
                                    (i + 1 + p as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 7,
                                )
                            })
                            .collect(),
                    )
                    .unwrap();
                    let message =
                        Mle::new(f.evals().iter().map(|v| (*v).to_extension::<E>()).collect())
                            .unwrap();
                    (weight, message)
                })
                .collect();
            let m = index_vars(width);
            let mut factors = Factors::<F, E>::Lockstep {
                parts,
                index_vars: m,
            };
            let mut transcript = DefaultTranscript::<E>::new(b"whir-chain-grouped-5a");
            let mut current: Vec<EE> = Vec::new();
            let mut current_domain = domain;
            for (r, &k) in schedule.iter().enumerate() {
                let (_, challenges) = factors.rounds(k, &mut transcript).unwrap();
                let (alphas, rho) = challenges.split_at(k.min(challenges.len()));
                assert_eq!(rho.len(), if r == 0 { m } else { 0 });
                let (folded, folded_domain) = if r == 0 {
                    let (codeword, at) =
                        fold_grouped::<F, E, KeccakWhir>(&held, &current_domain, alphas, rho)
                            .unwrap();
                    (codeword.host().expect("a host fold").to_vec(), at)
                } else {
                    crate::whir::fold_codeword_k_on_host::<F, E, E>(
                        &current,
                        &current_domain,
                        alphas,
                    )
                    .unwrap()
                };
                let Factors::Host { message: bound, .. } = &factors else {
                    panic!("round 0 leaves one host pair");
                };
                let left = num_vars - schedule[..=r].iter().sum::<usize>();
                assert_eq!(bound.num_vars(), left);
                assert_eq!(folded.len(), 1 << (left + cfg.log_blowup));
                let encoded = encode::<F, E>(&lift_coefficients(bound), &folded_domain).unwrap();
                assert_eq!(
                    bytes(&encoded),
                    bytes(&folded),
                    "width {width} round {r}: the encoded message is not the successor"
                );
                if r == 0 && m == 2 {
                    let swapped: Vec<EE> = rho.iter().rev().cloned().collect();
                    let (other, _) =
                        fold_grouped::<F, E, KeccakWhir>(&held, &current_domain, alphas, &swapped)
                            .unwrap();
                    assert_ne!(
                        bytes(other.host().unwrap()),
                        bytes(&folded),
                        "width {width}: the index order matters"
                    );
                }
                if schedule.get(r + 1).is_some() {
                    let z0: EE = transcript.sample_field_element();
                    let point = ood_point(&z0, factors.num_vars());
                    let y0 = factors.evaluate_message(&point).unwrap();
                    transcript.append_field_element(&y0);
                    let gamma: EE = transcript.sample_field_element();
                    factors.add_scaled_eq(&point, &gamma).unwrap();
                }
                current = folded;
                current_domain = folded_domain;
            }
        }
    }

    #[test]
    fn every_round_but_the_last_answers_out_of_domain() {
        let proof = run(6, 2).unwrap();
        assert_eq!(proof.rounds.len(), 3);
        for round in &proof.rounds[..2] {
            assert!(round.ood_value.is_some());
        }
        assert!(proof.rounds[2].ood_value.is_none());
    }

    /// The out-of-domain value is batched into the next group's claim, so
    /// getting it wrong breaks the chain.
    #[test]
    fn a_forged_out_of_domain_value_is_rejected() {
        let cfg = config(2);
        let num_vars = 6;
        let f = pseudo_mle(num_vars, 61);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let mut proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())
                .unwrap();
        proof.rounds[0].ood_value = Some(proof.rounds[0].ood_value.unwrap() + FE::one());

        assert!(
            verify::<F, F, _, KeccakWhir>(
                &proof,
                &commitment.root(),
                &z,
                y,
                &domain,
                &cfg,
                &mut transcript()
            )
            .is_err()
        );
    }

    #[test]
    fn a_round_missing_its_out_of_domain_value_is_rejected() {
        let cfg = config(2);
        let num_vars = 6;
        let f = pseudo_mle(num_vars, 67);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let mut proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())
                .unwrap();
        proof.rounds[0].ood_value = None;

        assert!(
            verify::<F, F, _, KeccakWhir>(
                &proof,
                &commitment.root(),
                &z,
                y,
                &domain,
                &cfg,
                &mut transcript()
            )
            .is_err()
        );
    }

    #[test]
    fn a_point_inside_the_domain_is_refused() {
        // The generator of the folded domain is in it, so it must be rejected;
        // a point outside must not be.
        let domain = Domain::<F>::new(5).unwrap();
        assert_eq!(
            require_out_of_domain::<F, F>(domain.generator(), &domain).err(),
            Some(Error::OodPointInDomain)
        );
        assert!(require_out_of_domain::<F, F>(&FE::from(7), &domain).is_ok());
    }

    // ---------------------------------------------------------------
    // Proof of work.
    // ---------------------------------------------------------------

    /// Enough bits to be a real search, few enough to be instant.
    fn ground(log_folding: usize) -> ChainConfig {
        ChainConfig {
            grind: GrindBits::uniform(8),
            ..config(log_folding)
        }
    }

    fn prove_ground(
        num_vars: usize,
        cfg: &ChainConfig,
    ) -> (ChainProof<F, F>, Commitment, Domain<F>, FE) {
        let f = pseudo_mle(num_vars, 71);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();
        let (commitment, domain) = commit::<F, KeccakWhir>(&f, cfg, true).unwrap();
        let proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, cfg, &mut transcript())
                .unwrap();
        (proof, commitment.root(), domain, y)
    }

    fn verify_ground(
        proof: &ChainProof<F, F>,
        root: &Commitment,
        domain: &Domain<F>,
        y: FE,
        cfg: &ChainConfig,
        num_vars: usize,
    ) -> Result<(), Error> {
        verify::<F, F, _, KeccakWhir>(
            proof,
            root,
            &point(num_vars),
            y,
            domain,
            cfg,
            &mut transcript(),
        )
    }

    /// Grind checks in the verifier's order: `(the state the check read, the
    /// nonce)`.
    type GrindChecks = Vec<([u8; 32], u64)>;

    /// The verifier's transcript, logging every proof-of-work check it passes:
    /// the state the check read and the nonce it absorbed right after.
    /// `check_grind` is the verifier's only reader of `state()`, and it absorbs
    /// the nonce next.
    struct GrindLog {
        inner: DefaultTranscript<F>,
        read: core::cell::Cell<Option<[u8; 32]>>,
        checks: GrindChecks,
    }

    impl IsTranscript<F> for GrindLog {
        fn append_field_element(&mut self, element: &FE) {
            self.inner.append_field_element(element);
        }

        fn append_bytes(&mut self, new_bytes: &[u8]) {
            if let Some(state) = self.read.take() {
                let nonce = new_bytes
                    .try_into()
                    .map(u64::from_be_bytes)
                    .expect("a passed grind check absorbs its eight-byte nonce next");
                self.checks.push((state, nonce));
            }
            self.inner.append_bytes(new_bytes);
        }

        fn state(&self) -> [u8; 32] {
            let state = self.inner.state();
            self.read.set(Some(state));
            state
        }

        fn sample_field_element(&mut self) -> FE {
            self.inner.sample_field_element()
        }

        fn sample_u64(&mut self, upper_bound: u64) -> u64 {
            self.inner.sample_u64(upper_bound)
        }
    }

    /// [`verify_ground`], with every grind check the verifier passed on the way.
    fn verify_logged(
        proof: &ChainProof<F, F>,
        root: &Commitment,
        domain: &Domain<F>,
        y: FE,
        cfg: &ChainConfig,
        num_vars: usize,
    ) -> (Result<(), Error>, GrindChecks) {
        let mut log = GrindLog {
            inner: transcript(),
            read: Default::default(),
            checks: Vec::new(),
        };
        let verdict =
            verify::<F, F, _, KeccakWhir>(proof, root, &point(num_vars), y, domain, cfg, &mut log);
        (verdict, log.checks)
    }

    fn passes(state: &[u8; 32], nonce: u64, bits: u8) -> bool {
        crypto::grinding::is_valid_nonce::<GrindingDigest<KeccakWhir>>(state, nonce, bits)
    }

    /// A forgery the check at `state` must refuse, whichever valid nonce the
    /// search returned: the first of `nonce + 1, nonce + 2, …` that fails the
    /// PoW there. `nonce + 1` alone passes with probability `2^-bits`, and a
    /// forgery that passes is absorbed and moves the transcript, failing later
    /// and elsewhere instead of at the check.
    fn failing(state: &[u8; 32], nonce: u64, bits: u8) -> u64 {
        (1..)
            .map(|d| nonce.wrapping_add(d))
            .find(|&n| !passes(state, n, bits))
            .expect("a nonce that fails the PoW")
    }

    /// Proves a uniformly ground chain, replaces the spent nonce of `slot` in
    /// `round` with a forgery its check must refuse, and verifies. The log
    /// holds three checks a round (folding, out-of-domain, query), so `round`
    /// must not be the last, which has no out-of-domain check.
    fn forge_uniform(round: usize, slot: &str) -> Result<(), Error> {
        let num_vars = 6;
        let cfg = ground(2);
        let (mut proof, root, domain, y) = prove_ground(num_vars, &cfg);
        let (verdict, checks) = verify_logged(&proof, &root, &domain, y, &cfg, num_vars);
        verdict.expect("the honest proof must verify");
        assert!(round + 1 < proof.rounds.len(), "not the last round");
        let nonces = &mut proof.rounds[round].nonces;
        let (at, field) = match slot {
            "folding" => (0, &mut nonces.folding),
            "ood" => (1, &mut nonces.ood),
            _ => (2, &mut nonces.query),
        };
        let (state, nonce) = checks[3 * round + at];
        assert_eq!(*field, nonce, "the logged check is round {round}'s {slot}");
        *field = failing(&state, nonce, 8);
        verify_ground(&proof, &root, &domain, y, &cfg, num_vars)
    }

    #[test]
    fn a_ground_proof_verifies() {
        let num_vars = 6;
        let cfg = ground(2);
        let (proof, root, domain, y) = prove_ground(num_vars, &cfg);

        // Every round grinds before its folding challenges and its queries, and
        // before the out-of-domain challenge where there is one: the verifier
        // checks exactly those nonces, in that order, and each passes the PoW
        // at the state its check read. Not "non-zero": the search may return
        // any valid nonce, 0 included (`find_any` under `parallel`).
        let (verdict, checks) = verify_logged(&proof, &root, &domain, y, &cfg, num_vars);
        verdict.unwrap();
        let last = proof.rounds.len() - 1;
        let spent: Vec<u64> = proof
            .rounds
            .iter()
            .enumerate()
            .flat_map(|(r, round)| {
                let n = round.nonces;
                if r == last {
                    vec![n.folding, n.query]
                } else {
                    vec![n.folding, n.ood, n.query]
                }
            })
            .collect();
        assert_eq!(checks.len(), 3 * proof.rounds.len() - 1);
        assert_eq!(
            checks.iter().map(|&(_, nonce)| nonce).collect::<Vec<_>>(),
            spent,
            "one check a spent slot, in slot order"
        );
        for (state, nonce) in &checks {
            assert!(passes(state, *nonce, 8));
        }
        assert_eq!(
            proof.rounds[last].nonces.ood, 0,
            "the last round has no out-of-domain grind"
        );
    }

    #[test]
    fn zero_bits_grinds_nothing() {
        let proof = run(6, 2).unwrap();
        for round in &proof.rounds {
            assert_eq!(round.nonces, RoundNonces::default());
        }
    }

    #[test]
    fn a_forged_folding_nonce_is_rejected() {
        assert_eq!(
            forge_uniform(1, "folding"),
            Err(Error::GrindingRejected { bits: 8 })
        );
    }

    #[test]
    fn a_forged_out_of_domain_nonce_is_rejected() {
        assert_eq!(
            forge_uniform(0, "ood"),
            Err(Error::GrindingRejected { bits: 8 })
        );
    }

    #[test]
    fn a_forged_query_nonce_is_rejected() {
        assert_eq!(
            forge_uniform(0, "query"),
            Err(Error::GrindingRejected { bits: 8 })
        );
    }

    /// A verifier asking for bits the prover did not spend.
    #[test]
    fn a_proof_ground_for_fewer_bits_is_rejected() {
        let num_vars = 6;
        let (proof, root, domain, y) = prove_ground(num_vars, &ground(2));
        let demanding = ChainConfig {
            grind: GrindBits::uniform(30),
            ..config(2)
        };

        assert!(verify_ground(&proof, &root, &domain, y, &demanding, num_vars).is_err());
    }

    #[test]
    fn the_bits_can_differ_per_place() {
        let num_vars = 5;
        let cfg = ChainConfig {
            grind: GrindBits {
                folding: 6,
                ood: 0,
                query: 8,
            },
            ..config(2)
        };
        let (proof, root, domain, y) = prove_ground(num_vars, &cfg);
        assert_eq!(proof.rounds[0].nonces.ood, 0);
        verify_ground(&proof, &root, &domain, y, &cfg, num_vars).unwrap();
    }

    // ---------------------------------------------------------------
    // P2: a grind before the queries only, and the nonce layout.
    // ---------------------------------------------------------------

    /// The query count reads the query grind alone, so dropping the other two
    /// grinds leaves it where it was.
    #[test]
    fn a_query_only_grind_keeps_the_query_count() {
        assert_eq!(
            GrindBits::query_only(20),
            GrindBits {
                folding: 0,
                ood: 0,
                query: 20
            }
        );
        for num_vars in [10, 25, 27] {
            assert_eq!(
                ChainConfig::with_security(2, 4, num_vars, 128, GrindBits::query_only(20))
                    .num_queries,
                ChainConfig::with_security(2, 4, num_vars, 128, GrindBits::uniform(20)).num_queries,
                "num_vars {num_vars}"
            );
        }
    }

    /// Enough query bits to be a real search, under a nonce layout.
    fn query_only(layout: NonceLayout) -> ChainConfig {
        ChainConfig {
            grind: GrindBits::query_only(8),
            format: ChainFormat {
                nonces: layout,
                ..ChainFormat::DEFAULT
            },
            ..config(2)
        }
    }

    /// ★ Under either layout a query-only chain grinds once a round, before its
    /// queries: the other two nonces stay zero, and a wrong query nonce is
    /// refused in every round.
    #[test]
    fn a_query_only_chain_checks_its_query_nonce_in_every_round() {
        let num_vars = 9;
        for layout in [NonceLayout::Three, NonceLayout::Spent] {
            let cfg = query_only(layout);
            let (proof, root, domain, y) = prove_ground(num_vars, &cfg);
            assert!(
                proof.rounds.len() > 2,
                "a round that is neither first nor last"
            );
            for round in &proof.rounds {
                assert_eq!((round.nonces.folding, round.nonces.ood), (0, 0));
            }
            // One check a round, the query's, each passing the PoW at the state
            // it read.
            let (verdict, checks) = verify_logged(&proof, &root, &domain, y, &cfg, num_vars);
            verdict.unwrap();
            assert_eq!(
                checks.iter().map(|&(_, nonce)| nonce).collect::<Vec<_>>(),
                proof
                    .rounds
                    .iter()
                    .map(|round| round.nonces.query)
                    .collect::<Vec<_>>(),
                "{layout:?}: one grind check a round, the query's"
            );
            for (r, (state, nonce)) in checks.iter().enumerate() {
                assert!(passes(state, *nonce, 8), "{layout:?}, round {r}");
                // A wrong query nonce, one that fails the PoW there, is refused
                // by that round's check.
                let mut forged = proof.clone();
                forged.rounds[r].nonces.query = failing(state, *nonce, 8);
                assert_eq!(
                    verify_ground(&forged, &root, &domain, y, &cfg, num_vars),
                    Err(Error::GrindingRejected { bits: 8 }),
                    "{layout:?}, round {r}"
                );
            }
        }
    }

    /// ★ A nonce the round does not spend is refused when set, under `Spent`
    /// only. `Three` carries it unread: today's format, and the reason `Spent`
    /// exists. Every unspent slot of a query-only chain, and the last round's
    /// out-of-domain slot, which no config spends.
    #[test]
    fn an_unspent_nonce_is_refused_where_the_layout_does_not_carry_it() {
        let num_vars = 9;
        let (proof, root, domain, y) = prove_ground(num_vars, &query_only(NonceLayout::Spent));
        let rounds = proof.rounds.len();
        let mut refused = 0;
        for r in 0..rounds {
            for slot in ["folding", "ood"] {
                let mut forged = proof.clone();
                match slot {
                    "folding" => forged.rounds[r].nonces.folding = 1,
                    _ => forged.rounds[r].nonces.ood = 1,
                }
                let under = |layout| {
                    verify_ground(&forged, &root, &domain, y, &query_only(layout), num_vars)
                };
                assert_eq!(
                    under(NonceLayout::Spent),
                    Err(Error::UnspentNonce),
                    "round {r}, the {slot} slot"
                );
                assert_eq!(
                    under(NonceLayout::Three),
                    Ok(()),
                    "round {r}, the {slot} slot: Three must not read it"
                );
                refused += 1;
            }
        }
        assert_eq!(refused, 2 * rounds, "two unspent slots a round");

        // With every grind spent, the one unspent slot is the last round's
        // out-of-domain nonce: refused under `Spent`, unread under `Three`.
        let uniform = |layout| ChainConfig {
            format: ChainFormat {
                nonces: layout,
                ..ChainFormat::DEFAULT
            },
            ..ground(2)
        };
        let (proof, root, domain, y) = prove_ground(num_vars, &uniform(NonceLayout::Spent));
        let mut forged = proof.clone();
        forged.rounds.last_mut().expect("a round").nonces.ood = 1;
        let under = |layout| verify_ground(&forged, &root, &domain, y, &uniform(layout), num_vars);
        assert_eq!(under(NonceLayout::Spent), Err(Error::UnspentNonce));
        assert_eq!(under(NonceLayout::Three), Ok(()));
    }

    /// The field tower: a base-field domain with extension-valued codewords.
    #[test]
    fn the_chain_runs_over_a_field_tower() {
        use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
        type ExtE = FieldElement<Ext>;

        let cfg = config(2);
        let num_vars = 5;
        // The committed polynomial is base-field; only the point is not.
        let f = Mle::new(
            (0..(1u64 << num_vars))
                .map(|i| FE::from(i * 7 + 3))
                .collect(),
        )
        .unwrap();
        let z: Vec<ExtE> = (0..num_vars).map(|i| ExtE::from(101 + i as u64)).collect();
        let y = f.evaluate_in(&z).unwrap();

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let mut prover = DefaultTranscript::<Ext>::new(b"tower");
        let proof = prove::<F, Ext, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut prover)
            .unwrap();

        let mut verifier = DefaultTranscript::<Ext>::new(b"tower");
        verify::<F, Ext, _, KeccakWhir>(
            &proof,
            &commitment.root(),
            &z,
            y,
            &domain,
            &cfg,
            &mut verifier,
        )
        .unwrap();
    }
}
