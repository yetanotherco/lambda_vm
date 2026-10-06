//! `fold_coset` as a machine leg — the arithmetic half of the query phase.
//!
//! `whir_commit::fold_coset` (`crypto/multilinear/src/whir_commit.rs:475-511`)
//! folds one opened block down to the single value it contributes, and the
//! verifier runs it once per query per round: at the round's own successor
//! block in `whir_round::verify` (`:159`) and against the final constant in
//! `verify_final` (`:1138`). The hashing half is [`super::whir_open`] beside
//! it.
//!
//! Each level halves the block by `fold_block_level` (`:434-465`):
//!
//! ```text
//!     out[t] = ½·(a + b) + α·(½·x⁻¹)·(a − b),   a = v[t], b = v[t + half]
//! ```
//!
//! with `x` running over the coset: slot `t` sits at `g^p·η^t`, where `p` is
//! the block's position in the level's domain and `η` is that level's stride.
//!
//! # ★ Why only ONE exponentiation, and the identity that buys it
//!
//! `x` starts at `generator^position`, which is RUNTIME — `position` is the
//! query index the transcript drew — so it is built from the index's BITS by
//! [`super::edsl::pow_bits`], whose factors `g^(2^i)` are program constants.
//!
//! Every LATER level is free of a second exponentiation. The host squares the
//! domain and reduces the position into it (`:501-507`), so its point is
//! `(g²)^(p mod n/2)`; and
//!
//! ```text
//!     (g²)^(p mod n/2) = g^(2p − k·n) = g^(2p) = (g^p)²
//! ```
//!
//! because `g` has order `n` (`Domain::squared` squares the generator and drops
//! one from the log size, `whir.rs:50-58`). So the emitter squares the point it
//! already has, and a squaring is one row where a second `pow_bits` would be
//! `2·index_bits`. `η` is a program constant at every level, because the domain
//! and the block width are both emit-time.
//!
//! ⚠ The reduction reads as the hard part of that identity and is not part of
//! it at all: `Domain::new` takes a PRIMITIVE `2^log_size`-th root
//! (`whir.rs:36-47`), so each level's generator has order exactly that level's
//! size and `g_l^p = g_l^(p mod n_l)` for every `p`. The host's `position %=`
//! (`whir_commit.rs:502`, `:507`) keeps the exponent small for `pow` and
//! changes no value. Recorded because an earlier draft of this doc — and a test
//! written against it — treated the reduction as something the emitter had to
//! survive.
//!
//! # The base round
//!
//! Round 0's current codeword is base-field, and the host folds its first level
//! in the base field before the multiply by `α` lifts it. This emits that level
//! in the extension instead: the values agree exactly — a base element embedded
//! in the cubic extension is `(v, 0, 0)`, and every operation here is the
//! embedding of the host's — so it is a COST difference and not a value one. It
//! is worth naming rather than hiding: an `LFM_XALU` row is 18 base-equivalent
//! cells against `LFM_BALU`'s 10, so folding round 0's first level in the base
//! field would save `8 · (18 − 10)` cells a query and is a lever for the
//! optimisation rounds, not a correctness matter.
//!
//! ⚠ Lifting a HINTED base value is only sound because the block's leaf hash
//! pins its upper lanes to zero — see [`super::whir_open`]'s module doc, which
//! carries the obligation.
//!
//! # Provenance
//!
//! The shape of this emitter follows an uncommitted draft left in lane V1b's
//! worktree. The closed form below, its terms, its gates and its mutations are
//! this lane's own: the form was re-derived from `fold_block_level` before the
//! draft's was read back, and the two agree.

use multilinear::whir::Domain;

use crate::tables::types::{FE, GoldilocksField};

use super::builder::{Bit, Ext, LfmBuilder};
use super::edsl::{pow_bits, pow_bits_alu, pow_bits_windowed};
use super::word::{LfmWord, base_word};

/// INSTRUCTIONS [`emit_fold_coset`] emits for a block of `block` values over
/// `log2(block)` levels, with a query index of `index_bits` bits.
///
/// Every term by the shape it comes from:
///
/// - `2·index_bits` for the ONE `pow_bits` — a `Select` and a `Mul` a bit
///   (`edsl.rs:856-880`);
/// - SIX rows an output slot: `a + b`, its halving, `½·x⁻¹` (ONE `Div`, because
///   `Div` is a reversed-multiply constraint so the halving rides along with
///   the reciprocal — `chips.rs:175`), `a − b`, its scaling, and the `MulAdd`
///   that joins the halves at `α`. A block of `2^levels` has `block − 1` output
///   slots across all its levels, so that is `6·(block − 1)`;
/// - ONE step of `x` along the coset per slot except the last of its level,
///   which is `(block − 1) − levels`;
/// - ONE squaring per level after the first, which is `levels − 1`.
///
/// The last two cancel to a constant: `−levels + levels − 1 = −1`. So
///
/// ```text
///     2·index_bits + 7·(block − 1) − 1
/// ```
///
/// and the cancellation is why the form has no `levels` in it even though two
/// of its terms do.
pub const fn fold_coset_rows(block: usize, index_bits: usize) -> usize {
    if block <= 1 {
        // `fold_coset` returns `values[0]` lifted when there is nothing to fold
        // (`whir_commit.rs:492-494`) and the emitter returns the wire. No chain
        // reaches it — a round folds at least one variable — and it is stated
        // rather than left to underflow `block − 1`.
        return 0;
    }
    2 * index_bits + 7 * (block - 1) - 1
}

/// `LFM_CONST` rows a fold interns.
///
/// The constants a fold NAMES are `pow_bits`' scale of one, one factor
/// `g^(2^i)` per index bit, the half, and one coset stride per level that has
/// two slots to step between. That count is an upper bound and not the answer,
/// because **the strides and the factors are powers of the same generator and
/// DO collide** — found by this form's own assertion failing, not predicted.
///
/// Every constant but the half is `g^e` for an exponent the shape fixes:
///
/// ```text
///     scale    e = 0
///     factor i e = 2^i                        for i < index_bits
///     stride l e = (N / block) · 2^l          for l < levels − 1
/// ```
///
/// with `N` the domain's size — level `l` has `N/2^l` points and `block/2^l`
/// values, so its stride exponent is `(N/block)·2^l` measured in the ORIGINAL
/// generator. Two constants are the same felt exactly when their exponents
/// agree modulo `N`, so the count is the size of that exponent set plus one for
/// the half. At a 2^8 domain with a block of sixteen and six index bits the
/// strides are `g^16, g^32, g^64` and the factors already include `g^16` and
/// `g^32`: eleven names, nine constants.
///
/// ⚠ This is reached in production, not only in a test. At `S = 25` a round's
/// domain is `2^(27 − 4r)` and its index is `D_r − 4` bits wide, so the first
/// stride `g^16` is the index factor `2^4` in every round whose index is more
/// than four bits — which is most of them.
///
/// The arithmetic here is over integer exponents modulo `N`. Nothing evaluates
/// the field, so this is a derivation from the shape rather than a second copy
/// of what the emitter interns.
///
/// ★ The half is counted as one more and assumed distinct from every `g^e`.
/// That is an assumption about a discrete log, not a proof; it is what the
/// tests would catch if it ever failed, in the same way the stride collision
/// was caught.
pub fn fold_coset_consts(log_domain: usize, levels: usize, index_bits: usize) -> usize {
    if levels == 0 {
        return 0;
    }
    fold_coset_exponents(log_domain, levels, index_bits).len() + 1
}

/// ⛔⛔ **NEVER UNION THIS ACROSS DOMAINS**, which is the warning the Newton
/// count needed in the other direction and this one needs in its own.
///
/// `sumcheck_round_consts`' pairs NEST in the degree, so one call at a
/// program's maximum is its whole pool. These do NOT nest: the exponent set is
/// the same integers under a DIFFERENT generator, so two chains over different
/// domains intern DIFFERENT field elements for the same exponents. A program's
/// fold pool is a genuine UNION over its distinct domains — a maximum of
/// anything is wrong here — and a count cannot express a union any more than it
/// could express the other.
///
/// ⇒ [`fold_coset_constants`] is the program-level form; this one is correct
/// for exactly one thing, which is what ONE fold over ONE domain interns.
///
/// The exponents `g` is raised to, deduplicated — the shape half of what
/// [`emit_fold_coset`] interns, over integer exponents modulo `N`.
///
/// - `0`, the base [`pow_bits`] starts its accumulator at;
/// - `2^i` for each index bit, the factors `pow_bits` multiplies in;
/// - `(N / block) << l` at each level that has two slots to step between —
///   level `l`'s stride, which is `g^{2^{l + log_domain − levels}}` once the
///   squared domain's generator is unfolded.
fn fold_coset_exponents(log_domain: usize, levels: usize, index_bits: usize) -> Vec<u128> {
    if levels == 0 {
        return Vec::new();
    }
    let n = 1u128 << log_domain;
    let block = 1u128 << levels;
    let mut exponents = vec![0u128];
    for i in 0..index_bits {
        exponents.push((1u128 << i) % n);
    }
    for l in 0..levels - 1 {
        exponents.push(((n / block) << l) % n);
    }
    exponents.sort_unstable();
    exponents.dedup();
    exponents
}

/// ★★ THE VALUES ONE FOLD INTERNS — the form a program-level pool can consume.
///
/// [`fold_coset_consts`] counts these and a count is exactly what a pool cannot
/// take: the builder interns on the canonical word, so two folds sharing a
/// value pay for it once and adding their counts charges it twice. Same defect
/// as the sumcheck round's, found the same way — as an unnamed remainder in an
/// assembled program's pool.
///
/// ★ AND IT RETIRES AN ASSUMPTION. The count's own doc says its last term —
/// `two_inv` — is "counted as one more and assumed distinct from every `g^e`",
/// and calls that "an assumption about a discrete log, not a proof". This form
/// does not need it: the words are deduplicated BY VALUE, so a collision makes
/// the pool one word smaller and the count identity is where it shows.
pub fn fold_coset_constants(
    domain: &Domain<GoldilocksField>,
    levels: usize,
    index_bits: usize,
) -> Vec<LfmWord> {
    if levels == 0 {
        return Vec::new();
    }
    let log_domain = domain.size().trailing_zeros() as usize;
    let generator = domain.generator();
    let two_inv = (FE::one() + FE::one())
        .inv()
        .expect("2 is invertible in Goldilocks");

    let mut words: Vec<LfmWord> = Vec::new();
    for exponent in fold_coset_exponents(log_domain, levels, index_bits) {
        let word = base_word(generator.pow(exponent as u64));
        if !words.contains(&word) {
            words.push(word);
        }
    }
    let half = base_word(two_inv);
    if !words.contains(&half) {
        words.push(half);
    }
    words
}

/// ★ `whir_commit::fold_coset`, emitted.
///
/// `values` are the block's openings in the order the host holds them —
/// `coset_of`'s order, which is what `CosetOpening::values` carries and what the
/// leaf hashed. A round-0 block is base-field on the host and is lifted by the
/// caller; see the module doc for why that lift is sound and what it costs.
///
/// Panics when the block is not `2^alphas.len()`, which the host returns an
/// error for. The block's width is an emit-time constant here, so a mismatch is
/// a bug in the emitter rather than a condition to carry at runtime — the
/// `epoch_verify.rs:171-179` idiom, where the absence of a second value
/// replaces a check somebody could forget.
pub fn emit_fold_coset(
    b: &mut LfmBuilder,
    values: &[Ext],
    domain: &Domain<GoldilocksField>,
    index_bits: &[Bit],
    alphas: &[Ext],
) -> Ext {
    assert_eq!(
        values.len(),
        1usize << alphas.len(),
        "a block holds one value per folded variable"
    );
    if alphas.is_empty() {
        return values[0];
    }

    let two_inv = (FE::one() + FE::one())
        .inv()
        .expect("2 is invertible in Goldilocks");
    let half_const = b.felt_const(two_inv);

    // `x` at the first level: `generator^position`, from the index's bits. The
    // factors are program constants; the bits are the transcript's.
    let generator = domain.generator();
    let factors: Vec<FE> = (0..index_bits.len())
        .map(|i| generator.pow(1u64 << i))
        .collect();
    let mut x = pow_bits(b, index_bits, &factors, FE::one());

    let mut current: Vec<Ext> = values.to_vec();
    let mut current_domain = domain.clone();

    for (level, alpha) in alphas.iter().enumerate() {
        if level > 0 {
            // The squared domain's point is the previous point squared — the
            // identity in the module doc, and the whole reason there is one
            // `pow_bits` and not `levels` of them.
            x = b.mul(x, x);
        }
        let half = current.len() / 2;
        // This level's coset stride, a program constant; only needed when the
        // level has two slots to step between.
        let stride = (half > 1).then(|| {
            let eta = current_domain
                .generator()
                .pow((current_domain.size() / current.len()) as u64);
            b.felt_const(eta)
        });

        let mut next = Vec::with_capacity(half);
        let mut point = x;
        for t in 0..half {
            let (a, c) = (current[t], current[t + half]);
            let sum = b.eadd(a, c);
            let even = b.emul_base(sum, half_const);
            // `½·x⁻¹` in ONE row. A domain element is never zero, so `Div`'s
            // `0/0 = 1` convention is unreachable and the reciprocal is the
            // only satisfying assignment.
            let point_inv = b.div(half_const, point);
            let difference = b.esub(a, c);
            let odd = b.emul_base(difference, point_inv);
            next.push(b.emul_add(odd, *alpha, even));
            if t + 1 < half {
                point = b.mul(point, stride.expect("a level with two slots has a stride"));
            }
        }
        current = next;
        current_domain = current_domain
            .squared()
            .expect("a domain of at least two points squares");
    }

    current[0]
}

// ============================================================================
// The lean fold (the default) and the classic one (the opt-out)
// ============================================================================

/// ★ `LAMBDA_VM_WHIR_FOLD_CLASSIC=1` emits every WHIR coset fold through
/// [`emit_fold_coset`], seven rows a folded value: the emission before the lean
/// fold became the default, instruction for instruction — the A/B arm and the
/// rollback switch.
///
/// By default every fold goes through [`emit_fold_coset_lean`], about three rows
/// a folded value. The folds were 57 % of a WHIR wrap's chain rows and 46 % of
/// all its rows; measured on block 25368371 (one RTX 5090, ABBA): WHIR −2.85 s,
/// level-0 instructions −20 %, census −154.7 M cells, hash rows unchanged.
///
/// Read once per process, at program EMISSION: the fold is verifier arithmetic
/// inside the emitted program, so the setting moves the WHIR wrap programs (and
/// their ids) and no proof format — the base proof a wrap verifies is the same
/// either way, and both emissions compute the same field value
/// (`the_lean_fold_*` tests). A test picks either with [`with_fold_emission`].
pub const CLASSIC_FOLD_ENV: &str = "LAMBDA_VM_WHIR_FOLD_CLASSIC";

/// Which fold sequence the emitter writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldEmission {
    /// [`emit_fold_coset`]: `7·(block − 1)` rows a block plus the point chain.
    Classic,
    /// [`emit_fold_coset_lean`]: `3·(block − 1)` rows a block plus the point
    /// chain and two a level.
    Lean,
}

std::thread_local! {
    static FOLD_OVERRIDE: std::cell::Cell<Option<FoldEmission>> =
        const { std::cell::Cell::new(None) };
}

impl FoldEmission {
    /// The emission in force on this thread: a [`with_fold_emission`] override
    /// if one is open, else the process's [`CLASSIC_FOLD_ENV`] setting.
    pub fn current() -> Self {
        FOLD_OVERRIDE
            .with(|o| o.get())
            .unwrap_or_else(Self::for_process)
    }

    /// [`CLASSIC_FOLD_ENV`]'s reading of a raw value: unset is the lean fold.
    pub fn from_setting(raw: Option<&str>) -> Self {
        if super::airs::env_switch(CLASSIC_FOLD_ENV, raw).unwrap_or(false) {
            FoldEmission::Classic
        } else {
            FoldEmission::Lean
        }
    }

    /// The process's setting, read once; the emission is named on stderr either
    /// way, so a log states which programs it proved.
    fn for_process() -> Self {
        static PROCESS: std::sync::OnceLock<FoldEmission> = std::sync::OnceLock::new();
        *PROCESS.get_or_init(|| {
            let emission = Self::from_setting(std::env::var(CLASSIC_FOLD_ENV).ok().as_deref());
            super::airs::announce(&match emission {
                FoldEmission::Classic => format!("WHIR FOLD: classic ({CLASSIC_FOLD_ENV}=1)"),
                FoldEmission::Lean => "WHIR FOLD: lean (the default)".to_string(),
            });
            emission
        })
    }
}

/// Run `f` with `emission` in force on this thread — how a test builds either
/// chain whatever the process setting. Restored on return and on unwind.
pub fn with_fold_emission<R>(emission: FoldEmission, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<FoldEmission>);
    impl Drop for Restore {
        fn drop(&mut self) {
            FOLD_OVERRIDE.with(|o| o.set(self.0));
        }
    }
    let _restore = Restore(FOLD_OVERRIDE.with(|o| o.replace(Some(emission))));
    f()
}

/// ★ `LFM_WHIR_POW_WINDOW=1` reads the lean fold's point `x₀⁻¹` through
/// [`POW_WINDOWS`] 2-bit windows ([`pow_bits_windowed`], I-PADLEAF §7.2): no
/// identity multiply and a 4-way `Select` mux a window. That moves three
/// `LFM_BALU` rows a fold into two `LFM_SELECT` rows (two and one below four
/// index bits), which takes a #1014 4-group leaf's `LFM_BALU` from 282,768 rows
/// to 256,776, under 2^18, while its `LFM_SELECT` stays under 2^19. Unset or `0`
/// keeps [`pow_bits`]: today's programs and ids.
///
/// Read once per process at program EMISSION, as [`CLASSIC_FOLD_ENV`] is: it moves
/// the programs that verify a WHIR proof (and their ids), never a proof format,
/// and both settings compute the same field value. The verifier derives the
/// leaf programs under its own setting. A test picks either with
/// [`with_pow_windows`].
pub const POW_WINDOW_ENV: &str = "LFM_WHIR_POW_WINDOW";

/// Bit pairs the windowed point muxes: two. A third would pay its Select from
/// the `LFM_SELECT` headroom for no `LFM_BALU` step left to clear.
pub const POW_WINDOWS: usize = 2;

std::thread_local! {
    static POW_OVERRIDE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// [`POW_WINDOW_ENV`]'s reading of a raw value: [`POW_WINDOWS`] when on, else `0`.
pub fn pow_windows_from_setting(raw: Option<&str>) -> usize {
    if super::airs::env_switch(POW_WINDOW_ENV, raw).unwrap_or(false) {
        POW_WINDOWS
    } else {
        0
    }
}

/// The windows the lean fold's point reads in force on this thread: a
/// [`with_pow_windows`] override if one is open, else the process's
/// [`POW_WINDOW_ENV`] setting (read once, named on stderr).
pub fn pow_windows() -> usize {
    POW_OVERRIDE.with(|o| o.get()).unwrap_or_else(|| {
        static PROCESS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *PROCESS.get_or_init(|| {
            let windows = pow_windows_from_setting(std::env::var(POW_WINDOW_ENV).ok().as_deref());
            super::airs::announce(&if windows == 0 {
                "WHIR FOLD POINT: one Select and one Mul a bit (the default)".to_string()
            } else {
                format!("WHIR FOLD POINT: {windows} 2-bit windows ({POW_WINDOW_ENV}=1)")
            });
            windows
        })
    })
}

/// Run `f` with `windows` in force on this thread. Restored on return and on
/// unwind.
pub fn with_pow_windows<R>(windows: usize, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<usize>);
    impl Drop for Restore {
        fn drop(&mut self) {
            POW_OVERRIDE.with(|o| o.set(self.0));
        }
    }
    let _restore = Restore(POW_OVERRIDE.with(|o| o.replace(Some(windows))));
    f()
}

/// A round's folding challenges as the lean fold consumes them: `½·α` per level,
/// computed ONCE per round rather than once per query, plus the half itself.
pub struct LeanAlphas {
    half_alphas: Vec<Ext>,
    half: super::builder::Felt,
}

/// [`LeanAlphas`] for one round: one `MulBase` a level.
pub fn prepare_lean_alphas(b: &mut LfmBuilder, alphas: &[Ext]) -> LeanAlphas {
    let half = b.felt_const(two_inv());
    LeanAlphas {
        half_alphas: alphas.iter().map(|a| b.emul_base(*a, half)).collect(),
        half,
    }
}

fn two_inv() -> FE {
    (FE::one() + FE::one())
        .inv()
        .expect("2 is invertible in Goldilocks")
}

/// ★ `whir_commit::fold_coset`, emitted lean — the same value as
/// [`emit_fold_coset`], in three extension rows a folded value and none in the
/// base field.
///
/// Each level's output is rewritten around the SECOND value of its pair:
///
/// ```text
///     ½·(a + c) + ½·α·x⁻¹·(a − c)  =  (a − c)·w + c,    w = ½ + ½·α·x⁻¹
/// ```
///
/// and along a level `x_t = x_0·η^t`, so `½·α·x_t⁻¹ = u·η^(−t)` with
/// `u = ½·α·x_0⁻¹` — ONE extension product a level. Per slot that leaves
/// `w = u·η^(−t) + ½` (a `MulAdd` against the program constant `η^(−t)`),
/// `a − c`, and the `MulAdd` that joins them: three `LFM_XALU` rows, where the
/// classic sequence pays five plus a base `Div` for `½·x⁻¹` and a base `Mul` to
/// step `x`. No reciprocal is taken at all: `x_0⁻¹ = g^(−p)` comes straight out
/// of [`pow_bits`] over the INVERSE factors `g^(−2^i)`, and each level after the
/// first squares it (`(x²)⁻¹ = (x⁻¹)²`, the classic identity inverted).
///
/// Rows: `2·index_bits` (the point) + `levels − 1` (squarings) + `levels` (the
/// `u` of each level) + `3·(block − 1)`; see [`fold_coset_rows_lean`]. The
/// `½·α` of [`prepare_lean_alphas`] are per ROUND and not counted here.
pub fn emit_fold_coset_lean(
    b: &mut LfmBuilder,
    values: &[Ext],
    domain: &Domain<GoldilocksField>,
    index_bits: &[Bit],
    prepared: &LeanAlphas,
) -> Ext {
    let levels = prepared.half_alphas.len();
    assert_eq!(
        values.len(),
        1usize << levels,
        "a block holds one value per folded variable"
    );
    if levels == 0 {
        return values[0];
    }

    // `x_0⁻¹ = g^(−p)`, from the index's bits against the inverse factors.
    let generator_inv = domain
        .generator()
        .inv()
        .expect("a domain generator is nonzero");
    let factors: Vec<FE> = (0..index_bits.len())
        .map(|i| generator_inv.pow(1u64 << i))
        .collect();
    // A Poseidon1 leaf's `Select` table binds its leaves (I-WHIR-P1 §S5.4): its
    // point reads the bits through the base ALU instead.
    let mut x_inv = match (b.wrap_hash(), pow_windows()) {
        (super::edsl::WrapHash::Poseidon1, _) => pow_bits_alu(b, index_bits, &factors),
        (_, 0) => pow_bits(b, index_bits, &factors, FE::one()),
        (_, windows) => pow_bits_windowed(b, index_bits, &factors, windows),
    };
    let half = prepared.half.as_ext();

    let mut current: Vec<Ext> = values.to_vec();
    let mut current_domain = domain.clone();
    for (level, half_alpha) in prepared.half_alphas.iter().enumerate() {
        if level > 0 {
            x_inv = b.mul(x_inv, x_inv);
        }
        let pairs = current.len() / 2;
        // This level's stride `η`, inverted: `η^(−t)` is a program constant per
        // slot, so the point never steps at run time.
        let eta_inv = current_domain
            .generator()
            .pow((current_domain.size() / current.len()) as u64)
            .inv()
            .expect("a coset stride is nonzero");
        let u = b.emul_base(*half_alpha, x_inv);

        let mut next = Vec::with_capacity(pairs);
        for t in 0..pairs {
            let (a, c) = (current[t], current[t + pairs]);
            let w = if t == 0 {
                b.eadd(u, half)
            } else {
                let step = b.felt_const(eta_inv.pow(t as u64));
                b.emul_add(u, step.as_ext(), half)
            };
            let difference = b.esub(a, c);
            next.push(b.emul_add(difference, w, c));
        }
        current = next;
        current_domain = current_domain
            .squared()
            .expect("a domain of at least two points squares");
    }

    current[0]
}

/// INSTRUCTIONS [`emit_fold_coset_lean`] emits for a block of `block` values
/// with a query index of `index_bits` bits:
///
/// ```text
///     2·index_bits + 3·(block − 1) + 2·levels − 1
/// ```
///
/// `2·index_bits` for the point, three rows an output slot (`block − 1` of them
/// across the levels), one `u` a level and one squaring a level after the first.
pub const fn fold_coset_rows_lean(block: usize, index_bits: usize) -> usize {
    fold_coset_rows_lean_windowed(block, index_bits, 0)
}

/// [`fold_coset_rows_lean`] with the point read through `windows` 2-bit
/// windows ([`POW_WINDOW_ENV`]): `2·index_bits − 1` rows for the point when it
/// has a bit ([`pow_bits_windowed`]), whatever the windows.
pub const fn fold_coset_rows_lean_windowed(
    block: usize,
    index_bits: usize,
    windows: usize,
) -> usize {
    if block <= 1 {
        return 0;
    }
    let levels = block.trailing_zeros() as usize;
    let point = if windows == 0 {
        2 * index_bits
    } else if index_bits == 0 {
        0
    } else {
        2 * index_bits - 1
    };
    point + 3 * (block - 1) + 2 * levels - 1
}

/// Rows [`prepare_lean_alphas`] emits for a round of `levels` levels: one
/// `MulBase` each.
pub const fn lean_prepare_rows(levels: usize) -> usize {
    levels
}

/// [`fold_coset_rows`] or [`fold_coset_rows_lean`], per [`FoldEmission::current`].
pub fn fold_rows_for(block: usize, index_bits: usize) -> usize {
    match FoldEmission::current() {
        FoldEmission::Classic => fold_coset_rows(block, index_bits),
        FoldEmission::Lean => fold_coset_rows_lean_windowed(block, index_bits, pow_windows()),
    }
}

/// ★ THE VALUES ONE LEAN FOLD INTERNS, deduplicated by value —
/// [`fold_coset_constants`]' twin: `g^(−e)` for `e` in `{0} ∪ {2^i : i < index_bits}`
/// (the scale and the inverse factors of the point) and `{s·N/block : 1 ≤ s < block/2}`
/// (every `η_l^(−t)` of every level is one of level 0's `η_0^(−s)`, since
/// `η_l = η_0^(2^l)`), plus the half.
pub fn fold_coset_constants_lean(
    domain: &Domain<GoldilocksField>,
    levels: usize,
    index_bits: usize,
) -> Vec<LfmWord> {
    fold_coset_constants_lean_windowed(domain, levels, index_bits, 0)
}

/// [`fold_coset_constants_lean`] with the point read through `windows` 2-bit
/// windows ([`POW_WINDOW_ENV`]): each window also interns its factors' product,
/// `g^(−(2^i + 2^(i+1)))` for the pair at bit `i`.
pub fn fold_coset_constants_lean_windowed(
    domain: &Domain<GoldilocksField>,
    levels: usize,
    index_bits: usize,
    windows: usize,
) -> Vec<LfmWord> {
    if levels == 0 {
        return Vec::new();
    }
    let n = domain.size() as u128;
    let block = 1u128 << levels;
    let generator_inv = domain
        .generator()
        .inv()
        .expect("a domain generator is nonzero");
    let mut exponents = vec![0u128];
    for i in 0..index_bits {
        exponents.push((1u128 << i) % n);
    }
    for w in 0..windows.min(index_bits / 2) {
        exponents.push(((1u128 << (2 * w)) + (1u128 << (2 * w + 1))) % n);
    }
    for s in 1..block / 2 {
        exponents.push((s * (n / block)) % n);
    }
    let mut words: Vec<LfmWord> = Vec::new();
    for e in exponents {
        let word = base_word(generator_inv.pow(e as u64));
        if !words.contains(&word) {
            words.push(word);
        }
    }
    let half = base_word(two_inv());
    if !words.contains(&half) {
        words.push(half);
    }
    words
}

/// [`fold_coset_constants`] or [`fold_coset_constants_lean`], per
/// [`FoldEmission::current`].
pub fn fold_constants_for(
    domain: &Domain<GoldilocksField>,
    levels: usize,
    index_bits: usize,
) -> Vec<LfmWord> {
    match FoldEmission::current() {
        FoldEmission::Classic => fold_coset_constants(domain, levels, index_bits),
        FoldEmission::Lean => {
            fold_coset_constants_lean_windowed(domain, levels, index_bits, pow_windows())
        }
    }
}

/// A round's fold, prepared once per round and applied once per query — the one
/// call site the chain emitter needs whichever [`FoldEmission`] is in force.
pub enum RoundFold {
    /// The classic fold takes the round's challenges as they are.
    Classic(Vec<Ext>),
    /// The lean fold takes them halved, once.
    Lean(LeanAlphas),
}

impl RoundFold {
    /// Prepare the round's challenges for the emission in force. The classic
    /// arm emits nothing.
    pub fn prepare(b: &mut LfmBuilder, alphas: &[Ext]) -> Self {
        match FoldEmission::current() {
            FoldEmission::Classic => RoundFold::Classic(alphas.to_vec()),
            FoldEmission::Lean => RoundFold::Lean(prepare_lean_alphas(b, alphas)),
        }
    }

    /// Fold one opened block at one query.
    pub fn fold(
        &self,
        b: &mut LfmBuilder,
        values: &[Ext],
        domain: &Domain<GoldilocksField>,
        index_bits: &[Bit],
    ) -> Ext {
        match self {
            RoundFold::Classic(alphas) => emit_fold_coset(b, values, domain, index_bits, alphas),
            RoundFold::Lean(prepared) => {
                emit_fold_coset_lean(b, values, domain, index_bits, prepared)
            }
        }
    }
}
