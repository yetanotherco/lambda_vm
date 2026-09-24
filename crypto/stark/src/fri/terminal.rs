//! Shared, pure FRI early-termination helpers used by both the prover
//! (`commit_phase_from_evaluations`, `try_fri_commit_gpu`) and the verifier
//! (`step_3_verify_fri`): the fold layout (`FriFoldLayout`) and the conversion
//! between a terminal codeword and the coefficients of the low-degree
//! polynomial it encodes. No transcript, no FRI protocol state.

use math::fft::bit_reversing::{in_place_bit_reverse_permute, reverse_index};
use math::field::element::FieldElement;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use math::polynomial::Polynomial;

use crate::fri::schedule::{FRI_SCHEDULE_DMAX, FriFormat, FriFormatError};
use crate::proof::options::ProofOptions;

/// The FRI early-termination fold layout.
///
/// Derived identically by the CPU prover (`commit_phase_from_evaluations`), the
/// GPU prover (`try_fri_commit_gpu`), and the verifier (`fri_termination_params`).
/// Keeping the arithmetic in one place is load-bearing: the three callers must
/// agree exactly or proofs fail to verify, and a CPU/GPU disagreement would
/// surface only on GPU machines.
///
/// The committed layers follow a fold schedule (`crate::fri::schedule`): layer
/// `j` folds by `2^{schedule[j]}`. Today's layout ([`Self::new`]) is the
/// all-ones schedule, built through the same constructor as every other format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FriFoldLayout {
    /// Folds from the LDE codeword down to the terminal codeword.
    pub(crate) total_folds: u32,
    /// Committed (Merkle-rooted) FRI layers = `schedule.len()`. Row-pair
    /// layout: `total_folds - 1` under the all-ones schedule, or 0 when there
    /// is no fold or only a single final fold.
    pub(crate) num_committed: usize,
    /// Terminal codeword length = `2^(blowup_log + effective_k)`.
    pub(crate) terminal_len: usize,
    /// Terminal polynomial log-degree bound actually used, `min(k, trace_bits)`.
    /// This is the verifier's `expected_k` and the prover's `effective_log_degree`.
    pub(crate) effective_k: u32,
    /// Fold exponent of each committed layer, first committed layer first.
    /// Invariant (checked by every constructor): `(one_row ? 0 : 1) +
    /// Σ schedule == total_folds` whenever `total_folds >= 1`, and empty
    /// otherwise; every entry is in `1..=FRI_SCHEDULE_DMAX`.
    pub(crate) schedule: Vec<u8>,
    /// Whether the DEEP codeword itself is committed (one-row openings): then
    /// the chain starts at the LDE size and there is no uncommitted fold 0.
    pub(crate) one_row: bool,
    /// Today's FRI encoding ([`FriFormat::is_legacy`]): pair-leaf layer trees
    /// and one sibling value per committed layer per query. `false` = group
    /// leaves (`H::Batched` over `2^d` values) and the full group per layer —
    /// decided by the format, never by the schedule's values, so a `Dp`
    /// schedule that happens to be all ones still uses the group encoding.
    pub(crate) legacy_encoding: bool,
}

impl FriFoldLayout {
    /// Today's layout, derived from the LDE codeword size.
    ///
    /// * `lde_log`    — log2 of the LDE (deep-composition) codeword length.
    /// * `blowup_log` — log2 of the LDE blowup factor.
    /// * `k`          — requested `fri_final_poly_log_degree`.
    ///
    /// Folding stops once the codeword encodes a polynomial of degree `< 2^k`,
    /// i.e. at codeword length `2^(blowup_log + k)`, clamped to the full LDE
    /// size for traces too small to fold that far (the `.min(lde_log)`).
    /// Computing `blowup_log + k` in `u32` (both small) sidesteps the
    /// `1 << (blowup_log + k)` overflow an out-of-range `k` would otherwise cause.
    ///
    /// This is [`Self::for_format`] at [`FriFormat::LEGACY`]: pair layers,
    /// row-pair openings, the all-ones schedule.
    pub(crate) fn new(lde_log: u32, blowup_log: u32, k: u32) -> Self {
        let terminal_log = (blowup_log + k).min(lde_log);
        let schedule = FriFormat::LEGACY.schedule(lde_log, terminal_log);
        // The all-ones schedule covers the committed folds by construction.
        Self::assemble(lde_log, blowup_log, terminal_log, false, schedule)
    }

    /// The layout under an explicit proof format. `total_folds`,
    /// `terminal_len` and `effective_k` do not depend on the format; only the
    /// split of the folds into committed layers does.
    ///
    /// `None` only for a schedule override that does not cover the committed
    /// folds exactly (the DP and the all-ones schedules land on the terminal
    /// by construction).
    pub(crate) fn for_format(
        lde_log: u32,
        blowup_log: u32,
        k: u32,
        fmt: &FriFormat,
    ) -> Option<Self> {
        let terminal_log = (blowup_log + k).min(lde_log);
        let schedule = fmt.schedule(lde_log, terminal_log);
        let mut layout = Self::assemble(lde_log, blowup_log, terminal_log, fmt.one_row, schedule);
        layout.legacy_encoding = fmt.is_legacy();
        layout.schedule_is_consistent().then_some(layout)
    }

    /// The layout of a table proved under `options` over an LDE of
    /// `2^lde_log` with blowup `2^blowup_log`, whose trace trees use the
    /// resolved leaf layout `one_row`: what the prover and the host verifier
    /// both build. The format comes from `options` and the table's AIR — a
    /// verifier-side constant — never from a proof.
    pub(crate) fn for_options(
        lde_log: u32,
        blowup_log: u32,
        options: &ProofOptions,
        one_row: bool,
    ) -> Result<Self, FriFormatError> {
        let fmt = FriFormat::from_options(options, one_row);
        Self::for_format(
            lde_log,
            blowup_log,
            u32::from(options.fri_final_poly_log_degree),
            &fmt,
        )
        .ok_or(FriFormatError::ScheduleOverrideMismatch)
    }

    /// The layout for a caller-supplied schedule, or `None` if the schedule
    /// does not cover exactly the committed folds (or has an exponent outside
    /// `1..=FRI_SCHEDULE_DMAX`). The encoding is the group encoding unless
    /// the schedule is today's (row pair, all ones), where it is legacy.
    #[cfg(test)]
    pub(crate) fn from_schedule(
        lde_log: u32,
        blowup_log: u32,
        k: u32,
        one_row: bool,
        schedule: Vec<u8>,
    ) -> Option<Self> {
        let terminal_log = (blowup_log + k).min(lde_log);
        let mut layout = Self::assemble(lde_log, blowup_log, terminal_log, one_row, schedule);
        layout.legacy_encoding = !one_row && layout.schedule.iter().all(|&d| d == 1);
        layout.schedule_is_consistent().then_some(layout)
    }

    /// Whether this layout uses today's FRI encoding (see
    /// [`Self::legacy_encoding`]). Every device FRI arm is gated on this.
    pub(crate) fn is_legacy(&self) -> bool {
        self.legacy_encoding
    }

    /// Log2 length of committed layer `j` (0-based): the chain start minus the
    /// bits the earlier committed layers consumed.
    pub(crate) fn layer_log_len(&self, lde_log: u32, j: usize) -> u32 {
        let consumed: u32 = self.schedule[..j].iter().map(|&d| u32::from(d)).sum();
        crate::fri::schedule::fri_chain_start(lde_log, self.one_row) - consumed
    }

    /// Depth of committed layer `j`'s tree: its length over `2^{d_j}` leaves.
    pub(crate) fn layer_depth(&self, lde_log: u32, j: usize) -> u32 {
        self.layer_log_len(lde_log, j) - u32::from(self.schedule[j])
    }

    /// Folding challenges a proof of this layout draws: one per committed
    /// layer plus the final fold's for row pairs (fold 0 consumes the first),
    /// one per committed layer for one row (layer 0, the input tree, is
    /// committed before any challenge); none when nothing folds.
    pub(crate) fn num_zetas(&self) -> usize {
        if self.total_folds == 0 {
            0
        } else {
            self.num_committed + usize::from(!self.one_row)
        }
    }

    /// Depth of every committed layer's tree, in layer order.
    pub(crate) fn layer_depths(&self, lde_log: u32) -> Vec<usize> {
        (0..self.num_committed)
            .map(|j| self.layer_depth(lde_log, j) as usize)
            .collect()
    }

    /// Opened values per query in the flat `layers_evaluations_sym` vector:
    /// one per layer (legacy) or every layer's full group.
    pub(crate) fn opened_values_per_query(&self) -> usize {
        if self.legacy_encoding {
            self.num_committed
        } else {
            self.schedule.iter().map(|&d| 1usize << d).sum()
        }
    }

    fn assemble(
        lde_log: u32,
        blowup_log: u32,
        terminal_log: u32,
        one_row: bool,
        schedule: Vec<u8>,
    ) -> Self {
        let total_folds = lde_log - terminal_log;
        Self {
            total_folds,
            num_committed: schedule.len(),
            terminal_len: 1usize << terminal_log,
            effective_k: terminal_log - blowup_log,
            schedule,
            one_row,
            legacy_encoding: !one_row,
        }
    }

    /// The constructor invariant (see [`Self::schedule`]).
    fn schedule_is_consistent(&self) -> bool {
        let entries_ok = self
            .schedule
            .iter()
            .all(|&d| d >= 1 && u32::from(d) <= FRI_SCHEDULE_DMAX);
        let covered: u64 = self.schedule.iter().map(|&d| u64::from(d)).sum();
        let expected = if self.total_folds == 0 {
            0
        } else {
            u64::from(self.total_folds) - u64::from(!self.one_row)
        };
        entries_ok && covered == expected
    }
}

/// Prover side: given a FRI terminal codeword in **bit-reversed** order,
/// recover the `2^final_poly_log_degree` coefficients of the underlying
/// low-degree polynomial.
///
/// The codeword is a coset evaluation of a polynomial of degree less than
/// `2^final_poly_log_degree` on the coset `terminal_offset·⟨ω⟩` of size
/// `blowup·2^k`.
///
/// Algorithm:
/// 1. Bit-reverse permute to convert from FRI order to natural (DFT) order.
/// 2. Decimate: extract the size-`2^k` sub-coset
///    `terminal_offset·⟨ω^blowup⟩` = every `blowup`-th natural-order point.
/// 3. Coset iFFT on the small (`2^k`-point) sub-domain — a `blowup×`-smaller
///    transform that recovers the `2^k` coefficients directly (no oversized
///    transform and no wasteful truncation).
pub(crate) fn coeffs_from_terminal_codeword<F, E>(
    codeword_bitrev: &[FieldElement<E>],
    terminal_offset: &FieldElement<F>,
    final_poly_log_degree: u32,
) -> Vec<FieldElement<E>>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField + Send + Sync,
{
    // A degree-<2^k poly is determined by 2^k points: the size-2^k sub-coset
    // terminal_offset*<w^blowup> = every `blowup`-th natural-order evaluation,
    // i.e. natural-order index m*blowup for m in 0..2^k. The codeword is in
    // bit-reversed order, so gather those points straight from it via
    // reverse_index — no full-codeword clone or O(n) permute (only 2^k of the
    // blowup*2^k evaluations are ever read).
    let len = codeword_bitrev.len();
    let keep = 1usize << final_poly_log_degree;
    let blowup = len / keep;
    let sub_coset: Vec<FieldElement<E>> = (0..keep)
        .map(|m| codeword_bitrev[reverse_index(m * blowup, len as u64)].clone())
        .collect();

    // Coset iFFT on the small domain -> the 2^k coefficients directly (no oversized trim).
    let poly = Polynomial::interpolate_offset_fft::<F>(&sub_coset, terminal_offset)
        .expect("terminal sub-coset must have power-of-two length and non-zero offset");

    // Pad with zeros only if interpolation dropped trailing-zero coeffs, so the
    // proof always carries exactly 2^k coefficients (the verifier length-checks).
    let mut coeffs = poly.coefficients().to_vec();
    coeffs.resize(keep, FieldElement::<E>::zero());
    coeffs
}

/// Verifier side: given `2^k` coefficients of the low-degree polynomial,
/// reconstruct the full FRI terminal codeword in **bit-reversed** order.
///
/// Algorithm:
/// 1. FFT (coset): evaluate the polynomial on the full coset of size
///    `codeword_len` with shift `terminal_offset` to get natural order.
/// 2. Bit-reverse permute to convert natural order to FRI order.
///
/// # Panics
///
/// Panics if any of the following preconditions are violated:
/// - `coeffs` is non-empty,
/// - `coeffs.len()` is a power of two,
/// - `codeword_len` is a power of two,
/// - `coeffs.len() <= codeword_len`, and
/// - `codeword_len` is divisible by `coeffs.len()`.
///
/// In the normal verifier flow these conditions are guaranteed by the
/// final-polynomial length check that the verifier performs before calling
/// this helper, so the assert should never fire in production.
pub(crate) fn terminal_codeword_from_coeffs<F, E>(
    coeffs: &[FieldElement<E>],
    terminal_offset: &FieldElement<F>,
    codeword_len: usize,
) -> Vec<FieldElement<E>>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField + Send + Sync,
{
    assert!(
        !coeffs.is_empty()
            && coeffs.len().is_power_of_two()
            && codeword_len.is_power_of_two()
            && coeffs.len() <= codeword_len
            && codeword_len.is_multiple_of(coeffs.len()),
        "terminal_codeword_from_coeffs: coeffs.len() ({}) must be a non-zero power of two dividing codeword_len ({}); the verifier must length-check coeffs before calling",
        coeffs.len(),
        codeword_len,
    );

    let poly = Polynomial::new(coeffs);
    let blowup = codeword_len / coeffs.len();

    // Step 1: coset FFT to get natural-order evaluations.
    let mut natural =
        Polynomial::evaluate_offset_fft::<F>(&poly, blowup, Some(coeffs.len()), terminal_offset)
            .expect("terminal coset size must be a power of two within the field's two-adicity");

    // Step 2: convert natural order to bit-reversed (FRI) order.
    in_place_bit_reverse_permute(&mut natural);
    natural
}
