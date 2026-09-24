//! Shared, pure FRI early-termination helpers used by both the prover
//! (`commit_phase_from_evaluations`, `try_fri_commit_gpu`) and the verifier
//! (`step_3_verify_fri`): the fold layout (`FriFoldLayout`) and the conversion
//! between a terminal codeword and the coefficients of the low-degree
//! polynomial it encodes. No transcript, no FRI protocol state.

use math::fft::bit_reversing::{in_place_bit_reverse_permute, reverse_index};
use math::field::element::FieldElement;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use math::polynomial::Polynomial;

use crate::fri::schedule::{FRI_SCHEDULE_DMAX, FriFormat};

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
        Self::for_format(lde_log, blowup_log, k, &FriFormat::LEGACY)
    }

    /// The layout under an explicit proof format. `total_folds`,
    /// `terminal_len` and `effective_k` do not depend on the format; only the
    /// split of the folds into committed layers does.
    pub(crate) fn for_format(lde_log: u32, blowup_log: u32, k: u32, fmt: &FriFormat<'_>) -> Self {
        let terminal_log = (blowup_log + k).min(lde_log);
        let schedule = fmt.schedule(lde_log, terminal_log);
        let layout = Self::assemble(lde_log, blowup_log, terminal_log, fmt.one_row, schedule);
        // Holds by construction: both schedules land exactly on the terminal.
        debug_assert!(layout.schedule_is_consistent());
        layout
    }

    /// The layout for a caller-supplied schedule, or `None` if the schedule
    /// does not cover exactly the committed folds (or has an exponent outside
    /// `1..=FRI_SCHEDULE_DMAX`).
    #[allow(dead_code)] // first caller arrives with the S3 prover/verifier.
    pub(crate) fn from_schedule(
        lde_log: u32,
        blowup_log: u32,
        k: u32,
        one_row: bool,
        schedule: Vec<u8>,
    ) -> Option<Self> {
        let terminal_log = (blowup_log + k).min(lde_log);
        let layout = Self::assemble(lde_log, blowup_log, terminal_log, one_row, schedule);
        layout.schedule_is_consistent().then_some(layout)
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
