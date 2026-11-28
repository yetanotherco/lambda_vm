//! `claim_reduce::verify` as a machine leg — the last step of a table's
//! argument, and the one that hands the stacked opening its claims.
//!
//! The mirror is `multilinear::claim_reduce::verify`
//! (`crypto/multilinear/src/claim_reduce.rs:262-321`). The main sumcheck leaves
//! one claim per FACTOR, and a factor is a column read at a frame offset; this
//! reduces all of them to one claim per COLUMN, at a single point, so the whole
//! epoch settles in one stacked opening.
//!
//! # What it does, in the order the transcript sees it
//!
//! Absorb every factor value; draw γ and raise it to the factor count; batch
//! the factor values by those weights into one claim; run a DEGREE-2 sumcheck
//! over the table's variables (`:291` — two, always, because the summand is a
//! kernel times a column and each is multilinear); rebuild that sumcheck's
//! residual as `Σ_offset shift_k(α, y)·(Σ_{factors at that offset} γ^i·c_i)`;
//! refuse if it disagrees; then absorb the column values.
//!
//! # ★ The kernel is per OFFSET, not per factor
//!
//! `shift_eval(α, y, k)` (`:308`) is evaluated once per DISTINCT offset, and
//! every factor reading that offset shares it. The VM's tables read at two
//! offsets — the row and the row after — so a table with dozens of factors
//! spends two kernels, and the per-factor cost is one `MulAdd`. That is the
//! whole reason this leg is cheap, and it is why the form below is a function
//! of the offset SET rather than of the factor count alone.
//!
//! # The refusal
//!
//! `rebuilt != claim.expected_evaluation` is `Error::ShiftedReadMismatch` on
//! the host (`:310-312`) and an `assert_eq_ext` here — a division with no
//! satisfying assignment, so a proof the host rejects has no execution.
//!
//! # ⛔ A non-mutation, recorded rather than dropped (instance 52)
//!
//! Moving the column-value absorbs from AFTER the refusal to before it was run
//! as a mutation and passed every gate — because it is a rewrite. The refusal
//! emits no transcript operation, so carrying the absorbs across it leaves the
//! absorb sequence, every drawn challenge and every row exactly as they were;
//! a straight-line program is a DAG and the order two independent groups are
//! emitted in is not a property of it. The claim written down before running it
//! — "it moves every challenge a later leg draws" — was simply false.
//!
//! The mutation that IS one moves the absorbs above the SUMCHECK, where a draw
//! sits between: every round challenge then comes from a transcript that
//! already holds the column values, and three of the four gates fail.

use multilinear::claim_reduce::FactorSource;

use super::builder::{Ext, LfmBuilder};
use super::whir_poly::{
    challenge_powers_rows, emit_challenge_powers, emit_shift_eval, emit_sumcheck_rounds,
    shift_eval_rows, sumcheck_round_rows,
};
use super::whir_transcript::{WhirTranscript, absorb_unpack_rows, sample_ext_rows};

/// The degree the reduce sumcheck runs at (`claim_reduce.rs:291`).
///
/// Not a parameter: the summand is one multilinear kernel times one multilinear
/// column, and two is what that is.
pub const REDUCE_DEGREE: usize = 2;

/// Rows an `assert_eq_ext` lowers to: the difference and the division by zero
/// (`builder.rs:289-292`).
const ASSERT_ROWS: usize = 2;

/// The proof `claim_reduce::verify` reads, as wires.
pub struct ReduceWires<'a> {
    /// One sumcheck round per variable, each carrying [`REDUCE_DEGREE`]
    /// evaluations.
    pub sumcheck: &'a [Vec<Ext>],
    /// Every committed column's value at the reduced point.
    pub column_values: &'a [Ext],
}

/// What the leg leaves for the stacked opening to settle.
pub struct ReducedClaimWires {
    pub point: Vec<Ext>,
    pub column_values: Vec<Ext>,
}

/// The distinct offsets the factors read, ascending — `claim_reduce::offsets`
/// (`:84-89`), which is emit-time structure because a factor's offset is the
/// AIR's own.
pub fn distinct_offsets(sources: &[FactorSource]) -> Vec<usize> {
    let mut all: Vec<usize> = sources.iter().map(|s| s.offset).collect();
    all.sort_unstable();
    all.dedup();
    all
}

/// INSTRUCTIONS [`emit_claim_reduce_verify`] emits, once the program has
/// interned its `1` and its `0`.
///
/// Every term by the shape it comes from:
///
/// - one `Unpack` per factor value absorbed, and one `Pack` for γ;
/// - `challenge_powers_rows(|sources|)` for the weights;
/// - `|sources| − 1` `MulAdd`s to batch the factor values, because the first
///   weight is the literal one and costs no multiply;
/// - per variable, the sumcheck's `REDUCE_DEGREE` absorbed evaluations, its
///   drawn challenge and `sumcheck_round_rows(REDUCE_DEGREE)`;
/// - per DISTINCT offset, one `shift_eval` over the table's variables, plus one
///   row per factor at that offset (the first is free when its weight is the
///   literal one) and one more to fold the offset's term into the total (free
///   for the first offset, which starts it);
/// - the refusal, and one `Unpack` per column value absorbed.
pub fn claim_reduce_rows(sources: &[FactorSource], num_columns: usize, num_vars: usize) -> usize {
    let factors = sources.len();
    let mut rows = factors * absorb_unpack_rows() + sample_ext_rows();
    rows += challenge_powers_rows(factors);
    rows += factors.saturating_sub(1);
    rows += num_vars
        * (REDUCE_DEGREE * absorb_unpack_rows()
            + sample_ext_rows()
            + sumcheck_round_rows(REDUCE_DEGREE));

    for offset in distinct_offsets(sources) {
        rows += shift_eval_rows(num_vars, offset);
        let members: Vec<usize> = (0..factors)
            .filter(|&i| sources[i].offset == offset)
            .collect();
        // The first member costs nothing when its weight is `γ^0 = 1`, which
        // only the very first factor can be.
        rows += members.len() - usize::from(members[0] == 0);
        // One row to join the kernel to the batch, whichever offset it is: a
        // `Mul` for the first, which opens the total, and a `MulAdd` for every
        // later one, which folds into it.
        rows += 1;
    }

    rows + ASSERT_ROWS + num_columns * absorb_unpack_rows()
}

/// ★ `claim_reduce::verify`, emitted.
///
/// `alpha` is the point the main sumcheck left the factors claimed at; the
/// point this returns is the one every COLUMN is claimed at, which is what the
/// stacked opening settles.
pub fn emit_claim_reduce_verify(
    b: &mut LfmBuilder,
    transcript: &mut WhirTranscript,
    proof: &ReduceWires<'_>,
    sources: &[FactorSource],
    factor_values: &[Ext],
    alpha: &[Ext],
    num_columns: usize,
) -> ReducedClaimWires {
    assert_eq!(
        sources.len(),
        factor_values.len(),
        "one value per factor — the host's `check_shape`, made emit-time"
    );
    assert!(!sources.is_empty(), "a table has factors");
    assert_eq!(
        proof.column_values.len(),
        num_columns,
        "one value per committed column"
    );
    assert_eq!(
        proof.sumcheck.len(),
        alpha.len(),
        "the reduce sumcheck runs over the table's variables"
    );
    assert!(
        sources.iter().all(|s| s.column < num_columns),
        "every factor reads a column the table committed"
    );

    for &value in factor_values {
        transcript.absorb_ext(b, value);
    }
    let gamma = transcript.sample_ext(b);
    let weights = emit_challenge_powers(b, gamma, sources.len());

    // `Σ γ^i · v_i`, with the first weight the literal one.
    let mut claimed = factor_values[0];
    for (weight, &value) in weights.iter().zip(factor_values).skip(1) {
        claimed = b.emul_add(*weight, value, claimed);
    }

    let mut point = Vec::with_capacity(alpha.len());
    for round in proof.sumcheck {
        assert_eq!(
            round.len(),
            REDUCE_DEGREE,
            "the reduce sumcheck is degree two"
        );
        for &evaluation in round {
            transcript.absorb_ext(b, evaluation);
        }
        point.push(transcript.sample_ext(b));
    }
    let residual = emit_sumcheck_rounds(b, claimed, proof.sumcheck, &point);

    let mut rebuilt: Option<Ext> = None;
    for offset in distinct_offsets(sources) {
        let members: Vec<usize> = (0..sources.len())
            .filter(|&i| sources[i].offset == offset)
            .collect();
        let mut batched: Option<Ext> = None;
        for &i in &members {
            let column = proof.column_values[sources[i].column];
            batched = Some(match batched {
                // `γ^0` is the literal one: the column value IS the term.
                None if i == 0 => column,
                None => b.emul(weights[i], column),
                Some(acc) => b.emul_add(weights[i], column, acc),
            });
        }
        let batched = batched.expect("an offset in the list has a factor reading it");
        let kernel = emit_shift_eval(b, alpha, &point, offset);
        rebuilt = Some(match rebuilt {
            None => b.emul(kernel, batched),
            Some(acc) => b.emul_add(kernel, batched, acc),
        });
    }
    b.assert_eq_ext(rebuilt.expect("a table has at least one offset"), residual);

    for &value in proof.column_values {
        transcript.absorb_ext(b, value);
    }

    ReducedClaimWires {
        point,
        column_values: proof.column_values.to_vec(),
    }
}
