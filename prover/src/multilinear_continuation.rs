//! Continuations on the multilinear path.
//!
//! The split into epochs, the local-to-global bookend, the cross-epoch register
//! and commit-index carry — all of that is [`crate::continuation`]'s and none of
//! it depends on the commitment scheme. What changes here is only how an
//! epoch's tables are argued: one WHIR commitment over the whole epoch and one
//! opening, the way [`crate::multilinear_prove`] does it for a whole program.
//!
//! # The binding
//!
//! A continuation is two halves: every epoch, and one cross-epoch proof over
//! the bookends and the global-memory tables. What makes them one proof is that
//! an epoch's bookend is the table the cross-epoch proof chained — and a table
//! here has no root of its own, since the stack gives one root per *stacked
//! polynomial*, shared by whatever columns land in it.
//!
//! So each bookend is committed in a group by itself, whose layout follows from
//! the table's shape alone and is therefore the same on both sides, and the
//! binding is comparing those roots. That only holds because a table commits
//! the same columns in the same order wherever it is argued, which is
//! `LeafLayout::build_live_over`'s job, not this module's.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use executor::elf::Elf;
use math::field::element::FieldElement;
use multilinear::mle::Mle;
use multilinear::whir_chain::ChainConfig;
use stark::config::Commitment;
use stark::multilinear_table::{
    self, CommittedTable, CommittedTables, MultiProof, TableLayout, TableStatement,
};
use stark::proof::options::ProofOptions;
use stark::traits::AIR;

use crate::multilinear_prove::chain_config;
use crate::statement;
use crate::tables::local_to_global::{self, CellBoundary};
use crate::tables::register;
use crate::tables::trace_builder::Traces;
use crate::test_utils::{E, F};
use crate::{Error, TableCounts};

/// Domain tag for a multilinear continuation epoch.
///
/// Distinct from both the univariate epoch tag and the monolithic multilinear
/// one: no two of the three may ever share a transcript prefix.
pub(crate) const MULTILINEAR_EPOCH_TAG: &[u8] = b"LAMBDAVM_MULTILINEAR_CONTINUATION_EPOCH_V1";

/// One epoch's proof and everything a standalone verifier re-binds.
///
/// Mirrors [`crate::continuation`]'s, minus the fields that only mean something
/// under FRI: there is no per-table root to carry, and the runtime page ranges
/// are always empty because a continuation epoch skips PAGE.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct EpochProof {
    /// The epoch's tables and the one opening that settles all of them, with
    /// the local-to-global table last.
    pub proof: MultiProof<F, E>,
    /// Each table's height in variables, same order.
    pub table_num_vars: Vec<u8>,
    pub table_counts: TableCounts,
    pub public_output: Vec<u8>,
    /// The epoch's final register file `R_{i+1}`, which the next epoch takes as
    /// its `INIT` — the cross-epoch register binding. x254 rides along.
    pub reg_fini: Vec<u32>,
}

impl EpochProof {
    /// The roots of the commitment the local-to-global bookend has to itself.
    ///
    /// This is what ties the epoch to the cross-epoch proof: the two commit the
    /// same table, and with the bookend in a commitment group of its own its
    /// roots say so. Every other table shares a group.
    ///
    /// `num_polys` is how many polynomials the stack split that group into —
    /// one for a bookend that fits in a stack, more for an epoch long enough
    /// that it does not. The group is the last, so its roots are the tail.
    pub fn l2g_roots(&self, num_polys: usize) -> Option<&[stark::config::Commitment]> {
        if num_polys == 0 {
            return None;
        }
        let start = self.proof.roots.len().checked_sub(num_polys)?;
        Some(&self.proof.roots[start..])
    }
}

/// The roots the local-to-global table commits to on its own — what an epoch
/// proof carries and the cross-epoch proof has to reproduce.
///
/// A function of the table, the blowup and the fold width, and of nothing else:
/// in particular not of the query count, which is what lets two proofs over
/// different table sets agree on them. One root per polynomial the stack split
/// the table into.
pub fn l2g_commitment(
    boundary: &[CellBoundary],
    config: &ChainConfig,
) -> Result<Vec<stark::config::Commitment>, Error> {
    let trace = local_to_global::generate_local_to_global_trace(boundary);
    let columns: Vec<Mle<F>> = trace
        .columns_main()
        .into_iter()
        .map(|values| Mle::new(values).map_err(|e| Error::Prover(format!("{e:?}"))))
        .collect::<Result<_, _>>()?;
    let shape = [(
        columns.len(),
        trace.main_table.height.trailing_zeros() as usize,
    )];
    let layout =
        multilinear_table::global_layout(&shape).map_err(|e| Error::Prover(format!("{e:?}")))?;
    let roots = crate::with_whir_hash!(|H| {
        multilinear::stacked_eval::StackedCommitment::<F, H>::commit(
            layout,
            &multilinear::stacking::borrow(&columns),
            None,
            config,
        )
        .map_err(|e| Error::Prover(format!("{e:?}")))?
        .roots()
        .to_vec()
    });
    if roots.is_empty() {
        return Err(Error::Prover("the bookend commits to nothing".to_string()));
    }
    Ok(roots)
}

/// DECODE's preprocessed columns, committed ONCE per ELF, outside every epoch's
/// proof.
///
/// # Why this exists
///
/// The five columns (`PC_0`, `PC_1`, `PACKED_DECODE`, `IMM_0`, `IMM_1`) are
/// ELF-derived: the same bytes in every epoch of every run of that program.
/// Today each epoch commits them with the rest of DECODE and the verifier then
/// evaluates each column's MLE at that epoch's reduced point — five 2^20 folds,
/// fifteen times, for a value that depended on nothing the epoch chose. Here
/// they are committed once and OPENED per epoch at DECODE's reduced point, and
/// the opening replaces the folds.
///
/// # ★ One derivation, both sides
///
/// The prover and the host verifier call this same function on the same ELF.
/// The verifier therefore absorbs a root it RECOMPUTED, never one the proof
/// handed it — a value that has not been checked must not reach the transcript,
/// or a forged root steers the challenges before the comparison that would have
/// rejected it. Nothing about the commitment is read from the proof: not the
/// root, not the layout, not the domain.
///
/// The digest travels with the roots for the same reason they are computed
/// together: the field machine cannot recompute either in-guest and pins the
/// PAIR as program text, so a pair that could be assembled from two different
/// ELFs is the defect to prevent.
///
/// # Cost, stated
///
/// The host verifier pays one commit over a 2^23 polynomial for the whole
/// proof, in place of `epochs x 5 x 2^20` MLE evaluations. The in-guest verifier
/// pays neither — it pins `(digest, roots)` as emit-time constants.
pub struct DecodePrepared<H>
where
    H: multilinear::whir_hash::WhirHash,
{
    /// The program these columns came from, as the transcript's own digest.
    pub elf_digest: [u8; 32],
    /// The five columns, in DECODE's column order.
    pub columns: Vec<Mle<F>>,
    /// Derived, never read from a proof.
    pub roots: Vec<Commitment>,
    pub commitment: multilinear::stacked_eval::StackedCommitment<F, H>,
    /// The parameters it was committed under — see [`Self::agrees_with`].
    log_blowup: usize,
    log_folding: usize,
}

impl<H> DecodePrepared<H>
where
    H: multilinear::whir_hash::WhirHash,
{
    /// Where DECODE's prepared columns are settled: its own table, its own
    /// leading preprocessed columns.
    ///
    /// ★ DECODE IS THE SINGLE-TABLE CASE OF A GENERAL SHAPE. A prepared
    /// commitment now names a `(table, preprocessed column)` per stacked column,
    /// because the cross-epoch genesis stack spans one table per dense page and
    /// settles column 1 — INIT — of each. DECODE spans one table and settles its
    /// prefix, so this is [`multilinear_table::leading_columns`] and nothing
    /// about what DECODE means has changed. The epoch byte gate says so by not
    /// moving: the single-table case produces the same weight shares under
    /// `Claimed::PerColumn` as under the `Shared` it replaces.
    pub(crate) fn settled_at(&self, table: usize) -> Vec<multilinear_table::PreparedColumn> {
        multilinear_table::leading_columns(table, self.columns.len())
    }

    /// What the prover opens at `table`'s reduced point.
    ///
    /// `borrowed` and `at` are the caller's because
    /// [`multilinear_table::Prepared`] holds slices and a self-referential
    /// struct cannot hand one out.
    pub(crate) fn opening<'a>(
        &'a self,
        borrowed: &'a [&'a Mle<F>],
        at: &'a [multilinear_table::PreparedColumn],
    ) -> multilinear_table::Prepared<'a, F, H> {
        multilinear_table::Prepared {
            commitment: &self.commitment,
            columns: borrowed,
            at,
        }
    }

    /// ★ ONE COMMITMENT SERVES EVERY EPOCH, and this is why it may.
    ///
    /// `StackedCommitment::commit` reads `log_blowup` — the codeword's rate and
    /// the room it reserves — and `log_folding`, and never `num_queries`. Each
    /// epoch derives its own `ChainConfig` from its own table shapes, and the
    /// query count legitimately differs between them; blowup and fold width do
    /// not. So a commitment built once is valid under every epoch's config
    /// exactly as long as that holds, and it is ASSERTED per epoch rather than
    /// assumed, because the day a blowup becomes shape-dependent this is the
    /// line that says so instead of a proof nobody can verify.
    pub(crate) fn agrees_with(&self, config: &ChainConfig) -> Result<(), Error> {
        if (config.log_blowup, config.log_folding) != (self.log_blowup, self.log_folding) {
            return Err(Error::Prover(format!(
                "the pinned DECODE commitment was built at blowup {} / folding {}, \
                 and this epoch argues at blowup {} / folding {}",
                self.log_blowup, self.log_folding, config.log_blowup, config.log_folding,
            )));
        }
        Ok(())
    }

    /// What the verifier settles the opening against.
    ///
    /// `at` is the caller's, and must be the SAME slice the opening was built
    /// from — it drives both the columns the opening settles and the checks
    /// `check_preprocessed` skips. [`Self::settled_at`] is how both sides get
    /// one.
    pub(crate) fn check<'a>(
        &'a self,
        at: &'a [multilinear_table::PreparedColumn],
    ) -> multilinear_table::PreparedCheck<'a, F> {
        multilinear_table::PreparedCheck {
            roots: &self.roots,
            layout: self.commitment.layout(),
            domain: self.commitment.domain(),
            at,
        }
    }
}

thread_local! {
    /// How many times DECODE's out-of-band commitment has been DERIVED on THIS
    /// thread, so §4's residency claim — one commitment held across the epochs,
    /// not one per epoch — is a number a test can read off the production call.
    ///
    /// ⚠ THREAD-LOCAL ON PURPOSE. A process-wide counter is an assertion about
    /// every test in the binary: `cargo test` runs them in parallel and several
    /// files prove continuations, so a global would read whatever the
    /// neighbours were doing. The derivation happens on the thread that calls
    /// `prove_epochs` / `verify_epochs_bookends`, above the epoch loop, so the
    /// caller's own thread is where the count belongs.
    ///
    /// ⚠ And a thread-local has its own failure mode: if the derivation ever
    /// moved onto a worker thread this would read ZERO, which a `<= 1` bound
    /// would happily accept. Every assertion on it is therefore an EQUALITY —
    /// exactly one per run — so both "rebuilt per epoch" and "counted nowhere"
    /// are failures.
    static DECODE_DERIVATIONS: core::cell::Cell<u64> = const { core::cell::Cell::new(0) };
}

/// Derivations on this thread since the last [`reset_decode_derivations`].
///
/// `cfg(test)` because the counter is an instrument and nothing in production
/// reads it; the BUMP stays unconditional, so what the test counts is the
/// production path and not a test-only copy of it.
#[cfg(test)]
pub(crate) fn decode_derivations() -> u64 {
    DECODE_DERIVATIONS.with(core::cell::Cell::get)
}

/// Zeroes this thread's derivation count.
#[cfg(test)]
pub(crate) fn reset_decode_derivations() {
    DECODE_DERIVATIONS.with(|c| c.set(0));
}

/// [`decode_prepared_for`]'s core: the commitment over columns already in hand.
///
/// The split is what lets the commitment's own properties — that it is a
/// function of the instruction table and of nothing else, and that its shape is
/// the one the ELF implies — be tested without an ELF artifact on disk. A test
/// that silently skips when a build product is missing is a test that passed
/// for the wrong reason.
pub(crate) fn decode_prepared_from_columns<H>(
    elf_digest: [u8; 32],
    columns: Vec<Vec<FieldElement<F>>>,
    config: &ChainConfig,
) -> Result<DecodePrepared<H>, Error>
where
    H: multilinear::whir_hash::WhirHash,
{
    // ★ THE FUNNEL. Both `decode_prepared` and `decode_prepared_for` come
    // through here, so this is the one place a derivation can be counted and
    // the one place it can be missed.
    DECODE_DERIVATIONS.with(|c| c.set(c.get() + 1));
    let columns: Vec<Mle<F>> = columns
        .into_iter()
        .map(|values| Mle::new(values).map_err(|e| Error::Prover(format!("DECODE: {e:?}"))))
        .collect::<Result<_, _>>()?;
    let rows = columns
        .first()
        .ok_or_else(|| Error::Prover("DECODE has no preprocessed columns".to_string()))?
        .len();
    if !rows.is_power_of_two() {
        return Err(Error::Prover(format!(
            "DECODE's preprocessed columns are {rows} rows, which the hypercube cannot hold",
        )));
    }
    let shape = [(columns.len(), rows.trailing_zeros() as usize)];
    let layout =
        multilinear_table::global_layout(&shape).map_err(|e| Error::Prover(format!("{e:?}")))?;
    let commitment = multilinear::stacked_eval::StackedCommitment::<F, H>::commit(
        layout,
        &multilinear::stacking::borrow(&columns),
        None,
        config,
    )
    .map_err(|e| Error::Prover(format!("{e:?}")))?;
    let roots = commitment.roots();
    if roots.is_empty() {
        return Err(Error::Prover(
            "the DECODE preprocessed group commits to nothing".to_string(),
        ));
    }
    Ok(DecodePrepared {
        elf_digest,
        columns,
        roots,
        commitment,
        log_blowup: config.log_blowup,
        log_folding: config.log_folding,
    })
}

/// The `ChainConfig` DECODE's out-of-band commitment is built under.
///
/// Derived from the group's OWN shape, because it has to exist before any epoch
/// does. Only `log_blowup` and `log_folding` matter to a commitment — see
/// [`DecodePrepared::agrees_with`] — and those are constants of
/// [`chain_config`], so this agrees with every epoch's by construction and is
/// asserted to anyway.
pub(crate) fn decode_prepared_config(columns: usize, num_vars: usize) -> ChainConfig {
    chain_config(&[(columns, num_vars)])
}

/// Derives [`DecodePrepared`] from an ELF, at the config its own shape implies.
///
/// The one call BOTH SIDES make, so the prover and the verifier cannot build the
/// commitment from two different ELFs or under two different parameter sets.
pub(crate) fn decode_prepared_for<H>(
    elf: &Elf,
    elf_bytes: &[u8],
) -> Result<DecodePrepared<H>, Error>
where
    H: multilinear::whir_hash::WhirHash,
{
    let columns = crate::tables::decode::preprocessed_columns_from_elf(elf)
        .map_err(|e| Error::Prover(format!("DECODE: {e:?}")))?;
    let rows = columns
        .first()
        .ok_or_else(|| Error::Prover("DECODE has no preprocessed columns".to_string()))?
        .len();
    if !rows.is_power_of_two() {
        return Err(Error::Prover(format!(
            "DECODE's preprocessed columns are {rows} rows, which the hypercube cannot hold",
        )));
    }
    let config = decode_prepared_config(columns.len(), rows.trailing_zeros() as usize);
    decode_prepared_from_columns(statement::elf_digest(elf_bytes), columns, &config)
}

/// Where DECODE sits among an epoch's tables, found by NAME.
///
/// ⚠ Not a constant and not a position. The opening binds the pinned columns to
/// ONE table's reduced point, and a prover able to aim them at a different
/// table's point would be settling them against challenges they were never
/// bound to. So the index is asserted against the AIR that carries the columns,
/// and "exactly one" is part of the assertion: a table set with two DECODEs, or
/// none, is a layout nobody meant to build.
pub(crate) fn decode_table_index(
    airs: &[&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>],
) -> Result<usize, Error> {
    let found: Vec<usize> = airs
        .iter()
        .enumerate()
        .filter(|(_, air)| air.name() == "DECODE")
        .map(|(i, _)| i)
        .collect();
    match found.as_slice() {
        [only] => Ok(*only),
        _ => Err(Error::Prover(format!(
            "an epoch's table set must carry exactly one DECODE, it carries {} (at {found:?})",
            found.len(),
        ))),
    }
}

/// Binds an epoch's statement into the transcript before any challenge.
///
/// The monolithic multilinear statement plus the epoch's position. A
/// continuation epoch never has private-input pages (the bookend replaces
/// PAGE), so that count is not stated — it is zero by construction.
///
/// ⚠ AND NEITHER ARE THE REGISTER VECTORS, deliberately. `register_init` and
/// `reg_fini` are bound by REGISTER's preprocessed COLUMNS, not by anything
/// here: the roots commit those columns before a challenge exists, and
/// `check_preprocessed` ties the verifier's own pair to them at the reduced
/// point. See `register::preprocessed_columns_with_fini`. Absorbing them would
/// add no binding and move every pinned constant.
///
/// ★ The length is accumulated beside the absorbs, never written as a constant:
/// a `FIXED` the caller has to keep in step is the same class of defect as the
/// pad this function exists to compute. See
/// [`statement::absorb_statement_padding`] for why the roots that follow have to
/// start on a field element boundary and why the pad cannot be a literal.
pub(crate) fn absorb_epoch(
    t: &mut impl crypto::fiat_shamir::is_transcript::IsTranscript<E>,
    elf_digest: &[u8; 32],
    public_output: &[u8],
    table_counts: &TableCounts,
    epoch_label: u64,
    table_num_vars: &[u8],
    config: &ChainConfig,
) {
    let mut len = 0usize;

    t.append_bytes(MULTILINEAR_EPOCH_TAG);
    len += MULTILINEAR_EPOCH_TAG.len();
    t.append_bytes(elf_digest);
    len += elf_digest.len();
    t.append_bytes(&epoch_label.to_le_bytes());
    len += size_of_val(&epoch_label);

    t.append_bytes(&(public_output.len() as u64).to_le_bytes());
    len += size_of::<u64>();
    t.append_bytes(public_output);
    len += public_output.len();

    len += statement::absorb_table_counts(t, table_counts);

    t.append_bytes(&(table_num_vars.len() as u64).to_le_bytes());
    len += size_of::<u64>();
    t.append_bytes(table_num_vars);
    len += table_num_vars.len();

    let &ChainConfig {
        log_blowup,
        log_folding,
        num_queries,
        grind,
    } = config;
    for value in [log_blowup as u64, log_folding as u64, num_queries as u64] {
        t.append_bytes(&value.to_le_bytes());
        len += size_of_val(&value);
    }
    let trailer = [grind.folding, grind.ood, grind.query];
    t.append_bytes(&trailer);
    len += trailer.len();

    statement::absorb_statement_padding(
        t,
        "epoch",
        len,
        &[
            ("public_output", public_output.len()),
            ("table_num_vars", table_num_vars.len()),
        ],
    );
}

/// How an epoch's tables are split across commitments: everything together,
/// and the local-to-global bookend on its own.
///
/// The bookend needs a root of its own because the cross-epoch proof commits
/// the same table and the two have to be compared. A table that shares a stack
/// has no root — the stack gives one per stacked *polynomial* — so committing
/// it alone is the only way to say "this is that table".
pub(crate) fn epoch_groups(num_tables: usize) -> Vec<usize> {
    vec![num_tables - 1, 1]
}

/// A table's layout, from the AIR and the shape the verifier states.
fn layout_of<'a>(
    air: &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
    width: usize,
    num_vars: usize,
) -> Result<TableLayout<'a, F, E>, multilinear::Error> {
    TableLayout::<F, E>::new(
        air.constraint_program(),
        air.constraints_meta(),
        air.bus_interactions(),
        width,
        num_vars,
        stark::multilinear_air::Uniforms::default(),
    )
}

/// What the epoch's tables owe the statement: the COMMIT bus's counterparty,
/// counted from the commit index this epoch carried in.
///
/// # ⛔ This is a REPLAY of the roots block, not a second spelling of it
///
/// The counterparty is a function of `z` and `alpha`, which `multi_verify` has
/// not drawn yet when this is called, so the transcript is forked and the block
/// is replayed on the fork. `carried` and `derived` are therefore the SAME two
/// lists that verification will be handed — the derived one included. A replay
/// that absorbed only the carried roots would compute the counterparty at
/// challenges no table was ever checked against, and every epoch with a
/// non-empty `public_output` would fail on `BusImbalance` while every epoch
/// without one passed: [`crate::compute_commit_bus_offset`] returns zero for an
/// empty output without reading either challenge, so those epochs accept any
/// replay at all. That is the failure this signature exists to prevent, and
/// [`multilinear_table::absorb_roots_and_challenge`] is called rather than
/// re-spelled so the two can no longer disagree.
///
/// Only the absorb half is shared. Two challenges are drawn, not three: the fork
/// is discarded, `beta` would never be read, and an unread draw is a real sponge
/// squeeze that `hash_metrics` counts on this clone like any other transcript.
/// The order the two are drawn in is checked against the block itself by
/// [`crate::tests::multilinear_continuation_tests`] rather than asserted here.
pub(crate) fn owed<T: crypto::fiat_shamir::transcript_hash::TranscriptHash>(
    public_output: &[u8],
    register_init: &[u32],
    carried: &[Commitment],
    derived: &[Commitment],
    transcript: &DefaultTranscript<E, T>,
) -> Option<FieldElement<E>> {
    let start_index = *register_init.get(register::X254_INDEX)? as u64;
    let mut probe = transcript.clone();
    multilinear_table::absorb_roots::<E, _>(&mut probe, carried, derived);
    let z: FieldElement<E> = probe.sample_field_element();
    let alpha: FieldElement<E> = probe.sample_field_element();
    crate::compute_commit_bus_offset(public_output, start_index, &z, &alpha)
}

/// Domain tag for the multilinear cross-epoch proof.
pub(crate) const MULTILINEAR_GLOBAL_TAG: &[u8] = b"LAMBDAVM_MULTILINEAR_CONTINUATION_GLOBAL_V1";

/// The one cross-epoch proof: every epoch's bookend and the global-memory
/// tables, in one transcript.
///
/// The bookends come first, one commitment group each, so root `k` is the one
/// epoch `k` carries — that comparison is what says the two proofs are about
/// the same table.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct GlobalProof {
    pub proof: MultiProof<F, E>,
    pub table_num_vars: Vec<u8>,
}

impl GlobalProof {
    /// The roots each epoch's bookend was committed under here, in epoch order.
    ///
    /// The bookends are the first commitment groups, one each, and the roots
    /// are flat — one per stacked polynomial — so a group's are a window.
    /// `polys` is how many polynomials each one stacked into.
    pub fn l2g_roots(&self, polys: &[usize]) -> Option<Vec<&[stark::config::Commitment]>> {
        let mut start = 0usize;
        let mut groups = Vec::with_capacity(polys.len());
        for &num_polys in polys {
            if num_polys == 0 {
                return None;
            }
            let end = start.checked_add(num_polys)?;
            groups.push(self.proof.roots.get(start..end)?);
            start = end;
        }
        Some(groups)
    }
}

/// Binds the cross-epoch statement: what the run was, not what any epoch was.
///
/// ⚠ This statement is padded for the same reason the epoch statement is, and
/// it needs it for the same reason: `table_num_vars` is one byte per table
/// (every epoch's bookend plus the global-memory tables), so it is a
/// variable-length field sitting between the fixed prefix and the roots.
/// `page_bases` is eight bytes an entry and does not move the alignment.
pub(crate) fn absorb_global(
    t: &mut impl crypto::fiat_shamir::is_transcript::IsTranscript<E>,
    elf_digest: &[u8; 32],
    num_epochs: usize,
    num_private_input_pages: usize,
    page_bases: &[u64],
    table_num_vars: &[u8],
    config: &ChainConfig,
) {
    let mut len = 0usize;

    t.append_bytes(MULTILINEAR_GLOBAL_TAG);
    len += MULTILINEAR_GLOBAL_TAG.len();
    t.append_bytes(elf_digest);
    len += elf_digest.len();
    t.append_bytes(&(num_epochs as u64).to_le_bytes());
    len += size_of::<u64>();
    t.append_bytes(&(num_private_input_pages as u64).to_le_bytes());
    len += size_of::<u64>();
    t.append_bytes(&(page_bases.len() as u64).to_le_bytes());
    len += size_of::<u64>();
    for base in page_bases {
        t.append_bytes(&base.to_le_bytes());
        len += size_of_val(base);
    }
    t.append_bytes(&(table_num_vars.len() as u64).to_le_bytes());
    len += size_of::<u64>();
    t.append_bytes(table_num_vars);
    len += table_num_vars.len();
    let &ChainConfig {
        log_blowup,
        log_folding,
        num_queries,
        grind,
    } = config;
    for value in [log_blowup as u64, log_folding as u64, num_queries as u64] {
        t.append_bytes(&value.to_le_bytes());
        len += size_of_val(&value);
    }
    let trailer = [grind.folding, grind.ood, grind.query];
    t.append_bytes(&trailer);
    len += trailer.len();

    statement::absorb_statement_padding(
        t,
        "global",
        len,
        &[
            ("page_bases", page_bases.len()),
            ("table_num_vars", table_num_vars.len()),
        ],
    );
}

/// How the cross-epoch proof's tables are split: every bookend alone — so its
/// root can be compared against the epoch that committed it — and the
/// global-memory tables together.
pub(crate) fn global_groups(num_epochs: usize, num_pages: usize) -> Vec<usize> {
    let mut sizes = vec![1usize; num_epochs];
    // A run that touched no memory has no global-memory tables, and a group of
    // none is a commitment to nothing.
    if num_pages > 0 {
        sizes.push(num_pages);
    }
    sizes
}

/// Proves the cross-epoch memory chain: each epoch's bookend, and one
/// global-memory table per page the run touched.
///
/// ★ GENERIC OVER THE HASH, and the dispatch belongs to its CALLER — the
/// arrangement [`prove_epoch`] and [`verify_global_bookends`] already have, and
/// for the same two reasons.
///
/// **A caller cannot be asked a question this function answers for itself.**
/// `whir_hash_knob::selected()` is a cached process setting: it says what THIS
/// PROCESS proves under, never what a bundle in front of it was proven under.
/// While the knob was read here, a test could not prove a bundle under a named
/// hash without re-implementing the prover, which is why the cross-epoch
/// execution arms had to be `#[ignore]`d with a posture guard instead.
///
/// **And an out-of-band commitment's type names the hash.** The genesis stack
/// this hands to [`multilinear_table::multi_prove`] is a
/// `StackedCommitment<F, H>`, so it cannot be built outside a dispatch and
/// passed in — the same argument that made `prove_epoch` generic so DECODE's
/// commitment could outlive one call.
pub fn prove_global<H>(
    boundaries: &[std::sync::Arc<Vec<CellBoundary>>],
    elf_bytes: &[u8],
    init_page_data: &std::collections::HashMap<u64, Vec<u8>>,
    page_bases: &[u64],
    num_private_input_pages: usize,
    opts: &ProofOptions,
) -> Result<GlobalProof, Error>
where
    H: multilinear::whir_hash::WhirHash,
{
    // Each cell's final state; the boundaries are in epoch order, so the last
    // fini wins.
    let mut final_state: crate::tables::global_memory::FiniStateMap =
        std::collections::HashMap::new();
    for epoch in boundaries {
        for b in epoch.iter() {
            final_state.insert(
                b.address,
                crate::tables::global_memory::FiniState {
                    value: (b.fini.value & 0xFF) as u8,
                    epoch: b.fini.epoch,
                },
            );
        }
    }

    let gm_configs = crate::continuation::global_memory_configs_from_init_page_data(
        page_bases,
        init_page_data,
        num_private_input_pages,
        true,
    );

    let l2g_airs: Vec<_> = (0..boundaries.len())
        .map(|i| crate::continuation::l2g_global_air(opts, local_to_global::epoch_label(i as u64)))
        .collect();
    let gm_airs: Vec<_> = gm_configs
        .iter()
        .map(|config| crate::continuation::global_memory_air(opts, config, None))
        .collect();
    let mut l2g_traces: Vec<_> = boundaries
        .iter()
        .map(|epoch| local_to_global::generate_local_to_global_trace(epoch.as_slice()))
        .collect();
    let mut gm_traces: Vec<_> = gm_configs
        .iter()
        .map(|config| crate::tables::global_memory::generate_global_trace(config, &final_state))
        .collect();

    let mut pairs: Vec<crate::AirTracePair<'_>> = Vec::new();
    for (air, trace) in l2g_airs.iter().zip(l2g_traces.iter_mut()) {
        pairs.push((air, trace, &()));
    }
    for (air, trace) in gm_airs.iter().zip(gm_traces.iter_mut()) {
        pairs.push((air, trace, &()));
    }

    let shapes: Vec<(usize, usize)> = pairs
        .iter()
        .map(|(_, trace, _)| {
            (
                trace.main_table.width,
                trace.main_table.height.trailing_zeros() as usize,
            )
        })
        .collect();
    let table_num_vars: Vec<u8> = shapes.iter().map(|&(_, n)| n as u8).collect();
    let config = chain_config(&shapes);

    let mut committed = Vec::with_capacity(pairs.len());
    for ((air, trace, _), &(width, num_vars)) in pairs.iter_mut().zip(&shapes) {
        let layout = layout_of(*air, width, num_vars)
            .map_err(|e| Error::Prover(format!("{}: {e:?}", air.name())))?;
        let mut columns = trace.columns_main();
        for (col, expected) in air.precomputed_columns().iter().enumerate() {
            if columns.get(col) != Some(expected) {
                return Err(Error::Prover(format!(
                    "{}: preprocessed column {col} is not what the run implies",
                    air.name(),
                )));
            }
        }
        committed.push(
            CommittedTable::from_layout(layout, |col| core::mem::take(&mut columns[col as usize]))
                .map_err(|e| Error::Prover(format!("{}: {e:?}", air.name())))?,
        );
    }
    let sizes = global_groups(boundaries.len(), gm_configs.len());
    // ★ `H` is the CALLER's. The transcript's hash is part of the configuration
    // either way; what changed is who chooses it, and the bound on
    // `multi_prove` still rejects any other spelling.
    let mut transcript =
        DefaultTranscript::<E, <H as multilinear::whir_hash::WhirHash>::Transcript>::new(&[]);
    absorb_global(
        &mut transcript,
        &statement::elf_digest(elf_bytes),
        boundaries.len(),
        num_private_input_pages,
        page_bases,
        &table_num_vars,
        &config,
    );
    let committed = CommittedTables::<_, _, H>::commit_grouped(committed, &sizes, &config)
        .map_err(|e| Error::Prover(format!("{e:?}")))?;
    let proof = multilinear_table::multi_prove(&committed, &config, &mut transcript, None)
        .map_err(|e| Error::Prover(format!("{e:?}")))?;

    Ok(GlobalProof {
        proof,
        table_num_vars,
    })
}

/// Verifies the cross-epoch proof from the ELF and the run's public shape.
///
/// `page_bases` and `num_epochs` are the bundle's, and both are bound into the
/// transcript and pinned by the bus: a wrong set leaves the GlobalMemory bus
/// unbalanced or the AIR count mismatched.
///
/// ★ THE DISPATCH IS HERE, and the bookend form below takes `H`. That is the
/// arrangement the epoch half already has ([`verify_epoch`] against
/// [`verify_epoch_bookend`]), and moving it out was not a tidy-up: a caller
/// that needs to ask "was this bundle proven under `H`" cannot be handed a
/// function that reads the process knob for itself. Nothing about what is
/// checked changes — this entry point picks the same `H` from the same knob the
/// macro read one level down — which is what the byte gate and the transcript
/// pin pair say when they do not move.
pub fn verify_global(
    elf: &Elf,
    elf_bytes: &[u8],
    global: &GlobalProof,
    num_epochs: usize,
    page_bases: &[u64],
    num_private_input_pages: usize,
    opts: &ProofOptions,
) -> Result<bool, Error> {
    crate::with_whir_hash!(|H| {
        Ok(verify_global_bookends::<H>(
            elf,
            elf_bytes,
            global,
            num_epochs,
            page_bases,
            num_private_input_pages,
            opts,
        )?
        .is_some())
    })
}

/// The cross-epoch proof's AIR set, OWNED — because everything downstream
/// borrows from it.
///
/// ★ TWO FAMILIES, KEPT APART, and that is the whole reason this is a struct
/// with two fields rather than one flat `Vec`. Both constructors return the
/// same concrete type, so one vector would compile — and the bookend/page SPLIT
/// would then be unrecoverable from the set. That split is what
/// [`global_groups`] takes, and unlike an epoch's it does NOT follow from the
/// table count: fifteen bookends with thirty-five pages and fourteen with
/// thirty-six are fifty tables either way. Here it is the struct's shape, so
/// [`refs`](Self::refs) and [`groups`](Self::groups) are written from the same
/// two vectors and cannot describe different layouts.
///
/// The AIRs are boxed only so their concrete type does not have to be spelled
/// here — [`WhirEpochAirs`] boxes its bookend for the same reason, and a `&dyn
/// AIR` is a `&dyn AIR` either way.
pub struct WhirGlobalAirs {
    /// One local-to-global bookend per epoch, in epoch-label order.
    bookends: Vec<Box<dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>>>,
    /// One GLOBAL_MEMORY table per touched page, in the canonical page-base
    /// order [`crate::continuation::global_memory_configs`] hands back.
    pages: Vec<Box<dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>>>,
}

impl WhirGlobalAirs {
    /// The AIRs in PROOF ORDER: every bookend, then every page.
    ///
    /// ⚠ The order IS the proof's layout — `multi_verify` matches AIRs to
    /// sub-proofs positionally — and it is written once, here, because it used
    /// to be written at the call site.
    pub fn refs(&self) -> Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> {
        let mut refs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
            Vec::with_capacity(self.bookends.len() + self.pages.len());
        for air in self.bookends.iter().chain(&self.pages) {
            refs.push(air.as_ref());
        }
        refs
    }

    /// How that set is committed: every bookend alone, then the pages together.
    ///
    /// [`global_groups`] over this set's OWN two families, so a caller cannot
    /// restate the split from a table count it arrived at some other way.
    pub fn groups(&self) -> Vec<usize> {
        global_groups(self.bookends.len(), self.pages.len())
    }
}

/// The AIR set the cross-epoch proof's tables are argued against.
///
/// ★ ONE DERIVATION, AND EVERY CONSUMER GOES THROUGH IT.
/// [`verify_global_bookends`] builds its statements from this, and so must any
/// later emitter of the cross-epoch program, so "the AIRs a program is emitted
/// against are the AIRs the verifier accepted" is true by construction. Two
/// call sites that agree today is exactly the shape that let REGISTER's
/// preprocessed columns and its root describe different tables.
/// ⚠ REACHABILITY IS NOT AN ARGUMENT FOR A SECOND COPY: `l2g_global_air`,
/// `global_memory_air` and `global_memory_configs` are all `pub(crate)`, so one
/// can be written anywhere in the crate — and it would be a second derivation
/// of the layout `multi_verify` matches positionally.
///
/// ⚠ THE PAGE CONFIGS ARE REBUILT FROM THE ELF, never taken from the bundle:
/// that is the genesis binding, and it is why this takes an [`Elf`] rather than
/// a config list. [`prove_global`] builds its own set from the run's init page
/// data with private genesis included — a different source on purpose, which is
/// why the prover is not a caller of this.
///
/// ⚠ `global_memory_air`'s preprocessed commitment is `None` here and there is
/// deliberately no parameter for it, which is the opposite of
/// [`epoch_airs_for`]'s `decode_commitment`. That argument supplies a page's
/// UNIVARIATE preprocessed commitment, and this path never reads one:
/// `AIR::precomputed_columns` runs the columns closure, while the commitment
/// sits behind `LazyCommitment::get`, whose callers are the univariate prover
/// and verifier and `crate::lfm`. A parameter would therefore be a knob with no
/// observable effect and no cost either way — the deferred per-page commitment
/// is never forced on this path. A cross-epoch program's own genesis opening is
/// a MULTILINEAR commitment over the same columns, a different object that this
/// argument cannot carry.
pub(crate) fn global_airs_for(
    elf: &Elf,
    opts: &ProofOptions,
    num_epochs: usize,
    page_bases: &[u64],
    num_private_input_pages: usize,
) -> WhirGlobalAirs {
    let bookends = (0..num_epochs)
        .map(|i| {
            Box::new(crate::continuation::l2g_global_air(
                opts,
                local_to_global::epoch_label(i as u64),
            )) as Box<dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>>
        })
        .collect();
    // Rebuilt from the ELF, never from the bundle: this is the genesis binding.
    let gm_configs =
        crate::continuation::global_memory_configs(page_bases, elf, num_private_input_pages);
    let pages = gm_configs
        .iter()
        .map(|config| {
            Box::new(crate::continuation::global_memory_air(opts, config, None))
                as Box<dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>>
        })
        .collect();
    WhirGlobalAirs { bookends, pages }
}

/// [`verify_global`], handing back the roots each epoch's bookend was
/// committed under — which is what the binding compares. `None` is a proof
/// that does not verify.
///
/// ★ `pub(crate)` AND GENERIC, for the two reasons the epoch half is:
///
/// **The hash is the caller's.** `whir_hash_knob::selected()` is a cached
/// process setting — it says what THIS PROCESS proves under, never what the
/// bundle in front of it was proven under. The agreement is the verification
/// itself: the transcript's sponge is part of the configuration, so a bundle
/// proven under another hash diverges from the first squeeze. A function that
/// read the knob for itself could not be asked the question, which is why
/// `an_epoch_proven_under_one_hash_is_refused_under_the_other` could only ever
/// exist on the epoch side. See
/// `multilinear_continuation_tests::a_cross_epoch_proof_proven_under_one_hash_is_refused_under_the_other`.
///
/// **The roots are the binding's.** [`verify_global`] answers `bool`, and the
/// cross-epoch wrap's published set is these roots — so a driver that could
/// only reach the `bool` had to re-derive them from `stacks`, which is a second
/// spelling of the two lines below.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_global_bookends<H>(
    elf: &Elf,
    elf_bytes: &[u8],
    global: &GlobalProof,
    num_epochs: usize,
    page_bases: &[u64],
    num_private_input_pages: usize,
    opts: &ProofOptions,
) -> Result<Option<Vec<Vec<Commitment>>>, Error>
where
    H: multilinear::whir_hash::WhirHash,
{
    // ★ THE ONE DERIVATION. An emitter of the cross-epoch program builds its
    // set through this same function.
    let air_set = global_airs_for(elf, opts, num_epochs, page_bases, num_private_input_pages);
    let air_refs = air_set.refs();
    if air_refs.len() != global.proof.tables.len() || global.table_num_vars.len() != air_refs.len()
    {
        return Err(Error::InvalidTableCounts(format!(
            "the cross-epoch layout has {} tables, the proof carries {} and {} heights",
            air_refs.len(),
            global.proof.tables.len(),
            global.table_num_vars.len(),
        )));
    }

    let shapes: Vec<(usize, usize)> = air_refs
        .iter()
        .zip(&global.table_num_vars)
        .map(|(air, &num_vars)| (air.trace_layout().0, num_vars as usize))
        .collect();
    let config = chain_config(&shapes);

    let layouts: Vec<TableLayout<'_, F, E>> = air_refs
        .iter()
        .zip(&shapes)
        .map(|(air, &(width, num_vars))| {
            layout_of(*air, width, num_vars).map_err(|e| Error::Prover(format!("{e:?}")))
        })
        .collect::<Result<_, _>>()?;
    let preprocessed: Vec<Vec<Mle<F>>> = air_refs
        .iter()
        .map(|air| {
            air.precomputed_columns()
                .into_iter()
                .map(|values| Mle::new(values).map_err(|e| Error::Prover(format!("{e:?}"))))
                .collect::<Result<_, _>>()
        })
        .collect::<Result<_, _>>()?;
    let statements: Vec<TableStatement<'_, F, E>> = layouts
        .iter()
        .zip(&preprocessed)
        .map(|(layout, cols)| layout.statement_with_preprocessed(cols))
        .collect();

    let sizes = air_set.groups();
    let (stacks, domains) = crate::multilinear_prove::stacks(&shapes, &sizes, &config)?;
    // Each bookend is a group of its own, so its roots are the group's — as
    // many as the stack split it into.
    let polys: Vec<usize> = stacks[..num_epochs].iter().map(|l| l.num_polys()).collect();

    // The cross-epoch bus has no counterparty in the statement: it must vanish.
    let verdict = {
        // ★ `H` is the CALLER's now. The transcript's hash is part of the
        // configuration either way; what changed is who chooses it, and the
        // bound on `multi_prove`/`multi_verify` still rejects any other
        // spelling.
        let mut transcript =
            DefaultTranscript::<E, <H as multilinear::whir_hash::WhirHash>::Transcript>::new(&[]);
        absorb_global(
            &mut transcript,
            &statement::elf_digest(elf_bytes),
            num_epochs,
            num_private_input_pages,
            page_bases,
            &global.table_num_vars,
            &config,
        );
        multilinear_table::multi_verify::<_, _, _, H>(
            &global.proof,
            &statements,
            &stacks,
            &domains,
            &sizes,
            &FieldElement::<E>::zero(),
            &config,
            &mut transcript,
            None,
        )
    };
    if verdict.is_err() {
        return Ok(None);
    }
    Ok(global
        .l2g_roots(&polys)
        .map(|groups| groups.into_iter().map(<[_]>::to_vec).collect()))
}

/// Proves one epoch: its tables plus the local-to-global bookend, against one
/// commitment.
///
/// ★ GENERIC OVER THE HASH, and the dispatch belongs to its CALLER. That is what
/// lets `prepared` — DECODE's out-of-band commitment, whose type names the hash
/// — be built once and HELD across every epoch of a run, instead of rebuilt
/// fifteen times for a value that depends on nothing an epoch chose. A dispatch
/// inside this function would make that commitment unable to outlive one call.
#[allow(clippy::too_many_arguments)]
pub fn prove_epoch<H>(
    elf: &Elf,
    elf_bytes: &[u8],
    register_init: &[u32],
    label: u64,
    mut traces: Traces,
    is_final: bool,
    boundary: &[CellBoundary],
    opts: &ProofOptions,
    decode_commitment: Option<Commitment>,
    prepared: &DecodePrepared<H>,
) -> Result<EpochProof, Error>
where
    H: multilinear::whir_hash::WhirHash,
{
    // The bookend's range checks are lookups into BITWISE, so its
    // multiplicities have to carry them.
    crate::tables::bitwise::update_multiplicities(
        &mut traces.bitwise,
        &local_to_global::collect_bitwise_from_l2g(boundary),
    );
    if !traces.page_configs.is_empty() {
        return Err(Error::ContinuationInvariant(
            "continuation epoch must have no PAGE configs (L2G bookend replaces PAGE)".to_string(),
        ));
    }

    let reg_fini = register::fini_from_trace(&traces.register);
    let table_counts = traces.table_counts();
    let public_output = traces.public_output_bytes.clone();

    let airs = crate::continuation::build_epoch_airs(
        elf,
        opts,
        &[],
        &table_counts,
        register_init,
        &reg_fini,
        is_final,
        decode_commitment,
    );
    let l2g_air = crate::continuation::l2g_memory_air(opts, label);
    let mut l2g_trace = local_to_global::generate_local_to_global_trace(boundary);

    let mut pairs = airs.air_trace_pairs(&mut traces);
    pairs.push((&l2g_air, &mut l2g_trace, &()));

    // Taken here, while the AIRs are still in hand, and by NAME — the opening
    // binds the pinned columns to ONE table's reduced point.
    let decode_at = decode_table_index(&pairs.iter().map(|(air, _, _)| *air).collect::<Vec<_>>())?;

    let shapes: Vec<(usize, usize)> = pairs
        .iter()
        .map(|(_, trace, _)| {
            (
                trace.main_table.width,
                trace.main_table.height.trailing_zeros() as usize,
            )
        })
        .collect();
    let table_num_vars: Vec<u8> = shapes.iter().map(|&(_, n)| n as u8).collect();
    let config = chain_config(&shapes);

    let mut committed = Vec::with_capacity(pairs.len());
    for ((air, trace, _), &(width, num_vars)) in pairs.iter_mut().zip(&shapes) {
        let layout = layout_of(*air, width, num_vars)
            .map_err(|e| Error::Prover(format!("{}: {e:?}", air.name())))?;
        let mut columns = trace.columns_main();
        // The verifier rebuilds these and demands the proof open to them, so a
        // trace that disagrees produces a proof nobody can verify.
        for (col, expected) in air.precomputed_columns().iter().enumerate() {
            if columns.get(col) != Some(expected) {
                return Err(Error::Prover(format!(
                    "{}: preprocessed column {col} is not what the program implies",
                    air.name(),
                )));
            }
        }
        committed.push(
            CommittedTable::from_layout(layout, |col| core::mem::take(&mut columns[col as usize]))
                .map_err(|e| Error::Prover(format!("{}: {e:?}", air.name())))?,
        );
    }
    let sizes = epoch_groups(committed.len());
    prepared.agrees_with(&config)?;

    // ★ The transcript's hash is part of the configuration, and `H` is the
    // caller's dispatch. The bound on `multi_prove`/`multi_verify` rejects any
    // other spelling.
    let mut transcript =
        DefaultTranscript::<E, <H as multilinear::whir_hash::WhirHash>::Transcript>::new(&[]);
    absorb_epoch(
        &mut transcript,
        &statement::elf_digest(elf_bytes),
        &public_output,
        &table_counts,
        label,
        &table_num_vars,
        &config,
    );
    let committed = CommittedTables::<_, _, H>::commit_grouped(committed, &sizes, &config)
        .map_err(|e| Error::Prover(format!("{e:?}")))?;
    let borrowed = multilinear::stacking::borrow(&prepared.columns);
    let decode_columns = prepared.settled_at(decode_at);
    let proof = multilinear_table::multi_prove(
        &committed,
        &config,
        &mut transcript,
        Some(prepared.opening(&borrowed, &decode_columns)),
    )
    .map_err(|e| Error::Prover(format!("{e:?}")))?;

    Ok(EpochProof {
        proof,
        table_num_vars,
        table_counts,
        public_output,
        reg_fini,
    })
}

/// A self-contained multilinear continuation proof.
///
/// Mirrors [`crate::continuation::ContinuationProof`]: the per-epoch proofs in
/// execution order, the one cross-epoch proof, and the two public values the
/// verifier rebuilds the cross-epoch tables from. **No cell values travel** —
/// the boundaries stay with the prover, because a boundary's init value is a
/// byte of the private input for a private read.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct ContinuationProof {
    pub epochs: Vec<EpochProof>,
    pub global: GlobalProof,
    pub num_private_input_pages: usize,
    /// Sorted, deduped page bases the run touched: page bases ONLY, so no
    /// private byte is in here. Prover-supplied but bus-enforced — a wrong set
    /// leaves the cross-epoch bus unbalanced or the table count mismatched, and
    /// it is bound into the cross-epoch statement.
    pub touched_page_bases: Vec<u64>,
}

impl ContinuationProof {
    pub fn num_epochs(&self) -> usize {
        self.epochs.len()
    }

    /// The run's committed output: every epoch's slice, in order.
    pub fn public_output(&self) -> Vec<u8> {
        self.epochs
            .iter()
            .flat_map(|e| e.public_output.iter().copied())
            .collect()
    }
}

/// Proves a whole run: every epoch, then the one cross-epoch proof that chains
/// their memory.
pub fn prove_continuation(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    epoch_size_log2: u32,
    opts: &ProofOptions,
) -> Result<ContinuationProof, Error> {
    let elf = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let decode_commitment = crate::tables::decode::commitment_from_elf(&elf, opts)
        .map_err(|e| Error::Recursion(format!("DECODE commitment from ELF: {e}")))?;
    let artifacts = crate::tables::trace_builder::DecodeArtifacts::from_elf(&elf)?;

    let mut epochs = Vec::new();
    // ★ THE DISPATCH IS HERE, ABOVE THE EPOCH LOOP, so DECODE's out-of-band
    // commitment — whose type names the hash — is built ONCE and held across
    // every epoch. Inside `prove_epoch` it could not outlive one call.
    let boundaries = crate::with_whir_hash!(|H| {
        let prepared = decode_prepared_for::<H>(&elf, elf_bytes)?;
        crate::continuation::for_each_epoch_overlapped(
            &elf,
            private_inputs,
            epoch_size_log2,
            &artifacts,
            |p| {
                epochs.push(prove_epoch::<H>(
                    &elf,
                    elf_bytes,
                    &p.register_init,
                    p.label,
                    p.traces,
                    p.is_final,
                    &p.boundary,
                    opts,
                    Some(decode_commitment),
                    &prepared,
                )?);
                Ok(())
            },
        )?
    });

    // The genesis image, which is the one the run started from — rebuilt here
    // rather than carried, because `for_each_epoch` advances its copy.
    let init_page_data = crate::tables::trace_builder::build_init_page_data(
        &crate::tables::trace_builder::build_initial_image_paged(&elf, private_inputs),
    );
    let num_private_input_pages = crate::tables::page::private_input_page_count(private_inputs);
    // One source of truth: the same list drives the committed tables and
    // travels in the bundle, so the two cannot diverge.
    let touched_page_bases = crate::continuation::touched_page_bases(&boundaries);
    // ★ THE DISPATCH IS HERE, as it is above the epoch loop — and the two are
    // now the same knob read in one shape rather than two places that happened
    // to agree. A hash a caller could choose for the epochs and not for the
    // cross-epoch proof would be a bundle whose two halves were argued under
    // different sponges.
    let global = crate::with_whir_hash!(|H| {
        prove_global::<H>(
            &boundaries,
            elf_bytes,
            &init_page_data,
            &touched_page_bases,
            num_private_input_pages,
            opts,
        )
    })?;

    Ok(ContinuationProof {
        epochs,
        global,
        num_private_input_pages,
        touched_page_bases,
    })
}

/// Verifies a whole run from the bundle and the ELF alone.
///
/// The verifier enumerates the epochs itself — `epoch_label` and `is_final` are
/// positions, not claims — derives each one's starting registers from the ELF
/// or the previous proof, closes the cross-epoch bus with genesis rebuilt from
/// the ELF, and **ties each epoch's bookend to the cross-epoch proof by its
/// root**. Without that last step the two halves are about unrelated tables.
pub fn verify_continuation(
    elf_bytes: &[u8],
    bundle: &ContinuationProof,
    opts: &ProofOptions,
) -> Result<bool, Error> {
    let Some(proved) = verify_epochs_bookends(elf_bytes, &bundle.epochs, opts)? else {
        return Ok(false);
    };
    let elf = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    // The dispatch, as above `verify_epochs_bookends`' own loop: this entry
    // point verifies under the process's hash, and the bookend form is told
    // which so a caller that needs to say can.
    let chained = crate::with_whir_hash!(|H| {
        verify_global_bookends::<H>(
            &elf,
            elf_bytes,
            &bundle.global,
            bundle.epochs.len(),
            &bundle.touched_page_bases,
            bundle.num_private_input_pages,
            opts,
        )
    })?;
    let Some(chained) = chained else {
        return Ok(false);
    };

    // The binding: epoch `k`'s bookend and the one the cross-epoch proof
    // chained are the same table, or neither half says anything about the
    // other. Comparing the groups whole is also what catches a bookend the two
    // sides stacked differently.
    Ok(proved == chained)
}

/// Proves every epoch of a run, in order, chaining the register file.
///
/// **Not a continuation proof yet.** Without the cross-epoch global-memory
/// proof nothing ties one epoch's *memory* to the next: what chains here is the
/// register file, which each epoch's REGISTER preprocessing binds at both ends.
/// The missing half, and why its binding needs a mechanism the univariate path
/// does not, is in this module's header.
pub fn prove_epochs(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    epoch_size_log2: u32,
    opts: &ProofOptions,
) -> Result<Vec<EpochProof>, Error> {
    let elf = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    // A pure function of (ELF, opts), so it is computed once rather than once
    // per epoch inside the AIR build.
    let decode_commitment = crate::tables::decode::commitment_from_elf(&elf, opts)
        .map_err(|e| Error::Recursion(format!("DECODE commitment from ELF: {e}")))?;
    let artifacts = crate::tables::trace_builder::DecodeArtifacts::from_elf(&elf)?;

    let mut proofs = Vec::new();
    // The dispatch above the loop, for the reason in `prove_continuation`.
    crate::with_whir_hash!(|H| {
        let prepared = decode_prepared_for::<H>(&elf, elf_bytes)?;
        crate::continuation::for_each_epoch_overlapped(
            &elf,
            private_inputs,
            epoch_size_log2,
            &artifacts,
            |p| {
                proofs.push(prove_epoch::<H>(
                    &elf,
                    elf_bytes,
                    &p.register_init,
                    p.label,
                    p.traces,
                    p.is_final,
                    &p.boundary,
                    opts,
                    Some(decode_commitment),
                    &prepared,
                )?);
                Ok(())
            },
        )?
    });
    Ok(proofs)
}

/// Verifies a run's epochs from the ELF alone, deriving each one's starting
/// registers from the last one's proof.
///
/// The verifier owns every value an epoch is checked against: the first
/// epoch's registers come from the ELF's entry point, the rest from the
/// previous epoch's proved `reg_fini`, and `is_final` is the position, not a
/// claim. Same caveat as [`prove_epochs`]: this checks the epochs, not that
/// their memory chains.
pub fn verify_epochs(
    elf_bytes: &[u8],
    epochs: &[EpochProof],
    opts: &ProofOptions,
) -> Result<bool, Error> {
    Ok(verify_epochs_bookends(elf_bytes, epochs, opts)?.is_some())
}

/// [`verify_epochs`], handing back each epoch's bookend roots in order — which
/// is what the binding compares. `None` is a run that does not verify.
fn verify_epochs_bookends(
    elf_bytes: &[u8],
    epochs: &[EpochProof],
    opts: &ProofOptions,
) -> Result<Option<Vec<Vec<Commitment>>>, Error> {
    if epochs.is_empty() {
        return Ok(None);
    }
    let elf = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    // The dispatch above the loop, so the verifier derives DECODE's commitment
    // ONCE for the whole run rather than once per epoch — the same reason the
    // prover holds it, and the same saving.
    crate::with_whir_hash!(|H| {
        let prepared = decode_prepared_for::<H>(&elf, elf_bytes)?;
        let mut carried = register::register_init_from_entry_point(elf.entry_point);
        let mut bookends = Vec::with_capacity(epochs.len());
        for (index, epoch) in epochs.iter().enumerate() {
            let label = local_to_global::epoch_label(index as u64);
            let is_final = index + 1 == epochs.len();
            let Some(roots) = verify_epoch_bookend::<H>(
                &elf, elf_bytes, epoch, &carried, is_final, label, opts, &prepared,
            )?
            else {
                return Ok(None);
            };
            bookends.push(roots);
            carried.clone_from(&epoch.reg_fini);
        }
        Ok(Some(bookends))
    })
}

/// Verifies one epoch from the bundle and the ELF alone.
///
/// `register_init` is the verifier's, not the bundle's: the ELF's for epoch 0,
/// the previous epoch's `reg_fini` after that. That is the cross-epoch register
/// binding, and the commit index rides in it.
#[allow(clippy::too_many_arguments)]
pub fn verify_epoch(
    elf: &Elf,
    elf_bytes: &[u8],
    epoch: &EpochProof,
    register_init: &[u32],
    is_final: bool,
    label: u64,
    opts: &ProofOptions,
) -> Result<bool, Error> {
    crate::with_whir_hash!(|H| {
        let prepared = decode_prepared_for::<H>(elf, elf_bytes)?;
        Ok(verify_epoch_bookend::<H>(
            elf,
            elf_bytes,
            epoch,
            register_init,
            is_final,
            label,
            opts,
            &prepared,
        )?
        .is_some())
    })
}

/// One epoch's AIR set, OWNED — because everything downstream borrows from it.
///
/// ★ THE SET IS TWO OWNED THINGS, NOT ONE, and that is why this struct exists
/// rather than a function returning `Vec<&dyn AIR>`. [`VmAirs::air_refs`] hands
/// out borrows of a `VmAirs`, and the local-to-global AIR is a SEPARATE value
/// that [`crate::continuation::l2g_memory_air`] returns by value; both have to
/// outlive the refs, the layouts built from them and the `TableStatement`s
/// built from those. A caller holds one of these and borrows from it.
///
/// The l2g AIR is boxed only so its concrete type does not have to be spelled
/// here; `VmAirs` boxes its own the same way, and a `&dyn AIR` is a `&dyn AIR`
/// either way.
pub(crate) struct WhirEpochAirs {
    airs: crate::VmAirs,
    l2g: Box<dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>>,
}

impl WhirEpochAirs {
    /// The AIRs in PROOF ORDER, local-to-global last.
    ///
    /// ⚠ The order IS the proof's layout. It is written once, here, because it
    /// used to be written at both call sites and getting it wrong at one of
    /// them would be a silent layout mismatch rather than a compile error.
    pub(crate) fn refs(&self) -> Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> {
        let mut refs = self.airs.air_refs();
        refs.push(self.l2g.as_ref());
        refs
    }
}

/// The AIR set an epoch's tables are argued against.
///
/// ★ ONE DERIVATION, TWO CALLERS. [`verify_epoch_bookend`] builds its
/// statements from this, and the level-0 driver
/// (`crate::lfm::whir_real_epoch`) derives the harvested epoch's shapes from
/// it, so the AIRs a wrap program is emitted against cannot drift from the AIRs
/// the verifier accepted. Two call sites that agree today is exactly the shape
/// that let REGISTER's preprocessed columns and its root describe different
/// tables.
///
/// ⚠ `decode_commitment` IS THE ONE THING THE TWO CALLERS DISAGREE ABOUT, so it
/// is a parameter rather than a constant. The verifier passes `None`; the
/// driver passes the root it was handed, because that is what its own
/// `decode_commitment` argument exists to carry. The difference reaches only
/// DECODE's preprocessed commitment, which the multilinear path never compares
/// — `lfm::whir_epoch_tests::the_supplied_decode_commitment_is_the_one_carried`
/// passes a deliberately bogus root and the epoch still verifies. That is a
/// real equivalence and not an obvious one, which is why the caller states
/// which it wants instead of relying on it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn epoch_airs_for(
    elf: &Elf,
    opts: &ProofOptions,
    epoch: &EpochProof,
    register_init: &[u32],
    is_final: bool,
    label: u64,
    decode_commitment: Option<Commitment>,
) -> WhirEpochAirs {
    WhirEpochAirs {
        airs: crate::continuation::build_epoch_airs(
            elf,
            opts,
            &[],
            &epoch.table_counts,
            register_init,
            &epoch.reg_fini,
            is_final,
            decode_commitment,
        ),
        l2g: Box::new(crate::continuation::l2g_memory_air(opts, label)),
    }
}

/// [`verify_epoch`], handing back the roots the epoch's bookend was committed
/// under — which is what the binding compares. `None` is a proof that does not
/// verify.
///
/// ★ `pub(crate)` because this is the ONLY verify entry point that takes its
/// hash as a parameter. [`verify_epoch`] chooses `H` from the cached process
/// knob, so from outside this module one process can verify under exactly one
/// hash — and the level-0 driver's refusal is precisely the claim that a bundle
/// proven under keccak fails under RPX, which cannot be tested through a
/// function that will not be told which to use. See
/// `multilinear_continuation_tests::an_epoch_proven_under_one_hash_is_refused_under_the_other`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_epoch_bookend<H>(
    elf: &Elf,
    elf_bytes: &[u8],
    epoch: &EpochProof,
    register_init: &[u32],
    is_final: bool,
    label: u64,
    opts: &ProofOptions,
    prepared: &DecodePrepared<H>,
) -> Result<Option<Vec<Commitment>>, Error>
where
    H: multilinear::whir_hash::WhirHash,
{
    // ★ THE SAME DERIVATION THE LEVEL-0 DRIVER USES. `None` is this side's
    // answer for DECODE's preprocessed commitment, unchanged.
    let air_set = epoch_airs_for(elf, opts, epoch, register_init, is_final, label, None);
    let air_refs = air_set.refs();
    // By NAME, and exactly one — the same rule the prover applied.
    let decode_at = decode_table_index(&air_refs)?;

    if air_refs.len() != epoch.proof.tables.len() || epoch.table_num_vars.len() != air_refs.len() {
        return Err(Error::InvalidTableCounts(format!(
            "the epoch layout has {} tables, the proof carries {} and {} heights",
            air_refs.len(),
            epoch.proof.tables.len(),
            epoch.table_num_vars.len(),
        )));
    }

    // The width is the AIR's, never the proof's; only the height is stated.
    let shapes: Vec<(usize, usize)> = air_refs
        .iter()
        .zip(&epoch.table_num_vars)
        .map(|(air, &num_vars)| (air.trace_layout().0, num_vars as usize))
        .collect();
    let config = chain_config(&shapes);

    let layouts: Vec<TableLayout<'_, F, E>> = air_refs
        .iter()
        .zip(&shapes)
        .map(|(air, &(width, num_vars))| {
            layout_of(*air, width, num_vars).map_err(|e| Error::Prover(format!("{e:?}")))
        })
        .collect::<Result<_, _>>()?;
    let preprocessed: Vec<Vec<Mle<F>>> = air_refs
        .iter()
        .map(|air| {
            air.precomputed_columns()
                .into_iter()
                .map(|values| Mle::new(values).map_err(|e| Error::Prover(format!("{e:?}"))))
                .collect::<Result<_, _>>()
        })
        .collect::<Result<_, _>>()?;
    let statements: Vec<TableStatement<'_, F, E>> = layouts
        .iter()
        .zip(&preprocessed)
        .map(|(layout, cols)| layout.statement_with_preprocessed(cols))
        .collect();

    let sizes = epoch_groups(shapes.len());
    let (layouts, domains) = crate::multilinear_prove::stacks(&shapes, &sizes, &config)?;
    // The bookend is committed in the last group, alone, so its roots are that
    // group's — as many as the stack split it into.
    let num_polys = layouts.last().map(|l| l.num_polys()).unwrap_or(0);

    prepared.agrees_with(&config)?;
    // ★ The transcript's hash is part of the configuration, and `H` is the
    // caller's dispatch. The bound on `multi_prove`/`multi_verify` rejects any
    // other spelling.
    let mut transcript =
        DefaultTranscript::<E, <H as multilinear::whir_hash::WhirHash>::Transcript>::new(&[]);
    absorb_epoch(
        &mut transcript,
        &statement::elf_digest(elf_bytes),
        &epoch.public_output,
        &epoch.table_counts,
        label,
        &epoch.table_num_vars,
        &config,
    );
    // ★ ONE VALUE, TWO USES. The replay below and the verification below it are
    // handed the same `PreparedCheck`, so they cannot be given two different
    // derived-root lists: the challenges `owed` computes the COMMIT bus's
    // counterparty at are the challenges every table is then checked at. Built
    // once here rather than twice at the two call sites, because "twice" is
    // precisely how the replay came to absorb a shorter roots block than the
    // verification did.
    let decode_columns = prepared.settled_at(decode_at);
    let check = prepared.check(&decode_columns);
    let derived = check.roots;
    // ★ `owed` replays this transcript to draw `z` and `alpha`, which are a
    // function of the configuration's sponge. Computing them against a
    // transcript of a different hash is the same defect one level down, and
    // just as quiet.
    let Some(owed) = owed(
        &epoch.public_output,
        register_init,
        &epoch.proof.roots,
        derived,
        &transcript,
    ) else {
        return Ok(None);
    };
    let verdict = multilinear_table::multi_verify::<_, _, _, H>(
        &epoch.proof,
        &statements,
        &layouts,
        &domains,
        &sizes,
        &owed,
        &config,
        &mut transcript,
        Some(check),
    );
    if verdict.is_err() {
        return Ok(None);
    }
    Ok(epoch.l2g_roots(num_polys).map(<[_]>::to_vec))
}
