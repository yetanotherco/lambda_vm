//! One multilinear proof for a whole block, proved in two phases.
//!
//! The block's tables — every chunk of every table type, run at full height —
//! are split into **groups**: contiguous runs in table order, each as many
//! tables as fit a few stacked polynomials. That is the unit a card holds. The
//! proof then goes:
//!
//! 1. **Phase A (commit).** Group by group: the columns go up, the stack is
//!    committed, the roots are taken, and the codewords go — only the top of
//!    each tree stays, on the host ([`RetiredStack`]). No card holds a block's
//!    codewords, and every root must exist before the first challenge.
//! 2. **The roots block.** Every root, in group order, into the transcript;
//!    then `(z, α, β)` once, shared by every table of the block
//!    ([`absorb_roots_and_challenge`], the same function every other proof
//!    uses). The transcript's state after the draw is the block's `S_post`.
//! 3. **Phase B (prove), per group `g`, on the fork `S_post ‖ g`.** The group's
//!    columns go up again; each of its tables is argued (today's
//!    [`prove`]); the group's codewords are recomputed — no hash, the tree's
//!    top is kept — and its stack is opened at the tables' points
//!    ([`stacked_eval::prove`]).
//!
//! A table whose leading preprocessed columns are settled out of band
//! ([`BlockPrepared`]) has its derived commitment's roots absorbed after the
//! groups' roots, and its opening proved on its group's fork after the group's
//! own opening.
//!
//! The verifier mirrors it and checks the bus balance ONCE, over every table of
//! the block: the LogUp challenges are shared, so the block's fractions sum to
//! what the statement owes exactly as one monolithic proof's do.
//!
//! # Why a fork per group is sound
//!
//! Each fork starts from `S_post`, which binds the statement and every root of
//! the block, and then the group's index; everything the group's argument and
//! opening draw is a function of that and of the group's own messages. This is
//! the STARK prover's per-table fork (`crypto/stark/src/prover.rs`, `append(idx)`)
//! at the granularity of a group. A prover cannot move a group's messages to
//! another group (the index is absorbed), cannot choose a root after a
//! challenge (all are absorbed before `z`), and cannot mix columns of two
//! groups (each opening is against its own group's roots, sliced by the
//! verifier's own layouts).
//!
//! ★ THE GROUPS ARE THE VERIFIER'S, never the proof's: [`block_groups`] is a
//! function of the tables' shapes and two format constants, and both sides call
//! it.

use std::sync::Arc;
use std::time::Instant;

use math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsPrimeField, IsSubFieldOf},
    },
    traits::AsBytes,
};
use multilinear::{
    Error as MlError,
    mle::Mle,
    stacked_eval::{self, Claimed, ColumnsAt, RetiredStack, StackedCommitment, StackedProof},
    stacking::StackedLayout,
    whir::Domain,
    whir_chain::{ChainConfig, StackVars},
    whir_commit::Commitment,
    whir_hash::WhirHash,
};

use crate::multilinear_table::{
    CommittedTable, MultiProof, TableStatement, absorb_roots_and_challenge, contribution,
    global_layout, prove, verify,
};

/// How the block's tables split into groups: contiguous runs in table order,
/// each as many tables as its stack holds in at most `max_polys` polynomials
/// under the stack cap. A table that alone needs more is a group of its own.
///
/// `shapes` is `(committed columns, height in variables)` per table. Returns
/// the size of each group, in order; they sum to `shapes.len()`.
pub fn block_groups(
    shapes: &[(usize, usize)],
    cap: StackVars,
    max_polys: usize,
) -> Result<Vec<usize>, MlError> {
    let mut sizes = Vec::new();
    let mut start = 0usize;
    for end in 1..=shapes.len() {
        let polys = global_layout(&shapes[start..end], cap)?.num_polys();
        if polys > max_polys && end - start > 1 {
            sizes.push(end - 1 - start);
            start = end - 1;
        }
    }
    if start < shapes.len() {
        sizes.push(shapes.len() - start);
    }
    Ok(sizes)
}

/// What a group cost, for the readout. Seconds of wall time on the prover's
/// thread.
#[derive(Clone, Debug, Default)]
pub struct GroupStamps {
    pub tables: usize,
    pub polys: usize,
    pub cells: usize,
    /// Phase A: the wait for the group's tables (a streamed producer), the
    /// columns up, the commit, the roots, the tops home.
    pub wait_a: f64,
    pub upload_a: f64,
    pub commit: f64,
    pub retire: f64,
    /// Phase B: the columns up again, the tables' arguments, the codewords
    /// recomputed, the opening.
    pub upload_b: f64,
    pub argue: f64,
    pub encode: f64,
    pub open: f64,
    /// Host bytes kept of the group's trees between the phases.
    pub tree_bytes: usize,
    /// When the group's commit ended, seconds since phase A started.
    pub committed_at: f64,
}

/// A table's PREPARED opening in a block: a commitment both sides derive from
/// the program, over the table's leading preprocessed columns, opened at the
/// table's own point on its group's fork, after the group's own opening.
///
/// It is the epoch's prepared opening ([`crate::multilinear_table::Prepared`])
/// at the granularity of one table: the root is DERIVED by the verifier and
/// absorbed after the group roots, before `z`; the columns stay in the group's
/// stack too, and the opening proves the derived commitment takes the values
/// the table's own argument settled on, at that table's point. One commitment
/// per table keeps every derived root independent of the grouping.
pub struct BlockPrepared<'a, F, H>
where
    F: IsFFTField + IsPrimeField + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
{
    /// The table's position in the proof's table order.
    pub table: usize,
    pub commitment: &'a StackedCommitment<F, H>,
    /// The table's leading preprocessed columns, in order: the prefix the
    /// opening settles.
    pub columns: &'a [&'a Mle<F>],
}

/// What the verifier settles a [`BlockPrepared`] opening against.
pub struct BlockPreparedCheck<'a, F>
where
    F: IsFFTField + IsPrimeField + 'static,
{
    /// The table's position in the proof's table order.
    pub table: usize,
    /// Derived from the program by the verifier, never read from the proof.
    pub roots: &'a [Commitment],
    /// Its columns are the table's leading preprocessed columns, so their count
    /// is the prefix `check_preprocessed` skips.
    pub layout: &'a StackedLayout,
    pub domain: &'a Domain<F>,
}

/// Prepared tables must name distinct tables in increasing order, so each
/// table has at most one opening and the derived roots have one order.
fn prepared_order(tables: impl Iterator<Item = usize>, num_tables: usize) -> Result<(), MlError> {
    let mut last: Option<usize> = None;
    for table in tables {
        if table >= num_tables || last.is_some_and(|l| table <= l) {
            return Err(MlError::UnknownPolynomial {
                index: table,
                len: num_tables,
            });
        }
        last = Some(table);
    }
    Ok(())
}

/// How a test makes the prover open a prepared table wrongly, for the verifier
/// to refuse. Each opening it makes is internally consistent — the columns it
/// opens, evaluated at the point it opens them — so only the verifier's own
/// binding of the opening to the table can refuse it.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PreparedDeviation {
    /// Which opening, by its index in the `prepared` list.
    pub open: usize,
    /// Open at this table's point (its position in the proof's table order,
    /// in the same group) instead of the prepared table's own.
    pub at_table: Option<usize>,
    /// Open this prepared entry's commitment and columns (an index in the
    /// `prepared` list) instead of its own.
    pub with: Option<usize>,
}

/// The block after phase A: every table, the groups' roots and the tops of
/// their trees. The codewords are gone.
pub struct BlockCommitted<'a, F, E>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    tables: Vec<CommittedTable<'a, F, E>>,
    sizes: Vec<usize>,
    groups: Vec<RetiredStack<F>>,
    roots: Vec<Commitment>,
    stamps: Vec<GroupStamps>,
}

impl<'a, F, E> BlockCommitted<'a, F, E>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    /// Phase A: commits each group in turn and lets its codewords go, keeping
    /// each tree but its bottom `drop_levels` levels on the host.
    ///
    /// `sizes` must be [`block_groups`]'s over the same shapes and the
    /// config's stack cap, or the verifier rebuilds other stacks.
    pub fn commit<H: WhirHash>(
        tables: Vec<CommittedTable<'a, F, E>>,
        sizes: &[usize],
        config: &ChainConfig,
        drop_levels: usize,
    ) -> Result<Self, MlError> {
        if sizes.iter().sum::<usize>() != tables.len() {
            return Err(MlError::QueryCountMismatch {
                expected: tables.len(),
                got: sizes.iter().sum(),
            });
        }
        let mut tables = tables.into_iter();
        let groups = sizes
            .iter()
            .map(|&size| tables.by_ref().take(size).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        Self::commit_streamed::<H>(groups, sizes, config, drop_levels)
    }

    /// [`Self::commit`] over groups handed over one at a time, in group order —
    /// by a producer still preparing the later ones while the card commits
    /// the earlier. Each group's wait for its tables is stamped (`wait_a`). A
    /// producer that stops early leaves fewer groups than `sizes` names, which
    /// is refused.
    pub fn commit_streamed<H: WhirHash>(
        groups: impl IntoIterator<Item = Vec<CommittedTable<'a, F, E>>>,
        sizes: &[usize],
        config: &ChainConfig,
        drop_levels: usize,
    ) -> Result<Self, MlError> {
        let block =
            Self::commit_groups::<H>(groups.into_iter().take(sizes.len()), config, drop_levels)?;
        if block.sizes != sizes {
            return Err(MlError::QueryCountMismatch {
                expected: sizes.len(),
                got: block.sizes.len(),
            });
        }
        Ok(block)
    }

    /// Phase A over whatever groups arrive, in arrival order, until the
    /// producer stops: the groups (and so their sizes) are the prover's, and the
    /// statement carries them. Each group's wait for its tables is stamped.
    pub fn commit_groups<H: WhirHash>(
        groups: impl IntoIterator<Item = Vec<CommittedTable<'a, F, E>>>,
        config: &ChainConfig,
        drop_levels: usize,
    ) -> Result<Self, MlError> {
        let mut tables = Vec::new();
        let mut sizes = Vec::new();
        let mut retired_groups = Vec::new();
        let mut roots = Vec::new();
        let mut stamps = Vec::new();
        let mut incoming = groups.into_iter();
        let started = Instant::now();
        loop {
            let waited = Instant::now();
            let Some(group) = incoming.next() else {
                break;
            };
            let size = group.len();
            if size == 0 {
                return Err(MlError::QueryCountMismatch {
                    expected: 1,
                    got: 0,
                });
            }
            sizes.push(size);
            let mut stamp = GroupStamps {
                tables: size,
                wait_a: waited.elapsed().as_secs_f64(),
                ..Default::default()
            };
            let t = Instant::now();
            let columns: Vec<&Mle<F>> = group.iter().flat_map(|t| t.columns()).collect();
            // Held in an `Arc` like phase B's, so the store is dropped on
            // purpose below, as soon as the tree tops are home.
            let store = multilinear::gpu::upload_columns(&columns).map(Arc::new);
            stamp.upload_a = t.elapsed().as_secs_f64();
            let shapes: Vec<(usize, usize)> = group
                .iter()
                .map(|t| (t.num_committed_columns(), t.num_vars()))
                .collect();
            stamp.cells = shapes.iter().map(|&(w, n)| w << n).sum();
            let layout = global_layout(&shapes, config.format.stack)?;
            stamp.polys = layout.num_polys();
            let t = Instant::now();
            let stacked = StackedCommitment::<F, H>::commit(
                layout,
                &columns,
                store.as_ref().map(|store| (&**store, 0)),
                config,
            )?;
            roots.extend(stacked.roots());
            stamp.commit = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let retired = stacked.retire(drop_levels, config)?;
            drop(store);
            drop(columns);
            stamp.retire = t.elapsed().as_secs_f64();
            stamp.tree_bytes = retired.tree_bytes();
            stamp.committed_at = started.elapsed().as_secs_f64();
            retired_groups.push(retired);
            stamps.push(stamp);
            tables.extend(group);
        }
        Ok(Self {
            tables,
            sizes,
            groups: retired_groups,
            roots,
            stamps,
        })
    }

    /// Every group's roots, in group order — what the transcript absorbs.
    pub fn roots(&self) -> &[Commitment] {
        &self.roots
    }

    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    pub fn tables(&self) -> &[CommittedTable<'a, F, E>] {
        &self.tables
    }

    /// Phase A's stamps, one per group.
    pub fn stamps(&self) -> &[GroupStamps] {
        &self.stamps
    }
}

/// A group's columns on the card, shared by its tables.
type Store = Option<Arc<multilinear::gpu::ResidentColumns>>;

/// Uploads a group's columns, in table order; `None` when the card declines
/// (every reader then takes the host copy).
fn upload_group<F, E>(group: &[CommittedTable<'_, F, E>]) -> Store
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    let columns: Vec<&Mle<F>> = group.iter().flat_map(|t| t.columns()).collect();
    multilinear::gpu::upload_columns(&columns).map(Arc::new)
}

/// The fork a group proves on: `S_post` then the group's index.
fn group_fork<E, T>(transcript: &T, group: usize) -> T
where
    E: IsField + 'static,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E> + Clone,
{
    let mut fork = transcript.clone();
    fork.append_bytes(&(group as u64).to_le_bytes());
    fork
}

/// Phase B and the proof: the roots block on `transcript`, then each group on
/// its own fork. Consumes the block — each group's host columns and kept tree
/// top are released as its opening ends. Returns the proof and every group's
/// stamps (phase A's, completed by phase B's).
///
/// The proof is a [`MultiProof`] — the roots, a table proof per table in
/// order, an opening per group — read under the forked schedule by
/// [`block_verify`] and by nothing else.
pub fn block_prove<F, E, T, H>(
    committed: BlockCommitted<'_, F, E>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<(MultiProof<F, E>, Vec<GroupStamps>), MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>
        + crypto::fiat_shamir::transcript_hash::HasTranscriptHash<Hash = <H as WhirHash>::Transcript>
        + Clone,
{
    let (proof, _, stamps) =
        block_prove_on_forks::<F, E, T, H>(committed, config, transcript, &[], &[], &|g| g)?;
    Ok((proof, stamps))
}

/// [`block_prove`] with the tables' prepared openings, and group `g` proved on
/// the fork of index `fork_of(g)`. Returns the proof, the prepared openings in
/// `prepared`'s order, and the stamps.
///
/// Only a test proves on a fork map other than the identity: it is how a proof
/// whose groups sit on the wrong forks is built, for the verifier to refuse.
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub fn block_prove_on_forks<F, E, T, H>(
    committed: BlockCommitted<'_, F, E>,
    config: &ChainConfig,
    transcript: &mut T,
    prepared: &[BlockPrepared<'_, F, H>],
    deviations: &[PreparedDeviation],
    fork_of: &dyn Fn(usize) -> usize,
) -> Result<(MultiProof<F, E>, Vec<StackedProof<F, E>>, Vec<GroupStamps>), MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>
        + crypto::fiat_shamir::transcript_hash::HasTranscriptHash<Hash = <H as WhirHash>::Transcript>
        + Clone,
{
    let BlockCommitted {
        mut tables,
        sizes,
        groups,
        roots,
        mut stamps,
    } = committed;
    prepared_order(prepared.iter().map(|p| p.table), tables.len())?;
    for p in prepared {
        if p.columns.is_empty() || p.columns.len() > tables[p.table].num_committed_columns() {
            return Err(MlError::QueryCountMismatch {
                expected: tables[p.table].num_committed_columns(),
                got: p.columns.len(),
            });
        }
    }
    let derived: Vec<Commitment> = prepared.iter().flat_map(|p| p.commitment.roots()).collect();
    let (z, alpha, beta) = absorb_roots_and_challenge::<E, T>(transcript, &roots, &derived);

    let mut table_proofs = Vec::with_capacity(tables.len());
    let mut openings = Vec::with_capacity(sizes.len());
    let mut prepared_openings = Vec::with_capacity(prepared.len());
    let mut at = 0usize;
    // The next group's columns, uploaded during this group's argument — the
    // card idles through the argument's host glue, and a group's store is a
    // few GiB beside the argument's working set, not beside its codewords.
    let mut pre_uploaded: Option<Store> = None;
    for (g, (retired, &size)) in groups.into_iter().zip(&sizes).enumerate() {
        let mut fork = group_fork::<E, T>(transcript, fork_of(g));
        let (head, tail) = tables.split_at_mut(at + size);
        let group = &mut head[at..];
        let next = sizes.get(g + 1).map(|&n| &tail[..n]);

        let t = Instant::now();
        let store = match pre_uploaded.take() {
            Some(store) => store,
            None => upload_group(group),
        };
        if let Some(store) = &store {
            let mut first = 0usize;
            for table in group.iter_mut() {
                table.set_resident(store.clone(), first);
                first += table.num_committed_columns();
            }
        }
        stamps[g].upload_b += t.elapsed().as_secs_f64();

        // The group's tables, one after another, each leaving its columns
        // claimed at its own point — with the next group's upload beside them.
        let t = Instant::now();
        let mut points: Vec<Vec<FieldElement<E>>> = Vec::new();
        let mut values: Vec<FieldElement<E>> = Vec::new();
        let group_ref: &[CommittedTable<'_, F, E>] = group;
        let (argued, next_store, joined) = std::thread::scope(|scope| {
            let uploader = next.map(|next| scope.spawn(move || upload_group(next)));
            let argued = (|| -> Result<(), MlError> {
                for table in group_ref.iter() {
                    let (proof, point) = prove(table, &z, &alpha, &beta, &mut fork, None)?;
                    for _ in 0..table.num_committed_columns() {
                        points.push(point.clone());
                    }
                    values.extend(proof.constraint.reduce.column_values.iter().cloned());
                    table_proofs.push(proof);
                }
                Ok(())
            })();
            let argue_end = Instant::now();
            let next_store = uploader.map(|handle| handle.join());
            (argued, next_store, argue_end.elapsed().as_secs_f64())
        });
        argued?;
        stamps[g].argue = t.elapsed().as_secs_f64() - joined;
        if let Some(next_store) = next_store {
            // The wait for the upload after the argument ended is the next
            // group's upload cost; the rest of it hid behind this argument.
            stamps[g + 1].upload_b += joined;
            pre_uploaded = Some(next_store.map_err(|_| MlError::DeviceFailed {
                stage: "uploading the next group's columns",
            })?);
        }

        let columns: Vec<&Mle<F>> = group.iter().flat_map(|t| t.columns()).collect();
        let t = Instant::now();
        let stacked = retired.revive::<H>(
            &columns,
            store.as_ref().map(|store| (&**store, ColumnsAt::From(0))),
            config,
        )?;
        stamps[g].encode = t.elapsed().as_secs_f64();
        let t = Instant::now();
        openings.push(stacked_eval::prove::<F, E, T, H>(
            &stacked,
            &columns,
            store.as_ref().map(|store| (&**store, 0)),
            &Claimed::PerColumn(&points),
            &values,
            config,
            &mut fork,
        )?);
        // The group's prepared tables, in table order: each opened at its own
        // point for its own prefix, on this fork.
        let firsts: Vec<usize> = group
            .iter()
            .scan(0usize, |first, table| {
                let at = *first;
                *first += table.num_committed_columns();
                Some(at)
            })
            .collect();
        for (index, &first) in firsts.iter().enumerate() {
            let Some(k) = prepared.iter().position(|p| p.table == at + index) else {
                continue;
            };
            let p = &prepared[k];
            let n = p.columns.len();
            let opening = match deviations.iter().find(|d| d.open == k) {
                None => stacked_eval::prove::<F, E, T, H>(
                    p.commitment,
                    p.columns,
                    None,
                    &Claimed::PerColumn(&points[first..first + n]),
                    &values[first..first + n],
                    config,
                    &mut fork,
                )?,
                Some(d) => {
                    let other = &prepared[d.with.unwrap_or(k)];
                    let point = match d.at_table {
                        Some(t) => {
                            let local = t.checked_sub(at).filter(|&l| l < firsts.len()).ok_or(
                                MlError::UnknownPolynomial {
                                    index: t,
                                    len: firsts.len(),
                                },
                            )?;
                            points[firsts[local]].clone()
                        }
                        None => points[first].clone(),
                    };
                    let claims: Vec<Vec<FieldElement<E>>> =
                        vec![point.clone(); other.columns.len()];
                    let opened: Vec<FieldElement<E>> = other
                        .columns
                        .iter()
                        .map(|column| column.evaluate_in(&point))
                        .collect::<Result<_, _>>()?;
                    stacked_eval::prove::<F, E, T, H>(
                        other.commitment,
                        other.columns,
                        None,
                        &Claimed::PerColumn(&claims),
                        &opened,
                        config,
                        &mut fork,
                    )?
                }
            };
            prepared_openings.push(opening);
        }
        stamps[g].open = t.elapsed().as_secs_f64();
        drop(stacked);
        drop(columns);
        for table in group.iter_mut() {
            table.clear_resident();
        }
        drop(store);
        at += size;
    }
    Ok((
        MultiProof {
            roots,
            tables: table_proofs,
            columns: openings,
            preprocessed: None,
        },
        prepared_openings,
        stamps,
    ))
}

/// Verifies a [`block_prove`] proof: the roots block, each group on its fork,
/// and the bus balance over every table of the block against `expected`.
///
/// `layouts`/`domains` are one per group, rebuilt by the caller from the
/// shapes and `sizes` ([`block_groups`]) — never from the proof. `prepared` are
/// the tables' prepared checks the caller derived from the program, and
/// `prepared_openings` the proof's openings of them, one each, in that order.
#[allow(clippy::too_many_arguments)]
pub fn block_verify<F, E, T, H>(
    proof: &MultiProof<F, E>,
    prepared_openings: &[StackedProof<F, E>],
    prepared: &[BlockPreparedCheck<'_, F>],
    statements: &[TableStatement<'_, F, E>],
    layouts: &[StackedLayout],
    domains: &[Domain<F>],
    sizes: &[usize],
    expected: &FieldElement<E>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<(), MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>
        + crypto::fiat_shamir::transcript_hash::HasTranscriptHash<Hash = <H as WhirHash>::Transcript>
        + Clone,
{
    block_verify_with::<F, E, T, H>(
        proof,
        prepared_openings,
        prepared,
        statements,
        layouts,
        domains,
        sizes,
        expected,
        config,
        transcript,
        false,
    )
}

/// [`block_verify`] with the prepared openings left unchecked when
/// `skip_prepared` — their prefixes still skipped by `check_preprocessed` — a
/// mutation: a test shows the openings are what refuses a prover that opens a
/// prepared table wrongly.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn block_verify_with<F, E, T, H>(
    proof: &MultiProof<F, E>,
    prepared_openings: &[StackedProof<F, E>],
    prepared: &[BlockPreparedCheck<'_, F>],
    statements: &[TableStatement<'_, F, E>],
    layouts: &[StackedLayout],
    domains: &[Domain<F>],
    sizes: &[usize],
    expected: &FieldElement<E>,
    config: &ChainConfig,
    transcript: &mut T,
    skip_prepared: bool,
) -> Result<(), MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>
        + crypto::fiat_shamir::transcript_hash::HasTranscriptHash<Hash = <H as WhirHash>::Transcript>
        + Clone,
{
    if proof.tables.len() != statements.len() {
        return Err(MlError::QueryCountMismatch {
            expected: statements.len(),
            got: proof.tables.len(),
        });
    }
    if layouts.len() != sizes.len()
        || domains.len() != sizes.len()
        || proof.columns.len() != sizes.len()
        || sizes.iter().sum::<usize>() != statements.len()
    {
        return Err(MlError::QueryCountMismatch {
            expected: sizes.len(),
            got: proof.columns.len(),
        });
    }
    // Every root the proof carries is one some group opens: a root no layout
    // claims would be absorbed and bound to nothing.
    let polys: usize = layouts.iter().map(StackedLayout::num_polys).sum();
    if polys != proof.roots.len() {
        return Err(MlError::QueryCountMismatch {
            expected: polys,
            got: proof.roots.len(),
        });
    }
    if proof.preprocessed.is_some() {
        return Err(MlError::QueryCountMismatch {
            expected: 0,
            got: 1,
        });
    }
    // One opening per prepared table, each table at most once, and each
    // settling a prefix of the table's preprocessed columns — never more than
    // the table has, which would skip a check nothing replaced.
    if prepared_openings.len() != prepared.len() {
        return Err(MlError::QueryCountMismatch {
            expected: prepared.len(),
            got: prepared_openings.len(),
        });
    }
    prepared_order(prepared.iter().map(|p| p.table), statements.len())?;
    let mut settled = vec![0usize; statements.len()];
    for p in prepared {
        let n = p.layout.placements().len();
        if n == 0 || n > statements[p.table].num_preprocessed {
            return Err(MlError::QueryCountMismatch {
                expected: statements[p.table].num_preprocessed,
                got: n,
            });
        }
        settled[p.table] = n;
    }
    let derived: Vec<Commitment> = prepared
        .iter()
        .flat_map(|p| p.roots.iter().copied())
        .collect();
    let (z, alpha, beta) = absorb_roots_and_challenge::<E, T>(transcript, &proof.roots, &derived);

    let mut balance = FieldElement::<E>::zero();
    let mut statement_at = 0usize;
    let mut root_at = 0usize;
    for (g, (((opening, layout), domain), &size)) in proof
        .columns
        .iter()
        .zip(layouts)
        .zip(domains)
        .zip(sizes)
        .enumerate()
    {
        let mut fork = group_fork::<E, T>(transcript, g);
        let mut points: Vec<Vec<FieldElement<E>>> = Vec::new();
        let mut values: Vec<FieldElement<E>> = Vec::new();
        for ((table, statement), &settled) in proof.tables[statement_at..statement_at + size]
            .iter()
            .zip(&statements[statement_at..statement_at + size])
            .zip(&settled[statement_at..statement_at + size])
        {
            let (output, reduced) =
                verify(table, *statement, &z, &alpha, &beta, &mut fork, settled)?;
            balance += contribution(&output).ok_or(MlError::BusImbalance)?;
            for _ in 0..statement.slot_of.len() {
                points.push(reduced.point.clone());
            }
            values.extend(reduced.column_values);
        }
        let roots = &proof.roots[root_at..root_at + layout.num_polys()];
        stacked_eval::verify::<F, E, T, H>(
            opening,
            layout,
            roots,
            &Claimed::PerColumn(&points),
            &values,
            domain,
            config,
            &mut fork,
        )?;
        // The group's prepared tables, in table order, on this fork: each
        // derived commitment must take the values the table settled on, at the
        // table's point.
        let mut first = 0usize;
        for (index, statement) in statements[statement_at..statement_at + size]
            .iter()
            .enumerate()
        {
            if let Some(k) = prepared
                .iter()
                .position(|p| p.table == statement_at + index)
                .filter(|_| !skip_prepared)
            {
                let (p, n) = (&prepared[k], settled[statement_at + index]);
                stacked_eval::verify::<F, E, T, H>(
                    &prepared_openings[k],
                    p.layout,
                    p.roots,
                    &Claimed::PerColumn(&points[first..first + n]),
                    &values[first..first + n],
                    p.domain,
                    config,
                    &mut fork,
                )?;
            }
            first += statement.slot_of.len();
        }
        statement_at += size;
        root_at += layout.num_polys();
    }
    if balance != *expected {
        return Err(MlError::BusImbalance);
    }
    Ok(())
}
