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
//! The tables whose leading preprocessed columns are settled out of band are
//! stacked per group ([`BlockPrepared`]): each such stack's derived roots are
//! absorbed after the groups' roots, and its opening proved on its group's fork
//! after the group's own opening.
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

/// A group's PREPARED opening in a block: one commitment, which both sides
/// derive from the program, over the leading preprocessed columns of the
/// group's prepared tables, opened on the group's fork after the group's own
/// opening — each table's columns at that table's point.
///
/// It is the epoch's prepared opening ([`crate::multilinear_table::Prepared`])
/// per group: its roots are DERIVED by the verifier and absorbed after the group
/// roots, before `z`; the columns stay in the group's stack too, and the opening
/// proves the derived commitment takes the values each table's own argument
/// settled on, at that table's point. One stack a group pays one chain for all
/// of its prepared tables.
pub struct BlockPrepared<'a, F, H>
where
    F: IsFFTField + IsPrimeField + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
{
    /// The group whose fork opens it.
    pub group: usize,
    /// The tables it settles, in stack order: each table's position in the
    /// proof's table order and the length of its leading prefix.
    pub tables: Vec<(usize, usize)>,
    pub commitment: &'a StackedCommitment<F, H>,
    /// The tables' prefixes, concatenated in `tables`' order.
    pub columns: &'a [&'a Mle<F>],
}

/// What the verifier settles a [`BlockPrepared`] opening against.
pub struct BlockPreparedCheck<'a, F>
where
    F: IsFFTField + IsPrimeField + 'static,
{
    pub group: usize,
    /// As [`BlockPrepared::tables`]: the prefixes `check_preprocessed` skips.
    pub tables: Vec<(usize, usize)>,
    /// Derived from the program by the verifier, never read from the proof.
    pub roots: &'a [Commitment],
    pub layout: &'a StackedLayout,
    pub domain: &'a Domain<F>,
}

/// The prepared openings' shape, checked the same way by both sides: at most one
/// per group, in group order; each settles distinct tables of its own group in
/// increasing order, each prefix at least one column and at most `max_prefix` of
/// the table's; and the stack holds exactly those columns (`columns`, when the
/// caller knows them).
/// One prepared opening's shape: its group, its `(table, prefix)` list, and the
/// stack's column count when the caller knows it.
type PreparedEntry<'a> = (usize, &'a [(usize, usize)], Option<usize>);

fn prepared_shape(
    entries: &[PreparedEntry<'_>],
    starts: &[usize],
    num_tables: usize,
    max_prefix: &dyn Fn(usize) -> usize,
) -> Result<(), MlError> {
    let mut last_group: Option<usize> = None;
    for &(group, tables, columns) in entries {
        let range = starts
            .get(group)
            .copied()
            .zip(starts.get(group + 1).copied().or(Some(num_tables)))
            .ok_or(MlError::UnknownPolynomial {
                index: group,
                len: starts.len(),
            })?;
        if last_group.is_some_and(|l| group <= l) || tables.is_empty() {
            return Err(MlError::UnknownPolynomial {
                index: group,
                len: starts.len(),
            });
        }
        last_group = Some(group);
        let mut last: Option<usize> = None;
        for &(table, n) in tables {
            if table < range.0 || table >= range.1 || last.is_some_and(|l| table <= l) {
                return Err(MlError::UnknownPolynomial {
                    index: table,
                    len: num_tables,
                });
            }
            if n == 0 || n > max_prefix(table) {
                return Err(MlError::QueryCountMismatch {
                    expected: max_prefix(table),
                    got: n,
                });
            }
            last = Some(table);
        }
        let total: usize = tables.iter().map(|&(_, n)| n).sum();
        if columns.is_some_and(|c| c != total) {
            return Err(MlError::QueryCountMismatch {
                expected: total,
                got: columns.unwrap_or(0),
            });
        }
    }
    Ok(())
}

/// A group's prepared claims: per table of `tables`, its prefix's columns at the
/// table's point (one point per column) and the values its argument settled on.
/// `firsts` are the group's tables' first columns, `start` its first table.
#[allow(clippy::type_complexity)]
fn prepared_claims<E: IsField>(
    tables: &[(usize, usize)],
    start: usize,
    firsts: &[usize],
    points: &[Vec<FieldElement<E>>],
    values: &[FieldElement<E>],
) -> (Vec<Vec<FieldElement<E>>>, Vec<FieldElement<E>>) {
    let mut at_points = Vec::new();
    let mut at_values = Vec::new();
    for &(table, n) in tables {
        let first = firsts[table - start];
        at_points.extend_from_slice(&points[first..first + n]);
        at_values.extend_from_slice(&values[first..first + n]);
    }
    (at_points, at_values)
}

/// How a test makes the prover open a group's prepared stack wrongly, for the
/// verifier to refuse.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct PreparedDeviation {
    /// Which opening, by its index in the `prepared` list.
    pub open: usize,
    /// `(table, other)`: open `table`'s block at `other`'s point (both positions
    /// in the proof's table order, in the opening's group).
    pub at_point_of: Vec<(usize, usize)>,
    /// Claim every value as the committed column evaluated at its claimed point,
    /// so the opening is internally consistent and only the verifier's binding
    /// of it to the tables (and to its derived roots) refuses it.
    pub consistent: bool,
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
    let starts: Vec<usize> = sizes
        .iter()
        .scan(0usize, |at, &size| {
            let start = *at;
            *at += size;
            Some(start)
        })
        .collect();
    let entries: Vec<PreparedEntry<'_>> = prepared
        .iter()
        .map(|p| (p.group, p.tables.as_slice(), Some(p.columns.len())))
        .collect();
    prepared_shape(&entries, &starts, tables.len(), &|t| {
        tables[t].num_committed_columns()
    })?;
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
        if let Some(k) = prepared.iter().position(|p| p.group == g) {
            let p = &prepared[k];
            let (mut at_points, mut at_values) =
                prepared_claims(&p.tables, at, &firsts, &points, &values);
            if let Some(d) = deviations.iter().find(|d| d.open == k) {
                // Each block's columns sit at its table's place in the stack.
                let mut column = 0usize;
                for &(table, n) in &p.tables {
                    if let Some(&(_, other)) = d.at_point_of.iter().find(|(t, _)| *t == table) {
                        let local = other.checked_sub(at).filter(|&l| l < firsts.len()).ok_or(
                            MlError::UnknownPolynomial {
                                index: other,
                                len: firsts.len(),
                            },
                        )?;
                        for point in &mut at_points[column..column + n] {
                            *point = points[firsts[local]].clone();
                        }
                    }
                    column += n;
                }
                if d.consistent {
                    at_values = p
                        .columns
                        .iter()
                        .zip(&at_points)
                        .map(|(c, point)| c.evaluate_in(point))
                        .collect::<Result<_, _>>()?;
                }
            }
            prepared_openings.push(stacked_eval::prove::<F, E, T, H>(
                p.commitment,
                p.columns,
                None,
                &Claimed::PerColumn(&at_points),
                &at_values,
                config,
                &mut fork,
            )?);
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
    // One opening per group with a prepared stack, each table settled at most
    // once and by its own group, each prefix within the table's preprocessed
    // columns — never more than it has, which would skip a check nothing
    // replaced — and the stack holding exactly those columns.
    if prepared_openings.len() != prepared.len() {
        return Err(MlError::QueryCountMismatch {
            expected: prepared.len(),
            got: prepared_openings.len(),
        });
    }
    let starts: Vec<usize> = sizes
        .iter()
        .scan(0usize, |at, &size| {
            let start = *at;
            *at += size;
            Some(start)
        })
        .collect();
    let entries: Vec<PreparedEntry<'_>> = prepared
        .iter()
        .map(|p| {
            (
                p.group,
                p.tables.as_slice(),
                Some(p.layout.placements().len()),
            )
        })
        .collect();
    prepared_shape(&entries, &starts, statements.len(), &|t| {
        statements[t].num_preprocessed
    })?;
    let mut settled = vec![0usize; statements.len()];
    for p in prepared {
        for &(table, n) in &p.tables {
            settled[table] = n;
        }
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
        // The group's prepared stack, on this fork: the derived commitment must
        // take the values each table settled on, at that table's point.
        if let Some(k) = prepared
            .iter()
            .position(|p| p.group == g)
            .filter(|_| !skip_prepared)
        {
            let firsts: Vec<usize> = statements[statement_at..statement_at + size]
                .iter()
                .scan(0usize, |first, s| {
                    let at = *first;
                    *first += s.slot_of.len();
                    Some(at)
                })
                .collect();
            let p = &prepared[k];
            let (at_points, at_values) =
                prepared_claims(&p.tables, statement_at, &firsts, &points, &values);
            stacked_eval::verify::<F, E, T, H>(
                &prepared_openings[k],
                p.layout,
                p.roots,
                &Claimed::PerColumn(&at_points),
                &at_values,
                p.domain,
                config,
                &mut fork,
            )?;
        }
        statement_at += size;
        root_at += layout.num_polys();
    }
    if balance != *expected {
        return Err(MlError::BusImbalance);
    }
    Ok(())
}
