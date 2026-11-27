//! The commit and opening boundary shared with the process transport.

pub use super::inproc::{Challenges, Committed, GroupParts, GroupShape, SealedPlan};
use super::*;
use crypto::fiat_shamir::is_transcript::IsTranscript;

impl<F, E> BlockCommitted<'_, F, E>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    pub fn is_replica_group(&self, shapes: &[(usize, usize)], config: &ChainConfig) -> bool {
        let [stack] = self.groups.as_slice() else {
            return false;
        };
        let Ok(expected) = global_layout(shapes, config.format.stack) else {
            return false;
        };
        self.sizes == [shapes.len()]
            && self.tables.len() == shapes.len()
            && self
                .tables
                .iter()
                .zip(shapes)
                .all(|(table, &(width, vars))| {
                    table.num_committed_columns() == width && table.num_vars() == vars
                })
            && stack.layout().placements() == expected.placements()
    }

    pub fn replica_receipt(&self, prepared_roots: Vec<Commitment>) -> Committed<E> {
        debug_assert_eq!(self.sizes.len(), 1);
        Committed {
            roots: self.roots.clone(),
            answers: self.ood[0].answers.clone(),
            prepared_roots,
            local_digests: self.local_digests().unwrap_or_default().to_vec(),
        }
    }

    /// Returns owned proof parts once the single-group lane, its stores and
    /// its prepared-opening scratch have all been released.
    pub fn open_replica_group<H, T>(
        self,
        index: usize,
        config: &ChainConfig,
        post: &T,
        challenges: Challenges<E>,
        prepared: &[BlockPrepared<'_, F, H>],
    ) -> Result<(GroupParts<F, E>, GroupStamps), MlError>
    where
        H: WhirHash,
        T: IsTranscript<E>
            + crypto::fiat_shamir::transcript_hash::HasTranscriptHash<Hash = H::Transcript>
            + Clone
            + Send
            + Default
            + 'static,
    {
        let (mut proof, argues, prepared, mut stamps, _, locals) =
            block_prove_with_front::<F, E, T, H>(
                self,
                config,
                &mut post.clone(),
                prepared,
                &[],
                &|_| index,
                &ArgueDeviation::default(),
                &|_| {},
                &[0],
                None,
                Some(challenges),
            )?;
        Ok((
            GroupParts {
                tables: proof.tables,
                argue: argues.into_iter().next(),
                local: locals.into_iter().find(|argue| !argue.is_empty()),
                opening: proof.columns.remove(0),
                prepared: prepared.into_iter().next(),
            },
            stamps.remove(0),
        ))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn verify_group<F, E, T, H>(
    group: usize,
    receipt: &Committed<E>,
    parts: &GroupParts<F, E>,
    statements: &[TableStatement<'_, F, E>],
    at: usize,
    prepared: Option<&BlockPreparedCheck<'_, F>>,
    layout: &StackedLayout,
    domain: &Domain<F>,
    config: &ChainConfig,
    post: &T,
    (z, alpha, beta): &Challenges<E>,
    buses: Option<&[u64]>,
) -> Result<(), MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>
        + crypto::fiat_shamir::transcript_hash::HasTranscriptHash<Hash = H::Transcript>
        + Clone
        + Default,
{
    let mut settled = vec![0; statements.len()];
    if let Some(prepared) = prepared {
        for &(table, count) in &prepared.tables {
            settled[table - at] = count;
        }
    }
    verify_opened_group::<F, E, T, H>(
        parts.view(group, &receipt.roots, &receipt.answers),
        statements,
        &settled,
        at,
        prepared,
        layout,
        domain,
        config,
        z,
        alpha,
        beta,
        group_fork(post, group),
        VerifierChecks::ALL,
        buses.map(|buses| (buses, receipt.local_digests.as_slice())),
    )?;
    Ok(())
}
