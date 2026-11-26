//! Stage 2 control messages. The adapter shares the existing phase-A witness
//! and phase-B card scheduler. One logical worker can own groups on any card.

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, mpsc};

use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::hash::blake3::chain::Blake3Std;
use digest::Digest;
use math::{
    field::{element::FieldElement, traits::IsField},
    traits::AsBytes,
};
use multilinear::{
    Error as MlError, stacked_eval::StackedProof, whir_chain::ArgueFormat, whir_commit::Commitment,
};

use super::{
    BatchedArgue, GroupOpened, LocalArgue, TableProof, absorb_roots_answers_and_challenge,
    group_fork,
};

#[derive(Clone, Copy)]
pub struct InprocWorkers(NonZeroUsize);

impl InprocWorkers {
    pub fn from_env() -> Result<Option<Self>, String> {
        const ENV: &str = "LAMBDA_VM_CLUSTER";
        match std::env::var(ENV) {
            Err(std::env::VarError::NotPresent) => Ok(None),
            Ok(value) => Self::parse(&value),
            Err(e) => Err(format!("{ENV}: {e}")),
        }
    }

    pub(crate) fn parse(value: &str) -> Result<Option<Self>, String> {
        match value.trim() {
            "" | "off" => Ok(None),
            value => value
                .strip_prefix("inproc:")
                .and_then(|n| n.parse::<NonZeroUsize>().ok())
                .filter(|n| n.get() <= 64)
                .map(|n| Some(Self(n)))
                .ok_or_else(|| "LAMBDA_VM_CLUSTER expects off or inproc:N (1–64)".into()),
        }
    }
}

pub(super) struct InprocRequest {
    pub workers: InprocWorkers,
    pub statement: Vec<Vec<u8>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct Header {
    attempt: u128,
    worker: usize,
    group: usize,
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct Reply<T> {
    header: Header,
    payload: T,
}

#[derive(Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub(super) struct Committed<E: IsField> {
    pub roots: Vec<Commitment>,
    pub answers: Vec<FieldElement<E>>,
    pub prepared_roots: Vec<Commitment>,
    pub local_digests: Vec<[u8; 32]>,
}

impl<E: IsField> Committed<E> {
    fn matches(&self, other: &Self) -> bool {
        self.roots == other.roots
            && self.answers == other.answers
            && self.prepared_roots == other.prepared_roots
            && self.local_digests == other.local_digests
    }
}

type Challenges<E> = (FieldElement<E>, FieldElement<E>, FieldElement<E>);

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct SealedPlan<E: IsField> {
    // Preserve the boundaries between the statement's append_bytes calls.
    statement: Vec<Vec<u8>>,
    groups: Vec<Committed<E>>,
    // The original prepared-stack order is part of the transcript.
    derived: Vec<Commitment>,
}

impl<E: IsField + 'static> SealedPlan<E> {
    fn replay<T: IsTranscript<E>>(&self, transcript: &mut T) -> Challenges<E> {
        for bytes in &self.statement {
            transcript.append_bytes(bytes);
        }
        let roots: Vec<_> = self
            .groups
            .iter()
            .flat_map(|g| g.roots.iter().copied())
            .collect();
        let answers: Vec<_> = self
            .groups
            .iter()
            .flat_map(|g| g.answers.iter().cloned())
            .collect();
        absorb_roots_answers_and_challenge(transcript, &roots, &self.derived, &answers)
    }

    fn digest(&self) -> Commitment
    where
        FieldElement<E>: AsBytes,
    {
        let mut hash = Blake3Std::new();
        hash.update((self.statement.len() as u64).to_le_bytes());
        for bytes in &self.statement {
            hash.update((bytes.len() as u64).to_le_bytes());
            hash.update(bytes);
        }
        hash.update((self.derived.len() as u64).to_le_bytes());
        for root in &self.derived {
            hash.update(root);
        }
        hash.update((self.groups.len() as u64).to_le_bytes());
        for group in &self.groups {
            for digests in [&group.roots, &group.prepared_roots, &group.local_digests] {
                hash.update((digests.len() as u64).to_le_bytes());
                for digest in digests {
                    hash.update(digest);
                }
            }
            hash.update((group.answers.len() as u64).to_le_bytes());
            for answer in &group.answers {
                answer.stream_bytes(&mut |bytes| hash.update(bytes));
            }
        }
        hash.finalize_digest()
    }
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct Bind<E: IsField> {
    attempt: u128,
    plan: SealedPlan<E>,
    digest: Commitment,
    challenges: Challenges<E>,
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub(super) struct GroupParts<F: IsField, E: IsField> {
    pub tables: Vec<TableProof<E>>,
    pub argue: Option<BatchedArgue<E>>,
    pub local: Option<LocalArgue<E>>,
    pub opening: StackedProof<F, E>,
    pub prepared: Option<StackedProof<F, E>>,
}

impl<F: IsField, E: IsField> GroupParts<F, E> {
    pub fn view<'a>(
        &'a self,
        group: usize,
        roots: &'a [Commitment],
        commit_ood: &'a [FieldElement<E>],
    ) -> GroupOpened<'a, F, E> {
        GroupOpened {
            group,
            roots,
            commit_ood,
            argue: self.argue.as_ref(),
            local: self.local.as_ref(),
            tables: &self.tables,
            opening: &self.opening,
            prepared: self.prepared.as_ref(),
        }
    }
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub(super) struct GroupDone<F: IsField, E: IsField> {
    plan: Commitment,
    parts: GroupParts<F, E>,
}

#[derive(Clone, Copy)]
pub(super) struct GroupShape {
    pub tables: usize,
    pub chains: usize,
    pub prepared_chains: usize,
    pub local: bool,
    pub argue: ArgueFormat,
}

impl GroupShape {
    fn check<F: IsField, E: IsField>(self, parts: &GroupParts<F, E>) -> bool {
        let per_table = self.argue == ArgueFormat::PerTable;
        parts.tables.len() == if per_table { self.tables } else { 0 }
            && parts
                .argue
                .as_ref()
                .map(|argue| (argue.factor_values.len(), argue.reduces.len()))
                == (!per_table).then_some((self.tables, self.tables))
            && parts.local.is_some() == self.local
            && parts.opening.polys.len() == self.chains
            && parts.prepared.as_ref().map(|p| p.polys.len())
                == (self.prepared_chains != 0).then_some(self.prepared_chains)
    }
}

fn protocol(reason: &'static str) -> MlError {
    MlError::ClusterProtocol(reason)
}

struct Slots<T> {
    expected: Vec<Header>,
    values: Vec<Option<T>>,
}

impl<T> Slots<T> {
    fn new(expected: &[Header]) -> Self {
        Self {
            expected: expected.to_vec(),
            values: (0..expected.len()).map(|_| None).collect(),
        }
    }

    fn insert(
        &mut self,
        reply: Reply<T>,
        check: impl FnOnce(usize, &T) -> bool,
    ) -> Result<(), MlError> {
        let g = reply.header.group;
        let expected = self
            .expected
            .get(g)
            .ok_or_else(|| protocol("group index out of range"))?;
        if reply.header != *expected {
            return Err(protocol("wrong attempt or worker"));
        }
        let slot = &mut self.values[g];
        if slot.is_some() {
            return Err(protocol("duplicate group"));
        }
        if !check(g, &reply.payload) {
            return Err(protocol(
                "group payload disagrees with its assignment or sealed plan",
            ));
        }
        *slot = Some(reply.payload);
        Ok(())
    }

    fn finish(self) -> Result<Vec<T>, MlError> {
        self.values
            .into_iter()
            .collect::<Option<_>>()
            .ok_or_else(|| protocol("missing group"))
    }
}

pub(super) struct Results<F: IsField, E: IsField> {
    headers: Vec<Header>,
    plan: Commitment,
    shapes: Vec<GroupShape>,
    workers: Workers<F, E>,
    rx: mpsc::Receiver<Reply<GroupDone<F, E>>>,
}

impl<F: IsField, E: IsField> Results<F, E> {
    pub fn sender(
        &self,
    ) -> impl Fn(usize, GroupParts<F, E>) -> Result<(), MlError> + Sync + use<F, E>
    where
        FieldElement<F>: Send,
        FieldElement<E>: Send,
    {
        let headers = self.headers.clone();
        let commands = self.workers.commands.clone();
        move |group, parts| {
            let header = headers[group];
            commands[header.worker]
                .send(Command::Opened(group, Box::new(parts)))
                .map_err(|_| protocol("worker result channel closed"))
        }
    }

    pub fn worker_failed(&self) -> Arc<AtomicBool> {
        self.workers.failed.clone()
    }

    pub fn finish(mut self) -> Result<Vec<GroupParts<F, E>>, MlError> {
        let mut completed = vec![0; self.workers.threads.len()];
        self.workers.join()?;
        let mut slots = Slots::new(&self.headers);
        for reply in self.rx {
            let worker = reply.header.worker;
            slots.insert(reply, |g, done| {
                done.plan == self.plan && self.shapes[g].check(&done.parts)
            })?;
            completed[worker] += 1;
        }
        let groups = slots.finish()?;
        eprintln!("BLOCK CLUSTER INPROC: completed groups by worker {completed:?}");
        Ok(groups.into_iter().map(|done| done.parts).collect())
    }
}

enum Command<F: IsField, E: IsField> {
    Assign(Vec<Reply<Committed<E>>>),
    Bind(Arc<Bind<E>>),
    Opened(usize, Box<GroupParts<F, E>>),
}

struct Workers<F: IsField, E: IsField> {
    commands: Vec<mpsc::Sender<Command<F, E>>>,
    threads: Vec<std::thread::JoinHandle<Result<(), MlError>>>,
    failed: Arc<AtomicBool>,
}

impl<F: IsField, E: IsField> Workers<F, E> {
    fn join(&mut self) -> Result<(), MlError> {
        // Closing the mailboxes also cancels workers waiting for Bind or work.
        self.commands.clear();
        let mut result = Ok(());
        for handle in self.threads.drain(..) {
            let joined = handle
                .join()
                .unwrap_or_else(|_| Err(protocol("worker panicked")));
            result = result.and(joined);
        }
        result
    }
}

impl<F: IsField, E: IsField> Drop for Workers<F, E> {
    fn drop(&mut self) {
        let _ = self.join();
    }
}

// Phase A has already used the streamed committer to produce this worker's
// private front. Keep it to check the coordinator's Bind before replay.
fn worker<F, E, T>(
    attempt: u128,
    id: usize,
    commands: mpsc::Receiver<Command<F, E>>,
    receipts: mpsc::Sender<Reply<Committed<E>>>,
    ready: mpsc::Sender<Reply<T>>,
    results: mpsc::Sender<Reply<GroupDone<F, E>>>,
) -> Result<(), MlError>
where
    F: IsField,
    E: IsField + 'static,
    FieldElement<E>: AsBytes,
    T: IsTranscript<E> + Default + Clone,
{
    let Ok(Command::Assign(private)) = commands.recv() else {
        return Err(protocol("worker expected a commit assignment"));
    };
    for receipt in &private {
        if receipt.header.attempt != attempt || receipt.header.worker != id {
            return Err(protocol("worker received an invalid assignment"));
        }
        receipts
            .send(Reply {
                header: receipt.header,
                payload: receipt.payload.clone(),
            })
            .map_err(|_| protocol("receipt channel closed"))?;
    }
    drop(receipts);
    let Ok(Command::Bind(bind)) = commands.recv() else {
        return Err(protocol("worker expected a sealed plan"));
    };
    if bind.attempt != attempt
        || bind.plan.digest() != bind.digest
        || private.iter().any(|r| {
            !bind
                .plan
                .groups
                .get(r.header.group)
                .is_some_and(|sealed| sealed.matches(&r.payload))
        })
    {
        return Err(protocol("worker received an invalid sealed plan"));
    }
    let mut post = T::default();
    if bind.plan.replay(&mut post) != bind.challenges {
        return Err(protocol(
            "worker challenge replay disagrees with coordinator",
        ));
    }
    let mut assigned: std::collections::BTreeMap<_, _> = private
        .into_iter()
        .map(|receipt| (receipt.header.group, receipt.header))
        .collect();
    for (&group, &header) in &assigned {
        ready
            .send(Reply {
                header,
                payload: group_fork(&post, group),
            })
            .map_err(|_| protocol("worker fork channel closed"))?;
    }
    drop(ready);
    let plan = bind.digest;
    drop(bind);
    for command in commands {
        let Command::Opened(group, parts) = command else {
            return Err(protocol("worker expected a group result"));
        };
        let header = assigned
            .remove(&group)
            .ok_or_else(|| protocol("worker cannot return this group"))?;
        results
            .send(Reply {
                header,
                payload: GroupDone {
                    plan,
                    parts: *parts,
                },
            })
            .map_err(|_| protocol("coordinator result channel closed"))?;
    }
    Ok(())
}

pub(super) struct Bound<F: IsField, E: IsField, T> {
    pub forks: Vec<T>,
    pub challenges: Challenges<E>,
    pub results: Results<F, E>,
}

pub(super) fn bind_workers<F, E, T>(
    request: InprocRequest,
    expected: Vec<Committed<E>>,
    derived: Vec<Commitment>,
    shapes: Vec<GroupShape>,
    transcript: &mut T,
) -> Result<Bound<F, E, T>, MlError>
where
    F: IsField + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: Send,
    FieldElement<E>: AsBytes + Send + Sync,
    T: IsTranscript<E> + Default + Clone + Send + 'static,
{
    // Stage 2 channels are process-local. The counter distinguishes
    // simultaneous and retried attempts within this process.
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let attempt = (u128::from(std::process::id()) << 64) | u128::from(NEXT.fetch_add(1, Relaxed));
    let workers = request.workers.0.get();
    let headers: Vec<_> = (0..expected.len())
        .map(|group| Header {
            attempt,
            worker: group % workers,
            group,
        })
        .collect();
    let (receipt_tx, receipt_rx) = mpsc::channel();
    let (ready_tx, ready_rx) = mpsc::channel();
    let (result_tx, result_rx) = mpsc::channel();
    let mut assignments: Vec<Vec<_>> = (0..workers).map(|_| Vec::new()).collect();
    for (&header, payload) in headers.iter().zip(&expected) {
        assignments[header.worker].push(Reply {
            header,
            payload: payload.clone(),
        });
    }
    let mut workers = Workers {
        commands: Vec::with_capacity(workers),
        threads: Vec::with_capacity(workers),
        failed: Arc::new(AtomicBool::new(false)),
    };
    for (id, assigned) in assignments.into_iter().enumerate() {
        let (tx, rx) = mpsc::channel();
        tx.send(Command::Assign(assigned))
            .map_err(|_| protocol("worker channel closed"))?;
        workers.commands.push(tx);
        let (receipts, ready, results) = (receipt_tx.clone(), ready_tx.clone(), result_tx.clone());
        let failed = Arc::clone(&workers.failed);
        let thread = std::thread::Builder::new()
            .name(format!("cluster-worker-{id}"))
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    worker::<F, E, T>(attempt, id, rx, receipts, ready, results)
                }))
                .unwrap_or_else(|_| Err(protocol("worker panicked")));
                if result.is_err() {
                    failed.store(true, Relaxed);
                }
                result
            })
            .map_err(|_| protocol("could not start worker"))?;
        workers.threads.push(thread);
    }
    drop(receipt_tx);
    drop(ready_tx);
    drop(result_tx);
    let mut slots = Slots::new(&headers);
    for receipt in receipt_rx {
        slots.insert(receipt, |g, payload| expected[g].matches(payload))?;
    }
    let plan = SealedPlan {
        statement: request.statement,
        groups: slots.finish()?,
        derived,
    };
    let challenges = plan.replay(transcript);
    let digest = plan.digest();
    let bind = Arc::new(Bind {
        attempt,
        plan,
        digest,
        challenges: challenges.clone(),
    });
    for tx in &workers.commands {
        tx.send(Command::Bind(Arc::clone(&bind)))
            .map_err(|_| protocol("worker channel closed"))?;
    }
    // These forks stay worker-side, with the shared in-process phase-B
    // executor. The Bind itself carries only replay inputs, never a live sponge.
    let mut forks = Slots::new(&headers);
    for fork in ready_rx {
        forks.insert(fork, |_, _| true)?;
    }
    Ok(Bound {
        forks: forks.finish()?,
        challenges,
        results: Results {
            headers,
            plan: digest,
            shapes,
            workers,
            rx: result_rx,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::{
        extensions_goldilocks::Degree3GoldilocksExtensionField as E,
        goldilocks::GoldilocksField as F,
    };
    use multilinear::whir_hash::{Blake3Whir, WhirHash};

    type T = <Blake3Whir as WhirHash>::Sponge;

    fn receipt(group: usize) -> Committed<E> {
        Committed {
            roots: vec![[group as u8; 32]],
            // More answers than roots, as with a joint tree.
            answers: vec![
                FieldElement::from(10 + group as u64),
                FieldElement::from(20 + group as u64),
            ],
            prepared_roots: vec![[100 + group as u8; 32]],
            local_digests: vec![[200 + group as u8; 32]],
        }
    }

    fn headers() -> Vec<Header> {
        (0..4)
            .map(|group| Header {
                attempt: 17,
                worker: group % 2,
                group,
            })
            .collect()
    }

    fn committed(header: Header) -> Reply<Committed<E>> {
        Reply {
            header,
            payload: receipt(header.group),
        }
    }

    fn admit(slots: &mut Slots<Committed<E>>, reply: Reply<Committed<E>>) -> Result<(), MlError> {
        slots.insert(reply, |g, payload| receipt(g).matches(payload))
    }

    #[test]
    fn receipts_admit_in_any_order_and_require_every_slot() {
        let headers = headers();
        let mut slots = Slots::new(&headers);
        for g in [2, 0, 3, 1] {
            admit(&mut slots, committed(headers[g])).unwrap();
        }
        let groups = slots.finish().unwrap();
        assert!(
            groups
                .iter()
                .enumerate()
                .all(|(g, payload)| receipt(g).matches(payload))
        );

        let mut missing = Slots::new(&headers);
        admit(&mut missing, committed(headers[3])).unwrap();
        assert!(matches!(
            missing.finish(),
            Err(MlError::ClusterProtocol("missing group"))
        ));
    }

    #[test]
    fn receipts_refuse_invalid_headers_and_swapped_payloads() {
        let headers = headers();
        let mut slots = Slots::new(&headers);
        let h = headers[1];
        for (bad, error) in [
            (Header { attempt: 18, ..h }, "wrong attempt or worker"),
            (Header { worker: 0, ..h }, "wrong attempt or worker"),
            (Header { group: 4, ..h }, "group index out of range"),
        ] {
            assert_eq!(admit(&mut slots, committed(bad)), Err(protocol(error)));
        }
        let mismatch = protocol("group payload disagrees with its assignment or sealed plan");
        let mut swapped = committed(h);
        swapped.payload.roots = receipt(3).roots;
        assert_eq!(admit(&mut slots, swapped), Err(mismatch));

        let mut swapped = committed(h);
        swapped.payload.answers = receipt(3).answers;
        assert_eq!(
            admit(&mut slots, swapped),
            Err(protocol(
                "group payload disagrees with its assignment or sealed plan"
            ))
        );

        admit(&mut slots, committed(h)).unwrap();
        assert_eq!(
            admit(&mut slots, committed(h)),
            Err(protocol("duplicate group"))
        );
    }

    fn shape() -> GroupShape {
        GroupShape {
            tables: 0,
            chains: 0,
            prepared_chains: 0,
            local: false,
            argue: ArgueFormat::PerTable,
        }
    }

    fn parts() -> GroupParts<F, E> {
        GroupParts {
            tables: Vec::new(),
            argue: None,
            local: None,
            opening: StackedProof { polys: Vec::new() },
            prepared: None,
        }
    }

    fn start(workers: usize, transcript: &mut T) -> Bound<F, E, T> {
        let request = InprocRequest {
            workers: InprocWorkers(NonZeroUsize::new(workers).unwrap()),
            statement: vec![vec![0x31, 0x32], vec![0x41, 0x42, 0x43]],
        };
        bind_workers(
            request,
            (0..4).map(receipt).collect(),
            vec![[103; 32], [100; 32], [102; 32], [101; 32]],
            vec![shape(); 4],
            transcript,
        )
        .unwrap()
    }

    #[test]
    fn workers_replay_statement_roots_derived_and_answers_before_forking() {
        for workers in [1, 2, 4] {
            let mut coordinator = T::default();
            let bound = start(workers, &mut coordinator);
            let mut expected = T::default();
            expected.append_bytes(&[0x31, 0x32]);
            expected.append_bytes(&[0x41, 0x42, 0x43]);
            for root in [0, 1, 2, 3, 103, 100, 102, 101] {
                expected.append_bytes(&[root; 32]);
            }
            for answer in [10, 20, 11, 21, 12, 22, 13, 23] {
                expected.append_field_element(&FieldElement::from(answer as u64));
            }
            assert_eq!(
                bound.challenges,
                (
                    expected.sample_field_element(),
                    expected.sample_field_element(),
                    expected.sample_field_element(),
                )
            );
            assert_eq!(
                coordinator.sample_field_element(),
                expected.clone().sample_field_element()
            );
            for (g, mut worker_fork) in bound.forks.into_iter().enumerate() {
                let mut fork = expected.clone();
                fork.append_bytes(&(g as u64).to_le_bytes());
                for _ in 0..3 {
                    assert_eq!(
                        worker_fork.sample_field_element(),
                        fork.sample_field_element()
                    );
                }
            }
            let send = bound.results.sender();
            for g in [3, 1, 0, 2] {
                send(g, parts()).unwrap();
            }
            drop(send);
            assert_eq!(bound.results.finish().unwrap().len(), 4);
        }
    }

    #[test]
    fn a_worker_error_and_missing_results_cancel_the_exchange() {
        let bound = start(2, &mut T::default());
        let send = bound.results.sender();
        send(2, parts()).unwrap();
        // Send the same owned result twice to the worker. It already returned
        // that assignment, so it errors while the other worker is still live.
        let _ = send(2, parts());
        drop(send);
        assert!(matches!(
            bound.results.finish(),
            Err(MlError::ClusterProtocol("worker cannot return this group"))
        ));

        let bound = start(4, &mut T::default());
        assert!(matches!(
            bound.results.finish(),
            Err(MlError::ClusterProtocol("missing group"))
        ));
        let bound = start(1, &mut T::default());
        let send = bound.results.sender();
        for g in 0..4 {
            let mut result = parts();
            if g == 2 {
                result.local = Some(LocalArgue::default());
            }
            send(g, result).unwrap();
        }
        drop(send);
        assert!(matches!(
            bound.results.finish(),
            Err(MlError::ClusterProtocol(
                "group payload disagrees with its assignment or sealed plan"
            ))
        ));
    }

    #[test]
    fn messages_round_trip_through_checked_rkyv() {
        use rkyv::{from_bytes, rancor::Error, to_bytes};
        let h = headers()[2];
        let msg = committed(h);
        let bytes = to_bytes::<Error>(&msg).unwrap();
        let decoded = from_bytes::<Reply<Committed<E>>, Error>(&bytes).unwrap();
        assert_eq!(decoded.header, msg.header);
        assert!(decoded.payload.matches(&msg.payload));
        assert!(from_bytes::<Reply<Committed<E>>, Error>(&bytes[..1]).is_err());

        let plan = SealedPlan {
            statement: vec![vec![42]],
            groups: vec![receipt(2)],
            derived: vec![[102; 32]],
        };
        let msg = Bind {
            attempt: h.attempt,
            digest: plan.digest(),
            challenges: plan.replay(&mut T::default()),
            plan,
        };
        let bytes = to_bytes::<Error>(&msg).unwrap();
        let decoded = from_bytes::<Bind<E>, Error>(&bytes).unwrap();
        assert_eq!(decoded.digest, decoded.plan.digest());
        assert_eq!(decoded.challenges, decoded.plan.replay(&mut T::default()));

        let msg = Reply {
            header: h,
            payload: GroupDone {
                plan: decoded.digest,
                parts: parts(),
            },
        };
        let bytes = to_bytes::<Error>(&msg).unwrap();
        let decoded = from_bytes::<Reply<GroupDone<F, E>>, Error>(&bytes).unwrap();
        assert_eq!(decoded.header, h);
        assert_eq!(decoded.payload.plan, msg.payload.plan);
        assert!(shape().check(&decoded.payload.parts));
    }
}
