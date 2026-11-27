//! One TCP connection per worker per attempt.
//! Deploy the same executable and proof settings to every host on a private
//! network. Topology tokens authenticate peers and frames. The link carries
//! private input in plaintext.

use super::stages::Clock;
use super::wire::{self, Attempt, Connection, Digest32, Kind, Reader};
use super::{super::*, HostAirs, Owned, Pass, Plan, range};
use crate::lfm::{
    cluster_levels::{self, LeafMap},
    last_level::LastLevel,
    levels::level_rounds,
    merkle_cap::CapMux,
    whir_block::{group_arena_words, leaf_arena_under},
    wlfm::{LeafArtifacts, WlfmFamily, WlfmProof},
    wlfm_leg::WlfmChild,
};
use crypto::hash::blake3::chain::Blake3Std;
use digest::Digest;
use rkyv::util::AlignedVec;
use stark::multilinear_block::replica::{self, Committed, GroupParts, GroupShape, SealedPlan};
use std::{
    cell::RefCell,
    fs::File,
    io::{self, Read},
    net::{SocketAddr, TcpListener, TcpStream},
    num::NonZeroUsize,
    path::Path,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    time::Duration,
};

type WireResult<T> = io::Result<T>;
type Receipt = Committed<E>;
type Parts = GroupParts<F, E>;

const ELF_LIMIT: usize = 64 << 20;
const INPUT_LIMIT: usize = 128 << 20;
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(1);
const FINISH_TIMEOUT: Duration = Duration::from_secs(1);
const WATCHDOG_GRACE: Duration = Duration::from_secs(1);
/// How long a repeated worker waits for the coordinator's next attempt unless
/// the topology says otherwise (`next-attempt-ms`). Between two attempts the
/// `cluster prove` runner proves the join, publishes the file and checks it;
/// after a refusal it also proves the block locally and checks that proof
/// before it starts the next run. The check alone took 47 s on a median block
/// and 90 s on a large one, so the worker's startup limit (5 s by default) is
/// far too short, and a refusal's local proof and second check on a large
/// block may need more than this default.
const NEXT_ATTEMPT_WAIT: Duration = Duration::from_secs(600);

#[derive(Clone)]
struct Endpoint {
    address: SocketAddr,
    ordinals: Vec<u32>,
    token: Option<Digest32>,
}

/// Text configuration. Each worker line is in placement order. Use a unique
/// 32-byte hex token per endpoint, and CUDA ordinals as visible to that process.
///
/// ```text
/// coordinator 127.0.0.1:48000
/// worker 127.0.0.1:48001 0 <64 hex digits>
/// worker 127.0.0.1:48002 0,1 <64 hex digits>
/// windows 16,32
/// deadlines-ms 5000 30000 30000
/// next-attempt-ms 600000
/// fallback local
/// ```
///
/// The coordinator dials the workers. Its address identifies it in the
/// handshake. Worker copies may replace every other token with `-`.
/// The optional `deadlines-ms` line sets startup, commit and opening limits
/// (defaults: 5 s, 30 s and 30 s). Set larger limits for larger workloads.
/// A worker serving several attempts (`worker --attempts N`) waits startup
/// for the first and the optional `next-attempt-ms` limit (default 600 s)
/// for each later one, from the end of the attempt before: long enough for
/// the coordinator to prove the join and check its file in between (after a
/// refusal, also a local proof and its check; raise the limit for large
/// blocks). Once a later attempt connects, its Start and inputs get the
/// startup limit again.
/// Workers allow at most one second per handshake within startup. The final
/// acknowledgement is best effort, with at most one second to collect it.
/// The optional `fallback local` line retries failures during cluster
/// proving. It defaults to off.
/// `windows` gives one exclusive end window per worker. The coordinator
/// owns the open-ended final range and the halting window. With the line
/// omitted the topology uses the replicated witness protocol.
/// Range builds hold their own witnesses through the commitment.
/// Use a private network or a tunnel if the input needs confidentiality.
///
/// The `cluster prove` runner publishes each file before checking it, so a
/// published file may not have been checked yet. Consumers must compare the
/// file's statement with the one they expect and verify the file themselves
/// (`verify_cluster_block_levels_file` for a range file,
/// `verify_block_levels_file` for a replica file). After a refusal, the
/// runner adds `run k: <reason>` to `OUTPUT.refused` before anything else,
/// then makes a fresh local proof regardless of the `fallback` setting; a
/// checked local proof replaces the run's file once, and any failure on the
/// way ends the process with the refused file still at OUTPUT. The marker
/// carries no trust either way:
/// - no entry does not mean the file passed: a kill or crash during the
///   check leaves an unchecked file and no entry;
/// - an entry stays after the local replacement and after later runs
///   overwrite OUTPUT, so it can describe a file no longer there;
/// - entries are keyed by run number, which restarts at 1 with each
///   invocation: a later invocation's accepted run k clears an earlier
///   invocation's entry for run k, and entries above its own `--runs` stay.
#[derive(Clone)]
pub struct Topology {
    coordinator: SocketAddr,
    endpoints: Vec<Endpoint>,
    cuts: Option<Vec<usize>>,
    startup: Duration,
    commit: Duration,
    open: Duration,
    next_attempt: Duration,
    fallback: bool,
}

impl Topology {
    pub fn read(path: &Path) -> WireResult<Self> {
        let mut bytes = String::new();
        File::open(path)?
            .take((16 << 10) + 1)
            .read_to_string(&mut bytes)?;
        Self::parse(&bytes)
    }

    pub fn parse(text: &str) -> WireResult<Self> {
        if text.len() > 16 << 10 {
            return Err(wire::invalid("topology exceeds 16 KiB"));
        }
        let mut coordinator = None;
        let mut endpoints: Vec<Endpoint> = Vec::new();
        let mut deadlines = None;
        let mut next_attempt = None;
        let mut cuts = None;
        let mut fallback = false;
        for (i, line) in text.lines().enumerate() {
            let fields: Vec<_> = line
                .split('#')
                .next()
                .unwrap_or_default()
                .split_whitespace()
                .collect();
            let bad = || wire::invalid(format!("invalid topology at line {}", i + 1));
            match fields.as_slice() {
                [] => {}
                ["coordinator", addr] if coordinator.is_none() => {
                    coordinator = Some(addr.parse::<SocketAddr>().map_err(|_| bad())?);
                }
                ["worker", addr, cards, key] if endpoints.len() < wire::MAX_WORKERS => {
                    let address = addr.parse::<SocketAddr>().map_err(|_| bad())?;
                    let ordinals: Vec<u32> = cards
                        .split(',')
                        .map(str::parse)
                        .collect::<Result<_, _>>()
                        .map_err(|_| bad())?;
                    if ordinals.len() > wire::MAX_WORKERS || address.port() == 0 {
                        return Err(bad());
                    }
                    let token = if *key == "-" {
                        None
                    } else {
                        if key.len() != 64 || !key.is_ascii() {
                            return Err(bad());
                        }
                        let mut token = [0; 32];
                        for (i, byte) in token.iter_mut().enumerate() {
                            *byte = u8::from_str_radix(&key[2 * i..2 * i + 2], 16)
                                .map_err(|_| bad())?;
                        }
                        Some(token)
                    };
                    if endpoints
                        .iter()
                        .any(|e| e.address == address || (token.is_some() && e.token == token))
                    {
                        return Err(bad());
                    }
                    endpoints.push(Endpoint {
                        address,
                        ordinals,
                        token,
                    });
                }
                ["deadlines-ms", startup, commit, open] if deadlines.is_none() => {
                    let millis = |s: &str| {
                        s.parse::<u64>()
                            .ok()
                            .filter(|ms| (1..=3_600_000).contains(ms))
                            .map(Duration::from_millis)
                            .ok_or_else(bad)
                    };
                    deadlines = Some((millis(startup)?, millis(commit)?, millis(open)?));
                }
                ["next-attempt-ms", wait] if next_attempt.is_none() => {
                    next_attempt = Some(
                        wait.parse::<u64>()
                            .ok()
                            .filter(|ms| (1..=3_600_000).contains(ms))
                            .map(Duration::from_millis)
                            .ok_or_else(bad)?,
                    );
                }
                ["windows", word] if cuts.is_none() => {
                    cuts = Some(range::parse_cuts(word).map_err(|_| bad())?);
                }
                ["fallback", "local"] if !fallback => fallback = true,
                _ => return Err(bad()),
            }
        }
        let coordinator = coordinator.ok_or_else(|| wire::invalid("missing coordinator"))?;
        if endpoints.is_empty() || endpoints.iter().any(|e| e.address == coordinator) {
            return Err(wire::invalid(
                "topology needs distinct coordinator and worker addresses",
            ));
        }
        if cuts
            .as_ref()
            .is_some_and(|cuts| cuts.len() != endpoints.len())
        {
            return Err(wire::invalid(
                "a range topology needs one window boundary per worker",
            ));
        }
        let (startup, commit, open) = deadlines.unwrap_or((
            Duration::from_secs(5),
            Duration::from_secs(30),
            Duration::from_secs(30),
        ));
        Ok(Self {
            coordinator,
            endpoints,
            cuts,
            startup,
            commit,
            open,
            next_attempt: next_attempt.unwrap_or(NEXT_ATTEMPT_WAIT),
            fallback,
        })
    }

    pub fn range_cuts(&self) -> Option<&[usize]> {
        self.cuts.as_deref()
    }

    pub(crate) fn local_fallback(&self) -> bool {
        self.fallback
    }

    pub fn workers(&self) -> usize {
        self.endpoints.len()
    }

    /// Pass this as LAMBDA_VM_CARDS when starting the worker, before any CUDA
    /// or Rayon initialization.
    pub fn worker_cards(&self, id: usize) -> WireResult<String> {
        let endpoint = self
            .endpoints
            .get(id)
            .ok_or_else(|| wire::invalid("unknown worker index"))?;
        Ok(endpoint
            .ordinals
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(","))
    }

    fn context(&self, id: usize, profile: &Digest32) -> WireResult<Digest32> {
        let mut digest = Blake3Std::new();
        digest.update(build_id()?);
        digest.update(profile);
        digest.update(self.coordinator.to_string());
        digest.update((id as u64).to_le_bytes());
        digest.update(self.endpoints[id].address.to_string());
        digest.update(self.worker_cards(id)?);
        Ok(digest.finalize().into())
    }
}

fn build_id() -> WireResult<Digest32> {
    static BUILD: OnceLock<Digest32> = OnceLock::new();
    if let Some(id) = BUILD.get() {
        return Ok(*id);
    }
    let mut file = File::open(std::env::current_exe()?)?;
    let mut digest = Blake3Std::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let size = file.read(&mut buffer)?;
        if size == 0 {
            break;
        }
        digest.update(&buffer[..size]);
    }
    Ok(*BUILD.get_or_init(|| digest.finalize().into()))
}

fn profile(opts: &ProofOptions, format: &BlockFormat) -> Digest32 {
    let mut digest = Blake3Std::new();
    digest.update(format!("{opts:?}\n{format:?}"));
    digest.update([u8::from(
        std::env::var("LAMBDA_VM_DETERMINISTIC_GRIND").as_deref() == Ok("1"),
    )]);
    digest.finalize().into()
}

// These settings configure the replicated witness build. Card placement and
// its memory budget are local to each host.
pub(super) const BUILD_SETTINGS: &[&str] = &[
    LOCAL_LANE_ENV,
    multilinear_block::CS_PROBE_ENV,
    multilinear_block::MERGE_PROBE_ENV,
    "LAMBDA_VM_INROW_SIDES_PINNED",
    "LAMBDA_VM_WORD_STREAM",
    "LAMBDA_VM_WORD_CHUNKS",
    FINISH_LATE_FIRST_ENV,
    "LAMBDA_VM_HOST_HOIST",
    "LAMBDA_VM_WALK_CUTS",
    "LAMBDA_VM_FINISH_EARLY",
    "LAMBDA_VM_LIGHT_WALK",
    "LAMBDA_VM_BLOCK_REGEN",
    "LAMBDA_VM_BLOCK_WALKERS",
    PAGES_FIRST_ENV,
    REST_AIRS_BESIDE_ENV,
    CLOSE_BEFORE_BITWISE_ENV,
    "LAMBDA_VM_BLOCK_CARD_FILL",
    STREAM_KECCAK_RND_ENV,
    KECCAK_RND_OWN_GROUPS_ENV,
    EXEC_STREAM_ENV,
    executor::vm::engine::ENGINE_KNOB,
    "LAMBDA_VM_EXEC_HISTORY",
    "LAMBDA_VM_EXEC_HISTORY_CHECK",
    "LAMBDA_VM_EXEC_SHADOW",
    "LAMBDA_VM_FINISH_PAGE_DIRECT",
    "LAMBDA_VM_FINISH_BITWISE_DIRECT",
    "LAMBDA_VM_FINISH_KECCAK_W_SPLIT",
];

thread_local! {
    static SETTINGS: RefCell<Option<Arc<[Option<String>]>>> = const { RefCell::new(None) };
}

pub(crate) fn setting(name: &str) -> Option<String> {
    SETTINGS.with(|slot| {
        if let Some(settings) = slot.borrow().as_ref()
            && let Some(index) = BUILD_SETTINGS.iter().position(|&key| key == name)
        {
            return settings[index].clone();
        }
        std::env::var(name).ok()
    })
}

pub(crate) fn witness_settings() -> Option<Arc<[Option<String>]>> {
    SETTINGS.with(|slot| slot.borrow().clone())
}

pub(crate) fn with_settings<T>(
    settings: Option<Arc<[Option<String>]>>,
    run: impl FnOnce() -> T,
) -> T {
    let Some(settings) = settings else {
        return run();
    };
    struct Reset(Option<Arc<[Option<String>]>>);
    impl Drop for Reset {
        fn drop(&mut self) {
            SETTINGS.with(|slot| slot.replace(self.0.take()));
        }
    }
    let _reset = Reset(SETTINGS.with(|slot| slot.replace(Some(settings))));
    run()
}

#[derive(Clone)]
enum Placement {
    Local,
    Tcp {
        topology: Arc<Topology>,
        failed: bool,
    },
}

thread_local! {
    static DISPATCH: RefCell<Option<Placement>> = const { RefCell::new(None) };
}

type Dispatch = (
    Option<Arc<Topology>>,
    Option<multilinear_block::InprocWorkers>,
);

pub(in crate::block_whir) fn dispatch() -> Result<Dispatch, Error> {
    let topology = match DISPATCH.with(|slot| slot.borrow().clone()) {
        Some(Placement::Local) => return Ok((None, None)),
        Some(Placement::Tcp { topology, .. }) => Some(topology),
        None => None,
    };
    let inproc = multilinear_block::InprocWorkers::from_env().map_err(Error::Prover)?;
    if topology.is_some() && inproc.is_some() {
        return Err(Error::Prover(
            "select either TCP topology or LAMBDA_VM_CLUSTER=inproc:N".into(),
        ));
    }
    Ok((topology, inproc))
}

/// The one-leaf driver only handles replica topologies. Tests also retain
/// the stage-A range archive path.
#[cfg(not(test))]
pub(crate) fn range_coordinator_selected() -> bool {
    DISPATCH.with(|slot| {
        matches!(&*slot.borrow(), Some(Placement::Tcp { topology, .. }) if topology.cuts.is_some())
    })
}

static LOCAL_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// How many times this process has retried a cluster proving failure locally.
pub fn local_fallbacks() -> u64 {
    LOCAL_FALLBACKS.load(Ordering::Relaxed)
}

/// Counts one more fallback to a local proof; returns the new count.
pub(crate) fn count_local_fallback() -> u64 {
    LOCAL_FALLBACKS.fetch_add(1, Ordering::Relaxed) + 1
}

fn with_placement<T>(placement: Placement, run: impl FnOnce() -> T) -> T {
    struct Reset(Option<Placement>);
    impl Drop for Reset {
        fn drop(&mut self) {
            DISPATCH.with(|slot| slot.replace(self.0.take()));
        }
    }
    let _reset = Reset(DISPATCH.with(|slot| slot.replace(Some(placement))));
    run()
}

pub(crate) fn locally<T>(prove: impl FnOnce() -> T) -> T {
    with_placement(Placement::Local, prove)
}

/// Run a complete proving entry point on this thread. On a configured
/// fallback the first invocation has returned and its observers and sockets
/// have been dropped before a fresh invocation starts with local placement.
pub fn coordinate<T, E: std::fmt::Display>(
    topology: &Topology,
    prove: impl Fn() -> Result<T, E>,
) -> Result<T, E> {
    with_placement(
        Placement::Tcp {
            topology: Arc::new(topology.clone()),
            failed: false,
        },
        || {
            let result = prove();
            let cluster_failed = DISPATCH.with(|slot| {
                matches!(
                    slot.borrow().as_ref(),
                    Some(Placement::Tcp { failed: true, .. })
                )
            });
            if let Err(error) = &result
                && topology.fallback
                && cluster_failed
            {
                eprintln!("CLUSTER TCP: {error}");
                let count = count_local_fallback();
                eprintln!("CLUSTER TCP: starting a fresh local attempt (fallback {count})");
                return locally(prove);
            }
            result
        },
    )
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct Start {
    range_levels: bool,
    cuts: Option<Vec<usize>>,
    trace_seed: [u8; 16],
    options: BlockOptions,
    settings: Vec<Option<String>>,
    elf_len: usize,
    input_len: usize,
    elf_hash: Digest32,
    input_hash: Digest32,
}

// These are local ceilings for the streamed production posture. In
// particular, peer-supplied sizes never authorize a larger worker allocation.
pub(in crate::block_whir) fn validate_options(options: &BlockOptions) -> WireResult<usize> {
    let window = options
        .window_log2
        .filter(|&log| log <= BLOCK_WINDOW_LOG2)
        .ok_or_else(|| wire::invalid("Start window_log2 exceeds worker limits"))?;
    let rows = &options.max_rows;
    if [
        rows.cpu,
        rows.memw,
        rows.memw_aligned,
        rows.dvrm,
        rows.mul,
        rows.lt,
        rows.shift,
        rows.load,
        rows.branch,
        rows.memw_register,
        rows.eq,
        rows.bytewise,
        rows.store,
        rows.cpu32,
        rows.mem_load,
        rows.mem_store,
    ]
    .into_iter()
    .any(|rows| !(1..=BLOCK_MAX_ROWS).contains(&rows))
    {
        return Err(wire::invalid("Start max_rows exceeds worker limits"));
    }
    if [
        (options.keccak_rnd_rows_log2, BLOCK_KECCAK_RND_ROWS_LOG2),
        (options.ecdas_rows_log2, BLOCK_ECDAS_ROWS_LOG2),
        (options.keccak_rows_log2, BLOCK_KECCAK_ROWS_LOG2),
        (options.ecsm_rows_log2, BLOCK_ECSM_ROWS_LOG2),
        (options.copy_rows_log2, BLOCK_COPY_ROWS_LOG2),
    ]
    .into_iter()
    .any(|(log, max)| log > max)
    {
        return Err(wire::invalid("Start table cut exceeds worker limits"));
    }
    if options.layout_workers > BLOCK_LAYOUT_WORKERS
        || options
            .layout_ahead
            .is_some_and(|ahead| ahead > BLOCK_LAYOUT_AHEAD)
        || options.layout_workers > 0
            && (options.layout_ahead.is_none() || options.rest_layout_bytes.is_none())
        || options
            .rest_layout_bytes
            .is_some_and(|bytes| !(1..=BLOCK_REST_LAYOUT_BYTES).contains(&bytes))
        || options
            .rest_build_bytes
            .is_some_and(|bytes| !(1..=BLOCK_REST_BUILD_BYTES).contains(&bytes))
        || options.drop_levels.device > 32
        || options.drop_levels.host > 32
    {
        return Err(wire::invalid(
            "Start layout or tree options exceed worker limits",
        ));
    }
    Ok(1 << window)
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct CommitReply {
    group: usize,
    descriptor: Digest32,
    receipt: Receipt,
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct Bind {
    leaf_statement: Option<OwnedBlockStatement>,
    plan: SealedPlan<E>,
    digest: Digest32,
    challenges: replica::Challenges<E>,
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct GroupReply {
    group: usize,
    descriptor: Digest32,
    plan: Digest32,
    parts: Parts,
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct LeafReply {
    plan: Digest32,
    artifacts: LeafArtifacts,
    proof: WlfmProof,
}

pub(in crate::block_whir) fn mark_failed() {
    DISPATCH.with(|slot| {
        if let Some(Placement::Tcp { failed, .. }) = slot.borrow_mut().as_mut() {
            *failed = true;
        }
    });
}

fn fail(error: impl std::fmt::Display) -> Error {
    mark_failed();
    Error::Prover(format!("cluster TCP: {error}"))
}

fn hash(bytes: &[u8]) -> Digest32 {
    Blake3Std::digest(bytes).into()
}

fn descriptor(plan: &Plan, index: usize) -> WireResult<Digest32> {
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&plan.groups[index])
        .map_err(|e| wire::invalid(e.to_string()))?;
    Ok(hash(&bytes))
}

fn read_input(
    reader: &mut Reader<'_>,
    deadline: Instant,
    attempt: Attempt,
    len: usize,
) -> WireResult<Vec<u8>> {
    let mut input = Vec::with_capacity(len);
    while input.len() < len {
        let (_, _, bytes) = reader.receive(deadline, Some(attempt), &[Kind::Input])?;
        let chunk: Vec<u8> = wire::decode(&bytes)?;
        if chunk.len() != (len - input.len()).min(wire::INPUT_CHUNK) {
            return Err(wire::invalid("invalid input chunk length"));
        }
        input.extend(chunk);
    }
    Ok(input)
}

fn send_input(
    socket: &Connection,
    deadline: Instant,
    attempt: Attempt,
    input: &[u8],
) -> WireResult<()> {
    for chunk in input.chunks(wire::INPUT_CHUNK) {
        socket.send(deadline, attempt, Kind::Input, &chunk.to_vec())?;
    }
    Ok(())
}

struct CloseOnDrop<'a>(&'a [Connection]);

impl Drop for CloseOnDrop<'_> {
    fn drop(&mut self) {
        for peer in self.0 {
            peer.close();
        }
    }
}

fn connect(address: SocketAddr, deadline: Instant) -> WireResult<TcpStream> {
    loop {
        match TcpStream::connect_timeout(&address, wire::remaining(deadline)?) {
            Ok(stream) => return Ok(stream),
            Err(e) if e.kind() == io::ErrorKind::ConnectionRefused => {
                std::thread::sleep(wire::remaining(deadline)?.min(Duration::from_millis(20)));
            }
            Err(e) => return Err(e),
        }
    }
}

fn start_peers(
    topology: &Topology,
    attempt: Attempt,
    elf: &[u8],
    input: &[u8],
    profile: Digest32,
    start: &Start,
    clock: Clock,
) -> WireResult<Vec<Connection>> {
    if elf.is_empty() || elf.len() > ELF_LIMIT || input.len() > INPUT_LIMIT {
        return Err(wire::invalid("ELF or private input exceeds cluster limits"));
    }
    if topology
        .endpoints
        .iter()
        .any(|endpoint| endpoint.token.is_none())
    {
        return Err(wire::invalid(
            "coordinator topology needs every worker token",
        ));
    }
    build_id()?;
    clock.mark("input-checked");
    let deadline = Instant::now() + topology.startup;
    std::thread::scope(|scope| {
        let workers: Vec<_> = topology
            .endpoints
            .iter()
            .enumerate()
            .map(|(id, endpoint)| {
                scope.spawn(move || {
                    let run = || {
                        clock.mark(format_args!("wait-worker-connect-W{id}"));
                        let socket = wire::authenticate(
                            connect(endpoint.address, deadline)?,
                            deadline,
                            endpoint.token.as_ref().unwrap(),
                            &topology.context(id, &profile)?,
                            true,
                        )?;
                        clock.mark(format_args!("worker-connected-W{id}"));
                        socket.send(deadline, attempt, Kind::Start, start)?;
                        clock.mark(format_args!("start-sent-W{id}"));
                        send_input(&socket, deadline, attempt, elf)?;
                        send_input(&socket, deadline, attempt, input)?;
                        clock.mark(format_args!("input-sent-W{id}"));
                        Ok(socket)
                    };
                    run().map_err(|why: io::Error| {
                        wire::invalid(format!("worker {id} startup: {why}"))
                    })
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().expect("startup thread"))
            .collect()
    })
}

pub(in crate::block_whir) struct Coordinator<'a> {
    topology: &'a Topology,
    attempt: Attempt,
    start: Start,
    peers: Vec<Connection>,
    commit_deadline: Instant,
    clock: Clock,
}

impl<'a> Coordinator<'a> {
    pub(in crate::block_whir) fn start(
        topology: &'a Topology,
        elf: &[u8],
        input: &[u8],
        opts: &ProofOptions,
        format: &BlockFormat,
        options: &BlockOptions,
    ) -> Result<Self, Error> {
        Self::start_as(topology, elf, input, opts, format, options, false)
    }

    fn start_as(
        topology: &'a Topology,
        elf: &[u8],
        input: &[u8],
        opts: &ProofOptions,
        format: &BlockFormat,
        options: &BlockOptions,
        range_levels: bool,
    ) -> Result<Self, Error> {
        let attempt = wire::random().map_err(fail)?;
        let clock = if topology.cuts.is_some() {
            let clock = Clock::new(Instant::now(), attempt, None);
            clock.mark("attempt-start");
            clock
        } else {
            Clock::default()
        };
        let trace_seed =
            if std::env::var("LAMBDA_VM_FIXED_TRACE_HASH").is_ok_and(|value| value.trim() == "1") {
                [0; 16]
            } else {
                let mut seed = wire::random().map_err(fail)?;
                while seed == [0; 16] {
                    seed = wire::random().map_err(fail)?;
                }
                seed
            };
        let mut settings = BUILD_SETTINGS
            .iter()
            .map(|name| std::env::var(name).ok())
            .collect::<Vec<_>>();
        let walkers = BUILD_SETTINGS
            .iter()
            .position(|&name| name == "LAMBDA_VM_BLOCK_WALKERS")
            .unwrap();
        settings[walkers] = Some(block_walkers()?.to_string());
        if let Some(cuts) = &topology.cuts {
            range::validate_cuts(
                cuts,
                validate_options(options).map_err(fail)?,
                options.max_rows.cpu,
            )?;
        }
        let start = Start {
            range_levels,
            cuts: topology.cuts.clone(),
            trace_seed,
            options: options.clone(),
            settings,
            elf_len: elf.len(),
            input_len: input.len(),
            elf_hash: hash(elf),
            input_hash: hash(input),
        };
        let peers = start_peers(
            topology,
            attempt,
            elf,
            input,
            profile(opts, format),
            &start,
            clock,
        )
        .map_err(fail)?;
        Ok(Self {
            topology,
            attempt,
            start,
            peers,
            commit_deadline: Instant::now() + topology.commit,
            clock,
        })
    }

    pub(in crate::block_whir) fn workers(&self) -> usize {
        self.peers.len()
    }

    pub(in crate::block_whir) fn trace_seed(&self) -> [u8; 16] {
        self.start.trace_seed
    }

    pub(in crate::block_whir) fn cuts(&self) -> Option<&[usize]> {
        self.start.cuts.as_deref()
    }

    pub(in crate::block_whir) fn settings(&self) -> Arc<[Option<String>]> {
        self.start.settings.clone().into()
    }
}

struct Phase<'a> {
    name: &'static str,
    reply: Kind,
    end: Kind,
    duration: Duration,
    peers: &'a [Connection],
    attempt: Attempt,
}

impl Phase<'_> {
    /// A rendezvous queue leaves at most one validated-length frame with each
    /// reader and one with the consumer. A slow peer cannot hide a disconnect
    /// on a different connection or reset the phase's absolute deadline.
    fn run(
        &self,
        request: impl Fn(usize, &Connection, Instant) -> WireResult<()> + Sync,
        mut admit: impl FnMut(usize, Kind, &AlignedVec) -> WireResult<()>,
    ) -> WireResult<()> {
        enum Event {
            Frame(Kind, AlignedVec),
            Done,
            Failed(io::Error),
        }
        let deadline = Instant::now() + self.duration;
        std::thread::scope(|scope| {
            let (tx, rx) = mpsc::sync_channel(0);
            let mut close = CloseOnDrop(self.peers);
            let request = &request;
            for (id, peer) in self.peers.iter().enumerate() {
                let tx = tx.clone();
                scope.spawn(move || {
                    let run = || {
                        request(id, peer, deadline)?;
                        let mut read = peer.reader();
                        loop {
                            let kinds = [self.reply, self.end, Kind::Witness];
                            let kinds = if self.reply == Kind::Committed {
                                &kinds[..]
                            } else {
                                &kinds[..2]
                            };
                            let (_, kind, bytes) =
                                read.receive(deadline, Some(self.attempt), kinds)?;
                            if kind == self.end {
                                return Ok(());
                            }
                            if tx.send((id, Event::Frame(kind, bytes))).is_err() {
                                return Ok(());
                            }
                        }
                    };
                    let event = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
                        Ok(Ok(())) => Event::Done,
                        Ok(Err(error)) => Event::Failed(error),
                        Err(_) => Event::Failed(wire::invalid("peer I/O thread panicked")),
                    };
                    let _ = tx.send((id, event));
                });
            }
            drop(tx);
            let result = (|| {
                let mut remaining = self.peers.len();
                while remaining > 0 {
                    let (id, event) = rx.recv_timeout(wire::remaining(deadline)?).map_err(|e| {
                        wire::invalid(format!(
                            "{} deadline or disconnected readers: {e}",
                            self.name
                        ))
                    })?;
                    match event {
                        Event::Frame(kind, bytes) => admit(id, kind, &bytes).map_err(|e| {
                            wire::invalid(format!("worker {id} {}: {e}", self.name))
                        })?,
                        Event::Done => remaining -= 1,
                        Event::Failed(e) => {
                            return Err(wire::invalid(format!("worker {id} {}: {e}", self.name)));
                        }
                    }
                }
                Ok(())
            })();
            if result.is_ok() {
                close.0 = &[];
            }
            drop(rx);
            result
        })
    }
}

enum Slot {
    Assigned,
    Committed(Receipt),
    Opened(Parts),
}

type RangeLevels = (OwnedBlockStatement, Option<LeafMap>, WlfmProof);

enum Proved {
    Base(BlockWhirProof<LocalArgues>),
    Levels(RangeLevels),
}

impl Proved {
    fn base(self) -> Result<BlockWhirProof<LocalArgues>, Error> {
        match self {
            Self::Base(proof) => Ok(proof),
            Self::Levels(_) => Err(fail("expected a base proof")),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::block_whir) fn prove<H: multilinear::whir_hash::WhirHash>(
    coordinator: &Coordinator<'_>,
    plan: Plan,
    laid: Laid,
    elf: &[u8],
    opts: &ProofOptions,
    format: &BlockFormat,
    on_statement: StatementObserver<'_>,
    on_group: GroupObserver<'_>,
    stamps: &mut BlockStamps,
    local_owned: Option<Owned<'_>>,
) -> Result<BlockWhirProof<LocalArgues>, Error> {
    prove_as::<H>(
        coordinator,
        plan,
        laid,
        elf,
        opts,
        format,
        on_statement,
        on_group,
        stamps,
        local_owned,
    )
    .and_then(Proved::base)
    .inspect_err(|_| mark_failed())
}

#[allow(clippy::too_many_arguments)]
fn prove_as<H: multilinear::whir_hash::WhirHash>(
    coordinator: &Coordinator<'_>,
    plan: Plan,
    laid: Laid,
    elf: &[u8],
    opts: &ProofOptions,
    format: &BlockFormat,
    on_statement: StatementObserver<'_>,
    on_group: GroupObserver<'_>,
    stamps: &mut BlockStamps,
    local_owned: Option<Owned<'_>>,
) -> Result<Proved, Error> {
    let begin = Instant::now();
    let clock = coordinator.clock;
    let Coordinator {
        topology,
        attempt,
        peers,
        commit_deadline,
        ..
    } = coordinator;
    let attempt = *attempt;
    let table_num_vars: Vec<_> = laid.shapes.iter().map(|&(_, n)| n as u8).collect();
    let statement = laid.statement(&table_num_vars);
    let frame = block_frame(statement, elf, opts, format)?;
    let airs = frame.airs.air_refs();
    let prepared = commit_groups_on_host::<H>(
        group_stacks(
            prepared_tables(&frame.airs, &frame.page_configs, format)?,
            &laid.groups,
        )?,
        &frame.config,
    )?;
    let prepared_roots = prepared_roots(&prepared);
    clock.mark("prepared-roots-done");
    let mut settled = Vec::with_capacity(prepared.len());
    for p in &prepared {
        settled.push(multilinear_block::BlockPreparedCheck {
            group: p.group,
            tables: p.tables.clone(),
            roots: &p.roots,
            layout: p.commitment.layout(),
            domain: p.commitment.domain(),
        });
    }
    let layouts = frame
        .order
        .iter()
        .map(|&i| {
            let (width, vars) = frame.shapes[i];
            layout_of(airs[i], width, vars).map_err(|e| fail(format_args!("{e:?}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let fixed = frame
        .order
        .iter()
        .map(|&i| preprocessed_mles(airs[i]))
        .collect::<Result<Vec<_>, _>>()?;
    let mut statements: Vec<_> = layouts
        .iter()
        .zip(&fixed)
        .map(|(layout, cols)| layout.statement_with_preprocessed(cols))
        .collect();
    for p in &settled {
        for &(table, count) in &p.tables {
            if statements[table].preprocessed.is_empty() {
                statements[table].num_preprocessed = count;
            }
        }
    }
    let digests = if format.local_lookups() {
        local_digests::<H::Sponge>(&laid.groups, &frame.shapes, |i| airs.get(i).copied())?
    } else {
        Vec::new()
    };
    let buses = format
        .local_lookups()
        .then_some(&local_bytes::LOCAL_BUSES[..]);
    let mut at = 0;
    let mut shapes = Vec::with_capacity(plan.owners.len());
    for (group, &size) in frame.sizes.iter().enumerate() {
        let tables = at..at + size;
        let poly = frame.stack_layouts[group].num_polys();
        let pre = settled.iter().find(|p| p.group == group);
        let chains =
            |poly| multilinear::stacked_eval::index_chains(poly, frame.config.format.group).len();
        shapes.push((
            tables.clone(),
            multilinear::stacked_eval::round0_trees(poly, &frame.config).len(),
            poly,
            pre,
            GroupShape {
                tables: size,
                chains: chains(poly),
                prepared_chains: pre.map_or(0, |p| chains(p.layout.num_polys())),
                local: buses.is_some_and(|buses| {
                    multilinear_block::group_is_local(&statements[tables], buses)
                }),
                argue: frame.config.format.argue,
            },
        ));
        at += size;
    }
    let descriptors = (0..plan.owners.len())
        .map(|g| descriptor(&plan, g))
        .collect::<WireResult<Vec<_>>>()
        .map_err(fail)?;
    let mut slots: Vec<_> = plan.owners.iter().map(|_| Slot::Assigned).collect();
    let mut witnesses = vec![false; peers.len()];
    let expected_witness = local_owned
        .is_none()
        .then(|| plan.witness_digest())
        .transpose()?;
    let check_owner = |id, group, digest| {
        if plan.owners.get(group) != Some(&id) || descriptors.get(group) != Some(&digest) {
            return Err(wire::invalid(
                "group disagrees with authenticated assignment",
            ));
        }
        Ok(())
    };
    let assignment = rkyv::to_bytes::<rkyv::rancor::Error>(&plan).map_err(fail)?;
    for (group, committed) in local_owned.iter().flatten() {
        let (_, roots, answers, prepared, _) = &shapes[*group];
        let receipt =
            committed.replica_receipt(prepared.map_or_else(Vec::new, |p| p.roots.to_vec()));
        if receipt.roots.len() != *roots || receipt.answers.len() != *answers {
            return Err(fail("coordinator range commitment has the wrong shape"));
        }
        slots[*group] = Slot::Committed(receipt);
    }
    let commit = Phase {
        name: "commit",
        reply: Kind::Committed,
        end: Kind::CommittedAll,
        duration: wire::remaining(*commit_deadline).map_err(fail)?,
        peers,
        attempt,
    };
    clock.mark("wait-receipts");
    commit
        .run(
            |_, socket, deadline| {
                if expected_witness.is_some() {
                    socket.write_frame(deadline, Kind::Assign, attempt, &assignment)?;
                }
                Ok(())
            },
            |id, kind, bytes| {
                if kind == Kind::Witness {
                    if witnesses[id]
                        || expected_witness.is_none_or(|expected| bytes[..] != expected)
                    {
                        return Err(wire::invalid(
                            "replica witness digest disagrees with coordinator",
                        ));
                    }
                    witnesses[id] = true;
                    return Ok(());
                }
                if expected_witness.is_some() && !witnesses[id] {
                    return Err(wire::invalid("missing replica witness digest"));
                }
                let CommitReply {
                    group,
                    descriptor,
                    receipt,
                } = wire::decode(bytes)?;
                check_owner(id, group, descriptor)?;
                if !matches!(slots[group], Slot::Assigned) {
                    return Err(wire::invalid("duplicate commit reply"));
                }
                let (range, roots, answers, prepared, _) = &shapes[group];
                if receipt.roots.len() != *roots
                    || receipt.answers.len() != *answers
                    || receipt.prepared_roots != prepared.map_or(&[][..], |p| p.roots)
                    || receipt.local_digests != buses.map_or(&[][..], |_| &digests[range.clone()])
                {
                    return Err(wire::invalid(
                        "commit receipt disagrees with derived roots, digests or counts",
                    ));
                }
                slots[group] = Slot::Committed(receipt);
                Ok(())
            },
        )
        .map_err(fail)?;
    clock.mark("receipts-received");
    if expected_witness.is_some() && !witnesses.iter().all(|&received| received) {
        return Err(fail("missing replica witness digest"));
    }
    let mut receipts = Vec::with_capacity(slots.len());
    for slot in &slots {
        match slot {
            Slot::Committed(receipt) => receipts.push(receipt.clone()),
            _ => return Err(fail("missing commitment")),
        }
    }
    clock.mark("receipts-gathered");
    let (roots, answers, bind, post) = {
        let sealed = SealedPlan {
            statement: plan.statement,
            groups: receipts,
            derived: prepared_roots
                .iter()
                .flat_map(|(_, roots)| roots.iter().copied())
                .collect(),
        };
        let (roots, answers) = sealed.front();
        let mut post = H::sponge();
        let challenges = sealed.replay(&mut post);
        let digest = sealed.digest();
        (
            roots,
            answers,
            Bind {
                leaf_statement: coordinator.start.range_levels.then(|| statement.to_owned()),
                plan: sealed,
                digest,
                challenges,
            },
            post,
        )
    };
    clock.mark("seal-done");
    stamps.phase_a += begin.elapsed().as_secs_f64();
    stamps.prepared = (settled.len(), 0.0);
    let mut statement_sent = false;
    let begin = Instant::now();
    let sealed = rkyv::to_bytes::<rkyv::rancor::Error>(&bind).map_err(fail)?;
    clock.mark("bind-encode-done");
    let local_post = post.clone();
    let open_local = |local_post| {
        clock.mark("base-openings-start");
        let mut replies = Vec::new();
        for (group, committed) in local_owned.into_iter().flatten() {
            let opened = Instant::now();
            let start = shapes[group].0.start;
            let prepared = prepared.iter().find(|p| p.group == group);
            let borrowed =
                prepared.map_or_else(Vec::new, |p| multilinear::stacking::borrow(&p.columns));
            let opening: Vec<_> = prepared
                .iter()
                .map(|p| multilinear_block::BlockPrepared {
                    card: 0,
                    group: 0,
                    tables: settled
                        .iter()
                        .find(|check| check.group == group)
                        .unwrap()
                        .tables
                        .iter()
                        .map(|&(index, len)| (index - start, len))
                        .collect(),
                    commitment: (&p.commitment).into(),
                    columns: &borrowed,
                    packed: None,
                })
                .collect();
            let (parts, _) = committed
                .open_replica_group::<H, _>(
                    group,
                    &frame.config,
                    &local_post,
                    bind.challenges,
                    &opening,
                )
                .map_err(|e| fail(format_args!("coordinator group {group}: {e:?}")))?;
            clock.opened(group, opened);
            replies.push(GroupReply {
                group,
                descriptor: descriptors[group],
                plan: bind.digest,
                parts,
            });
        }
        clock.mark("base-openings-end");
        Ok::<_, Error>(replies)
    };
    if coordinator.start.range_levels {
        let (map, owners) = LeafMap::from_owners(&plan.owners, peers.len() + 1).map_err(fail)?;
        let leaf_plan = cluster_levels::cluster_plan(
            elf,
            opts,
            format,
            statement,
            Some(&prepared_roots),
            map.as_ref(),
        )
        .map_err(fail)?;
        clock.mark("leaf-plan-done");
        let proof = open_levels(coordinator, &leaf_plan, &owners, &bind, &sealed, || {
            open_local(local_post)
        })?;
        return Ok(Proved::Levels((statement.to_owned(), map, proof)));
    }
    let open = Phase {
        name: "open",
        reply: Kind::Result,
        end: Kind::OpenedAll,
        duration: topology.open,
        peers,
        attempt,
    };
    let mut completed = vec![0; peers.len()];
    let mut result_bytes = 0;
    let mut admit = |id, reply: GroupReply| -> WireResult<()> {
        let GroupReply {
            group,
            descriptor,
            plan,
            parts,
        } = reply;
        check_owner(id, group, descriptor)?;
        let Slot::Committed(receipt) = &slots[group] else {
            return Err(wire::invalid("duplicate or uncommitted group result"));
        };
        let (range, _, _, prepared, shape) = &shapes[group];
        if plan != bind.digest || !shape.check(&parts) {
            return Err(wire::invalid(
                "result disagrees with sealed plan or proof shape",
            ));
        }
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            replica::verify_group::<F, E, _, H>(
                group,
                receipt,
                &parts,
                &statements[range.clone()],
                range.start,
                *prepared,
                &frame.stack_layouts[group],
                &frame.domains[group],
                &frame.config,
                &post,
                &bind.challenges,
                buses,
            )
        }))
        .map_err(|_| {
            wire::invalid(format!(
                "group {group} verification failed on malformed data"
            ))
        })?
        .map_err(|e| wire::invalid(format!("group {group} verification: {e:?}")))?;
        if !statement_sent {
            // Avoid starting speculative recursion if the opening phase
            // fails before delivering a group.
            on_statement(statement, &prepared_roots);
            statement_sent = true;
        }
        on_group(parts.view(group, &roots, &answers));
        slots[group] = Slot::Opened(parts);
        if let Some(count) = completed.get_mut(id) {
            *count += 1;
        }
        Ok(())
    };
    let local = std::thread::scope(|scope| {
        let local = scope.spawn(move || open_local(local_post));
        let remote = open
            .run(
                |id, socket, deadline| {
                    socket.write_frame(deadline, Kind::Bind, attempt, &sealed)?;
                    clock.mark(format_args!("bind-sent-W{id}"));
                    Ok(())
                },
                |id, _, bytes| {
                    result_bytes += bytes.len();
                    if result_bytes > 512 << 20 {
                        return Err(wire::invalid("attempt exceeds result byte budget"));
                    }
                    admit(id, wire::decode(bytes)?)
                },
            )
            .map_err(fail);
        remote?;
        local
            .join()
            .map_err(|_| fail("coordinator opening thread panicked"))?
    })?;
    for reply in local {
        admit(peers.len(), reply).map_err(fail)?;
    }
    let mut tables = Vec::new();
    let mut columns = Vec::new();
    let mut argues = Vec::new();
    let mut openings = Vec::new();
    let mut local = Vec::new();
    for slot in slots {
        let Slot::Opened(parts) = slot else {
            return Err(fail("missing opening"));
        };
        tables.extend(parts.tables);
        columns.push(parts.opening);
        argues.extend(parts.argue);
        openings.extend(parts.prepared);
        if format.local_lookups() {
            local.push(parts.local.unwrap_or_default());
        }
    }
    finish_peers(coordinator);
    eprintln!("BLOCK CLUSTER TCP: completed groups by worker {completed:?}");
    stamps.phase_b = begin.elapsed().as_secs_f64();
    stamps.tables = laid.shapes.len();
    stamps.cells = laid.shapes.iter().map(|&(width, n)| width << n).sum();
    Ok(Proved::Base(BlockWhirProof {
        proof: MultiProof {
            roots,
            tables,
            columns,
            preprocessed: None,
        },
        table_num_vars,
        runtime_page_ranges: laid.runtime_page_ranges,
        table_counts: laid.table_counts,
        public_output: laid.public_output,
        num_private_input_pages: laid.num_private_input_pages,
        groups: laid.groups,
        prepared: openings,
        argues,
        commit_ood: answers,
        fused_memory: laid.fused_memory,
        word_memory: laid.word_memory,
        local,
    }))
}

fn finish_peers(coordinator: &Coordinator<'_>) {
    let finish_deadline = Instant::now() + coordinator.topology.startup.min(FINISH_TIMEOUT);
    let attempt = coordinator.attempt;
    let clock = coordinator.clock;
    std::thread::scope(|scope| {
        for (id, peer) in coordinator.peers.iter().enumerate() {
            scope.spawn(move || {
                let result = peer.write_frame(finish_deadline, Kind::Finish, attempt, &[])
                    .and_then(|()| {
                        clock.mark(format_args!("finish-sent-W{id}"));
                        clock.mark(format_args!("wait-finish-W{id}"));
                        peer.reader().receive(finish_deadline, Some(attempt), &[Kind::Finish])
                    });
                match result {
                    Ok(_) => clock.mark(format_args!("finish-received-W{id}")),
                    Err(error) => {
                        clock.mark(format_args!("finish-failed-W{id}"));
                        eprintln!("CLUSTER TCP: worker {id} missing Finish acknowledgement, continuing to proof verification: {error}");
                    }
                }
            });
        }
    });
    clock.mark("finish-done");
}

fn range_leaf_arena(
    plan: &crate::lfm::whir_block::WhirBlockPlan,
    leaf: usize,
    bind: &Bind,
    groups: Vec<(usize, Parts)>,
) -> Result<Vec<Vec<crate::lfm::LfmWord>>, Error> {
    if !groups
        .iter()
        .map(|(group, _)| *group)
        .eq(plan.partition().leaf(leaf).iter().copied())
    {
        return Err(fail(
            "base group openings disagree with the leaf assignment",
        ));
    }
    let words = groups
        .into_iter()
        .map(|(g, parts)| {
            group_arena_words(
                plan,
                g,
                &parts.tables,
                parts.argue.as_ref(),
                parts.local.as_ref(),
                &parts.opening,
                parts.prepared.as_ref(),
            )
        })
        .collect::<Result<_, _>>()
        .map_err(fail)?;
    let (roots, answers) = bind.plan.front();
    Ok(leaf_arena_under(&roots, &answers, words, plan.wrap_hash()))
}

fn open_levels(
    coordinator: &Coordinator<'_>,
    plan: &crate::lfm::whir_block::WhirBlockPlan,
    owners: &[usize],
    bind: &Bind,
    sealed: &[u8],
    open_local: impl FnOnce() -> Result<Vec<GroupReply>, Error> + Send,
) -> Result<WlfmProof, Error> {
    let clock = coordinator.clock;
    let local = coordinator.peers.len();
    let local_leaf = owners
        .iter()
        .position(|&id| id == local)
        .ok_or_else(|| fail("the terminal machine has no leaf"))?;
    let mut replies: Vec<Option<LeafReply>> = coordinator.peers.iter().map(|_| None).collect();
    let rounds = level_rounds(plan);
    let last = LastLevel::pinned(WlfmFamily::B3);
    enum Local {
        Closed(WlfmProof),
        Leaf(Box<(WlfmChild, WlfmProof)>),
    }
    let phase = Phase {
        name: "leaf",
        reply: Kind::Leaf,
        end: Kind::OpenedAll,
        duration: coordinator.topology.open,
        peers: &coordinator.peers,
        attempt: coordinator.attempt,
    };
    let local = std::thread::scope(|scope| {
        let local_proof = scope.spawn(|| {
            let groups = open_local()?
                .into_iter()
                .map(|reply| (reply.group, reply.parts))
                .collect();
            let arena = range_leaf_arena(plan, local_leaf, bind, groups)?;
            clock.mark("leaf-arena-done");
            let proof = if owners.len() == 1 {
                cluster_levels::prove_single_staged(plan, &arena, last, clock)
                    .map(Local::Closed)
                    .map_err(fail)
            } else {
                cluster_levels::prove_leaf_staged(plan, local_leaf, &arena, clock)
                    .map(|leaf| Local::Leaf(Box::new(leaf)))
                    .map_err(fail)
            }?;
            clock.mark("wait-leaf-relay");
            Ok::<_, Error>(proof)
        });
        phase
            .run(
                |id, socket, deadline| {
                    socket.write_frame(deadline, Kind::Bind, coordinator.attempt, sealed)?;
                    clock.mark(format_args!("bind-sent-W{id}"));
                    Ok(())
                },
                |id, _, bytes| {
                    let reply: LeafReply = wire::decode(bytes)?;
                    let plan = reply.plan;
                    take_leaf_reply(&mut replies, owners, &bind.digest, id, plan, reply)?;
                    clock.mark(format_args!("leaf-received-W{id}"));
                    Ok(())
                },
            )
            .map_err(fail)?;
        local_proof
            .join()
            .map_err(|_| fail("coordinator leaf thread panicked"))?
    })?;
    clock.mark("leaf-relay-done");

    // Workers wait for Finish while the coordinator constructs the join.
    // Acknowledge completed leaves before the join and the host file check.
    finish_peers(coordinator);
    let Local::Leaf(local) = local else {
        let Local::Closed(proof) = local else {
            unreachable!()
        };
        return Ok(proof);
    };
    let leaves = leaves_in_order(owners, local_leaf, *local, &mut replies, |reply| {
        let artifacts = reply.artifacts.into_artifacts(rounds).map_err(fail)?;
        let child = WlfmChild {
            artifacts,
            num_public_words: plan.child_layout().total(),
            cap_mux: Some(CapMux::Packed),
        };
        Ok((child, reply.proof))
    })?;
    clock.mark("leaf-artifacts-ready");
    cluster_levels::prove_join_staged(&plan.child_layout(), rounds, leaves, last, clock)
        .map_err(fail)
}

/// Takes a worker's leaf once, from a machine that owns a leaf, for the plan
/// this attempt bound. Anything else is refused before it is kept.
fn take_leaf_reply<R>(
    replies: &mut [Option<R>],
    owners: &[usize],
    bound: &Digest32,
    id: usize,
    plan: Digest32,
    reply: R,
) -> WireResult<()> {
    if plan != *bound || !owners.contains(&id) || replies.get(id).is_none_or(Option::is_some) {
        return Err(wire::invalid("unexpected, duplicate or stale worker leaf"));
    }
    replies[id] = Some(reply);
    Ok(())
}

/// The leaves in leaf order: the coordinator's own at `local_leaf`, every
/// other one a worker's reply, converted. A leaf with no reply is refused.
fn leaves_in_order<L, R>(
    owners: &[usize],
    local_leaf: usize,
    local: L,
    replies: &mut [Option<R>],
    mut convert: impl FnMut(R) -> Result<L, Error>,
) -> Result<Vec<L>, Error> {
    let mut local = Some(local);
    owners
        .iter()
        .enumerate()
        .map(|(leaf, &id)| {
            if leaf == local_leaf {
                return local.take().ok_or_else(|| fail("duplicate local leaf"));
            }
            let reply = replies
                .get_mut(id)
                .and_then(Option::take)
                .ok_or_else(|| fail(format!("missing worker {id} leaf")))?;
            convert(reply)
        })
        .collect()
}

pub(in crate::block_whir) struct Worker<'a> {
    socket: &'a Connection,
    attempt: Attempt,
    id: usize,
    machines: usize,
    range_levels: bool,
    cards: Vec<usize>,
    commit_deadline: Instant,
    open: Duration,
    startup: Duration,
    stream_base: Mutex<mpsc::Receiver<WireResult<usize>>>,
    bind: Mutex<mpsc::Receiver<WireResult<AlignedVec>>>,
    finished: Mutex<mpsc::Receiver<WireResult<()>>>,
    cancelled: &'a AtomicBool,
    watchdog: mpsc::Sender<(&'static str, Instant)>,
    clock: Clock,
}

impl Worker<'_> {
    pub fn card(&self, plan: &Plan, group: usize) -> usize {
        let placed = plan.owners[..group]
            .iter()
            .filter(|&&id| id == self.id)
            .count();
        self.cards[placed % self.cards.len()]
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    pub fn stream_closed(&self, count: usize) -> Result<(), Error> {
        #[cfg(test)]
        let count = if fault_is("stream-count-cap") {
            BLOCK_MAX_GROUPS
        } else {
            count
        };
        self.socket
            .write_frame(
                self.commit_deadline,
                Kind::StreamClosed,
                self.attempt,
                &(count as u64).to_le_bytes(),
            )
            .map_err(fail)
    }

    pub fn stream_base(&self) -> Result<usize, Error> {
        self.stream_base
            .lock()
            .unwrap()
            .recv_timeout(wire::remaining(self.commit_deadline).map_err(fail)?)
            .map_err(fail)?
            .map_err(fail)
    }

    pub fn range_card(&self, local_group: usize) -> usize {
        self.cards[local_group % self.cards.len()]
    }

    pub fn check_cancelled(&self) -> Result<(), Error> {
        if self.cancelled.load(Ordering::Relaxed) {
            return Err(fail("coordinator disconnected"));
        }
        wire::remaining(self.commit_deadline).map_err(fail)?;
        Ok(())
    }

    pub fn open<H: multilinear::whir_hash::WhirHash>(
        &self,
        plan: &Plan,
        owned: Owned<'_>,
        witness: Option<Digest32>,
        prepared: Vec<PreparedColumns>,
        groups: &[Vec<u32>],
        config: &multilinear::whir_chain::ChainConfig,
    ) -> Result<(), Error> {
        self.open_as::<H>(plan, owned, witness, prepared, groups, config, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn open_as<H: multilinear::whir_hash::WhirHash>(
        &self,
        plan: &Plan,
        owned: Owned<'_>,
        witness: Option<Digest32>,
        prepared: Vec<PreparedColumns>,
        groups: &[Vec<u32>],
        config: &multilinear::whir_chain::ChainConfig,
        level_inputs: Option<(&[u8], &ProofOptions, &BlockFormat)>,
    ) -> Result<(), Error> {
        let mut stacks = group_stacks(prepared, groups)?.into_iter().peekable();
        let mut private = Vec::with_capacity(owned.len());
        let mut at = 0;
        let starts: Vec<_> = groups
            .iter()
            .map(|tables| {
                let start = at;
                at += tables.len();
                start
            })
            .collect();
        #[cfg(test)]
        {
            fault_pause("commit");
            if fault_is("panic") {
                eprintln!(
                    "CLUSTER TEST PANIC {:?}",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                );
                panic!("injected worker panic");
            }
            if fault_is("card-hang") {
                // This runs on the card path and never returns to a cancellation check.
                multilinear::gpu::on_indexed_card(self.card(plan, 0), false, || {
                    fault_pause("card-hang")
                });
            }
            if fault_is("timeout") {
                while !self.cancelled.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        }
        if let Some(witness) = witness {
            #[cfg(test)]
            let witness = if bypass_witness() {
                plan.witness_digest()?
            } else if fault_is("witness-reply") {
                let mut digest = witness;
                digest[0] ^= 1;
                digest
            } else {
                witness
            };
            self.socket
                .write_frame(self.commit_deadline, Kind::Witness, self.attempt, &witness)
                .map_err(fail)?;
        }
        for (group, committed) in owned {
            self.check_cancelled()?;
            while stacks.peek().is_some_and(|stack| stack.group < group) {
                stacks.next();
            }
            let card = self.card(plan, group);
            let prepared = if stacks.peek().is_some_and(|stack| stack.group == group) {
                let stack = stacks.next().unwrap();
                Some(
                    multilinear::gpu::on_indexed_card(card, false, || {
                        commit_group::<H>(stack, config)
                    })
                    .map_err(Error::from)?,
                )
            } else {
                None
            };
            let receipt = committed
                .replica_receipt(prepared.as_ref().map_or_else(Vec::new, |p| p.roots.clone()));
            let descriptor = descriptor(plan, group).map_err(fail)?;
            #[cfg_attr(not(test), allow(unused_mut))]
            let mut reply = CommitReply {
                group,
                descriptor,
                receipt,
            };
            #[cfg(test)]
            if fault_is("group-index") {
                reply.group = usize::MAX;
            }
            #[cfg(test)]
            if fault_is("receipt-shape") {
                reply.receipt.answers.clear();
            }
            #[cfg(test)]
            if fault_is("stale") {
                self.socket
                    .send(self.commit_deadline, [255; 16], Kind::Committed, &reply)
                    .map_err(fail)?;
            }
            self.socket
                .send(self.commit_deadline, self.attempt, Kind::Committed, &reply)
                .map_err(fail)?;
            #[cfg(test)]
            if fault_is("duplicate") {
                self.socket
                    .send(self.commit_deadline, self.attempt, Kind::Committed, &reply)
                    .map_err(fail)?;
            }
            private.push((committed, prepared, card, reply));
        }
        drop(stacks);
        self.socket
            .write_frame(self.commit_deadline, Kind::CommittedAll, self.attempt, &[])
            .map_err(fail)?;
        self.clock.mark("receipts-sent");
        let bind_deadline = self.commit_deadline + self.startup;
        let _ = self.watchdog.send(("bind", bind_deadline));
        self.clock.mark("wait-bind");
        let bind_bytes = self
            .bind
            .lock()
            .unwrap()
            .recv_timeout(wire::remaining(bind_deadline).map_err(fail)?)
            .map_err(fail)?
            .map_err(fail)?;
        self.clock.mark("bind-received");
        let deadline = Instant::now() + self.open;
        let _ = self.watchdog.send(("open", deadline));
        let bind: Bind = wire::decode(&bind_bytes).map_err(fail)?;
        let mut post = H::sponge();
        if bind.leaf_statement.is_some() != level_inputs.is_some()
            || bind.plan.statement != plan.statement
            || bind.plan.groups.len() != plan.owners.len()
            || bind.plan.digest() != bind.digest
            || !bind
                .plan
                .groups
                .iter()
                .flat_map(|r| r.prepared_roots.iter())
                .eq(&bind.plan.derived)
            || bind.plan.replay(&mut post) != bind.challenges
            || private
                .iter()
                .any(|(_, _, _, reply)| !bind.plan.groups[reply.group].matches(&reply.receipt))
        {
            return Err(fail("invalid sealed plan"));
        }
        self.clock.mark("seal-checked");
        #[cfg(test)]
        fault_pause("bind");
        let mut leaf_groups = Vec::new();
        self.clock.mark("base-openings-start");
        for (
            committed,
            prepared,
            card,
            CommitReply {
                group, descriptor, ..
            },
        ) in private
        {
            let opened = Instant::now();
            if self.cancelled.load(Ordering::Relaxed) {
                return Err(fail("coordinator disconnected"));
            }
            wire::remaining(deadline).map_err(fail)?;
            let post = post.clone();
            let challenges = bind.challenges;
            let start = starts[group];
            let (parts, _) = multilinear::gpu::on_indexed_card(card, false, move || {
                let borrowed = prepared
                    .as_ref()
                    .map_or_else(Vec::new, |p| multilinear::stacking::borrow(&p.columns));
                let prepared_opening: Vec<_> = prepared
                    .iter()
                    .map(|p| multilinear_block::BlockPrepared {
                        card,
                        group: 0,
                        tables: p
                            .tables
                            .iter()
                            .map(|&(index, len)| (index - start, len))
                            .collect(),
                        commitment: (&p.commitment).into(),
                        columns: &borrowed,
                        packed: None,
                    })
                    .collect();
                committed.open_replica_group::<H, _>(
                    group,
                    config,
                    &post,
                    challenges,
                    &prepared_opening,
                )
                // Everything in this closure, including its prepared stack,
                // drops before the writer can block on the result.
            })
            .map_err(|e| fail(format_args!("group {group}: {e:?}")))?;
            self.clock.opened(group, opened);
            #[cfg(test)]
            let parts = if fault_is("group-verification") {
                let mut parts = parts;
                parts.opening.polys[0].rounds.clear();
                parts
            } else if fault_is("group-shape") {
                let mut parts = parts;
                parts.tables.pop().expect("per-table fixture");
                parts
            } else {
                parts
            };
            if level_inputs.is_some() {
                leaf_groups.push((group, parts));
                continue;
            }
            self.socket
                .send(
                    deadline,
                    self.attempt,
                    Kind::Result,
                    &GroupReply {
                        group,
                        descriptor,
                        plan: bind.digest,
                        parts,
                    },
                )
                .map_err(fail)?;
        }
        self.clock.mark("base-openings-end");
        if let Some((elf, opts, format)) = level_inputs
            && !leaf_groups.is_empty()
        {
            let (map, owners) = LeafMap::from_owners(&plan.owners, self.machines).map_err(fail)?;
            let leaf = owners
                .iter()
                .position(|&id| id == self.id)
                .ok_or_else(|| fail("worker has no leaf"))?;
            let map = map
                .as_ref()
                .ok_or_else(|| fail("single active leaf must be terminal"))?;
            let statement = bind
                .leaf_statement
                .as_ref()
                .ok_or_else(|| fail("missing leaf statement"))?
                .view();
            let roots: Vec<_> = bind
                .plan
                .groups
                .iter()
                .enumerate()
                .filter(|(_, group)| !group.prepared_roots.is_empty())
                .map(|(index, group)| (index, group.prepared_roots.clone()))
                .collect();
            let plan =
                cluster_levels::cluster_plan(elf, opts, format, statement, Some(&roots), Some(map))
                    .map_err(fail)?;
            self.clock.mark("leaf-plan-done");
            let arena = range_leaf_arena(&plan, leaf, &bind, leaf_groups)?;
            self.clock.mark("leaf-arena-done");
            #[cfg(test)]
            let prove_leaf = if fault_is("weakened-leaf") {
                cluster_levels::prove_weakened_leaf
            } else {
                cluster_levels::prove_leaf_staged
            };
            #[cfg(not(test))]
            let prove_leaf = cluster_levels::prove_leaf_staged;
            let (child, proof) = prove_leaf(&plan, leaf, &arena, self.clock).map_err(fail)?;
            self.socket
                .send(
                    deadline,
                    self.attempt,
                    Kind::Leaf,
                    &LeafReply {
                        plan: bind.digest,
                        artifacts: LeafArtifacts::from_build(&child.artifacts),
                        proof,
                    },
                )
                .map_err(fail)?;
            self.clock.mark("leaf-sent");
        }
        self.socket
            .write_frame(deadline, Kind::OpenedAll, self.attempt, &[])
            .map_err(fail)?;
        self.clock.mark("opened-all-sent");
        let finish_deadline = deadline + self.startup;
        let _ = self.watchdog.send(("finish", finish_deadline));
        self.clock.mark("wait-finish");
        self.finished
            .lock()
            .unwrap()
            .recv_timeout(wire::remaining(finish_deadline).map_err(fail)?)
            .map_err(fail)?
            .map_err(fail)?;
        self.clock.mark("finish-received");
        #[cfg(test)]
        if fault_is("no-finish") {
            return Ok(());
        }
        self.socket
            .write_frame(finish_deadline, Kind::Finish, self.attempt, &[])
            .map_err(fail)?;
        self.clock.mark("finish-sent");
        Ok(())
    }
}

#[cfg(test)]
pub(in crate::block_whir) fn fault_is(name: &str) -> bool {
    std::env::var("LAMBDA_VM_CLUSTER_TEST_FAULT").as_deref() == Ok(name)
}

#[cfg(test)]
pub(in crate::block_whir) fn bypass_witness() -> bool {
    fault_is("global-balance") || fault_is("group-verification")
}

#[cfg(test)]
pub(in crate::block_whir) fn fault_pause(name: &str) {
    if fault_is(name) {
        eprintln!("CLUSTER TEST READY {name}");
        loop {
            std::thread::park();
        }
    }
}

fn watch_deadline(
    socket: &Connection,
    cancelled: &AtomicBool,
    deadlines: mpsc::Receiver<(&'static str, Instant)>,
    mut phase: &'static str,
    mut deadline: Instant,
) {
    loop {
        match deadlines.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(next) => (phase, deadline) = next,
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
            Err(mpsc::RecvTimeoutError::Timeout) => break,
        }
    }
    cancelled.store(true, Ordering::Relaxed);
    socket.close();
    eprintln!("CLUSTER TCP: worker {phase} deadline expired");
    // Ordinary cancellation unwinds the attempt and drops its card allocations.
    // If the task is stuck in a driver call, exit so the OS and driver can reclaim them.
    let grace = Instant::now() + WATCHDOG_GRACE;
    loop {
        match deadlines.recv_timeout(grace.saturating_duration_since(Instant::now())) {
            Ok(_) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                eprintln!("CLUSTER TCP: worker still blocked after {phase} deadline, exiting");
                std::process::exit(1);
            }
        }
    }
}

/// Serve a single authenticated proof attempt.
pub fn serve(
    topology: &Topology,
    id: usize,
    opts: &ProofOptions,
    format: &BlockFormat,
) -> Result<(), Error> {
    serve_attempts(topology, id, opts, format, NonZeroUsize::MIN)
}

/// Serve successive authenticated attempts, stopping at the first failure.
/// Launch with the configured LAMBDA_VM_CARDS before CUDA initializes. A
/// stuck worker exits after the phase deadline and a one-second grace period.
pub fn serve_attempts(
    topology: &Topology,
    id: usize,
    opts: &ProofOptions,
    format: &BlockFormat,
    attempts: NonZeroUsize,
) -> Result<(), Error> {
    let endpoint = topology
        .endpoints
        .get(id)
        .ok_or_else(|| fail("unknown worker index"))?;
    let token = endpoint
        .token
        .as_ref()
        .ok_or_else(|| fail("missing worker token"))?;
    let cards = topology.worker_cards(id).map_err(fail)?;
    if cfg!(feature = "cuda") && std::env::var("LAMBDA_VM_CARDS").as_deref() != Ok(cards.as_str()) {
        return Err(fail(format!(
            "worker {id} requires LAMBDA_VM_CARDS={cards}"
        )));
    }
    check_memory_format(format)?;
    refuse_unprovable(format)?;
    let listener = TcpListener::bind(endpoint.address).map_err(fail)?;
    let context = topology.context(id, &profile(opts, format)).map_err(fail)?;
    listener.set_nonblocking(true).map_err(fail)?;
    for attempt in 0..attempts.get() {
        let (socket, deadline) = if attempt == 0 {
            // The first attempt connects and starts within startup.
            let deadline = Instant::now() + topology.startup;
            (accept(&listener, token, &context, id, deadline)?, deadline)
        } else {
            // A later one comes after the coordinator's join proof and file
            // check: it gets that wait to connect, then startup to start.
            let wait = Instant::now() + topology.next_attempt;
            let socket = accept(&listener, token, &context, id, wait)?;
            (socket, Instant::now() + topology.startup)
        };
        serve_attempt(topology, id, opts, format, &socket, deadline)?;
    }
    Ok(())
}

fn accept(
    listener: &TcpListener,
    token: &Digest32,
    context: &Digest32,
    id: usize,
    deadline: Instant,
) -> Result<Connection, Error> {
    loop {
        let left = wire::remaining(deadline).map_err(fail)?;
        match listener.accept() {
            Ok((stream, _)) => {
                let handshake = stream.set_nonblocking(false).and_then(|()| {
                    wire::authenticate(
                        stream,
                        deadline.min(Instant::now() + HANDSHAKE_TIMEOUT),
                        token,
                        context,
                        false,
                    )
                });
                match handshake {
                    Ok(socket) => return Ok(socket),
                    Err(error) => eprintln!("CLUSTER TCP: worker {id} rejected handshake: {error}"),
                }
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                std::thread::sleep(left.min(Duration::from_millis(10)));
            }
            Err(e) => return Err(fail(e)),
        }
    }
}

fn serve_attempt(
    topology: &Topology,
    id: usize,
    opts: &ProofOptions,
    format: &BlockFormat,
    socket: &Connection,
    deadline: Instant,
) -> Result<(), Error> {
    let mut reader = socket.reader();
    let (attempt, _, bytes) = reader
        .receive(deadline, None, &[Kind::Start])
        .map_err(fail)?;
    let clock = if topology.cuts.is_some() {
        let clock = Clock::new(Instant::now(), attempt, Some(id));
        clock.mark("attempt-start");
        clock
    } else {
        Clock::default()
    };
    let (start, window, elf, inputs) = receive_start(&mut reader, deadline, attempt, &bytes, clock)
        .inspect_err(|error| report_error(socket, attempt, error))?;
    let settings: Arc<[_]> = start.settings.into();
    let options = &start.options;
    if start.range_levels && (start.cuts.is_none() || format.zf.base != BaseFormat::B3_WHIR) {
        return Err(fail("range-owned levels require a BLAKE3 range topology"));
    }
    if start.cuts != topology.cuts {
        return Err(fail("Start range list disagrees with topology"));
    }
    clock.mark("input-checked");
    let assignment_deadline = Instant::now() + topology.commit;
    let plan = if start.cuts.is_none() {
        let (_, _, assignment) = reader
            .receive(assignment_deadline, Some(attempt), &[Kind::Assign])
            .map_err(fail)?;
        Some(decode_assignment(&assignment, topology, format)?)
    } else {
        None
    };
    // Replica workers have a separate assignment wait. Range workers
    // execute from Start, so their build uses the original deadline.
    let commit_deadline = if start.cuts.is_some() {
        assignment_deadline
    } else {
        Instant::now() + topology.commit
    };
    let cancelled = AtomicBool::new(false);
    let result = std::thread::scope(|scope| {
        let _close = CloseOnDrop(std::slice::from_ref(socket));
        let (watchdog, deadlines) = mpsc::channel();
        scope.spawn(|| watch_deadline(socket, &cancelled, deadlines, "commit", commit_deadline));
        let (atx, assign) = mpsc::sync_channel(1);
        let (base_tx, stream_base) = mpsc::sync_channel(1);
        let is_range = start.cuts.is_some();
        let (btx, bind) = mpsc::sync_channel(1);
        let (ftx, finished) = mpsc::sync_channel(1);
        let cancelled = &cancelled;
        scope.spawn(move || {
            if is_range {
                let base = reader
                    .receive(commit_deadline, Some(attempt), &[Kind::IndexBase])
                    .and_then(|(_, _, bytes)| group_count(&bytes, format.max_groups));
                if base.is_err() {
                    cancelled.store(true, Ordering::Relaxed);
                }
                if base_tx.send(base).is_err() || cancelled.load(Ordering::Relaxed) {
                    return;
                }
                let message = reader
                    .receive(commit_deadline, Some(attempt), &[Kind::Assign])
                    .map(|(_, _, bytes)| bytes);
                if message.is_err() {
                    cancelled.store(true, Ordering::Relaxed);
                }
                if atx.send(message).is_err() || cancelled.load(Ordering::Relaxed) {
                    return;
                }
            }
            let message = reader
                .receive(
                    commit_deadline + topology.startup,
                    Some(attempt),
                    &[Kind::Bind],
                )
                .map(|(_, _, bytes)| bytes);
            if message.is_err() {
                cancelled.store(true, Ordering::Relaxed);
                let _ = btx.send(message);
                return;
            }
            if btx.send(message).is_err() {
                return;
            }
            let finish = reader
                .receive(
                    Instant::now() + topology.open + topology.startup,
                    Some(attempt),
                    &[Kind::Finish],
                )
                .map(|_| ());
            if finish.is_err() {
                cancelled.store(true, Ordering::Relaxed);
            }
            let _ = ftx.send(finish);
        });
        let run = || {
            let cards = multilinear::gpu::indexed_cards()
                .map_err(fail)?
                .unwrap_or_else(|| vec![0]);
            let worker = Worker {
                socket,
                attempt,
                id,
                machines: topology.workers() + 1,
                range_levels: start.range_levels,
                cards,
                commit_deadline,
                open: topology.open,
                startup: topology.startup,
                stream_base: Mutex::new(stream_base),
                bind: Mutex::new(bind),
                finished: Mutex::new(finished),
                cancelled,
                watchdog,
                clock,
            };
            let program = Elf::load(&elf).map_err(|e| Error::ElfLoad(format!("{e}")))?;
            clock.mark("elf-loaded");
            let trace_seed = start.trace_seed;
            #[cfg(test)]
            let trace_seed = if fault_is("ignore-seed") {
                [0; 16]
            } else {
                trace_seed
            };
            if let Some(cuts) = &start.cuts {
                let run = || {
                    serve_range(
                        &worker, cuts, &assign, topology, &program, &elf, &inputs, window,
                        trace_seed, opts, format, options,
                    )
                };
                return if format.zf.base == stark::proof::options::BaseFormat::B3_WHIR {
                    crate::base_only::with_blake3_levels(run)
                } else {
                    run()
                };
            }
            let mut airs = HostAirs::new(opts, format, options);
            airs.stream.trace_seed = Some(trace_seed);
            let run = || {
                prove_streamed_on_host(
                    &program,
                    &elf,
                    &inputs,
                    window,
                    opts,
                    format,
                    options,
                    &Deviations::default(),
                    Pass::Worker {
                        id,
                        plan: plan.as_ref().expect("replica assignment"),
                        tcp: Some(&worker),
                    },
                    &airs,
                    &|_, _| {},
                    None,
                    &|_| {},
                    None,
                    &mut BlockStamps::default(),
                )
                .map(|_| ())
            };
            match format.zf.base {
                stark::proof::options::BaseFormat::B3_WHIR => {
                    crate::base_only::with_blake3_levels(run)
                }
                _ => run(),
            }
        };
        let result = with_settings(Some(settings), run);
        if let Err(error) = &result {
            report_error(socket, attempt, error);
        }
        result
    });
    result?;
    if cancelled.load(Ordering::Relaxed) {
        return Err(fail("worker attempt cancelled"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(in crate::block_whir) fn prove_ranges(
    coordinator: &Coordinator<'_>,
    cuts: &[usize],
    program: &Elf,
    elf: &[u8],
    input: &[u8],
    window: usize,
    opts: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
    on_statement: StatementObserver<'_>,
    on_group: GroupObserver<'_>,
    stamps: &mut BlockStamps,
) -> Result<BlockWhirProof<LocalArgues>, Error> {
    prove_ranges_as(
        coordinator,
        cuts,
        program,
        elf,
        input,
        window,
        opts,
        format,
        options,
        on_statement,
        on_group,
        stamps,
    )
    .and_then(Proved::base)
    // Build-phase frames are read on spawned threads, so their own marks are
    // lost; any error leaving the attempt marks it on the proving thread.
    .inspect_err(|_| mark_failed())
}

pub(crate) fn prove_range_levels(
    topology: &Topology,
    elf: &[u8],
    input: &[u8],
    opts: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
) -> Result<(RangeLevels, Clock), String> {
    let cuts = topology
        .range_cuts()
        .ok_or("range levels require a windows list")?;
    if topology.workers() >= cluster_levels::MAX_CLUSTER_LEAVES {
        return Err("range topology exceeds the leaf cap".into());
    }
    let window = validate_options(options).map_err(|e| e.to_string())?;
    check_memory_format(format).map_err(|e| e.to_string())?;
    refuse_unprovable(format).map_err(|e| e.to_string())?;
    let program = Elf::load(elf).map_err(|e| e.to_string())?;
    let coordinator = Coordinator::start_as(topology, elf, input, opts, format, options, true)
        .map_err(|e| e.to_string())?;
    let _close = CloseOnDrop(&coordinator.peers);
    let result = crate::base_only::with_blake3_levels(|| {
        prove_ranges_as(
            &coordinator,
            cuts,
            &program,
            elf,
            input,
            window,
            opts,
            format,
            options,
            &|_, _| {},
            &|_| {},
            &mut BlockStamps::default(),
        )
    })
    .map_err(|e| e.to_string())?;
    match result {
        Proved::Levels(proof) => Ok((proof, coordinator.clock)),
        Proved::Base(_) => Err("range levels produced a base proof".into()),
    }
}

#[allow(clippy::too_many_arguments)]
fn prove_ranges_as(
    coordinator: &Coordinator<'_>,
    cuts: &[usize],
    program: &Elf,
    elf: &[u8],
    input: &[u8],
    window: usize,
    opts: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
    on_statement: StatementObserver<'_>,
    on_group: GroupObserver<'_>,
    stamps: &mut BlockStamps,
) -> Result<Proved, Error> {
    let start = Instant::now();
    let clock = coordinator.clock;
    set_up_card();
    let id = coordinator.workers();
    let deadline = coordinator.commit_deadline;
    let (local_base, base_rx) = mpsc::sync_channel(1);
    // The first failure of a reader, the index relay or the coordinator's own
    // build cancels the others: the peers are closed and the build stops at
    // its next window or streamed group.
    let failure = OnceLock::new();
    let cancel = |error: &Error| {
        let cause = match error {
            Error::Prover(cause) => cause.clone(),
            other => other.to_string(),
        };
        if failure.set(cause).is_ok() {
            drop(CloseOnDrop(&coordinator.peers));
        }
    };
    let job = range::Job::new(cuts, id, None)
        .with_clock(clock)
        .with_stream_base(base_rx, deadline, &failure);
    let airs = range::airs(
        &job,
        window,
        coordinator.trace_seed(),
        opts,
        format,
        options,
    );
    let attempt = coordinator.attempt;
    let (mut terminal, remote) = std::thread::scope(|scope| {
        let (stream_tx, stream_rx) = mpsc::channel();
        let readers: Vec<_> = coordinator
            .peers
            .iter()
            .enumerate()
            .map(|(id, peer)| {
                let stream_tx = stream_tx.clone();
                scope.spawn(move || {
                    let read = || {
                        let mut reader = peer.reader();
                        let streamed = reader
                            .receive(deadline, Some(attempt), &[Kind::StreamClosed])
                            .and_then(|(_, _, bytes)| group_count(&bytes, format.max_groups))
                            .map_err(fail)?;
                        clock.mark(format_args!("stream-closed-received-W{id}"));
                        let _ = stream_tx.send((id, streamed));
                        let (_, _, bytes) = reader
                            .receive(deadline, Some(attempt), &[Kind::RangeReady])
                            .map_err(fail)?;
                        clock.mark(format_args!("range-ready-received-W{id}"));
                        let metadata: range::Metadata = wire::decode(&bytes).map_err(fail)?;
                        if metadata.streamed != streamed {
                            return Err(fail("streamed group count disagrees with range receipt"));
                        }
                        let (_, _, bytes) = reader
                            .receive(deadline, Some(attempt), &[Kind::RangeCounts])
                            .map_err(fail)?;
                        clock.mark(format_args!("range-counts-received-W{id}"));
                        let counts: range::Counts = wire::decode(&bytes).map_err(fail)?;
                        if metadata.end == 0
                            && (!counts.bitwise.is_empty()
                                || !counts.decode.is_empty()
                                || counts.keccak != 0)
                            || metadata.end != 0 && counts.bitwise.is_empty()
                        {
                            return Err(fail("invalid range counts for assigned cycles"));
                        }
                        Ok((metadata, counts))
                    };
                    read().inspect_err(cancel)
                })
            })
            .collect();
        drop(stream_tx);
        let relay = scope.spawn(move || {
            let relay = || {
                let mut seen = vec![None; id];
                let mut base = 0;
                for (next, route) in coordinator.peers.iter().enumerate() {
                    route
                        .write_frame(
                            deadline,
                            Kind::IndexBase,
                            attempt,
                            &(base as u64).to_le_bytes(),
                        )
                        .map_err(fail)?;
                    while seen[next].is_none() {
                        let (owner, count) = stream_rx
                            .recv_timeout(wire::remaining(deadline).map_err(fail)?)
                            .map_err(fail)?;
                        seen[owner] = Some(count);
                    }
                    base += seen[next].take().unwrap();
                    check_group_count(base, format.max_groups)?;
                }
                let _ = local_base.send(base);
                Ok::<_, Error>(())
            };
            relay().inspect_err(cancel)
        });
        let terminal = with_settings(Some(coordinator.settings()), || {
            range::build(
                &job, &airs, program, elf, input, window, opts, format, options,
            )
        });
        let terminal = terminal.inspect_err(cancel);
        clock.mark("wait-range-readers");
        let remote = readers
            .into_iter()
            .map(|reader| reader.join().map_err(|_| fail("range reader panicked"))?)
            .collect::<Result<Vec<_>, Error>>();
        let relayed = relay
            .join()
            .map_err(|_| fail("range index relay panicked"))?;
        if let Some(cause) = failure.get() {
            return Err(Error::Prover(cause.clone()));
        }
        clock.mark("range-readers-done");
        relayed?;
        Ok::<_, Error>((terminal?, remote?))
    })?;
    let mut metas = Vec::with_capacity(id + 1);
    for (metadata, counts) in remote {
        if metadata.end != 0 {
            terminal.add_counts(&counts).map_err(fail)?;
        }
        metas.push(metadata);
    }
    metas.push(terminal.describe(&job)?.0);
    let (laid, plan) = range::assemble(
        &metas,
        cuts,
        window,
        &terminal.laid,
        elf,
        program,
        opts,
        format,
    )
    .map_err(fail)?;
    let assignment = rkyv::to_bytes::<rkyv::rancor::Error>(&plan).map_err(fail)?;
    clock.mark("assign-encode-done");
    for (id, peer) in coordinator.peers.iter().enumerate() {
        peer.write_frame(deadline, Kind::Assign, attempt, &assignment)
            .map_err(fail)?;
        clock.mark(format_args!("assign-sent-W{id}"));
    }
    let config = format.chain_config(&laid.shapes);
    let owned_indices = plan
        .owners
        .iter()
        .enumerate()
        .filter_map(|(group, &owner)| (owner == id).then_some(group));
    crate::with_base_hash!(format.zf.base, |H| {
        let streamed = metas[id].streamed;
        let mut early = terminal.committed.into_iter();
        let local = owned_indices
            .zip(terminal.groups)
            .map(|(index, group)| {
                if group.index < streamed {
                    let (assigned, committed) = early
                        .next()
                        .ok_or_else(|| fail("missing coordinator stream commitment"))?;
                    if assigned != index {
                        return Err(fail("coordinator index base disagrees with assignment"));
                    }
                    return Ok((index, committed));
                }
                Ok((
                    index,
                    range::commit::<H>(group, index, &config, options, format)?,
                ))
            })
            .collect::<Result<Owned<'_>, Error>>()?;
        clock.mark("final-commits-done");
        stamps.phase_a = start.elapsed().as_secs_f64();
        prove_as::<H>(
            coordinator,
            plan,
            laid,
            elf,
            opts,
            format,
            on_statement,
            on_group,
            stamps,
            Some(local),
        )
    })
}

fn group_count(bytes: &[u8], max: usize) -> WireResult<usize> {
    let raw: [u8; 8] = bytes
        .try_into()
        .map_err(|_| wire::invalid("invalid group count frame"))?;
    usize::try_from(u64::from_le_bytes(raw))
        .ok()
        .filter(|&n| n <= max)
        .ok_or_else(|| wire::invalid("group count exceeds block limit"))
}

fn decode_assignment(
    bytes: &AlignedVec,
    topology: &Topology,
    format: &BlockFormat,
) -> Result<Plan, Error> {
    let plan: Plan = wire::decode(bytes).map_err(fail)?;
    let owners = topology.workers() + usize::from(topology.cuts.is_some());
    if plan.groups.is_empty()
        || plan.groups.len() > format.max_groups
        || plan.groups.len() != plan.owners.len()
        || plan.owners.iter().any(|&id| id >= owners)
    {
        return Err(fail("invalid assignment"));
    }
    Ok(plan)
}

#[allow(clippy::too_many_arguments)]
fn serve_range(
    worker: &Worker<'_>,
    cuts: &[usize],
    assign: &mpsc::Receiver<WireResult<AlignedVec>>,
    topology: &Topology,
    program: &Elf,
    elf: &[u8],
    input: &[u8],
    window: usize,
    trace_seed: [u8; 16],
    opts: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
) -> Result<(), Error> {
    let clock = worker.clock;
    let job = range::Job::new(cuts, worker.id, Some(worker)).with_clock(clock);
    let airs = range::airs(&job, window, trace_seed, opts, format, options);
    let mut built = range::build(
        &job, &airs, program, elf, input, window, opts, format, options,
    )?;
    #[cfg_attr(not(test), allow(unused_mut))]
    let (mut meta, mut counts) = built.describe(&job)?;
    #[cfg(test)]
    if fault_is("missing-range") {
        return Err(fail("injected missing range"));
    }
    #[cfg(test)]
    match std::env::var("LAMBDA_VM_CLUSTER_TEST_FAULT").as_deref() {
        Ok("stream-count-mismatch") => meta.streamed += 1,
        Ok("stream-count-cap") => meta.streamed = BLOCK_MAX_GROUPS,
        Ok("bitwise-increment") => counts.bitwise[0] += 1,
        Ok("bitwise-multiple-rows") => {
            let middle = counts.bitwise.len() / 2;
            counts.bitwise[middle] += 3;
            *counts.bitwise.last_mut().unwrap() += 2;
        }
        Ok("bitwise-short") => {
            counts.bitwise.pop();
        }
        Ok("bitwise-field-limit") => {
            counts.bitwise[0] = math::field::goldilocks::GOLDILOCKS_PRIME;
        }
        Ok("decode-increment") => counts.decode[0].1 += 1,
        Ok("decode-unordered") => counts.decode.swap(0, 1),
        Ok("decode-out-of-range") => counts.decode = vec![(usize::MAX, 1)],
        Ok("keccak-increment") => counts.keccak += 1,
        _ => {}
    }
    worker
        .socket
        .send(
            worker.commit_deadline,
            worker.attempt,
            Kind::RangeReady,
            &meta,
        )
        .map_err(fail)?;
    clock.mark("range-ready-sent");
    worker
        .socket
        .send(
            worker.commit_deadline,
            worker.attempt,
            Kind::RangeCounts,
            &counts,
        )
        .map_err(fail)?;
    clock.mark("range-counts-sent");
    drop(counts);
    clock.mark("wait-assign");
    let bytes = assign
        .recv_timeout(wire::remaining(worker.commit_deadline).map_err(fail)?)
        .map_err(fail)?
        .map_err(fail)?;
    clock.mark("assign-received");
    let plan = decode_assignment(&bytes, topology, format)?;
    let indices: Vec<_> = plan
        .owners
        .iter()
        .enumerate()
        .filter_map(|(group, &owner)| (owner == worker.id).then_some(group))
        .collect();
    if indices.len() != meta.groups.len()
        || !indices.iter().zip(&meta.groups).all(|(&index, group)| {
            let assigned = &plan.groups[index];
            assigned.len() == group.len()
                && assigned.iter().zip(group).all(|(table, &local)| {
                    let (name, shape) = &meta.tables[local as usize];
                    &table.name == name
                        && table.shape == *shape
                        && table.key == Key::Air(local as usize)
                })
        })
    {
        return Err(fail("index base assignment disagrees with range receipt"));
    }
    let shapes = plan
        .groups
        .iter()
        .flatten()
        .map(|table| table.shape)
        .collect::<Vec<_>>();
    let config = format.chain_config(&shapes);
    let groups = plan
        .groups
        .iter()
        .map(|group| vec![0; group.len()])
        .collect::<Vec<_>>();
    crate::with_base_hash!(format.zf.base, |H| {
        let mut early = built.committed.into_iter();
        let owned = indices
            .into_iter()
            .zip(built.groups)
            .map(|(index, group)| {
                if group.index < meta.streamed {
                    let (assigned, committed) = early
                        .next()
                        .ok_or_else(|| fail("missing streamed commitment"))?;
                    if assigned != index {
                        return Err(fail("streamed index base disagrees with assignment"));
                    }
                    return Ok((index, committed));
                }
                let card = worker.card(&plan, index);
                #[cfg(test)]
                let base = if fault_is("wrong-index-base") {
                    index + 1
                } else {
                    index
                };
                #[cfg(not(test))]
                let base = index;
                worker.check_cancelled()?;
                multilinear::gpu::on_indexed_card(card, false, || {
                    range::commit::<H>(group, base, &config, options, format)
                })
                .map(|committed| (index, committed))
            })
            .collect::<Result<Owned<'_>, Error>>()?;
        clock.mark("final-commits-done");
        worker.open_as::<H>(
            &plan,
            owned,
            None,
            Vec::new(),
            &groups,
            &config,
            worker.range_levels.then_some((elf, opts, format)),
        )
    })
}

fn receive_start(
    reader: &mut Reader<'_>,
    deadline: Instant,
    attempt: Attempt,
    bytes: &AlignedVec,
    clock: Clock,
) -> Result<(Start, usize, Vec<u8>, Vec<u8>), Error> {
    let start: Start = wire::decode(bytes).map_err(fail)?;
    if start.settings.len() != BUILD_SETTINGS.len() {
        return Err(fail("invalid build settings"));
    }
    let options = &start.options;
    let window = validate_options(options).map_err(fail)?;
    if let Some(cuts) = &start.cuts {
        range::validate_cuts(cuts, window, options.max_rows.cpu)?;
    }
    if start.elf_len == 0 || start.elf_len > ELF_LIMIT || start.input_len > INPUT_LIMIT {
        return Err(fail("ELF or private input exceeds cluster limits"));
    }
    clock.mark("wait-input");
    let elf = read_input(reader, deadline, attempt, start.elf_len).map_err(fail)?;
    let inputs = read_input(reader, deadline, attempt, start.input_len).map_err(fail)?;
    clock.mark("input-received");
    if hash(&elf) != start.elf_hash || hash(&inputs) != start.input_hash {
        return Err(fail("input digests disagree"));
    }
    Ok((start, window, elf, inputs))
}

fn report_error(socket: &Connection, attempt: Attempt, error: &Error) {
    let reason: String = error.to_string().chars().take(960).collect();
    let _ = socket.send(
        Instant::now() + Duration::from_millis(100),
        attempt,
        Kind::Error,
        &reason,
    );
}

#[cfg(all(test, feature = "cuda", feature = "test-security"))]
mod repeat_tests {
    use super::*;

    fn topology(windows: &str) -> Topology {
        let reserve = || TcpListener::bind("127.0.0.1:0").unwrap();
        let coordinator = reserve();
        let worker = reserve();
        Topology::parse(&format!(
            "coordinator {}\nworker {} 0 {:064x}\ndeadlines-ms 15000 60000 90000\n{windows}",
            coordinator.local_addr().unwrap(),
            worker.local_addr().unwrap(),
            1,
        ))
        .unwrap()
    }

    fn authenticate_worker(
        stream: TcpStream,
        deadline: Instant,
        topology: &Topology,
        opts: &ProofOptions,
        format: &BlockFormat,
    ) -> io::Result<Connection> {
        wire::authenticate(
            stream,
            deadline,
            topology.endpoints[0].token.as_ref().unwrap(),
            &topology.context(0, &profile(opts, format)).unwrap(),
            true,
        )
    }

    #[test]
    #[ignore = "GPU tier: a refused attempt stops the repeated worker"]
    fn refused_first_attempt_stops_worker() {
        let topology = topology("");
        let opts = ProofOptions::default_test_options();
        let format = BlockFormat::pinned();
        std::thread::scope(|scope| {
            let worker =
                scope.spawn(|| serve_attempts(&topology, 0, &opts, &format, 2.try_into().unwrap()));
            let deadline = Instant::now() + topology.startup;
            let address = topology.endpoints[0].address;
            let socket = authenticate_worker(
                connect(address, deadline).unwrap(),
                deadline,
                &topology,
                &opts,
                &format,
            )
            .unwrap();
            let queued = connect(address, deadline).unwrap();
            let attempt = [1; 16];
            let start = Start {
                range_levels: false,
                cuts: None,
                trace_seed: [0; 16],
                options: BlockOptions::production(),
                settings: vec![None; BUILD_SETTINGS.len()],
                elf_len: 0,
                input_len: 0,
                elf_hash: hash(&[]),
                input_hash: hash(&[]),
            };
            socket.send(deadline, attempt, Kind::Start, &start).unwrap();
            let error = socket
                .reader()
                .receive(deadline, Some(attempt), &[Kind::Finish])
                .unwrap_err();
            assert!(
                error.to_string().contains("peer refused attempt: ")
                    && error.to_string().contains("exceeds cluster limits"),
                "{error}"
            );
            assert!(
                authenticate_worker(queued, deadline, &topology, &opts, &format).is_err(),
                "worker authenticated a second connection after refusing Start"
            );
            let error = worker.join().unwrap().unwrap_err();
            assert!(
                error.to_string().contains("exceeds cluster limits"),
                "{error}"
            );
        });
    }

    #[test]
    #[ignore = "GPU tier: replay an earlier input on a repeated worker"]
    fn repeated_worker_discards_an_earlier_attempt_frame() {
        let topology = topology("windows 2");
        let format = crate::lfm::levels::pin_family(
            BlockFormat::pinned_with(GroupFold::Indexed),
            WlfmFamily::B3,
            B3Rounds::Fri,
        )
        .unwrap();
        let opts = crate::lfm::proof::block_base_options_under(&format.zf);
        let mut options = BlockOptions::production();
        options.window_log2 = Some(2);
        options.max_rows.cpu = 8;
        let elf = crate::test_utils::asm_elf_bytes("test_commit_split");
        let program = Elf::load(&elf).unwrap();
        let inputs = [1];
        let prove = |coordinator: &Coordinator<'_>| {
            let proof = crate::base_only::with_blake3_levels(|| {
                prove_ranges_as(
                    coordinator,
                    &[2],
                    &program,
                    &elf,
                    &inputs,
                    4,
                    &opts,
                    &format,
                    &options,
                    &|_, _| {},
                    &|_| {},
                    &mut BlockStamps::default(),
                )
            })
            .unwrap();
            assert!(matches!(proof, Proved::Levels(_)));
        };
        std::thread::scope(|scope| {
            let worker =
                scope.spawn(|| serve_attempts(&topology, 0, &opts, &format, 2.try_into().unwrap()));
            let first =
                Coordinator::start_as(&topology, &elf, &inputs, &opts, &format, &options, true)
                    .unwrap();
            prove(&first);
            let Coordinator {
                attempt: earlier,
                start,
                peers,
                ..
            } = first;
            drop(peers);

            let attempt = wire::random().unwrap();
            assert_ne!(attempt, earlier);
            let deadline = Instant::now() + topology.startup;
            let stream = connect(topology.endpoints[0].address, deadline).unwrap();
            let socket = authenticate_worker(stream, deadline, &topology, &opts, &format).unwrap();
            socket.send(deadline, attempt, Kind::Start, &start).unwrap();
            // Authenticate the old Input on the new connection to exercise the
            // attempt ID check even when the frame MAC and sequence are valid.
            send_input(&socket, deadline, earlier, &inputs).unwrap();
            send_input(&socket, deadline, attempt, &elf).unwrap();
            send_input(&socket, deadline, attempt, &inputs).unwrap();
            prove(&Coordinator {
                topology: &topology,
                attempt,
                start,
                peers: vec![socket],
                commit_deadline: Instant::now() + topology.commit,
                clock: Clock::default(),
            });
            worker.join().unwrap().unwrap();
        });
    }
}

#[cfg(all(test, not(feature = "cuda")))]
mod tests {
    use super::*;

    #[test]
    fn a_worker_leaf_is_refused_unless_once_from_an_owner_for_the_bound_plan() {
        // Three workers and the coordinator (machine 3); worker 1 owns no leaf.
        let owners = [3, 0, 2];
        let bound = [7; 32];
        let refused = |replies: &mut [Option<u32>], id, plan, reply| {
            let before = replies.to_vec();
            let error = take_leaf_reply(replies, &owners, &bound, id, plan, reply).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("unexpected, duplicate or stale worker leaf"),
                "{error}"
            );
            assert_eq!(replies, &before[..], "a refused leaf was kept");
        };
        let mut replies = vec![None; 3];
        take_leaf_reply(&mut replies, &owners, &bound, 0, bound, 10).unwrap();
        // Stale: another attempt's plan.
        refused(&mut replies, 2, [8; 32], 12);
        // Duplicate: a second leaf from worker 0.
        refused(&mut replies, 0, bound, 11);
        // Unexpected: worker 1 owns no leaf.
        refused(&mut replies, 1, bound, 13);
        // Unexpected: a reply in the coordinator's own name, and past the peers.
        refused(&mut replies, 3, bound, 14);
        refused(&mut replies, 9, bound, 15);
        take_leaf_reply(&mut replies, &owners, &bound, 2, bound, 12).unwrap();
        assert_eq!(replies, [Some(10), None, Some(12)]);
    }

    #[test]
    fn a_missing_worker_leaf_is_refused_and_leaves_keep_leaf_order() {
        let owners = [3, 0, 2];
        let mut replies = vec![Some(10), None, None];
        let error = leaves_in_order(&owners, 0, 30, &mut replies, Ok).unwrap_err();
        assert!(
            error.to_string().contains("missing worker 2 leaf"),
            "{error}"
        );
        let mut replies = vec![Some(10), None, Some(12)];
        let leaves = leaves_in_order(&owners, 0, 30, &mut replies, Ok).unwrap();
        assert_eq!(leaves, [30, 10, 12]);
        assert_eq!(replies, [None, None, None], "every reply is taken once");
        let mut replies = vec![Some(10), None, Some(12)];
        let error = leaves_in_order(&[3, 0, 9], 0, 30, &mut replies, Ok).unwrap_err();
        assert!(
            error.to_string().contains("missing worker 9 leaf"),
            "{error}"
        );
    }

    #[test]
    fn cluster_host_rejects_invalid_start_before_execution() {
        let opts = ProofOptions::default_test_options();
        let format = BlockFormat::pinned();
        for (window, elf_len, want) in [
            (63, 1, "window_log2 exceeds worker limits"),
            (BLOCK_WINDOW_LOG2, ELF_LIMIT + 1, "exceeds cluster limits"),
            (BLOCK_WINDOW_LOG2, 1, "input digests disagree"),
        ] {
            let reserve = || TcpListener::bind("127.0.0.1:0").unwrap();
            let coordinator = reserve();
            let worker = reserve();
            let topology = Topology::parse(&format!(
                "coordinator {}\nworker {} 0 {}\ndeadlines-ms 2000 2000 2000",
                coordinator.local_addr().unwrap(),
                worker.local_addr().unwrap(),
                "ab".repeat(32),
            ))
            .unwrap();
            drop(worker);
            std::thread::scope(|scope| {
                let worker = scope.spawn(|| serve(&topology, 0, &opts, &format));
                let deadline = Instant::now() + topology.startup;
                let endpoint = &topology.endpoints[0];
                let socket = wire::authenticate(
                    connect(endpoint.address, deadline).unwrap(),
                    deadline,
                    endpoint.token.as_ref().unwrap(),
                    &topology.context(0, &profile(&opts, &format)).unwrap(),
                    true,
                )
                .unwrap();
                let attempt = [7; 16];
                let start = Start {
                    range_levels: false,
                    cuts: None,
                    trace_seed: [0; 16],
                    options: BlockOptions {
                        window_log2: Some(window),
                        ..BlockOptions::production()
                    },
                    settings: vec![None; BUILD_SETTINGS.len()],
                    elf_len,
                    input_len: 0,
                    elf_hash: [0; 32],
                    input_hash: hash(&[]),
                };
                socket.send(deadline, attempt, Kind::Start, &start).unwrap();
                if want == "input digests disagree" {
                    send_input(&socket, deadline, attempt, &[1]).unwrap();
                }
                let error = socket
                    .reader()
                    .receive(deadline, Some(attempt), &[Kind::Finish])
                    .unwrap_err();
                assert!(error.to_string().contains(want), "{error}");
                let error = worker.join().unwrap().unwrap_err();
                assert!(error.to_string().contains(want), "{error}");
            });
        }
    }

    #[test]
    fn cluster_host_proves_over_tcp() {
        let reserve = || TcpListener::bind("127.0.0.1:0").unwrap();
        let sockets = [reserve(), reserve(), reserve()];
        let [coordinator, first, second] = sockets
            .each_ref()
            .map(|socket| socket.local_addr().unwrap());
        let topology = Topology::parse(&format!(
            "coordinator {coordinator}\nworker {first} 0 {}\nworker {second} 0 {}\ndeadlines-ms 5000 10000 25000",
            "ab".repeat(32),
            "cd".repeat(32),
        )).unwrap();
        drop(sockets);
        let elf = crate::test_utils::asm_elf_bytes("test_commit_4");
        let opts = ProofOptions::default_test_options();
        let format = crate::tests::block_whir_p1_tests::many_groups(BaseFormat::B3_WHIR);
        let mut options = crate::tests::block_whir_p1_tests::options(3);
        options.window_log2 = Some(4);
        options.max_rows.cpu = 16;
        std::thread::scope(|scope| {
            let workers = [0, 1].map(|id| {
                let (topology, opts, format) = (&topology, &opts, &format);
                scope.spawn(move || serve(topology, id, opts, format))
            });
            let (proof, _) = crate::base_only::with_blake3_levels(|| {
                coordinate(&topology, || {
                    prove_block_whir_with(
                        &elf,
                        &[],
                        &opts,
                        &format,
                        &options,
                        &Deviations::default(),
                    )
                })
            })
            .expect("host cluster proves");
            for worker in workers {
                worker.join().unwrap().expect("worker completed");
            }
            assert!(
                crate::base_only::with_blake3_levels(|| {
                    verify_block_whir(&proof, &elf, &opts, &format)
                })
                .unwrap()
            );
        });
    }

    /// The wait for a repeated worker's later attempts: the topology's
    /// `next-attempt-ms`, else [`NEXT_ATTEMPT_WAIT`]; a zero, an out-of-range
    /// or a second value is refused.
    #[test]
    fn the_next_attempt_wait_is_the_topologys_or_the_default() {
        let text = |extra: &str| {
            format!(
                "coordinator 127.0.0.1:1\nworker 127.0.0.1:2 0 {}\n{extra}",
                "ab".repeat(32)
            )
        };
        let topology = Topology::parse(&text("")).unwrap();
        assert_eq!(topology.next_attempt, NEXT_ATTEMPT_WAIT);
        assert_eq!(topology.startup, Duration::from_secs(5));
        assert!(NEXT_ATTEMPT_WAIT >= Duration::from_secs(300));
        let topology = Topology::parse(&text("next-attempt-ms 120000\n")).unwrap();
        assert_eq!(topology.next_attempt, Duration::from_secs(120));
        for bad in [
            "next-attempt-ms 0\n",
            "next-attempt-ms 3600001\n",
            "next-attempt-ms x\n",
            "next-attempt-ms 1000\nnext-attempt-ms 1000\n",
        ] {
            assert!(Topology::parse(&text(bad)).is_err(), "{bad:?}");
        }
    }

    /// ★ A worker serving two attempts serves the second when the
    /// coordinator starts it well after the worker's startup limit, as it
    /// does after proving the join and checking its file: both proofs verify
    /// and the worker ends cleanly.
    ///
    /// Mutation: the worker waits only startup for the second attempt, gives
    /// up before it is dialed, and the second proof fails.
    #[test]
    fn a_repeated_worker_serves_an_attempt_started_after_its_startup_limit() {
        let reserve = || TcpListener::bind("127.0.0.1:0").unwrap();
        let sockets = [reserve(), reserve()];
        let [coordinator, worker] = sockets
            .each_ref()
            .map(|socket| socket.local_addr().unwrap());
        let topology = Topology::parse(&format!(
            "coordinator {coordinator}\nworker {worker} 0 {}\ndeadlines-ms 3000 10000 25000",
            "ab".repeat(32),
        ))
        .unwrap();
        drop(sockets);
        let elf = crate::test_utils::asm_elf_bytes("test_commit_4");
        let opts = ProofOptions::default_test_options();
        let format = crate::tests::block_whir_p1_tests::many_groups(BaseFormat::B3_WHIR);
        let mut options = crate::tests::block_whir_p1_tests::options(3);
        options.window_log2 = Some(4);
        options.max_rows.cpu = 16;
        // Longer than startup: the coordinator's work between two runs.
        let between = 2 * topology.startup + Duration::from_secs(1);
        std::thread::scope(|scope| {
            let worker =
                scope.spawn(|| serve_attempts(&topology, 0, &opts, &format, 2.try_into().unwrap()));
            let prove = || {
                crate::base_only::with_blake3_levels(|| {
                    coordinate(&topology, || {
                        prove_block_whir_with(
                            &elf,
                            &[],
                            &opts,
                            &format,
                            &options,
                            &Deviations::default(),
                        )
                    })
                })
            };
            let (first, _) = prove().expect("the first attempt proves");
            std::thread::sleep(between);
            let second = prove();
            let served = worker.join().unwrap();
            let (second, _) = second.expect("the second attempt, after the wait, proves");
            served.expect("the worker served both attempts");
            for proof in [first, second] {
                assert!(
                    crate::base_only::with_blake3_levels(|| {
                        verify_block_whir(&proof, &elf, &opts, &format)
                    })
                    .unwrap()
                );
            }
        });
    }
}
