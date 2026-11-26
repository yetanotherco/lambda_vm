//! One attempt per worker process, with one TCP connection per endpoint.
//! Deploy the same executable and proof settings to every host on a private
//! network. Topology tokens authenticate peers and frames. The link carries
//! private input in plaintext.

use super::wire::{self, Attempt, Connection, Digest32, Kind, Reader};
use super::{super::*, HostAirs, Owned, Pass, Plan};
use crypto::hash::blake3::chain::Blake3Std;
use digest::Digest;
use rkyv::util::AlignedVec;
use stark::multilinear_block::replica::{self, Committed, GroupParts, GroupShape, SealedPlan};
use std::{
    cell::RefCell,
    fs::File,
    io::{self, Read},
    net::{SocketAddr, TcpListener, TcpStream},
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
/// deadlines-ms 5000 30000 30000
/// fallback local
/// ```
///
/// The coordinator dials the workers. Its address identifies it in the
/// handshake. Worker copies may replace every other token with `-`.
/// The optional `deadlines-ms` line sets startup, commit and opening limits
/// (defaults: 5 s, 30 s and 30 s). Set larger limits for larger workloads.
/// Workers allow at most one second per handshake within startup. The final
/// acknowledgement is best effort, with at most one second to collect it.
/// Fallback defaults to off and retries cluster failures locally.
/// Worker hosts need memory for their replicated witness build. Cluster
/// replay disables live regeneration and always admits record-backed tables.
/// Use a private network or a tunnel if the input needs confidentiality.
#[derive(Clone)]
pub struct Topology {
    coordinator: SocketAddr,
    endpoints: Vec<Endpoint>,
    startup: Duration,
    commit: Duration,
    open: Duration,
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
        let (startup, commit, open) = deadlines.unwrap_or((
            Duration::from_secs(5),
            Duration::from_secs(30),
            Duration::from_secs(30),
        ));
        Ok(Self {
            coordinator,
            endpoints,
            startup,
            commit,
            open,
            fallback,
        })
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
const BUILD_SETTINGS: &[&str] = &[
    LOCAL_LANE_ENV,
    multilinear_block::CS_PROBE_ENV,
    multilinear_block::MERGE_PROBE_ENV,
    "LAMBDA_VM_INROW_SIDES_PINNED",
    "LAMBDA_VM_WORD_STREAM",
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

/// Run a complete proving entry point on this thread. On a configured
/// fallback the first invocation has returned and its observers and sockets
/// have been dropped before a fresh invocation starts with local placement.
pub fn coordinate<T, E: std::fmt::Display>(
    topology: &Topology,
    prove: impl Fn() -> Result<T, E>,
) -> Result<T, E> {
    struct Reset(Option<Placement>);
    impl Drop for Reset {
        fn drop(&mut self) {
            DISPATCH.with(|slot| slot.replace(self.0.take()));
        }
    }
    let previous = DISPATCH.with(|slot| {
        slot.replace(Some(Placement::Tcp {
            topology: Arc::new(topology.clone()),
            failed: false,
        }))
    });
    let reset = Reset(previous);
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
        static FALLBACKS: AtomicU64 = AtomicU64::new(0);
        eprintln!("CLUSTER TCP: {error}");
        let count = FALLBACKS.fetch_add(1, Ordering::Relaxed) + 1;
        eprintln!("CLUSTER TCP: starting a fresh local attempt (fallback {count})");
        DISPATCH.with(|slot| slot.replace(Some(Placement::Local)));
        return prove();
    }
    drop(reset);
    result
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct Start {
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
fn validate_options(options: &BlockOptions) -> WireResult<usize> {
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
    let deadline = Instant::now() + topology.startup;
    std::thread::scope(|scope| {
        let workers: Vec<_> = topology
            .endpoints
            .iter()
            .enumerate()
            .map(|(id, endpoint)| {
                scope.spawn(move || {
                    let run = || {
                        let socket = wire::authenticate(
                            connect(endpoint.address, deadline)?,
                            deadline,
                            endpoint.token.as_ref().unwrap(),
                            &topology.context(id, &profile)?,
                            true,
                        )?;
                        socket.send(deadline, attempt, Kind::Start, start)?;
                        send_input(&socket, deadline, attempt, elf)?;
                        send_input(&socket, deadline, attempt, input)?;
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
        let attempt = wire::random().map_err(fail)?;
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
        let start = Start {
            trace_seed,
            options: options.clone(),
            settings,
            elf_len: elf.len(),
            input_len: input.len(),
            elf_hash: hash(elf),
            input_hash: hash(input),
        };
        let peers = start_peers(topology, attempt, elf, input, profile(opts, format), &start)
            .map_err(fail)?;
        Ok(Self {
            topology,
            attempt,
            start,
            peers,
            commit_deadline: Instant::now() + topology.commit,
        })
    }

    pub(in crate::block_whir) fn workers(&self) -> usize {
        self.peers.len()
    }

    pub(in crate::block_whir) fn trace_seed(&self) -> [u8; 16] {
        self.start.trace_seed
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
        request: impl Fn(&Connection, Instant) -> WireResult<()> + Sync,
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
                        request(peer, deadline)?;
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
) -> Result<BlockWhirProof<LocalArgues>, Error> {
    let begin = Instant::now();
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
    let mut prepared = commit_groups_on_host::<H>(
        group_stacks(
            prepared_tables(&frame.airs, &frame.page_configs, format)?,
            &laid.groups,
        )?,
        &frame.config,
    )?;
    let prepared_roots = prepared_roots(&prepared);
    let mut settled = Vec::with_capacity(prepared.len());
    for p in &mut prepared {
        settled.push(multilinear_block::BlockPreparedCheck {
            group: p.group,
            tables: std::mem::take(&mut p.tables),
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
    let expected_witness = plan.witness_digest()?;
    let check_owner = |id, group, digest| {
        if plan.owners.get(group) != Some(&id) || descriptors.get(group) != Some(&digest) {
            return Err(wire::invalid(
                "group disagrees with authenticated assignment",
            ));
        }
        Ok(())
    };
    let assignment = rkyv::to_bytes::<rkyv::rancor::Error>(&plan).map_err(fail)?;
    let commit = Phase {
        name: "commit",
        reply: Kind::Committed,
        end: Kind::CommittedAll,
        duration: wire::remaining(*commit_deadline).map_err(fail)?,
        peers,
        attempt,
    };
    commit
        .run(
            |socket, deadline| socket.write_frame(deadline, Kind::Assign, attempt, &assignment),
            |id, kind, bytes| {
                if kind == Kind::Witness {
                    if witnesses[id] || bytes[..] != expected_witness {
                        return Err(wire::invalid(
                            "replica witness digest disagrees with coordinator",
                        ));
                    }
                    witnesses[id] = true;
                    return Ok(());
                }
                if !witnesses[id] {
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
    if !witnesses.iter().all(|&received| received) {
        return Err(fail("missing replica witness digest"));
    }
    let mut receipts = Vec::with_capacity(slots.len());
    for slot in &slots {
        match slot {
            Slot::Committed(receipt) => receipts.push(receipt.clone()),
            _ => return Err(fail("missing commitment")),
        }
    }
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
                plan: sealed,
                digest,
                challenges,
            },
            post,
        )
    };
    stamps.phase_a += begin.elapsed().as_secs_f64();
    stamps.prepared = (settled.len(), 0.0);
    let mut statement_sent = false;
    let begin = Instant::now();
    let sealed = rkyv::to_bytes::<rkyv::rancor::Error>(&bind).map_err(fail)?;
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
    open.run(
        |socket, deadline| socket.write_frame(deadline, Kind::Bind, attempt, &sealed),
        |id, _, bytes| {
            result_bytes += bytes.len();
            if result_bytes > 512 << 20 {
                return Err(wire::invalid("attempt exceeds result byte budget"));
            }
            let GroupReply {
                group,
                descriptor,
                plan,
                parts,
            } = wire::decode(bytes)?;
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
            completed[id] += 1;
            Ok(())
        },
    )
    .map_err(fail)?;
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
    let finish_deadline = Instant::now() + topology.startup.min(FINISH_TIMEOUT);
    std::thread::scope(|scope| {
        for (id, peer) in peers.iter().enumerate() {
            scope.spawn(move || {
                let result = peer.write_frame(finish_deadline, Kind::Finish, attempt, &[])
                    .and_then(|()| peer.reader().receive(finish_deadline, Some(attempt), &[Kind::Finish]));
                if let Err(error) = result {
                    eprintln!("CLUSTER TCP: worker {id} missing Finish acknowledgement, continuing to proof verification: {error}");
                }
            });
        }
    });
    eprintln!("BLOCK CLUSTER TCP: completed groups by worker {completed:?}");
    stamps.phase_b = begin.elapsed().as_secs_f64();
    stamps.tables = laid.shapes.len();
    stamps.cells = laid.shapes.iter().map(|&(width, n)| width << n).sum();
    Ok(BlockWhirProof {
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
    })
}

pub(in crate::block_whir) struct Worker<'a> {
    socket: &'a Connection,
    attempt: Attempt,
    id: usize,
    cards: Vec<usize>,
    commit_deadline: Instant,
    open: Duration,
    startup: Duration,
    bind: Mutex<mpsc::Receiver<WireResult<AlignedVec>>>,
    finished: Mutex<mpsc::Receiver<WireResult<()>>>,
    cancelled: &'a AtomicBool,
    watchdog: mpsc::Sender<(&'static str, Instant)>,
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
        witness: Digest32,
        prepared: Vec<PreparedColumns>,
        groups: &[Vec<u32>],
        config: &multilinear::whir_chain::ChainConfig,
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
        let bind_deadline = self.commit_deadline + self.startup;
        let _ = self.watchdog.send(("bind", bind_deadline));
        let bind_bytes = self
            .bind
            .lock()
            .unwrap()
            .recv_timeout(wire::remaining(bind_deadline).map_err(fail)?)
            .map_err(fail)?
            .map_err(fail)?;
        let deadline = Instant::now() + self.open;
        let _ = self.watchdog.send(("open", deadline));
        let bind: Bind = wire::decode(&bind_bytes).map_err(fail)?;
        let mut post = H::sponge();
        if bind.plan.statement != plan.statement
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
        #[cfg(test)]
        fault_pause("bind");
        for (
            committed,
            prepared,
            card,
            CommitReply {
                group, descriptor, ..
            },
        ) in private
        {
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
        self.socket
            .write_frame(deadline, Kind::OpenedAll, self.attempt, &[])
            .map_err(fail)?;
        let finish_deadline = deadline + self.startup;
        let _ = self.watchdog.send(("finish", finish_deadline));
        self.finished
            .lock()
            .unwrap()
            .recv_timeout(wire::remaining(finish_deadline).map_err(fail)?)
            .map_err(fail)?
            .map_err(fail)?;
        #[cfg(test)]
        if fault_is("no-finish") {
            return Ok(());
        }
        self.socket
            .write_frame(finish_deadline, Kind::Finish, self.attempt, &[])
            .map_err(fail)?;
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
fn fault_pause(name: &str) {
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

/// Serve a single authenticated proof attempt. Start a fresh worker process
/// for each retry. Launch with the configured LAMBDA_VM_CARDS before CUDA
/// initializes. A stuck worker process exits after the phase deadline and a
/// one-second grace period.
pub fn serve(
    topology: &Topology,
    id: usize,
    opts: &ProofOptions,
    format: &BlockFormat,
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
    let deadline = Instant::now() + topology.startup;
    let socket = loop {
        let left = wire::remaining(deadline).map_err(fail)?;
        match listener.accept() {
            Ok((stream, _)) => {
                let handshake = stream.set_nonblocking(false).and_then(|()| {
                    wire::authenticate(
                        stream,
                        deadline.min(Instant::now() + HANDSHAKE_TIMEOUT),
                        token,
                        &context,
                        false,
                    )
                });
                match handshake {
                    Ok(socket) => break socket,
                    Err(error) => eprintln!("CLUSTER TCP: worker {id} rejected handshake: {error}"),
                }
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                std::thread::sleep(left.min(Duration::from_millis(10)));
            }
            Err(e) => return Err(fail(e)),
        }
    };
    let socket = &socket;
    let mut reader = socket.reader();
    let (attempt, _, bytes) = reader
        .receive(deadline, None, &[Kind::Start])
        .map_err(fail)?;
    let (start, window, elf, inputs) = receive_start(&mut reader, deadline, attempt, &bytes)
        .inspect_err(|error| report_error(socket, attempt, error))?;
    let settings: Arc<[_]> = start.settings.into();
    let options = &start.options;
    let (_, _, assignment) = reader
        .receive(
            Instant::now() + topology.commit,
            Some(attempt),
            &[Kind::Assign],
        )
        .map_err(fail)?;
    let commit_deadline = Instant::now() + topology.commit;
    let plan: Plan = wire::decode(&assignment).map_err(fail)?;
    if plan.groups.is_empty()
        || plan.groups.len() > format.max_groups
        || plan.groups.len() != plan.owners.len()
        || plan.owners.iter().any(|&id| id >= topology.workers())
    {
        return Err(fail("invalid assignment"));
    }
    drop(assignment);
    let cancelled = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let _close = CloseOnDrop(std::slice::from_ref(socket));
        let (watchdog, deadlines) = mpsc::channel();
        scope.spawn(|| watch_deadline(socket, &cancelled, deadlines, "commit", commit_deadline));
        let (btx, bind) = mpsc::sync_channel(1);
        let (ftx, finished) = mpsc::sync_channel(1);
        let cancelled = &cancelled;
        scope.spawn(move || {
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
                cards,
                commit_deadline,
                open: topology.open,
                startup: topology.startup,
                bind: Mutex::new(bind),
                finished: Mutex::new(finished),
                cancelled,
                watchdog,
            };
            let program = Elf::load(&elf).map_err(|e| Error::ElfLoad(format!("{e}")))?;
            let mut airs = HostAirs::new(opts, format, options);
            let trace_seed = start.trace_seed;
            #[cfg(test)]
            let trace_seed = if fault_is("ignore-seed") {
                [0; 16]
            } else {
                trace_seed
            };
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
                        plan: &plan,
                        tcp: Some(&worker),
                    },
                    &airs,
                    &|_, _| {},
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
    })
}

fn receive_start(
    reader: &mut Reader<'_>,
    deadline: Instant,
    attempt: Attempt,
    bytes: &AlignedVec,
) -> Result<(Start, usize, Vec<u8>, Vec<u8>), Error> {
    let start: Start = wire::decode(bytes).map_err(fail)?;
    if start.settings.len() != BUILD_SETTINGS.len() {
        return Err(fail("invalid build settings"));
    }
    let options = &start.options;
    let window = validate_options(options).map_err(fail)?;
    if start.elf_len == 0 || start.elf_len > ELF_LIMIT || start.input_len > INPUT_LIMIT {
        return Err(fail("ELF or private input exceeds cluster limits"));
    }
    let elf = read_input(reader, deadline, attempt, start.elf_len).map_err(fail)?;
    let inputs = read_input(reader, deadline, attempt, start.input_len).map_err(fail)?;
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

#[cfg(all(test, not(feature = "cuda")))]
mod tests {
    use super::*;

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
}
