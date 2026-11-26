//! The hash state of the tables that deduplicate their operations through a
//! `HashMap` and lay their rows out in its iteration order: LT, BYTEWISE,
//! BRANCH, MUL and DVRM (EQ keeps one row an operation since main's
//! multiplicity fix).
//!
//! Every such map in the process hashes with one state, so a table's row order
//! is a function of its operation list: two builds of the same run in one
//! process lay out the same rows in the same order, which regenerating a
//! streamed chunk needs. The keys are random per process, as `HashMap::new()`'s
//! are, unless `LAMBDA_VM_FIXED_TRACE_HASH=1` fixes them (SipHash-1-3 with zero
//! keys): then the traces, and under `LAMBDA_VM_DETERMINISTIC_GRIND` the proofs,
//! are the same in every process. Fixed keys let a program that knows them
//! steer many operations into one bucket, so they are a measurement setting.

use std::cell::Cell;
use std::collections::hash_map::{DefaultHasher, RandomState};
use std::hash::{BuildHasher, Hasher};
use std::sync::OnceLock;

#[derive(Clone, Debug)]
pub enum TraceHashState {
    Process(RandomState),
    Fixed,
    Attempt([u8; 16]),
}

/// A portable seed for a cluster attempt. The zero seed preserves the
/// existing fixed-hash archive bytes.
pub(crate) type Seed = Option<[u8; 16]>;

thread_local! {
    static SEED: Cell<Seed> = const { Cell::new(None) };
}

pub(crate) fn with_seed<T>(seed: Seed, run: impl FnOnce() -> T) -> T {
    let Some(seed) = seed else { return run() };
    struct Reset(Seed);
    impl Drop for Reset {
        fn drop(&mut self) {
            SEED.set(self.0);
        }
    }
    let _reset = Reset(SEED.replace(Some(seed)));
    run()
}

impl TraceHashState {
    /// SipHash-1-3 with zero keys: the same hashes in every process.
    pub fn fixed() -> Self {
        Self::Fixed
    }
}

impl BuildHasher for TraceHashState {
    type Hasher = DefaultHasher;

    fn build_hasher(&self) -> DefaultHasher {
        match self {
            Self::Process(state) => state.build_hasher(),
            Self::Fixed => DefaultHasher::new(),
            Self::Attempt(key) => {
                let mut hasher = DefaultHasher::new();
                hasher.write(key);
                hasher
            }
        }
    }
}

/// The process's state (see the module docs), read from
/// `LAMBDA_VM_FIXED_TRACE_HASH` once.
pub fn trace_hash_state() -> TraceHashState {
    if let Some(seed) = SEED.get() {
        return if seed == [0; 16] {
            TraceHashState::Fixed
        } else {
            TraceHashState::Attempt(seed)
        };
    }
    static STATE: OnceLock<TraceHashState> = OnceLock::new();
    STATE
        .get_or_init(|| {
            if std::env::var("LAMBDA_VM_FIXED_TRACE_HASH").is_ok_and(|v| v.trim() == "1") {
                TraceHashState::fixed()
            } else {
                TraceHashState::Process(RandomState::new())
            }
        })
        .clone()
}

/// A deduplicating table's operation map, hashed under [`trace_hash_state`].
pub type OpMap<K, V> = std::collections::HashMap<K, V, TraceHashState>;
