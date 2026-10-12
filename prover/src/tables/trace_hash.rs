//! The hash state of the tables that deduplicate their operations through a
//! `HashMap` and lay their rows out in its iteration order: LT, EQ, BYTEWISE,
//! BRANCH, MUL and DVRM.
//!
//! Every such map in the process hashes with one state, so a table's row order
//! is a function of its operation list: two builds of the same run in one
//! process lay out the same rows in the same order, which regenerating a
//! streamed chunk needs. The keys are random per process, as `HashMap::new()`'s
//! are, unless `LAMBDA_VM_FIXED_TRACE_HASH=1` fixes them (SipHash-1-3 with zero
//! keys): then the traces, and under `LAMBDA_VM_DETERMINISTIC_GRIND` the proofs,
//! are the same in every process. Fixed keys let a program that knows them
//! steer many operations into one bucket, so they are a measurement setting.

use std::collections::hash_map::{DefaultHasher, RandomState};
use std::hash::BuildHasher;
use std::sync::OnceLock;

/// See the module docs: `None` is the fixed state.
#[derive(Clone, Debug)]
pub struct TraceHashState(Option<RandomState>);

impl TraceHashState {
    /// SipHash-1-3 with zero keys: the same hashes in every process.
    pub fn fixed() -> Self {
        Self(None)
    }
}

impl BuildHasher for TraceHashState {
    type Hasher = DefaultHasher;

    fn build_hasher(&self) -> DefaultHasher {
        match &self.0 {
            Some(state) => state.build_hasher(),
            None => DefaultHasher::new(),
        }
    }
}

/// The process's state (see the module docs), read from
/// `LAMBDA_VM_FIXED_TRACE_HASH` once.
pub fn trace_hash_state() -> TraceHashState {
    static STATE: OnceLock<TraceHashState> = OnceLock::new();
    STATE
        .get_or_init(|| {
            if std::env::var("LAMBDA_VM_FIXED_TRACE_HASH").is_ok_and(|v| v.trim() == "1") {
                TraceHashState::fixed()
            } else {
                TraceHashState(Some(RandomState::new()))
            }
        })
        .clone()
}

/// A deduplicating table's operation map, hashed under [`trace_hash_state`].
pub type OpMap<K, V> = std::collections::HashMap<K, V, TraceHashState>;
