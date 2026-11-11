//! Which interpreter runs the guest: the reference [`Executor`] or the
//! [`FastExecutor`], chosen by `LAMBDA_VM_EXEC_ENGINE` (`reference`, the
//! default, or `fast`). Both write the same logs and publish the same
//! window-start memory; [`Engine`] gives a caller one type for either.
use super::{
    execution::{Executor, ExecutorError, ReturnValues},
    fast::FastExecutor,
    history::{HistExtra, HistoryFinal, WindowHistory},
    logs::Log,
    memory::{GuestMemory, MemorySnapshot},
    registers::Registers,
};
use crate::elf::Elf;

/// The knob that picks the engine.
pub const ENGINE_KNOB: &str = "LAMBDA_VM_EXEC_ENGINE";

/// An interpreter of the guest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineKind {
    /// [`Executor`]: the reference, which defines a run.
    Reference,
    /// [`FastExecutor`]: the tight interpreter.
    Fast,
}

impl EngineKind {
    /// The engine a knob value names: unset or empty, the reference; `reference`
    /// or `fast`; anything else is refused, naming the value.
    pub fn from_value(value: Option<&str>) -> Result<Self, String> {
        match value.map(str::trim) {
            None | Some("") | Some("reference") => Ok(Self::Reference),
            Some("fast") => Ok(Self::Fast),
            Some(other) => Err(format!(
                "{ENGINE_KNOB}={other:?}: expected `reference` or `fast`"
            )),
        }
    }

    /// [`EngineKind::from_value`] of the knob.
    pub fn from_env() -> Result<Self, String> {
        Self::from_value(std::env::var(ENGINE_KNOB).ok().as_deref())
    }

    /// The knob value naming this engine.
    pub fn name(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Fast => "fast",
        }
    }
}

/// A run on either engine.
pub enum Engine {
    Reference(Box<Executor>),
    Fast(Box<FastExecutor>),
}

impl Engine {
    /// A run of `program` on `private_inputs` on the engine `kind`.
    pub fn new(
        kind: EngineKind,
        program: &Elf,
        private_inputs: Vec<u8>,
    ) -> Result<Self, ExecutorError> {
        Ok(match kind {
            EngineKind::Reference => {
                Self::Reference(Box::new(Executor::new(program, private_inputs)?))
            }
            EngineKind::Fast => Self::Fast(Box::new(FastExecutor::new(program, private_inputs)?)),
        })
    }

    /// Which engine this is.
    pub fn kind(&self) -> EngineKind {
        match self {
            Self::Reference(_) => EngineKind::Reference,
            Self::Fast(_) => EngineKind::Fast,
        }
    }

    /// At most `limit` cycles, their logs lent; `None` once finished.
    pub fn resume_with_limit(&mut self, limit: usize) -> Result<Option<&[Log]>, ExecutorError> {
        match self {
            Self::Reference(e) => e.resume_with_limit(limit),
            Self::Fast(e) => e.resume_with_limit(limit),
        }
    }

    /// At most `limit` cycles, their logs handed out; `None` once finished.
    pub fn resume_owned(&mut self, limit: usize) -> Result<Option<Vec<Log>>, ExecutorError> {
        match self {
            Self::Reference(e) => e.resume_owned(limit),
            Self::Fast(e) => e.resume_owned(limit),
        }
    }

    /// Turns the fast engine's history mode on
    /// ([`FastExecutor::enable_history`]); the reference has none.
    pub fn enable_history(&mut self) -> Result<(), ExecutorError> {
        match self {
            Self::Reference(_) => Err(ExecutorError::History(
                "history mode needs the fast engine".into(),
            )),
            Self::Fast(e) => e.enable_history(),
        }
    }

    /// [`FastExecutor::resume_history`].
    pub fn resume_history(
        &mut self,
        limit: usize,
    ) -> Result<Option<(Vec<Log>, WindowHistory)>, ExecutorError> {
        match self {
            Self::Reference(_) => Err(ExecutorError::History(
                "history mode needs the fast engine".into(),
            )),
            Self::Fast(e) => e.resume_history(limit),
        }
    }

    /// [`FastExecutor::take_history_final`].
    pub fn take_history_final(&mut self) -> Option<HistoryFinal> {
        match self {
            Self::Reference(_) => None,
            Self::Fast(e) => e.take_history_final(),
        }
    }

    /// [`FastExecutor::recycle_extras`] (the reference keeps no history).
    pub fn recycle_extras(&mut self, buffer: Vec<HistExtra>) {
        if let Self::Fast(e) = self {
            e.recycle_extras(buffer);
        }
    }

    /// Gives back a handed-out window's buffer for the next window.
    pub fn recycle(&mut self, buffer: Vec<Log>) {
        match self {
            Self::Reference(e) => e.recycle(buffer),
            Self::Fast(e) => e.recycle(buffer),
        }
    }

    /// The memory as it is now.
    pub fn snapshot(&mut self) -> MemorySnapshot {
        match self {
            Self::Reference(e) => e.memory().snapshot(),
            Self::Fast(e) => e.snapshot(),
        }
    }

    /// The bytes the run's memory takes on the heap.
    pub fn heap_bytes(&self) -> usize {
        match self {
            Self::Reference(e) => e.memory().heap_bytes(),
            Self::Fast(e) => e.memory().heap_bytes(),
        }
    }

    /// Current program counter (0 once halted).
    pub fn pc(&self) -> u64 {
        match self {
            Self::Reference(e) => e.pc(),
            Self::Fast(e) => e.pc(),
        }
    }

    /// Current register state.
    pub fn registers(&self) -> Registers {
        match self {
            Self::Reference(e) => e.registers().clone(),
            Self::Fast(e) => e.registers(),
        }
    }

    /// The bytes committed so far.
    pub fn public_output(&self) -> &[u8] {
        match self {
            Self::Reference(e) => GuestMemory::public_output(e.memory()),
            Self::Fast(e) => e.memory().public_output_bytes(),
        }
    }

    /// The return values (after the run has finished).
    pub fn finish(self) -> Result<ReturnValues, ExecutorError> {
        match self {
            Self::Reference(e) => e.finish(),
            Self::Fast(e) => e.finish(),
        }
    }
}
