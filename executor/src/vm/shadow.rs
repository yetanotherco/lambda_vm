//! A sampled shadow check of a run on the fast engine: every `N`-th window is
//! executed again by the reference [`Executor`] on a thread of its own, from
//! the state the run published at the window's start, and its logs, end pc
//! and end registers must equal the run's. A mismatch is kept as an error
//! naming the window and the cycle; the caller polls [`ShadowCheck::failure`]
//! and stops the run on one.
//!
//! The run never waits for the check: a sample that arrives while the check
//! is still busy with the previous one is skipped and counted.
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use super::{
    execution::{Executor, ExecutorError},
    logs::Log,
    memory::MemorySnapshot,
    registers::Registers,
};
use crate::elf::Elf;

/// The knob: check every `N`-th window (`0` or unset: off).
pub const SHADOW_KNOB: &str = "LAMBDA_VM_EXEC_SHADOW";

/// The window-start state a sample is re-executed from.
pub struct WindowStart {
    /// The window's number in the run (0 for the first).
    pub window: usize,
    /// The run's cycle count at the window's start.
    pub first_cycle: u64,
    pub pc: u64,
    pub registers: Registers,
    pub memory: MemorySnapshot,
    pub public_output: Vec<u8>,
}

/// One sampled window: its start, and what the run wrote.
struct Sample {
    start: WindowStart,
    logs: Vec<Log>,
    end_pc: u64,
    end_registers: Registers,
}

/// What the check did over a run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShadowCounts {
    /// Windows re-executed and found equal.
    pub checked: usize,
    /// Sampled windows skipped because the check was busy.
    pub skipped: usize,
}

/// The sampled shadow check (see the module docs).
pub struct ShadowCheck {
    every: usize,
    tx: Option<SyncSender<Sample>>,
    failure: Arc<Mutex<Option<String>>>,
    thread: Option<JoinHandle<usize>>,
    skipped: usize,
}

impl ShadowCheck {
    /// The sampling period a knob value names: unset, empty or `0` is off
    /// (`None`), `N > 0` every `N`-th window; anything else is refused.
    pub fn every_from_value(value: Option<&str>) -> Result<Option<usize>, String> {
        match value.map(str::trim) {
            None | Some("") | Some("0") => Ok(None),
            Some(v) => v
                .parse::<usize>()
                .map(Some)
                .map_err(|_| format!("{SHADOW_KNOB}={v:?}: expected a window count")),
        }
    }

    /// [`ShadowCheck::every_from_value`] of the knob.
    pub fn every_from_env() -> Result<Option<usize>, String> {
        Self::every_from_value(std::env::var(SHADOW_KNOB).ok().as_deref())
    }

    /// A check of every `every`-th window of a run of `program`, its thread
    /// started (with its own reference executor, the program decoded once).
    pub fn new(program: &Elf, every: usize) -> Result<Self, ExecutorError> {
        let reference = Executor::new(program, Vec::new())?;
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let failure = Arc::new(Mutex::new(None));
        let thread = {
            let failure = Arc::clone(&failure);
            std::thread::Builder::new()
                .name("exec-shadow".into())
                .spawn(move || check(reference, rx, failure))
                .map_err(|_| crate::vm::memory::MemoryError::AllocationFailed)?
        };
        Ok(Self {
            every: every.max(1),
            tx: Some(tx),
            failure,
            thread: Some(thread),
            skipped: 0,
        })
    }

    /// Whether window `window` is sampled.
    pub fn samples(&self, window: usize) -> bool {
        window.is_multiple_of(self.every)
    }

    /// Hands a sampled window to the check: its start, the logs the run
    /// wrote for it and the state after it. Never waits; a window the busy
    /// check cannot take is skipped.
    pub fn submit(
        &mut self,
        start: WindowStart,
        logs: &[Log],
        end_pc: u64,
        end_registers: Registers,
    ) {
        let Some(tx) = &self.tx else {
            return;
        };
        let sample = Sample {
            start,
            logs: logs.to_vec(),
            end_pc,
            end_registers,
        };
        match tx.try_send(sample) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => self.skipped += 1,
            // The check stopped (on a mismatch): its failure says why.
            Err(TrySendError::Disconnected(_)) => self.tx = None,
        }
    }

    /// The first mismatch found, if any.
    pub fn failure(&self) -> Option<String> {
        self.failure.lock().ok().and_then(|f| f.clone())
    }

    /// Waits for the samples handed over to be checked: the counts, or the
    /// mismatch.
    pub fn finish(mut self) -> Result<ShadowCounts, String> {
        drop(self.tx.take());
        let checked = match self.thread.take().map(JoinHandle::join) {
            Some(Ok(checked)) => checked,
            Some(Err(_)) => return Err("shadow check: its thread panicked".into()),
            None => 0,
        };
        match self.failure() {
            Some(failure) => Err(failure),
            None => Ok(ShadowCounts {
                checked,
                skipped: self.skipped,
            }),
        }
    }
}

impl Drop for ShadowCheck {
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// The check's thread: each sample re-executed from its start and compared;
/// stops at the first mismatch. Returns the samples found equal.
fn check(
    mut reference: Executor,
    rx: Receiver<Sample>,
    failure: Arc<Mutex<Option<String>>>,
) -> usize {
    let mut checked = 0;
    for sample in rx {
        if let Err(mismatch) = compare(&mut reference, &sample) {
            if let Ok(mut slot) = failure.lock() {
                *slot = Some(mismatch);
            }
            break;
        }
        checked += 1;
    }
    checked
}

/// One sample on the reference: `Err` names the first difference.
fn compare(reference: &mut Executor, sample: &Sample) -> Result<(), String> {
    let start = &sample.start;
    let at = |offset: usize| {
        format!(
            "shadow check: window {} (cycle {})",
            start.window,
            start.first_cycle + offset as u64
        )
    };
    reference.restore(
        start.pc,
        start.registers.clone(),
        &start.memory,
        start.public_output.clone(),
    );
    let want = match reference.resume_with_limit(sample.logs.len()) {
        Ok(Some(logs)) => logs,
        Ok(None) => return Err(format!("{}: the reference ran no cycle", at(0))),
        Err(e) => return Err(format!("{}: the reference failed: {e}", at(0))),
    };
    let record = |l: &Log| [l.current_pc, l.next_pc, l.src1_val, l.src2_val, l.dst_val];
    if let Some(i) = (0..want.len().max(sample.logs.len()))
        .find(|&i| want.get(i).map(record) != sample.logs.get(i).map(record))
    {
        return Err(format!(
            "{}: the fast engine wrote {:x?}, the reference {:x?}",
            at(i),
            sample.logs.get(i).map(record),
            want.get(i).map(record)
        ));
    }
    if reference.pc() != sample.end_pc
        || reference.registers().to_array() != sample.end_registers.to_array()
    {
        return Err(format!(
            "{}: the state after the window differs (pc {:#x} vs the reference's {:#x})",
            at(sample.logs.len()),
            sample.end_pc,
            reference.pc()
        ));
    }
    Ok(())
}
