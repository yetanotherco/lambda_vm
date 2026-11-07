//! The lean pass split over two threads, a measurement's arm (test code): the
//! FRONT decodes each cycle, keeps the register cells and records the memory's
//! accesses in order; the BACK applies them to the cells and records each
//! window's entries. Together they are the pass, access for access; each
//! alone does part of its work, so a run whose pass binds can read how far a
//! split moves the floor.

use std::sync::Arc;

use executor::vm::logs::Log;

use super::super::super::RegisterState;
use super::{Accesses, Bytes, Decode, Registers, pass_window};
use crate::Error;
use crate::tables::decode::DecodeTable;

/// One memory access the front recorded.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Access {
    /// [`Accesses::load_or_store`].
    LoadOrStore { base: u64, count: u8, ts: u64 },
    /// [`Accesses::access`].
    Bytes {
        base: u64,
        count: u32,
        write: Option<u64>,
    },
}

/// The front's sink: the accesses, in order.
struct Recorder<'a>(&'a mut Vec<Access>);

impl Accesses for Recorder<'_> {
    fn access(&mut self, base: u64, count: usize, write: Option<u64>) {
        self.0.push(Access::Bytes {
            base,
            count: count as u32,
            write,
        });
    }

    fn load_or_store(&mut self, base: u64, count: usize, ts: u64) {
        self.0.push(Access::LoadOrStore {
            base,
            count: count as u8,
            ts,
        });
    }
}

/// The decode and the register cells, the accesses recorded.
pub(crate) struct Front {
    decode: Decode,
    registers: Registers,
}

impl Front {
    pub(crate) fn new(decode: Arc<DecodeTable>, registers: &RegisterState) -> Result<Self, Error> {
        Ok(Self {
            decode: Decode::new(decode)?,
            registers: Registers::of(registers),
        })
    }

    /// One window's logs from cycle `first`, its accesses appended to `out`.
    pub(crate) fn pass(
        &mut self,
        logs: &[Log],
        first: usize,
        out: &mut Vec<Access>,
    ) -> Result<(), Error> {
        pass_window::<_, true>(
            &self.decode,
            &mut Recorder(out),
            &mut self.registers,
            logs,
            first,
        )
    }

    /// The register cells now, as the walk's.
    pub(crate) fn registers(&self) -> RegisterState {
        self.registers.state()
    }
}

/// The cells, the front's accesses applied.
pub(crate) struct Back {
    bytes: Bytes,
}

impl Back {
    pub(crate) fn new() -> Self {
        Self {
            bytes: Bytes::new(),
        }
    }

    /// Starts window `window`, its entries cleared.
    pub(crate) fn begin(&mut self, window: u32) -> Result<(), Error> {
        self.bytes.begin(window)
    }

    /// The front's accesses, in order.
    pub(crate) fn apply(&mut self, accesses: &[Access]) {
        for &access in accesses {
            match access {
                Access::LoadOrStore { base, count, ts } => {
                    self.bytes.load_or_store(base, usize::from(count), ts)
                }
                Access::Bytes { base, count, write } => {
                    self.bytes.access(base, count as usize, write)
                }
            }
        }
    }

    /// The window's entries, as [`super::LeanPass::entries`].
    pub(crate) fn entries(&self) -> &[(u64, u64)] {
        &self.bytes.entries
    }
}
