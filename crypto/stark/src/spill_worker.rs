//! Async disk-spill worker.
//!
//! A dedicated OS thread that writes LDE byte buffers to tempfiles and
//! returns the resulting `Mmap`. Callers submit a `Box<[u8]>` and receive a
//! `SpillHandle` that resolves to the `Mmap` at a barrier.
//!
//! The channel is bounded at 1 so at most one job is in flight beyond the
//! one being processed: this caps the extra live memory at two spill
//! buffers regardless of how many tables are produced concurrently.
//!
//! `SpillHandle` must be Send + Sync so it can live inside LDETraceTable
//! even when that table is shared across rayon workers via `&`. The
//! condvar-based slot provides both.

#![cfg(feature = "disk-spill")]

use std::io;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread;

struct SpillJob {
    data: Box<[u8]>,
    slot: Arc<SpillSlot>,
}

struct SpillSlot {
    mutex: Mutex<Option<io::Result<memmap2::Mmap>>>,
    cvar: Condvar,
}

pub(crate) struct SpillWorker {
    tx: SyncSender<SpillJob>,
}

impl SpillWorker {
    fn new() -> Self {
        let (tx, rx) = sync_channel::<SpillJob>(1);
        thread::Builder::new()
            .name("lde-spill".to_string())
            .spawn(move || worker_loop(rx))
            .expect("spawn lde-spill worker");
        Self { tx }
    }

    pub(crate) fn global() -> &'static SpillWorker {
        static WORKER: OnceLock<SpillWorker> = OnceLock::new();
        WORKER.get_or_init(SpillWorker::new)
    }

    pub(crate) fn submit(&self, data: Box<[u8]>) -> SpillHandle {
        let slot = Arc::new(SpillSlot {
            mutex: Mutex::new(None),
            cvar: Condvar::new(),
        });
        self.tx
            .send(SpillJob {
                data,
                slot: Arc::clone(&slot),
            })
            .expect("lde-spill worker died");
        SpillHandle { slot }
    }
}

fn worker_loop(rx: Receiver<SpillJob>) {
    for job in rx {
        let result = write_and_mmap(&job.data);
        {
            let mut guard = job.slot.mutex.lock().expect("spill slot poisoned");
            *guard = Some(result);
        }
        job.slot.cvar.notify_all();
    }
}

fn write_and_mmap(data: &[u8]) -> io::Result<memmap2::Mmap> {
    use std::io::Write;

    let file = tempfile::tempfile()?;
    file.set_len(data.len() as u64)?;
    {
        let mut writer = io::BufWriter::with_capacity(crypto::SPILL_BUF_CAPACITY, &file);
        writer.write_all(data)?;
        writer.flush()?;
    }
    // SAFETY: tempfile() creates an anonymous file with no filesystem path,
    // so no other process can open or modify it. The mapping keeps its own
    // reference to the underlying object (Unix: kernel VMA; Windows:
    // duplicated handle in memmap2), so the `file` local can drop at end of
    // scope without invalidating the mapping.
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    Ok(mmap)
}

pub(crate) struct SpillHandle {
    slot: Arc<SpillSlot>,
}

impl SpillHandle {
    pub(crate) fn wait(self) -> io::Result<memmap2::Mmap> {
        let mut guard = self.slot.mutex.lock().expect("spill slot poisoned");
        loop {
            if let Some(result) = guard.take() {
                return result;
            }
            guard = self
                .slot
                .cvar
                .wait(guard)
                .expect("spill slot cvar poisoned");
        }
    }
}
