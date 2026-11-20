//! What running a recursion tree needs besides its programs: proving a level's
//! proofs a bounded number at once and handing them back in index order
//! ([`in_index_order`]). The W3 driver ([`super::whir_block_tree`]) and the
//! tree suites share it.

/// Run `task` over `0..n` on `workers` threads, and return the results **in
/// index order** whatever order they finished in.
///
/// ★★★ THE ORDER IS THE SOUNDNESS PROPERTY, not a convenience. A level's
/// children, layouts and label runs are three parallel vectors, and a node at
/// the level above takes a contiguous SUBSLICE of each — which is exactly what
/// makes contiguity across sibling subtrees a consequence of the label pins
/// rather than a check of its own. Drain them in completion order and the pins
/// still verify, one subtree at a time, while the tree they describe is not the
/// tree that was built. ⇒ results land in per-index slots and are drained by
/// index, so nothing downstream can observe that a scheduler ran at all.
///
/// `workers <= 1` runs `task` inline, on this thread, in order: the control arm
/// is the original path and not this function with one worker.
///
/// # Panics
///
/// Re-raises the FIRST worker panic on the caller's thread, payload intact.
/// ⚠ `std::thread::scope` otherwise propagates with the fixed string "a scoped
/// thread panicked", which names neither the cause nor its location, and
/// libtest's global hook files a spawned thread's own message against no test
/// and drops it on the floor. The prover's `run_admitted` learned that the
/// expensive way — eleven anonymous failures in one suite run.
pub(crate) fn in_index_order<T: Send>(
    n: usize,
    workers: usize,
    task: impl Fn(usize) -> T + Sync,
) -> Vec<T> {
    let slots: Vec<std::sync::Mutex<Option<T>>> =
        (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    if workers <= 1 {
        for (j, slot) in slots.iter().enumerate() {
            *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(task(j));
        }
    } else {
        let cursor = std::sync::atomic::AtomicUsize::new(0);
        let first_panic: std::sync::Mutex<Option<Box<dyn std::any::Any + Send>>> =
            std::sync::Mutex::new(None);
        std::thread::scope(|scope| {
            for _ in 0..workers {
                let (cursor, slots, first_panic, task) = (&cursor, &slots, &first_panic, &task);
                scope.spawn(move || {
                    // ⛔ THIS THREAD IS PART OF THIS LEVEL. Without the enrolment
                    // its artifact builds are not counted and the level reports
                    // fewer proofs than it made.
                    let _enrolled = super::program_census::enrol();
                    loop {
                        let j = cursor.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if j >= slots.len() {
                            break;
                        }
                        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| task(j))) {
                            Ok(out) => {
                                *slots[j].lock().unwrap_or_else(|e| e.into_inner()) = Some(out);
                            }
                            Err(payload) => {
                                let mut first =
                                    first_panic.lock().unwrap_or_else(|e| e.into_inner());
                                if first.is_none() {
                                    *first = Some(payload);
                                }
                                break;
                            }
                        }
                    }
                });
            }
        });
        if let Some(payload) = first_panic.into_inner().unwrap_or_else(|e| e.into_inner()) {
            std::panic::resume_unwind(payload);
        }
    }
    slots
        .into_iter()
        .enumerate()
        .map(|(j, slot)| {
            // Every index ran (a panicking task was re-raised above).
            slot.into_inner()
                .unwrap_or_else(|e| e.into_inner())
                .unwrap_or_else(|| panic!("index {j} produced no result"))
        })
        .collect()
}
