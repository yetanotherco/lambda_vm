//! The executor's window buffers from the page-locked pool
//! (`LAMBDA_VM_POOL_EXEC_LOGS`, under `LAMBDA_VM_PINNED_POOL_GIB`): the knob's
//! words; and, in one process of its own, a block proved with the knob off
//! and on over the same pool proves the same bytes, its windows' logs lying
//! in the pool; with the pool spent the windows fall back to pageable buffers,
//! counted, and the block still proves the same bytes; a purge hands none of
//! the pool's pages back.

use crate::block_whir::{
    POOL_EXEC_LOGS_ENV, pool_exec_logs, pool_exec_logs_word, set_pool_exec_logs_for_test,
};

/// The knob's name; off by default, and only `1` turns it on.
#[test]
fn the_pool_exec_logs_knob_is_named_and_off_by_default() {
    assert_eq!(POOL_EXEC_LOGS_ENV, "LAMBDA_VM_POOL_EXEC_LOGS");
    if std::env::var_os(POOL_EXEC_LOGS_ENV).is_none() {
        assert!(!pool_exec_logs());
    }
    for (word, on) in [
        (None, false),
        (Some(""), false),
        (Some("0"), false),
        (Some("on"), false),
        (Some("1"), true),
        (Some(" 1 "), true),
    ] {
        assert_eq!(pool_exec_logs_word(word), on, "{word:?}");
    }
    set_pool_exec_logs_for_test(Some(true));
    assert!(pool_exec_logs());
    set_pool_exec_logs_for_test(None);
}

/// The card test's block: the spin guest at 100 000 spins (200 013 cycles), in
/// windows of 2^15 logs (1.3 MB a buffer, past the pool's smallest binding
/// allocation, so 7 windows), every table capped at 2^16 rows (a few chunks
/// a table, far under the block's 1024 groups: `MaxRowsConfig::small()`'s 32
/// rows a chunk closed 1025 groups and the layout refused it).
fn pooled_logs_fixture() -> (
    Vec<u8>,
    Vec<u8>,
    crate::block_whir::BlockFormat,
    crate::block_whir::BlockOptions,
) {
    let elf = crate::test_utils::asm_elf_bytes("test_private_input_spin");
    let mut input = 100_000u32.to_le_bytes().to_vec();
    input.extend_from_slice(b"pooled!!");
    let format = super::block_whir_tests::many_groups();
    let o =
        super::block_whir_tests::streamed(crate::tables::MaxRowsConfig::uniform(1 << 16), 5, 15);
    (elf, input, format, o)
}

/// The card test's block on the host (no pool, the knob off): it proves and
/// verifies under the block's group cap, in at least 6 windows. Its group
/// count is printed. Box or build server; ignored for its prove's time.
#[test]
#[ignore = "proves a 200 000-cycle block on the host: the card test's fixture"]
fn the_pooled_logs_fixture_proves_on_the_host_under_the_group_cap() {
    use crate::block_whir::{Deviations, prove_block_whir_with, verify_block_whir};
    let (elf, input, format, o) = pooled_logs_fixture();
    let opts = stark::proof::options::ProofOptions::default_test_options();
    let (proof, stamps) =
        prove_block_whir_with(&elf, &input, &opts, &format, &o, &Deviations::default())
            .expect("the block proves");
    assert!(
        verify_block_whir(&proof, &elf, &opts, &format).unwrap_or(false),
        "the proof verifies"
    );
    let groups = proof.proof.columns.len();
    assert!(groups < 1024, "{groups} groups");
    assert!(stamps.executor.windows >= 6, "{:?}", stamps.executor);
    println!(
        "POOLED LOGS FIXTURE: {groups} groups · {} windows · proved and verified on the host",
        stamps.executor.windows
    );
}

/// A stand-in for the driver's page-locking, for the host test.
#[cfg(feature = "cuda")]
fn stand_in_lock(_ptr: *mut u8, _len: usize) -> bool {
    true
}

/// The spin guest's private input: `spins` spins (13 cycles, and 2 a spin).
#[cfg(feature = "cuda")]
fn spin_input(spins: u32) -> Vec<u8> {
    let mut input = spins.to_le_bytes().to_vec();
    input.extend_from_slice(b"pooled!!");
    input
}

/// A thread for each of the pool's arenas (each new thread binds to the next)
/// takes what its arena holds free and what the pool has left, a window's
/// size then larger, until a buffer comes back pageable: the buffers, held.
#[cfg(feature = "cuda")]
fn spend_the_pool(window_bytes: usize) -> Vec<Vec<u8>> {
    let mut held: Vec<Vec<u8>> = Vec::new();
    for _ in 0..2 * math_cuda::pinned::ARENAS {
        let taken = std::thread::spawn(move || {
            let mut mine: Vec<Vec<u8>> = Vec::new();
            for size in [window_bytes, 32 << 20] {
                for _ in 0..4096 {
                    let chunk: Vec<u8> =
                        multilinear::pinned::with_pinned(size, || Vec::with_capacity(size));
                    let inside = multilinear::pinned::is_pinned(chunk.as_ptr(), chunk.capacity());
                    mine.push(chunk);
                    if !inside {
                        break;
                    }
                }
            }
            mine
        })
        .join()
        .expect("a spending thread");
        held.extend(taken);
    }
    // And the executor's buffers' own arena.
    for _ in 0..4096 {
        let chunk: Vec<u8> = multilinear::pinned::with_pinned_logs(window_bytes, || {
            Vec::with_capacity(window_bytes)
        });
        let inside = multilinear::pinned::is_pinned(chunk.as_ptr(), chunk.capacity());
        held.push(chunk);
        if !inside {
            break;
        }
    }
    let spent = math_cuda::pinned::stats();
    assert!(
        spent.carved_bytes + (4 << 20) >= spent.pool_bytes,
        "the pool handed out to its end: {spent:?}"
    );
    held
}

/// The fast engine's word history over `elf`, a window of `window` logs at a
/// time, each window's buffer from [`pooled_log_buffer`] when `pooled`: each
/// window's logs as records, and per window whether its buffer lay in the
/// pool.
#[cfg(feature = "cuda")]
fn run_windows(
    elf: &[u8],
    input: &[u8],
    window: usize,
    pooled: bool,
) -> (Vec<[u64; 5]>, Vec<bool>, Vec<usize>) {
    use executor::vm::engine::{Engine, EngineKind};
    let program = executor::elf::Elf::load(elf).expect("the guest loads");
    let mut engine = Engine::new(EngineKind::Fast, &program, input.to_vec()).expect("an engine");
    engine.enable_word_history().expect("the word history");
    let capacity = executor::vm::fast::log_window_capacity(window);
    let (mut logs, mut inside, mut at) = (Vec::new(), Vec::new(), Vec::new());
    loop {
        if pooled {
            engine.recycle(crate::block_whir::pooled_log_buffer(capacity));
        }
        let Some((got, _history)) = engine.resume_word_history(window).expect("a window") else {
            break;
        };
        inside.push(multilinear::pinned::is_pinned(
            got.as_ptr().cast(),
            got.capacity() * std::mem::size_of::<executor::vm::logs::Log>(),
        ));
        at.push(got.as_ptr() as usize);
        logs.extend(
            got.iter()
                .map(|l| [l.current_pc, l.next_pc, l.src1_val, l.src2_val, l.dst_val]),
        );
    }
    (logs, inside, at)
}

/// Whether every whole page of `[at, at + len)` is resident (`mincore`).
#[cfg(feature = "cuda")]
fn resident(at: usize, len: usize) -> bool {
    // SAFETY: sysconf has no preconditions.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(4096) as usize;
    let (lo, hi) = (at.next_multiple_of(page), (at + len) / page * page);
    if hi <= lo {
        return true;
    }
    let mut vec = vec![0u8; (hi - lo) / page];
    // SAFETY: `[lo, hi)` is mapped (inside the pool's mapping, never
    // unmapped) and page-aligned; `vec` has a byte a page.
    let rc = unsafe { libc::mincore(lo as *mut libc::c_void, hi - lo, vec.as_mut_ptr().cast()) };
    rc == 0 && vec.iter().all(|&b| b & 1 == 1)
}

/// ★ Host, one process (the pool locked by a stand-in for its life): run alone.
///
/// 1. The executor's windows written into buffers from the pool hold its logs
///    byte for byte as windows written into its own buffers do; every window
///    but the first (its buffer the executor's own) lies in the pool.
/// 2. A purge of every arena (`alloc_purge::purge_now`) once those buffers
///    went back to the pool's arenas hands none of the pool's pages back: a
///    freed window buffer's pages are still resident after it (`mincore`; a
///    page handed back with `MADV_DONTNEED` would not be), nothing more is
///    carved or locked, and a buffer taken again lies in the pool.
/// 3. The pool spent: the windows' buffers are pageable, the spill is counted,
///    and the logs are the same again.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "locks a stand-in page-locked pool for the whole process: run alone"]
fn pooled_executor_log_buffers_hold_the_same_logs_survive_a_purge_and_spill_when_short() {
    use crate::block_whir::pooled_log_buffer;
    use crate::test_utils::asm_elf_bytes;
    math_cuda::pinned::set_lock(stand_in_lock);
    assert_eq!(
        math_cuda::pinned::set_pool_gib_for_tests(1),
        1,
        "the pool's knob was read already in this process"
    );
    let pool = math_cuda::pinned::lock_pool().expect("the pool locked");
    assert_eq!((pool.pieces, pool.refused), (1, 0), "{pool:?}");
    assert!(multilinear::pinned::ready());
    // 2 000 013 cycles in windows of 2^16 logs, 2.6 MB a buffer.
    let elf = asm_elf_bytes("test_private_input_spin");
    let input = spin_input(1_000_000);
    let window = 1 << 16;
    let window_bytes = window * std::mem::size_of::<executor::vm::logs::Log>();

    // 1. The same logs, the windows in the pool.
    let (plain, plain_in, _) = run_windows(&elf, &input, window, false);
    assert!(
        plain_in.iter().all(|&p| !p),
        "plain buffers outside the pool"
    );
    let (pooled, pooled_in, pooled_at) = run_windows(&elf, &input, window, true);
    assert!(plain_in.len() >= 30, "{} windows", plain_in.len());
    assert_eq!(pooled.len(), plain.len(), "the logs' count");
    assert!(pooled == plain, "the logs, byte for byte");
    let in_pool = pooled_in.iter().filter(|&&p| p).count();
    assert!(
        !pooled_in[0] && pooled_in[1..pooled_in.len() - 1].iter().all(|&p| p),
        "every window but the first (and the last, shrunk) in the pool: {pooled_in:?}"
    );

    // 1b. A second run on a new thread (as each prove's executor is) takes
    // the first run's freed buffers again: the logs' own arena, the pool
    // carved no further than a window or two.
    let carved = math_cuda::pinned::stats().carved_bytes;
    let (elf2, input2) = (elf.clone(), input.clone());
    let (again, again_in, _) =
        std::thread::spawn(move || run_windows(&elf2, &input2, window, true))
            .join()
            .expect("a second run's thread");
    assert!(again == plain, "the second run: the logs, byte for byte");
    assert!(
        again_in[1..again_in.len() - 1].iter().all(|&p| p),
        "the second run's windows in the pool: {again_in:?}"
    );
    let grew = math_cuda::pinned::stats().carved_bytes - carved;
    // One window's buffer at most: the last window's spare, never taken.
    assert!(
        grew <= window_bytes as u64,
        "the second run carved {grew} bytes more of the pool"
    );

    // 2. A purge returns none of the pool's pages.
    // The windows' buffers are freed (back to the pool's arenas) as
    // `run_windows` returned; their pages were written, so resident.
    let freed: Vec<usize> = pooled_at
        .iter()
        .zip(&pooled_in)
        .filter(|&(_, &p)| p)
        .map(|(&a, _)| a)
        .collect();
    assert!(
        freed.iter().all(|&a| resident(a, window_bytes)),
        "freed, still resident"
    );
    let before = math_cuda::pinned::stats();
    let purged = crate::alloc_purge::purge_now();
    assert!(purged.is_some(), "the purge ran");
    let after = math_cuda::pinned::stats();
    assert!(
        freed.iter().all(|&a| resident(a, window_bytes)),
        "a purge handed a freed window buffer's pages back"
    );
    assert_eq!(
        (after.locked_bytes, after.carved_bytes),
        (before.locked_bytes, before.carved_bytes),
        "nothing re-locked, nothing more carved"
    );
    let again = pooled_log_buffer(window);
    assert!(
        multilinear::pinned::is_pinned(again.as_ptr().cast(), window_bytes),
        "a buffer taken after the purge lies in the pool"
    );
    drop(again);

    // 3. The pool spent: pageable windows, the spill counted, the same logs.
    let held = spend_the_pool(window_bytes);
    let (short, short_in, _) = run_windows(&elf, &input, window, true);
    assert!(short == plain, "the pool spent: the logs, byte for byte");
    assert!(
        short_in.iter().all(|&p| !p),
        "the pool spent: every window pageable"
    );
    // The arenas took pageable memory once the pool was spent (jemalloc maps
    // past a request and serves later ones from the rest, so the windows'
    // own allocations need not each map more).
    assert!(
        math_cuda::pinned::stats().spilled_bytes > 0,
        "the spill counted"
    );
    drop(held);
    println!(
        "POOL EXEC LOGS HOST: {} windows · {in_pool} in the pool · a second run carved {grew} B more · after a purge still the pool's · spent: {} pageable · the same logs",
        pooled_in.len(),
        short_in.len()
    );
}

/// ★ Card, one process (the pool locked by the driver for its life): run
/// alone, deterministic grind. A block proved with the knob off and on over
/// the same 1 GiB pool proves the same bytes and verifies; on, its windows lie
/// in the pool (every one but the first), off none does; with the pool spent
/// the knob on proves the same bytes from pageable windows, counted.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "card: locks a page-locked pool for the whole process; run alone, deterministic grind"]
fn pooled_executor_logs_prove_the_same_bytes_on_the_card() {
    use crate::block_whir::{BlockWhirProof, Deviations, prove_block_whir_with, verify_block_whir};
    use stark::proof::options::ProofOptions;

    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_pool_exec_logs_for_test(None);
        }
    }
    let _restore = Restore;
    assert!(
        std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some(),
        "the identity needs the deterministic grind"
    );
    assert!(multilinear::gpu::reserve_budget() > 0, "needs a card");
    assert_eq!(
        math_cuda::pinned::set_pool_gib_for_tests(1),
        1,
        "the pool's knob was read already in this process"
    );
    let pool = math_cuda::pinned::lock_pool().expect("the pool locked");
    assert_eq!((pool.pieces, pool.refused), (1, 0), "{pool:?}");
    let (elf, input, format, o) = pooled_logs_fixture();
    let opts = ProofOptions::default_test_options();
    let bytes = |p: &BlockWhirProof| {
        rkyv::to_bytes::<rkyv::rancor::Error>(p)
            .expect("serialize")
            .to_vec()
    };
    let proved = |on: bool| {
        set_pool_exec_logs_for_test(Some(on));
        let (proof, stamps) =
            prove_block_whir_with(&elf, &input, &opts, &format, &o, &Deviations::default())
                .expect("the block proves");
        assert!(
            verify_block_whir(&proof, &elf, &opts, &format).unwrap_or(false),
            "pooled logs {on}: the proof verifies"
        );
        (bytes(&proof), stamps.executor)
    };
    let (off, off_x) = proved(false);
    assert!(off_x.windows >= 6, "{off_x:?}");
    assert!(!off_x.logs_pooled && off_x.pooled_windows == 0, "{off_x:?}");
    let (on, on_x) = proved(true);
    assert_eq!(on, off, "the knob on: the proof's bytes");
    assert!(on_x.logs_pooled, "{on_x:?}");
    assert!(
        on_x.pooled_windows + 2 >= on_x.windows && on_x.pooled_windows >= 4,
        "the windows from the second on lie in the pool: {on_x:?}"
    );
    let held = spend_the_pool((1 << 15) * std::mem::size_of::<executor::vm::logs::Log>());
    let (short, short_x) = proved(true);
    assert_eq!(short, off, "the pool spent: the proof's bytes");
    assert!(
        short_x.pageable_windows > short_x.pooled_windows,
        "the windows fell back to pageable buffers: {short_x:?}"
    );
    drop(held);
    println!(
        "POOL EXEC LOGS CARD: off {} windows · on {} of {} in the pool · spent {} pageable of {} · the same bytes",
        off_x.windows, on_x.pooled_windows, on_x.windows, short_x.pageable_windows, short_x.windows
    );
}

/// ★ Card, one process (the pool locked by the driver for its life): run
/// alone, deterministic grind. With the executor's logs in a 1 GiB pool, the
/// block proved twice with the shared arenas picked by thread (today's) and
/// twice by size class (`LAMBDA_VM_POOL_ARENA=class`) proves the same bytes
/// each time, and verifies. Prints how much more of the pool each choice's
/// second prove carved (by size class: the first's freed buffers taken again).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "card: locks a page-locked pool for the whole process; run alone, deterministic grind"]
fn the_pool_arena_by_size_class_proves_the_same_bytes_on_the_card() {
    use crate::block_whir::{BlockWhirProof, Deviations, prove_block_whir_with, verify_block_whir};
    use stark::proof::options::ProofOptions;

    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_pool_exec_logs_for_test(None);
            math_cuda::pinned::set_pool_arena_by_class_for_test(None);
        }
    }
    let _restore = Restore;
    assert!(
        std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some(),
        "the identity needs the deterministic grind"
    );
    assert!(multilinear::gpu::reserve_budget() > 0, "needs a card");
    assert_eq!(
        math_cuda::pinned::set_pool_gib_for_tests(1),
        1,
        "the pool's knob was read already in this process"
    );
    let pool = math_cuda::pinned::lock_pool().expect("the pool locked");
    assert_eq!((pool.pieces, pool.refused), (1, 0), "{pool:?}");
    set_pool_exec_logs_for_test(Some(true));
    let (elf, input, format, o) = pooled_logs_fixture();
    let opts = ProofOptions::default_test_options();
    let bytes = |p: &BlockWhirProof| {
        rkyv::to_bytes::<rkyv::rancor::Error>(p)
            .expect("serialize")
            .to_vec()
    };
    let proved = |class: bool| {
        math_cuda::pinned::set_pool_arena_by_class_for_test(Some(class));
        let (proof, _) =
            prove_block_whir_with(&elf, &input, &opts, &format, &o, &Deviations::default())
                .expect("the block proves");
        assert!(
            verify_block_whir(&proof, &elf, &opts, &format).unwrap_or(false),
            "by size class {class}: the proof verifies"
        );
        (bytes(&proof), math_cuda::pinned::stats().carved_bytes)
    };
    let (slot1, c1) = proved(false);
    let (slot2, c2) = proved(false);
    let (class1, c3) = proved(true);
    let (class2, c4) = proved(true);
    assert_eq!(slot2, slot1, "by thread, twice: the proof's bytes");
    assert_eq!(class1, slot1, "by size class: the proof's bytes");
    assert_eq!(class2, slot1, "by size class, twice: the proof's bytes");
    println!(
        "POOL ARENA CARD: the same bytes by thread and by size class · the second prove carved {} B more by thread, {} B by size class",
        c2 - c1,
        c4 - c3
    );
}
