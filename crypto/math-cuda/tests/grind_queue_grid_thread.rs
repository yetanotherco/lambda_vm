//! ★ The K4 grid must read the same on every thread.
//!
//! `grinding::queue_grid` sizes the K4 grind by the queue kernel's occupancy, a
//! driver query that needs the CUDA context current on the CALLING thread.
//! `backend()` makes the context current only on the thread that created it,
//! and cudarc binds it before `num_regs` but not before the occupancy query. So
//! a first K4 grind on any other thread got an error there, the `None` was
//! cached for the whole process, and every RPX grind after it fell to the host
//! search. Nothing failed; the device counter just stopped moving.
//! `rpx_grind_device` showed it on the runs where its keccak test happened to
//! be the one that created the backend.
//!
//! ONE test, in its own binary, on purpose: the grid is cached per process, so
//! a read by any earlier test on a thread that had the context would hide it.
//!
//! Runs on the merge-queue GPU box via `make test-math-cuda`, like the other
//! tests here.

use std::thread;

use cudarc::driver::sys::CUdevice_attribute;
use math_cuda::device::backend;
use math_cuda::grinding::{RPX_BLOCK_DIM, queue_grid};

#[test]
fn the_queue_grid_reads_on_a_thread_that_did_not_create_the_backend() {
    // The backend, and with it the current context, on a thread of its own.
    thread::spawn(|| {
        backend().expect("a CUDA device");
    })
    .join()
    .unwrap();

    // What the driver answers on a fresh thread, printed rather than asserted:
    // this test can see the regression only while this is an error.
    let unbound = thread::spawn(|| {
        backend()
            .expect("a CUDA device")
            .rpx_grind_search_queue
            .occupancy_max_active_blocks_per_multiprocessor(RPX_BLOCK_DIM, 0, None)
    })
    .join()
    .unwrap();
    println!("occupancy query on a thread with no current context: {unbound:?}");

    let grid = thread::spawn(queue_grid)
        .join()
        .unwrap()
        .expect("the K4 grid must read on a thread that did not create the backend");
    let sms = backend()
        .expect("a CUDA device")
        .ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .expect("the multiprocessor count") as u32;
    assert!(
        grid >= sms,
        "a grid of {grid} blocks must hold at least one per multiprocessor ({sms})"
    );
}
