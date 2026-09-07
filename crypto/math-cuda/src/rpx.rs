//! RPX256 (Rescue-Prime eXtended, width 12) device launch code — PHASE 2.
//!
//! Phase 1 (lane K) ships the kernel SOURCE, `kernels/rpx.cu`, pinned by the
//! host known-answer harness `tests/host_kat/rpx_host_kat.cpp` (run with
//! `make test-rpx-host-kat`): the permutation, the rate-8 overwrite-duplex
//! leaf sponge and the Merkle parent, compiled on the host through
//! `cuda_host_shim.h` and checked against miden-crypto's RPO vectors and the
//! Rust oracle `prover/src/lfm/rpx.rs`.
//!
//! This module is the placeholder for the phase-2 launch wrappers — the leaf
//! kernels mirroring `blake3.rs`'s, `merkle_level` / `merkle_tail` over
//! `rpx::compress`, and the third arm in every `match hash` — and it is
//! deliberately NOT declared in `lib.rs` yet: nothing compiles it. Phase 2
//! adds `pub mod rpx;` to `lib.rs` and
//! `compile_kernel("rpx.cu", "rpx.cubin", have_nvcc, &[])` to `build.rs`
//! (both lane D's files, requested through the coordinator).
//!
//! Digest layout contract for that work: a digest is four CANONICAL Goldilocks
//! felts (the kernel canonicalises every permutation output), serialised as
//! `digest_to_commitment` does — each felt's eight big-endian bytes, 32 bytes
//! per node, the same slot width as a BLAKE3 digest.
