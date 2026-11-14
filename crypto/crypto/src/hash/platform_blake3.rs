//! [`PlatformBlake3`] — the BLAKE3 byte hash, under the name shape
//! [`PlatformKeccak256`](crate::hash::platform_keccak::PlatformKeccak256)
//! established for a hash that is accelerated on the riscv64 guest and software
//! everywhere else.
//!
//! # This is a re-export, and that is the design
//!
//! `platform_keccak` needs an adapter because the thing it selects between is
//! two different types: `sha3::Keccak256` on the host and a syscall-backed
//! sponge from the syscall crate on the guest. Those two carry their own
//! framing, so something has to give them one `digest` interface.
//!
//! BLAKE3 has no such pair. [`Blake3Chain`](crate::hash::blake3::Blake3Chain) is
//! one type on every target; the accelerator is reached from *inside* it, at the
//! compression function, where `compress_block` selects the syscall on riscv64
//! and software otherwise. The framing — single chunk, 64-byte blocks,
//! `CHUNK_START` / `CHUNK_END | ROOT`, `t = 0`, the true byte count as the final
//! block's `block_len` (PA-PLAN §1.7) — is above that seam and is therefore the
//! same code on host and guest by construction.
//!
//! INVARIANT (load-bearing): this must remain a **PURE PASSTHROUGH** — a
//! re-export and nothing else. A wrapper type here would be a second place the
//! framing is expressed, which is exactly what PA-PLAN §1.4 forbids and what
//! `executor::vm::instruction::execution`'s duplicate compression already cost
//! us one gating test to contain. It would also break the argument the `TypeId`
//! specializations in `merkle_tree::backends::field_element_vector` rest on:
//! they dispatch on the concrete `PlatformKeccak256` type, so a BLAKE3 `D`
//! reaches the generic `D::new()/update/finalize` path — correct only while
//! `PlatformBlake3` *is* `Blake3Chain` and hashes identically through both.
//!
//! The round count is not selected here either. It is
//! [`BLAKE3_ROUNDS`](crate::hash::blake3::BLAKE3_ROUNDS), one crate-global knob,
//! so a build cannot commit under two hashes.

pub use crate::hash::blake3::chain::Blake3Chain as PlatformBlake3;
