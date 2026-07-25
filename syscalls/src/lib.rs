pub mod allocator;
pub mod ef_io;
// Guest-only: `_start` + the imported `main` are entry symbols that collide
// with the host C runtime / test harness (Linux errors with "entry symbol
// `main` declared multiple times"; macOS happens to tolerate it, which is why
// host tests passed locally). Same treatment as the global allocator gating.
#[cfg(target_arch = "riscv64")]
pub mod entrypoint;
pub mod keccak;
pub mod random;
pub mod syscalls;
