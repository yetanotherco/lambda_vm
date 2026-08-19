#[cfg(target_arch = "riscv64")]
use core::arch::asm;

/// Memory-mapped private input region start address.
/// Layout: 4-byte LE length prefix at this address, data at +4.
/// The host pre-loads the input; the guest reads directly (no ecall).
/// Must match `executor::vm::memory::PRIVATE_INPUT_START_INDEX`.
#[cfg(target_arch = "riscv64")]
pub const PRIVATE_INPUT_START: usize = 0xFF000000;

/// Maximum private-input length the guest will read, in bytes (512 MiB).
/// The host caps stored input at this size in `Memory::store_private_inputs`,
/// so an honest length prefix is always `<=` this bound; a larger value can only
/// come from a malformed or forged prefix. The reader clamps to this cap so a
/// bogus length can never make the guest fabricate an arbitrarily long slice.
/// Must match `executor::vm::memory::MAX_PRIVATE_INPUT_SIZE`.
#[cfg(target_arch = "riscv64")]
const MAX_PRIVATE_INPUT_SIZE: usize = 512 * 1024 * 1024;

#[cfg(target_arch = "riscv64")]
pub enum SyscallNumbers {
    Print = 1,
    Panic = 2,
    Commit = 64,
    Halt = 93,
}

/// Syscall number for KeccakPermute (u64::MAX - 1).
#[cfg(target_arch = "riscv64")]
const KECCAK_SYSCALL_NUMBER: usize = usize::MAX - 1;

/// Syscall number for the ECSM secp256k1 scalar-multiply accelerator (-11 as usize).
#[cfg(target_arch = "riscv64")]
const ECSM_SYSCALL_NUMBER: usize = usize::MAX - 10;

/// Hint selectors for the request log (must match the executor's `HINT_*`).
pub const HINT_FIELD_INV: usize = 0;
pub const HINT_SCALAR_INV: usize = 1;
pub const HINT_FIELD_SQRT: usize = 2;

/// No-op. The `Print` ecall (a7=1) has no receiver on the Ecall bus, so emitting
/// it makes the LogUp bus unbalance and the proof fail to verify. Printing isn't
/// needed in provable programs, so `print_string` does nothing on every target.
/// Keep it as a no-op rather than deleting call sites: that way no guest path can
/// ever reintroduce an unmatched Print ecall. (See `SyscallNumbers::Print`.)
pub fn print_string(_s: &str) {}

/// # Safety
///
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_write(_fildes: i32, buf: *const u8, size: usize) -> isize {
    print_string("sys_write called\n");
    let content = unsafe { core::slice::from_raw_parts(buf, size) };
    print_string(&("SYS_WRITE: ".to_owned() + str::from_utf8(content).unwrap_or("<invalid utf8>"))); // Does the print of the sys write
    size.try_into().unwrap_or(-1)
}

#[cfg(target_arch = "riscv64")]
/// # Safety
///
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_panic(msg_ptr: *const u8, len: usize) {
    print_string("Sys panic called\n");
    unsafe {
        asm!(
            "ecall",
            in("a0") msg_ptr,
            in("a1") len,
            in("a7") SyscallNumbers::Panic as usize,
        )
    }
}

#[cfg(target_arch = "riscv64")]
pub fn commit(slice: &[u8]) {
    unsafe {
        asm!(
            "ecall",
            in("a0") 1usize,
            in("a1") slice.as_ptr(),
            in("a2") slice.len(),
            in("a7") SyscallNumbers::Commit as usize,
        )
    }
}

/// Read private input bytes from the memory-mapped region at
/// `PRIVATE_INPUT_START = 0xFF000000`.
///
/// The host pre-loads the input before execution; this function reads the
/// 4-byte LE length prefix and then copies the data bytes into a new `Vec`.
/// No ecall is performed — it's a plain memory read (ZisK-style).
#[cfg(target_arch = "riscv64")]
pub fn get_private_input() -> Vec<u8> {
    // Copy the borrowed private-input bytes into an owned `Vec`. The raw-pointer
    // read (length prefix + data slice) and its single `unsafe` block live in
    // `get_private_input_slice`, so the memory layout is defined in one place.
    get_private_input_slice().to_vec()
}

#[cfg(not(target_arch = "riscv64"))]
pub fn get_private_input() -> Vec<u8> {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

/// Borrow the private input bytes in place from the memory-mapped region —
/// no copy, no allocation. Same layout as [`get_private_input`]; the returned
/// slice starts at `PRIVATE_INPUT_START + 4` (a 4-aligned address) and lives
/// for the whole execution (the host never remaps the region).
#[cfg(target_arch = "riscv64")]
pub fn get_private_input_slice() -> &'static [u8] {
    // SAFETY: The host pre-loads private input at PRIVATE_INPUT_START before
    // execution and never remaps it afterward, so the returned slice is valid
    // for the `'static` lifetime of the guest's single-threaded execution
    // region, which stays mapped and unmodified for the whole execution.
    let len_ptr = PRIVATE_INPUT_START as *const u32;
    // Clamp the prover-written length prefix to `MAX_PRIVATE_INPUT_SIZE`. An
    // honest prefix (written by the host, which caps stored input at this size)
    // is always within bound, so clamping never changes behavior for real
    // inputs — it only bounds the slice length when a malformed or forged prefix
    // claims more, keeping the read deterministic.
    let len = (unsafe { core::ptr::read_volatile(len_ptr) } as usize).min(MAX_PRIVATE_INPUT_SIZE);
    let data_ptr = (PRIVATE_INPUT_START + 4) as *const u8;
    unsafe { core::slice::from_raw_parts(data_ptr, len) }
}

#[cfg(not(target_arch = "riscv64"))]
pub fn get_private_input_slice() -> &'static [u8] {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

// =============================================================================
// Hint arena — untrusted, host-supplied 32-byte hint values appended to the
// private-input region after the length-prefixed main data:
//
//   [u32 LE main_len][main data][zero-pad to 8][u32 LE hint_count][u32 pad]
//   then `hint_count` slots of 32 bytes, 8-aligned.
//
// Must match `executor::vm::memory::{hint_arena_header_offset,
// HINT_ARENA_HEADER_BYTES, HINT_SLOT_BYTES}` — the executor writes the region,
// this crate reads it.
//
// The values are UNTRUSTED (the prover chooses them): the caller MUST verify
// each hint in-guest (e.g. `x·inv == 1`) and recompute in software on failure.
// Consumption is positional — one slot per request, pass or fail — so a lying
// host can only trigger fallbacks, never shift the hint stream.
// =============================================================================

/// Byte size of one hint slot in the hint arena.
/// Must match `executor::vm::memory::HINT_SLOT_BYTES`.
#[cfg(target_arch = "riscv64")]
pub const HINT_SLOT_BYTES: usize = 32;

/// Byte size of the arena header (`[u32 LE hint_count][u32 zero pad]`).
/// Must match `executor::vm::memory::HINT_ARENA_HEADER_BYTES`.
#[cfg(target_arch = "riscv64")]
const HINT_ARENA_HEADER_BYTES: usize = 8;

/// Offset from `PRIVATE_INPUT_START` of the arena header for a given main-input
/// length: the 4-byte length prefix plus the data, padded up to 8 bytes.
/// Must match `executor::vm::memory::hint_arena_header_offset`.
#[cfg(target_arch = "riscv64")]
const fn hint_arena_header_offset(main_len: usize) -> usize {
    (4 + main_len + 7) & !7
}

/// Absolute address of the arena header, derived from the (clamped) length
/// prefix — the same value `get_private_input_slice` trusts.
#[cfg(target_arch = "riscv64")]
fn hint_arena_header_addr() -> usize {
    let len = (unsafe { core::ptr::read_volatile(PRIVATE_INPUT_START as *const u32) } as usize)
        .min(MAX_PRIVATE_INPUT_SIZE);
    PRIVATE_INPUT_START + hint_arena_header_offset(len)
}

/// Number of hint slots the host supplied. 0 when no arena was written (the
/// header reads back as zero-filled memory).
#[cfg(target_arch = "riscv64")]
pub fn hint_count() -> usize {
    unsafe { core::ptr::read_volatile(hint_arena_header_addr() as *const u32) as usize }
}

#[cfg(not(target_arch = "riscv64"))]
pub fn hint_count() -> usize {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

/// Read hint slot `i` as raw bytes, or `None` when the arena has no slot `i`.
/// The slot is read as four aligned 8-byte words (the fast aligned-load path);
/// slots are 8-aligned by construction.
#[cfg(target_arch = "riscv64")]
pub fn hint_slot(i: usize) -> Option<[u8; 32]> {
    if i >= hint_count() {
        return None;
    }
    let addr = hint_arena_header_addr() + HINT_ARENA_HEADER_BYTES + i * HINT_SLOT_BYTES;
    debug_assert_eq!(addr % 8, 0, "hint slot address must stay 8-aligned");
    // Read as four u64 words and re-serialize little-endian: the VM is
    // little-endian, so this reproduces the exact byte sequence the host wrote.
    let words: [u64; 4] = unsafe { core::ptr::read_volatile(addr as *const [u64; 4]) };
    let mut out = [0u8; 32];
    for (k, w) in words.iter().enumerate() {
        out[8 * k..8 * k + 8].copy_from_slice(&w.to_le_bytes());
    }
    Some(out)
}

#[cfg(not(target_arch = "riscv64"))]
pub fn hint_slot(_i: usize) -> Option<[u8; 32]> {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

/// Consume the next hint slot positionally. Returns `None` once the arena is
/// exhausted — the caller then recomputes in software. One slot is consumed per
/// call whether or not the hint verifies, so a host that supplies fewer hints
/// than requested only forces fallbacks; it cannot desynchronize the stream.
#[cfg(target_arch = "riscv64")]
pub fn next_hint() -> Option<[u8; 32]> {
    // Single-threaded guest: a relaxed atomic cursor is sufficient (the guest
    // target lowers atomics via `-C passes=lower-atomic`).
    static CURSOR: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
    let i = CURSOR.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    hint_slot(i)
}

#[cfg(not(target_arch = "riscv64"))]
pub fn next_hint() -> Option<[u8; 32]> {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

// =============================================================================
// Hint request log — the recording pass for hints that are NOT known
// beforehand. When the arena is exhausted, `request_hint` appends
// `(hint_id, input)` to a fixed scratch region above the private-input window;
/// the host reads it after the run (executor's `Memory::hint_requests`),
/// computes the answers, and re-runs with a complete arena. The log region:
///
//   [u32 LE count][u32 pad], then entries of [u64 LE hint_id][32-byte input].
//
// Must match `executor::vm::memory::{HINT_LOG_START_INDEX,
// HINT_LOG_HEADER_BYTES, HINT_LOG_ENTRY_BYTES}`.
// =============================================================================

/// Start of the hint request log: just past the reserved private-input window
/// (`PRIVATE_INPUT_START + 4 + MAX_PRIVATE_INPUT_SIZE`, rounded up to 8). The
/// stack lives at the top of the 64-bit space and the heap grows up from the
/// ELF image, so this region is collision-free in practice.
/// Must match `executor::vm::memory::HINT_LOG_START_INDEX`.
#[cfg(target_arch = "riscv64")]
pub const HINT_LOG_START: usize = 0xFF000000 + 4 + 512 * 1024 * 1024 + 4;

/// Log header size (`[u32 LE count][u32 pad]`).
/// Must match `executor::vm::memory::HINT_LOG_HEADER_BYTES`.
#[cfg(target_arch = "riscv64")]
const HINT_LOG_HEADER_BYTES: usize = 8;

/// Log entry size (`[u64 LE hint_id][32-byte input]`).
/// Must match `executor::vm::memory::HINT_LOG_ENTRY_BYTES`.
#[cfg(target_arch = "riscv64")]
const HINT_LOG_ENTRY_BYTES: usize = 40;

/// Consume the next hint slot for a `(hint_id, input)` request; on an exhausted
/// arena, append the request to the hint request log and return `None` (the
/// caller then recomputes in software). One slot is consumed per call whether
/// or not the hint verifies — see [`next_hint`].
#[cfg(target_arch = "riscv64")]
pub fn request_hint(hint_id: usize, input: &[u8; 32]) -> Option<[u8; 32]> {
    if let Some(slot) = next_hint() {
        return Some(slot);
    }
    let count_addr = HINT_LOG_START;
    let count = unsafe { core::ptr::read_volatile(count_addr as *const u32) } as usize;
    let entry = count_addr + HINT_LOG_HEADER_BYTES + count * HINT_LOG_ENTRY_BYTES;
    unsafe {
        core::ptr::write_volatile(entry as *mut u64, hint_id as u64);
        for k in 0..4 {
            let word = u64::from_le_bytes(input[8 * k..8 * k + 8].try_into().unwrap());
            core::ptr::write_volatile((entry + 8 + 8 * k) as *mut u64, word);
        }
        core::ptr::write_volatile(count_addr as *mut u32, count as u32 + 1);
    }
    None
}

#[cfg(not(target_arch = "riscv64"))]
pub fn request_hint(_hint_id: usize, _input: &[u8; 32]) -> Option<[u8; 32]> {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

#[cfg(target_arch = "riscv64")]
pub fn sys_halt() -> ! {
    // NOTE: no print_string here — the Print ecall is unmatched on the Ecall bus
    // and would cause a verification failure.
    unsafe {
        asm!(
            "ecall",
            in("a0") 0usize, // exit_code = 0 (enforced by HALT read on x10)
            in("a7") SyscallNumbers::Halt as usize,
        );
    }
    unreachable!()
}

#[cfg(not(target_arch = "riscv64"))]
pub fn sys_halt() -> ! {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

#[cfg(target_arch = "riscv64")]
/// Apply the Keccak-f[1600] permutation to a 25-element u64 state in-place.
pub fn keccak_permute(state: &mut [u64; 25]) {
    unsafe {
        asm!(
            "ecall",
            in("a0") state.as_mut_ptr(),
            in("a7") KECCAK_SYSCALL_NUMBER,
        )
    }
}

#[cfg(not(target_arch = "riscv64"))]
/// Apply the Keccak-f[1600] permutation to a 25-element u64 state in-place.
pub fn keccak_permute(_state: &mut [u64; 25]) {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

#[cfg(target_arch = "riscv64")]
/// Compute `xR = (k·G)_x` on secp256k1 via the ECSM accelerator. All values are 32-byte
/// little-endian. Requires `0 < k < N` and a canonical valid `xG` curve coordinate.
/// `xG` and `k` must not overlap; `xR` may alias either input.
pub fn ecsm_mul(xr: &mut [u8; 32], xg: &[u8; 32], k: &[u8; 32]) {
    unsafe {
        asm!(
            "ecall",
            in("a0") xr.as_mut_ptr(), // x10 = address to write xR
            in("a1") xg.as_ptr(),     // x11 = address of xG
            in("a2") k.as_ptr(),      // x12 = address of k
            in("a7") ECSM_SYSCALL_NUMBER,
        )
    }
}

#[cfg(not(target_arch = "riscv64"))]
/// Compute `xR = (k·G)_x` on secp256k1 via the ECSM accelerator (32-byte little-endian values).
pub fn ecsm_mul(_xr: &mut [u8; 32], _xg: &[u8; 32], _k: &[u8; 32]) {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

// =============================================================================
// Stub implementations for unsupported std functions
// These functions are required by Rust's std zkvm module but are not supported
// in Lambda VM. They will panic at runtime if called.
// =============================================================================

/// # Safety
///
/// This function is not supported in Lambda VM.
/// It will panic if called.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_read(_fd: u32, _buf: *mut u8, _nbytes: usize) -> usize {
    panic!("sys_read is not supported: io::Read for Stdin is not implemented in Lambda VM");
}

/// # Safety
///
/// This function is not supported in Lambda VM.
/// It will panic if called.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_argc() -> usize {
    panic!("sys_argc is not supported: command-line arguments are not available in Lambda VM");
}

/// # Safety
///
/// This function is not supported in Lambda VM.
/// It will panic if called.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_argv(_buf: *mut u32, _buf_nwords: usize, _arg_idx: usize) -> usize {
    panic!("sys_argv is not supported: command-line arguments are not available in Lambda VM");
}
