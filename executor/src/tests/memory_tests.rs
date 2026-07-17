//! Tests for guest memory: public-output commits and bounds-checked loads.

use crate::vm::memory::{Memory, MemoryError};

#[test]
fn test_commit_public_output_single() {
    let mut memory = Memory::default();
    memory.store_byte(0x100, b'a').unwrap();
    memory.store_byte(0x101, b'b').unwrap();

    memory
        .commit_public_output(0x100, 2)
        .expect("commit should succeed");

    assert_eq!(
        memory
            .read_return_value()
            .expect("public output should be readable"),
        b"ab".to_vec()
    );
}

#[test]
fn test_commit_public_output_appends() {
    let mut memory = Memory::default();
    memory.store_byte(0x100, b'a').unwrap();
    memory.store_byte(0x101, b'b').unwrap();
    memory.store_byte(0x104, b'c').unwrap();
    memory.store_byte(0x105, b'd').unwrap();

    memory
        .commit_public_output(0x100, 2)
        .expect("first commit should succeed");
    memory
        .commit_public_output(0x104, 2)
        .expect("second commit should succeed");

    // Append semantics: calls concatenate (EF zkVM IO interface).
    assert_eq!(
        memory
            .read_return_value()
            .expect("public output should be readable"),
        b"abcd".to_vec()
    );
}

#[test]
fn test_commit_public_output_empty_is_ok() {
    let mut memory = Memory::default();
    memory
        .commit_public_output(0, 0)
        .expect("zero-length commit should succeed");
    assert!(
        memory
            .read_return_value()
            .expect("public output should be readable")
            .is_empty()
    );
}

#[test]
fn test_commit_public_output_address_overflow() {
    let mut memory = Memory::default();
    let err = memory
        .commit_public_output(u64::MAX, 2)
        .expect_err("address overflow must error, not panic");
    assert!(matches!(err, MemoryError::AddressOverflow));
}

#[test]
fn test_load_bytes_huge_len_returns_alloc_error() {
    let memory = Memory::default();
    // A multi-petabyte allocation request from a guest must fail cleanly,
    // not abort the host process via OOM. `addr=0` and `len=1<<50` keep
    // `checked_add` happy so the path reaches the allocation.
    let huge = 1u64 << 50;
    let err = memory
        .load_bytes(0, huge)
        .expect_err("huge alloc must error, not abort");
    assert!(matches!(err, MemoryError::AllocationFailed));
}

#[test]
fn test_load_bytes_overflow_errors() {
    let memory = Memory::default();
    let err = memory
        .load_bytes(u64::MAX, 2)
        .expect_err("address overflow must error, not panic");
    assert!(matches!(err, MemoryError::AddressOverflow));
}

#[test]
fn test_commit_public_output_total_cap() {
    let mut memory = Memory::default();
    // Seed enough source bytes for two 512 KB writes.
    let chunk = vec![0xAB; 512 * 1024];
    memory
        .set_bytes_aligned(0x1_0000, &chunk)
        .expect("seed should succeed");

    memory
        .commit_public_output(0x1_0000, 512 * 1024)
        .expect("first 512 KB commit should succeed");
    memory
        .commit_public_output(0x1_0000, 512 * 1024)
        .expect("second 512 KB commit should succeed (total = 1 MB)");

    // One more byte exceeds the 1 MB total cap.
    let err = memory.commit_public_output(0x1_0000, 1).unwrap_err();
    assert!(matches!(err, MemoryError::CommitSizeExceeded));
}

#[test]
fn test_misaligned_load_store_overflow_errors() {
    let mut memory = Memory::default();

    assert!(matches!(
        memory.load_half(u64::MAX).unwrap_err(),
        MemoryError::AddressOverflow
    ));
    assert!(matches!(
        memory.store_half(u64::MAX, 0).unwrap_err(),
        MemoryError::AddressOverflow
    ));
    assert!(matches!(
        memory.load_word(u64::MAX - 1).unwrap_err(),
        MemoryError::AddressOverflow
    ));
    assert!(matches!(
        memory.store_word(u64::MAX - 1, 0).unwrap_err(),
        MemoryError::AddressOverflow
    ));
    assert!(matches!(
        memory.load_doubleword(u64::MAX - 6).unwrap_err(),
        MemoryError::AddressOverflow
    ));
    assert!(matches!(
        memory.store_doubleword(u64::MAX - 6, 0).unwrap_err(),
        MemoryError::AddressOverflow
    ));
}
