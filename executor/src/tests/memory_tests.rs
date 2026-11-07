//! Tests for guest memory: public-output commits and bounds-checked loads.

use crate::vm::memory::{Memory, MemoryError};

#[test]
fn test_commit_public_output_single() {
    let mut memory = Memory::default();
    memory.store_byte(0x100, b'a');
    memory.store_byte(0x101, b'b');

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
    memory.store_byte(0x100, b'a');
    memory.store_byte(0x101, b'b');
    memory.store_byte(0x104, b'c');
    memory.store_byte(0x105, b'd');

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

/// The paged store answers every access as the word map does: loads and
/// stores of every width, aligned or not, across 64 KiB pages, in the low and
/// top directories and between them, the overflow refusals at the top of the
/// address space, `load_bytes`, the private input, and `iter_bytes` (as a set:
/// the two iterate in different orders).
#[test]
fn the_paged_store_is_the_word_map() {
    use crate::vm::memory::StoreKind;

    const PAGE: u64 = 1 << 16;
    let mut pages = Memory::with_store(StoreKind::Pages);
    let mut words = Memory::with_store(StoreKind::Words);
    pages.store_private_inputs(vec![7u8; 70_001]).unwrap();
    words.store_private_inputs(vec![7u8; 70_001]).unwrap();

    let spots = [
        0,
        3 * PAGE - 3,
        0x1_0000_0000 - 2,
        (1 << 33) - 5,
        1 << 33,
        0x8000_0000_0000_0000 - 3,
        u64::MAX - (1 << 33) - 2,
        u64::MAX - (1 << 33) + 6,
        u64::MAX - 9,
        u64::MAX - 1,
    ];
    let mut x = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let same = |a: Result<u64, MemoryError>, b: Result<u64, MemoryError>, what: &str| match (a, b) {
        (Ok(a), Ok(b)) => assert_eq!(a, b, "{what}"),
        (Err(a), Err(b)) => assert_eq!(format!("{a:?}"), format!("{b:?}"), "{what}"),
        (a, b) => panic!("{what}: {a:?} vs {b:?}"),
    };
    for _ in 0..200_000 {
        let r = next();
        let addr = spots[(r % spots.len() as u64) as usize].wrapping_add((r >> 8) % 24);
        let value = next();
        let what = format!("op {} at {addr:#x}", (r >> 16) % 10);
        match (r >> 16) % 10 {
            0 => {
                pages.store_byte(addr, value as u8);
                words.store_byte(addr, value as u8);
            }
            1 => same(
                pages.store_half(addr, value as u16).map(|_| 0),
                words.store_half(addr, value as u16).map(|_| 0),
                &what,
            ),
            2 => same(
                pages.store_word(addr, value as u32).map(|_| 0),
                words.store_word(addr, value as u32).map(|_| 0),
                &what,
            ),
            3 => same(
                pages.store_doubleword(addr, value).map(|_| 0),
                words.store_doubleword(addr, value).map(|_| 0),
                &what,
            ),
            4 => assert_eq!(pages.load_byte(addr), words.load_byte(addr), "{what}"),
            5 => same(
                pages.load_half(addr).map(u64::from),
                words.load_half(addr).map(u64::from),
                &what,
            ),
            6 => same(
                pages.load_word(addr).map(u64::from),
                words.load_word(addr).map(u64::from),
                &what,
            ),
            7 | 8 => same(
                pages.load_doubleword(addr),
                words.load_doubleword(addr),
                &what,
            ),
            _ => {
                let len = value % 40;
                match (pages.load_bytes(addr, len), words.load_bytes(addr, len)) {
                    (Ok(a), Ok(b)) => assert_eq!(a, b, "{what}"),
                    (Err(a), Err(b)) => assert_eq!(format!("{a:?}"), format!("{b:?}"), "{what}"),
                    (a, b) => panic!("{what}: {a:?} vs {b:?}"),
                }
            }
        }
    }
    let mut a: Vec<(u64, u8)> = pages.iter_bytes().collect();
    let mut b: Vec<(u64, u8)> = words.iter_bytes().collect();
    a.sort_unstable();
    b.sort_unstable();
    assert_eq!(a.len(), b.len(), "the words written");
    assert!(a == b, "the bytes written");
}
