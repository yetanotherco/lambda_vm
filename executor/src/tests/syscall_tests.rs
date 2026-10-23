use crate::vm::instruction::execution::SyscallNumbers;

/// `raw()` is the inverse of `TryFrom<u64>`: the number the guest puts in `a7`
/// must decode back to the variant it came from. Runs over `ALL`, so a syscall
/// whose `raw()` collides with another's is caught here rather than by a guest
/// silently taking the wrong ecall path.
#[test]
fn raw_round_trips_through_try_from() {
    for &syscall in SyscallNumbers::ALL {
        assert_eq!(
            SyscallNumbers::try_from(syscall.raw()),
            Ok(syscall),
            "a7 = {} must decode back to {syscall:?}",
            syscall.raw()
        );
    }
}

/// Two syscalls sharing an `a7` would make `TryFrom` pick one and leave the other
/// unreachable, and `ALL` is what the CLI parity test enumerates.
#[test]
fn every_syscall_has_a_distinct_a7() {
    let mut raws: Vec<u64> = SyscallNumbers::ALL.iter().map(|s| s.raw()).collect();
    let listed = raws.len();
    raws.sort_unstable();
    raws.dedup();
    assert_eq!(raws.len(), listed, "two syscalls share an a7 value");
}
