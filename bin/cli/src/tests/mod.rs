use super::*;
use clap::CommandFactory;

// The arg graph is well-formed (e.g. `requires`/`conflicts_with` reference real args).
#[test]
fn cli_command_is_valid() {
    Cli::command().debug_assert();
}

// The continuation epoch flag requires --continuations.
#[test]
fn epoch_size_log2_requires_continuations() {
    let r = Cli::command().try_get_matches_from([
        "cli",
        "prove",
        "prog.elf",
        "-o",
        "out",
        "--epoch-size-log2",
        "20",
    ]);
    assert!(r.is_err());
}

#[test]
fn epoch_size_log2_accepts_continuations() {
    let r = Cli::command().try_get_matches_from([
        "cli",
        "prove",
        "prog.elf",
        "-o",
        "out",
        "--continuations",
        "--epoch-size-log2",
        "20",
    ]);
    assert!(r.is_ok());
}

#[test]
fn cycles_accepts_continuations() {
    let r = Cli::command().try_get_matches_from([
        "cli",
        "prove",
        "prog.elf",
        "-o",
        "out",
        "--continuations",
        "--cycles",
    ]);
    assert!(r.is_ok());
}

#[test]
fn elements_conflicts_with_continuations() {
    let r = Cli::command().try_get_matches_from([
        "cli",
        "prove",
        "prog.elf",
        "-o",
        "out",
        "--continuations",
        "--elements",
    ]);
    assert!(r.is_err());
}

#[test]
fn epoch_size_log2_rejects_tiny_cli_values() {
    let r = Cli::command().try_get_matches_from([
        "cli",
        "prove",
        "prog.elf",
        "-o",
        "out",
        "--continuations",
        "--epoch-size-log2",
        "17",
    ]);
    assert!(r.is_err());
}

#[test]
fn old_epoch_size_flag_is_rejected() {
    let r = Cli::command().try_get_matches_from([
        "cli",
        "prove",
        "prog.elf",
        "-o",
        "out",
        "--continuations",
        "--epoch-size",
        "1048576",
    ]);
    assert!(r.is_err());
}

#[test]
fn old_num_epochs_flag_is_rejected() {
    let r = Cli::command().try_get_matches_from([
        "cli",
        "prove",
        "prog.elf",
        "-o",
        "out",
        "--continuations",
        "--num-epochs",
        "4",
    ]);
    assert!(r.is_err());
}

#[test]
fn prove_help_omits_removed_epoch_flags() {
    let mut cmd = Cli::command();
    let prove = cmd.find_subcommand_mut("prove").unwrap();
    let mut help = Vec::new();
    prove.write_long_help(&mut help).unwrap();
    let help = String::from_utf8(help).unwrap();

    assert!(help.contains("--epoch-size-log2 <N>"));
    assert!(!help.contains("--num-epochs"));
    assert!(!help.contains("--epoch-size <"));
}

#[test]
fn continuation_epoch_size_rejects_tiny_cli_values() {
    assert!(continuation_epoch_size(17).is_err());
}

#[test]
fn continuation_epoch_size_uses_exact_power_of_two() {
    assert_eq!(continuation_epoch_size(20).unwrap(), 1 << 20);
}

// `accelerator_of` must match the prover's `CpuOperation::from_log`: count an
// invocation only when the instruction is an ECALL AND a7 is the accelerator
// syscall number. Covers both accelerators, the non-accelerator syscalls, a
// non-ECALL whose src1 collides with an accelerator number, and a cache miss.
#[test]
fn accelerator_of_mirrors_prover_classification() {
    use executor::vm::instruction::execution::{ECSM_SYSCALL_NUMBER, KECCAK_SYSCALL_NUMBER};

    let ecall = Instruction::EcallEbreak;

    assert_eq!(
        accelerator_of(Some(&ecall), KECCAK_SYSCALL_NUMBER),
        Some(Accelerator::Keccak)
    );
    assert_eq!(
        accelerator_of(Some(&ecall), ECSM_SYSCALL_NUMBER),
        Some(Accelerator::Ecsm)
    );

    // Non-accelerator syscalls (Commit=64, Halt=93) count as neither.
    assert_eq!(
        accelerator_of(Some(&ecall), SyscallNumbers::Commit as u64),
        None
    );
    assert_eq!(
        accelerator_of(Some(&ecall), SyscallNumbers::Halt as u64),
        None
    );

    // A non-ECALL instruction whose src1 happens to equal an accelerator a7
    // must not count — this is the `f.ecall &&` guard the prover applies.
    assert_eq!(
        accelerator_of(Some(&Instruction::Fence), KECCAK_SYSCALL_NUMBER),
        None
    );

    // No decoded instruction at the pc (cache miss) counts as neither.
    assert_eq!(accelerator_of(None, KECCAK_SYSCALL_NUMBER), None);
}
