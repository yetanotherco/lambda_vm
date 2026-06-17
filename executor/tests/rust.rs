use executor::{
    elf::Elf,
    vm::execution::{Executor, ReturnValues},
};

// NOTE: These tests require 64-bit RISC-V ELF files (RV64IM).
// The test binaries need to be recompiled for 64-bit architecture.
fn run_program_without_expect(
    elf_path: &str,
    private_inputs: Vec<u8>,
) -> Result<ReturnValues, executor::vm::execution::ExecutorError> {
    println!("Testing {}", elf_path);
    let elf_data = std::fs::read(elf_path).unwrap();
    let program = Elf::load(&elf_data).unwrap();
    println!("Program entry: 0x{:016x}", program.entry_point);
    let mut executor = Executor::new(&program, private_inputs)?;
    while let Some(_logs) = executor.resume()? {}
    executor.finish()
}

fn run_program_and_check_public_output(
    elf_path: &str,
    expected_output: Vec<u8>,
    private_inputs: Vec<u8>,
) {
    let result =
        run_program_without_expect(elf_path, private_inputs).expect("Failed to run program");

    assert_eq!(result.memory_values, expected_output);
}

fn run_program_and_check_output(elf_path: &str, expected_output: i64, private_inputs: Vec<u8>) {
    let result =
        run_program_without_expect(elf_path, private_inputs).expect("Failed to run program");

    assert!(result.register_values.0 == expected_output);
}

#[test]
fn test_basic_rust() {
    run_program_and_check_output("./program_artifacts/rust/basic_rust.elf", 0, vec![]);
}

#[test]
fn test_add() {
    run_program_and_check_output("./program_artifacts/rust/add.elf", 3, vec![]);
}

#[test]
fn test_if() {
    run_program_and_check_output("./program_artifacts/rust/if.elf", 10, vec![]);
}

#[test]
fn test_fibonacci() {
    run_program_and_check_output("./program_artifacts/rust/fibonacci.elf", 55, vec![]);
}

#[test]
fn test_fibonacci_iterative() {
    run_program_and_check_output(
        "./program_artifacts/rust/fibonacci_iterative.elf",
        1597,
        vec![],
    );
}

#[test]
fn test_byte() {
    run_program_and_check_output("./program_artifacts/rust/byte.elf", 0xDE, vec![]);
}

#[test]
fn test_byte_signed() {
    run_program_and_check_output("./program_artifacts/rust/byte_signed.elf", -8, vec![]);
}

#[test]
fn test_half() {
    run_program_and_check_output("./program_artifacts/rust/half.elf", 0xDEAD, vec![]);
}

#[test]
fn test_half_signed() {
    run_program_and_check_output(
        "./program_artifacts/rust/half_signed.elf",
        0xBEEF - 0xDEAD,
        vec![],
    );
}

#[test]
fn test_rlp() {
    run_program_and_check_output("./program_artifacts/rust/rlp.elf", 65536, vec![]);
}

#[test]
fn test_allocator() {
    run_program_and_check_public_output(
        "./program_artifacts/rust/allocator.elf",
        b"Hello World".to_vec(),
        vec![],
    );
}

#[test]
fn test_ethereum_types() {
    run_program_and_check_output("./program_artifacts/rust/ethereum_types.elf", 1, vec![]);
}

#[test]
fn test_vector() {
    run_program_and_check_public_output(
        "./program_artifacts/rust/vector.elf",
        [1, 2, 3, 4, 5].to_vec(),
        vec![],
    );
}

#[test]
fn test_hashmap() {
    run_program_and_check_output("./program_artifacts/rust/hashmap.elf", 3, vec![]);
}

#[test]
fn test_asm() {
    run_program_and_check_output("./program_artifacts/rust/asm.elf", 42, vec![]);
}

#[test]
fn test_print() {
    run_program_and_check_output("./program_artifacts/rust/print.elf", 1, vec![]);
}

#[test]
fn test_stdout() {
    run_program_and_check_output("./program_artifacts/rust/stdout.elf", 1, vec![]);
}

#[test]
fn test_panic() {
    let result = run_program_without_expect("./program_artifacts/rust/panic.elf", vec![]);
    assert!(result.is_err());
    if let Err(executor::vm::execution::ExecutorError::ExecutionError(
        executor::vm::instruction::execution::ExecutionError::Panic(msg),
    )) = result
    {
        assert_eq!(msg, "This is a panic test");
    } else {
        panic!("Expected panic error");
    }
}

#[test]
fn test_commit() {
    run_program_and_check_public_output(
        "./program_artifacts/rust/commit.elf",
        vec![4, 5, 6, 7],
        vec![4, 5, 6, 7],
    );
}

#[test]
fn test_ef_io_demo_concatenates_writes() {
    // Demo guest reads its private input via EF `read_input`, then emits it
    // back as the public output via TWO `write_output` calls (split in halves).
    // The COMMIT AIR concatenates the two calls; the executor's
    // `commit_public_output` appends in the same order.
    let input: Vec<u8> = b"hello world!".to_vec();
    run_program_and_check_public_output(
        "./program_artifacts/rust/ef_io_demo.elf",
        input.clone(),
        input,
    );
}

#[test]
fn test_commit_sum() {
    run_program_and_check_public_output(
        "./program_artifacts/rust/commit_sum.elf",
        10u8.to_le_bytes().to_vec(),
        vec![3, 7],
    );
}

#[test]
fn test_serde() {
    #[derive(serde::Serialize)]
    struct MyData {
        val: i32,
        values: Vec<u8>,
    }
    let my_data = MyData {
        val: 42,
        values: vec![1, 2, 3, 4, 5],
    };
    let serialized = serde_json::to_vec(&my_data).unwrap();
    run_program_and_check_public_output("./program_artifacts/rust/serde.elf", serialized, vec![]);
}

#[test]
fn test_random() {
    let result = run_program_without_expect("./program_artifacts/rust/random.elf", vec![]);
    assert!(result.is_err());
    if let Err(executor::vm::execution::ExecutorError::ExecutionError(
        executor::vm::instruction::execution::ExecutionError::Panic(msg),
    )) = result
    {
        assert_eq!(msg, "getrandom is not supported");
    } else {
        panic!("Expected rand error");
    }
}

#[test]
fn test_memory() {
    let size = 100000u32;
    let output = vec![1; size as usize];
    run_program_and_check_public_output(
        "./program_artifacts/rust/memory.elf",
        output[(size - 1000) as usize..].to_vec(),
        size.to_be_bytes().to_vec(),
    );
}

#[test]
fn test_keccak() {
    use tiny_keccak::Hasher;
    let input_a = b"hello world";
    let input_b = b"!";
    let mut output = [0u8; 32];
    let mut hasher = tiny_keccak::Keccak::v256();
    hasher.update(input_a);
    hasher.update(input_b);
    hasher.finalize(&mut output);
    run_program_and_check_public_output(
        "./program_artifacts/rust/keccak.elf",
        output.to_vec(),
        vec![],
    );
}

#[test]
fn test_stdin_read_panics() {
    let result = run_program_without_expect("./program_artifacts/rust/stdin_read.elf", vec![]);
    assert!(result.is_err());
    if let Err(executor::vm::execution::ExecutorError::ExecutionError(
        executor::vm::instruction::execution::ExecutionError::Panic(msg),
    )) = result
    {
        assert!(
            msg.contains("sys_read is not supported"),
            "Expected sys_read panic, got: {}",
            msg
        );
    } else {
        panic!("Expected panic error for stdin_read");
    }
}

#[test]
fn test_args_panics() {
    let result = run_program_without_expect("./program_artifacts/rust/args_test.elf", vec![]);
    assert!(result.is_err());
    if let Err(executor::vm::execution::ExecutorError::ExecutionError(
        executor::vm::instruction::execution::ExecutionError::Panic(msg),
    )) = result
    {
        assert!(
            msg.contains("sys_argc is not supported"),
            "Expected sys_argc panic, got: {}",
            msg
        );
    } else {
        panic!("Expected panic error for args_test");
    }
}

/// Larger-block smoke test: a synthetic ethrex block with 10 ETH transfers.
/// (Replaces the old `ethrex_hoodi.bin` real-block fixture, which was in the
/// pre-Crypto-trait ethrex format and no longer deserializes.) Fixture is
/// generated by `tooling/ethrex-fixtures`; see `tests/README.md`.
#[ignore = "heavier synthetic block (10 txs); run in the dedicated --ignored CI step"]
#[test]
fn test_ethrex() {
    use ethrex_guest_program::crypto::NativeCrypto;
    use ethrex_guest_program::l1::{ProgramInput, execution_program};
    use rkyv::rancor::Error;
    use std::fs;
    use std::sync::Arc;
    let inputs = fs::read("tests/ethrex_10_transfers.bin").unwrap();
    let input = rkyv::from_bytes::<ProgramInput, Error>(&inputs).unwrap();
    let output = execution_program(input, Arc::new(NativeCrypto)).unwrap();
    run_program_and_check_public_output(
        "./program_artifacts/rust/ethrex.elf",
        output.encode(),
        inputs,
    );
}

/// Executes a stateless ethrex block containing a single (plain ETH transfer)
/// transaction. Execution only — no proving — against the ethrex guest ELF
/// built from the same pinned ethrex revision as the native reference. The
/// fixture is a serialized `ProgramInput`; see `tests/README.md` for provenance.
///
/// The fixture is generated by `tooling/ethrex-fixtures` at the same ethrex rev
/// as the guest (see `tests/README.md`).
#[test]
fn test_ethrex_simple_tx() {
    use ethrex_guest_program::crypto::NativeCrypto;
    use ethrex_guest_program::l1::{ProgramInput, execution_program};
    use rkyv::rancor::Error;
    use std::sync::Arc;
    let inputs = std::fs::read("tests/ethrex_simple_tx.bin").unwrap();
    let input = rkyv::from_bytes::<ProgramInput, Error>(&inputs).unwrap();
    let output = execution_program(input, Arc::new(NativeCrypto)).unwrap();
    run_program_and_check_public_output(
        "./program_artifacts/rust/ethrex.elf",
        output.encode(),
        inputs,
    );
}

#[ignore = "Ignored until the vm is fast enough to run this test"]
#[test]
fn test_ckzg() {
    run_program_and_check_public_output("./program_artifacts/rust/ckzg.elf", vec![1, 1], vec![]);
}
