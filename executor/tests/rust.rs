use executor::{
    elf::Elf,
    vm::execution::{ReturnValues, run_program},
};

fn run_program_without_expect(
    elf_path: &str,
    private_inputs: Vec<u8>,
) -> Result<(ReturnValues, Vec<executor::vm::logs::Log>), executor::vm::execution::ExecutorError> {
    println!("Testing {}", elf_path);
    let elf_data = std::fs::read(elf_path).unwrap();
    let program = Elf::load(&elf_data).unwrap();
    println!("Program entry: 0x{:08x}", program.entry_point);
    program.image.iter().for_each(|(addr, word)| {
        println!("0x{:08x}: 0x{:08x}", addr, word);
    });

    run_program(program.image, program.entry_point, private_inputs)
}

fn run_program_and_check_public_output(
    elf_path: &str,
    expected_output: Vec<u8>,
    private_inputs: Vec<u8>,
) {
    let (results, _logs) =
        run_program_without_expect(elf_path, private_inputs).expect("Failed to run program");

    assert!(results.memory_values == expected_output);
}

fn run_program_and_check_output(elf_path: &str, expected_output: i32, private_inputs: Vec<u8>) {
    let (results, _logs) =
        run_program_without_expect(elf_path, private_inputs).expect("Failed to run program");

    assert!(results.register_values.0 == expected_output);
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
    let mut output = vec![];
    let size = 100000u32;
    for _ in 0..size {
        output.push(1);
    }
    run_program_and_check_public_output(
        "./program_artifacts/rust/memory.elf",
        output[(size - 1000) as usize..].to_vec(),
        size.to_be_bytes().to_vec(),
    );
}
