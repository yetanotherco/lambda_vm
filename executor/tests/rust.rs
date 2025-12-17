use executor::{elf::Elf, vm::execution::run_program};

fn run_program_and_check_output(elf_path: &str, expected_output: i32) {
    println!("Testing {}", elf_path);
    let elf_data = std::fs::read(elf_path).unwrap();
    let program = Elf::load(&elf_data).unwrap();
    println!("Program entry: 0x{:08x}", program.entry_point);
    program.image.iter().for_each(|(addr, word)| {
        println!("0x{:08x}: 0x{:08x}", addr, word);
    });

    let (results, _logs) =
        run_program(program.image, program.entry_point).expect("Failed to run program");

    assert!(results.0 == expected_output);
}

#[test]
fn test_basic_rust() {
    run_program_and_check_output("./program_artifacts/rust/basic_rust.elf", 0);
}

#[test]
fn test_add() {
    run_program_and_check_output("./program_artifacts/rust/add.elf", 3);
}

#[test]
fn test_if() {
    run_program_and_check_output("./program_artifacts/rust/if.elf", 10);
}

#[test]
fn test_fibonacci() {
    run_program_and_check_output("./program_artifacts/rust/fibonacci.elf", 1597);
}

#[test]
fn test_fibonacci_iterative() {
    run_program_and_check_output("./program_artifacts/rust/fibonacci_iterative.elf", 1597);
}

#[test]
fn test_byte() {
    run_program_and_check_output("./program_artifacts/rust/byte.elf", 0xDE);
}

#[test]
fn test_byte_signed() {
    run_program_and_check_output("./program_artifacts/rust/byte_signed.elf", -8);
}

#[test]
fn test_half() {
    run_program_and_check_output("./program_artifacts/rust/half.elf", 0xDEAD);
}

#[test]
fn test_half_signed() {
    run_program_and_check_output("./program_artifacts/rust/half_signed.elf", 0xBEEF - 0xDEAD);
}
