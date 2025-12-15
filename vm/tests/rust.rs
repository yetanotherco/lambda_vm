use vm::{elf::Elf, vm::execution::run_program};

fn run_program_and_check_output(elf_path: &str, expected_output: i32) {
    println!("Testing {}", elf_path);
    let elf_data = std::fs::read(elf_path).unwrap();
    let program = Elf::load(&elf_data).unwrap();
    println!("Program entry: 0x{:08x}", program.entry_point);
    program.image.iter().for_each(|(addr, word)| {
        println!("0x{:08x}: 0x{:08x}", addr, word);
    });
    let results = run_program(program.image, program.entry_point);

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
    println!("Testing fibonacci.elf");
    let elf_data = std::fs::read("./program_artifacts/rust/fibonacci.elf").unwrap();
    let program = Elf::load(&elf_data).unwrap();
    println!("Program entry: 0x{:08x}", program.entry_point);
    program.image.iter().for_each(|(addr, word)| {
        println!("0x{:08x}: 0x{:08x}", addr, word);
    });

    let results = run_program(program.image, program.entry_point);

    assert!(results.0 == 1597);
}
