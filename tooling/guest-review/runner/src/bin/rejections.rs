use ethrex_guest_program::l1::{ProgramInput, execution_program};
use executor::{elf::Elf, vm::execution::Executor};
use std::sync::Arc;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let original = std::fs::read(&args[1]).unwrap();
    for variant in [
        "state-root",
        "receipts-root",
        "missing-state",
        "missing-code",
    ] {
        let mut input = rkyv::from_bytes::<ProgramInput, rkyv::rancor::Error>(&original).unwrap();
        match variant {
            "state-root" => input.blocks[0].header.state_root = Default::default(),
            "receipts-root" => input.blocks[0].header.receipts_root = Default::default(),
            "missing-state" => input.execution_witness.state_trie_root = None,
            "missing-code" => input.execution_witness.codes.clear(),
            _ => unreachable!(),
        }
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&input)
            .unwrap()
            .to_vec();
        assert!(
            execution_program(input, Arc::new(ethrex_crypto::NativeCrypto)).is_err(),
            "native accepted {variant}"
        );
        for elf_path in &args[2..] {
            let elf = Elf::load(&std::fs::read(elf_path).unwrap()).unwrap();
            let mut vm = Executor::new(&elf, bytes.clone()).unwrap();
            let mut cycles = 0;
            loop {
                match vm.resume() {
                    Ok(Some(logs)) => {
                        cycles += logs.len();
                        assert!(cycles < 100_000_000, "rejection exceeded cycle budget");
                    }
                    Ok(None) => panic!("guest accepted {variant}: {elf_path}"),
                    Err(error) => {
                        assert!(
                            error.to_string().contains("Panic called"),
                            "unexpected rejection: {error}"
                        );
                        println!("{variant}\t{elf_path}\trejected");
                        break;
                    }
                }
            }
        }
    }
}
