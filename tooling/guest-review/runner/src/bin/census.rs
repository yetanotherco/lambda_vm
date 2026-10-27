use executor::{
    elf::Elf,
    vm::{execution::Executor, instruction::decoding::Instruction},
};
use std::collections::BTreeMap;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let elf = Elf::load(&std::fs::read(&args[1]).unwrap()).unwrap();
    let vm = Executor::new(&elf, vec![]).unwrap();
    let n = vm.instructions.instruction_count();
    println!(
        "static_instructions\t{n}\ndecode_padded_rows\t{}",
        (n + 1).next_power_of_two().max(2)
    );
    let mut counts = BTreeMap::<&str, u64>::new();
    for line in std::fs::read_to_string(&args[2]).unwrap().lines() {
        let (pc, count) = line.split_once('\t').unwrap();
        let pc = pc.parse().unwrap();
        let count = count.parse::<u64>().unwrap();
        let kind = match vm.instructions.get(pc).unwrap() {
            Instruction::Load { .. } => "loads",
            Instruction::Store { .. } => "stores",
            Instruction::EcallEbreak => "ecalls",
            _ => "other",
        };
        *counts.entry(kind).or_default() += count;
    }
    for (kind, count) in counts {
        println!("{kind}\t{count}");
    }
}
