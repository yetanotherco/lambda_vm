use anyhow::Result;
use zisk_sdk::{include_elf, ElfBinary, ProverClient, ZiskStdin};

const ELF: ElfBinary = include_elf!("fibonacci-zisk-guest");

fn main() -> Result<()> {
    let n: u64 = std::env::args()
        .nth(1)
        .expect("Usage: fibonacci-zisk-host <n>")
        .parse()
        .expect("n must be a u64");

    let stdin = ZiskStdin::new();
    stdin.write(&n);

    // emu() backend works on every supported platform; asm() is Linux/x86_64-only.
    let client = ProverClient::builder().emu().build()?;
    let (pk, _vk) = client.setup(&ELF)?;

    let result = client.execute(&pk, stdin)?;

    let total_elements: u64 = result
        .planning_info
        .planning_info
        .iter()
        .map(|a| a.num_instances as u64 * a.num_rows as u64 * a.num_columns_trace)
        .sum();

    println!("Cycles: {}", result.get_execution_steps());
    println!("Elements: {}", total_elements);
    Ok(())
}
