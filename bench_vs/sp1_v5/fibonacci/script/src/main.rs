use sp1_sdk::{Prover, ProverClient, SP1Proof, SP1Stdin};
use std::time::Instant;

fn main() {
    sp1_sdk::utils::setup_logger();

    let mut args = std::env::args().skip(1);
    let elf_path = args.next().expect("Usage: fibonacci-script-v5 <elf> <n>");
    let n: u64 = args
        .next()
        .expect("Usage: fibonacci-script-v5 <elf> <n>")
        .parse()
        .expect("n must be a u64");

    let elf = std::fs::read(&elf_path).expect("failed to read ELF");

    let client = ProverClient::builder().cpu().build();
    let mut stdin = SP1Stdin::new();
    stdin.write(&n);

    // Cycle count — outside the timer.
    let (_, report) = client.execute(&elf, &stdin).run().unwrap();
    println!("Cycles: {}", report.total_instruction_count());

    // Timed window: setup + core proof only.
    let start = Instant::now();
    let (pk, _vk) = client.setup(&elf);
    let proof = client
        .prove(&pk, &stdin)
        .core()
        .run()
        .expect("prove failed");
    let elapsed = start.elapsed();

    println!("Proving time: {:.3}s", elapsed.as_secs_f64());

    // Count main-trace field elements.
    // For each chip: rows = 1 << log_degree, cols = main.local.len().
    let total_elements: usize = match &proof.proof {
        SP1Proof::Core(shards) => shards
            .iter()
            .map(|shard| {
                shard
                    .chip_ordering
                    .values()
                    .map(|&idx| {
                        let chip = &shard.opened_values.chips[idx];
                        let rows = 1usize << chip.log_degree;
                        let cols = chip.main.local.len();
                        rows * cols
                    })
                    .sum::<usize>()
            })
            .sum(),
        _ => 0,
    };
    println!("Elements: {}", total_elements);

    // Count auxiliary (permutation/interaction) field elements.
    // For each chip: rows = 1 << log_degree, aux_cols = permutation.local.len().
    // These are extension-field columns used for bus argument (LogUp/permutation).
    let aux_elements: usize = match &proof.proof {
        SP1Proof::Core(shards) => shards
            .iter()
            .map(|shard| {
                shard
                    .chip_ordering
                    .values()
                    .map(|&idx| {
                        let chip = &shard.opened_values.chips[idx];
                        let rows = 1usize << chip.log_degree;
                        chip.permutation.local.len() * rows
                    })
                    .sum::<usize>()
            })
            .sum(),
        _ => 0,
    };
    println!("Aux elements: {}", aux_elements);
}
