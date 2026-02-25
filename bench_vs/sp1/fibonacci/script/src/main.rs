use sp1_sdk::blocking::{ProveRequest, Prover, ProverClient};
use sp1_sdk::{include_elf, ProvingKey, SP1Stdin};
use std::time::Instant;

const FIB_ELF: sp1_sdk::Elf = include_elf!("fibonacci-program");

fn main() {
    sp1_sdk::utils::setup_logger();

    let n: u64 = std::env::args()
        .nth(1)
        .expect("Usage: fibonacci-script <n>")
        .parse()
        .expect("n must be a u64");

    let client = ProverClient::from_env();
    let mut stdin = SP1Stdin::new();
    stdin.write(&n);

    // Setup
    let pk = client.setup(FIB_ELF.clone()).expect("setup failed");

    // Execute for cycle count
    let (_, report) = client
        .execute(FIB_ELF.clone(), stdin.clone())
        .run()
        .unwrap();
    println!("Cycles: {}", report.total_instruction_count());

    // Core proof (no recursion)
    let start = Instant::now();
    let proof = client
        .prove(&pk, stdin)
        .core()
        .run()
        .expect("prove failed");
    let elapsed = start.elapsed();

    println!("Proving time: {:.3}s", elapsed.as_secs_f64());

    // Verify
    client
        .verify(&proof, pk.verifying_key(), None)
        .expect("verify failed");

    println!("Proof verified successfully");
}
