//! FAIR CPU structural comparison: batched (multi-merkle-tree) vs per-table
//! prover, BOTH on CPU — run WITHOUT `--features cuda` so neither touches the
//! GPU. Interleaved A/B/A/B + warm to kill thermal bias. Isolates the
//! multi-merkle-tree's structural cost from the (portable) GPU-optimization gap.
//!
//!   cargo bench -p lambda-vm-prover --bench batched_cpu_cmp
//! Env: BLOCK=ethrex_simple_tx.bin  WARMUP=1  ITERS=3  COOLDOWN_SECS=0

use std::time::Duration;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn median(v: &[Duration]) -> Duration {
    v[v.len() / 2]
}

fn main() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let block = std::env::var("BLOCK").unwrap_or_else(|_| "ethrex_simple_tx.bin".to_string());
    let elf = std::fs::read(format!(
        "{manifest}/../executor/program_artifacts/rust/ethrex.elf"
    ))
    .expect("read ethrex.elf");
    let input =
        std::fs::read(format!("{manifest}/../executor/tests/{block}")).expect("read block input");

    let warmup = env_usize("WARMUP", 1);
    let iters = env_usize("ITERS", 3);
    let cooldown = env_usize("COOLDOWN_SECS", 0) as u64;
    if cfg!(feature = "cuda") {
        println!("⚠️  built with cuda — run WITHOUT --features cuda for the FAIR CPU comparison");
    }
    println!("=== CPU structural comparison — block={block} warmup={warmup} iters={iters} cooldown={cooldown}s ===");
    println!("(both provers on CPU; isolates the multi-merkle-tree structure vs per-table)");

    let mut batched = Vec::new();
    let mut pertable = Vec::new();
    for i in 0..(warmup + iters) {
        let tag = if i < warmup { "warmup" } else { "iter  " };
        let b = lambda_vm_prover::time_batched_prove(&elf, &input).expect("batched prove");
        if cooldown > 0 {
            std::thread::sleep(Duration::from_secs(cooldown));
        }
        let p = lambda_vm_prover::time_per_table_prove(&elf, &input).expect("per-table prove");
        println!("  {tag} {i}: batched={b:?}  per-table={p:?}");
        if i >= warmup {
            batched.push(b);
            pertable.push(p);
        }
        if cooldown > 0 {
            std::thread::sleep(Duration::from_secs(cooldown));
        }
    }

    batched.sort();
    pertable.sort();
    let bm = median(&batched);
    let pm = median(&pertable);
    let ratio = bm.as_secs_f64() / pm.as_secs_f64();
    println!(
        "RESULT block={block}  batched median={bm:?} (min {:?})  |  per-table median={pm:?} (min {:?})  |  batched/per-table = {ratio:.3}",
        batched[0], pertable[0],
    );
}
