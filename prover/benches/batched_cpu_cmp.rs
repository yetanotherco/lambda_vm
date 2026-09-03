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
    // VMPROVE=1: exercise the REAL VM API (`prove_with_inputs` + `verify`) after
    // the batched cutover — the VM now proves/verifies with the multi-merkle-tree.
    // This is the path a comparison-against-main harness hits. Build --features cuda.
    if std::env::var("VMPROVE").is_ok() {
        let t = std::time::Instant::now();
        let vm_proof = lambda_vm_prover::prove_with_inputs(&elf, &input).expect("VM prove");
        let dt = t.elapsed();
        let ok = lambda_vm_prover::verify(&vm_proof, &elf).expect("VM verify");
        println!("VMPROVE block={block}  prove={dt:?}  verify={ok}");
        return;
    }

    // VERIFY=1: prove THEN verify a real block with the batched prover, device
    // paths engaged (build with --features cuda). Closes the correctness loop at
    // real-block scale (the timing/size paths only prove).
    if std::env::var("VERIFY").is_ok() {
        match lambda_vm_prover::prove_and_verify_batched_block(&elf, &input) {
            Ok(true) => println!("VERIFY block={block}  batched prove->verify: PASS"),
            Ok(false) => println!("VERIFY block={block}  batched prove->verify: FAIL (rejected)"),
            Err(e) => println!("VERIFY block={block}  errored: {e:?}"),
        }
        return;
    }

    // SIZE=1: measure the serialized PROOF SIZE of both provers (the
    // multi-merkle-tree's structural payoff — one shared auth path per query).
    // Byte-identical on CPU or GPU, so this needs no GPU. Runs one prove each.
    if std::env::var("SIZE").is_ok() {
        let b = lambda_vm_prover::size_batched_prove(&elf, &input);
        let p = lambda_vm_prover::size_per_table_prove(&elf, &input);
        match (b, p) {
            (Ok(b), Ok(p)) => println!(
                "PROOF SIZE block={block}  batched={b} bytes ({:.2} MiB)  per-table={p} bytes ({:.2} MiB)  batched/per-table={:.3}",
                b as f64 / (1u64 << 20) as f64,
                p as f64 / (1u64 << 20) as f64,
                b as f64 / p as f64,
            ),
            (b, p) => println!("PROOF SIZE failed: batched={b:?} per-table={p:?}"),
        }
        return;
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
