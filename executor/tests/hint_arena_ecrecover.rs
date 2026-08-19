//! Two-pass hint-arena measurement driver for the `ecrecover_hints` guest.
//!
//! Pass 1 runs with an empty arena: every hint request misses, is appended to
//! the guest's request log, and the guest recomputes in software (correct,
//! just slow). The host then answers the logged requests with `compute_hint`,
//! and pass 2 re-runs with a complete arena — the provable configuration.
//!
//! Run explicitly (prints cycle counts; the guest ELF must be built):
//!   cargo test -p executor --test hint_arena_ecrecover -- --ignored --nocapture

use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::instruction::execution::compute_hint;
use std::time::Instant;

/// Number of ecrecovers the guest performs (3 hint requests each).
const N: usize = 30;

/// Build the guest's private input: `[u32 LE count]` then `count` records of
/// `sig(64) || recid(1) || msg(32)`. Any `(r, s, msg)` with `r, s` nonzero and
/// in range is a legal ecrecover input, but `r³ + 7` must be a quadratic
/// residue for decompression to succeed — so `r` values are rejection-sampled
/// for residuosity. `recid` alternates parity.
fn build_input() -> Vec<u8> {
    use k256::{FieldElement, Scalar};

    let mut out = Vec::with_capacity(4 + N * 97);
    out.extend_from_slice(&(N as u32).to_le_bytes());

    let mut r_int = 0u64;
    for i in 0..N {
        // Next r whose rhs = r³ + 7 is a residue.
        let (r_bytes, y_is_odd) = loop {
            r_int += 1;
            let r = FieldElement::from(r_int);
            let rhs = r * r * r + FieldElement::from(7u64);
            if let Some(y) = Option::<FieldElement>::from(rhs.sqrt()) {
                // ecrecover parses r as a *scalar* (r < n); small ints qualify.
                // `sqrt` yields an unnormalized element; `is_odd` requires a
                // normalized one.
                break (r.to_bytes(), bool::from(y.normalize().is_odd()));
            }
        };
        let s = Scalar::from(1000u64 + i as u64);
        let msg_byte = (i as u8).wrapping_mul(7).wrapping_add(1);

        out.extend_from_slice(&r_bytes);
        out.extend_from_slice(&s.to_bytes());
        out.push(u8::from(y_is_odd));
        out.extend_from_slice(&[msg_byte; 32]);
    }
    out
}

#[test]
#[ignore = "measurement driver — run explicitly"]
fn ecrecover_two_pass_cycles() {
    let elf_bytes = std::fs::read("program_artifacts/rust/ecrecover_hints.elf")
        .expect("ecrecover_hints.elf missing — run `make executor/program_artifacts/rust/ecrecover_hints.elf`");
    let elf = Elf::load(&elf_bytes).expect("ELF load");
    let input = build_input();

    // ── Pass 1: empty arena — all hints miss, logged, software fallback. ──
    let t0 = Instant::now();
    let pass1 = Executor::new(&elf, input.clone(), &[])
        .expect("executor")
        .run()
        .expect("pass 1");
    let pass1_time = t0.elapsed();
    let pass1_cycles = pass1.logs.len();
    assert_eq!(
        pass1.hint_requests.len(),
        3 * N,
        "every recovery must log sqrt + scalar_inv + field_inv"
    );

    // Host answers the requests, in order — that IS the arena.
    let hints: Vec<[u8; 32]> = pass1
        .hint_requests
        .iter()
        .map(|(id, input)| compute_hint(*id, input))
        .collect();

    // ── Pass 2: complete arena — every hint hits and verifies. ──
    let t0 = Instant::now();
    let pass2 = Executor::new(&elf, input.clone(), &hints)
        .expect("executor")
        .run()
        .expect("pass 2");
    let pass2_time = t0.elapsed();
    let pass2_cycles = pass2.logs.len();
    assert!(pass2.hint_requests.is_empty(), "arena must cover pass 2");

    // Same program, same input: identical committed output.
    assert_eq!(
        pass1.return_values.memory_values, pass2.return_values.memory_values,
        "hint source must not change the result"
    );

    println!("[ecrecover-hints] N = {N} recoveries");
    println!(
        "[ecrecover-hints] pass 1 (software fallback): {pass1_cycles} cycles in {pass1_time:?}"
    );
    println!(
        "[ecrecover-hints] pass 2 (arena hints):       {pass2_cycles} cycles in {pass2_time:?}"
    );
    println!(
        "[ecrecover-hints] guest cycle ratio pass1/pass2: {:.2}x",
        pass1_cycles as f64 / pass2_cycles as f64
    );

    // Dump the fixtures the prover-side continuation measurement consumes:
    // `<stem>.input.bin` (the private input) and `<stem>.hints.bin` (the arena
    // slots, concatenated).
    let dir = std::path::Path::new("program_artifacts/rust");
    let input_path = dir.join("ecrecover_hints.input.bin");
    let hints_path = dir.join("ecrecover_hints.hints.bin");
    std::fs::write(&input_path, &input).expect("write input fixture");
    let mut hints_bin = Vec::with_capacity(32 * hints.len());
    for h in &hints {
        hints_bin.extend_from_slice(h);
    }
    std::fs::write(&hints_path, &hints_bin).expect("write hints fixture");
    println!("[ecrecover-hints] fixtures written: {input_path:?}, {hints_path:?}");
}
