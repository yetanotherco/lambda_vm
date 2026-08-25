//! Hint-arena measurement driver for the `ecrecover_hints` guest.
//!
//! Three runs, all of the same program and input:
//!
//! - **silenced**: no hint is answered, so the guest recomputes every one in
//!   software. The fallback baseline, and the arm that pins the property the
//!   whole design rests on — an unanswered hint cannot change the result.
//! - **on demand**: the executor answers each request during the run by seeding
//!   the arena slot the guest is about to read. One execution, no recording
//!   pass. This is what the prover now runs.
//! - **explicit arena**: the arena from the on-demand run, passed up front.
//!   Must be indistinguishable from it — same cycles, same output.
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

    // ── Silenced: nothing is answered, every hint recomputed in software. ──
    let t0 = Instant::now();
    let mut silenced_ex = Executor::new(&elf, input.clone(), &[]).expect("executor");
    silenced_ex.silence_hints();
    let silenced = silenced_ex.run().expect("silenced run");
    let silenced_time = t0.elapsed();
    let silenced_cycles = silenced.logs.len();
    assert_eq!(
        silenced.hint_requests.len(),
        3 * N,
        "every recovery must log sqrt + scalar_inv + field_inv"
    );
    assert!(
        silenced.hints.is_empty(),
        "a silenced run answers nothing, so it builds no arena"
    );

    // ── On demand: answered during the run, no recording pass. ──
    let t0 = Instant::now();
    let ondemand = Executor::new(&elf, input.clone(), &[])
        .expect("executor")
        .run()
        .expect("on-demand run");
    let ondemand_time = t0.elapsed();
    let ondemand_cycles = ondemand.logs.len();
    assert_eq!(
        ondemand.hints.len(),
        3 * N,
        "one slot answered per request, in request order"
    );
    // The executor's answers are exactly what the host would have precomputed.
    let expected: Vec<[u8; 32]> = ondemand
        .hint_requests
        .iter()
        .map(|(id, input)| compute_hint(*id, input))
        .collect();
    assert_eq!(ondemand.hints, expected, "answers must be compute_hint's");

    // ── Explicit arena: the same bytes, shipped up front. ──
    let hints = ondemand.hints.clone();
    let t0 = Instant::now();
    let explicit = Executor::new(&elf, input.clone(), &hints)
        .expect("executor")
        .run()
        .expect("explicit-arena run");
    let explicit_time = t0.elapsed();
    let explicit_cycles = explicit.logs.len();
    assert_eq!(
        explicit.hints, hints,
        "an explicitly supplied arena must survive the run untouched"
    );
    assert_eq!(
        explicit_cycles, ondemand_cycles,
        "answering on demand must produce the same trace as shipping the arena"
    );

    // Same program, same input: identical committed output in all three.
    assert_eq!(
        silenced.return_values.memory_values, ondemand.return_values.memory_values,
        "an unanswered hint must not change the result"
    );
    assert_eq!(
        ondemand.return_values.memory_values, explicit.return_values.memory_values,
        "hint source must not change the result"
    );

    println!("[ecrecover-hints] N = {N} recoveries");
    println!(
        "[ecrecover-hints] silenced (software fallback): {silenced_cycles} cycles in {silenced_time:?}"
    );
    println!(
        "[ecrecover-hints] on demand (one execution):    {ondemand_cycles} cycles in {ondemand_time:?}"
    );
    println!(
        "[ecrecover-hints] explicit arena:               {explicit_cycles} cycles in {explicit_time:?}"
    );
    println!(
        "[ecrecover-hints] guest cycle ratio silenced/hinted: {:.2}x",
        silenced_cycles as f64 / ondemand_cycles as f64
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
