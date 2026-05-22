//! Tests for statement encoding and Fiat-Shamir transcript seeding.

use crate::statement::{encode_statement, statement_seed};
use crate::{RuntimePageRange, TableCounts};

fn sample_counts() -> TableCounts {
    TableCounts {
        cpu: 3,
        lt: 1,
        memw: 2,
        memw_aligned: 1,
        load: 1,
        mul: 1,
        dvrm: 1,
        shift: 1,
        branch: 2,
        memw_register: 1,
    }
}

fn sample_ranges() -> Vec<RuntimePageRange> {
    vec![
        RuntimePageRange {
            base: 0x1000,
            count: 4,
        },
        RuntimePageRange {
            base: 0x8000,
            count: 2,
        },
    ]
}

#[test]
fn encoding_is_deterministic() {
    let counts = sample_counts();
    let ranges = sample_ranges();
    assert_eq!(
        encode_statement(b"elf-bytes", b"output", &counts, 3, &ranges),
        encode_statement(b"elf-bytes", b"output", &counts, 3, &ranges),
    );
    assert_eq!(
        statement_seed(b"elf-bytes", b"output", &counts, 3, &ranges),
        statement_seed(b"elf-bytes", b"output", &counts, 3, &ranges),
    );
}

#[test]
fn encoding_starts_with_domain_tag() {
    let enc = encode_statement(b"", b"", &sample_counts(), 0, &[]);
    assert!(enc.starts_with(b"LAMBDAVM_STARK_STATEMENT_V1"));
}

#[test]
fn seed_depends_on_elf() {
    let c = sample_counts();
    let r = sample_ranges();
    assert_ne!(
        statement_seed(b"program-a", b"out", &c, 1, &r),
        statement_seed(b"program-b", b"out", &c, 1, &r),
    );
}

#[test]
fn seed_depends_on_public_output() {
    let c = sample_counts();
    let r = sample_ranges();
    assert_ne!(
        statement_seed(b"elf", b"output-1", &c, 1, &r),
        statement_seed(b"elf", b"output-2", &c, 1, &r),
    );
}

#[test]
fn seed_depends_on_table_counts() {
    let r = sample_ranges();
    let mut c2 = sample_counts();
    c2.branch += 1;
    assert_ne!(
        statement_seed(b"elf", b"out", &sample_counts(), 1, &r),
        statement_seed(b"elf", b"out", &c2, 1, &r),
    );
}

#[test]
fn seed_depends_on_private_input_pages() {
    let c = sample_counts();
    let r = sample_ranges();
    assert_ne!(
        statement_seed(b"elf", b"out", &c, 1, &r),
        statement_seed(b"elf", b"out", &c, 2, &r),
    );
}

#[test]
fn seed_depends_on_runtime_page_ranges() {
    let c = sample_counts();
    assert_ne!(
        statement_seed(b"elf", b"out", &c, 1, &sample_ranges()),
        statement_seed(b"elf", b"out", &c, 1, &[]),
    );
}

#[test]
fn length_prefix_prevents_public_output_boundary_collision() {
    // Without the public_output length prefix, "empty output + cpu count 0x41"
    // and "output [0x41] + cpu count 0" would encode to the same bytes. The
    // length prefix keeps the two statements distinct.
    let mut counts_a = sample_counts();
    counts_a.cpu = 0x41;
    let mut counts_b = sample_counts();
    counts_b.cpu = 0;
    assert_ne!(
        encode_statement(b"elf", b"", &counts_a, 0, &[]),
        encode_statement(b"elf", b"\x41", &counts_b, 0, &[]),
    );
}

/// End-to-end: an honest proof verifies against its own program, and a proof
/// must not verify against a different program -- the verifier's statement seed
/// (built from the other ELF) diverges from the prover's, so every Fiat-Shamir
/// challenge differs. Also exercises seed consistency across the prove path,
/// the verify path, and the bus-balance transcript replay.
#[test]
fn proof_binds_the_program_it_was_generated_for() {
    let rust_artifacts = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("executor/program_artifacts/rust");
    let read = |name: &str| {
        std::fs::read(rust_artifacts.join(name))
            .unwrap_or_else(|_| panic!("{name} not found -- run `make compile-programs-rust`"))
    };
    let allocator = read("allocator.elf");
    let pure_commit = read("pure_commit.elf");

    let proof = crate::prove(&allocator).expect("prove allocator");
    assert!(
        crate::verify(&proof, &allocator).expect("verify allocator"),
        "honest proof must verify against its own program",
    );
    assert!(
        !matches!(crate::verify(&proof, &pure_commit), Ok(true)),
        "a proof must not verify against a different program",
    );
}
