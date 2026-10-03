//! `LAMBDA_VM_ZF_LOGUP` on the production tables: the pair policy keeps every
//! production constraint program byte for byte (a golden list taken at
//! #1013's head before arities existed), and the wider policies give each
//! table the layout the rule picks.

use stark::constraint_ir::ConstraintArtifact;
use stark::proof::options::{GoldilocksCubicProofOptions, LogUpPolicy, ProofFormat, ProofOptions};
use stark::traits::AIR;

use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use crate::test_utils::production_airs;

type DynAir<'a> =
    &'a dyn AIR<Field = GoldilocksField, FieldExtension = GoldilocksExtension, PublicInputs = ()>;

/// FNV-1a, 64-bit: a digest stable across toolchains (unlike `DefaultHasher`).
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// One table's shape and the digest of its serialized constraint artifact
/// (nodes, constants, roots, widths, next-row columns, part multiplier).
#[derive(Debug, PartialEq, Eq)]
struct Program {
    label: String,
    main: usize,
    aux: usize,
    parts: usize,
    digest: u64,
}

fn program(label: String, air: DynAir<'_>) -> Program {
    let (main, aux) = air.trace_layout();
    let bytes = ConstraintArtifact::capture(air)
        .to_bytes()
        .expect("serialize the artifact");
    Program {
        label,
        main,
        aux,
        parts: air.composition_poly_degree_bound(1 << 10) >> 10,
        digest: fnv1a64(&bytes),
    }
}

/// Every production VM table and every LFM chip, under `opts`.
fn programs(opts: &ProofOptions) -> Vec<Program> {
    let mut out: Vec<Program> = production_airs(opts)
        .iter()
        .map(|(label, air)| program(label.to_string(), &**air))
        .collect();
    let roots = [[0u8; 32]; crate::lfm::airs::NUM_LFM_CHIPS];
    let lfm = crate::lfm::airs::LfmAirs::new_with_hasher(
        &roots,
        opts,
        1,
        crate::hash_pin::BLOCK_HASHER,
        crate::lfm::airs::ChipSet::FULL,
    );
    for air in lfm.air_refs() {
        out.push(program(format!("LFM {}", air.name()), air));
    }
    out
}

/// Prints the list [`PAIR_GOLDEN`] pins. Run at the commit the golden is
/// taken from: `cargo test -p lambda-vm-prover --lib print_pair_programs --
/// --ignored --nocapture`.
#[test]
#[ignore = "prints the golden list"]
fn print_pair_programs() {
    let opts = GoldilocksCubicProofOptions::with_blowup(4).expect("blowup 4");
    for p in programs(&opts) {
        println!(
            "    ({:?}, {}, {}, {}, {:#018x}),",
            p.label, p.main, p.aux, p.parts, p.digest
        );
    }
}

/// (label, main, aux, parts, artifact digest) of every production program at
/// blowup 4, taken at #1013's head `d740eb5d5` (before LogUp arities
/// existed) with [`print_pair_programs`].
const PAIR_GOLDEN: &[(&str, usize, usize, usize, u64)] = &[
    ("CPU", 38, 10, 2, 0x9465e37c04806a8a),
    ("BITWISE", 21, 5, 2, 0x37f34aa68bbeed5a),
    ("LT", 17, 5, 2, 0x2064d6b3a8ab5c47),
    ("SHIFT", 29, 9, 2, 0x189f69964ab8fdc5),
    ("EQ", 12, 3, 2, 0x26d01b993f67d094),
    ("BYTEWISE", 26, 5, 2, 0xda3e7a5cdb04470e),
    ("STORE", 16, 5, 2, 0xbc7b5aab59f7df2a),
    ("CPU32", 38, 12, 2, 0x699a415d5225f494),
    ("MEMW", 49, 13, 2, 0x21e660ad393aa352),
    ("MEMW_A", 29, 10, 2, 0xb2ccfa4111575574),
    ("MEMW_R", 10, 4, 2, 0xe9debf1786822d8f),
    ("LOAD", 18, 3, 2, 0x6a6e133dbdd7c1a7),
    ("DECODE", 6, 1, 1, 0xd0bbdf17dc28c0c2),
    ("MUL", 26, 12, 2, 0x8077732dad253c96),
    ("DVRM", 34, 17, 2, 0x3f109afaefc3c81c),
    ("BRANCH", 14, 3, 2, 0x613499166a924357),
    ("HALT", 4, 18, 2, 0x67f3ad0bac9810d1),
    ("COMMIT", 19, 9, 2, 0x55513d1063e84c8b),
    ("PAGE", 5, 2, 2, 0x152d416d96f4424b),
    ("REGISTER", 5, 1, 2, 0xe3863303371a31ea),
    ("KECCAK", 511, 67, 2, 0xcd60544d65847fb6),
    ("KECCAK_RND", 1480, 516, 2, 0xe3f4611130f8d5a9),
    ("KECCAK_RC", 10, 1, 1, 0x169f15b2a9de46e8),
    ("ECSM", 667, 290, 2, 0x391b0bc19d628ce5),
    ("ECDAS", 521, 194, 2, 0x28fd0bc29cfdc389),
    ("HINT", 41, 14, 2, 0x378fe3068a210c30),
    ("L2G_GLOBAL", 9, 1, 2, 0x43b1f1c98647af72),
    ("L2G_MEMORY", 9, 3, 2, 0x8f41abf35ce0197f),
    ("GLOBAL_MEMORY", 4, 1, 2, 0x53eab0ece49443d9),
    ("LFM LFM_CONST", 7, 1, 1, 0xec8825236090a817),
    ("LFM LFM_BALU", 14, 2, 2, 0xf028386d1f694aaa),
    ("LFM LFM_XALU", 23, 2, 2, 0x695a8d933b7fd446),
    ("LFM LFM_SELECT", 25, 3, 2, 0x9b523aa9840baa2e),
    ("LFM LFM_BITDEC", 200, 34, 2, 0x18c234b1d8122b7f),
    ("LFM LFM_HASH", 329, 3, 2, 0x46a147f05a54af9b),
    ("LFM LFM_KECCAK", 792, 88, 2, 0x9677ae50ec20492b),
    ("LFM LFM_LANES", 16, 5, 2, 0xccb7cf52e8fdda60),
    ("LFM LFM_HINT", 6, 1, 1, 0x423693b15eed9cd7),
    ("LFM LFM_PUBLIC", 7, 1, 2, 0xdea976d2849e42d8),
    ("LFM LFM_RANGE", 2, 1, 1, 0x8c7cc70a7340aaf6),
    ("LFM LFM_BLAKE3", 3076, 631, 2, 0xdda3bbcc3d056b6f),
    ("LFM KECCAK_RND", 1480, 516, 2, 0xe3f4611130f8d5a9),
    ("LFM KECCAK_RC", 10, 1, 1, 0x0d0a427b3c28fdf6),
    ("LFM BITWISE", 21, 5, 2, 0x03928fcbe395a20a),
];

fn options(blowup: u8, logup: LogUpPolicy) -> ProofOptions {
    let mut opts = GoldilocksCubicProofOptions::with_blowup(blowup).expect("a valid blowup");
    opts.format = ProofFormat {
        logup,
        ..crate::zf_format::ZfFormat::DEFAULT.base_proof_format()
    };
    opts
}

/// ★ T0a: under the pair policy, every production VM table and every LFM chip
/// has today's widths, part count and constraint program, byte for byte —
/// the default format moves no compiled-kernel key and no proof. The same
/// holds for a wide policy at a blowup that cannot hold wider groups.
#[test]
fn the_pair_policy_keeps_every_production_program() {
    let golden: Vec<Program> = PAIR_GOLDEN
        .iter()
        .map(|&(label, main, aux, parts, digest)| Program {
            label: label.to_string(),
            main,
            aux,
            parts,
            digest,
        })
        .collect();
    assert!(!golden.is_empty(), "the golden list is pinned");
    assert_eq!(programs(&options(4, LogUpPolicy::Pair)), golden);
    let legacy = GoldilocksCubicProofOptions::with_blowup(4).expect("blowup 4");
    assert_eq!(programs(&legacy), golden);
    // At blowup 2 no wider arity fits: every policy is the pair layout, whose
    // programs are the blowup-4 ones (artifacts do not depend on the blowup).
    for policy in [LogUpPolicy::K3, LogUpPolicy::K4, LogUpPolicy::Best] {
        assert_eq!(
            programs(&options(2, policy)),
            golden,
            "{policy} at blowup 2"
        );
    }
}

/// ★ T0f on the production tables: under `k4` at blowup 4 each table commits
/// `⌈N/k⌉` aux columns and `max(3, k + 1) − 1` parts for the arity the rule
/// picks (D-LOGUP §2.2), the LFM chips keep pairs under the production
/// wrap options, and `best` adds the three-part tables.
#[test]
fn the_wide_policies_give_each_table_its_layout() {
    let shape = |policy| -> Vec<(String, usize, usize)> {
        production_airs(&options(4, policy))
            .iter()
            .map(|(label, air)| {
                let air: DynAir<'_> = &**air;
                (
                    label.to_string(),
                    air.trace_layout().1,
                    air.composition_poly_degree_bound(1 << 10) >> 10,
                )
            })
            .collect()
    };
    let pair = shape(LogUpPolicy::Pair);
    let k4 = shape(LogUpPolicy::K4);
    let best = shape(LogUpPolicy::Best);
    let row = |rows: &[(String, usize, usize)], label: &str| {
        rows.iter()
            .find(|r| r.0 == label)
            .map(|r| (r.1, r.2))
            .unwrap_or_else(|| panic!("no {label}"))
    };
    // (label, pair, k4, best) as (aux, parts).
    for (label, p, k, b) in [
        ("KECCAK_RND", (516, 2), (258, 4), (258, 4)),
        ("ECSM", (290, 2), (145, 4), (145, 4)),
        ("ECDAS", (194, 2), (97, 4), (97, 4)),
        ("KECCAK", (67, 2), (34, 4), (34, 4)),
        ("CPU", (10, 2), (5, 4), (5, 4)),
        ("MEMW_A", (10, 2), (5, 4), (5, 4)),
        ("MEMW", (13, 2), (7, 4), (7, 4)),
        ("SHIFT", (9, 2), (5, 4), (6, 3)),
        ("LT", (5, 2), (5, 2), (3, 3)),
        ("MEMW_R", (4, 2), (4, 2), (4, 2)),
        ("STORE", (5, 2), (5, 2), (5, 2)),
        ("LOAD", (3, 2), (3, 2), (3, 2)),
    ] {
        assert_eq!(row(&pair, label), p, "{label} pair");
        assert_eq!(row(&k4, label), k, "{label} k4");
        assert_eq!(row(&best, label), b, "{label} best");
    }
    // Every table: never more extension columns than pairs, and a table the
    // rule keeps on pairs keeps its pair program.
    for ((label, pa, pp), (_, ka, kp)) in pair.iter().zip(&k4) {
        assert!(ka + kp <= pa + pp, "{label}: k4 commits more than pairs");
        assert!(*kp <= 4, "{label}: {kp} parts at blowup 4");
    }
    let pair_programs = programs(&options(4, LogUpPolicy::Pair));
    for (p, k) in pair_programs
        .iter()
        .zip(programs(&options(4, LogUpPolicy::K4)))
    {
        assert_eq!(p.label, k.label);
        if (p.aux, p.parts) == (k.aux, k.parts) {
            assert_eq!(
                p.digest, k.digest,
                "{}: kept pairs, kept its program",
                p.label
            );
        } else {
            assert_ne!(p.digest, k.digest, "{}", p.label);
        }
    }
    // The LFM chips' production options carry pairs whatever the policy.
    let wide = crate::zf_format::ZfFormat {
        logup: LogUpPolicy::K4,
        ..crate::zf_format::ZfFormat::DEFAULT
    };
    let mut wrap = GoldilocksCubicProofOptions::with_blowup(4).expect("blowup 4");
    wrap.fri_final_poly_log_degree = 8;
    let wrap = wide.options(wrap);
    assert_eq!(wrap.format.logup, LogUpPolicy::Pair);
}
