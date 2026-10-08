//! `LAMBDA_VM_ZF_LOGUP` on the production tables: the pair policy keeps every
//! production constraint program byte for byte (a golden list taken at
//! #1013's head before arities existed), the wider policies give each table
//! the layout the rule picks, and (box) a real `k4` block proof verifies while
//! each of D-LOGUP's mutations N1–N5 is refused.

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

/// [`PAIR_GOLDEN`] as programs.
fn pair_golden() -> Vec<Program> {
    PAIR_GOLDEN
        .iter()
        .map(|&(label, main, aux, parts, digest)| Program {
            label: label.to_string(),
            main,
            aux,
            parts,
            digest,
        })
        .collect()
}

/// ★ T0a: under the pair policy, every production VM table and every LFM chip
/// has the widths, part count and constraint program it had before arities
/// existed, byte for byte — the rollback (`LAMBDA_VM_ZF_LOGUP=pair`) moves no
/// compiled-kernel key and no proof. The same holds for a wide policy at a
/// blowup that cannot hold wider groups.
#[test]
fn the_pair_policy_keeps_every_production_program() {
    let golden = pair_golden();
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

/// ★ k4 is the production default: the block's base options with no knob set
/// ([`crate::zf_format::ZfFormat::DEFAULT`]) build exactly the k4 programs, and
/// the rollback spelling `LAMBDA_VM_ZF_LOGUP=pair` builds the golden pair list,
/// byte for byte.
#[test]
fn the_default_base_format_is_k4_and_pair_rolls_back() {
    use crate::zf_format::{ENV_LOGUP, ZfFormat};
    let base = GoldilocksCubicProofOptions::with_blowup(4).expect("blowup 4");
    let default = ZfFormat::DEFAULT.base_options(base.clone());
    assert_eq!(default.format.logup, LogUpPolicy::K4);
    assert_eq!(programs(&default), programs(&options(4, LogUpPolicy::K4)));
    assert_ne!(
        programs(&default),
        pair_golden(),
        "k4 moves the switched tables"
    );
    let rollback = ZfFormat::from_lookup(|name| (name == ENV_LOGUP).then(|| "pair".to_string()))
        .expect("the rollback spelling parses");
    assert_eq!(programs(&rollback.base_options(base)), pair_golden());
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

/// ★ D-LOGUP S2 negatives on a real VM block proof under `k4` (box): a
/// `fib_iterative_160k` block (CPU at 2^18 rows, device-only under the
/// production thresholds, so its four parts are split on the card) proved
/// under `LAMBDA_VM_ZF_LOGUP=k4` verifies, and each mutation is refused, never
/// a panic:
/// - N1 one cell of a committed k4 term column (CPU's, at the OOD point);
/// - N2 the multiplicity of an absorbed interaction (CPU's, in the trace;
///   the prover then commits a consistent aux trace and the bus refuses it);
/// - N3 the k4 proof under the pair options;
/// - N4 three or five part OODs for CPU's four-part AIR;
/// - N5 one composition-part OOD value.
#[test]
#[ignore = "proves a VM program three times at blowup 4; GPU box gate (cuda)"]
fn logup_k4_vm_proof_negatives() {
    use executor::elf::Elf;
    use executor::vm::execution::Executor;
    use stark::lookup::{LinearTerm, LogUpLayout, Multiplicity};
    use stark::residency_mode::ResidencyMode;

    let elf_bytes = crate::test_utils::asm_elf_bytes("fib_iterative_160k");
    let program = Elf::load(&elf_bytes).expect("load the ELF");
    let run = Executor::new(&program, vec![])
        .expect("executor")
        .run()
        .expect("run");
    let traces = crate::tables::trace_builder::Traces::from_elf_and_logs(
        &program,
        &run.logs,
        &crate::tables::MaxRowsConfig::default(),
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("build the traces");
    let base = GoldilocksCubicProofOptions::with_params(4, 128, 0).expect("options");
    let k4 = crate::zf_format::ZfFormat {
        logup: LogUpPolicy::K4,
        ..crate::zf_format::ZfFormat::DEFAULT
    }
    .base_options(base.clone());
    let pair = crate::zf_format::ZfFormat::DEFAULT.base_options(base);
    let prove = |traces: &crate::tables::trace_builder::Traces| {
        let decode = crate::tables::decode::commitment_from_elf_device_or_host(&program, &k4)
            .expect("DECODE commitment");
        crate::block::prove_block_traces(
            &elf_bytes,
            &program,
            &mut traces.clone(),
            &k4,
            Some(decode),
            ResidencyMode::RecomputeLdeDevice,
            Vec::new(),
            &mut crate::block::BlockTimes::default(),
            &mut |_| {},
        )
    };
    let accepts = |proof: &crate::VmProof, opts: &ProofOptions| {
        matches!(
            crate::block::verify_block(proof, &elf_bytes, opts),
            Ok(true)
        )
    };

    let proof = prove(&traces).expect("the k4 block proves");
    assert!(accepts(&proof, &k4), "the k4 block proof verifies");
    let airs = crate::VmAirs::new(
        &program,
        &k4,
        false,
        &traces.page_configs,
        &traces.table_counts(),
        None,
        true,
        None,
        None,
        None,
    );
    let refs = airs.air_refs();
    let idx = refs
        .iter()
        .position(|a| a.name() == "CPU[0]")
        .expect("a CPU[0] table");
    let (main, aux) = refs[idx].trace_layout();
    assert_eq!(aux, 5, "CPU commits 4 groups of four plus the accumulator");
    assert_eq!(
        proof.proof.proofs[idx]
            .composition_poly_parts_ood_evaluation
            .len(),
        4
    );
    println!("LOGUP NEG control: the k4 block verifies (CPU[0] = sub-proof {idx}, aux 5, 4 parts)");

    // N3
    assert!(
        !accepts(&proof, &pair),
        "N3: the pair options must refuse it"
    );
    println!("LOGUP NEG N3: refused under the pair options");
    // N1
    for term in [0, aux - 2] {
        let mut bad = proof.clone();
        let ood = &mut bad.proof.proofs[idx].trace_ood_evaluations;
        let v = *ood.get(0, main + term) + crate::tables::types::FEE::one();
        ood.set(0, main + term, v);
        assert!(!accepts(&bad, &k4), "N1: term column {term}");
    }
    println!("LOGUP NEG N1: a tampered k4 term cell (first and last term column) refused");
    // N4
    let mut fewer = proof.clone();
    fewer.proof.proofs[idx]
        .composition_poly_parts_ood_evaluation
        .pop();
    assert!(!accepts(&fewer, &k4), "N4: three part OODs");
    let mut more = proof.clone();
    more.proof.proofs[idx]
        .composition_poly_parts_ood_evaluation
        .push(crate::tables::types::FEE::zero());
    assert!(!accepts(&more, &k4), "N4: five part OODs");
    println!("LOGUP NEG N4: 3 and 5 part OODs for a 4-part AIR refused");
    // N5
    for part in 0..4 {
        let mut bad = proof.clone();
        bad.proof.proofs[idx].composition_poly_parts_ood_evaluation[part] +=
            crate::tables::types::FEE::one();
        assert!(!accepts(&bad, &k4), "N5: part {part}");
    }
    println!("LOGUP NEG N5: each tampered part OOD refused");
    // N2
    let layout = LogUpLayout::with_arity(refs[idx].bus_interactions().to_vec(), 4);
    assert_eq!(layout.num_term_columns + 1, aux);
    let (k, col) = layout
        .absorbed()
        .iter()
        .enumerate()
        .find_map(|(k, it)| {
            let col = match &it.multiplicity {
                Multiplicity::Column(c)
                | Multiplicity::Sum(c, _)
                | Multiplicity::Negated(c)
                | Multiplicity::Diff(c, _)
                | Multiplicity::Sum3(c, _, _) => Some(*c),
                Multiplicity::Linear(ts) => ts.iter().find_map(|t| match t {
                    LinearTerm::Column { column, .. }
                    | LinearTerm::ColumnUnsigned { column, .. } => Some(*column),
                    LinearTerm::Constant(_) => None,
                }),
                Multiplicity::One => None,
            };
            col.map(|c| (k, c))
        })
        .expect("an absorbed CPU interaction with a column multiplicity");
    let mut tampered = traces.clone();
    let cell = *tampered.cpus[0].main_table.get(1, col);
    tampered.cpus[0]
        .main_table
        .set(1, col, cell + crate::tables::types::FE::one());
    let refused = match prove(&tampered) {
        Ok(p) => !accepts(&p, &k4),
        Err(e) => {
            println!("    (the prover refused: {e:?})");
            true
        }
    };
    assert!(refused, "N2: a moved absorbed multiplicity must be refused");
    println!(
        "LOGUP NEG N2: absorbed interaction {k} of CPU (multiplicity column {col}) moved at row 1: refused"
    );
}
