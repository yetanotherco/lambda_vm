//! The block's in-guest verifier over the 7-round BLAKE3 base
//! ([`WhirBlockPlan::derive_blake3`]): the leaves replay `Blake3Whir`'s
//! transcript, leaves, joint leaves and binary paths, carry every root as two
//! words, and take the lean tables' compressions. Executed, not proved (the
//! programs' prover is the block prover over the lean tables, not this
//! module's).
//!
//! Box tier: every test proves a block's base first.
//!
//! - Small blocks: every leaf executes over an honest BLAKE3 base proof under
//!   both compression lowerings, the leaves publish one id, state and output,
//!   and their bus shares cancel; tampers of the base are refused.
//! - A real block (the census): the base proved at the production posture,
//!   then a one-group leaf a group executed, with each group's compressions,
//!   rows by chip and cells.

use crate::block_whir::{
    BlockFormat, BlockWhirProof, Deviations, prove_block_whir_with, verify_block_whir,
};
use crate::hash_pin::{BlockFamily, ProofFamily};
use crate::tables::types::{FE, FEE};
use crate::test_utils::asm_elf_bytes;
use crate::tests::block_whir_p1_tests::{many_groups, options};
use multilinear::whir_chain::{GroupFold, Round0Trees, RoundOpenings};
use stark::proof::options::{BaseFormat, ProofOptions};

use super::compiler::LfmProgram;
use super::executor::execute;
use super::instr::Instr;
use super::whir_block::{WhirBlockPlan, block_leaf_arena, id_words, out_halves};
use super::whir_chain::ChipRows;
use super::word::{LfmWord, base_word, word_as_ext};

/// A small block proved under the BLAKE3 base (the base-only permit held for
/// the prove alone) and accepted by the host verifier.
fn b3_block(name: &str, format: &BlockFormat, drop_levels: usize) -> (Vec<u8>, BlockWhirProof) {
    let elf = asm_elf_bytes(name);
    let opts = ProofOptions::default_test_options();
    let proof = crate::base_only::with_permit(|| {
        let proof = prove_block_whir_with(
            &elf,
            &[],
            &opts,
            format,
            &options(drop_levels),
            &Deviations::default(),
        )
        .expect("the BLAKE3 base proves")
        .0;
        assert!(
            verify_block_whir(&proof, &elf, &opts, format).expect("verify runs"),
            "the host verifier accepts the BLAKE3 base"
        );
        proof
    });
    (elf, proof)
}

/// Whether the host verifier accepts `proof` as a BLAKE3 base. The base-only
/// permit is held for the verify alone: without it the BLAKE3 base is refused
/// before it is read, which a tamper test would mistake for a refusal of the
/// tamper.
fn host_accepts(proof: &BlockWhirProof, elf: &[u8], format: &BlockFormat) -> bool {
    let opts = ProofOptions::default_test_options();
    crate::base_only::with_permit(|| {
        verify_block_whir(proof, elf, &opts, format)
            .expect("the BLAKE3 base is read under the permit")
    })
}

fn b3_plan(
    elf: &[u8],
    proof: &BlockWhirProof,
    format: &BlockFormat,
    leaves: usize,
) -> WhirBlockPlan {
    WhirBlockPlan::derive_blake3(
        elf,
        &ProofOptions::default_test_options(),
        format,
        proof.statement(),
        leaves,
        None,
    )
    .expect("the BLAKE3 plan derives")
}

/// Leaf `k` executed over `proof`'s arena: its published words, or why not.
fn run_leaf(
    plan: &WhirBlockPlan,
    proof: &BlockWhirProof,
    k: usize,
) -> Result<Vec<LfmWord>, String> {
    let program = plan.leaf_program(k)?;
    let arenas = block_leaf_arena(plan, proof, k)?;
    assert_eq!(
        program.arena_schema.lens,
        vec![arenas[0].len() as u32],
        "the leaf hints exactly the words its arena holds"
    );
    execute(
        &program,
        &arenas,
        &program.hasher(<BlockFamily as ProofFamily>::SOCKET),
    )
    .map(|e| e.public_words.iter().map(|(_, w)| *w).collect())
    .map_err(|e| format!("{e:?}"))
}

/// The BLAKE3 rows a program emits, by chip: `(lean, LFM_BLAKE3, any other
/// hash row)`.
fn hash_rows(program: &LfmProgram) -> (usize, usize, usize) {
    let mut out = (0, 0, 0);
    for instr in &program.instrs {
        match instr {
            Instr::B3l(_) => out.0 += 1,
            Instr::Blake3(_) => out.1 += 1,
            Instr::Hash { .. } | Instr::Hash16(_) | Instr::KeccakF(_) => out.2 += 1,
            _ => {}
        }
    }
    out
}

/// ★ THE LEAF GATE. Under the BLAKE3 base, one tree a polynomial and one a run
/// (joint leaves), at one, two and three leaves: every leaf executes over the
/// honest proof under both lowerings (the lean tables' rows, then
/// `LFM_BLAKE3`'s), hashing with that lowering's rows alone; every leaf
/// publishes the plan's id, one state (two words) and the output; and the
/// leaves' bus shares cancel — the block's bus, checked once over every table,
/// closes across them, which only a leaf that verified every group against
/// the host's challenges can give.
#[test]
#[ignore = "proves small BLAKE3 blocks; box tier"]
fn blake3_leaves_execute_and_close_the_bus() {
    for (group, trees) in [
        (GroupFold::PerPoly, Round0Trees::Each),
        (GroupFold::Indexed, Round0Trees::One),
    ] {
        let mut format = many_groups(BaseFormat::B3_WHIR);
        format.group = group;
        format.zf.whir_trees = trees;
        assert_eq!(format.zf.check_trees(), Ok(()));
        for (name, leaves) in [
            ("all_instructions_64", 3),
            ("test_commit_4", 2),
            ("test_keccak", 1),
        ] {
            let (elf, proof) = b3_block(name, &format, 3);
            for lean in [true, false] {
                let plan = b3_plan(&elf, &proof, &format, leaves).with_b3_lean(lean);
                let layout = plan.child_layout();
                assert_eq!(plan.digest_words(), 2, "a BLAKE3 root is two words");
                let label = format!("{name} {group:?}/{trees:?} lean {lean}");
                for g in 0..plan.num_groups() {
                    let (polys, n_stack) = plan.group_stack(g);
                    println!(
                        "B3 GROUP {label} {g}: polys {polys} · stack 2^{n_stack} · cost model {} \
                         compressions · emitted {}",
                        plan.costs()[g],
                        plan.chip_loads().0[g].accel
                    );
                }
                let mut words = Vec::new();
                for k in 0..plan.partition().num_leaves() {
                    let program = plan.leaf_program(k).expect("the leaf emits");
                    let (b3l, blake3, other) = hash_rows(&program);
                    assert_eq!(other, 0, "{label} leaf {k}: no felt-sponge row");
                    if lean {
                        assert!(b3l > 0 && blake3 == 0, "{label} leaf {k}: lean rows only");
                    } else {
                        assert!(blake3 > 0 && b3l == 0, "{label} leaf {k}: LFM_BLAKE3 only");
                    }
                    println!(
                        "B3 LEAF {label} {k}: groups {:?} · compressions {} · instructions {} · rows {:?}",
                        plan.partition().leaf(k),
                        b3l + blake3,
                        program.instrs.len(),
                        ChipRows::of(&program)
                    );
                    words.push(
                        run_leaf(&plan, &proof, k)
                            .unwrap_or_else(|e| panic!("{label} leaf {k}: refused: {e}")),
                    );
                }
                let id = id_words(plan.id());
                let out: Vec<LfmWord> = out_halves(&proof.public_output)
                    .into_iter()
                    .map(base_word)
                    .collect();
                let mut sum = FEE::zero();
                for w in &words {
                    assert_eq!(w.len(), layout.total(), "{label}");
                    assert_eq!(w[layout.id(0)], id[0]);
                    assert_eq!(w[layout.id(1)], id[1]);
                    for s in 0..plan.digest_words() {
                        assert_eq!(
                            w[layout.state(s)],
                            words[0][layout.state(s)],
                            "{label}: one state across the leaves"
                        );
                    }
                    for (i, half) in out.iter().enumerate() {
                        assert_eq!(w[layout.out_half(i)], *half);
                    }
                    sum += word_as_ext(&w[layout.sum()]).expect("an extension share");
                }
                assert_eq!(sum, FEE::zero(), "{label}: the leaves' shares cancel");
            }
        }
    }
}

/// ★ THE BASE TAMPERS. A BLAKE3 base proof changed after the prove has no
/// execution in the leaf that checks what changed: a committed root's last
/// byte (its second word), a commit-time out-of-domain answer, a round-0 leaf
/// value, a round's out-of-domain value and a query nonce, each in group 0;
/// under one tree a run, a joint leaf value in its second block. The host
/// verifier refuses each too.
#[test]
#[ignore = "proves small BLAKE3 blocks; box tier"]
fn a_tampered_blake3_base_has_no_leaf_execution() {
    for (group, trees) in [
        (GroupFold::PerPoly, Round0Trees::Each),
        (GroupFold::Indexed, Round0Trees::One),
    ] {
        let mut format = many_groups(BaseFormat::B3_WHIR);
        format.group = group;
        format.zf.whir_trees = trees;
        let (elf, honest) = b3_block("all_instructions_64", &format, 3);
        let plan = b3_plan(&elf, &honest, &format, 2);
        let holder = (0..plan.partition().num_leaves())
            .find(|&k| plan.partition().leaf(k).contains(&0))
            .expect("a leaf holds group 0");
        run_leaf(&plan, &honest, holder).expect("the honest leaf executes");
        let mut sites: Vec<(String, BlockWhirProof)> = Vec::new();
        let mut p = honest.clone();
        p.proof.roots[0][31] ^= 0x40;
        sites.push(("a root's last byte".into(), p));
        let mut p = honest.clone();
        p.commit_ood[0] += FEE::one();
        sites.push(("a commit-time answer".into(), p));
        let mut p = honest.clone();
        match &mut p.proof.columns[0].polys[0].rounds[0].openings {
            RoundOpenings::Base(r) | RoundOpenings::BaseJoint(r) => {
                r.current[0].values[1] += FE::one()
            }
            other => panic!("round 0 opens base blocks: {other:?}"),
        }
        sites.push(("a round-0 value".into(), p));
        let mut p = honest.clone();
        *p.proof.columns[0].polys[0].rounds[0]
            .ood_value
            .as_mut()
            .expect("round 0 has a successor") += FEE::one();
        sites.push(("a round's out-of-domain value".into(), p));
        let mut p = honest.clone();
        p.proof.columns[0].polys[0].rounds[0].nonces.query ^= 1;
        sites.push(("a query nonce".into(), p));
        if trees == Round0Trees::One {
            let joint = honest
                .proof
                .columns
                .iter()
                .enumerate()
                .find_map(|(g, column)| {
                    column.polys.iter().enumerate().find_map(|(c, chain)| {
                        matches!(chain.rounds[0].openings, RoundOpenings::BaseJoint(_))
                            .then_some((g, c))
                    })
                });
            let (g, c) = joint.expect("a grouped run opens a joint leaf");
            let mut p = honest.clone();
            match &mut p.proof.columns[g].polys[c].rounds[0].openings {
                RoundOpenings::BaseJoint(r) => {
                    let width = r.current[0].values.len();
                    r.current[0].values[width / 2 + 3] += FE::one();
                }
                _ => unreachable!("found joint above"),
            }
            let k = (0..plan.partition().num_leaves())
                .find(|&k| plan.partition().leaf(k).contains(&g))
                .expect("a leaf holds the group");
            assert!(
                !host_accepts(&p, &elf, &format),
                "{trees:?}: the host refuses a joint value"
            );
            assert!(
                run_leaf(&plan, &p, k).is_err(),
                "{trees:?}: a joint leaf value must have no execution"
            );
        }
        for (what, forged) in &sites {
            assert!(
                !host_accepts(forged, &elf, &format),
                "{trees:?} {what}: the host refuses"
            );
            let refused = run_leaf(&plan, forged, holder);
            println!(
                "B3 LEAF TAMPER {trees:?} {what}: {:?}",
                refused.as_ref().err()
            );
            assert!(refused.is_err(), "{trees:?} {what}: must have no execution");
        }
    }
}

/// ★ A BLAKE3 plan is over its own base alone: it is refused over a Poseidon1
/// format, and over its own base its leaf executes.
#[test]
#[ignore = "proves small blocks; box tier"]
fn a_blake3_plan_is_its_base_alone() {
    let p1 = many_groups(BaseFormat::P1_WHIR);
    let elf = asm_elf_bytes("test_commit_4");
    let opts = ProofOptions::default_test_options();
    let p1_proof =
        prove_block_whir_with(&elf, &[], &opts, &p1, &options(0), &Deviations::default())
            .expect("the Poseidon1 base proves")
            .0;
    assert!(
        WhirBlockPlan::derive_blake3(&elf, &opts, &p1, p1_proof.statement(), 1, None).is_err(),
        "a BLAKE3 plan over a Poseidon1 format"
    );
    let b3 = many_groups(BaseFormat::B3_WHIR);
    let (_, b3_proof) = b3_block("test_commit_4", &b3, 0);
    let plan = b3_plan(&elf, &b3_proof, &b3, 1);
    run_leaf(&plan, &b3_proof, 0).expect("its own base executes");
}

/// ★★ The census on a real block: the BLAKE3 base proved at the production
/// posture (`prove-block-base --base b3`'s format: blowup 2, coset leaves, one
/// tree a run, indexed groups), verified on the host, then the in-guest
/// verifier's leaves at ONE GROUP A LEAF executed over it under the lean
/// lowering — every one must execute, and their shares cancel — with each
/// group's compressions, rows by chip and cells printed. A tampered copy (a
/// root's byte) is refused by its group's leaf.
///
/// Reads `BLOCK_WHIR_ELF`, `BLOCK_WHIR_INPUT`; with `B3_CENSUS_WHOLE=1` also
/// executes the whole block as ONE leaf (the whole block's verifier) and prints its
/// size. Prints `B3 CENSUS …` lines; the reader turns them into rows.
#[test]
#[ignore = "a real block's BLAKE3 base and its in-guest verifier: box only"]
fn blake3_level0_census_on_a_real_block() {
    use super::whir_block_tree::WhirTreeConfig;
    use crate::zf_format::{LeafOrder, ZfFormat};
    let _permit = super::device_permit::serial_for_test();
    let elf = std::fs::read(std::env::var("BLOCK_WHIR_ELF").expect("BLOCK_WHIR_ELF"))
        .expect("read the ELF");
    let input = std::fs::read(std::env::var("BLOCK_WHIR_INPUT").expect("BLOCK_WHIR_INPUT"))
        .expect("read the input");
    // `prove-block-base --base b3`'s format, installed before anything reads it.
    let zf = ZfFormat {
        whir_blowup: 2,
        whir_leaves: LeafOrder::Coset,
        whir_trees: Round0Trees::One,
        base: BaseFormat::B3_WHIR,
        ..ZfFormat::from_env().expect("the environment's format")
    };
    zf.check_leaves().expect("leaves");
    zf.check_trees().expect("trees");
    ZfFormat::install(zf).expect("install the format");
    let mut cfg = WhirTreeConfig::from_env().expect("the tree config");
    cfg.format.group = GroupFold::Indexed;
    assert_eq!(cfg.format.zf.base, BaseFormat::B3_WHIR);
    let opts = super::proof::block_base_options();
    let t = std::time::Instant::now();
    let proof = crate::base_only::with_permit(|| {
        crate::block_whir::prove_block_whir_observed_led(
            &elf,
            &input,
            &opts,
            &cfg.format,
            &cfg.options,
            &|_, _| {},
            &|_| {},
            None,
        )
        .expect("the BLAKE3 base proves")
        .0
    });
    let prove_secs = t.elapsed().as_secs_f64();
    let (host, one_copy, per_point) = multilinear::whir_chain::final_message_reads();
    println!("B3 CENSUS FINAL READS: one-copy {one_copy} · host {host} · per-point {per_point}");
    let t = std::time::Instant::now();
    let verified = crate::base_only::with_permit(|| {
        verify_block_whir(&proof, &elf, &opts, &cfg.format).expect("verify runs")
    });
    println!(
        "B3 CENSUS BASE: groups {} · prove {prove_secs:.2} s · host verify {:.2} s · verified {verified} · proof bytes {}",
        proof.proof.columns.len(),
        t.elapsed().as_secs_f64(),
        rkyv::to_bytes::<rkyv::rancor::Error>(&proof).map_or(0, |b| b.len())
    );
    assert!(verified, "the host verifier accepts the BLAKE3 base");

    let groups = proof.proof.columns.len();
    let t = std::time::Instant::now();
    let plan =
        WhirBlockPlan::derive_blake3(&elf, &opts, &cfg.format, proof.statement(), groups, None)
            .expect("the BLAKE3 plan derives");
    println!(
        "B3 CENSUS PLAN: {} groups · {} leaves · derived {:.2} s · group polys {} · tag {}",
        plan.num_groups(),
        plan.partition().num_leaves(),
        t.elapsed().as_secs_f64(),
        cfg.format.group_polys,
        String::from_utf8_lossy(&crate::block_whir::block_format_tag(&cfg.format))
    );
    assert!(
        (0..plan.partition().num_leaves()).all(|k| plan.partition().leaf(k).len() == 1),
        "one group a leaf"
    );
    let layout = plan.child_layout();
    let lean_cells_per_compression = std::env::var("B3_CENSUS_LEAN_CELLS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(2_832);
    // Each group's own rows, the front's apart (the plan's emission marks):
    // a verified group's compressions are its walk's and opening's, not the
    // front every leaf repeats.
    let (loads, front) = plan.chip_loads();
    println!(
        "B3 CENSUS FRONT: compressions {} · rows {:?}",
        front.accel, front
    );
    let mut sum = FEE::zero();
    let mut per_group: Vec<(usize, usize, u64, u64)> = Vec::new();
    let mut executed = 0usize;
    for k in 0..plan.partition().num_leaves() {
        let g = plan.partition().leaf(k)[0];
        let program = plan.leaf_program(k).expect("the leaf emits");
        let (leaf_b3l, blake3, other) = hash_rows(&program);
        let b3l = loads[g].accel;
        assert_eq!((blake3, other), (0, 0), "group {g}: lean rows only");
        let chips = super::airs::lfm_chip_census_with_hasher(
            &program,
            program.hasher(<BlockFamily as ProofFamily>::SOCKET),
        );
        // The leaf's one-output selects are LEAN_MUX rows (a lean table, outside
        // the chip census): their cells belong to the field side too.
        let muxes = super::whir_b3_tests::lean_mux_cells(&program);
        let field_main: u64 = chips.iter().map(|c| c.main_cells()).sum::<u64>() + muxes.padded_main;
        let field_padded: u64 = chips
            .iter()
            .map(|c| c.main_cells() + c.aux_cells())
            .sum::<u64>()
            + muxes.padded_main
            + muxes.padded_aux;
        // The leaf's (front and group): its chips at their padded heights,
        // and its compressions at the lean tables' cells each.
        let hash_cells = leaf_b3l as u64 * lean_cells_per_compression;
        let t = std::time::Instant::now();
        let words = run_leaf(&plan, &proof, k);
        let exec_secs = t.elapsed().as_secs_f64();
        println!(
            "B3 CENSUS GROUP {g}: compressions {b3l} · leaf compressions {leaf_b3l} · polys {} · stack 2^{} · cost model {} (argue {}, opening {}, prepared {}) · instructions {} · group rows {:?} · leaf field main cells {field_main} (with aux {field_padded}) · leaf lean hash cells {hash_cells} · leaf hash share {:.4} · execute {exec_secs:.2} s · executed {}",
            plan.group_stack(g).0,
            plan.group_stack(g).1,
            plan.costs()[g],
            plan.cost_parts()[g][0],
            plan.cost_parts()[g][1],
            plan.cost_parts()[g][2],
            program.instrs.len(),
            loads[g],
            hash_cells as f64 / (hash_cells + field_main) as f64,
            words.is_ok()
        );
        match words {
            Ok(w) => {
                executed += 1;
                sum += word_as_ext(&w[layout.sum()]).expect("a share");
            }
            Err(e) => println!("B3 CENSUS REFUSED group {g}: {e}"),
        }
        per_group.push((g, b3l, field_main, hash_cells));
    }
    let mut comps: Vec<usize> = per_group.iter().map(|p| p.1).collect();
    comps.sort_unstable();
    let median = comps[comps.len() / 2];
    let max = *comps.last().expect("a group");
    let total: usize = comps.iter().sum();
    let field: u64 = per_group.iter().map(|p| p.2).sum();
    let hash: u64 = per_group.iter().map(|p| p.3).sum();
    println!(
        "B3 CENSUS SUMMARY: groups {groups} · executed {executed} · bus closes {} · compressions a group median {median} max {max} total {total} · one-group leaves' lean cells Σ {} (hash {hash}, field {field}) · leaf hash share {:.4}",
        sum == FEE::zero(),
        hash + field,
        hash as f64 / (hash + field) as f64
    );

    // The tamper: a root's byte off; group 0's leaf has no execution.
    let mut forged = proof.clone();
    forged.proof.roots[0][31] ^= 0x01;
    let holder = (0..plan.partition().num_leaves())
        .find(|&k| plan.partition().leaf(k).contains(&0))
        .expect("a leaf holds group 0");
    let refused = run_leaf(&plan, &forged, holder).is_err();
    println!("B3 CENSUS TAMPER root byte: refused {refused}");
    // Under the base's final polynomial: one of group 0's sent values off;
    // the host refuses it and group 0's leaf has no execution.
    let final_at = cfg
        .format
        .chain_config(&[(1, plan.group_stack(0).1)])
        .final_round(plan.group_stack(0).1);
    let final_refused = match final_at {
        Some((r, _)) => {
            let mut forged = proof.clone();
            assert!(
                forged.proof.columns[0].polys[0].rounds[r]
                    .next_root
                    .is_none(),
                "round {r} is group 0's final round: it commits no successor"
            );
            match &mut forged.proof.columns[0].polys[0].rounds[r].openings {
                RoundOpenings::Extension(p) => p.next[0].values[0] += FEE::one(),
                other => panic!("group 0's final round {r} sends extension values: {other:?}"),
            }
            let host = crate::base_only::with_permit(|| {
                verify_block_whir(&forged, &elf, &opts, &cfg.format).unwrap_or(false)
            });
            let refused = !host && run_leaf(&plan, &forged, holder).is_err();
            println!(
                "B3 CENSUS TAMPER final polynomial value (group 0, round {r}): host refuses {} · \
                 refused {refused}",
                !host
            );
            Some(refused)
        }
        None => {
            println!("B3 CENSUS TAMPER final polynomial value: none (no final round)");
            None
        }
    };

    if std::env::var("B3_CENSUS_WHOLE").is_ok_and(|v| v.trim() == "1") {
        let whole =
            WhirBlockPlan::derive_blake3(&elf, &opts, &cfg.format, proof.statement(), 1, None)
                .expect("the one-leaf plan");
        let program = whole.leaf_program(0).expect("the whole block's leaf");
        let (b3l, _, _) = hash_rows(&program);
        let t = std::time::Instant::now();
        let ok = run_leaf(&whole, &proof, 0).is_ok();
        println!(
            "B3 CENSUS WHOLE: compressions {b3l} · instructions {} · rows {:?} · execute {:.2} s · executed {ok}",
            program.instrs.len(),
            ChipRows::of(&program),
            t.elapsed().as_secs_f64()
        );
    }
    assert_eq!(executed, groups, "every group's leaf executes");
    assert_eq!(sum, FEE::zero(), "the leaves' shares cancel");
    assert!(refused, "a tampered root is refused");
    assert_ne!(
        final_refused,
        Some(false),
        "a tampered sent value is refused"
    );
}

/// Whether leaf `k` has no execution over `proof`: its arena is refused as
/// laid out, or the executor refuses it (a length the program does not
/// declare included).
fn leaf_refuses(plan: &WhirBlockPlan, proof: &BlockWhirProof, k: usize) -> bool {
    let program = plan.leaf_program(k).expect("the leaf emits");
    match block_leaf_arena(plan, proof, k) {
        Err(_) => true,
        Ok(arenas) => execute(
            &program,
            &arenas,
            &program.hasher(<BlockFamily as ProofFamily>::SOCKET),
        )
        .is_err(),
    }
}

/// ★ THE ROUND-FORM LEAF GATE. A small block's BLAKE3 base under the WHIR
/// round form (shrink 3, caps by Q: the base's own format word for it), one
/// joint round-0 tree a run and grouped runs:
/// - the leaf's statement is the host's: its id is the digest of the host's
///   statement bytes, which carry the format's own tag (`block_format_tag`,
///   the round form's word in it) and the chains' fold word with the round
///   form's bit and the caps-by-Q bit;
/// - every leaf executes over the honest proof under both lowerings, publishes
///   the plan's id, and the leaves' shares cancel;
/// - each form's leaves refuse the other form's proof of the same block
///   (in-guest, by execution), and their ids differ;
/// - a later round's opened value (the fold value its shift claim carries) is
///   refused by the leaf holding its group, and by the host.
#[test]
#[ignore = "proves small BLAKE3 blocks; box tier"]
fn blake3_round_form_leaves_execute_and_refuse_the_other_form() {
    use crate::block_whir::{B3Rounds, block_format_tag, block_frame, block_statement_bytes};
    use multilinear::whir_chain::{FOLD_WORD_WHIR, FOLD_WORD_WHIR_CAPS_BY_Q, RoundForm, WhirFolds};
    let mut fri = many_groups(BaseFormat::B3_WHIR);
    fri.group = GroupFold::Indexed;
    fri.zf.whir_trees = Round0Trees::One;
    fri.zf.whir_folds = WhirFolds::Uniform;
    let w = BlockFormat {
        b3_rounds: B3Rounds::WHIR3,
        ..fri
    };
    let opts = ProofOptions::default_test_options();
    for (name, leaves) in [("all_instructions_64", 2), ("test_commit_4", 1)] {
        let (elf, honest) = b3_block(name, &w, 3);
        let (_, today) = b3_block(name, &fri, 3);

        // The statement: the host's bytes, the format's tag and fold word.
        let config = block_frame(honest.statement(), &elf, &opts, &w)
            .expect("the frame")
            .config;
        match config.format.rounds {
            RoundForm::Whir(r) => assert!(r.caps_by_queries(), "{name}: caps by Q"),
            RoundForm::Fri => panic!("{name}: the round form's config"),
        }
        let word = config.fold_word();
        assert_ne!(word & FOLD_WORD_WHIR, 0, "{name}: the round form's bit");
        assert_ne!(
            word & FOLD_WORD_WHIR_CAPS_BY_Q,
            0,
            "{name}: the caps-by-Q bit"
        );
        let tag = block_format_tag(&w);
        assert_ne!(tag, block_format_tag(&fri), "{name}: each form its tag");
        let suffix = w.b3_rounds.tag_word();
        assert!(
            !suffix.is_empty() && tag.ends_with(suffix.as_bytes()),
            "{name}: the tag ends with the round form's word {suffix}"
        );
        let bytes = block_statement_bytes(
            honest.statement(),
            &tag,
            &crate::statement::elf_digest(&elf),
            &config,
            w.memory,
        );
        assert!(
            bytes.starts_with(&tag),
            "{name}: the statement opens on its tag"
        );
        assert!(
            bytes.windows(8).any(|b| b == word.to_le_bytes()),
            "{name}: the statement carries the fold word"
        );
        let plan = b3_plan(&elf, &honest, &w, leaves);
        assert_eq!(
            plan.id(),
            &crate::statement::elf_digest(&bytes),
            "{name}: the leaf's statement is the host's"
        );

        // Every leaf executes, under both lowerings.
        for lean in [true, false] {
            let plan = b3_plan(&elf, &honest, &w, leaves).with_b3_lean(lean);
            let layout = plan.child_layout();
            let id = id_words(plan.id());
            let mut sum = FEE::zero();
            for k in 0..plan.partition().num_leaves() {
                let words = run_leaf(&plan, &honest, k)
                    .unwrap_or_else(|e| panic!("{name} lean {lean} leaf {k}: refused: {e}"));
                assert_eq!(words[layout.id(0)], id[0]);
                assert_eq!(words[layout.id(1)], id[1]);
                sum += word_as_ext(&words[layout.sum()]).expect("an extension share");
            }
            assert_eq!(sum, FEE::zero(), "{name} lean {lean}: the shares cancel");
        }

        // Each form's leaves refuse the other's proof.
        let fri_plan = b3_plan(&elf, &today, &fri, leaves);
        assert_ne!(fri_plan.id(), plan.id(), "{name}: the forms' ids differ");
        for k in 0..fri_plan.partition().num_leaves() {
            run_leaf(&fri_plan, &today, k)
                .unwrap_or_else(|e| panic!("{name}: today's leaf {k} refused today's: {e}"));
            assert!(
                leaf_refuses(&fri_plan, &honest, k),
                "{name}: today's leaf {k} on a round-form proof"
            );
        }
        for k in 0..plan.partition().num_leaves() {
            assert!(
                leaf_refuses(&plan, &today, k),
                "{name}: a round-form leaf {k} on today's proof"
            );
        }

        // A later round's opened value: its claim's fold value moves.
        let site = honest
            .proof
            .columns
            .iter()
            .enumerate()
            .find_map(|(g, column)| {
                column.polys.iter().enumerate().find_map(|(c, chain)| {
                    chain.rounds.iter().enumerate().skip(1).find_map(|(r, round)| {
                        matches!(&round.openings, RoundOpenings::Extension(p) if !p.current.is_empty())
                            .then_some((g, c, r))
                    })
                })
            });
        let (g, c, r) = site.expect("a later round opens its tree");
        let mut forged = honest.clone();
        match &mut forged.proof.columns[g].polys[c].rounds[r].openings {
            RoundOpenings::Extension(p) => p.current[0].values[0] += FEE::one(),
            _ => unreachable!("found above"),
        }
        assert!(!host_accepts(&forged, &elf, &w), "{name}: the host refuses");
        let k = (0..plan.partition().num_leaves())
            .find(|&k| plan.partition().leaf(k).contains(&g))
            .expect("a leaf holds the group");
        assert!(
            leaf_refuses(&plan, &forged, k),
            "{name}: a later round's opened value must have no execution"
        );
        println!(
            "B3 ROUND FORM LEAF {name}: {} leaves · groups {} · tag {:?} · fold word {word:#x} · \
             tamper at group {g} chain {c} round {r} refused",
            plan.partition().num_leaves(),
            plan.num_groups(),
            String::from_utf8_lossy(&tag)
        );
    }
}
