//! D-WHIR sizing instrument (design lane, not for landing): what a node would
//! pay to verify ONE WHIR-proved LFM child, from the landed in-guest cost forms
//! (`whir_table::table_verify_cost`, `whir_stacked::stacked_verify_cost`,
//! `whir_epoch::roots_block_cost`) evaluated on the LFM chip set at the chip
//! heights the C4 WHIR tree measured (wt714 panels). Card-free, no ELF.
//!
//! cargo test --release -p lambda-vm-prover --lib dwhir_sizing -- --nocapture

use multilinear::whir_chain::StackVars;
use stark::multilinear_air::Uniforms;
use stark::multilinear_logup::interaction_shapes;
use stark::multilinear_table::{TableLayout, global_layouts};

use crate::tables::types::{GoldilocksExtension as E, GoldilocksField as F};

use super::airs::{ChipSet, LfmAirs, NUM_LFM_CHIPS};
use super::hash::HasherKind;
use super::whir_chain::{ChainShape, RoundStorage, chain_opening_perms};
use super::whir_epoch::roots_block_cost;
use super::whir_stacked::stacked_verify_cost;
use super::whir_table::{TableShape, table_verify_cost};
use super::whir_transcript::{CANDIDATES_PER_SQUEEZE, SpongeEntry};

/// log2 padded rows per chip, frozen order CONST BALU XALU SELECT BITDEC HASH
/// LANES HINT PUBLIC RANGE (wt714 panels, C4 B arm).
const PROFILES: &[(&str, [usize; 10])] = &[
    ("wrap 0", [8, 17, 19, 18, 12, 17, 18, 18, 8, 16]),
    ("wrap 4", [9, 17, 19, 18, 12, 17, 18, 18, 8, 16]),
    ("global wrap", [8, 19, 20, 19, 14, 19, 20, 20, 6, 16]),
    ("L1N0", [10, 19, 21, 19, 13, 19, 19, 20, 8, 16]),
    ("L2N0", [10, 19, 21, 20, 13, 19, 19, 20, 8, 16]),
    ("L2N1", [10, 19, 21, 19, 13, 19, 18, 20, 8, 16]),
    ("root", [10, 19, 21, 20, 13, 19, 19, 20, 8, 16]),
    // Hypothetical WHIR-verifying nodes, one and two sizes down.
    ("node -1", [10, 18, 20, 18, 12, 18, 18, 19, 8, 16]),
    ("node -2", [9, 17, 19, 17, 12, 17, 17, 18, 8, 16]),
];

fn ceil_log2(x: usize) -> usize {
    x.next_power_of_two().trailing_zeros() as usize
}

#[test]
fn dwhir_sizing() {
    let opts = super::proof::aggregation_wrap_options();
    let chip_set = ChipSet {
        keccak: false,
        blake3: false,
        bitwise: false,
    };
    let roots = [[0u8; 32]; NUM_LFM_CHIPS];
    let airs = LfmAirs::new_chunked(&roots, &[], &opts, 0, HasherKind::Rpx, chip_set);
    let refs = airs.air_refs();
    assert_eq!(refs.len(), 10, "the WHIR recursion chip set is ten tables");

    // ---- A. static per-chip shape
    println!("\n== D-WHIR A: LFM chips (RPX, no keccak/blake3/bitwise) ==");
    println!(
        "{:<11} {:>5} {:>5} {:>5} {:>4} {:>4} {:>6} {:>6} {:>4} {:>5} {:>6} {:>5}",
        "chip", "width", "prep", "value", "I", "+lg", "IRall", "IRlive", "deg", "roots", "facts", "pubs"
    );
    let mut ir_live = Vec::new();
    for air in &refs {
        let (width, _) = air.trace_layout();
        let prep = air.num_precomputed_columns();
        let program = air.constraint_program();
        let base_roots = program.roots[..program.num_base].to_vec();
        let live = stark::multilinear_air::live_nodes(program, &base_roots)
            .iter()
            .filter(|b| **b)
            .count();
        ir_live.push(live);
        let layout = TableLayout::<F, E>::new(
            program,
            air.constraints_meta(),
            air.bus_interactions(),
            width,
            4,
            Uniforms::default(),
        )
        .expect("lays out");
        let facts = layout.kinds().iter().filter(|k| k.source().is_some()).count();
        println!(
            "{:<11} {:>5} {:>5} {:>5} {:>4} {:>4} {:>6} {:>6} {:>4} {:>5} {:>6} {:>5}",
            air.name(),
            width,
            prep,
            width - prep,
            air.bus_interactions().len(),
            ceil_log2(air.bus_interactions().len()),
            program.len(),
            live,
            layout.shape().degree(),
            layout.shape().num_roots(),
            facts,
            layout.shape().public_selectors().len(),
        );
    }

    // ---- B. per profile
    for (label, logs) in PROFILES {
        println!("\n== D-WHIR B: {label} ==");
        let shapes_all: Vec<(usize, usize)> = refs
            .iter()
            .zip(logs.iter())
            .map(|(air, &n)| (air.trace_layout().0, n))
            .collect();
        let shapes_val: Vec<(usize, usize)> = refs
            .iter()
            .zip(logs.iter())
            .map(|(air, &n)| (air.trace_layout().0 - air.num_precomputed_columns(), n))
            .collect();
        let shapes_prep: Vec<(usize, usize)> = refs
            .iter()
            .zip(logs.iter())
            .map(|(air, &n)| (air.num_precomputed_columns(), n))
            .collect();
        let cells = |s: &[(usize, usize)]| s.iter().map(|&(w, n)| w << n).sum::<usize>();
        let config = crate::multilinear_prove::chain_config(&shapes_all);
        println!(
            "cells: all {} · value {} · prep {} | config: blowup 2^{} Q {} grind {}/{}/{} stack cap {}",
            cells(&shapes_all),
            cells(&shapes_val),
            cells(&shapes_prep),
            config.log_blowup,
            config.num_queries,
            config.grind.folding,
            config.grind.ood,
            config.grind.query,
            config.format.stack.get()
        );
        // ARGUE-model inputs.
        let mut node_rows = 0usize;
        let mut gkr_leaves = 0usize;
        let mut rounds = 0usize;
        for ((air, &n), live) in refs.iter().zip(logs.iter()).zip(&ir_live) {
            node_rows += live << n;
            let lg = ceil_log2(air.bus_interactions().len());
            gkr_leaves += 1usize << (n + lg);
            let m = n + lg;
            rounds += m * (m.saturating_sub(1)) / 2 + m + 2 * n;
        }
        println!(
            "argue inputs: IR live-node x rows {node_rows} · GKR input leaves {gkr_leaves} · sumcheck rounds ~{rounds}"
        );

        for cap in [27usize, 26, 25] {
            let cap_v = StackVars::new(cap).unwrap();
            let la = global_layouts(&shapes_all, &[10], cap_v).unwrap();
            let lv = global_layouts(&shapes_val, &[10], cap_v).unwrap();
            let lp = global_layouts(&shapes_prep, &[10], cap_v).unwrap();
            println!(
                "  cap {cap}: main(all) n {} polys {} · main(value) n {} polys {} · prepared n {} polys {}",
                la[0].n_stack(),
                la[0].num_polys(),
                lv[0].n_stack(),
                lv[0].num_polys(),
                lp[0].n_stack(),
                lp[0].num_polys()
            );
        }

        // The leg, policy A (prep committed in main AND prepared) and B (value-only main).
        let cap = StackVars::new(27).unwrap();
        let publics = super::per_table_aggregator::SchemaLayout::node(0).total();
        for (policy, main_shapes) in [("A all-in-main", &shapes_all), ("B value-only-main", &shapes_val)] {
            let main = global_layouts(main_shapes, &[10], cap).unwrap().remove(0);
            let prep = global_layouts(&shapes_prep, &[10], cap).unwrap().remove(0);
            // statement: ~18 constant felts + 4 per public word, then the roots.
            let stmt_felts = 18 + 4 * publics;
            let entry = SpongeEntry {
                buffered_felts: stmt_felts,
                out_pos: CANDIDATES_PER_SQUEEZE,
            };
            let (roots_ops, roots_sched) = roots_block_cost(main.num_polys(), prep.num_polys(), entry);
            let mut entry = roots_sched.entry();
            let mut ops = roots_ops + roots_sched.rows();
            let mut perms = roots_sched.perms();
            let mut consts = 0usize;
            let mut hints = main.num_polys() + 4 * publics;
            // table walk
            let mut tperms = 0usize;
            let mut tops = 0usize;
            for (air, &n) in refs.iter().zip(logs.iter()) {
                let (width, _) = air.trace_layout();
                let layout = TableLayout::<F, E>::new(
                    air.constraint_program(),
                    air.constraints_meta(),
                    air.bus_interactions(),
                    width,
                    n,
                    Uniforms::default(),
                )
                .unwrap();
                let slots = layout.slot_of().to_vec();
                let bus = interaction_shapes::<E, _>(air.bus_interactions(), width, |c| {
                    slots.get(c).copied().ok_or(multilinear::Error::UnknownPolynomial {
                        index: c,
                        len: slots.len(),
                    })
                })
                .unwrap();
                let shape = TableShape {
                    ir: layout.shape(),
                    bus: &bus,
                    kinds: layout.kinds(),
                    num_columns: layout.num_columns(),
                    num_vars: n,
                };
                let c = table_verify_cost(&shape, entry);
                entry = c.entry();
                tops += c.leg.operations() + c.schedule.rows();
                tperms += c.perms();
                consts += c.leg.constants();
                // arena words of this table
                let layers = shape.gkr_layers();
                let gkr_words: usize = (0..layers).map(|i| 3 * i + 4).sum();
                let facts = layout.kinds().iter().filter(|k| k.source().is_some()).count();
                hints += 2 + gkr_words + n * shape.sumcheck_degree() + facts + 2 * n + layout.num_columns();
            }
            ops += tops;
            perms += tperms;
            // closure: the LfmPublic balance over the child's words, and Σ p/q.
            let closure = 4 + 13 * publics + 10 + 9 + 2;
            ops += closure;
            // main group and prepared group, threaded
            let group_of = |s: &[(usize, usize)]| -> Vec<usize> {
                s.iter()
                    .enumerate()
                    .flat_map(|(t, &(w, _))| std::iter::repeat_n(t, w))
                    .collect()
            };
            let mshape = ChainShape::new(&config, main.n_stack());
            let mc = stacked_verify_cost(&main, &group_of(main_shapes), &mshape, entry);
            entry = mc.entry();
            let pshape = ChainShape::new(&config, prep.n_stack());
            let pc = stacked_verify_cost(&prep, &group_of(&shapes_prep), &pshape, entry);
            ops += mc.operations() + pc.operations();
            perms += mc.perms() + pc.perms();
            hints += main.num_polys() * (1 + RoundStorage::words(&mshape) as usize)
                + prep.num_polys() * (1 + RoundStorage::words(&pshape) as usize);
            println!(
                "  leg {policy}: perms {perms} (tables {tperms} · main {} [{} chains n{} open {}] · prepared {} [{} n{}]) · ops {ops} (tables {tops} · main {} · prepared {} · closure {closure}) · hints {hints} · table consts {consts} · publics {publics}",
                mc.perms(),
                mc.chains(),
                main.n_stack(),
                chain_opening_perms(&mshape),
                pc.perms(),
                pc.chains(),
                prep.n_stack(),
                mc.operations(),
                pc.operations(),
            );
        }
    }
}

/// D-RATE (SOTA c.2 / Q3): one chain's in-guest cost at rate 1/4 (Q 112) against rate 1/2 (Q 225, and the
/// calculator-tight 219), query-only 20-bit grind, the production format (caps auto, first6), from the landed
/// closed forms.
///
/// cargo test --release -p lambda-vm-prover --lib drate_chain_costs -- --nocapture
#[test]
fn drate_chain_costs() {
    use super::whir_chain::{
        chain_grind_perms, chain_query_rows, chain_schedule_perms, chain_schedule_rows, chain_shape_rows,
    };
    use multilinear::whir_chain::GrindBits;
    println!("\n== D-RATE: one chain, the production format, grind 0/0/20, entered fresh ==");
    println!(
        "{:>3} {:>5} {:>4} {:>6} {:>8} {:>6} {:>7} {:>8} {:>9} {:>8} {:>9} {:>7}",
        "n", "rate", "Q", "rounds", "opening", "grind", "sponge", "perms", "shaperows", "queryrows", "schedrows", "words"
    );
    for n in 20..=28usize {
        for (log_blowup, q) in [(2usize, 112usize), (1, 225), (1, 219)] {
            let mut config = crate::multilinear_prove::chain_config(&[(1, n.min(27))]);
            config.log_blowup = log_blowup;
            config.num_queries = q;
            config.grind = GrindBits { folding: 0, ood: 0, query: 20 };
            let shape = ChainShape::new(&config, n);
            let open = chain_opening_perms(&shape);
            let grind = chain_grind_perms(&shape);
            let sponge = chain_schedule_perms(&shape, SpongeEntry::fresh());
            println!(
                "{:>3} {:>5} {:>4} {:>6} {:>8} {:>6} {:>7} {:>8} {:>9} {:>8} {:>9} {:>7}",
                n,
                format!("1/{}", 1 << log_blowup),
                q,
                shape.rounds(),
                open,
                grind,
                sponge,
                open + grind + sponge,
                chain_shape_rows(&shape),
                chain_query_rows(&shape),
                chain_schedule_rows(&shape, SpongeEntry::fresh()),
                RoundStorage::words(&shape),
            );
        }
    }
}

/// D-RATE: the same chain under a FIXED cap height `c` (`LAMBDA_VM_ZF_WHIR_CAP=c`), both rates — the one format knob
/// whose optimum moves with the query count.
#[test]
fn drate_cap_sweep() {
    use super::whir_chain::{chain_grind_perms, chain_schedule_perms, chain_shape_rows, chain_schedule_rows};
    use crypto::merkle_tree::cap::CapPolicy;
    use multilinear::whir_chain::GrindBits;
    println!("\n== D-RATE: cap sweep, one chain (perms · rows · words), grind 0/0/20, entered fresh ==");
    for n in [23usize, 25, 26, 27] {
        for (log_blowup, q) in [(2usize, 112usize), (1, 225)] {
            let mut line = format!("n {n:>2} rate 1/{} Q {q:>3}:", 1 << log_blowup);
            for c in [0usize, 3, 4, 5, 6, 7, 8, 9] {
                let mut config = crate::multilinear_prove::chain_config(&[(1, n.min(27))]);
                config.log_blowup = log_blowup;
                config.num_queries = q;
                config.grind = GrindBits { folding: 0, ood: 0, query: 20 };
                config.format.cap = if c == 0 { CapPolicy::Off } else { CapPolicy::Fixed(c as u8) };
                let shape = ChainShape::new(&config, n);
                let perms = chain_opening_perms(&shape)
                    + chain_grind_perms(&shape)
                    + chain_schedule_perms(&shape, SpongeEntry::fresh());
                let rows = chain_shape_rows(&shape) + chain_schedule_rows(&shape, SpongeEntry::fresh());
                line += &format!("  c{c}: {perms}/{rows}/{}", RoundStorage::words(&shape));
            }
            println!("{line}");
        }
    }
}

/// D-RATE: the EXACT per-chip rows of one emitted chain verifier (`whir_chain_tests::chain_program`), at rate 1/4
/// (Q 112) and rate 1/2 (Q 225), grind 0/0/20, production caps and folds. Real rows per chip, from the census.
#[test]
fn drate_chain_census() {
    use multilinear::whir_chain::GrindBits;
    println!("\n== D-RATE: emitted chain census (real rows per chip) ==");
    for n in [22usize, 23, 24, 25, 26, 27] {
        for (log_blowup, q) in [(2usize, 112usize), (1, 225)] {
            let mut config = crate::multilinear_prove::chain_config(&[(1, n.min(27))]);
            config.log_blowup = log_blowup;
            config.num_queries = q;
            config.grind = GrindBits { folding: 0, ood: 0, query: 20 };
            let shape = ChainShape::new(&config, n);
            let program = super::whir_chain_tests::chain_program(&shape);
            let census = super::airs::lfm_chip_census(&program);
            let line: Vec<String> = census
                .iter()
                .filter(|c| c.real_rows > 0 && c.name != "LFM_RANGE")
                .map(|c| format!("{}={}", &c.name[4..], c.real_rows))
                .collect();
            println!("CHAINCENSUS n={n} rate=1/{} Q={q} instrs={} {}", 1 << log_blowup, program.instrs.len(), line.join(" "));
        }
    }
}
