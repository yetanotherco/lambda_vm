//! A census of the batched zerocheck's work a row, per base table kind at the
//! production format, split by phase: the bus's linear combination (every
//! interaction's sides merged into one column, `fused::BusColumn`) against the
//! constraint program's grid pass and later rounds, the folds and the rest.
//! Counts come from `fused::model`, the closed form a test holds equal to what
//! the host reference executes, for the variant the card runs (rounds 0 and 1
//! on the base-field grid, the corners skipped, then Gruen's rounds). The
//! variant that walks every interaction's terms in the per-row program is
//! printed beside it, for comparison. Analysis only; prints `ZC WORK` lines.
//!
//! ```text
//! cargo test -p lambda-vm-prover --lib -- --exact --nocapture tests::zc_work_census::zc_work_census_of_the_block_tables
//! ```

use std::collections::BTreeSet;

use math::field::element::FieldElement;
use multilinear::fused::{self, Constraints, Counters, Options, Shape, Statement, Variant};
use multilinear::logup;
use stark::multilinear_air::beta_powers;
use stark::proof::options::ProofOptions;
use stark::traits::AIR;

use crate::block_whir::BlockFormat;
use crate::tables::types::{GoldilocksExtension, GoldilocksField};

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;
type EE = FieldElement<Ext3>;
type DynAir<'a> = dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()> + 'a;

/// A fixed, non-degenerate stand-in for a challenge: the census counts steps,
/// which no random value changes except at zero or one.
fn ext(seed: u64) -> EE {
    let s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 3;
    EE::new([
        FieldElement::from(s),
        FieldElement::from(s.rotate_left(21) | 5),
        FieldElement::from(s.rotate_left(42) | 7),
    ])
}

/// The op model's units a row of each phase at `2^n` rows, and the total.
fn per_row(counters: &Counters, n: usize) -> (Vec<(&'static str, f64)>, f64) {
    let rows = (1u64 << n) as f64;
    let phases: Vec<(&'static str, f64)> = counters
        .phases()
        .iter()
        .map(|(name, ops)| (*name, ops.units(true, true) / rows))
        .collect();
    let total = counters.units(true, true) / rows;
    (phases, total)
}

/// One kind's census line.
fn census(air: &DynAir<'_>) -> Option<String> {
    let name = air.name().to_string();
    let width = air.trace_layout().0;
    let layout = crate::multilinear_prove::layout_of(air, width, 4).expect("the table lays out");
    let slots = layout.slot_of().to_vec();
    let interactions = stark::multilinear_logup::interactions(
        air.bus_interactions(),
        slots.len(),
        &ext(1),
        &ext(2),
        |col| {
            slots
                .get(col)
                .copied()
                .ok_or(multilinear::Error::UnknownPolynomial {
                    index: col,
                    len: slots.len(),
                })
        },
    )
    .expect("the interactions instantiate");
    if interactions.is_empty() {
        return Some(format!("ZC WORK {name} · no interactions"));
    }
    let shape = layout.shape();
    let constraints = Constraints::<Gl>::from_extension(
        &shape.steps_as_ops(),
        shape.root_steps(),
        shape.selector_of_root(),
        shape.degree(),
    )
    .expect("the base-field precondition holds");
    let betas = beta_powers(&ext(3), shape.num_roots());
    let n0 = 4;
    let r: Vec<EE> = (0..n0).map(|i| ext(10 + i as u64)).collect();
    let claim_point: Vec<EE> = (0..logup::input_layer_vars(interactions.len(), n0))
        .map(|i| ext(100 + i as u64))
        .collect();
    let statement = Statement {
        constraints: &constraints,
        betas: &betas,
        interactions: &interactions,
        claim_point: &claim_point,
        r: &r,
    };
    let lambdas = multilinear::challenge_powers(&ext(4), 3);
    let factors = layout.kinds().len();
    let base = Shape::of(&statement, factors, &lambdas, false).expect("the shape");
    let bus_reads: BTreeSet<usize> = fused::BusColumn::new(
        &interactions,
        &claim_point[..claim_point.len() - n0],
        &lambdas[1],
        &lambdas[2],
    )
    .expect("the bus column")
    .terms()
    .iter()
    .map(|(slot, _)| *slot)
    .collect();
    let constraint_reads: BTreeSet<usize> = constraints.reads().into_iter().collect();
    let bus_only = bus_reads.difference(&constraint_reads).count();
    let raw_terms: usize = interactions
        .iter()
        .map(|i| i.numerator.terms().len() + i.denominator.terms().len())
        .sum();
    // The card's lowered program and the threads its launches can take: the
    // slot file is budgeted at 512 MiB (`argue_fused::ceiling`) and a launch is
    // capped at 2^17 threads, split over the grid's points (rounds: nodes + 2).
    let lowered = multilinear::gpu_fused::lower_fused(&constraints);
    let (lowered_steps, slots) = lowered
        .as_ref()
        .map_or((0, 0), |l| (l.nodes.len() / 2, l.num_slots));
    let threads = |bytes_a_thread: u64| -> u64 {
        ((512u64 << 20) / bytes_a_thread.max(1))
            .clamp(32, 1 << 20)
            .min(1 << 17)
    };
    let d = base.constraint_degree as u64;
    let points = ((d + 1) * (d + 1)).saturating_sub(4).max(1);
    let grid_threads = threads(slots as u64 * 8);
    let round_threads = threads(slots as u64 * 24);
    let mut line = format!(
        "ZC WORK {name} · factors {factors} · constraint reads {} · dag steps {} · roots {} · degree {} \
         · lowered steps {lowered_steps} · live slots {slots} · grid threads {grid_threads} over {points} points \
         (slot file {} MiB) · round threads {round_threads} (slot file {} MiB) \
         · interactions {} · raw bus terms {raw_terms} · bus column terms {} · bus-only factors {bus_only}",
        constraint_reads.len(),
        base.dag.steps(),
        base.roots,
        base.constraint_degree,
        (grid_threads * slots as u64 * 8) >> 20,
        (round_threads * slots as u64 * 24) >> 20,
        interactions.len(),
        bus_reads.len(),
    );
    for n in [16usize, 21] {
        let mut at = base;
        at.num_vars = n;
        let grid = fused::model(
            &at,
            Options {
                skip_corners: true,
                ..Options::new(Variant::Grid)
            },
        );
        let today = fused::model(&at, Options::new(Variant::Today));
        let (phases, total) = per_row(&grid, n);
        let (_, today_total) = per_row(&today, n);
        line.push_str(&format!(" || n={n}: units a row {total:.1}"));
        for (phase, units) in phases {
            line.push_str(&format!(" · {phase} {units:.1}"));
        }
        line.push_str(&format!(
            " · bytes a row {:.0} · terms-in-program variant {today_total:.1}",
            grid.total().bytes as f64 / (1u64 << n) as f64
        ));
    }
    Some(line)
}

#[test]
fn zc_work_census_of_the_block_tables() {
    let opts = ProofOptions::default_test_options();
    let elf = crate::test_utils::asm_elf_bytes("keep_alive_regs");
    let program = executor::elf::Elf::load(&elf).expect("the ELF loads");
    let pages = crate::tables::trace_builder::Traces::page_configs_from_elf_and_runtime(
        &program,
        &[],
        0,
        1 << 20,
    )
    .expect("pages");
    let counts = crate::TableCounts {
        cpu: 1,
        lt: 1,
        memw: 1,
        memw_aligned: 1,
        load: 1,
        mul: 1,
        dvrm: 1,
        shift: 1,
        branch: 1,
        memw_register: 1,
        eq: 1,
        bytewise: 1,
        store: 1,
        cpu32: 1,
        keccak: 1,
        keccak_rnd: 1,
        ecsm: 1,
        ecdas: 1,
        hint: 1,
        copy: 1,
        blake3: 1,
        is_wh: 1,
        ecrec: 1,
    };
    let inputs =
        crate::BlockAirInputs::new(&opts, &program, vec![1 << 4], &[]).expect("the AIR inputs");
    let format = BlockFormat::production();
    let airs = crate::block_airs(
        &program,
        &opts,
        &pages,
        &counts,
        &inputs,
        format.keccak_rnd,
        format.memw_aligned,
        format.registers,
    );
    let mut seen = BTreeSet::new();
    let mut lines = 0;
    for air in airs.air_refs() {
        if !seen.insert(air.name().to_string()) {
            continue;
        }
        if let Some(line) = census(air) {
            println!("{line}");
            lines += 1;
        }
    }
    assert!(lines >= 20, "a line per base table kind ({lines})");
}
