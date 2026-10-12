//! The grouped WHIR leg (`GroupFold::Indexed`) a RELEASE build of the library
//! emits — no `cfg(test)`, so no emitter faults and no step marks — held to
//! the pins the library's own test build is held to
//! (`lfm::whir_onechain_tests::the_grouped_leg_stream_of_a_test_build`): the
//! program the unit tests gate is the program a release build emits.
//!
//! ```text
//! cargo test --release -p lambda-vm-prover --test onechain_release_stream -- --nocapture
//! ```

use lambda_vm_prover::lfm::builder::{Cell, Ext, LfmBuilder};
use lambda_vm_prover::lfm::compiler::compile;
use lambda_vm_prover::lfm::edsl::WrapHash;
use lambda_vm_prover::lfm::whir_chain::{ChainShape, RoundStorage};
use lambda_vm_prover::lfm::whir_stacked::{StackedChainWires, emit_stacked_verify_chains};
use lambda_vm_prover::lfm::whir_transcript::WhirTranscript;
use lambda_vm_prover::tables::types::GoldilocksField;

/// The grouped leg's instruction stream, emitted over hinted wires alone — it
/// is a function of the layout and the config, never of a proof — as
/// `(instructions, FNV-1a of their Debug text)`. ★ The SAME body sits in
/// `prover/src/lfm/whir_onechain_tests.rs`, built WITH `cfg(test)` (the
/// emitter faults and the step marks), and both builds are held to the same
/// pins: the test build emits exactly the release build's program.
fn grouped_leg_stream(p1: bool, heights: &[usize], n_stack: usize) -> (usize, u64) {
    use multilinear::whir_chain::{
        CapPolicy, ChainConfig, ChainFormat, FirstFold, GrindBits, GroupFold, WhirFolds,
    };
    let mut format = ChainFormat {
        folds: WhirFolds::First(FirstFold::new(6).expect("first6")),
        group: GroupFold::Indexed,
        ..ChainFormat::DEFAULT
    };
    if p1 {
        format.arity4_cap = CapPolicy::Fixed(2);
    } else {
        format.cap = CapPolicy::Fixed(1);
    }
    let cfg = ChainConfig {
        log_blowup: 2,
        log_folding: 4,
        num_queries: 3,
        grind: GrindBits::uniform(8),
        format,
    };
    let layout = multilinear::stacking::StackedLayout::build(heights, n_stack).expect("packs");
    let shape = ChainShape::new_at(&cfg, n_stack, if p1 { 4 } else { 2 });
    let domain = multilinear::whir::Domain::<GoldilocksField>::new(n_stack + cfg.log_blowup)
        .expect("domain");
    let hash = if p1 {
        WrapHash::Poseidon1
    } else {
        WrapHash::production()
    };
    let runs = shape.chains(layout.num_polys());
    let mut total: u32 = heights.iter().sum::<usize>() as u32 + heights.len() as u32;
    for run in &runs {
        total += run.len() as u32 + 1 + RoundStorage::words(&shape.run(run.len()));
    }
    let mut b = LfmBuilder::new().with_wrap_hash(hash);
    let arena = b.declare_arena(total);
    let mut transcript = WhirTranscript::for_builder(&mut b);
    let mut at = 0u32;
    let next = |b: &mut LfmBuilder, at: &mut u32| {
        let cell = b.hint_word(arena, *at);
        *at += 1;
        cell
    };
    let points: Vec<Vec<Ext>> = heights
        .iter()
        .map(|&h| (0..h).map(|_| next(&mut b, &mut at).as_ext()).collect())
        .collect();
    let values: Vec<Ext> = heights
        .iter()
        .map(|_| next(&mut b, &mut at).as_ext())
        .collect();
    let mut roots: Vec<Vec<Cell>> = Vec::new();
    let mut finals: Vec<Ext> = Vec::new();
    let mut storage: Vec<RoundStorage> = Vec::new();
    for run in &runs {
        roots.push(run.clone().map(|_| next(&mut b, &mut at)).collect());
        finals.push(next(&mut b, &mut at).as_ext());
        let run_shape = shape.run(run.len());
        let base = at;
        storage.push(RoundStorage::hint(&mut b, arena, base, &run_shape));
        at += RoundStorage::words(&run_shape);
    }
    let openings: Vec<_> = storage.iter().map(RoundStorage::openings).collect();
    let rounds: Vec<Vec<_>> = storage
        .iter()
        .zip(&openings)
        .map(|(chain, (current, next))| chain.wires(current, next))
        .collect();
    let chains: Vec<StackedChainWires<'_>> = (0..runs.len())
        .map(|i| StackedChainWires {
            rounds: &rounds[i],
            roots: &roots[i],
            final_value: finals[i],
        })
        .collect();
    let claimed_at: Vec<&[Ext]> = points.iter().map(Vec::as_slice).collect();
    let gamma = emit_stacked_verify_chains(
        &mut b,
        &mut transcript,
        &layout,
        &chains,
        &claimed_at,
        &values,
        &shape,
        &domain,
    );
    b.public(gamma.as_cell());
    let program = compile(b.finish());
    let text = format!("{:?}", program.instrs);
    let digest = text.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, byte| {
        (h ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    });
    (program.instrs.len(), digest)
}

/// One pinned case: the base (`true` for Poseidon1), the columns' heights, the
/// stack's height, and the stream's `(instructions, FNV-1a)`.
type StreamPin = (bool, &'static [usize], usize, (usize, u64));

/// The cases both builds are held to, each with its stream's pin: RPX and
/// Poseidon1, five polynomials split [4, 1] and six split [4, 2] whose short
/// columns' selectors run past `k0`. A pin moves only with the emitted
/// program; move it in both files together.
const GROUPED_LEG_STREAMS: [StreamPin; 4] = [
    (false, &[8; 5], 8, (5241, 0x8d95_4e39_3cca_f387)),
    (false, &[8, 3, 8, 2, 4, 8], 8, (5229, 0xac5d_a244_80ae_4271)),
    (true, &[10; 5], 10, (28895, 0x52ec_ae97_ae80_6593)),
    (
        true,
        &[10, 3, 10, 2, 4, 10],
        10,
        (28868, 0x02e6_e3fd_5c51_0e98),
    ),
];

#[test]
fn the_grouped_leg_stream_of_a_release_build() {
    for (p1, heights, n, pin) in GROUPED_LEG_STREAMS {
        let stream = grouped_leg_stream(p1, heights, n);
        println!(
            "ONECHAIN STREAM p1={p1} heights={heights:?}: {} instructions, digest {:016x}",
            stream.0, stream.1
        );
        assert_eq!(
            stream, pin,
            "p1={p1} heights={heights:?}: the pinned stream"
        );
    }
}
