//! Where a level program's emission spends its time: the builder's walk, the
//! source's finish, the compile and the admission validator, for the BLAKE3
//! W-leg over a small lean program's build. An instrument, test-only: run
//! alone with `--ignored --exact --nocapture`.

use std::time::Instant;

use super::compiler::{compile, compile_parallel_sliced};
use super::wlfm::{WlfmFamily, WlfmLevers, WlfmRate, build_wlfm_under};
use super::wlfm_leg::{WlfmChild, emit_wlfm_leg, wlfm_leg_builder};

#[test]
#[ignore = "instrument: run alone with --exact"]
fn level_emission_split() {
    let program = super::b3l_wlfm_tests::honest();
    let build = build_wlfm_under(
        &program,
        super::wlfm_tests::HASHER,
        WlfmRate::Two,
        WlfmLevers::with_family(WlfmFamily::B3),
        true,
    )
    .expect("builds");
    let child = WlfmChild {
        artifacts: build.artifacts.clone(),
        num_public_words: 8,
    };
    for run in 0..3 {
        let t = Instant::now();
        let mut b = wlfm_leg_builder(WlfmFamily::B3);
        let arena = b.declare_arena(0);
        let leg = emit_wlfm_leg(&mut b, &child, arena).expect("emits");
        for word in &leg.publics {
            let packed = b.pack_word([word.lanes[0], word.lanes[1], word.lanes[2], word.lanes[3]]);
            b.public(packed);
        }
        let walk = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let source = b.finish();
        let finish = t.elapsed().as_secs_f64();
        // The same source again, for the parallel compile.
        let mut b2 = wlfm_leg_builder(WlfmFamily::B3);
        let arena2 = b2.declare_arena(0);
        let leg2 = emit_wlfm_leg(&mut b2, &child, arena2).expect("emits");
        for word in &leg2.publics {
            let packed = b2.pack_word([word.lanes[0], word.lanes[1], word.lanes[2], word.lanes[3]]);
            b2.public(packed);
        }
        let source2 = b2.finish();
        let t = Instant::now();
        let compiled = compile(source);
        let compile_secs = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let _parallel = compile_parallel_sliced(source2, 1 << 15);
        let parallel_secs = t.elapsed().as_secs_f64();
        let split = super::validator::validate_split(&compiled);
        let n = compiled.instrs.len();
        let ns = |s: f64| 1e9 * s / n as f64;
        println!(
            "LEVEL EMIT run {run}: {n} instructions · walk {walk:.3}s ({:.1} ns) · finish {finish:.3}s ({:.1} ns) · compile {compile_secs:.3}s ({:.1} ns) · compile parallel {parallel_secs:.3}s ({:.1} ns) · validate {}",
            ns(walk),
            ns(finish),
            ns(compile_secs),
            ns(parallel_secs),
            split
                .iter()
                .map(|(k, s)| format!("{k} {s:.3}s ({:.1} ns)", ns(*s)))
                .collect::<Vec<_>>()
                .join(" · ")
        );
    }
}

/// Where a streamed execution's setup goes, for the BLAKE3 W-leg's program
/// with its one arena landing in 27 pieces (level 1's shape): the schedule,
/// the need pass, the wave buckets and the first wave, in ns an instruction.
/// An instrument, test-only: run alone with `--ignored --exact --nocapture`.
#[test]
#[ignore = "instrument: run alone with --exact"]
fn streamed_setup_split() {
    use super::executor::{ArenaPiece, StreamedExecution};
    let program = super::b3l_wlfm_tests::honest();
    let build = build_wlfm_under(
        &program,
        super::wlfm_tests::HASHER,
        WlfmRate::Two,
        WlfmLevers::with_family(WlfmFamily::B3),
        true,
    )
    .expect("builds");
    let child = WlfmChild {
        artifacts: build.artifacts.clone(),
        num_public_words: 8,
    };
    let mut b = wlfm_leg_builder(WlfmFamily::B3);
    let arena = b.declare_arena(0);
    let leg = emit_wlfm_leg(&mut b, &child, arena).expect("emits");
    for word in &leg.publics {
        let packed = b.pack_word([word.lanes[0], word.lanes[1], word.lanes[2], word.lanes[3]]);
        b.public(packed);
    }
    let compiled = compile(b.finish());
    let len = compiled.arena_schema.lens[0];
    let n_pieces = 27u32.min(len);
    let pieces: Vec<ArenaPiece> = (0..n_pieces)
        .map(|k| {
            let start = len * k / n_pieces;
            let end = len * (k + 1) / n_pieces;
            ArenaPiece {
                arena: 0,
                start,
                len: end - start,
                group: k as usize,
            }
        })
        .filter(|p| p.len > 0)
        .collect();
    let hasher =
        compiled.hasher(<crate::hash_pin::BlockFamily as crate::hash_pin::ProofFamily>::SOCKET);
    let n = compiled.instrs.len();
    let ns = |s: f64| 1e9 * s / n as f64;
    for bucketed in [false, true, true] {
        let t = Instant::now();
        let ex = StreamedExecution::new_pieced_bucketed(
            &compiled,
            pieces.clone(),
            &hasher,
            false,
            bucketed,
        )
        .expect("starts");
        let total = t.elapsed().as_secs_f64();
        let x = *ex.split();
        println!(
            "STREAMED SETUP bucketed {bucketed}: {n} instructions · {} pieces · total {total:.3}s ({:.1} ns) · schedule \
             {:.3}s ({:.1} ns) · need pass {:.3}s ({:.1} ns) · buckets {:.3}s ({:.1} ns) · first wave {:.3}s ({:.1} ns) · \
             machine {:.3}s ({:.1} ns) · first wave ran {}",
            pieces.len(),
            ns(total),
            x.depth_pass,
            ns(x.depth_pass),
            x.need_pass,
            ns(x.need_pass),
            x.bucket_build,
            ns(x.bucket_build),
            x.first_wave,
            ns(x.first_wave),
            x.setup,
            ns(x.setup),
            ex.ran()
        );
    }
}
