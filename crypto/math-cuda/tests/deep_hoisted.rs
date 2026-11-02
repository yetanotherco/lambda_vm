//! `LAMBDA_VM_DEEP_HOISTED`: the fused DEEP with its row-independent part
//! hoisted (`deep_composition_ext3_hoisted_m*`) against the fused kernel it
//! replaces, through `deep_composition_ext3_fused_keep` with the setting off
//! then on: the hoisted codeword is the fused one's canonical words, word for
//! word, at every instantiation (0..=3 eval points), with and without aux
//! columns; it is canonical throughout; and a row stride other than 1 keeps
//! the fused kernel.
//!
//! Needs a GPU, like every test in this directory. A binary of its own, so the
//! setting's override touches no other test.

use std::sync::Arc;

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
use math::field::goldilocks::GoldilocksField as Gl;
use math::field::traits::IsPrimeField;
use math_cuda::deep::{deep_composition_ext3_fused_keep, deep_hoisted_report, set_deep_hoisted};
use math_cuda::device::backend;
use math_cuda::lde::{GpuLdeBase, GpuLdeExt3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<Gl>;
type Fp3 = FieldElement<Ext3>;

fn rand_fp3(rng: &mut ChaCha8Rng) -> [u64; 3] {
    let e = Fp3::new([
        Fp::from_raw(rng.r#gen::<u64>()),
        Fp::from_raw(rng.r#gen::<u64>()),
        Fp::from_raw(rng.r#gen::<u64>()),
    ]);
    [
        *e.value()[0].value(),
        *e.value()[1].value(),
        *e.value()[2].value(),
    ]
}

fn canon(xs: &[u64]) -> Vec<u64> {
    xs.iter().map(Gl::canonical).collect()
}

/// The fused DEEP's codeword (FRI order) on random inputs, `hoisted` set as
/// given; `row_stride` rows apart over an LDE `row_stride` times the domain.
#[allow(clippy::too_many_arguments)]
fn codeword(
    domain: usize,
    num_main: usize,
    num_aux: usize,
    parts: usize,
    k: usize,
    row_stride: usize,
    seed: u64,
    hoisted: bool,
) -> Vec<u64> {
    let lde = domain * row_stride;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let be = backend().expect("cuda backend");
    let stream = be.next_stream();
    let main_flat: Vec<u64> = (0..num_main.max(1) * lde)
        .map(|_| rng.r#gen::<u64>())
        .collect();
    let ext = |cols: usize, rng: &mut ChaCha8Rng| -> Vec<u64> {
        (0..cols.max(1) * 3 * lde)
            .map(|_| rng.r#gen::<u64>() % 0xFFFF_FFFF_0000_0001)
            .collect()
    };
    let aux_flat = ext(num_aux, &mut rng);
    let parts_flat = ext(parts, &mut rng);
    let main = GpuLdeBase {
        ready: None,
        buf: Arc::new(stream.clone_htod(&main_flat).expect("htod")),
        m: num_main,
        lde_size: lde,
        tree: None,
        trace_dev: None,
        trace_rows: 0,
    };
    let aux = (num_aux > 0).then(|| GpuLdeExt3 {
        ready: None,
        buf: Arc::new(stream.clone_htod(&aux_flat).expect("htod")),
        m: num_aux,
        lde_size: lde,
        tree: None,
    });
    let parts_dev = GpuLdeExt3 {
        ready: None,
        buf: Arc::new(stream.clone_htod(&parts_flat).expect("htod")),
        m: parts,
        lde_size: lde,
        tree: None,
    };
    let coset: Vec<u64> = (0..lde).map(|_| rng.r#gen::<u64>() | 1).collect();
    let coset_dev = stream.clone_htod(&coset).expect("htod");
    let cols = num_main + num_aux;
    let z: Vec<u64> = (0..1 + k).flat_map(|_| rand_fp3(&mut rng)).collect();
    let h_ood: Vec<u64> = (0..parts).flat_map(|_| rand_fp3(&mut rng)).collect();
    let trace_ood: Vec<u64> = (0..cols * k).flat_map(|_| rand_fp3(&mut rng)).collect();
    let gammas_h: Vec<u64> = (0..parts).flat_map(|_| rand_fp3(&mut rng)).collect();
    let gammas_tr: Vec<u64> = (0..cols * k).flat_map(|_| rand_fp3(&mut rng)).collect();
    stream.synchronize().expect("sync");
    set_deep_hoisted(Some(hoisted));
    let out = deep_composition_ext3_fused_keep(
        &stream,
        &main,
        aux.as_ref(),
        &parts_dev,
        &coset_dev,
        &z,
        &h_ood,
        &trace_ood,
        &gammas_h,
        &gammas_tr,
        parts,
        num_main,
        num_aux,
        k,
        row_stride,
        domain,
    )
    .expect("the fused DEEP runs")
    .download()
    .expect("download");
    set_deep_hoisted(None);
    out
}

#[test]
fn the_hoisted_codeword_is_the_fused_one_word_for_word() {
    for (domain, num_main, num_aux, parts, k, seed) in [
        (1usize << 10, 5usize, 3usize, 2usize, 3usize, 1u64),
        (1 << 10, 4, 0, 1, 0, 2),
        (1 << 11, 3, 2, 2, 1, 3),
        (1 << 9, 6, 4, 2, 2, 4),
        (1 << 16, 23, 3, 2, 3, 5),
        (1 << 12, 40, 6, 2, 3, 6),
    ] {
        let fused = codeword(domain, num_main, num_aux, parts, k, 1, seed, false);
        let hoisted = codeword(domain, num_main, num_aux, parts, k, 1, seed, true);
        let what = format!("domain {domain} main {num_main} aux {num_aux} parts {parts} k {k}");
        assert_eq!(hoisted.len(), fused.len(), "{what}: length");
        assert!(
            hoisted.iter().all(|v| *v < 0xFFFF_FFFF_0000_0001),
            "{what}: a non-canonical word"
        );
        let want = canon(&fused);
        if let Some(i) = hoisted.iter().zip(&want).position(|(a, b)| a != b) {
            panic!(
                "{what}: word {i} differs: {:#x} != {:#x} (fused raw {:#x})",
                hoisted[i], want[i], fused[i]
            );
        }
    }
    let line = deep_hoisted_report();
    assert!(
        line.contains("hoisted launches 6"),
        "six hoisted launches: {line}"
    );
}

/// A row stride other than 1 keeps the fused kernel, counted, with the same
/// codeword.
#[test]
fn a_row_stride_other_than_one_keeps_the_fused_kernel() {
    let before = deep_hoisted_report();
    let fused = codeword(1 << 9, 4, 2, 2, 2, 4, 9, false);
    let on = codeword(1 << 9, 4, 2, 2, 2, 4, 9, true);
    assert_eq!(
        on, fused,
        "the fused kernel both times, so the same raw words"
    );
    let after = deep_hoisted_report();
    assert_ne!(before, after, "the fallback is counted: {before} → {after}");
    assert!(after.contains("fused kept"), "{after}");
}
