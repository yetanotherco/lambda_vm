//! The column-major LDE engine (`math_cuda::lde_cm`) against the legacy
//! pipelines, entry point by entry point, on the same inputs in one process
//! (`with_engine` forces the pipeline per call).
//!
//! What must hold for the engine to be a pure performance change:
//! - the Merkle roots — and for the split trees the downloaded precomputed
//!   node bytes — are identical;
//! - the device LDE handles hold the same field values (compared canonically:
//!   the engine may leave a different non-canonical representative, which
//!   nothing downstream observes — the hashes and serialisation canonicalise);
//! - the trace-domain snapshot is identical, raw u64 for raw u64;
//! - the host outputs of the batched entry points are identical (canonically).
//!
//! The engine's arithmetic is also pinned on the host by
//! `tests/host_kat/ntt_cm_host_kat.cpp`; `engine_matches_the_cpu_lde` anchors the
//! device engine to the CPU `Polynomial::coset_lde_full_expand` independently
//! of the legacy GPU path.

use math::fft::bowers_fft::LayerTwiddles;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsPrimeField;
use math::polynomial::Polynomial;
use math_cuda::DeviceHash;
use math_cuda::lde_cm::{columns_extended, k1b_codewords, spread_supports, with_engine, with_k1b};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<GoldilocksField>;

const COSET_OFFSET: u64 = 7;

/// `weights[i] = g^i / n` (evaluations) or `g^i` (coefficients).
fn weights_u64(n: usize, g: u64, with_inv_n: bool) -> Vec<u64> {
    let mut cur = if with_inv_n {
        Fp::from(n as u64).inv().unwrap()
    } else {
        Fp::one()
    };
    let g_fp = Fp::from_raw(g);
    let mut w = Vec::with_capacity(n);
    for _ in 0..n {
        w.push(*cur.value());
        cur = &cur * &g_fp;
    }
    w
}

fn random(rng: &mut ChaCha8Rng, len: usize) -> Vec<u64> {
    (0..len).map(|_| rng.r#gen::<u64>()).collect()
}

fn canon(xs: &[u64]) -> Vec<u64> {
    xs.iter().map(GoldilocksField::canonical).collect()
}

fn assert_same_values(engine: &[u64], legacy: &[u64], what: &str) {
    assert_eq!(engine.len(), legacy.len(), "{what}: length");
    let (a, b) = (canon(engine), canon(legacy));
    if let Some(i) = (0..a.len()).find(|&i| a[i] != b[i]) {
        panic!(
            "{what}: first difference at {i}: engine {:#x} legacy {:#x}",
            a[i], b[i]
        );
    }
}

/// D2H a handle's buffer once its `ready` event has fired.
fn download(
    buf: &math_cuda::CudaSlice<u64>,
    ready: Option<&math_cuda::device::PooledEvent>,
) -> Vec<u64> {
    let be = math_cuda::device::backend().expect("cuda backend");
    let stream = be.next_stream();
    if let Some(ev) = ready {
        stream.wait(ev.event()).expect("wait on the ready event");
    }
    let out = stream.clone_dtoh(buf).expect("D2H");
    stream.synchronize().expect("D2H sync");
    out
}

const HASHES: [DeviceHash; 3] = [
    DeviceHash::Rpx256,
    DeviceHash::Keccak256,
    DeviceHash::Blake3,
];

/// (log_n, blowup, cols). The small shapes run in milliseconds; the last two
/// put several columns in several L2 chunks (a 2^20-row column at blowup 4
/// is ~40 MB, so every column is its own chunk), which is what exercises the
/// chunk walk — and, for the aux commit, its descending order over a source
/// that lives inside the destination.
const SHAPES: &[(usize, usize, usize)] = &[
    (4, 2, 1),
    (4, 16, 3),
    (5, 4, 7),
    (6, 2, 37),
    (8, 8, 5),
    (9, 2, 316),
    (10, 4, 13),
    (12, 2, 64),
    (13, 4, 9),
    (16, 2, 11),
    (20, 4, 3),
    (21, 2, 2),
];

/// The main (base) commit: host and pre-uploaded input, both leaf layouts.
#[test]
fn base_commit_engine_matches_legacy() {
    let be = math_cuda::device::backend().expect("cuda backend");
    for (i, &(log_n, blowup, cols)) in SHAPES.iter().enumerate() {
        let n = 1usize << log_n;
        let mut rng = ChaCha8Rng::seed_from_u64(0xB45E_0000 + i as u64);
        let row_major = random(&mut rng, n * cols);
        let weights = weights_u64(n, COSET_OFFSET, true);
        let predev = {
            let stream = be.next_stream();
            let d = stream.clone_htod(&row_major).expect("predev upload");
            stream.synchronize().expect("predev sync");
            d
        };
        for (j, rpl) in [2usize, 1].into_iter().enumerate() {
            let hash = HASHES[(i + j) % HASHES.len()];
            for dev_input in [false, true] {
                let what = format!(
                    "base log_n={log_n} blowup={blowup} cols={cols} rpl={rpl} {hash:?} predev={dev_input}"
                );
                let run = |on: bool| {
                    with_engine(on, || {
                        math_cuda::lde::coset_lde_row_major_with_merkle_tree_keep_rpl(
                            &row_major,
                            dev_input.then_some(&predev),
                            hash,
                            n,
                            cols,
                            blowup,
                            &weights,
                            false,
                            rpl,
                        )
                    })
                    .expect("base commit")
                };
                let before = columns_extended();
                let (engine, engine_host) = run(true);
                assert!(
                    columns_extended() - before >= cols as u64,
                    "{what}: the entry point did not route through the engine"
                );
                let (legacy, legacy_host) = run(false);
                assert!(
                    engine_host.is_empty() && legacy_host.is_empty(),
                    "{what}: device-only"
                );
                assert_eq!(
                    engine.tree.as_ref().unwrap().root,
                    legacy.tree.as_ref().unwrap().root,
                    "{what}: root"
                );
                assert_eq!(
                    engine.tree.as_ref().unwrap().leaves_len,
                    legacy.tree.as_ref().unwrap().leaves_len
                );
                assert_eq!(
                    (engine.m, engine.lde_size, engine.trace_rows),
                    (legacy.m, legacy.lde_size, legacy.trace_rows)
                );
                assert_same_values(
                    &download(&engine.buf, engine.ready.as_deref()),
                    &download(&legacy.buf, legacy.ready.as_deref()),
                    &what,
                );
                assert_eq!(
                    download(engine.trace_dev.as_ref().unwrap(), engine.ready.as_deref()),
                    download(legacy.trace_dev.as_ref().unwrap(), legacy.ready.as_deref()),
                    "{what}: trace snapshot"
                );
            }
        }
    }
}

/// The aux (ext3) commits: no snapshot, so the engine reads a column-major
/// copy of the trace staged at the head of the LDE buffer itself.
#[test]
fn ext3_commit_engine_matches_legacy() {
    let be = math_cuda::device::backend().expect("cuda backend");
    for (i, &(log_n, blowup, cols)) in SHAPES.iter().enumerate() {
        // `cols` ext3 columns: keep the widest shapes affordable.
        let cols = cols.min(40);
        let n = 1usize << log_n;
        let mut rng = ChaCha8Rng::seed_from_u64(0xE473_0000 + i as u64);
        let row_major = random(&mut rng, n * cols * 3);
        let weights = weights_u64(n, COSET_OFFSET, true);
        let dev_input = {
            let stream = be.next_stream();
            let d = stream.clone_htod(&row_major).expect("upload");
            stream.synchronize().expect("sync");
            d
        };
        for (j, rpl) in [2usize, 1].into_iter().enumerate() {
            let hash = HASHES[(i + j + 1) % HASHES.len()];
            let what = format!("ext3 log_n={log_n} blowup={blowup} cols={cols} rpl={rpl} {hash:?}");
            let host = |on: bool| {
                with_engine(on, || {
                    math_cuda::lde::coset_lde_ext3_row_major_with_merkle_tree_keep_rpl(
                        &row_major, hash, n, cols, blowup, &weights, false, rpl,
                    )
                })
                .expect("ext3 commit")
                .0
            };
            let dev = |on: bool| {
                with_engine(on, || {
                    math_cuda::lde::coset_lde_ext3_row_major_with_merkle_tree_keep_dev_rpl(
                        &dev_input, hash, n, cols, blowup, &weights, false, rpl,
                    )
                })
                .expect("ext3 dev commit")
                .0
            };
            let legacy = host(false);
            let legacy_vals = download(&legacy.buf, legacy.ready.as_deref());
            for (engine, form) in [(host(true), "host"), (dev(true), "dev")] {
                assert_eq!(
                    engine.tree.as_ref().unwrap().root,
                    legacy.tree.as_ref().unwrap().root,
                    "{what} {form}: root"
                );
                assert_eq!((engine.m, engine.lde_size), (legacy.m, legacy.lde_size));
                assert_same_values(
                    &download(&engine.buf, engine.ready.as_deref()),
                    &legacy_vals,
                    &format!("{what} {form}"),
                );
            }
        }
    }
}

/// The preprocessed commit: two subset trees over one LDE.
#[test]
fn split_trees_engine_matches_legacy() {
    for (i, &(log_n, blowup, cols)) in SHAPES.iter().enumerate() {
        if cols < 2 {
            continue; // the split needs a column on each side
        }
        let n = 1usize << log_n;
        let split = (cols / 3).max(1);
        let mut rng = ChaCha8Rng::seed_from_u64(0x5B11_0000 + i as u64);
        let row_major = random(&mut rng, n * cols);
        let weights = weights_u64(n, COSET_OFFSET, true);
        for (j, rpl) in [2usize, 1].into_iter().enumerate() {
            let hash = HASHES[(i + j + 2) % HASHES.len()];
            let what =
                format!("split log_n={log_n} blowup={blowup} cols={cols} rpl={rpl} {hash:?}");
            let run = |on: bool| {
                with_engine(on, || {
                    math_cuda::lde::coset_lde_row_major_split_trees_rpl(
                        &row_major, None, hash, n, cols, blowup, &weights, split, true, false, rpl,
                    )
                })
                .expect("split commit")
            };
            let (engine_pre, engine, _) = run(true);
            let (legacy_pre, legacy, _) = run(false);
            assert_eq!(engine_pre, legacy_pre, "{what}: precomputed tree nodes");
            assert_eq!(
                engine.tree.as_ref().unwrap().root,
                legacy.tree.as_ref().unwrap().root,
                "{what}: multiplicity root"
            );
            assert_same_values(
                &download(&engine.buf, engine.ready.as_deref()),
                &download(&legacy.buf, legacy.ready.as_deref()),
                &what,
            );
            assert_eq!(
                download(engine.trace_dev.as_ref().unwrap(), engine.ready.as_deref()),
                download(legacy.trace_dev.as_ref().unwrap(), legacy.ready.as_deref()),
                "{what}: trace snapshot"
            );
        }
    }
}

/// A caller that wants the row-major host LDE keeps the legacy path even with
/// the engine on (the engine's LDE is column-major), and gets the same bytes.
#[test]
fn host_lde_callers_keep_the_legacy_path() {
    let (log_n, blowup, cols) = (10, 2, 5);
    let n = 1usize << log_n;
    let mut rng = ChaCha8Rng::seed_from_u64(0x4057);
    let row_major = random(&mut rng, n * cols);
    let weights = weights_u64(n, COSET_OFFSET, true);
    let run = |on: bool| {
        with_engine(on, || {
            math_cuda::lde::coset_lde_row_major_with_merkle_tree_keep(
                &row_major,
                None,
                DeviceHash::Rpx256,
                n,
                cols,
                blowup,
                &weights,
                true,
            )
        })
        .expect("base commit with host copy")
    };
    let (on, on_host) = run(true);
    let (off, off_host) = run(false);
    assert_eq!(
        on_host.len(),
        n * blowup * cols,
        "the host copy is produced"
    );
    assert_eq!(on_host, off_host, "identical row-major host LDE");
    assert_eq!(on.tree.unwrap().root, off.tree.unwrap().root);
}

/// Every batched (column-major, in-slab) entry point.
#[test]
fn batched_entry_points_engine_match_legacy() {
    let be = math_cuda::device::backend().expect("cuda backend");
    for (i, &(log_n, blowup, cols)) in SHAPES.iter().enumerate() {
        let cols = cols.min(24);
        let n = 1usize << log_n;
        let lde = n * blowup;
        let mut rng = ChaCha8Rng::seed_from_u64(0xBA7C_0000 + i as u64);
        let what = format!("batched log_n={log_n} blowup={blowup} cols={cols}");
        let w_evals = weights_u64(n, COSET_OFFSET, true);
        let w_coeffs = weights_u64(n, COSET_OFFSET, false);

        // Base columns.
        let base: Vec<Vec<u64>> = (0..cols).map(|_| random(&mut rng, n)).collect();
        let slices: Vec<&[u64]> = base.iter().map(Vec::as_slice).collect();
        let batch = |on: bool| {
            with_engine(on, || {
                math_cuda::lde::coset_lde_batch_base(&slices, blowup, &w_evals)
            })
            .expect("batch base")
        };
        let (a, b) = (batch(true), batch(false));
        for c in 0..cols {
            assert_same_values(&a[c], &b[c], &format!("{what} batch_base col {c}"));
        }
        let into = |on: bool| {
            let mut outs = vec![vec![0u64; lde]; cols];
            {
                let mut views: Vec<&mut [u64]> = outs.iter_mut().map(Vec::as_mut_slice).collect();
                with_engine(on, || {
                    math_cuda::lde::coset_lde_batch_base_into(&slices, blowup, &w_evals, &mut views)
                })
                .expect("batch base into");
            }
            outs
        };
        assert_eq!(
            into(true).iter().map(|c| canon(c)).collect::<Vec<_>>(),
            into(false).iter().map(|c| canon(c)).collect::<Vec<_>>(),
            "{what}: batch_base_into"
        );
        let hash = HASHES[i % HASHES.len()];
        let leaf_hash = |on: bool| {
            let mut outs = vec![vec![0u64; lde]; cols];
            let mut leaves = vec![0u8; (lde / 2) * 32];
            {
                let mut views: Vec<&mut [u64]> = outs.iter_mut().map(Vec::as_mut_slice).collect();
                with_engine(on, || {
                    math_cuda::lde::coset_lde_batch_base_into_with_leaf_hash(
                        &slices,
                        hash,
                        blowup,
                        &w_evals,
                        &mut views,
                        &mut leaves,
                    )
                })
                .expect("batch base leaf hash");
            }
            (outs.iter().map(|c| canon(c)).collect::<Vec<_>>(), leaves)
        };
        assert_eq!(
            leaf_hash(true),
            leaf_hash(false),
            "{what}: leaf hash {hash:?}"
        );

        // Ext3 columns (interleaved), their slab-resident form, and the
        // coefficient form.
        let ext: Vec<Vec<u64>> = (0..cols).map(|_| random(&mut rng, 3 * n)).collect();
        let ext_slices: Vec<&[u64]> = ext.iter().map(Vec::as_slice).collect();
        let ext3_into = |on: bool, coeffs: bool| {
            let mut outs = vec![vec![0u64; 3 * lde]; cols];
            {
                let mut views: Vec<&mut [u64]> = outs.iter_mut().map(Vec::as_mut_slice).collect();
                with_engine(on, || {
                    if coeffs {
                        math_cuda::lde::evaluate_poly_coset_batch_ext3_into(
                            &ext_slices,
                            n,
                            blowup,
                            &w_coeffs,
                            &mut views,
                        )
                    } else {
                        math_cuda::lde::coset_lde_batch_ext3_into(
                            &ext_slices,
                            n,
                            blowup,
                            &w_evals,
                            &mut views,
                        )
                    }
                })
                .expect("ext3 into");
            }
            outs.iter().map(|c| canon(c)).collect::<Vec<_>>()
        };
        assert_eq!(
            ext3_into(true, false),
            ext3_into(false, false),
            "{what}: ext3_into"
        );
        assert_eq!(
            ext3_into(true, true),
            ext3_into(false, true),
            "{what}: evaluate_poly ext3 (coefficients)"
        );
        let comp_tree = |on: bool| {
            let mut outs = vec![vec![0u64; 3 * lde]; cols];
            let mut nodes = vec![0u8; (lde - 1) * 32];
            {
                let mut views: Vec<&mut [u64]> = outs.iter_mut().map(Vec::as_mut_slice).collect();
                with_engine(on, || {
                    math_cuda::lde::evaluate_poly_coset_batch_ext3_into_with_merkle_tree(
                        &ext_slices,
                        hash,
                        n,
                        blowup,
                        &w_coeffs,
                        &mut views,
                        &mut nodes,
                    )
                })
                .expect("evaluate with tree");
            }
            (outs.iter().map(|c| canon(c)).collect::<Vec<_>>(), nodes)
        };
        assert_eq!(
            comp_tree(true),
            comp_tree(false),
            "{what}: evaluate_poly tree {hash:?}"
        );

        let slabs_keep = |on: bool| {
            let stream = be.next_stream();
            let mut host = vec![0u64; 3 * cols * lde];
            for s in 0..3 * cols {
                for r in 0..n {
                    host[s * lde + r] = ext[s / 3][r * 3 + s % 3];
                }
            }
            let buf = stream.clone_htod(&host).expect("slab upload");
            let handle = with_engine(on, || {
                math_cuda::lde::coset_lde_batch_ext3_slabs_keep(
                    &stream, buf, cols, n, blowup, &w_evals, None,
                )
            })
            .expect("slabs keep");
            download(&handle.buf, handle.ready.as_deref())
        };
        assert_same_values(
            &slabs_keep(true),
            &slabs_keep(false),
            &format!("{what}: slabs_keep"),
        );
    }
}

/// The engine against the CPU LDE, independently of the legacy GPU path.
#[test]
fn engine_matches_the_cpu_lde() {
    for &(log_n, blowup) in &[(4usize, 2usize), (7, 8), (11, 4), (14, 2), (18, 4)] {
        let n = 1usize << log_n;
        let mut rng = ChaCha8Rng::seed_from_u64(0xC9A0 + log_n as u64);
        let evals = random(&mut rng, n);
        let weights = weights_u64(n, COSET_OFFSET, true);
        let gpu = with_engine(true, || {
            math_cuda::lde::coset_lde_batch_base(&[evals.as_slice()], blowup, &weights)
        })
        .expect("engine lde")
        .remove(0);

        let log_lde = (n * blowup).trailing_zeros() as u64;
        let inv_tw = LayerTwiddles::<GoldilocksField>::new_inverse(log_n as u64).unwrap();
        let fwd_tw = LayerTwiddles::<GoldilocksField>::new(log_lde).unwrap();
        let w_fe: Vec<Fp> = weights.iter().map(|&w| Fp::from_raw(w)).collect();
        let mut cpu: Vec<Fp> = evals.iter().map(|&x| Fp::from_raw(x)).collect();
        Polynomial::coset_lde_full_expand::<GoldilocksField>(
            &mut cpu, blowup, &w_fe, &inv_tw, &fwd_tw,
        )
        .expect("cpu lde");
        let cpu: Vec<u64> = cpu.iter().map(|e| *e.value()).collect();
        assert_same_values(&gpu, &cpu, &format!("cpu log_n={log_n} blowup={blowup}"));
    }
}

/// GAP K1B: the WHIR encoding (spread + NTT of one codeword) through the column
/// engine against the legacy `lift_spread` + `run_ntt_body`: the codeword
/// (canonically) and the Merkle nodes (bytes) are identical. `(1, 2)` is below
/// the engine's smallest transform and must stay on the legacy path.
#[test]
fn whir_codeword_k1b_matches_legacy() {
    let shapes: &[(usize, usize, usize)] = &[
        // (log_evals, log_blowup, log_folding)
        (1, 2, 1),
        (2, 2, 1),
        (4, 0, 2),
        (6, 2, 4),
        (8, 4, 4),
        (10, 1, 4),
        (12, 2, 4),
        (16, 3, 4),
        (18, 2, 4),
        (21, 2, 4),
    ];
    for (i, &(log_evals, log_blowup, log_folding)) in shapes.iter().enumerate() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x3171_0000 + i as u64);
        // The commit takes canonical hypercube values.
        let evals = canon(&random(&mut rng, 1usize << log_evals));
        let hash = [DeviceHash::Rpx256, DeviceHash::Keccak256][i % 2];
        let what = format!("whir log_evals={log_evals} log_blowup={log_blowup} {hash:?}");
        let before = k1b_codewords();
        let (k1b_vals, k1b_nodes) = with_k1b(true, || {
            math_cuda::whir::commit_codeword_to_host(&evals, log_blowup, log_folding, hash)
        })
        .expect("K1B commit");
        let routed = k1b_codewords() > before;
        let supported = spread_supports((log_evals + log_blowup) as u32, log_blowup as u32);
        // Other tests may encode concurrently, so only the supported shapes'
        // "it routed" is certain; an unsupported one is checked by its values.
        if supported {
            assert!(routed, "{what}: the knob did not route through the engine");
        }
        let (vals, nodes) = with_k1b(false, || {
            math_cuda::whir::commit_codeword_to_host(&evals, log_blowup, log_folding, hash)
        })
        .expect("legacy commit");
        assert_same_values(&k1b_vals, &vals, &what);
        assert_eq!(k1b_nodes, nodes, "{what}: Merkle nodes");
    }
}
