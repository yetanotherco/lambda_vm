//! GAP K1 microbenchmark driver: the row-major commit's stages under the
//! legacy engine and the column-major one (`math_cuda::lde::k1_bench`), on the
//! W3/K0 shapes. `K1_SHAPES="log_n:m:blowup;..."`, `K1_HASHES="rpx,..."`,
//! `K1_RPL="2,1"`, `K1_ITERS=3`, `K1_SNAPSHOT=1`. Prints one `K1B ...` line
//! per timed iteration; both engines' `root0` must agree on every shape.
use math_cuda::DeviceHash;
use math_cuda::lde::k1_bench;

#[test]
#[ignore = "measurement arm: seconds to minutes per shape; run on the box"]
fn k1_row_major_commit() {
    let shapes = std::env::var("K1_SHAPES").unwrap_or_else(|_| "20:32:2".into());
    let hashes = std::env::var("K1_HASHES").unwrap_or_else(|_| "rpx".into());
    let rpls = std::env::var("K1_RPL").unwrap_or_else(|_| "2".into());
    let iters: usize = std::env::var("K1_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let snapshot = std::env::var("K1_SNAPSHOT").map_or(true, |v| v != "0");
    let mut mismatches = 0;
    for shape in shapes.split(';').filter(|s| !s.is_empty()) {
        let p: Vec<usize> = shape.split(':').map(|x| x.parse().unwrap()).collect();
        let (log_n, m, blowup) = (p[0], p[1], p[2]);
        let n = 1usize << log_n;
        let input = k1_bench::upload(n, m).expect("upload");
        for h in hashes.split(',') {
            let hash = match h {
                "rpx" => DeviceHash::Rpx256,
                "blake3" => DeviceHash::Blake3,
                "keccak" => DeviceHash::Keccak256,
                other => panic!("unknown hash {other}"),
            };
            for rpl in rpls.split(',').map(|x| x.parse::<usize>().unwrap()) {
                let mut roots = [None, None];
                for (e, engine) in ["legacy", "k1"].into_iter().enumerate() {
                    // One warm-up (module load, pool growth), then `iters` timed.
                    for it in 0..=iters {
                        let t = k1_bench::run(&input, hash, n, m, blowup, rpl, snapshot, e == 1)
                            .expect("run");
                        println!(
                            "K1B engine={engine} hash={h} log_n={log_n} m={m} blowup={blowup} rpl={rpl} \
                             snapshot={} iter={it} expand_ms={:.3} stage_ms={:.3} lde_ms={:.3} \
                             leaves_ms={:.3} inner_ms={:.3} transpose_ms={:.3} root0={:016x}",
                            snapshot as u8,
                            t.expand_ms,
                            t.stage_ms,
                            t.lde_ms,
                            t.leaves_ms,
                            t.inner_ms,
                            t.transpose_ms,
                            t.root0
                        );
                        roots[e] = Some(t.root0);
                    }
                }
                if roots[0] != roots[1] {
                    mismatches += 1;
                    println!("K1B ROOT MISMATCH log_n={log_n} m={m} blowup={blowup} rpl={rpl} {h}");
                }
            }
        }
        drop(input);
    }
    assert_eq!(mismatches, 0, "the engines disagree on a root");
}
