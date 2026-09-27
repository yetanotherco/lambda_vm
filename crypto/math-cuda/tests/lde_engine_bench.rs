//! LDE microbenchmark driver: the row-major commit's stages under the legacy
//! pipeline and the column-major engine (`math_cuda::lde::lde_bench`), on the
//! W3/K0 shapes. `LDEB_SHAPES="log_n:m:blowup;..."`, `LDEB_HASHES="rpx,..."`,
//! `LDEB_RPL="2,1"`, `LDEB_ITERS=3`, `LDEB_SNAPSHOT=1`. Prints one `LDEB ...`
//! line per timed iteration; both pipelines' `root0` must agree on every shape.
use math_cuda::DeviceHash;
use math_cuda::lde::lde_bench;

#[test]
#[ignore = "measurement arm: seconds to minutes per shape; run on the box"]
fn lde_row_major_commit() {
    let shapes = std::env::var("LDEB_SHAPES").unwrap_or_else(|_| "20:32:2".into());
    let hashes = std::env::var("LDEB_HASHES").unwrap_or_else(|_| "rpx".into());
    let rpls = std::env::var("LDEB_RPL").unwrap_or_else(|_| "2".into());
    let iters: usize = std::env::var("LDEB_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let snapshot = std::env::var("LDEB_SNAPSHOT").map_or(true, |v| v != "0");
    let mut mismatches = 0;
    for shape in shapes.split(';').filter(|s| !s.is_empty()) {
        let p: Vec<usize> = shape.split(':').map(|x| x.parse().unwrap()).collect();
        let (log_n, m, blowup) = (p[0], p[1], p[2]);
        let n = 1usize << log_n;
        let input = lde_bench::upload(n, m).expect("upload");
        for h in hashes.split(',') {
            let hash = match h {
                "rpx" => DeviceHash::Rpx256,
                "blake3" => DeviceHash::Blake3,
                "keccak" => DeviceHash::Keccak256,
                other => panic!("unknown hash {other}"),
            };
            for rpl in rpls.split(',').map(|x| x.parse::<usize>().unwrap()) {
                let mut roots = [None, None];
                for (e, engine) in ["legacy", "column"].into_iter().enumerate() {
                    // One warm-up (module load, pool growth), then `iters` timed.
                    for it in 0..=iters {
                        let t = lde_bench::run(&input, hash, n, m, blowup, rpl, snapshot, e == 1)
                            .expect("run");
                        println!(
                            "LDEB engine={engine} hash={h} log_n={log_n} m={m} blowup={blowup} rpl={rpl} \
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
                    println!(
                        "LDEB ROOT MISMATCH log_n={log_n} m={m} blowup={blowup} rpl={rpl} {h}"
                    );
                }
            }
        }
        drop(input);
    }
    assert_eq!(mismatches, 0, "the pipelines disagree on a root");
}
