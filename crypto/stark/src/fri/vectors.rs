//! The S3 test vectors the host prover exports ((a)–(d) in the README) for the
//! device prover and the in-guest verifier, checked in under
//! `crypto/stark/tests/vectors/zf_fri/` (see the README there).
//!
//! Compiled only for tests and the `test-utils` feature. Everything here is
//! deterministic: the KAT inputs come from [`splitmix64`], proofs are made at
//! `grinding_factor = 0`. `tests::zf_fri_vectors` (Keccak, Blake3) and the
//! prover crate's `tests::zf_rpx_vectors` (RPX) regenerate every file in memory
//! and require it byte-equal to the checked-in one.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::string::String;
use std::vec::Vec;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::merkle_tree::cap::CapPolicy;
use crypto::merkle_tree::traits::IsStreamingLeafBackend;
use math::fft::bit_reversing::reverse_index;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsFFTField;

use crate::config::StarkHash;
use crate::examples::read_only_memory_logup::{
    LogReadOnlyPublicInputs, LogReadOnlyRAP, read_only_logup_trace,
};
use crate::fri::capture::{FriCapture, capture};
use crate::fri::fri_functions::compute_coset_twiddles_inv;
use crate::fri::group::{group_fold, roots_of_unity_table};
use crate::fri::schedule::{
    FRI_COST_WEIGHTS, FRI_SCHEDULE_DMAX, fri_chain_start, fri_schedule_with_cost,
};
use crate::fri::terminal::FriFoldLayout;
use crate::proof::options::{FriMode, FriScheduleOverride, ProofFormat, ProofOptions};
use crate::prover::{GenericProver, IsStarkProver};
use crate::trace::TraceTable;
use crate::traits::AIR;
use crate::verifier::{GenericVerifier, IsStarkVerifier};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type Felt = FieldElement<F>;
type Ext = FieldElement<E>;

/// The vectors directory: `crypto/stark/tests/vectors/zf_fri`.
pub fn vectors_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/zf_fri")
}

/// One vector file: its name in [`vectors_dir`] and its exact bytes.
pub struct VectorFile {
    pub name: String,
    pub bytes: Vec<u8>,
}

/// Compare `files` with the checked-in ones (byte equality), or write them
/// when `write` is set. Returns the names that differ or are missing.
pub fn check_or_write(files: &[VectorFile], write: bool) -> Vec<String> {
    let dir = vectors_dir();
    let mut bad = Vec::new();
    for f in files {
        let path = dir.join(&f.name);
        if write {
            std::fs::create_dir_all(&dir).expect("create the vectors directory");
            std::fs::write(&path, &f.bytes).expect("write a vector file");
        } else if std::fs::read(&path).ok().as_deref() != Some(f.bytes.as_slice()) {
            bad.push(f.name.clone());
        }
    }
    bad
}

/// SplitMix64: the KAT input generator (stated in the README so any consumer can
/// regenerate the inputs without this crate).
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// An ext3 element from three SplitMix64 outputs, each reduced mod p.
pub(crate) fn next_ext(state: &mut u64) -> Ext {
    Ext::new([
        Felt::from(splitmix64(state)),
        Felt::from(splitmix64(state)),
        Felt::from(splitmix64(state)),
    ])
}

fn limbs(e: &Ext) -> [u64; 3] {
    let v = e.value();
    [v[0].canonical(), v[1].canonical(), v[2].canonical()]
}

fn ext_json(e: &Ext) -> String {
    let [a, b, c] = limbs(e);
    format!("[{a},{b},{c}]")
}

fn exts_json(v: &[Ext]) -> String {
    let items: Vec<String> = v.iter().map(ext_json).collect();
    format!("[{}]", items.join(","))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// (a) schedules
// ---------------------------------------------------------------------------

/// (a) The production fold schedules: T ∈ {4, 9, 10}, B = 6..=24, Q ∈ {3, 110},
/// cap off / auto, the S3 chain (from B − 1) and the S2 chain (from B), each
/// with its cost-law cost (Q × ns) — the DP's output as the format constant it is.
pub fn schedules_json() -> VectorFile {
    let mut s = String::from("{\n  \"generator\": \"stark::fri::vectors::schedules_json\",\n");
    let w = FRI_COST_WEIGHTS;
    let _ = writeln!(
        s,
        "  \"weights_ns\": {{\"compress\": {}, \"select\": {}, \"unpack\": {}, \"hint\": {}, \"compare\": {}, \"fold\": {}, \"twiddle\": {}, \"xalu\": {}, \"balu\": {}}},",
        w.cap.compress,
        w.cap.select,
        w.cap.unpack,
        w.cap.hint,
        w.cap.compare,
        w.fold,
        w.twiddle,
        w.xalu,
        w.balu
    );
    let _ = writeln!(s, "  \"dmax\": {FRI_SCHEDULE_DMAX},");
    s.push_str("  \"rows\": [\n");
    let mut rows = Vec::new();
    for t in [4u32, 9, 10] {
        for q in [3u64, 110] {
            for (cap_name, cap) in [("off", CapPolicy::Off), ("auto", CapPolicy::Auto)] {
                for b in 6..=24u32 {
                    for (chain, one_row) in [("s3", false), ("s2", true)] {
                        let b0 = fri_chain_start(b, one_row);
                        let c = fri_schedule_with_cost(b0, t, q, cap, FRI_SCHEDULE_DMAX);
                        rows.push(format!(
                            "    {{\"terminal_log\": {t}, \"queries\": {q}, \"cap\": \"{cap_name}\", \"lde_log\": {b}, \"chain\": \"{chain}\", \"b0\": {b0}, \"schedule\": {:?}, \"cost_q_ns\": {}}}",
                            c.schedule, c.cost_q
                        ));
                    }
                }
            }
        }
    }
    s.push_str(&rows.join(",\n"));
    s.push_str("\n  ]\n}\n");
    VectorFile {
        name: "a_schedules.json".into(),
        bytes: s.into_bytes(),
    }
}

// ---------------------------------------------------------------------------
// (b) group-fold KATs
// ---------------------------------------------------------------------------

/// The KAT codeword: `2^KAT_LOG` ext3 values from SplitMix64 seed
/// [`KAT_SEED`] (value i = three consecutive outputs), read as a bit-reversed
/// layer on the coset `3·⟨ω_{2^KAT_LOG}⟩`.
pub const KAT_LOG: u32 = 7;
pub const KAT_SEED: u64 = 0x5a46_4652_4933;

pub fn kat_codeword() -> Vec<Ext> {
    let mut st = KAT_SEED;
    (0..1usize << KAT_LOG).map(|_| next_ext(&mut st)).collect()
}

/// ζ of the fold KAT for exponent `d`: SplitMix64 seeded `KAT_SEED + d`.
pub fn kat_zeta(d: u32) -> Ext {
    let mut st = KAT_SEED + u64::from(d);
    next_ext(&mut st)
}

/// (b) For d = 1..=6: the KAT codeword folded d times with ζ, ζ², … (the
/// prover's commit loop), and the verifier's group fold of every group from
/// its slot-0 point (equal by construction; both listed so a device kernel
/// can be checked against either).
pub fn group_fold_json() -> VectorFile {
    let o = Felt::from(3u64);
    let n = 1usize << KAT_LOG;
    let cw = kat_codeword();
    let w = F::get_primitive_root_of_unity(u64::from(KAT_LOG)).expect("root");
    let mut s = String::from("{\n  \"generator\": \"stark::fri::vectors::group_fold_json\",\n");
    let _ = writeln!(
        s,
        "  \"layer_log\": {KAT_LOG},\n  \"coset_offset\": 3,\n  \"codeword\": {},",
        exts_json(&cw)
    );
    s.push_str("  \"folds\": [\n");
    let mut items = Vec::new();
    for d in 1..=6u32 {
        let zeta = kat_zeta(d);
        let mut folded = cw.clone();
        let mut tw = compute_coset_twiddles_inv::<F>(&o, n);
        crate::fri::fold_times(&mut folded, &zeta, d, &mut tw);
        let roots = roots_of_unity_table::<F>(d).expect("table");
        let by_group: Vec<Ext> = (0..n >> d)
            .map(|g| {
                // slot 0: y = x_g, so x_g⁻¹ = y⁻¹.
                let y = &o * w.pow(reverse_index(g << d, n as u64) as u64);
                group_fold::<F, E>(
                    &cw[g << d..(g + 1) << d],
                    &zeta,
                    &y.inv().expect("nonzero"),
                    &roots,
                )
            })
            .collect();
        assert_eq!(
            folded, by_group,
            "the prover's folds and the group fold agree"
        );
        items.push(format!(
            "    {{\"d\": {d}, \"zeta\": {}, \"folded\": {}}}",
            ext_json(&zeta),
            exts_json(&folded)
        ));
    }
    s.push_str(&items.join(",\n"));
    s.push_str("\n  ]\n}\n");
    VectorFile {
        name: "b_group_folds.json".into(),
        bytes: s.into_bytes(),
    }
}

// ---------------------------------------------------------------------------
// (c) group-leaf digests
// ---------------------------------------------------------------------------

/// (c) Under hash `H` (named `hash_name`), for d = 1..=6: the leaf digest of
/// the KAT codeword's first group (`H::Batched` over its 2^d values) and the
/// root of the whole KAT codeword committed as a group-leaf layer tree.
pub fn leaf_digests_json<H: StarkHash>(hash_name: &str) -> VectorFile {
    let cw = kat_codeword();
    let mut s = format!(
        "{{\n  \"generator\": \"stark::fri::vectors::leaf_digests_json\",\n  \"hash\": \"{hash_name}\",\n  \"codeword\": \"b_group_folds.json codeword\",\n  \"leaves\": [\n"
    );
    let mut items = Vec::new();
    for d in 1..=6u32 {
        let n = 1usize << d;
        let leaf =
            <H::Batched<E> as IsStreamingLeafBackend<E>>::hash_data_from_slices(&cw[..n], &[]);
        let tree = crate::fri::group_tree::<E, H>(&cw, n).expect("tree");
        items.push(format!(
            "    {{\"d\": {d}, \"first_leaf\": \"{}\", \"layer_root\": \"{}\"}}",
            hex(&leaf),
            hex(&tree.root)
        ));
    }
    s.push_str(&items.join(",\n"));
    s.push_str("\n  ]\n}\n");
    VectorFile {
        name: format!("c_leaf_digests_{hash_name}.json"),
        bytes: s.into_bytes(),
    }
}

// ---------------------------------------------------------------------------
// (d) small proofs per format
// ---------------------------------------------------------------------------

/// The (d) proof shape: `LogReadOnlyRAP` (ext3, one aux column), 2^10 rows,
/// blowup 4 (B = 12), k = 2, grinding 0, coset offset 3; Q = 3, or
/// [`CAPPED_QUERIES`] for the capped formats.
pub const PROOF_ROWS: usize = 1 << 10;

/// The query count of the capped (d) formats: the `auto` cap policy caps a
/// tree opened at least 20 times at height 3, so a Q = 3 proof
/// carries no cap at all.
pub const CAPPED_QUERIES: usize = 20;

pub fn proof_options(format: ProofFormat, queries: usize) -> ProofOptions {
    ProofOptions {
        blowup_factor: 4,
        fri_number_of_queries: queries,
        coset_offset: 3,
        grinding_factor: 0,
        fri_final_poly_log_degree: 2,
        format,
    }
}

/// The formats of (d), with their query counts: `pair` (today), `dp` (the
/// DP's schedule) and `dp_3_1_3` (an explicit uneven schedule, to catch
/// fold-count bugs), all at Q = 3; and `cap_pair` / `cap_dp` (the `auto` Merkle
/// cap on every tree, with today's FRI and with the DP's schedule) at
/// Q = [`CAPPED_QUERIES`] — the combined S1 × S3 vector.
pub fn proof_formats() -> Vec<(&'static str, ProofFormat, usize)> {
    let dp = ProofFormat {
        fri_mode: FriMode::Dp,
        ..ProofFormat::DEFAULT
    };
    let cap = ProofFormat {
        merkle_cap: CapPolicy::Auto,
        ..ProofFormat::DEFAULT
    };
    vec![
        ("pair", ProofFormat::DEFAULT, 3),
        ("dp", dp, 3),
        (
            "dp_3_1_3",
            ProofFormat {
                fri_schedule_override: FriScheduleOverride::new(&[3, 1, 3]),
                ..dp
            },
            3,
        ),
        ("cap_pair", cap, CAPPED_QUERIES),
        (
            "cap_dp",
            ProofFormat {
                fri_mode: FriMode::Dp,
                ..cap
            },
            CAPPED_QUERIES,
        ),
    ]
}

fn logup_case(
    format: ProofFormat,
    queries: usize,
) -> (
    LogReadOnlyRAP<F, E>,
    TraceTable<F, E>,
    LogReadOnlyPublicInputs<F>,
) {
    let rows = PROOF_ROWS;
    let addr: Vec<Felt> = (0..rows).map(|i| Felt::from((i % 5) as u64 + 1)).collect();
    let val: Vec<Felt> = (0..rows)
        .map(|i| Felt::from(((i % 5) as u64 + 1) * 10))
        .collect();
    let trace: TraceTable<F, E> = read_only_logup_trace(addr, val);
    let cols = trace.columns_main();
    let pi = LogReadOnlyPublicInputs {
        a0: cols[0][0],
        v0: cols[1][0],
        a_sorted_0: cols[2][0],
        v_sorted_0: cols[3][0],
        m0: cols[4][0],
    };
    (
        LogReadOnlyRAP::<F, E>::new(&proof_options(format, queries)),
        trace,
        pi,
    )
}

/// (d) Under hash `H`: per format, the proof's rkyv bytes (`.rkyv`) and a JSON
/// with everything a verifier derives from it — layout, ζ, ι, DEEP values,
/// roots, terminal coefficients, and per query per layer the leaf, slot and
/// opened values.
pub fn proof_vectors<H: StarkHash>(hash_name: &str) -> Vec<VectorFile> {
    let mut out = Vec::new();
    for (fmt_name, format, queries) in proof_formats() {
        out.extend(proof_files::<H>(
            hash_name, fmt_name, format, queries, "d_proof",
        ));
    }
    out
}

/// The formats of (e) (S2): one-row openings with the pair schedule
/// (`one_row_pair`: all-ones groups, the input tree committed in pairs) and
/// with an explicit uneven schedule from the input tree (`one_row_3_2_1_2`,
/// `Σ = 8 = B − T`).
pub fn one_row_proof_formats() -> Vec<(&'static str, ProofFormat)> {
    let on = ProofFormat {
        one_row: crate::proof::options::OneRowMode::On,
        ..ProofFormat::DEFAULT
    };
    vec![
        ("one_row_pair", on),
        (
            "one_row_3_2_1_2",
            ProofFormat {
                fri_mode: FriMode::Dp,
                fri_schedule_override: FriScheduleOverride::new(&[3, 2, 1, 2]),
                ..on
            },
        ),
    ]
}

/// (e) The S2 proofs under hash `H`: the (d) shape (at `Q = 3`) proved with one-row
/// openings. Per query the JSON adds the trace leaf (`r`, a leaf index over
/// the whole LDE) and its path length (`log2(lde)`); `deep` is DEEP at the
/// ONE point `x_r` (there is no `deep_sym`), and layer 0 is the input tree
/// (the DEEP codeword itself; its root is `fri_roots[0]`, absorbed before the
/// first ζ, so `zetas` has one entry per layer).
pub fn one_row_proof_vectors<H: StarkHash>(hash_name: &str) -> Vec<VectorFile> {
    let mut out = Vec::new();
    for (fmt_name, format) in one_row_proof_formats() {
        out.extend(proof_files::<H>(hash_name, fmt_name, format, 3, "e_proof"));
    }
    out
}

/// (e) One-row trace-leaf digests under hash `H`: a KAT base matrix (16 rows ×
/// 5 columns) and ext3 matrix (16 rows × 2 columns) from SplitMix64 seed
/// [`KAT_SEED`] + 100 / + 200, read as bit-reversed LDE columns, committed
/// with one row per leaf AND with row pairs (today's): every leaf digest and
/// the root of each. One row: leaf `i` = the row at bit-reversed position `i`
/// (columns in order, big-endian bytes / the `Batched` felt stream); row pair:
/// rows `2i`, `2i + 1`.
pub fn one_row_leaf_digests_json<H: StarkHash>(hash_name: &str) -> VectorFile {
    const ROWS: usize = 16;
    let mut st = KAT_SEED + 100;
    let base: Vec<Vec<Felt>> = (0..5)
        .map(|_| (0..ROWS).map(|_| Felt::from(splitmix64(&mut st))).collect())
        .collect();
    let mut st = KAT_SEED + 200;
    let ext: Vec<Vec<Ext>> = (0..2)
        .map(|_| (0..ROWS).map(|_| next_ext(&mut st)).collect())
        .collect();
    let mut s = format!(
        "{{\n  \"generator\": \"stark::fri::vectors::one_row_leaf_digests_json\",\n  \"hash\": \"{hash_name}\",\n  \"rows\": {ROWS},\n"
    );
    let base_json: Vec<String> = base
        .iter()
        .map(|c| {
            let v: Vec<String> = c.iter().map(|x| x.canonical().to_string()).collect();
            format!("[{}]", v.join(","))
        })
        .collect();
    let _ = writeln!(s, "  \"base_columns\": [{}],", base_json.join(","));
    let ext_cols: Vec<String> = ext.iter().map(|c| exts_json(c)).collect();
    let _ = writeln!(s, "  \"ext_columns\": [{}],", ext_cols.join(","));
    let mut items = Vec::new();
    for (layout_name, rows_per_leaf) in [("row", 1usize), ("row_pair", 2)] {
        let b = crate::commitment::leaves_bit_reversed_grouped::<F, H::Batched<F>>(
            &base,
            rows_per_leaf,
        );
        let (_, b_root) =
            crate::commitment::commit_bit_reversed_with::<F, H::Batched<F>>(&base, rows_per_leaf)
                .expect("base tree");
        let e =
            crate::commitment::leaves_bit_reversed_grouped::<E, H::Batched<E>>(&ext, rows_per_leaf);
        let (_, e_root) =
            crate::commitment::commit_bit_reversed_with::<E, H::Batched<E>>(&ext, rows_per_leaf)
                .expect("ext tree");
        let hexes = |v: &[crate::config::Commitment]| {
            let h: Vec<String> = v.iter().map(|x| format!("\"{}\"", hex(x))).collect();
            format!("[{}]", h.join(","))
        };
        items.push(format!(
            "    {{\"layout\": \"{layout_name}\", \"rows_per_leaf\": {rows_per_leaf}, \"base_leaves\": {}, \"base_root\": \"{}\", \"ext_leaves\": {}, \"ext_root\": \"{}\"}}",
            hexes(&b),
            hex(&b_root),
            hexes(&e),
            hex(&e_root)
        ));
    }
    s.push_str("  \"layouts\": [\n");
    s.push_str(&items.join(",\n"));
    s.push_str("\n  ]\n}\n");
    VectorFile {
        name: format!("e_leaf_digests_{hash_name}.json"),
        bytes: s.into_bytes(),
    }
}

/// One proof's `{prefix}_{hash}_{format}.{json,rkyv}` pair (the (d) and (e)
/// files). The table's leaf layout is resolved as the prover and verifier
/// resolve it; the JSON keeps the (d) schema for row pairs byte for byte and
/// adds the one-row fields otherwise.
fn proof_files<H: StarkHash>(
    hash_name: &str,
    fmt_name: &str,
    format: ProofFormat,
    queries: usize,
    prefix: &str,
) -> Vec<VectorFile> {
    let mut out = Vec::new();
    {
        let (air, mut trace, pi) = logup_case(format, queries);
        let one_row = crate::leaf_layout::table_leaf_layout(&air, PROOF_ROWS).is_one_row();
        let proof = GenericProver::<F, E, _, H>::prove(
            &air,
            &mut trace,
            &pi,
            &mut DefaultTranscript::<E>::new(&[]),
        )
        .expect("proving must succeed");
        let (ok, records) = capture(|| {
            GenericVerifier::<F, E, _, H>::verify(
                &proof,
                &air,
                &mut DefaultTranscript::<E>::new(&[]),
            )
        });
        assert!(ok, "the {prefix} proof must verify");
        let rec = FriCapture::<E>::from_any(records[0].as_ref()).expect("one ext3 record");
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
            .expect("rkyv")
            .to_vec();
        let lde_log = PROOF_ROWS.trailing_zeros() + 2;
        let layout =
            FriFoldLayout::for_options(lde_log, 2, air.options(), one_row).expect("layout");
        let stem = format!("{prefix}_{hash_name}_{fmt_name}");

        let mut s = format!(
            "{{\n  \"generator\": \"stark::fri::vectors::proof_vectors\",\n  \"hash\": \"{hash_name}\",\n  \"format\": \"{fmt_name}\",\n  \"proof_rkyv\": \"{stem}.rkyv\",\n  \"proof_rkyv_len\": {},\n",
            bytes.len()
        );
        let _ = writeln!(
            s,
            "  \"air\": \"LogReadOnlyRAP<Goldilocks, Goldilocks^3>, reads (i % 5 + 1, 10·(i % 5 + 1))\",\n  \"trace_rows\": {PROOF_ROWS},\n  \"lde_log\": {lde_log},\n  \"blowup\": 4,\n  \"fri_final_poly_log_degree\": 2,\n  \"queries\": {queries},\n  \"grinding_factor\": 0,\n  \"coset_offset\": 3,"
        );
        if !format.merkle_cap.is_off() {
            // The capped formats only (the Q = 3 files are unchanged): the
            // policy and every tree's height, from the verifier's own
            // `StarkCaps`. Each capped tree's cap rides at the end of query
            // 0's path (the owner path), so that `path_len` is `D − c + 2^c`.
            let caps = crate::merkle_caps::StarkCaps::for_options(
                air.options(),
                lde_log as usize,
                one_row,
            )
            .expect("caps");
            let _ = writeln!(
                s,
                "  \"merkle_cap\": \"{}\",\n  \"trace_tree_depth\": {},\n  \"trace_cap\": {},\n  \"fri_tree_depths\": {:?},\n  \"fri_caps\": {:?},",
                format.merkle_cap, caps.trace_depth, caps.trace, caps.fri_depths, caps.fri
            );
        }
        if one_row {
            let _ = writeln!(
                s,
                "  \"one_row\": true,\n  \"query_bound\": {},\n  \"trace_tree_depth\": {lde_log},",
                1u64 << lde_log
            );
        }
        let _ = writeln!(
            s,
            "  \"legacy_encoding\": {},\n  \"total_folds\": {},\n  \"terminal_len\": {},\n  \"schedule\": {:?},",
            layout.is_legacy(),
            layout.total_folds,
            layout.terminal_len,
            layout.schedule
        );
        let roots: Vec<String> = proof
            .fri_layers_merkle_roots
            .iter()
            .map(|r| format!("\"{}\"", hex(r)))
            .collect();
        let _ = writeln!(s, "  \"fri_roots\": [{}],", roots.join(","));
        let _ = writeln!(s, "  \"zetas\": {},", exts_json(&rec.zetas));
        let _ = writeln!(
            s,
            "  \"terminal_coeffs\": {},",
            exts_json(&proof.fri_final_poly_coeffs)
        );
        s.push_str("  \"queries_detail\": [\n");
        let mut qs = Vec::new();
        for (qi, &iota) in rec.iotas.iter().enumerate() {
            let dec = &proof.query_list[qi];
            let mut layers = Vec::new();
            let mut index = iota;
            let mut off = 0usize;
            for (j, &d) in layout.schedule.iter().enumerate() {
                let (leaf, slot, n) = if layout.is_legacy() {
                    (index >> 1, index & 1, 1usize)
                } else {
                    (index >> d, index & ((1 << d) - 1), 1usize << d)
                };
                layers.push(format!(
                    "{{\"layer\": {j}, \"d\": {d}, \"position\": {index}, \"leaf\": {leaf}, \"slot\": {slot}, \"values\": {}, \"path_len\": {}}}",
                    exts_json(&dec.layers_evaluations_sym[off..off + n]),
                    dec.layers_auth_paths[j].merkle_path.len()
                ));
                off += n;
                index = if layout.is_legacy() {
                    index >> 1
                } else {
                    index >> d
                };
            }
            if one_row {
                let opening = &proof.deep_poly_openings[qi];
                qs.push(format!(
                    "    {{\"iota\": {iota}, \"trace_leaf\": {iota}, \"trace_path_len\": {}, \"deep\": {}, \"terminal_position\": {index}, \"layers\": [{}]}}",
                    opening.main_trace_polys.proof.merkle_path.len(),
                    ext_json(&rec.deep[qi]),
                    layers.join(", ")
                ));
            } else {
                qs.push(format!(
                    "    {{\"iota\": {iota}, \"deep\": {}, \"deep_sym\": {}, \"terminal_position\": {index}, \"layers\": [{}]}}",
                    ext_json(&rec.deep[qi]),
                    ext_json(&rec.deep_sym[qi]),
                    layers.join(", ")
                ));
            }
        }
        s.push_str(&qs.join(",\n"));
        s.push_str("\n  ]\n}\n");
        out.push(VectorFile {
            name: format!("{stem}.json"),
            bytes: s.into_bytes(),
        });
        out.push(VectorFile {
            name: format!("{stem}.rkyv"),
            bytes,
        });
    }
    out
}
