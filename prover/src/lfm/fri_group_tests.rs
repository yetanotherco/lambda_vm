//! S3 in the in-guest FRI verifier: the shape from the shared schedule (G1)
//! and the group-layer emitter (G2), design/FRI.md §6, §11.
//!
//! Checked against the host's own artefacts, never against a second model:
//! - the in-guest shape (schedule, layer depths, caps) against the host's
//!   `StarkCaps::for_options` / `FriFormat::schedule` over a sweep of shapes;
//! - the emitted verifier against I-FRI-H's checked-in RPX vectors
//!   (`crypto/stark/tests/vectors/zf_fri/d_proof_rpx_*`: pair, dp, the uneven
//!   `[3, 1, 3]` override, and the two capped Q = 20 formats of REVIEW-FRI F9),
//!   executed, with its permutation count equal to the closed form;
//! - tampers of every value a group opening carries, and the slot check shown
//!   load-bearing (a moved `p₀` executes when, and only when, it is skipped);
//! - the {cap off, auto} × {pair, dp, uneven dp} round-trip matrix on a real
//!   laptop-scale proof (F9), both legs as one program;
//! - RULINGS 13 + 22: every row the emitter emits per FRI layer (group and
//!   pair, capped and uncapped) equals the DP's model (`stark::fri::schedule`),
//!   and a DEEP point's rows equal the S2 `auto` rule's DEEP term.

use crypto::merkle_tree::cap::CapPolicy;
use serde_json::Value;
use stark::examples::read_only_memory_logup::LogReadOnlyPublicInputs;
use stark::fri::schedule::{
    FRI_COST_WEIGHTS, FriLayerRows, fri_group_layer_rows, fri_layer_cost_q, fri_pair_layer_cost_q,
    fri_pair_layer_rows,
};
use stark::leaf_layout::deep_point_xalu_rows;
use stark::merkle_caps::StarkCaps;
use stark::proof::options::{FriMode, FriScheduleOverride, ProofFormat, ProofOptions};
use stark::proof::stark::StarkProof;
use stark::proof::view::StarkProofView;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::builder::Felt;
use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::deep::{DeepInvariants, DeepOpening, DeepShape, emit_deep_point};
use super::executor::execute;
use super::fri::{FriShape, LayerCommitment, LayerOpening, emit_group_layer, emit_pair_layer};
use super::fri_tests::{folding_fixture_with, fri_only_program, host_fri_from, permutations};
use super::instr::Instr;
use super::word::{LfmWord, base_word, ext_word, word_as_ext};

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;
type VectorProof = StarkProof<Gl, Ext3, LogReadOnlyPublicInputs<Gl>>;

// =============================================================================
// G1 — the in-guest shape IS the host's layout
// =============================================================================

/// ★ One schedule, one depth, one cap per layer, on both sides: the in-guest
/// `FriShape` (the emitter's program shape) against the host's `StarkCaps` —
/// the function the prover embeds caps with and the verifier checks them with,
/// itself built on the host's `FriFoldLayout` — over every LDE size of
/// interest, both terminals in production (T = 9 base legs, T = 10 LFM
/// proofs), both FRI modes and both cap policies.
#[test]
fn the_in_guest_fri_shape_is_the_hosts_layout() {
    let mut checked = 0usize;
    for (blowup, k) in [(4u8, 7u8), (4, 8), (2, 7)] {
        for queries in [3usize, 24, 110] {
            for cap in [CapPolicy::Off, CapPolicy::Auto] {
                for fri in [FriMode::Pair, FriMode::Dp] {
                    let blowup_log = (blowup as u32).trailing_zeros();
                    for lde_log in (blowup_log + 1)..=25 {
                        let opts = ProofOptions {
                            blowup_factor: blowup,
                            fri_number_of_queries: queries,
                            coset_offset: 3,
                            grinding_factor: 0,
                            fri_final_poly_log_degree: k,
                            format: ProofFormat {
                                merkle_cap: cap,
                                fri_mode: fri,
                                ..ProofFormat::DEFAULT
                            },
                        };
                        let shape = FriShape::from_options(&opts, lde_log);
                        shape.check();
                        let host = StarkCaps::for_options(&opts, lde_log as usize, false)
                            .expect("a row-pair format lays out");
                        let depths: Vec<usize> = (0..shape.num_committed())
                            .map(|j| shape.layer_depth(j))
                            .collect();
                        let caps: Vec<usize> = (0..shape.num_committed())
                            .map(|j| shape.layer_cap(j))
                            .collect();
                        assert_eq!(depths, host.fri_depths, "{opts:?} lde {lde_log}");
                        assert_eq!(caps, host.fri, "{opts:?} lde {lde_log}");
                        assert_eq!(shape.index_bits(), host.trace_depth);
                        assert_eq!(shape.is_legacy(), fri == FriMode::Pair);
                        checked += 1;
                    }
                }
            }
        }
    }
    println!("{checked} shapes: in-guest schedule, depths and caps == the host's");
}

// =============================================================================
// G2 — the emitted verifier on I-FRI-H's RPX vectors
// =============================================================================

fn ext_of(v: &Value) -> FEE {
    let limbs: Vec<u64> = v
        .as_array()
        .expect("an ext value is three limbs")
        .iter()
        .map(|x| x.as_u64().expect("a canonical limb"))
        .collect();
    assert_eq!(limbs.len(), 3);
    FEE::new([FE::from(limbs[0]), FE::from(limbs[1]), FE::from(limbs[2])])
}

/// One checked-in RPX (d) vector: its JSON, its proof, and the in-guest shape
/// the emitter builds for it from the vector's FORMAT (the host generator's
/// own `proof_formats`, never re-spelled here).
struct Vector {
    name: &'static str,
    json: Value,
    proof: VectorProof,
    shape: FriShape,
}

fn rpx_vectors() -> Vec<Vector> {
    stark::fri::vectors::proof_formats()
        .into_iter()
        .map(|(name, format, queries)| {
            let dir = stark::fri::vectors::vectors_dir();
            let stem = format!("d_proof_rpx_{name}");
            let json: Value = serde_json::from_slice(
                &std::fs::read(dir.join(format!("{stem}.json"))).expect("the vector JSON"),
            )
            .expect("valid JSON");
            let bytes = std::fs::read(dir.join(format!("{stem}.rkyv"))).expect("the vector proof");
            let proof: VectorProof =
                rkyv::from_bytes::<VectorProof, rkyv::rancor::Error>(&bytes).expect("rkyv");
            let opts = stark::fri::vectors::proof_options(format, queries);
            let lde_log = json["lde_log"].as_u64().expect("lde_log") as u32;
            let shape = FriShape::from_options(&opts, lde_log);
            Vector {
                name,
                json,
                proof,
                shape,
            }
        })
        .collect()
}

impl Vector {
    /// The arenas [`fri_only_program`] declares: `(ι, p₀(υ), p₀(−υ))` per
    /// query, then the roots, the ζs, the terminal coefficients, the per-query
    /// layer openings and (when capped) the caps.
    fn arenas(&self) -> Vec<Vec<LfmWord>> {
        let queries = self.json["queries_detail"].as_array().expect("queries");
        let mut deep = Vec::new();
        for q in queries {
            deep.push(base_word(FE::from(q["iota"].as_u64().expect("iota"))));
            deep.push(ext_word(&ext_of(&q["deep"])));
            deep.push(ext_word(&ext_of(&q["deep_sym"])));
        }
        let view = StarkProofView::Owned(&self.proof);
        let (openings, caps) = super::epoch_verify_tests::fri_layer_openings(view, self.shape);
        let mut per_query = Vec::new();
        for query in &openings {
            for (values, path) in query {
                per_query.extend(values.iter().map(ext_word));
                per_query.extend(super::proof_arena::commitments_to_arena(path));
            }
        }
        let zetas: Vec<LfmWord> = self.json["zetas"]
            .as_array()
            .expect("zetas")
            .iter()
            .map(|z| ext_word(&ext_of(z)))
            .collect();
        let mut out = vec![
            deep,
            super::proof_arena::commitments_to_arena(&self.proof.fri_layers_merkle_roots),
            zetas,
            self.proof
                .fri_final_poly_coeffs
                .iter()
                .map(ext_word)
                .collect(),
            per_query,
        ];
        if self.shape.cap_words(super::proof_arena::words_per_root()) > 0 {
            out.push(super::proof_arena::commitments_to_arena(&caps));
        }
        out
    }

    fn program(&self) -> LfmProgram {
        fri_only_program(self.shape, self.shape.num_queries)
    }
}

/// ★ The emitted FRI verifier accepts every RPX (d) vector — today's pair
/// proof, the DP schedule, the uneven `[3, 1, 3]` override (the only shape
/// that catches a fold-count off-by-one, REVIEW-FRI F6) and both capped Q = 20
/// formats (F9) — with the vector's schedule, depths and caps derived by the
/// emitter's own shape, and the permutation count exactly the closed form.
#[test]
fn the_emitted_fri_verifier_accepts_every_rpx_vector() {
    for v in rpx_vectors() {
        let s = v.shape;
        s.check();
        let schedule: Vec<u8> = v.json["schedule"]
            .as_array()
            .expect("schedule")
            .iter()
            .map(|d| d.as_u64().expect("d") as u8)
            .collect();
        assert_eq!(s.schedule(), schedule, "{}: the schedule", v.name);
        assert_eq!(
            s.is_legacy(),
            v.json["legacy_encoding"]
                .as_bool()
                .expect("legacy_encoding"),
            "{}",
            v.name
        );
        if let Some(caps) = v.json.get("fri_caps") {
            let want: Vec<usize> = caps
                .as_array()
                .expect("fri_caps")
                .iter()
                .map(|c| c.as_u64().expect("c") as usize)
                .collect();
            let depths: Vec<usize> = v.json["fri_tree_depths"]
                .as_array()
                .expect("depths")
                .iter()
                .map(|c| c.as_u64().expect("d") as usize)
                .collect();
            assert_eq!(
                (0..s.num_committed())
                    .map(|j| s.layer_cap(j))
                    .collect::<Vec<_>>(),
                want,
                "{}: caps",
                v.name
            );
            assert_eq!(
                (0..s.num_committed())
                    .map(|j| s.layer_depth(j))
                    .collect::<Vec<_>>(),
                depths,
                "{}: depths",
                v.name
            );
        }
        let program = v.program();
        let exec = execute(&program, &v.arenas(), &crate::hash_pin::BLOCK_HASHER)
            .unwrap_or_else(|e| panic!("{}: the honest vector must execute: {e:?}", v.name));
        assert_eq!(exec.public_words.len(), s.num_queries);
        let closed = s.num_queries * s.permutations_per_query() + s.cap_permutations();
        assert_eq!(
            permutations(&program),
            closed,
            "{}: emitted permutations against the closed form",
            v.name
        );
        println!(
            "{:<9} Q={:<2} schedule {:?} caps {:?}: {} permutations, {} instructions",
            v.name,
            s.num_queries,
            s.schedule(),
            (0..s.num_committed())
                .map(|j| s.layer_cap(j))
                .collect::<Vec<_>>(),
            closed,
            program.instrs.len()
        );
    }
}

/// ★ Every value a group opening carries is bound: the slot value, a non-slot
/// value, the last value of the group, a sibling, a cap word, a folding
/// challenge, a terminal coefficient, and the DEEP value the first slot check
/// compares against. Run on the uneven override and on the capped DP vector.
#[test]
fn no_tampered_group_opening_value_can_pass() {
    for v in rpx_vectors() {
        if !matches!(v.name, "dp_3_1_3" | "cap_dp") {
            continue;
        }
        let program = v.program();
        let honest = v.arenas();
        execute(&program, &honest, &crate::hash_pin::BLOCK_HASHER).expect("honest");
        let q0 = &v.json["queries_detail"][0]["layers"][0];
        let slot = q0["slot"].as_u64().expect("slot") as usize;
        let d0 = 1usize << v.shape.layer_fold(0);
        let other = (slot + 1) % d0;
        // Arenas: deep, roots, zetas, coeffs, queries[, caps].
        let mut bump: Vec<(String, usize, usize)> = vec![
            ("p0 (the DEEP value)".into(), 0, 1),
            ("zeta_1".into(), 2, 1),
            ("terminal coefficient 0".into(), 3, 0),
            (format!("query 0 layer 0 slot value (slot {slot})"), 4, slot),
            (format!("query 0 layer 0 non-slot value {other}"), 4, other),
            ("query 0 layer 0 last group value".into(), 4, d0 - 1),
            ("query 0 layer 0 first sibling".into(), 4, d0),
        ];
        if honest.len() == 6 {
            bump.push(("cap word 0".into(), 5, 0));
            bump.push(("last cap word".into(), 5, honest[5].len() - 1));
        }
        for (label, arena, word) in bump {
            let mut bad = honest.clone();
            bad[arena][word][0] += FE::one();
            execute(&program, &bad, &crate::hash_pin::BLOCK_HASHER).expect_err(&format!(
                "{}: moving {label} must make the program unexecutable",
                v.name
            ));
        }
    }
}

/// ★ The slot check is LOAD-BEARING (the in-guest M1). Under the group
/// encoding the value the first fold produces from the DEEP pair meets the
/// committed layers ONLY at the slot check: the leaf hashes the group, the
/// walk authenticates it, the group fold reads it. So a moved `p₀(υ)` is
/// refused with the check and ACCEPTED without it — which is exactly a
/// verifier that would accept FRI for a different codeword than the trace
/// openings commit to.
#[test]
fn the_slot_check_is_load_bearing() {
    let v = rpx_vectors()
        .into_iter()
        .find(|v| v.name == "cap_dp")
        .expect("the capped dp vector");
    let honest = v.arenas();
    let mut moved = honest.clone();
    moved[0][1][0] += FE::one();

    let with = v.program();
    execute(&with, &honest, &crate::hash_pin::BLOCK_HASHER).expect("honest");
    execute(&with, &moved, &crate::hash_pin::BLOCK_HASHER)
        .expect_err("a moved p0 must be refused by the slot check");

    super::fri::SKIP_SLOT_CHECK.with(|c| c.set(true));
    let without = v.program();
    super::fri::SKIP_SLOT_CHECK.with(|c| c.set(false));
    execute(&without, &moved, &crate::hash_pin::BLOCK_HASHER)
        .expect("WITHOUT the slot check a moved p0 is accepted — the check is the only binding");
}

// =============================================================================
// F9 — the {cap} × {fri} round-trip matrix, both legs, on a real proof
// =============================================================================

/// ★ REVIEW-FRI F9's matrix on a real laptop-scale proof (L2G_MEMORY, 2048
/// rows, blowup 2, `k = 2` so the committed chain covers 11 → 3, Q = 24):
/// {cap off, auto} × {pair, dp, dp `[3, 1, 4]`}. Per cell the FRI leg alone
/// and both legs as one program execute over every query, reach the terminal
/// codeword production computed, and emit exactly the closed form.
#[test]
fn the_cap_and_fri_matrix_round_trips_in_guest() {
    use super::epoch_verify::{blocks_for, group_leaf_felts};

    let hash = super::edsl::WrapHash::production();
    for cap in [CapPolicy::Off, CapPolicy::Auto] {
        for (label, fri, over) in [
            ("pair", FriMode::Pair, None),
            ("dp", FriMode::Dp, None),
            ("dp [3,1,4]", FriMode::Dp, Some(&[3u8, 1, 4][..])),
        ] {
            let mut opts = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(2)
                .expect("blowup 2");
            opts.fri_number_of_queries = 24;
            opts.grinding_factor = 0;
            opts.fri_final_poly_log_degree = 2;
            opts.format = ProofFormat {
                merkle_cap: cap,
                fri_mode: fri,
                fri_schedule_override: over.and_then(FriScheduleOverride::new),
                ..ProofFormat::DEFAULT
            };
            let (air, proof) = folding_fixture_with(2048, opts);
            let h = host_fri_from(&*air, &proof);
            let s = h.shape;
            let all: Vec<usize> = (0..h.trace.iotas.len()).collect();
            let codeword = h.terminal_codeword();
            let position = |iota: usize| iota >> (s.total_folds() - 1);

            // The FRI leg alone.
            let program = fri_only_program(s, all.len());
            let exec = execute(
                &program,
                &h.all_arenas(&all),
                &crate::hash_pin::BLOCK_HASHER,
            )
            .unwrap_or_else(|e| panic!("cap={cap} fri={label}: FRI leg: {e:?}"));
            for (k, &q) in all.iter().enumerate() {
                let v = word_as_ext(&exec.public_words[k].1).expect("ext");
                assert_eq!(
                    v,
                    codeword[position(h.trace.iotas[q])],
                    "cap={cap} fri={label}"
                );
            }
            assert_eq!(
                permutations(&program),
                all.len() * s.permutations_per_query() + s.cap_permutations(),
                "cap={cap} fri={label}: FRI leg closed form"
            );

            // Both legs as one program.
            let mut b = LfmBuilder::new().with_wrap_hash(hash);
            let (_, _, terminal) =
                super::fri::emit_sub_proof_with_fri(&mut b, &h.trace.shape, s, all.len());
            for t in &terminal {
                b.public(t.as_cell());
            }
            let joined = compile(b.finish());
            let mut arenas = h.trace.arenas(&all);
            arenas.extend(h.fri_arenas(&all));
            let exec = execute(&joined, &arenas, &crate::hash_pin::BLOCK_HASHER)
                .unwrap_or_else(|e| panic!("cap={cap} fri={label}: joined: {e:?}"));
            for (k, &q) in all.iter().enumerate() {
                let v = word_as_ext(&exec.public_words[k].1).expect("ext");
                assert_eq!(v, codeword[position(h.trace.iotas[q])]);
            }
            let sub = &h.trace.shape;
            let leaves: usize = sub
                .groups()
                .iter()
                .map(|g| blocks_for(group_leaf_felts(g), hash))
                .sum();
            let closed = all.len()
                * (leaves + sub.groups().len() * sub.path_len() + s.permutations_per_query())
                + sub.cap_permutations()
                + s.cap_permutations();
            assert_eq!(
                permutations(&joined),
                closed,
                "cap={cap} fri={label}: both legs' closed form"
            );
            println!(
                "cap={cap:<4} fri={label:<10} schedule {:?} FRI caps {:?} trace cap {}: FRI leg \
                 {} perms, both legs {} perms / {} instructions",
                s.schedule(),
                (0..s.num_committed())
                    .map(|j| s.layer_cap(j))
                    .collect::<Vec<_>>(),
                sub.trace_cap,
                permutations(&program),
                closed,
                joined.instrs.len(),
            );
        }
    }
}

// =============================================================================
// RULINGS 13 + 22 — every emitted row per FRI layer and per DEEP point, against
// the host's cost model (`stark::fri::schedule`, `stark::leaf_layout`)
// =============================================================================

/// The kinds of every instruction a program emits: `(selects, XALU, BALU,
/// hashes, unpacks, packs, hints, other)`.
fn count_kinds(instrs: &[Instr]) -> [usize; 8] {
    let mut k = [0usize; 8];
    for i in instrs {
        let slot = match i {
            Instr::Select { .. } => 0,
            Instr::ExtAlu { .. } => 1,
            Instr::BaseAlu { .. } => 2,
            Instr::Hash { .. } => 3,
            Instr::Unpack { .. } => 4,
            Instr::Pack { .. } => 5,
            Instr::Hint { .. } => 6,
            _ => 7,
        };
        k[slot] += 1;
    }
    k
}

/// The rows `emit(times)` adds per repetition: the program is built at
/// `times = 1` and `times = 2` over the same hinted inputs, and the difference
/// is one repetition's rows — interned program constants and one-time setup
/// (the index decomposition, the root's unpack, the cap's hints and root
/// check) fall out of it. Asserts the repetition emits nothing but the priced
/// row kinds.
fn rows_of_one(emit: &dyn Fn(usize) -> LfmProgram) -> FriLayerRows {
    let (once, twice) = (emit(1), emit(2));
    let (a, b) = (count_kinds(&once.instrs), count_kinds(&twice.instrs));
    let d: Vec<u64> = (0..8).map(|i| (b[i] - a[i]) as u64).collect();
    assert_eq!(d[7], 0, "a repetition emits only priced row kinds");
    assert_eq!(
        (twice.instrs.len() - once.instrs.len()) as u64,
        d.iter().sum::<u64>(),
        "every instruction is counted"
    );
    FriLayerRows {
        selects: d[0],
        xalu: d[1],
        balu: d[2],
        hashes: d[3],
        unpacks: d[4],
        packs: d[5],
        hints: d[6],
    }
}

/// Tree depth of the measured layers.
const MEASURED_DEPTH: usize = 2;

/// A program emitting `times` openings of one committed FRI layer (group
/// layer of exponent `d`, or today's pair layer when `d == 0`) over a
/// `MEASURED_DEPTH`-level tree capped at `c`, every shared input hinted before
/// the first opening. Every opening's values and siblings are hinted in the
/// loop (as `hint_layer_openings` does), so they count.
fn fri_layer_program(d: u32, c: usize, times: usize) -> LfmProgram {
    let pair = d == 0;
    let fold = if pair { 1 } else { d };
    let shape = FriShape {
        log2_lde_length: fold + MEASURED_DEPTH as u32 + 1,
        blowup_log: 1,
        final_poly_log_degree: 1,
        coset_offset: 3,
        num_queries: 1,
        format: if pair {
            ProofFormat::DEFAULT
        } else {
            ProofFormat {
                fri_mode: FriMode::Dp,
                fri_schedule_override: FriScheduleOverride::new(&[d as u8]),
                ..ProofFormat::DEFAULT
            }
        },
    };
    shape.check();
    assert_eq!(shape.is_legacy(), pair);
    assert_eq!(shape.schedule(), vec![fold as u8]);
    assert_eq!(shape.layer_depth(0), MEASURED_DEPTH);

    let n = if pair { 1 } else { 1usize << d };
    let num_siblings = MEASURED_DEPTH - c;
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    assert_eq!(
        super::edsl::digest_words(&b),
        1,
        "the model prices the production one-cell digest"
    );
    let arena = b.declare_arena((4 + fold as usize + (1 << c) + times * (n + num_siblings)) as u32);
    let root = b.hint_word(arena, 0);
    let mut commitment = LayerCommitment::from_lanes(vec![b.unpack(root)]);
    let v = b.hint_word(arena, 1).as_ext();
    let y_inv = b.hint_felt(arena, 2);
    let index = b.hint_felt(arena, 3);
    let bits = b.bit_dec(index, shape.index_bits());
    let zetas: Vec<_> = (0..fold)
        .map(|i| b.hint_word(arena, 4 + i).as_ext())
        .collect();
    let mut at = commitment.hint_cap(&mut b, arena, 4 + fold, c);
    for _ in 0..times {
        let values = (0..n)
            .map(|_| {
                at += 1;
                b.hint_word(arena, at - 1).as_ext()
            })
            .collect();
        let siblings = (0..num_siblings)
            .map(|_| {
                at += 1;
                super::edsl::WrapDigest::from_cell(b.hint_word(arena, at - 1))
            })
            .collect();
        let opening = LayerOpening { values, siblings };
        if pair {
            emit_pair_layer(&mut b, 0, &commitment, zetas[0], v, y_inv, &opening, &bits);
        } else {
            emit_group_layer(
                &mut b,
                shape,
                0,
                &commitment,
                &zetas,
                v,
                y_inv,
                &opening,
                &bits,
            );
        }
    }
    compile(b.finish())
}

/// ★ RULINGS 13 + 22: the rows one query's opening of a committed FRI layer
/// emits in-guest EQUAL the host model's, kind by kind and in total, for group
/// layers `d = 1..=6` and today's pair layer, uncapped and capped (`c = 1, 2`
/// on a two-level tree):
///
/// ```text
///   group d, depth D, cap c (stark::fri::schedule::fri_group_layer_rows):
///     selects  (2^d − 1) slot mux + (D − c) walk + (2^c − 1) cap mux + d x_g
///     XALU     5·(2^d − 1) folds + 2 slot assert + max(0, d − 2) scaling
///     BALU     d twiddles + d x_g + [d ≥ 2] scaling + 8 root compare
///     hashes   leaf(d) + (D − c)
///     unpacks  2^d values + 1 walked digest + [c ≥ 1] cap node
///     packs    ⌈3·2^d / 4⌉ leaf words
///     hints    2^d values + (D − c) siblings
///   pair, depth D, cap c (fri_pair_layer_rows):
///     selects 1 + (D − c) + (2^c − 1), XALU 5, BALU 1 + 8, hashes 1 + (D − c),
///     unpacks 2 + 1 + [c ≥ 1], packs 2, hints 1 + (D − c)
/// ```
///
/// The DP prices exactly these rows: `fri_layer_cost_q` is the uncapped rows
/// minus the cap's gain and the `c` sibling hints it removes, checked here
/// against the capped rows priced directly plus the cap's per-tree cost. A
/// change to the emitter that is not also a change to the model fails here.
#[test]
fn every_emitted_fri_row_is_priced() {
    let w = FRI_COST_WEIGHTS;
    println!("\n  layer c | sel XALU BALU hash unpack pack hint | ns/query");
    for c in 0..=2usize {
        for d in 0..=6u32 {
            let got = rows_of_one(&|times| fri_layer_program(d, c, times));
            let (label, model) = if d == 0 {
                (
                    "pair".to_string(),
                    fri_pair_layer_rows(MEASURED_DEPTH as u32, c as u32),
                )
            } else {
                (
                    format!("d={d}"),
                    fri_group_layer_rows(d, MEASURED_DEPTH as u32, c as u32),
                )
            };
            assert_eq!(got, model, "{label} c={c}: emitted rows == model rows");
            println!(
                "  {label:>5} {c} | {:>3} {:>4} {:>4} {:>4} {:>6} {:>4} {:>4} | {:>8}",
                got.selects,
                got.xalu,
                got.balu,
                got.hashes,
                got.unpacks,
                got.packs,
                got.hints,
                got.price(&w)
            );
        }
    }

    // The DP's per-layer price is these rows: uncapped exactly; capped, the
    // capped rows plus the cap's once-per-tree cost (`cap_gain`'s per-tree
    // term: 2^c − 1 compressions, 2^c hints, one compare).
    let depth = 10u32;
    for q in [1u64, 20, 110] {
        for cap in [
            CapPolicy::Off,
            CapPolicy::Fixed(1),
            CapPolicy::Fixed(3),
            CapPolicy::Auto,
        ] {
            let c = cap.height(q as usize, depth as usize) as u32;
            let per_tree = if c == 0 {
                0
            } else {
                ((1u64 << c) - 1) * w.cap.compress + (1u64 << c) * w.cap.hint + w.cap.compare
            };
            for d in 1..=6u32 {
                let direct = q * fri_group_layer_rows(d, depth, c).price(&w) + per_tree;
                assert_eq!(
                    fri_layer_cost_q(&w, d, depth, q, cap),
                    direct,
                    "group d={d} q={q} cap={cap:?}"
                );
            }
            let direct = q * fri_pair_layer_rows(depth, c).price(&w) + per_tree;
            assert_eq!(
                fri_pair_layer_cost_q(&w, depth, q, cap),
                direct,
                "pair q={q} cap={cap:?}"
            );
        }
    }
}

/// A program emitting `times` DEEP points of `shape` over hinted openings and
/// hinted invariants (the invariants hinted before the first point).
fn deep_point_program(shape: &DeepShape, times: usize) -> LfmProgram {
    let e = shape.num_eval_points;
    let cols = shape.num_total_cols;
    let parts = shape.num_composition_parts;
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena((4 * e + 4 + times * (1 + cols + parts)) as u32);
    let mut at = 0u32;
    let mut next = |b: &mut LfmBuilder| {
        at += 1;
        b.hint_word(arena, at - 1).as_ext()
    };
    let gamma = next(&mut b);
    let inv = DeepInvariants {
        ood_row_sum: (0..e).map(|_| next(&mut b)).collect(),
        h_sum_zpow: next(&mut b),
        z_pow: next(&mut b),
        row_points: (0..e).map(|_| next(&mut b)).collect(),
        gamma_pow_surviving: next(&mut b),
        gamma_pow_block: (0..e).map(|_| next(&mut b)).collect(),
        gamma_stride: (0..e).map(|_| next(&mut b)).collect(),
    };
    for _ in 0..times {
        let point = Felt(next(&mut b).as_cell().0);
        let opening = DeepOpening {
            point,
            trace: (0..cols).map(|_| next(&mut b)).collect(),
            parts: (0..parts).map(|_| next(&mut b)).collect(),
        };
        emit_deep_point(&mut b, shape, gamma, &inv, &opening);
    }
    compile(b.finish())
}

/// ★ RULINGS 22: the XALU rows of ONE in-guest DEEP point EQUAL the S2 `auto`
/// rule's DEEP term (`stark::leaf_layout::deep_point_xalu_rows`:
/// `num_surviving + 4·E + P + 3`), over shapes with and without a next row,
/// a widened step, and one or many composition parts. DEEP emits no other
/// row kind; the point's hinted inputs here stand in for the cells the trace
/// walk already authenticated.
#[test]
fn the_deep_point_rows_are_the_auto_rules_deep_term() {
    let shapes = [
        // (step, offsets, cols, next-row cols, parts)
        (1usize, 1usize, 7usize, vec![], 1usize),
        (1, 2, 5, vec![1, 3], 2),
        (1, 2, 40, vec![0, 5, 39], 3),
        (2, 2, 6, vec![2], 2),
        (1, 3, 9, vec![0, 8], 4),
    ];
    for (step, offsets, cols, next_cols, parts) in shapes {
        let shape = DeepShape {
            step_size: step,
            num_eval_points: offsets * step,
            num_total_cols: cols,
            next_row_cols: next_cols.clone(),
            num_composition_parts: parts,
            log2_trace_length: 8,
        };
        let got = rows_of_one(&|times| deep_point_program(&shape, times));
        let want = deep_point_xalu_rows(
            shape.num_surviving() as u64,
            shape.num_eval_points as u64,
            parts as u64,
        );
        let ctx =
            format!("step {step} offsets {offsets} cols {cols} next {next_cols:?} parts {parts}");
        assert_eq!(got.xalu, want, "{ctx}: DEEP XALU rows");
        assert_eq!(
            (got.selects, got.balu, got.hashes, got.unpacks, got.packs),
            (0, 0, 0, 0, 0),
            "{ctx}: DEEP emits only XALU rows"
        );
        assert_eq!(
            got.hints as usize,
            1 + cols + parts,
            "{ctx}: the stand-in hints"
        );
    }
}
