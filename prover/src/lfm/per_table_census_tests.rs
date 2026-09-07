//! ★★ LEVER 0 — the tenant-width census: what an aggregator pays to open a
//! wrap proof whose hash matrix is 316 columns wide instead of 3,056.
//!
//! # The question
//!
//! A per-table aggregator verifies one leg per TABLE of the wrap proof it
//! opens. Every leg's per-query bill is
//!
//! ```text
//! Σ_groups blocks_for(leaf_felts(g))          leaf ABSORPTION — width-driven
//! + num_committed · blocks_for(FRI_LEAF_FELTS) FRI leaf absorption
//! + groups · merkle_depth                      trace-tree parents — one compression each
//! + fri.path_steps_per_query()                 FRI path steps — one compression each
//! ```
//!
//! ([`super::epoch_verify::query_permutations_for`], whose agreement with the
//! EMITTER is already pinned by `epoch_verify_tests::the_assembled_epoch_verifier_runs`
//! — "the emitted permutation count must equal the closed form over the shapes".)
//!
//! Only the first two terms move with the tenant's column widths. So the size of
//! lever 0 is decided by **the leaf term's share of the whole per-query bill**,
//! and that share is a property of the FORMAT: in the batched format one shared
//! Merkle path serves every matrix, in the per-table format every table walks
//! its own, so the per-table bill carries a far larger path term and the hash
//! matrix is a far smaller fraction of it.
//!
//! ⚠ **That is the correction this module exists to measure.** `MEMORY-MODEL`
//! §7 lever 0 reads the P3-CENSUS-AB decomposition "1,247 hash-matrix blocks of
//! 1,846 per query" and concludes 0.370× (RPX). 1,846 is a **batched** per-query
//! bill. Applying its ratio to the **per-table** aggregator prices the path term
//! as if it shrank with the hash matrix, which it does not.
//!
//! # What the two tenants are
//!
//! A wrap proof's sub-proofs ARE the LFM chips ([`super::airs::LFM_CHIP_NAMES`]),
//! so the tenant is a [`ChipSet`] plus a [`HasherKind`]:
//!
//! - **BLAKE3 tenant** — `keccak: false, blake3: true`, one `LFM_BLAKE3` chunk:
//!   12 sub-proofs. The hash work sits in `LFM_BLAKE3` (3,056 main + 631 ext aux)
//!   and `LFM_HASH` idles at the four-row floor.
//! - **Algebraic tenant** — `keccak: false, blake3: false`: **11 sub-proofs**.
//!   The same hash work sits in `LFM_HASH` at the pinned algebraic widths
//!   (RPX 316, RPO 436, Poseidon 612 value columns over a 13-column preprocessed
//!   prefix, 3 ext3 aux each) and `LFM_BLAKE3` is gone entirely.
//!
//! The algebraic tenant therefore wins on TWO axes at once, and they must not be
//! conflated: the hash matrix narrows ~10×, **and** four sub-proofs disappear,
//! taking their leaves, their Merkle paths and their FRI legs with them.
//!
//! # ⚠ Why the emission arm runs under the BLAKE3 wrap hash on this branch
//!
//! ✓ VERIFIED by reading: [`super::sub_proof::SubProofShape::opening_words`]
//! (`sub_proof.rs:149`) and [`super::fri::FriShape::query_words`] (`fri.rs:164`)
//! size their arena strides from `proof_arena::words_per_root()`, which reads
//! `WrapHash::production()` — the PIN — while
//! [`super::epoch_verify::emit_table_verification`] advances its cursor by
//! `edsl::digest_words(b)`, which reads the BUILDER. On this branch the pin is
//! BLAKE3, so a builder at `WrapHash::Algebraic` strides 1 against a declared
//! stride of 2 and trips `emit_table_verification`'s own assertion
//! ("the emitter's cursor must agree with the declared query stride").
//!
//! An explicit-algebraic emission is therefore not expressible here. It does not
//! matter for this measurement, because
//! [`super::epoch_verify::blocks_for`]'s algebraic and BLAKE3 arms are
//! numerically identical for every leaf of one felt or more — rate 8 both, no
//! spurious trailing block either side, which
//! `rpo_chip_tests::the_rate_eight_census_is_hash_invariant` already proves and
//! [`the_block_rule_is_hash_invariant_on_every_tenant_group`] re-checks over
//! exactly the groups measured here. Both arms are emitted under
//! `WrapHash::Blake3`, and only the TENANT's widths differ.
//!
//! ⚠ The residual bias is stated rather than corrected: an algebraic root is ONE
//! cell where a byte root is two, so the emitted arithmetic around each absorb
//! and each Merkle parent is slightly DEARER here than an algebraic build would
//! be. Every emitted figure below is thus an upper bound for the algebraic arm.

use stark::config::Commitment;
use stark::constraint_ir::ConstraintArtifact;
use stark::proof::options::ProofOptions;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::airs::{
    BLAKE3_SLOT, ChipSet, KECCAK_RC_SLOT, KECCAK_RND_SLOT, KECCAK_SLOT, LFM_CHIP_NAMES, LfmAirs,
    NUM_LFM_CHIPS,
};
use super::builder::{Ext, Felt, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::constraints::{Analysis, QuotientShape, analyze};
use super::deep::DeepShape;
use super::edsl::WrapHash;
use super::epoch::{RootCells, TableAbsorbs, TableChallengeShape, fork_table};
use super::epoch_verify::{
    FRI_LEAF_FELTS, TableVerifyShape, blocks_for, boundary_terms, group_leaf_felts,
    query_permutations_for,
};
use super::fri::FriShape;
use super::hash::HasherKind;
use super::instr::ArenaId;
use super::sub_proof::{GroupShape, SubProofShape};
use super::transcript_replay::TranscriptReplay;

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;
type V = Verifier<Gl, Ext3, ()>;

// ======================= the recorded tenant shapes =======================

/// Chip-class `log2` heights of the RECORDED per-table wrap proof, in the frozen
/// [`LFM_CHIP_NAMES`] order — `bench_cache/optladder_2026-08-21/TIP/tip-wrappt-24.stdout`
/// (a 2^24 epoch, wrap preset blowup 4 / 110 q, 15 sub-proofs).
///
/// ⚠ `KECCAK_RND` reads **5**, not the 0 the record's own `chip log-heights`
/// line prints. That line is the chip-CLASS array, and `KECCAK_RND` is the one
/// AIR with no preprocessed root, so 0 is its placeholder there; the CHIP CENSUS
/// in the same file shows the chunk at 32 rows. Taking the 0 literally would
/// delete a 1,480-column sub-proof from the bill.
///
/// ★ The keccak family is PRESENT in that run and the record says why: the
/// instruction mix reads `keccak 1`. ONE permutation instantiates all three of
/// its chips, and the aggregator then opens all three.
const RECORDED_WRAP_LOG_HEIGHTS: [u32; NUM_LFM_CHIPS] =
    [11, 20, 21, 21, 21, 2, 2, 23, 22, 13, 16, 20, 5, 5, 20];

/// `LFM_HASH`'s slot — the receiving end of the swap whose other end is
/// [`BLAKE3_SLOT`]. Local because `airs.rs` exports no constant for it; every
/// other slot this module names is imported from there rather than spelled as a
/// literal, so a reordering of the frozen chip order cannot silently re-point
/// this census at the wrong table.
const HASH_SLOT: usize = 5;

/// A placeholder root. Nothing measured here is a function of a root's VALUE,
/// only of the AIR's shape and of how many CELLS the root occupies — so one
/// interned constant serves every preprocessed slot.
const ZERO_ROOT: [u8; 32] = [0u8; 32];

/// Padded height of the per-table wrap's hash table, MEASURED: 776,289
/// invocations (`MEMORY-MODEL` §1.1) pad to 2^20, which is what slot 11 records.
const WRAP_HASH_LOG_HEIGHT: u32 = 20;

/// Where the algebraic tenant's hash work goes, and what it costs elsewhere.
///
/// `LFM_HASH` takes over `LFM_BLAKE3`'s invocation count — the count is
/// hash-invariant (`blocks_for` has one rule at rate 8) and one row per
/// permutation holds for every candidate, so the height moves unchanged from
/// slot 11 to slot 5.
///
/// ⚠ Every NON-hash height is held at its recorded value. That is conservative
/// in a knowable direction for `BITWISE` (2^20, fed by BLAKE3's per-byte XOR and
/// range lookups — `MEMORY-MODEL` §7 lever 4) and neutral for the rest.
fn tenant_log_heights(algebraic: bool) -> [u32; NUM_LFM_CHIPS] {
    let mut h = RECORDED_WRAP_LOG_HEIGHTS;
    if algebraic {
        h[HASH_SLOT] = WRAP_HASH_LOG_HEIGHT;
    }
    h
}

/// The tenant a per-table aggregator leg opens.
///
/// `keccak` is a SECOND axis and it is not decorative: the recorded wrap program
/// emits exactly one keccak permutation (`program_id` is deliberately keccak
/// over bytes whatever the configuration commits under — see
/// [`RootCells::byte_halves`]), and that one permutation instantiates
/// `LFM_KECCAK` (736+88), `KECCAK_RND` (1,480+516) and `KECCAK_RC`. Three
/// sub-proofs the aggregator must open, for one row of work. Whether an
/// algebraic pipeline still emits it is a property of the wrap PROGRAM, not of
/// the hash, so both readings are carried and the report names the condition.
#[derive(Clone, Copy)]
struct Tenant {
    label: &'static str,
    /// The `LFM_HASH` permutation the wrap proof was proved under.
    hasher: HasherKind,
    algebraic: bool,
    keccak: bool,
}

/// ⚠⚠ The `LFM_HASH` socket permutation of a BLAKE3-tenant wrap proof is
/// **`Test`, not `Blake3`** — `hash_pin::BLOCK_HASHER` on a BLAKE3-pinned build.
///
/// The two axes are orthogonal and this is the trap in confusing them. Under a
/// byte hash the emitter's Merkle work goes through `ByteWrapHash::hash_bytes`,
/// which lowers to the dedicated `LFM_BLAKE3` chip and emits **no `Instr::Hash`
/// at all**, so the socket idles at the four-row floor and its width is
/// `TEST_NUM_COLUMNS`. Naming `HasherKind::Blake3` here instead instantiates the
/// BLAKE3 *socket* — a 2,980-column chip the real proof does not carry — and
/// inflates the BLAKE3 baseline by 1,185 blocks per query (1,201 for the socket
/// against the idle chip's 16), which makes lever 0 look BIGGER than it is —
/// 0.632× against the true 0.774×.
///
/// ✓ VERIFIED against the recorded census, which reports `LFM_HASH` at 4 rows,
/// 0 used, 28 main, 3 aux. [`the_blake3_tenant_socket_matches_the_record`] pins
/// it so this cannot regress silently.
const BLAKE3_TENANT_SOCKET: HasherKind = HasherKind::Test;

const TENANTS: [Tenant; 8] = [
    Tenant {
        label: "BLAKE3",
        hasher: BLAKE3_TENANT_SOCKET,
        algebraic: false,
        keccak: true,
    },
    Tenant {
        label: "RPX",
        hasher: HasherKind::Rpx,
        algebraic: true,
        keccak: true,
    },
    Tenant {
        label: "RPO",
        hasher: HasherKind::Rpo,
        algebraic: true,
        keccak: true,
    },
    Tenant {
        label: "Poseidon",
        hasher: HasherKind::Poseidon,
        algebraic: true,
        keccak: true,
    },
    Tenant {
        label: "RPX/no-kec",
        hasher: HasherKind::Rpx,
        algebraic: true,
        keccak: false,
    },
    Tenant {
        label: "RPO/no-kec",
        hasher: HasherKind::Rpo,
        algebraic: true,
        keccak: false,
    },
    Tenant {
        label: "Pos/no-kec",
        hasher: HasherKind::Poseidon,
        algebraic: true,
        keccak: false,
    },
    // The second BLAKE3 baseline: what a BLAKE3 tenant costs if the keccak
    // family is retired from it TOO. Carried so the report can quote a
    // like-for-like pair on both readings of that axis instead of comparing a
    // keccak-less algebraic tenant against a keccak-carrying BLAKE3 one.
    Tenant {
        label: "BLAKE3/no-kec",
        hasher: BLAKE3_TENANT_SOCKET,
        algebraic: false,
        keccak: false,
    },
];

impl Tenant {
    fn chip_set(&self) -> ChipSet {
        ChipSet {
            keccak: self.keccak,
            blake3: !self.algebraic,
        }
    }

    /// `KECCAK_RND` chunks — one when the family is present, as the record shows.
    fn keccak_rnd_chunks(&self) -> usize {
        usize::from(self.keccak)
    }

    /// Is chip class `slot` a sub-proof of this tenant's wrap proof?
    fn has_slot(&self, slot: usize) -> bool {
        match slot {
            KECCAK_SLOT | KECCAK_RND_SLOT | KECCAK_RC_SLOT => self.keccak,
            BLAKE3_SLOT => !self.algebraic,
            _ => true,
        }
    }

    /// The AIR set the wrap proof was proved under, at this tenant's widths.
    ///
    /// The roots are placeholders: nothing measured here is a function of a
    /// root's VALUE, only of the AIR's shape. `LFM_BLAKE3`'s chunk-0 root must
    /// be slot 11's entry, which a uniform array satisfies.
    fn airs(&self, options: &ProofOptions) -> LfmAirs {
        let roots: [Commitment; NUM_LFM_CHIPS] = [[0u8; 32]; NUM_LFM_CHIPS];
        let blake3_roots: &[Commitment] = if self.algebraic {
            &[]
        } else {
            &roots[BLAKE3_SLOT..=BLAKE3_SLOT]
        };
        LfmAirs::new_chunked(
            &roots,
            blake3_roots,
            options,
            self.keccak_rnd_chunks(),
            self.hasher,
            self.chip_set(),
        )
    }

    /// This tenant's sub-proof heights, in `air_refs()` order — the frozen chip
    /// order with the absent families' slots removed.
    fn present_log_heights(&self) -> Vec<u32> {
        let h = tenant_log_heights(self.algebraic);
        let mut out = Vec::with_capacity(NUM_LFM_CHIPS);
        for (slot, height) in h.iter().enumerate() {
            if self.has_slot(slot) {
                out.push(*height);
            }
        }
        out
    }

    /// Base-field-equivalent cells one invocation of this tenant's hash chip
    /// costs — `main + 3·aux`, over the chip that actually carries the
    /// permutations. MEASURED from the chip layouts.
    fn hash_cells_per_perm(&self) -> u64 {
        if self.algebraic {
            let main = (super::chips::hash::num_columns(self.hasher)
                - super::layout::hash::PREP_WIDTH) as u64;
            let aux = super::chips::hash::bus_interactions(self.hasher)
                .len()
                .div_ceil(2) as u64;
            main + 3 * aux
        } else {
            let main =
                (super::blake3_chip::cols::NUM_COLUMNS - super::layout::blake3::PREP_WIDTH) as u64;
            let aux = super::blake3_chip::bus_interactions().len().div_ceil(2) as u64;
            main + 3 * aux
        }
    }
}

// ============================ shape derivation ============================

/// One sub-proof of the tenant's wrap proof, as SHAPE — no proof, no proving.
///
/// Every field comes from the AIR plus the proof OPTIONS plus one trace length,
/// which is precisely the split `epoch_verify_tests::build_table_legs` documents:
/// "Every shape here is derived from the AIR and the proof OPTIONS. The one
/// parameter that is neither is `log2_trace_length`." So this is that function
/// with the proof-reading half replaced by a declared height — the reason no
/// wrap proof has to exist for this census.
struct TableShape {
    name: &'static str,
    challenge: TableChallengeShape,
    verify: TableVerifyShape,
    analysis: Analysis,
    /// Widths as the CENSUS reports them: preprocessed columns excluded from
    /// main, aux counted in extension elements.
    main_cols: usize,
    aux_cols: usize,
    num_precomputed: usize,
}

fn table_shape(
    name: &'static str,
    air: &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
    index: usize,
    num_tables: usize,
    log2_trace_length: u32,
) -> TableShape {
    let opts = air.options();
    let layout = V::ood_layout(air);
    let artifact = ConstraintArtifact::capture(air);

    let (main_width, aux_width) = air.trace_layout();
    let num_total_cols = main_width + aux_width;
    let num_precomputed = if air.is_preprocessed() {
        air.num_precomputed_columns()
    } else {
        0
    };

    let trace_length = 1usize << log2_trace_length;
    let log2_blowup = (opts.blowup_factor as usize).trailing_zeros();
    let log2_lde_length = log2_trace_length + log2_blowup;
    // `prover.rs:1960`'s own expression, which is what fixes the part count and
    // therefore the parts group's width.
    let num_parts = air.composition_poly_degree_bound(trace_length) / trace_length;

    let mut trace_groups = Vec::new();
    if num_precomputed > 0 {
        trace_groups.push(GroupShape {
            num_columns: num_precomputed,
            is_ext: false,
        });
    }
    trace_groups.push(GroupShape {
        num_columns: main_width - num_precomputed,
        is_ext: false,
    });
    if aux_width > 0 {
        trace_groups.push(GroupShape {
            num_columns: aux_width,
            is_ext: true,
        });
    }

    let step_size = layout.step_size();
    let num_eval_points = artifact.shape.transition_offsets.len() * step_size;
    let deep = DeepShape {
        step_size,
        num_eval_points,
        num_total_cols,
        next_row_cols: layout.next_row_cols().to_vec(),
        num_composition_parts: num_parts,
        log2_trace_length,
    };
    let sub = SubProofShape {
        deep,
        trace_groups,
        merkle_depth: log2_lde_length as usize - 1,
        log2_lde_length,
        coset_offset: FE::from(opts.coset_offset),
    };
    let has_aux_trace = air.has_aux_trace();
    let fri = FriShape::from_options(opts, log2_lde_length);

    TableShape {
        name,
        challenge: TableChallengeShape {
            index,
            num_tables,
            has_aux_root: aux_width > 0,
            has_contribution: has_aux_trace,
            log2_trace_length,
            log2_blowup,
            coset_offset: FE::from(opts.coset_offset),
            ood_current_dims: (num_total_cols, step_size),
            ood_next_dims: (layout.expected_next_width(), layout.expected_next_height()),
            num_parts,
            fri,
            grinding_factor: opts.grinding_factor,
            num_queries: opts.fri_number_of_queries,
        },
        verify: TableVerifyShape {
            quotient: QuotientShape {
                log2_trace_length,
                num_composition_parts: num_parts,
                boundary: boundary_terms(has_aux_trace, num_total_cols),
            },
            fri,
            main_width,
            num_alpha_powers: if has_aux_trace {
                artifact.shape.max_bus_elements as usize
            } else {
                0
            },
            num_queries: opts.fri_number_of_queries,
            sub,
        },
        analysis: analyze(&artifact),
        main_cols: main_width - num_precomputed,
        aux_cols: aux_width,
        num_precomputed,
    }
}

/// Every sub-proof of one tenant's wrap proof, at the declared heights.
fn tenant_tables(tenant: &Tenant, airs: &LfmAirs, log_heights: &[u32]) -> Vec<TableShape> {
    let refs = airs.air_refs();
    assert_eq!(
        refs.len(),
        log_heights.len(),
        "{}: one declared height per sub-proof the AIR set builds",
        tenant.label
    );
    let names: Vec<&'static str> = {
        let mut out = Vec::with_capacity(refs.len());
        for (slot, name) in LFM_CHIP_NAMES.iter().enumerate() {
            if tenant.has_slot(slot) {
                out.push(*name);
            }
        }
        out
    };
    let n = refs.len();
    refs.iter()
        .enumerate()
        .map(|(i, air)| table_shape(names[i], *air, i, n, log_heights[i]))
        .collect()
}

// ======================= the per-query decomposition =======================

/// One tenant's per-query bill, split into the two terms that move with the
/// tenant's widths and the two that do not.
#[derive(Default, Clone, Copy)]
struct Bill {
    /// Trace/parts leaf ABSORPTION blocks — width-driven.
    trace_leaves: usize,
    /// FRI layer leaf absorption blocks — six felts, rate-driven, width-blind.
    fri_leaves: usize,
    /// Trace-tree Merkle parents — one compression each, width-blind.
    parents: usize,
    /// FRI path steps — one compression each, width-blind.
    fri_paths: usize,
    /// Of `trace_leaves`, the blocks the HASH MATRIX's own sub-proof costs.
    hash_matrix_leaves: usize,
}

impl Bill {
    fn total(&self) -> usize {
        self.trace_leaves + self.fri_leaves + self.parents + self.fri_paths
    }
}

/// Per-QUERY bill over every sub-proof, plus the whole-leg total.
///
/// The per-query figure sums the per-query cost of each sub-proof; the leg total
/// multiplies each by ITS OWN query count, which the per-table format makes a
/// per-table quantity even though every table here carries the same preset.
fn bill(tables: &[TableShape], hash: WrapHash, hash_chip: &str) -> (Bill, usize) {
    let mut b = Bill::default();
    let mut leg_total = 0usize;
    for t in tables {
        let groups = t.verify.sub.groups();
        let leaves: usize = groups
            .iter()
            .map(|g| blocks_for(group_leaf_felts(g), hash))
            .sum();
        let fri_leaves = t.verify.fri.num_committed() * blocks_for(FRI_LEAF_FELTS, hash);
        let parents = groups.len() * t.verify.sub.merkle_depth;
        let fri_paths = t.verify.fri.path_steps_per_query();

        b.trace_leaves += leaves;
        b.fri_leaves += fri_leaves;
        b.parents += parents;
        b.fri_paths += fri_paths;
        if t.name == hash_chip {
            b.hash_matrix_leaves += leaves;
        }

        // The closed form the emitter is pinned against, so the leg total is not
        // a second spelling of the sum above.
        let closed = query_permutations_for(&t.verify, hash);
        assert_eq!(
            closed,
            t.verify.num_queries * (leaves + fri_leaves + parents + fri_paths),
            "{}: the decomposition must reproduce `query_permutations_for`",
            t.name
        );
        leg_total += closed;
    }
    (b, leg_total)
}

// ============================= the RSS laws ==============================

const GIB: f64 = 1_073_741_824.0;

/// The fitted affine law, blowup 2: `1.242 GiB + 33.94 B` per base-equivalent
/// cell (`rss-affine-in-cells-measured`, hasher-independent to ±7.5%).
fn rss_fitted(cells: u64) -> f64 {
    1.242 + 33.94 * cells as f64 / GIB
}

/// The record's own anchor form, which is what produced the published
/// aggregator numbers: `RSS = (cells/12.2e9)·336.8 + C·(1 − cells/12.2e9)` at the
/// fixture-measured `C = 1.3 GiB`. Carried beside the fitted law because the two
/// disagree by ~15% and every published figure used this one.
fn rss_anchored(cells: u64) -> f64 {
    let f = cells as f64 / 12.2e9;
    f * 336.8 + 1.3 * (1.0 - f)
}

// ================================ the gates ==============================

/// ✓ The block rule is hash-invariant on every group these tenants open.
///
/// The load-bearing premise of the whole module: both arms are emitted under
/// `WrapHash::Blake3` (see the header), which is only legitimate because
/// [`blocks_for`]'s BLAKE3 and algebraic arms agree. `rpo_chip_tests` proves that
/// in general; this checks it on exactly the groups measured here, so a future
/// rate change would fail HERE rather than silently re-price the census.
#[test]
fn the_block_rule_is_hash_invariant_on_every_tenant_group() {
    let opts = wrap_options();
    let mut checked = 0usize;
    for tenant in &TENANTS {
        let airs = tenant.airs(&opts);
        let tables = tenant_tables(tenant, &airs, &tenant.present_log_heights());
        for t in &tables {
            for g in t.verify.sub.groups() {
                let felts = group_leaf_felts(&g);
                assert_eq!(
                    blocks_for(felts, WrapHash::Blake3),
                    blocks_for(felts, WrapHash::Algebraic),
                    "{}/{}: a {felts}-felt leaf must cost one block count at rate 8",
                    tenant.label,
                    t.name
                );
                checked += 1;
            }
            assert_eq!(
                query_permutations_for(&t.verify, WrapHash::Blake3),
                query_permutations_for(&t.verify, WrapHash::Algebraic),
                "{}/{}: the whole per-query bill must be hash-invariant at rate 8",
                tenant.label,
                t.name
            );
        }
    }
    assert!(checked >= 40, "every tenant's groups must be covered");
    println!("\n★ rate-8 block rule checked on {checked} tenant groups — invariant");
}

/// ✓ The BLAKE3 tenant's `LFM_HASH` is the idle 28-column `Test` socket the
/// record shows — not the 2,980-column BLAKE3 socket.
///
/// This is the one number in the census that, if wrong, moves the headline
/// ratio by a third and in the flattering direction, so it is asserted against
/// the recorded run rather than left to the tenant table's spelling.
#[test]
fn the_blake3_tenant_socket_matches_the_record() {
    /// `LFM_HASH` as `tip-wrappt-24.stdout`'s CHIP CENSUS reports it: 28 main
    /// value columns over the 13-column preprocessed prefix, 3 ext aux.
    const RECORDED: (usize, usize) = (28, 3);

    // ★ The tenant's socket IS the pin's, not a value chosen here. A branch that
    // re-pins `BLOCK_HASHER` moves what a BLAKE3-tenant wrap proof actually
    // carries, and this census would then be measuring a shape nothing proves.
    assert_eq!(
        BLAKE3_TENANT_SOCKET,
        crate::hash_pin::BLOCK_HASHER,
        "the BLAKE3 tenant's LFM_HASH socket must be the build's own pin — the \
         recorded census this census is a ratio against was produced under it"
    );

    let opts = wrap_options();
    for tenant in TENANTS.iter().filter(|t| !t.algebraic) {
        let airs = tenant.airs(&opts);
        let tables = tenant_tables(tenant, &airs, &tenant.present_log_heights());
        let hash = tables
            .iter()
            .find(|t| t.name == "LFM_HASH")
            .expect("every tenant carries LFM_HASH");
        assert_eq!(
            (hash.main_cols, hash.aux_cols),
            RECORDED,
            "{}: a BLAKE3-tenant wrap proof idles LFM_HASH at the pin's `Test` \
             socket — naming HasherKind::Blake3 here instantiates the 2,980-column \
             BLAKE3 socket, a chip the real proof does not carry, and inflates the \
             baseline this whole census is a ratio against",
            tenant.label,
        );
    }
    println!("\n★ BLAKE3-tenant LFM_HASH = {RECORDED:?} (main, ext aux) — matches the record");
}

/// The wrap layer's own preset — blowup 4 / 110 queries (`PLAN` T5). This is the
/// proof the aggregator OPENS, so its options fix the aggregator's leg bill.
fn wrap_options() -> ProofOptions {
    ProofOptions {
        blowup_factor: 4,
        fri_number_of_queries: 110,
        coset_offset: 3,
        grinding_factor: 20,
        fri_final_poly_log_degree: 7,
    }
}

/// A fixture-scale twin of [`wrap_options`]: the same blowup and the same
/// terminal degree, two queries, at a height that still folds several FRI layers.
fn fixture_wrap_options_at(queries: usize) -> ProofOptions {
    ProofOptions {
        fri_number_of_queries: queries,
        ..wrap_options()
    }
}

/// Query counts the emission arm sweeps, so the glue can be SPLIT rather than
/// quoted as one percentage.
///
/// ★ Why a sweep and not a single point. Emitted − closed form is
/// `glue(q) = F + P·q`: `F` is the query-INVARIANT per-table spine (the OOD
/// blocks, the terminal coefficients, the layer roots, the grinding check) and
/// `P` is the per-query remainder (index-bit sampling). One point cannot
/// separate them, and the two extrapolate to production's 110 queries by
/// factors 55 apart — the difference between the glue being under 1% of the leg
/// there and 57% of it. Three points fit the line and the middle one checks it.
const GLUE_SWEEP: [usize; 3] = [2, 4, 8];

/// Queries the wrap preset carries, which is what the glue must be
/// extrapolated to.
const PRODUCTION_QUERIES: usize = 110;

const FIXTURE_LOG_HEIGHT: u32 = 12;

/// ★★ THE GATE — lever 0, measured: the per-query bill of a per-table
/// aggregator leg over an ALGEBRAIC-tenant wrap proof against a BLAKE3-tenant
/// one, decomposed so the width-driven half is visible.
///
/// Closed-form only, over shapes derived from the real AIRs. No proof, no
/// proving, no emission — which is what makes it run in milliseconds and what
/// makes every number here a function of the pinned chip widths alone.
#[test]
fn the_lever_zero_factor_is_measured_per_tenant() {
    let opts = wrap_options();
    println!(
        "\n★★ LEVER 0 — per-table aggregator leg over one wrap proof\n   \
         wrap preset: blowup {} / {} queries / grinding {} / terminal 2^{}\n   \
         tenant heights: recorded per-table wrap (Σ log2 = {}), LFM_HASH raised \
         to 2^{WRAP_HASH_LOG_HEIGHT} on the algebraic arm",
        opts.blowup_factor,
        opts.fri_number_of_queries,
        opts.grinding_factor,
        opts.fri_final_poly_log_degree,
        RECORDED_WRAP_LOG_HEIGHTS.iter().sum::<u32>(),
    );

    let mut rows: Vec<(Tenant, Bill, usize, usize)> = Vec::new();
    for tenant in &TENANTS {
        let airs = tenant.airs(&opts);
        let heights = tenant.present_log_heights();
        let tables = tenant_tables(tenant, &airs, &heights);
        let hash_chip = if tenant.algebraic {
            "LFM_HASH"
        } else {
            "LFM_BLAKE3"
        };

        println!(
            "\n   ── {} tenant: {} sub-proofs",
            tenant.label,
            tables.len()
        );
        println!(
            "      {:>12} {:>7} {:>7} {:>6} {:>6} {:>7} {:>9}",
            "sub-proof", "log2", "prep", "main", "aux", "parts", "blocks/q"
        );
        for (t, h) in tables.iter().zip(&heights) {
            let groups = t.verify.sub.groups();
            let leaves: usize = groups
                .iter()
                .map(|g| blocks_for(group_leaf_felts(g), WrapHash::Blake3))
                .sum();
            println!(
                "      {:>12} {:>7} {:>7} {:>6} {:>6} {:>7} {:>9}",
                t.name,
                h,
                t.num_precomputed,
                t.main_cols,
                t.aux_cols,
                t.verify.sub.deep.num_composition_parts,
                leaves,
            );
        }

        let (b, leg) = bill(&tables, WrapHash::Blake3, hash_chip);
        rows.push((*tenant, b, leg, tables.len()));
    }

    // ---- the decomposition, side by side.
    println!(
        "\n   ── PER-QUERY BILL (blocks, summed over sub-proofs)\n   \
         {:>10} {:>7} {:>12} {:>11} {:>9} {:>10} {:>10} {:>8}",
        "tenant",
        "tables",
        "trace leaves",
        "of which hm",
        "FRI leaf",
        "parents",
        "FRI paths",
        "TOTAL"
    );
    for (t, b, _, n) in &rows {
        println!(
            "   {:>10} {:>7} {:>12} {:>11} {:>9} {:>10} {:>10} {:>8}",
            t.label,
            n,
            b.trace_leaves,
            b.hash_matrix_leaves,
            b.fri_leaves,
            b.parents,
            b.fri_paths,
            b.total(),
        );
    }

    let (_, base_bill, base_leg, _) = rows[0];
    println!(
        "\n   ── THE FACTOR (per query, and per leg at {} queries)\n   \
         {:>10} {:>10} {:>9} {:>14} {:>9} {:>16}",
        opts.fri_number_of_queries,
        "tenant",
        "blocks/q",
        "factor",
        "leg blocks",
        "factor",
        "hm share of q"
    );
    for (t, b, leg, _) in &rows {
        println!(
            "   {:>10} {:>10} {:>9.4} {:>14} {:>9.4} {:>15.1}%",
            t.label,
            b.total(),
            b.total() as f64 / base_bill.total() as f64,
            leg,
            *leg as f64 / base_leg as f64,
            100.0 * b.hash_matrix_leaves as f64 / b.total() as f64,
        );
    }

    // ---- what the model predicted, and where its arithmetic went.
    let hm_share = base_bill.hash_matrix_leaves as f64 / base_bill.total() as f64;
    let rpx = rows[1].1;
    let naive = (base_bill.total() - base_bill.hash_matrix_leaves + rpx.hash_matrix_leaves) as f64
        / base_bill.total() as f64;
    println!(
        "\n   ⚠ MEMORY-MODEL §7 lever 0 predicts 0.370× (RPX) from \
         '1,247 hash-matrix blocks of 1,846 per query' = a 67.6% hash-matrix \
         share.\n     MEASURED share of the PER-TABLE bill: {:.1}% — the \
         per-table format's Merkle-path term ({} of {} blocks/query, {:.1}%) is \
         what dilutes it.\n     Swapping ONLY the hash matrix gives {naive:.4}×; \
         the measured {:.4}× is lower because the algebraic tenant also \
         DELETES {} sub-proofs.",
        100.0 * hm_share,
        base_bill.parents + base_bill.fri_paths,
        base_bill.total(),
        100.0 * (base_bill.parents + base_bill.fri_paths) as f64 / base_bill.total() as f64,
        rows[1].1.total() as f64 / base_bill.total() as f64,
        rows[0].3 - rows[1].3,
    );

    // ---- the two LIKE-FOR-LIKE pairs, since the keccak axis has to be held
    // fixed on both sides of a ratio for it to mean anything.
    let by = |label: &str| -> Bill {
        rows.iter()
            .find(|(t, _, _, _)| t.label == label)
            .expect("tenant present")
            .1
    };
    println!("\n   ── LIKE-FOR-LIKE (the keccak family held fixed on both sides)");
    for (alg, b3, tag) in [
        ("RPX", "BLAKE3", "keccak family PRESENT on both"),
        (
            "RPX/no-kec",
            "BLAKE3/no-kec",
            "keccak family RETIRED on both",
        ),
    ] {
        println!(
            "      {tag}: {alg} {} / {b3} {} = {:.4}x",
            by(alg).total(),
            by(b3).total(),
            by(alg).total() as f64 / by(b3).total() as f64,
        );
    }

    // ---- FALSIFICATION. The model contradicts its own multiplier, and that is
    // a stronger check than my arithmetic disagreeing with its arithmetic.
    //
    // §1.2 puts the per-table/batched compression ratio at the aggregator at
    // 2.44x. The LEAF term is format-invariant — each table's leaf is absorbed
    // once per query either way, and only the Merkle paths multiply — so a
    // hash matrix that is 67.6% of the BATCHED bill is necessarily 67.6/2.44 =
    // 27.7% of the per-table one. That is the number MEASURED below, not 67.6%.
    println!(
        "\n   ── FALSIFICATION: §1.2's own 2.44x multiplier implies a per-table \
         hash-matrix share of 67.6%/2.44 = 27.7%.\n      MEASURED: {:.1}% — so \
         the 0.370x is an internal inconsistency in the model, not a \
         disagreement with its data.",
        100.0 * hm_share,
    );

    // ---- the assertions: the model's own claim, falsified with a band.
    let measured = rows[1].1.total() as f64 / base_bill.total() as f64;
    assert!(
        measured > 0.370 * 1.15,
        "lever 0 at the PER-TABLE aggregator must be materially weaker than the \
         batched-derived 0.370×; measured {measured:.4}× — if this ever fails, \
         the path term has collapsed and the model's reading is back in play"
    );
    assert!(
        (0.30..0.95).contains(&measured),
        "the RPX factor must sit between 'hash matrix is everything' and 'hash \
         matrix is nothing'; measured {measured:.4}×"
    );
    for (t, b, _, _) in rows.iter().skip(1) {
        assert!(
            b.total() < base_bill.total(),
            "{}: an algebraic tenant cannot cost MORE per query than BLAKE3",
            t.label
        );
    }
    assert!(
        rows[1].1.total() <= rows[2].1.total() && rows[2].1.total() <= rows[3].1.total(),
        "the per-query bill must order RPX ≤ RPO ≤ Poseidon, as their widths do"
    );
}

/// ★★ THE AGGREGATOR — [`WRAPS`] wraps, fan-in 2 / 3 / 6, from the measured leg
/// bill and an IMPORTED glue constant.
///
/// Derives, with every step printed: level-1 node invocations, the `LFM_HASH`
/// table height against the 2^20 / 2^21 / 2^22 cliffs, cells at each tenant's own
/// cells-per-permutation, and peak RSS under both affine laws. Then applies the
/// brief's rule — "≥ 25% headroom under 2^21 at fan-in 3 ⇒ 3, else 2".
///
/// ⚠⚠ **The LEG is measured here; the GLUE is not, and the fan-in verdict turns
/// on the glue.** [`GLUE_ADDITIVE`] is imported from `MEMORY-MODEL` §1.3(iii)'s
/// batched six-leg model. This module's own emission arm measures glue at ~1.1%
/// of the leg at 110 queries — but it emits ONE leg, and a single-leg fixture
/// **structurally cannot contain inter-leg glue**: the binding between legs of a
/// real multi-leg aggregator has no counterpart in it. So the two numbers do not
/// contradict each other and neither settles the question.
///
/// ★ What that means for the recommendation, stated so the name of this test
/// cannot be read as a claim it does not support: **fan-in 2 is the CONSERVATIVE
/// DEFAULT, not a measured result.** If the inter-leg glue is at the single-leg
/// level, fan-in 3 clears with ~36% headroom and the rule selects it. Lane A's
/// first real two-leg emission is what settles it.
#[test]
fn the_per_table_aggregator_tree_applies_the_rule() {
    let opts = wrap_options();

    // ⚠⚠ IMPORTED, NOT MEASURED HERE — and the fan-in verdict turns on it.
    //
    // `MEMORY-MODEL` §1.3(iii): the emitted BATCHED aggregation program cost
    // 1,852,068 invocations against a six-leg model of 1,382,358. Read additively
    // that is +469,710 per aggregator proof; read multiplicatively, ×1.340.
    //
    // Two reasons it cannot simply be replaced by this module's own measurement.
    // It is a residual against a DIFFERENT and coarser leg model — its own note
    // says that model "counts the 6 verify legs, misses the glue", so the
    // shortfall was attributed wholesale rather than decomposed. And this
    // module's emission arm emits ONE leg, which structurally cannot contain
    // INTER-LEG glue: the binding between the legs of a real multi-leg
    // aggregator has no counterpart in a single-leg fixture. The two numbers
    // are not in conflict; neither settles the question.
    const GLUE_ADDITIVE: u64 = 469_710;
    const GLUE_MULTIPLICATIVE: f64 = 1.340;
    /// Wraps a block aggregates: base epochs at 2^22 (`PLAN` T2) over the
    /// measured 39.6M-cycle guest, so ⌈39.6e6 / 2^22⌉ = **10**, one wrap each.
    /// MEASURED 2026-09-07 18:35 UTC. `PLAN` T2's 18 is the older guest.
    ///
    /// ★ The fan-in DECISION does not move with this number. A level-1 node
    /// costs `f · leg + glue`, which is a function of the fan-in alone; the wrap
    /// count sets how MANY nodes there are, not how big the biggest one is.
    const WRAPS: u64 = 10;

    // The non-hash floor, as base-equivalent cells per hash INVOCATION. Two
    // readings, both anchored on a recorded number and both carried:
    //
    // - LO, MEASURED: the recorded per-table wrap census — 670,468,916 non-hash
    //   base-equivalent cells over 776,289 invocations
    //   (`optladder_2026-08-21/TIP/tip-wrappt-24.stdout`).
    // - HI, DERIVED: `MEMORY-MODEL` §4's back-out for the per-table AGGREGATOR —
    //   6.755e9 over 3,910,237 invocations, i.e. exactly 2x the wrap's rate.
    //
    // ⚠ ESTIMATE either way: the non-hash chips are power-of-two padded, so a
    // per-invocation rate is a linear proxy for a step function. Closing this
    // band is what the emission arm below is for.
    const NONHASH_PER_INVOCATION_LO: f64 = 863.7;
    const NONHASH_PER_INVOCATION_HI: f64 = 1727.4;

    println!(
        "\n★★ PER-TABLE AGGREGATOR over {WRAPS} wraps (base epochs 2^22, 39.6M-cycle \
         guest)\n   terminal preset blowup 2 / 219 q; leg bill MEASURED at the \
         wrap's own blowup {} / {} q; glue IMPORTED (see the test's doc)",
        opts.blowup_factor, opts.fri_number_of_queries
    );

    for tenant in &TENANTS {
        let airs = tenant.airs(&opts);
        let tables = tenant_tables(tenant, &airs, &tenant.present_log_heights());
        let hash_chip = if tenant.algebraic {
            "LFM_HASH"
        } else {
            "LFM_BLAKE3"
        };
        let (_, leg) = bill(&tables, WrapHash::Blake3, hash_chip);
        let leg = leg as u64;
        let cpp = tenant.hash_cells_per_perm();

        println!(
            "\n   ── {} — leg = {leg} invocations, hash chip = {cpp} \
             base-equivalent cells/permutation",
            tenant.label
        );
        println!(
            "      {:>8} {:>9} {:>13} {:>7} {:>9} {:>13} {:>14} {:>17}",
            "fan-in",
            "L1 nodes",
            "invocations",
            "height",
            "headroom",
            "hash cells",
            "total cells",
            "RSS fit / anch"
        );
        for f in [2u64, 3, 6] {
            let nodes = WRAPS.div_ceil(f);
            for (tag, invocations) in [
                ("add", f * leg + GLUE_ADDITIVE),
                (
                    "mul",
                    ((f * leg) as f64 * GLUE_MULTIPLICATIVE).round() as u64,
                ),
            ] {
                let height = invocations.next_power_of_two();
                let headroom = 1.0 - invocations as f64 / height as f64;
                let hash_cells = height * cpp;
                for (band, per_inv) in [
                    ("lo", NONHASH_PER_INVOCATION_LO),
                    ("hi", NONHASH_PER_INVOCATION_HI),
                ] {
                    let cells = hash_cells + (invocations as f64 * per_inv).round() as u64;
                    println!(
                        "      {f:>4} {tag:<3} {band:<2} {nodes:>7} {invocations:>13} {:>7} \
                         {:>8.1}% {hash_cells:>13} {cells:>14} {:>8.1} /{:>7.1}",
                        format!("2^{}", height.trailing_zeros()),
                        100.0 * headroom,
                        rss_fitted(cells),
                        rss_anchored(cells),
                    );
                }
            }
        }

        // ---- the whole tree, so the node COUNT is visible beside the node SIZE.
        //
        // Printed at BOTH wrap counts to make the invariance explicit: the level-1
        // node size is identical at 10 and at 18, because it is `f · leg + glue`
        // and carries no wrap count at all. Only the number of nodes moves.
        if tenant.algebraic {
            for wraps in [WRAPS, 18] {
                for f in [2u64, 3] {
                    let mut level = wraps;
                    let mut shape = Vec::new();
                    let mut proofs = 0u64;
                    shape.push(level);
                    while level > 1 {
                        level = level.div_ceil(f);
                        shape.push(level);
                        proofs += level;
                    }
                    let path: Vec<String> = shape.iter().map(|n| n.to_string()).collect();
                    println!(
                        "      {wraps} wraps, fan-in {f}: {} — {proofs} aggregator \
                     proofs over {} levels, every node {} invocations",
                        path.join(" → "),
                        shape.len() - 1,
                        f * leg + GLUE_ADDITIVE,
                    );
                }
            }
        }

        // ---- the brief's decision rule, on the additive reading at fan-in 3.
        if tenant.algebraic {
            let inv3 = 3 * leg + GLUE_ADDITIVE;
            let cliff = 1u64 << 21;
            let headroom = 1.0 - inv3 as f64 / cliff as f64;
            println!(
                "      RULE ({}): fan-in 3 = {inv3} invocations, {:.1}% headroom \
                 under 2^21 ⇒ fan-in {}",
                tenant.label,
                100.0 * headroom,
                if headroom >= 0.25 { 3 } else { 2 },
            );
        }
    }

    // ---- THE GATE. The rule, on BOTH keccak readings — because that axis, not
    // the hash, is what decides it.
    println!(
        "\n   ⇒ DECISION (rule: ≥25% headroom under 2^21 at fan-in 3 ⇒ 3, else 2)\n      \
         ⚠ under the IMPORTED glue constant — the leg is measured, the glue is not"
    );
    for tenant in TENANTS.iter().filter(|t| t.algebraic) {
        let airs = tenant.airs(&opts);
        let tables = tenant_tables(tenant, &airs, &tenant.present_log_heights());
        let (_, leg) = bill(&tables, WrapHash::Blake3, "LFM_HASH");
        for f in [2u64, 3] {
            let inv = f * leg as u64 + GLUE_ADDITIVE;
            println!(
                "      {:<12} fan-in {f}: {inv:>9} = {:>5.1}% of 2^21, headroom {:>5.1}%",
                tenant.label,
                100.0 * inv as f64 / (1u64 << 21) as f64,
                100.0 * (1.0 - inv as f64 / (1u64 << 21) as f64),
            );
        }
    }
    println!(
        "      ⚠ the glue term is known to be UNDER-counted by 34-42% \
         (MEMORY-MODEL §1.3(iii)). A fan-in whose headroom is below that band is \
         not protected against it."
    );

    // ---- the same rule under the MEASURED glue, read multiplicatively.
    //
    // The emission arm measures glue at +57.2% of the leg (RPX, fixture heights,
    // 2 queries — box B at `1de0e3b4`). Read as a multiplier that is the
    // PESSIMISTIC bound, and it is the reading that decides: a fan-in 3 node
    // crosses 2^21 outright and doubles its hash table, while fan-in 2 still
    // clears the rule. The emission arm's query sweep says how much of the
    // +57.2% survives to 110 queries; this row is the bound if none of it does.
    const MEASURED_GLUE_MULTIPLIER: f64 = 1.5716;
    println!(
        "\n   ⇒ SAME RULE under the MEASURED glue read multiplicatively \
         (×{MEASURED_GLUE_MULTIPLIER}, the pessimistic bound)"
    );
    for tenant in TENANTS.iter().filter(|t| t.algebraic) {
        let airs = tenant.airs(&opts);
        let tables = tenant_tables(tenant, &airs, &tenant.present_log_heights());
        let (_, leg) = bill(&tables, WrapHash::Blake3, "LFM_HASH");
        for f in [2u64, 3] {
            let inv = ((f * leg as u64) as f64 * MEASURED_GLUE_MULTIPLIER).round() as u64;
            let height = inv.next_power_of_two();
            println!(
                "      {:<12} fan-in {f}: {inv:>9} → 2^{} ({:>5.1}% of 2^21, \
                 headroom {:>5.1}% of its own height){}",
                tenant.label,
                height.trailing_zeros(),
                100.0 * inv as f64 / (1u64 << 21) as f64,
                100.0 * (1.0 - inv as f64 / height as f64),
                if height > (1u64 << 21) {
                    "  ⚠ CROSSES 2^21 — the hash table doubles"
                } else {
                    ""
                },
            );
        }
    }

    // RPX with the keccak family — the shape the record actually exhibits.
    let airs = TENANTS[1].airs(&opts);
    let tables = tenant_tables(&TENANTS[1], &airs, &TENANTS[1].present_log_heights());
    let (_, leg) = bill(&tables, WrapHash::Blake3, "LFM_HASH");
    let inv3 = 3 * leg as u64 + GLUE_ADDITIVE;
    let inv2 = 2 * leg as u64 + GLUE_ADDITIVE;
    let cliff = 1u64 << 21;
    assert!(
        inv3 < cliff,
        "fan-in 3 must at least stay under 2^21: {inv3}"
    );
    assert!(
        1.0 - inv2 as f64 / cliff as f64 >= 0.25,
        "fan-in 2 must clear the 25% rule with the keccak family present — that \
         is what makes it the recommendation; headroom {:.1}%",
        100.0 * (1.0 - inv2 as f64 / cliff as f64),
    );

    // ★ The recommendation's real strength: fan-in 2 clears the rule under the
    // PESSIMISTIC glue reading as well, and fan-in 3 does not merely fail the
    // rule there — it crosses the cliff. Asserted so the two do not have to be
    // re-derived by hand from the printed table.
    let pess = |f: u64| ((f * leg as u64) as f64 * 1.5716).round() as u64;
    assert!(
        1.0 - pess(2) as f64 / cliff as f64 >= 0.25,
        "fan-in 2 must clear 25% under the measured glue too: {} = {:.1}% of 2^21",
        pess(2),
        100.0 * pess(2) as f64 / cliff as f64,
    );
    // ⚠ Deliberately NOT `assert!(pess(3) > cliff)`. It is true — 2,109,260
    // against 2,097,152 — but by 0.58%, and a tripwire that flips on a 0.6%
    // drift is noise rather than a guard. The robust statement is the one the
    // rule is about: under the pessimistic reading fan-in 3 is nowhere near 25%
    // headroom, it is OVER the cliff. That takes a 20%+ move to flip.
    assert!(
        pess(3) as f64 / cliff as f64 > 0.75,
        "under the measured glue a fan-in 3 node must be far outside the 25% \
         rule — if this ever fails, fan-in 3 has become viable and the \
         recommendation should be revisited; it stands at {} = {:.1}% of 2^21",
        pess(3),
        100.0 * pess(3) as f64 / cliff as f64,
    );
    println!(
        "\n   ⇒ margin note: under the pessimistic glue, fan-in 3 lands at {} \
         against the 2^21 cliff of {cliff} — over it, but by only {:.2}%. The \
         recommendation does not rest on that 0.6%: fan-in 3 already fails the \
         rule at 13.6% headroom under the imported additive glue.",
        pess(3),
        100.0 * (pess(3) as f64 / cliff as f64 - 1.0),
    );
    println!(
        "\n   ★ READ THE VERDICT AS: fan-in 2 is the CONSERVATIVE DEFAULT, not a \
         measured result. The inter-leg glue of a real multi-leg aggregator is \
         UNMEASURED — a single-leg fixture cannot contain it — and the constant \
         above is imported from the batched model. If that glue is at the \
         single-leg level this arm measures (~1.1% of the leg at 110 queries), \
         fan-in 3 clears with ~36% headroom and the rule selects 3. Lane A's \
         first real two-leg emission settles it."
    );
}

// =========================== the emission arm ============================

/// Per-table arena set of one emitted leg, in declaration order — the shape
/// `aggregator_tests::GlobalTableArenas` has, since this is the same emitter.
struct LegArenas {
    aux_root: Option<ArenaId>,
    contribution: Option<ArenaId>,
    composition_root: ArenaId,
    ood_current: ArenaId,
    ood_next: ArenaId,
    parts: ArenaId,
    fri_roots: ArenaId,
    fri_coeffs: ArenaId,
    nonce: Option<ArenaId>,
    legs: super::epoch_verify::TableQueryArenas,
}

/// The per-table verification program over one tenant's wrap proof — the GLOBAL
/// leg's own structure (`aggregator_tests.rs:1476`): Phase A over hinted main
/// roots, one [`fork_table`] per sub-proof, [`super::epoch::emit_table_challenges`]
/// then [`super::epoch_verify::emit_table_verification`], and the LogUp closure.
///
/// ⚠ Built at `WrapHash::Blake3` on BOTH arms — see the module header for why an
/// explicit-algebraic build cannot be emitted on this branch, and why it does not
/// change a permutation count.
fn tenant_leg_program(tables: &[TableShape]) -> LfmProgram {
    tenant_node_program(tables, 1)
}

/// [`tenant_leg_program`] over `legs` wrap proofs in ONE node — the shape a
/// fan-in-`legs` aggregator has.
///
/// Every sub-proof of every leg gets its own fork index over the whole set
/// (`num_tables = legs · tables.len()`), which is what a real multi-leg
/// aggregator does. ⚠ It carries no BINDING legs — no register chain, no
/// attestation join, no published-root compare — because those are lane A's
/// design and not mine to invent. So the inter-leg glue it measures is a LOWER
/// BOUND: the shared spine and the widened fork space, and nothing of the
/// binding.
fn tenant_node_program(tables: &[TableShape], legs: usize) -> LfmProgram {
    use super::statement_replay::{PhaseATable, replay_phase_a};

    assert!(legs >= 1, "a node verifies at least one leg");
    // One flat sub-proof list over every leg, re-indexed so each fork is
    // distinct across the whole node. The SHAPE is borrowed and only the
    // challenge is owned: `Analysis` is not `Clone`, and it is also the one
    // field a second leg genuinely shares — the same AIR, the same constraint
    // program, verified twice.
    let num_tables = legs * tables.len();
    let flat: Vec<(&TableShape, TableChallengeShape)> = (0..legs)
        .flat_map(|_| tables.iter())
        .enumerate()
        .map(|(i, t)| {
            let mut ch = t.challenge.clone();
            ch.index = i;
            ch.num_tables = num_tables;
            (t, ch)
        })
        .collect();

    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Blake3);
    let n = flat.len();
    let per_root = RootCells::words_per_root(&b);

    let a_main_roots = b.declare_arena(per_root * n as u32);
    let per_table: Vec<LegArenas> = flat
        .iter()
        .map(|(shape, cs)| LegArenas {
            aux_root: cs.has_aux_root.then(|| b.declare_arena(per_root)),
            contribution: cs.has_contribution.then(|| b.declare_arena(1)),
            composition_root: b.declare_arena(per_root),
            ood_current: b.declare_arena((cs.ood_current_dims.0 * cs.ood_current_dims.1) as u32),
            ood_next: b.declare_arena((cs.ood_next_dims.0 * cs.ood_next_dims.1) as u32),
            parts: b.declare_arena(cs.num_parts as u32),
            fri_roots: b.declare_arena(per_root * cs.fri.num_committed() as u32),
            fri_coeffs: b.declare_arena(cs.fri.num_terminal_coeffs() as u32),
            nonce: (cs.grinding_factor > 0).then(|| b.declare_arena(1)),
            legs: super::epoch_verify::declare_table_arenas(&mut b, &shape.verify),
        })
        .collect();

    // A one-append statement stands in for the aggregator's own; it is 32 bytes
    // of spine and is identical on both arms, so it cannot tilt the comparison.
    let mut t = TranscriptReplay::new(&[]);
    t.append_const_bytes(&ZERO_ROOT[..]);

    let main_cells: Vec<RootCells> = (0..n)
        .map(|i| RootCells::hint(&mut b, a_main_roots, per_root * i as u32))
        .collect();
    let main_halves: Vec<Vec<Felt>> = main_cells.iter().map(RootCells::lanes_flat).collect();
    let prep_cells: Vec<Option<RootCells>> = flat
        .iter()
        .map(|(s, _)| (s.num_precomputed > 0).then(|| RootCells::constant(&mut b, &ZERO_ROOT)))
        .collect();
    let phase_a: Vec<PhaseATable> = flat
        .iter()
        .enumerate()
        .map(|(i, (s, _))| PhaseATable {
            preprocessed_root: (s.num_precomputed > 0).then_some(
                super::statement_replay::PhaseAPreprocessed::Constant(&ZERO_ROOT),
            ),
            main_root: &main_halves[i][..],
        })
        .collect();
    let (z, alpha) = replay_phase_a(&mut t, &mut b, &phase_a);
    b.public(z.as_cell());
    b.public(alpha.as_cell());

    let mut contributions: Vec<Ext> = Vec::new();
    for (i, (s, cs)) in flat.iter().enumerate() {
        let a = &per_table[i];
        let aux = a.aux_root.map(|id| RootCells::hint(&mut b, id, 0));
        let contribution = a.contribution.map(|id| b.hint_word(id, 0).as_ext());
        let composition = RootCells::hint(&mut b, a.composition_root, 0);
        let ood_current: Vec<Ext> = (0..(cs.ood_current_dims.0 * cs.ood_current_dims.1) as u32)
            .map(|k| b.hint_word(a.ood_current, k).as_ext())
            .collect();
        let ood_next: Vec<Ext> = (0..(cs.ood_next_dims.0 * cs.ood_next_dims.1) as u32)
            .map(|k| b.hint_word(a.ood_next, k).as_ext())
            .collect();
        let parts: Vec<Ext> = (0..cs.num_parts as u32)
            .map(|k| b.hint_word(a.parts, k).as_ext())
            .collect();
        let fri_roots: Vec<RootCells> = (0..cs.fri.num_committed())
            .map(|k| RootCells::hint(&mut b, a.fri_roots, per_root * k as u32))
            .collect();
        let fri_coeffs: Vec<Ext> = (0..cs.fri.num_terminal_coeffs() as u32)
            .map(|k| b.hint_word(a.fri_coeffs, k).as_ext())
            .collect();
        let nonce = a.nonce.map(|id| b.hint_felt(id, 0));
        if let Some(c) = contribution {
            contributions.push(c);
        }
        let mut fork = fork_table(&t, cs.index, cs.num_tables);
        let absorbs = TableAbsorbs {
            aux_root: aux.as_ref(),
            contribution,
            composition_root: &composition,
            ood_current: &ood_current,
            ood_next: &ood_next,
            parts: &parts,
            fri_roots: &fri_roots,
            fri_coeffs: &fri_coeffs,
            nonce,
        };
        let ch = super::epoch::emit_table_challenges(&mut b, &mut fork, cs, &absorbs);
        super::epoch_verify::emit_table_verification(
            &mut b,
            &s.verify,
            &s.analysis,
            &ch,
            &absorbs,
            &super::epoch_verify::TableInputs {
                precomputed_root: prep_cells[i].as_ref(),
                main_root: &main_cells[i],
                rap_challenges: &[z, alpha],
            },
            &a.legs,
        );
    }

    let shape = super::logup::LogUpShape {
        num_contributing_tables: contributions.len(),
        num_output_bytes: 0,
    };
    let target = b.ext_const(&FEE::zero());
    super::logup::emit_bus_closure(&mut b, &shape, &contributions, target);

    compile(b.finish())
}

/// ★ THE EMISSION ARM — the closed forms above, put through the real emitter.
///
/// At fixture heights (uniform 2^{FIXTURE_LOG_HEIGHT}, two queries) the leg is
/// emitted for real and the census run over it, which measures three things the
/// closed form cannot: that the emitted permutation count IS the closed form on
/// these tenant shapes, the GLUE (spine, Phase A, challenge sampling, grinding,
/// closure) as emitted rather than as modelled, and the emitted program's total
/// base-equivalent cells.
///
/// `#[ignore]`d: emission over a 3,056-column hash matrix is seconds, not
/// milliseconds, and this is an instrument rather than a guard.
#[test]
#[ignore = "emission instrument: run explicitly, prints the census"]
fn the_tenant_leg_emits_and_censuses() {
    println!(
        "\n★ EMITTED PER-TABLE LEG — fixture heights 2^{FIXTURE_LOG_HEIGHT}, \
         blowup 4, queries swept over {GLUE_SWEEP:?}\n   \
         glue = emitted − closed form, fitted as F + P·q and extrapolated to the \
         wrap preset's {PRODUCTION_QUERIES} queries"
    );

    for tenant in &TENANTS {
        let hash_chip = if tenant.algebraic {
            "LFM_HASH"
        } else {
            "LFM_BLAKE3"
        };
        let mut points: Vec<(usize, i64, usize)> = Vec::new();
        let mut shape = (0usize, 0usize, 0usize, 0usize, 0u64);

        for q in GLUE_SWEEP {
            let opts = fixture_wrap_options_at(q);
            let airs = tenant.airs(&opts);
            let heights = vec![FIXTURE_LOG_HEIGHT; airs.air_refs().len()];
            let tables = tenant_tables(tenant, &airs, &heights);
            let (b_bill, leg) = bill(&tables, WrapHash::Blake3, hash_chip);

            let program = tenant_leg_program(&tables);
            let emitted = super::wrap_tests::hash_ops(&program, WrapHash::Blake3);
            let (main, aux) = super::airs::lfm_cell_counts_with_hasher(&program, tenant.hasher);
            assert!(
                emitted >= leg,
                "{}: the emitted count cannot be below the leg's closed form — the \
                 difference IS the glue",
                tenant.label
            );
            points.push((q, emitted as i64 - leg as i64, leg));
            shape = (
                tables.len(),
                b_bill.total(),
                b_bill.hash_matrix_leaves,
                program.instrs.len(),
                main + 3 * aux,
            );
        }

        // ---- the fit. The endpoints give the line; the middle point checks it.
        let (q0, g0, leg0) = points[0];
        let (q2, g2, _) = points[2];
        let (q1, g1, _) = points[1];
        let per_query = (g2 - g0) as f64 / (q2 - q0) as f64;
        let fixed = g0 as f64 - per_query * q0 as f64;
        let predicted = fixed + per_query * q1 as f64;
        let residual = (predicted - g1 as f64) / g1 as f64;

        let leg_prod = leg0 as f64 / q0 as f64 * PRODUCTION_QUERIES as f64;
        let glue_prod = fixed + per_query * PRODUCTION_QUERIES as f64;

        println!(
            "\n   ── {} tenant: {} sub-proofs, {} blocks/query ({} hash matrix), \
             {} instructions, {} cells at q={}",
            tenant.label, shape.0, shape.1, shape.2, shape.3, shape.4, GLUE_SWEEP[2],
        );
        for (q, g, leg) in &points {
            println!(
                "      q={q:<2} leg {leg:>7}  glue {g:>7}  ({:+.1}% of the leg)",
                100.0 * *g as f64 / *leg as f64
            );
        }
        println!(
            "      FIT glue(q) = {fixed:.0} + {per_query:.1}·q  (q={q1}: predicted \
             {predicted:.0} vs measured {g1}, {:+.2}%)\n      \
             ⇒ at {PRODUCTION_QUERIES} queries: leg {leg_prod:.0}, glue \
             {glue_prod:.0} = {:.1}% of the leg   [FIXTURE heights — see below]",
            100.0 * residual,
            100.0 * glue_prod / leg_prod,
        );

        assert!(
            residual.abs() < 0.05,
            "{}: glue(q) must be affine in the query count — the middle point \
             missed the line through the endpoints by {:.1}%, so the split into a \
             fixed spine and a per-query remainder does not hold and the \
             extrapolation is not licensed",
            tenant.label,
            100.0 * residual,
        );
    }

    println!(
        "\n   ⚠ The extrapolation moves the glue's SHARE, not the fan-in verdict \
         on its own. It is taken at FIXTURE heights (2^{FIXTURE_LOG_HEIGHT}), where \
         each table commits 4 FRI layers against production's 12 and walks 13 \
         Merkle levels against 21-25, so the fixed term F is UNDERSTATED at real \
         heights. What carries is the property, not the constant: F is \
         width-driven and query-invariant, so its share falls as the query count \
         rises from {} to {PRODUCTION_QUERIES}.",
        GLUE_SWEEP[0],
    );
}

/// Query counts the production-height arm sweeps.
///
/// ★ Small ON PURPOSE, and the reason is the whole design of this test:
/// **`F` is the query-INVARIANT intercept, so measuring it needs production
/// HEIGHTS, not production queries.** A 110-query emission at these heights
/// would materialise ~447,370 compressions, which pads `LFM_BLAKE3` to 2^19
/// rows × 3,112 columns ≈ 13 GB of column-group data for that chip alone.
/// Eight queries pads it to 2^15 and costs ~800 MB, and the intercept it
/// yields is the same number.
const PRODUCTION_HEIGHT_SWEEP: [usize; 3] = [2, 4, 8];

/// ★★ THE DECIDING MEASUREMENT — `F` at PRODUCTION heights, and how it scales
/// with the LEG COUNT.
///
/// [`the_tenant_leg_emits_and_censuses`] fits the glue at fixture heights
/// (2^12), where each table commits 4 FRI layers against production's 12. This
/// arm re-fits it at the recorded per-table wrap's own heights, so the fan-in
/// verdict stops resting on a height extrapolation.
///
/// It also emits at **one leg and at two**, because the single-leg fixture is
/// exactly what could not answer the open question. The difference between
/// `F(2 legs)` and `2·F(1 leg)` is the inter-leg glue — the part of a
/// multi-leg node that is not just its legs added up.
///
/// ⚠⚠ **This is still a LOWER BOUND on the inter-leg glue, and the report must
/// say so.** The two-leg node here shares one transcript, one Phase A and one
/// LogUp closure over a fork space of `2 · tables` — but it carries **no
/// binding legs**: no register chain, no attestation join, no published-root
/// compare. Those are lane A's design and inventing them here would be
/// measuring my guess rather than the machine. So a fan-in verdict from this
/// number is safe in one direction only: if fan-in 3 misses the rule even at
/// this lower bound, it misses. If it clears here, lane A's real node still
/// decides.
#[test]
#[ignore = "production-height emission: seconds and GBs, run explicitly"]
fn the_rpx_leg_emits_at_production_heights() {
    // RPX with the keccak family — the shape the record actually exhibits.
    let tenant = TENANTS[1];
    assert_eq!(tenant.label, "RPX", "the production-height arm is RPX's");

    // The leg the aggregator really pays, by the closed form at the real preset.
    let prod_opts = wrap_options();
    let prod_airs = tenant.airs(&prod_opts);
    let prod_tables = tenant_tables(&tenant, &prod_airs, &tenant.present_log_heights());
    let (prod_bill, leg_110) = bill(&prod_tables, WrapHash::Blake3, "LFM_HASH");
    println!(
        "\n★★ RPX LEG AT PRODUCTION HEIGHTS — {} sub-proofs at the recorded \
         per-table wrap's heights\n   \
         closed form: {} blocks/query × {} queries = {leg_110} compressions per leg",
        prod_tables.len(),
        prod_bill.total(),
        prod_opts.fri_number_of_queries,
    );

    // ---- fit glue(q) = F + P·q at ONE leg and at TWO.
    let mut fits: Vec<(usize, f64, f64)> = Vec::new();
    for legs in [1usize, 2] {
        let mut points: Vec<(usize, i64)> = Vec::new();
        println!("\n   ── {legs} leg(s)");
        for q in PRODUCTION_HEIGHT_SWEEP {
            let opts = ProofOptions {
                fri_number_of_queries: q,
                ..wrap_options()
            };
            let airs = tenant.airs(&opts);
            let tables = tenant_tables(&tenant, &airs, &tenant.present_log_heights());
            let (_, leg) = bill(&tables, WrapHash::Blake3, "LFM_HASH");

            let program = tenant_node_program(&tables, legs);
            let emitted = super::wrap_tests::hash_ops(&program, WrapHash::Blake3);
            let glue = emitted as i64 - (legs * leg) as i64;
            assert!(
                glue >= 0,
                "{legs} leg(s), q={q}: the emitted count cannot be below the legs' \
                 closed form — the difference IS the glue"
            );
            println!(
                "      q={q:<2} legs {legs}  closed form {:>9}  emitted {emitted:>9}  \
                 glue {glue:>7}  ({} instructions)",
                legs * leg,
                program.instrs.len(),
            );
            points.push((q, glue));
        }
        let ((q0, g0), (q1, g1), (q2, g2)) = (points[0], points[1], points[2]);
        let per_query = (g2 - g0) as f64 / (q2 - q0) as f64;
        let fixed = g0 as f64 - per_query * q0 as f64;
        let predicted = fixed + per_query * q1 as f64;
        let residual = (predicted - g1 as f64) / g1 as f64;
        println!(
            "      FIT glue(q) = {fixed:.0} + {per_query:.1}·q  (q={q1}: predicted \
             {predicted:.0} vs measured {g1}, {:+.2}%)",
            100.0 * residual
        );
        assert!(
            residual.abs() < 0.05,
            "{legs} leg(s): glue(q) must be affine in the query count; the middle \
             point missed the line by {:.1}%",
            100.0 * residual,
        );
        fits.push((legs, fixed, per_query));
    }

    // ---- the inter-leg glue: what a second leg costs BEYOND a copy of the first.
    let (_, f1, p1) = fits[0];
    let (_, f2, p2) = fits[1];
    let inter_f = f2 - 2.0 * f1;
    let inter_p = p2 - 2.0 * p1;
    println!(
        "\n   ── INTER-LEG GLUE (lower bound — no binding legs, see the doc)\n      \
         F(1 leg) = {f1:.0}, F(2 legs) = {f2:.0}; 2·F(1) = {:.0} ⇒ inter-leg \
         F = {inter_f:+.0} ({:+.1}% of one leg's)\n      \
         P(1 leg) = {p1:.1}, P(2 legs) = {p2:.1} ⇒ inter-leg P = {inter_p:+.1}",
        2.0 * f1,
        100.0 * inter_f / f1,
    );

    // ---- the rule, at production heights and the real query count.
    //
    // A node of `f` legs costs `f · leg + F(f) + P(f)·q`. F and P are measured
    // at f ∈ {1,2}; f = 3 extends the per-leg increment linearly, which is the
    // one modelled step here and is named as such.
    let f_of = |f: f64| f1 + (f - 1.0) * (f2 - f1);
    let p_of = |f: f64| p1 + (f - 1.0) * (p2 - p1);
    println!(
        "\n   ⇒ THE RULE at production heights, {} queries\n      \
         {:>7} {:>13} {:>9} {:>8} {:>10} {:>14}",
        prod_opts.fri_number_of_queries,
        "fan-in",
        "invocations",
        "glue",
        "height",
        "headroom",
        "verdict",
    );
    for f in [2u64, 3] {
        let glue = f_of(f as f64) + p_of(f as f64) * prod_opts.fri_number_of_queries as f64;
        let inv = (f * leg_110 as u64) + glue.round() as u64;
        let height = inv.next_power_of_two();
        let headroom = 1.0 - inv as f64 / height as f64;
        println!(
            "      {f:>7} {inv:>13} {:>9.0} {:>8} {:>9.1}% {:>14}",
            glue,
            format!("2^{}", height.trailing_zeros()),
            100.0 * headroom,
            if headroom >= 0.25 { "clears" } else { "MISSES" },
        );
    }
    println!(
        "\n   ⚠ Read this as a LOWER BOUND on the inter-leg glue: the two-leg node \
         emitted here shares a transcript, a Phase A and a LogUp closure but \
         carries NO binding legs (register chain, attestation join, \
         published-root compare). If fan-in 3 misses the rule even here, it \
         misses. If it clears here, lane A's real node still decides."
    );
}
