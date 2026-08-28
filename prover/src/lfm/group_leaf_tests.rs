//! ★★★ **THE GROUP-LEAF FELT-SEQUENCE DIFFERENTIAL** — the one construction on
//! the batched wrap path with no differential covering it.
//!
//! The host builds a mixed round's leaf in `stark::fri::mmcs`: for each matrix
//! of the height group, all of `evaluations` then all of `evaluations_sym`,
//! matrices in round INPUT order, flat, one hash. The machine re-derives it in
//! [`super::batched_epoch_verify::emit_group_leaf_hash`]. If the two feed
//! different felts the walk reconstructs nothing, and the leg fails as a
//! `DivByZero` deep in a query walk that names neither the hash nor the site.
//!
//! ⚠ **Why this compares SEQUENCES and not only digests.** A digest
//! differential says THAT the two disagree. It cannot say whether the
//! disagreement is the order across matrices, the split between a matrix's two
//! rows, the decomposition of one extension element, or the padding of the
//! last block — and those have different fixes. Both sides therefore expose the
//! felt run they actually absorb ([`stark::fri::mmcs::group_opening_felts`] and
//! [`super::batched_epoch_verify::group_leaf_felts`], each split out of its own
//! only production caller), and the assertion names the first index at which
//! they part.
//!
//! ⚠ **Neither side restates the convention.** The expectation is not a rule
//! written out here for both implementations to be checked against — that would
//! pass whenever this file and the code share a misunderstanding. The host
//! sequence comes from the host's own production function, its base-felt
//! decomposition from the host's own
//! [`super::algebraic_commit::element_felts`], and the machine sequence is read
//! out of an EXECUTED program's memory. What this file chooses is only the
//! shapes.
//!
//! The shapes are chosen so that ordering and padding cannot both hide:
//! several matrices of differing widths, groups whose felt count lands under,
//! exactly on, and over the rate-8 boundary (the padding flag `len mod 8` is
//! the one part of the construction that is not identical on every block), and
//! the single-matrix degenerate case.

use stark::config::StarkHash;
use stark::fri::mmcs::{group_opening_felts, hash_group_openings};
use stark::proof::stark::PolynomialOpenings;

use super::algebraic_commit::{
    AlgebraicHasher, PoseidonCommit, PoseidonStarkHash, RpoCommit, RpoStarkHash, RpxCommit,
    RpxStarkHash, digest_to_commitment, element_felts,
};
use super::batched_epoch_verify::{MixedMatrixOpening, emit_group_leaf_hash, group_leaf_felts};
use super::builder::{Cell, LfmBuilder};
use super::compiler::compile;
use super::edsl::WrapHash;
use super::executor::execute;
use super::sub_proof::GroupShape;
use super::word::{LfmWord, base_word, ext_word};
use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

/// The three tenants that HAVE a commitment configuration, as the pair this
/// file needs: the permutation tag the machine's socket proves, and the
/// `StarkHash` the host commits under. They are the same hash by construction —
/// see `algebraic_commit`'s note — and passing both is what lets one body drive
/// the host and the machine at once.
///
/// `HasherKind::Test` is absent for the reason `algebraic_commit`'s own tenant
/// macro gives: it is a permutation without a commitment configuration, so
/// there is no host side to differentiate against.
macro_rules! for_each_tenant {
    ($body:ident) => {
        $body::<RpoCommit, RpoStarkHash>("Rpo");
        $body::<RpxCommit, RpxStarkHash>("Rpx");
        $body::<PoseidonCommit, PoseidonStarkHash>("Poseidon");
    };
}

/// An empty authentication path — a per-matrix opening's own `proof` is always
/// empty here, exactly as `MixedOpening`'s doc says: the group's one path is
/// the authenticator, and the leaf hash never reads it.
fn no_proof<T: PartialEq + Eq>() -> crypto::merkle_tree::proof::Proof<T> {
    crypto::merkle_tree::proof::Proof {
        merkle_path: Vec::new(),
    }
}

/// Distinct, matrix- and position-dependent values, so a swap of any two felts
/// anywhere in the sequence is visible. Non-zero throughout, which a
/// zero-padding bug could otherwise mask.
fn base_val(matrix: usize, i: usize) -> FE {
    FE::from(1000 * (matrix as u64 + 1) + i as u64 + 1)
}

fn ext_val(matrix: usize, i: usize) -> FEE {
    let b = 1000 * (matrix as u64 + 1) + 3 * i as u64;
    FEE::new([FE::from(b + 1), FE::from(b + 2), FE::from(b + 3)])
}

/// Every matrix in one height group sits at one height, so the leaf is
/// independent of it; a fixed value keeps the shape honest without implying
/// otherwise.
const GROUP_HEIGHT: usize = 4;

/// Name the first index at which two felt runs part, rather than only that they
/// do — the whole reason this is a sequence differential.
fn assert_sequence(tenant: &str, case: &str, host: &[FE], machine: &[FE]) {
    if let Some(i) = (0..host.len().min(machine.len())).find(|&i| host[i] != machine[i]) {
        panic!(
            "{tenant}/{case}: felt sequences part at index {i} of {} (host) / {} (machine):\n  \
             host    = {:?}\n  machine = {:?}",
            host.len(),
            machine.len(),
            &host[i.saturating_sub(2)..(i + 3).min(host.len())],
            &machine[i.saturating_sub(2)..(i + 3).min(machine.len())],
        );
    }
    assert_eq!(
        host.len(),
        machine.len(),
        "{tenant}/{case}: felt sequences agree on their common prefix but not in LENGTH"
    );
}

/// Emit the machine's group leaf over `runs`, execute it, and return the felt
/// sequence it absorbed (as values) together with the digest it produced.
///
/// The felts are read out of final memory: every felt the machine absorbs is a
/// base-valued word, so lane 0 is its value — an unpacked lane is written as
/// `base_word(lane)` and a hinted base cell holds `base_word(v)`.
fn machine_run<H: AlgebraicHasher>(
    arena: Vec<LfmWord>,
    widths: &[usize],
    is_ext: bool,
) -> (Vec<FE>, [u8; 32]) {
    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Algebraic);
    let a = b.declare_arena(arena.len() as u32);
    let cells: Vec<Cell> = (0..arena.len()).map(|i| b.hint_word(a, i as u32)).collect();

    let mut runs: Vec<Vec<Cell>> = Vec::new();
    let mut at = 0usize;
    for &w in widths {
        runs.push(cells[at..at + 2 * w].to_vec());
        at += 2 * w;
    }
    assert_eq!(at, cells.len(), "the runs cover the arena exactly");

    let matrices: Vec<MixedMatrixOpening<'_>> = widths
        .iter()
        .zip(&runs)
        .map(|(&w, values)| MixedMatrixOpening {
            shape: GroupShape {
                num_columns: w,
                is_ext,
            },
            log_height: GROUP_HEIGHT,
            values,
        })
        .collect();
    let group: Vec<&MixedMatrixOpening<'_>> = matrices.iter().collect();

    // The sequence and the digest come from the SAME program: the collection
    // under test, then production's own leaf over it.
    let felts = group_leaf_felts(&mut b, &group);
    let digest = emit_group_leaf_hash(&mut b, &group);
    assert_eq!(digest.len(), 1, "an algebraic digest is ONE cell");
    b.public(digest[0]);

    let program = compile(b.finish());
    let exec = execute(&program, &[arena], &H::KIND).expect("the leaf program must execute");

    let values = felts
        .iter()
        .map(|f| {
            exec.memory[f.addr().0 as usize].expect("an absorbed felt must have been written")[0]
        })
        .collect();
    // ⚠ `digest_to_commitment`, NOT `word::pack_digest`: the two disagree on
    // endianness (big vs little), and the host's `Commitment` is the former's.
    // Restating the byte order here is exactly the mistake this file's header
    // warns about, so the machine's digest goes through the host's own function.
    (values, digest_to_commitment(&exec.public_words[0].1))
}

/// ★ BASE matrices — the `main` round's shape.
fn check_base<H: AlgebraicHasher, S: StarkHash>(tenant: &str, case: &str, widths: &[usize]) {
    let openings: Vec<PolynomialOpenings<GoldilocksField>> = widths
        .iter()
        .enumerate()
        .map(|(m, &w)| PolynomialOpenings {
            proof: no_proof(),
            evaluations: (0..w).map(|c| base_val(m, c)).collect(),
            evaluations_sym: (0..w).map(|c| base_val(m, w + c)).collect(),
        })
        .collect();
    let group: Vec<&PolynomialOpenings<GoldilocksField>> = openings.iter().collect();

    let mut host: Vec<FE> = Vec::new();
    for e in &group_opening_felts(&group) {
        element_felts(e, &mut host);
    }
    let want = hash_group_openings::<GoldilocksField, S>(&group);

    // The arena in the machine's layout: per matrix, its `2 · w` opened cells
    // in leaf order — which is what the caller of `emit_mixed_verify_batch`
    // hints from the proof arena.
    let arena: Vec<LfmWord> = widths
        .iter()
        .enumerate()
        .flat_map(|(m, &w)| (0..2 * w).map(move |i| base_word(base_val(m, i))))
        .collect();

    let (machine, got) = machine_run::<H>(arena, widths, false);
    assert_sequence(tenant, case, &host, &machine);
    assert_eq!(got, want, "{tenant}/{case}: leaf digests must agree");
}

/// ★ EXTENSION matrices — the `aux` and `parts` rounds' shape, where each value
/// contributes THREE felts and the decomposition order is load-bearing.
fn check_ext<H: AlgebraicHasher, S: StarkHash>(tenant: &str, case: &str, widths: &[usize]) {
    let openings: Vec<PolynomialOpenings<GoldilocksExtension>> = widths
        .iter()
        .enumerate()
        .map(|(m, &w)| PolynomialOpenings {
            proof: no_proof(),
            evaluations: (0..w).map(|c| ext_val(m, c)).collect(),
            evaluations_sym: (0..w).map(|c| ext_val(m, w + c)).collect(),
        })
        .collect();
    let group: Vec<&PolynomialOpenings<GoldilocksExtension>> = openings.iter().collect();

    let mut host: Vec<FE> = Vec::new();
    for e in &group_opening_felts(&group) {
        element_felts(e, &mut host);
    }
    let want = hash_group_openings::<GoldilocksExtension, S>(&group);

    let arena: Vec<LfmWord> = widths
        .iter()
        .enumerate()
        .flat_map(|(m, &w)| (0..2 * w).map(move |i| ext_word(&ext_val(m, i))))
        .collect();

    let (machine, got) = machine_run::<H>(arena, widths, true);
    assert_sequence(tenant, case, &host, &machine);
    assert_eq!(got, want, "{tenant}/{case}: leaf digests must agree");
}

/// ★★★ The gate. Base groups: felt count is `2 · Σw`, so the rate-8 boundary is
/// crossed in both directions and landed on exactly.
#[test]
fn the_machine_group_leaf_absorbs_the_host_felt_sequence_base() {
    fn check<H: AlgebraicHasher, S: StarkHash>(tenant: &str) {
        // (case name, widths) — felt counts 2, 6, 8, 10, 20, 8, 32.
        check_base::<H, S>(tenant, "single-w1 (2 felts, degenerate)", &[1]);
        check_base::<H, S>(tenant, "single-w3 (6 felts, under rate)", &[3]);
        check_base::<H, S>(tenant, "single-w4 (8 felts, exactly rate)", &[4]);
        check_base::<H, S>(tenant, "single-w5 (10 felts, over rate)", &[5]);
        check_base::<H, S>(tenant, "multi-2,3,5 (20 felts, mixed widths)", &[2, 3, 5]);
        check_base::<H, S>(
            tenant,
            "multi-1,1,1,1 (8 felts, exactly rate)",
            &[1, 1, 1, 1],
        );
        check_base::<H, S>(tenant, "multi-7,9 (32 felts)", &[7, 9]);
    }
    for_each_tenant!(check);
}

/// ★★★ The gate, extension side. Felt count is `6 · Σw`, so w=4 lands exactly
/// on a rate multiple and w=1, 2, 3 do not.
#[test]
fn the_machine_group_leaf_absorbs_the_host_felt_sequence_ext() {
    fn check<H: AlgebraicHasher, S: StarkHash>(tenant: &str) {
        check_ext::<H, S>(tenant, "single-w1 (6 felts, under rate)", &[1]);
        check_ext::<H, S>(tenant, "single-w2 (12 felts, over rate)", &[2]);
        check_ext::<H, S>(tenant, "single-w3 (18 felts)", &[3]);
        check_ext::<H, S>(tenant, "single-w4 (24 felts, exact multiple)", &[4]);
        check_ext::<H, S>(tenant, "multi-1,2,4 (42 felts, mixed widths)", &[1, 2, 4]);
        check_ext::<H, S>(tenant, "multi-2,2 (24 felts, exact multiple)", &[2, 2]);
    }
    for_each_tenant!(check);
}

/// ⚠ **The differential's own control.** A gate that compares two sequences is
/// worth nothing if it would pass on sequences that differ, and the failure
/// this whole file exists to catch is precisely an ORDER disagreement — which a
/// length check and a digest check can both miss. So: perturb the machine's
/// arena by swapping two felts that a wrong matrix order would swap, and
/// require the gate's own comparison to reject it.
#[test]
fn the_differential_rejects_a_reordered_sequence() {
    let a: Vec<FE> = (0..6u64).map(FE::from).collect();
    let mut b = a.clone();
    b.swap(1, 4);
    let out = std::panic::catch_unwind(|| assert_sequence("ctl", "swap", &a, &b));
    assert!(out.is_err(), "a swapped sequence must be rejected");

    let short = &a[..5];
    let out = std::panic::catch_unwind(|| assert_sequence("ctl", "short", &a, short));
    assert!(out.is_err(), "a truncated sequence must be rejected");

    assert_sequence("ctl", "identical", &a, &a);
}
