//! KECCAK_RND without the byte checks its D lookups imply
//! ([`KeccakRndForm::Lean`]): an opt-in
//! format value, [`KeccakRndForm::Full`] the production constant.
//!
//! Pinned here: the knob reads `full` or `lean` and nothing else; the lean
//! table is today's less exactly the 20 `Cxz_left` checks; every `Cxz_left`
//! byte still enters a `BYTE_ALU[XOR]` lookup, which is what makes the checks
//! redundant; the θ identity rules out the one extra value, by parity (and an
//! identity without its factor 2 would not); the verifier presets hold `Full`;
//! and a lean block proves and verifies while each form's verifier refuses
//! the other's proof.

use crate::block_whir::{
    BlockFormat, BlockWhirProof, Deviations, prove_block_whir_with, verify_block_whir,
};
use crate::lfm::whir_block_tree::keccak_rnd_setting;
use crate::tables::bitwise::BitwiseOperationType;
use crate::tables::keccak_rnd::{self, KeccakRndForm, cols};
use crate::tables::types::BusId;
use crate::test_utils::asm_elf_bytes;
use crate::tests::block_whir_p1_tests::{many_groups, options};
use math::field::element::FieldElement;
use multilinear::whir_chain::ArgueFormat;
use stark::lookup::{BusInteraction, BusValue, LinearTerm};
use stark::proof::options::{BaseFormat, ProofOptions};

type F = crate::tables::types::GoldilocksField;

/// `BLOCK_WHIR_KECCAK_RND`: unset or `full` is today's table, `lean` the
/// opt-in, anything else refused.
#[test]
fn the_keccak_rnd_knob_reads_full_or_lean() {
    assert_eq!(keccak_rnd_setting(None), Ok(KeccakRndForm::Full));
    assert_eq!(keccak_rnd_setting(Some("full")), Ok(KeccakRndForm::Full));
    assert_eq!(keccak_rnd_setting(Some(" lean ")), Ok(KeccakRndForm::Lean));
    for bad in ["", "1", "on", "Lean", "off"] {
        assert!(keccak_rnd_setting(Some(bad)).is_err(), "{bad:?}");
    }
    assert_eq!(KeccakRndForm::default(), KeccakRndForm::Full);
}

/// An interaction as text (its bus, side, multiplicity and values), for
/// comparing lists: `BusInteraction` has no `Debug` of its own.
fn key(i: &BusInteraction) -> String {
    format!(
        "{} {} {:?} {:?}",
        i.bus_id, i.is_sender, i.multiplicity, i.values
    )
}

/// The ARE_BYTES pair a sender checks, as its two columns, if it is one.
fn are_bytes_pair(i: &BusInteraction) -> Option<(usize, usize)> {
    if i.bus_id != u64::from(BusId::AreBytes) {
        return None;
    }
    match i.values.as_slice() {
        [
            BusValue::Packed {
                start_column: a, ..
            },
            BusValue::Packed {
                start_column: b, ..
            },
        ] => Some((*a, *b)),
        _ => None,
    }
}

/// The lean table is today's, in today's order, less exactly the 20 checks on
/// `Cxz_left`'s byte pairs: 1,031 interactions become 1,011, and a row sends
/// 200 byte checks instead of 220, padding rows included.
#[test]
fn the_lean_form_drops_exactly_the_cxz_left_byte_checks() {
    let full = keccak_rnd::bus_interactions_for(KeccakRndForm::Full);
    let lean = keccak_rnd::bus_interactions_for(KeccakRndForm::Lean);
    assert_eq!(full.len(), 1031);
    assert_eq!(lean.len(), 1011);
    assert_eq!(keccak_rnd::bus_interactions().len(), full.len());
    let cxz_left = |c: usize| (cols::CXZ_LEFT..cols::CXZ_LEFT + 40).contains(&c);
    let kept: Vec<String> = full
        .iter()
        .filter(|i| !are_bytes_pair(i).is_some_and(|(a, b)| cxz_left(a) && cxz_left(b)))
        .map(key)
        .collect();
    let lean_debug: Vec<String> = lean.iter().map(key).collect();
    assert_eq!(
        kept, lean_debug,
        "lean = full less the Cxz_left pairs, in order"
    );
    assert!(
        lean.iter()
            .filter_map(are_bytes_pair)
            .all(|(a, b)| !cxz_left(a) && !cxz_left(b)),
        "no Cxz_left byte check is left"
    );
    assert_eq!(
        KeccakRndForm::Full.are_bytes_per_row(),
        keccak_rnd::ARE_BYTES_PER_ROW
    );
    assert_eq!(KeccakRndForm::Lean.are_bytes_per_row(), 200);
    assert_eq!(
        lean.iter().filter(|i| are_bytes_pair(i).is_some()).count(),
        KeccakRndForm::Lean.are_bytes_per_row()
    );
    // 2 calls = 48 rows, padded to 64: 16 padding rows.
    assert_eq!(
        keccak_rnd::padding_are_bytes_for(2, KeccakRndForm::Full),
        16 * 220
    );
    assert_eq!(
        keccak_rnd::padding_are_bytes_for(2, KeccakRndForm::Lean),
        16 * 200
    );
    assert_eq!(keccak_rnd::padding_are_bytes(2), 16 * 220);
}

/// The premise that makes the dropped checks redundant: for every lane `x` and
/// byte `b`, a `BYTE_ALU[XOR]` lookup weighted by μ takes `Cxz_left[x][b]`
/// with coefficient 1 in its first-operand slot's neighbour — the second
/// operand — plus at most the one carry bit of the halfword below (even `b`)
/// and nothing else. Its operands are bytes, so `Cxz_left[x][b]` is a byte, or
/// `−1` beside a carry of 1 (even `b`).
#[test]
fn every_cxz_left_byte_enters_a_d_xor_lookup() {
    let lean = keccak_rnd::bus_interactions_for(KeccakRndForm::Lean);
    let xor = format!(
        "{:?}",
        BusValue::constant(crate::tables::types::alu_op::XOR as u64)
    );
    for x in 0..5 {
        for b in 0..8 {
            let left = cols::cxz_left(x, b);
            let carry = cols::cxz_right_bit_for_byte(b).map(|hw| cols::cxz_right_bit(x, hw));
            let found = lean.iter().any(|i| {
                if i.bus_id != u64::from(BusId::ByteAlu)
                    || !matches!(i.multiplicity, stark::lookup::Multiplicity::Column(c) if c == cols::MU)
                {
                    return false;
                }
                let [op, _, second, _] = i.values.as_slice() else {
                    return false;
                };
                if format!("{op:?}") != xor {
                    return false;
                }
                let mut want = vec![LinearTerm::Column {
                    coefficient: 1,
                    column: left,
                }];
                if let Some(c) = carry {
                    want.push(LinearTerm::Column {
                        coefficient: 1,
                        column: c,
                    });
                }
                format!("{second:?}") == format!("{:?}", BusValue::linear(want))
            });
            assert!(found, "Cxz_left[{x}][{b}] enters a μ-weighted D XOR lookup");
            assert_eq!(
                carry.is_some(),
                b % 2 == 0,
                "a carry beside the even bytes only"
            );
        }
    }
    // And the lookup's table holds bytes: BYTE_ALU is indexed by two bytes.
    assert_eq!(
        crate::tables::bitwise::NUM_ROWS,
        256 * 256 * 16,
        "(X: Byte, Y: Byte, Z: B4)"
    );
    let _ = BitwiseOperationType::ByteAluXor;
}

/// The θ identity's residue `2·in − 2¹⁶·r − (lo + 256·hi)` over the field.
fn theta_residue(input: u64, r: u64, hi: u64, lo: FieldElement<F>, two: u64) -> FieldElement<F> {
    FieldElement::<F>::from(two * input)
        - FieldElement::<F>::from(r << 16)
        - FieldElement::<F>::from(256 * hi)
        - lo
}

/// ★ The parity argument, exhaustively: with the low byte `−1` — the one value
/// the D lookup leaves an even `Cxz_left` byte besides `[0, 255]`, when the
/// carry beside it is 1 — no halfword `in`, carry `r` and byte `hi` satisfies
/// the θ identity. And every honest halfword's low byte is even, so the D
/// lookup's operand `lo + carry` stays a byte.
#[test]
fn a_low_byte_of_minus_one_breaks_the_theta_identity() {
    let minus_one = -FieldElement::<F>::one();
    let zero = FieldElement::<F>::zero();
    for input in 0..1u64 << 16 {
        let shifted = (input << 1) & 0xFFFF;
        let (lo, hi, r) = (shifted & 0xFF, shifted >> 8, input >> 15);
        assert_eq!(theta_residue(input, r, hi, FieldElement::from(lo), 2), zero);
        assert_eq!(lo % 2, 0, "in = {input}: the honest low byte is even");
        for r in 0..2u64 {
            for hi in 0..256u64 {
                assert_ne!(
                    theta_residue(input, r, hi, minus_one, 2),
                    zero,
                    "in = {input}, r = {r}, hi = {hi}: lo = −1 passes"
                );
            }
        }
    }
}

/// The argument's mutation: the same identity without its factor 2 admits a low
/// byte of `−1` (`in = 255`, `r = 0`, `hi = 1`), so the parity of `2·in` is
/// what the dropped checks' redundancy rests on.
#[test]
fn without_the_factor_two_a_low_byte_of_minus_one_would_pass() {
    let minus_one = -FieldElement::<F>::one();
    assert_eq!(
        theta_residue(255, 0, 1, minus_one, 1),
        FieldElement::<F>::zero()
    );
}

/// The verifier's presets hold `Full`: a lean proof verifies only under a
/// format that names `Lean`.
#[test]
fn the_verifiers_keccak_rnd_form_is_a_named_constant() {
    assert_eq!(BlockFormat::pinned().keccak_rnd, KeccakRndForm::Full);
    assert_eq!(
        BlockFormat::pinned()
            .with_keccak_rnd(KeccakRndForm::Lean)
            .keccak_rnd,
        KeccakRndForm::Lean
    );
}

fn format(keccak_rnd: KeccakRndForm) -> BlockFormat {
    BlockFormat {
        argue: ArgueFormat::BATCHED,
        keccak_rnd,
        ..many_groups(BaseFormat::P1_WHIR)
    }
}

fn prove(elf: &[u8], format: &BlockFormat) -> BlockWhirProof {
    let mut o = options(0);
    o.window_log2 = Some(3);
    prove_block_whir_with(
        elf,
        &[],
        &ProofOptions::default_test_options(),
        format,
        &o,
        &Deviations::default(),
    )
    .expect("prove")
    .0
}

fn verify(proof: &BlockWhirProof, elf: &[u8], format: &BlockFormat) -> bool {
    verify_block_whir(proof, elf, &ProofOptions::default_test_options(), format).unwrap_or(false)
}

/// ★ A lean block proves and verifies (its BITWISE counts the lean table's
/// byte checks, or its bus would not balance); today's verifier refuses it and
/// the lean verifier refuses today's proof. A block with KECCAK_RND.
#[test]
fn a_lean_block_proves_and_verifies() {
    let elf = asm_elf_bytes("test_keccak");
    let lean = format(KeccakRndForm::Lean);
    let full = format(KeccakRndForm::Full);
    let proof = prove(&elf, &lean);
    assert!(verify(&proof, &elf, &lean), "the lean verifier");
    assert!(!verify(&proof, &elf, &full), "today's verifier refuses it");
    let todays = prove(&elf, &full);
    assert!(
        verify(&todays, &elf, &full),
        "today's verifier, today's proof"
    );
    assert!(
        !verify(&todays, &elf, &lean),
        "the lean verifier refuses today's"
    );
    eprintln!(
        "lean keccak block: {} tables, {} groups; verified under lean, refused under full and back",
        proof.table_num_vars.len(),
        proof.groups.len()
    );
}
