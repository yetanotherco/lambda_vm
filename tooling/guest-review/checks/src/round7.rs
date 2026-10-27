use ethrex_common::U256;
use ethrex_levm::opcode_handlers::arithmetic::div_with_u64_fast_path;

#[test]
fn specialized_division_matches_full_width_reference() {
    let mut values = vec![U256::zero(), U256::one(), U256::MAX];
    for bit in 0..256 {
        let power = U256::one() << bit;
        values.extend([power, power - U256::one(), power.saturating_add(U256::one())]);
    }
    for &lhs in &values {
        for &rhs in &values {
            assert_eq!(div_with_u64_fast_path(lhs, rhs), lhs.checked_div(rhs).unwrap_or_default(),
                "lhs={lhs}, rhs={rhs}");
        }
    }
    let mut seed = 0x25368371_25453112u64;
    let mut random = || {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed
    };
    for _ in 0..20_000 {
        let lhs = U256([random(), random(), random(), random()]);
        for rhs in [U256::from(random() as u32), U256::from(random()),
                    U256([random(), random(), random(), random()])] {
            assert_eq!(div_with_u64_fast_path(lhs, rhs), lhs.checked_div(rhs).unwrap_or_default());
        }
    }
}
