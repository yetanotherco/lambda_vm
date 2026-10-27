use ethrex_common::U256;
use ethrex_levm::opcode_handlers::arithmetic::div_with_u64_fast_path;

#[test]
fn full_width_division_add_back_cases() {
    let one=U256::one();
    let base=one<<32;
    let mut count=0;
    for digits in 3..=8 {
        for bit in 0..32 {
            let top=32*(digits-1)+bit;
            let d=(one<<top)+one;
            if top<255 {
                // The top two divisor digits suggest q=2, but its low digit
                // makes subtraction underflow and requires an add-back to q=1.
                let n=d*U256::from(2)-one;
                assert_eq!(div_with_u64_fast_path(n,d),one,"digits={digits}, bit={bit}");
                count+=1;
            }
            if digits<8 {
                // A nonzero high quotient digit followed by an add-back at the low digit.
                let n=d*base+d-one;
                assert_eq!(div_with_u64_fast_path(n,d),base);
                count+=1;
            }
        }
    }
    assert_eq!(count,351);
}

#[test]
fn full_width_division_all_divisor_widths_and_quotient_boundaries() {
    let mut seed=0x12_25368371_254531u64;
    let mut random=||{seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;seed};
    let mut count=0;
    for width in 65..=256 {
        for _ in 0..64 {
            let raw=U256([random(),random(),random(),random()]);
            let d=(raw & (U256::MAX>>(256-width))) | (U256::one()<<(width-1));
            let n=U256([random(),random(),random(),random()]);
            assert_eq!(div_with_u64_fast_path(n,d),n/d);
            count+=1;
            let cap=U256::MAX/d;
            for q in [U256::zero(),U256::one(),cap,U256::from(random()).min(cap)] {
                let product=q*d;
                let r=(d-U256::one()).min(U256::MAX-product);
                for extra in [U256::zero(),r] {
                    assert_eq!(div_with_u64_fast_path(product+extra,d),q,"width={width}, q={q}, d={d}");
                    count+=1;
                }
            }
        }
    }
    assert_eq!(count,110592);
}
