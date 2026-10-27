use ethrex_common::U256;
use ethrex_levm::opcode_handlers::arithmetic::div_with_u64_fast_path;

#[test]
fn wide_division_matches_reference_at_correction_boundaries() {
    let mut divisors=vec![u32::MAX as u64+1,u32::MAX as u64+2,u64::MAX,u64::MAX-1,10u64.pow(18)];
    for bit in 32..64 {
        let p=1u64<<bit;
        divisors.extend([p-1,p,p+1,p+(1u64<<(bit-1))-1]);
        for low in [0,1,0x7fffffff,0xfffffffe,0xffffffff] {
            divisors.push(p|low);
        }
    }
    let mut count=0;
    for d in divisors {
        let denominator=U256::from(d);
        // Quotient estimates at BASE and just below it, and near divisor multiples.
        for r in [0,1,d/2,d-2,d-1] {
            for digit in [0,1,u32::MAX as u64-1,u32::MAX as u64] {
                for position in [32,64,128,192] {
                    let n=(U256::from(r)<<position)|U256::from(digit);
                    assert_eq!(div_with_u64_fast_path(n,denominator),n/denominator,"n={n}, d={d}");
                    count+=1;
                }
            }
        }
        for q in [U256::zero(),U256::one(),U256::from(u32::MAX),U256::from(u64::MAX),
                  (U256::one()<<128)-U256::one(),(U256::one()<<192)-U256::one()] {
            for r in [0,d/2,d-1] {
                let n=q*denominator+U256::from(r);
                assert_eq!(div_with_u64_fast_path(n,denominator),q);
                count+=1;
            }
        }
    }
    assert_eq!(count,28714);
}

#[test]
fn wide_division_matches_reference_for_random_full_width_numerators() {
    let mut seed=0x11_25368371_254531u64;
    let mut random=||{seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;seed};
    for _ in 0..100_000 {
        let n=U256([random(),random(),random(),random()]);
        let d=U256::from(random() | (1u64<<32));
        assert_eq!(div_with_u64_fast_path(n,d),n/d);
    }
}
