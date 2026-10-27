//! # Arithmetic operations
//!
//! Includes the following opcodes:
//!   - `ADD`
//!   - `SUB`
//!   - `MUL`
//!   - `DIV`
//!   - `SDIV`
//!   - `MOD`
//!   - `SMOD`
//!   - `ADDMOD`
//!   - `MULMOD`
//!   - `EXP`
//!   - `SIGNEXTEND`
//!   - `CLZ`

use crate::{
    errors::{OpcodeResult, VMError},
    gas_cost,
    opcode_handlers::OpcodeHandler,
    vm::VM,
};
use ethrex_common::{U256, U512};

/// Implementation for the `ADD` opcode.
pub struct OpAddHandler;
impl OpcodeHandler for OpAddHandler {
    #[inline(always)]
    fn eval(vm: &mut VM<'_>) -> Result<OpcodeResult, VMError> {
        vm.current_call_frame.increase_consumed_gas(gas_cost::ADD)?;

        let (lhs, rhs) = vm.current_call_frame.stack.pop1_and_top_mut()?;
        *rhs = lhs.overflowing_add(*rhs).0;

        Ok(OpcodeResult::Continue)
    }
}

/// Implementation for the `SUB` opcode.
pub struct OpSubHandler;
impl OpcodeHandler for OpSubHandler {
    #[inline(always)]
    fn eval(vm: &mut VM<'_>) -> Result<OpcodeResult, VMError> {
        vm.current_call_frame.increase_consumed_gas(gas_cost::SUB)?;

        let (lhs, rhs) = vm.current_call_frame.stack.pop1_and_top_mut()?;
        *rhs = lhs.overflowing_sub(*rhs).0;

        Ok(OpcodeResult::Continue)
    }
}

/// Implementation for the `MUL` opcode.
pub struct OpMulHandler;
impl OpcodeHandler for OpMulHandler {
    #[inline(always)]
    fn eval(vm: &mut VM<'_>) -> Result<OpcodeResult, VMError> {
        vm.current_call_frame.increase_consumed_gas(gas_cost::MUL)?;

        let (lhs, rhs) = vm.current_call_frame.stack.pop1_and_top_mut()?;
        *rhs = lhs.overflowing_mul(*rhs).0;

        Ok(OpcodeResult::Continue)
    }
}

/// EVM unsigned division specialized for native-width divisors.
#[inline(always)]
pub fn div_with_u64_fast_path(lhs: U256, rhs: U256) -> U256 {
    if (rhs.0[1] | rhs.0[2] | rhs.0[3]) == 0 {
        let divisor = rhs.0[0];
        if divisor == 0 {
            return U256::zero();
        }
        if (lhs.0[1] | lhs.0[2] | lhs.0[3]) == 0 {
            return U256::from(lhs.0[0] / divisor);
        }
        if divisor.is_power_of_two() {
            return lhs >> divisor.trailing_zeros();
        }
        if divisor <= u32::MAX as u64 {
            return div_by_u32(lhs, divisor);
        }
        return div_by_wide_u64(lhs, divisor);
    }
    div_full_width_u32(lhs, rhs)
}

#[inline(never)]
fn div_by_u32(mut value: U256, divisor: u64) -> U256 {
    // Base-2^32 long division: remainder < divisor <= u32::MAX,
    // so combining it with the next digit always fits in a u64.
    let mut remainder = 0u64;
    for word in value.0.iter_mut().rev() {
        let high = (remainder << 32) | (*word >> 32);
        let q_high = high / divisor;
        remainder = high % divisor;
        let low = (remainder << 32) | (*word & u32::MAX as u64);
        let q_low = low / divisor;
        remainder = low % divisor;
        *word = (q_high << 32) | q_low;
    }
    value
}

/// Base-2^32 long division by a two-digit divisor, using only u64 arithmetic.
#[inline(never)]
fn div_by_wide_u64(value: U256, divisor: u64) -> U256 {
    debug_assert!(divisor > u32::MAX as u64);
    const BASE: u64 = 1u64 << 32;
    let shift = divisor.leading_zeros();
    let normalized = divisor << shift;
    let high = normalized >> 32;
    let low = normalized & (BASE - 1);
    let mut digits = [0u32; 9];
    let mut carry = 0u64;
    for (i, digit) in digits[..8].iter_mut().enumerate() {
        let raw = (value.0[i / 2] >> (32 * (i % 2))) & (BASE - 1);
        let shifted = (raw << shift) | carry;
        *digit = shifted as u32;
        carry = shifted >> 32;
    }
    digits[8] = carry as u32;
    let mut remainder = u64::from(digits[8]);
    for digit in digits[..8].iter_mut().rev() {
        let next = u64::from(*digit);
        if remainder < high {
            // remainder*BASE + next < high*BASE <= normalized, so q is zero.
            remainder = (remainder << 32) | next;
            *digit = 0;
            continue;
        }
        // high >= BASE/2. The upper two numerator digits give an upper
        // estimate; cap at one digit, then correct while the product is too big.
        let mut quotient = (remainder / high).min(BASE - 1);
        let mut estimate_remainder = remainder - quotient * high;
        while estimate_remainder < BASE && quotient * low > ((estimate_remainder << 32) | next) {
            quotient -= 1;
            estimate_remainder += high;
        }
        // The exact remainder is < normalized < 2^64, so subtracting the
        // low 64 bits of the 96-bit numerator/product yields that remainder.
        remainder = ((remainder << 32) | next).wrapping_sub(quotient.wrapping_mul(normalized));
        *digit = quotient as u32;
    }
    U256([
        u64::from(digits[0]) | (u64::from(digits[1]) << 32),
        u64::from(digits[2]) | (u64::from(digits[3]) << 32),
        u64::from(digits[4]) | (u64::from(digits[5]) << 32),
        u64::from(digits[6]) | (u64::from(digits[7]) << 32),
    ])
}

/// Long division in base 2^32 for divisors with at least three digits.
#[inline(never)]
fn div_full_width_u32(lhs: U256, rhs: U256) -> U256 {
    debug_assert!((rhs.0[1] | rhs.0[2] | rhs.0[3]) != 0);
    if lhs < rhs {
        return U256::zero();
    }
    const BASE: u64 = 1u64 << 32;
    let mut v: [u32; 8] = core::array::from_fn(|i| (rhs.0[i / 2] >> (32 * (i % 2))) as u32);
    let raw: [u32; 8] = core::array::from_fn(|i| (lhs.0[i / 2] >> (32 * (i % 2))) as u32);
    let n = v
        .iter()
        .rposition(|&digit| digit != 0)
        .expect("nonzero divisor")
        + 1;
    let len = raw
        .iter()
        .rposition(|&digit| digit != 0)
        .expect("nonzero numerator")
        + 1;
    let shift = v[n - 1].leading_zeros();
    let mut carry = 0u64;
    for digit in &mut v[..n] {
        let wide = (u64::from(*digit) << shift) | carry;
        *digit = wide as u32;
        carry = wide >> 32;
    }
    let mut u = [0u32; 9];
    carry = 0;
    for i in 0..len {
        let wide = (u64::from(raw[i]) << shift) | carry;
        u[i] = wide as u32;
        carry = wide >> 32;
    }
    u[len] = carry as u32;
    let mut q = [0u32; 8];
    let high = u64::from(v[n - 1]);
    let second = u64::from(v[n - 2]);
    for j in (0..=len - n).rev() {
        let top = (u64::from(u[j + n]) << 32) | u64::from(u[j + n - 1]);
        let mut estimate = (top / high).min(BASE - 1);
        let mut remainder = top - estimate * high;
        while remainder < BASE && estimate * second > ((remainder << 32) | u64::from(u[j + n - 2]))
        {
            estimate -= 1;
            remainder += high;
        }
        // Subtract estimate*v. Each product plus the previous carry/borrow fits u64.
        let mut borrow = 0u64;
        for i in 0..n {
            let product = estimate * u64::from(v[i]) + borrow;
            let (digit, underflow) = u[j + i].overflowing_sub(product as u32);
            u[j + i] = digit;
            borrow = (product >> 32) + u64::from(underflow);
        }
        let (last, underflow) = u[j + n].overflowing_sub(borrow as u32);
        u[j + n] = last;
        if underflow {
            // The estimate was still one too large: decrement and add v back.
            estimate -= 1;
            let mut carry = 0u64;
            for i in 0..n {
                let sum = u64::from(u[j + i]) + u64::from(v[i]) + carry;
                u[j + i] = sum as u32;
                carry = sum >> 32;
            }
            u[j + n] = u[j + n].wrapping_add(carry as u32);
        }
        q[j] = estimate as u32;
    }
    U256([
        u64::from(q[0]) | (u64::from(q[1]) << 32),
        u64::from(q[2]) | (u64::from(q[3]) << 32),
        u64::from(q[4]) | (u64::from(q[5]) << 32),
        u64::from(q[6]) | (u64::from(q[7]) << 32),
    ])
}

/// Implementation for the `DIV` opcode.
pub struct OpDivHandler;
impl OpcodeHandler for OpDivHandler {
    #[inline(always)]
    fn eval(vm: &mut VM<'_>) -> Result<OpcodeResult, VMError> {
        vm.current_call_frame.increase_consumed_gas(gas_cost::DIV)?;

        let (lhs, rhs) = vm.current_call_frame.stack.pop1_and_top_mut()?;
        *rhs = div_with_u64_fast_path(lhs, *rhs);

        Ok(OpcodeResult::Continue)
    }
}

/// Implementation for the `SDIV` opcode.
pub struct OpSDivHandler;
impl OpcodeHandler for OpSDivHandler {
    #[inline(always)]
    fn eval(vm: &mut VM<'_>) -> Result<OpcodeResult, VMError> {
        vm.current_call_frame
            .increase_consumed_gas(gas_cost::SDIV)?;

        let (top, slot) = vm.current_call_frame.stack.pop1_and_top_mut()?;
        let mut lhs = top;
        let mut rhs = *slot;

        let mut sign = false;
        if lhs.bit(255) {
            lhs = U256::zero().overflowing_sub(lhs).0;
            sign = !sign;
        }
        if rhs.bit(255) {
            rhs = U256::zero().overflowing_sub(rhs).0;
            sign = !sign;
        }

        *slot = match lhs.checked_div(rhs) {
            Some(mut res) => {
                if sign {
                    res = U256::zero().overflowing_sub(res).0;
                }
                res
            }
            None => U256::zero(),
        };

        Ok(OpcodeResult::Continue)
    }
}

/// EVM remainder with native-width and power-of-two specializations.
#[inline(always)]
pub fn rem_with_u64_fast_path(lhs: U256, rhs: U256) -> U256 {
    if (rhs.0[1] | rhs.0[2] | rhs.0[3]) == 0 {
        let divisor = rhs.0[0];
        if divisor == 0 {
            return U256::zero();
        }
        if (lhs.0[1] | lhs.0[2] | lhs.0[3]) == 0 {
            return U256::from(lhs.0[0] % divisor);
        }
        if divisor.is_power_of_two() {
            return U256::from(lhs.0[0] & (divisor - 1));
        }
        if divisor <= u32::MAX as u64 {
            let mut remainder = 0u64;
            for &word in lhs.0.iter().rev() {
                remainder = ((remainder << 32) | (word >> 32)) % divisor;
                remainder = ((remainder << 32) | (word & u32::MAX as u64)) % divisor;
            }
            return U256::from(remainder);
        }
    }
    lhs.checked_rem(rhs).unwrap_or_default()
}

/// Implementation for the `MOD` opcode.
pub struct OpModHandler;
impl OpcodeHandler for OpModHandler {
    #[inline(always)]
    fn eval(vm: &mut VM<'_>) -> Result<OpcodeResult, VMError> {
        vm.current_call_frame.increase_consumed_gas(gas_cost::MOD)?;

        let (lhs, rhs) = vm.current_call_frame.stack.pop1_and_top_mut()?;
        *rhs = rem_with_u64_fast_path(lhs, *rhs);

        Ok(OpcodeResult::Continue)
    }
}

/// Implementation for the `SMOD` opcode.
pub struct OpSModHandler;
impl OpcodeHandler for OpSModHandler {
    #[inline(always)]
    fn eval(vm: &mut VM<'_>) -> Result<OpcodeResult, VMError> {
        vm.current_call_frame
            .increase_consumed_gas(gas_cost::SMOD)?;

        let (top, slot) = vm.current_call_frame.stack.pop1_and_top_mut()?;
        let mut lhs = top;
        let mut rhs = *slot;

        let sign = lhs.bit(255);
        if sign {
            (lhs, _) = (!lhs).overflowing_add(U256::one());
        }
        if rhs.bit(255) {
            (rhs, _) = (!rhs).overflowing_add(U256::one());
        }

        *slot = match lhs.checked_rem(rhs) {
            Some(mut res) => {
                if sign {
                    (res, _) = (!res).overflowing_add(U256::one());
                }
                res
            }
            None => U256::zero(),
        };

        Ok(OpcodeResult::Continue)
    }
}

/// Implementation for the `ADDMOD` opcode.
pub struct OpAddModHandler;
impl OpcodeHandler for OpAddModHandler {
    #[inline(always)]
    fn eval(vm: &mut VM<'_>) -> Result<OpcodeResult, VMError> {
        vm.current_call_frame
            .increase_consumed_gas(gas_cost::ADDMOD)?;

        let [lhs, rhs, r#mod] = *vm.current_call_frame.stack.pop()?;
        if r#mod.is_zero() || r#mod == U256::one() {
            vm.current_call_frame.stack.push_zero()?;
        } else {
            #[expect(
                clippy::arithmetic_side_effects,
                reason = "mod is checked non-zero above"
            )]
            let res = U512::from(lhs).overflowing_add(rhs.into()).0 % r#mod;
            vm.current_call_frame
                .stack
                .push(U256([res.0[0], res.0[1], res.0[2], res.0[3]]))?;
        }

        Ok(OpcodeResult::Continue)
    }
}

/// Implementation for the `MULMOD` opcode.
pub struct OpMulModHandler;
impl OpcodeHandler for OpMulModHandler {
    #[inline(always)]
    fn eval(vm: &mut VM<'_>) -> Result<OpcodeResult, VMError> {
        vm.current_call_frame
            .increase_consumed_gas(gas_cost::MULMOD)?;

        let [multiplicand, multiplier, modulus] = *vm.current_call_frame.stack.pop()?;
        if modulus.is_zero() || multiplicand.is_zero() || multiplier.is_zero() {
            vm.current_call_frame.stack.push_zero()?;
        } else {
            let a_bytes = multiplicand.to_big_endian();
            let b_bytes = multiplier.to_big_endian();
            let m_bytes = modulus.to_big_endian();
            let result_bytes = vm.crypto.mulmod256(&a_bytes, &b_bytes, &m_bytes);
            let product_mod = U256::from_big_endian(&result_bytes);
            vm.current_call_frame.stack.push(product_mod)?;
        }

        Ok(OpcodeResult::Continue)
    }
}

/// Implementation for the `EXP` opcode.
pub struct OpExpHandler;
impl OpcodeHandler for OpExpHandler {
    #[inline(always)]
    fn eval(vm: &mut VM<'_>) -> Result<OpcodeResult, VMError> {
        let [base, exp] = *vm.current_call_frame.stack.pop()?;
        vm.current_call_frame
            .increase_consumed_gas(gas_cost::exp(exp)?)?;

        let (res, _) = base.overflowing_pow(exp);
        vm.current_call_frame.stack.push(res)?;

        Ok(OpcodeResult::Continue)
    }
}

/// Implementation for the `SIGNEXTEND` opcode.
pub struct OpSignExtendHandler;
impl OpcodeHandler for OpSignExtendHandler {
    #[inline(always)]
    fn eval(vm: &mut VM<'_>) -> Result<OpcodeResult, VMError> {
        vm.current_call_frame
            .increase_consumed_gas(gas_cost::SIGNEXTEND)?;

        let (index, slot) = vm.current_call_frame.stack.pop1_and_top_mut()?;
        let mut value = *slot;
        *slot = match usize::try_from(index) {
            #[expect(
                clippy::arithmetic_side_effects,
                reason = "x < 32 guard prevents overflow"
            )]
            Ok(x) if x < 32 => {
                if value.bit(8 * x + 7) {
                    value |= U256::MAX << (8 * (x + 1));
                } else if x != 31 {
                    value &= (U256::one() << (8 * (x + 1))) - 1;
                }

                value
            }
            _ => value,
        };

        Ok(OpcodeResult::Continue)
    }
}

/// Implementation for the `CLZ` opcode.
pub struct OpClzHandler;
impl OpcodeHandler for OpClzHandler {
    #[inline(always)]
    fn eval(vm: &mut VM<'_>) -> Result<OpcodeResult, VMError> {
        vm.current_call_frame.increase_consumed_gas(gas_cost::CLZ)?;

        let slot = vm.current_call_frame.stack.top_mut()?;
        let lz = slot.leading_zeros();
        *slot = lz.into();

        Ok(OpcodeResult::Continue)
    }
}
