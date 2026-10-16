use crate::errors::{ByteConversionError, CreationError};
use crate::field::errors::FieldError;
use crate::field::traits::IsField;
use crate::traits::ByteConversion;
use crate::unsigned_integer::traits::IsUnsignedInteger;
#[cfg(feature = "alloc")]
use alloc::{format, string::String};
#[cfg(any(
    feature = "lambdaworks-serde-binary",
    feature = "lambdaworks-serde-string"
))]
use core::fmt;
use core::fmt::Debug;
use core::iter::Sum;
#[cfg(any(
    feature = "lambdaworks-serde-binary",
    feature = "lambdaworks-serde-string"
))]
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub};
use num_bigint::BigUint;
use num_traits::Num;
#[cfg(any(
    feature = "lambdaworks-serde-binary",
    feature = "lambdaworks-serde-string"
))]
use serde::Deserialize;
#[cfg(any(
    feature = "lambdaworks-serde-binary",
    feature = "lambdaworks-serde-string"
))]
use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
#[cfg(any(
    feature = "lambdaworks-serde-binary",
    feature = "lambdaworks-serde-string"
))]
use serde::ser::{Serialize, SerializeStruct, Serializer};

use super::traits::{IsPrimeField, IsSubFieldOf, LegendreSymbol};

/// A field element with operations algorithms defined in `F`
///
/// `#[repr(transparent)]` makes `FieldElement<F>` byte-identical to
/// `F::BaseType`, which [`SpillSafe`](crate::spill_safe::SpillSafe)
/// requires. Changing the `repr` or adding fields breaks this and
/// is UB in any function that requires `T: SpillSafe`.
#[allow(clippy::derived_hash_with_manual_eq)]
#[repr(transparent)]
#[derive(Debug, Clone, Hash, Copy)]
pub struct FieldElement<F: IsField> {
    value: F::BaseType,
}

#[cfg(feature = "alloc")]
impl<F: IsField> FieldElement<F> {
    // Source: https://en.wikipedia.org/wiki/Modular_multiplicative_inverse#Multiple_inverses
    /// Computes the multiplicative inverses of a slice of field elements
    /// The algorithm just performs one inversion and several multiplications and should be used
    /// when wanting to invert several elements together.
    ///
    /// On `Err(InvZeroError)` the input slice is left unchanged (all-or-nothing).
    /// The parallel path enforces this with a zero pre-scan; the sequential
    /// path checks before any mutation.
    pub fn inplace_batch_inverse(numbers: &mut [Self]) -> Result<(), FieldError> {
        #[cfg(feature = "parallel")]
        {
            // Montgomery batch inverse has a serial prefix-product dependency, but
            // chunks are independent — each chunk inverts its own elements without
            // needing values from other chunks. Trade K-1 extra field inversions
            // (negligible vs ~2N mults per chunk) for K-way parallelism.
            const PARALLEL_BATCH_INV_THRESHOLD: usize = 1 << 16;
            if numbers.len() >= PARALLEL_BATCH_INV_THRESHOLD {
                use rayon::prelude::*;
                // Pre-scan for zeros so the mutation step is all-or-nothing.
                // Without this, a chunk containing zero would return Err while
                // sibling chunks may have already overwritten their elements.
                let zero = Self::zero();
                if numbers.par_iter().any(|x| x == &zero) {
                    return Err(FieldError::InvZeroError);
                }
                let chunk_size = numbers.len().div_ceil(rayon::current_num_threads().max(1));
                return numbers
                    .par_chunks_mut(chunk_size)
                    .try_for_each(Self::inplace_batch_inverse_sequential);
            }
        }
        Self::inplace_batch_inverse_sequential(numbers)
    }

    /// Single-threaded batch inversion. Callers that run inside a lazy-init
    /// cell (e.g. `OnceLock::get_or_init`) MUST use this variant: the parallel
    /// one farms work to the rayon pool, and if pool workers are blocked
    /// waiting on that same cell the initializer starves and the prove
    /// deadlocks.
    pub fn inplace_batch_inverse_sequential(numbers: &mut [Self]) -> Result<(), FieldError> {
        if numbers.is_empty() {
            return Ok(());
        }
        let count = numbers.len();
        let mut prod_prefix = alloc::vec::Vec::with_capacity(count);
        prod_prefix.push(numbers[0].clone());
        for i in 1..count {
            prod_prefix.push(&prod_prefix[i - 1] * &numbers[i]);
        }
        let mut bi_inv = prod_prefix[count - 1].inv()?;
        for i in (1..count).rev() {
            let ai_inv = &bi_inv * &prod_prefix[i - 1];
            bi_inv = &bi_inv * &numbers[i];
            numbers[i] = ai_inv;
        }
        numbers[0] = bi_inv;
        Ok(())
    }

    #[inline(always)]
    pub fn to_subfield_vec<S>(self) -> alloc::vec::Vec<FieldElement<S>>
    where
        S: IsSubFieldOf<F>,
    {
        S::to_subfield_vec(self.value)
            .into_iter()
            .map(|x| FieldElement::from_raw(x))
            .collect()
    }
}

/// From overloading for field elements
impl<F> From<&F::BaseType> for FieldElement<F>
where
    F::BaseType: Clone,
    F: IsField,
{
    fn from(value: &F::BaseType) -> Self {
        Self {
            value: F::from_base_type(value.clone()),
        }
    }
}

/// From overloading for U64
impl<F> From<u64> for FieldElement<F>
where
    F: IsField,
{
    fn from(value: u64) -> Self {
        Self {
            value: F::from_u64(value),
        }
    }
}

/// From overloading for i64.
/// Negative values are converted to their field equivalents: -x becomes p - x.
impl<F> From<i64> for FieldElement<F>
where
    F: IsField,
{
    fn from(value: i64) -> Self {
        if value >= 0 {
            Self::from(value as u64)
        } else {
            -Self::from(value.unsigned_abs())
        }
    }
}

/// From overloading for i32 (convenience for integer literals).
impl<F> From<i32> for FieldElement<F>
where
    F: IsField,
{
    fn from(value: i32) -> Self {
        Self::from(value as i64)
    }
}

#[cfg(feature = "alloc")]
/// From overloading for BigUint.
/// Creates a field element from a BigUint that is smaller than the modulus.
/// Returns error if the BigUint value is bigger than the modulus.
impl<F> TryFrom<BigUint> for FieldElement<F>
where
    Self: ByteConversion,
    F: IsPrimeField,
{
    type Error = ByteConversionError;
    fn try_from(value: BigUint) -> Result<Self, ByteConversionError> {
        FieldElement::<F>::from_reduced_big_uint(&value)
    }
}

impl<F> FieldElement<F>
where
    F::BaseType: Clone,
    F: IsField,
{
    pub fn from_raw(value: F::BaseType) -> Self {
        Self { value }
    }

    pub const fn const_from_raw(value: F::BaseType) -> Self {
        Self { value }
    }
}

/// Equality operator overloading for field elements
impl<F> PartialEq<FieldElement<F>> for FieldElement<F>
where
    F: IsField,
{
    fn eq(&self, other: &FieldElement<F>) -> bool {
        F::eq(&self.value, &other.value)
    }
}

impl<F> Eq for FieldElement<F> where F: IsField {}

/// Addition operator overloading for field elements
impl<F, L> Add<&FieldElement<L>> for &FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = FieldElement<L>;

    fn add(self, rhs: &FieldElement<L>) -> Self::Output {
        Self::Output {
            value: <F as IsSubFieldOf<L>>::add(&self.value, &rhs.value),
        }
    }
}

impl<F, L> Add<FieldElement<L>> for FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = FieldElement<L>;

    fn add(self, rhs: FieldElement<L>) -> Self::Output {
        &self + &rhs
    }
}

impl<F, L> Add<&FieldElement<L>> for FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = FieldElement<L>;

    fn add(self, rhs: &FieldElement<L>) -> Self::Output {
        &self + rhs
    }
}

impl<F, L> Add<FieldElement<L>> for &FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = FieldElement<L>;

    fn add(self, rhs: FieldElement<L>) -> Self::Output {
        self + &rhs
    }
}

/// AddAssign operator overloading for field elements
impl<F, L> AddAssign<FieldElement<F>> for FieldElement<L>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    fn add_assign(&mut self, rhs: FieldElement<F>) {
        self.value = <F as IsSubFieldOf<L>>::add(&rhs.value, &self.value);
    }
}

/// Sum operator for field elements
impl<F> Sum<FieldElement<F>> for FieldElement<F>
where
    F: IsField,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |augend, addend| augend + addend)
    }
}

/// Subtraction operator overloading for field elements*/
impl<F, L> Sub<&FieldElement<L>> for &FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = FieldElement<L>;

    fn sub(self, rhs: &FieldElement<L>) -> Self::Output {
        Self::Output {
            value: <F as IsSubFieldOf<L>>::sub(&self.value, &rhs.value),
        }
    }
}

impl<F, L> Sub<FieldElement<L>> for FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = FieldElement<L>;

    fn sub(self, rhs: FieldElement<L>) -> Self::Output {
        &self - &rhs
    }
}

impl<F, L> Sub<&FieldElement<L>> for FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = FieldElement<L>;

    fn sub(self, rhs: &FieldElement<L>) -> Self::Output {
        &self - rhs
    }
}

impl<F, L> Sub<FieldElement<L>> for &FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = FieldElement<L>;

    fn sub(self, rhs: FieldElement<L>) -> Self::Output {
        self - &rhs
    }
}

/// Multiplication operator overloading for field elements*/
impl<F, L> Mul<&FieldElement<L>> for &FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = FieldElement<L>;

    fn mul(self, rhs: &FieldElement<L>) -> Self::Output {
        Self::Output {
            value: <F as IsSubFieldOf<L>>::mul(&self.value, &rhs.value),
        }
    }
}

impl<F, L> Mul<FieldElement<L>> for FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = FieldElement<L>;

    fn mul(self, rhs: FieldElement<L>) -> Self::Output {
        &self * &rhs
    }
}

impl<F, L> Mul<&FieldElement<L>> for FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = FieldElement<L>;

    fn mul(self, rhs: &FieldElement<L>) -> Self::Output {
        &self * rhs
    }
}

impl<F, L> Mul<FieldElement<L>> for &FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = FieldElement<L>;

    fn mul(self, rhs: FieldElement<L>) -> Self::Output {
        self * &rhs
    }
}

/// MulAssign operator overloading for field elements
impl<F, L> MulAssign<FieldElement<F>> for FieldElement<L>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    fn mul_assign(&mut self, rhs: FieldElement<F>) {
        self.value = <F as IsSubFieldOf<L>>::mul(&rhs.value, &self.value);
    }
}

/// MulAssign operator overloading for field elements
impl<F, L> MulAssign<&FieldElement<F>> for FieldElement<L>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    fn mul_assign(&mut self, rhs: &FieldElement<F>) {
        self.value = <F as IsSubFieldOf<L>>::mul(&rhs.value, &self.value);
    }
}

/// Division operator overloading for field elements*/
impl<F, L> Div<&FieldElement<L>> for &FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = Result<FieldElement<L>, FieldError>;

    fn div(self, rhs: &FieldElement<L>) -> Self::Output {
        let value = <F as IsSubFieldOf<L>>::div(&self.value, &rhs.value)?;
        Ok(FieldElement::<L> { value })
    }
}

impl<F, L> Div<FieldElement<L>> for FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = Result<FieldElement<L>, FieldError>;

    fn div(self, rhs: FieldElement<L>) -> Self::Output {
        &self / &rhs
    }
}

impl<F, L> Div<&FieldElement<L>> for FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = Result<FieldElement<L>, FieldError>;

    fn div(self, rhs: &FieldElement<L>) -> Self::Output {
        &self / rhs
    }
}

impl<F, L> Div<FieldElement<L>> for &FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = Result<FieldElement<L>, FieldError>;

    fn div(self, rhs: FieldElement<L>) -> Self::Output {
        self / &rhs
    }
}

/// Negation operator overloading for field elements*/
impl<F> Neg for &FieldElement<F>
where
    F: IsField,
{
    type Output = FieldElement<F>;

    fn neg(self) -> Self::Output {
        Self::Output {
            value: F::neg(&self.value),
        }
    }
}

impl<F> Neg for FieldElement<F>
where
    F: IsField,
{
    type Output = FieldElement<F>;

    fn neg(self) -> Self::Output {
        -&self
    }
}

impl<F> Default for FieldElement<F>
where
    F: IsField,
{
    fn default() -> Self {
        Self { value: F::zero() }
    }
}

/// FieldElement general implementation
/// Most of this is delegated to the trait `F` that
/// implements the field operations.
impl<F> FieldElement<F>
where
    F: IsField,
{
    /// Creates a field element from `value`
    #[inline(always)]
    pub fn new(value: F::BaseType) -> Self {
        Self {
            value: F::from_base_type(value),
        }
    }

    /// Returns the underlying `value`
    #[inline(always)]
    pub fn value(&self) -> &F::BaseType {
        &self.value
    }

    /// Returns the multiplicative inverse of `self`
    #[inline(always)]
    pub fn inv(&self) -> Result<Self, FieldError> {
        let value = F::inv(&self.value)?;
        Ok(Self { value })
    }

    /// Returns the square of `self`
    #[inline(always)]
    pub fn square(&self) -> Self {
        Self {
            value: F::square(&self.value),
        }
    }

    /// Returns the double of `self`
    #[inline(always)]
    pub fn double(&self) -> Self {
        Self {
            value: F::double(&self.value),
        }
    }

    /// Returns `self` raised to the power of `exponent`
    #[inline(always)]
    pub fn pow<T>(&self, exponent: T) -> Self
    where
        T: IsUnsignedInteger,
    {
        Self {
            value: F::pow(&self.value, exponent),
        }
    }

    /// Returns the multiplicative neutral element of the field.
    #[inline(always)]
    pub fn one() -> Self {
        Self { value: F::one() }
    }

    /// Returns the additive neutral element of the field.
    #[inline(always)]
    pub fn zero() -> Self {
        Self { value: F::zero() }
    }

    /// Returns the raw base type
    pub fn to_raw(self) -> F::BaseType {
        self.value
    }

    #[inline(always)]
    pub fn to_extension<L: IsField>(self) -> FieldElement<L>
    where
        F: IsSubFieldOf<L>,
    {
        FieldElement {
            value: <F as IsSubFieldOf<L>>::embed(self.value),
        }
    }

    /// Compute `self - rhs` where `rhs` is in a subfield `S` of `F`.
    ///
    /// Uses mixed F-S arithmetic: computes `self - embed(rhs)` without
    /// explicitly converting rhs to the extension field.
    #[inline(always)]
    pub fn sub_subfield<S: IsSubFieldOf<F>>(&self, rhs: &FieldElement<S>) -> Self {
        // embed(rhs) - self gives the negation of what we want, in F.
        // Then negate to get self - embed(rhs).
        Self {
            value: F::neg(&<S as IsSubFieldOf<F>>::sub(&rhs.value, &self.value)),
        }
    }

    #[cfg(feature = "alloc")]
    /// Creates a field element from a BigUint that is smaller than the modulus.
    /// Returns error if the value is bigger than the modulus.
    pub fn from_reduced_big_uint(value: &BigUint) -> Result<Self, ByteConversionError>
    where
        Self: ByteConversion,
        F: IsPrimeField,
    {
        let mod_minus_one = format!("{:x}", F::modulus_minus_one());

        let modulus = BigUint::from_str_radix(&mod_minus_one, 16)
            .expect("invalid modulus representation")
            + 1u32;

        if value >= &modulus {
            Err(ByteConversionError::ValueNotReduced)
        } else {
            let mut bytes = value.to_bytes_le();
            // We pad the bytes to the size of the base type to be able to apply `from_bytes_le`.
            bytes.resize(core::mem::size_of::<F::BaseType>(), 0);
            Self::from_bytes_le(&bytes)
        }
    }

    #[cfg(feature = "alloc")]
    /// Converts a field element into a BigUint.
    pub fn to_big_uint(&self) -> BigUint
    where
        Self: ByteConversion,
    {
        BigUint::from_bytes_be(&self.to_bytes_be())
    }

    #[cfg(feature = "alloc")]
    /// Converts a hex string into a field element.
    /// It returns error if the hex value is larger than the modulus.
    pub fn from_hex_str(hex: &str) -> Result<Self, CreationError>
    where
        Self: ByteConversion,
        F: IsPrimeField,
    {
        let hex_str = hex.strip_prefix("0x").unwrap_or(hex);
        if hex_str.is_empty() {
            return Err(CreationError::EmptyString);
        }

        let value =
            BigUint::from_str_radix(hex_str, 16).map_err(|_| CreationError::InvalidHexString)?;

        Self::from_reduced_big_uint(&value).map_err(|_| CreationError::InvalidHexString)
    }

    #[cfg(feature = "alloc")]
    /// Converts a field element into a hex string.
    pub fn to_hex_str(&self) -> String
    where
        Self: ByteConversion,
    {
        format!("0x{:02X}", self.to_big_uint())
    }
}

impl<F: IsPrimeField> FieldElement<F> {
    /// Returns the canonical form of the value stored
    pub fn canonical(&self) -> F::CanonicalType {
        F::canonical(self.value())
    }

    /// Returns the two square roots of a field element, provided it exists
    /// The function returns the roots whenever the field element is a quadratic residue modulo p
    pub fn sqrt(&self) -> Option<(Self, Self)> {
        let sqrts = F::sqrt(&self.value);
        sqrts.map(|(sqrt1, sqrt2)| (Self { value: sqrt1 }, Self { value: sqrt2 }))
    }

    /// Returns the Legendre symbol of a field element modulo p
    pub fn legendre_symbol(&self) -> LegendreSymbol {
        F::legendre_symbol(&self.value)
    }

    /// Creates a `FieldElement` from a hexstring. It can contain `0x` or not.
    /// Returns an `CreationError::InvalidHexString`if the value is not a hexstring.
    /// Returns a `CreationError::EmptyString` if the input string is empty.
    /// Returns a `CreationError::HexStringIsTooBig` if the the input hex string is bigger than the
    /// maximum amount of characters for this element.
    /// Returns a `CreationError::CanonicalOutOfRange` if the canonical form of the value is
    /// out of the range [0, p-1] where p is the modulus.
    pub fn from_hex(hex_string: &str) -> Result<Self, CreationError> {
        if hex_string.is_empty() {
            return Err(CreationError::EmptyString);
        }
        let value = F::from_hex(hex_string)?;
        Ok(Self { value })
    }

    #[cfg(feature = "std")]
    /// Creates a hexstring from a `FieldElement` without `0x`.
    pub fn to_hex(&self) -> String {
        F::to_hex(&self.value)
    }
}

#[cfg(feature = "lambdaworks-serde-binary")]
impl<F> Serialize for FieldElement<F>
where
    F: IsField,
    F::BaseType: ByteConversion,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("FieldElement", 1)?;
        let data = self.value().to_bytes_be();
        state.serialize_field("value", &data)?;
        state.end()
    }
}

#[cfg(all(
    feature = "lambdaworks-serde-string",
    not(feature = "lambdaworks-serde-binary")
))]
impl<F: IsPrimeField> Serialize for FieldElement<F> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use crate::alloc::string::ToString;
        let mut state = serializer.serialize_struct("FieldElement", 1)?;
        state.serialize_field("value", &F::canonical(self.value()).to_string())?;
        state.end()
    }
}

#[cfg(feature = "lambdaworks-serde-binary")]
impl<'de, F> Deserialize<'de> for FieldElement<F>
where
    F: IsField,
    F::BaseType: ByteConversion,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Value,
        }

        struct FieldElementVisitor<F>(PhantomData<fn() -> F>);

        impl<'de, F: IsField> Visitor<'de> for FieldElementVisitor<F> {
            type Value = FieldElement<F>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct FieldElement")
            }

            fn visit_map<M>(self, mut map: M) -> Result<FieldElement<F>, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut value: Option<alloc::vec::Vec<u8>> = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Value => {
                            if value.is_some() {
                                return Err(de::Error::duplicate_field("value"));
                            }
                            value = Some(map.next_value()?);
                        }
                    }
                }
                let value = value.ok_or_else(|| de::Error::missing_field("value"))?;
                let val = F::BaseType::from_bytes_be(&value).unwrap();
                Ok(FieldElement::from_raw(val))
            }

            fn visit_seq<S>(self, mut seq: S) -> Result<FieldElement<F>, S::Error>
            where
                S: SeqAccess<'de>,
            {
                let mut value: Option<alloc::vec::Vec<u8>> = None;
                while let Some(val) = seq.next_element()? {
                    if value.is_some() {
                        return Err(de::Error::duplicate_field("value"));
                    }
                    value = Some(val);
                }
                let value = value.ok_or_else(|| de::Error::missing_field("value"))?;
                let val = F::BaseType::from_bytes_be(&value).unwrap();
                Ok(FieldElement::from_raw(val))
            }
        }

        const FIELDS: &[&str] = &["value"];
        deserializer.deserialize_struct("FieldElement", FIELDS, FieldElementVisitor(PhantomData))
    }
}

#[cfg(all(
    feature = "lambdaworks-serde-string",
    not(feature = "lambdaworks-serde-binary")
))]
impl<'de, F: IsPrimeField> Deserialize<'de> for FieldElement<F> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Value,
        }

        struct FieldElementVisitor<F>(PhantomData<fn() -> F>);

        impl<'de, F: IsPrimeField> Visitor<'de> for FieldElementVisitor<F> {
            type Value = FieldElement<F>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct FieldElement")
            }

            fn visit_map<M>(self, mut map: M) -> Result<FieldElement<F>, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut value: Option<&str> = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Value => {
                            if value.is_some() {
                                return Err(de::Error::duplicate_field("value"));
                            }
                            value = Some(map.next_value()?);
                        }
                    }
                }
                let value = value.ok_or_else(|| de::Error::missing_field("value"))?;
                FieldElement::from_hex(&value).map_err(|_| de::Error::custom("invalid hex"))
            }

            fn visit_seq<S>(self, mut seq: S) -> Result<FieldElement<F>, S::Error>
            where
                S: SeqAccess<'de>,
            {
                let mut value: Option<&str> = None;
                while let Some(val) = seq.next_element()? {
                    if value.is_some() {
                        return Err(de::Error::duplicate_field("value"));
                    }
                    value = Some(val);
                }
                let value = value.ok_or_else(|| de::Error::missing_field("value"))?;
                FieldElement::from_hex(&value).map_err(|_| de::Error::custom("invalid hex"))
            }
        }

        const FIELDS: &[&str] = &["value"];
        deserializer.deserialize_struct("FieldElement", FIELDS, FieldElementVisitor(PhantomData))
    }
}

// ============================================================================
// rkyv zero-copy (de)serialization
// ============================================================================
//
// `FieldElement<F>` is `#[repr(transparent)]` over `F::BaseType`. Its archived
// form is a local `#[repr(transparent)]` newtype wrapping the archived form of
// `F::BaseType` (e.g. archived `u64` for Goldilocks, `[ArchivedFieldElement; 3]`
// for the cubic extension). Keeping it a LOCAL type (rather than reusing
// `<F::BaseType as Archive>::Archived` directly) is what lets us implement
// `Deserialize` without colliding with rkyv's blanket impls — while the
// transparent repr keeps the archived bytes identical to the base type, so the
// recursion verifier still reads field elements straight from the proof buffer.

/// Archived form of [`FieldElement<F>`]; see the module note above.
#[cfg(feature = "rkyv")]
#[repr(transparent)]
pub struct ArchivedFieldElement<F: IsField>
where
    F::BaseType: rkyv::Archive,
{
    value: <F::BaseType as rkyv::Archive>::Archived,
}

#[cfg(feature = "rkyv")]
const _: () = {
    use rkyv::{Archive, Deserialize, Place, Portable, Serialize};

    // SAFETY: `ArchivedFieldElement<F>` is `#[repr(transparent)]` over the base
    // type's archived form, which is itself `Portable` (required by `Archive`).
    // A transparent wrapper over a `Portable` type is position-independent and
    // valid for the same byte patterns, so it is `Portable` too.
    unsafe impl<F> Portable for ArchivedFieldElement<F>
    where
        F: IsField,
        F::BaseType: Archive,
        <F::BaseType as Archive>::Archived: Portable,
    {
    }

    impl<F> Archive for FieldElement<F>
    where
        F: IsField,
        F::BaseType: Archive,
    {
        type Archived = ArchivedFieldElement<F>;
        type Resolver = <F::BaseType as Archive>::Resolver;

        #[inline]
        fn resolve(&self, resolver: Self::Resolver, out: Place<Self::Archived>) {
            // `ArchivedFieldElement` is `#[repr(transparent)]` over the base
            // type's archived form, so resolving into the inner field resolves
            // the whole newtype.
            let inner = unsafe { out.cast_unchecked::<<F::BaseType as Archive>::Archived>() };
            self.value.resolve(resolver, inner);
        }
    }

    impl<F, S> Serialize<S> for FieldElement<F>
    where
        F: IsField,
        F::BaseType: Serialize<S>,
        S: rkyv::rancor::Fallible + ?Sized,
    {
        #[inline]
        fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
            self.value.serialize(serializer)
        }
    }

    impl<F, D> Deserialize<FieldElement<F>, D> for ArchivedFieldElement<F>
    where
        F: IsField,
        F::BaseType: Archive,
        <F::BaseType as Archive>::Archived: Deserialize<F::BaseType, D>,
        D: rkyv::rancor::Fallible + ?Sized,
    {
        #[inline]
        fn deserialize(&self, deserializer: &mut D) -> Result<FieldElement<F>, D::Error> {
            Ok(FieldElement {
                value: self.value.deserialize(deserializer)?,
            })
        }
    }

    // SAFETY: `#[repr(transparent)]` over the inner archived value, so checking
    // the inner type's bytes checks the whole newtype.
    unsafe impl<F, C> rkyv::bytecheck::CheckBytes<C> for ArchivedFieldElement<F>
    where
        F: IsField,
        F::BaseType: Archive,
        <F::BaseType as Archive>::Archived: rkyv::bytecheck::CheckBytes<C>,
        C: rkyv::rancor::Fallible + ?Sized,
    {
        unsafe fn check_bytes(value: *const Self, context: &mut C) -> Result<(), C::Error> {
            unsafe {
                <<F::BaseType as Archive>::Archived as rkyv::bytecheck::CheckBytes<C>>::check_bytes(
                    value as *const <F::BaseType as Archive>::Archived,
                    context,
                )
            }
        }
    }
};

// ----------------------------------------------------------------------------
// Zero-copy native views (little-endian only)
// ----------------------------------------------------------------------------
//
// rkyv archives integers as `rend::*_le` types, which are `#[repr(C, align(N))]`
// and bit-identical to the native little-endian primitive. `FieldElement<F>` is
// `#[repr(transparent)]` over `F::BaseType` and `ArchivedFieldElement<F>` is
// `#[repr(transparent)]` over `<F::BaseType as Archive>::Archived`. So on a
// little-endian target the two types share size, alignment, and bit layout —
// an archived field element *is* a native field element. These views let the
// verifier read field elements straight out of the proof buffer with no copy
// and no allocation.
//
// Restricted to `target_endian = "little"` (the lambda-vm guest target). On a
// big-endian host these would be wrong, so they simply don't exist there.
// `IsField` is a public trait, so an arbitrary `F::BaseType: Archive` gives no
// guarantee that `Archived` shares size/align/layout with the base type —
// only rkyv's own primitive archived forms (and types built from them) do.
// `NativeArchived` is sealed to just those, so the views below are only
// callable for base types this crate has vetted.
#[cfg(all(feature = "rkyv", target_endian = "little"))]
mod sealed {
    pub trait Sealed {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl<F: super::IsField> Sealed for super::FieldElement<F> where F::BaseType: super::NativeArchived {}
    impl<T: super::NativeArchived, const N: usize> Sealed for [T; N] {}
}

/// See the module note above: implemented only for base types whose rkyv
/// `Archived` form is bit-identical to the native type on little-endian
/// targets (same size, same alignment, same byte layout).
///
/// # Safety
/// Implementors must guarantee `Self` and `Self::Archived` have identical
/// size and layout, and `Self`'s alignment is at least `Self::Archived`'s,
/// under `target_endian = "little"`.
#[cfg(all(feature = "rkyv", target_endian = "little"))]
pub unsafe trait NativeArchived: rkyv::Archive + sealed::Sealed {}

#[cfg(all(feature = "rkyv", target_endian = "little"))]
unsafe impl NativeArchived for u32 {}
#[cfg(all(feature = "rkyv", target_endian = "little"))]
unsafe impl NativeArchived for u64 {}
#[cfg(all(feature = "rkyv", target_endian = "little"))]
unsafe impl<F: IsField> NativeArchived for FieldElement<F> where F::BaseType: NativeArchived {}
#[cfg(all(feature = "rkyv", target_endian = "little"))]
unsafe impl<T: NativeArchived, const N: usize> NativeArchived for [T; N] {}

#[cfg(all(feature = "rkyv", target_endian = "little"))]
impl<F: IsField> ArchivedFieldElement<F>
where
    F::BaseType: NativeArchived,
{
    /// Reinterpret this archived element as a native [`FieldElement`] (no copy).
    ///
    /// Sound on little-endian: see the module note above.
    #[inline]
    pub fn as_native(&self) -> &FieldElement<F> {
        // SAFETY: identical size/align/bit-layout on little-endian.
        unsafe { &*(self as *const Self as *const FieldElement<F>) }
    }

    /// Reinterpret a slice of archived elements as a slice of native
    /// [`FieldElement`]s (no copy, no allocation).
    #[inline]
    pub fn slice_as_native(slice: &[Self]) -> &[FieldElement<F>] {
        // SAFETY: element-wise identical layout on little-endian, so the slice
        // (same length, same element stride) reinterprets directly.
        unsafe {
            core::slice::from_raw_parts(slice.as_ptr() as *const FieldElement<F>, slice.len())
        }
    }
}
