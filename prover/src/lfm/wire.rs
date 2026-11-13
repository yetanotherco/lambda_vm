//! A compact wire encoding of a W-LFM proof: no archive headers, no padding.
//!
//! The rkyv archive a proof file is written with carries an 8-byte relative
//! pointer and length for every vector and pads to alignment. A C1 proof is
//! mostly short vectors (a block of `2^k` values, a Merkle path, a sumcheck
//! round), so that framing is a few percent of its bytes. This encoding writes
//! every length as a LEB128 varint and every value at its width: a base field
//! element as its eight canonical little-endian bytes, an extension element as
//! its three, a digest as its 32 bytes.
//!
//! ★ AN ENCODING, NOT A FORMAT. The decoded proof is the same value the prover
//! made, and the verifier checks that value; nothing here reaches a
//! transcript. The decoder is strict so one proof has exactly one encoding:
//! a field element at or above the modulus, an overlong varint, an unknown tag,
//! a length that cannot fit in what is left, or a trailing byte is refused.

use math::field::element::FieldElement;
use multilinear::claim_reduce::ReduceProof;
use multilinear::constraint_argument::ConstraintCore;
use multilinear::gkr::{GkrProof, LayerProof};
use multilinear::gkr_lockstep::{LockstepLayer, LockstepProof, TreeHalves};
use multilinear::stacked_eval::StackedProof;
use multilinear::sumcheck::{RoundProof as SumcheckRound, SumcheckProof};
use multilinear::whir_chain::{ChainProof, ChainRound, RoundNonces, RoundOpenings};
use multilinear::whir_commit::{Commitment, CosetOpening};
use multilinear::whir_round::{GroupedRoundProof, RoundProof};
use stark::multilinear_table::batched::BatchedArgue;
use stark::multilinear_table::{MultiProof, TableProof};

use super::wlfm::WlfmProof;
use crate::tables::types::{FE, FEE, GoldilocksExtension as E, GoldilocksField as F};

/// The Goldilocks modulus: a base element's canonical value is below it.
const GOLDILOCKS: u64 = 0xFFFF_FFFF_0000_0001;

/// A cursor over encoded bytes.
pub struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn left(&self) -> usize {
        self.bytes.len() - self.at
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if n > self.left() {
            return Err(format!(
                "the encoding ends at byte {} and {n} more were wanted",
                self.bytes.len()
            ));
        }
        let out = &self.bytes[self.at..self.at + n];
        self.at += n;
        Ok(out)
    }

    /// Refused unless every byte was read.
    pub fn finish(self) -> Result<(), String> {
        if self.left() != 0 {
            return Err(format!("{} trailing bytes", self.left()));
        }
        Ok(())
    }
}

/// A value with one compact encoding.
pub trait Wire: Sized {
    fn put(&self, out: &mut Vec<u8>);
    fn get(r: &mut Reader<'_>) -> Result<Self, String>;
}

/// LEB128, at most ten bytes, the shortest form only.
fn put_varint(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn get_varint(r: &mut Reader<'_>) -> Result<u64, String> {
    let mut v = 0u64;
    for i in 0..10 {
        let byte = r.take(1)?[0];
        let bits = u64::from(byte & 0x7f);
        if i == 9 && bits > 1 {
            return Err("a varint past 64 bits".into());
        }
        v |= bits << (7 * i);
        if byte & 0x80 == 0 {
            if i > 0 && byte == 0 {
                return Err("an overlong varint".into());
            }
            return Ok(v);
        }
    }
    Err("a varint past ten bytes".into())
}

impl Wire for u64 {
    fn put(&self, out: &mut Vec<u8>) {
        put_varint(*self, out);
    }
    fn get(r: &mut Reader<'_>) -> Result<Self, String> {
        get_varint(r)
    }
}

impl Wire for u32 {
    fn put(&self, out: &mut Vec<u8>) {
        put_varint(u64::from(*self), out);
    }
    fn get(r: &mut Reader<'_>) -> Result<Self, String> {
        u32::try_from(get_varint(r)?).map_err(|_| "a u32 out of range".to_string())
    }
}

impl Wire for Commitment {
    fn put(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(self);
    }
    fn get(r: &mut Reader<'_>) -> Result<Self, String> {
        let mut d = [0u8; 32];
        d.copy_from_slice(r.take(32)?);
        Ok(d)
    }
}

impl Wire for FE {
    fn put(&self, out: &mut Vec<u8>) {
        use math::field::traits::IsPrimeField;
        out.extend_from_slice(&F::canonical(self.value()).to_le_bytes());
    }
    fn get(r: &mut Reader<'_>) -> Result<Self, String> {
        let mut b = [0u8; 8];
        b.copy_from_slice(r.take(8)?);
        let v = u64::from_le_bytes(b);
        if v >= GOLDILOCKS {
            return Err("a field element at or above the modulus".into());
        }
        Ok(FE::from(v))
    }
}

impl Wire for FEE {
    fn put(&self, out: &mut Vec<u8>) {
        for c in self.value() {
            c.put(out);
        }
    }
    fn get(r: &mut Reader<'_>) -> Result<Self, String> {
        Ok(FieldElement::<E>::new([
            FE::get(r)?,
            FE::get(r)?,
            FE::get(r)?,
        ]))
    }
}

impl<A: Wire, B: Wire> Wire for (A, B) {
    fn put(&self, out: &mut Vec<u8>) {
        self.0.put(out);
        self.1.put(out);
    }
    fn get(r: &mut Reader<'_>) -> Result<Self, String> {
        Ok((A::get(r)?, B::get(r)?))
    }
}

impl<T: Wire, const N: usize> Wire for [T; N] {
    fn put(&self, out: &mut Vec<u8>) {
        for x in self {
            x.put(out);
        }
    }
    fn get(r: &mut Reader<'_>) -> Result<Self, String> {
        let items = (0..N).map(|_| T::get(r)).collect::<Result<Vec<_>, _>>()?;
        items
            .try_into()
            .map_err(|_| "an array of the wrong length".to_string())
    }
}

impl<T: Wire> Wire for Vec<T> {
    fn put(&self, out: &mut Vec<u8>) {
        put_varint(self.len() as u64, out);
        for x in self {
            x.put(out);
        }
    }
    fn get(r: &mut Reader<'_>) -> Result<Self, String> {
        let n = usize::try_from(get_varint(r)?).map_err(|_| "a length out of range")?;
        // Every element takes at least one byte, so a longer claim is a lie
        // and never an allocation.
        if n > r.left() {
            return Err(format!("{n} elements in {} bytes", r.left()));
        }
        (0..n).map(|_| T::get(r)).collect()
    }
}

impl<T: Wire> Wire for Option<T> {
    fn put(&self, out: &mut Vec<u8>) {
        match self {
            None => out.push(0),
            Some(x) => {
                out.push(1);
                x.put(out);
            }
        }
    }
    fn get(r: &mut Reader<'_>) -> Result<Self, String> {
        match r.take(1)?[0] {
            0 => Ok(None),
            1 => Ok(Some(T::get(r)?)),
            t => Err(format!("an option tag {t}")),
        }
    }
}

/// A struct's fields in order, each through [`Wire`].
macro_rules! wire_struct {
    ($ty:ty { $($field:ident),+ $(,)? }) => {
        impl Wire for $ty {
            fn put(&self, out: &mut Vec<u8>) {
                $(self.$field.put(out);)+
            }
            fn get(r: &mut Reader<'_>) -> Result<Self, String> {
                Ok(Self { $($field: Wire::get(r)?),+ })
            }
        }
    };
}

wire_struct!(SumcheckRound<E> { evaluations });
wire_struct!(SumcheckProof<E> { rounds });
wire_struct!(CosetOpening<F> { values, proof });
wire_struct!(CosetOpening<E> { values, proof });
wire_struct!(RoundProof<F, E> { current, next });
wire_struct!(RoundProof<E, E> { current, next });
wire_struct!(GroupedRoundProof<F, E> { current, next });
wire_struct!(RoundNonces {
    folding,
    ood,
    query
});
wire_struct!(ChainRound<F, E> { sumcheck, next_root, ood_value, nonces, openings });
wire_struct!(ChainProof<F, E> { rounds, final_value });
wire_struct!(StackedProof < F, E > { polys });
wire_struct!(LayerProof<E> { sumcheck, p_lo, p_hi, q_lo, q_hi });
wire_struct!(GkrProof<E> { layers });
wire_struct!(TreeHalves<E> { p_lo, p_hi, q_lo, q_hi });
wire_struct!(LockstepLayer<E> { sumcheck, halves });
wire_struct!(LockstepProof<E> { layers });
wire_struct!(ReduceProof<E> { sumcheck, column_values });
wire_struct!(ConstraintCore<E> { sumcheck, factor_values, reduce });
wire_struct!(TableProof<E> { gkr, bus_output, constraint });
wire_struct!(BatchedArgue<E> { bus_outputs, gkr, constraint, factor_values, reduces });
wire_struct!(MultiProof<F, E> { roots, tables, columns, preprocessed });
wire_struct!(WlfmProof {
    proof,
    argues,
    prepared,
    commit_ood,
    public_words
});

impl Wire for crypto::merkle_tree::proof::Proof<Commitment> {
    fn put(&self, out: &mut Vec<u8>) {
        self.merkle_path.put(out);
    }
    fn get(r: &mut Reader<'_>) -> Result<Self, String> {
        Ok(Self {
            merkle_path: Wire::get(r)?,
        })
    }
}

impl Wire for RoundOpenings<F, E> {
    fn put(&self, out: &mut Vec<u8>) {
        match self {
            RoundOpenings::Base(p) => {
                out.push(0);
                p.put(out);
            }
            RoundOpenings::Extension(p) => {
                out.push(1);
                p.put(out);
            }
            RoundOpenings::BaseGrouped(p) => {
                out.push(2);
                p.put(out);
            }
            RoundOpenings::BaseJoint(p) => {
                out.push(3);
                p.put(out);
            }
        }
    }
    fn get(r: &mut Reader<'_>) -> Result<Self, String> {
        Ok(match r.take(1)?[0] {
            0 => RoundOpenings::Base(Wire::get(r)?),
            1 => RoundOpenings::Extension(Wire::get(r)?),
            2 => RoundOpenings::BaseGrouped(Wire::get(r)?),
            3 => RoundOpenings::BaseJoint(Wire::get(r)?),
            t => return Err(format!("a round-openings tag {t}")),
        })
    }
}

/// `proof`'s compact bytes.
pub fn encode(proof: &WlfmProof) -> Vec<u8> {
    let mut out = Vec::new();
    proof.put(&mut out);
    out
}

/// The proof `bytes` encode, refused unless they are exactly one proof's
/// encoding.
pub fn decode(bytes: &[u8]) -> Result<WlfmProof, String> {
    let mut r = Reader::new(bytes);
    let proof = WlfmProof::get(&mut r)?;
    r.finish()?;
    Ok(proof)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get<T: Wire>(bytes: &[u8]) -> Result<T, String> {
        let mut r = Reader::new(bytes);
        let v = T::get(&mut r)?;
        r.finish()?;
        Ok(v)
    }

    fn put<T: Wire>(v: &T) -> Vec<u8> {
        let mut out = Vec::new();
        v.put(&mut out);
        out
    }

    /// Varints round-trip at the edges, and an overlong, a too-long or a
    /// past-64-bit one is refused.
    #[test]
    fn varints_have_one_encoding() {
        for v in [0u64, 1, 127, 128, 300, u32::MAX as u64, u64::MAX] {
            assert_eq!(get::<u64>(&put(&v)), Ok(v), "{v}");
        }
        assert_eq!(put(&127u64), vec![0x7f]);
        assert_eq!(put(&128u64), vec![0x80, 0x01]);
        assert!(get::<u64>(&[0x80, 0x00]).is_err(), "overlong");
        assert!(get::<u64>(&[0xff; 11]).is_err(), "eleven bytes");
        let mut past = vec![0xff; 9];
        past.push(0x02);
        assert!(get::<u64>(&past).is_err(), "past 64 bits");
        assert!(get::<u32>(&put(&(u64::from(u32::MAX) + 1))).is_err());
    }

    /// Field elements: eight canonical bytes each; the modulus and above are
    /// refused, the largest canonical value is not.
    #[test]
    fn field_elements_are_canonical() {
        let x = FE::from(0x1234_5678_9abc_def0u64);
        assert_eq!(get::<FE>(&put(&x)), Ok(x));
        assert_eq!(put(&x).len(), 8);
        let e = FEE::new([FE::from(1u64), FE::from(2u64), FE::from(3u64)]);
        assert_eq!(get::<FEE>(&put(&e)), Ok(e));
        assert_eq!(put(&e).len(), 24);
        assert!(get::<FE>(&GOLDILOCKS.to_le_bytes()).is_err());
        assert!(get::<FE>(&u64::MAX.to_le_bytes()).is_err());
        assert!(get::<FE>(&(GOLDILOCKS - 1).to_le_bytes()).is_ok());
    }

    /// Containers: a length that cannot fit is refused before any
    /// allocation, an unknown option or openings tag is refused, and a
    /// trailing byte is refused.
    #[test]
    fn containers_refuse_lies_unknown_tags_and_trailing_bytes() {
        let v: Vec<FE> = vec![FE::from(5u64); 3];
        assert_eq!(get::<Vec<FE>>(&put(&v)), Ok(v.clone()));
        let mut huge = Vec::new();
        put_varint(u64::MAX >> 1, &mut huge);
        assert!(get::<Vec<FE>>(&huge).is_err());
        assert!(get::<Option<FE>>(&[2]).is_err());
        assert!(get::<RoundOpenings<F, E>>(&[4]).is_err());
        let mut trailing = put(&v);
        trailing.push(0);
        assert!(get::<Vec<FE>>(&trailing).is_err());
        let truncated = &put(&v)[..10];
        assert!(get::<Vec<FE>>(truncated).is_err());
    }
}
