#![no_std]
#![no_main]

// Disclaimer: This rlp test program was obtained from ethrex repository
// https://github.com/lambdaclass/ethrex/blob/main/crates/common/rlp/decode.rs

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

pub const RLP_NULL: u8 = 0x80;
pub const RLP_EMPTY_LIST: u8 = 0xC0;

pub trait RLPDecode: Sized {
    fn decode_unfinished(rlp: &[u8]) -> Result<(Self, &[u8]), bool>;

    fn decode(rlp: &[u8]) -> Result<Self, bool> {
        let (decoded, remaining) = Self::decode_unfinished(rlp)?;
        if !remaining.is_empty() {
            return Err(false);
        }

        Ok(decoded)
    }
}

impl RLPDecode for u32 {
    fn decode_unfinished(rlp: &[u8]) -> Result<(Self, &[u8]), bool> {
        let (bytes, rest) = decode_bytes(rlp)?;
        let padded_bytes = static_left_pad(bytes)?;
        Ok((u32::from_be_bytes(padded_bytes), rest))
    }
}

pub fn decode_bytes(data: &[u8]) -> Result<(&[u8], &[u8]), bool> {
    let (is_list, payload, rest) = decode_rlp_item(data)?;
    if is_list {
        return Err(false);
    }
    Ok((payload, rest))
}

pub fn decode_rlp_item(data: &[u8]) -> Result<(bool, &[u8], &[u8]), bool> {
    if data.is_empty() {
        return Err(false);
    }

    let first_byte = data[0];

    match first_byte {
        0..=0x7F => Ok((false, &data[..1], &data[1..])),
        0x80..=0xB7 => {
            let length = (first_byte - 0x80) as usize;
            if data.len() < length + 1 {
                return Err(false);
            }
            Ok((false, &data[1..length + 1], &data[length + 1..]))
        }
        0xB8..=0xBF => {
            let length_of_length = (first_byte - 0xB7) as usize;
            if data.len() < length_of_length + 1 {
                return Err(false);
            }
            let length_bytes = &data[1..length_of_length + 1];
            let length = usize::from_be_bytes(static_left_pad(length_bytes)?);
            if data.len() < length_of_length + length + 1 {
                return Err(false);
            }
            Ok((
                false,
                &data[length_of_length + 1..length_of_length + length + 1],
                &data[length_of_length + length + 1..],
            ))
        }
        RLP_EMPTY_LIST..=0xF7 => {
            let length = (first_byte - RLP_EMPTY_LIST) as usize;
            if data.len() < length + 1 {
                return Err(false);
            }
            Ok((true, &data[1..length + 1], &data[length + 1..]))
        }
        0xF8..=0xFF => {
            let list_length = (first_byte - 0xF7) as usize;
            if data.len() < list_length + 1 {
                return Err(false);
            }
            let length_bytes = &data[1..list_length + 1];
            let payload_length = usize::from_be_bytes(static_left_pad(length_bytes)?);
            if data.len() < list_length + payload_length + 1 {
                return Err(false);
            }
            Ok((
                true,
                &data[list_length + 1..list_length + payload_length + 1],
                &data[list_length + payload_length + 1..],
            ))
        }
    }
}

pub fn static_left_pad<const N: usize>(data: &[u8]) -> Result<[u8; N], bool> {
    let mut result = [0; N];

    if data.is_empty() {
        return Ok(result);
    }
    if data[0] == 0 {
        return Err(false);
    }
    if data.len() > N {
        return Err(false);
    }
    let data_start_index = N.saturating_sub(data.len());
    result
        .get_mut(data_start_index..)
        .ok_or(false)?
        .copy_from_slice(data);
    Ok(result)
}

#[unsafe(export_name = "main")]
pub fn main() -> u32 {
    let rlp = [0x83, 0x01, 0x00, 0x00];
    let decoded = u32::decode(&rlp).unwrap();
    decoded
}
