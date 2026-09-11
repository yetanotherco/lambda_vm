//! Hint-request guest: asks the host for one hint per selector and commits a
//! value that does NOT depend on their bytes.
//!
//! Cheap end-to-end exercise of the mechanism (`ecrecover_hints` is the
//! realistic consumer but costs millions of cycles). Each request goes through
//! the request log and is answered by the executor seeding the matching arena
//! slot as an initial value, so the proof needs no HINT table rows: the guest's
//! reads are ordinary MEMR loads chained to the private-input pages.
//!
//! What it pins: the committed output is how many requests came back non-zero,
//! so a silent host commits 0 and an answering host commits 3 — either way the
//! guest decides, which is the property every hint consumer relies on. Each
//! input is chosen so its selector CAN answer (a non-zero element always has an
//! inverse; 4 is a square), making "non-zero" mean "answered" rather than
//! "answerable".

use lambda_vm_syscalls as syscalls;
use syscalls::syscalls::{HINT_FIELD_INV, HINT_FIELD_SQRT, HINT_SCALAR_INV, request_hint};

/// 32-byte big-endian encoding of a small integer — the serialization k256 (and
/// therefore the hint ABI) uses.
const fn be(n: u8) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[31] = n;
    bytes
}

pub fn main() {
    let requests = [
        (HINT_FIELD_INV, be(7)),
        (HINT_SCALAR_INV, be(7)),
        (HINT_FIELD_SQRT, be(4)),
    ];

    let mut answered = 0u8;
    for (selector, input) in requests {
        if request_hint(selector, &input) != [0u8; 32] {
            answered += 1;
        }
    }

    let mut out = [0u8; 32];
    out[0] = answered;
    syscalls::syscalls::commit(&out);
}
