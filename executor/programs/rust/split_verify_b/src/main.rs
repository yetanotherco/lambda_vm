// verify_b of the 3MI split: runs the hash half of an LFM epoch verifier
// (RPX over Goldilocks, keccak-f, select, pack/unpack, bit decomposition)
// and commits keccak256(program) ‖ keccak256(record), the record being lanes
// 0..3 of the cells the field half hands over (`out`, assumed) followed by
// those it takes back (`back`, computed here).
//
// Private input: u32 program length in words, u32 repetitions of the hash
// half (a measurement knob; 0 and 1 run it once), the program words, then the
// `out` values and the hint values, four u64 lanes each. See
// prover/src/field_vm/split.rs for the encoding.

mod consts;

use consts::{ARK1, ARK2};
use lambda_vm_syscalls::keccak::keccak256;
use lambda_vm_syscalls::syscalls;

const P: u64 = 0xFFFF_FFFF_0000_0001;
const EPS: u64 = 0xFFFF_FFFF;

type Word = [u64; 4];

// Inside the permutation values are any u64 congruent mod p; they are
// canonicalized on the way out.
#[inline(always)]
fn add(a: u64, b: u64) -> u64 {
    let (s, c) = a.overflowing_add(b);
    let (s, c) = s.overflowing_add(EPS & (c as u64).wrapping_neg());
    s.wrapping_add(EPS & (c as u64).wrapping_neg())
}

#[inline(always)]
fn reduce(x: u128) -> u64 {
    let lo = x as u64;
    let hi = (x >> 64) as u64;
    let (t0, borrow) = lo.overflowing_sub(hi >> 32);
    let t0 = t0.wrapping_sub(EPS & (borrow as u64).wrapping_neg());
    let hl = hi & EPS;
    let (r, carry) = t0.overflowing_add((hl << 32) - hl);
    r.wrapping_add(EPS & (carry as u64).wrapping_neg())
}

#[inline(always)]
fn canonical(x: u64) -> u64 {
    if x >= P { x - P } else { x }
}

#[inline(always)]
fn mul(a: u64, b: u64) -> u64 {
    reduce(a as u128 * b as u128)
}

type State = [u64; 12];

const MDS_ROW: [u64; 12] = [7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8];

const MDS: [[u64; 12]; 12] = {
    let mut m = [[0; 12]; 12];
    let mut i = 0;
    while i < 12 {
        let mut j = 0;
        while j < 12 {
            m[i][j] = MDS_ROW[(j + 12 - i) % 12];
            j += 1;
        }
        i += 1;
    }
    m
};

#[inline(always)]
fn mds(s: &State) -> State {
    let mut out = [0u64; 12];
    for (row, o) in MDS.iter().zip(out.iter_mut()) {
        let mut acc: u128 = 0;
        for (c, v) in row.iter().zip(s) {
            acc += *v as u128 * *c as u128;
        }
        *o = reduce(acc);
    }
    out
}

#[inline(always)]
fn sbox(x: u64) -> u64 {
    let x2 = mul(x, x);
    let x3 = mul(x2, x);
    let x6 = mul(x3, x3);
    mul(x6, x)
}

#[inline(always)]
fn exp_acc(base: u64, tail: u64, m: usize) -> u64 {
    let mut acc = base;
    for _ in 0..m {
        acc = mul(acc, acc);
    }
    mul(acc, tail)
}

/// `x^{1/7}` by the addition chain of `rpo::Rpo256::inv_sbox_layer`, one lane
/// at a time so the chain stays in registers.
fn inv_sbox(x: u64) -> u64 {
    let t1 = mul(x, x);
    let t2 = mul(t1, t1);
    let t3 = exp_acc(t2, t2, 3);
    let t4 = exp_acc(t3, t3, 6);
    let t5 = exp_acc(t4, t4, 12);
    let t6 = exp_acc(t5, t3, 6);
    let t7 = exp_acc(t6, t6, 31);
    let mut a = mul(mul(t7, t7), t6);
    a = mul(a, a);
    a = mul(a, a);
    mul(a, mul(mul(t1, t2), x))
}

/// `GF(p)[φ]/(φ³ − φ − 1)`.
#[inline(always)]
fn ext_mul(a: &[u64; 3], b: &[u64; 3]) -> [u64; 3] {
    let x = add(mul(a[1], b[2]), mul(a[2], b[1]));
    [
        add(mul(a[0], b[0]), x),
        add(
            add(mul(a[0], b[1]), mul(a[1], b[0])),
            add(x, mul(a[2], b[2])),
        ),
        add(
            add(mul(a[0], b[2]), mul(a[1], b[1])),
            add(mul(a[2], b[0]), mul(a[2], b[2])),
        ),
    ]
}

fn power7(a: &[u64; 3]) -> [u64; 3] {
    let a2 = ext_mul(a, a);
    let a3 = ext_mul(&a2, a);
    let a6 = ext_mul(&a3, &a3);
    ext_mul(&a6, a)
}

/// RPX256: FB E FB E FB E M.
fn rpx(state: State) -> State {
    let mut s = state;
    for r in 0..7 {
        if r % 2 == 0 && r < 6 {
            s = mds(&s);
            for (l, v) in s.iter_mut().enumerate() {
                *v = sbox(add(*v, ARK1[r][l]));
            }
            s = mds(&s);
            for (l, v) in s.iter_mut().enumerate() {
                *v = add(*v, ARK2[r][l]);
            }
            for v in s.iter_mut() {
                *v = inv_sbox(*v);
            }
        } else if r % 2 == 1 {
            for (l, v) in s.iter_mut().enumerate() {
                *v = add(*v, ARK1[r][l]);
            }
            for e in 0..4 {
                let x = [s[3 * e], s[3 * e + 1], s[3 * e + 2]];
                let p = power7(&x);
                s[3 * e..3 * e + 3].copy_from_slice(&p);
            }
        } else {
            s = mds(&s);
            for (l, v) in s.iter_mut().enumerate() {
                *v = add(*v, ARK1[r][l]);
            }
        }
    }
    s.map(canonical)
}

const DOMAIN_TRANSCRIPT: u64 = u32::from_le_bytes(*b"LFMT") as u64;
const DOMAIN_LEAF: u64 = u32::from_le_bytes(*b"LFML") as u64;

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    #[inline(always)]
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.bytes[self.at..self.at + 4].try_into().unwrap());
        self.at += 4;
        v
    }

    #[inline(always)]
    fn a(&mut self) -> usize {
        self.u32() as usize
    }

    fn u64(&mut self) -> u64 {
        let v = u64::from_le_bytes(self.bytes[self.at..self.at + 8].try_into().unwrap());
        self.at += 8;
        v
    }

    fn felt(&mut self) -> u64 {
        let v = self.u64();
        assert!(v < P, "non-canonical felt");
        v
    }

    fn word(&mut self) -> Word {
        [self.felt(), self.felt(), self.felt(), self.felt()]
    }
}

fn base(w: &Word) -> u64 {
    assert!(w[1] == 0 && w[2] == 0 && w[3] == 0, "not a base word");
    w[0]
}

pub fn main() {
    let input = syscalls::get_private_input_slice();
    let len = u32::from_le_bytes(input[..4].try_into().unwrap()) as usize;
    let reps = u32::from_le_bytes(input[4..8].try_into().unwrap()).max(1);
    let program = &input[8..8 + 4 * len];
    let mut values = Reader {
        bytes: input,
        at: 8 + 4 * len,
    };
    let mut p = Reader {
        bytes: program,
        at: 0,
    };
    assert_eq!(p.u32(), 1, "encoding version");
    let n_cells = p.a();
    let n_out = p.a();
    let n_back = p.a();
    let n_instr = p.a();
    let mut mem: Vec<Word> = vec![[0; 4]; n_cells];
    let mut record: Vec<u8> = Vec::with_capacity(24 * (n_out + n_back));
    for cell in mem.iter_mut().take(n_out) {
        let is_word = p.u32() != 0;
        let l3c = p.u32() as u64 | ((p.u32() as u64) << 32);
        let w = values.word();
        assert!(is_word || w[3] == l3c, "fourth lane of a field cell");
        for lane in &w[..3] {
            record.extend_from_slice(&lane.to_le_bytes());
        }
        *cell = w;
    }
    let back_at = p.at;
    let instr_at = back_at + 4 * n_back;
    let hints_at = values.at;
    for _ in 0..reps {
        p.at = instr_at;
        values.at = hints_at;
        for _ in 0..n_instr {
            match p.u32() {
                // Hash: mode, inputs, outputs.
                1 => {
                    let mode = p.u32();
                    let mut s = [0u64; 12];
                    if mode == 3 {
                        for k in 0..3 {
                            s[4 * k..4 * k + 4].copy_from_slice(&mem[p.a()]);
                        }
                        let o = rpx(s);
                        for k in 0..3 {
                            let a = p.a();
                            mem[a].copy_from_slice(&o[4 * k..4 * k + 4]);
                        }
                    } else {
                        s[0..4].copy_from_slice(&mem[p.a()]);
                        s[4..8].copy_from_slice(&mem[p.a()]);
                        s[9] = match mode {
                            0 => 0,
                            1 => DOMAIN_TRANSCRIPT,
                            2 => DOMAIN_LEAF,
                            _ => panic!("hash mode"),
                        };
                        let o = rpx(s);
                        let a = p.a();
                        mem[a].copy_from_slice(&o[0..4]);
                    }
                }
                // Select: bit, in_l, in_r, out_l, out_r.
                2 => {
                    let bit = base(&mem[p.a()]);
                    let (l, r) = (mem[p.a()], mem[p.a()]);
                    let (ol, or) = match bit {
                        0 => (l, r),
                        1 => (r, l),
                        _ => panic!("non-boolean select"),
                    };
                    mem[p.a()] = ol;
                    mem[p.a()] = or;
                }
                // Pack: four lanes, out.
                3 => {
                    let w = [
                        base(&mem[p.a()]),
                        base(&mem[p.a()]),
                        base(&mem[p.a()]),
                        base(&mem[p.a()]),
                    ];
                    mem[p.a()] = w;
                }
                // Unpack: input, four outs.
                4 => {
                    let w = mem[p.a()];
                    for lane in w {
                        mem[p.a()] = [lane, 0, 0, 0];
                    }
                }
                // BitDec: input, bits, the bit cells, halves flag, [hi, lo].
                5 => {
                    let v = base(&mem[p.a()]);
                    let n = p.a();
                    for i in 0..n {
                        mem[p.a()] = [(v >> i) & 1, 0, 0, 0];
                    }
                    if p.u32() != 0 {
                        let hi = (v >> 32) as u32;
                        let lo = v as u32;
                        mem[p.a()] = [hi.swap_bytes() as u64, 0, 0, 0];
                        mem[p.a()] = [lo.swap_bytes() as u64, 0, 0, 0];
                    }
                }
                // KeccakF: mode, 13 state words, [9 block words], 13 outs,
                // rev flag, [2 reversed-digest outs]; u32 halves per lane.
                6 => {
                    let absorb = p.u32() != 0;
                    let mut state = [0u64; 25];
                    for j in 0..13 {
                        let w = mem[p.a()];
                        for (l, v) in w.iter().enumerate() {
                            let h = 4 * j + l;
                            if h >= 50 {
                                assert!(*v == 0, "keccak spare lane");
                            } else {
                                assert!(*v < 1 << 32, "keccak half");
                                state[h / 2] |= v << (32 * (h % 2));
                            }
                        }
                    }
                    if absorb {
                        for j in 0..9 {
                            let w = mem[p.a()];
                            for (l, v) in w.iter().enumerate() {
                                let h = 4 * j + l;
                                if h >= 34 {
                                    assert!(*v == 0, "keccak spare block lane");
                                } else {
                                    assert!(*v < 1 << 32, "keccak block half");
                                    state[h / 2] ^= v << (32 * (h % 2));
                                }
                            }
                        }
                    }
                    syscalls::keccak_permute(&mut state);
                    for j in 0..13 {
                        let a = p.a();
                        mem[a] = core::array::from_fn(|l| {
                            let h = 4 * j + l;
                            if h < 50 {
                                (state[h / 2] >> (32 * (h % 2))) as u32 as u64
                            } else {
                                0
                            }
                        });
                    }
                    if p.u32() != 0 {
                        let mut digest = [0u8; 32];
                        for (lane, chunk) in state[..4].iter().zip(digest.chunks_exact_mut(8)) {
                            chunk.copy_from_slice(&lane.to_le_bytes());
                        }
                        digest.reverse();
                        for w in 0..2 {
                            let a = p.a();
                            mem[a] = core::array::from_fn(|l| {
                                let h = 4 * w + l;
                                u32::from_le_bytes(digest[4 * h..4 * h + 4].try_into().unwrap())
                                    as u64
                            });
                        }
                    }
                }
                // Hint: out, value from the hint stream.
                7 => {
                    let w = values.word();
                    mem[p.a()] = w;
                }
                // Const: out, four lanes.
                8 => {
                    let a = p.a();
                    let w = [p.u64(), p.u64(), p.u64(), p.u64()];
                    mem[a] = w;
                }
                t => panic!("unknown instruction {t}"),
            }
        }
    }
    let end = p.at;
    p.at = back_at;
    for _ in 0..n_back {
        let w = mem[p.a()];
        for lane in &w[..3] {
            record.extend_from_slice(&lane.to_le_bytes());
        }
    }
    assert_eq!(end, program.len(), "trailing program words");
    let mut out = [0u8; 64];
    out[..32].copy_from_slice(&keccak256(program));
    out[32..].copy_from_slice(&keccak256(&record));
    syscalls::commit(&out);
}
