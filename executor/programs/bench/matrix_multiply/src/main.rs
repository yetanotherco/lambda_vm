use lambda_vm_syscalls as syscalls;

const SIZE: usize = {
    const fn parse(s: &str) -> usize {
        let b = s.as_bytes();
        let mut r = 0;
        let mut i = 0;
        while i < b.len() {
            r = r * 10 + (b[i] - b'0') as usize;
            i += 1;
        }
        r
    }
    match option_env!("SIZE") {
        Some(s) => parse(s),
        None => 81,
    }
};

#[inline(never)]
fn matrix_multiply(a: &[[u32; SIZE]; SIZE], b: &[[u32; SIZE]; SIZE], result: &mut [[u32; SIZE]; SIZE]) {
    let mut i = 0;
    while i < SIZE {
        let mut j = 0;
        while j < SIZE {
            let mut sum = 0u32;
            let mut k = 0;
            while k < SIZE {
                sum = sum.wrapping_add(a[i][k].wrapping_mul(b[k][j]));
                k += 1;
            }
            result[i][j] = sum;
            j += 1;
        }
        i += 1;
    }
}

#[inline(never)]
fn init_matrix(m: &mut [[u32; SIZE]; SIZE], seed: u32) {
    let mut val = seed;
    let mut i = 0;
    while i < SIZE {
        let mut j = 0;
        while j < SIZE {
            m[i][j] = val % 100;
            val = val.wrapping_mul(1103515245).wrapping_add(12345);
            j += 1;
        }
        i += 1;
    }
}

pub fn main() {
    let mut a = [[0u32; SIZE]; SIZE];
    let mut b = [[0u32; SIZE]; SIZE];
    let mut result = [[0u32; SIZE]; SIZE];

    init_matrix(&mut a, 12345);
    init_matrix(&mut b, 67890);
    matrix_multiply(&a, &b, &mut result);

    let mut checksum = 0u32;
    let mut i = 0;
    while i < SIZE {
        checksum = checksum.wrapping_add(result[i][i]);
        i += 1;
    }
    syscalls::syscalls::commit(&checksum.to_le_bytes());
}
