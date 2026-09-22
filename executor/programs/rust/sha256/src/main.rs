fn main() {
    let mut data = [0u8; 1032];
    for (i, b) in data.iter_mut().enumerate() {
        *b = (i * 17 + 3) as u8;
    }
    for len in [0, 1, 31, 32, 55, 56, 63, 64, 65, 127, 128, 129, 1024] {
        for offset in 0..8 {
            let hash = lambda_vm_syscalls::sha256::sha256(&data[offset..offset + len]);
            lambda_vm_syscalls::syscalls::commit(&hash);
        }
    }
}
