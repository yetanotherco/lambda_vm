use lambda_vm_syscalls as syscalls;

pub fn main() {
    let input = syscalls::syscalls::get_private_input();
    let size = u32::from_be_bytes(input.try_into().unwrap());
    let mut vector = vec![];
    for i in 0..size {
        vector.push(1);
    }
    syscalls::syscalls::commit(&vector[(size - 1000) as usize ..]);
}
