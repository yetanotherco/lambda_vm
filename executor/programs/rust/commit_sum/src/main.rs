use lambda_vm_syscalls as syscalls;

pub fn main() {
    let input = syscalls::syscalls::get_private_input();
    let a = input[0];
    let b = input[1];
    syscalls::syscalls::commit((a + b).to_le_bytes().as_ref());
}
