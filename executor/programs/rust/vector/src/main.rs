use lambda_vm_syscalls as syscalls;

pub fn main() {
    let vector = vec![1, 2, 3, 4, 5];
    syscalls::syscalls::commit(&vector);
}
