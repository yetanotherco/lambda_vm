use lambda_vm_syscalls as syscalls;

pub fn main() {
    let input = syscalls::syscalls::get_private_input();
    syscalls::syscalls::print_string(format!("Private input received: {:?}\n", input).as_str());
    syscalls::syscalls::commit(input);
}
