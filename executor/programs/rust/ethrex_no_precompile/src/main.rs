use guest_program::{execution::execution_program, input::ProgramInput};
use rkyv::rancor::Error;
use lambda_vm_syscalls as syscalls;
pub fn main() {
    let input = syscalls::syscalls::get_private_input();
    let input = rkyv::from_bytes::<ProgramInput, Error>(&input).unwrap();
    let output = execution_program(input).unwrap();
    let output_bytes = output.encode();
    syscalls::syscalls::commit(&output_bytes);
}
