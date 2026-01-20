use lambda_vm_syscalls as syscalls;

// This test tries to access command-line arguments using std::env::args()
// This requires sys_argc and sys_argv to be implemented
// The test should fail because these functions are not defined in Lambda VM

pub fn main() {
    // Try to get the number of arguments
    let args: Vec<String> = std::env::args().collect();

    // Commit the argument count as a byte
    let count = args.len() as u8;
    syscalls::syscalls::commit(&[count]);
}
