#![no_main]

use hashbrown::HashMap;
use lambda_vm_syscalls as syscalls;

#[unsafe(export_name = "main")]
pub fn main() -> i32 {
    syscalls::allocator::init_allocator();
    let mut map = HashMap::new();
    map.insert("one", 1);
    map.insert("two", 2);
    return map["two"] + map["one"];
}
