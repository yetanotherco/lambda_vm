use lambda_vm_syscalls as syscalls;
use hashbrown::HashMap;

const ITERATIONS: usize = 100;
const MAP_SIZE: usize = 100;

pub fn main() {
    syscalls::allocator::init_allocator();

    let mut sum: u64 = 0;

    for iteration in 0..ITERATIONS {
        let mut map: HashMap<u64, u64> = HashMap::new();

        // Insert entries
        for i in 0..MAP_SIZE {
            let key = (iteration * MAP_SIZE + i) as u64;
            let value = key.wrapping_mul(31);
            map.insert(key, value);
        }

        // Lookup and accumulate
        for i in 0..MAP_SIZE {
            let key = (iteration * MAP_SIZE + i) as u64;
            if let Some(&value) = map.get(&key) {
                sum = sum.wrapping_add(value);
            }
        }
    }

    syscalls::syscalls::commit(&sum.to_le_bytes());
}
