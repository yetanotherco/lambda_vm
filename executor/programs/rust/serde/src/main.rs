use lambda_vm_syscalls as syscalls;

#[derive(serde::Serialize)]
struct MyData {
    val: i32,
    values: Vec<u8>,
}
pub fn main() {
    let my_data = MyData {
        val: 42,
        values: vec![1, 2, 3, 4, 5],
    };
    let serialized = serde_json::to_vec(&my_data).unwrap();
    syscalls::syscalls::commit(serialized.as_ref());
}
