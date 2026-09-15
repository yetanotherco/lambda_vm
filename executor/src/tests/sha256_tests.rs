use crate::{
    sha256,
    vm::{
        instruction::{
            decoding::Instruction,
            execution::{ExecutionError, SHA256_SYSCALL_NUMBER},
        },
        memory::Memory,
        registers::Registers,
    },
};
#[test]
fn sha256_abc_vector() {
    let mut h = [0u8; 32];
    for (i, x) in sha256::IV.iter().enumerate() {
        h[4 * i..4 * i + 4].copy_from_slice(&x.to_be_bytes());
    }
    let mut m = [0; 64];
    m[..4].copy_from_slice(b"abc\x80");
    m[63] = 24;
    sha256::compress(&mut h, &m);
    assert_eq!(
        h,
        [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
            0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
            0xf2, 0x00, 0x15, 0xad
        ]
    );
}
#[test]
fn sha256_syscall_alignment_overlap_and_boundary() {
    for (h, m) in [
        (0x1000, 0x2000),
        (0x1003, 0x2007),
        (0x1000, 0x1000),
        (0x1007, 0x1000),
        (0x1000, 0x1007),
        (0xfffffff0, 0x2003),
    ] {
        let mut memory = Memory::default();
        for i in 0..64 {
            memory.store_byte(m + i, (i * 17) as u8);
        }
        for i in 0..32 {
            memory.store_byte(h + i, (i * 23) as u8);
        }
        let mut expected = std::array::from_fn(|i| memory.load_byte(h + i as u64));
        let message = std::array::from_fn(|i| memory.load_byte(m + i as u64));
        sha256::compress(&mut expected, &message);
        let mut registers = Registers::default();
        registers.write(17, SHA256_SYSCALL_NUMBER).unwrap();
        registers.write(10, h).unwrap();
        registers.write(11, m).unwrap();
        Instruction::EcallEbreak
            .run(&mut 0, &mut registers, &mut memory)
            .unwrap();
        assert_eq!(
            std::array::from_fn::<_, 32, _>(|i| memory.load_byte(h + i as u64)),
            expected
        );
    }
}
#[test]
fn sha256_rejects_overflow_without_writes() {
    for (h, m) in [(u64::MAX - 30, 0x1000), (0x1000, u64::MAX - 62)] {
        let mut memory = Memory::default();
        memory.store_byte(0x1000, 123);
        let mut registers = Registers::default();
        registers.write(17, SHA256_SYSCALL_NUMBER).unwrap();
        registers.write(10, h).unwrap();
        registers.write(11, m).unwrap();
        assert!(matches!(
            Instruction::EcallEbreak.run(&mut 0, &mut registers, &mut memory),
            Err(ExecutionError::Sha256AddressOverflow)
        ));
        assert_eq!(memory.load_byte(0x1000), 123);
    }
}
