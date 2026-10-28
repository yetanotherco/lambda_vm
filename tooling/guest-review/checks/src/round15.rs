use ethrex_common::types::Code;

fn check(bytes: Vec<u8>, queries: &[usize]) {
    let reference = reference_common::types::Code::from_bytecode_unchecked(
        bytes.clone().into(),
        Default::default(),
    );
    let code = Code::from_bytecode_unchecked(bytes.into(), Default::default());
    let clone = code.clone();
    for (i, &target) in queries.iter().enumerate() {
        let expected = u32::try_from(target)
            .ok()
            .is_some_and(|pc| reference.jump_targets.binary_search(&pc).is_ok());
        let instance = if i % 2 == 0 { &code } else { &clone };
        assert_eq!(
            instance.is_valid_jump_target(target),
            expected,
            "target={target}"
        );
    }
    assert_eq!(code.jump_targets(), reference.jump_targets.as_ref());
    assert_eq!(clone.jump_targets(), reference.jump_targets.as_ref());
    assert_eq!(
        serde_json::to_value(&code).unwrap(),
        serde_json::to_value(&reference).unwrap()
    );
}

#[test]
fn distant_destinations_cover_every_push_distance_and_opcode() {
    for target in [64, 65, 127, 128, 129, 600] {
        for distance in 1..=32 {
            for opcode in 0..=255u8 {
                let mut bytes = vec![0; 1100];
                bytes[target - distance] = opcode;
                bytes[target] = 0x5b;
                bytes[1000] = 0x7f;
                bytes[1010] = 0x5b; // Invalid: inside PUSH32.
                bytes[1033] = 0x5b;
                check(
                    bytes,
                    &[
                        target,
                        target,
                        1033,
                        1010,
                        target - distance,
                        target,
                        0,
                        usize::MAX,
                    ],
                );
            }
        }
    }
}

#[test]
fn sparse_queries_and_random_windows_match_original_scanner() {
    let mut seed = 0x15b17c0deu64;
    for _ in 0..4096 {
        let mut bytes = vec![0; 2049];
        for window in [568..600, 968..1000, 2016..2048] {
            for i in window {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                bytes[i] = seed as u8;
            }
        }
        for target in [550, 600, 950, 1000, 2000, 2048] {
            bytes[target] = 0x5b;
        }
        // Forward, backward, repeated and nearby queries before materializing.
        check(
            bytes,
            &[2000, 550, 2048, 600, 1000, 950, 550, 600, 2049, usize::MAX],
        );
    }
}
