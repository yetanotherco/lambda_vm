// Replay identical checkpoint histories against the pinned, unmodified ethrex.
// Compare observations, including log order, refund restoration, slot warmth,
// transient overwrites, and nested commit followed by outer revert.
macro_rules! replay {
    ($name:ident, $levm:ident, $common:ident) => {
        fn $name(fork_index: usize, seed: u64) -> Vec<String> {
            use $common::{
                Address, H256, U256,
                types::{Fork, Log},
            };
            use $levm::vm::Substate;
            let fork = [Fork::Istanbul, Fork::Cancun, Fork::Osaka][fork_index];
            let mut state = Substate::from_accesses(fork, Default::default(), Default::default());
            let mut rng = seed;
            let mut depth = 0usize;
            let mut observations = Vec::new();
            for step in 0..4096 {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let address = Address::from_low_u64_be(0x1000 + ((rng >> 16) & 7));
                let slot = H256::from_low_u64_be((rng >> 24) & 15);
                let key = U256::from((rng >> 24) & 15);
                match rng % 12 {
                    0 if depth < 8 => {
                        state.push_backup();
                        depth += 1;
                    }
                    1 => {
                        state.commit_backup();
                        depth = depth.saturating_sub(1);
                    }
                    2 => {
                        state.revert_backup();
                        depth = depth.saturating_sub(1);
                    }
                    3 => observations
                        .push(format!("slot:{}", state.add_accessed_slot(address, slot))),
                    4 => observations
                        .push(format!("address:{}", state.add_accessed_address(address))),
                    5 => {
                        observations.push(format!("created:{}", state.add_created_account(address)))
                    }
                    6 => {
                        observations.push(format!("destroyed:{}", state.add_selfdestruct(address)))
                    }
                    7 => state.set_transient(&address, &key, U256::from(rng)),
                    8 => state.set_transient(&address, &key, U256::zero()),
                    9 => state.refunded_gas = rng & 0xffff,
                    10 => state.add_log(Log {
                        address,
                        topics: vec![slot],
                        data: vec![step as u8].into(),
                    }),
                    _ => {}
                }
                if step % 16 == 0 {
                    observations.push(format!(
                        "refund:{}; access:{:?}; logs:{:?}; current:{:?}; len:{}",
                        state.refunded_gas,
                        state.make_access_list(),
                        state.extract_logs(),
                        state.current_logs(),
                        state.logs_len()
                    ));
                    let mut destroyed = state.iter_selfdestruct().copied().collect::<Vec<_>>();
                    destroyed.sort();
                    observations.push(format!("destroyed:{destroyed:?}"));
                    for n in [1, 9, 10, 17, 0x100, 0x1000, 0x1001, 0x1002, 0x1007] {
                        let a = Address::from_low_u64_be(n);
                        observations.push(format!(
                            "{}:{}:{}:{:?}",
                            state.is_address_accessed(&a),
                            state.is_account_created(&a),
                            state.is_selfdestruct(&a),
                            state.get_accessed_storage_slots(&a)
                        ));
                        for k in 0..16 {
                            observations.push(format!(
                                "{}:{}",
                                state.is_slot_accessed(&a, &H256::from_low_u64_be(k)),
                                state.get_transient(&a, &U256::from(k))
                            ));
                        }
                    }
                }
            }
            // Unwind the entire chain, including redundant commits/reverts at root.
            for _ in 0..10 {
                state.commit_backup();
                observations.push(format!(
                    "{}:{:?}:{:?}",
                    state.refunded_gas,
                    state.make_access_list(),
                    state.extract_logs()
                ));
            }
            state.revert_backup();
            observations.push(format!(
                "{}:{:?}:{:?}",
                state.refunded_gas,
                state.make_access_list(),
                state.extract_logs()
            ));
            observations
        }
    };
}

replay!(optimized, ethrex_levm, ethrex_common);
replay!(reference, reference_levm, reference_common);

#[test]
fn checkpoint_histories_match_reference() {
    for fork in 0..3 {
        for seed in [0x13c0ffee, 0xdeadbeef01, 0x514e3a8b] {
            assert_eq!(
                optimized(fork, seed),
                reference(fork, seed),
                "fork={fork} seed={seed}"
            );
        }
    }
}
