// Compare the changed handlers with the pinned original VM, including the
// traced branch that the guest benchmark deliberately leaves inactive.
macro_rules! jump_runner {
    ($name:ident, $common:ident, $levm:ident, $crypto:ident) => {
        fn $name(bytes: &[u8], target: u64, condition: Option<u64>, gas: i64,
                 tracing: bool, config: usize, amsterdam: bool) -> serde_json::Value {
            use $common::{Address, H256, U256, types::{AccountState, ChainConfig,
                Code, CodeMetadata, EIP1559Transaction, Fork, Transaction}};
            use $levm::{db::{Database, gen_db::GeneralizedDatabase}, errors::DatabaseError,
                opcode_handlers::{OpcodeHandler, stack_memory_storage_flow::{OpJumpHandler, OpJumpIHandler}},
                opcode_tracer::{LevmOpcodeTracer, OpcodeTracerConfig}, vm::VM};
            struct Stub;
            impl Database for Stub {
                fn get_account_state(&self, _: Address) -> Result<AccountState, DatabaseError> { unreachable!() }
                fn get_storage_value(&self, _: Address, _: H256) -> Result<U256, DatabaseError> { unreachable!() }
                fn get_block_hash(&self, _: u64) -> Result<H256, DatabaseError> { unreachable!() }
                fn get_chain_config(&self) -> Result<ChainConfig, DatabaseError> { unreachable!() }
                fn get_account_code(&self, _: H256) -> Result<Code, DatabaseError> { unreachable!() }
                fn get_code_metadata(&self, _: H256) -> Result<CodeMetadata, DatabaseError> { unreachable!() }
            }
            let mut db = GeneralizedDatabase::new(std::sync::Arc::new(Stub));
            let tx = Transaction::EIP1559Transaction(EIP1559Transaction::default());
            let crypto = $crypto::NativeCrypto;
            let fork = if amsterdam { Fork::Amsterdam } else { Fork::Prague };
            let mut vm = VM::new_state_gas_harness(fork, &mut db, &tx, &crypto, 0);
            vm.current_call_frame.bytecode = Code::from_bytecode_unchecked(bytes.to_vec().into(), Default::default());
            vm.current_call_frame.pc = 1;
            vm.current_call_frame.gas_remaining = gas;
            vm.current_call_frame.memory.store_data(3, &[1,2,3,4,5]).unwrap();
            vm.current_call_frame.sub_return_data = vec![0xa5,0x5a].into();
            vm.current_call_frame.stack.push(U256::from(0x1234u64)).unwrap();
            if let Some(value) = condition { vm.current_call_frame.stack.push(value.into()).unwrap(); }
            vm.current_call_frame.stack.push(target.into()).unwrap();
            if tracing {
                vm.opcode_tracer = LevmOpcodeTracer::new(OpcodeTracerConfig {
                    disable_stack: config & 1 != 0,
                    enable_memory: config & 2 != 0,
                    enable_return_data: config & 4 != 0,
                    limit: if config & 8 != 0 { 1 } else { 0 },
                    ..Default::default()
                });
                let values = if let Some(value) = condition { vec![U256::from(0x1234u64), value.into(), target.into()] }
                             else { vec![U256::from(0x1234u64), target.into()] };
                let memory = vm.current_call_frame.memory.live_bytes();
                vm.opcode_tracer.pre_step_capture(0, if condition.is_some() { 0x57 } else { 0x56 },
                    gas as u64, 1, 0, &values, &memory, memory.len() as u64,
                    &vm.current_call_frame.sub_return_data, None);
            }
            let result = if condition.is_some() { OpJumpIHandler::eval(&mut vm) } else { OpJumpHandler::eval(&mut vm) };
            serde_json::json!({
                "result": format!("{result:?}"), "pc": vm.current_call_frame.pc,
                "gas": vm.current_call_frame.gas_remaining, "stack_len": vm.current_call_frame.stack.len(),
                "logs": format!("{:?}", vm.opcode_tracer.logs), "gas_override": vm.opcode_tracer.last_opcode_gas_cost,
                "pending_step": vm.opcode_tracer.last_step_index,
                "memory": vm.current_call_frame.memory.live_bytes(),
            })
        }
    }
}
jump_runner!(optimized, ethrex_common, ethrex_levm, ethrex_crypto);
jump_runner!(reference, reference_common, reference_levm, reference_crypto);

#[test]
fn jump_traces_gas_and_errors_match_original_vm() {
    let bytes = [0x56, 0x5b, 0x60, 0x5b, 0x5b, 0x00, 0x7f, 0x5b];
    for tracing in [false, true] {
        for amsterdam in [false, true] {
            for config in 0..16 {
                for gas in [0, 7, 8, 9, 10, 11, 12, 1000] {
                    for condition in [None, Some(0), Some(1)] {
                        for target in [0,1,3,4,5,7,8,u64::MAX] {
                            assert_eq!(optimized(&bytes,target,condition,gas,tracing,config,amsterdam),
                                       reference(&bytes,target,condition,gas,tracing,config,amsterdam),
                                       "target={target} condition={condition:?} gas={gas} tracing={tracing} config={config} amsterdam={amsterdam}");
                        }
                    }
                }
            }
        }
    }
}
