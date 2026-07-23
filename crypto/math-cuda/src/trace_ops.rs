//! Device `CpuOperation` builder (Phase 0 of full-GPU trace-gen; see
//! `kernels/trace_ops.cu`).
//!
//! Reconstructs, on device, the per-cycle op record the prover's
//! `CpuOperation::from_log` computes on the host — a stateless function of the cycle's
//! executor Log + decoded instruction. This SoA is the resident seam every later
//! trace-gen stage reads. `gpu_build_cpu_ops` is the host-in/host-out form used by the
//! byte-parity test; a device-resident form follows once downstream stages consume it.

use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

/// The computed per-cycle fields (indexed by cycle). `flags` bit 0 = branch_cond,
/// 1 = ecall_commit, 2 = ecall_keccak, 3 = ecall_ecsm. The decode (pc/imm/packed),
/// timestamp, and raw log values are inputs, so only the computed outputs are returned.
pub struct DeviceCpuOps {
    pub rv1: Vec<u64>,
    pub rv2: Vec<u64>,
    pub arg2: Vec<u64>,
    pub res: Vec<u64>,
    pub rvd: Vec<u64>,
    pub next_pc: Vec<u64>,
    pub flags: Vec<u8>,
    pub commit_buf_addr: Vec<u64>,
    pub commit_count: Vec<u64>,
    pub keccak_state_addr: Vec<u64>,
}

/// Build the `CpuOperation` computed fields on device from the per-cycle Log SoA
/// (`current_pc`, `next_pc_log`, `src1_val`, `src2_val`, `dst_val`) and decode SoA
/// (`pc`, `imm`, `packed` = `ShrunkDecode::pack()`). One thread per cycle.
#[allow(clippy::too_many_arguments)]
pub fn gpu_build_cpu_ops(
    current_pc: &[u64],
    next_pc_log: &[u64],
    src1_val: &[u64],
    src2_val: &[u64],
    dst_val: &[u64],
    pc: &[u64],
    imm: &[u64],
    packed: &[u64],
) -> Result<DeviceCpuOps> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = current_pc.len();
    if n == 0 {
        return Ok(DeviceCpuOps {
            rv1: vec![],
            rv2: vec![],
            arg2: vec![],
            res: vec![],
            rvd: vec![],
            next_pc: vec![],
            flags: vec![],
            commit_buf_addr: vec![],
            commit_count: vec![],
            keccak_state_addr: vec![],
        });
    }
    debug_assert_eq!(next_pc_log.len(), n);
    debug_assert_eq!(src1_val.len(), n);
    debug_assert_eq!(src2_val.len(), n);
    debug_assert_eq!(dst_val.len(), n);
    debug_assert_eq!(pc.len(), n);
    debug_assert_eq!(imm.len(), n);
    debug_assert_eq!(packed.len(), n);

    let cpc_d = stream.clone_htod(current_pc)?;
    let npc_d = stream.clone_htod(next_pc_log)?;
    let s1_d = stream.clone_htod(src1_val)?;
    let s2_d = stream.clone_htod(src2_val)?;
    let dv_d = stream.clone_htod(dst_val)?;
    let pc_d = stream.clone_htod(pc)?;
    let imm_d = stream.clone_htod(imm)?;
    let pk_d = stream.clone_htod(packed)?;

    let mut rv1 = stream.alloc_zeros::<u64>(n)?;
    let mut rv2 = stream.alloc_zeros::<u64>(n)?;
    let mut arg2 = stream.alloc_zeros::<u64>(n)?;
    let mut res = stream.alloc_zeros::<u64>(n)?;
    let mut rvd = stream.alloc_zeros::<u64>(n)?;
    let mut next_pc = stream.alloc_zeros::<u64>(n)?;
    let mut flags = stream.alloc_zeros::<u8>(n)?;
    let mut commit_buf_addr = stream.alloc_zeros::<u64>(n)?;
    let mut commit_count = stream.alloc_zeros::<u64>(n)?;
    let mut keccak_state_addr = stream.alloc_zeros::<u64>(n)?;

    let n_u64 = n as u64;
    unsafe {
        stream
            .launch_builder(&be.build_cpu_ops)
            .arg(&n_u64)
            .arg(&cpc_d)
            .arg(&npc_d)
            .arg(&s1_d)
            .arg(&s2_d)
            .arg(&dv_d)
            .arg(&pc_d)
            .arg(&imm_d)
            .arg(&pk_d)
            .arg(&mut rv1)
            .arg(&mut rv2)
            .arg(&mut arg2)
            .arg(&mut res)
            .arg(&mut rvd)
            .arg(&mut next_pc)
            .arg(&mut flags)
            .arg(&mut commit_buf_addr)
            .arg(&mut commit_count)
            .arg(&mut keccak_state_addr)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }

    Ok(DeviceCpuOps {
        rv1: stream.clone_dtoh(&rv1)?,
        rv2: stream.clone_dtoh(&rv2)?,
        arg2: stream.clone_dtoh(&arg2)?,
        res: stream.clone_dtoh(&res)?,
        rvd: stream.clone_dtoh(&rvd)?,
        next_pc: stream.clone_dtoh(&next_pc)?,
        flags: stream.clone_dtoh(&flags)?,
        commit_buf_addr: stream.clone_dtoh(&commit_buf_addr)?,
        commit_count: stream.clone_dtoh(&commit_count)?,
        keccak_state_addr: stream.clone_dtoh(&keccak_state_addr)?,
    })
}

/// Per-cycle chip **route flags** computed ONCE from the resident packed decode (+ branch_cond
/// flags byte) and shared by every chip builder. Without this, each of the ~9 resident chip cores
/// re-allocated 6×`n` u32 flag arrays (~96 MB) and re-ran the routing kernels over all cycles —
/// a ~47 ms/chip floor. Computing them once (in the `DeviceCpuOpsResident` constructors) turns
/// each chip's route into a cheap slice read. `alu` = the 6 chipop_alu_route flags in fixed order
/// [LT, SHIFT, EQ, BYTEWISE, MUL, DVRM]; the rest are the per-source routes.
pub struct DeviceChipRoutes {
    pub alu: Vec<CudaSlice<u32>>,
    pub cpu32: CudaSlice<u32>,
    pub load: CudaSlice<u32>,
    pub store: CudaSlice<u32>,
    pub cpu32_shift: CudaSlice<u32>,
    pub cpu32_mul: CudaSlice<u32>,
    pub cpu32_dvrm: CudaSlice<u32>,
    pub branch: CudaSlice<u32>,
}

/// Compute all chip route flags in ONE pass over the cycles (see [`DeviceChipRoutes`]). Reads the
/// resident `packed` decode + `flags` (branch_cond) buffers; synchronizes before returning so the
/// flags are safe to read from any downstream stream.
fn compute_chip_routes(
    be: &crate::device::Backend,
    packed: &CudaSlice<u64>,
    flags: &CudaSlice<u8>,
    n: usize,
) -> Result<DeviceChipRoutes> {
    let stream = be.next_stream();
    let n_u64 = n as u64;
    let cfg = LaunchConfig::for_num_elems(n.max(1) as u32);
    let mut f0 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f1 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f2 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f3 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f4 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f5 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut cpu32 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut load = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut store = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut cpu32_shift = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut cpu32_mul = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut cpu32_dvrm = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut branch = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut store_unused = stream.alloc_zeros::<u32>(n.max(1))?;
    if n > 0 {
        unsafe {
            stream
                .launch_builder(&be.chipop_alu_route)
                .arg(&n_u64).arg(packed)
                .arg(&mut f0).arg(&mut f1).arg(&mut f2).arg(&mut f3).arg(&mut f4).arg(&mut f5)
                .launch(cfg)?;
            stream.launch_builder(&be.cpu32_route).arg(&n_u64).arg(packed).arg(&mut cpu32).launch(cfg)?;
            stream.launch_builder(&be.load_route).arg(&n_u64).arg(packed).arg(&mut load).launch(cfg)?;
            stream.launch_builder(&be.store_route).arg(&n_u64).arg(packed).arg(&mut store).launch(cfg)?;
            stream.launch_builder(&be.cpu32_shift_route).arg(&n_u64).arg(packed).arg(&mut cpu32_shift).launch(cfg)?;
            stream.launch_builder(&be.cpu32_mul_route).arg(&n_u64).arg(packed).arg(&mut cpu32_mul).launch(cfg)?;
            stream.launch_builder(&be.cpu32_dvrm_route).arg(&n_u64).arg(packed).arg(&mut cpu32_dvrm).launch(cfg)?;
            stream
                .launch_builder(&be.chipop_branch_store_route)
                .arg(&n_u64).arg(packed).arg(flags).arg(&mut branch).arg(&mut store_unused)
                .launch(cfg)?;
        }
    }
    stream.synchronize()?;
    Ok(DeviceChipRoutes {
        alu: vec![f0, f1, f2, f3, f4, f5],
        cpu32,
        load,
        store,
        cpu32_shift,
        cpu32_mul,
        cpu32_dvrm,
        branch,
    })
}

/// The **resident cpu_ops seam**: the Phase-0 device `CpuOperation` fields kept ON DEVICE (no
/// download), plus the decode inputs the chips also read. A single log upload feeds every chip
/// builder in place — eliminating the per-chip full-SoA re-uploads. Buffers are synchronized and
/// context-resident, so downstream chip builders can consume them on any stream. `routes` holds
/// the per-cycle chip route flags computed ONCE (see [`DeviceChipRoutes`]).
pub struct DeviceCpuOpsResident {
    pub n: usize,
    pub packed: CudaSlice<u64>,
    pub imm: CudaSlice<u64>,
    pub pc: CudaSlice<u64>,
    pub rv1: CudaSlice<u64>,
    pub rv2: CudaSlice<u64>,
    pub arg2: CudaSlice<u64>,
    pub res: CudaSlice<u64>,
    pub rvd: CudaSlice<u64>,
    pub flags: CudaSlice<u8>,
    pub routes: DeviceChipRoutes,
}

/// Build the Phase-0 cpu_ops on device and KEEP them resident (see [`DeviceCpuOpsResident`]).
/// Inputs are the one-time per-cycle Log SoA + decode SoA (uploaded once). Everything a chip
/// builder needs — `packed`, `imm`, `pc`, `rv1`, `rv2`, `arg2`, `res`, `rvd`, `flags` — comes
/// back as device buffers with no host round-trip.
#[allow(clippy::too_many_arguments)]
pub fn gpu_build_cpu_ops_resident(
    current_pc: &[u64],
    next_pc_log: &[u64],
    src1_val: &[u64],
    src2_val: &[u64],
    dst_val: &[u64],
    pc: &[u64],
    imm: &[u64],
    packed: &[u64],
) -> Result<DeviceCpuOpsResident> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = current_pc.len();
    let cpc_d = stream.clone_htod(current_pc)?;
    let npc_d = stream.clone_htod(next_pc_log)?;
    let s1_d = stream.clone_htod(src1_val)?;
    let s2_d = stream.clone_htod(src2_val)?;
    let dv_d = stream.clone_htod(dst_val)?;
    let pc_d = stream.clone_htod(pc)?;
    let imm_d = stream.clone_htod(imm)?;
    let pk_d = stream.clone_htod(packed)?;

    let mut rv1 = stream.alloc_zeros::<u64>(n.max(1))?;
    let mut rv2 = stream.alloc_zeros::<u64>(n.max(1))?;
    let mut arg2 = stream.alloc_zeros::<u64>(n.max(1))?;
    let mut res = stream.alloc_zeros::<u64>(n.max(1))?;
    let mut rvd = stream.alloc_zeros::<u64>(n.max(1))?;
    let mut next_pc = stream.alloc_zeros::<u64>(n.max(1))?;
    let mut flags = stream.alloc_zeros::<u8>(n.max(1))?;
    let mut commit_buf_addr = stream.alloc_zeros::<u64>(n.max(1))?;
    let mut commit_count = stream.alloc_zeros::<u64>(n.max(1))?;
    let mut keccak_state_addr = stream.alloc_zeros::<u64>(n.max(1))?;
    if n > 0 {
        let n_u64 = n as u64;
        unsafe {
            stream
                .launch_builder(&be.build_cpu_ops)
                .arg(&n_u64)
                .arg(&cpc_d)
                .arg(&npc_d)
                .arg(&s1_d)
                .arg(&s2_d)
                .arg(&dv_d)
                .arg(&pc_d)
                .arg(&imm_d)
                .arg(&pk_d)
                .arg(&mut rv1)
                .arg(&mut rv2)
                .arg(&mut arg2)
                .arg(&mut res)
                .arg(&mut rvd)
                .arg(&mut next_pc)
                .arg(&mut flags)
                .arg(&mut commit_buf_addr)
                .arg(&mut commit_count)
                .arg(&mut keccak_state_addr)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    stream.synchronize()?;
    let routes = compute_chip_routes(be, &pk_d, &flags, n)?;
    Ok(DeviceCpuOpsResident {
        n,
        packed: pk_d,
        imm: imm_d,
        pc: pc_d,
        rv1,
        rv2,
        arg2,
        res,
        rvd,
        flags,
        routes,
    })
}

/// Upload ALREADY-COMPUTED cpu_op fields into a resident device buffer set — the p5 single-build
/// seam. Unlike [`gpu_build_cpu_ops_resident`] (which recomputes rv1/rv2/arg2/res/rvd/flags on
/// device from the raw Log SoA), this takes the host-side `CpuOperation` fields verbatim and just
/// uploads them ONCE, so every chip's `*_from_devops` builder reads them in place with zero
/// re-uploads. Exact host-parity by construction (no device recompute).
#[allow(clippy::too_many_arguments)]
pub fn gpu_upload_cpu_ops_resident(
    packed: &[u64],
    imm: &[u64],
    pc: &[u64],
    rv1: &[u64],
    rv2: &[u64],
    arg2: &[u64],
    res: &[u64],
    rvd: &[u64],
    flags: &[u8],
) -> Result<DeviceCpuOpsResident> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let pk_d = stream.clone_htod(packed)?;
    let imm_d = stream.clone_htod(imm)?;
    let pc_d = stream.clone_htod(pc)?;
    let rv1_d = stream.clone_htod(rv1)?;
    let rv2_d = stream.clone_htod(rv2)?;
    let arg2_d = stream.clone_htod(arg2)?;
    let res_d = stream.clone_htod(res)?;
    let rvd_d = stream.clone_htod(rvd)?;
    let flags_d = stream.clone_htod(flags)?;
    stream.synchronize()?;
    let routes = compute_chip_routes(be, &pk_d, &flags_d, n)?;
    Ok(DeviceCpuOpsResident {
        n,
        packed: pk_d,
        imm: imm_d,
        pc: pc_d,
        rv1: rv1_d,
        rv2: rv2_d,
        arg2: arg2_d,
        res: res_d,
        rvd: rvd_d,
        flags: flags_d,
        routes,
    })
}

/// Fill the CPU32 table straight from the resident cpu_ops (no upload) — the fully-resident
/// path: log SoA uploaded once → device cpu_ops → this reads them in place. Returns the filled
/// device buffer + auto-sized `num_rows`.
pub fn gpu_build_cpu32_resident_from_devops(
    ops: &DeviceCpuOpsResident,
) -> Result<(CudaSlice<u64>, usize)> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = ops.n;
    let ncols = crate::trace_cpu::CPU32_NCOLS;
    let n_u64 = n as u64;
    let flag = &ops.routes.cpu32; // route-once: precomputed
    let (excl, total) = crate::trace_walk::excl_scan(be, &stream, flag, n.max(1))?;
    let rows = total as usize;
    let num_rows = rows.next_power_of_two().max(4);
    let mut ops_dev = stream.alloc_zeros::<u64>(rows.max(1) * 8)?;
    if n > 0 && rows > 0 {
        unsafe {
            stream
                .launch_builder(&be.build_cpu32_ops)
                .arg(&n_u64)
                .arg(&ops.packed)
                .arg(&ops.rv1)
                .arg(&ops.rv2)
                .arg(&ops.imm)
                .arg(&ops.pc)
                .arg(flag)
                .arg(&excl)
                .arg(&mut ops_dev)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    let mut out = stream.alloc_zeros::<u64>(num_rows * ncols)?;
    let (rows_u64, num_rows_u64) = (rows as u64, num_rows as u64);
    unsafe {
        stream
            .launch_builder(&be.cpu32_fill)
            .arg(&ops_dev)
            .arg(&rows_u64)
            .arg(&num_rows_u64)
            .arg(&mut out)
            .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
    }
    stream.synchronize()?;
    Ok((out, num_rows))
}

/// Fill the LOAD table from the resident cpu_ops (no upload), auto-sized. Reads
/// `packed`/`res`/`rvd` from the shared device buffers.
pub fn gpu_build_load_resident_from_devops(
    ops: &DeviceCpuOpsResident,
) -> Result<(CudaSlice<u64>, usize)> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = ops.n;
    let ncols = crate::trace_cpu::LOAD_NCOLS;
    let n_u64 = n as u64;
    let flag = &ops.routes.load; // route-once: precomputed
    let (excl, total) = crate::trace_walk::excl_scan(be, &stream, flag, n.max(1))?;
    let rows = total as usize;
    let num_rows = rows.next_power_of_two().max(4);
    let mut ops_dev = stream.alloc_zeros::<u64>(rows.max(1) * 7)?;
    if n > 0 && rows > 0 {
        unsafe {
            stream
                .launch_builder(&be.build_load_ops)
                .arg(&n_u64)
                .arg(&ops.packed)
                .arg(&ops.res)
                .arg(&ops.rvd)
                .arg(flag)
                .arg(&excl)
                .arg(&mut ops_dev)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    let mut out = stream.alloc_zeros::<u64>(num_rows * ncols)?;
    let (rows_u64, num_rows_u64) = (rows as u64, num_rows as u64);
    unsafe {
        stream
            .launch_builder(&be.load_fill)
            .arg(&ops_dev)
            .arg(&rows_u64)
            .arg(&num_rows_u64)
            .arg(&mut out)
            .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
    }
    stream.synchronize()?;
    Ok((out, num_rows))
}

/// Fill the STORE table from the resident cpu_ops (no upload), auto-sized. Reads
/// `packed`/`res`/`rv2` from the shared device buffers.
pub fn gpu_build_store_resident_from_devops(
    ops: &DeviceCpuOpsResident,
) -> Result<(CudaSlice<u64>, usize)> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = ops.n;
    let ncols = crate::trace_cpu::STORE_NCOLS;
    let n_u64 = n as u64;
    let flag = &ops.routes.store; // route-once: precomputed
    let (excl, total) = crate::trace_walk::excl_scan(be, &stream, flag, n.max(1))?;
    let rows = total as usize;
    let num_rows = rows.next_power_of_two().max(4);
    let mut ops_dev = stream.alloc_zeros::<u64>(rows.max(1) * 4)?;
    if n > 0 && rows > 0 {
        unsafe {
            stream
                .launch_builder(&be.build_store_ops)
                .arg(&n_u64)
                .arg(&ops.packed)
                .arg(&ops.res)
                .arg(&ops.rv2)
                .arg(flag)
                .arg(&excl)
                .arg(&mut ops_dev)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    let mut out = stream.alloc_zeros::<u64>(num_rows * ncols)?;
    let (rows_u64, num_rows_u64) = (rows as u64, num_rows as u64);
    unsafe {
        stream
            .launch_builder(&be.store_fill)
            .arg(&ops_dev)
            .arg(&rows_u64)
            .arg(&num_rows_u64)
            .arg(&mut out)
            .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
    }
    stream.synchronize()?;
    Ok((out, num_rows))
}

/// One state-free ALU chip's compacted op stream, in program order. `a`/`b` are the raw
/// operands the chip constructor takes; `alu_flags` holds `signed` @ bit 5 and
/// `signed2_or_invert` @ bit 6. For LT: `a`=lhs, `b`=rhs. For SHIFT: `a`=value,
/// `b`=shift_amount.
pub struct DeviceAluChipOps {
    pub a: Vec<u64>,
    pub b: Vec<u64>,
    pub alu_flags: Vec<u8>,
}

/// The six state-free ALU chip-op streams extracted from the resident cpu_ops, each in
/// program order. All share the `(rv1, arg2, alu_flags)` gather; the route predicate
/// (on `alu_op`) is what differs. See [`gpu_extract_alu_chipops`].
pub struct DeviceAluChips {
    pub lt: DeviceAluChipOps,
    pub shift: DeviceAluChipOps,
    pub eq: DeviceAluChipOps,
    pub bytewise: DeviceAluChipOps,
    pub mul: DeviceAluChipOps,
    pub dvrm: DeviceAluChipOps,
}

/// Extract the six instruction-driven state-free ALU chip-op streams on device from the
/// resident cpu_op fields (`packed` decode + `rv1` + `arg2`), reproducing exactly the
/// per-cycle `cpu_ops.iter().filter().map()` projections in trace_builder.rs:
/// - LT     — `!word_instr && is_lt()`      → `(rv1, arg2, signed, invert)`
/// - SHIFT  — `!word_instr && is_shift()`   → `(rv1, arg2, signed, invert)`
/// - EQ     — `!word_instr && is_eq()`      → `(rv1, arg2, invert)`
/// - BYTEWISE — `!word_instr && (and|or|xor)` → `(rv1, arg2, alu_op)`
/// - MUL    — `!word_instr && is_mul()`     → `(rv1, arg2, signed, signed2, muldiv)`
/// - DVRM   — `!word_instr && is_divrem()`  → `(rv1, arg2, signed, muldiv)`
///
/// All six flag bits (signed@5, signed2/invert@6, muldiv@7) and `alu_op`@0-4 are carried in
/// the returned `alu_flags` byte, so a single gather serves every chip. Route → exclusive-
/// scan compact → gather, all on device. Phase 3 (state-free ALU chips).
pub fn gpu_extract_alu_chipops(packed: &[u64], rv1: &[u64], arg2: &[u64]) -> Result<DeviceAluChips> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let empty = || DeviceAluChipOps {
        a: vec![],
        b: vec![],
        alu_flags: vec![],
    };
    let empty_chips = || DeviceAluChips {
        lt: empty(),
        shift: empty(),
        eq: empty(),
        bytewise: empty(),
        mul: empty(),
        dvrm: empty(),
    };
    if n == 0 {
        return Ok(empty_chips());
    }
    debug_assert_eq!(rv1.len(), n);
    debug_assert_eq!(arg2.len(), n);

    let pk_d = stream.clone_htod(packed)?;
    let rv1_d = stream.clone_htod(rv1)?;
    let arg2_d = stream.clone_htod(arg2)?;

    // Route: per-cycle flags for all 6 chips in one pass.
    let mut flag_lt = stream.alloc_zeros::<u32>(n)?;
    let mut flag_shift = stream.alloc_zeros::<u32>(n)?;
    let mut flag_eq = stream.alloc_zeros::<u32>(n)?;
    let mut flag_bytewise = stream.alloc_zeros::<u32>(n)?;
    let mut flag_mul = stream.alloc_zeros::<u32>(n)?;
    let mut flag_dvrm = stream.alloc_zeros::<u32>(n)?;
    let n_u64 = n as u64;
    unsafe {
        stream
            .launch_builder(&be.chipop_alu_route)
            .arg(&n_u64)
            .arg(&pk_d)
            .arg(&mut flag_lt)
            .arg(&mut flag_shift)
            .arg(&mut flag_eq)
            .arg(&mut flag_bytewise)
            .arg(&mut flag_mul)
            .arg(&mut flag_dvrm)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }

    // Compact + gather each chip. `in_a` = rv1, `in_b` = arg2 for all chips.
    let gather = |flag: &cudarc::driver::CudaSlice<u32>| -> Result<DeviceAluChipOps> {
        let (excl, total) = crate::trace_walk::excl_scan(be, &stream, flag, n)?;
        let m = total as usize;
        if m == 0 {
            return Ok(empty());
        }
        let mut out_a = stream.alloc_zeros::<u64>(m)?;
        let mut out_b = stream.alloc_zeros::<u64>(m)?;
        let mut out_f = stream.alloc_zeros::<u8>(m)?;
        unsafe {
            stream
                .launch_builder(&be.chipop_gather)
                .arg(&n_u64)
                .arg(&rv1_d)
                .arg(&arg2_d)
                .arg(&pk_d)
                .arg(flag)
                .arg(&excl)
                .arg(&mut out_a)
                .arg(&mut out_b)
                .arg(&mut out_f)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
        Ok(DeviceAluChipOps {
            a: stream.clone_dtoh(&out_a)?,
            b: stream.clone_dtoh(&out_b)?,
            alu_flags: stream.clone_dtoh(&out_f)?,
        })
    };

    Ok(DeviceAluChips {
        lt: gather(&flag_lt)?,
        shift: gather(&flag_shift)?,
        eq: gather(&flag_eq)?,
        bytewise: gather(&flag_bytewise)?,
        mul: gather(&flag_mul)?,
        dvrm: gather(&flag_dvrm)?,
    })
}

/// A compacted 4-column chip-op stream, program order. Column meaning is chip-specific
/// (see [`gpu_extract_branch_store`]).
pub struct DeviceGather4 {
    pub c0: Vec<u64>,
    pub c1: Vec<u64>,
    pub c2: Vec<u64>,
    pub c3: Vec<u64>,
}

/// Extract the BRANCH and STORE chip-op streams on device — the two remaining state-free
/// (cpu_ops-projection) chips, each with its own field set:
/// - BRANCH — route `branch_cond` (cpu_ops `flags` bit 0) → `(pc, imm, rv1, packed)`,
///   from which the consumer reads `jalr` (`mem_flags` bit 0). Matches
///   `BranchOperation::new(pc, imm, rv1, jalr)`.
/// - STORE — route `is_store()` (`memory && mem_flags bit 0`) → `(res, timestamp, rv2,
///   packed)`, from which the consumer reads `mem_bytes`. Matches
///   `StoreOperation::new(res, timestamp, rv2, mem_bytes)`.
///
/// Returns `(branch, store)`. `flags` is the per-cycle `build_cpu_ops` flags byte; `pc`/
/// `imm` the decode SoA; `res`/`ts`/`rv2` the cpu_op fields; `packed` the decode fields.
#[allow(clippy::too_many_arguments)]
pub fn gpu_extract_branch_store(
    packed: &[u64],
    flags: &[u8],
    pc: &[u64],
    imm: &[u64],
    rv1: &[u64],
    res: &[u64],
    ts: &[u64],
    rv2: &[u64],
) -> Result<(DeviceGather4, DeviceGather4)> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let empty = || DeviceGather4 {
        c0: vec![],
        c1: vec![],
        c2: vec![],
        c3: vec![],
    };
    if n == 0 {
        return Ok((empty(), empty()));
    }
    for s in [flags.len()] {
        debug_assert_eq!(s, n);
    }
    for s in [pc.len(), imm.len(), rv1.len(), res.len(), ts.len(), rv2.len()] {
        debug_assert_eq!(s, n);
    }

    let pk_d = stream.clone_htod(packed)?;
    let fl_d = stream.clone_htod(flags)?;
    let pc_d = stream.clone_htod(pc)?;
    let imm_d = stream.clone_htod(imm)?;
    let rv1_d = stream.clone_htod(rv1)?;
    let res_d = stream.clone_htod(res)?;
    let ts_d = stream.clone_htod(ts)?;
    let rv2_d = stream.clone_htod(rv2)?;

    let mut flag_branch = stream.alloc_zeros::<u32>(n)?;
    let mut flag_store = stream.alloc_zeros::<u32>(n)?;
    let n_u64 = n as u64;
    unsafe {
        stream
            .launch_builder(&be.chipop_branch_store_route)
            .arg(&n_u64)
            .arg(&pk_d)
            .arg(&fl_d)
            .arg(&mut flag_branch)
            .arg(&mut flag_store)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }

    let gather4 = |flag: &cudarc::driver::CudaSlice<u32>,
                   in0: &cudarc::driver::CudaSlice<u64>,
                   in1: &cudarc::driver::CudaSlice<u64>,
                   in2: &cudarc::driver::CudaSlice<u64>|
     -> Result<DeviceGather4> {
        let (excl, total) = crate::trace_walk::excl_scan(be, &stream, flag, n)?;
        let m = total as usize;
        if m == 0 {
            return Ok(empty());
        }
        let mut out0 = stream.alloc_zeros::<u64>(m)?;
        let mut out1 = stream.alloc_zeros::<u64>(m)?;
        let mut out2 = stream.alloc_zeros::<u64>(m)?;
        let mut out3 = stream.alloc_zeros::<u64>(m)?;
        unsafe {
            stream
                .launch_builder(&be.chipop_gather4)
                .arg(&n_u64)
                .arg(in0)
                .arg(in1)
                .arg(in2)
                .arg(&pk_d)
                .arg(flag)
                .arg(&excl)
                .arg(&mut out0)
                .arg(&mut out1)
                .arg(&mut out2)
                .arg(&mut out3)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
        Ok(DeviceGather4 {
            c0: stream.clone_dtoh(&out0)?,
            c1: stream.clone_dtoh(&out1)?,
            c2: stream.clone_dtoh(&out2)?,
            c3: stream.clone_dtoh(&out3)?,
        })
    };

    let branch = gather4(&flag_branch, &pc_d, &imm_d, &rv1_d)?;
    let store = gather4(&flag_store, &res_d, &ts_d, &rv2_d)?;
    Ok((branch, store))
}

/// Build the CPU32 (word `*W`) chip-op rows on device — the packed `pack_cpu32_op` SoA
/// (8 u64/op) per word-instr cycle, compacted in program order, with `res` computed on
/// device (compute_aux + cpu32_res: SHIFT/MUL/DVRM arithmetic). Reproduces exactly
/// `build_cpu32_op` fed by `chunk_and_generate(cpu32_ops, …)`. Inputs are the resident
/// cpu_op fields: `packed` decode, `rv1`, `rv2`, `imm`, `pc` (= decode.pc). Returns the
/// flat `rows * 8` packed buffer (feeds `gpu_build_cpu32_trace` / `cpu32_fill`) + row count.
pub fn gpu_build_cpu32_ops(
    packed: &[u64],
    rv1: &[u64],
    rv2: &[u64],
    imm: &[u64],
    pc: &[u64],
) -> Result<(Vec<u64>, usize)> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    if n == 0 {
        return Ok((vec![], 0));
    }
    debug_assert_eq!(rv1.len(), n);
    debug_assert_eq!(rv2.len(), n);
    debug_assert_eq!(imm.len(), n);
    debug_assert_eq!(pc.len(), n);

    let pk_d = stream.clone_htod(packed)?;
    let rv1_d = stream.clone_htod(rv1)?;
    let rv2_d = stream.clone_htod(rv2)?;
    let imm_d = stream.clone_htod(imm)?;
    let pc_d = stream.clone_htod(pc)?;

    let mut flag = stream.alloc_zeros::<u32>(n)?;
    let n_u64 = n as u64;
    unsafe {
        stream
            .launch_builder(&be.cpu32_route)
            .arg(&n_u64)
            .arg(&pk_d)
            .arg(&mut flag)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
    let (excl, total) = crate::trace_walk::excl_scan(be, &stream, &flag, n)?;
    let rows = total as usize;
    if rows == 0 {
        return Ok((vec![], 0));
    }
    let mut out = stream.alloc_zeros::<u64>(rows * 8)?;
    unsafe {
        stream
            .launch_builder(&be.build_cpu32_ops)
            .arg(&n_u64)
            .arg(&pk_d)
            .arg(&rv1_d)
            .arg(&rv2_d)
            .arg(&imm_d)
            .arg(&pc_d)
            .arg(&flag)
            .arg(&excl)
            .arg(&mut out)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok((host, rows))
}

/// Device-resident CPU32 fill: same chain as [`gpu_build_cpu32_resident`] but returns the
/// filled `num_rows * CPU32_NCOLS` buffer **on device** (no download), for the production
/// pipeline to hand to the LDE via `TraceTable::set_main_input_dev`. `num_rows` = padded
/// height (caller sizes it as `cpu32_count.next_power_of_two().max(4)`).
#[allow(clippy::too_many_arguments)]
pub fn gpu_build_cpu32_resident_dev(
    packed: &[u64],
    rv1: &[u64],
    rv2: &[u64],
    imm: &[u64],
    pc: &[u64],
    num_rows: usize,
) -> Result<CudaSlice<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let ncols = crate::trace_cpu::CPU32_NCOLS;
    let n_u64 = n as u64;
    let pk_d = stream.clone_htod(packed)?;
    let rv1_d = stream.clone_htod(rv1)?;
    let rv2_d = stream.clone_htod(rv2)?;
    let imm_d = stream.clone_htod(imm)?;
    let pc_d = stream.clone_htod(pc)?;
    let mut flag = stream.alloc_zeros::<u32>(n.max(1))?;
    if n > 0 {
        unsafe {
            stream
                .launch_builder(&be.cpu32_route)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&mut flag)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    let (excl, total) = crate::trace_walk::excl_scan(be, &stream, &flag, n.max(1))?;
    let rows = total as usize;
    let mut ops_dev = stream.alloc_zeros::<u64>(rows.max(1) * 8)?;
    if n > 0 && rows > 0 {
        unsafe {
            stream
                .launch_builder(&be.build_cpu32_ops)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&rv1_d)
                .arg(&rv2_d)
                .arg(&imm_d)
                .arg(&pc_d)
                .arg(&flag)
                .arg(&excl)
                .arg(&mut ops_dev)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    let mut out = stream.alloc_zeros::<u64>(num_rows * ncols)?;
    let (rows_u64, num_rows_u64) = (rows as u64, num_rows as u64);
    unsafe {
        stream
            .launch_builder(&be.cpu32_fill)
            .arg(&ops_dev)
            .arg(&rows_u64)
            .arg(&num_rows_u64)
            .arg(&mut out)
            .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
    }
    stream.synchronize()?;
    Ok(out)
}

/// RESIDENT CPU32 chain (proof of the device→device seam): build the CPU32 op rows on
/// device from the cpu_op fields AND fill the CPU32 trace table, with **no host round-trip
/// for the intermediate op buffer** — the only host transfers are the input upload and the
/// final download (which in the real pipeline is replaced by feeding the LDE resident). The
/// filled table is byte-identical to the host path: the device op-build is already validated
/// == `build_cpu32_op`, and `cpu32_fill` is deterministic. `num_rows` = padded table height.
/// Returns the filled `num_rows * CPU32_NCOLS` buffer.
#[allow(clippy::too_many_arguments)]
pub fn gpu_build_cpu32_resident(
    packed: &[u64],
    rv1: &[u64],
    rv2: &[u64],
    imm: &[u64],
    pc: &[u64],
    num_rows: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let ncols = crate::trace_cpu::CPU32_NCOLS;
    let n_u64 = n as u64;

    let pk_d = stream.clone_htod(packed)?;
    let rv1_d = stream.clone_htod(rv1)?;
    let rv2_d = stream.clone_htod(rv2)?;
    let imm_d = stream.clone_htod(imm)?;
    let pc_d = stream.clone_htod(pc)?;

    // Route + compact word-instr cycles.
    let mut flag = stream.alloc_zeros::<u32>(n.max(1))?;
    if n > 0 {
        unsafe {
            stream
                .launch_builder(&be.cpu32_route)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&mut flag)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    let (excl, total) = crate::trace_walk::excl_scan(be, &stream, &flag, n.max(1))?;
    let rows = total as usize;

    // Build the packed CPU32 op rows on device (stays resident).
    let mut ops_dev = stream.alloc_zeros::<u64>(rows.max(1) * 8)?;
    if n > 0 && rows > 0 {
        unsafe {
            stream
                .launch_builder(&be.build_cpu32_ops)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&rv1_d)
                .arg(&rv2_d)
                .arg(&imm_d)
                .arg(&pc_d)
                .arg(&flag)
                .arg(&excl)
                .arg(&mut ops_dev)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }

    // Fill the CPU32 trace table directly from the resident op buffer (no re-upload).
    let mut out = stream.alloc_zeros::<u64>(num_rows * ncols)?;
    let rows_u64 = rows as u64;
    let num_rows_u64 = num_rows as u64;
    unsafe {
        stream
            .launch_builder(&be.cpu32_fill)
            .arg(&ops_dev)
            .arg(&rows_u64)
            .arg(&num_rows_u64)
            .arg(&mut out)
            .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// RESIDENT LT chain — the deduped-chip capstone: cpu_op fields → device route+extract →
/// device dedup (`dedup3_core`) → device pack → device `lt_fill`, entirely on device (no
/// intermediate host round-trip). LT's bus is order-independent (LogUp), so the filled table
/// matches the host path as a MULTISET of rows (device order is sorted, host is HashMap order).
/// `num_rows` = padded height. Returns the filled `num_rows * LT_NCOLS` buffer.
pub fn gpu_build_lt_resident(
    packed: &[u64],
    rv1: &[u64],
    arg2: &[u64],
    num_rows: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let ncols = crate::trace_cpu::LT_NCOLS;
    let n_u64 = n as u64;
    let pk_d = stream.clone_htod(packed)?;
    let rv1_d = stream.clone_htod(rv1)?;
    let arg2_d = stream.clone_htod(arg2)?;

    // Route LT cycles (chipop_alu_route writes all 6 chip flags; we use LT).
    let mut f_lt = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f_shift = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f_eq = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f_byte = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f_mul = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f_dvrm = stream.alloc_zeros::<u32>(n.max(1))?;
    if n > 0 {
        unsafe {
            stream
                .launch_builder(&be.chipop_alu_route)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&mut f_lt)
                .arg(&mut f_shift)
                .arg(&mut f_eq)
                .arg(&mut f_byte)
                .arg(&mut f_mul)
                .arg(&mut f_dvrm)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    let (excl, total) = crate::trace_walk::excl_scan(be, &stream, &f_lt, n.max(1))?;
    let rows = total as usize;

    // Extract the LT dedup key (alu_flags, lhs=rv1, rhs=arg2) for the compacted LT ops.
    let mut k0 = stream.alloc_zeros::<u64>(rows.max(1))?;
    let mut k1 = stream.alloc_zeros::<u64>(rows.max(1))?;
    let mut k2 = stream.alloc_zeros::<u64>(rows.max(1))?;
    if n > 0 && rows > 0 {
        unsafe {
            stream
                .launch_builder(&be.lt_key_gather)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&rv1_d)
                .arg(&arg2_d)
                .arg(&f_lt)
                .arg(&excl)
                .arg(&mut k0)
                .arg(&mut k1)
                .arg(&mut k2)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }

    // Dedup on device → unique (alu_flags, lhs, rhs, mult).
    let mut out = stream.alloc_zeros::<u64>(num_rows * ncols)?;
    if rows > 0 {
        let (uk0, uk1, uk2, umult, m) =
            crate::trace_walk::dedup3_core(be, &stream, &k0, &k1, &k2, rows)?;
        // Pack unique rows into the LT fill stride, then fill — all resident.
        let mut ops_dev = stream.alloc_zeros::<u64>(m.max(1) * crate::trace_cpu::LT_STRIDE)?;
        let m_u64 = m as u64;
        if m > 0 {
            unsafe {
                stream
                    .launch_builder(&be.dedup_pack_abf)
                    .arg(&m_u64)
                    .arg(&uk0)
                    .arg(&uk1)
                    .arg(&uk2)
                    .arg(&umult)
                    .arg(&mut ops_dev)
                    .launch(LaunchConfig::for_num_elems(m as u32))?;
            }
        }
        let num_rows_u64 = num_rows as u64;
        unsafe {
            stream
                .launch_builder(&be.lt_fill)
                .arg(&ops_dev)
                .arg(&m_u64)
                .arg(&num_rows_u64)
                .arg(&mut out)
                .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
        }
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// RESIDENT LT chain with a MERGED op source: instruction-driven LT ⊕ **dvrm-derived** LT
/// (`LtOperation::new(abs_r, abs_d, false)` per is_divrem cycle). Both key streams are
/// gathered into one buffer (dvrm appended after the instruction ops) and deduped together,
/// demonstrating on-device multi-source merging for a production chip table. (The remaining
/// LT source — memw-derived ts-range LTs — is gated on the Phase-2 MEMW routing.) Returns the
/// filled `num_rows * LT_NCOLS` buffer.
pub fn gpu_build_lt_instr_dvrm_resident(
    packed: &[u64],
    rv1: &[u64],
    arg2: &[u64],
    num_rows: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let ncols = crate::trace_cpu::LT_NCOLS;
    let n_u64 = n as u64;
    let pk_d = stream.clone_htod(packed)?;
    let rv1_d = stream.clone_htod(rv1)?;
    let arg2_d = stream.clone_htod(arg2)?;

    let mut f0 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f1 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f2 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f3 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f4 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f5 = stream.alloc_zeros::<u32>(n.max(1))?;
    if n > 0 {
        unsafe {
            stream
                .launch_builder(&be.chipop_alu_route)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&mut f0)
                .arg(&mut f1)
                .arg(&mut f2)
                .arg(&mut f3)
                .arg(&mut f4)
                .arg(&mut f5)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    // f0 = LT, f5 = DVRM.
    let (excl_lt, total_lt) = crate::trace_walk::excl_scan(be, &stream, &f0, n.max(1))?;
    let (excl_dv, total_dv) = crate::trace_walk::excl_scan(be, &stream, &f5, n.max(1))?;
    let rows_lt = total_lt as usize;
    let rows_dv = total_dv as usize;
    let total = rows_lt + rows_dv;

    let mut k0 = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut k1 = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut k2 = stream.alloc_zeros::<u64>(total.max(1))?;
    if n > 0 && rows_lt > 0 {
        unsafe {
            stream
                .launch_builder(&be.lt_key_gather)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&rv1_d)
                .arg(&arg2_d)
                .arg(&f0)
                .arg(&excl_lt)
                .arg(&mut k0)
                .arg(&mut k1)
                .arg(&mut k2)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    if n > 0 && rows_dv > 0 {
        let base = rows_lt as u64;
        unsafe {
            stream
                .launch_builder(&be.dvrm_lt_key_gather)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&rv1_d)
                .arg(&arg2_d)
                .arg(&f5)
                .arg(&excl_dv)
                .arg(&base)
                .arg(&mut k0)
                .arg(&mut k1)
                .arg(&mut k2)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }

    let mut out = stream.alloc_zeros::<u64>(num_rows * ncols)?;
    if total > 0 {
        let (uk0, uk1, uk2, umult, m) =
            crate::trace_walk::dedup3_core(be, &stream, &k0, &k1, &k2, total)?;
        let mut ops_dev = stream.alloc_zeros::<u64>(m.max(1) * crate::trace_cpu::LT_STRIDE)?;
        let m_u64 = m as u64;
        if m > 0 {
            unsafe {
                stream
                    .launch_builder(&be.dedup_pack_abf)
                    .arg(&m_u64)
                    .arg(&uk0)
                    .arg(&uk1)
                    .arg(&uk2)
                    .arg(&umult)
                    .arg(&mut ops_dev)
                    .launch(LaunchConfig::for_num_elems(m as u32))?;
            }
        }
        let num_rows_u64 = num_rows as u64;
        unsafe {
            stream
                .launch_builder(&be.lt_fill)
                .arg(&ops_dev)
                .arg(&m_u64)
                .arg(&num_rows_u64)
                .arg(&mut out)
                .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
        }
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// LT-resident-table STEP 2B: the FULL resident LT table from the resident cpu_ops seam (instruction
/// LT ⊕ dvrm→lt) MERGED with the memw→lt operand pairs (uploaded), device-resident (no download).
/// Reads `devops.packed/rv1/arg2` + the precomputed `devops.routes.alu[0]` (LT) / `[5]` (DVRM); appends
/// the memw→lt keys (k0=0) via `lt_memw_key_write`; single global dedup (`dedup3_core`) → pack → fill.
/// Returns the filled `num_rows*LT_NCOLS` device buffer + auto-sized `num_rows`. This is the last chip
/// table to go resident; it replaces the host `gpu_build_lt_tables(&lt_ops)` under gpu_full.
pub fn gpu_build_lt_full_resident_from_devops(
    ops: &DeviceCpuOpsResident,
    memw_lhs: &[u64],
    memw_rhs: &[u64],
    max_rows: usize,
) -> Result<Vec<(CudaSlice<u64>, usize)>> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = ops.n;
    let ncols = crate::trace_cpu::LT_NCOLS;
    let n_u64 = n as u64;
    let f_lt = &ops.routes.alu[0]; // LT (route-once)
    let f_dv = &ops.routes.alu[5]; // DVRM (route-once)
    let n_memw = memw_lhs.len();
    debug_assert_eq!(memw_rhs.len(), n_memw);

    let (excl_lt, total_lt) = crate::trace_walk::excl_scan(be, &stream, f_lt, n.max(1))?;
    let (excl_dv, total_dv) = crate::trace_walk::excl_scan(be, &stream, f_dv, n.max(1))?;
    let rows_lt = total_lt as usize;
    let rows_dv = total_dv as usize;
    let total = rows_lt + rows_dv + n_memw;

    let mut k0 = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut k1 = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut k2 = stream.alloc_zeros::<u64>(total.max(1))?;
    if n > 0 && rows_lt > 0 {
        unsafe {
            stream.launch_builder(&be.lt_key_gather).arg(&n_u64).arg(&ops.packed).arg(&ops.rv1)
                .arg(&ops.arg2).arg(f_lt).arg(&excl_lt).arg(&mut k0).arg(&mut k1).arg(&mut k2)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    if n > 0 && rows_dv > 0 {
        let base = rows_lt as u64;
        unsafe {
            stream.launch_builder(&be.dvrm_lt_key_gather).arg(&n_u64).arg(&ops.packed).arg(&ops.rv1)
                .arg(&ops.arg2).arg(f_dv).arg(&excl_dv).arg(&base).arg(&mut k0).arg(&mut k1).arg(&mut k2)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    if n_memw > 0 {
        let ml = stream.clone_htod(memw_lhs)?;
        let mr = stream.clone_htod(memw_rhs)?;
        let base = (rows_lt + rows_dv) as u64;
        let nm = n_memw as u64;
        unsafe {
            stream.launch_builder(&be.lt_memw_key_write).arg(&nm).arg(&ml).arg(&mr).arg(&base)
                .arg(&mut k0).arg(&mut k1).arg(&mut k2)
                .launch(LaunchConfig::for_num_elems(n_memw as u32))?;
        }
    }

    let stride = crate::trace_cpu::LT_STRIDE;
    if total == 0 {
        // Single empty (padded) table.
        let out = stream.alloc_zeros::<u64>(4 * ncols)?;
        stream.synchronize()?;
        return Ok(vec![(out, 4)]);
    }
    // Single GLOBAL dedup (bus-equivalent to the host per-chunk dedup — LT is a LogUp multiset), then
    // split the UNIQUE rows into ≤ `max_rows` chunks and fill each into its own table. LT dedups poorly
    // (memw→lt timestamps are mostly unique) so `m` can exceed one table (~2 chunks for ethrex_5tx).
    let (uk0, uk1, uk2, umult, m) =
        crate::trace_walk::dedup3_core(be, &stream, &k0, &k1, &k2, total)?;
    let mut ops_dev = stream.alloc_zeros::<u64>(m.max(1) * stride)?;
    if m > 0 {
        let m_u64 = m as u64;
        unsafe {
            stream.launch_builder(&be.dedup_pack_abf).arg(&m_u64).arg(&uk0).arg(&uk1).arg(&uk2)
                .arg(&umult).arg(&mut ops_dev).launch(LaunchConfig::for_num_elems(m as u32))?;
        }
    }
    let mut out_chunks = Vec::new();
    let n_chunks = m.div_ceil(max_rows).max(1);
    for c in 0..n_chunks {
        let start = c * max_rows;
        let m_chunk = (m - start).min(max_rows);
        let num_rows = m_chunk.next_power_of_two().max(4);
        let mut out = stream.alloc_zeros::<u64>(num_rows * ncols)?;
        // Copy this chunk's packed rows to a fresh buffer so `lt_fill` reads row 0..m_chunk.
        let mut chunk_ops = stream.alloc_zeros::<u64>(m_chunk.max(1) * stride)?;
        if m_chunk > 0 {
            stream.memcpy_dtod(
                &ops_dev.slice(start * stride..(start + m_chunk) * stride),
                &mut chunk_ops,
            )?;
        }
        let (mc_u64, nr_u64) = (m_chunk as u64, num_rows as u64);
        unsafe {
            stream.launch_builder(&be.lt_fill).arg(&chunk_ops).arg(&mc_u64).arg(&nr_u64)
                .arg(&mut out).launch(LaunchConfig::for_num_elems(num_rows as u32))?;
        }
        out_chunks.push((out, num_rows));
    }
    stream.synchronize()?;
    Ok(out_chunks)
}

/// Generic resident chain for an ALU deduped chip: cpu_op fields → `chipop_alu_route`
/// (pick `flag_index`: 0=LT 1=SHIFT 2=EQ 3=BYTEWISE 4=MUL 5=DVRM) → `key_gather` → device
/// dedup → `pack` → `fill`, entirely on device. Every deduped ALU chip is one call to this
/// with its own (key_gather, pack, fill, stride, ncols). Returns the filled `num_rows*ncols`.
#[allow(clippy::too_many_arguments)]
/// Host-upload entry for a SINGLE-multiplicity deduped ALU chip: uploads `packed/rv1/arg2`
/// once, synchronizes, then runs the device-resident core. Used by the Vec/parity APIs.
#[allow(clippy::too_many_arguments)]
fn resident_alu_dedup_chip_host(
    packed: &[u64],
    rv1: &[u64],
    arg2: &[u64],
    flag_index: usize,
    key_gather: &CudaFunction,
    pack: &CudaFunction,
    fill: &CudaFunction,
    stride: usize,
    ncols: usize,
) -> Result<(CudaSlice<u64>, usize)> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let n_u64 = n as u64;
    let pk_d = stream.clone_htod(packed)?;
    let rv1_d = stream.clone_htod(rv1)?;
    let arg2_d = stream.clone_htod(arg2)?;
    // Host path routes its own flags (no resident routes available).
    let mut f0 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f1 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f2 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f3 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f4 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f5 = stream.alloc_zeros::<u32>(n.max(1))?;
    if n > 0 {
        unsafe {
            stream
                .launch_builder(&be.chipop_alu_route)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&mut f0)
                .arg(&mut f1)
                .arg(&mut f2)
                .arg(&mut f3)
                .arg(&mut f4)
                .arg(&mut f5)
                .launch(LaunchConfig::for_num_elems(n.max(1) as u32))?;
        }
    }
    stream.synchronize()?;
    let flags6 = [f0, f1, f2, f3, f4, f5];
    resident_alu_dedup_chip_dev(
        &pk_d,
        &rv1_d,
        &arg2_d,
        &flags6[flag_index],
        n,
        key_gather,
        pack,
        fill,
        stride,
        ncols,
    )
}

/// Device-resident core for a SINGLE-multiplicity deduped ALU chip. Reads the packed/rv1/arg2
/// device buffers + a PRECOMPUTED route `flag` IN PLACE (no upload, no re-route) — the
/// resident-cpu_ops seam feeds `DeviceCpuOpsResident` buffers + `routes.alu[i]`; the host path
/// uploads + routes first via [`resident_alu_dedup_chip_host`].
#[allow(clippy::too_many_arguments)]
fn resident_alu_dedup_chip_dev(
    pk_d: &CudaSlice<u64>,
    rv1_d: &CudaSlice<u64>,
    arg2_d: &CudaSlice<u64>,
    flag: &CudaSlice<u32>,
    n: usize,
    key_gather: &CudaFunction,
    pack: &CudaFunction,
    fill: &CudaFunction,
    stride: usize,
    ncols: usize,
) -> Result<(CudaSlice<u64>, usize)> {
    let be = backend()?;
    let stream = be.next_stream();
    let n_u64 = n as u64;
    let (excl, total) = crate::trace_walk::excl_scan(be, &stream, flag, n.max(1))?;
    let rows = total as usize;

    let mut k0 = stream.alloc_zeros::<u64>(rows.max(1))?;
    let mut k1 = stream.alloc_zeros::<u64>(rows.max(1))?;
    let mut k2 = stream.alloc_zeros::<u64>(rows.max(1))?;
    if n > 0 && rows > 0 {
        unsafe {
            stream
                .launch_builder(key_gather)
                .arg(&n_u64)
                .arg(pk_d)
                .arg(rv1_d)
                .arg(arg2_d)
                .arg(flag)
                .arg(&excl)
                .arg(&mut k0)
                .arg(&mut k1)
                .arg(&mut k2)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }

    // Dedup first so the table height matches the device-computed unique count.
    let dedup = if rows > 0 {
        Some(crate::trace_walk::dedup3_core(be, &stream, &k0, &k1, &k2, rows)?)
    } else {
        None
    };
    let m = dedup.as_ref().map(|d| d.4).unwrap_or(0);
    let num_rows = m.next_power_of_two().max(4);
    let mut out = stream.alloc_zeros::<u64>(num_rows * ncols)?;
    if let Some((uk0, uk1, uk2, umult, _m)) = dedup {
        let mut ops_dev = stream.alloc_zeros::<u64>(m.max(1) * stride)?;
        let m_u64 = m as u64;
        if m > 0 {
            unsafe {
                stream
                    .launch_builder(pack)
                    .arg(&m_u64)
                    .arg(&uk0)
                    .arg(&uk1)
                    .arg(&uk2)
                    .arg(&umult)
                    .arg(&mut ops_dev)
                    .launch(LaunchConfig::for_num_elems(m as u32))?;
            }
        }
        let num_rows_u64 = num_rows as u64;
        unsafe {
            stream
                .launch_builder(fill)
                .arg(&ops_dev)
                .arg(&m_u64)
                .arg(&num_rows_u64)
                .arg(&mut out)
                .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
        }
    }
    stream.synchronize()?;
    Ok((out, num_rows))
}

/// Download a device trace buffer to host (for the Vec-returning parity APIs).
fn dtoh(buf: &CudaSlice<u64>) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let host = stream.clone_dtoh(buf)?;
    stream.synchronize()?;
    Ok(host)
}

/// RESIDENT BYTEWISE fill returning `(device buffer, auto-sized num_rows)` for p5 integration.
pub fn gpu_build_bytewise_resident_dev(
    packed: &[u64],
    rv1: &[u64],
    arg2: &[u64],
) -> Result<(CudaSlice<u64>, usize)> {
    let be = backend()?;
    resident_alu_dedup_chip_host(
        packed,
        rv1,
        arg2,
        3, // BYTEWISE flag
        &be.bytewise_key_gather,
        &be.dedup_pack_abf,
        &be.bytewise_fill,
        crate::trace_cpu::BYTEWISE_STRIDE,
        crate::trace_cpu::BYTEWISE_NCOLS,
    )
}

/// RESIDENT-SEAM BYTEWISE: reads the device-resident cpu_ops (`packed/rv1/arg2`) in place, no
/// re-upload. Byte-identical (multiset) to [`gpu_build_bytewise_resident_dev`].
pub fn gpu_build_bytewise_resident_from_devops(
    ops: &DeviceCpuOpsResident,
) -> Result<(CudaSlice<u64>, usize)> {
    let be = backend()?;
    resident_alu_dedup_chip_dev(
        &ops.packed,
        &ops.rv1,
        &ops.arg2,
        &ops.routes.alu[3], // BYTEWISE (route-once)
        ops.n,
        &be.bytewise_key_gather,
        &be.dedup_pack_abf,
        &be.bytewise_fill,
        crate::trace_cpu::BYTEWISE_STRIDE,
        crate::trace_cpu::BYTEWISE_NCOLS,
    )
}

/// RESIDENT BYTEWISE chain (key = alu_op; generic pack `[a, b, op, mult]`). `_num_rows` is
/// ignored — the resident chain auto-sizes the table from the device unique count.
pub fn gpu_build_bytewise_resident(
    packed: &[u64],
    rv1: &[u64],
    arg2: &[u64],
    _num_rows: usize,
) -> Result<Vec<u64>> {
    dtoh(&gpu_build_bytewise_resident_dev(packed, rv1, arg2)?.0)
}

/// Generic resident chain for a DUAL-multiplicity ALU deduped chip (MUL: mu_lo/mu_hi;
/// DVRM: mu_q/mu_r). Like `resident_alu_dedup_chip` but the key-gather also emits a per-op
/// selector, dedup uses `dedup3_core2`, and the pack is the stride-5 `dedup_pack_abf2`.
#[allow(clippy::too_many_arguments)]
fn resident_alu_dedup2_chip(
    packed: &[u64],
    rv1: &[u64],
    arg2: &[u64],
    flag_index: usize,
    key_gather: &CudaFunction,
    fill: &CudaFunction,
    stride: usize,
    ncols: usize,
    num_rows: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let n_u64 = n as u64;
    let pk_d = stream.clone_htod(packed)?;
    let rv1_d = stream.clone_htod(rv1)?;
    let arg2_d = stream.clone_htod(arg2)?;

    let mut f0 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f1 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f2 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f3 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f4 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f5 = stream.alloc_zeros::<u32>(n.max(1))?;
    if n > 0 {
        unsafe {
            stream
                .launch_builder(&be.chipop_alu_route)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&mut f0)
                .arg(&mut f1)
                .arg(&mut f2)
                .arg(&mut f3)
                .arg(&mut f4)
                .arg(&mut f5)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    let flag = [&f0, &f1, &f2, &f3, &f4, &f5][flag_index];
    let (excl, total) = crate::trace_walk::excl_scan(be, &stream, flag, n.max(1))?;
    let rows = total as usize;

    let mut k0 = stream.alloc_zeros::<u64>(rows.max(1))?;
    let mut k1 = stream.alloc_zeros::<u64>(rows.max(1))?;
    let mut k2 = stream.alloc_zeros::<u64>(rows.max(1))?;
    let mut sel = stream.alloc_zeros::<u32>(rows.max(1))?;
    if n > 0 && rows > 0 {
        unsafe {
            stream
                .launch_builder(key_gather)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&rv1_d)
                .arg(&arg2_d)
                .arg(flag)
                .arg(&excl)
                .arg(&mut k0)
                .arg(&mut k1)
                .arg(&mut k2)
                .arg(&mut sel)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }

    let mut out = stream.alloc_zeros::<u64>(num_rows * ncols)?;
    if rows > 0 {
        let (uk0, uk1, uk2, um0, um1, m) =
            crate::trace_walk::dedup3_core2(be, &stream, &k0, &k1, &k2, &sel, rows)?;
        let mut ops_dev = stream.alloc_zeros::<u64>(m.max(1) * stride)?;
        let m_u64 = m as u64;
        if m > 0 {
            unsafe {
                stream
                    .launch_builder(&be.dedup_pack_abf2)
                    .arg(&m_u64)
                    .arg(&uk0)
                    .arg(&uk1)
                    .arg(&uk2)
                    .arg(&um0)
                    .arg(&um1)
                    .arg(&mut ops_dev)
                    .launch(LaunchConfig::for_num_elems(m as u32))?;
            }
        }
        let num_rows_u64 = num_rows as u64;
        unsafe {
            stream
                .launch_builder(fill)
                .arg(&ops_dev)
                .arg(&m_u64)
                .arg(&num_rows_u64)
                .arg(&mut out)
                .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
        }
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// RESIDENT MUL chain (dual mult mu_lo/mu_hi; key = lhs/rhs/signed², selector = muldiv).
pub fn gpu_build_mul_resident(
    packed: &[u64],
    rv1: &[u64],
    arg2: &[u64],
    num_rows: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    resident_alu_dedup2_chip(
        packed,
        rv1,
        arg2,
        4, // MUL flag
        &be.mul_key_gather,
        &be.mul_fill,
        crate::trace_cpu::MUL_STRIDE,
        crate::trace_cpu::MUL_NCOLS,
        num_rows,
    )
}

/// RESIDENT MUL chain with a MERGED op source: instruction-driven MUL ⊕ **dvrm-derived** MUL
/// (each is_divrem cycle contributes `MulOperation::new(d, d_signed, q, q_signed)` to both
/// mu_lo and mu_hi). Both key+selector streams are gathered into one buffer (dvrm appended,
/// 2 entries/cycle) and deduped together with `dedup3_core2`. (The cpu32-derived MUL source
/// remains.) Returns the filled `num_rows * MUL_NCOLS` buffer.
pub fn gpu_build_mul_instr_dvrm_resident(
    packed: &[u64],
    rv1: &[u64],
    arg2: &[u64],
    num_rows: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let ncols = crate::trace_cpu::MUL_NCOLS;
    let n_u64 = n as u64;
    let pk_d = stream.clone_htod(packed)?;
    let rv1_d = stream.clone_htod(rv1)?;
    let arg2_d = stream.clone_htod(arg2)?;

    let mut f0 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f1 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f2 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f3 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f4 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f5 = stream.alloc_zeros::<u32>(n.max(1))?;
    if n > 0 {
        unsafe {
            stream
                .launch_builder(&be.chipop_alu_route)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&mut f0)
                .arg(&mut f1)
                .arg(&mut f2)
                .arg(&mut f3)
                .arg(&mut f4)
                .arg(&mut f5)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    // f4 = MUL, f5 = DVRM.
    let (excl_mul, total_mul) = crate::trace_walk::excl_scan(be, &stream, &f4, n.max(1))?;
    let (excl_dv, total_dv) = crate::trace_walk::excl_scan(be, &stream, &f5, n.max(1))?;
    let rows_mul = total_mul as usize;
    let rows_dv = total_dv as usize;
    let total = rows_mul + 2 * rows_dv; // dvrm contributes 2 entries/cycle

    let mut k0 = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut k1 = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut k2 = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut sel = stream.alloc_zeros::<u32>(total.max(1))?;
    if n > 0 && rows_mul > 0 {
        unsafe {
            stream
                .launch_builder(&be.mul_key_gather)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&rv1_d)
                .arg(&arg2_d)
                .arg(&f4)
                .arg(&excl_mul)
                .arg(&mut k0)
                .arg(&mut k1)
                .arg(&mut k2)
                .arg(&mut sel)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    if n > 0 && rows_dv > 0 {
        let base = rows_mul as u64;
        unsafe {
            stream
                .launch_builder(&be.mul_dvrm_key_gather)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&rv1_d)
                .arg(&arg2_d)
                .arg(&f5)
                .arg(&excl_dv)
                .arg(&base)
                .arg(&mut k0)
                .arg(&mut k1)
                .arg(&mut k2)
                .arg(&mut sel)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }

    let mut out = stream.alloc_zeros::<u64>(num_rows * ncols)?;
    if total > 0 {
        let (uk0, uk1, uk2, um0, um1, m) =
            crate::trace_walk::dedup3_core2(be, &stream, &k0, &k1, &k2, &sel, total)?;
        let mut ops_dev = stream.alloc_zeros::<u64>(m.max(1) * crate::trace_cpu::MUL_STRIDE)?;
        let m_u64 = m as u64;
        if m > 0 {
            unsafe {
                stream
                    .launch_builder(&be.dedup_pack_abf2)
                    .arg(&m_u64)
                    .arg(&uk0)
                    .arg(&uk1)
                    .arg(&uk2)
                    .arg(&um0)
                    .arg(&um1)
                    .arg(&mut ops_dev)
                    .launch(LaunchConfig::for_num_elems(m as u32))?;
            }
        }
        let num_rows_u64 = num_rows as u64;
        unsafe {
            stream
                .launch_builder(&be.mul_fill)
                .arg(&ops_dev)
                .arg(&m_u64)
                .arg(&num_rows_u64)
                .arg(&mut out)
                .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
        }
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// RESIDENT MUL chain — three sources merged: instruction ⊕ dvrm-derived (from the
/// instruction-driven dvrm cycles) ⊕ cpu32-derived. Three key+selector streams concatenated
/// into one buffer, single `dedup3_core2`. NOTE: not yet the COMPLETE production MUL — the
/// C13/C14 dvrm→mul derivation also applies to the *cpu32-derived* dvrm ops (an intertwined
/// 4th contribution), which this does not yet include. Validated against a host matching
/// exactly these three sources.
pub fn gpu_build_mul_full_resident_dev(
    packed: &[u64],
    rv1: &[u64],
    rv2: &[u64],
    arg2: &[u64],
    imm: &[u64],
) -> Result<(CudaSlice<u64>, usize)> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let n_u64 = n as u64;
    let pk_d = stream.clone_htod(packed)?;
    let rv1_d = stream.clone_htod(rv1)?;
    let rv2_d = stream.clone_htod(rv2)?;
    let arg2_d = stream.clone_htod(arg2)?;
    let imm_d = stream.clone_htod(imm)?;
    let mut f0 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f1 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f2 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f3 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f4 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f5 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f_c = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f_cd = stream.alloc_zeros::<u32>(n.max(1))?;
    let cfg = LaunchConfig::for_num_elems(n.max(1) as u32);
    if n > 0 {
        unsafe {
            stream.launch_builder(&be.chipop_alu_route).arg(&n_u64).arg(&pk_d)
                .arg(&mut f0).arg(&mut f1).arg(&mut f2).arg(&mut f3).arg(&mut f4).arg(&mut f5).launch(cfg)?;
            stream.launch_builder(&be.cpu32_mul_route).arg(&n_u64).arg(&pk_d).arg(&mut f_c).launch(cfg)?;
            stream.launch_builder(&be.cpu32_dvrm_route).arg(&n_u64).arg(&pk_d).arg(&mut f_cd).launch(cfg)?;
        }
    }
    stream.synchronize()?;
    mul_full_resident_core(&pk_d, &rv1_d, &rv2_d, &arg2_d, &imm_d, &f4, &f5, &f_c, &f_cd, n)
}

/// RESIDENT-SEAM MUL: reads the device-resident cpu_ops + precomputed routes in place (no re-upload).
pub fn gpu_build_mul_full_resident_from_devops(
    ops: &DeviceCpuOpsResident,
) -> Result<(CudaSlice<u64>, usize)> {
    mul_full_resident_core(
        &ops.packed, &ops.rv1, &ops.rv2, &ops.arg2, &ops.imm,
        &ops.routes.alu[4], &ops.routes.alu[5], &ops.routes.cpu32_mul, &ops.routes.cpu32_dvrm, ops.n,
    )
}

/// Device-resident core for MUL (four sources merged, dual-multiplicity dedup, autosized). Reads
/// precomputed route flags (route-once): `mul_flag`=alu[4], `dvrm_flag`=alu[5] (instr-dvrm→mul),
/// `cpu32_mul_flag`, `cpu32_dvrm_flag` (cpu32-dvrm→mul).
#[allow(clippy::too_many_arguments)]
fn mul_full_resident_core(
    pk_d: &CudaSlice<u64>,
    rv1_d: &CudaSlice<u64>,
    rv2_d: &CudaSlice<u64>,
    arg2_d: &CudaSlice<u64>,
    imm_d: &CudaSlice<u64>,
    mul_flag: &CudaSlice<u32>,
    dvrm_flag: &CudaSlice<u32>,
    cpu32_mul_flag: &CudaSlice<u32>,
    cpu32_dvrm_flag: &CudaSlice<u32>,
    n: usize,
) -> Result<(CudaSlice<u64>, usize)> {
    let be = backend()?;
    let stream = be.next_stream();
    let ncols = crate::trace_cpu::MUL_NCOLS;
    let n_u64 = n as u64;

    let (excl_m, tm) = crate::trace_walk::excl_scan(be, &stream, mul_flag, n.max(1))?;
    let (excl_d, td) = crate::trace_walk::excl_scan(be, &stream, dvrm_flag, n.max(1))?;
    let (excl_c, tc) = crate::trace_walk::excl_scan(be, &stream, cpu32_mul_flag, n.max(1))?;
    let (excl_cd, tcd) = crate::trace_walk::excl_scan(be, &stream, cpu32_dvrm_flag, n.max(1))?;
    let (rm, rd, rc, rcd) = (tm as usize, td as usize, tc as usize, tcd as usize);
    // Four MUL sources: instruction, instruction-dvrm→mul (2/op), cpu32, cpu32-dvrm→mul (2/op).
    let total = rm + 2 * rd + rc + 2 * rcd;

    let mut k0 = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut k1 = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut k2 = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut sel = stream.alloc_zeros::<u32>(total.max(1))?;
    if n > 0 && rm > 0 {
        unsafe {
            stream.launch_builder(&be.mul_key_gather)
                .arg(&n_u64).arg(pk_d).arg(rv1_d).arg(arg2_d).arg(mul_flag).arg(&excl_m)
                .arg(&mut k0).arg(&mut k1).arg(&mut k2).arg(&mut sel)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    if n > 0 && rd > 0 {
        let base = rm as u64;
        unsafe {
            stream.launch_builder(&be.mul_dvrm_key_gather)
                .arg(&n_u64).arg(pk_d).arg(rv1_d).arg(arg2_d).arg(dvrm_flag).arg(&excl_d).arg(&base)
                .arg(&mut k0).arg(&mut k1).arg(&mut k2).arg(&mut sel)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    if n > 0 && rc > 0 {
        let base = (rm + 2 * rd) as u64;
        unsafe {
            stream.launch_builder(&be.cpu32_mul_ops)
                .arg(&n_u64).arg(pk_d).arg(rv1_d).arg(rv2_d).arg(imm_d).arg(cpu32_mul_flag).arg(&excl_c).arg(&base)
                .arg(&mut k0).arg(&mut k1).arg(&mut k2).arg(&mut sel)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    if n > 0 && rcd > 0 {
        let base = (rm + 2 * rd + rc) as u64;
        unsafe {
            stream.launch_builder(&be.cpu32_dvrm_mul_key_gather)
                .arg(&n_u64).arg(pk_d).arg(rv1_d).arg(rv2_d).arg(imm_d).arg(cpu32_dvrm_flag).arg(&excl_cd).arg(&base)
                .arg(&mut k0).arg(&mut k1).arg(&mut k2).arg(&mut sel)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }

    let dedup = if total > 0 {
        Some(crate::trace_walk::dedup3_core2(be, &stream, &k0, &k1, &k2, &sel, total)?)
    } else {
        None
    };
    let m = dedup.as_ref().map(|d| d.5).unwrap_or(0);
    let num_rows = m.next_power_of_two().max(4);
    let mut out = stream.alloc_zeros::<u64>(num_rows * ncols)?;
    if let Some((uk0, uk1, uk2, um0, um1, _m)) = dedup {
        let mut ops_dev = stream.alloc_zeros::<u64>(m.max(1) * crate::trace_cpu::MUL_STRIDE)?;
        let m_u64 = m as u64;
        if m > 0 {
            unsafe {
                stream.launch_builder(&be.dedup_pack_abf2)
                    .arg(&m_u64).arg(&uk0).arg(&uk1).arg(&uk2).arg(&um0).arg(&um1).arg(&mut ops_dev)
                    .launch(LaunchConfig::for_num_elems(m as u32))?;
            }
        }
        let num_rows_u64 = num_rows as u64;
        unsafe {
            stream.launch_builder(&be.mul_fill)
                .arg(&ops_dev).arg(&m_u64).arg(&num_rows_u64).arg(&mut out)
                .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
        }
    }
    stream.synchronize()?;
    Ok((out, num_rows))
}

/// RESIDENT MUL full chain (Vec parity API; `_num_rows` ignored — auto-sized). Merges ALL four
/// MUL sources: instruction ⊕ instruction-dvrm→mul ⊕ cpu32 ⊕ cpu32-dvrm→mul.
pub fn gpu_build_mul_full_resident(
    packed: &[u64],
    rv1: &[u64],
    rv2: &[u64],
    arg2: &[u64],
    imm: &[u64],
    _num_rows: usize,
) -> Result<Vec<u64>> {
    dtoh(&gpu_build_mul_full_resident_dev(packed, rv1, rv2, arg2, imm)?.0)
}

/// RESIDENT DVRM chain — instruction ⊕ cpu32-derived sources merged.
pub fn gpu_build_dvrm_full_resident_dev(
    packed: &[u64],
    rv1: &[u64],
    rv2: &[u64],
    arg2: &[u64],
    imm: &[u64],
) -> Result<(CudaSlice<u64>, usize)> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let n_u64 = n as u64;
    let pk_d = stream.clone_htod(packed)?;
    let rv1_d = stream.clone_htod(rv1)?;
    let rv2_d = stream.clone_htod(rv2)?;
    let arg2_d = stream.clone_htod(arg2)?;
    let imm_d = stream.clone_htod(imm)?;
    let mut f0 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f1 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f2 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f3 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f4 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f5 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f_c = stream.alloc_zeros::<u32>(n.max(1))?;
    let cfg = LaunchConfig::for_num_elems(n.max(1) as u32);
    if n > 0 {
        unsafe {
            stream.launch_builder(&be.chipop_alu_route).arg(&n_u64).arg(&pk_d)
                .arg(&mut f0).arg(&mut f1).arg(&mut f2).arg(&mut f3).arg(&mut f4).arg(&mut f5).launch(cfg)?;
            stream.launch_builder(&be.cpu32_dvrm_route).arg(&n_u64).arg(&pk_d).arg(&mut f_c).launch(cfg)?;
        }
    }
    stream.synchronize()?;
    dvrm_full_resident_core(&pk_d, &rv1_d, &rv2_d, &arg2_d, &imm_d, &f5, &f_c, n)
}

/// RESIDENT-SEAM DVRM: reads the device-resident cpu_ops + precomputed routes in place (no re-upload).
pub fn gpu_build_dvrm_full_resident_from_devops(
    ops: &DeviceCpuOpsResident,
) -> Result<(CudaSlice<u64>, usize)> {
    dvrm_full_resident_core(
        &ops.packed, &ops.rv1, &ops.rv2, &ops.arg2, &ops.imm,
        &ops.routes.alu[5], &ops.routes.cpu32_dvrm, ops.n,
    )
}

/// Device-resident core for DVRM (instruction ⊕ cpu32 sources, dual-multiplicity, autosized).
/// Reads precomputed route flags (route-once): `dvrm_flag`=alu[5], `cpu32_dvrm_flag`.
#[allow(clippy::too_many_arguments)]
fn dvrm_full_resident_core(
    pk_d: &CudaSlice<u64>,
    rv1_d: &CudaSlice<u64>,
    rv2_d: &CudaSlice<u64>,
    arg2_d: &CudaSlice<u64>,
    imm_d: &CudaSlice<u64>,
    dvrm_flag: &CudaSlice<u32>,
    cpu32_dvrm_flag: &CudaSlice<u32>,
    n: usize,
) -> Result<(CudaSlice<u64>, usize)> {
    let be = backend()?;
    let stream = be.next_stream();
    let ncols = crate::trace_cpu::DVRM_NCOLS;
    let n_u64 = n as u64;

    let (excl_i, ti) = crate::trace_walk::excl_scan(be, &stream, dvrm_flag, n.max(1))?;
    let (excl_c, tc) = crate::trace_walk::excl_scan(be, &stream, cpu32_dvrm_flag, n.max(1))?;
    let (ri, rc) = (ti as usize, tc as usize);
    let total = ri + rc;

    let mut k0 = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut k1 = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut k2 = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut sel = stream.alloc_zeros::<u32>(total.max(1))?;
    if n > 0 && ri > 0 {
        unsafe {
            stream.launch_builder(&be.dvrm_key_gather)
                .arg(&n_u64).arg(pk_d).arg(rv1_d).arg(arg2_d).arg(dvrm_flag).arg(&excl_i)
                .arg(&mut k0).arg(&mut k1).arg(&mut k2).arg(&mut sel)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    if n > 0 && rc > 0 {
        let base = ri as u64;
        unsafe {
            stream.launch_builder(&be.cpu32_dvrm_ops)
                .arg(&n_u64).arg(pk_d).arg(rv1_d).arg(rv2_d).arg(imm_d).arg(cpu32_dvrm_flag).arg(&excl_c).arg(&base)
                .arg(&mut k0).arg(&mut k1).arg(&mut k2).arg(&mut sel)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }

    let dedup = if total > 0 {
        Some(crate::trace_walk::dedup3_core2(be, &stream, &k0, &k1, &k2, &sel, total)?)
    } else {
        None
    };
    let m = dedup.as_ref().map(|d| d.5).unwrap_or(0);
    let num_rows = m.next_power_of_two().max(4);
    let mut out = stream.alloc_zeros::<u64>(num_rows * ncols)?;
    if let Some((uk0, uk1, uk2, um0, um1, _m)) = dedup {
        let mut ops_dev = stream.alloc_zeros::<u64>(m.max(1) * crate::trace_cpu::DVRM_STRIDE)?;
        let m_u64 = m as u64;
        if m > 0 {
            unsafe {
                stream.launch_builder(&be.dedup_pack_abf2)
                    .arg(&m_u64).arg(&uk0).arg(&uk1).arg(&uk2).arg(&um0).arg(&um1).arg(&mut ops_dev)
                    .launch(LaunchConfig::for_num_elems(m as u32))?;
            }
        }
        let num_rows_u64 = num_rows as u64;
        unsafe {
            stream.launch_builder(&be.dvrm_fill)
                .arg(&ops_dev).arg(&m_u64).arg(&num_rows_u64).arg(&mut out)
                .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
        }
    }
    stream.synchronize()?;
    Ok((out, num_rows))
}

/// RESIDENT DVRM full chain (Vec parity API; `_num_rows` ignored — auto-sized).
pub fn gpu_build_dvrm_full_resident(
    packed: &[u64],
    rv1: &[u64],
    rv2: &[u64],
    arg2: &[u64],
    imm: &[u64],
    _num_rows: usize,
) -> Result<Vec<u64>> {
    dtoh(&gpu_build_dvrm_full_resident_dev(packed, rv1, rv2, arg2, imm)?.0)
}

/// RESIDENT DVRM chain (dual mult mu_q/mu_r; key = n/d/signed, selector = muldiv).
pub fn gpu_build_dvrm_resident(
    packed: &[u64],
    rv1: &[u64],
    arg2: &[u64],
    num_rows: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    resident_alu_dedup2_chip(
        packed,
        rv1,
        arg2,
        5, // DVRM flag
        &be.dvrm_key_gather,
        &be.dvrm_fill,
        crate::trace_cpu::DVRM_STRIDE,
        crate::trace_cpu::DVRM_NCOLS,
        num_rows,
    )
}

/// RESIDENT SHIFT chain (per-row — SHIFT does not dedup). Route → `build_shift_ops` → fill.
pub fn gpu_build_shift_resident(
    packed: &[u64],
    rv1: &[u64],
    arg2: &[u64],
    num_rows: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let ncols = crate::trace_cpu::SHIFT_NCOLS;
    let n_u64 = n as u64;
    let pk_d = stream.clone_htod(packed)?;
    let rv1_d = stream.clone_htod(rv1)?;
    let arg2_d = stream.clone_htod(arg2)?;
    let mut f0 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f1 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f2 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f3 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f4 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f5 = stream.alloc_zeros::<u32>(n.max(1))?;
    if n > 0 {
        unsafe {
            stream
                .launch_builder(&be.chipop_alu_route)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&mut f0)
                .arg(&mut f1)
                .arg(&mut f2)
                .arg(&mut f3)
                .arg(&mut f4)
                .arg(&mut f5)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    // f1 = SHIFT flag.
    let (excl, total) = crate::trace_walk::excl_scan(be, &stream, &f1, n.max(1))?;
    let rows = total as usize;
    let mut ops_dev = stream.alloc_zeros::<u64>(rows.max(1) * crate::trace_cpu::SHIFT_STRIDE)?;
    if n > 0 && rows > 0 {
        unsafe {
            stream
                .launch_builder(&be.build_shift_ops)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&rv1_d)
                .arg(&arg2_d)
                .arg(&f1)
                .arg(&excl)
                .arg(&mut ops_dev)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    let mut out = stream.alloc_zeros::<u64>(num_rows * ncols)?;
    let (rows_u64, num_rows_u64) = (rows as u64, num_rows as u64);
    unsafe {
        stream
            .launch_builder(&be.shift_fill)
            .arg(&ops_dev)
            .arg(&rows_u64)
            .arg(&num_rows_u64)
            .arg(&mut out)
            .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// RESIDENT SHIFT chain with a MERGED source: instruction-driven SHIFT (word=0) ⊕
/// **cpu32-derived** SHIFT (word instructions dispatched to the SHIFT chip via
/// `cpu32_chip_op`, word=1). Both are per-row; the cpu32 rows are appended after the
/// instruction rows and filled together. Returns the filled `num_rows * SHIFT_NCOLS` buffer.
/// Device-resident core for SHIFT (instruction SHIFT ∪ cpu32-derived SHIFT, per-row). Reads
/// the packed/rv1/rv2/arg2/imm device buffers IN PLACE. `num_rows_hint = None` autosizes the
/// table from the device row count (the resident-seam path); `Some(n)` pins the padded height
/// (the host parity path). Returns `(device buffer, num_rows)`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn shift_full_resident_core(
    pk_d: &CudaSlice<u64>,
    rv1_d: &CudaSlice<u64>,
    rv2_d: &CudaSlice<u64>,
    arg2_d: &CudaSlice<u64>,
    imm_d: &CudaSlice<u64>,
    shift_flag: &CudaSlice<u32>,       // route-once: alu[1] (instruction SHIFT, word=0)
    cpu32_shift_flag: &CudaSlice<u32>, // route-once: cpu32-derived SHIFT (word=1)
    n: usize,
    num_rows_hint: Option<usize>,
) -> Result<(CudaSlice<u64>, usize)> {
    let be = backend()?;
    let stream = be.next_stream();
    let ncols = crate::trace_cpu::SHIFT_NCOLS;
    let n_u64 = n as u64;

    let (excl_i, total_i) = crate::trace_walk::excl_scan(be, &stream, shift_flag, n.max(1))?;
    let (excl_c, total_c) = crate::trace_walk::excl_scan(be, &stream, cpu32_shift_flag, n.max(1))?;
    let rows_i = total_i as usize;
    let rows_c = total_c as usize;
    let total = rows_i + rows_c;
    let num_rows = num_rows_hint.unwrap_or_else(|| total.next_power_of_two().max(4));

    let mut ops_dev = stream.alloc_zeros::<u64>(total.max(1) * crate::trace_cpu::SHIFT_STRIDE)?;
    if n > 0 && rows_i > 0 {
        unsafe {
            stream
                .launch_builder(&be.build_shift_ops)
                .arg(&n_u64)
                .arg(pk_d)
                .arg(rv1_d)
                .arg(arg2_d)
                .arg(shift_flag)
                .arg(&excl_i)
                .arg(&mut ops_dev)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    if n > 0 && rows_c > 0 {
        let base = rows_i as u64;
        unsafe {
            stream
                .launch_builder(&be.cpu32_shift_ops)
                .arg(&n_u64)
                .arg(pk_d)
                .arg(rv1_d)
                .arg(rv2_d)
                .arg(imm_d)
                .arg(cpu32_shift_flag)
                .arg(&excl_c)
                .arg(&base)
                .arg(&mut ops_dev)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    let mut out = stream.alloc_zeros::<u64>(num_rows * ncols)?;
    let (total_u64, num_rows_u64) = (total as u64, num_rows as u64);
    unsafe {
        stream
            .launch_builder(&be.shift_fill)
            .arg(&ops_dev)
            .arg(&total_u64)
            .arg(&num_rows_u64)
            .arg(&mut out)
            .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
    }
    stream.synchronize()?;
    Ok((out, num_rows))
}

pub fn gpu_build_shift_full_resident_dev(
    packed: &[u64],
    rv1: &[u64],
    rv2: &[u64],
    arg2: &[u64],
    imm: &[u64],
    num_rows: usize,
) -> Result<CudaSlice<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let n_u64 = n as u64;
    let pk_d = stream.clone_htod(packed)?;
    let rv1_d = stream.clone_htod(rv1)?;
    let rv2_d = stream.clone_htod(rv2)?;
    let arg2_d = stream.clone_htod(arg2)?;
    let imm_d = stream.clone_htod(imm)?;
    let mut f0 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f1 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f2 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f3 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f4 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f5 = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut cpu32_shift = stream.alloc_zeros::<u32>(n.max(1))?;
    let cfg = LaunchConfig::for_num_elems(n.max(1) as u32);
    if n > 0 {
        unsafe {
            stream.launch_builder(&be.chipop_alu_route).arg(&n_u64).arg(&pk_d)
                .arg(&mut f0).arg(&mut f1).arg(&mut f2).arg(&mut f3).arg(&mut f4).arg(&mut f5).launch(cfg)?;
            stream.launch_builder(&be.cpu32_shift_route).arg(&n_u64).arg(&pk_d).arg(&mut cpu32_shift).launch(cfg)?;
        }
    }
    stream.synchronize()?;
    let flags6 = [f0, f1, f2, f3, f4, f5];
    Ok(shift_full_resident_core(
        &pk_d, &rv1_d, &rv2_d, &arg2_d, &imm_d, &flags6[1], &cpu32_shift, n, Some(num_rows),
    )?
    .0)
}

/// RESIDENT-SEAM SHIFT: reads the device-resident cpu_ops + precomputed routes in place, autosizing.
pub fn gpu_build_shift_full_resident_from_devops(
    ops: &DeviceCpuOpsResident,
) -> Result<(CudaSlice<u64>, usize)> {
    shift_full_resident_core(
        &ops.packed, &ops.rv1, &ops.rv2, &ops.arg2, &ops.imm,
        &ops.routes.alu[1], &ops.routes.cpu32_shift, ops.n, None,
    )
}

/// RESIDENT SHIFT chain (Vec parity API; downloads the device buffer).
pub fn gpu_build_shift_full_resident(
    packed: &[u64],
    rv1: &[u64],
    rv2: &[u64],
    arg2: &[u64],
    imm: &[u64],
    num_rows: usize,
) -> Result<Vec<u64>> {
    dtoh(&gpu_build_shift_full_resident_dev(packed, rv1, rv2, arg2, imm, num_rows)?)
}

/// RESIDENT BRANCH chain (dedup4: key = pc/offset/register/jalr). Route via
/// `chipop_branch_store_route` (branch_cond) → `branch_key_gather` → `dedup4_core` →
/// `branch_pack` → `branch_fill`. `flags` = per-cycle build_cpu_ops flags byte (bit0 = branch_cond).
pub fn gpu_build_branch_resident_dev(
    packed: &[u64],
    flags: &[u8],
    pc: &[u64],
    imm: &[u64],
    rv1: &[u64],
) -> Result<(CudaSlice<u64>, usize)> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let n_u64 = n as u64;
    let pk_d = stream.clone_htod(packed)?;
    let fl_d = stream.clone_htod(flags)?;
    let pc_d = stream.clone_htod(pc)?;
    let imm_d = stream.clone_htod(imm)?;
    let rv1_d = stream.clone_htod(rv1)?;
    let mut f_branch = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f_store = stream.alloc_zeros::<u32>(n.max(1))?;
    if n > 0 {
        unsafe {
            stream
                .launch_builder(&be.chipop_branch_store_route)
                .arg(&n_u64).arg(&pk_d).arg(&fl_d).arg(&mut f_branch).arg(&mut f_store)
                .launch(LaunchConfig::for_num_elems(n.max(1) as u32))?;
        }
    }
    stream.synchronize()?;
    branch_resident_core(&pk_d, &pc_d, &imm_d, &rv1_d, &f_branch, n)
}

/// RESIDENT-SEAM BRANCH: reads the device-resident cpu_ops + precomputed route in place.
pub fn gpu_build_branch_resident_from_devops(
    ops: &DeviceCpuOpsResident,
) -> Result<(CudaSlice<u64>, usize)> {
    branch_resident_core(&ops.packed, &ops.pc, &ops.imm, &ops.rv1, &ops.routes.branch, ops.n)
}

/// Device-resident core for BRANCH (4-key dedup: pc/offset/register/jalr, autosized). Reads the
/// precomputed `branch_flag` (route-once: `chipop_branch_store_route` f_branch = branch_cond).
fn branch_resident_core(
    pk_d: &CudaSlice<u64>,
    pc_d: &CudaSlice<u64>,
    imm_d: &CudaSlice<u64>,
    rv1_d: &CudaSlice<u64>,
    branch_flag: &CudaSlice<u32>,
    n: usize,
) -> Result<(CudaSlice<u64>, usize)> {
    let be = backend()?;
    let stream = be.next_stream();
    let ncols = crate::trace_cpu::BRANCH_NCOLS;
    let n_u64 = n as u64;

    let (excl, total) = crate::trace_walk::excl_scan(be, &stream, branch_flag, n.max(1))?;
    let rows = total as usize;

    let mut k0 = stream.alloc_zeros::<u64>(rows.max(1))?;
    let mut k1 = stream.alloc_zeros::<u64>(rows.max(1))?;
    let mut k2 = stream.alloc_zeros::<u64>(rows.max(1))?;
    let mut k3 = stream.alloc_zeros::<u64>(rows.max(1))?;
    if n > 0 && rows > 0 {
        unsafe {
            stream
                .launch_builder(&be.branch_key_gather)
                .arg(&n_u64)
                .arg(pk_d)
                .arg(pc_d)
                .arg(imm_d)
                .arg(rv1_d)
                .arg(branch_flag)
                .arg(&excl)
                .arg(&mut k0)
                .arg(&mut k1)
                .arg(&mut k2)
                .arg(&mut k3)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }

    let dedup = if rows > 0 {
        Some(crate::trace_walk::dedup4_core(be, &stream, &k0, &k1, &k2, &k3, rows)?)
    } else {
        None
    };
    let m = dedup.as_ref().map(|d| d.5).unwrap_or(0);
    let num_rows = m.next_power_of_two().max(4);
    let mut out = stream.alloc_zeros::<u64>(num_rows * ncols)?;
    if let Some((uk0, uk1, uk2, uk3, umult, _m)) = dedup {
        let mut ops_dev = stream.alloc_zeros::<u64>(m.max(1) * crate::trace_cpu::BRANCH_STRIDE)?;
        let m_u64 = m as u64;
        if m > 0 {
            unsafe {
                stream
                    .launch_builder(&be.branch_pack)
                    .arg(&m_u64)
                    .arg(&uk0)
                    .arg(&uk1)
                    .arg(&uk2)
                    .arg(&uk3)
                    .arg(&umult)
                    .arg(&mut ops_dev)
                    .launch(LaunchConfig::for_num_elems(m as u32))?;
            }
        }
        let num_rows_u64 = num_rows as u64;
        unsafe {
            stream
                .launch_builder(&be.branch_fill)
                .arg(&ops_dev)
                .arg(&m_u64)
                .arg(&num_rows_u64)
                .arg(&mut out)
                .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
        }
    }
    stream.synchronize()?;
    Ok((out, num_rows))
}

/// RESIDENT BRANCH chain (Vec parity API; `_num_rows` ignored — auto-sized).
pub fn gpu_build_branch_resident(
    packed: &[u64],
    flags: &[u8],
    pc: &[u64],
    imm: &[u64],
    rv1: &[u64],
    _num_rows: usize,
) -> Result<Vec<u64>> {
    dtoh(&gpu_build_branch_resident_dev(packed, flags, pc, imm, rv1)?.0)
}

/// RESIDENT EQ fill returning `(device buffer, auto-sized num_rows)` for p5 integration. Same
/// chain as `gpu_build_eq_resident`, via the generic deduped helper (flag 2 = EQ).
pub fn gpu_build_eq_resident_dev(
    packed: &[u64],
    rv1: &[u64],
    arg2: &[u64],
) -> Result<(CudaSlice<u64>, usize)> {
    let be = backend()?;
    resident_alu_dedup_chip_host(
        packed,
        rv1,
        arg2,
        2, // EQ flag
        &be.eq_key_gather,
        &be.dedup_pack_abf,
        &be.eq_fill,
        crate::trace_cpu::EQ_STRIDE,
        crate::trace_cpu::EQ_NCOLS,
    )
}

/// RESIDENT-SEAM EQ: reads the device-resident cpu_ops (`packed/rv1/arg2`) in place, no
/// re-upload. Byte-identical (multiset) to [`gpu_build_eq_resident_dev`].
pub fn gpu_build_eq_resident_from_devops(
    ops: &DeviceCpuOpsResident,
) -> Result<(CudaSlice<u64>, usize)> {
    let be = backend()?;
    resident_alu_dedup_chip_dev(
        &ops.packed,
        &ops.rv1,
        &ops.arg2,
        &ops.routes.alu[2], // EQ (route-once)
        ops.n,
        &be.eq_key_gather,
        &be.dedup_pack_abf,
        &be.eq_fill,
        crate::trace_cpu::EQ_STRIDE,
        crate::trace_cpu::EQ_NCOLS,
    )
}

/// RESIDENT EQ chain — same deduped template as LT, but EQ's dedup key is `invert` only
/// (`eq_key_gather`) and its pack is the generic `[a, b, flags, mult]` (`dedup_pack_abf`).
/// Multiset-identical to the host path. `num_rows` = padded height.
pub fn gpu_build_eq_resident(
    packed: &[u64],
    rv1: &[u64],
    arg2: &[u64],
    num_rows: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let ncols = crate::trace_cpu::EQ_NCOLS;
    let n_u64 = n as u64;
    let pk_d = stream.clone_htod(packed)?;
    let rv1_d = stream.clone_htod(rv1)?;
    let arg2_d = stream.clone_htod(arg2)?;

    let mut f_lt = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f_shift = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f_eq = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f_byte = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f_mul = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut f_dvrm = stream.alloc_zeros::<u32>(n.max(1))?;
    if n > 0 {
        unsafe {
            stream
                .launch_builder(&be.chipop_alu_route)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&mut f_lt)
                .arg(&mut f_shift)
                .arg(&mut f_eq)
                .arg(&mut f_byte)
                .arg(&mut f_mul)
                .arg(&mut f_dvrm)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    let (excl, total) = crate::trace_walk::excl_scan(be, &stream, &f_eq, n.max(1))?;
    let rows = total as usize;

    let mut k0 = stream.alloc_zeros::<u64>(rows.max(1))?;
    let mut k1 = stream.alloc_zeros::<u64>(rows.max(1))?;
    let mut k2 = stream.alloc_zeros::<u64>(rows.max(1))?;
    if n > 0 && rows > 0 {
        unsafe {
            stream
                .launch_builder(&be.eq_key_gather)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&rv1_d)
                .arg(&arg2_d)
                .arg(&f_eq)
                .arg(&excl)
                .arg(&mut k0)
                .arg(&mut k1)
                .arg(&mut k2)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }

    let mut out = stream.alloc_zeros::<u64>(num_rows * ncols)?;
    if rows > 0 {
        let (uk0, uk1, uk2, umult, m) =
            crate::trace_walk::dedup3_core(be, &stream, &k0, &k1, &k2, rows)?;
        let mut ops_dev = stream.alloc_zeros::<u64>(m.max(1) * crate::trace_cpu::EQ_STRIDE)?;
        let m_u64 = m as u64;
        if m > 0 {
            unsafe {
                stream
                    .launch_builder(&be.dedup_pack_abf)
                    .arg(&m_u64)
                    .arg(&uk0)
                    .arg(&uk1)
                    .arg(&uk2)
                    .arg(&umult)
                    .arg(&mut ops_dev)
                    .launch(LaunchConfig::for_num_elems(m as u32))?;
            }
        }
        let num_rows_u64 = num_rows as u64;
        unsafe {
            stream
                .launch_builder(&be.eq_fill)
                .arg(&ops_dev)
                .arg(&m_u64)
                .arg(&num_rows_u64)
                .arg(&mut out)
                .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
        }
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// RESIDENT LOAD chain (device op-build → device fill, no intermediate host round-trip).
/// Byte-identical to the host path. `num_rows` = padded table height. Returns the filled
/// `num_rows * LOAD_NCOLS` buffer.
pub fn gpu_build_load_resident_dev(
    packed: &[u64],
    res: &[u64],
    rvd: &[u64],
    num_rows: usize,
) -> Result<CudaSlice<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let ncols = crate::trace_cpu::LOAD_NCOLS;
    let n_u64 = n as u64;
    let pk_d = stream.clone_htod(packed)?;
    let res_d = stream.clone_htod(res)?;
    let rvd_d = stream.clone_htod(rvd)?;
    let mut flag = stream.alloc_zeros::<u32>(n.max(1))?;
    if n > 0 {
        unsafe {
            stream
                .launch_builder(&be.load_route)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&mut flag)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    let (excl, total) = crate::trace_walk::excl_scan(be, &stream, &flag, n.max(1))?;
    let rows = total as usize;
    let mut ops_dev = stream.alloc_zeros::<u64>(rows.max(1) * 7)?;
    if n > 0 && rows > 0 {
        unsafe {
            stream
                .launch_builder(&be.build_load_ops)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&res_d)
                .arg(&rvd_d)
                .arg(&flag)
                .arg(&excl)
                .arg(&mut ops_dev)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    let mut out = stream.alloc_zeros::<u64>(num_rows * ncols)?;
    let (rows_u64, num_rows_u64) = (rows as u64, num_rows as u64);
    unsafe {
        stream
            .launch_builder(&be.load_fill)
            .arg(&ops_dev)
            .arg(&rows_u64)
            .arg(&num_rows_u64)
            .arg(&mut out)
            .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
    }
    stream.synchronize()?;
    Ok(out)
}

/// RESIDENT LOAD chain (Vec parity API; downloads the device buffer).
pub fn gpu_build_load_resident(
    packed: &[u64],
    res: &[u64],
    rvd: &[u64],
    num_rows: usize,
) -> Result<Vec<u64>> {
    dtoh(&gpu_build_load_resident_dev(packed, res, rvd, num_rows)?)
}

/// RESIDENT STORE chain (device op-build → device fill, no intermediate host round-trip).
/// Byte-identical to the host path. `num_rows` = padded table height. Returns the filled
/// `num_rows * STORE_NCOLS` buffer.
pub fn gpu_build_store_resident_dev(
    packed: &[u64],
    res: &[u64],
    rv2: &[u64],
    num_rows: usize,
) -> Result<CudaSlice<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let ncols = crate::trace_cpu::STORE_NCOLS;
    let n_u64 = n as u64;
    let pk_d = stream.clone_htod(packed)?;
    let res_d = stream.clone_htod(res)?;
    let rv2_d = stream.clone_htod(rv2)?;
    let mut flag = stream.alloc_zeros::<u32>(n.max(1))?;
    if n > 0 {
        unsafe {
            stream
                .launch_builder(&be.store_route)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&mut flag)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    let (excl, total) = crate::trace_walk::excl_scan(be, &stream, &flag, n.max(1))?;
    let rows = total as usize;
    let mut ops_dev = stream.alloc_zeros::<u64>(rows.max(1) * 4)?;
    if n > 0 && rows > 0 {
        unsafe {
            stream
                .launch_builder(&be.build_store_ops)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&res_d)
                .arg(&rv2_d)
                .arg(&flag)
                .arg(&excl)
                .arg(&mut ops_dev)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    let mut out = stream.alloc_zeros::<u64>(num_rows * ncols)?;
    let (rows_u64, num_rows_u64) = (rows as u64, num_rows as u64);
    unsafe {
        stream
            .launch_builder(&be.store_fill)
            .arg(&ops_dev)
            .arg(&rows_u64)
            .arg(&num_rows_u64)
            .arg(&mut out)
            .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
    }
    stream.synchronize()?;
    Ok(out)
}

/// RESIDENT STORE chain (Vec parity API; downloads the device buffer).
pub fn gpu_build_store_resident(
    packed: &[u64],
    res: &[u64],
    rv2: &[u64],
    num_rows: usize,
) -> Result<Vec<u64>> {
    dtoh(&gpu_build_store_resident_dev(packed, res, rv2, num_rows)?)
}

/// Build the LOAD chip-op rows on device — the packed `pack_load_op` SoA (7 u64/op) per
/// is_load cycle, compacted in program order, with the sign/zero-extended `res_bytes`
/// computed on device. Reproduces `collect_load_op_from_cpu`'s `LoadOperation` (the LOAD
/// chip table only; the MEMW read row's old_ts is the Phase-2 walk, separate). Inputs are
/// resident cpu_op fields: `packed` decode, `res` (= effective address), `rvd` (loaded
/// value). Returns the flat `rows * 7` buffer (feeds `load_fill`) + row count.
pub fn gpu_build_load_ops(packed: &[u64], res: &[u64], rvd: &[u64]) -> Result<(Vec<u64>, usize)> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    if n == 0 {
        return Ok((vec![], 0));
    }
    debug_assert_eq!(res.len(), n);
    debug_assert_eq!(rvd.len(), n);

    let pk_d = stream.clone_htod(packed)?;
    let res_d = stream.clone_htod(res)?;
    let rvd_d = stream.clone_htod(rvd)?;

    let mut flag = stream.alloc_zeros::<u32>(n)?;
    let n_u64 = n as u64;
    unsafe {
        stream
            .launch_builder(&be.load_route)
            .arg(&n_u64)
            .arg(&pk_d)
            .arg(&mut flag)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
    let (excl, total) = crate::trace_walk::excl_scan(be, &stream, &flag, n)?;
    let rows = total as usize;
    if rows == 0 {
        return Ok((vec![], 0));
    }
    let mut out = stream.alloc_zeros::<u64>(rows * 7)?;
    unsafe {
        stream
            .launch_builder(&be.build_load_ops)
            .arg(&n_u64)
            .arg(&pk_d)
            .arg(&res_d)
            .arg(&rvd_d)
            .arg(&flag)
            .arg(&excl)
            .arg(&mut out)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok((host, rows))
}
