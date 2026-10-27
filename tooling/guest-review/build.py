#!/usr/bin/env python3
"""Build a review guest directly from this checkout, using the measured flags."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'out'


def build(name='branch', guest='ethrex'):
    for directory in ['logs', 'elfs']:
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    default_llvm = '/opt/homebrew/opt/llvm/bin'
    if not Path(default_llvm).exists():
        default_llvm = str(Path(shutil.which('clang') or '/usr/bin/clang').parent)
    llvm = Path(os.environ.get('LLVM_BIN', default_llvm))
    default_sysroot = Path.home() / '.lambda-vm-sysroot'
    if not default_sysroot.exists():
        default_sysroot = Path('/opt/lambda-vm-sysroot')
    sysroot = os.environ.get('SYSROOT_DIR', str(default_sysroot))
    target = ROOT / 'guest-target'
    env = dict(os.environ, CARGO_TARGET_DIR=str(target), CARGO_BUILD_JOBS='4',
        AR=str(llvm / 'llvm-ar'), AR_riscv64im_lambda_vm_elf=str(llvm / 'llvm-ar'),
        CC_riscv64im_lambda_vm_elf=str(llvm / 'clang'),
        CFLAGS_riscv64im_lambda_vm_elf=f'--target=riscv64 -march=rv64im -mabi=lp64 --sysroot={sysroot}')
    offline = [] if os.environ.get('GUEST_REVIEW_ONLINE') == '1' else ['--offline']
    with (OUT / 'logs' / f'{name}-build.log').open('w') as log:
        subprocess.run(['rustup', 'run', 'nightly-2026-02-01', 'cargo', 'build', *offline,
            '--release', '--locked', '--target', str(ROOT / 'executor/programs/riscv64im-lambda-vm-elf.json'),
            '-Z', 'build-std=core,alloc,std,compiler_builtins,panic_abort',
            '-Z', 'build-std-features=compiler-builtins-mem', '-Z', 'json-target-spec', '--bin', guest],
            cwd=ROOT / 'executor/programs/rust' / guest, env=env,
            stdout=log, stderr=subprocess.STDOUT, check=True)
    elf = OUT / 'elfs' / f'{name}.elf'
    shutil.copyfile(target / 'riscv64im-lambda-vm-elf/release' / guest, elf)
    print(f'Built {name}: {elf}', flush=True)
    return elf


if __name__ == '__main__':
    build(*sys.argv[1:])
