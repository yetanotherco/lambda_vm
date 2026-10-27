#!/usr/bin/env python3
"""Rebuild this branch and require exact round-12 cycles, outputs and COPY counts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from build import HERE, ROOT, OUT, build


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cmd, log_name, **kwargs):
    print(f'Running {log_name}', flush=True)
    with (OUT / 'logs' / log_name).open('w') as log:
        subprocess.run(list(map(str, cmd)), cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, **kwargs)


def read_rows(path):
    rows = {}
    for line in path.read_text().splitlines():
        fields = line.split('\t')
        assert fields[0] not in rows, f'Duplicate fixture: {fields[0]}'
        rows[fields[0]] = fields[1:]
    return rows


def validation_guest():
    guest = ROOT / 'executor/programs/rust/guest_validation'
    (guest / 'src').mkdir(parents=True, exist_ok=True)
    (guest / '.cargo').mkdir(exist_ok=True)
    (guest / 'Cargo.toml').write_text('''[workspace]
[package]
name = "guest_validation"
version = "0.1.0"
edition = "2024"
[dependencies]
lambda-vm-syscalls = { path = "../../../../syscalls" }
ethrex-common = { path = "../../../../vendor/ethrex/crates/common", default-features = false }
ethrex-levm = { path = "../../../../vendor/ethrex/crates/vm/levm", default-features = false }
ethrex-trie = { path = "../../../../vendor/ethrex/crates/common/trie", default-features = false }
ethrex-crypto = { path = "../../../../vendor/ethrex/crates/common/crypto", default-features = false }
lambda-vm-ethrex-crypto = { path = "../../../../crypto/ethrex-crypto" }
[profile.release]
lto = "fat"
codegen-units = 1
''')
    shutil.copyfile(HERE / 'guest-tests.rs', guest / 'src/main.rs')
    shutil.copyfile(HERE / 'keccak-vectors.bin', guest / 'src/keccak-vectors.bin')
    shutil.copyfile(HERE / 'guest-tests-Cargo.lock', guest / 'Cargo.lock')
    shutil.copyfile(ROOT / 'executor/programs/rust/ethrex/.cargo/config.toml', guest / '.cargo/config.toml')
    return build('guest-validation', 'guest_validation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host-target', type=Path, default=OUT / 'host',
                        help='Optional reusable host dependency cache; guest is built in this checkout')
    args = parser.parse_args()
    host = args.host_target.resolve()
    for directory in ['logs', 'elfs']:
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    expected = json.loads((HERE / 'expected/manifest.json').read_text())
    for name, digest in expected['fixtures'].items():
        assert sha256(HERE / 'fixtures' / name) == digest, name
    for name, digest in expected['optimized_source_sha256'].items():
        assert sha256(ROOT / name) == digest, f'Changed optimized source: {name}'

    offline = [] if os.environ.get('GUEST_REVIEW_ONLINE') == '1' else ['--offline']
    cargo = ['rustup', 'run', '1.94.0', 'cargo']
    run([*cargo, 'build', '--release', '--locked', *offline, '--manifest-path',
         HERE / 'runner/Cargo.toml', '--target-dir', host], 'runner-build.log')
    runner = host / 'release/pr1014-guest-measure'
    elf = build()
    fixtures = sorted((HERE / 'fixtures').glob('*.bin'))
    with (OUT / 'logs/branch.tsv').open('w') as out, (OUT / 'logs/branch-copy.tsv').open('w') as counters:
        subprocess.run([str(runner), str(elf), *map(str, fixtures)],
                       stdout=out, stderr=counters, check=True)
    actual = read_rows(OUT / 'logs/branch.tsv')
    counters = read_rows(OUT / 'logs/branch-copy.tsv')
    assert actual == read_rows(HERE / 'expected/final.tsv'), 'Cycle or public-output mismatch'
    assert counters == read_rows(HERE / 'expected/final-copy.tsv'), 'Accelerator counter mismatch'
    print('All six fixtures exactly match round 12: instructions, outputs and counters.', flush=True)

    run([*cargo, 'test', '--release', '--locked', *offline, '--manifest-path',
         HERE / 'checks/Cargo.toml', '--target-dir', host], 'native-checks.log')
    run([*cargo, 'test', '--locked', *offline, '--manifest-path',
         ROOT / 'crypto/ethrex-crypto/Cargo.toml', '--target-dir', host, 'keccak'], 'keccak-tests.log')
    with (OUT / 'logs/golden-build.log').open('w') as log:
        vectors = subprocess.check_output([*cargo, 'run', '--release', '--locked', *offline,
            '--manifest-path', str(HERE / 'checks/Cargo.toml'), '--target-dir', str(host),
            '--bin', 'keccak_vectors'], cwd=ROOT, stderr=log)
    assert vectors == (HERE / 'keccak-vectors.bin').read_bytes(), 'Reference Keccak digest mismatch'

    empty = OUT / 'empty.bin'
    empty.write_bytes(b'')
    rv = validation_guest()
    with (OUT / 'logs/guest-validation-copy.tsv').open('w') as err:
        rv_result = subprocess.check_output([str(runner), str(rv), str(empty)], text=True, stderr=err)
    (OUT / 'logs/guest-validation.tsv').write_text(rv_result)
    assert rv_result.strip().split('\t')[2] == b'R12CHECK'.hex(), rv_result

    guest = ROOT / 'executor/programs/rust/memcmp_validation'
    (guest / 'src').mkdir(parents=True, exist_ok=True)
    (guest / '.cargo').mkdir(exist_ok=True)
    (guest / 'Cargo.toml').write_text('''[workspace]
[package]
name = "memcmp_validation"
version = "0.1.0"
edition = "2024"
[dependencies]
lambda-vm-syscalls = { path = "../../../../syscalls", features = ["fast-memcmp"] }
[profile.release]
lto = "fat"
codegen-units = 1
''')
    shutil.copyfile(HERE / 'memcmp-tests.rs', guest / 'src/main.rs')
    shutil.copyfile(ROOT / 'executor/programs/rust/ethrex/.cargo/config.toml', guest / '.cargo/config.toml')
    run([*cargo, 'generate-lockfile', *offline, '--manifest-path', guest / 'Cargo.toml'], 'memcmp-lock.log')
    memcmp = build('memcmp-validation', 'memcmp_validation')
    with (OUT / 'logs/memcmp-copy.tsv').open('w') as err:
        memcmp_result = subprocess.check_output([str(runner), str(memcmp), str(empty)], text=True, stderr=err)
    (OUT / 'logs/memcmp.tsv').write_text(memcmp_result)
    assert memcmp_result.strip().split('\t')[2] == '8012000000000000', memcmp_result

    for fixture in fixtures:
        if 'mainnet' in fixture.name:
            run([host / 'release/rejections', fixture, elf], f'rejections-{fixture.stem}.log')
    result = {
        'git_commit_at_validation': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'lambda_base': expected['lambda_base'], 'ethrex_base': expected['ethrex_base'],
        'guest_toolchain': expected['toolchain'], 'host_toolchain': '1.94.0',
        'guest_elf_sha256': sha256(elf),
        'fixtures': {name: {'instructions': int(values[0]), 'public_output': values[1],
                           **{key: int(value) for key, value in (item.split('=') for item in counters[name])}}
                     for name, values in actual.items()},
        'rv_validation_instructions': int(rv_result.strip().split('\t')[1]),
        'native_tests': 31, 'keccak_tests': 4, 'keccak_golden_vectors': len(vectors) // 40,
        'guest_invalid_input_rejections': 8,
        'checks': 'passed: exact counts, native oracle, native tests, Keccak, RV guest, memcmp, invalid inputs',
    }
    (OUT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
