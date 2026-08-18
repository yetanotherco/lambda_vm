"""
LAYER 2 VECTORS: emit and self-check the socket KATs, then cross-check against
the independently-produced table in `thoughts/blake3/socket-kats/socket_kats.json`.

What this establishes:
  * the two routes (byte level via the tree hasher, word level via one
    compression) agree on every vector at BOTH round counts;
  * every framing degree of freedom is DISCRIMINATED by at least one vector --
    i.e. the table can actually catch a chip that gets that choice wrong;
  * at rounds = 7 the socket equals standard BLAKE3 of the 36-byte message,
    truncated, which is the external cross-check a build phase can re-run as a
    one-line `blake3::hash` assertion;
  * my digests equal the parallel agent's, computed from two separately written
    implementations of both the primitive and the framing.

Run: python3 socket_kats.py [--write]
"""

from __future__ import annotations

import json
import os
import sys

import blake3_oracle as ora
import socket_ref as sk

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "socket_kats.json")

_PEER_CANDIDATES = [
    "/Users/maurofab/workspace/lambda_vm-blake3-impl/thoughts/blake3/socket-kats/socket_kats.json",
    os.path.join(HERE, "..", "..", "..", "blake3", "socket-kats", "socket_kats.json"),
]


def vectors() -> list[tuple[str, list[int], list[int]]]:
    """Fixed, written-out inputs -- nothing depends on an RNG.

    The structural vectors are deliberately degenerate (they are the inputs a
    buggy chip is most likely to be tested on); the formula vectors exist
    BECAUSE the degenerate ones cannot detect a byte-order or a swap error.
    """
    v: list[tuple[str, list[int], list[int]]] = [
        ("zeros",       [0, 0, 0, 0], [0, 0, 0, 0]),
        ("unit_a",      [1, 0, 0, 0], [0, 0, 0, 0]),
        ("unit_b",      [0, 0, 0, 0], [1, 0, 0, 0]),
        ("all_ones",    [0xFFFFFFFF] * 4, [0xFFFFFFFF] * 4),
        ("nibble_ramp", [0x00000000, 0x11111111, 0x22222222, 0x33333333],
                        [0x44444444, 0x55555555, 0x66666666, 0x77777777]),
        ("max_min",     [0xFFFFFFFF, 0, 0xFFFFFFFF, 0], [0, 0xFFFFFFFF, 0, 0xFFFFFFFF]),
        # Formula vectors: asymmetric, byte-distinct, a != b.
        ("formula_1",   [0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10],
                        [0x11121314, 0x15161718, 0x191A1B1C, 0x1D1E1F20]),
        ("formula_2",   [0xDEADBEEF, 0xCAFEBABE, 0x8BADF00D, 0xFEEDFACE],
                        [0x0BADC0DE, 0xD15EA5E5, 0xC0FFEE00, 0xBAAAAAAD]),
        ("formula_3",   [0x7F800001, 0x00000002, 0x80000000, 0x7FFFFFFF],
                        [0x00FF00FF, 0xFF00FF00, 0x0F0F0F0F, 0xF0F0F0F0]),
        ("boundary",    [0, 1, 0xFFFFFFFE, 0xFFFFFFFF],
                        [0x80000000, 0x7FFFFFFF, 0x00010000, 0x0000FFFF]),
    ]
    return v


def build() -> dict:
    table: dict[str, list] = {"6": [], "7": []}
    discriminated: dict[str, set[str]] = {name: set() for name in sk.CONTROLS}
    problems: list[str] = []

    for rounds in (7, 6):
        fr = sk.honest(rounds)
        for name, a, b in vectors():
            # Both routes; socket_digest() asserts they agree.
            digest = sk.socket_digest(a, b, fr)

            msg = sk.message_bytes(a, b, fr)
            full = ora.hash_bytes(msg, 32, rounds=rounds)
            # The 7-round external identity, stated as an executable claim.
            if rounds == 7:
                want = [int.from_bytes(full[4 * i:4 * i + 4], "little") for i in range(4)]
                if want != digest:
                    problems.append(f"{name}: 7-round != standard BLAKE3(msg)[..16]")

            controls_out = {}
            inapplicable = []
            for cname, cfr in sk.CONTROLS.items():
                cfr = sk.Framing(**{**cfr.__dict__, "rounds":
                                    (6 if cname == "rounds_6_not_7" else rounds)})
                if not sk.control_applicable(a, b, cfr, fr):
                    inapplicable.append(cname)
                    continue
                cd = sk.socket_digest_wordlevel(a, b, cfr)
                controls_out[cname] = "".join(f"{x:08x}" for x in cd)
                if cd != digest:
                    discriminated[cname].add(f"{name}@{rounds}")
                else:
                    problems.append(
                        f"CONTROL {cname} did NOT change the digest on {name}@{rounds}")

            table[str(rounds)].append({
                "name": name,
                "a": a,
                "b": b,
                "message_bytes_hex": msg.hex(),
                "digest": digest,
                "digest_lanes_hex": "".join(f"{x:08x}" for x in digest),
                "digest_bytes_hex": b"".join(
                    int(x).to_bytes(4, "little") for x in digest).hex(),
                "full_blake3_32B_hex": full.hex(),
                "negative_controls": controls_out,
                "controls_inapplicable": inapplicable,
            })

    undiscriminated = [c for c, s in discriminated.items() if not s]
    if undiscriminated:
        problems.append(f"controls NEVER discriminated by any vector: {undiscriminated}")

    return {
        "socket": "LFM_HASH 2-to-1 BLAKE3 compress (Option A + domain tag)",
        "spec": {
            "digest_lanes": sk.DIGEST_LANES,
            "digest_bits": 128,
            "domain_tag_ascii": sk.TAG_LFMC_ASCII.decode(),
            "domain_tag_word": sk.TAG_LFMC,
            "chaining_value_in": "BLAKE3 IV[0..8]",
            "counter": 0,
            "block_len": sk.BLOCK_LEN_LFMC,
            "flags": sk.FLAGS_LFMC,
            "flags_meaning": "CHUNK_START|CHUNK_END|ROOT",
            "message_layout": "m[0..4]=a, m[4..8]=b, m[8]=tag, m[9..16]=0",
            "truncation_window": "out[0..4] (the LOW four of 16 output words)",
            "lane_serialisation": "one felt = one u32 = four little-endian bytes "
                                  "(keccak_host convention, NOT word::pack_digest)",
        },
        "rounds": table,
        "control_discrimination": {c: sorted(s) for c, s in discriminated.items()},
        "_problems": problems,
    }


def cross_check(built: dict) -> tuple[bool, str]:
    """Recompute the PARALLEL AGENT's vectors with MY code.  Two independently
    written implementations of both the primitive and the framing must agree."""
    path = next((p for p in _PEER_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        return False, "peer socket_kats.json NOT FOUND -- cross-check CANNOT RUN"
    with open(path) as f:
        peer = json.load(f)

    if peer.get("domain_tag_word") != sk.TAG_LFMC:
        return False, (f"SPEC DISAGREEMENT: peer tag {peer.get('domain_tag_word')} "
                       f"vs mine {sk.TAG_LFMC}")
    for k, mine in (("block_len", sk.BLOCK_LEN_LFMC), ("flags", sk.FLAGS_LFMC),
                    ("counter", 0), ("digest_lanes", 4)):
        if peer.get(k) != mine:
            return False, f"SPEC DISAGREEMENT on {k}: peer {peer.get(k)} vs mine {mine}"

    n = 0
    for rounds_key, entries in peer["rounds"].items():
        rounds = int(rounds_key)
        fr = sk.honest(rounds)
        for e in entries:
            mine = sk.socket_digest(e["a"], e["b"], fr)
            if [f"{x:08x}" for x in mine] != [f"{x:08x}" for x in e["digest"]]:
                return False, (f"DIGEST MISMATCH on peer vector {e['name']}@{rounds}\n"
                               f"  mine={[hex(x) for x in mine]}\n"
                               f"  peer={[hex(x) for x in e['digest']]}")
            if sk.message_bytes(e["a"], e["b"], fr).hex() != e["message_bytes_hex"]:
                return False, f"MESSAGE MISMATCH on peer vector {e['name']}@{rounds}"
            n += 1
    return True, (f"cross-check PASS: {n} peer vectors reproduced exactly "
                  f"(spec fields agree too)")


def main() -> int:
    built = build()
    problems = built.pop("_problems")
    print("=" * 74)
    print("LAYER 2 -- socket KATs")
    print("=" * 74)
    nvec = sum(len(v) for v in built["rounds"].values())
    ncontrol = sum(len(e["negative_controls"])
                   for v in built["rounds"].values() for e in v)
    print(f"  vectors                     : {nvec} ({len(vectors())} inputs x 2 round counts)")
    print(f"  framing controls evaluated  : {ncontrol}")
    print(f"  framing degrees of freedom  : {len(sk.CONTROLS)}")
    for c, s in built["control_discrimination"].items():
        print(f"    {c:20s} discriminated by {len(s):2d} vector-instances")

    ok_cc, msg_cc = cross_check(built)
    print(f"  [{'PASS' if ok_cc else 'FAIL'}] {msg_cc}")

    if problems:
        print("\n  PROBLEMS:")
        for p in problems:
            print(f"    - {p}")

    if "--write" in sys.argv:
        with open(OUT, "w") as f:
            json.dump(built, f, indent=1)
        print(f"\n  wrote {OUT}")

    ok = ok_cc and not problems
    print("-" * 74)
    print(f"LAYER 2: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
