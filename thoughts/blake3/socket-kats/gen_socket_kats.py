"""
Reference vectors for the LFM 2-to-1 BLAKE3 compress socket (see SOCKET.md).

This generates the KATs for the SOCKET FUNCTION -- not for the bare compression
function `f`, which `CANONICAL_VECTORS` already pins. The socket adds five
framing degrees of freedom on top of `f` (where a and b land in the message,
the counter, the block length, the flags, and the truncation window), and every
one of them is a fresh way to be wrong.

Three computations must agree for each vector, and the script fails loudly if
they do not:

  W. WORD level, Python  -- the in-repo oracle's `compress`, called with the
     socket's (h, m, t, block_len, flags).
  C. WORD level, C       -- upstream BLAKE3's parameterised portable compress,
     called with the same tuple (reference-impl/b3ref{6,7} `compress`).
  B. BYTE level, C       -- upstream BLAKE3's WHOLE TREE HASHER over the byte
     string `a || b || tag`, truncated (reference-impl/b3ref{6,7} `hashhex`).

W-vs-C is the two-source check. **B is the one that matters for the framing**:
it says the socket is not merely "some compression call" but exactly a standard
BLAKE3 hash of a domain-separated 36-byte string. At rounds=7 that makes the
socket reproducible with a one-line `blake3` crate call and no oracle anywhere
in the chain -- which is the §2.3/§7 argument for 7 rounds, made concrete.

Run:  python3 gen_socket_kats.py       (after reference-impl/build.sh)
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE_DIR = os.path.join(HERE, "..", "blake3-oracle")
REFIMPL_DIR = os.path.join(HERE, "..", "reference-impl")

sys.path.insert(0, ORACLE_DIR)
import blake3_ref as ref  # noqa: E402

MASK32 = 0xFFFFFFFF

# --- the socket's frozen constants (SOCKET.md §2) ---------------------------

# Domain tag for the 2-to-1 compress socket: the four ASCII bytes "LFMC",
# read as one little-endian u32 message word.
DOMAIN_TAG_BYTES = b"LFMC"
DOMAIN_TAG_WORD = int.from_bytes(DOMAIN_TAG_BYTES, "little")  # 0x434D464C

SOCKET_COUNTER = 0
SOCKET_BLOCK_LEN = 36  # 8 digest words (32 bytes) + the 4-byte tag
SOCKET_FLAGS = ref.CHUNK_START | ref.CHUNK_END | ref.ROOT  # 0x0B
DIGEST_LANES = 4  # truncate the 8-word chaining value to its low 4 words

FAILURES = []


def check(name, cond, detail=""):
    if not cond:
        print(f"  FAIL  {name}   {detail}")
        FAILURES.append(name)
    return cond


# --- the socket, defined twice --------------------------------------------

def socket_message_words(a, b, tag=DOMAIN_TAG_WORD):
    """The 16-word message block m the compression consumes."""
    return list(a) + list(b) + [tag] + [0] * 7


def socket_message_bytes(a, b, tag_bytes=DOMAIN_TAG_BYTES):
    """The byte string the whole-hash form consumes: a || b || tag."""
    out = b"".join(w.to_bytes(4, "little") for w in list(a) + list(b))
    return out + tag_bytes


def socket_compress_python(a, b, rounds):
    out = ref.compress(list(ref.IV), socket_message_words(a, b), SOCKET_COUNTER,
                       SOCKET_BLOCK_LEN, SOCKET_FLAGS, rounds=rounds)
    return out[:DIGEST_LANES]


def c_binary(rounds):
    return os.path.join(REFIMPL_DIR, "b3ref7" if rounds == 7 else "b3ref6")


def c_compress_batch(records, rounds):
    """records: list of (h, m, t, block_len, flags). Returns 16-word outputs."""
    lines = []
    for h, m, t, bl, fl in records:
        words = [f"{w:08x}" for w in list(h) + list(m)]
        lines.append(" ".join(words) + f" {t:016x} {bl} {fl}\n")
    r = subprocess.run([c_binary(rounds), "compress"], input="".join(lines),
                       capture_output=True, text=True, check=True)
    out = r.stdout.strip().split("\n")
    assert len(out) == len(records), (len(out), len(records))
    return [[int(ln[8 * i:8 * i + 8], 16) for i in range(16)] for ln in out]


def c_hash_bytes(msg, out_len, rounds):
    r = subprocess.run([c_binary(rounds), "hashhex", msg.hex(), str(out_len)],
                       capture_output=True, text=True, check=True)
    return bytes.fromhex(r.stdout.strip())


def digest_from_bytes(digest_bytes):
    return [int.from_bytes(digest_bytes[4 * i:4 * i + 4], "little")
            for i in range(DIGEST_LANES)]


# --- test inputs (explicit; every one is written into the JSON) ------------

def test_inputs():
    """Five structural cases then five formula cases. The formula is
    a[i] = 0x9E3779B9*(16k+i+1) mod 2^32, b[i] = 0x9E3779B9*(16k+i+9) mod 2^32,
    so any language can regenerate them; the JSON lists them explicitly anyway."""
    cases = [
        ("zeros", [0] * 4, [0] * 4),
        ("a_one", [1, 0, 0, 0], [0] * 4),
        ("b_one", [0] * 4, [1, 0, 0, 0]),
        ("all_ones", [MASK32] * 4, [MASK32] * 4),
        ("nibble_ramp",
         [0x00000000, 0x11111111, 0x22222222, 0x33333333],
         [0x44444444, 0x55555555, 0x66666666, 0x77777777]),
    ]
    for k in range(5):
        a = [(0x9E3779B9 * (16 * k + i + 1)) & MASK32 for i in range(4)]
        b = [(0x9E3779B9 * (16 * k + i + 9)) & MASK32 for i in range(4)]
        cases.append((f"formula_{k}", a, b))
    return cases


# --- negative controls: one framing degree of freedom each -----------------

def byteswap32(w):
    return int.from_bytes(w.to_bytes(4, "little"), "big")


def control_applicable(name, a, b):
    """Whether a control can discriminate on THESE inputs.

    Two controls are no-ops on degenerate inputs and would otherwise look like
    failures: swapping a and b when a == b, and re-packing lanes big-endian
    when every lane is a byte-palindrome (0x00000000, 0xFFFFFFFF, 0x11111111,
    ...). Those cases are declared inapplicable rather than quietly skipped,
    and `main` separately asserts that every control is still discriminated by
    at least one vector -- otherwise a degree of freedom would sit unpinned
    behind a green run.
    """
    if name == "swap_a_b":
        return list(a) != list(b)
    if name == "lanes_big_endian":
        return any(byteswap32(w) != w for w in list(a) + list(b))
    return True


def negative_controls(a, b, rounds):
    """Each entry perturbs exactly one framing choice and must change the
    digest. A control that does NOT change it means that degree of freedom is
    unpinned -- the vectors would accept a wrong implementation."""
    iv = list(ref.IV)
    m = socket_message_words(a, b)
    controls = {}

    # N1 operand order.
    controls["swap_a_b"] = ref.compress(
        iv, socket_message_words(b, a), SOCKET_COUNTER, SOCKET_BLOCK_LEN,
        SOCKET_FLAGS, rounds=rounds)[:DIGEST_LANES]

    # N2 domain tag value ("LFMC" -> "LFMD").
    controls["tag_changed"] = ref.compress(
        iv, socket_message_words(a, b, int.from_bytes(b"LFMD", "little")),
        SOCKET_COUNTER, SOCKET_BLOCK_LEN, SOCKET_FLAGS,
        rounds=rounds)[:DIGEST_LANES]

    # N3 tag omitted entirely (message is 32 bytes, m[8] = 0).
    controls["tag_omitted"] = ref.compress(
        iv, socket_message_words(a, b, 0), SOCKET_COUNTER, 32, SOCKET_FLAGS,
        rounds=rounds)[:DIGEST_LANES]

    # N4 truncation window moved to the high half of the chaining value.
    full = ref.compress(iv, m, SOCKET_COUNTER, SOCKET_BLOCK_LEN, SOCKET_FLAGS,
                        rounds=rounds)
    controls["truncate_high_half"] = full[4:8]

    # N5 flags: PARENT instead of CHUNK_START|CHUNK_END|ROOT.
    controls["flags_parent"] = ref.compress(
        iv, m, SOCKET_COUNTER, SOCKET_BLOCK_LEN, ref.PARENT,
        rounds=rounds)[:DIGEST_LANES]

    # N6 block_len declared 64 rather than the true 36.
    controls["block_len_64"] = ref.compress(
        iv, m, SOCKET_COUNTER, 64, SOCKET_FLAGS, rounds=rounds)[:DIGEST_LANES]

    # N7 counter nonzero.
    controls["counter_one"] = ref.compress(
        iv, m, 1, SOCKET_BLOCK_LEN, SOCKET_FLAGS, rounds=rounds)[:DIGEST_LANES]

    # N8 lanes packed big-endian instead of little-endian.
    be = [byteswap32(w) for w in list(a) + list(b)]
    controls["lanes_big_endian"] = ref.compress(
        iv, be + [DOMAIN_TAG_WORD] + [0] * 7, SOCKET_COUNTER, SOCKET_BLOCK_LEN,
        SOCKET_FLAGS, rounds=rounds)[:DIGEST_LANES]

    # N9 the other round count.
    controls["other_round_count"] = socket_compress_python(
        a, b, 6 if rounds == 7 else 7)

    return controls


def main():
    for r in (6, 7):
        if not os.path.exists(c_binary(r)):
            print(f"missing {c_binary(r)} -- run reference-impl/build.sh first")
            return 2

    doc = {
        "socket": "LFM 2-to-1 BLAKE3 compress (see SOCKET.md)",
        "digest_lanes": DIGEST_LANES,
        "digest_bits": 32 * DIGEST_LANES,
        "domain_tag_ascii": DOMAIN_TAG_BYTES.decode(),
        "domain_tag_word": DOMAIN_TAG_WORD,
        "counter": SOCKET_COUNTER,
        "block_len": SOCKET_BLOCK_LEN,
        "flags": SOCKET_FLAGS,
        "flags_meaning": "CHUNK_START|CHUNK_END|ROOT",
        "chaining_value_in": "BLAKE3 IV",
        "message_layout": "m[0..4]=a, m[4..8]=b, m[8]=tag, m[9..16]=0",
        "rounds": {},
    }

    cases = test_inputs()
    discriminated = {}
    print("=" * 74)
    print("LFM 2-to-1 BLAKE3 compress socket -- reference vectors")
    print("=" * 74)

    for rounds in (7, 6):
        # Batch the word-level C calls.
        records = [(list(ref.IV), socket_message_words(a, b), SOCKET_COUNTER,
                    SOCKET_BLOCK_LEN, SOCKET_FLAGS) for _, a, b in cases]
        c_out = c_compress_batch(records, rounds)

        vectors = []
        for idx, (name, a, b) in enumerate(cases):
            w = socket_compress_python(a, b, rounds)
            c = c_out[idx][:DIGEST_LANES]
            msg = socket_message_bytes(a, b)
            digest32 = c_hash_bytes(msg, 32, rounds)
            bl = digest_from_bytes(digest32)

            check(f"r{rounds} {name}: python word == C word", w == c, f"{w} vs {c}")
            check(f"r{rounds} {name}: word form == BLAKE3(a||b||tag) truncated",
                  w == bl, f"{w} vs {bl}")

            ctrls = negative_controls(a, b, rounds)
            inapplicable = []
            for cname, cval in ctrls.items():
                if not control_applicable(cname, a, b):
                    inapplicable.append(cname)
                    continue
                if check(f"r{rounds} {name}: control '{cname}' changes the digest",
                         cval != w, f"control equals the canonical digest {w}"):
                    discriminated.setdefault(rounds, set()).add(cname)

            vectors.append({
                "name": name,
                "a": list(a),
                "b": list(b),
                "message_bytes_hex": msg.hex(),
                "digest": w,
                "digest_hex": "".join(f"{x:08x}" for x in w),
                "full_blake3_digest_hex": digest32.hex(),
                "negative_controls": {k: v for k, v in ctrls.items()},
                "controls_inapplicable_here": inapplicable,
            })

        doc["rounds"][str(rounds)] = vectors
        # Every control must be discriminated by at least one vector, or that
        # framing degree of freedom is unpinned by this table.
        all_controls = set(vectors[0]["negative_controls"].keys())
        missed = all_controls - discriminated.get(rounds, set())
        check(f"r{rounds}: every control is discriminated by >=1 vector",
              not missed, f"never discriminated: {sorted(missed)}")
        print(f"  rounds={rounds}: {len(vectors)} vectors, "
              f"{len(all_controls)} controls, all discriminated")

    # The headline cross-check, stated once more as an explicit assertion.
    a, b = cases[4][1], cases[4][2]
    seven = socket_compress_python(a, b, 7)
    lib = digest_from_bytes(c_hash_bytes(socket_message_bytes(a, b), 32, 7))
    check("HEADLINE: at rounds=7 the socket IS truncated standard BLAKE3",
          seven == lib)

    out_path = os.path.join(HERE, "socket_kats.json")
    json.dump(doc, open(out_path, "w"), indent=2)

    print("=" * 74)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S)")
        for f in FAILURES[:10]:
            print("   -", f)
        return 1
    print("RESULT: ALL GREEN")
    print(f"  wrote {os.path.basename(out_path)}")
    print("  At rounds=7 every vector equals blake3::hash(a||b||\"LFMC\")[0..16],")
    print("  so the build phase can re-derive this table from the crate alone.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
