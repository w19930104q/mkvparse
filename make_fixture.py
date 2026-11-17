# Licence==MIT
"""
Build the deterministic Matroska fixture used by recovery_check.py.

fixed.mkv layout (334 bytes), designed to exercise the recovery rules in
mkvstream.py:

    0   EBML header (known size, 40 bytes total, fixed)
    40  Segment                       unknown size
    45    SeekHead (known size)
    50      Seek -> Info @ 0
    64      Void                      <- Void inside SeekHead
    73    Info (known size)
    78      TimestampScale
    85      unknown_6666              <- unknown element id, ordered metadata
    92      MuxingApp / WritingApp
    126   Tracks (known size, 2 tracks: V_MJPEG + A_MPEG/L3)
    197   Cluster#0 (known size)      2 SimpleBlocks
    232   Cluster#1                   unknown size, Void inside, Xiph lacing
    285   Cluster#2 (known size)
    296     Cluster                   <- nested Cluster, unknown size
    314   Tags (known size)
    319     Tag                       unknown size
    322       Targets
    329       unknown_7777

Variants:
    t1_boundary.mkv  clean cut at 232 (end of Cluster#0)
    t2_dirty.mkv     dirty cut at 299 (inside the nested Cluster header)
    part_a/part_b    split at 299, for the checkpoint-resume test
    oversize.mkv     TargetTypeValue size byte 0x81 -> 0x85: child end
                     exceeds the known-size Targets parent (rule R2)
"""

import os
from struct import pack

CUT_AFTER_CLUSTER0 = 232
CUT_INSIDE_HEADER = 299
EOF = 334


def encode_id(id_):
    return id_.to_bytes((id_.bit_length() + 7) // 8, "big")


def encode_size(n):
    if n < 0x7F:
        return bytes([0x80 | n])
    if n < 0x3FFF:
        return (0x4000 | n).to_bytes(2, "big")
    if n < 0x1FFFFF:
        return (0x200000 | n).to_bytes(3, "big")
    raise ValueError("size too big for this fixture")


def E(id_, payload):
    return encode_id(id_) + encode_size(len(payload)) + payload


def E_unknown(id_, payload):
    return encode_id(id_) + b"\xff" + payload


def u(n):
    return n.to_bytes(max(1, (n.bit_length() + 7) // 8), "big")


def block(track, tcode, flags, payload):
    return bytes([0x80 | track]) + pack(">h", tcode) + bytes([flags]) + payload


def build_fixed():
    ebml = E(0x1A45DFA3,
             E(0x4286, u(1)) + E(0x42F7, u(1)) + E(0x42F2, u(4))
             + E(0x42F3, u(8)) + E(0x4282, b"matroska")
             + E(0x4287, u(2)) + E(0x4285, u(2)))

    seekhead = E(0x114D9B74,
                 E(0x4DBB, E(0x53AB, encode_id(0x1549A966)) + E(0x53AC, u(0)))
                 + E(0xEC, b"\x00" * 7))

    info = E(0x1549A966,
             E(0x2AD7B1, u(1000000))
             + E(0x6666, b"abcd")                    # unknown element id
             + E(0x4D80, b"mkvstack-test")
             + E(0x5741, b"make_fixture.py"))

    entry1 = E(0xAE,
               E(0xD7, u(1)) + E(0x73C5, u(0x0100)) + E(0x83, u(1))
               + E(0x86, b"V_MJPEG")
               + E(0xE0, E(0xB0, u(320)) + E(0xBA, u(240))))
    entry2 = E(0xAE,
               E(0xD7, u(2)) + E(0x73C5, u(0x0101)) + E(0x83, u(2))
               + E(0x86, b"A_MPEG/L3")
               + E(0xE1, E(0x9F, u(2)) + E(0xB5, pack(">f", 8000.0))))
    tracks = E(0x1654AE6B, entry1 + entry2)

    cluster0 = E(0x1F43B675,
                 E(0xE7, u(0)) + E(0xAB, u(0))
                 + E(0xA3, block(1, 0, 0x80, b"VID0-key"))
                 + E(0xA3, block(2, 0, 0x00, b"AUD0")))

    laced = (bytes([0x82]) + pack(">h", 0) + bytes([0x02, 0x02])
             + bytes([0x06, 0x06]) + b"lace-a" + b"lace-b" + b"lace-c")
    cluster1 = E_unknown(0x1F43B675,
                         E(0xE7, u(26)) + E(0xAB, u(35))
                         + E(0xA3, block(1, 0, 0x00, b"VID1"))
                         + E(0xEC, b"\x00" * 3)      # Void inside Cluster
                         + E(0xA3, laced))

    nested = E_unknown(0x1F43B675,
                       E(0xE7, u(99))
                       + E(0xA3, block(1, 0, 0x00, b"VID2")))
    cluster2 = E(0x1F43B675,
                 E(0xE7, u(52)) + E(0xAB, u(53)) + nested)

    tags = E(0x1254C367,
             E_unknown(0x7373,
                       E(0x63C0, E(0x68CA, u(50)))
                       + E(0x7777, b"zz")))           # unknown element id

    segment = E_unknown(0x18538067,
                        seekhead + info + tracks
                        + cluster0 + cluster1 + cluster2 + tags)
    return ebml + segment


def build_all():
    fixed = build_fixed()
    assert len(fixed) == EOF, len(fixed)
    oversize = bytearray(fixed)
    # TargetTypeValue (0x68CA) size byte at offset 327: 1 -> 5, so the child
    # claims end 333 while its known-size parent Targets ends at 329.
    assert oversize[325] == 0x68 and oversize[326] == 0xCA
    assert oversize[327] == 0x81
    oversize[327] = 0x85
    return {
        "fixed.mkv": fixed,
        "t1_boundary.mkv": fixed[:CUT_AFTER_CLUSTER0],
        "t2_dirty.mkv": fixed[:CUT_INSIDE_HEADER],
        "part_a.mkv": fixed[:CUT_INSIDE_HEADER],
        "part_b.mkv": fixed[CUT_INSIDE_HEADER:],
        "oversize.mkv": bytes(oversize),
    }


def write_all(outdir):
    os.makedirs(outdir, exist_ok=True)
    for name, data in build_all().items():
        with open(os.path.join(outdir, name), "wb") as f:
            f.write(data)


if __name__ == "__main__":
    import sys
    write_all(sys.argv[1] if len(sys.argv) > 1 else ".")
