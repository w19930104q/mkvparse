# Licence==MIT; Vitaly "_Vi" Shukela 2012, stack parser 2026

"""
Streaming Matroska/EBML parser built around an explicit element stack.

mkvparse.py treats unknown EBML sizes and nested Clusters as opaque units
and falls back to a blind byte-scan resync, which loses sync when Void /
SeekHead / unknown elements are interleaved with streaming track data.
This parser instead:

  * keeps an explicit stack of open elements (id, type, offsets, level,
    unknown-size flag) -- see StackEntry;
  * applies deterministic recovery rules instead of blind resync:

      R1 (unknown-size close)  an unknown-size master is closed when a
                               "boundary" element ID appears that may only
                               occur at the same or a higher level (a new
                               Cluster/SeekHead/Info/Tracks/Cues/... while a
                               Cluster is open, a new Segment/EBML header,
                               or the end of a known-size parent).  Void and
                               CRC-32 are *not* boundary IDs: they are legal
                               at any level and must not close anything.
      R2 (oversize child)      an element whose computed end exceeds the
                               nearest enclosing known-size ancestor is not
                               trusted; parsing resumes at that ancestor's
                               end, which is a verified boundary.
      R3 (unknown-size atomic) unknown size on a non-master element is
                               invalid EBML; the parser scans forward to the
                               next boundary ID and continues there.
      R4 (truncation)          an incomplete header or payload (EOF in the
                               middle) is not an error: the parser rolls
                               back to the element start and suspends, so
                               the parse can be resumed later.

  * preserves metadata order: children of a master element are delivered as
    an ordered list of (name, value) pairs, never as a dict;
  * supports checkpoint/restart: after every fully verified element the
    parser records a checkpoint (offset + stack snapshot).  A truncated
    parse can be resumed from the last checkpoint on a new stream that
    yields bytes starting at the checkpoint offset (see snapshot/restore).

Verified boundaries: an element is "verified" when its end is confirmed
either by its own declared size or by a same-level boundary ID (R1).  An
unknown-size element closed by a *higher*-level boundary or by EOF is
reported verified=no: its position is known, its content boundary is not.
"""

import sys
import datetime
from struct import unpack

from mkvparse import (
    element_types_names,
    EbmlElementType as EET,
    handle_block,
    parse_fixedlength_number,
    get_major_bit_number,
)

UNKNOWN_SIZE = -1

ID_EBML    = 0x1A45DFA3
ID_SEGMENT = 0x18538067
ID_CLUSTER = 0x1F43B675
ID_TIMESTAMP   = 0xE7
ID_SIMPLEBLOCK = 0xA3
ID_BLOCKGROUP  = 0xA0

# Boundary IDs (R1) mapped to the stack level at which they may appear.
# Level 0 = root (EBML, Segment), level 1 = direct child of Segment.
# Void (0xEC) and CRC-32 (0xBF) are intentionally absent: legal anywhere.
BOUNDARY_LEVEL = {
    ID_EBML:    0,
    ID_SEGMENT: 0,
    0x114D9B74: 1,  # SeekHead
    0x1549A966: 1,  # Info
    0x1654AE6B: 1,  # Tracks
    ID_CLUSTER: 1,  # Cluster
    0x1C53BB6B: 1,  # Cues
    0x1941A469: 1,  # Attachments
    0x1043A770: 1,  # Chapters
    0x1254C367: 1,  # Tags
    0x1B538667: 1,  # SignatureSlot
}

TYPE_NAMES = {
    EET.VOID: "binary",
    EET.MASTER: "master",
    EET.JUST_GO_ON: "master",
    EET.UNSIGNED: "uint",
    EET.SIGNED: "sint",
    EET.TEXTA: "texta",
    EET.TEXTU: "textu",
    EET.BINARY: "binary",
    EET.FLOAT: "float",
    EET.DATE: "date",
}

MASTER_TYPES = (EET.MASTER, EET.JUST_GO_ON)

_DATE_EPOCH_SHIFT = (datetime.datetime(2001, 1, 1)
                     - datetime.datetime(1970, 1, 1)).total_seconds()


class Truncated(Exception):
    """EOF in the middle of an element header or payload (R4)."""


class StackEntry:
    """One open element on the explicit element stack."""
    __slots__ = ("id_", "name", "type_", "hsize", "size", "start",
                 "data_start", "end", "level", "tc", "children")

    def __init__(self, id_, name, type_, hsize, size, start, level):
        self.id_ = id_
        self.name = name
        self.type_ = type_
        self.hsize = hsize
        self.size = size                    # UNKNOWN_SIZE == -1
        self.start = start                  # offset of the element ID
        self.data_start = start + hsize
        self.end = None if size == UNKNOWN_SIZE else self.data_start + size
        self.level = level
        self.tc = 0                         # cluster timestamp, if Cluster
        self.children = []                  # ordered (name, (type, value))

    def to_row(self):
        return [self.id_, self.hsize, self.size, self.start, self.data_start,
                self.end, self.size == UNKNOWN_SIZE, self.level, False,
                self.tc]

    @staticmethod
    def from_row(row, type_, name):
        (id_, hsize, size, start, data_start, end,
         _unknown, level, _verified, tc) = row
        e = StackEntry(id_, name, type_, hsize, size, start, level)
        e.end = end
        e.tc = tc
        return e


class StreamHandler:
    """Override these; mirrors the interesting parts of MatroskaHandler."""
    def frame(self, track_id, timestamp, data, more_laced_frames, duration,
              keyframe, invisible, discardable):
        pass

    def tracks_available(self, tracks):
        pass

    def segment_info_available(self, info):
        pass


class OffsetReader:
    """Seekable reader tracking absolute stream offsets."""

    def __init__(self, f, base_offset=0):
        self.f = f
        self.base = base_offset

    def tell(self):
        return self.base + self.f.tell()

    def seek(self, abspos):
        self.f.seek(abspos - self.base)

    def read(self, n):
        return self.f.read(n)

    def read_exact(self, n):
        data = self.f.read(n)
        if len(data) < n:
            raise Truncated("wanted %d bytes, got %d" % (n, len(data)))
        return data


def decode_value(type_, data):
    """Decode an atomic element payload, mirroring read_simple_element."""
    size = len(data)
    if size == 0:
        return ""
    if type_ == EET.UNSIGNED:
        return parse_fixedlength_number(data, 0, size, False)[0]
    if type_ == EET.SIGNED:
        return parse_fixedlength_number(data, 0, size, True)[0]
    if type_ == EET.TEXTA:
        return data.replace(b"\x00", b"").decode("ascii")
    if type_ == EET.TEXTU:
        return data.replace(b"\x00", b"").decode("UTF-8")
    if type_ == EET.DATE:
        v = parse_fixedlength_number(data, 0, size, True)[0]
        return v * 1e-9 + _DATE_EPOCH_SHIFT
    if type_ == EET.FLOAT:
        if size == 4:
            return unpack(">f", data)[0]
        if size == 8:
            return unpack(">d", data)[0]
        return None
    return data


class StreamParser:
    """
    Usage:
        p = StreamParser(handler)
        p.start(open("file.mkv", "rb"))
        p.run()                      # runs to EOF or suspends on truncation
        snap = p.snapshot()          # JSON-able checkpoint + stack

        # later, on a stream positioned at snap["checkpoint"]:
        p2 = StreamParser(handler)
        p2.restore(snap)
        p2.start(new_stream, base_offset=snap["checkpoint"])
        p2.run()

    Parse events are appended to self.lines (element and "# end" rows).
    """

    def __init__(self, handler=None):
        self.handler = handler or StreamHandler()
        self.stack = []
        self.lines = []
        self.elements = []          # (start, path, id_, size, end) records
        self.timecode_scale = 1000000
        self.checkpoint = 0
        self.base_offset = 0
        self.suspended = False
        self.eof = False
        self._snap_stack = []
        self.r = None

    # ------------------------------------------------------------------ i/o

    def start(self, f, base_offset=0):
        self.r = OffsetReader(f, base_offset)
        self.base_offset = base_offset

    def snapshot(self):
        return {
            "checkpoint": self.checkpoint,
            "base_offset": self.base_offset,
            "timecode_scale": self.timecode_scale,
            "stack": [list(row) for row in self._snap_stack],
            "prev_cluster_start": None,
        }

    def restore(self, snap):
        self.checkpoint = snap["checkpoint"]
        self.timecode_scale = snap.get("timecode_scale", 1000000)
        self.stack = []
        for row in snap["stack"]:
            id_ = row[0]
            type_, name = element_types_names.get(
                id_, (EET.BINARY, "unknown_%x" % id_))
            self.stack.append(StackEntry.from_row(row, type_, name))
        self._snap_stack = [list(r) for r in snap["stack"]]

    # -------------------------------------------------------------- events

    def _emit_element(self, start, path, id_, type_, size, end):
        tname = TYPE_NAMES.get(type_, "binary")
        self.lines.append("%d\t%s\t%x\t%s\t%s\t%s" % (
            start, path, id_, tname,
            "unknown" if size == UNKNOWN_SIZE else size,
            "?" if end is None else end))
        self.elements.append((start, path, id_, size, end))

    def _emit_close(self, pos, entry, verified):
        self.lines.append("# end   %-41s%s verified=%s" % (
            pos, self._path(entry), "yes" if verified else "no"))

    def _emit_note(self, text):
        self.lines.append("# " + text)

    def _path(self, extra=None):
        names = [e.name for e in self.stack]
        if extra is not None:
            names.append(extra.name if isinstance(extra, StackEntry) else extra)
        return "/".join(names)

    def _mark_verified(self, pos):
        """Record a verified boundary we can restart from."""
        self.checkpoint = pos
        self._snap_stack = [e.to_row() for e in self.stack]

    # ------------------------------------------------------------- stack ops

    def _close_entry(self, pos, verified):
        e = self.stack.pop()
        self._emit_close(pos, e, verified)
        if e.type_ == EET.MASTER:
            self._master_finished(e)
        if verified:
            self._mark_verified(pos)
        return e

    def _close_finished(self, pos):
        """Pop entries ended at or before pos.  A known-size entry deeper
        in the stack whose end has been reached also closes any
        unknown-size entries above it: a child cannot outlive its
        parent's verified end."""
        while True:
            idx = None
            for i in range(len(self.stack) - 1, -1, -1):
                e = self.stack[i]
                if e.end is None:
                    continue
                if pos >= e.end:
                    idx = i
                break  # nearest known-size ancestor decides
            if idx is None:
                return
            end = self.stack[idx].end
            while len(self.stack) > idx:
                e = self.stack[-1]
                self._close_entry(end, verified=(e.end is not None
                                                 and e.end == end))

    def _apply_boundary(self, id_, pos):
        """R1: a boundary ID closes open unknown-size entries at its level
        or deeper.  Known-size entries are trusted and never popped."""
        level = BOUNDARY_LEVEL.get(id_)
        if level is None:
            return
        while self.stack:
            e = self.stack[-1]
            if e.end is not None or e.level < level:
                break
            # sibling boundary confirms the end; ancestor-level boundary
            # or EOF only tells us where we are, not that the content ended
            self._close_entry(pos, verified=(e.level == level))

    def _finalize(self, pos):
        """Clean EOF: close everything still open."""
        while self.stack:
            e = self.stack[-1]
            self._close_entry(pos,
                              verified=(e.end is not None and e.end == pos))

    # ------------------------------------------------------------- parsing

    def _read_header(self):
        """Returns (id, size, hsize, start) or None at a clean EOF."""
        start = self.r.tell()
        first = self.r.read(1)
        if first == b"":
            return None
        (nlen, _) = get_major_bit_number(first[0])
        rest = self.r.read(nlen)
        if len(rest) < nlen:
            raise Truncated("element ID cut off")
        id_ = first[0]
        for b in rest:
            id_ = id_ * 0x100 + b
        sb = self.r.read(1)
        if not sb:
            raise Truncated("element size cut off")
        (slen, _) = get_major_bit_number(sb[0])
        val = sb[0] & ~(0x80 >> slen)
        rest = self.r.read(slen)
        if len(rest) < slen:
            raise Truncated("element size cut off")
        for b in rest:
            val = val * 0x100 + b
        size = UNKNOWN_SIZE if val == 2 ** (7 * slen + 7) - 1 else val
        return (id_, size, self.r.tell() - start, start)

    def _resync(self):
        """R3: scan forward for the next boundary ID, leave stream at it."""
        first_bytes = set()
        for id_ in BOUNDARY_LEVEL:
            n = (id_.bit_length() + 7) // 8
            first_bytes.add((id_ >> (8 * (n - 1))) & 0xFF)
        while True:
            b = self.r.read(1)
            if b == b"":
                return
            if b[0] not in first_bytes:
                continue
            pos = self.r.tell() - 1
            matched = False
            for id_ in BOUNDARY_LEVEL:
                n = (id_.bit_length() + 7) // 8
                if n == 1:
                    cand = bytes([b[0]])
                else:
                    self.r.seek(pos)
                    cand = self.r.read(n)
                if int.from_bytes(cand, "big") == id_:
                    matched = True
                    break
            if matched:
                self.r.seek(pos)
                return
            self.r.seek(pos + 1)

    def run(self):
        self.suspended = False
        while True:
            pos = self.r.tell()
            try:
                hdr = self._read_header()
            except Truncated:
                # R4: incomplete element; roll back and suspend so the
                # parse can be resumed from the last verified checkpoint.
                self.r.seek(pos)
                self.suspended = True
                return
            if hdr is None:
                self.eof = True
                break
            (id_, size, hsize, start) = hdr
            self._close_finished(pos)
            self._apply_boundary(id_, pos)

            type_, name = element_types_names.get(
                id_, (EET.BINARY, "unknown_%x" % id_))
            is_master = type_ in MASTER_TYPES

            if size == UNKNOWN_SIZE:
                if is_master:
                    e = StackEntry(id_, name, type_, hsize, size, start,
                                   len(self.stack))
                    self._emit_element(start, self._path(name), id_, type_,
                                       size, None)
                    self.stack.append(e)
                else:
                    # R3: unknown-size atomic element is invalid EBML.
                    self._emit_note(
                        "resync at %d: unknown-size atomic element %x, "
                        "scanning for next boundary ID" % (start, id_))
                    self._resync()
                continue

            end = start + hsize + size

            # R2: element end must not exceed the nearest enclosing
            # known-size ancestor; if it does, restart at that verified
            # boundary instead of trusting the element.
            oversize_at = None
            for e in reversed(self.stack):
                if e.end is not None and end > e.end:
                    oversize_at = e
                    break
            if oversize_at is not None:
                self._emit_note(
                    "oversize at %d: element %x end %d exceeds %s end %d; "
                    "skipping to verified boundary" % (
                        start, id_, end, self._path(), oversize_at.end))
                self.r.seek(oversize_at.end)
                continue

            if is_master:
                e = StackEntry(id_, name, type_, hsize, size, start,
                               len(self.stack))
                self._emit_element(start, self._path(name), id_, type_,
                                   size, e.end)
                self.stack.append(e)
                continue

            try:
                data = self.r.read_exact(size)
            except Truncated:
                self.r.seek(pos)            # R4, as above
                self.suspended = True
                return
            value = decode_value(type_, data)
            self._emit_element(start, self._path(name), id_, type_, size, end)
            self._atomic_finished(name, type_, value, data)
            self._mark_verified(end)

        self._finalize(self.r.tell())

    # ------------------------------------------------------- semantic hooks

    def _innermost_cluster(self):
        for e in reversed(self.stack):
            if e.id_ == ID_CLUSTER:
                return e
        return None

    def _atomic_finished(self, name, type_, value, data):
        if self.stack and self.stack[-1].type_ == EET.MASTER:
            self.stack[-1].children.append((name, (type_, value)))
        if name == "Timestamp" and type_ == EET.UNSIGNED:
            c = self._innermost_cluster()
            if c is not None:
                c.tc = value
        elif name == "SimpleBlock" and type_ == EET.BINARY:
            c = self._innermost_cluster()
            handle_block(data, self.handler, c.tc if c else 0,
                         self.timecode_scale, None, {})

    def _master_finished(self, entry):
        tree = entry.children
        if self.stack and self.stack[-1].type_ == EET.MASTER:
            self.stack[-1].children.append((entry.name, (EET.MASTER, tree)))
        if entry.name == "Info":
            self.handler.segment_info_available(tree)
            d = dict(tree)
            if "TimestampScale" in d:
                self.timecode_scale = d["TimestampScale"][1]
        elif entry.name == "Tracks":
            tracks = {}
            for (ten, (_t, track)) in tree:
                if ten != "TrackEntry":
                    continue
                d = dict(track)
                n = d["TrackNumber"][1]
                tt = d["TrackType"][1]
                d["type"] = {0x01: "video", 0x02: "audio", 0x03: "complex",
                             0x10: "logo", 0x11: "subtitle", 0x12: "button",
                             0x20: "control"}.get(tt)
                tracks[n] = d
            self.handler.tracks_available(tracks)
        elif entry.name == "BlockGroup":
            d = dict(tree)
            duration = None
            if "BlockDuration" in d:
                duration = d["BlockDuration"][1] * 0.000000001 \
                    * self.timecode_scale
            if "Block" in d:
                c = self._innermost_cluster()
                handle_block(d["Block"][1], self.handler, c.tc if c else 0,
                             self.timecode_scale, duration, {})
        elif entry.name == "EBML":
            d = dict(tree)
            dt = d.get("DocType", (None, ""))[1]
            if dt not in ("matroska", "webm"):
                sys.stderr.write(
                    "mkvstream: Warning: EBML DocType is not "
                    "\"matroska\" or \"webm\"\n")
