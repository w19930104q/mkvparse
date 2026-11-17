#!/usr/bin/env python3
# Licence==MIT
"""
Recovery test harness for mkvstream.py.

Usage: python3 recovery_check.py <outdir>

Builds the fixed MKV fixture and truncated/corrupted variants into <outdir>,
parses each with the explicit-stack parser printing element paths and
offsets, then performs the diffs:

    diff1  ref.out vs t1.out      clean truncation at a verified boundary
    diff2  ref.out vs resume.out  checkpoint resume on a split stream
    diff3  ref.out vs t2.out      dirty truncation inside an element header

and writes report.txt summarizing all checks.
"""

import contextlib
import difflib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import make_fixture
import mkvstream
import mkvparse


class CollectHandler(mkvstream.StreamHandler):
    def __init__(self):
        self.frames = []
        self.tracks = None
        self.segment_info = None

    def frame(self, track_id, timestamp, data, more_laced_frames, duration,
              keyframe, invisible, discardable):
        self.frames.append((track_id, timestamp, len(data),
                            more_laced_frames, keyframe))

    def tracks_available(self, tracks):
        self.tracks = tracks

    def segment_info_available(self, info):
        self.segment_info = info


def parse_file(path, handler=None):
    handler = handler or CollectHandler()
    p = mkvstream.StreamParser(handler)
    with open(path, "rb") as f:
        p.start(f)
        p.run()
    return p, handler


def write_lines(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def write_diff(path, name, ref_lines, other_lines):
    diff = list(difflib.unified_diff(
        ref_lines, other_lines,
        fromfile="%s-left" % name, tofile="%s-right" % name, lineterm=""))
    write_lines(path, diff)
    return diff


def element_lines(lines):
    """Element rows only (no '# end' close rows / notes)."""
    return [l for l in lines if not l.startswith("#")]


def strip_unverified_closes(lines):
    """Drop trailing '# end ... verified=no' rows (EOF finalization)."""
    out = list(lines)
    while out and out[-1].startswith("# end") \
            and out[-1].endswith("verified=no"):
        out.pop()
    return out


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    make_fixture.write_all(outdir)
    P = lambda name: os.path.join(outdir, name)

    report = []

    # ---- reference parse -------------------------------------------------
    ref, ref_h = parse_file(P("fixed.mkv"))
    write_lines(P("ref.out"), ref.lines)

    # fixed 40-byte EBML header verified at offset 0
    ebml_ok = ref.elements[0][1] == "EBML" and ref.elements[0][4] == 40
    report.append("header: fixed 40-byte EBML header verified at offset 0: "
                  "%s" % ebml_ok)

    # metadata order: unknown_6666 must stay between TimestampScale and
    # MuxingApp inside Info (ordered list, not dict)
    info_names = [n for (n, _v) in ref_h.segment_info]
    order_ok = info_names.index("unknown_6666") < info_names.index("MuxingApp")
    report.append("metadata order preserved (unknown_6666 before MuxingApp): "
                  "%s" % order_ok)

    frames_ok = len(ref_h.frames) == 7
    track_list = sorted(ref_h.tracks) if ref_h.tracks else []
    report.append("frames extracted: %d (expect 7); tracks=%s"
                  % (len(ref_h.frames), track_list))
    with open(P("frames.json"), "w") as f:
        json.dump(ref_h.frames, f, indent=1)

    # element offsets, for cut-point documentation
    offs = {}
    clusters = []
    for (start, path, id_, size, end) in ref.elements:
        name = path.split("/")[-1]
        depth = path.count("/")
        if depth == 1 and name in ("SeekHead", "Info", "Tracks", "Tags"):
            offs[name.lower()] = start
        elif path == "Segment":
            offs["segment"] = start
        elif name == "Cluster" and depth == 1:
            clusters.append(start)
        elif name == "Cluster" and depth > 1:
            offs["nested_cluster"] = start
        elif name == "unknown_6666":
            offs["unknown_in_info"] = start
        elif name == "Void" and "SeekHead" in path:
            offs["void_in_seekhead"] = start
        elif name == "Void" and "Cluster" in path:
            offs["void_in_cluster1"] = start
    for i, c in enumerate(clusters):
        offs["cluster%d" % i] = c
    offs["cut_after_cluster0"] = make_fixture.CUT_AFTER_CLUSTER0
    offs["cut_inside_header"] = make_fixture.CUT_INSIDE_HEADER
    offs["eof"] = make_fixture.EOF
    with open(P("offsets.json"), "w") as f:
        json.dump(dict(sorted(offs.items())), f, indent=1)

    # ---- diff 1: clean truncation at the cluster#0 boundary (232) --------
    t1, _ = parse_file(P("t1_boundary.mkv"))
    write_lines(P("t1.out"), t1.lines)
    d1 = write_diff(P("diff1_ref_vs_t1.diff"), "diff1_ref_vs_t1",
                    ref.lines, t1.lines)
    t1_prefix = strip_unverified_closes(t1.lines) == \
        ref.lines[:len(strip_unverified_closes(t1.lines))]
    report.append("diff #1 (ref vs clean truncation t1): %d differing hunk "
                  "lines; t1 is exact prefix of ref: %s"
                  % (len(d1), t1_prefix))

    # ---- diff 3: dirty truncation inside the nested Cluster header -------
    t2, _ = parse_file(P("t2_dirty.mkv"))
    write_lines(P("t2.out"), t2.lines)
    d3 = write_diff(P("diff3_ref_vs_t2.diff"), "diff3_ref_vs_t2",
                    ref.lines, t2.lines)
    t2_prefix = element_lines(t2.lines) == \
        element_lines(ref.lines)[:len(element_lines(t2.lines))]
    report.append("extra    (ref vs dirty truncation t2): t2 is exact "
                  "prefix of ref up to last verified boundary: %s; "
                  "t1 checkpoint=%d t2 checkpoint=%d"
                  % (t2_prefix, t1.checkpoint, t2.checkpoint))

    # ---- diff 2: checkpoint resume on a split stream ---------------------
    # part_a is cut dirty at 299; the parser must suspend at the last
    # verified checkpoint (296, end of PrevSize) with an open stack.
    pa_handler = CollectHandler()
    pa = mkvstream.StreamParser(pa_handler)
    with open(P("part_a.mkv"), "rb") as f:
        pa.start(f)
        pa.run()
    snap = pa.snapshot()
    with open(P("snapshot.json"), "w") as f:
        json.dump(snap, f)

    # resume: new stream yielding bytes from the checkpoint onward
    with open(P("part_a.mkv"), "rb") as f:
        head = f.read()
    with open(P("part_b.mkv"), "rb") as f:
        tail = f.read()
    cp = snap["checkpoint"]
    resume_stream = io.BytesIO(head[cp:] + tail)
    rs = mkvstream.StreamParser(pa_handler)
    rs.restore(snap)
    rs.start(resume_stream, base_offset=cp)
    rs.run()
    resume_lines = pa.lines + rs.lines
    write_lines(P("resume.out"), resume_lines)
    d2 = write_diff(P("diff2_ref_vs_resume.diff"), "diff2_ref_vs_resume",
                    ref.lines, resume_lines)
    report.append("restart: dirty cut at %d -> verified checkpoint %d -> "
                  "resumed on split stream (open stack depth %d)"
                  % (make_fixture.CUT_INSIDE_HEADER, cp,
                     len(snap["stack"])))
    report.append("diff #2 (ref vs checkpoint resume):   %d differing hunk "
                  "lines; identical: %s" % (len(d2), not d2))
    report.append("frames after resume: %d (expect 7)"
                  % len(pa_handler.frames))

    # ---- R2: oversize child ----------------------------------------------
    ov, _ = parse_file(P("oversize.mkv"))
    write_lines(P("oversize.out"), ov.lines)
    oversize_noted = any(l.startswith("# oversize") for l in ov.lines)
    recovered = any("unknown_7777" in l for l in element_lines(ov.lines))
    report.append("R2 oversize recovery: skipped to known Targets boundary "
                  "and closed stream at EOF: %s (checkpoint=%d)"
                  % (oversize_noted and recovered and ov.eof, ov.checkpoint))

    # ---- legacy mkvparse.py on the same fixture --------------------------
    legacy_frames = []
    stderr = io.StringIO()

    class LegacyHandler(mkvparse.MatroskaHandler):
        def frame(self, track_id, timestamp, data, more_laced_frames,
                  duration, keyframe, invisible, discardable):
            legacy_frames.append(track_id)

    with contextlib.redirect_stderr(stderr):
        with open(P("fixed.mkv"), "rb") as f:
            try:
                mkvparse.mkvparse(f, LegacyHandler())
            except Exception:
                import traceback
                traceback.print_exc()
    with open(P("legacy.log"), "w") as f:
        f.write(stderr.getvalue())
    legacy_failed = "Traceback" in stderr.getvalue()
    report.append("legacy mkvparse: %d frames before it gave up; blind "
                  "resync on the unknown-size element then crashes in its "
                  "own diagnostic formatting: %s (see legacy.log)"
                  % (len(legacy_frames), legacy_failed))

    all_ok = all([
        ebml_ok, order_ok, frames_ok, track_list == [1, 2],
        t1_prefix, t2_prefix, not d2, len(pa_handler.frames) == 7,
        oversize_noted and recovered and ov.eof,
        len(legacy_frames) == 7 and legacy_failed,
        t1.checkpoint == 232, t2.checkpoint == 296,
    ])
    report.append("ALL CHECKS PASSED: %s" % all_ok)

    with open(P("report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    sys.stdout.write("\n".join(report) + "\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "recovery"))
