#!/usr/bin/env python3
"""Compare oskar_vis.py against OSKAR's own Python bindings on a .vis file.

    python3 sim/lofar/check_reader.py quick.vis [more.vis ...]

oskar_vis.py reads the .vis format directly, so nothing else in sim/lofar
needs the OSKAR Python bindings. Where the bindings do happen to be
installed, this checks the two readers agree exactly: every header value,
every visibility, every coordinate, in every block.

Exit status is 0 if they agree, 1 if they differ, and 2 if the bindings are
not available, in which case nothing was compared.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oskar_vis import VisFile   # noqa: E402  (needs the path set above)

HEADER_FIELDS = [
    "num_stations", "num_channels_total", "num_times_total", "num_blocks",
    "freq_start_hz", "freq_inc_hz", "channel_bandwidth_hz",
    "time_start_mjd_utc", "time_inc_sec", "time_average_sec",
    "phase_centre_ra_deg", "phase_centre_dec_deg",
]

failures = []


def same(name, mine, theirs):
    a, b = np.asarray(mine), np.asarray(theirs)
    ok = a.shape == b.shape and np.array_equal(a, b)
    if ok:
        print("  [OK ] %-34s %s" % (name, a.shape if a.ndim else a))
    else:
        failures.append(name)
        print("  [FAIL] %-33s %s vs %s" % (name, a.shape, b.shape))
        if a.shape == b.shape:
            d = np.abs(a.astype(np.complex128) - b.astype(np.complex128))
            print("         max absolute difference %.6g" % d.max())
    return ok


def compare(path):
    import oskar
    print(path)
    mine = VisFile(path)
    hdr, handle = oskar.VisHeader.read(path)
    for field in HEADER_FIELDS:
        same(field, getattr(mine, field), getattr(hdr, field))
    for blk in mine.blocks():
        ref = oskar.VisBlock.create_from_header(hdr)
        ref.read(hdr, handle, blk.index)
        tag = "block %d" % blk.index
        same(tag + " start_time_index", blk.start_time_index,
             ref.start_time_index)
        same(tag + " num_times", blk.num_times, ref.num_times)
        same(tag + " num_pols", mine.num_pols, ref.num_pols)
        same(tag + " cross-correlations", blk.cross_correlations,
             np.array(ref.cross_correlations(), copy=True))
        for axis, accessor in (("uu", ref.baseline_uu_metres),
                               ("vv", ref.baseline_vv_metres),
                               ("ww", ref.baseline_ww_metres)):
            same(tag + " baseline " + axis,
                 getattr(blk, "baseline_%s_metres" % axis),
                 np.array(accessor(), copy=True))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    try:
        import oskar                                       # noqa: F401
        from oskar import VisHeader                         # noqa: F401
    except ImportError:
        print("OSKAR Python bindings not installed, so there is nothing to")
        print("compare against. This is expected, and does not affect")
        print("vis_to_ms.py or check_output.py, which never use them.")
        return 2
    for path in sys.argv[1:]:
        compare(path)
    print()
    if failures:
        print("%d difference(s): %s" % (len(failures), ", ".join(failures)))
        print("RESULT: FAIL")
        return 1
    print("the two readers agree exactly")
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
