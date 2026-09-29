#!/usr/bin/env python3
"""Convert an OSKAR .vis file to a Measurement Set using python-casacore.

    python3 sim/lofar/vis_to_ms.py lofar_16s_8ch.vis telescope.tm out.ms

OSKAR writes a Measurement Set itself, but only when it was built against
casacore's C++ libraries. Where only python-casacore is available, this does
the same job from the .vis file afterwards.

The conventions match OSKAR's own writer (oskar/ms/src/oskar_ms_write.cpp):
baselines ordered with antenna1 < antenna2, station 1 outer; (u,v,w) copied
unchanged; TIME = (index + 0.5) x interval + start; weights and sigmas 1.

Requires python-casacore and numpy; the .vis file is read by oskar_vis.py in
this directory, so the OSKAR Python bindings are not needed. Visibilities are
streamed one block at a time, so an 8-hour observation does not need to fit
in memory.
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oskar_vis import VisFile   # noqa: E402  (needs the path set above)

# WGS84
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def geodetic_to_ecef(lon_deg, lat_deg, alt_m):
    lon, lat = math.radians(lon_deg), math.radians(lat_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt_m) * cos_lat * math.cos(lon)
    y = (n + alt_m) * cos_lat * math.sin(lon)
    z = ((1.0 - WGS84_E2) * n + alt_m) * sin_lat
    return np.array([x, y, z])


def enu_to_ecef(lon_deg, lat_deg, alt_m, east, north, up):
    """Station ENU offsets to absolute ECEF positions."""
    centre = geodetic_to_ecef(lon_deg, lat_deg, alt_m)
    lon, lat = math.radians(lon_deg), math.radians(lat_deg)
    sl, cl = math.sin(lon), math.cos(lon)
    sp, cp = math.sin(lat), math.cos(lat)
    # Columns: east, north, up expressed in ECEF.
    rot = np.array([[-sl, -sp * cl, cp * cl],
                    [cl, -sp * sl, cp * sl],
                    [0.0, cp, sp]])
    local = np.vstack([east, north, up])
    return (rot @ local).T + centre


def read_telescope(path):
    """Array centre and station ENU offsets from an OSKAR telescope model."""
    with open(os.path.join(path, "position.txt")) as f:
        parts = [p for p in f.readline().replace(",", " ").split() if p]
    lon, lat = float(parts[0]), float(parts[1])
    alt = float(parts[2]) if len(parts) > 2 else 0.0
    east, north, up = [], [], []
    with open(os.path.join(path, "layout.txt")) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            v = [float(x) for x in line.replace(",", " ").split()]
            east.append(v[0])
            north.append(v[1])
            up.append(v[2] if len(v) > 2 else 0.0)
    return lon, lat, alt, np.array(east), np.array(north), np.array(up)


def create_ms(name, positions, freqs, chan_width, ra_rad, dec_rad,
              time_range, npol, telescope="OSKAR"):
    from casacore.tables import (default_ms, maketabdesc,
                                 makearrcoldesc, table)

    n_ant, n_chan = len(positions), len(freqs)
    if os.path.exists(name):
        raise SystemExit(f"{name} already exists; remove it first")
    default_ms(name).close()
    ms = table(name, readonly=False, ack=False)
    # default_ms() provides every standard column except the visibilities
    # themselves; add DATA with a fixed (channel, polarisation) shape.
    ms.addcols(maketabdesc([makearrcoldesc(
        "DATA", 0.0 + 0.0j, ndim=2, shape=[n_chan, npol],
        valuetype="complex")]))

    ant = table(os.path.join(name, "ANTENNA"), readonly=False, ack=False)
    ant.addrows(n_ant)
    ant.putcol("NAME", np.array([f"s{i:03d}" for i in range(n_ant)]))
    ant.putcol("STATION", np.array([f"s{i:03d}" for i in range(n_ant)]))
    ant.putcol("TYPE", np.array(["GROUND-BASED"] * n_ant))
    ant.putcol("MOUNT", np.array(["ALT-AZ"] * n_ant))
    ant.putcol("POSITION", positions)
    ant.putcol("OFFSET", np.zeros((n_ant, 3)))
    ant.putcol("DISH_DIAMETER", np.full(n_ant, 35.0))
    ant.putcol("FLAG_ROW", np.zeros(n_ant, dtype=bool))
    ant.close()

    feed = table(os.path.join(name, "FEED"), readonly=False, ack=False)
    feed.addrows(n_ant)
    feed.putcol("ANTENNA_ID", np.arange(n_ant))
    feed.putcol("FEED_ID", np.zeros(n_ant, dtype=int))
    feed.putcol("SPECTRAL_WINDOW_ID", np.full(n_ant, -1))
    feed.putcol("TIME", np.full(n_ant, 0.5 * (time_range[0] + time_range[1])))
    feed.putcol("INTERVAL", np.full(n_ant, time_range[1] - time_range[0]))
    feed.putcol("NUM_RECEPTORS", np.full(n_ant, 2))
    feed.putcol("BEAM_ID", np.full(n_ant, -1))
    feed.putcol("BEAM_OFFSET", np.zeros((n_ant, 2, 2)))
    feed.putcol("POLARIZATION_TYPE", np.array([["X", "Y"]] * n_ant))
    feed.putcol("POL_RESPONSE",
                np.tile(np.eye(2, dtype=np.complex64), (n_ant, 1, 1)))
    feed.putcol("POSITION", np.zeros((n_ant, 3)))
    feed.putcol("RECEPTOR_ANGLE", np.zeros((n_ant, 2)))
    feed.close()

    spw = table(os.path.join(name, "SPECTRAL_WINDOW"), readonly=False, ack=False)
    spw.addrows(1)
    spw.putcell("NUM_CHAN", 0, n_chan)
    spw.putcell("NAME", 0, "OSKAR")
    spw.putcell("REF_FREQUENCY", 0, float(freqs[0]))
    spw.putcell("CHAN_FREQ", 0, np.asarray(freqs, dtype=float))
    spw.putcell("CHAN_WIDTH", 0, np.full(n_chan, chan_width))
    spw.putcell("EFFECTIVE_BW", 0, np.full(n_chan, chan_width))
    spw.putcell("RESOLUTION", 0, np.full(n_chan, chan_width))
    spw.putcell("TOTAL_BANDWIDTH", 0, float(n_chan * chan_width))
    spw.putcell("NET_SIDEBAND", 0, 1)
    spw.putcell("IF_CONV_CHAIN", 0, 0)
    spw.putcell("FREQ_GROUP", 0, 0)
    spw.putcell("FREQ_GROUP_NAME", 0, "")
    spw.putcell("MEAS_FREQ_REF", 0, 5)          # TOPO
    spw.putcell("FLAG_ROW", 0, False)
    spw.close()

    pol = table(os.path.join(name, "POLARIZATION"), readonly=False, ack=False)
    pol.addrows(1)
    # 9..12 are XX, XY, YX, YY in casacore's Stokes enumeration.
    corr = [9, 10, 11, 12] if npol == 4 else [9]
    prod = [[0, 0], [0, 1], [1, 0], [1, 1]] if npol == 4 else [[0, 0]]
    pol.putcell("NUM_CORR", 0, npol)
    pol.putcell("CORR_TYPE", 0, np.array(corr, dtype=int))
    pol.putcell("CORR_PRODUCT", 0, np.array(prod, dtype=int).T)
    pol.putcell("FLAG_ROW", 0, False)
    pol.close()

    dd = table(os.path.join(name, "DATA_DESCRIPTION"), readonly=False, ack=False)
    dd.addrows(1)
    dd.putcell("SPECTRAL_WINDOW_ID", 0, 0)
    dd.putcell("POLARIZATION_ID", 0, 0)
    dd.putcell("FLAG_ROW", 0, False)
    dd.close()

    direction = np.array([[ra_rad, dec_rad]])
    fld = table(os.path.join(name, "FIELD"), readonly=False, ack=False)
    fld.addrows(1)
    fld.putcell("NAME", 0, "field")
    fld.putcell("CODE", 0, "")
    fld.putcell("TIME", 0, time_range[0])
    fld.putcell("NUM_POLY", 0, 0)
    fld.putcell("SOURCE_ID", 0, 0)
    fld.putcell("DELAY_DIR", 0, direction)
    fld.putcell("PHASE_DIR", 0, direction)
    fld.putcell("REFERENCE_DIR", 0, direction)
    fld.putcell("FLAG_ROW", 0, False)
    fld.close()

    obs = table(os.path.join(name, "OBSERVATION"), readonly=False, ack=False)
    obs.addrows(1)
    obs.putcell("TELESCOPE_NAME", 0, telescope)
    obs.putcell("TIME_RANGE", 0, np.asarray(time_range, dtype=float))
    obs.putcell("OBSERVER", 0, "OSKAR")
    obs.putcell("PROJECT", 0, "OSKAR simulation")
    obs.putcell("RELEASE_DATE", 0, 0.0)
    obs.putcell("SCHEDULE_TYPE", 0, "")
    obs.putcell("FLAG_ROW", 0, False)
    obs.close()
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vis")
    ap.add_argument("telescope")
    ap.add_argument("ms")
    args = ap.parse_args()

    from casacore.tables import table  # noqa: F401  (checked early)

    lon, lat, alt, east, north, up = read_telescope(args.telescope)
    positions = enu_to_ecef(lon, lat, alt, east, north, up)

    hdr = VisFile(args.vis)
    n_ant = hdr.num_stations
    if len(positions) != n_ant:
        raise SystemExit(f"telescope model has {len(positions)} stations, "
                         f"the .vis file has {n_ant}")
    n_chan = hdr.num_channels_total
    n_time = hdr.num_times_total
    n_bl = n_ant * (n_ant - 1) // 2
    freqs = hdr.freq_start_hz + hdr.freq_inc_hz * np.arange(n_chan)
    chan_width = hdr.channel_bandwidth_hz or abs(hdr.freq_inc_hz) or 1.0
    interval = hdr.time_inc_sec
    exposure = hdr.time_average_sec or interval
    t0 = hdr.time_start_mjd_utc * 86400.0
    times = t0 + (np.arange(n_time) + 0.5) * interval
    npol = hdr.num_pols

    print(f"{n_ant} stations, {n_bl} baselines, {n_chan} channels, "
          f"{n_time} time steps, {npol} polarisations")
    print(f"{n_bl * n_time} rows to write")

    ms = create_ms(args.ms, positions, freqs, chan_width,
                   math.radians(hdr.phase_centre_ra_deg),
                   math.radians(hdr.phase_centre_dec_deg),
                   (times[0] - 0.5 * interval, times[-1] + 0.5 * interval),
                   npol)

    # Baseline order, matching OSKAR: antenna1 < antenna2, antenna1 outer.
    a1 = np.concatenate([np.full(n_ant - s - 1, s) for s in range(n_ant - 1)])
    a2 = np.concatenate([np.arange(s + 1, n_ant) for s in range(n_ant - 1)])

    row = 0
    for blk in hdr.blocks():
        t_start = blk.start_time_index
        nt = blk.num_times
        # OSKAR's layout is (time, channel, baseline, polarisation).
        vis = blk.cross_correlations
        uu = blk.baseline_uu_metres
        vv = blk.baseline_vv_metres
        ww = blk.baseline_ww_metres
        for t in range(nt):
            ms.addrows(n_bl)
            # MS wants (row, channel, polarisation).
            data = np.ascontiguousarray(
                vis[t].transpose(1, 0, 2).astype(np.complex64))
            ms.putcol("DATA", data, row, n_bl)
            ms.putcol("UVW", np.column_stack([uu[t], vv[t], ww[t]]), row, n_bl)
            ms.putcol("ANTENNA1", a1, row, n_bl)
            ms.putcol("ANTENNA2", a2, row, n_bl)
            ms.putcol("TIME", np.full(n_bl, times[t_start + t]), row, n_bl)
            ms.putcol("TIME_CENTROID", np.full(n_bl, times[t_start + t]),
                      row, n_bl)
            ms.putcol("INTERVAL", np.full(n_bl, interval), row, n_bl)
            ms.putcol("EXPOSURE", np.full(n_bl, exposure), row, n_bl)
            ms.putcol("WEIGHT", np.ones((n_bl, npol), dtype=np.float32),
                      row, n_bl)
            ms.putcol("SIGMA", np.ones((n_bl, npol), dtype=np.float32),
                      row, n_bl)
            ms.putcol("FLAG", np.zeros((n_bl, n_chan, npol), dtype=bool),
                      row, n_bl)
            ms.putcol("FLAG_ROW", np.zeros(n_bl, dtype=bool), row, n_bl)
            for col, val in (("DATA_DESC_ID", 0), ("FIELD_ID", 0),
                             ("SCAN_NUMBER", 1), ("ARRAY_ID", 0),
                             ("OBSERVATION_ID", 0), ("STATE_ID", -1),
                             ("FEED1", 0), ("FEED2", 0), ("PROCESSOR_ID", 0)):
                ms.putcol(col, np.full(n_bl, val), row, n_bl)
            row += n_bl
        print(f"  block {blk.index + 1}/{hdr.num_blocks}: {row} rows",
              flush=True)

    ms.flush()
    ms.close()
    print(f"wrote {args.ms} ({row} rows)")


if __name__ == "__main__":
    sys.exit(main())
