#!/usr/bin/env python3
"""Check an OSKAR simulation output for consistency with its settings file.

    python3 sim/lofar/check_output.py lofar_16s_8ch.ini lofar_16s_8ch.vis
    python3 sim/lofar/check_output.py lofar_16s_8ch.ini lofar_16s_8ch.ms

Works on either output. A Measurement Set needs python-casacore; a .vis file
needs only numpy, being read by oskar_vis.py in this directory.
Both paths check the same things: that the dimensions match what was asked
for, that the metadata is self-consistent, and that the visibilities are
physically plausible rather than merely present.

Exit status is 0 if every check passes, 1 otherwise, so it can be used in a
script.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oskar_vis import VisFile   # noqa: E402  (needs the path set above)

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return bool(ok)


def read_settings(path):
    """Minimal reader for the flat key=value settings used here."""
    s = {}
    section = ""
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
        elif "=" in line:
            k, v = line.split("=", 1)
            s[f"{section}/{k.strip()}"] = v.strip()
    return s


def expected(settings, telescope_dir):
    n_st = len([d for d in os.listdir(telescope_dir)
                if d.startswith("station")]) if os.path.isdir(telescope_dir) else 0
    exp = {
        "stations": n_st,
        "baselines": n_st * (n_st - 1) // 2,
        "channels": int(settings["observation/num_channels"]),
        "steps": int(settings["observation/num_time_steps"]),
        "freq_start": float(settings["observation/start_frequency_hz"]),
        "freq_inc": float(settings["observation/frequency_inc_hz"]),
        "ra": float(settings["observation/phase_centre_ra_deg"]),
        "dec": float(settings["observation/phase_centre_dec_deg"]),
        "dump": float(settings["interferometer/time_average_sec"]),
    }
    exp["rows"] = exp["baselines"] * exp["steps"]
    return exp


def plausible(amp, tag):
    """Visibility amplitudes should be finite, non-trivial and bounded."""
    ok = True
    ok &= check(f"{tag}: all values finite", np.isfinite(amp).all(),
                f"{int((~np.isfinite(amp)).sum())} non-finite")
    nz = np.count_nonzero(amp)
    ok &= check(f"{tag}: not all zero", nz > 0.5 * amp.size,
                f"{100.0 * nz / amp.size:.1f}% non-zero")
    peak, med = float(np.max(amp)), float(np.median(amp))
    ok &= check(f"{tag}: amplitudes bounded", 0 < peak < 1e6,
                f"peak {peak:.3f} Jy, median {med:.3g} Jy")
    # A sky of thousands of faint sources gives a strong zero-spacing-like
    # response on short baselines and much less on long ones; a constant
    # amplitude everywhere would mean something is wrong.
    ok &= check(f"{tag}: amplitudes vary across the data",
                float(np.std(amp)) > 1e-6 * max(peak, 1e-12),
                f"std {float(np.std(amp)):.3g}")
    return ok


def check_ms(path, exp):
    try:
        from casacore.tables import table
    except ImportError:
        print("  python-casacore not available: cannot check a Measurement Set")
        return False
    ok = True
    t = table(path, ack=False)
    ok &= check("MS opens", t.nrows() > 0, f"{t.nrows()} rows")
    ok &= check("row count matches settings", t.nrows() == exp["rows"],
                f"{t.nrows()} vs {exp['rows']} expected "
                f"({exp['baselines']} baselines x {exp['steps']} steps)")

    ant = table(os.path.join(path, "ANTENNA"), ack=False)
    ok &= check("antenna count", ant.nrows() == exp["stations"],
                f"{ant.nrows()} vs {exp['stations']}")

    spw = table(os.path.join(path, "SPECTRAL_WINDOW"), ack=False)
    nchan = int(spw.getcell("NUM_CHAN", 0))
    freqs = np.asarray(spw.getcell("CHAN_FREQ", 0), dtype=float)
    ok &= check("channel count", nchan == exp["channels"],
                f"{nchan} vs {exp['channels']}")
    ok &= check("first channel frequency",
                abs(freqs[0] - exp["freq_start"]) < 1.0,
                f"{freqs[0] / 1e6:.6f} MHz vs {exp['freq_start'] / 1e6:.6f}")
    if nchan > 1:
        ok &= check("channel spacing",
                    abs((freqs[1] - freqs[0]) - exp["freq_inc"]) < 1.0,
                    f"{(freqs[1] - freqs[0]) / 1e3:.3f} kHz vs "
                    f"{exp['freq_inc'] / 1e3:.3f}")

    fld = table(os.path.join(path, "FIELD"), ack=False)
    ra, dec = np.asarray(fld.getcell("PHASE_DIR", 0), dtype=float).ravel()[:2]
    ok &= check("phase centre",
                abs(math.degrees(ra) % 360 - exp["ra"] % 360) < 1e-3 and
                abs(math.degrees(dec) - exp["dec"]) < 1e-3,
                f"{math.degrees(ra) % 360:.4f}, {math.degrees(dec):.4f} deg")

    data = t.getcol("DATA")
    uvw = t.getcol("UVW")
    ok &= check("DATA shape", data.ndim == 3 and data.shape[1] == nchan,
                f"{data.shape} (rows, channels, polarisations)")
    ok &= check("4 polarisations", data.shape[2] == 4, f"{data.shape[2]}")
    ok &= plausible(np.abs(data), "DATA")

    ok &= check("UVW finite and non-zero", np.isfinite(uvw).all() and
                np.abs(uvw).max() > 0,
                f"max |u,v,w| {np.abs(uvw).max():.1f} m")
    # (u,v,w) is the baseline vector rotated into the phase-centre frame, so
    # its length must equal the distance between the two antennas. This checks
    # the antenna positions and the coordinates together.
    pos = table(os.path.join(path, "ANTENNA"), ack=False).getcol("POSITION")
    a1 = t.getcol("ANTENNA1")
    a2 = t.getcol("ANTENNA2")
    sep = np.linalg.norm(pos[a2] - pos[a1], axis=1)
    err = np.abs(sep - np.linalg.norm(uvw, axis=1))
    ok &= check("UVW length matches antenna separation", err.max() < 1.0,
                f"max discrepancy {err.max() * 1e3:.1f} mm over baselines to "
                f"{sep.max() / 1e3:.1f} km")
    ok &= check("baselines ordered antenna1 < antenna2", bool((a1 < a2).all()),
                f"{int((a1 >= a2).sum())} rows out of order")
    # Time span should match the requested observation length.
    times = t.getcol("TIME")
    span = float(times.max() - times.min())
    want = (exp["steps"] - 1) * exp["dump"]
    ok &= check("time span matches settings", abs(span - want) < exp["dump"],
                f"{span:.1f} s vs {want:.1f} s")
    if "FLAG" in t.colnames():
        flagged = float(np.mean(t.getcol("FLAG")))
        ok &= check("nothing flagged", flagged == 0.0,
                    f"{100.0 * flagged:.2f}% flagged")
    return ok


def check_vis(path, exp):
    ok = True
    hdr = VisFile(path)
    ok &= check("station count", hdr.num_stations == exp["stations"],
                f"{hdr.num_stations} vs {exp['stations']}")
    ok &= check("channel count", hdr.num_channels_total == exp["channels"],
                f"{hdr.num_channels_total} vs {exp['channels']}")
    ok &= check("time step count", hdr.num_times_total == exp["steps"],
                f"{hdr.num_times_total} vs {exp['steps']}")
    ok &= check("first channel frequency",
                abs(hdr.freq_start_hz - exp["freq_start"]) < 1.0,
                f"{hdr.freq_start_hz / 1e6:.6f} MHz")
    ok &= check("phase centre",
                abs(hdr.phase_centre_ra_deg % 360 - exp["ra"] % 360) < 1e-3 and
                abs(hdr.phase_centre_dec_deg - exp["dec"]) < 1e-3,
                f"{hdr.phase_centre_ra_deg:.4f}, "
                f"{hdr.phase_centre_dec_deg:.4f} deg")

    amps, uu = [], []
    for blk in hdr.blocks():
        amps.append(np.abs(blk.cross_correlations).astype(np.float64).ravel())
        uu.append(np.asarray(blk.baseline_uu_metres).ravel())
    amp = np.concatenate(amps)
    uu = np.concatenate(uu)
    expected_vals = exp["rows"] * exp["channels"] * 4
    ok &= check("visibility count", amp.size == expected_vals,
                f"{amp.size} vs {expected_vals} "
                f"(rows x channels x polarisations)")
    ok &= plausible(amp, "cross-correlations")
    ok &= check("baseline coordinates non-zero", np.isfinite(uu).all() and
                np.abs(uu).max() > 0, f"max |u| {np.abs(uu).max():.1f} m")
    return ok


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    ini, out = sys.argv[1], sys.argv[2]
    settings = read_settings(ini)
    tel = settings.get("telescope/input_directory", "telescope.tm")
    if not os.path.isabs(tel):
        tel = os.path.join(os.path.dirname(os.path.abspath(ini)), tel)
    exp = expected(settings, tel)
    print(f"settings : {ini}")
    print(f"output   : {out}")
    print(f"expected : {exp['stations']} stations, {exp['baselines']} baselines, "
          f"{exp['channels']} channels, {exp['steps']} time steps, "
          f"{exp['rows']} rows")
    print()
    if not os.path.exists(out):
        print(f"  [FAIL] output does not exist: {out}")
        if out.endswith(".ms"):
            print("         OSKAR only writes a Measurement Set when it was "
                  "built against casacore;")
            print("         otherwise it reports the file in its log but "
                  "writes nothing.")
        return 1
    ok = check_ms(out, exp) if out.rstrip("/").endswith(".ms") \
        else check_vis(out, exp)
    failed = [n for n, good, _ in CHECKS if not good]
    print()
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    print("RESULT:", "PASS" if ok and not failed else "FAIL")
    return 0 if (ok and not failed) else 1


if __name__ == "__main__":
    sys.exit(main())
