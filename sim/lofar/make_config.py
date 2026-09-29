#!/usr/bin/env python3
"""Generate a LOFAR-like OSKAR configuration: telescope model, sky model and
settings files.

    python3 sim/lofar/make_config.py [--outdir DIR] [--stations 48]
                                     [--elements 48] [--sources 5000]

Defaults follow the agreed setup: 48 stations, thousands of sources, 8 or 20
channels per subband, 2 s or 16 s integration over 8 hours. Four settings
files are written, from cheapest to most expensive, so a run can be validated
before committing to the full observation.

The station layout is representative, not the real LOFAR array: 48 stations in
a core/remote/distant distribution around the LOFAR reference position, each
with a grid of HBA-tile-like elements. Replace telescope.tm with the real
model when it is available; nothing else needs to change.
"""

import argparse
import math
import os
import random

# LOFAR reference position (approximate, near Exloo).
LON_DEG, LAT_DEG, ALT_M = 6.8689, 52.9088, 50.0

# HBA band. A subband is 195.3125 kHz; channels divide it.
SUBBAND_HZ = 195312.5
FREQ_START_HZ = 150.0e6

# Field centre, chosen to transit near the middle of the observation.
RA_DEG, DEC_DEG = 123.4, 52.9


def station_positions(n, rng):
    """Horizontal offsets (east, north) in metres from the array centre.

    Roughly LOFAR-shaped: half the stations in a dense core, the rest spread
    out to give long baselines.
    """
    groups = [(n // 2, 1500.0), (n - n // 2 - n // 6, 40000.0), (n // 6, 90000.0)]
    out = []
    for count, radius in groups:
        for _ in range(count):
            # sqrt() for uniform area density, so the core does not clump.
            r = radius * math.sqrt(rng.random())
            t = rng.random() * 2.0 * math.pi
            out.append((r * math.cos(t), r * math.sin(t)))
    return out[:n]


def element_positions(n, spacing=5.15):
    """Element offsets within a station, on a filled square grid.

    5.15 m matches the HBA tile pitch, so a 48-element station spans ~36 m,
    close to a real HBA station.
    """
    side = math.ceil(math.sqrt(n))
    half = (side - 1) / 2.0
    out = []
    for iy in range(side):
        for ix in range(side):
            if len(out) < n:
                out.append(((ix - half) * spacing, (iy - half) * spacing))
    return out


def write_telescope(path, n_stations, n_elements, rng):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "position.txt"), "w") as f:
        f.write(f"{LON_DEG} {LAT_DEG} {ALT_M}\n")
    stations = station_positions(n_stations, rng)
    with open(os.path.join(path, "layout.txt"), "w") as f:
        f.write("# Station positions: east, north (metres) from the centre\n")
        for east, north in stations:
            f.write(f"{east:.3f}, {north:.3f}\n")
    elements = element_positions(n_elements)
    for i in range(n_stations):
        sdir = os.path.join(path, f"station{i:03d}")
        os.makedirs(sdir, exist_ok=True)
        with open(os.path.join(sdir, "layout.txt"), "w") as f:
            f.write("# Element positions: east, north (metres) from the station centre\n")
            for east, north in elements:
                # A little jitter, so every station beam differs slightly.
                f.write(f"{east + rng.gauss(0, 0.02):.4f}, "
                        f"{north + rng.gauss(0, 0.02):.4f}\n")
    return len(stations), len(elements)


def write_sky(filename, n_sources, rng, radius_deg=5.0):
    """Point sources in a disc around the phase centre, power-law fluxes."""
    with open(filename, "w") as f:
        # Named-column header, so OSKAR does not fall back to the old
        # fixed-format reader (which works, but warns).
        f.write("Format = RaD, DecD, I, Q, U, V, ReferenceFrequency, "
                "SpectralIndex\n")
        for _ in range(n_sources):
            r = radius_deg * math.sqrt(rng.random())
            t = rng.random() * 2.0 * math.pi
            dec = DEC_DEG + r * math.sin(t)
            ra = RA_DEG + r * math.cos(t) / math.cos(math.radians(dec))
            # dN/dS ~ S^-1.6 over 10 mJy to 10 Jy.
            flux = 0.01 * (1.0 - rng.random()) ** (-1.0 / 0.6)
            flux = min(flux, 10.0)
            f.write(f"{ra:.6f}, {dec:.6f}, {flux:.6e}, 0, 0, 0, "
                    f"{FREQ_START_HZ:.1f}, -0.7\n")


SETTINGS = """[General]
app=oskar_sim_interferometer

[simulator]
double_precision={double}
use_gpus=true
cuda_device_ids=all
max_sources_per_chunk={chunk}

[sky]
oskar_sky_model/file={sky}

[telescope]
input_directory={telescope}
pol_mode=Full
station_type=Aperture array
aperture_array/array_pattern/enable=true
aperture_array/array_pattern/normalise=true
aperture_array/element_pattern/functional_type=Dipole
aperture_array/element_pattern/dipole_length=0.5
aperture_array/element_pattern/dipole_length_units=Wavelengths

[observation]
num_channels={channels}
start_frequency_hz={freq_start}
frequency_inc_hz={freq_inc}
phase_centre_ra_deg={ra}
phase_centre_dec_deg={dec}
num_time_steps={steps}
start_time_utc={start_utc}
length={length}

[interferometer]
oskar_vis_filename={vis}
ms_filename={ms}
channel_bandwidth_hz={chan_bw}
time_average_sec={dump}
max_time_samples_per_block=8
"""


def write_settings(path, name, telescope, sky, channels, dump_s, hours,
                   double=False, chunk=16384):
    steps = int(round(hours * 3600.0 / dump_s))
    chan_bw = SUBBAND_HZ / channels
    body = SETTINGS.format(
        double="true" if double else "false", chunk=chunk,
        sky=sky, telescope=telescope,
        channels=channels, freq_start=f"{FREQ_START_HZ:.1f}",
        freq_inc=f"{chan_bw:.4f}", chan_bw=f"{chan_bw:.4f}",
        ra=RA_DEG, dec=DEC_DEG, steps=steps,
        start_utc="01-01-2026 18:00:00.000",
        length=f"{int(hours * 3600)}.0",
        dump=dump_s, vis=f"{name}.vis", ms=f"{name}.ms")
    with open(os.path.join(path, f"{name}.ini"), "w") as f:
        f.write(body)
    baselines = None
    return steps, chan_bw, baselines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--stations", type=int, default=48)
    ap.add_argument("--elements", type=int, default=48)
    ap.add_argument("--sources", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tel = os.path.join(args.outdir, "telescope.tm")
    sky = os.path.join(args.outdir, "sky.osm")
    ns, ne = write_telescope(tel, args.stations, args.elements, rng)
    write_sky(sky, args.sources, rng)

    # Relative paths, so the directory can be copied anywhere.
    tel_rel, sky_rel = "telescope.tm", "sky.osm"
    runs = [
        # name          channels  dump(s)  hours  double
        ("quick",              8,      16,   0.25, False),
        # The same short run in double precision. Single precision carries a
        # few parts in a thousand at this array size (see below), which is far
        # too coarse to compare two GPU backends against each other; this is
        # the file to use for that.
        ("quick_double",       8,      16,   0.25, True),
        ("lofar_16s_8ch",      8,      16,   8.0,  False),
        ("lofar_2s_20ch",     20,       2,   8.0,  False),
    ]
    baselines = ns * (ns - 1) // 2
    print(f"{ns} stations ({baselines} baselines), {ne} elements per station, "
          f"{args.sources} sources")
    for name, ch, dump, hours, double in runs:
        steps, chan_bw, _ = write_settings(args.outdir, name, tel_rel, sky_rel,
                                           ch, dump, hours, double=double)
        vis = baselines * steps * ch
        word = 8 if double else 4
        print(f"  {name:16s} {ch:2d} ch, {dump:2d} s, {hours:5.2f} h -> "
              f"{steps:6d} steps, {vis / 1e6:8.1f} M visibilities, "
              f"~{vis * 4 * 2 * word / 1e9:6.2f} GB, "
              f"channel width {chan_bw / 1e3:.2f} kHz"
              f"{', double precision' if double else ''}")


if __name__ == "__main__":
    main()
