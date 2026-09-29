#!/usr/bin/env python3
"""Read an OSKAR .vis file with nothing but numpy.

    from oskar_vis import VisFile
    v = VisFile("lofar_16s_8ch.vis")
    print(v.num_stations, v.num_channels_total, v.num_times_total)
    for blk in v.blocks():
        blk.cross_correlations   # (time, channel, baseline, polarisation)

OSKAR ships Python bindings that do this, but they are a C extension: they
need the library installed, Python development headers and a compiler, which
is a lot of machinery to read a documented file format. The format is
specified in docs/binary_file/binary_file.rst; this implements the part of it
that visibility files use.

A file is a 64-byte header followed by chunks. Each chunk is a 20-byte tag
giving a group id, a tag id, an index and the block length, then the payload,
then an optional CRC. Chunks may appear in any order, so the tags are indexed
first and read on demand.
"""

import os
import struct

import numpy as np

# Payload type codes are a bitfield: char 1, int 2, single 4, double 8,
# complex 32, matrix 64. So 40 is complex double, 104 complex double matrix.
CHAR, INT, SINGLE, DOUBLE, COMPLEX, MATRIX = 1, 2, 4, 8, 32, 64

# Visibility header tags (group 11) that this reader exposes.
HDR = {
    "telescope_path": 1, "tags_per_block": 2, "auto_present": 3,
    "cross_present": 4, "amp_type": 5, "coord_precision": 6,
    "max_times_per_block": 7, "num_times_total": 8,
    "max_channels_per_block": 9, "num_channels_total": 10,
    "num_stations": 11, "pol_type": 12, "casa_phase_convention": 13,
    "phase_centre_type": 21, "phase_centre": 22, "freq_start_hz": 23,
    "freq_inc_hz": 24, "channel_bandwidth_hz": 25, "time_start_mjd_utc": 26,
    "time_inc_sec": 27, "time_average_sec": 28, "telescope_lon_deg": 29,
    "telescope_lat_deg": 30, "telescope_alt_m": 31,
    "station_x": 32, "station_y": 33, "station_z": 34,
}

GROUP_HEADER, GROUP_BLOCK = 11, 12
BLOCK_DIMS, BLOCK_AUTO, BLOCK_CROSS = 1, 2, 3
BLOCK_UU, BLOCK_VV, BLOCK_WW = 4, 5, 6
BLOCK_STATION_U, BLOCK_STATION_V, BLOCK_STATION_W = 7, 8, 9


class _Chunk(object):
    __slots__ = ("offset", "length", "data_type", "elem_size", "big_endian")

    def __init__(self, offset, length, data_type, elem_size, big_endian):
        self.offset = offset
        self.length = length
        self.data_type = data_type
        self.elem_size = elem_size
        self.big_endian = big_endian


class VisBlock(object):
    """One visibility block: a range of time samples and channels."""

    def __init__(self, index, dims, cross, auto, uu, vv, ww,
                 station_uvw=None):
        (self.start_time_index, self.start_channel_index, self.num_times,
         self.num_channels, self.num_baselines, self.num_stations) = dims
        self.index = index
        self.cross_correlations = cross
        self.auto_correlations = auto
        self.baseline_uu_metres = uu
        self.baseline_vv_metres = vv
        self.baseline_ww_metres = ww
        self.station_uvw_metres = station_uvw


class VisFile(object):
    def __init__(self, path):
        self.path = path
        self._f = open(path, "rb")
        magic = self._f.read(9)
        if magic[:8] != b"OSKARBIN":
            raise ValueError("%s is not an OSKAR binary file" % path)
        self.format_version = struct.unpack("<B", self._f.read(1))[0]
        self._f.seek(64)
        self._index = {}
        self._scan()
        if (GROUP_HEADER, HDR["num_stations"], 0) not in self._index:
            raise ValueError("%s has no visibility header; is it a .vis file?"
                             % path)
        for name, tag in HDR.items():
            setattr(self, name, self._value(GROUP_HEADER, tag))
        self.phase_centre_ra_deg = float(self.phase_centre[0])
        self.phase_centre_dec_deg = float(self.phase_centre[1])
        # The polarisation dimension is implicit in the amplitude type: the
        # matrix flag means four polarisations, otherwise one.
        self.num_pols = 4 if (self.amp_type & MATRIX) else 1
        self.num_baselines = self.num_stations * (self.num_stations - 1) // 2
        self.num_blocks = 1 + max(
            i for (g, t, i) in self._index
            if g == GROUP_BLOCK and t == BLOCK_DIMS)

    # -- file structure ----------------------------------------------------

    def _scan(self):
        """Build the tag index by walking the chunks."""
        size = os.path.getsize(self.path)
        while True:
            pos = self._f.tell()
            if pos + 20 > size:
                break
            tag = self._f.read(20)
            if len(tag) < 20 or tag[0:1] != b"T" or tag[2:3] != b"G":
                break
            elem_size = tag[3]
            flags = tag[4]
            data_type = tag[5]
            group, tag_id = tag[6], tag[7]
            index = struct.unpack("<i", tag[8:12])[0]
            block_size = struct.unpack("<q", tag[12:20])[0]
            extended = bool(flags & 0x80)
            has_crc = bool(flags & 0x40)
            big_endian = bool(flags & 0x20)
            payload = pos + 20
            length = block_size
            if extended:
                # Group and tag names are written before the payload; this
                # reader only needs the standard tags, so skip the chunk.
                payload += group + tag_id
                length -= group + tag_id
            if has_crc:
                length -= 4
            if not extended:
                self._index[(group, tag_id, index)] = _Chunk(
                    payload, length, data_type, elem_size, big_endian)
            self._f.seek(pos + 20 + block_size)

    def _dtype(self, data_type, elem_size, big_endian):
        order = ">" if big_endian else "<"
        base = data_type & 0x0F
        if base == CHAR:
            return np.dtype(order + "i1")
        if base == INT:
            return np.dtype(order + "i4")
        if base == SINGLE:
            fmt = "c8" if (data_type & COMPLEX) else "f4"
        elif base == DOUBLE:
            fmt = "c16" if (data_type & COMPLEX) else "f8"
        else:
            raise ValueError("unsupported payload type %d" % data_type)
        return np.dtype(order + fmt)

    def _read(self, group, tag, index=0, required=True):
        """The payload of one chunk, as a flat numpy array (or None)."""
        chunk = self._index.get((group, tag, index))
        if chunk is None:
            if required:
                raise KeyError("tag (%d, %d, %d) not in %s"
                               % (group, tag, index, self.path))
            return None
        dtype = self._dtype(chunk.data_type, chunk.elem_size, chunk.big_endian)
        self._f.seek(chunk.offset)
        raw = self._f.read(chunk.length)
        if chunk.data_type & CHAR:
            return raw.split(b"\0", 1)[0].decode("utf-8", "replace")
        return np.frombuffer(raw, dtype=dtype)

    def _value(self, group, tag, index=0):
        """A scalar, a string, or a small array, whichever the tag holds."""
        v = self._read(group, tag, index, required=False)
        if v is None or isinstance(v, str):
            return v
        return v[0].item() if v.size == 1 else v

    # -- data --------------------------------------------------------------

    def _baseline_indices(self, n):
        """Baseline order used throughout OSKAR: station 1 outer, s1 < s2."""
        a1 = np.concatenate([np.full(n - s - 1, s) for s in range(n - 1)])
        a2 = np.concatenate([np.arange(s + 1, n) for s in range(n - 1)])
        return a1, a2

    def blocks(self):
        """Yield each visibility block in order, reading it on demand."""
        a1, a2 = self._baseline_indices(self.num_stations)
        for b in range(self.num_blocks):
            dims = self._read(GROUP_BLOCK, BLOCK_DIMS, b)
            nt, nch, nbl, nst = (int(dims[2]), int(dims[3]),
                                 int(dims[4]), int(dims[5]))
            npol = self.num_pols
            # Dimension order is fixed: time slowest, then channel, then
            # baseline, with polarisation fastest.
            cross = self._read(GROUP_BLOCK, BLOCK_CROSS, b, required=False)
            if cross is not None:
                cross = cross.reshape(nt, nch, nbl, npol)
            auto = self._read(GROUP_BLOCK, BLOCK_AUTO, b, required=False)
            if auto is not None:
                auto = auto.reshape(nt, nch, nst, npol)
            uvw = []
            for tag in (BLOCK_UU, BLOCK_VV, BLOCK_WW):
                c = self._read(GROUP_BLOCK, tag, b, required=False)
                uvw.append(c.reshape(nt, nbl) if c is not None else None)
            # A file normally stores station coordinates rather than baseline
            # ones, since there are far fewer of them, and leaves the caller
            # to form the differences. Do that here, the way OSKAR does
            # (oskar_convert_station_uvw_to_baseline_uvw.c): the baseline
            # vector is station 2 minus station 1, negated under the CASA
            # phase convention.
            station_uvw = []
            for tag in (BLOCK_STATION_U, BLOCK_STATION_V, BLOCK_STATION_W):
                c = self._read(GROUP_BLOCK, tag, b, required=False)
                station_uvw.append(c.reshape(nt, nst)
                                   if c is not None else None)
            if uvw[0] is None and station_uvw[0] is not None:
                factor = -1.0 if self.casa_phase_convention else 1.0
                uvw = [factor * (c[:, a2] - c[:, a1]) for c in station_uvw]
            yield VisBlock(b, [int(d) for d in dims], cross, auto,
                           uvw[0], uvw[1], uvw[2], station_uvw)

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main():
    """Print a summary of a .vis file, as a quick check that it parses."""
    import sys
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    v = VisFile(sys.argv[1])
    print("file             : %s (format version %d)"
          % (v.path, v.format_version))
    print("stations         : %d (%d baselines)"
          % (v.num_stations, v.num_baselines))
    print("channels / times : %d / %d in %d block(s)"
          % (v.num_channels_total, v.num_times_total, v.num_blocks))
    print("polarisations    : %d (amp type %d, pol type %d)"
          % (v.num_pols, v.amp_type, v.pol_type))
    print("frequency        : %.6f MHz + n x %.4f kHz"
          % (v.freq_start_hz / 1e6, v.freq_inc_hz / 1e3))
    print("phase centre     : %.4f, %.4f deg"
          % (v.phase_centre_ra_deg, v.phase_centre_dec_deg))
    print("start / interval : MJD %.8f / %.3f s"
          % (v.time_start_mjd_utc, v.time_inc_sec))
    total = 0
    peak = 0.0
    for blk in v.blocks():
        total += blk.cross_correlations.size
        peak = max(peak, float(np.abs(blk.cross_correlations).max()))
    print("visibilities     : %d values, peak |V| %.6g Jy" % (total, peak))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
