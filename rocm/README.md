# Running OSKAR on AMD GPUs (ROCm / HIP)

This tree adds a HIP backend to OSKAR 2.13.0, so it runs on AMD GPUs as well
as NVIDIA ones. The CUDA, OpenCL and CPU backends are unchanged and still
build exactly as before.

Tested on AMD Instinct MI210 (gfx90a) with ROCm 6.1.0 under RHEL 8.8.

---

## 1. Requirements

| | |
|---|---|
| ROCm | 6.1 or later, with `hipcc` and hipFFT |
| CMake | **3.21 minimum** (the HIP language support needs it) |
| Compiler | any C++11 host compiler; GCC 8.5 is known to work |
| Optional | HDF5 (element patterns), casacore C++ (Measurement Set output) |

If CMake is missing or too old, `bash rocm/bootstrap_cmake.sh` installs a
self-contained official build into `~/.local/oskar-cmake`. No root required.
It downloads from Kitware; on a node with no outbound network, fetch
`cmake-3.28.6-linux-x86_64.tar.gz` elsewhere, drop it in `rocm/`, and the
script will use it instead.

If ROCm is not on the default path, set `ROCM_PATH` before building, for
example `export ROCM_PATH=/opt/rocm-6.1.0`.

---

## 2. Build

The quickest route is the supplied script, which configures, builds, runs the
test suite, runs a simulation and compares it against reference data:

```bash
cd <source tree>
BACKEND=hip bash rocm/check_node.sh 2>&1 | tee build_report.txt
```

It writes into `build-hip/`. Useful variables: `JOBS` (parallel build jobs),
`GPU_ARCH` (defaults to whatever `rocminfo` reports), `BUILD_DIR`.

To configure by hand instead:

```bash
mkdir build-hip && cd build-hip
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DFIND_HIP=ON -DFIND_CUDA=OFF -DFIND_OPENCL=OFF \
         -DCMAKE_PREFIX_PATH=$ROCM_PATH \
         -DCMAKE_HIP_COMPILER=$ROCM_PATH/llvm/bin/clang++ \
         -DHIP_ARCH=gfx90a
make -j
```

`HIP_ARCH` accepts several architectures, e.g. `-DHIP_ARCH="gfx90a;gfx942"`.

HIP and CUDA are mutually exclusive in one build: both provide the same
`OSKAR_GPU` compute path, so `-DFIND_HIP=ON` suppresses the CUDA search.

### Checking the build

```bash
cd build-hip
./apps/oskar_system_info      # should list the AMD GPUs
ctest                         # unit tests
```

`telescope_test` fails when HDF5 is absent: three element-pattern tests assert
success rather than skipping. That failure is unrelated to the GPU backend.

### What a correct build looks like

Measured on six MI210s (gfx90a) with ROCm 6.1.0, so you have something to
compare against:

| | |
|---|---|
| `ctest` | 14 of 15 pass; only `telescope_test`, for the HDF5 reason above |
| Reference simulation, double | nrmse 1.93e-15 against the CUDA result |
| Reference simulation, single | nrmse 9.79e-07 against the CUDA result |
| LOFAR-scale run, double | nrmse 2.84e-14 against the CUDA result |

The reference data for the first two is in `rocm/golden/`, and
`rocm/check_node.sh` runs the whole list. The last figure is the 48-station,
5000-source, 15-minute configuration of section 3, compared between an MI210
and an NVIDIA GPU; see section 6 before comparing anything in single
precision.

---

## 3. Run a simulation

A LOFAR-like configuration is provided in `sim/lofar/`: 48 stations of 48
elements, 5000 sources, 150 MHz. Regenerate or resize it with:

```bash
python3 sim/lofar/make_config.py --stations 48 --elements 48 --sources 5000
```

Three settings files are written, cheapest first:

| File | Channels | Integration | Duration | Visibilities | Size |
|---|---|---|---|---|---|
| `quick.ini` | 8 | 16 s | 15 min | 0.5 M | 20 MB |
| `quick_double.ini` | 8 | 16 s | 15 min | 0.5 M | 32 MB |
| `lofar_16s_8ch.ini` | 8 | 16 s | 8 h | 16.2 M | 0.5 GB |
| `lofar_2s_20ch.ini` | 20 | 2 s | 8 h | 324.9 M | 10.4 GB |

`quick_double.ini` is the same short run in double precision. Use it, not
`quick.ini`, for any comparison between two machines or two GPU backends —
see "Comparing two backends" below.

Start with the quick one:

```bash
cd sim/lofar
HIP_VISIBLE_DEVICES=0 ../../build-hip/apps/oskar_sim_interferometer quick.ini
```

Pin the GPU with `HIP_VISIBLE_DEVICES` on a shared machine. OSKAR uses every
visible GPU by default, and will distribute work across all of them.

The station layout is representative rather than the real LOFAR array.
Replacing `sim/lofar/telescope.tm` with a real telescope model is enough;
nothing else needs to change.

---

## 4. Measurement Set output

OSKAR writes a Measurement Set itself **only if it was built against
casacore's C++ libraries**. Without them it writes just its own `.vis` file —
and note that its log still lists the `.ms` path as an output even though
nothing is written. `python-casacore` does not satisfy the build: it provides
the Python bindings, not the C++ headers CMake looks for.

Two options:

**a. Build against casacore** and OSKAR writes the MS directly. Add casacore's
prefix to `CMAKE_PREFIX_PATH`; `ms_filename` in the settings file then
produces a Measurement Set.

**b. Convert afterwards.** using `numpy` and `python-casacore`:

```bash
pip install python-casacore

python3 sim/lofar/vis_to_ms.py quick.vis sim/lofar/telescope.tm quick.ms
```

The `.vis` file is read by `sim/lofar/oskar_vis.py`, a reader of OSKAR's
binary format written against the specification in
`docs/binary_file/binary_file.rst`. So the conversion needs no compiler, no
OSKAR installation and no OSKAR Python bindings — useful on a node with an old
Python, since the bindings are a C extension that has to be built. Run it on
its own to summarise a file:

```bash
python3 sim/lofar/oskar_vis.py quick.vis
```

The converter follows the same conventions as OSKAR's own writer: baselines
ordered with antenna1 < antenna2, (u,v,w) copied unchanged,
`TIME = (index + 0.5) x interval + start`, unit weights. Visibilities are
streamed block by block, so an 8-hour observation does not need to fit in
memory.

A `.vis` file normally stores station (u,v,w) rather than baseline (u,v,w),
because there are far fewer of them; the reader forms the differences the way
`oskar_convert_station_uvw_to_baseline_uvw.c` does, station 2 minus station 1,
negated under the CASA phase convention. Its output was checked field by field
against OSKAR's own bindings on both a single- and a double-precision file:
every header value and every visibility, (u,v,w) and dimension array is
bit-identical.

That comparison is a script, so it can be repeated anywhere the bindings do
happen to be installed:

```bash
python3 sim/lofar/check_reader.py quick.vis
```

It exits 0 if the two readers agree, 1 if they differ, and 2 if the bindings
are absent — in which case nothing was compared, which is the normal case and
affects nothing else.

---

## 5. Validate the output

```bash
python3 sim/lofar/check_output.py quick.ini quick.vis     # .vis
python3 sim/lofar/check_output.py quick.ini quick.ms      # Measurement Set
```

It checks the output against its settings file: dimensions, start frequency
and channel spacing, phase centre, observation length, that no value is
non-finite and that amplitudes actually vary. For a Measurement Set it also
checks the ANTENNA, SPECTRAL_WINDOW, FIELD and POLARIZATION subtables, the
baseline ordering, the flags, and that each `|UVW|` equals the distance
between the two antennas — which verifies the antenna positions and the
coordinates together. Exit status is 0 on success, so it can be scripted.

The conversion and validation scripts were verified end to end on an NVIDIA
machine — simulate, convert, validate, then re-read the Measurement Set with
casacore's own TaQL — since both GPU backends produce the same `.vis` file. The
HIP build is verified against CUDA reference data separately, in section 2.

---

## 6. Comparing two backends

`apps/oskar_vis_compare` reports a normalised RMS difference between two
outputs, gated at 1e-11 for double precision and 1e-4 for single. Those gates
suit the small reference configuration in `rocm/golden/` (30 stations, 3
sources). **They are far too tight for the LOFAR-like configuration in single
precision**, and the reason is worth knowing before anyone reads a failure as
a porting bug.

Measured on the 48-station, 5000-source, 15-minute run, taking a
double-precision run as the reference:

| Comparison | nrmse |
|---|---|
| CUDA single vs CUDA double | 4.51e-03 |
| HIP single vs CUDA double | 4.48e-03 |
| HIP single vs CUDA single | 1.58e-03 |

Compare in double precision instead:
```bash
oskar_sim_interferometer quick_double.ini        # on each machine
oskar_vis_compare a/quick_double.vis b/quick_double.vis
```
---

## 7. Edited and created file compared to OSKAR current version

The kernels themselves are untouched. OSKAR writes each kernel once, in
`oskar/*/define_*.h`, against a neutral vocabulary that
`oskar/utility/oskar_kernel_macros.h` maps onto CUDA, OpenCL or plain C. The
port adds a fourth mapping, for HIP.

| File | Purpose |
|---|---|
| `oskar/utility/oskar_gpu.h` | Maps the CUDA runtime calls OSKAR uses onto HIP. |
| `oskar/utility/oskar_gpu_fft.h` | The same for cuFFT to hipFFT. |
| `oskar/utility/oskar_kernel_macros.h` | The HIP branch of the kernel vocabulary. |
| `oskar/oskar_global.h`, CMake files | `OSKAR_HAVE_GPU` ("some GPU backend") replaces `OSKAR_HAVE_CUDA` at call sites that were never CUDA-specific. |
| `apps/oskar_vis_compare` | Compares two outputs numerically; used to verify the port against CUDA. |
| `sim/lofar/` | A LOFAR-like configuration, plus the `.vis` reader, Measurement Set converter and output validation described above. |

Points worth knowing if you modify this:

- `hipcc` does not define `__CUDA_ARCH__`. OSKAR used it to decide whether
  cross-lane shuffles are available, and an undefined macro evaluates to 0 in
  `#if`, which selected a fallback where the reduction macro expands to
  nothing — silently wrong results. Capability is now expressed by explicit
  `OSKAR_GPU_HAS_*` flags.
- The correlation kernels are written in terms of 32-lane groups. That stays
  correct on a 64-lane wavefront, because the reduction's XOR masks never
  cross the 32-lane boundary, so each half reduces independently — which is
  what the code wants, each half handling a different baseline. It is however
  not optimal on AMD: one hardware wavefront then spans two baselines.
- Link `hip::host`, not `hip::device`. The latter adds `-xhip` for every
  language in the target, so ordinary C++ sources get handed to the host
  compiler as HIP.

### Known issue

On ROCm 6.1 with gfx90a, the first kernel in a process could read incorrect
values from a buffer that had just been filled by a host-to-device copy.
`oskar_mem_create()` and `oskar_mem_realloc()` therefore clear new device
allocations on HIP, which removes the failure. The cost is one `hipMemset`
per allocation, and allocations happen during setup rather than in the
compute loops.

The root cause was not identified. A standalone HIP program performing the same
allocate/copy/read sequence does not reproduce it.

Set `OSKAR_HIP_NO_CLEAR_ON_ALLOC=1` to disable the workaround, to check
whether a given ROCm version still needs it. `rocm/check_node.sh` runs that
comparison as its last step.

---

## 8. Development without an AMD GPU

`rocm/compile_check_local.sh` compiles the whole library for AMD on a machine
with no AMD GPU and no ROCm installation. It fetches the real ROCm headers
once, then compiles every host source with the system compiler and every
kernel file through to gfx90a assembly using a local clang that has the
amdgcn target. It takes about 15 seconds and catches compilation problems
before they reach a GPU node. It says nothing about runtime behaviour.
