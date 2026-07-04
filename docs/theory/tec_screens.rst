.. _tec_screens:

***********************
Ionospheric TEC Screens
***********************

The ionosphere enters the measurement equation as an optional,
per-station, per-source phase term - the :math:`\mathbf{Z}` Jones matrix
(see :ref:`theory`) - driven by an external screen of total electron
content (TEC) values. This page describes how that screen is loaded,
and how the physics is applied.

The input screen
================

OSKAR does not generate the ionosphere model itself - you supply a FITS cube
whose pixel values are interpreted as a change in total electron content
(:math:`\Delta{\rm TEC}`) above the array. The cube's axes are ``XX``, ``YY``
and ``TIME``: a stack of two-dimensional screens that evolve over the
observation. (Any higher dimensions of the cube need to be of length 1.)
These cubes can be produced using
the `ARatmospy <https://github.com/shrieks/ARatmospy>`_ auto-regressive
atmosphere generator, which lets the pattern drift and evolve with time.
See the :ref:`example_ionosphere` example for a script that
generates a suitable cube.

The physics: TEC to phase
=========================

For each source seen from each station, OSKAR traces the line of sight to
where it pierces the screen at the configured height, reads the nearest
:math:`\Delta{\rm TEC}` pixel, and converts it to a phase that scales
inversely with frequency :math:`\nu` (in Hz):

.. math::

   {\rm phase}_{\rm rad} =
   \Delta{\rm TEC} \times \frac{-8.44797245 \times 10^9}{\nu}

The screen is then applied as a scalar phase on the identity matrix:

.. math::

   \mathbf{Z} =
   \exp\left\{i
   \left[-8.44797245 \times 10^9 \, \frac{\Delta{\rm TEC}}{\nu} \right]
   \right\}
   \left[
   \begin{array}{cc}
   1 & 0 \\
   0 & 1
   \end{array}
   \right]

Because the phase goes as :math:`1/\nu`, the effect is far stronger at low
frequencies.

Origin of the constant
----------------------

The coefficient :math:`-8.44797245 \times 10^9` is
the cold-plasma phase-delay constant. The ionosphere has a phase
refractive index :math:`n_{\rm ph} = \sqrt{1 - \nu_p^2/\nu^2}`, where the
plasma frequency satisfies :math:`\nu_p^2 = N_e e^2 / (4\pi^2\varepsilon_0
m_e)`. Integrating the excess phase :math:`(2\pi\nu/c)\int(n_{\rm ph}-1)
\,dl` along the line of sight, with :math:`{\rm TEC} = \int N_e\,dl`,
gives

.. math::

   \Delta\phi = -\frac{e^2}{4\pi\varepsilon_0 m_e c} \, \frac{\rm TEC}{\nu}
   = -k \, \frac{\rm TEC}{\nu},
   \qquad k = \frac{e^2}{4\pi\varepsilon_0 m_e c}
   \approx 8.44797 \times 10^{-7}.

Two conventions turn :math:`k` into the value in the code:

- **Sign:** negative because the plasma phase index is less than one, so
  the wave's phase advances relative to vacuum.
- **TEC units:** :math:`\Delta{\rm TEC}` is expressed in TEC units,
  with 1 TECU :math:`= 10^{16}` electrons m\ :sup:`-2`.
  Multiplying :math:`k` by :math:`10^{16}` gives :math:`8.44797245 \times 10^9`.

Pierce points
-------------

The world-to-pixel mapping places each source's pierce point using the
station's projected position and the source direction cosines::

    world_x = (station_u + s_l * screen_height_m) / pixel_size_m
    world_y = (station_v + s_m * screen_height_m) / pixel_size_m
    pix_x   = screen_num_pixels_x / 2 + ROUND(world_x)
    pix_y   = screen_num_pixels_y / 2 + ROUND(world_y)
    tec     = screen[pix_x + pix_y * screen_num_pixels_x]

Here ``station_u`` and ``station_v`` shift the sample point per station,
while the direction cosines ``s_l`` and ``s_m`` shift it per source.
The lookup is **nearest-pixel** (there is no interpolation), and pierce
points that fall outside the screen return zero :math:`\Delta{\rm TEC}` -
that is, no phase. Note also that the pierce offset uses
:math:`l \cdot H` rather than the exact :math:`(l/n) \cdot H`, a
small-angle approximation which progressively underestimates the
horizontal offset for sources far from the reference direction.

Screen geometry
---------------

Where the screen sits, and how it is oriented, depends on the beam
coordinate frame:

- **RADEC mode** (phase-tracking - the usual case): ``station_u``
  and ``station_v`` are the standard interferometric :math:`(u,v)` -
  the station position projected into the plane perpendicular to the
  phase-centre direction - and the :math:`(l,m)` passed to the kernel are
  direction cosines relative to the phase centre.
  The screen is therefore a tangent plane oriented perpendicular
  to the line of sight to the phase centre, co-moving with the field as
  it tracks.

- **AZEL mode** (fixed or drift-scan beams): :math:`(u,v)` are computed
  towards the zenith and reduce to horizontal East/North ground
  coordinates, and the direction cosines passed are the raw East/North
  components relative to zenith. Only in this mode is the screen the
  intuitive horizontal slab at the given height above the array,
  oriented East/North.

.. note::

    **Isoplanatic mode.** With ``telescope/isoplanatic_screen = true``,
    the source direction cosines are forced to zero, so every source at a
    given station samples the same pixel and receives an identical phase.

.. note::

    **Faraday rotation.** When not running in scalar mode, a full
    two-by-two rotation matrix is evaluated instead of just a scalar phase,
    using the geomagnetic field along the line of sight per ITU-R P.531-6:
    :math:`{\rm faraday\_angle} = 236 \cdot B_{\rm los} \cdot {\rm TEC}
    \cdot \nu_{\rm GHz}^{-2}`.

Time evolution
==============

The time resolution of the screen does not need to match the
simulation time steps. As the simulation steps through time, OSKAR
selects the nearest slice of the cube and reads a new two-dimensional
screen only when the slice index changes.

For example, a screen with 60-second slices used with 4-second
integrations reuses the same slice for about 15 consecutive
integrations, switching as the elapsed time crosses 30 s, 90 s, 150 s,
and so on. This is nearest-slice-in-time selection with no
interpolation - the temporal analogue of the nearest-pixel spatial
lookup. Points to note:

- **The interval comes from the FITS header by default.** With
  ``screen_time_interval_sec`` left at its default (``file``), OSKAR
  reads the TIME-axis increment (``CDELT3``) from the cube. Setting the
  key explicitly overrides the header.
- **Alignment is by elapsed time, not absolute time.** The index is
  computed from time elapsed since the *observation* start; slice 0 is
  assumed to coincide with the observation start, and the cube's own
  time reference (``CRVAL3``/``CRPIX3``) is ignored. The cube therefore
  needs enough slices to cover the observation.
- **If the interval is missing or non-positive** (``<= 0``), OSKAR
  instead advances one screen slice per simulation time step (a 1:1
  mapping), which is unlikely to be what you want unless the two were
  generated to match. If in doubt, set
  ``external_tec_screen/screen_time_interval_sec`` explicitly.
- Requests past the end of the cube are clamped to the last available
  slice.

Settings
========

All keys live under the ``telescope`` group.

.. list-table::
   :header-rows: 1
   :widths: 45 15 40

   * - Key
     - Default
     - Meaning
   * - ``ionosphere_screen_type``
     - ``None``
     - ``None`` or ``External``.
   * - ``external_tec_screen/input_fits_file``
     -
     - Path to the FITS cube.
   * - ``external_tec_screen/screen_height_km``
     - ``300``
     - Height of the screen, in kilometres.
   * - ``external_tec_screen/screen_pixel_size_m``
     - ``file``
     - Pixel scale in metres, else the FITS ``CDELT``.
   * - ``external_tec_screen/screen_time_interval_sec``
     - ``file``
     - Seconds between slices, else the FITS ``CDELT``.
   * - ``isoplanatic_screen``
     - ``false``
     - Use the same phase for all sources at each station.
