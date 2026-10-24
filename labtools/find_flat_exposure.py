"""
Find the exposure that puts a flat-field image at a chosen brightness.

    find_flat_exposure(cam, gains_db, target=0.5)

For each gain, the exposure is adjusted until the mean of the central region
sits at `target` (a fraction of full scale) with no pixels saturated. The
camera is left with the gain and exposure it had before.

How the search works
--------------------
The signal grows linearly with exposure:  level = offset + slope * exposure.
  1. Measure one frame and assume no offset: scale the exposure by
     target / level. This is the first guess.
  2. From then on, use the last two measurements to draw the straight line
     (both slope and offset are then known) and solve it for the target
     exposure (the secant method). For a linear camera this lands on the
     target in one or two steps.
  3. A saturated frame says nothing about the slope, so it is never used for
     the line: the exposure is cut by 4 and the search continues.
The search stops with a clear error if the light seems to be off, if the source
is too bright even at the shortest exposure, or if it does not converge.

Levels are in raw Mono16 units divided by 65535, so 0.5 means half of the
16-bit range whether or not the camera left-shifts its 12-bit values.

Status: SCRATCH. Run on the bench, check the reported levels, then move it
into labtools or acquire/flat_acquire.py.
"""

import numpy as np

from labtools.sensor import clipped_mask

RAW_FULL_SCALE = 65535       # Mono16
CLIPPED_LIMIT = 1e-4         # this fraction of clipped pixels is tolerated (hot pixels)
SHRINK_IF_SATURATED = 4.0    # exposure is divided by this after a saturated frame


def find_flat_exposure(cam, gains_db, target=0.5, tolerance=0.03, max_steps=12):
    """Exposure [us] that brings the flat to `target` of full scale, per gain.

    cam        an open FLIRCamera, light source on and stable
    gains_db   one gain (float) or several (list/array)
    target     wanted mean level of the central region, fraction of full scale
    tolerance  accepted distance from target, also a fraction of full scale
    max_steps  measurements allowed per gain before giving up

    Returns one exposure per gain, in the same form as gains_db:
    a float for a float, an array for a list or array.
    """
    single_gain = np.ndim(gains_db) == 0
    gain_list = np.atleast_1d(gains_db)

    # Remember the current settings so the camera can be restored at the end,
    # even if the search fails halfway.
    original_gain = cam.gain_value
    original_exposure = cam.exposure_value

    exposures = []
    try:
        start = cam.exposure_value
        previous_gain = None

        for gain in gain_list:
            # Gain in dB scales the signal by 10^(gain/20), so the exposure found
            # at the previous gain predicts a good starting point for this one.
            # It is only a first guess: the search corrects any error in it.
            if previous_gain is not None:
                start = exposures[-1] * 10 ** (-(gain - previous_gain) / 20)

            cam.set_gain(gain)
            print(f"Gain {gain:g} dB:")
            exposure = search_exposure(cam, start, target, tolerance, max_steps)

            exposures.append(exposure)
            previous_gain = gain
    finally:
        cam.set_gain(original_gain)
        cam.set_exposure_time(original_exposure)

    if single_gain:
        return exposures[0]
    return np.array(exposures)


def search_exposure(cam, start, target, tolerance, max_steps) -> float:
    """Adjust the exposure at the current gain until the level hits target,
    beginning at `start` microseconds."""
    shortest, longest = cam.exposure_range_us
    exposure = min(max(start, shortest), longest)   # keep the start inside the limits
    points = []                                      # (exposure, level) of unsaturated frames

    for step in range(1, max_steps + 1):
        # The camera rounds the exposure, so keep what was REQUESTED (to know
        # whether a limit was reached) apart from what was USED (for the line fit).
        requested = exposure
        used, level, clipped = measure(cam, requested)
        print(f"  step {step:2d}: {used:10.1f} us -> level {level:.3f}, "
              f"{100 * clipped:.3f} % clipped")

        # Saturated: no slope information. Back off and try again.
        if clipped > CLIPPED_LIMIT:
            if requested <= shortest:
                raise RuntimeError("Saturated even at the shortest exposure: "
                                   "dim the light source")
            exposure = max(requested / SHRINK_IF_SATURATED, shortest)
            continue

        # Close enough: done.
        if abs(level - target) <= tolerance:
            return used

        points.append((used, level))
        exposure = next_guess(points, target)

        # Keep the next guess inside what the camera allows.
        if exposure > longest:
            if requested >= longest:
                raise RuntimeError(f"Too dark even at the longest exposure ({longest:g} us): "
                                   "check that the light is on, then brighten it or raise the gain")
            exposure = longest
        exposure = max(exposure, shortest)

    raise RuntimeError(f"No exposure within ±{tolerance} of target {target} "
                       f"after {max_steps} steps; is the light source stable?")


def measure(cam, exposure) -> tuple[float, float, float]:
    """Take one frame. Returns (actual exposure, level, clipped fraction).

    level    mean of the central region / full scale (0 = black, 1 = full)
    clipped  fraction of ALL pixels at the top code. The whole frame is checked,
             not just the centre: a bright spot anywhere would spoil the flat.
    """
    cam.set_exposure_time(exposure)
    actual_exposure = cam.exposure_value          # the camera rounds the setting
    frame = cam.get_frame()

    height, width = frame.shape
    centre = frame[height // 4: 3 * height // 4, width // 4: 3 * width // 4]
    level = centre.mean() / RAW_FULL_SCALE
    clipped = clipped_mask(frame).mean()
    return actual_exposure, float(level), float(clipped)


def next_guess(points, target) -> float:
    """Exposure expected to give `target`, from the measurements so far."""
    last_exposure, last_level = points[-1]

    # Only one point: assume the line goes through zero.
    if len(points) == 1:
        if last_level <= 0:
            raise RuntimeError("Frame is completely black: is the light on?")
        return last_exposure * target / last_level

    # Two or more points: straight line through the last two.
    previous_exposure, previous_level = points[-2]
    slope = (last_level - previous_level) / (last_exposure - previous_exposure)
    if slope <= 0:
        raise RuntimeError("Brightness does not increase with exposure: "
                           "is the light on and the lens uncovered?")
    return last_exposure + (target - last_level) / slope


if __name__ == "__main__":
    # Bench check:  python scratch/2026-10-24_flat_exposure.py
    from labtools.flir import FLIRCamera

    with FLIRCamera(cam_index=1, exposure_value=1000, gain_value=0) as cam:
        cam.disable_fps_limit()           # otherwise long exposures are refused
        gains = [0, 10, 20, 34]
        exposures = find_flat_exposure(cam, gains)
        for gain, exposure in zip(gains, exposures):
            print(f"{gain:5.1f} dB -> {exposure:10.1f} us")