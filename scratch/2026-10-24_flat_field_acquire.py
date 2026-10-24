
"""
Author: Mykhailo Vorobiov
Date: 2026-10-24

Flat-field acquisition for the FLIR camera at several gains. The exposure for
each gain is found automatically before any images are taken.

Method: Janesick, Photon Transfer (SPIE, 2007).
    Take many images (50+) of a uniformly lit field at each gain. Averaging
    them suppresses photon and read noise, leaving the fixed pattern noise
    (FPN) that is the same in every frame: pixel-to-pixel response differences
    (PRNU), vignetting and dust shadows.

    A later script removes the FPN with Janesick (2007), p. 111, eq. (8.2):

            S_iCOR = mu_FF * S_iRAW / S_iFF

    As written it has no dark term. Here it is applied to DARK-SUBTRACTED
    signals (our addition, not Janesick's notation):

            S_r = (S_i - D_i) * mu_0 / (S_0 - D_0)

    S_r  corrected pixel value
    S_i  raw image to correct,   D_i  per-pixel dark map for that image
    S_0  averaged flat (pixel),  D_0  averaged dark taken with the flat
    mu_0 mean of (S_0 - D_0)

    Why: a pixel reads R = O + s(1 + e) with offset O and PRNU deviation e.
    Only (R - O)/(F - O) = s/f cancels e exactly. With raw values a fraction
    O(f - s) / ((O + s)(O + f)) of the PRNU survives: about 20 % for a dim
    image (s ~ 3000 DN) against a bright flat (f ~ 45000 DN) with O ~ 1000 DN.
    The darks recorded here provide D_0.

How many frames?
    The averaged flat still carries noise of about 1 / sqrt(N_frames * N_e) per
    pixel (N_e = electrons per pixel per frame). It must be well below the PRNU
    it is meant to measure (typically ~1 %). At high gain the full well holds
    only a few hundred electrons, so 50 frames may not be enough. The script
    prints this residual noise for every gain; raise --frames if it is close
    to 1 %.

Before running:
  - Uniform illumination: a diffuser (opal glass) right in front of the lens,
    or an integrating sphere. A sheet of paper must be well OUT OF FOCUS,
    otherwise its fibre texture is recorded as if it were camera FPN.
  - Light source warmed up and on, lens uncovered.
  - Fill in LIGHT_SOURCE below; it goes into the provenance record.

Procedure
  1. Search the exposure for every gain (nothing is written yet).
  2. Show the exposures found and ask to continue.
  3. LIGHT phase: record --frames lit frames at each gain.
  4. One prompt to cover the lens.
  5. DARK phase: record the same number of frames at each gain and exposure.
  Every stack is saved the moment it is recorded, so a failure later in the
  run loses only the stack in progress.

Damaged frames
    On GigE a frame occasionally arrives with missing network packets. Up to
    MAX_DROPPED_FRAMES such frames per stack are thrown away and replaced by
    the next frame; flats and darks are equivalent frames, so nothing is lost.
    The number replaced is printed and stored with the stack.

Usage:
    python acquire/flat_acquire_gains.py <run_dir> --gains 0 10 20 34
    options: --frames 50  --target 0.7  --cam 1  --fov 11.5

Output, two files per gain, e.g. for 34 dB:
    <run_dir>/light_34.0dB.h5    /light  uint16 (frame, y, x), raw Mono16
    <run_dir>/dark_34.0dB.h5     /dark   uint16 (frame, y, x), raw Mono16
    attrs: gain and exposure actually used, target level, frames replaced
plus the usual provenance record.
"""

import argparse
import pathlib
import time

import numpy as np

from labtools.data_io import save_h5
from labtools.find_flat_exposure import find_flat_exposure
from labtools.flir import FLIRCamera
from labtools.provenance import stamp

LIGHT_SOURCE = ""      # e.g. "625 nm LED, 20 mA, opal diffuser at the lens"

# Rough time per frame on top of the exposure itself (readout and transfer).
# Only used for the estimate shown BEFORE recording; the progress bar measures
# the real rate while recording. ~5 ms suits the BFS-PGE-04S2M (about 290 fps).
READOUT_S_PER_FRAME = 0.005

# Damaged frames (missing GigE packets) replaced per stack before giving up.
# A handful is harmless; many more points to a network problem worth fixing.
MAX_DROPPED_FRAMES = 5


# --------------------------------------------------------------------------- #
# Progress and time estimates
# --------------------------------------------------------------------------- #

def format_duration(seconds: float) -> str:
    """42.0 -> '42 s',  185 -> '3 min 05 s'."""
    if seconds < 60:
        return f"{seconds:.0f} s"
    minutes, seconds = divmod(round(seconds), 60)
    return f"{minutes} min {seconds:02d} s"


class ProgressBar:
    """One-line progress bar with the time left, redrawn in place:

        [2/4] light 20.0 dB [###########---------]   25/50  3 s left

    The driver calls it after every frame, as progress(frames_done, total),
    when passed to get_frame_sequence(..., progress=bar).

    Time left = average time per frame x frames still to come. The average is
    measured from the FIRST frame on, not from the start, because the first
    frame also carries the stream start-up and would inflate the estimate.
    """
    WIDTH = 30          # characters in the bar

    def __init__(self, label: str):
        self.label = label
        self.start = time.perf_counter()
        self.first_frame_time = None

    def __call__(self, done: int, total: int) -> None:
        now = time.perf_counter()
        elapsed = now - self.start

        if done == 1:
            self.first_frame_time = now

        filled = int(self.WIDTH * done / total)
        bar = "#" * filled + "-" * (self.WIDTH - filled)

        if done == total:
            status = f"done in {format_duration(elapsed)}"
            line_end = "\n"                     # finished: move to the next line
        elif done == 1:
            status = "estimating..."            # one frame gives no rate yet
            line_end = ""
        else:
            seconds_per_frame = (now - self.first_frame_time) / (done - 1)
            time_left = seconds_per_frame * (total - done)
            status = f"{format_duration(time_left)} left"
            line_end = ""                       # stay on this line, redraw next time

        # "\r" returns to the start of the line, so each call overwrites the last.
        print(f"\r  {self.label} [{bar}] {done:4d}/{total}  {status:<16}",
              end=line_end, flush=True)


def estimate_duration(exposures_us, n_frames: int) -> float:
    """Rough seconds to record light AND dark stacks for all gains."""
    seconds_per_stack = 0.0
    for exposure_us in exposures_us:
        seconds_per_frame = exposure_us / 1e6 + READOUT_S_PER_FRAME
        seconds_per_stack += n_frames * seconds_per_frame
    return 2 * seconds_per_stack                # light phase + dark phase


# --------------------------------------------------------------------------- #
# Recording and saving
# --------------------------------------------------------------------------- #

def record_stack(cam: FLIRCamera, gain_db: float, exposure_us: float,
                 n_frames: int, label: str) -> dict:
    """Set gain and exposure, record n_frames in one stream with a progress bar.

    Returns a dict with the images, the gain and exposure actually used (the
    camera rounds both, so the values read back are the ones to store), and
    how many damaged frames were replaced.
    """
    cam.set_gain(gain_db)
    cam.set_exposure_time(exposure_us)
    progress_bar = ProgressBar(label)
    sequence = cam.get_frame_sequence(n_frames, progress=progress_bar,
                                      max_dropped=MAX_DROPPED_FRAMES)
    if sequence["dropped"] > 0:
        print(f"        replaced {sequence['dropped']} damaged frame(s)")

    return {
        "images": sequence["images"],
        "gain_used": cam.gain_value,
        "exposure_used": cam.exposure_value,
        "dropped": sequence["dropped"],
    }


def residual_flat_noise(light: np.ndarray) -> float:
    """Relative noise per pixel that will remain in the AVERAGE of this stack.

    Each pixel scatters from frame to frame by its standard deviation; averaging
    N frames shrinks that by sqrt(N). Dividing by the pixel's mean level gives
    the relative error of the averaged flat. The median over the central region
    is reported (central only, to keep memory use small). The offset is not
    subtracted here, so this slightly underestimates; fine as a guide.
    """
    n_frames, height, width = light.shape
    centre = light[:, height // 4: 3 * height // 4, width // 4: 3 * width // 4]
    centre = centre.astype(np.float64)

    pixel_mean = centre.mean(axis=0)
    pixel_std = centre.std(axis=0, ddof=1)
    relative_noise = pixel_std / np.sqrt(n_frames) / pixel_mean
    return float(np.median(relative_noise))


def save_stack(run_dir: pathlib.Path, kind: str, gain_requested: float,
               stack: dict, target: float) -> pathlib.Path:
    """Write one stack straight away, e.g. light_34.0dB.h5 holding /light.

    kind is "light" or "dark"; it names both the file and the dataset.
    """
    path = run_dir / f"{kind}_{gain_requested:04.1f}dB.h5"
    save_h5(path, {
        kind: stack["images"],
        "attrs": {
            "gain_db": stack["gain_used"],
            "gain_db_requested": gain_requested,
            "exposure_us": stack["exposure_used"],
            "dropped_frames_replaced": stack["dropped"],
            "target_level": target,
            "light_source": LIGHT_SOURCE,
            f"{kind}.axes": "frame, y, x",
            f"{kind}.units": "raw DN (Mono16)",
        },
    })
    return path


def main(run_dir: pathlib.Path, gains_db: list[float], n_frames: int = 50,
         target: float = 0.7, cam_index: int = 1, fov_mm: float = 11.5) -> None:

    with FLIRCamera(cam_index=cam_index, fov_horizontal=fov_mm) as cam:
        cam.disable_fps_limit()           # otherwise long exposures are refused

        # Only one stack is in memory at a time; each is saved when recorded.
        width, height = cam.resolution
        megabytes_per_stack = n_frames * width * height * 2 / 1e6
        print(f"{len(gains_db)} gains x {n_frames} frames x (light + dark), "
              f"{megabytes_per_stack:.0f} MB per stack")

        # 1. Find the exposures. Nothing is written to disk yet, so a failure
        #    here (light off, source too bright) leaves no half-finished run.
        input("LIGHT: switch the source on and uncover the lens, then press Enter ")
        print("\nStep 1/3: finding the exposure for each gain")
        exposures = find_flat_exposure(cam, gains_db, target=target)

        # 2. Show what was found, how long recording will take, and ask.
        print("\n   gain [dB]   exposure [us]")
        for gain, exposure in zip(gains_db, exposures):
            print(f"   {gain:9.1f}   {exposure:13.1f}")
        estimate = estimate_duration(exposures, n_frames)
        print(f"\nRecording will take roughly {format_duration(estimate)} "
              "(light and dark together, plus the time to cover the lens).")
        if input("Record flats with these settings? [y/N] ").strip().lower() != "y":
            print("Aborted, nothing written.")
            return

        # From here on the run is recorded. exist_ok=False: never overwrite a run.
        run_dir.mkdir(parents=True, exist_ok=False)
        stamp(run_dir, {
            "gains_db": gains_db,
            "exposures_us": exposures,
            "n_frames": n_frames,
            "target_level": target,
            "light_source": LIGHT_SOURCE,
            "camera": cam.get_metadata(),
        })

        # 3. LIGHT phase: every gain, one after the other, saved at once.
        print("\nStep 2/3: recording lit frames")
        light_phase_start = time.perf_counter()
        settings_used = []            # what the darks must match, one entry per gain
        n_gains = len(gains_db)

        for number, (gain, exposure) in enumerate(zip(gains_db, exposures), start=1):
            label = f"[{number}/{n_gains}] light {gain:5.1f} dB"
            light = record_stack(cam, gain, exposure, n_frames, label)
            path = save_stack(run_dir, "light", gain, light, target)

            noise = residual_flat_noise(light["images"])
            print(f"        averaged flat noise {100 * noise:.2f} % per pixel, saved {path.name}")
            settings_used.append({
                "gain_requested": gain,
                "gain_used": light["gain_used"],
                "exposure_used": light["exposure_used"],
            })
            del light                 # free the memory before the next stack

        # 4. One prompt for all darks. The dark phase uses the same settings,
        #    so it takes about as long as the light phase just did.
        light_phase_duration = time.perf_counter() - light_phase_start
        print(f"\nLight phase took {format_duration(light_phase_duration)}; "
              f"the dark phase will take about the same.")
        input("DARK: cover the lens completely, then press Enter ")

        # 5. DARK phase: same gain and exposure as each lit stack, saved at once.
        print("\nStep 3/3: recording dark frames")
        for number, entry in enumerate(settings_used, start=1):
            label = f"[{number}/{n_gains}] dark  {entry['gain_requested']:5.1f} dB"
            dark = record_stack(cam, entry["gain_requested"], entry["exposure_used"],
                                n_frames, label)

            # Dark and light must have identical settings, or the dark does not
            # describe the same offset. Fail loudly rather than save a mismatch.
            same_gain = np.isclose(dark["gain_used"], entry["gain_used"])
            same_exposure = np.isclose(dark["exposure_used"], entry["exposure_used"])
            if not (same_gain and same_exposure):
                raise RuntimeError(f"Dark settings differ from light at {entry['gain_requested']} dB")

            path = save_stack(run_dir, "dark", entry["gain_requested"], dark, target)
            print(f"        saved {path.name}")
            del dark

    print(f"\nDone: {run_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=pathlib.Path, help="new folder for this run")
    ap.add_argument("--gains", type=float, nargs="+", required=True,
                    help="gains [dB], e.g. --gains 0 10 20 34")
    ap.add_argument("--frames", type=int, default=50, help="frames per stack (default 50)")
    ap.add_argument("--target", type=float, default=0.7,
                    help="mean level of the frame centre, fraction of full scale (default 0.7)")
    ap.add_argument("--cam", type=int, default=1, help="camera index (default 1)")
    ap.add_argument("--fov", type=float, default=11.5, help="horizontal FOV [mm]")
    a = ap.parse_args()
    main(a.run_dir, a.gains, a.frames, a.target, a.cam, a.fov)