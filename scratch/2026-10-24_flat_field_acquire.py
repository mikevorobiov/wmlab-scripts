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
  5. DARK phase: record the same number of frames at each gain and exposure,
     then save that gain's lit and dark stacks to flat_<gain>dB.h5.

Usage:
    python acquire/flat_acquire_gains.py <run_dir> --gains 0 10 20 34
    options: --frames 50  --target 0.7  --cam 1  --fov 11.5

Output, one file per gain, e.g. <run_dir>/flat_34.0dB.h5:
    /light, /dark      uint16 (frame, y, x), raw Mono16 values
    attrs              gain and exposure actually used, target level
plus the usual provenance record.
"""

import argparse
import pathlib

import numpy as np

from labtools.data_io import save_h5
from labtools.find_flat_exposure import find_flat_exposure
from labtools.flir import FLIRCamera
from labtools.provenance import stamp

LIGHT_SOURCE = "Barrina T5 LED lamps"      # e.g. "625 nm LED, 20 mA, opal diffuser at the lens"


def record_stack(cam: FLIRCamera, gain_db: float, exposure_us: float, n_frames: int):
    """Set gain and exposure, record n_frames in one stream.

    Returns (images, gain actually used, exposure actually used). The camera
    rounds both settings, so the values read back are the ones to store.
    """
    cam.set_gain(gain_db)
    cam.set_exposure_time(exposure_us)
    images = cam.get_frame_sequence(n_frames)["images"]
    return images, cam.gain_value, cam.exposure_value


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


def save_gain(run_dir: pathlib.Path, entry: dict, dark: np.ndarray, target: float) -> pathlib.Path:
    """Write one gain's lit and dark stacks, with the settings actually used."""
    path = run_dir / f"flat_{entry['gain_requested']:04.1f}dB.h5"      # e.g. flat_34.0dB.h5
    save_h5(path, {
        "light": entry["light"],
        "dark": dark,
        "attrs": {
            "gain_db": entry["gain_used"],
            "gain_db_requested": entry["gain_requested"],
            "exposure_us": entry["exposure_used"],
            "target_level": target,
            "light_source": LIGHT_SOURCE,
            "light.axes": "frame, y, x",
            "dark.axes": "frame, y, x",
            "light.units": "raw DN (Mono16)",
            "dark.units": "raw DN (Mono16)",
        },
    })
    return path


def main(run_dir: pathlib.Path, gains_db: list[float], n_frames: int = 50,
         target: float = 0.7, cam_index: int = 1, fov_mm: float = 11.5) -> None:

    with FLIRCamera(cam_index=cam_index, fov_horizontal=fov_mm) as cam:
        cam.disable_fps_limit()           # otherwise long exposures are refused

        # Lit and dark stacks of all gains are held in memory until the dark
        # phase is done. Say how much that is before starting.
        width, height = cam.resolution
        megabytes = len(gains_db) * 2 * n_frames * width * height * 2 / 1e6
        print(f"{len(gains_db)} gains x {n_frames} frames x (light + dark): "
              f"about {megabytes:.0f} MB in memory")

        # 1. Find the exposures. Nothing is written to disk yet, so a failure
        #    here (light off, source too bright) leaves no half-finished run.
        input("LIGHT: switch the source on and uncover the lens, then press Enter ")
        exposures = find_flat_exposure(cam, gains_db, target=target)

        # 2. Show what was found and ask before recording.
        print("\n   gain [dB]   exposure [us]")
        for gain, exposure in zip(gains_db, exposures):
            print(f"   {gain:9.1f}   {exposure:13.1f}")
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

        # 3. LIGHT phase: every gain, one after the other.
        recorded = []
        for gain, exposure in zip(gains_db, exposures):
            light, gain_used, exposure_used = record_stack(cam, gain, exposure, n_frames)
            noise = residual_flat_noise(light)
            print(f"  light {gain:5.1f} dB: {n_frames} frames, "
                  f"averaged flat noise {100 * noise:.2f} % per pixel")
            recorded.append({
                "gain_requested": gain,
                "gain_used": gain_used,
                "exposure_used": exposure_used,
                "light": light,
            })

        # 4. One prompt for all darks.
        input("\nDARK: cover the lens completely, then press Enter ")

        # 5. DARK phase: same gain and exposure as each lit stack, then save.
        for entry in recorded:
            dark, gain_used, exposure_used = record_stack(
                cam, entry["gain_requested"], entry["exposure_used"], n_frames)

            # Dark and light must have identical settings, or the dark does not
            # describe the same offset. Fail loudly rather than save a mismatch.
            same_gain = np.isclose(gain_used, entry["gain_used"])
            same_exposure = np.isclose(exposure_used, entry["exposure_used"])
            if not (same_gain and same_exposure):
                raise RuntimeError(f"Dark settings differ from light at {entry['gain_requested']} dB")

            path = save_gain(run_dir, entry, dark, target)
            print(f"  dark  {entry['gain_requested']:5.1f} dB: saved {path.name}")

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