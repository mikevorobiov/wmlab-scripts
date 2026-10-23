"""
Photon transfer curve (PTC) acquisition for the FLIR camera, one gain per run.

Method: Janesick, Photon Transfer (SPIE, 2007), pair-difference variance.
At one gain, the script sweeps the exposure log-spaced from --min-exposure to
--max-exposure and records a frame pair at every point: first LIGHT (constant,
uniform illumination), then DARK (sensor covered).

Choose --max-exposure so the brightest point just saturates. The script first
grabs test frames at T, T/2 and T/4 and checks the ROI signal for linearity. That
catches both ADC clipping and full well filling below 4095 DN, so you can adjust
before committing to the run.

Before running:
  - Stable, flat illumination (LED + diffuser or integrating sphere), warmed up and on.
  - Fill in LIGHT_SOURCE below; it goes into the provenance record.

Usage:
    python acquire/ptc_acquire.py <run_dir> --gain 20 --max-exposure 2000
    options: --min-exposure 20  --n-exp 25  --frames 2  --cam 1  --fov 11.5  --preview

One run per gain, e.g.
    ...\\2026-10-23_run001_FLIR_ptc-0dB    --gain 0  --max-exposure 20000
    ...\\2026-10-23_run002_FLIR_ptc-20dB   --gain 20 --max-exposure 2000
Then analyze them together: python analyze/ptc_fit.py <out_dir> <run_dir> [<run_dir> ...]

Output: <run_dir>/ptc.h5
    /dark, /light   uint16  (1, n_exp, n_frames, y, x)   leading axis = gain (one per run)
    /gain_db        float64 (1, n_exp)   gain read back from the camera
    /exposure_us    float64 (1, n_exp)   exposure read back from the camera
"""

import argparse
import pathlib

import h5py
import numpy as np

from labtools.flir import FLIRCamera
from labtools.provenance import stamp
from labtools.sensor import FULL_SCALE, clipped_mask, to_native

LIGHT_SOURCE = ""      # e.g. "625 nm LED, 20 mA, opal diffuser"


def manual_light(on: bool) -> None:
    """Default light control: ask the operator. Replace with e.g. a LabJack output
    driving the LED for a hands-off run, but only if the setup is light-tight."""
    if not on:
        input("DARK: cover the sensor completely, then press Enter ")


def _roi(shape) -> tuple[slice, slice]:
    """Central half in each dimension, the same region analyze/ptc_fit.py uses."""
    ny, nx = shape
    return slice(ny // 4, 3 * ny // 4), slice(nx // 4, 3 * nx // 4)


LINEAR_RATIO = 1.8      # doubling the exposure must give at least 1.8x the ROI mean


def grab_roi(cam: FLIRCamera, exposure_us: float) -> tuple[np.ndarray, float, float]:
    """One frame at exposure_us -> (frame, ROI mean in native DN, ROI fraction ADC-clipped)."""
    cam.set_exposure_time(exposure_us)
    frame = cam.get_frame()
    roi = _roi(frame.shape)
    return frame, float(to_native(frame)[roi].mean()), float(clipped_mask(frame)[roi].mean())


def show_preview(frames: dict[str, np.ndarray], title: str) -> None:
    """Show test frames with the analysis ROI outlined and ADC-clipped pixels in red.
    Blocks until the window is closed."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, len(frames), figsize=(5.5 * len(frames), 4.5), squeeze=False)
    for ax, (label, frame) in zip(axes[0], frames.items()):
        im = ax.imshow(frame, cmap="gray", interpolation="nearest")
        sat = np.ma.masked_where(~clipped_mask(frame), np.ones(frame.shape))
        ax.imshow(sat, cmap="autumn", alpha=0.8, interpolation="nearest")
        ys, xs = _roi(frame.shape)
        ax.add_patch(Rectangle((xs.start - 0.5, ys.start - 0.5), xs.stop - xs.start,
                               ys.stop - ys.start, fill=False, ec="cyan", lw=1.2))
        ax.set_title(f"{label}\nROI mean {to_native(frame)[ys, xs].mean():.0f} DN "
                     f"of {FULL_SCALE} (native 12-bit)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, label="raw value")
    fig.suptitle(f"{title}   (cyan: analysis ROI, red: ADC clipped)  close window to continue",
                 fontsize=10)
    fig.tight_layout()
    plt.show()


def check_max_exposure(cam: FLIRCamera, max_exposure_us: float,
                       preview: bool = False) -> tuple[dict, str]:
    """Is max_exposure just past saturation? Ideal: saturated at T but not at T/2.

    Saturation of either kind, ADC clipping or full well filling below 4095 DN,
    makes the signal grow sublinearly with exposure. So the ROI mean is measured
    at T, T/2 and T/4:
      T/2 -> T   still linear  : not saturated at T           -> TOO SHORT
      T/4 -> T/2 already sublinear : saturated already at T/2 -> TOO LONG
      otherwise saturation sits between T/2 and T             -> OK
    """
    ts = (max_exposure_us, max_exposure_us / 2, max_exposure_us / 4)
    shots = [grab_roi(cam, t) for t in ts]
    m = [s[1] for s in shots]
    r_top, r_mid = m[0] / m[1], m[1] / m[2]
    if m[0] < 0.25 * FULL_SCALE or r_top > LINEAR_RATIO:
        verdict = "TOO SHORT: still linear at the top; increase --max-exposure."
    elif r_mid < LINEAR_RATIO:
        verdict = "TOO LONG: saturated already at half of it; decrease --max-exposure."
    else:
        verdict = "OK"
    info = {"roi_mean_dn": m, "exposures_us": list(ts),
            "ratio_T_over_half": r_top, "ratio_half_over_quarter": r_mid,
            "clipped_fraction": [s[2] for s in shots]}
    if preview:
        show_preview({f"{t:g} us ({100 * s[2]:.0f} % ROI clipped)": s[0] for t, s in zip(ts, shots)},
                     f"gain {cam.gain_value:.1f} dB: {verdict.split(':')[0]}")
    return info, verdict


def acquire_phase(cam: FLIRCamera, exposures, n_frames: int, label: str):
    """n_frames at every exposure, at the current gain. Leading gain axis of length 1."""
    nx, ny = cam.resolution
    images = np.empty((1, len(exposures), n_frames, ny, nx), dtype=np.uint16)
    gain_rb = np.empty((1, len(exposures)))
    exp_rb = np.empty_like(gain_rb)
    for j, ex in enumerate(exposures):
        cam.set_exposure_time(ex)
        images[0, j] = cam.get_frame_sequence(n_frames)["images"]
        gain_rb[0, j] = cam.gain_value          # actual values, as quantized by the camera
        exp_rb[0, j] = cam.exposure_value
    print(f"  {label} done")
    return images, gain_rb, exp_rb


def main(run_dir: pathlib.Path, gain_db: float, max_exposure_us: float,
         min_exposure_us: float = 20.0, n_exp: int = 25, n_frames: int = 2,
         cam_index: int = 1, fov_mm: float = 11.5, set_light=manual_light,
         confirm=True, preview=False) -> None:
    if n_frames < 2:
        raise ValueError("Need at least 2 frames per point (pair difference)")
    params = {
        "cam_index": cam_index,
        "fov_horizontal_mm": fov_mm,
        "frames_per_point": n_frames,
        "gain_db": gain_db,
        "exposure_us": np.geomspace(min_exposure_us, max_exposure_us, n_exp),
        "light_source": LIGHT_SOURCE,
    }

    with FLIRCamera(cam_index=cam_index, fov_horizontal=fov_mm,
                    exposure_value=min_exposure_us, gain_value=gain_db) as cam:
        cam.disable_fps_limit()     # otherwise the frame rate caps the max exposure

        set_light(True)
        check, verdict = check_max_exposure(cam, max_exposure_us, preview)
        m = check["roi_mean_dn"]
        print(f"{gain_db:g} dB, ROI mean at T, T/2, T/4 = {m[0]:.0f}, {m[1]:.0f}, {m[2]:.0f} DN "
              f"(ratios {check['ratio_T_over_half']:.2f}, {check['ratio_half_over_quarter']:.2f}; "
              f"linear = 2) -> {verdict}")
        if confirm and input("Start run? [y/N] ").strip().lower() != "y":
            print("Aborted, nothing written.")
            return

        run_dir.mkdir(parents=True, exist_ok=False)     # raw data is never overwritten
        stamp(run_dir, {**params, "max_exposure_check": check, "camera": cam.get_metadata()})

        light, g_l, e_l = acquire_phase(cam, params["exposure_us"], n_frames, "light")
        set_light(False)
        dark, g_d, e_d = acquire_phase(cam, params["exposure_us"], n_frames, "dark")

    if not (np.allclose(g_d, g_l) and np.allclose(e_d, e_l)):
        raise RuntimeError("Dark and light settings differ; camera did not apply them consistently")

    with h5py.File(run_dir / "ptc.h5", "w-") as f:
        f.create_dataset("dark", data=dark)
        f.create_dataset("light", data=light)
        f.create_dataset("gain_db", data=g_l)
        f.create_dataset("exposure_us", data=e_l)
        f.attrs["axes"] = "gain, exposure, frame, y, x"
    print(f"Saved {run_dir / 'ptc.h5'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("--gain", type=float, required=True, help="gain [dB]")
    ap.add_argument("--max-exposure", type=float, required=True, help="longest exposure [us]")
    ap.add_argument("--min-exposure", type=float, default=20.0, help="shortest exposure [us]")
    ap.add_argument("--n-exp", type=int, default=25, help="number of exposures")
    ap.add_argument("--frames", type=int, default=2, help="frames per point")
    ap.add_argument("--cam", type=int, default=1, help="camera index")
    ap.add_argument("--fov", type=float, default=11.5, help="horizontal FOV [mm]")
    ap.add_argument("--preview", action="store_true",
                    help="show the max-exposure test frames before asking to start")
    a = ap.parse_args()
    main(a.run_dir, a.gain, a.max_exposure, a.min_exposure, a.n_exp, a.frames, a.cam, a.fov,
         preview=a.preview)
