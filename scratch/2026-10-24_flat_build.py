"""
Build flat-field correction images from a run of flat_acquire_gains.py.

Input:  a raw run folder holding, per gain, light_<gain>dB.h5 and dark_<gain>dB.h5
Output: one flat_<gain>dB.h5 per gain in <out_dir>, plus flat_overview.png

What is computed, per gain
--------------------------
    D0    = average dark frame                       (per pixel, raw DN)
    S0    = average lit frame - D0                   (flat signal, raw DN)
    mu0   = mean of S0 over the good pixels
    flat  = S0 / mu0                                 (mean 1)

A measurement image S_i, with its own dark map D_i, is then corrected as
(Janesick 2007, eq. 8.2, applied to dark-subtracted signals):

    corrected = (S_i - D_i) / flat

Also saved:
    flat_error   relative standard error of each flat pixel: the noise left
                 after averaging. Propagate it like any other uncertainty.
    bad_pixels   pixels the flat cannot be trusted for (see below)

Bad pixels
    hot     average dark level far above the rest
    noisy   frame-to-frame scatter in the darks far above the rest
    odd     response far from its NEIGHBOURS: flat divided by a median-
            smoothed copy of itself. Comparing with neighbours rather than the
            whole frame keeps vignetting from being flagged. The edges of dust
            shadows can land here too; that is on the safe side.
A pixel can fall into more than one group (a blinking pixel also raises its
average dark level, so it is usually "hot" as well); n_bad counts it once.

How "far" is decided: a pixel is flagged only if its deviation from the median
is BOTH
  - more than Z_LIMIT robust sigmas, where robust sigma = 1.4826 x MAD
    (median absolute deviation, which the outliers themselves cannot inflate),
    but never less than what the ADC can resolve, and
  - more than a physical minimum, a few ADC codes (hot, noisy tests).

Why the floor: when the read noise is below one ADC code (low gain), most dark
pixels read the same code in every frame. Their scatter is then 0 and the MAD
is close to 0, so without a floor almost every pixel would look like an
outlier. One code is 16 raw DN when the camera left-shifts its 12-bit values.
A frame cannot resolve better than step/sqrt(12); an average of N frames not
better than step/sqrt(12 N).

Why the MAD is valid for these three quantities (unlike per-pixel variances
from only two frames, which are strongly skewed): an average over ~100 frames,
a standard deviation over ~100 frames, and a ratio near 1 are all close to
Gaussian across pixels, apart from the outliers being looked for.

PRNU (pixel response non-uniformity) is reported from the same high-pass
image: the spread of flat / smoothed flat over the good pixels. Smooth
illumination gradients do not contribute to it.

Usage:
    python scratch/2026-10-24_flat_build.py <out_dir> <raw_run_dir>
    options: --smooth 15  --z 5
"""

import argparse
import pathlib

import numpy as np
from scipy.ndimage import median_filter

from labtools.data_io import load, save_h5
from labtools.provenance import file_ref, stamp
from labtools.sensor import lsb_shift

SMOOTH_PIXELS = 15      # median-filter size for the "neighbours" comparison
Z_LIMIT = 5.0           # robust z-score above which a pixel is called bad
HOT_MIN_CODES = 3       # a hot pixel's dark level is at least this many codes high
NOISY_MIN_CODES = 2     # a noisy pixel's dark scatter is at least this many codes


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #

def robust_sigma(values: np.ndarray, resolution: float) -> float:
    """Standard deviation estimated from the MAD, but never below `resolution`.

    1.4826 * MAD equals the standard deviation for Gaussian data, and a few
    extreme values (the very pixels we look for) barely change it. The floor
    stops quantized data, where the MAD can be ~0, from making every small
    difference look enormous.
    """
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return max(1.4826 * mad, resolution)


def flag_high(values: np.ndarray, resolution: float, minimum: float, z_limit: float) -> np.ndarray:
    """True where a value is above the median by more than z_limit robust
    sigmas AND by more than `minimum`."""
    excess = values - np.median(values)
    sigma = robust_sigma(values, resolution)
    is_statistical_outlier = excess > z_limit * sigma
    is_large_enough = excess > minimum
    return is_statistical_outlier & is_large_enough


def flag_either_side(values: np.ndarray, resolution: float, z_limit: float) -> np.ndarray:
    """True where a value is more than z_limit robust sigmas from the median,
    above or below."""
    deviation = np.abs(values - np.median(values))
    sigma = robust_sigma(values, resolution)
    return deviation > z_limit * sigma


def average_and_scatter(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel mean and frame-to-frame standard deviation of a (frame, y, x)
    stack. dtype=float64 makes NumPy compute in floating point without first
    copying the whole uint16 stack."""
    mean = stack.mean(axis=0, dtype=np.float64)
    std = stack.std(axis=0, dtype=np.float64, ddof=1)
    return mean, std


# --------------------------------------------------------------------------- #
# One gain
# --------------------------------------------------------------------------- #

def build_flat(light: np.ndarray, dark: np.ndarray,
               smooth_pixels: int = SMOOTH_PIXELS, z_limit: float = Z_LIMIT) -> dict:
    """Turn one gain's lit and dark stacks into a flat and its companions."""
    n_light = light.shape[0]
    n_dark = dark.shape[0]

    # Averages and scatter, per pixel
    light_mean, light_std = average_and_scatter(light)
    dark_mean, dark_std = average_and_scatter(dark)
    signal = light_mean - dark_mean                       # S0

    if np.any(signal <= 0):
        raise ValueError("Flat signal is zero or negative somewhere: "
                         "was the lens really uncovered for the lit frames?")

    # Pixels that disagree with their neighbours (dead, weak, overly sensitive)
    smoothed = median_filter(signal, size=smooth_pixels, mode="nearest")
    neighbour_ratio = signal / smoothed

    # Smallest step the ADC can make, in raw DN: 16 if the camera left-shifts
    # its 12-bit codes into Mono16, otherwise 1.
    step = 2 ** lsb_shift(dark)

    # Finest detail each quantity can resolve (see the module docstring)
    resolution_of_single_frame = step / np.sqrt(12)
    resolution_of_dark_mean = resolution_of_single_frame / np.sqrt(n_dark)
    resolution_of_ratio = resolution_of_single_frame / np.sqrt(n_light) / np.median(signal)

    hot = flag_high(dark_mean, resolution_of_dark_mean,
                    minimum=HOT_MIN_CODES * step, z_limit=z_limit)
    noisy = flag_high(dark_std, resolution_of_single_frame,
                      minimum=NOISY_MIN_CODES * step, z_limit=z_limit)
    odd = flag_either_side(neighbour_ratio, resolution_of_ratio, z_limit=z_limit)
    bad = hot | noisy | odd

    # Normalise to mean 1 over the good pixels
    mu0 = signal[~bad].mean()
    flat = signal / mu0

    # Relative error of each flat pixel: scatter of the averages, light and dark
    error_of_light_mean = light_std / np.sqrt(n_light)
    error_of_dark_mean = dark_std / np.sqrt(n_dark)
    flat_error = np.sqrt(error_of_light_mean**2 + error_of_dark_mean**2) / signal

    # PRNU: pixel-scale spread only (smooth gradients removed by the ratio)
    prnu = neighbour_ratio[~bad].std()

    # Vignetting: how far the smooth illumination falls off, as min/max of the
    # smoothed flat relative to its mean
    smoothed_flat = smoothed / mu0

    return {
        "flat": flat.astype(np.float32),
        "flat_error": flat_error.astype(np.float32),
        "dark": dark_mean.astype(np.float32),
        "bad_pixels": bad,
        "summary": {
            "mu0_dn": float(mu0),
            "prnu": float(prnu),
            "median_flat_error": float(np.median(flat_error[~bad])),
            "vignetting_min": float(smoothed_flat.min()),
            "vignetting_max": float(smoothed_flat.max()),
            "n_hot": int(hot.sum()),
            "n_noisy": int(noisy.sum()),
            "n_odd": int(odd.sum()),
            "n_bad": int(bad.sum()),
            "n_light_frames": n_light,
            "n_dark_frames": n_dark,
            "adc_step_dn": int(step),
        },
    }


# --------------------------------------------------------------------------- #
# Whole run
# --------------------------------------------------------------------------- #

def find_pairs(raw_dir: pathlib.Path) -> list[tuple[pathlib.Path, pathlib.Path]]:
    """Match every light_<gain>dB.h5 with its dark_<gain>dB.h5."""
    pairs = []
    for light_path in sorted(raw_dir.glob("light_*dB.h5")):
        gain_part = light_path.name.removeprefix("light_")      # e.g. "34.0dB.h5"
        dark_path = raw_dir / f"dark_{gain_part}"
        if not dark_path.exists():
            raise FileNotFoundError(f"{light_path.name} has no matching {dark_path.name}")
        pairs.append((light_path, dark_path))

    if not pairs:
        raise FileNotFoundError(f"No light_*dB.h5 files in {raw_dir}")
    return pairs


def check_same_settings(light_attrs: dict, dark_attrs: dict, name: str) -> None:
    """A dark only describes the flat's offset if gain and exposure match."""
    for key in ("gain_db", "exposure_us"):
        if not np.isclose(light_attrs[key], dark_attrs[key]):
            raise ValueError(f"{name}: light and dark differ in {key} "
                             f"({light_attrs[key]} vs {dark_attrs[key]})")


def plot_overview(results: list[dict], path: pathlib.Path) -> None:
    """One panel per gain: the flat, with bad pixels in red."""
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.6), squeeze=False)
    for ax, result in zip(axes[0], results):
        flat = result["flat"]
        low, high = np.percentile(flat, [1, 99])
        image = ax.imshow(flat, cmap="gray", vmin=low, vmax=high)

        bad_y, bad_x = np.nonzero(result["bad_pixels"])
        ax.plot(bad_x, bad_y, "r.", markersize=2)

        s = result["summary"]
        ax.set_title(f"{result['gain_db']:.1f} dB   PRNU {100 * s['prnu']:.2f} %\n"
                     f"{s['n_bad']} bad pixels (red)", fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(out_dir: pathlib.Path, raw_dir: pathlib.Path,
         smooth_pixels: int = SMOOTH_PIXELS, z_limit: float = Z_LIMIT) -> None:
    pairs = find_pairs(raw_dir)

    stamp(out_dir, {
        "inputs": [file_ref(path) for pair in pairs for path in pair],
        "smooth_pixels": smooth_pixels,
        "z_limit": z_limit,
    })

    results = []
    for light_path, dark_path in pairs:
        light_file = load(light_path)
        dark_file = load(dark_path)
        check_same_settings(light_file["attrs"], dark_file["attrs"], light_path.name)

        result = build_flat(light_file["light"], dark_file["dark"], smooth_pixels, z_limit)
        result["gain_db"] = light_file["attrs"]["gain_db"]
        del light_file, dark_file             # stacks no longer needed

        gain_part = light_path.name.removeprefix("light_")      # "34.0dB.h5"
        s = result["summary"]
        save_h5(out_dir / f"flat_{gain_part}", {
            "flat": result["flat"],
            "flat_error": result["flat_error"],
            "dark": result["dark"],
            "bad_pixels": result["bad_pixels"],
            "attrs": {
                **s,
                "gain_db": result["gain_db"],
                "source_light": light_path.name,
                "source_dark": dark_path.name,
                "flat.meaning": "dark-subtracted average lit frame / mu0 (mean 1)",
                "flat_error.meaning": "relative standard error of flat",
                "dark.units": "raw DN (Mono16), average of the dark stack",
                "bad_pixels.meaning": "True = hot, noisy or odd; do not trust the flat here",
            },
        })

        print(f"{result['gain_db']:5.1f} dB:  PRNU {100 * s['prnu']:.2f} %,  "
              f"flat noise {100 * s['median_flat_error']:.2f} %,  "
              f"vignetting {s['vignetting_min']:.2f}..{s['vignetting_max']:.2f},  "
              f"bad pixels {s['n_bad']} (hot {s['n_hot']}, noisy {s['n_noisy']}, "
              f"odd {s['n_odd']}),  ADC step {s['adc_step_dn']} DN")
        if s["median_flat_error"] > 0.3 * s["prnu"]:
            print("        NOTE: flat noise is not small compared with the PRNU; "
                  "more frames would make this flat cleaner")
        results.append(result)

    plot_overview(results, out_dir / "flat_overview.png")
    print(f"\nSaved {len(results)} flat(s) and flat_overview.png in {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=pathlib.Path, help="new folder for the flats")
    ap.add_argument("raw_dir", type=pathlib.Path, help="run folder from flat_acquire_gains.py")
    ap.add_argument("--smooth", type=int, default=SMOOTH_PIXELS,
                    help=f"median-filter size in pixels (default {SMOOTH_PIXELS})")
    ap.add_argument("--z", type=float, default=Z_LIMIT,
                    help=f"robust z-score limit for bad pixels (default {Z_LIMIT})")
    a = ap.parse_args()
    main(a.out_dir, a.raw_dir, a.smooth, a.z)