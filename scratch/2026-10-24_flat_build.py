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

Bad pixels (MAD test: robust z-score > Z_LIMIT; the MAD is the median
absolute deviation, which outliers cannot inflate):
    hot     average dark level far above the rest
    noisy   frame-to-frame scatter in the darks far above the rest
    odd     response far from its NEIGHBOURS: flat divided by a median-
            smoothed copy of itself. Comparing with neighbours rather than the
            whole frame keeps vignetting from being flagged. The edges of dust
            shadows can land here too; that is on the safe side.
A pixel can fall into more than one group (a blinking pixel also raises its
average dark level, so it is usually "hot" as well); n_bad counts it once.

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

SMOOTH_PIXELS = 15      # median-filter size for the "neighbours" comparison
Z_LIMIT = 5.0           # robust z-score above which a pixel is called bad


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #

def robust_z(values: np.ndarray) -> np.ndarray:
    """How many robust standard deviations each value is from the median.

    1.4826 * MAD equals the standard deviation for Gaussian data, but a few
    extreme values (the very pixels we look for) barely change it.
    """
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    robust_sigma = 1.4826 * mad
    if robust_sigma == 0:
        raise ValueError("All values identical: cannot judge outliers")
    return (values - median) / robust_sigma


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

    hot = robust_z(dark_mean) > z_limit
    noisy = robust_z(dark_std) > z_limit
    odd = np.abs(robust_z(neighbour_ratio)) > z_limit
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
              f"odd {s['n_odd']})")
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
