"""
Photon transfer curve (PTC) analysis: camera gain K, read noise, PRNU, saturation.

Method: Janesick, Photon Transfer (SPIE, 2007).
For every (gain, exposure) point, with dark-subtracted mean signal S [DN]:
    total temporal variance   var_total = var(A - B) / 2        (light pair)
    read (+dark) variance     var_read  = var(Ad - Bd) / 2      (dark pair)
Pair differences cancel fixed-pattern noise, so it never inflates the estimate.
Pixels whose temporal variance is far above the typical one (hot, blinking/RTS)
are excluded from the variances; the fraction excluded is reported.

K [e-/DN] is the inverse slope of a straight-line fit
    var_total = a + S / K
over the shot-noise-limited points (5-70 % of the saturation signal). The free
intercept a makes K immune to any signal-independent variance. a should equal
var_read; a mismatch is reported as "excess" variance present only in lit frames.

Saturation is taken from the variance peak, which covers both ADC clipping and
full well filling before the top code.

Units: native 12-bit DN when the camera left-shifts its 12-bit codes into Mono16,
otherwise 16-bit DN (see labtools/sensor.py). Use the model in the same units,
pixel format and gain it was measured at.

Single-pixel noise model for a raw frame:
    sigma^2(S) = read_dn^2 [+ excess] + S / K
The PRNU term (prnu * S)^2 is spatial. It cancels in flat-fielded or ratio images
and is absent from frame-to-frame fluctuations of the same pixel.

Usage:
    python analyze/ptc_fit.py <out_dir> <raw_run_dir> [<raw_run_dir> ...]
One raw run per gain (see acquire/ptc_acquire.py); all runs go into one summary and plot.
Outputs in <out_dir>:
    ptc_points.csv   every (gain, exposure) point
    ptc_summary.csv  one row per gain: K, read noise, offset, PRNU, saturation
    ptc.png          noise vs signal, log-log, with the fitted model
"""

import pathlib
import sys

import numpy as np

from labtools.sensor import ADC_BITS, clip_level, full_scale, lsb_shift

FIT_RANGE = (0.05, 0.70)        # fraction of the saturation signal where K is fitted
OUTLIER_FACTOR = 50.0           # pixel excluded if its temporal var > 50 x the median one
SATURATED_FRACTION = 1e-3       # point counts as ADC-saturated above this ROI fraction clipped


# --------------------------------------------------------------------------- #
# core (pure NumPy, no I/O)
# --------------------------------------------------------------------------- #

def central_roi(ny: int, nx: int, fraction: float) -> tuple[slice, slice]:
    """Central region covering `fraction` of each dimension (avoids vignetting)."""
    dy, dx = int(ny * fraction / 2), int(nx * fraction / 2)
    return slice(ny // 2 - dy, ny // 2 + dy), slice(nx // 2 - dx, nx // 2 + dx)


def pixel_var(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel temporal variance of a (frame, y, x) float stack, and its outliers.

    Each frame's mean is removed first, so slow illumination drift is not counted.
    For two frames the per-pixel variance is (A - B)^2 / 2. A Gaussian pixel exceeds
    50x the median per-pixel variance with probability ~2e-6, so the cut flags hot
    and blinking pixels without biasing normal ones. The 1 DN^2 floor keeps the cut
    sane when the noise is below one code and the median is 0.
    """
    s = stack - stack.mean(axis=(1, 2), keepdims=True)
    v = s.var(axis=0, ddof=1)
    return v, v > OUTLIER_FACTOR * max(float(np.median(v)), 1.0)


def temporal_var(stack: np.ndarray, bad: np.ndarray | None = None) -> tuple[float, float]:
    """Pixel-averaged temporal variance, excluding outliers and `bad` pixels.
    Returns (variance, fraction of pixels excluded)."""
    v, out = pixel_var(stack)
    if bad is not None:
        out = out | bad
    return float(v[~out].mean()), float(out.mean())


def bad_pixel_mask(dark_stacks) -> np.ndarray:
    """Pixels flagged as outliers in ANY dark pair. Hot and blinking pixels are a
    property of the pixel, and in lit frames their jumps hide under shot noise, so
    they are found in the darks and excluded everywhere."""
    bad = None
    for d in dark_stacks:
        out = pixel_var(d)[1]
        bad = out if bad is None else bad | out
    return bad


def ptc_point(light: np.ndarray, dark: np.ndarray, clip: float,
              bad: np.ndarray | None = None) -> dict:
    """One PTC point from float (frame, y, x) stacks already cut to the ROI."""
    n = light.shape[0]
    offset = dark.mean()
    signal = light.mean() - offset
    var_total, out_light = temporal_var(light, bad)
    var_read, out_dark = temporal_var(dark, bad)
    # spatial variance of the frame-averaged, dark-subtracted image, minus its temporal part
    flat = light.mean(axis=0) - dark.mean(axis=0)
    if bad is not None:
        flat = flat[~bad]
    var_fpn = flat.var(ddof=1) - (var_total + var_read) / n
    clipped = float(np.mean(light >= clip))
    return {
        "offset_dn": offset,
        "signal_dn": signal,
        "var_total": var_total,
        "var_read": var_read,
        "var_shot": var_total - var_read,
        "prnu": np.sqrt(max(var_fpn, 0.0)) / signal if signal > 0 else np.nan,
        "clipped_fraction": clipped,
        "saturated": clipped > SATURATED_FRACTION,
        "outliers_light": out_light,
        "outliers_dark": out_dark,
        "dark_at_zero": float(np.mean(dark <= 0)),
    }


def analyze(dark, light, gain_db, exposure_us, roi_fraction=0.5):
    """dark, light: uint16 (n_gain, n_exp, n_frames, y, x).
    Returns (points, summary): lists of dicts, ready for CSV."""
    if light.shape[2] < 2:
        raise ValueError("Need at least 2 frames per point")
    shift = lsb_shift(light)
    scale = float(1 << shift)
    fs = full_scale(light, shift)            # decided from the bright stack
    clip = clip_level(fs)
    roi = (slice(None),) + central_roi(*light.shape[-2:], roi_fraction)

    points, summary = [], []
    for i in range(light.shape[0]):
        bad = bad_pixel_mask(dark[i, j][roi].astype(np.float64) / scale
                             for j in range(light.shape[1]))
        rows = []
        for j in range(light.shape[1]):
            p = ptc_point(light[i, j][roi].astype(np.float64) / scale,
                          dark[i, j][roi].astype(np.float64) / scale, clip, bad)
            p.update(gain_db=gain_db[i, j], exposure_us=exposure_us[i, j])
            rows.append(p)

        S = np.array([r["signal_dn"] for r in rows])
        v_total = np.array([r["var_total"] for r in rows])
        v_read = np.array([r["var_read"] for r in rows])
        sat = np.array([r["saturated"] for r in rows])
        offset = float(np.median([r["offset_dn"] for r in rows]))

        # Saturation of either kind compresses the noise: variance peaks, then drops.
        # The peak is the last point BEFORE saturation, so the saturation signal is a
        # lower bound, good to one exposure step; more --n-exp near the top tightens it.
        S_peak = float(S[int(np.argmax(v_total))])
        past = S > S_peak
        if not past.any():
            limit, span = "not reached", fs - offset
        else:
            first_past = int(np.argmin(np.where(past, S, np.inf)))
            limit = "ADC" if sat[first_past] else "full well"
            span = S_peak

        use = (~sat) & (~past) & (S > FIT_RANGE[0] * span) & (S < FIT_RANGE[1] * span)
        for r, u, p in zip(rows, use, past):
            r["used_in_fit"] = bool(u)
            r["past_saturation"] = bool(p)

        # read variance: median over the 5 shortest exposures (little dark current, robust)
        short = np.argsort(exposure_us[i])[:5]
        read_var = float(np.median(v_read[short]))

        if use.sum() >= 4:
            # weights 1/var: each variance estimate has about the same RELATIVE error
            (slope, icpt), cov = np.polyfit(S[use], v_total[use], 1, w=1 / v_total[use], cov=True)
            K, K_err = 1 / slope, float(np.sqrt(cov[0, 0]) / slope)
            icpt_err = float(np.sqrt(cov[1, 1]))
        else:
            K = K_err = icpt = icpt_err = np.nan
        prnu = float(np.nanmedian([r["prnu"] for r, u in zip(rows, use) if u])) if use.any() else np.nan

        summary.append({
            "gain_db": float(np.median(gain_db[i])),
            "K_e_per_dn": K,
            "K_rel_err": K_err,
            "read_noise_dn": np.sqrt(read_var),
            "read_noise_e": np.sqrt(read_var) * K,
            "intercept_var_dn2": icpt,
            "excess_var_dn2": icpt - read_var,
            "excess_var_err_dn2": icpt_err,
            "offset_dn": offset,
            "saturation_dn": span,
            "saturation_e": span * K,
            "saturation_limit": limit,
            "prnu": prnu,
            "n_fit_points": int(use.sum()),
            "lsb_shift": shift,
            "full_scale_dn": fs,
            # unsaturated points only: in clipped frames nearly every pixel has zero
            # variance, so the few that don't would all look like outliers
            "bad_pixel_fraction": float(bad.mean()),
            "max_outlier_fraction": max((max(r["outliers_light"], r["outliers_dark"])
                                         for r, s_, p_ in zip(rows, sat, past) if not (s_ or p_)),
                                        default=0.0),
            "dark_at_zero": max(r["dark_at_zero"] for r in rows),
        })
        points.extend(rows)
    return points, summary


def noise_dn(signal_dn, K, read_noise_dn, prnu=0.0, extra_var_dn2=0.0):
    """1-sigma single-pixel noise [DN] at dark-subtracted signal [DN].
    Use prnu=0 for frame-to-frame noise or for flat-fielded / ratio images.
    extra_var_dn2: the summary's excess_var_dn2, if it is real in your measurement too."""
    S = np.clip(signal_dn, 0, None)
    return np.sqrt(read_noise_dn**2 + extra_var_dn2 + S / K + (prnu * S) ** 2)


# --------------------------------------------------------------------------- #
# I/O and plotting
# --------------------------------------------------------------------------- #

def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    keys = list(rows[0])
    lines = [",".join(keys)]
    for r in rows:
        lines.append(",".join(f"{r[k]:.10g}" if isinstance(r[k], float) else str(r[k]) for k in keys))
    path.write_text("\n".join(lines) + "\n")


def plot(points, summary, path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for s in summary:
        rows = [p for p in points if p.get("run") == s.get("run")
                and abs(p["gain_db"] - s["gain_db"]) < 0.05]
        S = np.array([p["signal_dn"] for p in rows])
        sig = np.sqrt([p["var_total"] for p in rows])
        used = np.array([p["used_in_fit"] for p in rows])
        line, = ax.loglog(S[used], sig[used], "o", label=f"{s['gain_db']:.0f} dB, "
                          f"K={s['K_e_per_dn']:.3g} e-/DN, read={s['read_noise_e']:.2g} e-")
        ax.loglog(S[~used & (S > 0)], sig[~used & (S > 0)], "o", mfc="none", color=line.get_color())
        if np.isfinite(s["K_e_per_dn"]):
            Sm = np.geomspace(max(S[S > 0].min(), 1e-1), S.max(), 200)
            ax.loglog(Sm, np.sqrt(np.clip(s["intercept_var_dn2"] + Sm / s["K_e_per_dn"], 1e-12, None)),
                      "-", color=line.get_color(), lw=1)
    ax.set_xlabel("dark-subtracted signal [DN]")
    ax.set_ylabel("temporal noise [DN]")
    ax.set_title("Photon transfer curve (filled: used in K fit)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def report(s: dict) -> str:
    lines = [f"{s['gain_db']:5.1f} dB  K={s['K_e_per_dn']:.4g} e-/DN (±{100*s['K_rel_err']:.1f}%)  "
             f"read={s['read_noise_dn']:.3g} DN = {s['read_noise_e']:.3g} e-  "
             f"sat={s['saturation_e']:.4g} e- ({s['saturation_limit']})  "
             f"PRNU={100*s['prnu']:.2f}%  fit pts={s['n_fit_points']}  "
             f"[units: {'native 12-bit' if s['lsb_shift'] or s['full_scale_dn'] < 4096 else '16-bit'} DN]"]
    if s["n_fit_points"] < 4:
        lines.append("   NO FIT: fewer than 4 unsaturated points in the 5-70 % range. "
                     "Check the exposure range and the saturation flags in ptc_points.csv.")
    ex, ex_err = s["excess_var_dn2"], s["excess_var_err_dn2"]
    if np.isfinite(ex) and ex > 3 * ex_err and ex > 0.1 * s["read_noise_dn"] ** 2:
        lines.append(f"   NOTE: lit frames carry {ex:.4g} ± {ex_err:.2g} DN^2 "
                     f"(= {np.sqrt(ex) * s['K_e_per_dn']:.2g} e- rms) of signal-independent variance "
                     "absent from the darks. K (slope) is unaffected.")
    if s["bad_pixel_fraction"] > 1e-4:
        lines.append(f"   hot/blinking pixels (found in darks): {100*s['bad_pixel_fraction']:.3f} % "
                     "of the ROI, excluded from all variances.")
    if s["dark_at_zero"] > 1e-4:
        lines.append(f"   WARNING: {100*s['dark_at_zero']:.2f} % of dark pixels at 0; "
                     "noise is clipped. Raise BlackLevel.")
    return "\n".join(lines)


def main(out_dir: pathlib.Path, raw_dirs: list[pathlib.Path], roi_fraction: float = 0.5) -> None:
    import h5py
    from labtools.provenance import file_ref, stamp

    h5_files = [d / "ptc.h5" for d in raw_dirs]
    stamp(out_dir, {"inputs": [file_ref(h) for h in h5_files], "roi_fraction": roi_fraction,
                    "fit_range": list(FIT_RANGE), "adc_bits": ADC_BITS,
                    "outlier_factor": OUTLIER_FACTOR})

    points, summary = [], []
    for h5 in h5_files:                         # one file at a time keeps memory to one run
        with h5py.File(h5, "r") as f:
            p, s = analyze(f["dark"][...], f["light"][...],
                           f["gain_db"][...], f["exposure_us"][...], roi_fraction)
        for row in p + s:
            row["run"] = h5.parent.name
        points += p
        summary += s
    summary.sort(key=lambda s: s["gain_db"])

    write_csv(out_dir / "ptc_points.csv", points)
    write_csv(out_dir / "ptc_summary.csv", summary)
    plot(points, summary, out_dir / "ptc.png")
    for s in summary:
        print(report(s))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(pathlib.Path(sys.argv[1]), [pathlib.Path(p) for p in sys.argv[2:]])
